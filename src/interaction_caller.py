#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import sys
import h5py
import subprocess

import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import skimage
from statsmodels.stats.multitest import multipletests


def get_proc_chroms(chrom_lens, rank, n_proc):
    chrom_list = [(k, chrom_lens[k]) for k in list(chrom_lens.keys())]
    chrom_list.sort(key=lambda x: x[1])
    chrom_list.reverse()

    chrom_names = [i[0] for i in chrom_list]
    indices = list(range(rank, len(chrom_names), n_proc))
    proc_chroms = [chrom_names[i] for i in indices]

    return proc_chroms


def combine_chrom_interactions(directory):
    headers = "\t".join([
        "chr1", "x1", "x2", "chr2", "y1", "y2",
        "support_cell_count",
        "case_avg",
        "control_avg",
        "pvalue",
        "stat",
        "fdr_dist",
        "fdr_chrom",
    ])

    output_filename_temp = os.path.join(directory, "combined_significances.bedpe.temp")
    output_filename = os.path.join(directory, "combined_significances.bedpe")
    input_filepattern = directory + "/significances.*.bedpe"

    proc = subprocess.Popen(
        "awk 'FNR>1' " + input_filepattern + " > " + output_filename_temp,
        shell=True,
    )
    proc.communicate()

    with open(output_filename, "w") as ofile:
        ofile.write(headers + "\n")

    proc = subprocess.Popen(
        " ".join(["cat", output_filename_temp, ">>", output_filename]),
        shell=True,
    )
    proc.communicate()


def determine_dense_matrix_size(num_cells, dist, binsize, max_mem):
    max_mem_floats = max_mem * 1e9
    max_mem_floats /= 8

    square_cells = max_mem_floats // num_cells
    mat_size = int(np.floor(np.sqrt(square_cells)) / 4)
    mat_size = max(int((dist // binsize) + 50), mat_size)

    return mat_size


def _get_hdf_key(hdf_file, chrom):
    if chrom in hdf_file:
        return chrom

    keynames = list(hdf_file.keys())
    non_cell_keys = [k for k in keynames if k != "cellnames"]

    if len(non_cell_keys) == 0:
        raise KeyError(
            "Cannot find matrix dataset in HDF file. "
            f"Available keys: {keynames}"
        )

    return non_cell_keys[0]


def convert_sparse_dataframe_to_dense_matrix(
    d,
    mat_size,
    dist,
    binsize,
    upper_limit,
    num_cells,
    chrom_size,
    chrom_filename,
    chrom,
    max_distance_bin,
    neighborhood_limit_lower,
):
    """
    将 combined BEDPE + cells.hdf 分块转换为 dense matrix。

    关键修改：
    1. interaction × cell 矩阵缺失值直接补 0。
    2. 额外生成 target_pairs_local，只对原始 union interactions 做显著性检验。
    3. 保留 SnapHiC 中 yield 相关的中间 print。
    """
    d["i"] = (d.iloc[:, 1] // binsize).astype(int)
    d["j"] = (d.iloc[:, 4] // binsize).astype(int)

    max_distance_bin = dist // binsize
    chrom_bins = int(chrom_size // binsize)

    step_size = int(mat_size - max_distance_bin)
    if step_size <= 0:
        raise ValueError(
            "Invalid matrix block step size. "
            f"mat_size={mat_size}, max_distance_bin={max_distance_bin}. "
            "Please increase max_mem or decrease dist."
        )

    for i in range(0, chrom_bins + 1, step_size):
        matrix_upper_bound = max(0, i - upper_limit)
        matrix_lower_bound = min(i + mat_size + upper_limit, chrom_bins + 1)

        keeprows = list(
            np.where(
                (d["i"] >= matrix_upper_bound)
                & (d["j"] < matrix_lower_bound)
            )[0]
        )

        if len(keeprows) == 0:
            continue

        d_keep = d.iloc[keeprows].copy()

        d_portion = d_keep.iloc[:, 0:6].reset_index(drop=True)
        d_portion.columns = ["chr1", "x1", "x2", "chr2", "y1", "y2"]

        with h5py.File(chrom_filename + ".cells.hdf", "r") as hdf_file:
            keyname = _get_hdf_key(hdf_file, chrom)
            portion = hdf_file[keyname][keeprows, :]

        if portion.shape[0] == 0:
            continue

        portion = pd.DataFrame(portion)
        portion = pd.concat([d_portion, portion], axis=1)

        portion["i"] = (portion.loc[:, "x1"] // binsize).astype(int)
        portion["j"] = (portion.loc[:, "y1"] // binsize).astype(int)

        full_sparse = pd.DataFrame({
            "i": range(min(portion["i"]), max(portion["j"]) - 1),
            "j": range(min(portion["i"]) + 1, max(portion["j"])),
        })

        portion = portion.merge(full_sparse, on=["i", "j"], how="outer")

        mats = []
        local_neighborhoods = []

        for cell_index in range(num_cells):
            cell_values = portion.iloc[:, 6 + cell_index]

            # 原理上 interaction × cell 缺失直接补 0，不在检验阶段再做额外处理。
            cell_values = pd.to_numeric(cell_values, errors="coerce").fillna(0.0)

            cell_mat = sp.sparse.csr_matrix(
                (
                    cell_values,
                    (
                        (portion["i"] - matrix_upper_bound),
                        (portion["j"] - matrix_upper_bound),
                    ),
                ),
                shape=(
                    matrix_lower_bound - matrix_upper_bound,
                    matrix_lower_bound - matrix_upper_bound,
                ),
            )

            cell_mat = np.array(cell_mat.todense(), dtype=float)

            # 与 SnapHiC 一致：下三角和对角线不参与邻域计算。
            cell_mat[np.tril_indices(cell_mat.shape[0], 0)] = np.nan

            mat = np.expand_dims(cell_mat, 2)

            if matrix_upper_bound == 0:
                pad_size = abs(i - upper_limit)
                mat = np.pad(
                    mat,
                    ((pad_size, 0), (pad_size, 0), (0, 0)),
                    mode="constant",
                    constant_values=np.nan,
                )

            if matrix_lower_bound == chrom_bins + 1:
                pad_size = upper_limit
                mat = np.pad(
                    mat,
                    ((0, pad_size), (0, pad_size), (0, 0)),
                    mode="constant",
                    constant_values=np.nan,
                )

            start_index = i

            mat, local_neighborhood = get_mat_and_neighborhood(
                mat,
                upper_limit,
                neighborhood_limit_lower,
                1,
                start_index,
                max_distance_bin,
            )

            print("returned size", mat.shape, local_neighborhood.shape)

            local_neighborhoods.append(local_neighborhood)
            mats.append(mat)

        local_neighborhoods = np.stack(local_neighborhoods, axis=-1)
        mat_3d = np.stack(mats, axis=-1)
        mat_3d = np.squeeze(mat_3d)

        if mat_3d.ndim == 2:
            mat_3d = mat_3d[:, :, np.newaxis]

        if local_neighborhoods.ndim == 2:
            local_neighborhoods = local_neighborhoods[:, :, np.newaxis]

        target_pairs = d_keep[["i", "j"]].drop_duplicates().copy()
        target_pairs["local_i"] = target_pairs["i"] - start_index
        target_pairs["local_j"] = target_pairs["j"] - start_index

        h, w = mat_3d.shape[:2]

        target_pairs = target_pairs[
            (target_pairs["local_i"] >= 0)
            & (target_pairs["local_i"] < h)
            & (target_pairs["local_j"] >= 0)
            & (target_pairs["local_j"] < w)
            & ((target_pairs["local_j"] - target_pairs["local_i"]) > 0)
            & ((target_pairs["local_j"] - target_pairs["local_i"]) <= max_distance_bin)
        ].copy()

        target_pairs_local = target_pairs[["local_i", "local_j"]].astype(int).to_numpy()

        if target_pairs_local.shape[0] == 0:
            continue

        print("yielding", mat_3d.shape, local_neighborhoods.shape)

        yield mat_3d, local_neighborhoods, i, target_pairs_local


def get_nth_diag_indices(mat, offset):
    rows, cols_orig = np.diag_indices_from(mat)
    cols = cols_orig.copy()

    if offset > 0:
        cols += offset
        rows = rows[:-offset]
        cols = cols[:-offset]

    return rows, cols


def get_neighbor_counts_matrix(shape, gap_large, gap_small, max_distance):
    a = np.zeros(shape)

    big_width = gap_large * 2 + 1
    small_width = gap_small * 2 + 1
    area = big_width ** 2 - small_width ** 2

    for i in range(0, big_width):
        val = np.sum(list(range(big_width - i)))
        rows, cols = get_nth_diag_indices(a, i + 1)
        a[rows, cols] -= val

        rows, cols = get_nth_diag_indices(a, max_distance - i)
        a[rows, cols] -= val

    for i in range(0, small_width):
        val = np.sum(list(range(small_width - i)))
        rows, cols = get_nth_diag_indices(a, i + 1)
        a[rows, cols] += val

        rows, cols = get_nth_diag_indices(a, max_distance - i)
        a[rows, cols] += val

    a += area

    return a


def get_mat_and_neighborhood(
    mat,
    upper_limit,
    lower_limit,
    num_cells,
    start_index,
    max_distance_bin,
):
    gc.collect()

    big_neighborhood = skimage.util.view_as_windows(
        mat,
        (2 * upper_limit + 1, 2 * upper_limit + 1, num_cells),
        step=1,
    )

    small_neighborhood = skimage.util.view_as_windows(
        mat,
        (2 * lower_limit + 1, 2 * lower_limit + 1, num_cells),
        step=1,
    )

    big_neighborhood = np.squeeze(
        big_neighborhood.reshape(
            big_neighborhood.shape[0],
            big_neighborhood.shape[0],
            1,
            -1,
            num_cells,
        )
    )

    small_neighborhood = np.squeeze(
        small_neighborhood.reshape(
            small_neighborhood.shape[0],
            small_neighborhood.shape[0],
            1,
            -1,
            num_cells,
        )
    )

    big_neighborhood_counts = np.sum(~np.isnan(big_neighborhood), axis=-1)
    small_neighborhood_counts = np.sum(~np.isnan(small_neighborhood), axis=-1)

    print("big:", big_neighborhood.shape)
    print("small:", small_neighborhood.shape)

    big_neighborhood = np.nansum(big_neighborhood, axis=-1)
    small_neighborhood = np.nansum(small_neighborhood, axis=-1)

    trim_size = upper_limit - lower_limit

    small_neighborhood = small_neighborhood[
        trim_size:-trim_size,
        trim_size:-trim_size,
    ]

    small_neighborhood_counts = small_neighborhood_counts[
        trim_size:-trim_size,
        trim_size:-trim_size,
    ]

    mat = mat[upper_limit:-upper_limit, upper_limit:-upper_limit]

    local_neighborhood = big_neighborhood - small_neighborhood
    local_neighborhood_counts = big_neighborhood_counts - small_neighborhood_counts

    print(
        "bn:",
        big_neighborhood_counts,
        "sn:",
        small_neighborhood_counts,
        "ln:",
        local_neighborhood_counts,
        "ul:",
        upper_limit,
        "ll:",
        lower_limit,
        "si:",
        start_index,
    )

    del small_neighborhood
    del big_neighborhood
    del big_neighborhood_counts
    del small_neighborhood_counts
    gc.collect()

    local_neighborhood = local_neighborhood / local_neighborhood_counts

    del local_neighborhood_counts
    gc.collect()

    return mat, local_neighborhood


def _wilcoxon_vectorized(case_values, control_values):
    """
    批量 Wilcoxon signed-rank test。

    case_values:
        shape = (n_interactions, n_cells)

    control_values:
        shape = (n_interactions, n_cells)

    不在这里做 diff==0 或 NaN 清理。
    interaction × cell 缺失已经在构建矩阵阶段补 0。
    """
    try:
        res = stats.wilcoxon(
            case_values,
            control_values,
            axis=1,
            alternative="greater",
            zero_method="wilcox",
            correction=False,
            mode="auto",
        )

        stat = np.asarray(res.statistic, dtype=float)
        pvalue = np.asarray(res.pvalue, dtype=float)

    except Exception:
        # 如果某些旧版 scipy 不支持 axis 参数，则明确报错，避免静默退回慢循环。
        raise RuntimeError(
            "Current scipy.stats.wilcoxon does not support vectorized axis=1. "
            "Please upgrade scipy, e.g. scipy>=1.9, or use a newer conda environment."
        )

    return stat, pvalue


def compute_significances(
    mat,
    local_neighborhood,
    upper_limit,
    lower_limit,
    num_cells,
    start_index,
    max_distance_bin,
    target_pairs_local,
):
    """
    计算显著性。

    关键修改：
    1. 不再扫描 dense block 完整上三角。
    2. 只对 combined BEDPE 中真实存在的 union interactions 做检验。
    3. Wilcoxon 使用 scipy 的 axis=1 批量计算。
    """
    print("in compute:", mat.shape, local_neighborhood.shape)

    if mat.ndim == 2:
        mat = mat[:, :, np.newaxis]

    if local_neighborhood.ndim == 2:
        local_neighborhood = local_neighborhood[:, :, np.newaxis]

    h, w, n_cells_actual = mat.shape

    rows = target_pairs_local[:, 0].astype(int)
    cols = target_pairs_local[:, 1].astype(int)

    valid = (
        (rows >= 0)
        & (rows < h)
        & (cols >= 0)
        & (cols < w)
        & ((cols - rows) > 0)
        & ((cols - rows) <= max_distance_bin)
    )

    rows = rows[valid]
    cols = cols[valid]

    if rows.size == 0:
        return pd.DataFrame(
            columns=["i", "j", "case_avg", "control_avg", "pvalue", "stat"]
        )

    pair_df = pd.DataFrame({"row": rows, "col": cols}).drop_duplicates()
    rows = pair_df["row"].to_numpy(dtype=int)
    cols = pair_df["col"].to_numpy(dtype=int)

    case_values = mat[rows, cols, :]
    control_values = local_neighborhood[rows, cols, :]

    case_avg = np.mean(case_values, axis=1)
    control_avg = np.mean(control_values, axis=1)

    stat, pvalue = _wilcoxon_vectorized(case_values, control_values)

    result = pd.DataFrame({
        "i": rows + start_index,
        "j": cols + start_index,
        "case_avg": case_avg,
        "control_avg": control_avg,
        "pvalue": pvalue,
        "stat": stat,
    })

    result.loc[:, "i"] = result["i"].astype(int)
    result.loc[:, "j"] = result["j"].astype(int)

    result = result[result["j"] - result["i"] <= max_distance_bin]

    return result


def _standardize_combined_columns(d):
    if d.shape[1] < 7:
        raise ValueError(
            "Combined BEDPE must contain at least 7 columns: "
            "chr1 x1 x2 chr2 y1 y2 support_cell_count"
        )

    d_main = d.iloc[:, list(range(7))].copy()

    d_main.columns = [
        "chr1", "x1", "x2",
        "chr2", "y1", "y2",
        "support_cell_count",
    ]

    d_main["support_cell_count"] = pd.to_numeric(
        d_main["support_cell_count"],
        errors="coerce",
    ).fillna(0).astype(int)

    return d_main


def call_interactions(
    indir,
    outdir,
    chrom_lens,
    binsize,
    dist,
    neighborhood_limit_lower=3,
    neighborhood_limit_upper=5,
    rank=0,
    n_proc=1,
    max_mem=2,
    logger=None,
):
    if logger:
        logger.set_rank(rank)

    try:
        os.makedirs(outdir)
    except Exception:
        pass

    proc_chroms = get_proc_chroms(chrom_lens, rank, n_proc)

    for chrom in proc_chroms:
        if logger:
            logger.write(
                f"\tprocessor {rank}: computing for chromosome {chrom}",
                verbose_level=1,
                allow_all_ranks=True,
            )

        chrom_filename = os.path.join(
            indir,
            ".".join([chrom, "raw", "combined", "bedpe"]),
        )

        if not os.path.exists(chrom_filename):
            chrom_filename_alt = os.path.join(
                indir,
                ".".join([chrom, "normalized", "combined", "bedpe"]),
            )

            if os.path.exists(chrom_filename_alt):
                chrom_filename = chrom_filename_alt

        if not os.path.exists(chrom_filename):
            raise FileNotFoundError(
                f"Cannot find combined BEDPE file for {chrom}: "
                f"{chrom_filename} or normalized alternative"
            )

        hdf_filename = chrom_filename + ".cells.hdf"

        if not os.path.exists(hdf_filename):
            raise FileNotFoundError(
                f"Cannot find HDF matrix for {chrom}: {hdf_filename}"
            )

        with h5py.File(hdf_filename, "r") as ifile:
            keyname = _get_hdf_key(ifile, chrom)
            num_cells = ifile[keyname].shape[1]

        if logger:
            logger.write(
                f"\tprocessor {rank}: detected {num_cells} cells for chromosome {chrom}",
                append_time=False,
                allow_all_ranks=True,
                verbose_level=2,
            )

        d = pd.read_csv(chrom_filename, sep="\t", header=None)

        matrix_max_size = determine_dense_matrix_size(
            num_cells,
            dist,
            binsize,
            max_mem,
        )

        max_distance_bin = dist // binsize

        submatrices = convert_sparse_dataframe_to_dense_matrix(
            d,
            matrix_max_size,
            dist,
            binsize,
            neighborhood_limit_upper,
            num_cells,
            chrom_lens[chrom],
            chrom_filename,
            chrom,
            max_distance_bin,
            neighborhood_limit_lower,
        )

        results = []

        for i, (
            submatrix,
            local_neighborhood,
            start_index,
            target_pairs_local,
        ) in enumerate(submatrices):

            if logger:
                logger.write(
                    f"\tprocessor {rank}: computing background for batch {i} of {chrom}, "
                    f"start index = {start_index}",
                    verbose_level=3,
                    allow_all_ranks=True,
                    append_time=False,
                )

            if i > 0 and len(results) > 0:
                results[-1] = results[-1][results[-1]["i"] < start_index]

            submat_result = compute_significances(
                submatrix,
                local_neighborhood,
                neighborhood_limit_upper,
                neighborhood_limit_lower,
                num_cells,
                start_index,
                max_distance_bin,
                target_pairs_local,
            )

            if submat_result.shape[0] > 0:
                results.append(submat_result)

        output_file = os.path.join(
            outdir,
            ".".join(["significances", chrom, "bedpe"]),
        )

        empty_cols = [
            "chr1", "x1", "x2", "chr2", "y1", "y2",
            "support_cell_count",
            "case_avg",
            "control_avg",
            "pvalue",
            "stat",
            "fdr_dist",
            "fdr_chrom",
        ]

        if len(results) == 0:
            pd.DataFrame(columns=empty_cols).to_csv(
                output_file,
                sep="\t",
                index=False,
            )
            continue

        results = pd.concat(results, axis=0, ignore_index=True)

        min_index = 0
        max_index = results["j"].max()

        results = results[
            (results["i"] >= min_index + neighborhood_limit_upper)
            & (results["j"] <= max_index - neighborhood_limit_upper)
        ].copy()

        if results.shape[0] == 0:
            pd.DataFrame(columns=empty_cols).to_csv(
                output_file,
                sep="\t",
                index=False,
            )
            continue

        def compute_fdr_by_dist(dsub):
            dsub = dsub.copy()
            fdrs = multipletests(list(dsub["pvalue"]), method="fdr_bh")[1]
            dsub.loc[:, "fdr_dist"] = fdrs
            return dsub

        results.reset_index(drop=True, inplace=True)

        results = results.groupby(
            results["j"] - results["i"],
            group_keys=False,
        ).apply(compute_fdr_by_dist)

        results.loc[:, "fdr_chrom"] = multipletests(
            list(results["pvalue"]),
            method="fdr_bh",
        )[1]

        results.loc[:, "i"] = (results["i"] * binsize).astype(int)
        results.loc[:, "j"] = (results["j"] * binsize).astype(int)

        d_out = _standardize_combined_columns(d)

        d_out = d_out.merge(
            results,
            left_on=["x1", "y1"],
            right_on=["i", "j"],
            how="inner",
        )

        d_out.drop(["i", "j"], axis=1, inplace=True)

        d_out = d_out[
            [
                "chr1", "x1", "x2", "chr2", "y1", "y2",
                "support_cell_count",
                "case_avg",
                "control_avg",
                "pvalue",
                "stat",
                "fdr_dist",
                "fdr_chrom",
            ]
        ]

        if logger:
            logger.write(
                f"\tprocessor {rank}: computation for {chrom} completed. writing to file.",
                append_time=False,
                allow_all_ranks=True,
                verbose_level=2,
            )

        d_out.to_csv(
            output_file,
            sep="\t",
            index=False,
        )