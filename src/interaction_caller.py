#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import skimage
import subprocess
from statsmodels.stats.multitest import multipletests
import gc
import sys
import h5py


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
        "outlier_count",
        "case_avg",
        "control_avg",
        "pvalue",
        "stat",
        "n_eff",
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


def convert_sparse_dataframe_to_dense_matrix(
    d,
    mat_size,
    dist,
    binsize,
    upper_limit,
    num_cells,
    chrom_size,
    chrom_filename,
    max_distance_bin,
    neighborhood_limit_lower,
):
    d["i"] = (d.iloc[:, 1] // binsize).astype(int)
    d["j"] = (d.iloc[:, 4] // binsize).astype(int)

    max_distance_bin = dist // binsize
    chrom_bins = int(chrom_size // binsize)

    for i in range(0, chrom_bins + 1, int(mat_size - max_distance_bin)):
        matrix_upper_bound = max(0, i - upper_limit)
        matrix_lower_bound = min(i + mat_size + upper_limit, chrom_bins + 1)

        keeprows = list(
            np.where(
                (d["i"] >= matrix_upper_bound)
                & (d["j"] < matrix_lower_bound)
            )[0]
        )

        d_portion = d.iloc[keeprows, 0:6].reset_index(drop=True)
        d_portion.columns = ["chr1", "x1", "x2", "chr2", "y1", "y2"]

        if len(keeprows) == 0:
            continue

        hdf_file = h5py.File(chrom_filename + ".cells.hdf", "r")
        keynames = list(hdf_file.keys())
        keyname = keynames[0] if keynames[0] != "cellnames" else keynames[1]
        portion = hdf_file[keyname]
        portion = portion[keeprows, :]
        hdf_file.close()

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

            # 下三角和对角线不参与计算
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

            local_neighborhoods.append(local_neighborhood)
            mats.append(mat)

        local_neighborhoods = np.stack(local_neighborhoods, axis=-1)
        mat_3d = np.stack(mats, axis=-1)
        mat_3d = np.squeeze(mat_3d)

        yield mat_3d, local_neighborhoods, i


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

    # sliding window
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

    big_neighborhood = np.nansum(big_neighborhood, axis=-1)
    small_neighborhood = np.nansum(small_neighborhood, axis=-1)

    # remove edge cases that are used only as neighbors
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

    del small_neighborhood
    del big_neighborhood
    del big_neighborhood_counts
    del small_neighborhood_counts
    gc.collect()

    # 避免除以 0
    local_neighborhood_counts = local_neighborhood_counts.astype(float)
    local_neighborhood_counts[local_neighborhood_counts == 0] = np.nan

    local_neighborhood = local_neighborhood / local_neighborhood_counts

    del local_neighborhood_counts
    gc.collect()

    return mat, local_neighborhood


def _wilcoxon_one_position(x, y):
    """
    对一个 interaction 的多个 pseudocell 值进行单边 Wilcoxon signed-rank test。

    x:
        center values across pseudocells

    y:
        local background values across pseudocells

    H1:
        x > y

    返回：
        stat, pvalue, n_eff
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() == 0:
        return 0.0, 1.0, 0

    diff = x[mask] - y[mask]
    diff = diff[np.isfinite(diff)]
    diff = diff[diff != 0]

    n_eff = int(diff.size)

    if n_eff == 0:
        return 0.0, 1.0, 0

    try:
        stat, pvalue = stats.wilcoxon(
            diff,
            alternative="greater",
            zero_method="wilcox",
            correction=False,
            mode="auto",
        )
    except Exception:
        return 0.0, 1.0, n_eff

    if not np.isfinite(stat):
        stat = 0.0
    if not np.isfinite(pvalue):
        pvalue = 1.0

    return float(stat), float(pvalue), n_eff


def compute_significances(
    mat,
    local_neighborhood,
    upper_limit,
    lower_limit,
    num_cells,
    start_index,
    max_distance_bin,
):
    """
    原始代码中这里使用：
        stats.ttest_rel(mat, local_neighborhood, axis=2)

    现在改为：
        one-sided Wilcoxon signed-rank test
        H1: mat > local_neighborhood

    其余矩阵化、分块、坐标转换逻辑保持原始结构。
    """

    # ------------------------------------------------------------
    # 1. 对每个矩阵位置计算 Wilcoxon
    # ------------------------------------------------------------
    if mat.ndim == 2:
        mat = mat[:, :, np.newaxis]

    if local_neighborhood.ndim == 2:
        local_neighborhood = local_neighborhood[:, :, np.newaxis]

    h, w, n_cells_actual = mat.shape

    pvals = np.ones((h, w), dtype=float)
    stat = np.zeros((h, w), dtype=float)
    n_eff = np.zeros((h, w), dtype=np.int16)

    upper_rows, upper_cols = np.triu_indices(h, k=1)

    for r, c in zip(upper_rows, upper_cols):
        if c - r > max_distance_bin:
            continue

        x = mat[r, c, :]
        y = local_neighborhood[r, c, :]

        s, p, n = _wilcoxon_one_position(x, y)

        stat[r, c] = s
        pvals[r, c] = p
        n_eff[r, c] = n

    # ------------------------------------------------------------
    # 2. 计算 case/control 平均值
    # ------------------------------------------------------------
    local_neighborhood_mean = np.nanmean(local_neighborhood, axis=-1)
    mat_mean = np.nanmean(mat, axis=-1)

    # ------------------------------------------------------------
    # 3. 只保留上三角
    # ------------------------------------------------------------
    mat_mean = np.triu(mat_mean, 1)
    local_neighborhood_mean = np.triu(local_neighborhood_mean, 1)

    pvals = np.triu(pvals, 1)
    pvals = np.nan_to_num(pvals, nan=1.0)

    stat = np.triu(stat, 1)
    stat = np.nan_to_num(stat, nan=0.0)

    n_eff = np.triu(n_eff, 1)

    # ------------------------------------------------------------
    # 4. 转 dataframe
    # ------------------------------------------------------------
    mat_sparse = sp.sparse.coo_matrix(mat_mean)
    local_sparse = sp.sparse.coo_matrix(local_neighborhood_mean)
    pval_sparse = sp.sparse.coo_matrix(pvals)
    stat_sparse = sp.sparse.coo_matrix(stat)
    neff_sparse = sp.sparse.coo_matrix(n_eff)

    result_mat = pd.DataFrame({
        "i": mat_sparse.row,
        "j": mat_sparse.col,
        "case_avg": mat_sparse.data,
    })

    result_neighb = pd.DataFrame({
        "i": local_sparse.row,
        "j": local_sparse.col,
        "control_avg": local_sparse.data,
    })

    result_pval = pd.DataFrame({
        "i": pval_sparse.row,
        "j": pval_sparse.col,
        "pvalue": pval_sparse.data,
    })

    result_stat = pd.DataFrame({
        "i": stat_sparse.row,
        "j": stat_sparse.col,
        "stat": stat_sparse.data,
    })

    result_neff = pd.DataFrame({
        "i": neff_sparse.row,
        "j": neff_sparse.col,
        "n_eff": neff_sparse.data,
    })

    result = result_mat.merge(result_neighb, on=["i", "j"], how="outer")
    result = result.merge(result_pval, on=["i", "j"], how="outer")
    result = result.merge(result_stat, on=["i", "j"], how="outer")
    result = result.merge(result_neff, on=["i", "j"], how="outer")

    result.loc[:, "pvalue"] = result["pvalue"].fillna(1.0)
    result.loc[:, "stat"] = result["stat"].fillna(0.0)
    result.loc[:, "n_eff"] = result["n_eff"].fillna(0).astype(int)

    result.loc[:, "i"] += start_index
    result.loc[:, "j"] += start_index

    result.loc[:, "i"] = result["i"].astype(int)
    result.loc[:, "j"] = result["j"].astype(int)

    result = result[result["j"] - result["i"] <= max_distance_bin]

    return result


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

        # 如果新流程输出没有 raw 命名，则兼容 normalized 命名
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
            if chrom in ifile:
                num_cells = ifile[chrom].shape[1]
            else:
                keynames = list(ifile.keys())
                keyname = keynames[0] if keynames[0] != "cellnames" else keynames[1]
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
            max_distance_bin,
            neighborhood_limit_lower,
        )

        results = []

        for i, (submatrix, local_neighborhood, start_index) in enumerate(submatrices):
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
            )

            results.append(submat_result)

        if len(results) == 0:
            empty_cols = [
                "chr1", "x1", "x2", "chr2", "y1", "y2",
                "outlier_count",
                "case_avg", "control_avg", "pvalue", "stat", "n_eff",
                "fdr_dist", "fdr_chrom",
            ]
            pd.DataFrame(columns=empty_cols).to_csv(
                os.path.join(outdir, ".".join(["significances", chrom, "bedpe"])),
                sep="\t",
                index=False,
            )
            continue

        results = pd.concat(results, axis=0)

        min_index = 0
        max_index = results["j"].max()

        results = results[
            (results["i"] >= min_index + neighborhood_limit_upper)
            & (results["j"] <= max_index - neighborhood_limit_upper)
        ]

        def compute_fdr_by_dist(dsub):
            fdrs = multipletests(list(dsub["pvalue"]), method="fdr_bh")[1]
            dsub.loc[:, "fdr_dist"] = fdrs
            return dsub

        results.reset_index(drop=True, inplace=True)

        if results.shape[0] > 0:
            results = results.groupby(
                results["j"] - results["i"],
                as_index=False,
            ).apply(compute_fdr_by_dist)

            results.loc[:, "fdr_chrom"] = multipletests(
                list(results["pvalue"]),
                method="fdr_bh",
            )[1]
        else:
            results["fdr_dist"] = []
            results["fdr_chrom"] = []

        results.loc[:, "i"] = (results["i"] * binsize).astype(int)
        results.loc[:, "j"] = (results["j"] * binsize).astype(int)

        d = d.iloc[:, list(range(7))]
        d.columns = [
            "chr1", "x1", "x2",
            "chr2", "y1", "y2",
            "outlier_count",
        ]

        d = d.merge(results, left_on=["x1", "y1"], right_on=["i", "j"])
        d.drop(["i", "j"], axis=1, inplace=True)

        if logger:
            logger.write(
                f"\tprocessor {rank}: computation for {chrom} completed. writing to file.",
                append_time=False,
                allow_all_ranks=True,
                verbose_level=2,
            )

        d.to_csv(
            os.path.join(outdir, ".".join(["significances", chrom, "bedpe"])),
            sep="\t",
            index=False,
        )