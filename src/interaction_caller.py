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

    当前逻辑：
    1. 原始 combined BEDPE / HDF 只代表有读数的 interaction；文件中缺失不是“真实 0 行”。
    2. 为计算背景，先扩展出该 block 内所有合法上三角坐标；缺失坐标在 dense matrix 中补 0。
    3. 背景均值分母使用窗口内所有合法坐标位置，即 NaN 以外的位置；补出来的 0 纳入分母。
    4. 仍然只对原始 union interactions，即 d_keep 中真实存在的坐标，做显著性检验。
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

            # 原始数据文件只保留有读数的 interaction；outer merge 产生的 NaN
            # 表示该合法坐标在该 cell 中没有有效读数。
            # 对背景均值而言，这类“合法位置无读数”应按 0 计入，并纳入分母；
            # 因此这里补 0。非法位置仍在后面用下三角/边界 NaN 表示。
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
    """
    计算每个中心点的 local background。

    修改后的逻辑：
    1. 背景窗口分母使用所有合法坐标命中的 interaction 位置。
    2. 原始文件中缺失的合法坐标在 dense matrix 中已经补为 0，并纳入背景均值分母。
    3. NaN 只表示非法位置，例如下三角、对角线或 padding 边界，不参与分母。
    4. 如果某个中心点的背景窗口没有任何合法背景坐标，则该中心点的 local background 设为 0.0。
    5. 显著性检验对象仍只保留原始 union interactions，不会把补出来的 0 坐标作为待检验 interaction。
    """
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

    # 背景均值分母 = 窗口内所有合法坐标位置数。
    # 说明：
    #   - outer merge 后缺失的合法坐标已经在 dense matrix 中补为 0，
    #     表示该位置没有有效读数，但仍是背景窗口内的合法坐标，因此纳入分母；
    #   - NaN 只表示非法位置，包括下三角、对角线和 padding 边界，不纳入分母。
    big_valid = np.isfinite(big_neighborhood)
    small_valid = np.isfinite(small_neighborhood)

    big_neighborhood_counts = np.sum(big_valid, axis=-1)
    small_neighborhood_counts = np.sum(small_valid, axis=-1)

    print("big:", big_neighborhood.shape)
    print("small:", small_neighborhood.shape)

    big_neighborhood = np.sum(
        np.where(big_valid, big_neighborhood, 0.0),
        axis=-1,
    )

    small_neighborhood = np.sum(
        np.where(small_valid, small_neighborhood, 0.0),
        axis=-1,
    )

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
    del big_valid
    del small_valid
    gc.collect()

    # 没有任何合法背景坐标时，背景均值设为 0.0。
    # 正常情况下，缺失但合法的坐标已补 0 并纳入分母；
    # 这里主要是边界/极端窗口的兜底处理。
    with np.errstate(divide="ignore", invalid="ignore"):
        local_neighborhood = np.where(
            local_neighborhood_counts > 0,
            local_neighborhood / local_neighborhood_counts,
            0.0,
        )

    del local_neighborhood_counts
    gc.collect()

    return mat, local_neighborhood


def _wilcoxon_vectorized(case_values, control_values):
    """
    批量 Wilcoxon signed-rank test。

    保持原逻辑不变：
    case_values:
        shape = (n_interactions, n_cells)

    control_values:
        shape = (n_interactions, n_cells)

    这里不额外清理 NaN 或 diff==0。
    本版本中 control_values 不引入 NaN；
    缺失但合法的背景坐标已补 0 并纳入分母，
    没有合法背景坐标的位置在 get_mat_and_neighborhood() 中设为 0.0。
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

        # 与 SnapHiC 原版保持同一原则：
        # 不可检验或全零差异产生的 NaN pvalue 不能进入 multipletests。
        # 原版 t-test 流程中会将 NaN pvalue 替换为 1；
        # 这里 Wilcoxon 出现 NaN 时同样按“不显著”处理。
        stat = np.nan_to_num(stat, nan=0.0, posinf=0.0, neginf=0.0)
        pvalue = np.nan_to_num(pvalue, nan=1.0, posinf=1.0, neginf=1.0)

    except Exception:
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

    逻辑：
    1. 只对 combined BEDPE 中真实存在的 union interactions 做检验。
    2. case_values 是中心 interaction 在每个 cell 中的值。
    3. control_values 是每个 cell 中该中心点背景窗口的均值。
    4. control_values 的背景均值由背景窗口内所有合法坐标位置计算；
       原文件缺失的合法坐标按 0 计入并纳入分母，NaN 非法位置不参与。
    5. 检验对象仍只来自原始 union interactions；补出来的 0 坐标不会作为新的待检验 interaction。
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


def _bh_fdr_safe(pvalues):
    """
    Benjamini-Hochberg FDR correction with SnapHiC-compatible NaN handling.

    SnapHiC 原版在 t-test 后将 NaN pvalue 置为 1，再做：
        1. distance-stratified BH-FDR -> fdr_dist
        2. chromosome-wide BH-FDR    -> fdr_chrom

    pseudocell 这里使用 Wilcoxon，某些全零差异或不可检验情况会产生 NaN。
    如果不先将 NaN pvalue 置为 1，statsmodels.multipletests 可能输出 NaN，
    保存到 TSV 后就表现为 fdr_chrom 空值。
    """
    p = pd.to_numeric(pd.Series(pvalues), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)
    p = np.clip(p, 0.0, 1.0)

    if p.size == 0:
        return np.array([], dtype=float)

    return multipletests(p, method="fdr_bh")[1]


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

        # ============================================================
        # FDR correction, same correction logic as original SnapHiC:
        #   fdr_dist  : BH-FDR within each genomic-distance stratum
        #   fdr_chrom : BH-FDR across all tested interactions on this chromosome
        #
        # Important fix:
        #   Wilcoxon may produce NaN pvalue for all-zero / non-testable pairs.
        #   SnapHiC's t-test path replaces NaN pvalue by 1 before FDR correction.
        #   Here we do the same; otherwise fdr_chrom may be written as empty.
        # ============================================================
        results = results.copy()
        results["pvalue"] = pd.to_numeric(results["pvalue"], errors="coerce").fillna(1.0)
        results["pvalue"] = np.nan_to_num(
            results["pvalue"].to_numpy(dtype=float),
            nan=1.0,
            posinf=1.0,
            neginf=1.0,
        )
        results["pvalue"] = np.clip(results["pvalue"], 0.0, 1.0)

        def compute_fdr_by_dist(dsub):
            dsub = dsub.copy()
            dsub.loc[:, "fdr_dist"] = _bh_fdr_safe(dsub["pvalue"])
            return dsub

        results.reset_index(drop=True, inplace=True)

        results = results.groupby(
            results["j"] - results["i"],
            group_keys=False,
        ).apply(compute_fdr_by_dist)

        results.loc[:, "fdr_chrom"] = _bh_fdr_safe(results["pvalue"])

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

        # 最后再兜底一次，确保输出 TSV 中 fdr_dist/fdr_chrom 不是空值。
        d_out["pvalue"] = pd.to_numeric(d_out["pvalue"], errors="coerce").fillna(1.0)
        d_out["stat"] = pd.to_numeric(d_out["stat"], errors="coerce").fillna(0.0)
        d_out["fdr_dist"] = pd.to_numeric(d_out["fdr_dist"], errors="coerce").fillna(1.0)
        d_out["fdr_chrom"] = pd.to_numeric(d_out["fdr_chrom"], errors="coerce").fillna(1.0)

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