#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prefilter_support.py

Pseudocell-radius support filter for raw-count pseudocell BEDPE.

该模块不再作为独立 support step 使用，而是在 postprocess 内部由 --support-filter 触发。

输入：
  postprocessed/candidates.{chrom}.bedpe
  pseudocell raw BEDPE directory

pseudocell raw BEDPE 格式：
  chr1  x1  x2  chr2  y1  y2  count

support filter 定义：
  对 candidates.{chrom}.bedpe 中每个 candidate interaction (i, j)，
  在每个 pseudocell raw BEDPE 中统计其 Chebyshev 半径 r 内有读数的 interaction 数量。

窗口：
  abs(i' - i) <= r
  abs(j' - j) <= r
  不包括中心 interaction (i, j)
  只统计 i' < j' 的 intra-chrom interaction

每个 candidate 会得到：
  support_n1, support_n2, ..., support_nN

然后：
  support_min_count = min(support_n1 ... support_nN)
  support_window_total = 半径窗口内理论 interaction 数量，不包括中心 interaction
  support_threshold_count = support_window_total * support_ratio

筛选条件：
  support_min_count > support_threshold_count

输出：
  1. candidate_support.{chrom}.bedpe
     保留 candidates 原始所有列，并补充：
       support_n1 ... support_nN
       support_min_count
       support_window_total
       support_threshold_count
       pass_support

  2. candidates_filter.{chrom}.bedpe
     只保留通过 support filter 的 candidate 原始列，不包含 support 相关列。

性能设计：
  support_radius_bins 通常很小，例如 1 或 2。
  因此这里不使用 KDTree，而是：
    1. 每个 pseudocell 构建 set((bin_i, bin_j))；
    2. 对每个 candidate 枚举半径窗口内有限数量 offsets；
    3. 用 set membership 快速判断是否有读数。
"""

import os
import glob
import re
import sys
import math
import numpy as np
import pandas as pd


BEDPE6_COLS = ["chr1", "x1", "x2", "chr2", "y1", "y2"]
BEDPE7_COLS = ["chr1", "x1", "x2", "chr2", "y1", "y2", "value"]

_NAME_RE = re.compile(r"^(?P<pb>.+?)\.(?P<chrom>chr[^.]+)\..*?\.bedpe(?:\.gz)?$")


# ============================================================
# Logging
# ============================================================

def _log(logger, msg, rank=0, v=1):
    if logger is not None:
        try:
            logger.write(
                msg,
                append_time=False,
                allow_all_ranks=True,
                verbose_level=v,
            )
            logger.flush()
            return
        except Exception:
            pass

    if rank == 0:
        print(msg, file=sys.stderr, flush=True)


# ============================================================
# File reading
# ============================================================

def _read_bedpe7(path):
    """
    读取 raw pseudocell BEDPE 文件。

    期望格式：
      chr1 x1 x2 chr2 y1 y2 value

    只使用前 7 列。
    """
    compression = "gzip" if str(path).endswith(".gz") else None

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            compression=compression,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=BEDPE7_COLS)

    if df.shape[0] == 0:
        return pd.DataFrame(columns=BEDPE7_COLS)

    if df.shape[1] < 7:
        raise ValueError(f"[support] {path} has fewer than 7 columns: {df.shape[1]}")

    df = df.iloc[:, :7].copy()
    df.columns = BEDPE7_COLS

    df["chr1"] = df["chr1"].astype(str)
    df["chr2"] = df["chr2"].astype(str)

    for c in ["x1", "x2", "y1", "y2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(-1).astype(np.int64)

    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0).astype(float)

    return df


def _infer_pb_and_chrom_from_name(path):
    """
    尽量从文件名中推断 pseudocell 名称和染色体。
    """
    base = os.path.basename(path)

    m = _NAME_RE.match(base)
    if m:
        return m.group("pb"), m.group("chrom")

    if ".chr" in base:
        pb = base.split(".chr")[0]
        chrom = "chr" + base.split(".chr")[1].split(".")[0]
        return pb, chrom

    m2 = re.search(r"(chr[0-9XYM]+)", base)
    if m2:
        chrom = m2.group(1)
        pb = base.split(chrom)[0].rstrip(".")
        return pb, chrom

    return base, "unknown"


def _find_pseudocell_files(pseudocell_dir, chrom, pattern="*.bedpe*"):
    """
    在 pseudocell_dir 中查找指定染色体的 pseudocell raw BEDPE 文件。

    支持命名形式：
      cell1.chr1.xxx.bedpe
      cell1.chr1.bedpe
      xxx_chr1_xxx.bedpe
    """
    files = []

    for fp in glob.glob(os.path.join(pseudocell_dir, pattern)):
        if os.path.isdir(fp):
            continue

        base = os.path.basename(fp)

        if (
            f".{chrom}." in base
            or base.endswith(f".{chrom}.bedpe")
            or base.endswith(f".{chrom}.bedpe.gz")
            or f"_{chrom}_" in base
            or f"_{chrom}." in base
            or f".{chrom}_" in base
        ):
            files.append(fp)

    if len(files) == 0:
        files = glob.glob(os.path.join(pseudocell_dir, f"*{chrom}*.bedpe*"))

    out = []
    for fp in files:
        if os.path.isfile(fp):
            pb, inferred_chrom = _infer_pb_and_chrom_from_name(fp)
            out.append((pb, fp))

    out = sorted(out, key=lambda x: x[0])

    return out


# ============================================================
# Coordinate conversion
# ============================================================

def _standardize_to_positive_bin_set(df, chrom, binsize):
    """
    将一个 pseudocell BEDPE 转为有读数的 bin-pair set。

    返回：
      set((i, j))
    """
    if df.shape[0] == 0:
        return set()

    df = df[(df["chr1"] == chrom) & (df["chr2"] == chrom)].copy()
    if df.shape[0] == 0:
        return set()

    df = df[df["value"] > 0].copy()
    if df.shape[0] == 0:
        return set()

    a = (df["x1"].astype(np.int64) // int(binsize)).astype(np.int64)
    b = (df["y1"].astype(np.int64) // int(binsize)).astype(np.int64)

    lo = np.minimum(a, b)
    hi = np.maximum(a, b)

    mask = lo < hi

    lo = lo[mask]
    hi = hi[mask]

    if len(lo) == 0:
        return set()

    pairs = set(zip(lo.astype(int).tolist(), hi.astype(int).tolist()))

    return pairs


def _candidate_to_bin_arrays(candidates, binsize):
    """
    candidates DataFrame 转为 bin_i, bin_j arrays。
    """
    for c in ["x1", "x2", "y1", "y2"]:
        if c in candidates.columns:
            candidates[c] = pd.to_numeric(candidates[c], errors="coerce").fillna(0).astype(np.int64)

    i = (candidates["x1"].astype(np.int64) // int(binsize)).astype(np.int64).to_numpy()
    j = (candidates["y1"].astype(np.int64) // int(binsize)).astype(np.int64).to_numpy()

    lo = np.minimum(i, j)
    hi = np.maximum(i, j)

    return lo.astype(np.int64), hi.astype(np.int64)


# ============================================================
# Window offsets
# ============================================================

def make_support_offsets(support_radius_bins):
    """
    生成 support 窗口 offsets。

    Chebyshev 半径：
      abs(di) <= r
      abs(dj) <= r

    不包括中心点：
      di == 0 and dj == 0

    注意：
      对于 candidate (i,j)，实际点为 (i+di, j+dj)。
      后续还会过滤：
        a >= 0, b >= 0, a < b
    """
    r = int(support_radius_bins)

    if r < 0:
        raise ValueError("[support] support_radius_bins must be non-negative")

    offsets = []

    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di == 0 and dj == 0:
                continue
            offsets.append((di, dj))

    return offsets


def count_window_total_for_candidate(i, j, offsets):
    """
    计算某个 candidate 的理论窗口 interaction 数量。

    不包括中心 interaction。
    只统计 a < b 且坐标非负的点。
    """
    total = 0

    i = int(i)
    j = int(j)

    for di, dj in offsets:
        a = i + int(di)
        b = j + int(dj)

        if a < 0 or b < 0:
            continue
        if a >= b:
            continue

        total += 1

    return total


def count_support_for_one_cell(bin_set, cand_i, cand_j, offsets):
    """
    对一个 pseudocell 计算所有 candidates 的窗口支持数量。

    输入：
      bin_set:
        set((i,j))，当前 pseudocell 中有读数的 bin pair。

      cand_i, cand_j:
        candidate 中心 bin 坐标数组。

      offsets:
        support 窗口 offsets。

    输出：
      counts:
        每个 candidate 半径窗口内有读数的 interaction 数量。
    """
    n = len(cand_i)
    counts = np.zeros(n, dtype=np.int32)

    if len(bin_set) == 0 or n == 0 or len(offsets) == 0:
        return counts

    # 半径通常为 1 或 2，offset 数量很小；
    # 外层遍历 offsets，内层遍历 candidates，避免为每个 candidate 重建窗口。
    for di, dj in offsets:
        a_arr = cand_i + int(di)
        b_arr = cand_j + int(dj)

        valid = (a_arr >= 0) & (b_arr >= 0) & (a_arr < b_arr)
        if not np.any(valid):
            continue

        valid_idx = np.where(valid)[0]

        # set membership 需要 Python tuple；这里仅对有效 candidate 查询。
        for idx in valid_idx:
            key = (int(a_arr[idx]), int(b_arr[idx]))
            if key in bin_set:
                counts[idx] += 1

    return counts


# ============================================================
# Core filter
# ============================================================

def filter_candidates_by_pseudocell_support(
    candidate_file,
    output_candidate_support_file,
    output_candidate_filtered_file,
    pseudocell_dir,
    chrom,
    binsize,
    support_radius_bins,
    support_ratio,
    pattern="*.bedpe*",
    logger=None,
    rank=0,
):
    """
    对单个 chromosome 的 candidates.{chrom}.bedpe 执行 support filter。

    输出：
      candidate_support.{chrom}.bedpe
      candidates_filter.{chrom}.bedpe
    """

    if not os.path.exists(candidate_file):
        raise FileNotFoundError(f"[support] candidate file not found: {candidate_file}")

    try:
        candidates = pd.read_csv(candidate_file, sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        candidates = pd.DataFrame(columns=BEDPE6_COLS)

    original_columns = list(candidates.columns)

    if candidates.shape[0] == 0:
        candidate_support = candidates.copy()
        candidate_support["support_min_count"] = []
        candidate_support["support_window_total"] = []
        candidate_support["support_threshold_count"] = []
        candidate_support["pass_support"] = []

        candidate_support.to_csv(output_candidate_support_file, sep="\t", index=False)
        candidates.to_csv(output_candidate_filtered_file, sep="\t", index=False)

        _log(
            logger,
            f"[support] [{chrom}] empty candidates, wrote empty support/filter files",
            rank=rank,
            v=2,
        )
        return

    required = {"chr1", "x1", "x2", "chr2", "y1", "y2"}
    if not required.issubset(set(candidates.columns)):
        raise ValueError(
            f"[support] candidate file must contain columns {sorted(required)}: {candidate_file}"
        )

    candidates = candidates.copy()

    cand_i, cand_j = _candidate_to_bin_arrays(candidates, binsize=binsize)
    n_candidates = len(cand_i)

    offsets = make_support_offsets(support_radius_bins)

    files = _find_pseudocell_files(
        pseudocell_dir=pseudocell_dir,
        chrom=chrom,
        pattern=pattern,
    )

    if len(files) == 0:
        raise FileNotFoundError(
            f"[support] no pseudocell BEDPE files found for {chrom} in {pseudocell_dir}"
        )

    n_cells = len(files)

    _log(
        logger,
        (
            f"[support] [{chrom}] candidates={n_candidates}, "
            f"pseudocells={n_cells}, radius={support_radius_bins}, "
            f"ratio={support_ratio}, offsets={len(offsets)}"
        ),
        rank=rank,
        v=2,
    )

    support_counts = np.zeros((n_candidates, n_cells), dtype=np.int32)

    for cell_idx, (cell_name, fp) in enumerate(files):
        df = _read_bedpe7(fp)

        bin_set = _standardize_to_positive_bin_set(
            df=df,
            chrom=chrom,
            binsize=binsize,
        )

        counts = count_support_for_one_cell(
            bin_set=bin_set,
            cand_i=cand_i,
            cand_j=cand_j,
            offsets=offsets,
        )

        support_counts[:, cell_idx] = counts

        _log(
            logger,
            (
                f"[support] [{chrom}] {cell_idx + 1}/{n_cells} "
                f"{os.path.basename(fp)} nonzero_pairs={len(bin_set)}"
            ),
            rank=rank,
            v=3,
        )

    window_total = np.array(
        [
            count_window_total_for_candidate(int(i), int(j), offsets)
            for i, j in zip(cand_i, cand_j)
        ],
        dtype=np.int32,
    )

    support_threshold_count = window_total.astype(float) * float(support_ratio)

    if n_cells > 0:
        support_min_count = support_counts.min(axis=1).astype(np.int32)
    else:
        support_min_count = np.zeros(n_candidates, dtype=np.int32)

    pass_support = support_min_count > support_threshold_count

    candidate_support = candidates.copy()

    # 保留每个 pseudocell 的支持数量。
    # 文件名可能很长，因此列名使用 support_n1...support_nN。
    for cell_idx in range(n_cells):
        candidate_support[f"support_n{cell_idx + 1}"] = support_counts[:, cell_idx]

    candidate_support["support_min_count"] = support_min_count
    candidate_support["support_window_total"] = window_total
    candidate_support["support_threshold_count"] = support_threshold_count
    candidate_support["pass_support"] = pass_support.astype(np.int8)

    candidate_filtered = candidates.loc[pass_support, original_columns].copy()

    candidate_support.to_csv(output_candidate_support_file, sep="\t", index=False)
    candidate_filtered.to_csv(output_candidate_filtered_file, sep="\t", index=False)

    _log(
        logger,
        (
            f"[support] [{chrom}] support filter: "
            f"{n_candidates} -> {candidate_filtered.shape[0]}"
        ),
        rank=rank,
        v=2,
    )


def filter_candidate_dir_by_pseudocell_support(
    postproc_dir,
    pseudocell_dir,
    proc_chroms,
    binsize,
    support_radius_bins,
    support_ratio,
    pattern="*.bedpe*",
    logger=None,
    rank=0,
):
    """
    对 postprocessed 目录下多个 candidates.{chrom}.bedpe 执行 support filter。

    输入：
      postprocessed/candidates.{chrom}.bedpe

    输出：
      postprocessed/candidate_support.{chrom}.bedpe
      postprocessed/candidates_filter.{chrom}.bedpe
    """

    for chrom in proc_chroms:
        candidate_file = os.path.join(postproc_dir, f"candidates.{chrom}.bedpe")
        candidate_support_file = os.path.join(postproc_dir, f"candidate_support.{chrom}.bedpe")
        candidate_filtered_file = os.path.join(postproc_dir, f"candidates_filter.{chrom}.bedpe")

        filter_candidates_by_pseudocell_support(
            candidate_file=candidate_file,
            output_candidate_support_file=candidate_support_file,
            output_candidate_filtered_file=candidate_filtered_file,
            pseudocell_dir=pseudocell_dir,
            chrom=chrom,
            binsize=binsize,
            support_radius_bins=support_radius_bins,
            support_ratio=support_ratio,
            pattern=pattern,
            logger=logger,
            rank=rank,
        )