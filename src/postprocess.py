#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
postprocess.py

Postprocessing for raw-count pseudocell Hi-C loop calling.

本版本适配 pseudocell_hic.py：

主要流程：
  1. 读取 interactions/significances.{chrom}.bedpe。
  2. 从预生成的 pseudobulk .hic 文件读取对应染色体 contact map。
  3. 基于 .hic contact map 计算五类结构背景：
       circle
       donut
       lower_left
       horizontal
       vertical
  4. 根据 FDR、Wilcoxon stat、距离范围、support_fraction、五类结构背景过滤，
     生成 candidates.{chrom}.bedpe。
  5. 如果 support_filter=True：
       进一步基于 pseudocell raw BEDPE 计算半径 support filter；
       输出 candidate_support.{chrom}.bedpe；
       输出 candidates_filter.{chrom}.bedpe；
       聚类使用 candidates_filter.{chrom}.bedpe。
     如果 support_filter=False：
       聚类使用 candidates.{chrom}.bedpe。
  6. 聚类生成 clustered.candidates.{chrom}.bedpe。
  7. 合并所有染色体结果生成：
       {prefix}.postprocessed.all_candidates.bedpe
       {prefix}.postprocessed.summits.bedpe

依赖：
  优先使用 hicstraw 读取 .hic。
  需要在环境中安装：
      pip install hic-straw

说明：
  support_filter 不能从 pseudobulk .hic 读取，因为它需要每个 pseudocell 各自的半径窗口有效 interaction 数量。
"""

import os
import sys
import glob
import math
import numpy as np
import pandas as pd


# ============================================================
# Optional import: support filter
# ============================================================

try:
    from src.prefilter_support import filter_candidate_dir_by_pseudocell_support
except ImportError:
    try:
        from prefilter_support import filter_candidate_dir_by_pseudocell_support
    except ImportError:
        filter_candidate_dir_by_pseudocell_support = None


# ============================================================
# Optional import: hicstraw
# ============================================================

try:
    import hicstraw
except ImportError:
    hicstraw = None


# ============================================================
# Constants
# ============================================================

BEDPE6_COLS = ["chr1", "x1", "x2", "chr2", "y1", "y2"]

DEFAULT_SIGNIF_COLS = [
    "chr1", "x1", "x2", "chr2", "y1", "y2",
    "center_value",
    "background_mean",
    "stat",
    "n_eff",
    "pvalue",
    "qvalue",
    "support_fraction",
    "distance",
]


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
# Basic helpers
# ============================================================

def get_proc_chroms(chrom_lens, rank, n_proc):
    """
    按染色体长度从大到小分配给不同 rank，保留 SnapHiC 风格并行逻辑。
    """
    chrom_list = [(k, chrom_lens[k]) for k in list(chrom_lens.keys())]
    chrom_list.sort(key=lambda x: x[1], reverse=True)

    chrom_names = [x[0] for x in chrom_list]
    indices = list(range(rank, len(chrom_names), n_proc))
    proc_chroms = [chrom_names[i] for i in indices]

    return proc_chroms


def _safe_numeric(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _ensure_bedpe_numeric(df):
    for c in ["x1", "x2", "y1", "y2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64)
    return df


def _standardize_bedpe_columns(df):
    """
    标准化 significances 文件列名。

    如果文件已有 header，则保留原列名。
    如果没有 header，则使用 DEFAULT_SIGNIF_COLS + extra_col。
    """
    if df.shape[0] == 0:
        return pd.DataFrame(columns=DEFAULT_SIGNIF_COLS)

    first_vals = [str(x) for x in df.iloc[0, : min(df.shape[1], 6)].tolist()]
    looks_header = any(x in {"chr1", "x1", "x2", "chr2", "y1", "y2"} for x in first_vals)

    if looks_header:
        df.columns = df.iloc[0].astype(str).tolist()
        df = df.iloc[1:, :].copy()
        return df.reset_index(drop=True)

    ncol = df.shape[1]
    if ncol <= len(DEFAULT_SIGNIF_COLS):
        cols = DEFAULT_SIGNIF_COLS[:ncol]
    else:
        cols = DEFAULT_SIGNIF_COLS + [f"extra_{i}" for i in range(ncol - len(DEFAULT_SIGNIF_COLS))]

    df.columns = cols
    return df


def infer_column(df, candidates, required=False, default=None):
    for c in candidates:
        if c in df.columns:
            return c

    if required:
        raise KeyError(f"None of candidate columns found: {candidates}")

    return default


# ============================================================
# Filter regions
# ============================================================

def read_filter_regions(filter_file, binsize):
    """
    读取 blacklist/filter regions。

    支持：
      1. chr start end
      2. chr bin
    返回：
      set((chrom, bin_id))
    """
    bad = set()

    if filter_file is None:
        return bad

    if not os.path.exists(filter_file):
        return bad

    try:
        df = pd.read_csv(filter_file, sep="\t", header=None, comment="#")
    except pd.errors.EmptyDataError:
        return bad

    if df.shape[0] == 0:
        return bad

    if df.shape[1] >= 3:
        for _, row in df.iterrows():
            chrom = str(row.iloc[0])
            start = int(row.iloc[1])
            end = int(row.iloc[2])

            b1 = start // int(binsize)
            b2 = max(start, end - 1) // int(binsize)

            for b in range(b1, b2 + 1):
                bad.add((chrom, b))

    elif df.shape[1] >= 2:
        for _, row in df.iterrows():
            chrom = str(row.iloc[0])
            b = int(row.iloc[1])
            bad.add((chrom, b))

    return bad


def apply_filter_regions(df, filter_regions, binsize):
    """
    删除任一 anchor 落入 blacklist/filter region 的 interaction。
    """
    if df.shape[0] == 0 or len(filter_regions) == 0:
        return df

    keep = []

    for _, row in df.iterrows():
        chrom1 = str(row["chr1"])
        chrom2 = str(row["chr2"])

        b1 = int(row["x1"]) // int(binsize)
        b2 = int(row["y1"]) // int(binsize)

        if (chrom1, b1) in filter_regions or (chrom2, b2) in filter_regions:
            keep.append(False)
        else:
            keep.append(True)

    return df.loc[np.asarray(keep, dtype=bool)].copy()


# ============================================================
# .hic reader
# ============================================================

def _normalize_chrom_name_for_hic(chrom):
    """
    hicstraw 通常可以接受 chr1；这里保留原名。
    如果用户 .hic 中染色体没有 chr 前缀，需要自行保证 CHROMS 与 .hic 一致。
    """
    return str(chrom)


def load_hic_contact_map(
    hic_path,
    chrom,
    binsize,
    normalization="NONE",
    unit="BP",
    logger=None,
    rank=0,
):
    """
    从 pseudobulk .hic 中读取单染色体 contact map。

    返回：
      dict[(bin_i, bin_j)] = count

    其中 bin_i/bin_j 是 binsize 对应的 bin index。
    """
    if hicstraw is None:
        raise ImportError(
            "hicstraw is required for reading .hic files. "
            "Install it with: pip install hic-straw"
        )

    if not os.path.exists(hic_path):
        raise FileNotFoundError(f"[postprocess] pseudobulk .hic not found: {hic_path}")

    chrom_hic = _normalize_chrom_name_for_hic(chrom)

    _log(
        logger,
        f"[postprocess] [{chrom}] loading .hic contact map from {hic_path}",
        rank=rank,
        v=2,
    )

    try:
        result = hicstraw.straw(
            normalization,
            hic_path,
            chrom_hic,
            chrom_hic,
            unit,
            int(binsize),
        )
    except Exception as e:
        raise RuntimeError(
            f"[postprocess] failed to read .hic for {chrom} with hicstraw. "
            f"hic={hic_path}, normalization={normalization}, binsize={binsize}. "
            f"Original error: {e}"
        )

    contact = {}

    for rec in result:
        # hicstraw record fields:
        # rec.binX, rec.binY, rec.counts
        i = int(rec.binX) // int(binsize)
        j = int(rec.binY) // int(binsize)

        if i == j:
            continue

        if i > j:
            i, j = j, i

        v = float(rec.counts)

        if v == 0:
            continue

        contact[(i, j)] = contact.get((i, j), 0.0) + v

    _log(
        logger,
        f"[postprocess] [{chrom}] .hic nonzero bin pairs = {len(contact)}",
        rank=rank,
        v=2,
    )

    return contact


def get_contact_value(contact_map, i, j):
    i = int(i)
    j = int(j)

    if i == j:
        return 0.0

    if i > j:
        i, j = j, i

    return float(contact_map.get((i, j), 0.0))


# ============================================================
# Structural background from .hic
# ============================================================

def compute_structural_backgrounds_for_pair(contact_map, i, j, gap_large, gap_small):
    """
    计算五类结构背景。

    输入：
      contact_map:
        dict[(bin_i, bin_j)] = count

      i, j:
        中心 interaction 的 bin 坐标。

      gap_large:
        大窗口半径，单位 bin。

      gap_small:
        中心排除窗口半径，单位 bin。

    背景定义：
      donut:
        大方形窗口中排除中心小窗口后的所有点。

      circle:
        这里保留为独立字段。
        当前实现使用与 donut 相同的候选区域，便于通过 multiplier 独立调参。
        如果你后续要严格复刻原 SnapHiC circle 几何形状，可以只改这个函数。

      horizontal:
        固定左 anchor i，沿右 anchor j 方向取背景，排除中心小窗口。

      vertical:
        固定右 anchor j，沿左 anchor i 方向取背景，排除中心小窗口。

      lower_left:
        两个 anchor 同时向左下方向移动的局部背景。
    """
    i = int(i)
    j = int(j)

    if i > j:
        i, j = j, i

    gl = int(gap_large)
    gs = int(gap_small)

    circle_vals = []
    donut_vals = []
    horizontal_vals = []
    vertical_vals = []
    lower_left_vals = []

    # donut / circle
    for di in range(-gl, gl + 1):
        for dj in range(-gl, gl + 1):
            if abs(di) <= gs and abs(dj) <= gs:
                continue

            a = i + di
            b = j + dj

            if a < 0 or b < 0:
                continue
            if a >= b:
                continue

            v = get_contact_value(contact_map, a, b)
            donut_vals.append(v)
            circle_vals.append(v)

    # horizontal
    for dj in range(-gl, gl + 1):
        if abs(dj) <= gs:
            continue

        a = i
        b = j + dj

        if a < 0 or b < 0:
            continue
        if a >= b:
            continue

        horizontal_vals.append(get_contact_value(contact_map, a, b))

    # vertical
    for di in range(-gl, gl + 1):
        if abs(di) <= gs:
            continue

        a = i + di
        b = j

        if a < 0 or b < 0:
            continue
        if a >= b:
            continue

        vertical_vals.append(get_contact_value(contact_map, a, b))

    # lower-left diagonal
    for d in range(gs + 1, gl + 1):
        a = i - d
        b = j - d

        if a < 0 or b < 0:
            continue
        if a >= b:
            continue

        lower_left_vals.append(get_contact_value(contact_map, a, b))

    def mean_or_zero(vals):
        if len(vals) == 0:
            return 0.0
        return float(np.mean(vals))

    return {
        "circle": mean_or_zero(circle_vals),
        "donut": mean_or_zero(donut_vals),
        "lower_left": mean_or_zero(lower_left_vals),
        "horizontal": mean_or_zero(horizontal_vals),
        "vertical": mean_or_zero(vertical_vals),
    }


def add_structural_background_columns_from_hic(
    df,
    contact_map,
    binsize,
    gap_large,
    gap_small,
):
    """
    给 significance/candidate DataFrame 增加：
      pseudobulk_center
      circle
      donut
      lower_left
      horizontal
      vertical
    """
    df = df.copy()

    if df.shape[0] == 0:
        for c in [
            "pseudobulk_center",
            "circle",
            "donut",
            "lower_left",
            "horizontal",
            "vertical",
        ]:
            df[c] = []
        return df

    centers = np.zeros(df.shape[0], dtype=float)
    circles = np.zeros(df.shape[0], dtype=float)
    donuts = np.zeros(df.shape[0], dtype=float)
    lower_lefts = np.zeros(df.shape[0], dtype=float)
    horizontals = np.zeros(df.shape[0], dtype=float)
    verticals = np.zeros(df.shape[0], dtype=float)

    for idx, (_, row) in enumerate(df.iterrows()):
        i = int(row["x1"]) // int(binsize)
        j = int(row["y1"]) // int(binsize)

        if i > j:
            i, j = j, i

        centers[idx] = get_contact_value(contact_map, i, j)

        bg = compute_structural_backgrounds_for_pair(
            contact_map=contact_map,
            i=i,
            j=j,
            gap_large=gap_large,
            gap_small=gap_small,
        )

        circles[idx] = bg["circle"]
        donuts[idx] = bg["donut"]
        lower_lefts[idx] = bg["lower_left"]
        horizontals[idx] = bg["horizontal"]
        verticals[idx] = bg["vertical"]

    df["pseudobulk_center"] = centers
    df["circle"] = circles
    df["donut"] = donuts
    df["lower_left"] = lower_lefts
    df["horizontal"] = horizontals
    df["vertical"] = verticals

    return df


# ============================================================
# Significance file reading
# ============================================================

def read_significance_file(path):
    """
    读取 significances.{chrom}.bedpe。

    支持有 header 或无 header。
    """
    if not os.path.exists(path):
        return pd.DataFrame(columns=DEFAULT_SIGNIF_COLS)

    compression = "gzip" if str(path).endswith(".gz") else None

    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            compression=compression,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=DEFAULT_SIGNIF_COLS)

    if df.shape[0] == 0:
        return pd.DataFrame(columns=DEFAULT_SIGNIF_COLS)

    df = _standardize_bedpe_columns(df)
    df = _ensure_bedpe_numeric(df)

    return df


# ============================================================
# Candidate filtering
# ============================================================

def apply_candidate_filters(
    df,
    fdr_thresh,
    stat_threshold,
    min_support_fraction,
    candidate_lower_thresh,
    candidate_upper_thresh,
    binsize,
    circle_threshold_mult,
    donut_threshold_mult,
    lower_left_threshold_mult,
    horizontal_threshold_mult,
    vertical_threshold_mult,
):
    """
    应用 candidate 过滤条件。

    注意：
      stat_threshold 直接使用用户输入的 args.stat_threshold。
      本函数不对 Wilcoxon statistic 做自适应转换。
    """
    if df.shape[0] == 0:
        return df

    df = df.copy()

    q_col = infer_column(df, ["qvalue", "qval", "fdr", "FDR", "padj", "adj_pvalue"], required=False)
    p_col = infer_column(df, ["pvalue", "pval", "p", "P"], required=False)
    stat_col = infer_column(df, ["stat", "wilcoxon_stat", "tstat", "t_stat"], required=False)
    support_col = infer_column(
        df,
        ["support_fraction", "support_frac", "cell_fraction", "pseudocell_fraction"],
        required=False,
    )

    # FDR / p-value
    if q_col is not None:
        df[q_col] = _safe_numeric(df[q_col], default=1.0)
        keep_fdr = df[q_col] <= float(fdr_thresh)
    elif p_col is not None:
        df[p_col] = _safe_numeric(df[p_col], default=1.0)
        keep_fdr = df[p_col] <= float(fdr_thresh)
    else:
        keep_fdr = pd.Series(True, index=df.index)

    # Wilcoxon statistic
    if stat_col is not None:
        df[stat_col] = _safe_numeric(df[stat_col], default=0.0)
        keep_stat = df[stat_col] >= float(stat_threshold)
    else:
        keep_stat = pd.Series(True, index=df.index)

    # center support fraction from interaction step
    if support_col is not None:
        df[support_col] = _safe_numeric(df[support_col], default=0.0)
        keep_support = df[support_col] >= float(min_support_fraction)
    else:
        keep_support = pd.Series(True, index=df.index)

    # distance
    if "distance" in df.columns:
        df["distance"] = _safe_numeric(df["distance"], default=0.0)
        dist_bp = df["distance"]
    else:
        dist_bp = (df["y1"].astype(np.int64) - df["x1"].astype(np.int64)).abs()

    keep_distance = (
        (dist_bp >= float(candidate_lower_thresh)) &
        (dist_bp <= float(candidate_upper_thresh))
    )

    # structural background filters
    for c in ["pseudobulk_center", "circle", "donut", "lower_left", "horizontal", "vertical"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c], default=0.0)

    if "pseudobulk_center" in df.columns:
        center = df["pseudobulk_center"]

        keep_circle = center > df["circle"] * float(circle_threshold_mult)
        keep_donut = center > df["donut"] * float(donut_threshold_mult)
        keep_lower_left = center > df["lower_left"] * float(lower_left_threshold_mult)
        keep_horizontal = center > df["horizontal"] * float(horizontal_threshold_mult)
        keep_vertical = center > df["vertical"] * float(vertical_threshold_mult)
    else:
        keep_circle = pd.Series(True, index=df.index)
        keep_donut = pd.Series(True, index=df.index)
        keep_lower_left = pd.Series(True, index=df.index)
        keep_horizontal = pd.Series(True, index=df.index)
        keep_vertical = pd.Series(True, index=df.index)

    keep = (
        keep_fdr &
        keep_stat &
        keep_support &
        keep_distance &
        keep_circle &
        keep_donut &
        keep_lower_left &
        keep_horizontal &
        keep_vertical
    )

    return df.loc[keep].copy()


def find_candidates(
    indir,
    outdir,
    proc_chroms,
    chrom_lens,
    fdr_thresh,
    gap_large,
    gap_small,
    candidate_lower_thresh,
    candidate_upper_thresh,
    binsize,
    dist,
    max_mem,
    stat_threshold,
    circle_threshold_mult,
    donut_threshold_mult,
    lower_left_threshold_mult,
    horizontal_threshold_mult,
    vertical_threshold_mult,
    min_support_fraction,
    filter_file=None,
    logger=None,
    rank=0,
    pseudobulk_hic=None,
    hic_normalization="NONE",
):
    """
    对每个 chromosome 生成 candidates.{chrom}.bedpe。
    """
    os.makedirs(outdir, exist_ok=True)

    if pseudobulk_hic is None:
        raise ValueError("[postprocess] --pseudobulk-hic is required for postprocess")

    filter_regions = read_filter_regions(filter_file, binsize)

    for chrom in proc_chroms:
        infile = os.path.join(indir, f"significances.{chrom}.bedpe")
        outfile_nofilter = os.path.join(outdir, f"nofilter.{chrom}.bedpe")
        outfile_candidates = os.path.join(outdir, f"candidates.{chrom}.bedpe")

        _log(logger, f"\tprocessor {rank}: finding candidates for {chrom}", rank=rank, v=2)

        sig = read_significance_file(infile)

        if sig.shape[0] == 0:
            sig.to_csv(outfile_nofilter, sep="\t", index=False)
            sig.to_csv(outfile_candidates, sep="\t", index=False)
            continue

        sig = _ensure_bedpe_numeric(sig)

        # intra-chrom only
        sig = sig[(sig["chr1"].astype(str) == chrom) & (sig["chr2"].astype(str) == chrom)].copy()

        if sig.shape[0] == 0:
            sig.to_csv(outfile_nofilter, sep="\t", index=False)
            sig.to_csv(outfile_candidates, sep="\t", index=False)
            continue

        contact_map = load_hic_contact_map(
            hic_path=pseudobulk_hic,
            chrom=chrom,
            binsize=binsize,
            normalization=hic_normalization,
            unit="BP",
            logger=logger,
            rank=rank,
        )

        sig = add_structural_background_columns_from_hic(
            df=sig,
            contact_map=contact_map,
            binsize=binsize,
            gap_large=gap_large,
            gap_small=gap_small,
        )

        sig = apply_filter_regions(sig, filter_regions, binsize)

        sig.to_csv(outfile_nofilter, sep="\t", index=False)

        cand = apply_candidate_filters(
            df=sig,
            fdr_thresh=fdr_thresh,
            stat_threshold=stat_threshold,
            min_support_fraction=min_support_fraction,
            candidate_lower_thresh=candidate_lower_thresh,
            candidate_upper_thresh=candidate_upper_thresh,
            binsize=binsize,
            circle_threshold_mult=circle_threshold_mult,
            donut_threshold_mult=donut_threshold_mult,
            lower_left_threshold_mult=lower_left_threshold_mult,
            horizontal_threshold_mult=horizontal_threshold_mult,
            vertical_threshold_mult=vertical_threshold_mult,
        )

        cand.to_csv(outfile_candidates, sep="\t", index=False)

        _log(
            logger,
            f"\tprocessor {rank}: {chrom} candidates {sig.shape[0]} -> {cand.shape[0]}",
            rank=rank,
            v=2,
        )


# ============================================================
# Clustering
# ============================================================

def _get_bin_pair(row, binsize):
    i = int(row["x1"]) // int(binsize)
    j = int(row["y1"]) // int(binsize)

    if i > j:
        i, j = j, i

    return i, j


def _choose_summit(cluster_df):
    """
    从一个 cluster 中选择 summit。

    优先级：
      1. qvalue / fdr 最小
      2. stat 最大
      3. pseudobulk_center 最大
      4. center_value 最大
    """
    df = cluster_df.copy()

    sort_cols = []
    ascending = []

    for c in ["qvalue", "qval", "fdr", "FDR", "padj", "adj_pvalue"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c], default=1.0)
            sort_cols.append(c)
            ascending.append(True)
            break

    for c in ["stat", "wilcoxon_stat", "tstat", "t_stat"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c], default=0.0)
            sort_cols.append(c)
            ascending.append(False)
            break

    for c in ["pseudobulk_center", "center_value"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c], default=0.0)
            sort_cols.append(c)
            ascending.append(False)
            break

    if len(sort_cols) == 0:
        return df.iloc[[0]].copy()

    return df.sort_values(sort_cols, ascending=ascending).iloc[[0]].copy()


def _rank_summits_for_gap_filter(summit_df):
    df = summit_df.copy()

    sort_cols = []
    ascending = []

    for c in ["qvalue", "qval", "fdr", "FDR", "padj", "adj_pvalue"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c], default=1.0)
            sort_cols.append(c)
            ascending.append(True)
            break

    for c in ["stat", "wilcoxon_stat", "tstat", "t_stat"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c], default=0.0)
            sort_cols.append(c)
            ascending.append(False)
            break

    for c in ["pseudobulk_center", "center_value"]:
        if c in df.columns:
            df[c] = _safe_numeric(df[c], default=0.0)
            sort_cols.append(c)
            ascending.append(False)
            break

    if len(sort_cols) == 0:
        return list(range(summit_df.shape[0]))

    return df.sort_values(sort_cols, ascending=ascending).index.tolist()


def cluster_one_chrom(df, binsize, clustering_gap, summit_gap):
    """
    单染色体 candidate 聚类。

    规则：
      两个 candidate 的 bin 坐标 Chebyshev 距离 <= clustering_gap，
      则归入同一 cluster。
    """
    if df.shape[0] == 0:
        out = df.copy()
        out["cluster_id"] = []
        out["cluster_size"] = []
        out["is_summit"] = []
        return out

    df = df.copy().reset_index(drop=True)

    pairs = np.asarray([_get_bin_pair(row, binsize) for _, row in df.iterrows()], dtype=np.int64)
    n = pairs.shape[0]

    visited = np.zeros(n, dtype=bool)
    cluster_ids = np.full(n, -1, dtype=np.int64)

    cid = 0

    for start in range(n):
        if visited[start]:
            continue

        queue = [start]
        visited[start] = True
        cluster_ids[start] = cid

        while queue:
            u = queue.pop()

            d = np.maximum(
                np.abs(pairs[:, 0] - pairs[u, 0]),
                np.abs(pairs[:, 1] - pairs[u, 1]),
            )

            neighbors = np.where((d <= int(clustering_gap)) & (~visited))[0]

            for v in neighbors:
                visited[v] = True
                cluster_ids[v] = cid
                queue.append(v)

        cid += 1

    df["cluster_id"] = cluster_ids

    cluster_sizes = df.groupby("cluster_id").size().to_dict()
    df["cluster_size"] = df["cluster_id"].map(cluster_sizes).astype(int)
    df["is_summit"] = 0

    summit_indices = []

    for _, sub in df.groupby("cluster_id", sort=False):
        summit = _choose_summit(sub)
        summit_indices.append(summit.index[0])

    df.loc[summit_indices, "is_summit"] = 1

    # summit_gap 后处理：summit 过近时只保留排序优先级更高的
    if summit_gap is not None and int(summit_gap) > 0 and len(summit_indices) > 1:
        summit_df = df.loc[summit_indices].copy()
        summit_df = summit_df.reset_index().rename(columns={"index": "_orig_idx"})

        ranked = _rank_summits_for_gap_filter(summit_df)

        chosen = []
        blocked = np.zeros(summit_df.shape[0], dtype=bool)

        for local_idx in ranked:
            if blocked[local_idx]:
                continue

            chosen.append(int(summit_df.loc[local_idx, "_orig_idx"]))

            i0, j0 = _get_bin_pair(summit_df.loc[local_idx], binsize)

            for k in range(summit_df.shape[0]):
                if blocked[k]:
                    continue

                i1, j1 = _get_bin_pair(summit_df.loc[k], binsize)
                d_bp = max(abs(i0 - i1), abs(j0 - j1)) * int(binsize)

                if d_bp < int(summit_gap):
                    blocked[k] = True

        df["is_summit"] = 0
        df.loc[chosen, "is_summit"] = 1

    return df


def cluster_candidates(
    outdir,
    proc_chroms,
    clustering_gap,
    binsize,
    summit_gap,
    logger=None,
    rank=0,
    candidate_input_prefix="candidates",
):
    """
    对 candidates 或 candidates_filter 聚类。

    输入：
      {candidate_input_prefix}.{chrom}.bedpe

    输出：
      clustered.candidates.{chrom}.bedpe
    """
    for chrom in proc_chroms:
        infile = os.path.join(outdir, f"{candidate_input_prefix}.{chrom}.bedpe")
        outfile = os.path.join(outdir, f"clustered.candidates.{chrom}.bedpe")

        if not os.path.exists(infile):
            raise FileNotFoundError(f"[cluster] input candidate file not found: {infile}")

        try:
            df = pd.read_csv(infile, sep="\t", low_memory=False)
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=BEDPE6_COLS)

        if df.shape[0] == 0:
            df.to_csv(outfile, sep="\t", index=False)
            _log(logger, f"\tprocessor {rank}: {chrom} clustering empty input", rank=rank, v=2)
            continue

        df = _ensure_bedpe_numeric(df)

        clustered = cluster_one_chrom(
            df=df,
            binsize=binsize,
            clustering_gap=clustering_gap,
            summit_gap=summit_gap,
        )

        clustered.to_csv(outfile, sep="\t", index=False)

        n_summit = int(clustered["is_summit"].sum()) if "is_summit" in clustered.columns else 0

        _log(
            logger,
            f"\tprocessor {rank}: {chrom} clustered candidates={clustered.shape[0]}, summits={n_summit}",
            rank=rank,
            v=2,
        )


# ============================================================
# Compatibility hook
# ============================================================

def append_zscores(outdir, proc_chroms):
    """
    新 pseudocell raw-count 流程不需要 zscore。
    保留空函数用于兼容旧接口。
    """
    return


# ============================================================
# Main postprocess
# ============================================================

def postprocess(
    indir,
    outdir,
    chrom_lens,
    fdr_thresh,
    gap_large,
    gap_small,
    candidate_lower_thresh,
    candidate_upper_thresh,
    binsize,
    dist,
    clustering_gap,
    rank,
    n_proc,
    max_mem,
    tstat_threshold=None,
    circle_threshold_mult=1.33,
    donut_threshold_mult=1.33,
    lower_left_threshold_mult=1.33,
    horizontal_threshold_mult=1.2,
    vertical_threshold_mult=1.2,
    outlier_threshold_mult=None,
    filter_file=None,
    summit_gap=20000,
    logger=None,
    support_dir=None,
    support_filter=False,
    pseudobulk_hic=None,
    hic_normalization="NONE",
    raw_sc_bedpe_dir=None,
    raw_sc_pattern="*.bedpe*",
    stat_threshold=None,
    min_support_fraction=None,
    pseudocell_dir=None,
    support_radius_bins=1,
    support_ratio=0.5,
    pseudocell_pattern="*.bedpe*",
):
    """
    Raw pseudocell postprocess.

    注意：
      raw_sc_bedpe_dir/raw_sc_pattern 参数仅保留兼容，不再使用。
      五类结构背景直接从 pseudobulk_hic 读取。
    """
    if logger:
        try:
            logger.set_rank(rank)
        except Exception:
            pass

    os.makedirs(outdir, exist_ok=True)

    if stat_threshold is None:
        stat_threshold = 0.0 if tstat_threshold is None else tstat_threshold

    if min_support_fraction is None:
        min_support_fraction = 0.0 if outlier_threshold_mult is None else outlier_threshold_mult

    if pseudobulk_hic is None:
        raise ValueError("[postprocess] --pseudobulk-hic is required")

    proc_chroms = get_proc_chroms(chrom_lens, rank, n_proc)

    _log(
        logger,
        f"\tprocessor {rank}: postprocess chromosomes = {','.join(proc_chroms)}",
        rank=rank,
        v=2,
    )

    find_candidates(
        indir=indir,
        outdir=outdir,
        proc_chroms=proc_chroms,
        chrom_lens=chrom_lens,
        fdr_thresh=fdr_thresh,
        gap_large=gap_large,
        gap_small=gap_small,
        candidate_lower_thresh=candidate_lower_thresh,
        candidate_upper_thresh=candidate_upper_thresh,
        binsize=binsize,
        dist=dist,
        max_mem=max_mem,
        stat_threshold=stat_threshold,
        circle_threshold_mult=circle_threshold_mult,
        donut_threshold_mult=donut_threshold_mult,
        lower_left_threshold_mult=lower_left_threshold_mult,
        horizontal_threshold_mult=horizontal_threshold_mult,
        vertical_threshold_mult=vertical_threshold_mult,
        min_support_fraction=min_support_fraction,
        filter_file=filter_file,
        logger=logger,
        rank=rank,
        pseudobulk_hic=pseudobulk_hic,
        hic_normalization=hic_normalization,
    )

    if support_filter:
        if filter_candidate_dir_by_pseudocell_support is None:
            raise ImportError(
                "[support] support_filter=True but src.prefilter_support.filter_candidate_dir_by_pseudocell_support "
                "could not be imported."
            )

        if pseudocell_dir is None:
            raise ValueError("[support] support_filter=True but pseudocell_dir is None")

        _log(
            logger,
            (
                f"\tprocessor {rank}: applying pseudocell-radius support filter "
                f"(radius={support_radius_bins}, ratio={support_ratio})"
            ),
            rank=rank,
            v=2,
        )

        filter_candidate_dir_by_pseudocell_support(
            postproc_dir=outdir,
            pseudocell_dir=pseudocell_dir,
            proc_chroms=proc_chroms,
            binsize=binsize,
            support_radius_bins=support_radius_bins,
            support_ratio=support_ratio,
            pattern=pseudocell_pattern,
            logger=logger,
            rank=rank,
        )

        candidate_input_prefix = "candidates_filter"
    else:
        candidate_input_prefix = "candidates"

    _log(
        logger,
        f"\tprocessor {rank}: clustering input = {candidate_input_prefix}.{{chrom}}.bedpe",
        rank=rank,
        v=2,
    )

    cluster_candidates(
        outdir=outdir,
        proc_chroms=proc_chroms,
        clustering_gap=clustering_gap,
        binsize=binsize,
        summit_gap=summit_gap,
        logger=logger,
        rank=rank,
        candidate_input_prefix=candidate_input_prefix,
    )

    append_zscores(outdir, proc_chroms)


# ============================================================
# Combine outputs
# ============================================================

def _read_clustered_file(path):
    try:
        df = pd.read_csv(path, sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    return df


def combine_postprocessed_chroms(directory, prefix=None):
    """
    合并所有 clustered.candidates.{chrom}.bedpe。

    输出：
      {prefix}.postprocessed.all_candidates.bedpe
      {prefix}.postprocessed.summits.bedpe
    """
    if prefix is None:
        prefix = "pseudocell"

    pattern = os.path.join(directory, "clustered.candidates.*.bedpe")
    files = sorted(glob.glob(pattern))

    all_dfs = []

    for fp in files:
        df = _read_clustered_file(fp)
        if df.shape[0] == 0:
            continue
        all_dfs.append(df)

    all_candidates_file = os.path.join(directory, f"{prefix}.postprocessed.all_candidates.bedpe")
    summits_file = os.path.join(directory, f"{prefix}.postprocessed.summits.bedpe")

    if len(all_dfs) == 0:
        empty = pd.DataFrame(columns=BEDPE6_COLS)
        empty.to_csv(all_candidates_file, sep="\t", index=False)
        empty.to_csv(summits_file, sep="\t", index=False)
        return

    all_df = pd.concat(all_dfs, ignore_index=True)

    all_df.to_csv(all_candidates_file, sep="\t", index=False)

    if "is_summit" in all_df.columns:
        summits = all_df[all_df["is_summit"].astype(int) == 1].copy()
    else:
        summits = all_df.copy()

    summits.to_csv(summits_file, sep="\t", index=False)
