#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import numpy as np
import pandas as pd


try:
    from src.prefilter_support import filter_candidate_dir_by_pseudocell_support
except ImportError:
    try:
        from prefilter_support import filter_candidate_dir_by_pseudocell_support
    except ImportError:
        filter_candidate_dir_by_pseudocell_support = None


try:
    import hicstraw
except ImportError:
    hicstraw = None


BEDPE6_COLS = ["chr1", "x1", "x2", "chr2", "y1", "y2"]

SIGNIF_COLS = [
    "chr1", "x1", "x2", "chr2", "y1", "y2",
    "support_cell_count",
    "case_avg",
    "control_avg",
    "pvalue",
    "stat",
    "fdr_dist",
    "fdr_chrom",
]


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


def get_proc_chroms(chrom_lens, rank, n_proc):
    chrom_list = [(k, chrom_lens[k]) for k in list(chrom_lens.keys())]
    chrom_list.sort(key=lambda x: x[1], reverse=True)

    chrom_names = [x[0] for x in chrom_list]
    indices = list(range(rank, len(chrom_names), n_proc))
    return [chrom_names[i] for i in indices]


def _safe_numeric(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def _ensure_bedpe_numeric(df):
    for c in ["x1", "x2", "y1", "y2"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(np.int64)
    return df


def count_pseudocell_files_for_chrom(pseudocell_dir, chrom, pattern="*.bedpe*"):
    if pseudocell_dir is None:
        raise ValueError("[postprocess] pseudocell_dir is required to count num_cells")

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
        files = [
            fp for fp in glob.glob(os.path.join(pseudocell_dir, f"*{chrom}*.bedpe*"))
            if os.path.isfile(fp)
        ]

    return len(files)


def read_filter_regions(filter_file, binsize):
    bad = set()

    if filter_file is None or not os.path.exists(filter_file):
        return bad

    try:
        df = pd.read_csv(filter_file, sep="\t", header=None, comment="#")
    except pd.errors.EmptyDataError:
        return bad

    if df.shape[0] == 0:
        return bad

    if df.shape[1] >= 3:
        for row in df.itertuples(index=False):
            chrom = str(row[0])
            start = int(row[1])
            end = int(row[2])

            b1 = start // int(binsize)
            b2 = max(start, end - 1) // int(binsize)

            for b in range(b1, b2 + 1):
                bad.add((chrom, b))

    elif df.shape[1] >= 2:
        for row in df.itertuples(index=False):
            bad.add((str(row[0]), int(row[1])))

    return bad


def apply_filter_regions(df, filter_regions, binsize):
    if df.shape[0] == 0 or len(filter_regions) == 0:
        return df

    b1 = (df["x1"].astype(np.int64) // int(binsize)).to_numpy()
    b2 = (df["y1"].astype(np.int64) // int(binsize)).to_numpy()
    chr1 = df["chr1"].astype(str).to_numpy()
    chr2 = df["chr2"].astype(str).to_numpy()

    keep = np.ones(df.shape[0], dtype=bool)

    for idx in range(df.shape[0]):
        if (chr1[idx], int(b1[idx])) in filter_regions:
            keep[idx] = False
        elif (chr2[idx], int(b2[idx])) in filter_regions:
            keep[idx] = False

    return df.loc[keep].copy()


def load_hic_contact_map(
    hic_path,
    chrom,
    binsize,
    normalization="NONE",
    unit="BP",
    logger=None,
    rank=0,
):
    if hicstraw is None:
        raise ImportError(
            "hicstraw is required for reading .hic files. "
            "Install with: pip install hic-straw"
        )

    if not os.path.exists(hic_path):
        raise FileNotFoundError(f"[postprocess] pseudobulk .hic not found: {hic_path}")

    _log(
        logger,
        f"[postprocess] [{chrom}] loading .hic contact map from {hic_path}",
        rank=rank,
        v=2,
    )

    try:
        records = hicstraw.straw(
            "observed",
            str(normalization),
            str(hic_path),
            str(chrom),
            str(chrom),
            str(unit),
            int(binsize),
        )
    except Exception as e:
        raise RuntimeError(
            f"[postprocess] failed to read .hic for {chrom} with hicstraw. "
            f"hic={hic_path}, normalization={normalization}, binsize={binsize}. "
            f"Original error: {e}"
        )

    contact = {}

    for rec in records:
        i = int(rec.binX) // int(binsize)
        j = int(rec.binY) // int(binsize)

        if i == j:
            continue

        if i > j:
            i, j = j, i

        v = float(rec.counts)

        if v != 0:
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


def compute_structural_backgrounds_for_pair(contact_map, i, j, gap_large, gap_small):
    """
    计算 pseudobulk .hic 的五类结构背景。

    当前逻辑：
    1. 背景窗口内所有合法坐标都参与背景均值计算。
    2. 合法坐标是指：bin 坐标非负，并且位于上三角 a < b。
    3. 如果合法坐标在 .hic contact_map 中缺失，get_contact_value() 返回 0.0，
       该 0.0 会被加入 vals，并计入分母。
    4. NaN/非法坐标不参与分母。
    5. 如果某一类背景没有任何合法坐标，则该背景均值设为 0.0。
    """
    i = int(i)
    j = int(j)

    if i > j:
        i, j = j, i

    gl = int(gap_large)
    gs = int(gap_small)

    circle_vals = []
    donut_vals = []
    lower_left_vals = []
    horizontal_vals = []
    vertical_vals = []

    def append_if_valid(vals, a, b):
        """
        合法坐标缺失 contact 时按 0.0 处理并纳入分母；
        非法坐标不纳入分母。
        """
        a = int(a)
        b = int(b)

        if a < 0 or b < 0 or a >= b:
            return

        v = get_contact_value(contact_map, a, b)

        if np.isfinite(v):
            vals.append(float(v))

    # circle/donut：中心周围的大方框去掉中心小方框。
    # 这里保留原脚本含义：circle 与 donut 使用同一组背景坐标。
    for di in range(-gl, gl + 1):
        for dj in range(-gl, gl + 1):
            if abs(di) <= gs and abs(dj) <= gs:
                continue

            a = i + di
            b = j + dj

            before_len = len(circle_vals)
            append_if_valid(circle_vals, a, b)
            if len(circle_vals) > before_len:
                donut_vals.append(circle_vals[-1])

    # lower_left：沿左下方向的对角背景。
    for d in range(gs + 1, gl + 1):
        append_if_valid(lower_left_vals, i - d, j - d)

    # horizontal：固定左端点，横向移动右端点。
    for dj in range(-gl, gl + 1):
        if abs(dj) <= gs:
            continue
        append_if_valid(horizontal_vals, i, j + dj)

    # vertical：移动左端点，固定右端点。
    for di in range(-gl, gl + 1):
        if abs(di) <= gs:
            continue
        append_if_valid(vertical_vals, i + di, j)

    def mean_or_zero(vals):
        return float(np.mean(vals)) if len(vals) > 0 else 0.0

    return (
        mean_or_zero(circle_vals),
        mean_or_zero(donut_vals),
        mean_or_zero(lower_left_vals),
        mean_or_zero(horizontal_vals),
        mean_or_zero(vertical_vals),
    )

def add_structural_background_columns_from_hic(
    df,
    contact_map,
    binsize,
    gap_large,
    gap_small,
):
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

    n = df.shape[0]

    centers = np.zeros(n, dtype=float)
    circles = np.zeros(n, dtype=float)
    donuts = np.zeros(n, dtype=float)
    lower_lefts = np.zeros(n, dtype=float)
    horizontals = np.zeros(n, dtype=float)
    verticals = np.zeros(n, dtype=float)

    x_bins = (df["x1"].astype(np.int64) // int(binsize)).to_numpy()
    y_bins = (df["y1"].astype(np.int64) // int(binsize)).to_numpy()

    for idx in range(n):
        i = int(x_bins[idx])
        j = int(y_bins[idx])

        if i > j:
            i, j = j, i

        centers[idx] = get_contact_value(contact_map, i, j)

        circle, donut, lower_left, horizontal, vertical = compute_structural_backgrounds_for_pair(
            contact_map=contact_map,
            i=i,
            j=j,
            gap_large=gap_large,
            gap_small=gap_small,
        )

        circles[idx] = circle
        donuts[idx] = donut
        lower_lefts[idx] = lower_left
        horizontals[idx] = horizontal
        verticals[idx] = vertical

    df["pseudobulk_center"] = centers
    df["circle"] = circles
    df["donut"] = donuts
    df["lower_left"] = lower_lefts
    df["horizontal"] = horizontals
    df["vertical"] = verticals

    return df


def read_significance_file(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=SIGNIF_COLS)

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
        return pd.DataFrame(columns=SIGNIF_COLS)

    if df.shape[0] == 0:
        return pd.DataFrame(columns=SIGNIF_COLS)

    first_vals = [str(x) for x in df.iloc[0, : min(df.shape[1], 6)].tolist()]
    has_header = any(x in {"chr1", "x1", "x2", "chr2", "y1", "y2"} for x in first_vals)

    if has_header:
        df.columns = df.iloc[0].astype(str).tolist()
        df = df.iloc[1:, :].copy()
    else:
        if df.shape[1] < len(SIGNIF_COLS):
            raise ValueError(
                f"[postprocess] {path} has {df.shape[1]} columns, "
                f"expected at least {len(SIGNIF_COLS)}"
            )
        df = df.iloc[:, :len(SIGNIF_COLS)].copy()
        df.columns = SIGNIF_COLS

    df = _ensure_bedpe_numeric(df)
    return df.reset_index(drop=True)


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
    num_cells,
):
    if df.shape[0] == 0:
        return df

    required_cols = [
        "chr1", "x1", "x2", "chr2", "y1", "y2",
        "support_cell_count",
        "pvalue",
        "stat",
        "fdr_dist",
        "pseudobulk_center",
        "circle",
        "donut",
        "lower_left",
        "horizontal",
        "vertical",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if len(missing) > 0:
        raise KeyError(
            "[postprocess] missing required columns: " + ",".join(missing)
        )

    if num_cells is None or int(num_cells) <= 0:
        raise ValueError("[postprocess] num_cells must be positive")

    df = df.copy()

    df["support_cell_count"] = _safe_numeric(df["support_cell_count"], default=0.0)
    df["pvalue"] = _safe_numeric(df["pvalue"], default=1.0)
    df["stat"] = _safe_numeric(df["stat"], default=0.0)
    df["fdr_dist"] = _safe_numeric(df["fdr_dist"], default=1.0)

    for c in ["pseudobulk_center", "circle", "donut", "lower_left", "horizontal", "vertical"]:
        df[c] = _safe_numeric(df[c], default=0.0)

    dist_bp = (df["y1"].astype(np.int64) - df["x1"].astype(np.int64)).abs()
    center = df["pseudobulk_center"]

    keep = (
        (dist_bp >= float(candidate_lower_thresh)) &
        (dist_bp <= float(candidate_upper_thresh)) &
        (center > 0) &
        (df["stat"] >= float(stat_threshold)) &
        (df["fdr_dist"] <= float(fdr_thresh)) &
        (df["support_cell_count"] > float(min_support_fraction) * float(num_cells)) &
        (center > df["circle"] * float(circle_threshold_mult)) &
        (center > df["donut"] * float(donut_threshold_mult)) &
        (center > df["lower_left"] * float(lower_left_threshold_mult)) &
        (center > df["horizontal"] * float(horizontal_threshold_mult)) &
        (center > df["vertical"] * float(vertical_threshold_mult))
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
    pseudocell_dir=None,
    pseudocell_pattern="*.bedpe*",
):
    os.makedirs(outdir, exist_ok=True)

    if pseudobulk_hic is None:
        raise ValueError("[postprocess] --pseudobulk-hic is required")

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
        sig = sig[(sig["chr1"].astype(str) == chrom) & (sig["chr2"].astype(str) == chrom)].copy()

        if sig.shape[0] == 0:
            sig.to_csv(outfile_nofilter, sep="\t", index=False)
            sig.to_csv(outfile_candidates, sep="\t", index=False)
            continue

        num_cells = count_pseudocell_files_for_chrom(
            pseudocell_dir=pseudocell_dir,
            chrom=chrom,
            pattern=pseudocell_pattern,
        )

        if num_cells <= 0:
            raise ValueError(
                f"[postprocess] cannot determine num_cells for {chrom} from {pseudocell_dir}"
            )

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
            num_cells=num_cells,
        )

        cand.to_csv(outfile_candidates, sep="\t", index=False)

        _log(
            logger,
            f"\tprocessor {rank}: {chrom} candidates {sig.shape[0]} -> {cand.shape[0]} "
            f"(num_cells={num_cells})",
            rank=rank,
            v=2,
        )


def _get_bin_pair(row, binsize):
    i = int(row["x1"]) // int(binsize)
    j = int(row["y1"]) // int(binsize)

    if i > j:
        i, j = j, i

    return i, j


def _choose_summit(cluster_df):
    df = cluster_df.copy()

    df["fdr_dist"] = _safe_numeric(df["fdr_dist"], default=1.0)
    df["stat"] = _safe_numeric(df["stat"], default=0.0)
    df["pseudobulk_center"] = _safe_numeric(df["pseudobulk_center"], default=0.0)

    return df.sort_values(
        ["fdr_dist", "stat", "pseudobulk_center"],
        ascending=[True, False, False],
    ).iloc[[0]].copy()


def _rank_summits_for_gap_filter(summit_df):
    df = summit_df.copy()

    df["fdr_dist"] = _safe_numeric(df["fdr_dist"], default=1.0)
    df["stat"] = _safe_numeric(df["stat"], default=0.0)
    df["pseudobulk_center"] = _safe_numeric(df["pseudobulk_center"], default=0.0)

    return df.sort_values(
        ["fdr_dist", "stat", "pseudobulk_center"],
        ascending=[True, False, False],
    ).index.tolist()


def cluster_one_chrom(df, binsize, clustering_gap, summit_gap):
    if df.shape[0] == 0:
        out = df.copy()
        out["cluster_id"] = []
        out["cluster_size"] = []
        out["is_summit"] = []
        return out

    df = df.copy().reset_index(drop=True)

    pairs = np.asarray(
        [_get_bin_pair(row, binsize) for _, row in df.iterrows()],
        dtype=np.int64,
    )

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

    # df = df[df["cluster_size"] > 1].copy().reset_index(drop=True)

    if df.shape[0] == 0:
        df["is_summit"] = []
        return df

    df["is_summit"] = 0

    summit_indices = []

    for _, sub in df.groupby("cluster_id", sort=False):
        summit = _choose_summit(sub)
        summit_indices.append(summit.index[0])

    df.loc[summit_indices, "is_summit"] = 1

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
            f"\tprocessor {rank}: {chrom} clustered candidates={clustered.shape[0]}, "
            f"summits={n_summit}",
            rank=rank,
            v=2,
        )


def append_zscores(outdir, proc_chroms):
    return


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
        pseudocell_dir=pseudocell_dir,
        pseudocell_pattern=pseudocell_pattern,
    )

    if support_filter:
        if filter_candidate_dir_by_pseudocell_support is None:
            raise ImportError(
                "[support] support_filter=True but "
                "filter_candidate_dir_by_pseudocell_support could not be imported"
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


def _read_clustered_file(path):
    try:
        return pd.read_csv(path, sep="\t", low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def combine_postprocessed_chroms(directory, prefix=None):
    if prefix is None:
        prefix = "pseudocell"

    files = sorted(glob.glob(os.path.join(directory, "clustered.candidates.*.bedpe")))

    all_dfs = []

    for fp in files:
        df = _read_clustered_file(fp)
        if df.shape[0] > 0:
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