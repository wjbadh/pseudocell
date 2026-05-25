#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Raw-count pseudocell SnapHiC-like run
# ============================================================

SNAP_DIR="/disk2/guoq/pesudocell/pseudocell_modified"

PSEUDOCELL_RAW="/disk2/guoq/pesudocell/high_pseudocell/pesudocell_raw"
RAW_SC_BEDPE="/disk2/guoq/pesudocell/high_pseudocell/raw_sc_bedpe"

OUTDIR="/disk2/guoq/pesudocell/high_pseudocell/raw_count_loopcaller"
CHR_LENS="/lanec4_home/guoq/SnapHiC_modified4/ext/hg38.chrom.sizes.txt"
FILTER_FILE="/lanec4_home/guoq/SnapHiC_modified4/ext/hg38_filter_regions.txt"

PREFIX="GM_"

# 建议直接把 pseudocell_raw 当作 rwr 输入目录。
# 因为 snap.py 内部仍使用 outdir/rwr 这个变量名，
# 最简单做法是跳过 bin/rwr，并将 --indir 指向 pseudocell_raw。
#
# 注意：如果 snap.py 中 rwr_dir 固定为 outdir/rwr，
# 你需要先建立软链接：
#   mkdir -p "${OUTDIR}"
#   ln -sfn "${PSEUDOCELL_RAW}" "${OUTDIR}/rwr"
# 然后 --steps hic interaction postprocess

mkdir -p "${OUTDIR}"
ln -sfn "${PSEUDOCELL_RAW}" "${OUTDIR}/rwr"

python "${SNAP_DIR}/snap.py" \
    --indir "${PSEUDOCELL_RAW}" \
    --outdir "${OUTDIR}" \
    --raw-sc-dir "${RAW_SC_BEDPE}" \
    --chr-lens "${CHR_LENS}" \
    --genome hg38 \
    --max-chrom-number 22 \
    --prefix "${PREFIX}" \
    --binsize 10000 \
    --dist 2000000 \
    --low-cutoff 5000 \
    --outlier 0 \
    --outlier-threshold-multiplier 0.1 \
    --stat-threshold 0 \
    --fdr-threshold 0.1 \
    --local-lower-limit 2 \
    --local-upper-limit 5 \
    --candidate-lower-distance 100000 \
    --candidate-upper-distance 300000 \
    --postproc-gap-large 5 \
    --postproc-gap-small 2 \
    --circle-threshold-multiplier 1.33 \
    --donut-threshold-multiplier 1.33 \
    --lower-left-threshold-multiplier 1.33 \
    --horizontal-threshold-multiplier 1.2 \
    --vertical-threshold-multiplier 1.2 \
    --clustering-gap 1 \
    --summit-gap 30000 \
    --filter-file "${FILTER_FILE}" \
    --steps hic interaction postprocess \
    --no-hic \
    --no-cool \
    --parallel
