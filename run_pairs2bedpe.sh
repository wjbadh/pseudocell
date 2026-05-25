#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Input / output
# ============================================================

# 原始 allValidPairs / txt 文件所在目录
INPUT_DIR="/disk2/guoq/bootstrap/data/GM12878/high_txt"

# 默认输出目录
OUTDIR="/disk2/guoq/pesudocell/high_pseudocell/raw_sc_bedpe"

# 输入文件匹配模式
# 如果你的文件是 *.allValidPairs，可以改成 "*.allValidPairs"
# 如果是压缩文件，可以改成 "*.txt.gz" 或 "*.allValidPairs.gz"
PATTERN="*.txt.gz"

# 分辨率
RESOLUTION=10,000

# 并行进程数
JOBS=22

# Python 脚本路径
SCRIPT="/disk2/guoq/Pseudo_Cell/src/pairs2bedpe.py"

# 可选：染色体长度文件
# 如果没有，可以留空
CHROM_SIZES=""

# ============================================================
# allValidPairs 列号设置，1-based
# ============================================================
# 常见 Juicer allValidPairs 格式：
# readID chr1 pos1 strand1 chr2 pos2 strand2 frag1 frag2 mapq1 mapq2
#
# 对应：
# chr1-col = 2
# pos1-col = 3
# chr2-col = 5
# pos2-col = 6

CHR1_COL=2
POS1_COL=3
CHR2_COL=4
POS2_COL=5

# ============================================================
# Run
# ============================================================

mkdir -p "${OUTDIR}"

CMD=(
    python "${SCRIPT}"
    --input-dir "${INPUT_DIR}"
    --pattern "${PATTERN}"
    --outdir "${OUTDIR}"
    --resolution "${RESOLUTION}"
    --jobs "${JOBS}"
    --chr1-col "${CHR1_COL}"
    --pos1-col "${POS1_COL}"
    --chr2-col "${CHR2_COL}"
    --pos2-col "${POS2_COL}"
    --add-chr
)

# 如果提供了染色体长度文件，则启用
if [[ -n "${CHROM_SIZES}" ]]; then
    CMD+=(--chrom-sizes "${CHROM_SIZES}")
fi

echo "[INFO] Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"