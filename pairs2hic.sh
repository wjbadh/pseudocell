#!/usr/bin/env bash
set -euo pipefail

IN_GLOB="/lanec4_home/guoq/bootstrap/data/GM12878/high_txt/*.txt.gz"
JUICER_JAR="/lanec4_home/guoq/juicer_tools_1.22.01.jar"
GENOME="/lanec4_home/guoq/SnapHiC_modified3/ext/hg38.chrom.sizes.txt"

OUT_HIC="/lanec4_home/guoq/DeepLoop_runs/high_aggregate.5kb.hic"
OUT_SHORT_GZ="/lanec4_home/guoq/DeepLoop_runs/high_aggregate.5kb.txt.gz"
RESOLUTION=5000

POS_ONE_BASED=1

COL_CHR1=2
COL_POS1=3
COL_CHR2=4
COL_POS2=5
COL_STRAND1=6
COL_STRAND2=7

OUT_DIR="$(dirname "$OUT_HIC")"
mkdir -p "$OUT_DIR"

if [[ ! -f "$JUICER_JAR" ]]; then
    echo "[ERROR] JUICER_JAR not found: $JUICER_JAR" >&2
    exit 1
fi

if [[ ! -f "$GENOME" ]]; then
    echo "[ERROR] GENOME chrom.sizes not found: $GENOME" >&2
    exit 1
fi

shopt -s nullglob
INPUT_FILES=( $IN_GLOB )
shopt -u nullglob

if [[ ${#INPUT_FILES[@]} -eq 0 ]]; then
    echo "[ERROR] No input files matched: $IN_GLOB" >&2
    exit 1
fi

echo "[INFO] input files: ${#INPUT_FILES[@]}"
echo "[INFO] output short: $OUT_SHORT_GZ"
echo "[INFO] output hic:   $OUT_HIC"
echo "[INFO] resolution:   $RESOLUTION"

TMP_TXT="$(mktemp "${OUT_DIR}/juicer_short.XXXXXX.txt")"
trap 'rm -f "$TMP_TXT"' EXIT

zcat "${INPUT_FILES[@]}" \
| awk -v OFS="\t" \
      -v one="${POS_ONE_BASED}" \
      -v c1="${COL_CHR1}" -v p1="${COL_POS1}" \
      -v c2="${COL_CHR2}" -v p2="${COL_POS2}" \
      -v s1="${COL_STRAND1}" -v s2="${COL_STRAND2}" '
  $0 ~ /^#/ {next}

  NF < c1 || NF < p1 || NF < c2 || NF < p2 || NF < s1 || NF < s2 {next}

  {
    chr1 = $c1
    pos1 = $p1 - one
    chr2 = $c2
    pos2 = $p2 - one

    if (chr1 ~ /^(chr)?(X|Y|M|MT)$/) next
    if (chr2 ~ /^(chr)?(X|Y|M|MT)$/) next

    if (pos1 < 0) pos1 = 0
    if (pos2 < 0) pos2 = 0

    if ($s1 == "True" || $s1 == "+" || $s1 == "1") {
      strand1 = 1
    } else {
      strand1 = 0
    }

    if ($s2 == "True" || $s2 == "+" || $s2 == "1") {
      strand2 = 1
    } else {
      strand2 = 0
    }

    print strand1, chr1, pos1, 0, strand2, chr2, pos2, 0
  }' \
| LC_ALL=C sort -T "$OUT_DIR" -k2,2 -k6,6 -k3,3n -k7,7n \
> "$TMP_TXT"

echo "[INFO] preview juicer short format:"
head "$TMP_TXT"

echo "[INFO] line count:"
wc -l "$TMP_TXT"

if [[ ! -s "$TMP_TXT" ]]; then
    echo "[ERROR] temporary short-format file is empty: $TMP_TXT" >&2
    exit 1
fi

gzip -c "$TMP_TXT" > "$OUT_SHORT_GZ"

echo "[INFO] running juicer pre..."

java -Xmx80g -jar "$JUICER_JAR" pre -q 0 -r "$RESOLUTION" \
    "$TMP_TXT" "$OUT_HIC" "$GENOME"

echo "[DONE] short: $OUT_SHORT_GZ"
echo "[DONE] hic:   $OUT_HIC"
