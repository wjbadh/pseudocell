#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel converter from allValidPairs / raw pairs txt files to per-chromosome 7-column BEDPE files.

For each input file, the script treats it as one single cell.
It aggregates raw contacts into fixed-resolution bins and writes one BEDPE file per chromosome:

    <cell_name>.chr*.bedpe

Output columns, without header:

    chr1    x1    x2    chr2    y1    y2    count

Default output directory:

    /raw_sc_bedpe

Typical allValidPairs columns from Juicer-like format may be:

    readID  chr1  pos1  strand1  chr2  pos2  strand2  frag1  frag2  mapq1  mapq2

For that format, use:
    --chr1-col 2 --pos1-col 3 --chr2-col 5 --pos2-col 6

Column indices are 1-based by default.
"""

import argparse
import gzip
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def open_maybe_gzip(path):
    """
    Open plain text or gzip-compressed text file.
    """
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def infer_cell_name(path, strip_suffixes=None):
    """
    Infer cell name from file basename.
    """
    name = Path(path).name

    if strip_suffixes:
        for suf in strip_suffixes:
            if name.endswith(suf):
                name = name[: -len(suf)]
                return name

    # default suffix removal
    for suf in [
        ".allValidPairs.gz",
        ".allValidPairs",
        ".validPairs.gz",
        ".validPairs",
        ".pairs.gz",
        ".pairs",
        ".txt.gz",
        ".txt",
        ".tsv.gz",
        ".tsv",
        ".bedpe.gz",
        ".bedpe",
        ".gz",
    ]:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break

    return name


def normalize_chrom(chrom, add_chr=False, remove_chr=False):
    """
    Normalize chromosome naming if requested.
    """
    chrom = str(chrom)

    if remove_chr and chrom.startswith("chr"):
        chrom = chrom[3:]

    if add_chr and not chrom.startswith("chr"):
        chrom = "chr" + chrom

    return chrom


def chrom_sort_key(chrom):
    """
    Natural chromosome sort: chr1, chr2, ..., chr10, chrX, chrY, chrM.
    """
    c = chrom
    if c.startswith("chr"):
        c = c[3:]

    order = {
        "X": 23,
        "Y": 24,
        "M": 25,
        "MT": 25,
    }

    if c.isdigit():
        return (0, int(c))

    return (1, order.get(c, 1000), c)


def parse_chrom_sizes(chrom_sizes, add_chr=False, remove_chr=False):
    """
    Read chromosome sizes file.

    Expected format:
        chr1    248956422
        chr2    242193529
        ...

    Returns:
        dict: chrom -> length
    """
    if chrom_sizes is None:
        return None

    sizes = {}
    with open(chrom_sizes, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split()
            if len(fields) < 2:
                continue
            chrom = normalize_chrom(fields[0], add_chr=add_chr, remove_chr=remove_chr)
            try:
                size = int(fields[1])
            except ValueError:
                continue
            sizes[chrom] = size

    return sizes


def parse_chrom_list(chroms):
    """
    Parse comma-separated chromosome list.
    """
    if chroms is None or chroms.strip() == "":
        return None

    return set(x.strip() for x in chroms.split(",") if x.strip())


def should_skip_line(line, comment_prefixes):
    """
    Skip empty, comment, metadata, or likely header lines.
    """
    if not line.strip():
        return True

    stripped = line.lstrip()

    for prefix in comment_prefixes:
        if stripped.startswith(prefix):
            return True

    # Common pairs metadata lines
    if stripped.startswith("#"):
        return True

    return False


def process_one_file(args_tuple):
    """
    Process one allValidPairs file.

    Returns summary dict.
    """
    (
        input_path,
        outdir,
        binsize,
        chr1_col,
        pos1_col,
        chr2_col,
        pos2_col,
        min_distance,
        max_distance,
        keep_trans,
        chrom_filter,
        chrom_sizes,
        add_chr,
        remove_chr,
        zero_based_pos,
        strip_suffixes,
        comment_prefixes,
        chunks_write_mode,
    ) = args_tuple

    input_path = str(input_path)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cell_name = infer_cell_name(input_path, strip_suffixes=strip_suffixes)

    # Convert to 0-based Python indices
    chr1_idx = chr1_col - 1
    pos1_idx = pos1_col - 1
    chr2_idx = chr2_col - 1
    pos2_idx = pos2_col - 1

    max_needed_idx = max(chr1_idx, pos1_idx, chr2_idx, pos2_idx)

    # data[chrom][(bin1, bin2)] = count
    # For cis contacts, one output file per chromosome.
    # For trans contacts, the output chromosome is encoded as chr1__chr2 unless keep_trans=False.
    data = defaultdict(lambda: defaultdict(int))

    total_lines = 0
    used_lines = 0
    skipped_lines = 0
    skipped_malformed = 0
    skipped_chrom = 0
    skipped_distance = 0
    skipped_trans = 0

    with open_maybe_gzip(input_path) as f:
        for line in f:
            total_lines += 1

            if should_skip_line(line, comment_prefixes):
                skipped_lines += 1
                continue

            fields = line.rstrip("\n").split()

            if len(fields) <= max_needed_idx:
                skipped_malformed += 1
                continue

            raw_chr1 = fields[chr1_idx]
            raw_chr2 = fields[chr2_idx]

            chrom1 = normalize_chrom(raw_chr1, add_chr=add_chr, remove_chr=remove_chr)
            chrom2 = normalize_chrom(raw_chr2, add_chr=add_chr, remove_chr=remove_chr)

            if chrom_filter is not None:
                if chrom1 not in chrom_filter or chrom2 not in chrom_filter:
                    skipped_chrom += 1
                    continue

            try:
                pos1 = int(float(fields[pos1_idx]))
                pos2 = int(float(fields[pos2_idx]))
            except ValueError:
                skipped_malformed += 1
                continue

            # If input position is 1-based, convert to 0-based coordinate for bin assignment.
            # Most allValidPairs coordinates are genomic positions; binning is usually floor(pos / binsize).
            # If you want strict 1-based to 0-based conversion, keep default zero_based_pos=False.
            if not zero_based_pos:
                pos1 -= 1
                pos2 -= 1

            if pos1 < 0 or pos2 < 0:
                skipped_malformed += 1
                continue

            bin1 = pos1 // binsize
            bin2 = pos2 // binsize

            if chrom1 == chrom2:
                distance = abs(bin2 - bin1) * binsize

                if min_distance is not None and distance < min_distance:
                    skipped_distance += 1
                    continue

                if max_distance is not None and distance > max_distance:
                    skipped_distance += 1
                    continue

                # Standardize order inside the same chromosome.
                if bin1 > bin2:
                    bin1, bin2 = bin2, bin1

                data[chrom1][(bin1, bin2)] += 1
                used_lines += 1

            else:
                if not keep_trans:
                    skipped_trans += 1
                    continue

                # For trans contacts, impose deterministic chromosome order.
                key1 = chrom_sort_key(chrom1)
                key2 = chrom_sort_key(chrom2)

                if key2 < key1:
                    chrom1, chrom2 = chrom2, chrom1
                    bin1, bin2 = bin2, bin1

                trans_key = f"{chrom1}__{chrom2}"
                data[trans_key][(chrom1, bin1, chrom2, bin2)] += 1
                used_lines += 1

    written_files = []

    for chrom_key in sorted(data.keys(), key=chrom_sort_key):
        if "__" not in chrom_key:
            chrom = chrom_key
            out_path = outdir / f"{cell_name}.{chrom}.bedpe"

            with open(out_path, chunks_write_mode) as out:
                for (bin1, bin2), count in sorted(data[chrom].items()):
                    x1 = bin1 * binsize
                    x2 = x1 + binsize
                    y1 = bin2 * binsize
                    y2 = y1 + binsize

                    if chrom_sizes is not None and chrom in chrom_sizes:
                        if x1 >= chrom_sizes[chrom] or y1 >= chrom_sizes[chrom]:
                            continue
                        x2 = min(x2, chrom_sizes[chrom])
                        y2 = min(y2, chrom_sizes[chrom])

                    out.write(
                        f"{chrom}\t{x1}\t{x2}\t{chrom}\t{y1}\t{y2}\t{count}\n"
                    )

            written_files.append(str(out_path))

        else:
            chrom_pair = chrom_key
            out_path = outdir / f"{cell_name}.{chrom_pair}.bedpe"

            with open(out_path, chunks_write_mode) as out:
                for (chrom1, bin1, chrom2, bin2), count in sorted(data[chrom_pair].items()):
                    x1 = bin1 * binsize
                    x2 = x1 + binsize
                    y1 = bin2 * binsize
                    y2 = y1 + binsize

                    if chrom_sizes is not None:
                        if chrom1 in chrom_sizes:
                            if x1 >= chrom_sizes[chrom1]:
                                continue
                            x2 = min(x2, chrom_sizes[chrom1])
                        if chrom2 in chrom_sizes:
                            if y1 >= chrom_sizes[chrom2]:
                                continue
                            y2 = min(y2, chrom_sizes[chrom2])

                    out.write(
                        f"{chrom1}\t{x1}\t{x2}\t{chrom2}\t{y1}\t{y2}\t{count}\n"
                    )

            written_files.append(str(out_path))

    return {
        "input": input_path,
        "cell": cell_name,
        "total_lines": total_lines,
        "used_lines": used_lines,
        "skipped_lines": skipped_lines,
        "skipped_malformed": skipped_malformed,
        "skipped_chrom": skipped_chrom,
        "skipped_distance": skipped_distance,
        "skipped_trans": skipped_trans,
        "n_output_files": len(written_files),
    }


def collect_input_files(input_dir, pattern, input_list):
    """
    Collect input files either from input directory or input list.
    """
    files = []

    if input_list is not None:
        with open(input_list, "r") as f:
            for line in f:
                path = line.strip()
                if path:
                    files.append(path)

    if input_dir is not None:
        input_dir = Path(input_dir)
        files.extend(str(p) for p in sorted(input_dir.glob(pattern)))

    # Deduplicate while preserving order
    seen = set()
    unique_files = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique_files.append(f)

    return unique_files


def main():
    parser = argparse.ArgumentParser(
        description="Convert allValidPairs/raw pairs txt files to per-chromosome aggregated 7-column BEDPE files."
    )

    parser.add_argument(
        "-i", "--input-dir",
        default=None,
        help="Directory containing allValidPairs/txt files."
    )
    parser.add_argument(
        "--input-list",
        default=None,
        help="Text file containing input paths, one file per line."
    )
    parser.add_argument(
        "-p", "--pattern",
        default="*.txt",
        help="Input file glob pattern under --input-dir. Default: *.txt"
    )
    parser.add_argument(
        "-o", "--outdir",
        default="/raw_sc_bedpe",
        help="Output directory. Default: /raw_sc_bedpe"
    )
    parser.add_argument(
        "-r", "--resolution",
        "--binsize",
        dest="binsize",
        type=int,
        default=10000,
        help="Bin size / resolution in bp. Default: 10000"
    )
    parser.add_argument(
        "-j", "--jobs",
        type=int,
        default=4,
        help="Number of parallel worker processes. Default: 4"
    )

    # 1-based columns by default
    parser.add_argument(
        "--chr1-col",
        type=int,
        default=2,
        help="1-based column index for chromosome 1. Default: 2"
    )
    parser.add_argument(
        "--pos1-col",
        type=int,
        default=3,
        help="1-based column index for position 1. Default: 3"
    )
    parser.add_argument(
        "--chr2-col",
        type=int,
        default=5,
        help="1-based column index for chromosome 2. Default: 5"
    )
    parser.add_argument(
        "--pos2-col",
        type=int,
        default=6,
        help="1-based column index for position 2. Default: 6"
    )

    parser.add_argument(
        "--zero-based-pos",
        action="store_true",
        help="Set this if input positions are already 0-based. Default assumes 1-based positions and subtracts 1 before binning."
    )

    parser.add_argument(
        "--chroms",
        default=None,
        help="Comma-separated chromosome list to keep, e.g. chr1,chr2,chrX. Default: keep all chromosomes."
    )
    parser.add_argument(
        "--chrom-sizes",
        default=None,
        help="Optional chromosome sizes file. If provided, bin end positions will be clipped to chromosome length."
    )
    parser.add_argument(
        "--add-chr",
        action="store_true",
        help="Add 'chr' prefix to chromosome names if absent."
    )
    parser.add_argument(
        "--remove-chr",
        action="store_true",
        help="Remove 'chr' prefix from chromosome names if present."
    )

    parser.add_argument(
        "--min-distance",
        type=int,
        default=None,
        help="Minimum cis genomic distance in bp to keep. Default: no lower limit."
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=None,
        help="Maximum cis genomic distance in bp to keep. Default: no upper limit."
    )
    parser.add_argument(
        "--keep-trans",
        action="store_true",
        help="Also keep trans contacts. Default: only cis contacts are output per chromosome."
    )

    parser.add_argument(
        "--strip-suffix",
        action="append",
        default=None,
        help="Suffix to strip from basename when inferring cell name. Can be used multiple times."
    )
    parser.add_argument(
        "--comment-prefix",
        action="append",
        default=None,
        help="Line prefix to skip. Can be used multiple times. Default skips # lines."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files. Default also overwrites, because each file is written once per cell/chrom."
    )

    args = parser.parse_args()

    if args.input_dir is None and args.input_list is None:
        parser.error("You must provide --input-dir or --input-list.")

    if args.add_chr and args.remove_chr:
        parser.error("--add-chr and --remove-chr cannot be used together.")

    if args.binsize <= 0:
        parser.error("--resolution / --binsize must be positive.")

    files = collect_input_files(args.input_dir, args.pattern, args.input_list)

    if len(files) == 0:
        print("[ERROR] No input files found.", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    chrom_filter = parse_chrom_list(args.chroms)
    chrom_sizes = parse_chrom_sizes(
        args.chrom_sizes,
        add_chr=args.add_chr,
        remove_chr=args.remove_chr,
    )

    if chrom_filter is not None:
        chrom_filter = {
            normalize_chrom(x, add_chr=args.add_chr, remove_chr=args.remove_chr)
            for x in chrom_filter
        }

    comment_prefixes = args.comment_prefix
    if comment_prefixes is None:
        comment_prefixes = ["#"]

    task_args = []
    for input_path in files:
        task_args.append(
            (
                input_path,
                str(outdir),
                args.binsize,
                args.chr1_col,
                args.pos1_col,
                args.chr2_col,
                args.pos2_col,
                args.min_distance,
                args.max_distance,
                args.keep_trans,
                chrom_filter,
                chrom_sizes,
                args.add_chr,
                args.remove_chr,
                args.zero_based_pos,
                args.strip_suffix,
                comment_prefixes,
                "w",
            )
        )

    print(f"[INFO] Number of input files: {len(files)}")
    print(f"[INFO] Output directory: {outdir}")
    print(f"[INFO] Bin size: {args.binsize}")
    print(f"[INFO] Jobs: {args.jobs}")
    print(
        f"[INFO] Columns: chr1={args.chr1_col}, pos1={args.pos1_col}, "
        f"chr2={args.chr2_col}, pos2={args.pos2_col}"
    )

    summaries = []

    if args.jobs == 1:
        for t in task_args:
            summary = process_one_file(t)
            summaries.append(summary)
            print(
                f"[DONE] {summary['cell']}: "
                f"used={summary['used_lines']}, "
                f"outputs={summary['n_output_files']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            future_to_path = {
                executor.submit(process_one_file, t): t[0]
                for t in task_args
            }

            for future in as_completed(future_to_path):
                input_path = future_to_path[future]
                try:
                    summary = future.result()
                    summaries.append(summary)
                    print(
                        f"[DONE] {summary['cell']}: "
                        f"used={summary['used_lines']}, "
                        f"outputs={summary['n_output_files']}"
                    )
                except Exception as e:
                    print(f"[ERROR] Failed processing {input_path}: {e}", file=sys.stderr)
                    raise

    summary_path = outdir / "conversion_summary.tsv"
    with open(summary_path, "w") as out:
        out.write(
            "input\tcell\ttotal_lines\tused_lines\tskipped_lines\t"
            "skipped_malformed\tskipped_chrom\tskipped_distance\t"
            "skipped_trans\tn_output_files\n"
        )
        for s in sorted(summaries, key=lambda x: x["cell"]):
            out.write(
                f"{s['input']}\t{s['cell']}\t{s['total_lines']}\t{s['used_lines']}\t"
                f"{s['skipped_lines']}\t{s['skipped_malformed']}\t"
                f"{s['skipped_chrom']}\t{s['skipped_distance']}\t"
                f"{s['skipped_trans']}\t{s['n_output_files']}\n"
            )

    print(f"[INFO] Summary written to: {summary_path}")
    print("[INFO] Finished.")


if __name__ == "__main__":
    main()