#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
import subprocess
from collections import Counter, defaultdict

import h5py
import numpy as np
import pandas as pd

try:
    import cooler
except Exception:
    cooler = None

try:
    import hicstraw
except Exception:
    hicstraw = None


BEDPE6_COLS = ["chr1", "x1", "x2", "chr2", "y1", "y2"]
BEDPE7_COLS = BEDPE6_COLS + ["value"]
AUTO_MIN_CONTACT_MULTIPLIER = 2.0


def _log(logger, message, rank=0, level=1):
    if logger is None:
        return
    logger.write(
        message,
        verbose_level=level,
        append_time=False,
        allow_all_ranks=True,
    )


def get_proc_chroms(chrom_lens, rank, n_proc):
    chroms = sorted(chrom_lens, key=chrom_lens.get, reverse=True)
    return chroms[rank::n_proc]


def _read_bedpe7(path):
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            compression="gzip" if str(path).endswith(".gz") else None,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=BEDPE7_COLS)

    if df.empty:
        return pd.DataFrame(columns=BEDPE7_COLS)
    if df.shape[1] < 7:
        raise ValueError(f"[combine_cells] {path} has fewer than 7 columns: {df.shape[1]}")

    df = df.iloc[:, :7].copy()
    df.columns = BEDPE7_COLS
    df["chr1"] = df["chr1"].astype(str)
    df["chr2"] = df["chr2"].astype(str)

    for col in ["x1", "x2", "y1", "y2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    df = df.dropna(subset=["x1", "x2", "y1", "y2", "value"]).copy()
    for col in ["x1", "x2", "y1", "y2"]:
        df[col] = df[col].astype(np.int64)
    df["value"] = df["value"].astype(float)
    return df


def _standardize_cis(df, chrom, binsize):
    if df.empty:
        return pd.DataFrame(columns=BEDPE7_COLS)

    df = df[(df["chr1"] == chrom) & (df["chr2"] == chrom)].copy()
    if df.empty:
        return pd.DataFrame(columns=BEDPE7_COLS)

    binsize = int(binsize)
    a = (df["x1"].to_numpy(dtype=np.int64) // binsize) * binsize
    b = (df["y1"].to_numpy(dtype=np.int64) // binsize) * binsize
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)

    df["x1"] = lo
    df["x2"] = lo + binsize
    df["y1"] = hi
    df["y2"] = hi + binsize
    df = df[df["x1"] < df["y1"]]

    if df.empty:
        return pd.DataFrame(columns=BEDPE7_COLS)

    return df.groupby(BEDPE6_COLS, as_index=False, sort=False)["value"].sum()


def _find_input_files(indir, chrom, pattern="*.bedpe*"):
    files = []
    for path in glob.glob(os.path.join(indir, pattern)):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        if (
            f".{chrom}." in name
            or name.endswith(f".{chrom}.bedpe")
            or name.endswith(f".{chrom}.bedpe.gz")
            or f"_{chrom}_" in name
            or f"_{chrom}." in name
            or f".{chrom}_" in name
        ):
            files.append(path)

    if not files:
        files = [
            path
            for path in glob.glob(os.path.join(indir, f"*{chrom}*.bedpe*"))
            if os.path.isfile(path)
        ]
    return sorted(files)


def _chrom_name_candidates(chrom):
    chrom = str(chrom)
    candidates = [chrom, chrom[3:] if chrom.startswith("chr") else f"chr{chrom}"]
    return list(dict.fromkeys(candidates))


def _straw_records(hic_path, chrom, binsize, normalization="NONE"):
    if hicstraw is None:
        raise ImportError(
            "[combine_cells] hicstraw is required. Install it with `pip install hic-straw`."
        )
    if not os.path.isfile(hic_path):
        raise FileNotFoundError(f"[combine_cells] pseudobulk .hic not found: {hic_path}")

    errors = []
    for hic_chrom in _chrom_name_candidates(chrom):
        try:
            records = hicstraw.straw(
                "observed",
                str(normalization),
                str(hic_path),
                hic_chrom,
                hic_chrom,
                "BP",
                int(binsize),
            )
            if records:
                return records, hic_chrom
        except Exception as exc:
            errors.append(f"{hic_chrom}: {exc}")

    if errors:
        raise RuntimeError(
            f"[combine_cells] failed to read {chrom} from {hic_path}; "
            f"normalization={normalization}, binsize={binsize}; "
            f"errors={' | '.join(errors)}"
        )
    return [], str(chrom)


def _read_hic_interactions(
    pseudobulk_hic,
    chrom,
    binsize,
    hic_normalization="NONE",
    min_distance=None,
    max_distance=None,
):
    """Read nonzero cis interactions and aggregate duplicate bin pairs."""
    binsize = int(binsize)
    min_distance = None if min_distance is None else int(min_distance)
    max_distance = None if max_distance is None else int(max_distance)

    records, hic_chrom = _straw_records(
        hic_path=pseudobulk_hic,
        chrom=chrom,
        binsize=binsize,
        normalization=hic_normalization,
    )

    rows = []
    for record in records:
        x = (int(record.binX) // binsize) * binsize
        y = (int(record.binY) // binsize) * binsize
        if x == y:
            continue

        lo, hi = sorted((x, y))
        distance_bp = hi - lo
        if min_distance is not None and distance_bp < min_distance:
            continue
        if max_distance is not None and distance_bp > max_distance:
            continue

        count = float(record.counts)
        if not np.isfinite(count) or count <= 0:
            continue
        rows.append((lo, hi, count))

    if not rows:
        columns = BEDPE6_COLS + ["distance_bp", "hic_count"]
        return pd.DataFrame(columns=columns), hic_chrom

    df = pd.DataFrame(rows, columns=["x1", "y1", "hic_count"])
    df = df.groupby(["x1", "y1"], as_index=False, sort=False)["hic_count"].sum()
    df["chr1"] = str(chrom)
    df["x2"] = df["x1"] + binsize
    df["chr2"] = str(chrom)
    df["y2"] = df["y1"] + binsize
    df["distance_bp"] = df["y1"] - df["x1"]
    return df[BEDPE6_COLS + ["distance_bp", "hic_count"]].sort_values(
        ["x1", "y1"], ignore_index=True
    ), hic_chrom


def _counter_median(counter):
    """Exact median from value -> frequency counts."""
    total = sum(counter.values())
    if total <= 0:
        raise ValueError("Cannot calculate a median from an empty counter")

    left_rank = (total - 1) // 2
    right_rank = total // 2
    cumulative = 0
    left_value = None
    right_value = None

    for value, frequency in sorted(counter.items()):
        next_cumulative = cumulative + frequency
        if left_value is None and left_rank < next_cumulative:
            left_value = float(value)
        if right_rank < next_cumulative:
            right_value = float(value)
            break
        cumulative = next_cumulative

    return (left_value + right_value) / 2.0


def compute_min_contact_count_by_distance(
    pseudobulk_hic,
    chroms,
    binsize,
    hic_normalization="NONE",
    min_distance=None,
    max_distance=None,
    multiplier=AUTO_MIN_CONTACT_MULTIPLIER,
    logger=None,
    rank=0,
):
    """
    Compute one threshold per exact genomic distance across all selected chromosomes.

    Only unique cis bin pairs with aggregated observed .hic count > 0 are used.
    The threshold is:
        min_contact_count(distance) = multiplier * median(hic_count at that distance)
    """
    multiplier = float(multiplier)
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("[combine_cells] min_contact_foldchange must be > 0")

    histograms = defaultdict(Counter)

    for chrom in chroms:
        interactions, _ = _read_hic_interactions(
            pseudobulk_hic=pseudobulk_hic,
            chrom=chrom,
            binsize=binsize,
            hic_normalization=hic_normalization,
            min_distance=min_distance,
            max_distance=max_distance,
        )
        frequencies = interactions.groupby(
            ["distance_bp", "hic_count"], sort=False
        ).size()
        for (distance_bp, hic_count), frequency in frequencies.items():
            histograms[int(distance_bp)][float(hic_count)] += int(frequency)

        _log(
            logger,
            f"\tprocessor {rank}: threshold scan {chrom}, interactions={len(interactions)}",
            rank=rank,
            level=2,
        )

    if not histograms:
        raise ValueError(
            "[combine_cells] no nonzero cis interactions were found in the .hic file "
            "for the selected chromosomes and distance range"
        )

    rows = []
    thresholds = {}
    for distance_bp in sorted(histograms):
        counter = histograms[distance_bp]
        median_count = _counter_median(counter)
        threshold = float(multiplier) * median_count
        thresholds[int(distance_bp)] = threshold
        rows.append(
            {
                "distance_bp": int(distance_bp),
                "n_interactions": int(sum(counter.values())),
                "median_hic_count": float(median_count),
                "foldchange": float(multiplier),
                "min_contact_count": float(threshold),
            }
        )

    summary = pd.DataFrame(rows)

    # Only rank 0 prints the table, otherwise every MPI rank would print the
    # same distance thresholds. Here an interaction means one unique cis bin
    # pair whose aggregated observed .hic count is strictly greater than 0.
    if int(rank) == 0:
        total_nonzero = int(summary["n_interactions"].sum())
        print(
            "\n[MIN_CONTACT_COUNT_BY_DISTANCE] "
            "interaction = unique cis bin pair with observed .hic count > 0",
            flush=True,
        )
        print(
            f"[MIN_CONTACT_COUNT_BY_DISTANCE] "
            f"foldchange={multiplier:g}, distances={len(summary)}, "
            f"nonzero_interactions={total_nonzero}",
            flush=True,
        )
        print(
            "distance_bins\tdistance_bp\tn_interactions_count_gt_0\t"
            "median_count_gt_0\tfoldchange\tmin_contact_count",
            flush=True,
        )

        binsize_int = int(binsize)
        for row in summary.itertuples(index=False):
            distance_bp = int(row.distance_bp)
            distance_bins = (
                distance_bp // binsize_int
                if distance_bp % binsize_int == 0
                else distance_bp / float(binsize_int)
            )
            print(
                f"{distance_bins}\t{distance_bp}\t{int(row.n_interactions)}\t"
                f"{float(row.median_hic_count):g}\t{float(row.foldchange):g}\t"
                f"{float(row.min_contact_count):g}",
                flush=True,
            )
        print("[MIN_CONTACT_COUNT_BY_DISTANCE] end\n", flush=True)

    _log(
        logger,
        f"\tprocessor {rank}: calculated adaptive min_contact_count for "
        f"{len(thresholds)} distances from {summary['n_interactions'].sum()} interactions",
        rank=rank,
        level=1,
    )
    return thresholds, summary


def _fill_one_cell_column(df, base_index, num_rows):
    values = np.zeros(num_rows, dtype=np.float32)
    if df.empty:
        return values

    aligned = df[["x1", "y1", "value"]].merge(
        base_index,
        on=["x1", "y1"],
        how="inner",
        validate="one_to_one",
    )
    if not aligned.empty:
        values[aligned["_row_id"].to_numpy(dtype=np.int64)] = aligned["value"].to_numpy(
            dtype=np.float32
        )
    return values


def combine_and_reformat_chroms(
    indir,
    output_filename,
    chrom,
    min_contact_count,
    logger,
    rank,
    binsize=None,
    input_pattern="*.bedpe*",
    pseudobulk_hic=None,
    hic_normalization="NONE",
    union_min_distance=None,
    union_max_distance=None,
    distance_thresholds=None,
):
    """
    Build the interaction x pseudocell matrix and calculate support_cell_count.

    min_contact_count:
      - numeric: use the same fixed threshold for every interaction;
      - None: use distance_thresholds[distance_bp], normally calculated as
        2 * median(nonzero .hic counts at the same distance).
    """
    if binsize is None:
        raise ValueError("[combine_cells] binsize is required")
    if pseudobulk_hic is None:
        raise ValueError("[combine_cells] pseudobulk_hic is required")

    input_files = _find_input_files(indir, chrom, pattern=input_pattern)
    if not input_files:
        raise FileNotFoundError(f"[combine_cells] no BEDPE files found for {chrom} in {indir}")

    interactions, hic_chrom = _read_hic_interactions(
        pseudobulk_hic=pseudobulk_hic,
        chrom=chrom,
        binsize=binsize,
        hic_normalization=hic_normalization,
        min_distance=union_min_distance,
        max_distance=union_max_distance,
    )
    union_keys = interactions[BEDPE6_COLS].copy()
    num_rows = len(union_keys)
    num_cells = len(input_files)

    if min_contact_count is None:
        if distance_thresholds is None:
            raise ValueError(
                "[combine_cells] distance_thresholds is required when min_contact_count is auto"
            )
        threshold_vector = interactions["distance_bp"].map(distance_thresholds)
        if threshold_vector.isna().any():
            missing = sorted(
                interactions.loc[threshold_vector.isna(), "distance_bp"].astype(int).unique().tolist()
            )
            raise KeyError(
                f"[combine_cells] missing adaptive thresholds for distances: {missing[:20]}"
            )
        threshold_vector = threshold_vector.to_numpy(dtype=np.float32)
        threshold_mode = "adaptive"
    else:
        fixed_threshold = float(min_contact_count)
        if not np.isfinite(fixed_threshold) or fixed_threshold <= 0:
            raise ValueError("[combine_cells] min_contact_count must be > 0")
        threshold_vector = np.full(num_rows, fixed_threshold, dtype=np.float32)
        threshold_mode = f"fixed={fixed_threshold:g}"

    _log(
        logger,
        f"\tprocessor {rank}: {chrom} cells={num_cells}, interactions={num_rows}, "
        f"hic_chrom={hic_chrom}, min_contact_count={threshold_mode}",
        rank=rank,
        level=1,
    )

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    hdf_path = output_filename + ".cells.hdf"
    if os.path.exists(hdf_path):
        os.remove(hdf_path)

    support_cell_count = np.zeros(num_rows, dtype=np.int32)
    sum_scores = np.zeros(num_rows, dtype=np.float64)

    with h5py.File(hdf_path, "w") as hdf_file:
        names = hdf_file.create_dataset("cellnames", (1, num_cells), "S1000")
        names[0, :] = [
            os.path.basename(path).encode("ascii", "ignore") for path in input_files
        ]
        cells_data = hdf_file.create_dataset(
            chrom,
            shape=(num_rows, num_cells),
            maxshape=(None, num_cells),
            chunks=(min(100000, max(1, num_rows)), num_cells),
            dtype="float32",
        )

        if num_rows:
            base_index = union_keys[["x1", "y1"]].copy()
            base_index["_row_id"] = np.arange(num_rows, dtype=np.int64)

            for cell_index, path in enumerate(input_files):
                cell_df = _standardize_cis(_read_bedpe7(path), chrom, binsize)
                column = _fill_one_cell_column(cell_df, base_index, num_rows)
                cells_data[:, cell_index] = column
                support_cell_count += (column >= threshold_vector).astype(np.int32)
                sum_scores += column.astype(np.float64)

                if (cell_index + 1) % 100 == 0 or cell_index + 1 == num_cells:
                    _log(
                        logger,
                        f"\tprocessor {rank}: {chrom} filled "
                        f"{cell_index + 1}/{num_cells} pseudocells",
                        rank=rank,
                        level=2,
                    )

    support_fraction = support_cell_count / float(num_cells)
    mean_scores = sum_scores / float(num_cells)

    combined = union_keys.copy()
    combined["support_cell_count"] = support_cell_count
    combined["support_fraction"] = support_fraction
    combined.to_csv(output_filename, sep="\t", index=False, header=False)

    hic_input = union_keys.copy()
    if not hic_input.empty:
        hic_input["score"] = np.ceil(mean_scores).astype(int)
        hic_input = hic_input[hic_input["score"] > 0].copy()
        hic_input["str1"] = 0
        hic_input["str2"] = 1
        hic_input["frag1"] = 0
        hic_input["frag2"] = 1
        hic_input = hic_input[
            ["str1", "chr1", "x1", "frag1", "str2", "chr2", "y1", "frag2", "score"]
        ]
    else:
        hic_input = pd.DataFrame(
            columns=["str1", "chr1", "x1", "frag1", "str2", "chr2", "y1", "frag2", "score"]
        )
    hic_input.to_csv(output_filename + ".hic.input", sep="\t", index=False, header=False)


def combine_cells(
    indir,
    outdir,
    outlier_threshold=None,
    chrom_lens=None,
    rank=0,
    n_proc=1,
    logger=None,
    min_contact_count=None,
    binsize=None,
    input_pattern="*.bedpe*",
    pseudobulk_hic=None,
    hic_normalization="NONE",
    union_min_distance=None,
    union_max_distance=None,
    distance_thresholds=None,
    min_contact_foldchange=AUTO_MIN_CONTACT_MULTIPLIER,
):
    """Backward-compatible wrapper for chromosome-parallel processing."""
    if chrom_lens is None:
        raise ValueError("[combine_cells] chrom_lens is required")
    if binsize is None:
        raise ValueError("[combine_cells] binsize is required")
    if pseudobulk_hic is None:
        raise ValueError("[combine_cells] pseudobulk_hic is required")

    # Preserve the old positional outlier_threshold as a fixed-threshold override.
    if min_contact_count is None and outlier_threshold is not None:
        min_contact_count = float(outlier_threshold)

    if min_contact_count is None and distance_thresholds is None:
        distance_thresholds, summary = compute_min_contact_count_by_distance(
            pseudobulk_hic=pseudobulk_hic,
            chroms=list(chrom_lens),
            binsize=binsize,
            hic_normalization=hic_normalization,
            min_distance=union_min_distance,
            max_distance=union_max_distance,
            multiplier=min_contact_foldchange,
            logger=logger,
            rank=rank,
        )
        if rank == 0:
            os.makedirs(outdir, exist_ok=True)
            summary.to_csv(
                os.path.join(outdir, "min_contact_count_by_distance.tsv"),
                sep="\t",
                index=False,
            )

    if logger is not None:
        logger.set_rank(rank)
    os.makedirs(outdir, exist_ok=True)

    for chrom in get_proc_chroms(chrom_lens, rank, n_proc):
        output_filename = os.path.join(outdir, f"{chrom}.raw.combined.bedpe")
        combine_and_reformat_chroms(
            indir=indir,
            output_filename=output_filename,
            chrom=chrom,
            min_contact_count=min_contact_count,
            logger=logger,
            rank=rank,
            binsize=binsize,
            input_pattern=input_pattern,
            pseudobulk_hic=pseudobulk_hic,
            hic_normalization=hic_normalization,
            union_min_distance=union_min_distance,
            union_max_distance=union_max_distance,
            distance_thresholds=distance_thresholds,
        )


def combine_chrom_hic(directory, no_cool, no_hic, genome, chrom_sizes_filename, binsize, prefix):
    output_filename = os.path.join(directory, "allChr.hic.input")
    hic_filename = os.path.join(directory, f"{prefix}.allChr.hic" if prefix else "allChr.hic")
    cooler_filename = os.path.join(
        directory, f"{prefix}.allChr.cool" if prefix else "allChr.cool"
    )

    chrom_files = sorted(glob.glob(os.path.join(directory, "*.bedpe.hic.input")))
    with open(output_filename, "wb") as output_handle:
        for path in chrom_files:
            with open(path, "rb") as input_handle:
                output_handle.write(input_handle.read())
            os.remove(path)

    if not no_hic:
        project_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        juicer_jar = os.path.join(project_dir, "utils", "juicer_tools_1.22.01.jar")
        subprocess.check_call(
            ["java", "-jar", juicer_jar, "pre", output_filename, hic_filename, genome]
        )

    if not no_cool:
        if cooler is None:
            raise ImportError("[combine_chrom_hic] cooler is required when --no-cool is not set")
        subprocess.check_call(
            [
                "cooler",
                "cload",
                "pairs",
                "--zero-based",
                "--assembly",
                genome,
                "-c1",
                "2",
                "-p1",
                "3",
                "-c2",
                "6",
                "-p2",
                "7",
                "--field",
                "count=9",
                f"{chrom_sizes_filename}:{int(binsize)}",
                output_filename,
                cooler_filename,
            ]
        )