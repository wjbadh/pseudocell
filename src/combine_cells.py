import os
import glob
import subprocess
import sys
import numpy as np
import pandas as pd
import h5py

try:
    import cooler
except Exception:
    cooler = None


BEDPE6_COLS = ["chr1", "x1", "x2", "chr2", "y1", "y2"]
BEDPE7_COLS = ["chr1", "x1", "x2", "chr2", "y1", "y2", "value"]


def get_proc_chroms(chrom_lens, rank, n_proc):
    chrom_list = [(k, chrom_lens[k]) for k in list(chrom_lens.keys())]
    chrom_list.sort(key=lambda x: x[1])
    chrom_list.reverse()
    chrom_names = [i[0] for i in chrom_list]

    indices = list(range(rank, len(chrom_names), n_proc))
    proc_chroms = [chrom_names[i] for i in indices]
    return proc_chroms


def _read_bedpe7(path):
    try:
        df = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
            compression="gzip" if path.endswith(".gz") else None,
            low_memory=False,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=BEDPE7_COLS)

    if df.shape[0] == 0:
        return pd.DataFrame(columns=BEDPE7_COLS)
    if df.shape[1] < 7:
        raise ValueError(f"[combine_cells] {path} has fewer than 7 columns: {df.shape[1]}")

    df = df.iloc[:, :7].copy()
    df.columns = BEDPE7_COLS

    for c in ["x1", "x2", "y1", "y2"]:
        df[c] = df[c].astype(np.int64)
    df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0).astype(float)
    df["chr1"] = df["chr1"].astype(str)
    df["chr2"] = df["chr2"].astype(str)
    return df


def _standardize_cis(df, chrom, binsize=None):
    if df.shape[0] == 0:
        return pd.DataFrame(columns=BEDPE7_COLS)

    df = df[(df["chr1"] == chrom) & (df["chr2"] == chrom)].copy()
    if df.shape[0] == 0:
        return pd.DataFrame(columns=BEDPE7_COLS)

    if binsize is not None:
        a = (df["x1"].astype(np.int64) // int(binsize)) * int(binsize)
        b = (df["y1"].astype(np.int64) // int(binsize)) * int(binsize)
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        df["x1"] = lo
        df["x2"] = lo + int(binsize)
        df["y1"] = hi
        df["y2"] = hi + int(binsize)
    else:
        lo = np.minimum(df["x1"].astype(np.int64), df["y1"].astype(np.int64))
        hi = np.maximum(df["x1"].astype(np.int64), df["y1"].astype(np.int64))
        width_x = (df["x2"] - df["x1"]).abs().median()
        width_y = (df["y2"] - df["y1"]).abs().median()
        width = int(width_x if pd.notnull(width_x) else width_y)
        if width <= 0:
            width = 1
        df["x1"] = lo
        df["x2"] = lo + width
        df["y1"] = hi
        df["y2"] = hi + width

    df = df[df["x1"] < df["y1"]].copy()
    df = df.groupby(BEDPE6_COLS, as_index=False)["value"].sum()
    return df


def _find_input_files(indir, chrom, pattern="*.bedpe*"):
    files = []
    for fp in glob.glob(os.path.join(indir, pattern)):
        if os.path.isdir(fp):
            continue
        base = os.path.basename(fp)
        if f".{chrom}." in base or base.endswith(f".{chrom}.bedpe") or base.endswith(f".{chrom}.bedpe.gz"):
            files.append(fp)

    if len(files) == 0:
        files = glob.glob(os.path.join(indir, f"*{chrom}*.bedpe*"))

    files = sorted([f for f in files if os.path.isfile(f)])
    return files


def combine_and_reformat_chroms(
    indir,
    output_filename,
    chrom,
    min_contact_count,
    logger,
    rank,
    binsize=None,
    input_pattern="*.bedpe*",
):
    """
    Combine raw-count pseudocell BEDPE files into:
      1) output_filename
         chr1 x1 x2 chr2 y1 y2 support_cell_count support_fraction

      2) output_filename + ".cells.hdf"
         interaction × pseudocell raw-count matrix

      3) output_filename + ".hic.input"
         juicer/cooler-like input using mean raw count as score

    min_contact_count:
        A pseudocell supports an interaction if raw count >= min_contact_count.
    """
    input_filenames = _find_input_files(indir, chrom, pattern=input_pattern)

    if len(input_filenames) == 0:
        raise FileNotFoundError(f"[combine_cells] no BEDPE files found for {chrom} in {indir}")

    if logger:
        logger.write(
            f'\tprocessor {rank}: {chrom} found {len(input_filenames)} pseudocell files',
            verbose_level=1,
            append_time=False,
            allow_all_ranks=True,
        )

    cell_dfs = []
    union_keys = None

    for fp in input_filenames:
        df = _read_bedpe7(fp)
        df = _standardize_cis(df, chrom=chrom, binsize=binsize)
        df = df[BEDPE6_COLS + ["value"]].copy()

        if union_keys is None:
            union_keys = df[BEDPE6_COLS].copy()
        else:
            union_keys = pd.concat([union_keys, df[BEDPE6_COLS]], axis=0, ignore_index=True)

        cell_dfs.append(df)

    union_keys = union_keys.drop_duplicates().sort_values(["x1", "y1"]).reset_index(drop=True)

    num_cells = len(input_filenames)
    num_rows = union_keys.shape[0]

    os.makedirs(os.path.dirname(output_filename), exist_ok=True)

    hdf_path = output_filename + ".cells.hdf"
    if os.path.exists(hdf_path):
        os.remove(hdf_path)

    hdf_file = h5py.File(hdf_path, "w")
    names_dataset = hdf_file.create_dataset("cellnames", (1, num_cells), "S1000")
    names_dataset[0, :] = [
        os.path.basename(fname).encode("ascii", "ignore") for fname in input_filenames
    ]

    cells_data = hdf_file.create_dataset(
        chrom,
        chunks=(min(100000, max(1, num_rows)), num_cells),
        shape=(num_rows, num_cells),
        maxshape=(None, num_cells),
        dtype="float32",
    )

    values_matrix = np.zeros((num_rows, num_cells), dtype=np.float32)
    base_index = union_keys[["x1", "y1"]].copy()
    base_index["_row_id"] = np.arange(num_rows)

    for cell_idx, df in enumerate(cell_dfs):
        tmp = df[["x1", "y1", "value"]].copy()
        tmp = tmp.merge(base_index, on=["x1", "y1"], how="left")
        values_matrix[tmp["_row_id"].to_numpy(dtype=np.int64), cell_idx] = tmp["value"].to_numpy(dtype=np.float32)

    cells_data[:, :] = values_matrix
    hdf_file.close()

    support_cell_count = np.sum(values_matrix >= float(min_contact_count), axis=1).astype(np.int32)
    support_fraction = support_cell_count / float(num_cells)
    mean_scores = np.mean(values_matrix, axis=1)

    out = union_keys.copy()
    out["support_cell_count"] = support_cell_count
    out["support_fraction"] = support_fraction
    out.to_csv(output_filename, sep="\t", index=False, header=False)

    hic_format = union_keys.copy()
    hic_format.loc[:, "score"] = np.ceil(mean_scores).astype(int)
    hic_format.loc[:, "str1"] = 0
    hic_format.loc[:, "str2"] = 1
    hic_format.loc[:, "frag1"] = 0
    hic_format.loc[:, "frag2"] = 1
    hic_format = hic_format[["str1", "chr1", "x1", "frag1", "str2", "chr2", "y1", "frag2", "score"]]
    hic_format = hic_format[hic_format["score"] > 0]
    hic_format.to_csv(output_filename + ".hic.input", index=False, header=False, sep="\t")

    if logger:
        logger.write(
            f'\tprocessor {rank}: {chrom} combined matrix rows={num_rows}, pseudocells={num_cells}',
            verbose_level=1,
            append_time=False,
            allow_all_ranks=True,
        )


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
):
    """
    Backward-compatible wrapper.

    Old argument:
      outlier_threshold

    New semantics:
      min_contact_count

    If min_contact_count is None, outlier_threshold is used as the raw-count
    minimum required for pseudocell support.
    """
    if chrom_lens is None:
        raise ValueError("[combine_cells] chrom_lens is required")

    if min_contact_count is None:
        min_contact_count = 1 if outlier_threshold is None else outlier_threshold

    if logger:
        logger.set_rank(rank)

    os.makedirs(outdir, exist_ok=True)

    proc_chroms = get_proc_chroms(chrom_lens, rank, n_proc)

    for chrom in proc_chroms:
        if logger:
            logger.write(
                f'\tprocessor {rank}: combining chromosome {chrom}',
                verbose_level=1,
                allow_all_ranks=True,
            )

        output_filename = os.path.join(outdir, ".".join([chrom, "raw", "combined", "bedpe"]))

        combine_and_reformat_chroms(
            indir=indir,
            output_filename=output_filename,
            chrom=chrom,
            min_contact_count=min_contact_count,
            logger=logger,
            rank=rank,
            binsize=binsize,
            input_pattern=input_pattern,
        )

        if logger:
            logger.write(
                f'\tprocessor {rank}: chromosome {chrom} is combined',
                verbose_level=1,
                allow_all_ranks=True,
            )


def combine_chrom_hic(directory, no_cool, no_hic, genome, chrom_sizes_filename, binsize, prefix):
    output_filename = os.path.join(directory, "allChr.hic.input")

    if prefix:
        hic_filename = os.path.join(directory, f"{prefix}.allChr.hic")
        cooler_filename = os.path.join(directory, f"{prefix}.allChr.cool")
    else:
        hic_filename = os.path.join(directory, "allChr.hic")
        cooler_filename = os.path.join(directory, "allChr.cool")

    chrom_files = glob.glob(directory + "/*.bedpe.hic.input")
    input_filepattern = directory + "/*.bedpe.hic.input"

    if len(chrom_files) == 0:
        with open(output_filename, "w") as ofile:
            pass
        return

    proc = subprocess.Popen("cat " + input_filepattern + " > " + output_filename, shell=True)
    proc.communicate()

    for fname in chrom_files:
        os.remove(fname)

    if not no_hic:
        juicer_path = os.path.realpath(__file__)
        juicer_path = juicer_path[:juicer_path.rfind("/")]
        juicer_path = juicer_path[:juicer_path.rfind("/")]
        subprocess.check_call(
            " ".join([
                f"java -jar {juicer_path}/utils/juicer_tools_1.22.01.jar pre",
                output_filename,
                hic_filename,
                genome,
            ]),
            shell=True,
        )

    if not no_cool:
        if cooler is None:
            raise ImportError("[combine_chrom_hic] cooler is required when --no-cool is not set")
        subprocess.check_call(
            " ".join([
                "cooler cload pairs --zero-based --assembly",
                genome,
                "-c1 2 -p1 3 -c2 6 -p2 7 --field count=9",
                chrom_sizes_filename + ":" + str(int(binsize)),
                output_filename,
                cooler_filename,
            ]),
            shell=True,
        )