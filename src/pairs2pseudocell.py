#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import gzip
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.sparse import coo_matrix, csr_matrix


def natural_key(value):
    parts = re.split(r"(\d+)", str(value))
    return [int(x) if x.isdigit() else x for x in parts]


def normalize_chrom(chrom):
    chrom = str(chrom).strip()
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def open_text(path):
    with open(path, "rb") as handle:
        is_gzip = handle.read(2) == b"\x1f\x8b"
    return gzip.open(path, "rt") if is_gzip else open(path, "rt")


def list_input_files(input_dir):
    files = [
        entry.path
        for entry in os.scandir(input_dir)
        if entry.is_file(follow_symlinks=True)
    ]
    files.sort(key=lambda x: natural_key(os.path.basename(x)))
    return files


def read_one_cell(path, chrom, binsize):
    cell = Path(path).name
    counts = defaultdict(int)
    n_data = 0

    with open_text(path) as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            n_data += 1
            fields = line.split()
            if len(fields) < 5:
                raise ValueError(
                    f"{path}:{line_no}: expected at least 5 columns, got {len(fields)}"
                )

            chrom1 = normalize_chrom(fields[1])
            chrom2 = normalize_chrom(fields[3])

            try:
                pos1 = int(fields[2])
                pos2 = int(fields[4])
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_no}: pos1/pos2 must be integers"
                ) from exc

            if pos1 < 1 or pos2 < 1:
                raise ValueError(
                    f"{path}:{line_no}: pairs coordinates must be >= 1"
                )

            if chrom1 != chrom2 or chrom1 != chrom:
                continue

            bin1 = ((pos1 - 1) // binsize) * binsize
            bin2 = ((pos2 - 1) // binsize) * binsize
            if bin1 > bin2:
                bin1, bin2 = bin2, bin1

            counts[(bin1, bin2)] += 1

    if n_data == 0:
        raise ValueError(f"{path}: no data records")

    return [
        (cell, chrom, bin1, bin2, count)
        for (bin1, bin2), count in counts.items()
    ]


def build_long_table(files, chrom, binsize, read_jobs, chunk_size):
    parts = []
    chunk_size = max(1, chunk_size)

    for start in range(0, len(files), chunk_size):
        end = min(start + chunk_size, len(files))
        print(f"[READ] {chrom}: files {start + 1}-{end}/{len(files)}", flush=True)

        rows = Parallel(
            n_jobs=read_jobs,
            backend="loky",
            batch_size=1,
            pre_dispatch="2*n_jobs",
        )(
            delayed(read_one_cell)(path, chrom, binsize)
            for path in files[start:end]
        )

        flat_rows = [row for cell_rows in rows for row in cell_rows]
        if flat_rows:
            parts.append(
                pd.DataFrame(
                    flat_rows,
                    columns=["cell", "chrom", "binA", "binB", "count"],
                )
            )

        del rows, flat_rows
        gc.collect()

    if not parts:
        raise ValueError(f"No cis contacts found for {chrom}")

    df = pd.concat(parts, ignore_index=True)
    df["cell"] = df["cell"].astype("category")
    df["chrom"] = df["chrom"].astype("category")
    df[["binA", "binB", "count"]] = df[["binA", "binB", "count"]].astype(np.int64)

    print(
        f"[READ] {chrom}: cells={df['cell'].nunique()}, rows={len(df):,}",
        flush=True,
    )
    return df


def filter_interactions(df, threshold):
    if threshold <= 0:
        return df

    n_cells = df["cell"].nunique()
    support = (
        df.groupby(["chrom", "binA", "binB"], observed=True)["cell"]
        .nunique()
        .div(n_cells)
        .rename("support")
        .reset_index()
    )
    keep = support.loc[
        support["support"] >= threshold,
        ["chrom", "binA", "binB"],
    ]
    return df.merge(keep, on=["chrom", "binA", "binB"], how="inner")


def make_pseudocells(df, m, phi, seed, replace, prefix, global_offset, batch_id):
    df = df[df["binA"] < df["binB"]].copy()
    if df.empty:
        raise ValueError("No non-self interactions remain")

    df["cell"] = df["cell"].astype("category")
    cells = df["cell"].cat.categories.astype(str).to_numpy()
    cell_codes = df["cell"].cat.codes.to_numpy()
    n_cells = len(cells)

    edge_index = pd.MultiIndex.from_frame(df[["chrom", "binA", "binB"]])
    edge_codes, edges = pd.factorize(edge_index, sort=False)
    edge_df = edges.to_frame(index=False)
    edge_df.columns = ["chrom", "binA", "binB"]

    X = coo_matrix(
        (
            df["count"].to_numpy(np.int64),
            (cell_codes, edge_codes),
        ),
        shape=(n_cells, len(edges)),
    ).tocsr()

    k = max(1, int(round(phi * n_cells)))
    if not replace and k > n_cells:
        raise ValueError(f"phi={phi} selects {k} cells from only {n_cells} without replacement")

    rng = np.random.default_rng(seed)
    meta_rows = []

    if replace:
        weights = rng.multinomial(
            k,
            np.full(n_cells, 1.0 / n_cells),
            size=m,
        ).astype(np.int32)
        W = csr_matrix(weights)

        for i in range(m):
            selected = np.flatnonzero(weights[i])
            selected_cells = ",".join(
                f"{cells[j]}:{weights[i, j]}" for j in selected
            )
            meta_rows.append(
                _meta_row(
                    i, global_offset, batch_id, seed, replace,
                    phi, n_cells, k, prefix, selected_cells,
                )
            )
    else:
        row_idx = []
        col_idx = []

        for i in range(m):
            selected = np.sort(rng.choice(n_cells, size=k, replace=False))
            row_idx.extend([i] * k)
            col_idx.extend(selected.tolist())
            selected_cells = ",".join(cells[selected])
            meta_rows.append(
                _meta_row(
                    i, global_offset, batch_id, seed, replace,
                    phi, n_cells, k, prefix, selected_cells,
                )
            )

        W = coo_matrix(
            (
                np.ones(len(row_idx), dtype=np.int32),
                (row_idx, col_idx),
            ),
            shape=(m, n_cells),
        ).tocsr()

    Y = (W @ X).tocsr()

    order = np.lexsort(
        (
            edge_df["binB"].to_numpy(np.int64),
            edge_df["binA"].to_numpy(np.int64),
        )
    )
    Y = Y[:, order]
    edge_df = edge_df.iloc[order].reset_index(drop=True)

    return Y, edge_df, pd.DataFrame(meta_rows)


def _meta_row(
    local_i,
    global_offset,
    batch_id,
    seed,
    replace,
    phi,
    n_cells,
    k,
    prefix,
    selected_cells,
):
    local_num = local_i + 1
    global_num = global_offset + local_num
    return {
        "batch_id": batch_id,
        "random_seed": seed,
        "replace": replace,
        "phi": phi,
        "n_total_cells": n_cells,
        "n_sampled_cells": k,
        "pb_local_num": local_num,
        "pb_global_num": global_num,
        "pb_num": f"pb{local_num}",
        "pb_name": f"{prefix}{global_num}",
        "selected_cells": selected_cells,
    }


def write_outputs(Y, edges, meta, chrom, outdir, binsize, prefix, global_offset, batch_id):
    outdir = Path(outdir).resolve()
    raw_dir = outdir / "pesudocell_raw"
    meta_dir = outdir / "meta_cell"
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    meta_path = meta_dir / f"meta_cell.{chrom}.batch{batch_id}.tsv"
    meta.to_csv(meta_path, sep="\t", index=False)

    bin1 = edges["binA"].to_numpy(np.int64)
    bin2 = edges["binB"].to_numpy(np.int64)
    n_edges = len(edges)

    for i in range(Y.shape[0]):
        name = f"{prefix}{global_offset + i + 1}"
        path = raw_dir / f"{name}.{chrom}.normalized.pseudocell.bedpe"
        values = Y.getrow(i).toarray().ravel().astype(np.int64)

        with open(path, "wt") as handle:
            for start in range(0, n_edges, 200_000):
                end = min(start + 200_000, n_edges)
                pd.DataFrame(
                    {
                        "chr1": chrom,
                        "x1": bin1[start:end],
                        "x2": bin1[start:end] + binsize,
                        "chr2": chrom,
                        "y1": bin2[start:end],
                        "y2": bin2[start:end] + binsize,
                        "val": values[start:end],
                    }
                ).to_csv(handle, sep="\t", header=False, index=False)

    print(
        f"[WRITE] {chrom}: pseudocells={Y.shape[0]}, interactions={n_edges:,}",
        flush=True,
    )


def cleanup_outputs(outdir, chrom):
    outdir = Path(outdir).resolve()

    raw_dir = outdir / "pesudocell_raw"
    if raw_dir.exists():
        for path in raw_dir.glob(f"*.{chrom}.normalized.pseudocell.bedpe"):
            path.unlink()

    meta_dir = outdir / "meta_cell"
    if meta_dir.exists():
        for path in meta_dir.glob(f"meta_cell.{chrom}.batch*.tsv"):
            path.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--pattern", default="")  # compatibility only; ignored
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--read_jobs", type=int, default=24)
    parser.add_argument("--read_chunk_size", type=int, default=24)
    parser.add_argument("--thr", type=float, required=True)
    parser.add_argument("--m_total", type=int, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--phi", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--binsize", type=int, required=True)
    parser.add_argument("--prefix", default="GM_")
    parser.add_argument("--overwrite_rwr", action="store_true")
    args = parser.parse_args()

    chrom = normalize_chrom(args.chrom)
    files = list_input_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No files found in {args.input_dir}")

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"[START] chrom={chrom}, files={len(files)}, outdir={outdir}",
        flush=True,
    )

    if args.overwrite_rwr:
        cleanup_outputs(outdir, chrom)

    df = build_long_table(
        files=files,
        chrom=chrom,
        binsize=args.binsize,
        read_jobs=args.read_jobs,
        chunk_size=args.read_chunk_size,
    )
    df = filter_interactions(df, args.thr)
    if df.empty:
        raise ValueError("No interactions remain after filtering")

    start = 1
    batch_id = 0

    while start <= args.m_total:
        m = min(args.batch_size, args.m_total - start + 1)
        global_offset = start - 1
        seed = args.seed + batch_id

        print(
            f"[BATCH] {chrom}: batch={batch_id}, pseudocells={start}-{start + m - 1}, seed={seed}",
            flush=True,
        )

        Y, edges, meta = make_pseudocells(
            df=df,
            m=m,
            phi=args.phi,
            seed=seed,
            replace=args.replace,
            prefix=args.prefix,
            global_offset=global_offset,
            batch_id=batch_id,
        )

        write_outputs(
            Y=Y,
            edges=edges,
            meta=meta,
            chrom=chrom,
            outdir=outdir,
            binsize=args.binsize,
            prefix=args.prefix,
            global_offset=global_offset,
            batch_id=batch_id,
        )

        del Y, edges, meta
        gc.collect()

        start += m
        batch_id += 1

    print(f"[DONE] {chrom}: {outdir}", flush=True)


if __name__ == "__main__":
    main()
