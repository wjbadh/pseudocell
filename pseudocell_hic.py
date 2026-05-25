#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pseudocell_hic.py

Raw-count pseudocell Hi-C loop calling workflow.

流程：
    hic -> interaction -> postprocess

其中 support filter 不再是独立 step，而是 postprocess 内部由 --support-filter 控制。

如果设置 --support-filter：
    postprocess 生成 candidates.{chrom}.bedpe 后，
    立即对 candidates 做 pseudocell 半径支持筛选，
    输出：
        candidate_support.{chrom}.bedpe
        candidates_filter.{chrom}.bedpe
    随后聚类使用 candidates_filter.{chrom}.bedpe。

如果不设置 --support-filter：
    聚类使用 candidates.{chrom}.bedpe。

postprocess 的五类结构背景：
    circle / donut / lower_left / horizontal / vertical
    直接从预生成的 pseudobulk .hic 文件读取。

注意：
    --pseudobulk-hic 是 postprocess 必需参数。
    support_filter 仍然使用 -i/--indir 指定的 pseudocell raw BEDPE 目录。
"""

import argparse
import os
import multiprocessing

from src.combine_cells import combine_cells, combine_chrom_hic
from src.interaction_caller import call_interactions, combine_chrom_interactions
from src.postprocess import postprocess, combine_postprocessed_chroms
import src.logger


VALID_STEPS = {"hic", "interaction", "postprocess"}


def main():
    parser = create_parser()
    args = parser.parse_args()

    validate_args(args)

    if args.make_hic:
        args.no_hic = False
    if args.make_cool:
        args.no_cool = False

    if args.summit_gap == -1:
        args.summit_gap = 2 * args.binsize

    parallel_mode, rank, n_proc, parallel_properties = determine_parallelization_options(
        args.parallel,
        args.threaded,
        args.num_proc,
    )

    if rank == 0:
        os.makedirs(args.outdir, exist_ok=True)

    if parallel_mode == "parallel":
        parallel_properties["comm"].Barrier()

    threaded = True if parallel_mode == "threaded" else False

    logger = src.logger.Logger(
        f"{args.outdir}/pseudocell_hic.log",
        rank=rank,
        verbose_threshold=args.verbose,
        threaded=threaded,
    )

    logger.dump_args(args)
    logger.write(f"Starting pseudocell_hic workflow in {parallel_mode} mode")

    chrom_dict = parse_chrom_lengths(
        chrom=args.chrom,
        chrom_lens_filename=args.chr_lens,
        genome=args.genome,
        max_chrom_number=args.max_chrom_number,
    )

    logger.write(f"chromosome lengths file is read. Processing {len(chrom_dict)} chromosomes.")
    logger.flush()

    hic_dir = os.path.join(args.outdir, "hic")
    interaction_dir = os.path.join(args.outdir, "interactions")
    postproc_dir = os.path.join(args.outdir, "postprocessed")

    # ------------------------------------------------------------
    # step 1: hic
    # ------------------------------------------------------------
    if "hic" in args.steps:
        logger.write("Starting hic step: combining raw-count pseudocell BEDPE files")
        logger.flush()

        if parallel_mode in ("nonparallel", "parallel"):
            if parallel_mode == "parallel":
                parallel_properties["comm"].Barrier()

            combine_cells(
                indir=args.indir,
                outdir=hic_dir,
                chrom_lens=chrom_dict,
                rank=rank,
                n_proc=n_proc,
                logger=logger,
                min_contact_count=args.min_contact_count,
                binsize=args.binsize,
                input_pattern=args.pseudocell_pattern,
            )

            if parallel_mode == "parallel":
                parallel_properties["comm"].Barrier()

        elif parallel_mode == "threaded":
            params = [
                (
                    args.indir,
                    hic_dir,
                    None,
                    chrom_dict,
                    i,
                    n_proc,
                    logger,
                    args.min_contact_count,
                    args.binsize,
                    args.pseudocell_pattern,
                )
                for i in range(n_proc)
            ]

            with multiprocessing.Pool(n_proc) as pool:
                pool.starmap(combine_cells, params)

        logger.write("Per-chromosome raw pseudocell combine step completed")
        logger.flush()

        if rank == 0:
            combine_chrom_hic(
                directory=hic_dir,
                no_cool=args.no_cool,
                no_hic=args.no_hic,
                genome=args.genome,
                chrom_sizes_filename=args.chr_lens,
                binsize=args.binsize,
                prefix=args.prefix,
            )

        if parallel_mode == "parallel":
            parallel_properties["comm"].Barrier()

        logger.write("hic step completed")
        logger.flush()

    # ------------------------------------------------------------
    # step 2: interaction
    # ------------------------------------------------------------
    if "interaction" in args.steps:
        logger.write("Starting interaction step: one-sided Wilcoxon local background test")
        logger.flush()

        if parallel_mode in ("nonparallel", "parallel"):
            if parallel_mode == "parallel":
                parallel_properties["comm"].Barrier()

            call_interactions(
                indir=hic_dir,
                outdir=interaction_dir,
                chrom_lens=chrom_dict,
                binsize=args.binsize,
                dist=args.dist,
                neighborhood_limit_lower=args.local_lower_limit,
                neighborhood_limit_upper=args.local_upper_limit,
                rank=rank,
                n_proc=n_proc,
                max_mem=args.max_memory,
                logger=logger,
            )

            if parallel_mode == "parallel":
                parallel_properties["comm"].Barrier()

        elif parallel_mode == "threaded":
            params = [
                (
                    hic_dir,
                    interaction_dir,
                    chrom_dict,
                    args.binsize,
                    args.dist,
                    args.local_lower_limit,
                    args.local_upper_limit,
                    i,
                    n_proc,
                    args.max_memory,
                    logger,
                )
                for i in range(n_proc)
            ]

            with multiprocessing.Pool(n_proc) as pool:
                pool.starmap(call_interactions, params)

        if rank == 0:
            combine_chrom_interactions(directory=interaction_dir)

        if parallel_mode == "parallel":
            parallel_properties["comm"].Barrier()

        logger.write("interaction step completed")
        logger.flush()

    # ------------------------------------------------------------
    # step 3: postprocess
    # ------------------------------------------------------------
    if "postprocess" in args.steps:
        logger.write("Starting postprocess step using pseudobulk .hic structural background")
        logger.flush()

        if args.pseudobulk_hic is None:
            raise ValueError("--pseudobulk-hic is required for postprocess step")

        if parallel_mode in ("nonparallel", "parallel"):
            if parallel_mode == "parallel":
                parallel_properties["comm"].Barrier()

            postprocess(
                indir=interaction_dir,
                outdir=postproc_dir,
                chrom_lens=chrom_dict,
                fdr_thresh=args.fdr_threshold,
                gap_large=args.postproc_gap_large,
                gap_small=args.postproc_gap_small,
                candidate_lower_thresh=args.candidate_lower_distance,
                candidate_upper_thresh=args.candidate_upper_distance,
                binsize=args.binsize,
                dist=args.dist,
                clustering_gap=args.clustering_gap,
                rank=rank,
                n_proc=n_proc,
                max_mem=args.max_memory,
                stat_threshold=args.stat_threshold,
                min_support_fraction=args.min_support_fraction,
                circle_threshold_mult=args.circle_threshold_multiplier,
                donut_threshold_mult=args.donut_threshold_multiplier,
                lower_left_threshold_mult=args.lower_left_threshold_multiplier,
                horizontal_threshold_mult=args.horizontal_threshold_multiplier,
                vertical_threshold_mult=args.vertical_threshold_multiplier,
                filter_file=args.filter_file,
                summit_gap=args.summit_gap,
                logger=logger,
                pseudobulk_hic=args.pseudobulk_hic,
                hic_normalization=args.hic_normalization,
                support_filter=args.support_filter,
                pseudocell_dir=args.indir,
                support_radius_bins=args.support_radius_bins,
                support_ratio=args.support_ratio,
                pseudocell_pattern=args.pseudocell_pattern,
            )

            if parallel_mode == "parallel":
                parallel_properties["comm"].Barrier()

        elif parallel_mode == "threaded":
            params = [
                (
                    interaction_dir,                          # indir
                    postproc_dir,                            # outdir
                    chrom_dict,                              # chrom_lens
                    args.fdr_threshold,                      # fdr_thresh
                    args.postproc_gap_large,                 # gap_large
                    args.postproc_gap_small,                 # gap_small
                    args.candidate_lower_distance,           # candidate_lower_thresh
                    args.candidate_upper_distance,           # candidate_upper_thresh
                    args.binsize,                            # binsize
                    args.dist,                               # dist
                    args.clustering_gap,                     # clustering_gap
                    i,                                       # rank
                    n_proc,                                  # n_proc
                    args.max_memory,                         # max_mem
                    None,                                    # tstat_threshold
                    args.circle_threshold_multiplier,        # circle_threshold_mult
                    args.donut_threshold_multiplier,         # donut_threshold_mult
                    args.lower_left_threshold_multiplier,    # lower_left_threshold_mult
                    args.horizontal_threshold_multiplier,    # horizontal_threshold_mult
                    args.vertical_threshold_multiplier,      # vertical_threshold_mult
                    None,                                    # outlier_threshold_mult
                    args.filter_file,                        # filter_file
                    args.summit_gap,                         # summit_gap
                    logger,                                  # logger
                    None,                                    # support_dir
                    args.support_filter,                     # support_filter
                    args.pseudobulk_hic,                     # pseudobulk_hic
                    args.hic_normalization,                  # hic_normalization
                    None,                                    # raw_sc_bedpe_dir, compatibility only
                    "*.bedpe*",                              # raw_sc_pattern, compatibility only
                    args.stat_threshold,                     # stat_threshold
                    args.min_support_fraction,               # min_support_fraction
                    args.indir,                              # pseudocell_dir
                    args.support_radius_bins,                # support_radius_bins
                    args.support_ratio,                      # support_ratio
                    args.pseudocell_pattern,                 # pseudocell_pattern
                )
                for i in range(n_proc)
            ]

            with multiprocessing.Pool(n_proc) as pool:
                pool.starmap(postprocess, params)

        if rank == 0:
            combine_postprocessed_chroms(directory=postproc_dir, prefix=args.prefix)

        if parallel_mode == "parallel":
            parallel_properties["comm"].Barrier()

        logger.write("postprocess step completed")
        logger.flush()

    logger.write("Exiting pseudocell_hic workflow")
    logger.flush()


def validate_args(args):
    invalid_steps = [s for s in args.steps if s not in VALID_STEPS]
    if invalid_steps:
        raise ValueError(
            f"Invalid steps: {invalid_steps}. Valid steps are: {sorted(VALID_STEPS)}"
        )

    if "postprocess" in args.steps and args.pseudobulk_hic is None:
        raise ValueError("--pseudobulk-hic is required when postprocess step is enabled")

    if args.support_radius_bins < 0:
        raise ValueError("--support-radius-bins must be >= 0")

    if args.support_ratio < 0:
        raise ValueError("--support-ratio must be >= 0")


def parse_chrom_lengths(chrom, chrom_lens_filename, genome, max_chrom_number):
    if not max_chrom_number or max_chrom_number == -1:
        if not chrom or chrom == "None":
            chrom_count = 22 if genome.startswith("hg") else 19 if genome.startswith("mm") else None
            if not chrom_count:
                raise ValueError("Genome name is not recognized. Use --max-chrom-number.")
            chrom = ["chr" + str(i) for i in range(1, chrom_count + 1)]
        else:
            chrom = [c.strip() for c in chrom.split()]
    else:
        chrom = ["chr" + str(i) for i in range(1, max_chrom_number + 1)]

    with open(chrom_lens_filename) as infile:
        lines = infile.readlines()

    chrom_lens = {
        line.split()[0]: int(line.split()[1])
        for line in lines
        if len(line.split()) >= 2 and line.split()[0] in chrom
    }

    return chrom_lens


def determine_parallelization_options(parallel, threaded, n_proc):
    if parallel and threaded:
        raise ValueError("Only one of --parallel or --threaded can be set.")

    if parallel:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        n_proc = comm.Get_size()
        rank = comm.Get_rank()
        mode = "parallel"
        properties = {"comm": comm}
    elif threaded:
        if n_proc < 1:
            raise ValueError("If --threaded is set, --num-proc should be a positive integer.")
        mode = "threaded"
        rank = 0
        properties = {}
    else:
        mode = "nonparallel"
        n_proc = 1
        rank = 0
        properties = {}

    return mode, rank, n_proc, properties


def create_parser():
    parser = argparse.ArgumentParser(
        description="Raw-count pseudocell Hi-C loop caller without bin/RWR steps."
    )

    parser.add_argument(
        "-i",
        "--indir",
        required=True,
        help="Input pseudocell raw BEDPE directory.",
    )

    parser.add_argument(
        "--pseudobulk-hic",
        required=True,
        help="Pre-generated pseudobulk .hic file for structural background calculation.",
    )

    parser.add_argument(
        "--hic-normalization",
        default="NONE",
        help="Normalization type for reading .hic, e.g. NONE, KR, VC, VC_SQRT.",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        required=True,
        help="Output directory.",
    )

    parser.add_argument(
        "-l",
        "--chr-lens",
        required=True,
        help="Chromosome lengths file.",
    )

    parser.add_argument(
        "-g",
        "--genome",
        default="hg38",
        help="Genome name, e.g. hg38 or mm10.",
    )

    parser.add_argument(
        "--chrom",
        default=None,
        help='Chromosome(s), e.g. "chr1" or "chr1 chr2".',
    )

    parser.add_argument("--max-chrom-number", type=int, default=-1)
    parser.add_argument("--prefix", default=None)

    parser.add_argument(
        "--pseudocell-pattern",
        default="*.bedpe*",
        help="Input pseudocell BEDPE filename pattern.",
    )

    parser.add_argument(
        "--steps",
        nargs="*",
        default=["hic", "interaction", "postprocess"],
        help="Valid steps: hic interaction postprocess.",
    )

    parser.add_argument("--parallel", action="store_true", default=False)
    parser.add_argument("--threaded", action="store_true", default=False)
    parser.add_argument("-n", "--num-proc", default=0, type=int)

    parser.add_argument("--dist", type=int, default=2_000_000)
    parser.add_argument("--binsize", type=int, default=10_000)

    parser.add_argument(
        "--min-contact-count",
        type=float,
        default=1,
        help="Minimum raw count retained in hic/combine step.",
    )

    parser.add_argument(
        "--min-support-fraction",
        type=float,
        default=0.05,
        help="Minimum fraction of pseudocells with nonzero center interaction in interaction/postprocess filtering.",
    )

    parser.add_argument(
        "--stat-threshold",
        type=float,
        default=0.0,
        help="Wilcoxon statistic threshold. Set manually according to pseudocell number.",
    )

    parser.add_argument("--local-lower-limit", type=int, default=2)
    parser.add_argument("--local-upper-limit", type=int, default=5)
    parser.add_argument("--fdr-threshold", type=float, default=0.1)

    parser.add_argument("--candidate-lower-distance", type=int, default=100_000)
    parser.add_argument("--candidate-upper-distance", type=int, default=2_000_000)

    parser.add_argument("--postproc-gap-large", type=int, default=5)
    parser.add_argument("--postproc-gap-small", type=int, default=2)

    parser.add_argument("--circle-threshold-multiplier", type=float, default=1.33)
    parser.add_argument("--donut-threshold-multiplier", type=float, default=1.33)
    parser.add_argument("--lower-left-threshold-multiplier", type=float, default=1.33)
    parser.add_argument("--horizontal-threshold-multiplier", type=float, default=1.2)
    parser.add_argument("--vertical-threshold-multiplier", type=float, default=1.2)

    parser.add_argument(
        "--support-filter",
        action="store_true",
        default=False,
        help="If set, apply pseudocell-radius support filter to candidates before clustering.",
    )

    parser.add_argument(
        "--support-radius-bins",
        type=int,
        default=1,
        help="Chebyshev radius in bins for candidate support filter.",
    )

    parser.add_argument(
        "--support-ratio",
        type=float,
        default=0.5,
        help="Threshold ratio for candidate support filter.",
    )

    parser.add_argument("--clustering-gap", type=int, default=1)
    parser.add_argument("--summit-gap", type=int, default=20_000)

    parser.add_argument("--filter-file", default=None)
    parser.add_argument("--max-memory", type=float, default=2)
    parser.add_argument("--verbose", type=int, default=0)

    parser.add_argument("--no-hic", action="store_true", default=True)
    parser.add_argument("--make-hic", action="store_true", default=False)
    parser.add_argument("--no-cool", action="store_true", default=True)
    parser.add_argument("--make-cool", action="store_true", default=False)

    return parser


if __name__ == "__main__":
    main()