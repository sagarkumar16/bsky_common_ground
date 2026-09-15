"""
tfidf_overlap.py
================
CLI wrapper for the TF-IDF cosine overlap analysis from notebook
04-sk-test-filter.ipynb ("Rerunning with TF-IDF" section).

Computes mean TF-IDF cosine similarity between user pairs as a function of how
many starterpacks they share, then saves the plot to disk. Pairs are selected by
CENSUS (every shared-pack bin is enumerated/populated rather than hoped for from
a random user slice) and error bars are USER-CLUSTERED bootstrap CIs (each user
appears in many pairs, so pair-level SEM badly understates uncertainty).

Usage
-----
    python tfidf_overlap.py \
        --starterpacks bluesky-graph/starterpacks.jsonl \
        --records-dir  bluesky-graph/records \
        --output       output/cosine_overlap.pdf
"""

import os
import sys
from pathlib import Path

import click
import matplotlib
import pandas as pd

matplotlib.use("Agg")  # non-interactive backend for headless / cluster use
import matplotlib.pyplot as plt

# Make the repo root importable so `spcg` resolves when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spcg.overlap_info import (
    cluster_bootstrap_ci,
    plot_info_vs_overlap,
    run_overlap_cosine_census,
)


@click.command()
@click.option(
    "--starterpacks",
    "starterpacks_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to starterpacks.jsonl (required unless --pairs-path is given)",
)
@click.option(
    "--records-dir",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Root directory of per-user record gzip files "
    "(required unless --pairs-path is given)",
)
@click.option(
    "--pairs-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="AGGREGATE-ONLY mode: skip the census/cosine computation and instead "
    "load this already-scored per-pair table (parquet/csv with user_a, "
    "user_b, shared_packs, cosine) to regenerate the agg table + figure. "
    "Use to (re)produce the summary from a cached pairs table without "
    "redoing the expensive TF-IDF pass. --starterpacks/--records-dir are "
    "then not needed.",
)
@click.option(
    "--output",
    "output_path",
    default="output/cosine_overlap.pdf",
    show_default=True,
    help="Destination for the saved plot (pdf/png/svg). The small agg table is "
    "written alongside it; the large per-pair table goes to --data-dir.",
)
@click.option(
    "--data-dir",
    default=None,
    help="Directory for the large per-pair data table. Defaults to the figure "
    "directory (set this to e.g. /scratch/.../study_data on the cluster).",
)
@click.option(
    "--per-level",
    default=1000,
    show_default=True,
    type=int,
    help="Cap on pairs enumerated per shared-pack level (keep all if fewer exist)",
)
@click.option(
    "--baseline-pairs",
    default=1000,
    show_default=True,
    type=int,
    help="Number of disjoint (zero shared pack) pairs to draw for the bin-0 baseline",
)
@click.option(
    "--max-shared",
    default=5,
    show_default=True,
    type=int,
    help="Lump pairs sharing >= this many packs into the top bin",
)
@click.option(
    "--min-tokens",
    default=50,
    show_default=True,
    type=int,
    help="Drop users with fewer than this many whitespace tokens of usable text",
)
@click.option(
    "--max-posts-per-user",
    default=None,
    type=int,
    help="Cap each user's history to this many most-recent posts (None = no cap)",
)
@click.option(
    "--n-boot",
    default=500,
    show_default=True,
    type=int,
    help="Number of user-clustered bootstrap resamples for the CIs",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Random seed for reproducible pair selection and bootstrap",
)
@click.option(
    "--n-jobs",
    default=0,
    show_default=True,
    type=int,
    help="Worker processes for reading user records (the dominant I/O cost). "
    "0 or negative = use all available CPUs (or $SLURM_CPUS_PER_TASK)",
)
@click.option(
    "--english-only/--no-english-only",
    default=True,
    show_default=True,
    help="Keep only English-detected posts (langid); content similarity is only "
    "meaningful within one language. On by default; pass --no-english-only "
    "to keep all languages.",
)
def main(
    starterpacks_path,
    records_dir,
    pairs_path,
    output_path,
    data_dir,
    per_level,
    baseline_pairs,
    max_shared,
    min_tokens,
    max_posts_per_user,
    n_boot,
    seed,
    n_jobs,
    english_only,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir else output_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)

    if pairs_path:
        # ---- AGGREGATE-ONLY: reuse an already-scored pairs table ------------
        # Skips the expensive census + TF-IDF cosine pass; just re-derives the
        # per-bin agg (cheap bootstrap over the cached pairs) and the figure.
        print(f"Aggregate-only mode: loading scored pairs from {pairs_path} ...")
        pair_df = _load_table(Path(pairs_path))
        missing = {"user_a", "user_b", "shared_packs", "cosine"} - set(pair_df.columns)
        if missing:
            raise click.ClickException(
                f"{pairs_path} is missing required column(s): {sorted(missing)}"
            )
        print(
            f"loaded {len(pair_df):,} scored pairs across shared-pack levels "
            f"{sorted(pair_df['shared_packs'].unique())}"
        )
        agg = cluster_bootstrap_ci(
            pair_df,
            value_col="cosine",
            group_col="shared_packs",
            n_boot=n_boot,
            seed=seed,
            max_shared=max_shared,
        )
        _emit(agg, pair_df, output_path, data_dir, max_shared, seed, write_pairs=False)
        return

    if not starterpacks_path or not records_dir:
        raise click.ClickException(
            "--starterpacks and --records-dir are required unless --pairs-path "
            "is given (aggregate-only mode)."
        )

    if n_jobs <= 0:
        env = os.environ.get("SLURM_CPUS_PER_TASK")
        n_jobs = int(env) if env and env.isdigit() else (os.cpu_count() or 1)

    # Sensible TF-IDF defaults for the census corpus: prune rare/ubiquitous terms
    # and damp prolific accounts.
    tfidf_kwargs = dict(min_df=10, max_df=0.4, sublinear_tf=True)

    print(
        f"Running census overlap-vs-cosine analysis "
        f"({n_jobs} worker process(es), seed={seed}) ..."
    )
    agg, pair_df = run_overlap_cosine_census(
        starterpacks_path=starterpacks_path,
        records_dir=records_dir,
        per_level=per_level,
        baseline_pairs=baseline_pairs,
        max_shared=max_shared,
        min_tokens=min_tokens,
        max_posts_per_user=max_posts_per_user,
        tfidf_kwargs=tfidf_kwargs,
        n_boot=n_boot,
        seed=seed,
        plot=False,
        n_jobs=n_jobs,
        english_only=english_only,
    )
    _emit(agg, pair_df, output_path, data_dir, max_shared, seed, write_pairs=True)


def _emit(agg, pair_df, output_path, data_dir, max_shared, seed, write_pairs):
    """Print the agg, save the figure, and persist the tables."""
    print(agg.to_string(index=False))

    print(f"Saving plot to {output_path} ...")
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    plot_info_vs_overlap(
        agg,
        ax=ax,
        info_label="mean cosine similarity",
        max_shared=max_shared,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    # Persist the tables next to the figure, seed in the filename for provenance.
    stem = output_path.stem
    out_dir = output_path.parent
    _save_table(agg, out_dir / f"{stem}_agg_seed{seed}")  # small summary -> figure dir
    if write_pairs:
        _save_table(
            pair_df, data_dir / f"{stem}_pairs_seed{seed}"
        )  # large per-pair table -> data dir
    print("Done.")


def _load_table(path):
    """Load a parquet or csv pairs table (by extension)."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _save_table(df, path_no_ext):
    """Write a DataFrame to parquet if the engine is available, else CSV."""
    try:
        out = path_no_ext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except Exception:
        out = path_no_ext.with_suffix(".csv")
        df.to_csv(out, index=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
