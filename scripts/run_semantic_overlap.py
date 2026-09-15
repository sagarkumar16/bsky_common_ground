"""
run_semantic_overlap.py
=======================
CLI for the semantic normalization of shared starterpacks (see pack_semantics).

Takes the pair table produced by the existing census/cosine pipeline
(``user_a, user_b, shared_packs, cosine``), attaches the effective number of
semantically distinct shared packs (``n_eff``) to every pair, writes the
augmented table, and produces a similarity-vs-``n_eff`` view that REUSES the
existing aggregation / bootstrap / plotting functions unchanged.

It also fits a small OLS (cosine ~ shared_packs, ~ n_eff, ~ both) so one can ask
whether the *diversity* of common ground predicts content similarity beyond the
raw shared-pack count.

Usage
-----
    python run_semantic_overlap.py \
        --pairs-path        output/cosine_overlap_new_sampling_pairs_seed42.parquet \
        --starterpacks-path bluesky-graph/starterpacks.jsonl \
        --method            cluster_average \
        --out               output/semantic_overlap.parquet

If --pairs-path is omitted, the existing census/cosine functions are called
read-only to produce the table first (slower: recomputes user-post cosine).
"""

import sys
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless / cluster use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the repo root importable so `spcg` resolves when invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spcg.pack_semantics as ps
from spcg.overlap_info import (
    build_user_pack_membership,
    cluster_bootstrap_ci,
    load_starterpacks,
    plot_info_vs_overlap,
    run_overlap_cosine_census,
)


def _save_table(df, path_no_ext):
    """Write a DataFrame to parquet if the engine is available, else CSV."""
    try:
        out = path_no_ext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except Exception:
        out = path_no_ext.with_suffix(".csv")
        df.to_csv(out, index=False)
    print(f"  wrote {out}")
    return out


def _read_pairs(pairs_path):
    p = Path(pairs_path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _ols(y, cols):
    """OLS via lstsq with an intercept. cols: list of (name, array). Returns
    (coef dict incl 'intercept', R^2)."""
    X = np.column_stack([np.ones(len(y))] + [c for _, c in cols])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    names = ["intercept"] + [n for n, _ in cols]
    return dict(zip(names, (float(b) for b in beta))), r2


@click.command()
@click.option(
    "--pairs-path",
    type=click.Path(dir_okay=False),
    help="Saved pair table (user_a, user_b, shared_packs, cosine). "
    "If omitted, the census/cosine pipeline is run to produce it.",
)
@click.option(
    "--augmented-path",
    default=None,
    type=click.Path(dir_okay=False),
    help="AGGREGATE-ONLY mode: load an already-computed augmented table "
    "(with the n_eff_* columns) and just regenerate the "
    "similarity-vs-n_eff figure + OLS report -- skipping the "
    "embedding + per-pair n_eff computation. Use to (re)produce "
    "the per-run summary from a cached n_eff table without redoing "
    "the expensive work. --starterpacks-path/--records-dir are "
    "then not needed.",
)
@click.option(
    "--starterpacks-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to starterpacks.jsonl (for descriptions + membership). "
    "Required unless --augmented-path is given.",
)
@click.option(
    "--records-dir",
    type=click.Path(file_okay=False),
    help="Per-user records dir; required only when --pairs-path is omitted.",
)
@click.option(
    "--model-name",
    default="all-MiniLM-L6-v2",
    show_default=True,
    help="SentenceTransformer model (use a multilingual one if descriptions are).",
)
@click.option(
    "--method",
    default="cluster_average",
    show_default=True,
    type=click.Choice(
        [
            "cluster_average",
            "cluster_single",
            "cluster_complete",
            "effrank",
            "rowsum",
            "participation",
        ]
    ),
    help="Primary n_eff estimator aliased to the 'n_eff' column "
    "(default: average-linkage cluster count). The eigenvalue "
    "family (effrank/rowsum/participation) is always stored too.",
)
@click.option(
    "--missing",
    default="distinct",
    show_default=True,
    type=click.Choice(["distinct", "drop"]),
    help="Policy for packs with no usable description.",
)
@click.option(
    "--use-specificity",
    is_flag=True,
    default=False,
    help="Weight pack contributions by IDF-on-packs specificity.",
)
@click.option(
    "--cache-path",
    default=None,
    help="npz embedding cache (default: <data-dir>/pack_emb_<model>.npz).",
)
@click.option(
    "--out",
    "out_path",
    default="output/semantic_overlap.parquet",
    show_default=True,
    help="Base path for the run; the figure (pdf) is written next to "
    "it. The augmented pair table goes to --data-dir.",
)
@click.option(
    "--data-dir",
    default=None,
    help="Directory for the large augmented pair table + npz cache. "
    "Defaults to the figure directory (set to e.g. "
    "/scratch/.../study_data on the cluster).",
)
@click.option(
    "--max-shared",
    default=5,
    show_default=True,
    type=int,
    help="Top bin for the binned n_eff view (>= this is lumped).",
)
@click.option(
    "--n-boot",
    default=500,
    show_default=True,
    type=int,
    help="User-clustered bootstrap resamples for the CIs.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    type=int,
    help="Seed (bootstrap, and census pair selection if regenerated).",
)
# census params used only when --pairs-path is omitted
@click.option("--per-level", default=1000, show_default=True, type=int)
@click.option("--baseline-pairs", default=1000, show_default=True, type=int)
@click.option("--min-tokens", default=50, show_default=True, type=int)
@click.option("--n-jobs", default=1, show_default=True, type=int)
@click.option(
    "--english-only/--no-english-only",
    default=True,
    show_default=True,
    help="Keep only English-detected posts (langid). On by default; "
    "only applies when regenerating pairs (no --pairs-path).",
)
def main(
    pairs_path,
    augmented_path,
    starterpacks_path,
    records_dir,
    model_name,
    method,
    missing,
    use_specificity,
    cache_path,
    out_path,
    data_dir,
    max_shared,
    n_boot,
    seed,
    per_level,
    baseline_pairs,
    min_tokens,
    n_jobs,
    english_only,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir else out_path.parent
    data_dir.mkdir(parents=True, exist_ok=True)
    model_safe = model_name.replace("/", "_")
    if cache_path is None:
        cache_path = data_dir / f"pack_emb_{model_safe}.npz"

    # AGGREGATE-ONLY: reuse an already-computed augmented (n_eff) table; skip the
    # embedding + per-pair n_eff pass and just re-derive the figure + OLS report.
    if augmented_path:
        print(f"Aggregate-only mode: loading augmented table from {augmented_path} ...")
        out = _read_pairs(augmented_path)
        neff_col = f"n_eff_{method}"
        needed = {"user_a", "user_b", "cosine", neff_col}
        missing_cols = needed - set(out.columns)
        if missing_cols:
            raise click.UsageError(
                f"augmented table is missing column(s): {sorted(missing_cols)} "
                f"(is --method '{method}' the one it was built with?)"
            )
        print(f"loaded {len(out):,} pairs")
        if "k_shared" in out.columns:
            _report_invariants(out, method)
        _emit_semantic(out, out_path, model_safe, method, max_shared, n_boot, seed)
        print(f"\nDone (aggregate-only). Figure regenerated; reused {augmented_path}")
        return

    if not starterpacks_path:
        raise click.UsageError(
            "--starterpacks-path is required unless --augmented-path is given "
            "(aggregate-only mode)."
        )

    # 1. Pair table: read the saved one (preferred) or regenerate it.
    if pairs_path:
        print(f"Reading pair table from {pairs_path} ...")
        pair_df = _read_pairs(pairs_path)
    else:
        if not records_dir:
            raise click.UsageError(
                "--records-dir is required when --pairs-path is omitted"
            )
        print("No --pairs-path; running census/cosine pipeline to build it ...")
        _agg, pair_df = run_overlap_cosine_census(
            starterpacks_path=starterpacks_path,
            records_dir=records_dir,
            per_level=per_level,
            baseline_pairs=baseline_pairs,
            max_shared=max_shared,
            min_tokens=min_tokens,
            tfidf_kwargs=dict(min_df=10, max_df=0.4, sublinear_tf=True),
            n_boot=n_boot,
            seed=seed,
            plot=False,
            n_jobs=n_jobs,
            english_only=english_only,
        )
    for col in ("user_a", "user_b", "cosine"):
        if col not in pair_df.columns:
            raise click.UsageError(f"pair table is missing required column '{col}'")

    # 2. Membership over ALL users (pack_id == load_starterpacks line index).
    print("Loading starterpacks + membership ...")
    n_packs = ps.verify_alignment(
        starterpacks_path, load_starterpacks=load_starterpacks
    )
    packs, _ = load_starterpacks(starterpacks_path)
    membership = build_user_pack_membership(packs, users=None)
    print(f"  {n_packs:,} packs; pack_id alignment verified.")

    # 3. Pack-description embeddings (cached; second run does not re-embed).
    print(f"Embedding pack descriptions with '{model_name}' (cache: {cache_path}) ...")
    descriptions = ps.load_pack_descriptions(starterpacks_path)
    n_text = sum(1 for t in descriptions.values() if t and t.strip())
    E, _pack_ids, row_of = ps.embed_packs(
        descriptions, model_name=model_name, cache_path=str(cache_path)
    )
    print(
        f"  embedded {len(row_of):,}/{len(descriptions):,} packs with usable text "
        f"({n_text:,} non-empty); missing policy = '{missing}'."
    )

    # 4. Optional specificity weighting (independent of redundancy).
    specificity = ps.pack_specificity(membership) if use_specificity else None

    # 5. Attach n_eff (all three estimators) to every pair.
    print("Computing n_eff per pair ...")
    out = ps.n_eff_for_pairs(
        pair_df,
        membership,
        E,
        row_of,
        method=method,
        missing=missing,
        specificity=specificity,
    )

    # Invariant report: n_eff==0 iff k==0; ==1 iff k==1; else 1 <= n_eff <= k.
    _report_invariants(out, method)

    # 6. Persist the augmented table (large -> data dir; seed + model in name).
    stem = out_path.stem
    written = _save_table(out, data_dir / f"{stem}_{model_safe}_seed{seed}")

    # 7-8. Similarity-vs-n_eff figure + OLS report (shared with aggregate-only).
    _emit_semantic(out, out_path, model_safe, method, max_shared, n_boot, seed)

    print(f"\nDone. Augmented table: {written}")


def _emit_semantic(out, out_path, model_safe, method, max_shared, n_boot, seed):
    """Similarity-vs-n_eff binned view (figure) + OLS report, reusing the
    existing bootstrap + plotting functions. Shared by the full and
    aggregate-only paths so a rerun/directed run always emits the per-run figure."""
    # 7. Similarity-vs-n_eff view, reusing the existing bootstrap + plot.
    neff_col = f"n_eff_{method}"
    binned = out.copy()
    binned["neff_bin"] = np.rint(binned[neff_col]).clip(0, max_shared).astype(int)
    agg = cluster_bootstrap_ci(
        binned,
        value_col="cosine",
        group_col="neff_bin",
        n_boot=n_boot,
        seed=seed,
        max_shared=max_shared,
    )
    print("\nsimilarity vs n_eff (binned, user-clustered bootstrap CIs):")
    print(agg.to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    plot_info_vs_overlap(
        agg, ax=ax, info_label="mean cosine similarity", max_shared=max_shared
    )
    ax.set_xlabel(
        f"effective number of distinct shared packs " f"(n_eff, {method}, binned)"
    )
    plt.tight_layout()
    plot_out = out_path.parent / f"{out_path.stem}_{model_safe}_seed{seed}.pdf"
    fig.savefig(plot_out, dpi=150)
    plt.close(fig)
    print(f"  wrote {plot_out}")

    # 8. Does diversity of common ground predict similarity beyond raw count?
    _report_models(out, method)


def _report_invariants(out, method):
    k = out["k_shared"].to_numpy()
    for col in ("n_eff_effrank", "n_eff_rowsum", "n_eff_participation"):
        v = out[col].to_numpy()
        bad_zero = int(np.sum((v == 0) != (k == 0)))
        bad_one = int(
            np.sum((np.isclose(v, 1.0)) & (k != 1) & (k != 0))
        )  # informational
        over = int(np.sum(v > k + 1e-6))
        under = int(np.sum((k >= 1) & (v < 1.0 - 1e-6)))
        flag = "OK" if (bad_zero == 0 and over == 0 and under == 0) else "CHECK"
        print(
            f"  [{flag}] {col}: zero-mismatch={bad_zero}, n_eff>k={over}, "
            f"(k>=1 & n_eff<1)={under}"
        )


def _report_models(out, method):
    d = out[out["k_shared"] > 0].copy()  # diversity only defined where packs are shared
    if len(d) < 3:
        print("\n[model] too few shared-pack pairs to fit.")
        return
    y = d["cosine"].to_numpy(dtype=float)
    sp = (
        d["shared_packs"].to_numpy(dtype=float)
        if "shared_packs" in d
        else d["k_shared"].to_numpy(float)
    )
    ne = d[f"n_eff_{method}"].to_numpy(dtype=float)
    print(
        "\n[model] cosine ~ predictor(s) on shared-pack pairs "
        f"(n={len(d)}, n_eff={method}):"
    )
    c1, r1 = _ols(y, [("shared_packs", sp)])
    c2, r2 = _ols(y, [("n_eff", ne)])
    c3, r3 = _ols(y, [("shared_packs", sp), ("n_eff", ne)])
    print(f"  shared_packs        : R^2={r1:.4f}  beta={c1['shared_packs']:+.5f}")
    print(f"  n_eff               : R^2={r2:.4f}  beta={c2['n_eff']:+.5f}")
    print(
        f"  shared_packs + n_eff: R^2={r3:.4f}  "
        f"beta_count={c3['shared_packs']:+.5f}  beta_neff={c3['n_eff']:+.5f}"
    )
    print(f"  -> n_eff adds dR^2={r3 - r1:+.4f} over count alone.")


if __name__ == "__main__":
    main()
