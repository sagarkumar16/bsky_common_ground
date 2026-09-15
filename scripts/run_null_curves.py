"""
run_null_curves.py
==================
Overlay a configuration-model NULL curve on BOTH overlap-vs-cosine plots:

  (raw)        mean cosine vs number of shared packs      (shared_packs)
  (normalized) mean cosine vs effective # distinct shared (n_eff)

The null rerandomizes pack MEMBERSHIP only (cosine is a fixed property of the
users' text and is held constant); under each null replicate we recompute BOTH
the raw shared-pack count and n_eff for every pair, re-bin, and take the mean
cosine per bin. Averaging over replicates gives the null curve + a 95% band.
Observed rising well above a flat null = the association exceeds what the
affiliation hypergraph's degree structure alone produces.

Both curves are computed on the SAME (census) pair set used by the observed
plots, so observed vs null are apples-to-apples on one axis. (This is the
descriptive conditional null for those plots; the regression-coefficient null in
run_null_model.py uses the selection-unbiased eval set instead.)

    python scripts/run_null_curves.py \
        --pairs-path <census pairs w/ user_a,user_b,cosine> \
        --starterpacks-path starterpacks.jsonl --embeddings-path pack_emb.npz \
        --ensemble canonical --n-rep 200 --out output/null_curves
"""

import sys
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spcg.null_model as nm
import spcg.pack_semantics as ps
from spcg.overlap_info import cluster_bootstrap_ci

try:
    from tqdm import tqdm
except Exception:

    def tqdm(it=None, **k):
        return it if it is not None else iter(())


def _read(path):
    p = Path(path)
    return pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)


def _save(df, path_no_ext):
    try:
        out = Path(f"{path_no_ext}.parquet")
        df.to_parquet(out, index=False)
    except Exception:
        out = Path(f"{path_no_ext}.csv")
        df.to_csv(out, index=False)
    print(f"  wrote {out}")


def _binned_mean(values, bins, n_bins):
    """Mean of `values` within each integer bin 0..n_bins (nan where empty)."""
    out = np.full(n_bins + 1, np.nan)
    for k in range(n_bins + 1):
        m = bins == k
        if m.any():
            out[k] = float(values[m].mean())
    return out


@click.command()
@click.option(
    "--pairs-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Census pair table with user_a,user_b,cosine (the same pairs "
    "the observed plots use).",
)
@click.option("--starterpacks-path", required=True, type=click.Path(dir_okay=False))
@click.option(
    "--embeddings-path", default=None, help="npz embedding cache (reused if complete)."
)
@click.option("--model-name", default="all-MiniLM-L6-v2", show_default=True)
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
    help="n_eff estimator (match the semantic stage).",
)
@click.option(
    "--missing",
    default="distinct",
    show_default=True,
    type=click.Choice(["distinct", "drop"]),
)
@click.option(
    "--ensemble",
    default="canonical",
    show_default=True,
    type=click.Choice(["canonical", "microcanonical"]),
)
@click.option(
    "--n-rep",
    default=200,
    show_default=True,
    type=int,
    help="Null replicates for the band.",
)
@click.option(
    "--max-shared",
    default=5,
    show_default=True,
    type=int,
    help="Top bin (>= this is lumped) for both axes.",
)
@click.option(
    "--n-boot",
    default=500,
    show_default=True,
    type=int,
    help="User-clustered bootstrap resamples for the OBSERVED CIs.",
)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--n-trades",
    default=None,
    type=int,
    help="Curveball trades per microcanonical draw (default ~5*nnz).",
)
@click.option("--out", "out_dir", default="output/null_curves", show_default=True)
def main(
    pairs_path,
    starterpacks_path,
    embeddings_path,
    model_name,
    method,
    missing,
    ensemble,
    n_rep,
    max_shared,
    n_boot,
    seed,
    n_trades,
    out_dir,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _read(pairs_path)
    for c in ("user_a", "user_b", "cosine"):
        if c not in pairs.columns:
            raise click.UsageError(f"pairs table missing required column '{c}'")
    print(f"loaded {len(pairs):,} pairs from {pairs_path}")

    print("loading incidence (full, all users) ...")
    B, user_ids, _pack_ids = nm.load_incidence(starterpacks_path)
    print(
        f"  incidence: {B.shape[0]:,} users x {B.shape[1]:,} packs, {B.nnz:,} memberships"
    )

    print("loading embeddings (cache reused if complete) ...")
    desc = ps.load_pack_descriptions(starterpacks_path)
    E, _ids, row_of = ps.embed_packs(
        desc, model_name=model_name, cache_path=embeddings_path
    )

    # map dids -> incidence row ids; keep aligned fixed cosine
    did2idx = {d: i for i, d in enumerate(user_ids.tolist())}
    a = pairs["user_a"].map(did2idx)
    b = pairs["user_b"].map(did2idx)
    keep = a.notna() & b.notna()
    if int((~keep).sum()):
        print(
            f"[pairs] dropped {int((~keep).sum())} with an endpoint not in the incidence"
        )
    idx_pairs = pd.DataFrame(
        {
            "user_a": a[keep].astype(np.int64).to_numpy(),
            "user_b": b[keep].astype(np.int64).to_numpy(),
        }
    )
    cosine = pairs.loc[keep, "cosine"].to_numpy(dtype=float)
    eval_users = pd.unique(
        pd.concat([idx_pairs["user_a"], idx_pairs["user_b"]])
    ).astype(np.int64)

    def curves_from_membership(memb):
        """(count_bin, neff_bin) per pair under a given membership."""
        neff = ps.n_eff_for_pairs(
            idx_pairs, memb, E, row_of, method=method, missing=missing
        )
        k = neff["k_shared"].to_numpy()
        ne = neff[f"n_eff_{method}"].to_numpy(dtype=float)
        cbin = np.clip(k, 0, max_shared).astype(int)
        nbin = np.clip(np.rint(ne), 0, max_shared).astype(int)
        return cbin, nbin

    # ---- observed curves (with user-clustered bootstrap CIs) --------------
    print("computing OBSERVED curves ...")
    obs_memb = nm._membership_from_rows(B, eval_users)
    obs_cbin, obs_nbin = curves_from_membership(obs_memb)
    dfo = idx_pairs.copy()
    dfo["cosine"] = cosine
    dfo["count_bin"] = obs_cbin
    dfo["neff_bin"] = obs_nbin
    obs_count = cluster_bootstrap_ci(
        dfo,
        value_col="cosine",
        group_col="count_bin",
        n_boot=n_boot,
        seed=seed,
        max_shared=max_shared,
    )
    obs_neff = cluster_bootstrap_ci(
        dfo,
        value_col="cosine",
        group_col="neff_bin",
        n_boot=n_boot,
        seed=seed,
        max_shared=max_shared,
    )

    # ---- null replicate curves -------------------------------------------
    print(f"drawing {n_rep} null replicates ({ensemble}) ...")
    x = y = None
    if ensemble == "canonical":
        x, y = nm.bicm_fit(B)
    null_count = np.full((n_rep, max_shared + 1), np.nan)
    null_neff = np.full((n_rep, max_shared + 1), np.nan)
    for r in tqdm(range(n_rep), desc="null curves", unit="rep"):
        s = seed + r
        if ensemble == "canonical":
            B_rand = nm.bicm_sample(x, y, s, rows=eval_users)
        else:
            B_rand = nm.curveball_randomize(B, n_trades=n_trades, seed=s)
        memb = nm._membership_from_rows(B_rand, eval_users)
        cbin, nbin = curves_from_membership(memb)
        null_count[r] = _binned_mean(cosine, cbin, max_shared)
        null_neff[r] = _binned_mean(cosine, nbin, max_shared)

    def agg(nullmat):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # empty (all-NaN) bins -> NaN, fine
            mean = np.nanmean(nullmat, axis=0)
            lo = np.nanpercentile(nullmat, 2.5, axis=0)
            hi = np.nanpercentile(nullmat, 97.5, axis=0)
        return mean, lo, hi

    nc_mean, nc_lo, nc_hi = agg(null_count)
    nn_mean, nn_lo, nn_hi = agg(null_neff)

    # ---- plot + csv -------------------------------------------------------
    _overlay(
        obs_count,
        (nc_mean, nc_lo, nc_hi),
        max_shared,
        "number of shared starterpacks (raw)",
        out_dir / "overlap_count_vs_cosine_null.pdf",
        "Raw shared-pack count vs cosine (observed vs null)",
    )
    _overlay(
        obs_neff,
        (nn_mean, nn_lo, nn_hi),
        max_shared,
        f"effective # distinct shared packs (n_eff, {method})",
        out_dir / "overlap_neff_vs_cosine_null.pdf",
        "Similarity-normalized (n_eff) vs cosine (observed vs null)",
    )

    rows = []
    for xkind, obs, (m, lo, hi) in (
        ("count", obs_count, (nc_mean, nc_lo, nc_hi)),
        ("n_eff", obs_neff, (nn_mean, nn_lo, nn_hi)),
    ):
        omap = {int(r.shared_packs): r for _, r in obs.iterrows()}
        for k in range(max_shared + 1):
            o = omap.get(k)
            rows.append(
                {
                    "x_kind": xkind,
                    "bin": k,
                    "observed_mean": (o["mean"] if o is not None else np.nan),
                    "observed_ci_low": (o["ci_low"] if o is not None else np.nan),
                    "observed_ci_high": (o["ci_high"] if o is not None else np.nan),
                    "null_mean": m[k],
                    "null_lo": lo[k],
                    "null_hi": hi[k],
                }
            )
    _save(pd.DataFrame(rows), out_dir / f"null_curves_{method}_seed{seed}")
    print("Done.")


def _overlay(obs, null_band, max_shared, xlabel, out_path, title):
    m, lo, hi = null_band
    bins = np.arange(max_shared + 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), facecolor="white")
    # null band + mean
    ax.fill_between(bins, lo, hi, color="#C44E52", alpha=0.18, label="null 95% band")
    ax.plot(bins, m, "--", color="#C44E52", lw=1.5, label="null mean")
    # observed
    ax.errorbar(
        obs["shared_packs"],
        obs["mean"],
        yerr=[obs["mean"] - obs["ci_low"], obs["ci_high"] - obs["mean"]],
        fmt="o-",
        capsize=4,
        color="#333333",
        ecolor="#999999",
        lw=1.6,
        label="observed (95% CI)",
    )
    labels = [str(k) if k < max_shared else f"≥{max_shared}" for k in bins]
    ax.set_xticks(bins)
    ax.set_xticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("mean cosine similarity")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
