"""
run_network_analysis.py
=======================
Network position as a moderator of communal common ground.

Given a census pair table with content similarity ``cosine``, effective common
ground ``n_eff`` (or another ``--cg-col``), and raw ``shared_packs``, this adds
follow-graph distance ``hops`` and runs three analyses:

  (A) Distance baseline (descriptive): mean cosine per hops level, with
      node-level cluster-bootstrap CIs (reuses cluster_bootstrap_ci).
  (B) Distance as control: does common ground survive conditioning on distance?
      cosine ~ cg  vs  cosine ~ cg + C(hops); the cg coefficient is reported
      both ways. CIs from the node-level cluster bootstrap (not naive OLS SEs).
  (C) Distance as moderator (headline): cosine ~ cg * C(hops), plus the
      per-stratum cg -> cosine slope for each hops level, with bootstrap CIs.

Inference everywhere uses the SAME node-level (dyadic) cluster bootstrap as
cluster_bootstrap_ci -- resample USERS with replacement, weight each pair by the
product of its endpoints' draw multiplicities -- because pairs are dyadically
dependent. Identical ``--seed`` reproduces identical tables and figures.

Theory: ``cg`` (default ``n_eff``) operationalizes Clark's *communal* common
ground; ``hops`` is the network-position dimension. A flat cg->cosine slope
across distance means affiliation confers common ground uniformly; a slope that
varies with distance means proximity moderates it. ``cosine`` is a proxy for the
construct, not the construct.

Usage
-----
    python run_network_analysis.py \
        --pairs-path    output/semantic_overlap_..._seed42.parquet \
        --edgelist-path data/filtered_follows.parquet \
        --out           output/network_analysis
"""

import re
import sys
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # headless / cluster
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

try:
    from tqdm import tqdm
except Exception:  # tqdm absent -> no-op passthrough

    def tqdm(it=None, **k):
        return it if it is not None else iter(())


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spcg.network_distance as nd
from spcg.overlap_info import cluster_bootstrap_ci, plot_info_vs_overlap


# ---------------------------------------------------------------------------
# Node-level (dyadic) cluster bootstrap of arbitrary per-weight statistics
# ---------------------------------------------------------------------------
def node_cluster_bootstrap(pairs_df, stat_fn, n_boot=500, seed=42):
    """
    Reuses cluster_bootstrap_ci's resampling EXACTLY: resample users with
    replacement, weight each pair by the product of its two endpoints' draw
    multiplicities, recompute ``stat_fn(weights)`` (a dict of named scalars).

    Returns a tidy frame [term, estimate, ci_low, ci_high] where ``estimate`` is
    the all-weights-one point value and CIs are 2.5/97.5 nan-aware percentiles.
    """
    users = pd.unique(pd.concat([pairs_df["user_a"], pairs_df["user_b"]]))
    user_idx = {u: i for i, u in enumerate(users)}
    n_users = len(users)
    a_idx = pairs_df["user_a"].map(user_idx).to_numpy()
    b_idx = pairs_df["user_b"].map(user_idx).to_numpy()

    point = stat_fn(np.ones(len(pairs_df)))
    keys = list(point.keys())

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(keys)), np.nan)
    for bi in tqdm(range(n_boot), desc="cluster bootstrap", unit="rep"):
        draws = rng.integers(0, n_users, size=n_users)
        mult = np.bincount(draws, minlength=n_users)
        w = (mult[a_idx] * mult[b_idx]).astype(float)
        s = stat_fn(w)
        boot[bi] = [s.get(k, np.nan) for k in keys]

    rows = []
    for j, k in enumerate(keys):
        col = boot[:, j]
        if np.all(np.isnan(col)):
            lo = hi = np.nan
        else:
            lo = float(np.nanpercentile(col, 2.5))
            hi = float(np.nanpercentile(col, 97.5))
        rows.append((k, float(point[k]), lo, hi))
    return pd.DataFrame(rows, columns=["term", "estimate", "ci_low", "ci_high"])


def _wls_beta(X, y, w):
    """Weighted least squares coefficients via sqrt-weight scaling."""
    sw = np.sqrt(w)
    try:
        beta, *_ = np.linalg.lstsq(sw[:, None] * X, sw * y, rcond=None)
        return beta
    except Exception:
        return np.full(X.shape[1], np.nan)


def _make_stat_fn(cosine, cg, hops, levels):
    """
    Build the per-weight statistic used by every bootstrap pass, so (B) and (C)
    share one resampling (hence one seed). Returns a dict of:
      cg|no_hops, cg|with_hops          -- (B) cg coef without / with C(hops)
      slope|<lvl>                        -- (C) per-stratum cg->cosine slope
      mean|<lvl>|<cg_stratum>            -- (C) cell means for the cg strata
                                            {n_eff=0, n_eff[1,3], n_eff>3}
    """
    n = len(cosine)
    ref = levels[0]
    dummies = [(hops == lvl).astype(float) for lvl in levels[1:]]

    X_base = np.column_stack([np.ones(n), cg])
    X_full = np.column_stack([np.ones(n), cg] + dummies) if dummies else X_base

    # Three fixed common-ground strata (n_eff is 0 for k=0 and >=1 for k>=1, so
    # there is no gap in (0,1) -- these partition every pair).
    cg_strata = [
        ("n_eff=0", cg < 1.0),
        ("n_eff[1,3]", (cg >= 1.0) & (cg <= 3.0)),
        ("n_eff>3", cg > 3.0),
    ]

    def stat(w):
        out = {}
        out["cg|no_hops"] = _wls_beta(X_base, cosine, w)[1]
        out["cg|with_hops"] = _wls_beta(X_full, cosine, w)[1]
        for lvl in levels:
            m = hops == lvl
            ww = w * m
            if ww.sum() > 0 and m.sum() >= 2 and np.ptp(cg[m]) > 0:
                out[f"slope|{lvl}"] = _wls_beta(X_base, cosine, ww)[1]
            else:
                out[f"slope|{lvl}"] = np.nan
            for label, smask in cg_strata:
                ww2 = w * (m & smask)
                tot = ww2.sum()
                out[f"mean|{lvl}|{label}"] = (
                    float((ww2 * cosine).sum() / tot) if tot > 0 else np.nan
                )
        return out

    meta = dict(ref=ref, cg_strata=[lab for lab, _ in cg_strata])
    return stat, meta


def _ols_named(cosine, design_cols):
    """statsmodels OLS on a named design (list of (name, array)) for a readable
    point summary. Returns the fitted results."""
    names = [n for n, _ in design_cols]
    X = pd.DataFrame({n: a for n, a in design_cols})
    return sm.OLS(cosine, X[names]).fit(), names


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--pairs-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Pair table with user_a,user_b,cosine,<cg-col>,shared_packs.",
)
@click.option(
    "--edgelist-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Persisted reduced follow edgelist (parquet/csv: source,target). "
    "Required unless --hops-path is given.",
)
@click.option(
    "--hops-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Precomputed pairs_with_hops table (carrying a 'hops' column). "
    "If given, BFS and the graph are SKIPPED entirely -- the "
    "compute-once / analyze-many workflow (see --hops-only).",
)
@click.option(
    "--hops-only",
    is_flag=True,
    default=False,
    help="Compute hops, write pairs_with_hops, then exit before the "
    "analyses. Run this once (heavy), then iterate with --hops-path.",
)
@click.option("--max-hops", default=4, show_default=True, type=int)
@click.option(
    "--directed",
    is_flag=True,
    default=False,
    help="Keep follow direction (default: symmetrize).",
)
@click.option(
    "--cg-col",
    default="n_eff",
    show_default=True,
    help="Common-ground column (e.g. n_eff, n_eff_effrank, shared_packs).",
)
@click.option("--n-boot", default=500, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option(
    "--n-jobs",
    default=4,
    show_default=True,
    type=int,
    help="Workers for BFS. 0/<0 = $SLURM_CPUS_PER_TASK or all CPUs.",
)
@click.option(
    "--out",
    "out_dir",
    default="output/network_analysis",
    show_default=True,
    help="Directory for figures + small summary tables.",
)
@click.option(
    "--data-dir",
    default=None,
    help="Directory for the large augmented pair table (pairs_with_hops). "
    "Defaults to --out (set to e.g. /scratch/.../study_data on "
    "the cluster).",
)
def main(
    pairs_path,
    edgelist_path,
    hops_path,
    hops_only,
    max_hops,
    directed,
    cg_col,
    n_boot,
    seed,
    n_jobs,
    out_dir,
    data_dir,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(data_dir) if data_dir else out_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cg_col}_seed{seed}"
    sentinel = max_hops + 1

    def _read(path):
        return (
            pd.read_parquet(path)
            if Path(path).suffix in (".parquet", ".pq")
            else pd.read_csv(path)
        )

    # ---- get hops (compute via BFS, or reuse a precomputed table) --------
    if hops_path:
        # Compute-once / analyze-many: reuse a prior pairs_with_hops; skip the
        # graph + BFS entirely.
        merged = _read(hops_path)
        if "hops" not in merged.columns:
            raise click.UsageError(f"{hops_path} has no 'hops' column")
        for c in ("user_a", "user_b", "cosine", cg_col):
            if c not in merged.columns:
                raise click.UsageError(f"hops table missing required column '{c}'")
        print(
            f"loaded {len(merged):,} pairs WITH precomputed hops from {hops_path} "
            f"(graph BFS skipped)"
        )
    else:
        if not edgelist_path:
            raise click.UsageError("provide --edgelist-path (or --hops-path)")
        pairs = _read(pairs_path)
        for c in ("user_a", "user_b", "cosine", cg_col):
            if c not in pairs.columns:
                raise click.UsageError(f"pairs table missing required column '{c}'")
        print(f"loaded {len(pairs):,} pairs from {pairs_path}")

        print("building follow graph (CSR) from persisted edgelist ...")
        graph = nd.load_follow_graph(edgelist_path, directed=directed)
        print(f"  graph: {graph.n:,} nodes, {len(graph.indices):,} directed CSR edges")

        merged = nd.pair_distances(
            pairs, graph, max_hops=max_hops, n_jobs=n_jobs, directed=directed
        )

    merged["hops_bin"] = nd.bin_hops(merged, max_hops)

    # Persist the (heavy) hops result up front so it can be reused via
    # --hops-path even if the analyses are interrupted; --hops-only stops here.
    if not hops_path:
        aug = merged.copy()
        aug["hops_bin"] = aug["hops_bin"].astype(str)
        _save_table(aug, data_dir / f"pairs_with_hops_{tag}")
        if hops_only:
            print(
                "\n[hops-only] wrote pairs_with_hops; re-run with "
                "--hops-path to do the analyses. Done."
            )
            return

    # analysis arrays
    cosine = merged["cosine"].to_numpy(dtype=float)
    cg = merged[cg_col].to_numpy(dtype=float)
    hops = merged["hops"].to_numpy(dtype=int)
    levels = sorted(np.unique(hops).tolist())  # ints; sentinel last
    level_label = {
        lvl: ("unreachable" if lvl >= sentinel else str(lvl)) for lvl in levels
    }

    # ===== (A) distance baseline ==========================================
    print("\n=== (A) distance baseline: mean cosine by hops (cluster bootstrap) ===")
    agg_A = cluster_bootstrap_ci(
        merged,
        value_col="cosine",
        group_col="hops",
        n_boot=n_boot,
        seed=seed,
        max_shared=sentinel,
    )
    print(agg_A.to_string(index=False))
    agg_A.to_csv(out_dir / f"A_distance_baseline_{tag}.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    plot_info_vs_overlap(
        agg_A, ax=ax, info_label="mean cosine similarity", max_shared=sentinel
    )
    ax.set_xlabel("follow-graph distance (hops)")
    ax.set_xticks(list(agg_A["shared_packs"]))
    ax.set_xticklabels([level_label.get(int(l), str(l)) for l in agg_A["shared_packs"]])
    ax.set_title("(A) Content similarity by network distance")
    plt.tight_layout()
    _savefig(fig, out_dir / f"A_distance_baseline_{tag}.pdf")

    # ===== build shared bootstrap statistic ===============================
    stat_fn, meta = _make_stat_fn(cosine, cg, hops, levels)
    boot = node_cluster_bootstrap(merged, stat_fn, n_boot=n_boot, seed=seed)
    boot_by_term = {r["term"]: r for _, r in boot.iterrows()}

    # ===== (B) distance as control ========================================
    print("\n=== (B) does common ground survive controlling for distance? ===")
    dummies = [
        (f"hops[{level_label[lvl]}]", (hops == lvl).astype(float)) for lvl in levels[1:]
    ]
    res_base, _ = _ols_named(cosine, [("Intercept", np.ones(len(cosine))), ("cg", cg)])
    res_full, full_names = _ols_named(
        cosine, [("Intercept", np.ones(len(cosine))), ("cg", cg)] + dummies
    )

    b_no = boot_by_term["cg|no_hops"]
    b_with = boot_by_term["cg|with_hops"]
    cg_compare = pd.DataFrame(
        [
            {
                "model": "cosine ~ cg",
                "cg_coef": b_no["estimate"],
                "ci_low": b_no["ci_low"],
                "ci_high": b_no["ci_high"],
                "ols_coef": res_base.params["cg"],
            },
            {
                "model": "cosine ~ cg + C(hops)",
                "cg_coef": b_with["estimate"],
                "ci_low": b_with["ci_low"],
                "ci_high": b_with["ci_high"],
                "ols_coef": res_full.params["cg"],
            },
        ]
    )
    print(cg_compare.to_string(index=False))
    print(
        f"  -> cg coefficient shift when distance is added: "
        f"{b_with['estimate'] - b_no['estimate']:+.5f}"
    )
    cg_compare.to_csv(out_dir / f"B_cg_with_without_distance_{tag}.csv", index=False)

    full_tbl = _coef_table(res_full, boot, name_map={"cg": "cg|with_hops"})
    full_tbl.to_csv(out_dir / f"B_full_model_coefs_{tag}.csv", index=False)

    # (B) figure: cg coefficient with vs without distance (bootstrap CIs)
    fig, ax = plt.subplots(figsize=(5.5, 3.5), facecolor="white")
    ys = [0, 1]
    est = [b_no["estimate"], b_with["estimate"]]
    lo = [b_no["estimate"] - b_no["ci_low"], b_with["estimate"] - b_with["ci_low"]]
    hi = [b_no["ci_high"] - b_no["estimate"], b_with["ci_high"] - b_with["estimate"]]
    ax.errorbar(est, ys, xerr=[lo, hi], fmt="o", capsize=4, color="#185FA5")
    ax.axvline(0, color="#bbbbbb", lw=1, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(["cosine ~ cg", "cosine ~ cg + C(hops)"])
    ax.set_xlabel(f"{cg_col} coefficient (node cluster-bootstrap 95% CI)")
    ax.set_title("(B) Common ground with vs without distance control")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_dir / f"B_cg_coefficient_{tag}.pdf")

    # ===== (C) distance as moderator ======================================
    print("\n=== (C) distance as moderator: cosine ~ cg * C(hops) ===")
    inter = [
        (f"cg:hops[{level_label[lvl]}]", cg * (hops == lvl).astype(float))
        for lvl in levels[1:]
    ]
    res_int, _ = _ols_named(
        cosine, [("Intercept", np.ones(len(cosine))), ("cg", cg)] + dummies + inter
    )
    with open(out_dir / f"C_interaction_summary_{tag}.txt", "w") as f:
        f.write(_scrub_summary(str(res_int.summary())))
    print(
        f"  interaction model R^2={res_int.rsquared:.4f}; "
        f"summary -> C_interaction_summary_{tag}.txt"
    )

    # per-stratum slope table (point + bootstrap CIs)
    slope_rows = []
    for lvl in levels:
        r = boot_by_term[f"slope|{lvl}"]
        slope_rows.append(
            {
                "hops": level_label[lvl],
                "hops_int": lvl,
                "slope": r["estimate"],
                "ci_low": r["ci_low"],
                "ci_high": r["ci_high"],
                "n_pairs": int((hops == lvl).sum()),
            }
        )
    slope_tbl = pd.DataFrame(slope_rows)
    print(slope_tbl.to_string(index=False))
    slope_tbl.to_csv(out_dir / f"C_slope_by_distance_{tag}.csv", index=False)

    # (C) figure 1: slope by distance
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    x = np.arange(len(slope_tbl))
    yerr = [
        slope_tbl["slope"] - slope_tbl["ci_low"],
        slope_tbl["ci_high"] - slope_tbl["slope"],
    ]
    ax.errorbar(
        x,
        slope_tbl["slope"],
        yerr=yerr,
        fmt="o-",
        capsize=4,
        color="#333333",
        ecolor="#999999",
    )
    ax.axhline(0, color="#bbbbbb", lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels(slope_tbl["hops"])
    ax.set_xlabel("follow-graph distance (hops)")
    ax.set_ylabel(f"{cg_col} -> cosine slope")
    ax.set_title("(C) Does proximity moderate common ground?")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_dir / f"C_slope_by_distance_{tag}.pdf")

    # (C) figure 2: mean cosine by cg stratum {n_eff=0, [1,3], >3} within each
    # distance level -- one point per stratum, offset for readability.
    strata = meta["cg_strata"]  # ["n_eff=0","n_eff[1,3]","n_eff>3"]
    disp = {
        "n_eff=0": (f"{cg_col}=0", "#8c8c8c"),
        "n_eff[1,3]": (f"{cg_col}∈[1,3]", "#4C72B0"),
        "n_eff>3": (f"{cg_col}>3", "#C44E52"),
    }
    offs = np.linspace(-0.16, 0.16, len(strata))
    fig, ax = plt.subplots(figsize=(6.8, 4), facecolor="white")
    xs = np.arange(len(levels))
    for label, off in zip(strata, offs):
        lbl, color = disp.get(label, (label, None))
        est = [boot_by_term[f"mean|{lvl}|{label}"]["estimate"] for lvl in levels]
        lo = [
            boot_by_term[f"mean|{lvl}|{label}"]["estimate"]
            - boot_by_term[f"mean|{lvl}|{label}"]["ci_low"]
            for lvl in levels
        ]
        hi = [
            boot_by_term[f"mean|{lvl}|{label}"]["ci_high"]
            - boot_by_term[f"mean|{lvl}|{label}"]["estimate"]
            for lvl in levels
        ]
        ax.errorbar(
            xs + off, est, yerr=[lo, hi], fmt="o", capsize=3, color=color, label=lbl
        )
    # highlight the theoretically loaded cell: highest cg stratum (>3) with no
    # network contact (unreachable/far distance level)
    if levels and levels[-1] >= sentinel and "n_eff>3" in strata:
        off = offs[strata.index("n_eff>3")]
        hl = boot_by_term[f"mean|{levels[-1]}|n_eff>3"]["estimate"]
        if np.isfinite(hl):
            ax.scatter(
                [xs[-1] + off],
                [hl],
                s=180,
                facecolors="none",
                edgecolors="#C44E52",
                linewidths=2,
                zorder=5,
            )
            ax.annotate(
                "high common ground,\nno network contact",
                (xs[-1] + off, hl),
                textcoords="offset points",
                xytext=(-10, 14),
                fontsize=8,
                ha="right",
                color="#C44E52",
            )
    ax.set_xticks(xs)
    ax.set_xticklabels([level_label[l] for l in levels])
    ax.set_xlabel("follow-graph distance (hops)")
    ax.set_ylabel("mean cosine similarity")
    ax.set_title(f"(C) mean cosine by {cg_col} stratum and network distance")
    ax.legend(frameon=False, title=cg_col)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig(fig, out_dir / f"C_cg_strata_by_distance_{tag}.pdf")

    # (pairs_with_hops was already persisted up front, before the analyses.)
    print(
        f"\nDone. Figures + small tables in {out_dir}/; "
        f"pair table in {data_dir}/ (tag={tag})."
    )


def _coef_table(res, boot_df, name_map=None):
    """Coefficient table: OLS point + bootstrap CI where a bootstrap term maps
    to a model parameter (name_map: model_param -> bootstrap term)."""
    name_map = name_map or {}
    bt = {r["term"]: r for _, r in boot_df.iterrows()}
    rows = []
    for p in res.params.index:
        bkey = name_map.get(p)
        r = bt.get(bkey) if bkey else None
        rows.append(
            {
                "term": p,
                "ols_coef": float(res.params[p]),
                "boot_ci_low": (r["ci_low"] if r is not None else np.nan),
                "boot_ci_high": (r["ci_high"] if r is not None else np.nan),
            }
        )
    return pd.DataFrame(rows)


def _scrub_summary(text):
    """Replace the wall-clock Date/Time that statsmodels stamps into .summary()
    with fixed tokens, so the saved summary is byte-identical across reruns
    (same lengths -> alignment preserved). Numbers carry no colons/date words."""
    text = re.sub(r"\d{2}:\d{2}:\d{2}", "00:00:00", text)
    text = re.sub(r"\w{3}, \d{2} \w{3} \d{4}", "Xxx, 00 Xxx 0000", text)
    return text


def _savefig(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path}")


def _save_table(df, path_no_ext):
    try:
        out = path_no_ext.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    except Exception:
        out = path_no_ext.with_suffix(".csv")
        df.to_csv(out, index=False)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
