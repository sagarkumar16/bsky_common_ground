"""
run_null_model.py
=================
CLI for the configuration-model null (see null_model.py).

  --mode sample : compute the OBSERVED statistic once, then draw a chunk of
                  null replicates (curveball or BiCM), recompute n_eff under
                  each, refit, and append per-replicate statistics to a per-task
                  parquet. Designed for a Slurm array job (one chunk per task).
  --mode reduce : read all per-task files + the observed value, compute
                  two-sided empirical p-values and z-scores, and write a summary
                  table + a null-distribution figure with the observed marked.

Only pack membership is rerandomized; cosine and hops are FIXED inputs (read
from the eval pair table). The follow graph is NEVER loaded.

  python scripts/run_null_model.py --mode sample \
      --ensemble canonical --statistic interaction \
      --eval-pairs-path eval.parquet --starterpacks-path starterpacks.jsonl \
      --embeddings-path pack_emb.npz --n-rep 1000 --n-tasks 50 \
      --task-id $SLURM_ARRAY_TASK_ID --seed-base 0 --out out_dir
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
    return out


def _load_embeddings(embeddings_path, starterpacks_path, model_name):
    """Reuse pack_semantics: embed_packs loads the npz cache and only embeds
    packs missing from it -- so if the cache is complete no model is loaded."""
    desc = ps.load_pack_descriptions(starterpacks_path)
    E, pack_ids, row_of = ps.embed_packs(
        desc, model_name=model_name, cache_path=embeddings_path
    )
    return E, row_of


def _map_pairs(eval_df, user_ids):
    """Map a DID eval table -> integer-coded pairs + aligned fixed cosine/hops.
    Drops pairs with an endpoint absent from the incidence."""
    did2idx = {d: i for i, d in enumerate(user_ids.tolist())}
    a = eval_df["user_a"].map(did2idx)
    b = eval_df["user_b"].map(did2idx)
    keep = a.notna() & b.notna()
    n_drop = int((~keep).sum())
    if n_drop:
        print(
            f"[eval] dropped {n_drop}/{len(eval_df)} pairs with an endpoint "
            f"not in the incidence"
        )
    pairs = pd.DataFrame(
        {
            "user_a": a[keep].astype(np.int64).to_numpy(),
            "user_b": b[keep].astype(np.int64).to_numpy(),
        }
    )
    fixed = pd.DataFrame(
        {
            "cosine": eval_df.loc[keep, "cosine"].to_numpy(dtype=float),
            "hops": eval_df.loc[keep, "hops"].to_numpy(dtype=int),
        }
    )
    return pairs.reset_index(drop=True), fixed.reset_index(drop=True)


# ---------------------------------------------------------------------------
@click.command()
@click.option("--mode", type=click.Choice(["sample", "reduce"]), required=True)
@click.option(
    "--ensemble",
    type=click.Choice(["microcanonical", "canonical"]),
    default="canonical",
    show_default=True,
)
@click.option(
    "--statistic",
    type=click.Choice(["slope", "interaction", "cooccurrence"]),
    default="interaction",
    show_default=True,
)
@click.option(
    "--eval-pairs-path",
    type=click.Path(dir_okay=False),
    help="Table user_a,user_b,cosine,hops -- a SELECTION-UNBIASED "
    "(uniform among V>=2) sample of pairs, NOT the s-stratified "
    "census. Required for --mode sample.",
)
@click.option(
    "--starterpacks-path",
    type=click.Path(dir_okay=False),
    help="starterpacks.jsonl (incidence + pack descriptions).",
)
@click.option(
    "--embeddings-path",
    default=None,
    help="npz embedding cache (pack_emb_<model>.npz from the semantic "
    "stage); reused, not recomputed, when complete.",
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
    help="n_eff estimator used as the common-ground predictor "
    "(default: average-linkage cluster count).",
)
@click.option(
    "--missing",
    default="distinct",
    show_default=True,
    type=click.Choice(["distinct", "drop"]),
)
@click.option(
    "--n-rep",
    default=1000,
    show_default=True,
    type=int,
    help="Total null replicates across all tasks.",
)
@click.option("--task-id", default=0, show_default=True, type=int)
@click.option("--n-tasks", default=1, show_default=True, type=int)
@click.option("--seed-base", default=0, show_default=True, type=int)
@click.option(
    "--n-trades",
    default=None,
    type=int,
    help="Curveball trades per microcanonical draw (default ~5*nnz).",
)
@click.option(
    "--out",
    "out_dir",
    required=True,
    help="Directory for per-task files (sample) / summary + figure (reduce).",
)
def main(
    mode,
    ensemble,
    statistic,
    eval_pairs_path,
    starterpacks_path,
    embeddings_path,
    model_name,
    method,
    missing,
    n_rep,
    task_id,
    n_tasks,
    seed_base,
    n_trades,
    out_dir,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if mode == "reduce":
        return _reduce(out_dir, statistic)
    return _sample(
        out_dir,
        ensemble,
        statistic,
        eval_pairs_path,
        starterpacks_path,
        embeddings_path,
        model_name,
        method,
        missing,
        n_rep,
        task_id,
        n_tasks,
        seed_base,
        n_trades,
    )


# ---------------------------------------------------------------------------
def _sample(
    out_dir,
    ensemble,
    statistic,
    eval_pairs_path,
    starterpacks_path,
    embeddings_path,
    model_name,
    method,
    missing,
    n_rep,
    task_id,
    n_tasks,
    seed_base,
    n_trades,
):
    if not (eval_pairs_path and starterpacks_path):
        raise click.UsageError(
            "--mode sample needs --eval-pairs-path and " "--starterpacks-path"
        )

    print("loading incidence (full, all users) ...")
    B, user_ids, pack_ids = nm.load_incidence(starterpacks_path)
    d, k = nm.margins(B)
    print(
        f"  incidence: {B.shape[0]:,} users x {B.shape[1]:,} packs, "
        f"{B.nnz:,} memberships"
    )

    print("loading embeddings (cache reused if complete) ...")
    E, row_of = _load_embeddings(embeddings_path, starterpacks_path, model_name)

    eval_df = _read(eval_pairs_path)
    for c in ("user_a", "user_b", "cosine", "hops"):
        if c not in eval_df.columns:
            raise click.UsageError(f"eval table missing column '{c}'")
    pairs, fixed = _map_pairs(eval_df, user_ids)
    print(f"  eval pairs: {len(pairs):,}")

    # ---- cooccurrence: analytic, canonical only, no replicate loop ----------
    if statistic == "cooccurrence":
        if ensemble != "canonical":
            raise click.UsageError(
                "--statistic cooccurrence requires "
                "--ensemble canonical (analytic test)"
            )
        if task_id != 0:
            print("[cooccurrence] analytic one-shot; only task 0 computes. exit.")
            return
        x, y = nm.bicm_fit(B)
        tbl = nm.bicm_cooccurrence_pvalues(pairs, x, y, B)
        _save(tbl, out_dir / "null_cooccurrence")
        print("[cooccurrence] wrote analytic per-pair p-values.")
        return

    # ---- observed statistic (once; task 0 persists it) ----------------------
    observed = nm.null_statistic(
        B, pairs, E, row_of, fixed, statistic, method=method, missing=missing
    )
    if task_id == 0:
        _save(pd.DataFrame([observed]), out_dir / f"observed_{statistic}")
    print(f"[observed] {observed}")

    # ---- fit the canonical model once (shared across this task's replicates) -
    x = y = None
    if ensemble == "canonical":
        print("fitting BiCM ...")
        x, y = nm.bicm_fit(B)
        ed, ek = _expected_margins(x, y)
        print(
            f"  BiCM expected-margin max abs error: "
            f"users={np.max(np.abs(ed - d)):.3g}, packs={np.max(np.abs(ek - k)):.3g}"
        )
        eval_users = pd.unique(pd.concat([pairs["user_a"], pairs["user_b"]])).astype(
            np.int64
        )

    # ---- replicate chunk ----------------------------------------------------
    chunk = int(np.ceil(n_rep / n_tasks))
    rows = []
    for i in range(chunk):
        gidx = task_id * chunk + i
        if gidx >= n_rep:
            break
        seed = seed_base + gidx
        if ensemble == "canonical":
            B_rand = nm.bicm_sample(x, y, seed, rows=eval_users)
        else:
            B_rand = nm.curveball_randomize(B, n_trades=n_trades, seed=seed)
        stat = nm.null_statistic(
            B_rand, pairs, E, row_of, fixed, statistic, method=method, missing=missing
        )
        stat.update(replicate=gidx, seed=seed)
        rows.append(stat)
        if (i + 1) % 25 == 0:
            print(f"  task {task_id}: {i + 1}/{chunk} replicates")

    _save(pd.DataFrame(rows), out_dir / f"null_stats_task{task_id}_{statistic}")
    print(
        f"[sample] task {task_id}: {len(rows)} replicates "
        f"(ensemble={ensemble}, statistic={statistic})."
    )


def _expected_margins(x, y):
    """E[d_u] = sum_e P_ue and E[k_e] = sum_u P_ue (degree-class safe, O(U*P) but
    we only call it once for the diagnostic; uses chunking to bound memory)."""
    U, P = len(x), len(y)
    ed = np.empty(U)
    # chunk users to avoid a dense U x P
    step = max(1, 2_000_000 // max(P, 1))
    for s in range(0, U, step):
        xb = x[s : s + step][:, None]
        Pb = xb * y[None, :] / (1.0 + xb * y[None, :])
        ed[s : s + step] = Pb.sum(axis=1)
    # E[k_e] via the same probs, accumulated
    ek = np.zeros(P)
    for s in range(0, U, step):
        xb = x[s : s + step][:, None]
        Pb = xb * y[None, :] / (1.0 + xb * y[None, :])
        ek += Pb.sum(axis=0)
    return ed, ek


# ---------------------------------------------------------------------------
def _reduce(out_dir, statistic):
    if statistic == "cooccurrence":
        return _reduce_cooccurrence(out_dir)

    obs_path = nm_resolve(out_dir, f"observed_{statistic}")
    if obs_path is None:
        raise click.ClickException(
            f"no observed_{statistic}.* in {out_dir} " "(was task 0 run?)"
        )
    observed = _read(obs_path).iloc[0].to_dict()

    files = sorted(out_dir.glob(f"null_stats_task*_{statistic}.parquet")) or sorted(
        out_dir.glob(f"null_stats_task*_{statistic}.csv")
    )
    if not files:
        raise click.ClickException(f"no null_stats_task*_{statistic}.* in {out_dir}")
    null = pd.concat([_read(f) for f in files], ignore_index=True)
    keys = [c for c in null.columns if c not in ("replicate", "seed")]
    print(
        f"[reduce] {len(null):,} replicates across {len(files)} task files; "
        f"statistics: {keys}"
    )

    rows = []
    for kkey in keys:
        vals = null[kkey].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        obs = float(observed[kkey])
        n = len(vals)
        ge = int(np.sum(vals >= obs))
        le = int(np.sum(vals <= obs))
        p = min(1.0, 2.0 * min((1 + ge) / (n + 1), (1 + le) / (n + 1)))
        mu, sd = float(vals.mean()), (
            float(vals.std(ddof=1)) if n > 1 else (float("nan"))
        )
        z = (obs - mu) / sd if (n > 1 and sd > 0) else np.nan
        rows.append(
            {
                "statistic": kkey,
                "observed": obs,
                "null_mean": mu,
                "null_sd": sd,
                "z": z,
                "p_two_sided": p,
                "n_rep": n,
            }
        )
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))
    _save(summary, out_dir / f"null_summary_{statistic}")

    # null-distribution figure (one panel per statistic key)
    ncol = min(3, len(keys))
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(4.2 * ncol, 3.2 * nrow), squeeze=False, facecolor="white"
    )
    for ax, kkey in zip(axes.ravel(), keys):
        vals = null[kkey].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        ax.hist(vals, bins=40, color="#bbbbbb", edgecolor="white")
        ax.axvline(
            float(observed[kkey]),
            color="#C44E52",
            lw=2,
            label=f"observed={observed[kkey]:.4f}",
        )
        ax.set_title(kkey, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes.ravel()[len(keys) :]:
        ax.set_visible(False)
    fig.suptitle(f"Configuration-model null: {statistic}")
    plt.tight_layout()
    out = out_dir / f"null_distribution_{statistic}.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def _reduce_cooccurrence(out_dir):
    path = nm_resolve(out_dir, "null_cooccurrence")
    if path is None:
        raise click.ClickException(f"no null_cooccurrence.* in {out_dir}")
    tbl = _read(path)
    n = len(tbl)
    n_sig = int((tbl["pvalue"] < 0.05).sum())
    print(
        f"[reduce cooccurrence] {n:,} pairs; {n_sig:,} with p<0.05 "
        f"({100 * n_sig / max(n, 1):.1f}%)"
    )
    fig, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.hist(
        tbl["pvalue"].to_numpy(dtype=float),
        bins=20,
        range=(0, 1),
        color="#bbbbbb",
        edgecolor="white",
    )
    ax.axhline(n / 20, color="#999999", ls="--", lw=1, label="uniform (null true)")
    ax.set_xlabel("two-sided p-value (analytic Poisson-binomial)")
    ax.set_ylabel("pairs")
    ax.set_title("Canonical co-occurrence test")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = out_dir / "null_distribution_cooccurrence.pdf"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")
    _save(tbl.sort_values("pvalue"), out_dir / "null_summary_cooccurrence")


def nm_resolve(out_dir, stem):
    for ext in (".parquet", ".csv"):
        p = out_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


if __name__ == "__main__":
    main()
