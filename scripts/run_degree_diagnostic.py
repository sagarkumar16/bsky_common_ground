"""
run_degree_diagnostic.py

B1 -- the degree-distribution diagnostic figure (``degree_distributions_by_bin.pdf``).

Plots the distribution of Pack degree ``k_u`` for the users populating three
populations, overlaid on one log-x axis:

  (1) observed s=0 bin      -- endpoints of the uniform rejection-sampled
                               zero-overlap pairs (the original baseline).
  (2) observed s>=1 bins    -- endpoints of the co-member pairs. Degree-weighted
                               by construction (a user in many co-member pairs
                               contributes many times).
  (3) configuration null    -- endpoints of the pairs that land in the null's
                               s>=1 bins under BiCM replicates.

Expected: (2) is shifted strongly RIGHT of (1) and (3). That shift IS the
artifact the paper's central methodological claim rests on. The script prints
the median ordering and FAILS LOUDLY if it does not hold.

Optional second panel (--panel2/--no-panel2): mean cosine vs Pack degree over
the DISJOINT (s=0) pairs -- degree predicts similarity-to-anyone, which is the
second half of the confound (degree -> posting volume -> lexical density).

-----------------------------------------------------------------------------
WHY THIS IS A SCRIPT AND NOT A NOTEBOOK CELL
-----------------------------------------------------------------------------
The obvious formulation of population (3) -- enumerate every co-member pair in a
replicate via ``S = triu(B_rand @ B_rand.T)`` -- is NOT memory-bounded and will
OOM on any machine. A single pack with k members contributes ~k^2 nonzeros to
B B^T, so one 100k-member pack alone implies ~1e10 entries. More RAM does not
fix it.

Instead this mirrors ``run_null_curves.py``: the EVAL PAIR SET IS FIXED, and for
each replicate we only recompute the shared-pack count of those pairs by
intersecting two CSR rows. That is O(n_pairs * mean_degree), never O(n_users^2),
and it is also the *correct* null population -- these are exactly the pairs that
populate the null curve's s>=1 bins.

Degrees are always the OBSERVED pack degree ``d_u`` (incidence margins), so the
x-axis is one fixed per-user quantity across all three populations and the
curves differ only in WHICH users (and with what multiplicity) each bin selects.
That isolates the selection effect, which is the claim under test.

Embeddings are never loaded -- only ``k_shared`` is needed, not ``n_eff``.

Usage:
    python scripts/run_degree_diagnostic.py \
        --pairs-path   $DATA_DIR/directed/pairs_with_hops_n_eff_seed16.parquet \
        --packs-path   /scratch/xee6vz/bluesky-graph/starterpacks.jsonl \
        --out          reports/figures \
        --cache-dir    $DATA_DIR/degree_diagnostic \
        --n-rep 5 --seed 16
"""

from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")  # headless on the cluster
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Make the repo root importable so `spcg` resolves when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spcg.null_model as nm

# palette: matches the notebook's black/red observed-vs-null convention
C_OBS, C_NULL, C_GREY, C_MATCH = "#333333", "#C44E52", "#8c8c8c", "#4C72B0"
FS = 16


def _read(path):
    p = Path(path)
    return pd.read_parquet(p) if p.suffix in (".parquet", ".pq") else pd.read_csv(p)


def _shared_counts(B, idx_pairs):
    """Shared-pack count per pair by intersecting CSR rows. O(n_pairs*deg), and
    it never materializes B B^T."""
    indptr, indices = B.indptr, B.indices
    a = idx_pairs["user_a"].to_numpy(np.int64)
    b = idx_pairs["user_b"].to_numpy(np.int64)
    out = np.empty(len(a), dtype=np.int32)
    for i in range(len(a)):
        sa = indices[indptr[a[i]] : indptr[a[i] + 1]]
        sb = indices[indptr[b[i]] : indptr[b[i] + 1]]
        # CSR indices are sorted -> intersect1d on sorted arrays is cheap
        out[i] = np.intersect1d(sa, sb, assume_unique=True).size
    return out


def _ccdf(v):
    v = np.sort(v[v > 0])
    return v, 1.0 - np.arange(len(v)) / len(v)


@click.command()
@click.option(
    "--pairs-path",
    required=True,
    help="Augmented pair table (needs user_a, user_b, shared_packs, cosine).",
)
@click.option(
    "--packs-path", required=True, help="starterpacks.jsonl (for the incidence)."
)
@click.option(
    "--out",
    "out_dir",
    default="reports/figures",
    show_default=True,
    help="Where the PDF lands (small file).",
)
@click.option(
    "--cache-dir",
    default=None,
    help="Where the pooled degree arrays are cached (.npz). Put this on "
    "DATA_DIR/scratch -- NOT in ./output/.",
)
@click.option(
    "--n-rep",
    default=5,
    show_default=True,
    type=int,
    help="BiCM replicates for population (3). 5 is plenty for a distribution.",
)
@click.option("--seed", default=16, show_default=True, type=int)
@click.option(
    "--panel2/--no-panel2",
    default=True,
    show_default=True,
    help="Second panel: mean cosine vs pack degree on the disjoint pairs.",
)
def main(pairs_path, packs_path, out_dir, cache_dir, n_rep, seed, panel2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)

    # ---- inputs -----------------------------------------------------------
    pairs = _read(pairs_path)
    for c in ("user_a", "user_b", "shared_packs"):
        if c not in pairs.columns:
            raise click.UsageError(f"pairs table missing required column '{c}'")
    print(f"loaded {len(pairs):,} pairs from {pairs_path}", flush=True)

    print("loading incidence (full, all users) ...", flush=True)
    B, user_ids, _pack_ids = nm.load_incidence(packs_path)
    print(
        f"  incidence: {B.shape[0]:,} users x {B.shape[1]:,} packs, "
        f"{B.nnz:,} memberships",
        flush=True,
    )

    d_u, _k_e = nm.margins(B)  # d_u[i] = # packs user i belongs to
    d_u = np.asarray(d_u, dtype=np.int64)

    # map dids -> incidence rows, drop pairs with an endpoint outside the incidence
    did2idx = {d: i for i, d in enumerate(user_ids.tolist())}
    a = pairs["user_a"].map(did2idx)
    b = pairs["user_b"].map(did2idx)
    keep = a.notna() & b.notna()
    if int((~keep).sum()):
        print(
            f"[pairs] dropped {int((~keep).sum())} with an endpoint not in the incidence",
            flush=True,
        )
    idx_pairs = pd.DataFrame(
        {
            "user_a": a[keep].astype(np.int64).to_numpy(),
            "user_b": b[keep].astype(np.int64).to_numpy(),
        }
    )
    s_obs = pairs.loc[keep, "shared_packs"].to_numpy(int)
    cosine = (
        pairs.loc[keep, "cosine"].to_numpy(float) if "cosine" in pairs.columns else None
    )
    eval_users = pd.unique(
        pd.concat([idx_pairs["user_a"], idx_pairs["user_b"]])
    ).astype(np.int64)
    print(f"  {len(eval_users):,} distinct eval users", flush=True)

    # ---- populations (1) and (2): observed ---------------------------------
    def endpoint_degrees(mask):
        """Endpoint pack-degrees, endpoints NOT deduped so the degree weighting
        of the co-member enumeration is preserved."""
        return np.concatenate(
            [
                d_u[idx_pairs["user_a"].to_numpy()[mask]],
                d_u[idx_pairs["user_b"].to_numpy()[mask]],
            ]
        )

    m0 = s_obs == 0
    m1 = s_obs >= 1
    deg_bin0 = endpoint_degrees(m0)  # (1) uniform s=0
    deg_bin1 = endpoint_degrees(m1)  # (2) observed s>=1
    print(
        f"  pop(1) uniform s=0 : {m0.sum():,} pairs -> {len(deg_bin0):,} endpoints",
        flush=True,
    )
    print(
        f"  pop(2) observed s>=1: {m1.sum():,} pairs -> {len(deg_bin1):,} endpoints",
        flush=True,
    )

    # ---- population (3): BiCM replicates ----------------------------------
    print(f"fitting BiCM and drawing {n_rep} replicates ...", flush=True)
    x, y = nm.bicm_fit(B)
    parts = []
    for rep in range(n_rep):
        B_rand = nm.bicm_sample(x, y, seed=seed + rep, rows=eval_users)
        s_null = _shared_counts(B_rand, idx_pairs)
        mn = s_null >= 1
        parts.append(endpoint_degrees(mn))
        print(
            f"  rep {rep + 1}/{n_rep}: {int(mn.sum()):,} pairs with s>=1 "
            f"-> {len(parts[-1]):,} endpoints",
            flush=True,
        )
        del B_rand
    deg_null = np.concatenate(parts)  # (3) config null

    if cache:
        np.savez_compressed(
            cache / f"degree_populations_seed{seed}.npz",
            deg_bin0=deg_bin0,
            deg_bin1=deg_bin1,
            deg_null=deg_null,
        )
        print(f"cached pooled degrees -> {cache}", flush=True)

    # ---- figure ------------------------------------------------------------
    if panel2 and cosine is not None:
        fig, (ax, axb) = plt.subplots(1, 2, figsize=(11.5, 4.2), facecolor="white")
    else:
        fig, ax = plt.subplots(figsize=(6.6, 4.2), facecolor="white")
        axb = None

    meds = {}
    for v, lbl, color in [
        (deg_bin0, r"uniform $s{=}0$", C_GREY),
        (deg_null, "config. null", C_NULL),
        (deg_bin1, r"observed $s{\geq}1$", C_OBS),
    ]:
        vx, vy = _ccdf(v)
        med = float(np.median(v))
        meds[lbl] = med
        ax.step(
            vx, vy, where="post", color=color, lw=2, label=f"{lbl}  (median {med:.0f})"
        )
        ax.axvline(med, color=color, ls=":", lw=1, alpha=0.7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Pack degree $k_u$", fontsize=FS)
    ax.set_ylabel(r"CCDF  $P(K \geq k)$", fontsize=FS)
    ax.legend(frameon=False, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)

    if axb is not None:
        # degree predicts similarity-to-anyone: bin the DISJOINT (s=0) pairs by
        # max endpoint degree, plot mean cosine.
        ka = d_u[idx_pairs["user_a"].to_numpy()[m0]]
        kb = d_u[idx_pairs["user_b"].to_numpy()[m0]]
        kmax = np.maximum(ka, kb).astype(float)
        cos0 = cosine[m0]
        edges = np.unique(np.quantile(kmax, np.linspace(0, 1, 11)))
        who = np.clip(np.digitize(kmax, edges[1:-1]), 0, max(len(edges) - 2, 0))
        cx, cy = [], []
        for g in range(max(len(edges) - 1, 1)):
            msk = who == g
            if msk.sum() >= 30:
                cx.append(float(np.median(kmax[msk])))
                cy.append(float(cos0[msk].mean()))
        axb.plot(cx, cy, "o-", color=C_OBS, lw=1.6)
        axb.set_xscale("log")
        axb.set_xlabel(r"max endpoint pack degree $k_u$", fontsize=FS)
        axb.set_ylabel("Mean cosine (disjoint $s{=}0$ pairs)", fontsize=FS)
        axb.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    pdf = out_dir / "degree_distributions_by_bin.pdf"
    fig.savefig(pdf)
    print(f"wrote {pdf}", flush=True)

    # ---- the claim under test ---------------------------------------------
    m_uni = meds[r"uniform $s{=}0$"]
    m_nul = meds["config. null"]
    m_obs = meds[r"observed $s{\geq}1$"]
    summary = pd.DataFrame(
        [
            {
                "population": "uniform_s0",
                "median_k_u": m_uni,
                "n_endpoints": len(deg_bin0),
            },
            {
                "population": "config_null",
                "median_k_u": m_nul,
                "n_endpoints": len(deg_null),
            },
            {
                "population": "observed_s>=1",
                "median_k_u": m_obs,
                "n_endpoints": len(deg_bin1),
            },
        ]
    )
    summary.to_csv(out_dir / "degree_distributions_by_bin_medians.csv", index=False)
    print("\n" + summary.to_string(index=False), flush=True)

    print(
        f"\nmedian pack degree -- uniform s=0: {m_uni:.0f} | config null: {m_nul:.0f} "
        f"| observed s>=1: {m_obs:.0f}",
        flush=True,
    )
    if m_obs > m_nul and m_obs > m_uni:
        print(
            "OK: observed s>=1 is right-shifted vs BOTH the uniform baseline and "
            "the configuration null -- the artifact is visible.",
            flush=True,
        )
    else:
        print(
            "!! WARNING !! observed s>=1 is NOT right-shifted vs both comparisons. "
            "The paper's central methodological claim depends on this shift -- "
            "inspect the figure before relying on it.",
            flush=True,
        )


if __name__ == "__main__":
    main()
