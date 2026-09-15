"""
null_model.py
=============
Configuration-model null for the common-ground -> similarity analysis.

We test whether the observed ``n_eff -> cosine`` association (and its moderation
by follow distance ``hops``) exceeds what the affiliation hypergraph's DEGREE
STRUCTURE alone produces. Only **pack membership** is rerandomized; the content
embeddings, the per-pair ``cosine``, and the per-pair ``hops`` are held FIXED
(read from the eval pair table). Because the null leaves ``cosine``/``hops``
fixed and only re-derives ``n_eff``, **the follow graph is never loaded here.**

Two ensembles over **0/1 user x pack incidence matrices with BOTH margins fixed**
(user degrees {d_u} and pack sizes {k_e}) -- the vertex-labeled *simple*
configuration model:

  * microcanonical -- curveball checkerboard trades; both margins exact.
  * canonical (BiCM) -- max-entropy independent-Bernoulli; margins in expectation,
    cheap analytic p-values.

We deliberately do NOT use a stub-labeled sampler: it admits repeated
vertices-in-edge and over-weights repeated edges, outside the support of
affiliation data. Fixing both margins is required -- fixing only user degrees
(Chung-Lu) would let pack-size heterogeneity manufacture co-memberships. (The
projection statistic s_uv = (B Bᵀ)_uv is relabeling-invariant, so the
stub-vs-vertex distinction washes out at projection level; the binary
fixed-both-margins sampler is the principled choice regardless.)

NOTE: microcanonical and canonical can be non-equivalent under extensive
constraints, so significance may differ between them -- hence we report both.

All n_eff recomputation goes through ``pack_semantics.n_eff_for_pairs`` (a pure
function of membership + embeddings); model fitting reuses ``_wls_beta`` from
``run_network_analysis``. Nothing here mutates existing modules.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

# Reuse existing code read-only: put the REPO ROOT on sys.path so `spcg` and
# `scripts` resolve however this module was reached.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spcg.pack_semantics as ps  # noqa: E402
from spcg.overlap_info import load_starterpacks  # noqa: E402  (membership convention)


# ---------------------------------------------------------------------------
# 1. Incidence
# ---------------------------------------------------------------------------
def load_incidence(starterpacks_path, candidate_users=None):
    """
    Binary user x pack incidence (CSR), with ``pack_id`` == the load_starterpacks
    line index (so it matches the membership / embedding convention).

    The FULL incidence (all users) is built so the true margins are preserved by
    the randomizers. ``candidate_users`` does NOT restrict the incidence (it only
    bears on which *pairs* are later evaluated) and is accepted for API symmetry.

    Returns ``(B, user_ids, pack_ids)`` where ``user_ids[i]`` is the DID of row i
    and ``pack_ids`` == ``arange(n_packs)``.
    """
    packs, _ = load_starterpacks(starterpacks_path)
    users = sorted(set().union(*packs)) if packs else []
    user_idx = {d: i for i, d in enumerate(users)}

    rows, cols = [], []
    for pid, members in enumerate(packs):
        for d in members:
            rows.append(user_idx[d])
            cols.append(pid)
    B = sparse.csr_matrix(
        (
            np.ones(len(rows), dtype=np.int8),
            (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
        ),
        shape=(len(users), len(packs)),
    )
    B.sum_duplicates()
    B.data[:] = 1  # vertex-labeled simple: collapse any repeated membership to 0/1
    return B, np.asarray(users, dtype=object), np.arange(len(packs))


def margins(B):
    """(user degrees d_u, pack sizes k_e) as 1-D int arrays."""
    return (
        np.asarray(B.sum(axis=1)).ravel().astype(np.int64),
        np.asarray(B.sum(axis=0)).ravel().astype(np.int64),
    )


# ---------------------------------------------------------------------------
# 2. Microcanonical sampler: curveball / checkerboard trades
# ---------------------------------------------------------------------------
def curveball_randomize(B, n_trades=None, seed=0, burn_in=0):
    """
    Curveball randomization on the user (row) adjacency: repeatedly pick two
    users and randomly redistribute the packs unique to each between them. Each
    trade preserves BOTH the two users' degrees and every pack's size, so the
    result has the observed margins EXACTLY.

    Operates on per-row sets (no dense matrix). ``n_trades`` defaults to ~5*nnz;
    ``burn_in`` extra trades are run first. Returns a new binary CSR.
    """
    rng = np.random.default_rng(seed)
    n_users = B.shape[0]
    indptr, indices = B.indptr, B.indices
    rows = [set(indices[indptr[i] : indptr[i + 1]].tolist()) for i in range(n_users)]

    if n_trades is None:
        n_trades = 5 * B.nnz
    total = int(n_trades) + int(burn_in)

    ii = rng.integers(0, n_users, size=total)
    jj = rng.integers(0, n_users, size=total)
    for t in range(total):
        i, j = int(ii[t]), int(jj[t])
        if i == j:
            continue
        A, C = rows[i], rows[j]
        only_i = A - C
        if not only_i:
            continue
        only_j = C - A
        if not only_j:
            continue
        pool = np.fromiter(only_i | only_j, dtype=np.int64)
        pool = pool[rng.permutation(pool.shape[0])]
        ku = len(only_i)
        shared = A & C
        rows[i] = shared | set(pool[:ku].tolist())
        rows[j] = shared | set(pool[ku:].tolist())

    # rebuild CSR
    counts = np.fromiter((len(s) for s in rows), count=n_users, dtype=np.int64)
    new_indptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(counts, out=new_indptr[1:])
    new_indices = np.empty(int(new_indptr[-1]), dtype=np.int32)
    for i, s in enumerate(rows):
        if s:
            new_indices[new_indptr[i] : new_indptr[i + 1]] = sorted(s)
    return sparse.csr_matrix(
        (np.ones(new_indices.shape[0], dtype=np.int8), new_indices, new_indptr),
        shape=B.shape,
    )


# ---------------------------------------------------------------------------
# 3. Canonical sampler: BiCM (bipartite configuration model)
# ---------------------------------------------------------------------------
def bicm_fit(B, max_iter=2000, tol=1e-9, use_package=True):
    """
    Fit the BiCM Lagrange multipliers so expected margins match observed.
    Returns ``(x, y)`` -- per-user ``x`` (len n_users) and per-pack ``y``
    (len n_packs) with ``P_ue = x_u y_e / (1 + x_u y_e)``.

    Uses the ``bicm`` package if importable; otherwise a documented reduced
    fixed-point over UNIQUE degree/size classes (small dense system, scalable).
    """
    d, k = margins(B)

    if use_package:
        try:
            from bicm import BipartiteGraph

            bg = BipartiteGraph()
            bg.set_biadjacency_matrix(
                (B > 0).astype(int).toarray()
                if B.shape[0] * B.shape[1] < 5_000_000
                else _edgelist(B)
            )
            bg.solve_tool()
            xx = np.asarray(bg.x).ravel()
            yy = np.asarray(bg.y).ravel()
            if xx.shape[0] == len(d) and yy.shape[0] == len(k):
                return xx, yy
        except Exception:
            pass  # fall through to the built-in fixed point

    # Reduced fixed point: unknowns per unique user-degree and unique pack-size.
    du, d_inv = np.unique(d, return_inverse=True)
    ku, k_inv = np.unique(k, return_inverse=True)
    nd_ = np.bincount(d_inv).astype(float)  # users per degree class
    nk_ = np.bincount(k_inv).astype(float)  # packs per size class
    L = float(d.sum())
    a = du.astype(float) / np.sqrt(L) if L > 0 else du.astype(float) + 1.0
    b = ku.astype(float) / np.sqrt(L) if L > 0 else ku.astype(float) + 1.0

    duf, kuf = du.astype(float), ku.astype(float)
    for _ in range(max_iter):
        ab = np.outer(a, b)  # (n_d x n_k)
        denom_a = (nk_[None, :] * b[None, :] / (1.0 + ab)).sum(axis=1)
        a_new = np.divide(duf, denom_a, out=np.zeros_like(a), where=denom_a > 0)
        ab2 = np.outer(a_new, b)
        denom_b = (nd_[:, None] * a_new[:, None] / (1.0 + ab2)).sum(axis=0)
        b_new = np.divide(kuf, denom_b, out=np.zeros_like(b), where=denom_b > 0)
        if np.max(np.abs(a_new - a)) < tol and np.max(np.abs(b_new - b)) < tol:
            a, b = a_new, b_new
            break
        a, b = a_new, b_new

    return a[d_inv], b[k_inv]


def _edgelist(B):
    """(user, pack) edge list for the bicm package's sparse path."""
    coo = B.tocoo()
    return np.column_stack([coo.row, coo.col])


def bicm_probabilities(x, y):
    """Return a row-probability getter ``P(u) -> array`` with
    ``P_ue = x_u y_e / (1 + x_u y_e)`` (avoids materializing the dense matrix)."""

    def prob_row(u):
        xu = x[int(u)]
        return xu * y / (1.0 + xu * y)

    return prob_row


def bicm_sample(x, y, seed, rows=None, pbar=False):
    """
    Independent-Bernoulli draw from the BiCM. Returns a binary CSR of shape
    (n_users, n_packs). If ``rows`` is given, only those user rows are sampled
    (others left empty) -- the canonical null leaves rows independent, so this is
    exact for the sampled users and keeps memory at O(sum of their degrees).
    """
    rng = np.random.default_rng(seed)
    U, P = len(x), len(y)
    rows = np.arange(U) if rows is None else np.asarray(rows, dtype=np.int64)
    counts = np.zeros(U, dtype=np.int64)
    col_of = {}

    if pbar:
        it = tqdm(rows, desc="Sampling")
    else:
        it = rows

    for u in it:
        u = int(u)
        p = x[u] * y / (1.0 + x[u] * y)
        cols = np.flatnonzero(rng.random(P) < p).astype(np.int32)
        col_of[u] = cols
        counts[u] = cols.shape[0]
    indptr = np.zeros(U + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    for u, cols in col_of.items():
        indices[indptr[u] : indptr[u + 1]] = cols
    return sparse.csr_matrix(
        (np.ones(indices.shape[0], dtype=np.int8), indices, indptr), shape=(U, P)
    )


def bicm_cooccurrence_pvalues(pairs, x, y, B):
    """
    Analytic per-pair co-occurrence test under the canonical null (no MCMC).
    For a pair (u, v), ``s_uv = (B Bᵀ)_uv`` is a sum of independent Bernoullis
    with success prob ``q_e = P_ue P_ve``, i.e. Poisson-binomial. We report the
    expected count ``E[s]=sum q_e``, the variance ``sum q_e(1-q_e)``, the observed
    ``s``, a z-score, and a two-sided p-value via the Poisson approximation to
    the Poisson-binomial (accurate when the per-pack probs are small, as here).

    ``pairs`` has integer ``user_a, user_b`` columns; returns a tidy DataFrame.
    """
    from scipy.stats import poisson

    a = pairs["user_a"].to_numpy(dtype=np.int64)
    b = pairs["user_b"].to_numpy(dtype=np.int64)
    s_obs = np.array([len(s) for s in shared_sets_for_pairs(B, pairs)], dtype=np.int64)

    out = []
    for i in range(len(a)):
        pu = x[a[i]] * y / (1.0 + x[a[i]] * y)
        pv = x[b[i]] * y / (1.0 + x[b[i]] * y)
        q = pu * pv
        lam = float(q.sum())
        var = float((q * (1.0 - q)).sum())
        s = int(s_obs[i])
        if lam > 0:
            p = 2.0 * min(poisson.cdf(s, lam), poisson.sf(s - 1, lam))
            p = float(min(p, 1.0))
        else:
            p = 1.0
        z = (s - lam) / np.sqrt(var) if var > 0 else np.nan
        out.append((int(a[i]), int(b[i]), s, lam, var, float(z), p))
    return pd.DataFrame(
        out, columns=["user_a", "user_b", "s_obs", "E_s", "var_s", "z", "pvalue"]
    )


# ---------------------------------------------------------------------------
# 4. Shared sets and the test statistic
# ---------------------------------------------------------------------------
def shared_sets_for_pairs(B, pairs):
    """Shared pack ids per pair, intersecting CSR rows. Returns a list of arrays
    aligned to ``pairs`` rows."""
    indptr, indices = B.indptr, B.indices
    a = pairs["user_a"].to_numpy(dtype=np.int64)
    b = pairs["user_b"].to_numpy(dtype=np.int64)
    out = []
    for i in range(len(a)):
        sa = indices[indptr[a[i]] : indptr[a[i] + 1]]
        sb = indices[indptr[b[i]] : indptr[b[i] + 1]]
        out.append(np.intersect1d(sa, sb, assume_unique=True))
    return out


def _membership_from_rows(B, users):
    """``{user_index: frozenset(pack_ids)}`` for the given users, from CSR rows."""
    indptr, indices = B.indptr, B.indices
    return {
        int(u): frozenset(indices[indptr[int(u)] : indptr[int(u) + 1]].tolist())
        for u in users
    }


def _fit_statistic(n_eff, cosine, hops, statistic):
    """Compute the scalar test statistic(s) from recomputed ``n_eff`` and the
    FIXED ``cosine``/``hops``. Reuses ``_wls_beta`` (OLS at unit weights)."""
    # Imported here, not at module scope: run_network_analysis imports `spcg`,
    # so a top-level import would be circular via spcg/__init__.
    from scripts.run_network_analysis import _wls_beta

    n = len(cosine)
    w = np.ones(n)
    if statistic == "slope":
        X = np.column_stack([np.ones(n), n_eff])  # cosine ~ n_eff
        beta = _wls_beta(X, cosine, w)
        return {"slope": float(beta[1])}

    if statistic == "interaction":
        levels = sorted(np.unique(hops).tolist())  # cosine ~ n_eff * C(hops)
        dummies = [(hops == lv).astype(float) for lv in levels[1:]]
        X = np.column_stack(
            [np.ones(n), n_eff] + dummies + [n_eff * d for d in dummies]
        )
        beta = _wls_beta(X, cosine, w)
        start = 2 + len(dummies)  # interaction block
        return {
            f"interaction[hops={lv}]": float(beta[start + i])
            for i, lv in enumerate(levels[1:])
        }

    raise ValueError(f"unknown statistic for sampling: {statistic!r}")


def null_statistic(
    B_rand,
    pairs,
    embeddings,
    pack_row,
    fixed_df,
    statistic,
    method="effrank",
    missing="distinct",
):
    """
    Recompute ``n_eff`` for ``pairs`` from the randomized incidence ``B_rand``
    (via ``pack_semantics.n_eff_for_pairs`` -- a pure function of membership +
    embeddings), join the FIXED ``cosine``/``hops`` from ``fixed_df`` (aligned by
    row), and return the chosen statistic as a dict of named scalars.

    ``pairs`` and ``fixed_df`` are row-aligned; ``pairs`` carries integer
    ``user_a, user_b`` (no ``shared_packs`` column, so n_eff_for_pairs stays
    quiet in the hot loop).
    """
    users = pd.unique(pd.concat([pairs["user_a"], pairs["user_b"]]))
    membership = _membership_from_rows(B_rand, users)
    neff = ps.n_eff_for_pairs(
        pairs[["user_a", "user_b"]],
        membership,
        embeddings,
        pack_row,
        method=method,
        missing=missing,
    )
    n_eff = neff[f"n_eff_{method}"].to_numpy(dtype=float)
    cosine = fixed_df["cosine"].to_numpy(dtype=float)
    hops = fixed_df["hops"].to_numpy(dtype=int)
    return _fit_statistic(n_eff, cosine, hops, statistic)
