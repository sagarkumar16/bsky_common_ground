"""
pack_semantics.py
=================
Semantic normalization of *shared* starterpacks.

Turns the integer "number of shared packs" between two users into a continuous
**effective number of semantically distinct shared packs**, ``n_eff``, so that
redundant shared packs (e.g. four near-identical soccer packs) count for less
than the same number of distinct ones. ``n_eff`` is bounded in ``[1, k]`` for a
shared set of size ``k`` and is intended as an alternative predictor for the
existing similarity-vs-overlap analysis in ``spcg/overlap_info.py``.

Intuition target (the worked example):
    A, B share 5 packs of which 4 are near-identical -> n_eff ~ 2.
    A, C share 3 distinct packs                      -> n_eff ~ 3.

Why an effective-number (Hill / diversity) construction and not ``count /
similarity``: dividing a count by a value in ``[0, 1]`` is unbounded and grows
with redundancy rather than discounting it (diversity -> infinity). A Hill
number built on the pack-embedding Gram is bounded by ``k`` and collapses
redundant packs toward 1, which is the behaviour we want.

This module only READS the existing pipeline (``load_starterpacks`` /
``build_user_pack_membership`` conventions); it never modifies it. The
``pack_id`` used here is the SAME line/enumeration index that
``load_starterpacks`` assigns (blank lines skipped, no increment), so ids align
with ``membership`` produced by ``build_user_pack_membership``.

All ``n_eff`` functions are PURE functions of ``(membership, E)`` so a future
null-model path can call them under rerandomized membership with the embeddings
held fixed.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _text_hash(t):
    """Stable content hash of a pack's embedding text (process-independent, so it
    can key the embedding cache -- changing the text invalidates the entry)."""
    return hashlib.md5(t.encode("utf-8")).hexdigest()


# sentence-transformers is heavy (pulls torch) and only needed to *create*
# embeddings. Import it lazily so this module imports fine without it -- the
# n_eff math operates on an embedding matrix E and needs no model.
_MODELS = {}


def _as_text(value):
    """Return a stripped string only for genuine str values, else ''. Guards
    against non-string description/name fields (null, numbers, bare NaN)."""
    return value.strip() if isinstance(value, str) else ""


def _load_model(model_name):
    """Lazily load (and memoize) a SentenceTransformer model."""
    if model_name not in _MODELS:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "embedding packs needs sentence-transformers: "
                "pip install -r requirements-semantic.txt"
            ) from e
        _MODELS[model_name] = SentenceTransformer(model_name)
    return _MODELS[model_name]


# ---------------------------------------------------------------------------
# 1. Pack-description text, keyed by the SAME id scheme as load_starterpacks
# ---------------------------------------------------------------------------
def load_pack_descriptions(starterpacks_path, include_name=True):
    """
    Return ``{pack_id: text}`` for every starterpack.

    ``pack_id`` is the line/enumeration index, assigned EXACTLY as
    ``load_starterpacks`` does it: blank lines are skipped WITHOUT incrementing
    the id, so the keys line up 1:1 with the ``packs`` list and therefore with
    the pack ids stored in ``membership``.

    The embedded text is the pack's TITLE (``name``) plus its DESCRIPTION
    (``record.description``, sometimes top-level ``description``), joined as
    "title. description". Including the title -- the strongest, most consistently
    present theme signal -- means packs with a title but a sparse/empty
    description still embed meaningfully, which yields fewer near-orthogonal
    (effectively "distinct") packs. ``include_name=False`` drops the title and
    uses the description only. A pack with neither maps to ``""`` and is handled
    by the downstream ``missing`` policy (see ``embed_packs`` /
    ``effective_number_shared``).
    """
    descriptions = {}
    pid = 0
    with open(starterpacks_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue  # mirror load_starterpacks: skip blanks, do NOT advance id
            sp = json.loads(line)
            rec = sp.get("record", {}) or {}
            # Real records occasionally carry a non-string here (null, a number,
            # or even a bare NaN that json.loads accepts), so coerce defensively
            # -- only genuine strings count as text, everything else is empty.
            name = _as_text(sp.get("name") or rec.get("name"))
            desc = _as_text(rec.get("description") or sp.get("description"))
            parts = []
            if include_name and name:
                parts.append(name)
            if desc:
                parts.append(desc)
            descriptions[pid] = ". ".join(parts)  # "title. description"
            pid += 1
    return descriptions


def verify_alignment(starterpacks_path, load_starterpacks=None, n_check=50):
    """
    Confirm ``load_pack_descriptions`` ids line up with ``load_starterpacks``.

    Reuses the existing ``load_starterpacks`` (passed in or imported read-only)
    and checks (a) identical pack counts and (b) for a sample of ids, that the
    name we parse matches the name the existing loader stored. Raises on
    mismatch; returns the pack count on success.
    """
    if load_starterpacks is None:
        from spcg.overlap_info import load_starterpacks  # read-only reuse
    packs, names = load_starterpacks(starterpacks_path)
    desc = load_pack_descriptions(starterpacks_path, include_name=True)
    if len(desc) != len(packs):
        raise AssertionError(
            f"pack_id misalignment: {len(desc)} descriptions vs {len(packs)} packs"
        )
    # Spot-check that our id->text and the existing id->name refer to the same row.
    idxs = range(0, len(packs), max(1, len(packs) // max(1, n_check)))
    for i in idxs:
        nm = names[i] or ""
        if nm and nm not in desc[i] and desc[i]:
            # name present but not reflected in our text -> ids drifted
            raise AssertionError(
                f"pack_id misalignment at {i}: name={nm!r} not in text={desc[i]!r}"
            )
    return len(packs)


# ---------------------------------------------------------------------------
# 2. Embeddings (one L2-normalized vector per pack), cached
# ---------------------------------------------------------------------------
def embed_packs(
    descriptions,
    model_name="all-MiniLM-L6-v2",
    cache_path=None,
    batch_size=64,
    show_progress=False,
):
    """
    Embed each pack's text into one **L2-normalized** sentence vector (so cosine
    == dot product). Use a multilingual model (e.g.
    ``paraphrase-multilingual-MiniLM-L12-v2``) if the descriptions are
    multilingual; ``model_name`` is exposed as config.

    Returns ``(E, pack_ids, row_of)``:
        E        : (n_embedded, d) float32, L2-normalized rows
        pack_ids : sorted list of the pack ids that HAD text (one per E row)
        row_of   : {pack_id: row index in E}

    Packs with empty text are intentionally absent from ``row_of`` -- they are
    handled by the ``missing`` policy in ``effective_number_shared`` (default:
    "distinct"), which never needs an embedding for them.

    Caching: results are stored in an ``.npz`` keyed by ``(pack_id, model_name,
    text_hash)`` and reused on rerun. The text hash means a cached vector is
    reused ONLY if the pack's embedding text is unchanged -- so changing what we
    embed (e.g. adding the title to the description) correctly RE-EMBEDS the
    affected packs instead of silently returning stale vectors. The cache is
    incremental: unchanged packs are reused, new/changed packs are (re)embedded,
    and the current set is written back.
    """
    items = {int(pid): t for pid, t in descriptions.items() if t and t.strip()}
    cur_hash = {pid: _text_hash(t) for pid, t in items.items()}

    # Reuse a cached vector only if same model_name AND the text hash matches.
    cached = {}
    if cache_path and Path(cache_path).exists():
        data = np.load(cache_path, allow_pickle=True)
        if str(data["model_name"]) == model_name and data["pack_ids"].size:
            ids = data["pack_ids"]
            emb = data["embeddings"]
            hashes = data["text_hashes"] if "text_hashes" in data.files else None
            for i, p in enumerate(ids):
                p = int(p)
                if (
                    hashes is not None
                    and p in cur_hash
                    and str(hashes[i]) == cur_hash[p]
                ):
                    cached[p] = emb[i]

    to_embed = [pid for pid in items if pid not in cached]
    if to_embed:
        if cache_path:
            print(
                f"[embed] reusing {len(cached):,} cached, embedding "
                f"{len(to_embed):,} new/changed pack(s) with '{model_name}'"
            )
        model = _load_model(model_name)
        texts = [items[pid] for pid in to_embed]
        vecs = model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        for pid, v in zip(to_embed, vecs):
            cached[pid] = np.asarray(v, dtype=np.float32)

    pack_ids = sorted(items.keys())
    if pack_ids:
        E = np.vstack([cached[pid] for pid in pack_ids]).astype(np.float32)
        # defensive re-normalization (cached vectors should already be unit norm)
        norms = np.linalg.norm(E, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        E = E / norms
    else:
        E = np.zeros((0, 0), dtype=np.float32)
    row_of = {pid: i for i, pid in enumerate(pack_ids)}

    if cache_path and pack_ids:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        all_emb = np.vstack([cached[p] for p in pack_ids]).astype(np.float32)
        np.savez(
            cache_path,
            pack_ids=np.asarray(pack_ids, dtype=np.int64),
            embeddings=all_emb,
            model_name=np.asarray(model_name),
            text_hashes=np.asarray([cur_hash[p] for p in pack_ids]),
        )

    return E, pack_ids, row_of


# ---------------------------------------------------------------------------
# 3. Effective number of distinct shared packs
# ---------------------------------------------------------------------------
def _shared_basis(shared_pack_ids, E, row_of, missing):
    """
    Build a matrix V whose Gram ``V @ V.T`` is the desired pack-similarity
    matrix for the shared set, plus the list of pack ids aligned to V's rows.

    missing == "distinct" (default): every pack appears as a row. Packs with an
        embedding use it; packs without one get a fresh orthonormal basis vector
        (their own extra dimension), i.e. similarity 1 to themselves and 0 to
        everything else -- "maximally distinct". This keeps V real and its Gram
        positive semidefinite by construction.
    missing == "drop": missing packs are removed entirely; V holds only the
        embedded packs.

    Returns ``(V, row_pids)`` or ``(None, [])`` if nothing is left.
    """
    present_rows, present_pos, present_pids = [], [], []
    missing_pos, missing_pids = [], []
    for pos, pid in enumerate(shared_pack_ids):
        r = row_of.get(int(pid))
        if r is None:
            missing_pos.append(pos)
            missing_pids.append(int(pid))
        else:
            present_pos.append(pos)
            present_rows.append(E[r])
            present_pids.append(int(pid))

    if missing == "drop":
        if not present_rows:
            return None, []
        return np.vstack(present_rows).astype(np.float64), present_pids

    # "distinct": full set, missing packs get unique unit dimensions.
    k = len(shared_pack_ids)
    d = E.shape[1] if E.shape[0] else 0
    V = np.zeros((k, d + len(missing_pos)), dtype=np.float64)
    for i, pos in enumerate(present_pos):
        if d:
            V[pos, :d] = present_rows[i]
    for j, pos in enumerate(missing_pos):
        V[pos, d + j] = 1.0
    row_pids = [int(p) for p in shared_pack_ids]
    return V, row_pids


def _apply_specificity(V, row_pids, specificity):
    """
    Weight pack contributions by specificity (IDF-on-packs). Weights are
    renormalized to mean 1 within this shared set, so equal weights reduce to
    the unweighted case and the diversity number stays comparable in scale.
    Scaling each row by sqrt(w) means ``(V @ V.T)_ij = sqrt(w_i w_j) E_i.E_j``.
    """
    w = np.array(
        [float(specificity.get(int(p), 1.0)) for p in row_pids], dtype=np.float64
    )
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    total = w.sum()
    if total > 0:
        w = w * (len(w) / total)  # mean -> 1
    return V * np.sqrt(w)[:, None]


def _methods_from_basis(V, row_pids, specificity=None):
    """Compute (effrank, rowsum, participation) from the shared-set basis V."""
    if specificity is not None:
        V = _apply_specificity(V, row_pids, specificity)

    # Gram of the (possibly weighted) vectors. PSD by construction.
    G = V @ V.T

    # participation: closed form, no eigendecomposition.
    #   n_eff = (sum lambda)^2 / sum lambda^2 = trace(G)^2 / ||G||_F^2
    trace = float(np.trace(G))
    fro2 = float(np.sum(G * G))
    participation = (trace * trace / fro2) if fro2 > 0 else 0.0

    # effrank: Shannon diversity of the eigenvalue distribution of the UNCLIPPED
    # Gram (must stay PSD -> eigvalsh on G, never on a clipped matrix).
    lam = np.linalg.eigvalsh(G)
    lam = lam[lam > 1e-12]
    s = lam.sum()
    if s <= 0:
        effrank = 0.0
    else:
        p = lam / s
        effrank = float(np.exp(-np.sum(p * np.log(p))))

    # rowsum: clipped similarity S_ij = max(0, E_i.E_j), n_eff = sum_i 1/sum_j S_ij.
    # Most aggressive discounting. Row sums always include the diagonal (>= the
    # row's self-weight > 0), but guard zero row-sums anyway.
    S = np.maximum(0.0, G)
    rs = S.sum(axis=1)
    nz = rs > 0
    rowsum = float(np.sum(1.0 / rs[nz])) if np.any(nz) else 0.0

    return float(effrank), float(rowsum), float(participation)


def _n_eff_all(shared_pack_ids, E, row_of, missing="distinct", specificity=None):
    """All three estimators for one shared set -> dict, with k==0/1 shortcuts."""
    k = len(shared_pack_ids)
    if k == 0:
        return {"effrank": 0.0, "rowsum": 0.0, "participation": 0.0}
    if k == 1:
        return {"effrank": 1.0, "rowsum": 1.0, "participation": 1.0}
    V, row_pids = _shared_basis(shared_pack_ids, E, row_of, missing)
    if V is None or V.shape[0] == 0:
        return {"effrank": 0.0, "rowsum": 0.0, "participation": 0.0}
    if V.shape[0] == 1:  # only one pack survived "drop"
        return {"effrank": 1.0, "rowsum": 1.0, "participation": 1.0}
    eff, rs, part = _methods_from_basis(V, row_pids, specificity)
    return {"effrank": eff, "rowsum": rs, "participation": part}


def effective_number_shared(
    shared_pack_ids, E, row_of, method="effrank", missing="distinct", specificity=None
):
    """
    Effective number of semantically distinct shared packs for one shared set.

    method:
      "rowsum"        clipped S_ij = max(0, E_i.E_j); n_eff = sum_i 1/sum_j S_ij.
                      Most aggressive discounting (~2 on the soccer example).
      "effrank"       Gram eigenvalue diversity n_eff = exp(-sum p ln p) on the
                      UNCLIPPED (PSD) Gram (~1.6 on soccer; nearest "#themes").
      "participation" closed form (sum lambda)^2 / sum lambda^2 = k^2 / ||G||_F^2.
    missing: "distinct" (treat empty-description packs as maximally distinct) or
             "drop" (exclude them).
    specificity: optional {pack_id: weight} to weight pack contributions.

    Returns 0.0 for k==0, 1.0 for k==1, else a value in roughly [1, k].
    """
    return _n_eff_all(shared_pack_ids, E, row_of, missing, specificity)[method]


# ---------------------------------------------------------------------------
# 4. Optional specificity weighting (IDF-on-packs), independent of redundancy
# ---------------------------------------------------------------------------
def pack_specificity(membership):
    """
    ``{pack_id: w_e}`` with ``w_e = log(N_users / k_e)`` -- IDF on packs, where
    ``k_e`` is the number of users in pack e and ``N_users`` is the total user
    count. A niche shared pack (small ``k_e``) signals more common ground than a
    giant generic one, so it carries a larger weight. This is a DISTINCT
    construct from redundancy; keep it off by default and pass it explicitly to
    ``effective_number_shared`` / ``n_eff_for_pairs`` to fold it in.
    """
    counts = Counter()
    for _did, packs in membership.items():
        for pid in packs:
            counts[int(pid)] += 1
    n_users = len(membership)
    return {pid: float(np.log(n_users / k)) for pid, k in counts.items() if k > 0}


# ---------------------------------------------------------------------------
# 5. Attach n_eff to a pair table
# ---------------------------------------------------------------------------
def n_eff_for_pairs(
    pairs_df,
    membership,
    E,
    row_of,
    method="cluster_average",
    missing="distinct",
    specificity=None,
):
    """
    Add ``[k_shared, n_eff_effrank, n_eff_rowsum, n_eff_participation]`` (and, for
    a cluster ``method``, ``n_eff_<method>``) to a pair table, plus the
    convenience ``n_eff`` alias set to ``method``. ``pairs_df`` has
    ``user_a, user_b`` (dids); each shared set is derived from ``membership``
    (NOT from any precomputed count).

    ``method`` (default ``"cluster_average"``): the primary estimator aliased to
    ``n_eff``. Options:
      * ``"cluster_single"|"cluster_complete"|"cluster_average"`` --
        threshold-integrated cluster count (``effective_num_cluster``) on the
        description-embedding cosine similarity of the shared packs. Average
        linkage is the default (robust "number of distinct themes").
      * ``"effrank"|"rowsum"|"participation"`` -- the eigenvalue-family estimators.
    The three eigenvalue-family columns are ALWAYS emitted for cross-estimator
    robustness reporting, regardless of ``method``.

    If a ``shared_packs`` column is present it is cross-checked against the
    derived ``k_shared`` -- a mismatch means the pack_id scheme drifted from the
    one that produced the table.
    """
    is_cluster = method.startswith("cluster_")
    link = method.split("cluster_", 1)[1] if is_cluster else None
    if is_cluster and link not in ("single", "complete", "average"):
        raise ValueError(f"unknown cluster method {method!r}")

    df = pairs_df.copy()
    eff, rs, part, ks = [], [], [], []
    clust = [] if is_cluster else None
    for a, b in zip(df["user_a"], df["user_b"]):
        shared = sorted(membership.get(a, frozenset()) & membership.get(b, frozenset()))
        ks.append(len(shared))
        vals = _n_eff_all(shared, E, row_of, missing=missing, specificity=specificity)
        eff.append(vals["effrank"])
        rs.append(vals["rowsum"])
        part.append(vals["participation"])
        if is_cluster:
            # average/single/complete linkage on the embedding-cosine similarity
            # of the shared packs (same signal as the eigen family, PSD-agnostic)
            clust.append(
                effective_num_cluster(_emb_sim(shared, E, row_of), method=link)
            )

    df["k_shared"] = ks
    df["n_eff_effrank"] = eff
    df["n_eff_rowsum"] = rs
    df["n_eff_participation"] = part
    if is_cluster:
        df[f"n_eff_{method}"] = clust
    df["n_eff"] = df[f"n_eff_{method}"]

    if "shared_packs" in df.columns:
        mism = int((df["k_shared"].to_numpy() != df["shared_packs"].to_numpy()).sum())
        if mism:
            print(
                f"[n_eff] WARNING: {mism}/{len(df)} pairs have derived k_shared != "
                f"stored shared_packs -- check pack_id alignment."
            )
        else:
            print(
                f"[n_eff] pack_id alignment OK: k_shared matches stored "
                f"shared_packs for all {len(df)} pairs."
            )

    return df


# ===========================================================================
# ADDITIVE (pair_stats_diagnostic_task): renormalization-style redundancy
# estimators that read as "number of distinct themes", an alternative text-free
# similarity signal (co-membership Jaccard), and a convenience aggregator.
#
# All new estimators take S, the P x P similarity among a pair's shared packs
# (S_ii == 1, entries clipped to [0, 1]); they are agnostic to how S was built.
# The existing estimators (effrank/rowsum/participation, n_eff_for_pairs, the
# loaders) are UNCHANGED and are reused here for the embedding-signal eigenvalue
# family.
# ===========================================================================
def effective_num_cluster(S, method="single"):
    """
    Threshold-integrated cluster count -- a closed form (no threshold sweep).

    With distance ``D = 1 - clip(S, 0, 1)`` and a hierarchical linkage, the
    number of clusters at threshold tau is ``#clusters(tau) = P - #{merges <=
    tau}``. Integrating over tau in [0, 1]:
        int_0^1 #clusters(tau) dtau = P - sum_k (1 - h_k) = 1 + sum_k h_k
    where ``h_k`` are the merge heights. So this equals ``1 + sum(heights)``.

    Range ``[1, P]``: identical packs -> heights 0 -> 1; mutually dissimilar ->
    heights 1 -> P; four-identical-plus-one -> ~2. For ``method="single"`` this
    equals ``1 + (total maximum-spanning-tree distance)``. ``method`` in
    {"single", "complete", "average"}.
    """
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform

    S = np.asarray(S, dtype=float)
    P = S.shape[0]
    if P <= 1:
        return float(P)  # 0 -> 0, 1 -> 1
    D = 1.0 - np.clip(S, 0.0, 1.0)
    D = 0.5 * (D + D.T)  # enforce symmetry for squareform
    np.fill_diagonal(D, 0.0)
    L = linkage(squareform(D, checks=False), method=method)
    return 1.0 + float(L[:, 2].sum())  # int_0^1 (#clusters at tau) dtau


def effective_num_mean_interp(S):
    """Mean-similarity interpolation, ``1 + (1 - sbar)*(P - 1)`` in ``[1, P]``
    where ``sbar`` is the mean off-diagonal (clipped) similarity. Simple,
    structure-free robustness check on the cluster count."""
    S = np.asarray(S, dtype=float)
    P = S.shape[0]
    if P <= 1:
        return float(P)
    off = np.clip(S, 0.0, 1.0).copy()
    np.fill_diagonal(off, np.nan)
    sbar = float(np.nanmean(off))
    return 1.0 + (1.0 - sbar) * (P - 1)  # in [1, P]


# --- Two ways to build S ----------------------------------------------------
def _sim_from_V(V):
    """L2-normalize rows of V (zero rows stay zero -> "distinct"), then
    ``S = clip(V @ V.T, 0, 1)`` with the diagonal forced to 1."""
    V = np.asarray(V, dtype=float)
    if V.shape[0] == 0:
        return np.zeros((0, 0))
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Vn = V / norms
    S = np.clip(Vn @ Vn.T, 0.0, 1.0)
    np.fill_diagonal(S, 1.0)
    return S


def sim_from_embeddings(shared_ids, Z):
    """A. Description-embedding cosine. ``V = Z[shared_ids]`` (Z indexed by
    pack_id; missing packs are zero rows -> distinct), ``S = clip(V @ V.T, 0, 1)``
    with unit diagonal."""
    ids = np.asarray(list(shared_ids), dtype=int)
    return _sim_from_V(np.asarray(Z)[ids])


def _emb_sim(shared_ids, E, row_of):
    """S under the embedding signal from the ``(E, row_of)`` produced by
    ``embed_packs`` (equivalent to ``sim_from_embeddings`` with a full Z)."""
    d = E.shape[1] if E.shape[0] else 0
    rows = [E[row_of[int(p)]] if int(p) in row_of else np.zeros(d) for p in shared_ids]
    return _sim_from_V(np.vstack(rows)) if rows else np.zeros((0, 0))


def sim_from_comembership(shared_ids, pack_members):
    """B. Frozen co-membership Jaccard (text-free). For shared packs i, j,
    ``S_ij = |members_i & members_j| / |members_i | members_j|`` on the observed
    member sets. Computed pair-locally over the small shared set (no global
    pack x pack matrix). This is a FIXED pack attribute of the OBSERVED
    membership; under the configuration-model null it must be FROZEN (computed
    once from the observed graph, not recomputed per replicate). Jaccard S is not
    guaranteed PSD -- see the eigenvalue-family note below."""
    ids = [int(p) for p in shared_ids]
    P = len(ids)
    S = np.eye(P)
    sets = [set(pack_members.get(p, set())) for p in ids]
    for i in range(P):
        for j in range(i + 1, P):
            a, b = sets[i], sets[j]
            u = len(a | b)
            S[i, j] = S[j, i] = (len(a & b) / u) if u else 0.0
    return S


def load_pack_members(starterpacks_path):
    """``{pack_id: set(did)}`` over observed membership, pack_id == the
    load_starterpacks line index (same convention as the existing membership).
    Reuses ``load_starterpacks`` (read-only)."""
    from spcg.overlap_info import load_starterpacks

    packs, _ = load_starterpacks(starterpacks_path)
    return {i: set(members) for i, members in enumerate(packs)}


# --- eigenvalue family on an arbitrary (possibly non-PSD) similarity matrix --
def _eigen_family_from_sim(S, clip_negative_eigs=True):
    """
    effrank / rowsum / participation computed directly from a similarity matrix
    S treated as a (pseudo-)Gram, for the NON-embedding signal (Jaccard) where
    there is no basis to hand to the existing estimators. Mirrors their formulas:
      rowsum        = sum_i 1 / sum_j clip(S,0,1)_ij            (unit diag -> [1,P])
      participation = trace^2 / ||S||_F^2 = P^2 / ||clip(S)||_F^2
      effrank       = exp(-sum p ln p), p = lambda / sum(lambda) over the
                      NON-NEGATIVE eigenvalues (negatives clipped to 0).
    Returns ``(effrank, rowsum, participation, nonpsd)`` where ``nonpsd`` is True
    if S had a negative eigenvalue (the flag the aggregator surfaces).
    """
    S = np.asarray(S, dtype=float)
    P = S.shape[0]
    if P == 0:
        return 0.0, 0.0, 0.0, False
    if P == 1:
        return 1.0, 1.0, 1.0, False
    Sc = np.clip(S, 0.0, 1.0)

    rs = Sc.sum(axis=1)
    nz = rs > 0
    rowsum = float(np.sum(1.0 / rs[nz])) if np.any(nz) else float(P)

    trace = float(np.trace(Sc))
    fro2 = float(np.sum(Sc * Sc))
    participation = (trace * trace / fro2) if fro2 > 0 else float(P)

    lam = np.linalg.eigvalsh(0.5 * (S + S.T))
    nonpsd = bool((lam < -1e-9).any())
    lam = lam[lam > 1e-12]  # clip negatives/zeros
    tot = lam.sum()
    if tot > 0:
        p = lam / tot
        effrank = float(np.exp(-np.sum(p * np.log(p))))
    else:
        effrank = float(P)
    return effrank, rowsum, participation, nonpsd


# --- convenience aggregator -------------------------------------------------
def all_effective_numbers(shared_ids, embeddings, pack_members):
    """
    Every effective-number statistic for one shared set, under BOTH signals.

    ``embeddings`` is the ``(E, row_of)`` pair from ``embed_packs``.
    ``pack_members`` is ``load_pack_members(...)`` output.

    Keys:
      "count"                                        raw P
      "cluster_{single,complete,average}__{emb,jac}" threshold-integrated count
      "mean_interp__{emb,jac}"                       mean-similarity interpolation
      "effrank__emb","rowsum__emb","participation__emb"
                                                     EXISTING estimators reused
                                                     on the embedding Gram (PSD)
      "effrank__jac(nonPSD)","rowsum__jac(nonPSD)","participation__jac(nonPSD)"
                                                     same formulas on the Jaccard
                                                     signal (non-PSD: negative
                                                     eigenvalues clipped, flagged)

    The embedding eigenvalue family REUSES ``_n_eff_all`` (the existing
    effrank/rowsum/participation on the unclipped embedding Gram); the Jaccard
    eigenvalue family uses ``_eigen_family_from_sim`` because Jaccard S is not a
    basis and may be non-PSD.
    """
    E, row_of = embeddings
    ids = [int(p) for p in shared_ids]
    P = len(ids)

    out = {"count": float(P)}
    S_emb = _emb_sim(ids, E, row_of)
    S_jac = sim_from_comembership(ids, pack_members)
    for name, S in (("emb", S_emb), ("jac", S_jac)):
        for m in ("single", "complete", "average"):
            out[f"cluster_{m}__{name}"] = effective_num_cluster(S, method=m)
        out[f"mean_interp__{name}"] = effective_num_mean_interp(S)

    emb_eig = _n_eff_all(ids, E, row_of, missing="distinct")  # reuse existing
    out["effrank__emb"] = emb_eig["effrank"]
    out["rowsum__emb"] = emb_eig["rowsum"]
    out["participation__emb"] = emb_eig["participation"]

    je, jr, jp, _nonpsd = _eigen_family_from_sim(S_jac)
    out["effrank__jac(nonPSD)"] = je
    out["rowsum__jac(nonPSD)"] = jr
    out["participation__jac(nonPSD)"] = jp
    return out
