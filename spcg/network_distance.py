"""
network_distance.py
===================
Follow-graph distance (``hops``) for census user pairs, as the network-position
dimension of communal common ground.

Theory (naming/titles only): ``n_eff`` operationalizes H.H. Clark's *communal*
(membership-based) common ground; ``hops`` -- shortest follow-graph distance --
is the network-position dimension his account omits. These functions add
``hops`` to a pair table so one can ask whether shared affiliation confers
common ground uniformly across the graph or is moderated by proximity. ``cosine``
is treated as an observable proxy for common ground, not the construct itself.

Efficiency contract (see network_analysis_task.md):
  * Build the CSR from a PERSISTED, reduced follow edgelist (parquet/csv with
    source,target) -- never scan the raw multi-TB CSV here.
  * Compute ``hops`` ONLY for the given census pairs: group by source, BFS each
    unique source over the CSR to depth ``max_hops``, read off the distance to
    the needed targets. No all-pairs shortest paths.
  * Pairs beyond ``max_hops`` or in another component are NOT dropped -- they get
    an explicit sentinel (``hops = max_hops + 1``, label "unreachable"), the
    theoretically loaded cell (strong communal common ground, no network contact).

All functions are pure functions of ``(pairs, graph)`` so a later
configuration-model null (membership rerandomized, follow graph + embeddings
fixed) can call them unchanged.
"""

import atexit
import os
import shutil
import tempfile
from collections import defaultdict, namedtuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

try:
    from tqdm import tqdm
except Exception:  # tqdm absent -> no-op passthrough

    def tqdm(it=None, **k):
        return it if it is not None else iter(())


# A CSR follow graph with the id<->did bookkeeping.
FollowGraph = namedtuple(
    "FollowGraph", ["indptr", "indices", "did_to_id", "id_to_did", "n"]
)

# Globals shared with fork()ed BFS workers (read-only CSR adjacency arrays). Set
# in the parent before the pool is created so children inherit them copy-on-write
# -- no pickling, no per-worker copy. Forward = out-edges (a-side); backward =
# in-edges (b-side); for an undirected/symmetrized graph they are the same arrays.
_INDPTR_F = _INDICES_F = None
_INDPTR_B = _INDICES_B = None
_N = 0
_MAX_HOPS = 4
_HF = 0  # ceil(max_hops/2): forward radius from a (cached per source)
_HB = 0  # floor(max_hops/2): backward radius from b
_TARGETS_BY_SRC = None


def _resolve_n_jobs(n_jobs):
    """n_jobs <= 0/None -> $SLURM_CPUS_PER_TASK else all CPUs (mirrors
    spcg.overlap_info._resolve_n_jobs)."""
    if n_jobs is not None and n_jobs > 0:
        return n_jobs
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env and env.isdigit():
        return int(env)
    return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# 1. Build the CSR follow graph from a persisted edgelist
# ---------------------------------------------------------------------------
def _readonly_memmap(arr, path):
    """Persist `arr` to `path`.npy and return a READ-ONLY memmap of it. A
    file-backed read-only mapping is NOT charged to fork()'s overcommit
    accounting (unlike a private anonymous array), so a worker pool can fork off
    a parent holding a multi-GB graph without OSError(ENOMEM), and the pages are
    shared (~1x memory, not per-worker)."""
    np.save(path, np.ascontiguousarray(arr))
    return np.load(path + ".npy", mmap_mode="r")


def _mmap_scratch(prefix):
    """Temp dir for memmap files (honors $TMPDIR -- point it at scratch if the
    node's /tmp is small or RAM-backed). Auto-removed at process exit."""
    d = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    return d


def load_follow_graph(
    edgelist_path, directed=False, source_col="source", target_col="target", mmap=True
):
    """
    Build a CSR adjacency from a PERSISTED follow edgelist (parquet/csv with
    ``source,target``). DIDs are integer-coded densely.

    Returns a ``FollowGraph(indptr, indices, did_to_id, id_to_did, n)``.
    ``directed=False`` (default) symmetrizes the graph to match the symmetric
    ``n_eff``/cosine constructs; ``directed=True`` keeps follow direction
    (an edge ``source -> target`` means source follows target).

    ``mmap=True`` (default) returns ``indptr``/``indices`` as READ-ONLY memmaps
    (and drops the in-RAM copies) so the graph is fork-safe for the parallel BFS
    -- see ``_readonly_memmap``. Set ``mmap=False`` to keep plain in-RAM arrays.

    This reads only the two id columns from a reduced edgelist; it does NOT scan
    the raw multi-TB follows.csv.
    """
    p = Path(edgelist_path)
    cols = [source_col, target_col]
    if p.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p, columns=cols)
    else:
        df = pd.read_csv(p, usecols=cols, dtype=str)

    src_s = df[source_col].astype(str)
    tgt_s = df[target_col].astype(str)
    # Dense integer codes over the union of endpoints; id_to_did = uniques.
    codes, uniques = pd.factorize(
        pd.concat([src_s, tgt_s], ignore_index=True), sort=True
    )
    m = len(df)
    src = codes[:m].astype(np.int32)
    tgt = codes[m:].astype(np.int32)
    n = len(uniques)
    id_to_did = np.asarray(uniques, dtype=object)
    did_to_id = {d: i for i, d in enumerate(id_to_did)}

    if directed:
        rows, ccols = src, tgt
    else:  # symmetrize
        rows = np.concatenate([src, tgt])
        ccols = np.concatenate([tgt, src])
    data = np.ones(len(rows), dtype=np.uint8)
    csr = sparse.csr_matrix((data, (rows, ccols)), shape=(n, n))
    csr.sum_duplicates()

    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int32)
    del csr, rows, ccols, data, src, tgt, codes
    if mmap:
        d = _mmap_scratch("nd_graph_")
        indptr_mm = _readonly_memmap(indptr, os.path.join(d, "indptr"))
        indices_mm = _readonly_memmap(indices, os.path.join(d, "indices"))
        del indptr, indices  # drop RAM copies -> only file-backed remain
        indptr, indices = indptr_mm, indices_mm

    return FollowGraph(
        indptr=indptr, indices=indices, did_to_id=did_to_id, id_to_did=id_to_did, n=n
    )


# ---------------------------------------------------------------------------
# 2. Bounded shortest-path distance for exactly the census pairs
# ---------------------------------------------------------------------------
# MEET-IN-THE-MIDDLE bounded BFS. We do NOT run single-source-to-all-nodes
# shortest paths (scipy dijkstra allocates an n-length distance array per source
# and, on a small-world graph, its k-hop ball already reaches a large fraction of
# the whole graph). Instead, for a pair (a, b) with a shortest path <= max_hops,
# some vertex m on it lies within ceil(k/2) of a AND floor(k/2) of b. So we
# expand a's forward ball to HF=ceil(k/2) (ONCE per unique source, reused across
# all its targets) and combine with b's backward ball to HB=floor(k/2). For the
# default k=3 that is a 2-hop ball from a plus b's immediate in-neighbors -- no
# n-length allocation, no heap, and far smaller than the 3-hop ball.
#
# Neighbor gathering is fully vectorized (no per-frontier-node Python loop). If a
# run is still slow, the graph is genuinely dense: use a SPARSER edgelist
# (filter_follows --reciprocal) or lower --max-hops.
_INF = 1 << 30


def _gather(frontier, indptr, indices):
    """All out-neighbors of every node in `frontier`, vectorized (CSR segment
    gather without a Python loop over the frontier)."""
    starts = indptr[frontier]
    counts = indptr[frontier + 1] - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=indices.dtype)
    seg_start = np.repeat(starts, counts)
    within = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
    return indices[seg_start + within]


def _ball(src, depth, indptr, indices, dist, batch_edges=10_000_000):
    """BFS from `src` to `depth`, writing distances into the scratch `dist`
    array (-1 = unvisited). Returns the list of touched-node arrays so the caller
    can reset `dist` cheaply. `dist[src]` is set to 0.

    The frontier is expanded in batches capped at ~``batch_edges`` out-edges per
    ``_gather`` call, so peak memory stays bounded even when a hub's neighborhood
    has hundreds of millions of collective out-edges (materializing them all at
    once would OOM). Distances are marked as each batch is processed, so a node
    is never enqueued twice within a level."""
    dist[src] = 0
    touched = [np.array([src], dtype=np.int64)]
    frontier = touched[0]
    for d in range(1, depth + 1):
        if frontier.size == 0:
            break
        counts = (indptr[frontier + 1] - indptr[frontier]).astype(np.int64)
        new_parts = []
        i, F = 0, frontier.shape[0]
        while i < F:
            tot, j = 0, i
            # at least one node per batch, even if its degree exceeds the cap
            while j < F and (j == i or tot + counts[j] <= batch_edges):
                tot += int(counts[j])
                j += 1
            neigh = _gather(frontier[i:j], indptr, indices)
            if neigh.size:
                neigh = np.unique(neigh)
                fresh = neigh[dist[neigh] < 0]
                if fresh.size:
                    dist[fresh] = d
                    new_parts.append(fresh)
            i = j
        if not new_parts:
            break
        frontier = new_parts[0] if len(new_parts) == 1 else np.concatenate(new_parts)
        touched.append(frontier)
    return touched


def _combine(b, dist_a, dist_b):
    """Shortest a->b length given a's forward ball already in `dist_a`, by
    expanding b's backward ball to HB and taking min(dist_a[m] + dist_b[m])."""
    hb = _HB
    if hb == 0:  # k == 1: b must be in a's ball
        v = dist_a[b]
        return int(v) if v >= 0 else _INF
    if hb == 1:  # k in {2,3}: b + its in-neighbors
        best = _INF
        va = dist_a[b]
        if va >= 0:
            best = int(va)
        pred = _INDICES_B[_INDPTR_B[b] : _INDPTR_B[b + 1]]
        if pred.size:
            da = dist_a[pred]
            r = da[da >= 0]
            if r.size:
                best = min(best, int(r.min()) + 1)
        return best
    # hb >= 2: full backward ball from b, intersect with a's ball
    touched = _ball(b, hb, _INDPTR_B, _INDICES_B, dist_b)
    nodes = np.concatenate(touched)
    da = dist_a[nodes]
    mask = da >= 0
    best = int((da[mask] + dist_b[nodes][mask]).min()) if mask.any() else _INF
    dist_b[nodes] = -1
    return best


def _hops_chunk(src_chunk):
    """Worker: for each source, expand its forward ball once and resolve all its
    targets by meet-in-the-middle. Returns (src, tgt, hops) for reached pairs."""
    dist_a = np.full(_N, -1, dtype=np.int16)  # scratch, reset per source
    dist_b = np.full(_N, -1, dtype=np.int16) if _HB >= 2 else None
    out = []
    for a in src_chunk:
        touched = _ball(a, _HF, _INDPTR_F, _INDICES_F, dist_a)
        for b in _TARGETS_BY_SRC[a]:
            d = _combine(int(b), dist_a, dist_b)
            if d <= _MAX_HOPS and d > 0:
                out.append((a, int(b), d))
        dist_a[np.concatenate(touched)] = -1  # reset only touched entries
    return out


def pair_distances(
    pairs_df, graph, max_hops=4, n_jobs=4, directed=False, chunk_size=None
):
    """
    Add a ``hops`` column = shortest follow-distance for each census pair, via
    meet-in-the-middle bounded BFS (only the listed pairs are resolved; no
    all-pairs shortest paths, no n-length per-source allocation).

    Pairs beyond ``max_hops``, in a different component, or with an endpoint
    absent from the graph get the sentinel ``hops = max_hops + 1``
    ("unreachable") -- kept, not dropped.

    ``directed``: pass the SAME value used to load the graph. False (default)
    means the graph is symmetrized, so the backward (b-side) search reuses the
    forward adjacency; True builds the transpose so the b-side follows in-edges
    (giving directed a->b distance).

    ``pairs_df`` must have ``user_a, user_b`` (dids). Returns a copy + ``hops``.
    """
    global _INDPTR_F, _INDICES_F, _INDPTR_B, _INDICES_B, _N, _MAX_HOPS, _HF, _HB
    global _TARGETS_BY_SRC

    df = pairs_df.copy()
    sentinel = max_hops + 1
    a_id = df["user_a"].map(graph.did_to_id)
    b_id = df["user_b"].map(graph.did_to_id)
    valid = a_id.notna() & b_id.notna()

    hops = np.full(len(df), sentinel, dtype=np.int64)
    a_id_v = a_id[valid].astype(np.int64).to_numpy()
    b_id_v = b_id[valid].astype(np.int64).to_numpy()
    pos_v = np.flatnonzero(valid.to_numpy())

    # Workers need targets_by_src (built before the pool); rows_by_pair is only
    # for write-back, so it's built AFTER the pool to keep the forked parent lean.
    targets_by_src = defaultdict(list)
    for s, t in zip(a_id_v, b_id_v):
        targets_by_src[int(s)].append(int(t))
    targets_by_src = {
        s: np.unique(np.asarray(ts, dtype=np.int64)) for s, ts in targets_by_src.items()
    }

    # Forward adjacency = the graph's CSR arrays (memmapped by load_follow_graph
    # -> fork-safe). Backward adjacency = transpose for a directed graph (also
    # memmapped so it doesn't reintroduce a multi-GB anonymous array in the
    # parent), else the same forward arrays (symmetrized graph).
    _INDPTR_F = graph.indptr
    _INDICES_F = graph.indices
    if directed:
        Bt = sparse.csr_matrix(
            (
                np.ones(graph.indices.shape[0], dtype=np.int8),
                graph.indices,
                graph.indptr,
            ),
            shape=(graph.n, graph.n),
        ).T.tocsr()
        bp = Bt.indptr.astype(np.int64)
        bi = Bt.indices.astype(np.int32)
        del Bt
        d = _mmap_scratch("nd_graphB_")
        _INDPTR_B = _readonly_memmap(bp, os.path.join(d, "indptr"))
        _INDICES_B = _readonly_memmap(bi, os.path.join(d, "indices"))
        del bp, bi
    else:
        _INDPTR_B, _INDICES_B = _INDPTR_F, _INDICES_F
    _N = graph.n
    _MAX_HOPS = max_hops
    _HF = (max_hops + 1) // 2
    _HB = max_hops // 2
    _TARGETS_BY_SRC = targets_by_src

    srcs = list(targets_by_src.keys())
    workers = _resolve_n_jobs(n_jobs)
    if chunk_size is None:
        chunk_size = max(1, len(srcs) // (workers * 8)) if workers > 1 else len(srcs)
    chunks = [srcs[i : i + chunk_size] for i in range(0, len(srcs), chunk_size)]
    print(
        f"[hops] {len(srcs):,} unique sources, {len(chunks)} chunk(s), "
        f"{workers} worker(s), meet-in-the-middle {_HF}+{_HB} (max {max_hops} hops)"
    )

    def _serial():
        out = []
        for c in tqdm(chunks, desc="hops BFS (serial)", unit="chunk"):
            out.extend(_hops_chunk(c))
        return out

    found = None
    if workers > 1 and len(chunks) > 1:
        # Forking workers off a parent already holding the full graph can fail
        # with OSError(ENOMEM) under a Slurm cgroup memory cap + strict
        # overcommit. If so, fall back to serial rather than crashing.
        try:
            ctx = get_context("fork")
        except ValueError:
            ctx = None
        try:
            with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
                blocks = list(
                    tqdm(
                        ex.map(_hops_chunk, chunks),
                        total=len(chunks),
                        desc="hops BFS",
                        unit="chunk",
                    )
                )
            found = [rec for blk in blocks for rec in blk]
        except OSError as e:
            print(
                f"[hops] parallel fork failed ({e}); running serially instead. "
                f"(The meet-in-the-middle BFS is the real speedup; raise --mem "
                f"or lower --n-jobs to re-enable parallelism.)"
            )
            found = None
    if found is None:
        found = _serial()

    # (built now, after the pool) (src,tgt) -> row positions, for write-back
    rows_by_pair = defaultdict(list)
    for pos, s, t in zip(pos_v, a_id_v, b_id_v):
        rows_by_pair[(int(s), int(t))].append(int(pos))
    for s, t, d in found:
        for pos in rows_by_pair[(s, t)]:
            hops[pos] = d

    df["hops"] = hops
    n_far = int((hops == sentinel).sum())
    print(
        f"[hops] {len(df) - n_far:,}/{len(df):,} pairs within {max_hops} hops; "
        f"{n_far:,} unreachable/far (sentinel={sentinel})."
    )
    return df


# ---------------------------------------------------------------------------
# 3. Ordered categorical binning
# ---------------------------------------------------------------------------
def bin_hops(pairs_df, max_hops, col="hops"):
    """
    Ordered categorical for ``hops`` with levels ``1, 2, ..., max_hops,
    "unreachable"`` (the last absorbing everything at/above the sentinel).
    Returns a pandas Categorical aligned to ``pairs_df`` rows.
    """
    labels = [str(h) for h in range(1, max_hops + 1)] + ["unreachable"]
    h = pairs_df[col].to_numpy()
    mapped = np.where(
        (h >= 1) & (h <= max_hops),
        np.char.mod("%d", np.clip(h, 1, max_hops)),
        "unreachable",
    )
    return pd.Categorical(mapped, categories=labels, ordered=True)
