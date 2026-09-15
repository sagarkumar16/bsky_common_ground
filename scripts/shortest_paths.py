"""
shortest_paths.py
=================
Shortest-path lengths (in follow-hops) between a SET of users on the Bluesky
follow graph.

Input graph: a CSV with columns ['source', 'target', 'created_at'] where an edge
`source -> target` means "source follows target". `created_at` is ignored.

The graph is large (multi-TB CSV), so the design is built around doing the
expensive part exactly once:

  1. PARSE ONCE -> COMPACT CSR. Stream the CSV in chunks, map the long DID
     strings to dense int32 node ids, and build a scipy CSR adjacency matrix.
     The result is cached to disk (`--graph-cache`); reruns skip the parse
     entirely (the single biggest speedup available here).

  2. MULTI-SOURCE BFS. Shortest paths on an UNWEIGHTED graph are a BFS, which
     scipy runs in C via `csgraph.dijkstra(unweighted=True)`. We run BFS only
     from the query users (not all nodes) and keep only the columns for the
     query users, so the output is a |S| x |S| hop-distance table.

  3. PARALLEL over sources. The CSR is built once in the parent and shared with
     worker processes copy-on-write (fork), so each worker runs BFS for a slice
     of the sources against the same in-memory graph with no extra copies.

Directed by default (paths follow the "follows" direction). Pass --undirected to
treat a follow as a mutual hop. Unreachable pairs get distance -1.

Usage
-----
    python shortest_paths.py \
        --follows /scratch/xee6vz/bluesky-graph/follows.csv \
        --users   users.txt \
        --output  output/shortest_paths.parquet
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import click
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

# Globals shared with fork()ed workers (read-only). Set in the parent before the
# pool is created so children inherit them copy-on-write -- no per-worker copy of
# the (large) CSR, no pickling it through the IPC channel.
_CSR = None
_TARGET_IDS = None
_DIRECTED = True


def _resolve_n_jobs(n_jobs):
    """n_jobs <= 0 (or None) -> Slurm allocation ($SLURM_CPUS_PER_TASK) else all
    CPUs. Mirrors spcg.overlap_info._resolve_n_jobs so behaviour is consistent."""
    if n_jobs is not None and n_jobs > 0:
        return n_jobs
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env and env.isdigit():
        return int(env)
    return os.cpu_count() or 1


# ---------------------------------------------------------------------------
# 1. Query user set
# ---------------------------------------------------------------------------
def load_user_set(path, col=None):
    """
    Load the set of query users (DIDs), order-preserving and de-duplicated.

    Accepts:
      * a plain text file (one DID per line), or
      * a .csv / .parquet table -- uses `col` if given, else a 'did' column,
        else the UNION of 'user_a' + 'user_b' (so the cosine pair table works
        directly).
    """
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        with open(p, "r", encoding="utf-8") as f:
            dids = [ln.strip() for ln in f if ln.strip()]
        return list(dict.fromkeys(dids))

    if col and col in df.columns:
        vals = df[col].astype(str)
    elif "did" in df.columns:
        vals = df["did"].astype(str)
    elif {"user_a", "user_b"} <= set(df.columns):
        vals = pd.concat([df["user_a"], df["user_b"]]).astype(str)
    else:
        raise ValueError(
            f"{path}: could not find a user column (pass --users-col; tried "
            f"'did', 'user_a'+'user_b'). Columns: {list(df.columns)}"
        )
    return list(dict.fromkeys(vals.tolist()))


# ---------------------------------------------------------------------------
# 2. Build (or load) the compact CSR adjacency
# ---------------------------------------------------------------------------
def _cache_paths(graph_cache):
    base = Path(graph_cache)
    return base.with_suffix(".npz"), base.with_suffix(".nodes.npy")


def build_graph(
    follows_path,
    graph_cache=None,
    rebuild=False,
    chunksize=5_000_000,
    has_header=False,
    columns=("source", "target", "created_at"),
):
    """
    Build the directed CSR adjacency from follows.csv and the sorted node-id
    array `nodes` (nodes[i] = DID of node i). Cached to disk when `graph_cache`
    is set; a cached graph is loaded as-is unless `rebuild`.

    Two streaming passes over the CSV (only 'source','target' are read):
      pass 1 collects the unique DIDs (-> dense id space);
      pass 2 maps each edge to (src_id, tgt_id) via vectorized searchsorted.
    int32 ids cap the graph at ~2.1B nodes, far above the real node count.
    """
    if graph_cache and not rebuild:
        csr_path, nodes_path = _cache_paths(graph_cache)
        if csr_path.exists() and nodes_path.exists():
            print(f"[graph] loading cached CSR from {csr_path}")
            csr = sparse.load_npz(csr_path)
            nodes = np.load(nodes_path, allow_pickle=True)
            print(f"[graph] {csr.shape[0]:,} nodes, {csr.nnz:,} edges (cached)")
            return csr, nodes

    # follows.csv is headerless by default -> name the columns positionally so
    # usecols can select source/target. Pass has_header=True for a headered file.
    reader_kw = dict(usecols=["source", "target"], dtype=str, chunksize=chunksize)
    if not has_header:
        reader_kw.update(header=None, names=list(columns))

    # Pass 1: unique DIDs.
    print(f"[graph] pass 1/2: scanning {follows_path} for unique users ...")
    uniq = set()
    n_rows = 0
    for chunk in pd.read_csv(follows_path, **reader_kw):
        n_rows += len(chunk)
        uniq.update(pd.unique(chunk.values.ravel("K")))
    uniq.discard(np.nan)
    uniq.discard(None)
    nodes = np.array(sorted(str(x) for x in uniq), dtype=object)
    n = len(nodes)
    print(f"[graph]   {n_rows:,} edges, {n:,} unique users")

    # Pass 2: map edges to dense int32 ids.
    print("[graph] pass 2/2: mapping edges to int ids ...")
    src_parts, tgt_parts = [], []
    for chunk in pd.read_csv(follows_path, **reader_kw):
        s = np.searchsorted(nodes, chunk["source"].to_numpy(dtype=object))
        t = np.searchsorted(nodes, chunk["target"].to_numpy(dtype=object))
        src_parts.append(s.astype(np.int32))
        tgt_parts.append(t.astype(np.int32))
    src = np.concatenate(src_parts) if src_parts else np.empty(0, np.int32)
    tgt = np.concatenate(tgt_parts) if tgt_parts else np.empty(0, np.int32)
    del src_parts, tgt_parts

    data = np.ones(len(src), dtype=np.uint8)  # unweighted; multi-edges collapse
    csr = sparse.csr_matrix((data, (src, tgt)), shape=(n, n))
    csr.sum_duplicates()
    print(f"[graph] built CSR: {n:,} nodes, {csr.nnz:,} edges")

    if graph_cache:
        csr_path, nodes_path = _cache_paths(graph_cache)
        csr_path.parent.mkdir(parents=True, exist_ok=True)
        sparse.save_npz(csr_path, csr)
        np.save(nodes_path, nodes)
        print(f"[graph] cached -> {csr_path} (+ {nodes_path.name})")

    return csr, nodes


# ---------------------------------------------------------------------------
# 3. Multi-source BFS (parallel over the query users)
# ---------------------------------------------------------------------------
def _bfs_batch(source_ids):
    """Worker: BFS from `source_ids` against the shared global CSR, keeping only
    the query-user (target) columns. inf (unreachable) -> -1. Returns int32
    block of shape (len(source_ids), len(_TARGET_IDS))."""
    d = dijkstra(
        _CSR, directed=_DIRECTED, indices=np.asarray(source_ids), unweighted=True
    )
    d = d[:, _TARGET_IDS]
    return np.where(np.isfinite(d), d, -1).astype(np.int32)


def shortest_path_matrix(csr, query_ids, directed=True, n_jobs=1, batch_size=None):
    """
    Hop-distance matrix D of shape (len(query_ids), len(query_ids)) where
    D[i, j] is the shortest follow-distance from query_ids[i] to query_ids[j]
    (-1 if unreachable, 0 on the diagonal). Sources are split into batches and
    BFS'd in parallel; the CSR is shared with workers via fork.
    """
    global _CSR, _TARGET_IDS, _DIRECTED
    _CSR, _TARGET_IDS, _DIRECTED = csr, np.asarray(query_ids), directed

    q = np.asarray(query_ids)
    workers = _resolve_n_jobs(n_jobs)
    if batch_size is None:
        # a handful of source-BFS per task keeps the C loop warm without making
        # any single returned block huge
        batch_size = max(1, len(q) // (workers * 4)) if workers > 1 else len(q)
    batches = [q[i : i + batch_size] for i in range(0, len(q), batch_size)]

    print(
        f"[bfs] {len(q):,} query users, directed={directed}, "
        f"{workers} worker(s), {len(batches)} batch(es)"
    )

    if workers > 1 and len(batches) > 1:
        try:
            ctx = get_context("fork")  # share CSR copy-on-write
        except ValueError:  # platform without fork
            ctx = None
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            blocks = list(ex.map(_bfs_batch, batches))
    else:
        blocks = [_bfs_batch(b) for b in batches]

    return np.vstack(blocks)


def matrix_to_long(D, query_dids, drop_self=True, drop_unreachable=False):
    """Melt the distance matrix into a tidy frame (source_did, target_did,
    distance). distance == -1 means unreachable."""
    n = len(query_dids)
    iu, ju = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    iu, ju, dist = iu.ravel(), ju.ravel(), D.ravel()
    dids = np.asarray(query_dids, dtype=object)
    out = pd.DataFrame(
        {"source_did": dids[iu], "target_did": dids[ju], "distance": dist}
    )
    if drop_self:
        out = out[iu != ju]
    if drop_unreachable:
        out = out[out["distance"] >= 0]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--follows",
    "follows_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="follows.csv with columns source,target,created_at "
    "(only needed when the graph cache must be built).",
)
@click.option(
    "--users",
    "users_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Query users: a DID-per-line text file, or a csv/parquet "
    "table (uses --users-col, else 'did', else user_a+user_b).",
)
@click.option(
    "--users-col", default=None, help="Column holding DIDs when --users is a table."
)
@click.option(
    "--output",
    "output_path",
    default="output/shortest_paths.parquet",
    show_default=True,
    help="Destination for the long-format result.",
)
@click.option(
    "--graph-cache",
    default="output/follows_graph",
    show_default=True,
    help="Path stem for the cached CSR (<stem>.npz + <stem>.nodes.npy). "
    "Reused across runs; build once.",
)
@click.option(
    "--rebuild",
    is_flag=True,
    default=False,
    help="Force a rebuild of the graph cache from --follows.",
)
@click.option(
    "--undirected",
    is_flag=True,
    default=False,
    help="Treat a follow as a mutual hop (ignore edge direction).",
)
@click.option(
    "--chunksize",
    default=5_000_000,
    show_default=True,
    type=int,
    help="CSV rows per chunk while building the graph.",
)
@click.option(
    "--has-header",
    is_flag=True,
    default=False,
    help="follows.csv has a header row (default: headerless, columns "
    "are source,target,created_at positionally).",
)
@click.option(
    "--n-jobs",
    default=0,
    show_default=True,
    type=int,
    help="Worker processes for the BFS. 0/<0 = $SLURM_CPUS_PER_TASK " "or all CPUs.",
)
@click.option(
    "--drop-unreachable",
    is_flag=True,
    default=False,
    help="Omit unreachable pairs instead of writing distance=-1.",
)
def main(
    follows_path,
    users_path,
    users_col,
    output_path,
    graph_cache,
    rebuild,
    undirected,
    chunksize,
    has_header,
    n_jobs,
    drop_unreachable,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    users = load_user_set(users_path, col=users_col)
    print(f"loaded {len(users):,} query users from {users_path}")

    csr, nodes = build_graph(
        follows_path,
        graph_cache=graph_cache,
        rebuild=rebuild,
        chunksize=chunksize,
        has_header=has_header,
    )

    # Map query DIDs -> node ids; some users may be absent from the graph.
    pos = np.searchsorted(nodes, np.asarray(users, dtype=object))
    pos = np.clip(pos, 0, len(nodes) - 1)
    present = nodes[pos] == np.asarray(users, dtype=object)
    n_missing = int((~present).sum())
    if n_missing:
        print(
            f"[users] {n_missing}/{len(users)} query users are not in the "
            f"follow graph; they are dropped."
        )
    users = [u for u, ok in zip(users, present) if ok]
    query_ids = pos[present]
    if len(users) < 2:
        raise click.UsageError("need >=2 query users present in the graph")

    D = shortest_path_matrix(csr, query_ids, directed=not undirected, n_jobs=n_jobs)

    long_df = matrix_to_long(
        D, users, drop_self=True, drop_unreachable=drop_unreachable
    )

    # quick reachability summary
    reach = long_df["distance"] >= 0
    if reach.any():
        finite = long_df.loc[reach, "distance"]
        print(
            f"[result] {int(reach.sum()):,}/{len(long_df):,} ordered pairs "
            f"reachable; distance mean={finite.mean():.2f}, max={int(finite.max())}"
        )
    else:
        print("[result] no reachable pairs among the query users")

    _save_table(long_df, output_path)
    print("Done.")


def _save_table(df, output_path):
    """Parquet if the engine is available, else CSV alongside."""
    try:
        df.to_parquet(output_path, index=False)
        print(f"  wrote {output_path}")
    except Exception:
        alt = Path(output_path).with_suffix(".csv")
        df.to_csv(alt, index=False)
        print(f"  wrote {alt}")


if __name__ == "__main__":
    main()
