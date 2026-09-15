"""
inspect_degrees.py
==================
Report the highest-degree users in a follow edgelist -- a diagnostic for the
2-hop-neighborhood blowup in the network analysis. A few mega-hubs (the official
Bluesky account, celebrities followed by a large fraction of the graph) can make
one source's 2-hop ball explode to hundreds of millions of edges; this dumps the
top-N by out-degree (accounts followed), in-degree (followers), and total, plus
how concentrated the edges are, so we can decide whether to filter them.

Memory-light: uses pyarrow to count each column separately (no CSR, no
factorization, one column resident at a time).

    python scripts/inspect_degrees.py \
        --edgelist-path /scratch/.../filtered_follows.parquet --top 100 --out out_dir
"""

from pathlib import Path

import click
import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    raise SystemExit("inspect_degrees needs pyarrow: pip install pyarrow") from e


def _iter_col_batches(path, col, columns, has_header, batch_rows, block_mb):
    """Yield one column's values in batches -- STREAMING, so the whole (huge)
    edge column is never materialized at once."""
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        pf = pq.ParquetFile(p)
        for b in pf.iter_batches(columns=[col], batch_size=batch_rows):
            yield b.column(0)
    else:
        read_opts = (
            pacsv.ReadOptions(block_size=block_mb * 1024 * 1024)
            if has_header
            else pacsv.ReadOptions(
                block_size=block_mb * 1024 * 1024, column_names=list(columns)
            )
        )
        conv = pacsv.ConvertOptions(
            include_columns=[col], column_types={col: pa.string()}
        )
        reader = pacsv.open_csv(p, read_options=read_opts, convert_options=conv)
        for batch in reader:
            yield batch.column(0)


def _degree_series(
    path, col, columns, has_header, name, batch_rows=5_000_000, block_mb=256
):
    """Per-node degree as Series(index=did, value=count), accumulated over
    STREAMED batches. Peak memory ~ one batch + the running per-node counts
    (bounded by the node count, not the edge count)."""
    acc = None
    n_rows = 0
    for arr in _iter_col_batches(path, col, columns, has_header, batch_rows, block_mb):
        n_rows += len(arr)
        vc = pc.value_counts(arr)
        s = pd.Series(
            vc.field("counts").to_numpy(), index=vc.field("values").to_pylist()
        )
        acc = s if acc is None else acc.add(s, fill_value=0)
    if acc is None:
        acc = pd.Series(dtype="int64")
    return acc.astype(np.int64).rename(name), int(n_rows)


@click.command()
@click.option(
    "--edgelist-path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Follow edgelist (parquet/csv with source,target).",
)
@click.option("--top", "top_n", default=100, show_default=True, type=int)
@click.option(
    "--has-header",
    is_flag=True,
    default=False,
    help="CSV has a header row (default: headerless source,target,created_at).",
)
@click.option(
    "--columns",
    default="source,target,created_at",
    show_default=True,
    help="Column names for a headerless CSV.",
)
@click.option("--source-col", default="source", show_default=True)
@click.option("--target-col", default="target", show_default=True)
@click.option("--out", "out_dir", default="output/degrees", show_default=True)
def main(edgelist_path, top_n, has_header, columns, source_col, target_col, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = [c.strip() for c in columns.split(",") if c.strip()]

    print(f"counting out-degree (source={source_col}) ...")
    out_s, n_edges = _degree_series(
        edgelist_path, source_col, cols, has_header, "out_degree"
    )
    print(f"counting in-degree (target={target_col}) ...")
    in_s, _ = _degree_series(edgelist_path, target_col, cols, has_header, "in_degree")

    deg = pd.concat([out_s, in_s], axis=1).fillna(0).astype(np.int64)
    deg.index.name = "did"
    deg["total_degree"] = deg["out_degree"] + deg["in_degree"]
    deg = deg.sort_values("total_degree", ascending=False)

    n_nodes = len(deg)
    tot_endpoints = 2 * n_edges  # each edge contributes one out + one in
    print(f"\n[graph] {n_nodes:,} users, {n_edges:,} edges")
    q = deg["total_degree"].quantile([0.5, 0.9, 0.99, 0.999, 1.0])
    print(
        "[total-degree quantiles] "
        + "  ".join(f"p{int(p*1000)/10}={int(v):,}" for p, v in q.items())
    )
    for thr in (10_000, 100_000, 1_000_000):
        print(
            f"  users with total_degree > {thr:,}: "
            f"{int((deg['total_degree'] > thr).sum()):,}"
        )

    top = deg.head(top_n).reset_index()
    share = top["total_degree"].sum() / tot_endpoints if tot_endpoints else 0.0
    print(
        f"\n[concentration] top {top_n} users touch {share:.1%} of all "
        f"edge-endpoints (out+in). Filtering them removes that share of the "
        f"2-hop expansion pressure."
    )

    show = min(top_n, 30)
    print(f"\nTop {show} users by total degree:")
    print(f"  {'rank':>4}  {'out_deg':>12}  {'in_deg':>12}  {'total':>12}  did")
    for i, r in top.head(show).iterrows():
        print(
            f"  {i:>4}  {r['out_degree']:>12,}  {r['in_degree']:>12,}  "
            f"{r['total_degree']:>12,}  {r['did']}"
        )

    top.to_csv(out_dir / "top_degrees.csv", index=False)
    print(f"\nwrote {out_dir/'top_degrees.csv'} (top {top_n})")


if __name__ == "__main__":
    main()
