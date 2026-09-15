"""
filter_follows.py
=================
Build a reduced, persisted follow edgelist (parquet) from the raw multi-TB
``follows.csv`` so the network-distance pipeline can load a CSR that fits in
memory (see network_distance.load_follow_graph / run_network_analysis.py).

Default reduction: the subgraph INDUCED on the study population -- the union of
all starterpack members. An edge is kept iff BOTH endpoints are members. This
covers every census user and their member-neighbors while shrinking the graph by
orders of magnitude, and it gives a principled "distance within the starterpack
ecosystem": pairs that are only connected through non-member intermediaries fall
into the "unreachable" cell, which the analysis treats as theoretically loaded
(strong communal common ground, no in-ecosystem network contact).

Efficiency: ONE streaming pass over follows.csv via pyarrow's C++ CSV reader,
reading only the source/target columns, filtering each batch with the C++
``is_in`` kernel against the member set, and streaming matches straight to
parquet. Memory is bounded by one batch + the member set (never the raw file).
Optional dedupe / reciprocal / min-degree refinements run as a cheap second
phase over the already-reduced parquet, not the raw file.

Usage
-----
    python filter_follows.py \
        --follows      /scratch/xee6vz/bluesky-graph/follows.csv \
        --starterpacks /scratch/xee6vz/bluesky-graph/starterpacks.jsonl \
        --output       /scratch/xee6vz/bluesky-graph/filtered_follows.parquet
"""

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "filter_follows needs pyarrow (parquet + streaming CSV): pip install pyarrow"
    ) from e


# ---------------------------------------------------------------------------
# Node universe
# ---------------------------------------------------------------------------
def build_universe(starterpacks_path=None, nodes_path=None):
    """Set of DIDs defining the kept-node universe.

    --nodes (one DID per line) wins if given; otherwise the union of all
    starterpack members, read via the existing load_starterpacks (read-only)."""
    if nodes_path:
        with open(nodes_path, "r", encoding="utf-8") as f:
            return {ln.strip() for ln in f if ln.strip()}
    if not starterpacks_path:
        raise click.UsageError("provide --starterpacks or --nodes for the universe")
    from spcg.overlap_info import load_starterpacks  # read-only reuse

    packs, _ = load_starterpacks(starterpacks_path)
    universe = set()
    for members in packs:
        universe |= members
    return universe


# ---------------------------------------------------------------------------
# Phase 1: streaming filter raw CSV -> parquet
# ---------------------------------------------------------------------------
def stream_filter(
    follows_path,
    output_path,
    universe,
    both_endpoints=True,
    keep_created_at=False,
    block_size_mb=256,
    has_header=False,
    columns=("source", "target", "created_at"),
    source_col="source",
    target_col="target",
    exclude=None,
):
    """One streaming pass: keep edges whose endpoints are in `universe`.

    `exclude`: an iterable of DIDs to DROP entirely -- any edge touching an
    excluded node is removed (e.g. the @bsky.app mega-hub that nearly every
    account follows on signup, which otherwise dominates 2-hop neighborhoods).

    The raw follows.csv is HEADERLESS by default, so we declare `columns`
    (positional names for every column in the file). Pass has_header=True for a
    file that already has a header row."""
    uni = pa.array(sorted(universe), type=pa.string())
    exc = pa.array(sorted(exclude), type=pa.string()) if exclude else None
    include = [source_col, target_col] + (["created_at"] if keep_created_at else [])

    if has_header:
        read_opts = pacsv.ReadOptions(block_size=block_size_mb * 1024 * 1024)
    else:
        # No header in the file -> name the columns ourselves; the first row is
        # then treated as data, not a header.
        read_opts = pacsv.ReadOptions(
            block_size=block_size_mb * 1024 * 1024, column_names=list(columns)
        )
    conv_opts = pacsv.ConvertOptions(
        include_columns=include,
        column_types={source_col: pa.string(), target_col: pa.string()},
    )
    reader = pacsv.open_csv(
        follows_path, read_options=read_opts, convert_options=conv_opts
    )

    writer = None
    scanned = kept = excluded = 0
    seen_nodes = set()
    try:
        for batch in reader:
            tbl = pa.Table.from_batches([batch])
            scanned += tbl.num_rows
            in_src = pc.is_in(tbl[source_col], value_set=uni)
            in_tgt = pc.is_in(tbl[target_col], value_set=uni)
            mask = pc.and_(in_src, in_tgt) if both_endpoints else pc.or_(in_src, in_tgt)
            if exc is not None:
                # drop any edge touching an excluded node (either endpoint)
                touches = pc.or_(
                    pc.is_in(tbl[source_col], value_set=exc),
                    pc.is_in(tbl[target_col], value_set=exc),
                )
                before = pc.sum(mask).as_py() or 0
                mask = pc.and_(mask, pc.invert(touches))
                excluded += before - (pc.sum(mask).as_py() or 0)
            filt = tbl.filter(mask)
            if filt.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path, filt.schema, compression="zstd"
                    )
                writer.write_table(filt)
                kept += filt.num_rows
                seen_nodes.update(filt[source_col].to_pylist())
                seen_nodes.update(filt[target_col].to_pylist())
            if scanned % (block_size_mb * 1024 * 1024 // 64) < tbl.num_rows:
                print(f"  scanned {scanned:,} edges, kept {kept:,} ...", flush=True)
    finally:
        if writer is not None:
            writer.close()
    return dict(scanned=scanned, kept=kept, nodes=len(seen_nodes), excluded=excluded)


# ---------------------------------------------------------------------------
# Phase 2 (optional): dedupe / reciprocal / min-degree over the reduced file
# ---------------------------------------------------------------------------
def refine(
    output_path,
    dedupe=True,
    reciprocal=False,
    min_degree=0,
    source_col="source",
    target_col="target",
):
    """Cheap second phase on the ALREADY-REDUCED parquet (fits in memory)."""
    import numpy as np
    import pandas as pd

    df = pq.read_table(output_path).to_pandas()
    n0 = len(df)

    if dedupe:
        df = df.drop_duplicates(subset=[source_col, target_col])

    if reciprocal:
        # keep only edges whose reverse also exists (mutual follows)
        rev = df[[target_col, source_col]].rename(
            columns={target_col: source_col, source_col: target_col}
        )
        df = df.merge(rev, on=[source_col, target_col], how="inner")
        df = df.drop_duplicates(subset=[source_col, target_col])

    if min_degree and min_degree > 0:
        # iteratively drop nodes whose total degree < min_degree
        while True:
            deg = pd.concat([df[source_col], df[target_col]]).value_counts()
            low = set(deg[deg < min_degree].index)
            if not low:
                break
            df = df[~df[source_col].isin(low) & ~df[target_col].isin(low)]

    nodes = pd.unique(pd.concat([df[source_col], df[target_col]]))
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, output_path, compression="zstd")
    return dict(before=n0, after=len(df), nodes=len(nodes))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--follows",
    "follows_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Raw follows.csv (columns source,target,created_at).",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Destination parquet for the filtered edgelist.",
)
@click.option(
    "--starterpacks",
    "starterpacks_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="starterpacks.jsonl; universe = union of all members (default).",
)
@click.option(
    "--nodes",
    "nodes_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Override universe with a DID-per-line node file.",
)
@click.option(
    "--either-endpoint",
    is_flag=True,
    default=False,
    help="Keep an edge if EITHER endpoint is in the universe "
    "(1-hop ego expansion). Default keeps only BOTH (induced).",
)
@click.option(
    "--has-header",
    is_flag=True,
    default=False,
    help="The CSV has a header row (default: headerless -- columns are "
    "named positionally via --columns).",
)
@click.option(
    "--columns",
    default="source,target,created_at",
    show_default=True,
    help="Comma-separated names for ALL columns of a headerless CSV "
    "(must match the file's column count).",
)
@click.option(
    "--source-col",
    default="source",
    show_default=True,
    help="Source column name (after --columns / header is applied).",
)
@click.option(
    "--target-col",
    default="target",
    show_default=True,
    help="Target column name (after --columns / header is applied).",
)
@click.option(
    "--keep-created-at",
    is_flag=True,
    default=False,
    help="Carry the created_at column through (default: drop it).",
)
@click.option(
    "--dedupe/--no-dedupe",
    default=True,
    show_default=True,
    help="Drop duplicate (source,target) edges.",
)
@click.option(
    "--reciprocal",
    is_flag=True,
    default=False,
    help="Keep only mutual (reverse-edge-exists) follows.",
)
@click.option(
    "--min-degree",
    default=0,
    show_default=True,
    type=int,
    help="Iteratively drop nodes with total degree below this.",
)
@click.option(
    "--exclude-dids",
    default=None,
    help="Comma-separated DIDs to drop entirely (any edge touching "
    "them is removed). E.g. the @bsky.app mega-hub "
    "(did:plc:z72i7hdynmk6r22z27h6tvur).",
)
@click.option(
    "--exclude-file",
    default=None,
    type=click.Path(dir_okay=False),
    help="File with one DID-to-exclude per line (merged with " "--exclude-dids).",
)
@click.option(
    "--block-size-mb",
    default=256,
    show_default=True,
    type=int,
    help="CSV reader batch size (memory per streamed block).",
)
def main(
    follows_path,
    output_path,
    starterpacks_path,
    nodes_path,
    either_endpoint,
    has_header,
    columns,
    source_col,
    target_col,
    keep_created_at,
    dedupe,
    reciprocal,
    min_degree,
    exclude_dids,
    exclude_file,
    block_size_mb,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    col_names = [c.strip() for c in columns.split(",") if c.strip()]

    exclude = set()
    if exclude_dids:
        exclude |= {d.strip() for d in exclude_dids.split(",") if d.strip()}
    if exclude_file:
        with open(exclude_file, "r", encoding="utf-8") as f:
            exclude |= {ln.strip() for ln in f if ln.strip()}
    if exclude:
        print(
            f"excluding {len(exclude):,} DID(s) entirely (edges touching them "
            f"are dropped)"
        )

    print("building node universe ...")
    universe = build_universe(starterpacks_path, nodes_path)
    print(
        f"  universe: {len(universe):,} users "
        f"({'either' if either_endpoint else 'both'} endpoint(s) must be in it)"
    )

    print(
        f"streaming-filtering {follows_path} -> {output_path} "
        f"({'header' if has_header else 'headerless: ' + ','.join(col_names)}) ..."
    )
    s = stream_filter(
        follows_path,
        str(output_path),
        universe,
        both_endpoints=not either_endpoint,
        keep_created_at=keep_created_at,
        block_size_mb=block_size_mb,
        has_header=has_header,
        columns=col_names,
        source_col=source_col,
        target_col=target_col,
        exclude=exclude,
    )
    print(
        f"[phase 1] scanned {s['scanned']:,} edges, kept {s['kept']:,} "
        f"({s['nodes']:,} distinct nodes"
        + (f"; dropped {s['excluded']:,} touching excluded DIDs)" if exclude else ")")
    )
    if s["kept"] == 0:
        raise click.ClickException("no edges kept -- check the universe / columns")

    if dedupe or reciprocal or (min_degree and min_degree > 0):
        print(
            f"[phase 2] refine (dedupe={dedupe}, reciprocal={reciprocal}, "
            f"min_degree={min_degree}) on the reduced file ..."
        )
        r = refine(
            str(output_path),
            dedupe=dedupe,
            reciprocal=reciprocal,
            min_degree=min_degree,
        )
        print(
            f"[phase 2] {r['before']:,} -> {r['after']:,} edges, "
            f"{r['nodes']:,} nodes"
        )

    print(f"Done. Wrote {output_path}")


if __name__ == "__main__":
    main()
