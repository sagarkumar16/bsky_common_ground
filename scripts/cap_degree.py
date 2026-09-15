"""
cap_degree.py
=============
Drop every account whose TOTAL degree (in + out) exceeds a cap from an edgelist
parquet -- the mega-hubs that blow up 2-hop neighborhoods (and time out the hops
stage). Removing a node only lowers other nodes' degrees, so a SINGLE pass is
enough: any node above the cap in the original graph is removed, and no surviving
node can then exceed it.

Streaming (memory bounded by node count, not edge count), so it runs on the huge
filtered_follows.parquet without loading it into RAM:
  pass 1  accumulate per-node out-degree + in-degree -> total; find offenders
  pass 2  stream, drop every edge touching an offender, write the capped parquet

    python scripts/cap_degree.py \
        --input filtered_follows.parquet --max-degree 500000 \
        --output filtered_follows_maxdeg500000.parquet
"""

import os
from pathlib import Path

import click
import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    raise SystemExit("cap_degree needs pyarrow: pip install pyarrow") from e


def total_degrees(path, source_col, target_col, batch_rows):
    """Per-node total degree (out+in), streamed. Returns (Series, n_edges)."""
    pf = pq.ParquetFile(path)
    outs = ins = None
    n_edges = 0
    for b in pf.iter_batches(columns=[source_col, target_col], batch_size=batch_rows):
        n_edges += b.num_rows
        so = pc.value_counts(b.column(0))
        ti = pc.value_counts(b.column(1))
        so_s = pd.Series(
            so.field("counts").to_numpy(), index=so.field("values").to_pylist()
        )
        ti_s = pd.Series(
            ti.field("counts").to_numpy(), index=ti.field("values").to_pylist()
        )
        outs = so_s if outs is None else outs.add(so_s, fill_value=0)
        ins = ti_s if ins is None else ins.add(ti_s, fill_value=0)
    if outs is None:
        return pd.Series(dtype="int64"), 0
    total = outs.add(ins, fill_value=0).astype(np.int64)
    return total, int(n_edges)


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Edgelist parquet to cap.",
)
@click.option(
    "--max-degree",
    default=500_000,
    show_default=True,
    type=int,
    help="Drop accounts whose total degree (in+out) exceeds this.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Destination parquet (default: <input>_maxdeg<N>.parquet). "
    "Ignored with --in-place.",
)
@click.option(
    "--in-place",
    is_flag=True,
    default=False,
    help="Replace the input atomically (temp file + rename).",
)
@click.option("--source-col", default="source", show_default=True)
@click.option("--target-col", default="target", show_default=True)
@click.option("--batch-rows", default=5_000_000, show_default=True, type=int)
def main(
    input_path, max_degree, output_path, in_place, source_col, target_col, batch_rows
):
    inp = Path(input_path)
    if in_place:
        out = inp.with_suffix(inp.suffix + ".tmp")
    elif output_path:
        out = Path(output_path)
    else:
        out = inp.with_name(f"{inp.stem}_maxdeg{max_degree}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"pass 1/2: computing total degree from {inp} ...")
    deg, n_edges = total_degrees(inp, source_col, target_col, batch_rows)
    offenders = set(deg.index[deg > max_degree].tolist())
    print(
        f"  {len(deg):,} accounts, {n_edges:,} edges; "
        f"max total_degree = {int(deg.max()) if len(deg) else 0:,}"
    )
    print(
        f"  {len(offenders):,} account(s) exceed {max_degree:,} "
        f"(total degree) -> dropping every edge touching them"
    )

    if not offenders:
        print("  nothing exceeds the cap.")
        if in_place:
            print(f"  {inp} left unchanged.")
            return
        # still materialize the output path so downstream steps can rely on it
        pq.write_table(pq.read_table(inp), out, compression="zstd")
        print(f"  wrote {out} (identical copy)")
        return

    exc = pa.array(sorted(offenders), type=pa.string())
    print(f"pass 2/2: writing capped edgelist -> {out} ...")
    pf = pq.ParquetFile(inp)
    writer = None
    kept = 0
    try:
        for b in pf.iter_batches(batch_size=batch_rows):
            tbl = pa.Table.from_batches([b])
            touches = pc.or_(
                pc.is_in(tbl[source_col], value_set=exc),
                pc.is_in(tbl[target_col], value_set=exc),
            )
            filt = tbl.filter(pc.invert(touches))
            if filt.num_rows:
                if writer is None:
                    writer = pq.ParquetWriter(out, filt.schema, compression="zstd")
                writer.write_table(filt)
                kept += filt.num_rows
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise click.ClickException("no edges survived -- cap too aggressive?")

    if in_place:
        os.replace(out, inp)
        final = inp
    else:
        final = out
    print(f"  kept {kept:,}/{n_edges:,} edges (dropped {n_edges - kept:,}) -> {final}")


if __name__ == "__main__":
    main()
