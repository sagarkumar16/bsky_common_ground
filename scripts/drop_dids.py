"""
drop_dids.py
============
Drop every edge touching a given DID (or DIDs) from an existing edgelist parquet
-- e.g. strip the @bsky.app mega-hub (did:plc:z72i7hdynmk6r22z27h6tvur) from
filtered_follows.parquet without a full rebuild.

Streams the parquet in row batches (memory bounded by one batch, not the edge
count) and writes the surviving edges to a new parquet. With --in-place it
writes to a temp file and atomically replaces the input.

    python scripts/drop_dids.py \
        --input /scratch/.../filtered_follows.parquet \
        --dids did:plc:z72i7hdynmk6r22z27h6tvur --in-place
"""

import os
from pathlib import Path

import click

try:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
except Exception as e:  # pragma: no cover
    raise SystemExit("drop_dids needs pyarrow: pip install pyarrow") from e


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Edgelist parquet to filter.",
)
@click.option(
    "--dids",
    default=None,
    help="Comma-separated DIDs to drop (any edge touching them).",
)
@click.option(
    "--dids-file",
    default=None,
    type=click.Path(dir_okay=False),
    help="File with one DID-to-drop per line (merged with --dids).",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    help="Destination parquet (default: <input stem>_nodids.parquet). "
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
    input_path,
    dids,
    dids_file,
    output_path,
    in_place,
    source_col,
    target_col,
    batch_rows,
):
    inp = Path(input_path)
    drop = set()
    if dids:
        drop |= {d.strip() for d in dids.split(",") if d.strip()}
    if dids_file:
        with open(dids_file, "r", encoding="utf-8") as f:
            drop |= {ln.strip() for ln in f if ln.strip()}
    if not drop:
        raise click.UsageError("provide --dids and/or --dids-file")

    if in_place:
        out = inp.with_suffix(inp.suffix + ".tmp")
    else:
        out = (
            Path(output_path)
            if output_path
            else inp.with_name(inp.stem + "_nodids.parquet")
        )
    out.parent.mkdir(parents=True, exist_ok=True)

    exc = pa.array(sorted(drop), type=pa.string())
    print(f"dropping edges touching {len(drop)} DID(s) from {inp} ...")

    pf = pq.ParquetFile(inp)
    writer = None
    scanned = kept = 0
    try:
        for batch in pf.iter_batches(batch_size=batch_rows):
            tbl = pa.Table.from_batches([batch])
            scanned += tbl.num_rows
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
        raise click.ClickException(
            "no edges survived -- refusing to write an empty file"
        )

    if in_place:
        os.replace(out, inp)
        final = inp
    else:
        final = out
    print(
        f"scanned {scanned:,} edges, kept {kept:,} "
        f"(dropped {scanned - kept:,}) -> {final}"
    )


if __name__ == "__main__":
    main()
