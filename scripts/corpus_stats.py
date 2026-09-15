"""
corpus_stats.py
===============
Headline corpus counts for one pipeline run (output/run_<stamp>_<mode>/):

  1. total starterpacks            -- lines in the UNFILTERED starterpacks.jsonl
  2. total nodes                   -- distinct member DIDs across those packs
  3. starterpacks after filtering  -- packs in the corpus the run used
  4. nodes after filtering         -- distinct member DIDs in that corpus
  5. following edges               -- rows in the run's follow edgelist (the
                                      induced subgraph on the filtered members;
                                      the degree-capped variant is reported too
                                      if it exists)
  6. posts                         -- app.bsky.feed.post records authored by the
                                      filtered nodes (raw count: no language /
                                      token / per-user-cap filtering)

Paths are read from the run's run_manifest.txt (written by
slurm/run_full_pipeline.sh); any of them can be overridden on the command line.
Older manifests lack the path lines, so pass --starterpacks / --edgelist then.

Usage
-----
    python scripts/corpus_stats.py --run-dir output/run_20260901_165251_directed
    python scripts/corpus_stats.py --run-dir ... --skip-posts   # fast: no records scan
"""

import gzip
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spcg.overlap_info import (  # noqa: E402
    _resolve_n_jobs,
    load_starterpacks,
    record_path_for_did,
)

POST_COLLECTION = "app.bsky.feed.post"


def read_manifest(run_dir):
    """Parse run_manifest.txt into a dict ('key : value' and 'K=V K=V' lines)."""
    path = Path(run_dir) / "run_manifest.txt"
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if " : " in line:
            k, v = line.split(" : ", 1)
            out[k.strip()] = v.strip()
        else:
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    out[k.strip()] = v.strip()
    return out


def pack_stats(path):
    """(n_packs, n_distinct_member_dids) for a starterpacks jsonl."""
    packs, _ = load_starterpacks(path)
    nodes = set()
    for members in packs:
        nodes |= members
    return len(packs), nodes


def edge_count(path):
    """Row count of an edgelist parquet, from metadata (no data read)."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def _count_user_posts(task):
    """Worker: number of post records in one user's records file (0 if absent)."""
    did, records_dir = task
    fpath = record_path_for_did(did, records_dir)
    n = 0
    try:
        f = gzip.open(fpath, "rt", encoding="utf-8")
    except FileNotFoundError:
        return 0, False
    import json

    with f:
        for line in f:
            # cheap substring screen before the JSON parse
            if POST_COLLECTION not in line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("commit", {}).get("collection") == POST_COLLECTION:
                n += 1
    return n, True


def post_count(dids, records_dir, n_jobs):
    """(total posts, users with a records file) over `dids`, in parallel."""
    tasks = [(d, records_dir) for d in sorted(dids)]
    total = found = 0
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for i, (n, ok) in enumerate(
            ex.map(_count_user_posts, tasks, chunksize=256), start=1
        ):
            total += n
            found += ok
            if i % 100_000 == 0:
                print(f"  scanned {i:,}/{len(tasks):,} users, {total:,} posts ...",
                      flush=True)
    return total, found


@click.command()
@click.option("--run-dir", required=True, type=click.Path(file_okay=False),
              help="Run output folder containing run_manifest.txt.")
@click.option("--starterpacks", "filtered_path", default=None,
              help="Filtered corpus the run used (default: manifest 'starterpacks').")
@click.option("--unfiltered", "unfiltered_path", default=None,
              help="Unfiltered corpus (default: starterpacks.jsonl next to the "
                   "filtered one).")
@click.option("--edgelist", default=None,
              help="Follow edgelist parquet (default: manifest 'edgelist').")
@click.option("--records-dir", default=None,
              help="Per-user records dir (default: <packs dir>/records).")
@click.option("--skip-posts", is_flag=True, default=False,
              help="Skip the (slow) scan of every user's records file.")
@click.option("--n-jobs", default=0, show_default=True, type=int,
              help="Workers for the posts scan (<=0: all available cores).")
def main(run_dir, filtered_path, unfiltered_path, edgelist, records_dir,
         skip_posts, n_jobs):
    run_dir = Path(run_dir)
    man = read_manifest(run_dir)

    filtered_path = filtered_path or man.get("starterpacks")
    if not filtered_path:
        raise click.UsageError(
            "no 'starterpacks' in run_manifest.txt; pass --starterpacks")
    filtered_path = Path(filtered_path)
    unfiltered_path = Path(unfiltered_path or filtered_path.parent / "starterpacks.jsonl")
    edgelist = edgelist or man.get("edgelist")
    records_dir = records_dir or filtered_path.parent / "records"

    rows = []

    print(f"[packs] unfiltered: {unfiltered_path}")
    n_packs_all, nodes_all = pack_stats(unfiltered_path)
    rows += [("total_starterpacks", n_packs_all), ("total_nodes", len(nodes_all))]
    del nodes_all

    print(f"[packs] filtered:   {filtered_path}")
    n_packs_f, nodes_f = pack_stats(filtered_path)
    rows += [("filtered_starterpacks", n_packs_f), ("filtered_nodes", len(nodes_f))]

    if edgelist and Path(edgelist).exists():
        print(f"[edges] {edgelist}")
        rows.append(("following_edges", edge_count(edgelist)))
        # degree-capped graph actually used for hops (run_full_pipeline.sh 3pre)
        max_deg = man.get("MAX_DEGREE")
        if max_deg and man.get("data_dir"):
            capped = Path(man["data_dir"]) / f"{Path(edgelist).stem}_maxdeg{max_deg}.parquet"
            if capped.exists():
                print(f"[edges] capped: {capped}")
                rows.append((f"following_edges_maxdeg{max_deg}", edge_count(capped)))
    else:
        print(f"[edges] edgelist not found ({edgelist}); skipping")

    if skip_posts:
        print("[posts] skipped (--skip-posts)")
    else:
        workers = _resolve_n_jobs(n_jobs)
        print(f"[posts] scanning {len(nodes_f):,} users in {records_dir} "
              f"({workers} workers) ...")
        n_posts, n_found = post_count(nodes_f, str(records_dir), workers)
        rows += [("posts", n_posts), ("users_with_records", n_found)]

    width = max(len(k) for k, _ in rows)
    lines = [f"{k:<{width}} : {v:,}" for k, v in rows]
    report = "\n".join(lines)
    print("\n" + report)
    out = run_dir / "corpus_stats.txt"
    out.write_text(report + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
