"""
make_null_eval_pairs.py
=======================
Build the SELECTION-UNBIASED evaluation pair table the configuration-model null
needs (see null_model.py / run_null_model.py).

The null reassigns pairs to shared-pack levels, so the evaluation set must be
chosen INDEPENDENTLY of the observed co-membership s_uv -- it must NOT be the
s-stratified census (which conditions on the statistic under test). Here we draw
a UNIFORM sample of user pairs among the candidate pool V_{>=2} (users in >= 2
packs), then attach the FIXED quantities the null holds constant:

  cosine  -- TF-IDF content similarity, via spcg.overlap_info.cosine_for_pairs
  hops    -- follow-graph distance,    via network_distance.pair_distances

Output: a table ``user_a,user_b,cosine,hops`` (dids) for run_null_model
``--eval-pairs-path``. Reuses existing functions read-only; defines no new
similarity/distance logic.
"""

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spcg.network_distance as nd  # noqa: E402
from spcg.overlap_info import build_user_pack_membership  # noqa: E402
from spcg.overlap_info import cosine_for_pairs, load_starterpacks


def sample_uniform_pairs(membership, n_pairs, seed=42, min_packs=2):
    """Uniformly sample ``n_pairs`` distinct unordered pairs from the candidate
    pool V_{>=min_packs} (users in >= min_packs packs). Selection depends only on
    membership counts, never on the pairwise co-membership s_uv."""
    cand = sorted(d for d, p in membership.items() if len(p) >= min_packs)
    if len(cand) < 2:
        raise ValueError("need >= 2 candidate users in V_{>=2}")
    rng = np.random.default_rng(seed)
    n_cand = len(cand)
    seen, a_out, b_out = set(), [], []
    attempts, max_attempts = 0, n_pairs * 200 + 1000
    while len(a_out) < n_pairs and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(n_cand))
        j = int(rng.integers(n_cand))
        if i == j:
            continue
        a, b = (cand[i], cand[j]) if cand[i] < cand[j] else (cand[j], cand[i])
        if (a, b) in seen:
            continue
        seen.add((a, b))
        a_out.append(a)
        b_out.append(b)
    if len(a_out) < n_pairs:
        print(
            f"[eval] only drew {len(a_out)}/{n_pairs} unique pairs "
            f"after {attempts} attempts"
        )
    return pd.DataFrame({"user_a": a_out, "user_b": b_out})


@click.command()
@click.option(
    "--starterpacks-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--records-dir",
    required=True,
    type=click.Path(exists=True, file_okay=False),
    help="Per-user records (for cosine).",
)
@click.option(
    "--edgelist-path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Reduced follow edgelist (for hops).",
)
@click.option(
    "--n-pairs",
    default=5000,
    show_default=True,
    type=int,
    help="Uniform pairs to draw from V_{>=2}.",
)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--min-tokens", default=50, show_default=True, type=int)
@click.option("--max-posts-per-user", default=1000, type=int)
@click.option("--max-hops", default=3, show_default=True, type=int)
@click.option("--directed", is_flag=True, default=False)
@click.option("--n-jobs", default=0, show_default=True, type=int)
@click.option(
    "--english-only/--no-english-only",
    default=True,
    show_default=True,
    help="Keep only English-detected posts (langid) for cosine. On by "
    "default; --no-english-only keeps all languages.",
)
@click.option("--out", "out_path", required=True)
def main(
    starterpacks_path,
    records_dir,
    edgelist_path,
    n_pairs,
    seed,
    min_tokens,
    max_posts_per_user,
    max_hops,
    directed,
    n_jobs,
    english_only,
    out_path,
):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("loading membership (all users) ...")
    packs, _ = load_starterpacks(starterpacks_path)
    membership = build_user_pack_membership(packs, users=None)

    pairs = sample_uniform_pairs(membership, n_pairs, seed=seed)
    print(f"sampled {len(pairs):,} uniform pairs among V>=2")

    print("attaching cosine (TF-IDF) for the sampled pairs ...")
    pairs = cosine_for_pairs(
        pairs,
        records_dir=records_dir,
        min_tokens=min_tokens,
        max_posts_per_user=max_posts_per_user,
        tfidf_kwargs=dict(min_df=10, max_df=0.4, sublinear_tf=True),
        n_jobs=n_jobs,
        english_only=english_only,
    )
    print(f"  {len(pairs):,} pairs with cosine (others dropped: no usable text)")

    print("attaching follow-graph hops ...")
    graph = nd.load_follow_graph(edgelist_path, directed=directed)
    pairs = nd.pair_distances(
        pairs, graph, max_hops=max_hops, n_jobs=n_jobs, directed=directed
    )

    eval_df = pairs[["user_a", "user_b", "cosine", "hops"]].copy()
    try:
        eval_df.to_parquet(out_path, index=False)
        print(f"wrote {out_path}")
    except Exception:
        alt = out_path.with_suffix(".csv")
        eval_df.to_csv(alt, index=False)
        print(f"wrote {alt}")


if __name__ == "__main__":
    main()
