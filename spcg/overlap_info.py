"""
overlap_info.py
===============

Functions to add to your starterpack notebook to produce a plot of

    x = number of shared starterpacks between a pair of users
    y = average information flow for those pairs (with error bars)

on a *slice* of users (the full 2M x 2M is infeasible -- this lets you
characterise the relationship on a tractable subset first).

Typical use (paste into a notebook cell after running cell 1, which puts your
ProcessEntropy fork on sys.path):

    from spcg.overlap_info import (
        load_starterpacks, slice_by_packs, build_user_pack_membership,
        build_text_df_for_users, compute_pairwise_info,
        build_info_vs_overlap, aggregate_by_overlap, plot_info_vs_overlap,
        run_slice_analysis,
    )

    agg, pair_df = run_slice_analysis(
        starterpacks_path='bluesky-graph/starterpacks.jsonl',
        records_dir='bluesky-graph/records',
        n_packs=25, target_users=150,     # size of the slice
        info_col='flow',                   # or 'entropyStoT' / 'entropyTtoS'
        errorbar='sem',                    # 'sem' | 'std' | 'ci95'
        max_shared=4,                      # lump '>=4 shared' into one bin
    )

Design notes
------------
* Shared-pack counts use ALL starterpacks, so they're exact for the slice
  users even though the slice itself is chosen from a few packs.
* Pairs are ordered (directed) by default because information flow is
  directional. Pass symmetric=True to average the two directions per
  unordered pair.
* The slice is built by sampling whole starterpacks and unioning their
  members, which guarantees the x-axis spans more than {0, 1}. Random user
  sampling would leave almost every pair at 0 shared packs.
"""

import gzip
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

# pairwise_information_flow comes from your (modified) ProcessEntropy fork.
# Cell 1 of your notebook already does sys.path.insert(...); import lazily so
# this module can be imported regardless of order, and fail with a clear hint.
try:
    from ProcessEntropy.CrossEntropy import pairwise_information_flow
except Exception:  # pragma: no cover
    pairwise_information_flow = None


# ---------------------------------------------------------------------------
# 1. Starterpacks WITH identity (the notebook discarded pack identity)
# ---------------------------------------------------------------------------
def load_starterpacks(path="bluesky-graph/starterpacks.jsonl"):
    """
    Read starterpacks.jsonl into:
        packs : list[set[str]]  -- packs[i] = member DIDs of starterpack i
        names : list[str|None]  -- best-effort human name per pack (for labels)
    Pack identity is the line index i, which is stable and dependency-free.
    """
    packs, names = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sp = json.loads(line)
            packs.append({u["did"] for u in sp.get("members", [])})
            rec = sp.get("record", {}) or {}
            names.append(sp.get("name") or rec.get("name") or sp.get("uri"))
    return packs, names


# ---------------------------------------------------------------------------
# 2. user -> set of pack ids (incidence), restricted to a slice for memory
# ---------------------------------------------------------------------------
def build_user_pack_membership(packs, users=None):
    """
    Return {did: frozenset(pack_ids)}. If `users` (a set) is given, only those
    users are tracked, but membership is still counted across ALL packs so the
    shared-pack counts are exact.
    """
    membership = defaultdict(set)
    for pid, members in enumerate(packs):
        it = members if users is None else (members & users)
        for did in it:
            membership[did].add(pid)
    return {d: frozenset(s) for d, s in membership.items()}


# ---------------------------------------------------------------------------
# 3. Build a tractable slice that actually spans the overlap axis
# ---------------------------------------------------------------------------
def slice_by_packs(packs, n_packs=25, target_users=150, min_pack_size=2, seed=42):
    """
    Sample n_packs starterpacks, union their members into the slice, then (if
    needed) downsample to target_users. Returns a set of DIDs.

    Sampling whole packs is what gives you pairs that share 1, 2, 3+ packs.
    """
    rng = random.Random(seed)
    candidates = [i for i, m in enumerate(packs) if len(m) >= min_pack_size]
    chosen = rng.sample(candidates, min(n_packs, len(candidates)))
    users = set()
    for i in chosen:
        users |= packs[i]
    if target_users is not None and len(users) > target_users:
        users = set(rng.sample(sorted(users), target_users))
    return users


# ---------------------------------------------------------------------------
# 4. Text extraction for the slice (reuses the notebook's post logic)
# ---------------------------------------------------------------------------
def record_path_for_did(did, records_dir="bluesky-graph/records"):
    """records/<a>/<ab>/<abc>/<did>.jsonl.gz, deterministic from the DID."""
    suffix = did.split("did:plc:", 1)[-1]
    return Path(records_dir) / suffix[:1] / suffix[:2] / suffix[:3] / f"{did}.jsonl.gz"


def _process_user_gzip(fpath, include_quote_posts=True, english_only=False):
    """Posts for one user: list of {created_at, text, is_quote_post}.

    english_only: drop posts not detected as English (see lang_filter); content
    similarity is only meaningful within a single language. Off by default.
    """
    out = []
    try:
        f = gzip.open(fpath, "rt", encoding="utf-8")
    except FileNotFoundError:
        return out
    with f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if data.get("commit", {}).get("collection") != "app.bsky.feed.post":
                continue
            record = data.get("record", {})
            text = record.get("text", "").strip()
            if not text:
                continue
            text = " ".join(text.splitlines())
            is_quote = (
                record.get("embed", {}).get("$type", "") == "app.bsky.embed.record"
            )
            if is_quote and not include_quote_posts:
                continue
            if english_only:
                from spcg.lang_filter import is_english  # cached after first import

                if not is_english(text):
                    continue
            out.append(
                {
                    "created_at": record.get("createdAt"),
                    "text": text,
                    "is_quote_post": is_quote,
                }
            )
    return out


def _resolve_n_jobs(n_jobs):
    """Map an n_jobs request to a concrete worker count.

    n_jobs <= 0 (or None) means "use everything available": the Slurm
    allocation ($SLURM_CPUS_PER_TASK) if present, else os.cpu_count().
    """
    if n_jobs is not None and n_jobs > 0:
        return n_jobs
    env = os.environ.get("SLURM_CPUS_PER_TASK")
    if env and env.isdigit():
        return int(env)
    return os.cpu_count() or 1


def _load_user_posts(task):
    """Worker: read + parse one user's records. Module-level so it is picklable
    for ProcessPoolExecutor. Returns (did, texts, timestamps) sorted by time, or
    None if the user has no usable posts (or falls below the thresholds).

    Filtering (min_posts/min_tokens) and the max_posts_per_user tail-trim are
    done here so the parent only has to concatenate frames -- the per-user CPU
    cost (gzip decompression, JSON + datetime parsing) is what we fan out.
    """
    did, records_dir, min_posts, min_tokens, max_posts_per_user, english_only = task
    posts = _process_user_gzip(
        record_path_for_did(did, records_dir), english_only=english_only
    )
    if len(posts) < min_posts:
        return None
    if min_tokens and sum(len(p["text"].split()) for p in posts) < min_tokens:
        return None
    parsed = []
    for p in posts:
        try:
            t = datetime.fromisoformat(p["created_at"])
        except (TypeError, ValueError):
            continue
        if t.tzinfo is None:  # naive -> assume UTC
            t = t.replace(tzinfo=timezone.utc)
        else:  # aware -> convert to UTC
            t = t.astimezone(timezone.utc)
        parsed.append((t, p["text"]))
    if not parsed:
        return None
    parsed.sort(key=lambda r: r[0])
    if max_posts_per_user is not None and len(parsed) > max_posts_per_user:
        parsed = parsed[-max_posts_per_user:]
    ts = [t for t, _ in parsed]
    txt = [x for _, x in parsed]
    return did, txt, ts


def build_text_df_for_users(
    users,
    records_dir="bluesky-graph/records",
    min_posts=1,
    min_tokens=0,
    max_posts_per_user=None,
    n_jobs=1,
    english_only=False,
):
    """
    Build the combined text frame ProcessEntropy expects (text, created_at, uid)
    and a uid->did map. Users with no usable posts (or below thresholds) are
    dropped; uids are assigned only to kept users.

    max_posts_per_user: if set, keep only each user's most recent N posts. This
        is the most direct lever on per-pair cost -- cross-entropy time scales
        with sequence length, and a few prolific accounts dominate both the
        mean and the long tail. Keep it comfortably above ProcessEntropy's
        ~1000-token convergence floor (e.g. 1000-2000 posts).

    n_jobs: number of worker processes for reading per-user record files, which
        is the dominant cost on a large slice. <=1 reads serially; <=0 uses all
        available CPUs (or $SLURM_CPUS_PER_TASK). Users are processed in sorted
        DID order so uid assignment is deterministic regardless of worker count
        (the all-pairs cosine is invariant to uid labels either way).
    """
    users = sorted(users)
    tasks = [
        (did, records_dir, min_posts, min_tokens, max_posts_per_user, english_only)
        for did in users
    ]

    workers = _resolve_n_jobs(n_jobs)
    if workers > 1 and len(tasks) > 1:
        # chunk so each worker handles a batch of users per IPC round-trip
        chunksize = max(1, len(tasks) // (workers * 4))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_load_user_posts, tasks, chunksize=chunksize))
    else:
        results = [_load_user_posts(t) for t in tasks]

    frames, uid_to_did, uid = [], {}, 0
    for res in results:
        if res is None:
            continue
        did, txt, ts = res
        df = pd.DataFrame({"text": txt, "created_at": ts})
        df["uid"] = uid
        frames.append(df)
        uid_to_did[uid] = did
        uid += 1
    if not frames:
        raise ValueError("no users in the slice had usable posts")
    return pd.concat(frames, ignore_index=True), uid_to_did


# ---------------------------------------------------------------------------
# 5. Pairwise information for the slice (all pairs, via your fork)
# ---------------------------------------------------------------------------
def compute_pairwise_info(text_df):
    if pairwise_information_flow is None:
        raise ImportError(
            "ProcessEntropy.CrossEntropy.pairwise_information_flow not importable. "
            "Run notebook cell 1 (sys.path.insert to your fork) before importing this module."
        )
    return pairwise_information_flow(
        text_df, text_col="text", label_col="uid", time_col="created_at"
    )


# ---------------------------------------------------------------------------
# 6. Join information with shared-pack counts -> one row per pair
# ---------------------------------------------------------------------------
def build_info_vs_overlap(
    info_df, uid_to_did, membership, info_col="flow", symmetric=False
):
    """
    Returns a frame with columns:
        source_did, target_did, shared_packs, information
    `information` is taken from info_col ('flow' | 'entropyStoT' | 'entropyTtoS').
    """
    df = info_df.copy()
    df["source_did"] = df["source"].map(uid_to_did)
    df["target_did"] = df["target"].map(uid_to_did)

    def shared(a, b):
        return len(membership.get(a, frozenset()) & membership.get(b, frozenset()))

    df["shared_packs"] = [
        shared(s, t) for s, t in zip(df["source_did"], df["target_did"])
    ]
    df["information"] = df[info_col]

    if symmetric:
        df["_key"] = [
            tuple(sorted((s, t))) for s, t in zip(df["source_did"], df["target_did"])
        ]
        g = (
            df.groupby("_key")
            .agg(
                shared_packs=("shared_packs", "first"),
                information=("information", "mean"),
            )
            .reset_index()
        )
        g["source_did"] = g["_key"].str[0]
        g["target_did"] = g["_key"].str[1]
        df = g

    return df[["source_did", "target_did", "shared_packs", "information"]]


# ---------------------------------------------------------------------------
# 7. Aggregate by shared-pack count and plot
# ---------------------------------------------------------------------------
def _bootstrap_ci(vals, n_boot=2000, ci=95, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    means = vals[idx].mean(axis=1)
    lo = np.percentile(means, (100 - ci) / 2)
    hi = np.percentile(means, 100 - (100 - ci) / 2)
    return lo, hi


def aggregate_by_overlap(pair_df, max_shared=None, errorbar="sem"):
    """
    Group pairs by shared_packs, return per-bin mean/std/count and asymmetric
    error distances (err_low, err_high). If max_shared is set, all counts above
    it are lumped into the top bin (its label means '>= max_shared').
    """
    d = pair_df.copy()
    if max_shared is not None:
        d["shared_packs"] = d["shared_packs"].clip(upper=max_shared)

    rows = []
    for k, sub in d.groupby("shared_packs"):
        vals = sub["information"].to_numpy(dtype=float)
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        n = len(vals)
        if errorbar == "sem":
            e = std / np.sqrt(n) if n > 0 else 0.0
            lo = hi = e
        elif errorbar == "std":
            lo = hi = std
        elif errorbar == "ci95":
            clo, chi = _bootstrap_ci(vals)
            lo, hi = mean - clo, chi - mean
        else:
            lo = hi = 0.0
        rows.append((int(k), mean, std, n, lo, hi))

    return (
        pd.DataFrame(
            rows,
            columns=["shared_packs", "mean", "std", "count", "err_low", "err_high"],
        )
        .sort_values("shared_packs")
        .reset_index(drop=True)
    )


def plot_info_vs_overlap(
    agg_df,
    ax=None,
    info_label="Average information (flow)",
    max_shared=None,
    annotate_n=True,
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.errorbar(
        agg_df["shared_packs"],
        agg_df["mean"],
        yerr=[agg_df["err_low"], agg_df["err_high"]],
        fmt="o-",
        capsize=4,
        color="#333333",
        ecolor="#999999",
        linewidth=1.5,
    )
    if annotate_n:
        for _, r in agg_df.iterrows():
            ax.annotate(
                f"n={int(r['count'])}",
                (r["shared_packs"], r["mean"]),
                textcoords="offset points",
                xytext=(0, 9),
                fontsize=8,
                ha="center",
                color="#999999",
            )
    ax.set_xlabel("Number of shared starterpacks")
    ax.set_ylabel(info_label)
    labels = list(agg_df["shared_packs"])
    if max_shared is not None and labels and labels[-1] == max_shared:
        ax.set_xticks(labels)
        ax.set_xticklabels(
            [str(l) if l < max_shared else f"\u2265{max_shared}" for l in labels]
        )
    ax.spines[["top", "right"]].set_visible(False)
    return ax


# ---------------------------------------------------------------------------
# 8. End-to-end convenience wrapper
# ---------------------------------------------------------------------------
def run_slice_analysis(
    starterpacks_path="bluesky-graph/starterpacks.jsonl",
    records_dir="bluesky-graph/records",
    packs=None,
    n_packs=25,
    target_users=150,
    min_posts=1,
    min_tokens=0,
    max_posts_per_user=None,
    info_col="flow",
    symmetric=False,
    errorbar="sem",
    max_shared=4,
    plot=True,
    n_jobs=1,
):
    """
    Full slice pipeline. Returns (agg_df, pair_df). Set plot=False to skip the
    figure. Pass a preloaded `packs` to avoid re-reading starterpacks.jsonl.
    `max_posts_per_user` caps per-user history to bound per-pair cost.
    `n_jobs` parallelizes per-user record loading (see build_text_df_for_users).
    """
    if packs is None:
        packs, _ = load_starterpacks(starterpacks_path)

    users = slice_by_packs(packs, n_packs=n_packs, target_users=target_users)
    membership = build_user_pack_membership(packs, users=users)

    text_df, uid_to_did = build_text_df_for_users(
        users,
        records_dir=records_dir,
        min_posts=min_posts,
        min_tokens=min_tokens,
        max_posts_per_user=max_posts_per_user,
        n_jobs=n_jobs,
    )
    n = len(uid_to_did)
    print(
        f"slice: {n} users with posts, {len(text_df):,} posts total "
        f"(~{n * (n - 1):,} ordered pairs to compute)"
    )

    info_df = compute_pairwise_info(text_df)
    pair_df = build_info_vs_overlap(
        info_df, uid_to_did, membership, info_col=info_col, symmetric=symmetric
    )
    agg = aggregate_by_overlap(pair_df, max_shared=max_shared, errorbar=errorbar)

    if plot:
        plot_info_vs_overlap(
            agg, info_label=f"Average information ({info_col})", max_shared=max_shared
        )
        plt.tight_layout()
        plt.show()

    return agg, pair_df


# ===========================================================================
# SINGLE-USER ANALYSIS: information vs. following-distance (1..3 hops)
#
#   x = shortest following-distance from one ego to an alter (1, 2, or 3)
#   y = average ego<->alter information at that distance, with error bars
#
# This is ego-centric (ego vs each alter), NOT all-pairs. The error bars show
# how information varies ACROSS the alters at a given distance, for this one
# ego.
# ===========================================================================
try:
    from tqdm.notebook import tqdm as _tqdm
except Exception:  # plain script / no ipywidgets
    try:
        from tqdm import tqdm as _tqdm
    except Exception:  # tqdm absent -> no-op

        def _tqdm(it, **k):
            return it


def _user_text_frame(did, records_dir="bluesky-graph/records", max_posts_per_user=None):
    """One user's posts as a (text, created_at) frame, or None if none usable."""
    posts = _process_user_gzip(record_path_for_did(did, records_dir))
    ts, txt = [], []
    for p in posts:
        try:
            t = datetime.fromisoformat(p["created_at"])
        except (TypeError, ValueError):
            continue
        # normalize to UTC-aware so sort_values never mixes naive/aware stamps
        t = (
            t.replace(tzinfo=timezone.utc)
            if t.tzinfo is None
            else t.astimezone(timezone.utc)
        )
        ts.append(t)
        txt.append(p["text"])
    if not txt:
        return None
    df = pd.DataFrame({"text": txt, "created_at": ts}).sort_values("created_at")
    if max_posts_per_user is not None and len(df) > max_posts_per_user:
        df = df.tail(max_posts_per_user)
    return df.reset_index(drop=True)


def ego_neighbors_by_distance(
    follows, ego_did, n_hops=3, max_frontier=20000, seed=42, verbose=True
):
    """
    BFS out from a single ego over the `follows` edgelist along FOLLOWING edges
    (source follows target). Returns {hop: set(dids)} where each set is the
    nodes whose shortest following-distance from the ego is exactly `hop`.

    `follows` is any DataFrame-like with 'source'/'target' columns -- a Dask
    frame over the raw multi-TB CSV, OR a smaller pre-filtered/parquet edgelist
    (recommended: 3 full scans of the raw file for one user is expensive, so
    point this at a reduced edgelist if you have one).

    Each set returned is COMPLETE and its distance labels are EXACT. Only the
    EXPANSION frontier is capped: if a frontier exceeds `max_frontier` it is
    sampled before the next hop, so deeper sets may be missing some nodes that
    were only reachable through dropped ones (no node is ever mislabeled).
    For follower-distance instead, pass a frame with source/target swapped.
    """
    rng = random.Random(seed)
    seen = {ego_did}
    frontier = {ego_did}
    by_dist = {}

    for hop in range(1, n_hops + 1):
        if len(frontier) == 1:
            (only,) = tuple(frontier)
            sub = follows[follows["source"] == only]
        else:
            sub = follows[follows["source"].isin(frontier)]

        targets = sub["target"].dropna().unique()
        # Dask frames need .compute(); pandas frames are already arrays
        targets = targets.compute() if hasattr(targets, "compute") else targets

        new = set(np.asarray(targets).tolist()) - seen
        by_dist[hop] = new
        seen |= new
        if verbose:
            print(f"  distance {hop}: {len(new):,} users")

        if max_frontier is not None and len(new) > max_frontier:
            frontier = set(rng.sample(sorted(new), max_frontier))
            if verbose:
                print(f"    (capped expansion frontier to {max_frontier:,})")
        else:
            frontier = new
        if not frontier:
            break

    return by_dist


def ego_alter_info_by_distance(
    ego_did,
    by_dist,
    records_dir="bluesky-graph/records",
    min_posts=5,
    max_posts_per_user=None,
    max_alters_per_distance=60,
    seed=42,
):
    """
    For each sampled alter at each distance, compute ego<->alter information via
    pairwise_information_flow on the 2-user (ego, alter) frame and keep the
    source==ego row (which carries both directions). Returns a long DataFrame:
        alter_did, distance, flow, entropyStoT, entropyTtoS
    (entropyStoT = ego->alter, entropyTtoS = alter->ego).
    """
    if pairwise_information_flow is None:
        raise ImportError(
            "run notebook cell 1 to put your ProcessEntropy fork on sys.path"
        )

    ego_df = _user_text_frame(ego_did, records_dir, max_posts_per_user)
    if ego_df is None or len(ego_df) < min_posts:
        raise ValueError(
            f"ego {ego_did} has too few usable posts ({0 if ego_df is None else len(ego_df)})"
        )
    ego_df = ego_df.assign(uid=0)[["text", "created_at", "uid"]]

    rng = random.Random(seed)
    rows = []
    for hop, nodes in sorted(by_dist.items()):
        nodes = sorted(nodes)
        if max_alters_per_distance and len(nodes) > max_alters_per_distance:
            nodes = rng.sample(nodes, max_alters_per_distance)

        for a in _tqdm(nodes, desc=f"distance {hop}"):
            adf = _user_text_frame(a, records_dir, max_posts_per_user)
            if adf is None or len(adf) < min_posts:
                continue
            adf = adf.assign(uid=1)[["text", "created_at", "uid"]]
            pair = pd.concat([ego_df, adf], ignore_index=True)
            try:
                res = pairwise_information_flow(
                    pair, text_col="text", label_col="uid", time_col="created_at"
                )
            except Exception:
                continue
            r = res[res["source"] == 0]
            if len(r) == 0:
                continue
            r = r.iloc[0]
            rows.append(
                (
                    a,
                    hop,
                    float(r["flow"]),
                    float(r["entropyStoT"]),
                    float(r["entropyTtoS"]),
                )
            )

    return pd.DataFrame(
        rows, columns=["alter_did", "distance", "flow", "entropyStoT", "entropyTtoS"]
    )


def aggregate_by_distance(pair_df, info_col="flow", errorbar="sem"):
    """Group by distance -> per-distance mean/std/count and error distances."""
    rows = []
    for k, sub in pair_df.groupby("distance"):
        vals = sub[info_col].to_numpy(dtype=float)
        mean = float(np.nanmean(vals)) if len(vals) else np.nan
        std = float(np.nanstd(vals, ddof=1)) if len(vals) > 1 else 0.0
        n = len(vals)
        if errorbar == "sem":
            lo = hi = std / np.sqrt(n) if n > 0 else 0.0
        elif errorbar == "std":
            lo = hi = std
        elif errorbar == "ci95":
            clo, chi = _bootstrap_ci(vals)
            lo, hi = mean - clo, chi - mean
        else:
            lo = hi = 0.0
        rows.append((int(k), mean, std, n, lo, hi))
    return (
        pd.DataFrame(
            rows, columns=["distance", "mean", "std", "count", "err_low", "err_high"]
        )
        .sort_values("distance")
        .reset_index(drop=True)
    )


def plot_info_vs_distance(agg_df, info_col="flow", ax=None, annotate_n=True):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4), facecolor="white")
    ax.errorbar(
        agg_df["distance"],
        agg_df["mean"],
        yerr=[agg_df["err_low"], agg_df["err_high"]],
        fmt="o-",
        capsize=4,
        color="#185FA5",
        ecolor="#999999",
        linewidth=1.5,
    )
    if annotate_n:
        for _, r in agg_df.iterrows():
            ax.annotate(
                f"n={int(r['count'])}",
                (r["distance"], r["mean"]),
                textcoords="offset points",
                xytext=(0, 9),
                fontsize=8,
                ha="center",
                color="#999999",
            )
    ax.set_xlabel("following-distance from ego")
    ax.set_ylabel(f"average information ({info_col})")
    ax.set_xticks(sorted(agg_df["distance"].unique()))
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def run_single_user_distance_analysis(
    follows,
    ego_did,
    records_dir="bluesky-graph/records",
    n_hops=3,
    max_frontier=20000,
    max_alters_per_distance=60,
    min_posts=5,
    max_posts_per_user=1500,
    info_col="flow",
    errorbar="sem",
    seed=42,
    plot=True,
):
    """
    End-to-end single-user distance analysis. Returns (agg_df, pair_df).
    BFS cost = up to n_hops scans of `follows`; info cost is bounded by
    max_alters_per_distance (e.g. 3 x 60 = 180 pairs ~ a couple of minutes).
    """
    print(f"BFS from {ego_did} (following edges, up to {n_hops} hops)")
    by_dist = ego_neighbors_by_distance(
        follows, ego_did, n_hops=n_hops, max_frontier=max_frontier, seed=seed
    )
    pair_df = ego_alter_info_by_distance(
        ego_did,
        by_dist,
        records_dir=records_dir,
        min_posts=min_posts,
        max_posts_per_user=max_posts_per_user,
        max_alters_per_distance=max_alters_per_distance,
        seed=seed,
    )
    print(f"computed information for {len(pair_df)} ego-alter pairs")
    agg = aggregate_by_distance(pair_df, info_col=info_col, errorbar=errorbar)
    if plot:
        plot_info_vs_distance(agg, info_col=info_col)
        plt.tight_layout()
        plt.show()
    return agg, pair_df


# ===========================================================================
# ALTERNATIVE MEASURE: TF-IDF cosine similarity between users
#
# Swaps the temporal, directional cross-entropy "flow" for a static, symmetric
# content-similarity score. Each user is ONE document (their posts concatenated);
# terms = words + hashtags, stopwords dropped, weighted by IDF, L2-normalized;
# pair score = cosine = dot product of the two normalized vectors.
#
# Plugs into the SAME downstream functions:
#   overlap : tfidf_cosine_info_df -> build_info_vs_overlap(info_col='cosine')
#             -> aggregate_by_overlap -> plot_info_vs_overlap
#   distance: run_single_user_distance_cosine (mirrors the cross-entropy version)
#
# Properties that differ from flow/cross-entropy, and why they matter here:
#   * SYMMETRIC   -> no +f/-f cancellation against a symmetric x-axis (the
#                    all-zeros bug); cosine(i,j) == cosine(j,i).
#   * ATEMPORAL   -> no "source after target" warning; ignores post timing
#                    entirely, so it cannot capture direction or influence.
#   * L2-NORMED   -> scale/length-invariant, so far less length-confounded.
#   * CHEAP       -> vectorize once (O(users)); all-pairs is a single sparse
#                    matmul. The 245 ms/pair cost is gone.
# Interpretation flips again: HIGHER cosine = MORE shared content (the intuitive
# direction), unlike cross-entropy where lower = more shared.
# ===========================================================================
import re

try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    _HAVE_SKLEARN = True
except Exception:
    TfidfVectorizer = None
    ENGLISH_STOP_WORDS = frozenset()
    _HAVE_SKLEARN = False

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@[\w.]+")
# a token is either a #hashtag or an alphabetic word (apostrophes allowed)
_TOKEN_RE = re.compile(r"#\w+|[a-z][a-z']+")


def make_content_tokenizer(extra_stopwords=None, keep_mentions=False, min_len=2):
    """
    text -> list of content tokens. Lowercases, strips URLs (and @mentions
    unless keep_mentions), keeps #hashtags as distinct tokens (so '#climate' !=
    'climate'), and drops stopwords + sub-min_len words. Hashtags are kept even
    if their bare word is a stopword, since they're intentional content markers.
    """
    stop = set(ENGLISH_STOP_WORDS)
    if extra_stopwords:
        stop |= set(extra_stopwords)

    def tok(text):
        text = text.lower()
        text = _URL_RE.sub(" ", text)
        if not keep_mentions:
            text = _MENTION_RE.sub(" ", text)
        out = []
        for m in _TOKEN_RE.findall(text):
            if m[0] == "#":
                if len(m) > 1:
                    out.append(m)
            elif len(m) >= min_len and m not in stop:
                out.append(m)
        return out

    return tok


def build_user_tfidf(
    text_df,
    min_df=2,
    max_df=1.0,
    sublinear_tf=False,
    keep_mentions=False,
    extra_stopwords=None,
    vectorizer=None,
):
    """
    One TF-IDF vector per user (document = a user's concatenated posts).
    Returns (X, uids, vectorizer):
        X    : L2-normalized sparse matrix, row i = user uids[i]
        uids : np.ndarray of uid per row (sorted)
    Pass a pre-fit `vectorizer` to reuse a FIXED idf (e.g. a global background)
    for comparability across slices/distances; otherwise idf is fit on this set.

    min_df=2 drops terms used by only one user: such terms can never appear in a
    pair's cosine numerator (both users must share the term), they only affect
    norms, and dropping them shrinks the vocabulary a lot. Set min_df=1 to keep
    the literal full vocabulary. sublinear_tf=True (1+log tf) damps heavy
    repetition from prolific users and is a good robustness toggle.
    """
    if not _HAVE_SKLEARN:
        raise ImportError("TF-IDF cosine needs scikit-learn: pip install scikit-learn")

    grouped = text_df.groupby("uid")["text"].apply(lambda s: " ".join(s)).sort_index()
    uids = grouped.index.to_numpy()
    docs = grouped.tolist()

    tok = make_content_tokenizer(
        extra_stopwords=extra_stopwords, keep_mentions=keep_mentions
    )
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            tokenizer=tok,
            token_pattern=None,
            lowercase=False,
            norm="l2",
            sublinear_tf=sublinear_tf,
            min_df=min_df,
            max_df=max_df,
        )
        X = vectorizer.fit_transform(docs)
    else:
        X = vectorizer.transform(docs)
    return X, uids, vectorizer


def tfidf_cosine_info_df(text_df, max_users_dense=6000, **tfidf_kwargs):
    """
    All-pairs cosine for the slice, shaped like the cross-entropy info_df so it
    feeds build_info_vs_overlap directly. Returns columns: source, target,
    cosine (one row per unordered pair, uid-keyed). INCLUDES zero-cosine pairs
    (users sharing no vocabulary) so the bin means aren't biased upward.
    """
    X, uids, _ = build_user_tfidf(text_df, **tfidf_kwargs)
    n = len(uids)
    if n > max_users_dense:
        print(
            f"[cosine] n={n} users -> {n*n} dense similarity matrix; "
            f"for large n compute it blocked or use approximate-NN top-k."
        )
    S = (X @ X.T).toarray()
    iu, ju = np.triu_indices(n, k=1)
    return pd.DataFrame({"source": uids[iu], "target": uids[ju], "cosine": S[iu, ju]})


def run_overlap_cosine_analysis(
    packs=None,
    starterpacks_path="bluesky-graph/starterpacks.jsonl",
    records_dir="bluesky-graph/records",
    n_packs=25,
    target_users=150,
    min_posts=1,
    min_tokens=0,
    max_posts_per_user=None,
    max_shared=4,
    errorbar="sem",
    tfidf_kwargs=None,
    plot=True,
    n_jobs=1,
):
    """Overlap analysis (x = shared starterpacks) using TF-IDF cosine.

    `n_jobs` parallelizes per-user record loading (see build_text_df_for_users).
    """
    if packs is None:
        packs, _ = load_starterpacks(starterpacks_path)
    users = slice_by_packs(packs, n_packs=n_packs, target_users=target_users)
    membership = build_user_pack_membership(packs, users=users)
    text_df, uid_to_did = build_text_df_for_users(
        users,
        records_dir=records_dir,
        min_posts=min_posts,
        min_tokens=min_tokens,
        max_posts_per_user=max_posts_per_user,
        n_jobs=n_jobs,
    )
    print(f"slice: {len(uid_to_did)} users, {len(text_df):,} posts")

    info_df = tfidf_cosine_info_df(text_df, **(tfidf_kwargs or {}))
    pair_df = build_info_vs_overlap(
        info_df, uid_to_did, membership, info_col="cosine", symmetric=False
    )
    agg = aggregate_by_overlap(pair_df, max_shared=max_shared, errorbar=errorbar)
    if plot:
        plot_info_vs_overlap(
            agg, info_label="mean cosine similarity", max_shared=max_shared
        )
        plt.tight_layout()
        plt.show()
    return agg, pair_df, info_df


def run_single_user_distance_cosine(
    follows,
    ego_did,
    records_dir="bluesky-graph/records",
    n_hops=3,
    max_frontier=20000,
    max_alters_per_distance=60,
    min_posts=5,
    max_posts_per_user=1500,
    errorbar="sem",
    tfidf_kwargs=None,
    seed=42,
    plot=True,
):
    """Single-user distance analysis (x = following-distance) using cosine."""
    by_dist = ego_neighbors_by_distance(
        follows, ego_did, n_hops=n_hops, max_frontier=max_frontier, seed=seed
    )
    rng = random.Random(seed)
    did_distance, selected = {}, [ego_did]
    for hop, nodes in sorted(by_dist.items()):
        nodes = sorted(nodes)
        if max_alters_per_distance and len(nodes) > max_alters_per_distance:
            nodes = rng.sample(nodes, max_alters_per_distance)
        for d in nodes:
            did_distance[d] = hop
            selected.append(d)

    text_df, uid_to_did = build_text_df_for_users(
        selected,
        records_dir=records_dir,
        min_posts=min_posts,
        max_posts_per_user=max_posts_per_user,
    )

    X, uids, _ = build_user_tfidf(text_df, **(tfidf_kwargs or {}))
    uid_to_row = {u: i for i, u in enumerate(uids)}
    did_to_uid = {v: k for k, v in uid_to_did.items()}
    if ego_did not in did_to_uid or did_to_uid[ego_did] not in uid_to_row:
        raise ValueError("ego has too few usable posts to vectorize")
    ego_row = uid_to_row[did_to_uid[ego_did]]
    ego_vec = X[ego_row]

    rows = []
    for u, r in uid_to_row.items():
        did = uid_to_did[u]
        if did == ego_did:
            continue
        cos = float(ego_vec.multiply(X[r]).sum())  # both L2-normed -> dot = cosine
        rows.append((did, did_distance[did], cos))
    pair_df = pd.DataFrame(rows, columns=["alter_did", "distance", "cosine"])

    agg = aggregate_by_distance(pair_df, info_col="cosine", errorbar=errorbar)
    if plot:
        plot_info_vs_distance(agg, info_col="cosine")
        plt.tight_layout()
        plt.show()
    return agg, pair_df


# ===========================================================================
# CENSUS PAIR SELECTION + USER-CLUSTERED BOOTSTRAP CIs
#
# Replaces the "sample users -> score ALL pairs -> SEM" path for the
# Shared-Packs-vs-Cosine plot with two rigour upgrades:
#
#   1. Census pair selection (census_comember_pairs): instead of sampling users
#      and hoping the slice spans the overlap axis, we enumerate co-member pairs
#      exactly from an incidence matrix, so every shared-pack bin 1..K is
#      populated (capped per level for tractability) and a matched set of
#      zero-overlap baseline pairs anchors bin 0. Reproducible from `seed`.
#
#   2. User-clustered bootstrap (cluster_bootstrap_ci): SEM assumes independent
#      observations, but each user appears in many pairs, so pair-level errors
#      are badly under-stated. We resample USERS (not pairs) with replacement and
#      weight each pair by the product of its two endpoints' draw multiplicities
#      (a dyadic node bootstrap), which propagates that dependence into the CIs.
#
# Only the listed pairs are ever scored (row dot products) -- no dense N x N
# similarity matrix is built. Feeds plot_info_vs_overlap unchanged.
# ===========================================================================
def census_comember_pairs(
    membership,
    candidate_users=None,
    per_level=1000,
    baseline_pairs=1000,
    max_shared=5,
    seed=42,
):
    """
    Enumerate user pairs by exact shared-pack count, plus a matched zero-overlap
    baseline, deterministically.

    membership      : {did: set/frozenset(pack_id)} over ALL users (build with
                      build_user_pack_membership(packs); do NOT restrict to a slice).
    candidate_users : pool to enumerate co-member pairs from. Default = every user
                      in >=2 packs (where all the 2/3/4/5+ overlap lives, and a
                      small enough pool to form an incidence matrix on).
    per_level       : cap per shared-pack level (keep all if fewer exist).
    baseline_pairs  : number of disjoint (zero shared pack) pairs to draw for bin 0.
    max_shared      : levels are min(count, max_shared); the top level is ">=K".

    Returns a DataFrame with columns user_a, user_b, shared_packs (dids; the
    stored shared_packs is the EXACT count, clipping happens downstream).
    """
    rng = np.random.default_rng(seed)

    if candidate_users is None:
        candidate_users = [d for d, packs in membership.items() if len(packs) >= 2]
    cand = sorted(candidate_users)
    if len(cand) < 2:
        raise ValueError("need >=2 candidate users to enumerate co-member pairs")
    row_of = {d: i for i, d in enumerate(cand)}

    # Binary incidence matrix M (rows = candidate users, cols = pack ids).
    rows, cols = [], []
    for d in cand:
        for pid in membership[d]:
            rows.append(row_of[d])
            cols.append(int(pid))
    if not cols:
        raise ValueError("candidate users have no pack memberships")
    n_cols = max(cols) + 1
    M = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(cand), n_cols),
    )

    # S = M M^T holds shared-pack counts; only the upper triangle (i<j) nonzeros
    # are real co-member pairs. Never densified beyond the candidate pool.
    S = sparse.triu(M @ M.T, k=1).tocoo()
    ii, jj = S.row, S.col
    counts = np.rint(S.data).astype(int)

    # Sort by (i, j) so sampling is reproducible regardless of sparse op ordering.
    order = np.lexsort((jj, ii))
    ii, jj, counts = ii[order], jj[order], counts[order]
    levels = np.minimum(counts, max_shared)

    rec_a, rec_b, rec_shared = [], [], []
    for lvl in range(1, max_shared + 1):
        sel = np.where(levels == lvl)[0]
        if len(sel) > per_level:
            sel = np.sort(rng.choice(sel, size=per_level, replace=False))
        for idx in sel:
            rec_a.append(cand[ii[idx]])
            rec_b.append(cand[jj[idx]])
            rec_shared.append(int(counts[idx]))

    # Baseline: random disjoint pairs (no shared pack) over ALL users -> bin 0.
    all_users = sorted(membership.keys())
    n_all = len(all_users)
    seen = set()
    n_base, attempts, max_attempts = 0, 0, baseline_pairs * 200
    while n_base < baseline_pairs and attempts < max_attempts:
        attempts += 1
        a = all_users[int(rng.integers(n_all))]
        b = all_users[int(rng.integers(n_all))]
        if a == b:
            continue
        key = (a, b) if a < b else (b, a)
        if key in seen:
            continue
        if membership.get(a, frozenset()) & membership.get(b, frozenset()):
            continue
        seen.add(key)
        rec_a.append(key[0])
        rec_b.append(key[1])
        rec_shared.append(0)
        n_base += 1
    if n_base < baseline_pairs:
        print(
            f"[census] only drew {n_base}/{baseline_pairs} baseline pairs "
            f"after {attempts} attempts"
        )

    return pd.DataFrame({"user_a": rec_a, "user_b": rec_b, "shared_packs": rec_shared})


def cosine_for_pairs(
    pairs_df,
    records_dir="bluesky-graph/records",
    min_tokens=50,
    max_posts_per_user=None,
    tfidf_kwargs=None,
    n_jobs=1,
    english_only=False,
):
    """
    Score TF-IDF cosine for exactly the pairs in `pairs_df` via per-row dot
    products (rows are L2-normalized, so dot == cosine). NEVER materializes a
    dense users x users similarity matrix.

    Loads text only for the users that actually appear in the pairs, vectorizes
    once, then for each pair with text on both sides computes X[ra].(X[rb]).
    Pairs where either user has no usable text are dropped (count reported).

    Returns `pairs_df` with an added `cosine` column.
    """
    needed = pd.unique(pd.concat([pairs_df["user_a"], pairs_df["user_b"]]))
    text_df, uid_to_did = build_text_df_for_users(
        list(needed),
        records_dir=records_dir,
        min_tokens=min_tokens,
        max_posts_per_user=max_posts_per_user,
        n_jobs=n_jobs,
        english_only=english_only,
    )

    X, uids, _ = build_user_tfidf(text_df, **(tfidf_kwargs or {}))
    did_to_row = {uid_to_did[u]: i for i, u in enumerate(uids)}

    cosines, keep = [], []
    for a, b in zip(pairs_df["user_a"], pairs_df["user_b"]):
        ra, rb = did_to_row.get(a), did_to_row.get(b)
        if ra is None or rb is None:
            keep.append(False)
            continue
        cosines.append(float(X[ra].multiply(X[rb]).sum()))
        keep.append(True)

    keep = np.asarray(keep)
    n_dropped = int((~keep).sum())
    if n_dropped:
        print(
            f"[cosine] dropped {n_dropped}/{len(pairs_df)} pairs "
            f"(a user had no usable text)"
        )

    out = pairs_df.loc[keep].copy()
    out["cosine"] = cosines
    return out.reset_index(drop=True)


def cluster_bootstrap_ci(
    pair_df,
    value_col="cosine",
    group_col="shared_packs",
    n_boot=500,
    seed=42,
    max_shared=5,
):
    """
    User-clustered (dyadic node) bootstrap CIs per shared-pack bin.

    Each iteration resamples USERS with replacement; a pair's weight is the
    product of its two endpoints' draw multiplicities, so a user pulled k times
    multiplies the influence of all its pairs. The per-bin statistic is the
    weighted mean sum(w*v)/sum(w). This respects that each user appears in many
    pairs -- the dependence SEM ignores.

    Point estimate per bin = unweighted mean over the actual pairs.
    CIs = 2.5/97.5 nan-aware percentiles of the bootstrap weighted means.

    Returns the same schema as aggregate_by_overlap so plot_info_vs_overlap
    works unchanged: shared_packs, mean, std, count, err_low, err_high
    (plus ci_low, ci_high). Deterministic given `seed`.
    """
    d = pair_df.copy()
    d[group_col] = d[group_col].clip(upper=max_shared)

    users = pd.unique(pd.concat([d["user_a"], d["user_b"]]))
    user_idx = {u: i for i, u in enumerate(users)}
    n_users = len(users)

    a_idx = d["user_a"].map(user_idx).to_numpy()
    b_idx = d["user_b"].map(user_idx).to_numpy()
    vals = d[value_col].to_numpy(dtype=float)
    bins = d[group_col].to_numpy(dtype=int)
    uniq_bins = np.array(sorted(np.unique(bins)))

    rng = np.random.default_rng(seed)
    boot = np.full((n_boot, len(uniq_bins)), np.nan)
    for bi in range(n_boot):
        draws = rng.integers(0, n_users, size=n_users)
        mult = np.bincount(draws, minlength=n_users)
        w = mult[a_idx] * mult[b_idx]
        for k, lvl in enumerate(uniq_bins):
            mask = bins == lvl
            wt = w[mask]
            tot = wt.sum()
            if tot > 0:
                boot[bi, k] = float((wt * vals[mask]).sum() / tot)

    rows = []
    for k, lvl in enumerate(uniq_bins):
        mask = bins == lvl
        v = vals[mask]
        mean = float(np.mean(v)) if len(v) else np.nan
        std = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        col = boot[:, k]
        if np.all(np.isnan(col)):
            ci_low = ci_high = np.nan
        else:
            ci_low = float(np.nanpercentile(col, 2.5))
            ci_high = float(np.nanpercentile(col, 97.5))
        rows.append(
            (
                int(lvl),
                mean,
                std,
                int(len(v)),
                mean - ci_low,
                ci_high - mean,
                ci_low,
                ci_high,
            )
        )

    return (
        pd.DataFrame(
            rows,
            columns=[
                "shared_packs",
                "mean",
                "std",
                "count",
                "err_low",
                "err_high",
                "ci_low",
                "ci_high",
            ],
        )
        .sort_values("shared_packs")
        .reset_index(drop=True)
    )


def run_overlap_cosine_census(
    starterpacks_path="bluesky-graph/starterpacks.jsonl",
    records_dir="bluesky-graph/records",
    candidate_users=None,
    per_level=1000,
    baseline_pairs=1000,
    max_shared=5,
    min_tokens=50,
    max_posts_per_user=None,
    tfidf_kwargs=None,
    n_boot=500,
    seed=42,
    plot=True,
    n_jobs=1,
    english_only=False,
):
    """
    Census overlap-vs-cosine pipeline with user-clustered bootstrap CIs.

    load packs -> full membership -> census_comember_pairs -> cosine_for_pairs
    -> cluster_bootstrap_ci -> (optional) plot_info_vs_overlap.

    Returns (agg, pair_df). `pair_df` is the scored pairs (with the `cosine`
    column); `agg` is the per-bin table with bootstrap CIs.
    """
    packs, _ = load_starterpacks(starterpacks_path)
    membership = build_user_pack_membership(packs, users=None)

    pairs = census_comember_pairs(
        membership,
        candidate_users=candidate_users,
        per_level=per_level,
        baseline_pairs=baseline_pairs,
        max_shared=max_shared,
        seed=seed,
    )
    print(
        f"census: {len(pairs):,} candidate pairs across "
        f"shared-pack levels {sorted(pairs['shared_packs'].unique())}"
    )

    pair_df = cosine_for_pairs(
        pairs,
        records_dir=records_dir,
        min_tokens=min_tokens,
        max_posts_per_user=max_posts_per_user,
        tfidf_kwargs=tfidf_kwargs,
        n_jobs=n_jobs,
        english_only=english_only,
    )
    print(f"scored {len(pair_df):,} pairs with cosine")

    agg = cluster_bootstrap_ci(
        pair_df,
        value_col="cosine",
        group_col="shared_packs",
        n_boot=n_boot,
        seed=seed,
        max_shared=max_shared,
    )

    if plot:
        plot_info_vs_overlap(
            agg, info_label="mean cosine similarity", max_shared=max_shared
        )
        plt.tight_layout()
        plt.show()
    return agg, pair_df
