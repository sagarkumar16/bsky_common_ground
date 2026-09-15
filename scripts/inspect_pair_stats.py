"""
inspect_pair_stats.py
=====================
Diagnostic: dump every redundancy statistic for 5 real pairs, under BOTH
similarity signals, so we can eyeball whether the signal is meaningful and how
the estimators compare.

For each selected pair it prints the two user dids, the shared pack count P, each
shared pack's id/name/description (so signal quality is inspectable by eye), the
P x P similarity matrix under embedding-cosine AND co-membership-Jaccard, and a
table of every statistic from ``pack_semantics.all_effective_numbers``. It also
writes a tidy long CSV (``pair_stats.csv``) and ``shared_packs.csv``.

Pairs are chosen (seeded) to have >= 2 shared packs, span a range of shared
counts, and deliberately include one HIGH-redundancy pair (low mean distinct)
and one LOW-redundancy pair, ranked by embedding mean off-diagonal similarity,
so the estimators visibly diverge.

    python scripts/inspect_pair_stats.py \
        --starterpacks-path starterpacks.jsonl \
        --embeddings-path pack_emb_all-MiniLM-L6-v2.npz --out out_dir
"""

import sys
from collections import defaultdict
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spcg.pack_semantics as ps
from spcg.overlap_info import build_user_pack_membership, load_starterpacks


def _mean_offdiag(S):
    P = S.shape[0]
    if P <= 1:
        return 1.0
    off = S.copy().astype(float)
    np.fill_diagonal(off, np.nan)
    return float(np.nanmean(off))


def _candidate_pairs(pairs_path, membership, Z, seed, pool=400):
    """Build a pool of candidate pairs with >= 2 shared packs, each annotated
    with its shared pack ids, P, and embedding mean off-diagonal similarity."""
    rng = np.random.default_rng(seed)

    def shared_of(a, b):
        return sorted(membership.get(a, frozenset()) & membership.get(b, frozenset()))

    raw = []
    if pairs_path:
        df = (
            pd.read_parquet(pairs_path)
            if Path(pairs_path).suffix in (".parquet", ".pq")
            else pd.read_csv(pairs_path)
        )
        idx = np.arange(len(df))
        rng.shuffle(idx)
        for i in idx:
            a, b = str(df.iloc[i]["user_a"]), str(df.iloc[i]["user_b"])
            raw.append((a, b))
            if len(raw) >= pool * 4:
                break
    else:
        cand = sorted(d for d, p in membership.items() if len(p) >= 2)
        if len(cand) < 2:
            raise click.ClickException("need >= 2 users in V_{>=2} to derive pairs")
        n = len(cand)
        attempts = 0
        seen = set()
        while len(raw) < pool * 4 and attempts < pool * 400:
            attempts += 1
            i, j = int(rng.integers(n)), int(rng.integers(n))
            if i == j:
                continue
            a, b = (cand[i], cand[j]) if cand[i] < cand[j] else (cand[j], cand[i])
            if (a, b) in seen:
                continue
            seen.add((a, b))
            raw.append((a, b))

    out = []
    for a, b in raw:
        shared = shared_of(a, b)
        if len(shared) < 2:
            continue
        S_emb = ps.sim_from_embeddings(shared, Z)
        out.append(
            {
                "a": a,
                "b": b,
                "shared": shared,
                "P": len(shared),
                "sbar": _mean_offdiag(S_emb),
            }
        )
        if len(out) >= pool:
            break
    if len(out) < 2:
        raise click.ClickException("could not find >= 2 pairs with >= 2 shared packs")
    out.sort(key=lambda c: (c["a"], c["b"]))  # deterministic order
    return out


def _select_five(cands, seed):
    """Pick 5 pairs: a HIGH- and a LOW-redundancy pair (max/min mean off-diag
    similarity, drawn from the shared-count level with the widest spread so P is
    controlled), then fill to 5 spanning distinct shared counts. Seeded."""
    rng = np.random.default_rng(seed + 1)
    by_p = defaultdict(list)
    for c in cands:
        by_p[c["P"]].append(c)

    best_P, best_spread = None, -1.0
    for P, cs in by_p.items():
        if len(cs) >= 2:
            spread = max(c["sbar"] for c in cs) - min(c["sbar"] for c in cs)
            if spread > best_spread:
                best_spread, best_P = spread, P
    if best_P is not None:
        cs = by_p[best_P]
        hi = max(cs, key=lambda c: (c["sbar"], c["a"], c["b"]))
        lo = min(cs, key=lambda c: (c["sbar"], c["a"], c["b"]))
    else:  # no shared-count level has >= 2 candidates: use global extremes
        hi = max(cands, key=lambda c: (c["sbar"], c["a"], c["b"]))
        lo = min(cands, key=lambda c: (c["sbar"], c["a"], c["b"]))

    selected = [dict(hi, label="high-redundancy"), dict(lo, label="low-redundancy")]
    chosen_keys = {(c["a"], c["b"]) for c in selected}

    # fill to 5, preferring shared counts not yet represented
    remaining = [c for c in cands if (c["a"], c["b"]) not in chosen_keys]
    rem_by_p = defaultdict(list)
    for c in remaining:
        rem_by_p[c["P"]].append(c)
    seen_p = {c["P"] for c in selected}
    order = sorted(rem_by_p.keys(), key=lambda P: (P in seen_p, P))  # new P first
    for P in order:
        if len(selected) >= 5:
            break
        bucket = rem_by_p[P]
        pick = bucket[int(rng.integers(len(bucket)))]
        selected.append(dict(pick, label=f"span(P={P})"))
        seen_p.add(P)
    # if still short (few distinct P), top up from remaining
    leftover = [
        c
        for c in remaining
        if (c["a"], c["b"]) not in {(s["a"], s["b"]) for s in selected}
    ]
    while len(selected) < 5 and leftover:
        pick = leftover.pop(int(rng.integers(len(leftover))))
        selected.append(dict(pick, label=f"span(P={pick['P']})"))
    return selected[:5]


def _fmt_matrix(S):
    return "\n".join(
        "      [" + "  ".join(f"{v:4.2f}" for v in row) + "]" for row in np.round(S, 2)
    )


def _parse_key(key):
    """('count') -> ('count','all'); ('effrank__jac(nonPSD)') -> ('effrank','jac(nonPSD)')."""
    if "__" not in key:
        return key, "all"
    stat, signal = key.split("__", 1)
    return stat, signal


@click.command()
@click.option(
    "--starterpacks-path", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option(
    "--pairs-path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Optional table with user_a,user_b[,shared_packs]; else pairs "
    "are derived from membership.",
)
@click.option(
    "--embeddings-path", default=None, help="npz embedding cache (reused if complete)."
)
@click.option("--model-name", default="all-MiniLM-L6-v2", show_default=True)
@click.option("--n-pairs", default=5, show_default=True, type=int)
@click.option("--seed", default=42, show_default=True, type=int)
@click.option("--out", "out_dir", default="output/pair_stats", show_default=True)
def main(
    starterpacks_path, pairs_path, embeddings_path, model_name, n_pairs, seed, out_dir
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading starterpacks (names, members, membership) ...")
    packs, names = load_starterpacks(starterpacks_path)
    membership = build_user_pack_membership(packs, users=None)
    pack_members = ps.load_pack_members(starterpacks_path)
    descriptions = ps.load_pack_descriptions(starterpacks_path)

    print(f"embedding {len(packs):,} packs (cache reused if complete) ...")
    E, _pack_ids, row_of = ps.embed_packs(
        descriptions, model_name=model_name, cache_path=embeddings_path
    )
    d = E.shape[1] if E.shape[0] else 0
    Z = np.zeros((len(packs), d))
    for pid, r in row_of.items():
        Z[pid] = E[r]

    cands = _candidate_pairs(pairs_path, membership, Z, seed)
    selected = _select_five(cands, seed)[:n_pairs]
    print(
        f"selected {len(selected)} pairs "
        f"(labels: {[c['label'] for c in selected]})\n"
    )

    long_rows, sp_rows = [], []
    for pid_i, c in enumerate(selected):
        a, b, shared, P = c["a"], c["b"], c["shared"], c["P"]
        S_emb = ps.sim_from_embeddings(shared, Z)
        S_jac = ps.sim_from_comembership(shared, pack_members)
        stats = ps.all_effective_numbers(shared, (E, row_of), pack_members)

        print("=" * 74)
        print(
            f"PAIR {pid_i}  [{c['label']}]   P = {P} shared packs "
            f"(emb mean off-diag sim = {c['sbar']:.3f})"
        )
        print(f"  user_a = {a}")
        print(f"  user_b = {b}")
        print("  shared packs:")
        for pk in shared:
            nm = names[pk] if pk < len(names) else None
            desc = (descriptions.get(pk, "") or "")[:140]
            print(f"    [{pk}] {nm!r}: {desc}")
            sp_rows.append(
                {
                    "pair_id": pid_i,
                    "pack_id": pk,
                    "name": nm,
                    "description": descriptions.get(pk, ""),
                }
            )
        print("  S (embedding cosine):")
        print(_fmt_matrix(S_emb))
        print("  S (co-membership Jaccard):")
        print(_fmt_matrix(S_jac))

        print("  effective numbers (statistic x signal):")
        # tidy table
        table = defaultdict(dict)
        for key, val in stats.items():
            stat, signal = _parse_key(key)
            table[stat][signal] = val
            long_rows.append(
                {
                    "pair_id": pid_i,
                    "user_a": a,
                    "user_b": b,
                    "P": P,
                    "statistic": stat,
                    "signal": signal,
                    "value": val,
                }
            )
        signals = ["all", "emb", "jac", "jac(nonPSD)"]
        present_sigs = [s for s in signals if any(s in table[st] for st in table)]
        hdr = "    " + f"{'statistic':<18}" + "".join(f"{s:>14}" for s in present_sigs)
        print(hdr)
        for stat in [
            "count",
            "cluster_single",
            "cluster_complete",
            "cluster_average",
            "mean_interp",
            "effrank",
            "rowsum",
            "participation",
        ]:
            if stat not in table:
                continue
            cells = []
            for s in present_sigs:
                v = table[stat].get(s)
                cells.append(f"{v:14.3f}" if v is not None else " " * 14)
            print(f"    {stat:<18}" + "".join(cells))
        print()

    pd.DataFrame(long_rows).to_csv(out_dir / "pair_stats.csv", index=False)
    pd.DataFrame(sp_rows).to_csv(out_dir / "shared_packs.csv", index=False)
    print(f"wrote {out_dir/'pair_stats.csv'} and {out_dir/'shared_packs.csv'}")

    # quick sanity: redundancy ordering on the two hand-picked pairs
    hi = next((c for c in selected if c["label"] == "high-redundancy"), None)
    lo = next((c for c in selected if c["label"] == "low-redundancy"), None)
    if hi and lo:
        sh = ps.all_effective_numbers(hi["shared"], (E, row_of), pack_members)
        sl = ps.all_effective_numbers(lo["shared"], (E, row_of), pack_members)
        print(
            f"[check] mean_interp__emb  high={sh['mean_interp__emb']:.3f}  "
            f"low={sl['mean_interp__emb']:.3f}  "
            f"({'OK: high<low' if sh['mean_interp__emb'] < sl['mean_interp__emb'] else 'NOTE: not lower'})"
        )


if __name__ == "__main__":
    main()
