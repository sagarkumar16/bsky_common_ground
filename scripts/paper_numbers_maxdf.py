"""
paper_numbers_maxdf.py
======================
``max_df`` sensitivity for the TF-IDF content-similarity measure.

``max_df = 0.4`` is a free parameter that the paper leans on: it is what licenses
describing the cosine as *marked* vocabulary (terms a community supplies) rather
than lexical overlap in general. This script re-fits the vectorizer at
``max_df in {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`` -- everything else held fixed
(``min_df=10``, ``sublinear_tf=True``, same pairs, same seed, same filters) --
re-scores BOTH the observed pairs and the degree-matched zero-overlap baseline at
each setting, and reports the three quantities the paper actually rests on:

  1. sim(s) increasing in s
  2. Delta(s) = sim(s) - matched(s) > 0 and monotone
  3. ratio(s) = sim(s) / matched(s) ~ 2, near-constant in s

None of the three is a statement about absolute levels, so all three should
survive a change of vectorizer. Absolute cosines WILL move (less pruning -> denser
vectors -> higher cosines); that is expected and is not a contradiction.

Plus a second, independent question: WHAT does the max_df cut actually remove? The
highest-df discarded terms and the highest-df surviving terms are dumped, with
their document frequencies, so a human can decide whether the discarded set is
"the platform-wide register" or just ordinary English the stopword list should
already have caught.

WHICH CORPUS THE max_df CUT IS TAKEN OVER (this is not the scored sample)
-------------------------------------------------------------------------
sklearn's float ``max_df`` drops a term iff ``df > max_df * n_docs``, where BOTH
``df`` and ``n_docs`` are counted over the documents the vectorizer was FITTED on.
The pipeline hands ``cosine_for_pairs`` the whole co-membership census, so it fits
on every census user with usable text (~220k docs) and only afterwards scores the
subset of census pairs whose two endpoints both have text (~212k pairs, spanning
~136k users). Fitting on those ~136k scored users instead moves the ceiling from
0.4*220796 = 88318 documents to 0.4*136387 = 54555 -- a materially harsher cut, and
NOT the parameter the paper reports (see ``paper_numbers_maxdf_diag.py``, which
established this and confirmed that fitting on the census corpus reproduces the
cached cosine column to machine precision).

So this script rebuilds the census with the pipeline's caps, loads text for ALL
census users, and takes every max_df in the sweep over that ~220k-document corpus.
The pairs it scores are still exactly the cached pair list; only the fit corpus is
larger. It checks the rebuild rather than trusting it: the census pairs it can
score must be the cached pair set, and every cached pair must have both endpoints
in the fit corpus.

Why the pair set does not move with max_df (checked, per the task)
-----------------------------------------------------------------
``spcg.overlap_info.cosine_for_pairs`` loads text via ``build_text_df_for_users``,
and the ``min_tokens`` filter is applied inside ``_load_user_posts`` -- i.e. BEFORE
any vectorizer exists. A pair is dropped iff one of its users has no usable text.
``build_user_tfidf`` runs afterwards and never drops a user (a user whose every
term is pruned keeps a row; it is simply all-zero, cosine 0). So the pair sample
is a function of the co-membership census and the token filter only, and re-fitting
at another max_df re-scores the SAME pairs. The script asserts this rather than
assuming it: it takes the pair list from the already-scored table and stops if the
set it can score differs.

Reuse (imports, never re-implements)
------------------------------------
  * ``spcg.overlap_info.load_starterpacks`` / ``build_user_pack_membership`` /
    ``census_comember_pairs`` -- rebuild the census the pipeline fitted on, with the
    pipeline's caps and seed, so the fit corpus here is the fit corpus there.
  * ``spcg.overlap_info.build_text_df_for_users``  -- the record loading + min_tokens
    + max_posts_per_user + english_only filters (run ONCE; max_df-independent).
  * ``spcg.overlap_info.make_content_tokenizer`` -- the exact tokenizer the pipeline
    vectorizes with.
  * ``spcg.overlap_info.build_user_tfidf`` -- used, at the primary max_df, to VERIFY
    the fast shared-counts path reproduces the pipeline's vectorizer bit-for-bit.
  * ``run_paper_numbers.section7`` -- the degree-matched zero-overlap baseline, so
    the matched curve here is produced by the same code as the paper's, not a
    re-derivation. It is called once per max_df against a re-scored table.
  * ``run_paper_numbers.Report`` -- the report accumulator (tables + the
    ``!! CONTRADICTS DRAFT !!`` mechanism).
  * ``run_network_analysis._wls_beta`` -- the slope beta(s in [1,8]).

The fast path (why it is exact)
-------------------------------
Re-tokenizing the whole corpus once per max_df is the dominant cost and is pure
waste: tokenization does not depend on max_df. So the corpus is counted ONCE with a
CountVectorizer carrying the SAME tokenizer, then each max_df is a column mask on
that count matrix followed by TfidfTransformer. This is identical to what
TfidfVectorizer(min_df, max_df) does internally -- it prunes by document frequency
at the count stage, and both idf (a function of the term's own df and n_docs) and
the L2 norm (computed on the pruned row) are applied after. ``--verify-refit``
proves it: at the primary max_df it also fits the real ``build_user_tfidf`` and
compares every scored cosine.

Usage
-----
    python paper_numbers_maxdf.py \
        --pairs-path  $DATA_DIR/directed/pairs_with_hops_n_eff_seed16.parquet \
        --records-dir /scratch/xee6vz/bluesky-graph/records \
        --packs-path  /scratch/xee6vz/bluesky-graph/starterpacks.jsonl \
        --out         output/paper_numbers \
        --data-dir    $DATA_DIR \
        --verify-refit

The census caps (``--per-level``, ``--baseline-pairs``, ``--max-shared``, ``--seed``)
and the text filters MUST match the run that produced the cached pair table
(run_full_pipeline.sh: PER_LEVEL=75000, BASELINE_PAIRS=150000, MAX_SHARED=8,
MIN_TOKENS=50, MAX_POSTS_PER_USER=1000, ENGLISH_ONLY=1, SEED=16). They are the
defaults here, and the rebuild is checked against the cached pair set.
"""

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_network_analysis import _wls_beta  # noqa: E402  (private; reused, not copied)
from scripts.run_paper_numbers import Report, section7  # noqa: E402
from spcg.overlap_info import build_text_df_for_users  # noqa: E402
from spcg.overlap_info import (
    build_user_pack_membership,
    build_user_tfidf,
    census_comember_pairs,
    load_starterpacks,
    make_content_tokenizer,
)
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

try:
    from tqdm import tqdm
except Exception:  # tqdm absent -> no-op passthrough

    def tqdm(it=None, **k):
        return it if it is not None else iter(())


# The five numbers the main text reports at max_df=0.4. If the re-scoring path
# does not reproduce them, the path is wrong and nothing downstream is meaningful.
DRAFT_AT_040 = {
    "sim_s1": 0.104,
    "sim_s8": 0.151,
    "matched_s1": 0.053,
    "delta_s1": 0.051,
    "ratio_s1": 1.97,
}

# Heuristic buckets for reading the discarded-term list. NOT a finding -- an aid so
# the human eye has somewhere to start. Bluesky/platform register, hand-listed.
PLATFORM_LEXICON = {
    "bsky",
    "bluesky",
    "skeet",
    "skeets",
    "skeeting",
    "starter",
    "pack",
    "packs",
    "starterpack",
    "app",
    "post",
    "posts",
    "posting",
    "posted",
    "repost",
    "reposts",
    "feed",
    "feeds",
    "thread",
    "threads",
    "follow",
    "follows",
    "followers",
    "following",
    "followed",
    "account",
    "accounts",
    "block",
    "blocked",
    "mute",
    "muted",
    "handle",
    "profile",
    "timeline",
    "alt",
    "text",
    "dm",
    "dms",
    "twitter",
    "tweet",
    "tweets",
    "elon",
    "musk",
    "mastodon",
    "site",
    "link",
    "links",
    "url",
    "gif",
    "video",
    "photo",
    "pic",
    "pics",
    "image",
    "images",
}


# ===========================================================================
# Corpus -> one count matrix (tokenized once; max_df is then a column mask)
# ===========================================================================
def count_corpus(text_df, min_df):
    """CountVectorizer over the SAME tokenizer the pipeline uses, with min_df applied
    and max_df wide open. Returns (C, terms, df_counts, uids).

    df_counts[j] = number of user-documents containing term j. Every max_df setting
    is a mask on this vector, so the corpus is tokenized exactly once.
    """
    grouped = text_df.groupby("uid")["text"].apply(lambda s: " ".join(s)).sort_index()
    uids = grouped.index.to_numpy()
    docs = grouped.tolist()

    # float64: TfidfVectorizer's default dtype. float32 here costs half the memory but
    # drifts ~1e-7 from the pipeline's cosines, which defeats the point of verify-refit.
    cv = CountVectorizer(
        tokenizer=make_content_tokenizer(),
        token_pattern=None,
        lowercase=False,
        min_df=min_df,
        max_df=1.0,
        dtype=np.float64,
    )
    C = cv.fit_transform(docs).tocsc()
    terms = np.array(cv.get_feature_names_out())
    df_counts = np.asarray((C > 0).sum(axis=0)).ravel().astype(np.int64)
    return C, terms, df_counts, uids


def tfidf_at(C, df_counts, max_df, min_df, sublinear_tf=True):
    """The pipeline's TF-IDF at one max_df, from the shared count matrix.

    Replicates sklearn's own pruning rule exactly (_limit_features keeps terms with
    min_doc_count <= df <= max_doc_count, where max_doc_count = max_df * n_docs as a
    FLOAT, not a floor), then applies the same TfidfTransformer settings
    TfidfVectorizer would (use_idf, smooth_idf, l2 norm, sublinear_tf).

    Returns (X, keep_mask). X rows are L2-normalized, so a row dot product IS cosine.
    """
    n_docs = C.shape[0]
    keep = (df_counts >= min_df) & (df_counts <= max_df * n_docs)
    X = TfidfTransformer(
        norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=sublinear_tf
    ).fit_transform(C[:, keep].tocsr())
    return X.tocsr(), keep


def score_pairs(X, ra, rb, chunk=50_000):
    """Cosine for each (row a, row b): rows are L2-normalized, so dot == cosine.
    Chunked so the elementwise product never materializes all pairs' nnz at once.
    """
    out = np.empty(len(ra), dtype=float)
    for i in range(0, len(ra), chunk):
        sl = slice(i, min(i + chunk, len(ra)))
        prod = X[ra[sl]].multiply(X[rb[sl]])
        out[sl] = np.asarray(prod.sum(axis=1)).ravel()
    return out


# ===========================================================================
# Per-max_df quantities
# ===========================================================================
def sim_by_s(shared, cosine, s_values):
    """Mean cosine at EXACT shared_packs == s (the paper's sim(s); not clipped)."""
    out = {}
    for s in s_values:
        m = shared == s
        out[s] = float(cosine[m].mean()) if m.any() else np.nan
    return out


def beta_1_8(shared, cosine):
    """Marginal slope of cosine in raw shared-pack count over the 1<=s<=8 window --
    the same design and estimator as run_paper_numbers §1 (point estimate)."""
    m = (shared >= 1) & (shared <= 8)
    X = np.column_stack([np.ones(len(shared)), shared.astype(float)])
    return float(_wls_beta(X, cosine.astype(float), m.astype(float))[1])


def matched_curve(df_scored, packs_path, seed, out_dir, tag):
    """Degree-matched zero-overlap baseline at this max_df, by CALLING the paper's own
    §7 (run_paper_numbers.section7) on the re-scored table.

    §7 matches each observed pair to a bin-0 pair drawn from the same
    (degree-decile, degree-decile) cell. Degrees, cells, pool and rng draw order all
    depend on the pair table's users/shared_packs and the seed -- never on the
    cosines -- so the SAME baseline pairs are selected at every max_df, and only
    their scores move. That is exactly the like-for-like Delta(s) the paper needs.

    The §7 Report is given its own directory so it cannot clobber the main
    output/paper_numbers/ artifacts; we then read back the curve it wrote.
    """
    sub = Path(out_dir) / "maxdf_s7" / tag
    sub.mkdir(parents=True, exist_ok=True)
    rep7 = Report(sub)  # never flushed: we only want its table
    section7(rep7, df_scored, "n_eff", packs_path, None, seed, sentinel=4)
    return pd.read_csv(sub / "s7_degree_matched_curve.csv")


# ===========================================================================
# What does the cut remove?
# ===========================================================================
def vocab_cut_terms(terms, df_counts, n_docs, max_df, min_df, top_n):
    """The terms either side of the max_df cut, with their document frequencies.

    Returns (table, counts). `table` has one row per reported term with side in
    {discarded_high_df, survived}; discarded terms are the ones pruned BY MAX_DF
    (df above the cut), not the ones pruned by min_df.
    """
    cut = max_df * n_docs
    above = df_counts > cut
    kept = (df_counts >= min_df) & (df_counts <= cut)

    def rows(mask, side, order_desc=True):
        idx = np.flatnonzero(mask)
        idx = idx[np.argsort(-df_counts[idx] if order_desc else df_counts[idx])]
        idx = idx[:top_n]
        return [
            {
                "side": side,
                "rank": r + 1,
                "term": terms[j],
                "df_count": int(df_counts[j]),
                "df_frac": float(df_counts[j] / n_docs),
                "is_hashtag": bool(str(terms[j]).startswith("#")),
                "in_sklearn_stopwords": bool(terms[j] in ENGLISH_STOP_WORDS),
                "in_platform_lexicon": bool(
                    str(terms[j]).lstrip("#") in PLATFORM_LEXICON
                ),
            }
            for r, j in enumerate(idx)
        ]

    tbl = pd.DataFrame(rows(above, "discarded_high_df") + rows(kept, "survived"))
    counts = {
        "n_terms_min_df": int((df_counts >= min_df).sum()),
        "n_discarded_by_max_df": int(above.sum()),
        "n_survived": int(kept.sum()),
        "cut_df_count": float(cut),
        "n_docs": int(n_docs),
    }
    return tbl, counts


def read_discarded(tbl):
    """A first-pass, mechanical read of the discarded set -- offered as a starting
    point, NOT as the judgment. The judgment is the human's; the list is above."""
    d = tbl[tbl["side"] == "discarded_high_df"]
    if d.empty:
        return ["  (nothing discarded by max_df at this setting)"]
    n = len(d)
    n_stop = int(d["in_sklearn_stopwords"].sum())
    n_plat = int(d["in_platform_lexicon"].sum())
    n_tag = int(d["is_hashtag"].sum())
    lines = [
        "  Of the %d highest-df DISCARDED terms shown:" % n,
        "    %d (%.0f%%) are in sklearn's ENGLISH_STOP_WORDS. The tokenizer already"
        % (n_stop, 100 * n_stop / n),
        "        drops those, so any nonzero count here means the stopword list is",
        "        leaking -- not that max_df is doing work.",
        "    %d (%.0f%%) are in the hand-listed platform/Bluesky lexicon (bsky, skeet,"
        % (n_plat, 100 * n_plat / n),
        "        post, follow, ...): the 'platform register' the draft claims to excise.",
        "    %d (%.0f%%) are hashtags." % (n_tag, 100 * n_tag / n),
        "    The remainder are generic/ordinary content words.",
        "",
        "  READ IT YOURSELF: if the discarded set is mostly ordinary English content",
        "  words rather than platform register, the 'marked vocabulary' framing in",
        '  "Estimating Shared Repertoire" is not supported by what the cut removes,',
        "  and that section must be rewritten. The counts above are mechanical; the",
        "  substantive call is a human one.",
    ]
    return lines


# ===========================================================================
# CLI
# ===========================================================================
@click.command()
@click.option(
    "--pairs-path",
    required=True,
    help="Scored pair table from the completed run (user_a,user_b,"
    "shared_packs[,cosine]). Its pair list IS the sample; cosine, if "
    "present, is used as the max_df=0.4 reproduction check.",
)
@click.option(
    "--records-dir",
    required=True,
    help="Per-user record gzips (the text is re-loaded once; it does not "
    "depend on max_df).",
)
@click.option(
    "--packs-path",
    required=True,
    help="starterpacks.jsonl -- rebuilds the census that defines the FIT "
    "corpus, and supplies pack-degree for the §7 degree-matched null.",
)
# --- census caps: MUST match run_full_pipeline.sh for the run that made the cache ---
@click.option(
    "--per-level",
    default=75000,
    show_default=True,
    type=int,
    help="Census cap per shared_packs level (pipeline value).",
)
@click.option(
    "--baseline-pairs",
    default=150000,
    show_default=True,
    type=int,
    help="Census bin-0 pair cap (pipeline value).",
)
@click.option(
    "--max-shared",
    default=8,
    show_default=True,
    type=int,
    help="Census top shared_packs level (pipeline value).",
)
@click.option(
    "--out",
    "out_dir",
    default="output/paper_numbers",
    show_default=True,
    help="Small tables + the human-readable report (repo-tracked).",
)
@click.option(
    "--data-dir",
    default=None,
    help="Large artifacts (the re-scored per-pair cosines at every max_df). "
    "Defaults to --out; on the cluster point this at scratch.",
)
@click.option(
    "--max-df",
    "max_dfs",
    default="0.2,0.3,0.4,0.5,0.6,0.7,0.8",
    show_default=True,
    help="Comma-separated max_df settings to sweep.",
)
@click.option(
    "--primary-max-df",
    default=0.4,
    show_default=True,
    type=float,
    help="The pipeline's setting: the one that must reproduce the draft and "
    "the one the vocabulary cut is inspected at.",
)
@click.option("--min-df", default=10, show_default=True, type=int)
@click.option("--sublinear-tf/--no-sublinear-tf", default=True, show_default=True)
@click.option(
    "--min-tokens",
    default=50,
    show_default=True,
    type=int,
    help="MUST match the pipeline (50): it is what determines the pair set.",
)
@click.option("--max-posts-per-user", default=1000, show_default=True, type=int)
@click.option("--english-only/--no-english-only", default=True, show_default=True)
@click.option("--seed", default=16, show_default=True, type=int)
@click.option(
    "--n-jobs",
    default=0,
    show_default=True,
    type=int,
    help="Worker processes for record loading (0 = all / $SLURM_CPUS_PER_TASK).",
)
@click.option("--top-terms", default=50, show_default=True, type=int)
@click.option(
    "--verify-refit/--no-verify-refit",
    default=False,
    show_default=True,
    help="At the primary max_df, ALSO fit spcg.overlap_info.build_user_tfidf "
    "and assert the shared-counts path reproduces it exactly. One extra "
    "tokenization pass; worth it once.",
)
@click.option(
    "--tol",
    default=0.002,
    show_default=True,
    type=float,
    help="Absolute tolerance for reproducing the draft's numbers at "
    "max_df=0.4 (they are quoted to 3 d.p.).",
)
@click.option(
    "--strict/--no-strict",
    default=True,
    show_default=True,
    help="Stop if max_df=0.4 does not reproduce the main text. The task is "
    "explicit: that would be a broken re-scoring path, not a finding.",
)
def main(
    pairs_path,
    records_dir,
    packs_path,
    per_level,
    baseline_pairs,
    max_shared,
    out_dir,
    data_dir,
    max_dfs,
    primary_max_df,
    min_df,
    sublinear_tf,
    min_tokens,
    max_posts_per_user,
    english_only,
    seed,
    n_jobs,
    top_terms,
    verify_refit,
    tol,
    strict,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(data_dir) if data_dir else out
    data.mkdir(parents=True, exist_ok=True)
    grid = [float(v) for v in str(max_dfs).split(",") if str(v).strip()]

    rep = Report(out)
    rep.p("MAX_DF SENSITIVITY  --  re-fit the vectorizer, re-score the same pairs")
    rep.p("pairs: %s" % pairs_path)
    rep.p(
        "grid: max_df in %s   (min_df=%d, sublinear_tf=%s, min_tokens=%d, "
        "max_posts_per_user=%s, english_only=%s, seed=%d)"
        % (
            grid,
            min_df,
            sublinear_tf,
            min_tokens,
            max_posts_per_user,
            english_only,
            seed,
        )
    )
    rep.p(
        "fit corpus: the FULL census (per_level=%d, baseline_pairs=%d, max_shared=%d),"
        % (per_level, baseline_pairs, max_shared)
    )
    rep.p(
        "            i.e. every census user with usable text -- NOT just the users in"
    )
    rep.p("            the scored pairs. max_df is a fraction OF THAT corpus, which is")
    rep.p("            what the pipeline's vectorizer saw.")

    # ---- the pair sample: taken from the completed run, never re-drawn ------
    pairs = (
        pd.read_parquet(pairs_path)
        if str(pairs_path).endswith(".parquet")
        else pd.read_csv(pairs_path)
    )
    for c in ("user_a", "user_b", "shared_packs"):
        if c not in pairs.columns:
            raise SystemExit(
                "pairs table missing '%s' (have %s)" % (c, list(pairs.columns))
            )
    cached = pairs["cosine"].to_numpy(float) if "cosine" in pairs else None
    print(
        "pairs: %d rows, shared_packs 0..%d"
        % (len(pairs), int(pairs["shared_packs"].max()))
    )

    # ---- the FIT corpus: the whole census, loaded ONCE ----------------------
    # max_df is a fraction of the documents the vectorizer is fitted on, so the fit
    # corpus has to be the pipeline's (every census user with usable text), not the
    # smaller set of users who happen to appear in a scoreable pair.
    rep.h("0. THE FIT CORPUS AND THE PAIR SET  (checked, not assumed)")
    rep.p(
        "  The pipeline hands cosine_for_pairs the WHOLE census, so its vectorizer is"
    )
    rep.p(
        "  fitted on every census user with usable text and only then scores the pairs"
    )
    rep.p(
        "  whose two endpoints both have text. max_df is a fraction of the former, not"
    )
    rep.p(
        "  the latter. Rebuilding that census here with the pipeline's caps and seed."
    )
    rep.p("")
    rep.p("  cosine_for_pairs applies min_tokens INSIDE build_text_df_for_users, i.e.")
    rep.p("  before the vectorizer exists; build_user_tfidf never drops a user. So the")
    rep.p(
        "  pair sample is a function of the census + token filter only. Confirming by"
    )
    rep.p(
        "  re-loading text with the pipeline's filters and re-deriving which pairs are"
    )
    rep.p("  scoreable -- if this set differs from the cached table, the comparison is")
    rep.p("  NOT like-for-like and everything below is caveated.")

    print(
        "rebuilding the census (per_level=%d, baseline_pairs=%d, max_shared=%d, "
        "seed=%d) ..." % (per_level, baseline_pairs, max_shared, seed)
    )
    packs, _ = load_starterpacks(packs_path)
    membership = build_user_pack_membership(packs, users=None)
    census = census_comember_pairs(
        membership,
        candidate_users=None,
        per_level=per_level,
        baseline_pairs=baseline_pairs,
        max_shared=max_shared,
        seed=seed,
    )
    census_users = pd.unique(pd.concat([census["user_a"], census["user_b"]]))
    scored_users = pd.unique(pd.concat([pairs["user_a"], pairs["user_b"]]))
    print(
        "census: %d pairs over %d distinct users (the set handed to "
        "build_text_df_for_users)" % (len(census), len(census_users))
    )

    print(
        "loading text for ALL %d census users (min_tokens=%d, english_only=%s) ..."
        % (len(census_users), min_tokens, english_only)
    )
    text_df, uid_to_did = build_text_df_for_users(
        list(census_users),
        records_dir=records_dir,
        min_tokens=min_tokens,
        max_posts_per_user=max_posts_per_user,
        n_jobs=n_jobs,
        english_only=english_only,
    )
    print("text: %d users with usable text, %d posts" % (len(uid_to_did), len(text_df)))

    C, terms, df_counts, uids = count_corpus(text_df, min_df)
    n_docs = C.shape[0]
    print(
        "fit corpus: %d docs, %d terms at min_df=%d -- max_df cuts are fractions OF "
        "THIS" % (n_docs, len(terms), min_df)
    )

    did_to_row = {uid_to_did[u]: i for i, u in enumerate(uids)}

    # did the rebuild land on the census that produced the cache?
    text_users = set(did_to_row)
    can_score = census[
        census["user_a"].isin(text_users) & census["user_b"].isin(text_users)
    ]
    same_pairs = set(map(tuple, can_score[["user_a", "user_b"]].to_numpy())) == set(
        map(tuple, pairs[["user_a", "user_b"]].to_numpy())
    )

    ra = pairs["user_a"].map(did_to_row).to_numpy()
    rb = pairs["user_b"].map(did_to_row).to_numpy()
    keep_pair = ~(pd.isna(ra) | pd.isna(rb))
    n_drop = int((~keep_pair).sum())

    rep.p("")
    rep.p("  census pairs handed to cosine_for_pairs : %d" % len(census))
    rep.p("  distinct users in them                  : %d" % len(census_users))
    rep.p(
        "  FIT CORPUS: of those, with usable text  : %d   <-- max_df denominator"
        % n_docs
    )
    rep.p("  users appearing in the cached pairs     : %d" % len(scored_users))
    rep.p(
        "  fit corpus - scored users               : %d  (text, but no scoreable pair)"
        % (n_docs - len(scored_users))
    )
    rep.p(
        "  the max_df=%g ceiling is therefore %.1f docs, not %.1f"
        % (primary_max_df, primary_max_df * n_docs, primary_max_df * len(scored_users))
    )
    rep.p("")
    rep.p("  pairs in the cached table               : %d" % len(pairs))
    rep.p("  census pairs scoreable from this corpus : %d" % len(can_score))
    rep.p("  the two pair SETS are identical         : %s" % same_pairs)
    rep.p("  pairs scoreable at EVERY max_df         : %d" % int(keep_pair.sum()))
    rep.p("  pairs dropped (a user had no text)      : %d" % n_drop)
    if not same_pairs:
        rep.p(
            "  !! the rebuilt census does not land on the cached pair set. Either the"
        )
        rep.p("     caps/seed differ from the run that made the cache, or the records")
        rep.p(
            "     changed. The sweep below is still internally like-for-like, but the"
        )
        rep.p(
            "     fit corpus is only APPROXIMATELY the one the draft was computed on."
        )
        rep.contradict(
            "§0",
            "the rebuilt census yields %d scoreable pairs, not the "
            "cached %d; the fit corpus the max_df cut is taken over is "
            "only approximately the pipeline's." % (len(can_score), len(pairs)),
        )
    if n_drop:
        rep.p(
            "  !! the re-loaded text does not cover every cached pair. The pair set is"
        )
        rep.p(
            "     still IDENTICAL ACROSS the five max_df settings (nothing here depends"
        )
        rep.p(
            "     on the vectorizer), so the sensitivity comparison stays like-for-like,"
        )
        rep.p("     but it no longer matches the pairs the draft was computed on.")
        rep.contradict(
            "§0",
            "re-loaded text covers only %d of the %d cached pairs; the "
            "max_df sweep is internally like-for-like but is not the "
            "identical sample the draft used." % (int(keep_pair.sum()), len(pairs)),
        )
    if same_pairs and not n_drop:
        rep.p("  -> CONFIRMED: the fit corpus is the pipeline's, and the pair set is")
        rep.p("     exactly the cached sample and invariant to max_df.")

    pairs = pairs.loc[keep_pair].reset_index(drop=True)
    ra = ra[keep_pair].astype(int)
    rb = rb[keep_pair].astype(int)
    if cached is not None:
        cached = cached[
            keep_pair.to_numpy() if hasattr(keep_pair, "to_numpy") else keep_pair
        ]
    shared = pairs["shared_packs"].to_numpy(int)

    # ---- sweep -------------------------------------------------------------
    rep.h(
        "1. max_df SWEEP  (same pairs, same seed; vectorizer re-fit on the full "
        "%d-doc census corpus at each setting)" % n_docs
    )
    rows, delta_rows, scored_long = [], [], []
    s_report = [0, 1, 4, 8]

    for v in tqdm(grid, desc="max_df"):
        tag = "maxdf_%s" % ("%g" % v).replace(".", "p")
        X, keep_terms = tfidf_at(C, df_counts, v, min_df, sublinear_tf)
        vocab = int(keep_terms.sum())
        cos = score_pairs(X, ra, rb)
        n_zero_row = int((np.asarray(X.sum(axis=1)).ravel() == 0).sum())
        print(
            "  max_df=%.2f: cut at %.0f/%d docs, vocab=%d, mean cosine=%.5f "
            "(%d all-zero vectors in the fit corpus)"
            % (v, v * n_docs, n_docs, vocab, cos.mean(), n_zero_row)
        )

        # verify the fast path against the pipeline's own vectorizer, once
        if verify_refit and abs(v - primary_max_df) < 1e-12:
            Xv, uids_v, _ = build_user_tfidf(
                text_df, min_df=min_df, max_df=v, sublinear_tf=sublinear_tf
            )
            if not np.array_equal(uids_v, uids):
                raise SystemExit("verify-refit: row order differs between paths")
            dv = float(np.abs(score_pairs(Xv.tocsr(), ra, rb) - cos).max())
            rep.p(
                "  [verify-refit] max |cosine(shared-counts) - cosine(build_user_tfidf)|"
                " at max_df=%g = %.3e  (vocab %d vs %d)" % (v, dv, vocab, Xv.shape[1])
            )
            if dv > 1e-9 or vocab != Xv.shape[1]:
                raise SystemExit(
                    "verify-refit FAILED: the shared-counts path does not "
                    "reproduce spcg.overlap_info.build_user_tfidf (max diff "
                    "%.3e). Do not trust anything below." % dv
                )

        df_scored = pairs.copy()
        df_scored["cosine"] = cos
        if "n_eff" not in df_scored.columns:  # §7 takes cg_col but never reads it
            df_scored["n_eff"] = np.nan
        mt = matched_curve(df_scored, packs_path, seed, out, tag)

        sim = sim_by_s(shared, cos, s_report)
        m_lookup = mt.set_index("shared_packs")
        matched = {
            s: float(m_lookup["matched_zero_mean_cosine"].get(s, np.nan))
            for s in s_report
        }

        for s in range(1, 9):
            o = float(m_lookup["obs_mean_cosine"].get(s, np.nan))
            m_ = float(m_lookup["matched_zero_mean_cosine"].get(s, np.nan))
            delta_rows.append(
                {
                    "max_df": v,
                    "shared_packs": s,
                    "n_obs": int(m_lookup["n_obs"].get(s, 0)),
                    "sim": o,
                    "matched": m_,
                    "delta": o - m_,
                    "ratio": (o / m_) if m_ else np.nan,
                }
            )

        rows.append(
            {
                "max_df": v,
                "n_fit_docs": n_docs,
                "cut_df_count": v * n_docs,
                "vocab_size": vocab,
                "sim_s0": sim[0],
                "sim_s1": sim[1],
                "sim_s4": sim[4],
                "sim_s8": sim[8],
                "matched_s1": matched[1],
                "matched_s8": matched[8],
                "delta_s1": sim[1] - matched[1],
                "delta_s8": sim[8] - matched[8],
                "ratio_s1": sim[1] / matched[1] if matched[1] else np.nan,
                "ratio_s8": sim[8] / matched[8] if matched[8] else np.nan,
                "beta_1_8": beta_1_8(shared, cos),
                "n_pairs": len(cos),
                "n_zero_vectors_in_fit_corpus": n_zero_row,
            }
        )
        scored_long.append(
            pd.DataFrame(
                {
                    "max_df": v,
                    "user_a": pairs["user_a"].to_numpy(),
                    "user_b": pairs["user_b"].to_numpy(),
                    "shared_packs": shared,
                    "cosine": cos,
                }
            )
        )

    sens = pd.DataFrame(rows)
    deltas = pd.DataFrame(delta_rows)
    rep.table(sens, "maxdf_sensitivity", floatfmt="%.5f")
    rep.p("")
    rep.p("  Delta(s) and ratio(s) at every s in [1,8], per max_df:")
    rep.table(deltas, "maxdf_delta_by_s", floatfmt="%.5f")

    # the large artifact goes to --data-dir, never the repo
    long = pd.concat(scored_long, ignore_index=True)
    big = data / ("maxdf_pairs_scored_seed%d.parquet" % seed)
    try:
        long.to_parquet(big, index=False)
    except Exception:
        big = big.with_suffix(".csv")
        long.to_csv(big, index=False)
    print("wrote re-scored pairs (%d rows) -> %s" % (len(long), big))
    rep.p("  per-pair re-scored cosines at every max_df -> %s" % big)

    # ---- does max_df=0.4 reproduce the main text? --------------------------
    rep.h("2. DOES max_df=0.4 REPRODUCE THE MAIN TEXT?  (if not: stop)")
    prim = sens[np.isclose(sens["max_df"], primary_max_df)]
    if prim.empty:
        raise SystemExit("primary max_df=%g is not in the sweep grid" % primary_max_df)
    p = prim.iloc[0]
    bad = []
    for k, claim in DRAFT_AT_040.items():
        got = float(p[k])
        t = 0.05 if k.startswith("ratio") else tol
        ok = abs(got - claim) <= t
        rep.p(
            "  %-11s draft %.3f   recomputed %.5f   [%s]"
            % (k, claim, got, "ok" if ok else "MISMATCH")
        )
        if not ok:
            bad.append("%s: %.5f vs draft %.3f" % (k, got, claim))
    if cached is not None:
        Xp, _ = tfidf_at(C, df_counts, primary_max_df, min_df, sublinear_tf)
        d = np.abs(score_pairs(Xp, ra, rb) - cached)
        rep.p(
            "  per-pair vs the CACHED cosine column: max|diff|=%.3e  mean|diff|=%.3e"
            % (float(d.max()), float(d.mean()))
        )
        rep.p("  (fitting on the full census corpus, this should be ~0: it is the same")
        rep.p(
            "   vectorizer on the same documents. A large diff means the census rebuild"
        )
        rep.p("   or the text filters do not match the run that produced the cache.)")
    if bad:
        msg = (
            "max_df=0.4 does NOT reproduce the main text: %s\n"
            "Per the task, this means the RE-SCORING PATH is wrong, not that the\n"
            "parameter is sensitive. Stopping." % "; ".join(bad)
        )
        rep.p("")
        rep.p("  !! " + msg.replace("\n", "\n  "))
        rep.contradict(
            "§reproduce",
            "max_df=0.4 does not reproduce the main text (%s) "
            "-- suspect the re-scoring path, not the parameter." % "; ".join(bad),
        )
        if strict:
            _emit(rep, out)  # persist what we have, then stop as instructed
            raise SystemExit(msg)
    else:
        rep.p(
            "  -> all five reproduce. The re-scoring path is sound; the sweep below is"
        )
        rep.p("     about max_df, not about a broken pipeline.")

    # ---- the three claims that carry the paper -----------------------------
    rep.h("3. DO THE THREE LOAD-BEARING CLAIMS SURVIVE?")
    rep.p("  None of the three is a claim about absolute levels; all three should hold")
    rep.p("  at every max_df. Absolute cosine moving with max_df is expected and fine.")

    # 3a. monotonicity of sim(s) in s
    rep.p("")
    rep.p("  (1) MONOTONICITY: sim(s) increasing in s, for s = 0,1,4,8")
    for _, r in sens.iterrows():
        seq = [r["sim_s0"], r["sim_s1"], r["sim_s4"], r["sim_s8"]]
        mono = bool(np.all(np.diff(seq) > 0))
        rep.p(
            "    max_df=%.2f  %s   monotone=%s"
            % (r["max_df"], np.array2string(np.array(seq), precision=4), mono)
        )
        if not mono:
            rep.contradict(
                "max_df=%.2f" % r["max_df"],
                "sim(s) is NOT increasing in s (%s). The core "
                "shared-packs -> similarity result does not survive this "
                "vectorizer setting." % np.array2string(np.array(seq), precision=4),
            )

    # 3b. Delta(s) > 0 and monotone over s in [1,8]
    rep.p("")
    rep.p("  (2) Delta(s) = sim(s) - matched(s) > 0 at every s>=1, and monotone in s")
    for v, g in deltas.groupby("max_df"):
        g = g.sort_values("shared_packs")
        d = g["delta"].to_numpy(float)
        pos = bool(np.all(d > 0))
        mono = bool(np.all(np.diff(d) > 0))
        rep.p(
            "    max_df=%.2f  Delta(1..8)=%s  all>0=%s  monotone=%s"
            % (v, np.array2string(d, precision=4), pos, mono)
        )
        if not pos:
            rep.contradict(
                "max_df=%.2f" % v,
                "Delta(s) is NOT positive at every s>=1 (%s): the observed "
                "curve does not sit above the degree-matched baseline."
                % np.array2string(d, precision=4),
            )
        if not mono:
            rep.p(
                "       (Delta not strictly monotone; note the draft claims monotone "
                "growth in s.)"
            )

    # 3c. constancy of the ratio -- the most surprising, most fragile claim
    rep.p("")
    rep.p("  (3) ratio(s) = sim(s)/matched(s) ~ 2, near-constant in s  [the headline]")
    ratio_rows = []
    for v, g in deltas.groupby("max_df"):
        g = g.sort_values("shared_packs")
        r = g["ratio"].to_numpy(float)
        mean = float(np.nanmean(r))
        spread = float((np.nanmax(r) - np.nanmin(r)) / mean) if mean else np.nan
        ratio_rows.append(
            {
                "max_df": v,
                "ratio_mean_1_8": mean,
                "ratio_min": float(np.nanmin(r)),
                "ratio_max": float(np.nanmax(r)),
                "rel_spread_over_s": spread,
            }
        )
        rep.p(
            "    max_df=%.2f  ratio(1..8)=%s  mean=%.3f  rel.spread over s=%.1f%%"
            % (v, np.array2string(r, precision=3), mean, 100 * spread)
        )
        if spread > 0.15:
            rep.contradict(
                "max_df=%.2f" % v,
                "ratio(s) is NOT near-constant in s (%.0f%% spread, %s). The "
                "paper's most surprising claim -- a constant ~2x observed-to-"
                "matched ratio -- is a max_df artifact at this setting."
                % (100 * spread, np.array2string(r, precision=3)),
            )
    rt = pd.DataFrame(ratio_rows)
    rep.table(rt, "maxdf_ratio_constancy", floatfmt="%.4f")

    # ratio stability ACROSS max_df -- the check the reviewer will actually run
    base = rt[np.isclose(rt["max_df"], primary_max_df)]["ratio_mean_1_8"].iloc[0]
    swing = float((rt["ratio_mean_1_8"].max() - rt["ratio_mean_1_8"].min()) / base)
    rep.p("")
    rep.p(
        "  ratio mean over s, ACROSS max_df: %s  (swing = %.1f%% of the 0.4 value)"
        % (np.array2string(rt["ratio_mean_1_8"].to_numpy(), precision=3), 100 * swing)
    )
    if swing > 0.10:
        rep.contradict(
            "across max_df",
            "the observed-to-matched ratio moves %.0f%% across max_df in "
            "[0.2,0.8] (%s). It is not a parameter-free ~2x; the claim as "
            "written in the main text is unsupported."
            % (
                100 * swing,
                np.array2string(rt["ratio_mean_1_8"].to_numpy(), precision=3),
            ),
        )
    else:
        rep.p("  -> the ratio is stable across max_df. As predicted: it is a ratio of")
        rep.p("     identically-scored quantities, so the level shift cancels.")

    # ---- sanity checks -----------------------------------------------------
    rep.h("4. SANITY CHECKS")
    # non-decreasing, not strictly increasing: a higher max_df can add no new terms if
    # the corpus happens to have none in the widened df band. A DECREASE would mean the
    # pruning rule is not what we think it is.
    vs = sens.sort_values("max_df")["vocab_size"].to_numpy()
    vmono = bool(np.all(np.diff(vs) >= 0))
    rep.p(
        "  vocab_size by max_df: %s  non-decreasing=%s (flat steps = no terms in the"
        % (np.array2string(vs), vmono)
    )
    rep.p("  widened df band; a DECREASE would be a bug)")
    if not vmono:
        rep.contradict(
            "§sanity",
            "vocab_size DECREASES with max_df (%s) -- the pruning "
            "rule is not doing what we think it is; stop and fix this before "
            "reading anything else here." % np.array2string(vs),
        )
    npair = sens["n_pairs"].unique()
    rep.p(
        "  pairs scored at each max_df: %s  identical=%s"
        % (npair.tolist(), len(npair) == 1)
    )
    rep.p(
        "  levels move with max_df (expected): sim(s=1) = %s"
        % np.array2string(sens["sim_s1"].to_numpy(), precision=4)
    )
    lvl_mono = bool(
        np.all(np.diff(sens.sort_values("max_df")["sim_s1"].to_numpy()) > 0)
    )
    rep.p(
        "  and they move MONOTONICALLY (less pruning -> higher cosine): %s" % lvl_mono
    )

    # ---- what does the cut actually remove? --------------------------------
    rep.h("5. WHAT DOES THE max_df=%g CUT ACTUALLY REMOVE?" % primary_max_df)
    tbl, cnt = vocab_cut_terms(
        terms, df_counts, n_docs, primary_max_df, min_df, top_terms
    )
    rep.p(
        "  %d docs in the fit corpus (all census users with usable text); %d terms"
        % (cnt["n_docs"], cnt["n_terms_min_df"])
    )
    rep.p(
        "  survive min_df=%d. The max_df=%g cut sits at df=%.1f docs (%.0f%% of the fit"
        % (min_df, primary_max_df, cnt["cut_df_count"], 100 * primary_max_df)
    )
    rep.p(
        "  corpus): %d terms discarded above it, %d survive."
        % (cnt["n_discarded_by_max_df"], cnt["n_survived"])
    )
    rep.p("")
    rep.p(
        "  Top-%d DISCARDED terms (highest df, i.e. the most ubiquitous) and top-%d"
        % (top_terms, top_terms)
    )
    rep.p("  SURVIVING terms (highest df, i.e. immediately below the cut):")
    rep.table(tbl, "maxdf_vocab_cut_terms", floatfmt="%.4f")
    rep.p("")
    rep.p("  A mechanical read (the substantive one is yours):")
    for line in read_discarded(tbl):
        rep.p(line)

    _emit(rep, out)
    print("done. see %s" % (out / "paper_numbers_maxdf.txt"))


def _emit(rep, out):
    """Write the report.

    NOTE: run_paper_numbers.Report.flush() writes paper_numbers.txt, which would
    clobber the existing report. So we write our own file and APPEND an idempotent
    block to paper_numbers.txt (re-running replaces the block rather than stacking
    copies), which is what the task asks for without destroying anything.
    """
    lines = rep.lines
    if rep.contradictions:
        lines = (
            [
                "",
                "#" * 78,
                "# SUMMARY OF DRAFT CONTRADICTIONS (see flagged sections below)",
                "#" * 78,
            ]
            + ["  - " + c for c in rep.contradictions]
            + lines
        )
    body = "\n".join(lines) + "\n"

    own = Path(out) / "paper_numbers_maxdf.txt"
    own.write_text(body)
    print("wrote %s" % own)

    marker = "8. MAX_DF SENSITIVITY (appended by paper_numbers_maxdf.py)"
    block = "\n" + "=" * 78 + "\n" + marker + "\n" + "=" * 78 + "\n" + body
    pn = Path(out) / "paper_numbers.txt"
    if pn.exists():
        prev = pn.read_text()
        cut = prev.find("=" * 78 + "\n" + marker)  # drop any block from a previous run
        pn.write_text((prev if cut < 0 else prev[:cut]).rstrip("\n") + "\n" + block)
        print("appended the max_df section to %s" % pn)


if __name__ == "__main__":
    main()
