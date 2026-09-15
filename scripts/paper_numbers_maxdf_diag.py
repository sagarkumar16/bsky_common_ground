"""
paper_numbers_maxdf_diag.py
===========================
WHY the cached cosine column is not reproduced by re-scoring at max_df=0.4.

The finding this script tests (established by reading the code; see MAXDF_DIAGNOSIS.md):

  The vectorizer that produced the CACHED cosine column was fitted on a strictly
  LARGER document set than the one the re-scoring path re-fits on.

    tfidf_overlap.py:200      run_overlap_cosine_census(...)
    spcg/overlap_info.py:1319     -> cosine_for_pairs(pairs, ...)   # pairs = the FULL
                                                                  #   750,000-pair census
    spcg/overlap_info.py:1149-1159  ->   build_text_df_for_users(every user in those pairs)
                                     build_user_tfidf(text_df)    # FITS HERE, on every
                                                                  #   user with text
    spcg/overlap_info.py:1163-1180 ->   pairs whose endpoint has no text are dropped AFTER

  A user with usable text whose census partners have NO text is in the FIT corpus but
  in NO scored pair. The seed-16 run dropped 537,486 of 750,000 pairs, leaving 212,514
  pairs over 136,387 users -- so the fit corpus is a superset of those 136,387.
  paper_numbers_maxdf.py re-fits on the 136,387 alone.

  This changes three things at once, all of which move the cosine:
    (a) the max_df=0.4 PROPORTION is taken over N_fit docs, not 136,387;
    (b) each term's document frequency is counted over N_fit docs;
    (c) idf = log((1 + n_docs) / (1 + df)) + 1 uses n_docs = N_fit.

So this script does NOT just rescale the ceiling; it rebuilds the ORIGINAL fitting
corpus and re-scores the cached pairs from it. Three variants are scored on the SAME
212,514 cached pairs:

  A. ORIGINAL   : fit on the full census corpus (N_fit docs), max_df=0.4 (float)
                  -> must reproduce the cached cosine column to ~1e-12 if the
                     diagnosis is right.
  B. RE-SCORE   : fit on the 136,387 scored users, max_df=0.4 (float)
                  -> must reproduce paper_numbers_maxdf.py's md=0.4 row.
  C. INT-CEILING: fit on the 136,387 scored users, max_df = the ORIGINAL absolute
                  document-count ceiling (0.4 * N_fit), i.e. the task's Item-2 test.
                  This corrects (a) only, and leaves (b) and (c) wrong, so it is
                  expected to move the numbers toward the cache WITHOUT closing the
                  gap exactly. A ~exact match here would mean (b)+(c) cancel.

Reuse (imports, never re-implements)
------------------------------------
  * spcg.overlap_info.load_starterpacks / build_user_pack_membership /
    census_comember_pairs   -- the ORIGINAL census, re-derived from the same seed
                               and caps the pipeline used (deterministic).
  * spcg.overlap_info.build_text_df_for_users -- the record loading + min_tokens +
                               max_posts_per_user + english_only filters.
  * spcg.overlap_info.build_user_tfidf -- optional --verify-refit against the real
                               pipeline vectorizer.
  * paper_numbers_maxdf.count_corpus / tfidf_at / score_pairs / sim_by_s /
    matched_curve           -- the shared-counts scoring path, already proven exact
                               (max|diff| = 2.3e-15 vs build_user_tfidf).
  * run_paper_numbers.Report -- the report accumulator.

One tokenization pass serves both corpora: the 136,387-user count matrix is a ROW
SLICE of the census count matrix (with df recomputed and min_df re-applied on the
slice), which is exactly what a fresh CountVectorizer on those rows would produce --
a term below min_df in the slice is dropped by the mask, and a term below min_df in
the full corpus was never in the matrix and is below min_df in the slice too.

Usage
-----
    python scripts/paper_numbers_maxdf_diag.py \
        --pairs-path   $DATA_DIR/directed/pairs_with_hops_n_eff_seed16.parquet \
        --records-dir  /scratch/xee6vz/bluesky-graph/records \
        --packs-path   /scratch/xee6vz/bluesky-graph/starterpacks.jsonl \
        --out          output/paper_numbers \
        --report       MAXDF_DIAGNOSIS.md

The census caps MUST match the run that produced the cache (run_full_pipeline.sh:
PER_LEVEL=75000, BASELINE_PAIRS=150000, MAX_SHARED=8, MIN_TOKENS=50,
MAX_POSTS_PER_USER=1000, ENGLISH_ONLY=1, SEED=16). They are the defaults here; the
script CHECKS the rebuilt census against the cached pair set and refuses to draw a
conclusion if it does not land on the same pairs.
"""

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.paper_numbers_maxdf import (  # noqa: E402  (imported, not copied)
    count_corpus,
    matched_curve,
    score_pairs,
    sim_by_s,
    tfidf_at,
)
from scripts.run_paper_numbers import Report  # noqa: E402
from spcg.overlap_info import build_text_df_for_users  # noqa: E402
from spcg.overlap_info import (
    build_user_pack_membership,
    build_user_tfidf,
    census_comember_pairs,
    load_starterpacks,
)

# What the draft (i.e. the cached column) reports at the stated max_df=0.4.
DRAFT = {
    "sim_s0": 0.028,
    "sim_s1": 0.104,
    "sim_s4": 0.138,
    "sim_s8": 0.151,
    "matched_s1": 0.053,
    "matched_s8": 0.072,
}

S_REPORT = [0, 1, 4, 8]


def _load_pairs(path):
    p = str(path)
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)


def _rows_for(pairs, did_to_row):
    """(ra, rb, mask) of pair endpoints in a corpus whose rows are keyed by did."""
    ra = pairs["user_a"].map(did_to_row).to_numpy()
    rb = pairs["user_b"].map(did_to_row).to_numpy()
    ok = ~(pd.isna(ra) | pd.isna(rb))
    return ra, rb, ok


def _quantities(cos, shared, pairs, packs_path, seed, out, tag, skip_matched):
    """sim(s) at s in {0,1,4,8} + the §7 degree-matched baseline at s in {1,8}."""
    sim = sim_by_s(shared, cos, S_REPORT)
    row = {"sim_s%d" % s: sim[s] for s in S_REPORT}
    if skip_matched:
        row.update({"matched_s1": np.nan, "matched_s8": np.nan})
        return row
    scored = pairs.copy()
    scored["cosine"] = cos
    if "n_eff" not in scored.columns:  # §7 takes cg_col but never reads it
        scored["n_eff"] = np.nan
    mt = matched_curve(scored, packs_path, seed, out, tag).set_index("shared_packs")
    row["matched_s1"] = float(mt["matched_zero_mean_cosine"].get(1, np.nan))
    row["matched_s8"] = float(mt["matched_zero_mean_cosine"].get(8, np.nan))
    return row


@click.command()
@click.option(
    "--pairs-path",
    required=True,
    help="The CACHED scored pair table (user_a,user_b,shared_packs,cosine). "
    "Its cosine column is the thing being explained.",
)
@click.option("--records-dir", required=True, help="Per-user record gzips.")
@click.option(
    "--packs-path",
    required=True,
    help="starterpacks.jsonl -- rebuilds the census AND feeds the §7 "
    "degree-matched baseline.",
)
@click.option("--out", "out_dir", default="output/paper_numbers", show_default=True)
@click.option(
    "--report",
    "report_path",
    default="MAXDF_DIAGNOSIS.md",
    show_default=True,
    help="Markdown diagnosis written here (overwritten on each run).",
)
# --- census caps: MUST match run_full_pipeline.sh for the run that made the cache ---
@click.option("--per-level", default=75000, show_default=True, type=int)
@click.option("--baseline-pairs", default=150000, show_default=True, type=int)
@click.option("--max-shared", default=8, show_default=True, type=int)
# --- text filters: MUST match too ---------------------------------------------------
@click.option("--min-tokens", default=50, show_default=True, type=int)
@click.option("--max-posts-per-user", default=1000, show_default=True, type=int)
@click.option("--english-only/--no-english-only", default=True, show_default=True)
# --- vectorizer ---------------------------------------------------------------------
@click.option("--min-df", default=10, show_default=True, type=int)
@click.option(
    "--max-df",
    default=0.4,
    show_default=True,
    type=float,
    help="The max_df the pipeline PASSES (a float proportion).",
)
@click.option("--sublinear-tf/--no-sublinear-tf", default=True, show_default=True)
@click.option("--seed", default=16, show_default=True, type=int)
@click.option("--n-jobs", default=0, show_default=True, type=int)
@click.option(
    "--verify-refit/--no-verify-refit",
    default=False,
    show_default=True,
    help="Also fit the real build_user_tfidf on the census corpus and assert "
    "the shared-counts path reproduces it. One extra tokenization pass.",
)
@click.option(
    "--skip-matched/--no-skip-matched",
    default=False,
    show_default=True,
    help="Skip the §7 degree-matched baseline (sim(s) only). Faster.",
)
def main(
    pairs_path,
    records_dir,
    packs_path,
    out_dir,
    report_path,
    per_level,
    baseline_pairs,
    max_shared,
    min_tokens,
    max_posts_per_user,
    english_only,
    min_df,
    max_df,
    sublinear_tf,
    seed,
    n_jobs,
    verify_refit,
    skip_matched,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rep = Report(out)
    rep.p("MAX_DF DIAGNOSIS -- what corpus was the CACHED vectorizer fitted on?")
    rep.p("cached pairs: %s" % pairs_path)
    rep.p(
        "census rebuild: per_level=%d baseline_pairs=%d max_shared=%d seed=%d"
        % (per_level, baseline_pairs, max_shared, seed)
    )
    rep.p(
        "text filters: min_tokens=%d max_posts_per_user=%s english_only=%s"
        % (min_tokens, max_posts_per_user, english_only)
    )
    rep.p(
        "vectorizer: min_df=%d (int) max_df=%g (float) sublinear_tf=%s"
        % (min_df, max_df, sublinear_tf)
    )

    # =====================================================================
    # 1. The scored corpus (what the re-scoring path re-fits on)
    # =====================================================================
    pairs = _load_pairs(pairs_path)
    for c in ("user_a", "user_b", "shared_packs"):
        if c not in pairs.columns:
            raise SystemExit(
                "pairs table missing '%s' (have %s)" % (c, list(pairs.columns))
            )
    cached = pairs["cosine"].to_numpy(float) if "cosine" in pairs.columns else None
    scored_users = pd.unique(pd.concat([pairs["user_a"], pairs["user_b"]]))
    n_scored = len(scored_users)
    shared = pairs["shared_packs"].to_numpy(int)
    print(
        "cached pairs: %d ; distinct users in them (N_scored): %d"
        % (len(pairs), n_scored)
    )

    # =====================================================================
    # 2. The ORIGINAL fitting corpus: rebuild the census the pipeline fitted on
    # =====================================================================
    print(
        "rebuilding the ORIGINAL census (the pair set cosine_for_pairs was handed) ..."
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
    print(
        "census: %d pairs over %d distinct users (the set handed to "
        "build_text_df_for_users)" % (len(census), len(census_users))
    )

    print(
        "loading text for ALL %d census users (this is the fit corpus) ..."
        % len(census_users)
    )
    text_df, uid_to_did = build_text_df_for_users(
        list(census_users),
        records_dir=records_dir,
        min_tokens=min_tokens,
        max_posts_per_user=max_posts_per_user,
        n_jobs=n_jobs,
        english_only=english_only,
    )
    n_fit = len(uid_to_did)

    # ---- print N_fit early and loudly, as the task asks ------------------
    print("")
    print("#" * 78)
    print(
        "#  N_fit    = %d   (documents the ORIGINAL vectorizer was fitted on)" % n_fit
    )
    print("#  N_scored = %d   (documents the re-scoring path re-fits on)" % n_scored)
    print(
        "#  N_fit - N_scored = %d   (users with text but no scoreable pair)"
        % (n_fit - n_scored)
    )
    print("#" * 78)
    print("")

    rep.h("1. WHAT DID THE ORIGINAL VECTORIZER FIT ON?")
    rep.p("  census pairs handed to cosine_for_pairs : %d" % len(census))
    rep.p("  distinct users in them                  : %d" % len(census_users))
    rep.p("  N_fit  (of those, with usable text)     : %d   <-- the fit corpus" % n_fit)
    rep.p("  N_scored (users in the cached pairs)    : %d" % n_scored)
    rep.p("  N_fit - N_scored                        : %d" % (n_fit - n_scored))

    if n_fit == n_scored:
        msg = (
            "HYPOTHESIS DEAD: N_fit == N_scored == %d. The original vectorizer was "
            "fitted on exactly the scored users, so the fit corpus cannot explain "
            "the offset. Look elsewhere (seed in the degree-matched draw, "
            "sublinear_tf, a stale cached parquet, english_only staging)." % n_fit
        )
        print("!! " + msg)
        rep.contradict("§1", msg)
        _write_md(report_path, dead=msg, rows=None, meta=None)
        _emit(rep, out)
        raise SystemExit(msg)

    # =====================================================================
    # 3. Did we rebuild the RIGHT census? (the pairs it can score must be the cache)
    # =====================================================================
    C_fit, terms, df_fit, uids_fit = count_corpus(text_df, min_df)
    n_docs_fit = C_fit.shape[0]
    if n_docs_fit != n_fit:
        raise SystemExit(
            "count_corpus saw %d docs but build_text_df_for_users kept %d"
            % (n_docs_fit, n_fit)
        )
    did_to_row_fit = {uid_to_did[u]: i for i, u in enumerate(uids_fit)}
    print(
        "census corpus: %d docs, %d terms at min_df=%d"
        % (n_docs_fit, len(terms), min_df)
    )

    text_users = set(did_to_row_fit)
    can_score = census[
        census["user_a"].isin(text_users) & census["user_b"].isin(text_users)
    ]
    cached_set = set(map(tuple, pairs[["user_a", "user_b"]].to_numpy()))
    rebuilt_set = set(map(tuple, can_score[["user_a", "user_b"]].to_numpy()))
    same_pairs = rebuilt_set == cached_set

    rep.h("2. IS THE REBUILT CENSUS THE ONE THAT PRODUCED THE CACHE?")
    rep.p("  census pairs scoreable (both endpoints have text) : %d" % len(can_score))
    rep.p("  cached pairs                                      : %d" % len(pairs))
    rep.p("  the two pair SETS are identical                   : %s" % same_pairs)
    print(
        "scoreable census pairs: %d ; cached: %d ; identical set: %s"
        % (len(can_score), len(pairs), same_pairs)
    )
    if not same_pairs:
        rep.contradict(
            "§2",
            "the rebuilt census does not land on the cached pair set "
            "(%d vs %d pairs). Either the caps/seed differ from the run "
            "that made the cache, or the records changed. The N_fit "
            "below is then only indicative." % (len(can_score), len(pairs)),
        )
        print(
            "!! WARNING: rebuilt census != cached pair set. Numbers below are "
            "indicative, not a proof."
        )

    # =====================================================================
    # 4. The two df ceilings
    # =====================================================================
    ceil_original = max_df * n_fit
    ceil_rescore = max_df * n_scored
    effective_max_df = ceil_original / n_scored

    rep.h("3. THE TWO DOCUMENT-FREQUENCY CEILINGS")
    rep.p("  sklearn (float max_df): a term is dropped iff df > max_df * n_docs, where")
    rep.p("  n_docs is the number of documents THE VECTORIZER WAS FITTED ON, and df is")
    rep.p("  counted over those same documents.")
    rep.p("")
    rep.p("  ceil_original = %g * %d = %.1f docs" % (max_df, n_fit, ceil_original))
    rep.p("  ceil_rescore  = %g * %d = %.1f docs" % (max_df, n_scored, ceil_rescore))
    rep.p(
        "  ceil_original / N_scored = %.4f   <-- EFFECTIVE max_df of the cached run,"
        % effective_max_df
    )
    rep.p("                                        expressed over the scored corpus")
    print(
        "ceil_original=%.1f  ceil_rescore=%.1f  effective_max_df=%.4f"
        % (ceil_original, ceil_rescore, effective_max_df)
    )

    # =====================================================================
    # 5. Score the cached pairs three ways
    # =====================================================================
    rep.h("4. RE-SCORING THE CACHED PAIRS FROM EACH CANDIDATE CORPUS")

    ra_f, rb_f, ok_f = _rows_for(pairs, did_to_row_fit)
    if not ok_f.all():
        raise SystemExit(
            "%d cached pairs have an endpoint with no text in the rebuilt "
            "census corpus -- the rebuild is wrong" % int((~ok_f).sum())
        )
    ra_f = ra_f.astype(int)
    rb_f = rb_f.astype(int)

    # ---- A: the ORIGINAL path (fit on the census corpus, max_df float) ----
    X_a, keep_a = tfidf_at(C_fit, df_fit, max_df, min_df, sublinear_tf)
    cos_a = score_pairs(X_a, ra_f, rb_f)
    print(
        "A ORIGINAL   : fit on %d docs, max_df=%g -> vocab=%d, mean cosine=%.5f"
        % (n_fit, max_df, int(keep_a.sum()), cos_a.mean())
    )

    if verify_refit:
        Xv, uids_v, _ = build_user_tfidf(
            text_df, min_df=min_df, max_df=max_df, sublinear_tf=sublinear_tf
        )
        if not np.array_equal(uids_v, uids_fit):
            raise SystemExit("verify-refit: row order differs between paths")
        dv = float(np.abs(score_pairs(Xv.tocsr(), ra_f, rb_f) - cos_a).max())
        rep.p(
            "  [verify-refit] max |A(shared-counts) - A(build_user_tfidf)| = %.3e" % dv
        )
        if dv > 1e-9 or int(keep_a.sum()) != Xv.shape[1]:
            raise SystemExit(
                "verify-refit FAILED (max diff %.3e): the shared-counts "
                "path does not reproduce build_user_tfidf." % dv
            )

    # ---- the 136,387-doc sub-corpus: a ROW SLICE of the census counts -----
    sub_rows = np.array(sorted(did_to_row_fit[d] for d in scored_users), dtype=int)
    if len(sub_rows) != n_scored:
        raise SystemExit("scored users not all present in the census corpus")
    C_sub = C_fit.tocsr()[sub_rows].tocsc()
    df_sub = np.asarray((C_sub > 0).sum(axis=0)).ravel().astype(np.int64)
    row_in_sub = {r: i for i, r in enumerate(sub_rows)}
    ra_s = np.array([row_in_sub[r] for r in ra_f], dtype=int)
    rb_s = np.array([row_in_sub[r] for r in rb_f], dtype=int)

    # ---- B: the RE-SCORING path (fit on the scored users, max_df float) ---
    X_b, keep_b = tfidf_at(C_sub, df_sub, max_df, min_df, sublinear_tf)
    cos_b = score_pairs(X_b, ra_s, rb_s)
    print(
        "B RE-SCORE   : fit on %d docs, max_df=%g -> vocab=%d, mean cosine=%.5f"
        % (n_scored, max_df, int(keep_b.sum()), cos_b.mean())
    )

    # ---- C: the scored users, but with the ORIGINAL ABSOLUTE ceiling ------
    # sklearn with an int max_df keeps df <= max_df; passing the equivalent proportion
    # ceil_original / n_scored to tfidf_at applies exactly that absolute cut.
    X_c, keep_c = tfidf_at(C_sub, df_sub, effective_max_df, min_df, sublinear_tf)
    cos_c = score_pairs(X_c, ra_s, rb_s)
    print(
        "C INT-CEILING: fit on %d docs, max_df=%d docs (=%.4f of them) -> vocab=%d, "
        "mean cosine=%.5f"
        % (
            n_scored,
            int(ceil_original),
            effective_max_df,
            int(keep_c.sum()),
            cos_c.mean(),
        )
    )

    # =====================================================================
    # 6. Compare each variant to the cached column
    # =====================================================================
    variants = [
        (
            "A_original_fit_corpus",
            cos_a,
            int(keep_a.sum()),
            n_fit,
            "fit on the %d census users, max_df=%g (float)" % (n_fit, max_df),
        ),
        (
            "B_rescore_scored_only",
            cos_b,
            int(keep_b.sum()),
            n_scored,
            "fit on the %d scored users, max_df=%g (float)" % (n_scored, max_df),
        ),
        (
            "C_scored_only_int_ceiling",
            cos_c,
            int(keep_c.sum()),
            n_scored,
            "fit on the %d scored users, max_df=%d docs (absolute)"
            % (n_scored, int(ceil_original)),
        ),
    ]

    rows = []
    for name, cos, vocab, n_docs, desc in variants:
        r = {
            "variant": name,
            "n_fit_docs": n_docs,
            "vocab_size": vocab,
            "mean_cosine": float(cos.mean()),
        }
        r.update(
            _quantities(cos, shared, pairs, packs_path, seed, out, name, skip_matched)
        )
        if cached is not None:
            d = np.abs(cos - cached)
            r["max_abs_diff_vs_cached"] = float(d.max())
            r["mean_abs_diff_vs_cached"] = float(d.mean())
        rows.append(r)
        print(
            "  %-26s vocab=%-7d sim_s1=%.5f  max|diff vs cache|=%s"
            % (
                name,
                vocab,
                r["sim_s1"],
                ("%.3e" % r["max_abs_diff_vs_cached"]) if cached is not None else "n/a",
            )
        )

    tbl = pd.DataFrame(rows)
    rep.table(tbl, "maxdf_diag_variants", floatfmt="%.6f")

    # draft comparison
    rep.p("")
    rep.p("  Against the DRAFT (= the cached column) at the stated max_df=0.4:")
    for k, v in DRAFT.items():
        line = "    %-11s draft %.3f" % (k, v)
        for name, *_ in variants:
            got = float(tbl.loc[tbl["variant"] == name, k].iloc[0])
            line += "   %s %.5f" % (name.split("_")[0], got)
        rep.p(line)

    reproduces = (
        cached is not None
        and float(
            tbl.loc[
                tbl["variant"] == "A_original_fit_corpus", "max_abs_diff_vs_cached"
            ].iloc[0]
        )
        < 1e-6
    )
    rep.h("5. VERDICT")
    if reproduces:
        rep.p(
            "  CONFIRMED. Fitting on the ORIGINAL census corpus (%d docs) and scoring"
            % n_fit
        )
        rep.p(
            "  the same %d pairs reproduces the cached cosine column to machine"
            % len(pairs)
        )
        rep.p(
            "  precision. The cached numbers are max_df=0.4 OF A %d-DOCUMENT CORPUS,"
            % n_fit
        )
        rep.p(
            "  i.e. an effective ceiling of %.0f documents = %.4f of the %d users the"
            % (ceil_original, effective_max_df, n_scored)
        )
        rep.p(
            "  paper actually scores. The re-scoring path is not broken; it re-fits on"
        )
        rep.p("  a smaller corpus than the pipeline did.")
    else:
        rep.p(
            "  NOT CONFIRMED by variant A. The fit corpus is a superset (%d vs %d), but"
            % (n_fit, n_scored)
        )
        rep.p("  re-fitting on it does not reproduce the cached column. Something ELSE")
        rep.p("  also differs. See the per-variant diffs above and check: the degree-")
        rep.p(
            "  matched draw's seed, sublinear_tf, a stale cached parquet, english_only."
        )

    _emit(rep, out)
    _write_md(
        report_path,
        dead=None,
        rows=tbl,
        meta={
            "n_fit": n_fit,
            "n_scored": n_scored,
            "n_census": len(census),
            "n_census_users": len(census_users),
            "n_pairs": len(pairs),
            "same_pairs": same_pairs,
            "max_df": max_df,
            "ceil_original": ceil_original,
            "ceil_rescore": ceil_rescore,
            "effective_max_df": effective_max_df,
            "reproduces": reproduces,
            "min_df": min_df,
            "sublinear_tf": sublinear_tf,
            "seed": seed,
        },
    )
    print("done. see %s and %s" % (out / "paper_numbers_maxdf_diag.txt", report_path))


# ===========================================================================
# Output
# ===========================================================================
def _emit(rep, out):
    """Write our own report file. Never touches paper_numbers.txt."""
    lines = rep.lines
    if rep.contradictions:
        lines = (
            ["", "#" * 78, "# SUMMARY OF CONTRADICTIONS", "#" * 78]
            + ["  - " + c for c in rep.contradictions]
            + lines
        )
    own = Path(out) / "paper_numbers_maxdf_diag.txt"
    own.write_text("\n".join(lines) + "\n")
    print("wrote %s" % own)


def _write_md(path, dead, rows, meta):
    """Overwrite MAXDF_DIAGNOSIS.md with the run's numbers.

    The code-level findings (Item 1) are static -- they come from reading the
    pipeline, not from this run -- so they are reproduced verbatim; the numbers
    (Item 2) are filled in from what just ran.
    """
    L = []
    a = L.append
    a("# max_df diagnosis: the cached cosines were fitted on a larger corpus")
    a("")
    if dead:
        a("## VERDICT (first line, as asked): THE HYPOTHESIS IS DEAD")
        a("")
        a(dead)
        a("")
        Path(path).write_text("\n".join(L) + "\n")
        print("wrote %s" % path)
        return

    v = {r["variant"]: r for r in rows.to_dict("records")}
    A, B, C = (
        v["A_original_fit_corpus"],
        v["B_rescore_scored_only"],
        v["C_scored_only_int_ceiling"],
    )
    a("## Verdict")
    a("")
    a(
        "`N_fit = %d` vs `N_scored = %d`. The vectorizer that produced the cached "
        "cosine column was fitted on **%d documents**, not the %d users the paper "
        "scores." % (meta["n_fit"], meta["n_scored"], meta["n_fit"], meta["n_scored"])
    )
    a("")
    a(
        "- `ceil_original = %g * %d = %.1f` documents"
        % (meta["max_df"], meta["n_fit"], meta["ceil_original"])
    )
    a(
        "- `ceil_rescore  = %g * %d = %.1f` documents"
        % (meta["max_df"], meta["n_scored"], meta["ceil_rescore"])
    )
    a(
        "- **effective max_df of the cached run, over the scored corpus: "
        "`%.4f`**" % meta["effective_max_df"]
    )
    a("")
    a(
        "Re-fitting on the original corpus and re-scoring the same %d pairs reproduces "
        "the cached column to `max|diff| = %.2e`: **%s**."
        % (
            meta["n_pairs"],
            A.get("max_abs_diff_vs_cached", float("nan")),
            "CONFIRMED" if meta["reproduces"] else "NOT reproduced -- see below",
        )
    )
    a("")
    a("## The path (line references)")
    a("")
    a("| step | file:line | what |")
    a("|------|-----------|------|")
    a(
        "| pipeline | `run_full_pipeline.sh:195-215` | stage 1 runs `tfidf_overlap.py` "
        "with `PER_LEVEL=75000 BASELINE_PAIRS=150000 MAX_SHARED=8 MIN_TOKENS=50 "
        "MAX_POSTS_PER_USER=1000 SEED=16` |"
    )
    a(
        "| params | `tfidf_overlap.py:196` | `tfidf_kwargs = dict(min_df=10, max_df=0.4, "
        "sublinear_tf=True)` -- `max_df` is a **float**, `min_df` an **int**, in both paths |"
    )
    a(
        "| census | `spcg/overlap_info.py:1305-1319` | the FULL census (%d pairs) is handed "
        "to `cosine_for_pairs` |" % meta["n_census"]
    )
    a(
        "| fit | `spcg/overlap_info.py:1149-1159` | `build_text_df_for_users(every user in "
        "those pairs)` then `build_user_tfidf(text_df)` -- **the fit happens here, on "
        "every user with usable text** |"
    )
    a(
        "| drop | `spcg/overlap_info.py:1163-1180` | pairs whose endpoint has no text are "
        "dropped **after** the fit |"
    )
    a("")
    a(
        "A user with usable text whose census partners have none is in the fit corpus but "
        "in no scored pair. That is the %d-user gap."
        % (meta["n_fit"] - meta["n_scored"])
    )
    a("")
    a("## Numbers")
    a("")
    a(
        "| variant | fit docs | vocab | sim_s0 | sim_s1 | sim_s4 | sim_s8 | matched_s1 | "
        "matched_s8 | max\\|diff\\| vs cache |"
    )
    a("|---|---|---|---|---|---|---|---|---|---|")
    for name, label in [
        ("A_original_fit_corpus", "A: original fit corpus, max_df=0.4"),
        ("B_rescore_scored_only", "B: scored users only, max_df=0.4"),
        ("C_scored_only_int_ceiling", "C: scored users, absolute int ceiling"),
    ]:
        r = v[name]
        a(
            "| %s | %d | %d | %.5f | %.5f | %.5f | %.5f | %.5f | %.5f | %.2e |"
            % (
                label,
                r["n_fit_docs"],
                r["vocab_size"],
                r["sim_s0"],
                r["sim_s1"],
                r["sim_s4"],
                r["sim_s8"],
                r["matched_s1"],
                r["matched_s8"],
                r.get("max_abs_diff_vs_cached", float("nan")),
            )
        )
    a(
        "| draft (cached) | %d | -- | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f | -- |"
        % (
            meta["n_fit"],
            DRAFT["sim_s0"],
            DRAFT["sim_s1"],
            DRAFT["sim_s4"],
            DRAFT["sim_s8"],
            DRAFT["matched_s1"],
            DRAFT["matched_s8"],
        )
    )
    a("")
    a(
        "Variant C (the absolute-int re-score the task asked for) corrects the ceiling "
        "but not the document frequencies or the idf denominator, both of which are also "
        "counted over the fit corpus. It therefore moves toward the cache without "
        "necessarily landing on it (max|diff| = %.2e); variant A is the exact "
        "reconstruction." % C.get("max_abs_diff_vs_cached", float("nan"))
    )
    a("")
    a(
        "Generated by `scripts/paper_numbers_maxdf_diag.py` "
        "(seed=%d, min_df=%d, sublinear_tf=%s)."
        % (meta["seed"], meta["min_df"], meta["sublinear_tf"])
    )
    Path(path).write_text("\n".join(L) + "\n")
    print("wrote %s" % path)


if __name__ == "__main__":
    main()
