"""
run_paper_numbers.py
====================
Read-only reporting script that resolves the paper's outstanding ``\\todo{}``
numbers from the CACHED parquet artifacts of a completed pipeline run. It emits
numbers and small tables; it does NOT recompute the expensive stages (no BFS,
no re-embedding, no re-scoring) except the optional degree-stratified null in
§7, gated behind ``--run-degree-null``.

It imports (never re-implements) the existing helpers:
  * ``run_network_analysis.node_cluster_bootstrap`` -- the user-level (dyadic)
    cluster bootstrap of arbitrary per-weight statistics. This is the single
    reused resampler behind every CI on a contrast (slopes, cell means, stratum
    gaps/ratios): resample USERS with replacement, weight each pair by the
    product of its two endpoints' draw multiplicities.
  * ``run_network_analysis._wls_beta`` -- weighted-least-squares coefficients.
    (Imported despite the leading underscore, per the task's instruction to
    import awkwardly-scoped helpers rather than refactor or copy them. See the
    NOTE ON REUSE section at the bottom of the emitted paper_numbers.txt.)
  * ``spcg.overlap_info.cluster_bootstrap_ci`` -- per-bin bootstrap CIs (same
    resampling), used for the raw shared-pack / n_eff bin curves.
  * ``spcg.overlap_info.load_starterpacks`` / ``build_user_pack_membership`` --
    pack membership for §6/§7.
  * ``network_distance.bin_hops`` -- the existing hop binning (sentinel =
    max_hops+1, label "unreachable").
  * ``pack_semantics.load_pack_descriptions`` / (npz reader) -- pack names +
    cached embeddings for §6.

Every result is written BOTH as a parquet/csv under ``--out`` AND as a
human-readable ``paper_numbers.txt`` with one clearly-labelled section per item.

Usage
-----
    python run_paper_numbers.py \
        --pairs-path  /scratch/xee6vz/study_data/directed/pairs_with_hops_n_eff_seed16.parquet \
        --null-curves output/run_20260709_053144_directed/null_curves/null_curves_cluster_average_seed16.parquet \
        --packs-path  /scratch/xee6vz/bluesky-graph/starterpacks.jsonl \
        --emb-cache   /scratch/xee6vz/study_data/pack_emb_all-MiniLM-L6-v2.npz \
        --out         output/paper_numbers/
"""

import sys
from pathlib import Path

import click
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import spcg.network_distance as nd  # noqa: E402  (bin_hops)

# node_cluster_bootstrap is public; _wls_beta is private but reused deliberately
# (task: import awkward helpers, do not copy). See NOTE ON REUSE in the report.
from scripts.run_network_analysis import _wls_beta, node_cluster_bootstrap  # noqa: E402
from spcg.overlap_info import build_user_pack_membership  # noqa: E402
from spcg.overlap_info import cluster_bootstrap_ci, load_starterpacks

try:
    import statsmodels.api as sm

    _HAVE_SM = True
except Exception:  # pragma: no cover
    _HAVE_SM = False


# ===========================================================================
# Small utilities
# ===========================================================================
class Report:
    """Accumulates the human-readable paper_numbers.txt and remembers where the
    machine-readable tables were written, so the report can point at them."""

    def __init__(self, out_dir):
        self.out_dir = Path(out_dir)
        self.lines = []
        self.contradictions = []

    def h(self, title):
        self.lines += ["", "=" * 78, title, "=" * 78]

    def p(self, *msg):
        self.lines.append(" ".join(str(m) for m in msg))

    def contradict(self, section, msg):
        block = ["", "!! CONTRADICTS DRAFT !!  (%s)" % section, "  " + msg]
        self.lines += block
        self.contradictions.append("[%s] %s" % (section, msg))

    def table(self, df, name, floatfmt="%.5f"):
        """Persist df as csv+parquet under out_dir and inline it into the txt."""
        stem = self.out_dir / name
        try:
            df.to_parquet(stem.with_suffix(".parquet"), index=False)
        except Exception:
            pass
        df.to_csv(stem.with_suffix(".csv"), index=False)
        self.lines.append(
            df.to_string(index=False, float_format=lambda v: floatfmt % v)
        )
        self.lines.append("  -> wrote %s.csv" % name)

    def flush(self):
        if self.contradictions:
            self.lines = (
                [
                    "",
                    "#" * 78,
                    "# SUMMARY OF DRAFT CONTRADICTIONS (see flagged sections below)",
                    "#" * 78,
                ]
                + ["  - " + c for c in self.contradictions]
                + self.lines
            )
        path = self.out_dir / "paper_numbers.txt"
        path.write_text("\n".join(self.lines) + "\n")
        print("\nwrote %s" % path)


def _wmean(w, v):
    """Weighted mean, nan if no weight (matches the bootstrap's own convention)."""
    tot = w.sum()
    return float((w * v).sum() / tot) if tot > 0 else np.nan


def _ols_r2(x, y):
    """R^2 of the unweighted simple regression y ~ 1 + x (point estimate)."""
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return (1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan, float(beta[1])


# ===========================================================================
# Sanity checks -- fail loudly (task §"Sanity checks to assert")
# ===========================================================================
def sanity_checks(df, cg_col, max_hops, expect_rows, expect_sentinel):
    sentinel = max_hops + 1
    s = df["shared_packs"].to_numpy()
    ne = df[cg_col].to_numpy()
    h = df["hops"].to_numpy()

    problems = []

    # n_eff <= shared_packs for all rows
    if not np.all(ne <= s + 1e-9):
        problems.append("n_eff > shared_packs for %d rows" % int((ne > s + 1e-9).sum()))
    # n_eff >= 1 wherever shared_packs >= 1
    m1 = s >= 1
    if not np.all(ne[m1] >= 1 - 1e-9):
        problems.append(
            "n_eff < 1 where shared_packs>=1 for %d rows"
            % int((ne[m1] < 1 - 1e-9).sum())
        )
    # n_eff == 0 iff shared_packs == 0
    iff_bad = int(((ne < 1e-9) != (s == 0)).sum())
    if iff_bad:
        problems.append("(n_eff==0) != (shared_packs==0) for %d rows" % iff_bad)
    # total rows
    if len(df) != expect_rows:
        problems.append("row count %d != expected %d" % (len(df), expect_rows))
    # hops in {1..sentinel}
    bad_h = set(np.unique(h).tolist()) - set(range(1, sentinel + 1))
    if bad_h:
        problems.append("hops values outside {1..%d}: %s" % (sentinel, sorted(bad_h)))
    # sentinel count
    n_sent = int((h == sentinel).sum())
    if n_sent != expect_sentinel:
        problems.append(
            "hops==%d (sentinel) count %d != expected %d"
            % (sentinel, n_sent, expect_sentinel)
        )
    # three strata partition every row with no gap / no overlap
    s0 = ne < 1.0
    s13 = (ne >= 1.0) & (ne <= 3.0)
    s3 = ne > 3.0
    cover = s0.astype(int) + s13.astype(int) + s3.astype(int)
    if not np.all(cover == 1):
        problems.append(
            "strata do not partition rows: %d rows in !=1 stratum"
            % int((cover != 1).sum())
        )

    if problems:
        msg = (
            "SANITY CHECKS FAILED -- stopping (do not work around):\n  - "
            + "\n  - ".join(problems)
        )
        raise SystemExit(msg)
    print(
        "[sanity] all checks passed (%d rows, sentinel count %d)." % (len(df), n_sent)
    )


# ===========================================================================
# The shared per-weight statistic feeding ONE cluster-bootstrap pass.
# Every CI on a contrast (slope, cell mean, stratum gap/ratio) is derived from
# this single resampling, so one --seed reproduces everything (task §Uncertainty).
# ===========================================================================
def build_stat_fn(df, cg_col, levels, eps):
    cosine = df["cosine"].to_numpy(float)
    cg = df[cg_col].to_numpy(float)
    shared = df["shared_packs"].to_numpy(float)
    hops = df["hops"].to_numpy(int)
    logcos = np.log(cosine + eps)

    # §1 designs: cosine ~ 1 + shared_packs on two shared-pack windows.
    m_s18 = (shared >= 1) & (shared <= 8)
    m_s08 = (shared >= 0) & (shared <= 8)
    X_s = np.column_stack([np.ones(len(df)), shared])

    # §4 design: log(cosine+eps) ~ 1 + n_eff within each hop level.
    X_cg = np.column_stack([np.ones(len(df)), cg])

    # Fixed n_eff strata (partition every pair; boundaries from the codebase).
    strata = [
        ("n_eff=0", cg < 1.0),
        ("n_eff[1,3]", (cg >= 1.0) & (cg <= 3.0)),
        ("n_eff>3", cg > 3.0),
    ]
    hop_masks = {lvl: (hops == lvl) for lvl in levels}

    def stat(w):
        out = {}
        # --- §1 marginal slope in raw shared-pack count -------------------
        out["s1_beta_1_8"] = _wls_beta(X_s, cosine, w * m_s18)[1]
        out["s1_beta_0_8"] = _wls_beta(X_s, cosine, w * m_s08)[1]

        # --- §3 cell means (raw) + §5 log-scale cell means ----------------
        cellmean = {}
        logcellmean = {}
        for lvl in levels:
            hm = hop_masks[lvl]
            for lab, sm_ in strata:
                ww = w * (hm & sm_)
                cellmean[(lab, lvl)] = _wmean(ww, cosine)
                logcellmean[(lab, lvl)] = _wmean(ww, logcos)
                out["cellmean|%s|%d" % (lab, lvl)] = cellmean[(lab, lvl)]
            # --- §4 per-hop log-scale slope of n_eff -> log cosine --------
            ww_h = w * hm
            if hm.sum() >= 2 and np.ptp(cg[hm]) > 0:
                out["logslope|%d" % lvl] = _wls_beta(X_cg, logcos, ww_h)[1]
            else:
                out["logslope|%d" % lvl] = np.nan

        # --- §5 stratum gaps / ratios, computed as ONE scalar per boot ----
        # (CI on the contrast itself, NOT a difference of two independent CIs).
        for lvl in levels:
            g = cellmean[("n_eff>3", lvl)] - cellmean[("n_eff[1,3]", lvl)]
            out["gap|%d" % lvl] = g
            m0 = cellmean[("n_eff=0", lvl)]
            m13 = cellmean[("n_eff[1,3]", lvl)]
            out["ratio|%d" % lvl] = (m13 / m0) if (m0 and m0 > 0) else np.nan
            # log-scale ratio = difference of mean log-cosine (geometric ratio).
            out["logdiff|%d" % lvl] = (
                logcellmean[("n_eff[1,3]", lvl)] - logcellmean[("n_eff=0", lvl)]
            )
        return out

    meta = dict(strata=[lab for lab, _ in strata], eps=eps)
    return stat, meta


def _boot_lookup(boot_df):
    return {r["term"]: r for _, r in boot_df.iterrows()}


def _ci(row):
    return float(row["estimate"]), float(row["ci_low"]), float(row["ci_high"])


# ===========================================================================
# §1  Marginal slope of similarity in raw shared-pack count
# ===========================================================================
def section1(rep, df, bt):
    rep.h("1. MARGINAL SLOPE OF SIMILARITY IN RAW SHARED-PACK COUNT")
    rep.p("Fit cosine ~ shared_packs. Slope beta via user-level cluster bootstrap")
    rep.p("(pairs share users -> OLS SEs understate; see paper §Uncertainty).")

    shared = df["shared_packs"].to_numpy(float)
    cosine = df["cosine"].to_numpy(float)

    for tag, key, lo, hi in [
        ("1<=s<=8", "s1_beta_1_8", 1, 8),
        ("0<=s<=8", "s1_beta_0_8", 0, 8),
    ]:
        m = (shared >= lo) & (shared <= hi)
        r2, beta_pt = _ols_r2(shared[m], cosine[m])
        est, cl, ch = _ci(bt[key])
        rep.p("")
        rep.p("  window %s:  n=%d" % (tag, int(m.sum())))
        rep.p(
            "    beta (cluster-boot point) = %.5f   95%% CI [%.5f, %.5f]"
            % (est, cl, ch)
        )
        rep.p("    beta (plain OLS check)     = %.5f" % beta_pt)
        rep.p("    R^2                        = %.4f" % r2)

    e18, l18, h18 = _ci(bt["s1_beta_1_8"])
    e08, l08, h08 = _ci(bt["s1_beta_0_8"])
    rep.p("")
    rep.p(
        "  The naive 0..8 slope is %.2fx the 1..8 slope; the extra steepness is"
        % (e08 / e18 if e18 else float("nan"))
    )
    rep.p("  the s=0 -> s=1 jump (bin-0 pairs share no packs), NOT constant growth.")
    tbl = pd.DataFrame(
        [
            {"window": "1<=s<=8", "beta": e18, "ci_low": l18, "ci_high": h18},
            {"window": "0<=s<=8", "beta": e08, "ci_low": l08, "ci_high": h08},
        ]
    )
    rep.table(tbl, "s1_marginal_slope")
    if e18 <= 0:
        rep.contradict("§1", "beta on 1<=s<=8 is not positive (%.5f)." % e18)


# ===========================================================================
# §2  Distributional shift under renormalization
# ===========================================================================
def section2(rep, df, cg_col):
    rep.h("2. DISTRIBUTIONAL SHIFT UNDER RENORMALIZATION (shared_packs>=1)")
    d = df[df["shared_packs"] >= 1]
    s = d["shared_packs"].to_numpy(float)
    ne = d[cg_col].to_numpy(float)

    def desc(v):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        return dict(
            mean=float(v.mean()),
            median=float(med),
            q1=float(q1),
            q3=float(q3),
            iqr=float(q3 - q1),
        )

    ds, dn = desc(s), desc(ne)
    rep.p("  n pairs with shared_packs>=1: %d" % len(d))
    rep.p(
        "  shared_packs : mean=%.4f median=%.1f IQR=[%.1f, %.1f] (=%.1f)"
        % (ds["mean"], ds["median"], ds["q1"], ds["q3"], ds["iqr"])
    )
    rep.p(
        "  n_eff        : mean=%.4f median=%.4f IQR=[%.4f, %.4f] (=%.4f)"
        % (dn["mean"], dn["median"], dn["q1"], dn["q3"], dn["iqr"])
    )

    # Confirm the draft's known means (8.26 shared_packs, 4.29 n_eff).
    for label, got, claim in [
        ("shared_packs", ds["mean"], 8.26),
        ("n_eff", dn["mean"], 4.29),
    ]:
        ok = abs(got - claim) <= 0.05
        rep.p(
            "  draft mean %s claim %.2f -> computed %.4f  [%s]"
            % (label, claim, got, "confirmed" if ok else "MISMATCH")
        )
        if not ok:
            rep.contradict(
                "§2", "mean %s = %.4f, draft says %.2f." % (label, got, claim)
            )

    frac_reduced = float((ne < s - 1e-9).mean())
    frac_to1 = float((np.abs(ne - 1.0) < 1e-9).mean())
    rep.p("")
    rep.p(
        "  fraction with n_eff < shared_packs (renorm did something): %.4f"
        % frac_reduced
    )
    rep.p(
        "  fraction reduced to n_eff==1 exactly (all shared packs identical): %.4f"
        % frac_to1
    )

    ratio = ne / s
    rr = desc(ratio)
    rep.p(
        "  reduction ratio n_eff/shared_packs: mean=%.4f median=%.4f IQR=[%.4f, %.4f]"
        % (rr["mean"], rr["median"], rr["q1"], rr["q3"])
    )

    rows = []
    for k in range(1, 9):
        sub = ratio[np.abs(s - k) < 1e-9]
        if len(sub):
            rows.append(
                {
                    "shared_packs": k,
                    "n": len(sub),
                    "ratio_mean": float(sub.mean()),
                    "ratio_median": float(np.median(sub)),
                    "n_eff_mean": float(ne[np.abs(s - k) < 1e-9].mean()),
                }
            )
    rep.p("")
    rep.p("  reduction ratio conditional on shared_packs (s=1..8):")
    rep.table(pd.DataFrame(rows), "s2_reduction_ratio_by_s")

    summ = pd.DataFrame(
        [
            {"var": "shared_packs", **ds},
            {"var": "n_eff", **dn},
            {"var": "n_eff/shared_packs", **rr},
            {"var": "frac_reduced", "mean": frac_reduced},
            {"var": "frac_n_eff_eq_1", "mean": frac_to1},
        ]
    )
    rep.table(summ, "s2_distribution_summary")


# ===========================================================================
# §3  Realized cell counts for the network figure (3x4 grid)
# ===========================================================================
def section3(rep, df, cg_col, levels, meta, bt, sentinel):
    rep.h("3. REALIZED CELL COUNTS FOR THE NETWORK FIGURE (3 x 4 grid)")
    ne = df[cg_col].to_numpy(float)
    hops = df["hops"].to_numpy(int)
    strata = meta["strata"]

    def label_hop(lvl):
        return "far/unreachable" if lvl >= sentinel else str(lvl)

    strat_mask = {
        "n_eff=0": ne < 1.0,
        "n_eff[1,3]": (ne >= 1.0) & (ne <= 3.0),
        "n_eff>3": ne > 3.0,
    }

    rows = []
    for lab in strata:
        for lvl in levels:
            m = strat_mask[lab] & (hops == lvl)
            n = int(m.sum())
            key = "cellmean|%s|%d" % (lab, lvl)
            est, cl, ch = _ci(bt[key])
            rows.append(
                {
                    "n_eff_stratum": lab,
                    "hops": label_hop(lvl),
                    "n": n,
                    "mean_cosine": est,
                    "ci_low": cl,
                    "ci_high": ch,
                    "ci_width": ch - cl,
                }
            )
    tbl = pd.DataFrame(rows)
    rep.table(tbl, "s3_cell_counts")

    # ---- diagnostic on the fragile hops=1, n_eff=0 cell ------------------
    rep.p("")
    rep.p("  >>> FLAG: hops=1, n_eff=0 cell (draft CI [0.028, 0.147], +/-90% rel) <<<")
    cell = df[(ne < 1.0) & (hops == 1)]
    n = len(cell)
    users = pd.concat([cell["user_a"], cell["user_b"]])
    n_unique = int(pd.unique(users).shape[0])
    per_user = users.value_counts()
    max_by_user = int(per_user.iloc[0]) if len(per_user) else 0
    top_user_share = (max_by_user / n) if n else float("nan")
    row = bt["cellmean|n_eff=0|1"]
    est, cl, ch = _ci(row)
    rep.p("    n (pairs)                 = %d" % n)
    rep.p("    unique users spanned      = %d" % n_unique)
    rep.p(
        "    max pairs from one user   = %d  (%.1f%% of the cell)"
        % (max_by_user, 100 * top_user_share)
    )
    rep.p("    recomputed mean cosine    = %.5f  95%% CI [%.5f, %.5f]" % (est, cl, ch))
    if n < 200:
        rep.p("    -> THIN CELL: n<200; the wide CI is small-sample, not instability.")
    if max_by_user >= 0.10 * max(n, 1):
        rep.p("    -> ONE USER DOMINATES (>=10% of pairs): the cluster bootstrap can")
        rep.p(
            "       swing on that user's draw multiplicity -- this destabilises the CI."
        )
    else:
        rep.p("    -> no single user dominates (<10% of pairs); width is sample size,")
        rep.p("       not one high-degree user.")
    diag = pd.DataFrame(
        [
            {
                "cell": "hops=1,n_eff=0",
                "n": n,
                "unique_users": n_unique,
                "max_pairs_one_user": max_by_user,
                "top_user_share": top_user_share,
                "mean_cosine": est,
                "ci_low": cl,
                "ci_high": ch,
            }
        ]
    )
    rep.table(diag, "s3_hops1_neff0_diagnostic")


# ===========================================================================
# §4  Log-scale interaction (the important one)
# ===========================================================================
def section4(rep, df, cg_col, levels, bt, eps_primary, eps_set, sentinel):
    rep.h("4. LOG-SCALE INTERACTION:  log(cosine+eps) ~ n_eff * C(hops)")
    cosine = df["cosine"].to_numpy(float)
    cg = df[cg_col].to_numpy(float)
    hops = df["hops"].to_numpy(int)
    ref = levels[0]
    nonref = levels[1:]

    def label_hop(lvl):
        return "far" if lvl >= sentinel else str(lvl)

    rep.p("  eps note: cosine in [0,1] with a near-zero floor (bin-0 mean ~0.028),")
    rep.p(
        "  so log(cosine+eps) is sensitive to eps. Primary eps=%g; sweep %s."
        % (eps_primary, eps_set)
    )

    # ---- eps sensitivity of the interaction sign + significance ----------
    def fit_log_interaction(eps):
        y = np.log(cosine + eps)
        cols = {"Intercept": np.ones(len(df)), "n_eff": cg}
        for lvl in nonref:
            cols["hops[%s]" % label_hop(lvl)] = (hops == lvl).astype(float)
        for lvl in nonref:
            cols["n_eff:hops[%s]" % label_hop(lvl)] = cg * (hops == lvl).astype(float)
        X = pd.DataFrame(cols)
        res = sm.OLS(y, X).fit()
        return res

    sens_rows = []
    for eps in eps_set:
        res = fit_log_interaction(eps)
        for lvl in nonref:
            t = "n_eff:hops[%s]" % label_hop(lvl)
            sens_rows.append(
                {
                    "eps": eps,
                    "term": t,
                    "coef": float(res.params[t]),
                    "p": float(res.pvalues[t]),
                    "sign": int(np.sign(res.params[t])),
                }
            )
    sens = pd.DataFrame(sens_rows)
    rep.p("")
    rep.p("  Interaction coefficients vs eps (OLS on log scale):")
    rep.table(sens, "s4_eps_sensitivity", floatfmt="%.6f")

    # stability verdict per interaction term
    stable = True
    for t, grp in sens.groupby("term"):
        signs = set(grp["sign"])
        sig = set((grp["p"] < 0.05).tolist())
        term_stable = (len(signs) == 1) and (len(sig) == 1)
        stable = stable and term_stable
        rep.p(
            "    %-22s sign stable=%s  signif(<0.05) stable=%s"
            % (t, len(signs) == 1, len(sig) == 1)
        )
    if stable:
        rep.p("  -> interaction sign AND significance are STABLE across eps.")
    else:
        rep.p("  -> WARNING: interaction sign/significance is NOT stable across eps.")
        rep.contradict(
            "§4",
            "log-scale interaction is eps-dependent: the "
            "multiplicative claim may be an offset artifact. Do not "
            "submit the multiplicative reading without resolving this.",
        )

    # ---- headline OLS at primary eps: interaction test statistic ---------
    res = fit_log_interaction(eps_primary)
    inter_rows = []
    for lvl in nonref:
        t = "n_eff:hops[%s]" % label_hop(lvl)
        inter_rows.append(
            {
                "term": t,
                "coef": float(res.params[t]),
                "t_stat": float(res.tvalues[t]),
                "p_ols": float(res.pvalues[t]),
            }
        )
    rep.p("")
    rep.p(
        "  Interaction test (OLS log scale, eps=%g); note OLS SEs understate"
        % eps_primary
    )
    rep.p("  (pairs share users) -- treat p_ols as a screen, trust the bootstrap CIs.")
    rep.table(pd.DataFrame(inter_rows), "s4_interaction_test", floatfmt="%.6f")

    # ---- per-hop log-scale slopes with cluster-bootstrap CIs -------------
    srows = []
    for lvl in levels:
        est, cl, ch = _ci(bt["logslope|%d" % lvl])
        srows.append(
            {
                "hops": label_hop(lvl),
                "log_slope": est,
                "ci_low": cl,
                "ci_high": ch,
                "n": int((hops == lvl).sum()),
            }
        )
    rep.p("")
    rep.p(
        "  Per-hop n_eff -> log(cosine+eps) slope (cluster-bootstrap CI, eps=%g):"
        % eps_primary
    )
    rep.table(pd.DataFrame(srows), "s4_log_slope_by_hop")

    # ---- principled alternatives: GLMs with log link ---------------------
    _section4_glm(rep, df, cg, hops, cosine, nonref, label_hop)


def _section4_glm(rep, df, cg, hops, cosine, nonref, label_hop):
    rep.p("")
    rep.p("  Principled alternatives (no arbitrary offset), log link:")
    cols = {"Intercept": np.ones(len(df)), "n_eff": cg}
    for lvl in nonref:
        cols["hops[%s]" % label_hop(lvl)] = (hops == lvl).astype(float)
    for lvl in nonref:
        cols["n_eff:hops[%s]" % label_hop(lvl)] = cg * (hops == lvl).astype(float)
    X = pd.DataFrame(cols)

    glm_rows = []
    # quasi-Poisson (Poisson family, log link) handles cosine==0 (y>=0). This is
    # the primary principled fit because cosine has exact zeros.
    try:
        pois = sm.GLM(
            cosine, X, family=sm.families.Poisson(sm.families.links.Log())
        ).fit()
        disp = float(pois.pearson_chi2 / pois.df_resid)  # quasi-Poisson dispersion
        for lvl in nonref:
            t = "n_eff:hops[%s]" % label_hop(lvl)
            glm_rows.append(
                {
                    "model": "quasiPoisson(log)",
                    "term": t,
                    "coef": float(pois.params[t]),
                    "p": float(pois.pvalues[t]),
                }
            )
        rep.p(
            "    quasi-Poisson(log) on raw cosine: fit on ALL %d pairs "
            "(handles cosine==0); dispersion=%.3f" % (len(df), disp)
        )
    except Exception as e:
        rep.p("    quasi-Poisson(log) failed: %s" % e)

    # Gamma needs y>0, so it necessarily DROPS exact-zero-cosine pairs; report
    # how many and treat it as the secondary check.
    pos = cosine > 0
    n_drop = int((~pos).sum())
    try:
        Xg = X[pos]
        gam = sm.GLM(
            cosine[pos], Xg, family=sm.families.Gamma(sm.families.links.Log())
        ).fit()
        for lvl in nonref:
            t = "n_eff:hops[%s]" % label_hop(lvl)
            glm_rows.append(
                {
                    "model": "Gamma(log)",
                    "term": t,
                    "coef": float(gam.params[t]),
                    "p": float(gam.pvalues[t]),
                }
            )
        rep.p(
            "    Gamma(log) on cosine>0: DROPS %d zero-cosine pairs (%.1f%%) -- "
            "biases toward the similar pairs." % (n_drop, 100 * n_drop / len(df))
        )
    except Exception as e:
        rep.p("    Gamma(log) failed: %s" % e)

    if glm_rows:
        g = pd.DataFrame(glm_rows)
        rep.table(g, "s4_glm_interaction", floatfmt="%.6f")
        # agreement check across the two GLM families
        piv = g.pivot_table(index="term", columns="model", values="coef")
        if piv.shape[1] == 2:
            disagree = [
                t
                for t in piv.index
                if np.sign(piv.iloc[:, 0][t]) != np.sign(piv.iloc[:, 1][t])
            ]
            if disagree:
                rep.p(
                    "    -> GLMs DISAGREE in sign on: %s (reporting both)."
                    % ", ".join(disagree)
                )
            else:
                rep.p("    -> both GLM families agree in interaction sign.")


# ===========================================================================
# §5  Stratum gaps, with intervals (CI on the CONTRAST)
# ===========================================================================
def section5(rep, df, levels, bt, sentinel):
    rep.h("5. STRATUM GAPS / RATIOS, WITH CIs ON THE CONTRAST")
    rep.p("  Each interval is a single bootstrapped scalar (resample users,")
    rep.p("  recompute BOTH means, take the contrast) -- NOT a difference of two")
    rep.p("  independently-bootstrapped CIs. Single most load-bearing network number.")

    def label_hop(lvl):
        return "far" if lvl >= sentinel else str(lvl)

    rows = []
    for lvl in levels:
        ge, gl, gh = _ci(bt["gap|%d" % lvl])
        re_, rl, rh = _ci(bt["ratio|%d" % lvl])
        le, ll, lh = _ci(bt["logdiff|%d" % lvl])
        rows.append(
            {
                "hops": label_hop(lvl),
                "gap_>3_minus_[1,3]": ge,
                "gap_ci_low": gl,
                "gap_ci_high": gh,
                "ratio_[1,3]/0": re_,
                "ratio_ci_low": rl,
                "ratio_ci_high": rh,
                "log_ratio_[1,3]/0": le,
                "log_ci_low": ll,
                "log_ci_high": lh,
            }
        )
    tbl = pd.DataFrame(rows)
    rep.table(tbl, "s5_stratum_gaps", floatfmt="%.5f")

    # Test the draft's two claims against the intervals.
    gaps = tbl["gap_>3_minus_[1,3]"].to_numpy()
    rep.p("")
    rep.p("  Draft claim A: additive gap ~0.027 and CONSTANT across hops.")
    rep.p("    gaps by hop: %s" % np.array2string(gaps, precision=4))
    # constant? overlapping CIs across hops is a weak check; report spread.
    spread = float(np.nanmax(gaps) - np.nanmin(gaps))
    rep.p(
        "    max-min gap spread = %.4f (mean gap = %.4f)"
        % (spread, float(np.nanmean(gaps)))
    )
    if abs(float(np.nanmean(gaps)) - 0.027) > 0.010:
        rep.contradict(
            "§5",
            "mean additive gap = %.4f, draft says ~0.027." % float(np.nanmean(gaps)),
        )

    ratios = tbl["ratio_[1,3]/0"].to_numpy()
    rep.p("  Draft claim B: ratio mean([1,3])/mean(0) climbs monotonically with hops.")
    rep.p("    ratios by hop: %s" % np.array2string(ratios, precision=3))
    mono = bool(np.all(np.diff(ratios[np.isfinite(ratios)]) > 0))
    rep.p("    strictly increasing across hops: %s" % mono)
    if not mono:
        rep.contradict(
            "§5",
            "ratio mean([1,3])/mean(0) is NOT strictly monotonic "
            "across hops: %s" % np.array2string(ratios, precision=3),
        )


# ===========================================================================
# §6  Person-like Starter Packs (surface, don't judge)
# ===========================================================================
def _load_pack_emb(emb_cache):
    """Read the cached pack-embedding npz (pack_ids, embeddings) -> (E, row_of).
    Returns (None, None) if unavailable so §6 degrades to frequency-only."""
    if not emb_cache or not Path(emb_cache).exists():
        return None, None
    data = np.load(emb_cache, allow_pickle=True)
    ids = data["pack_ids"].astype(int)
    E = data["embeddings"].astype(np.float32)
    # rows are already L2-normalized by embed_packs; defensively renormalize.
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E = E / norms
    return E, {int(p): i for i, p in enumerate(ids)}


def section6(rep, df, cg_col, packs_path, emb_cache, top_n=200):
    rep.h("6. PERSON-LIKE STARTER PACKS (surface a list; a human decides)")
    if not packs_path or not Path(packs_path).exists():
        rep.p("  SKIPPED: --packs-path not found (%s)." % packs_path)
        return

    ne = df[cg_col].to_numpy(float)
    d_top = df[df["shared_packs"] >= 1].copy()
    ne_pos = d_top[cg_col].to_numpy(float)
    top_thr = np.percentile(ne_pos, 90)
    # median band: a narrow window around the median n_eff for a baseline set.
    med = np.median(ne_pos)
    lo, hi = np.percentile(ne_pos, [45, 55])
    top_pairs = d_top[d_top[cg_col] >= top_thr]
    med_pairs = d_top[(d_top[cg_col] >= lo) & (d_top[cg_col] <= hi)]
    rep.p(
        "  top-decile n_eff threshold: n_eff>=%.4f  (%d pairs)"
        % (top_thr, len(top_pairs))
    )
    rep.p(
        "  median band n_eff in [%.4f, %.4f] (median=%.4f, %d pairs)"
        % (lo, hi, med, len(med_pairs))
    )

    # membership only for the users appearing in the two pair sets we score.
    need = pd.unique(
        pd.concat(
            [
                top_pairs["user_a"],
                top_pairs["user_b"],
                med_pairs["user_a"],
                med_pairs["user_b"],
            ]
        )
    )
    print("[§6] loading starterpacks + membership for %d users ..." % len(need))
    packs, names = load_starterpacks(packs_path)
    membership = build_user_pack_membership(packs, users=set(need.tolist()))

    E, row_of = _load_pack_emb(emb_cache)
    if E is None:
        rep.p("  (no --emb-cache: emitting frequency only, cosine distance = NaN)")

    def pack_name(pid):
        nm = names[pid] if 0 <= pid < len(names) else None
        return nm if nm else "<pack %d>" % pid

    def build_table(pairs, tag):
        from collections import defaultdict

        freq = defaultdict(int)
        dist_sum = defaultdict(float)
        dist_cnt = defaultdict(int)
        for a, b in zip(pairs["user_a"], pairs["user_b"]):
            shared = sorted(
                membership.get(a, frozenset()) & membership.get(b, frozenset())
            )
            if not shared:
                continue
            for pid in shared:
                freq[pid] += 1
            # mean pairwise cosine distance of each shared pack to the OTHERS
            # shared by the SAME pair (requires embeddings).
            if E is not None and len(shared) >= 2:
                rows = [row_of.get(pid) for pid in shared]
                have = [(pid, r) for pid, r in zip(shared, rows) if r is not None]
                if len(have) >= 2:
                    idx = np.array([r for _, r in have])
                    V = E[idx]
                    G = V @ V.T  # cosine sim among shared packs
                    k = len(have)
                    # mean distance of pack i to the other k-1 shared packs
                    mean_sim = (G.sum(axis=1) - 1.0) / (k - 1)
                    for (pid, _), ms in zip(have, mean_sim):
                        dist_sum[pid] += float(1.0 - ms)
                        dist_cnt[pid] += 1
        rows = []
        for pid, f in sorted(freq.items(), key=lambda kv: -kv[1])[:top_n]:
            md = (dist_sum[pid] / dist_cnt[pid]) if dist_cnt[pid] else np.nan
            rows.append(
                {
                    "pack_id": pid,
                    "pack_name": pack_name(pid),
                    "frequency": f,
                    "mean_cosine_distance_to_co_shared": md,
                }
            )
        out = pd.DataFrame(rows)
        rep.table(out, "s6_%s_packs" % tag, floatfmt="%.4f")
        return out

    rep.p("")
    rep.p("  Top-%d most frequent shared packs among TOP-DECILE n_eff pairs:" % top_n)
    build_table(top_pairs, "top_decile")
    rep.p("")
    rep.p(
        "  Top-%d most frequent shared packs among MEDIAN n_eff pairs (baseline):"
        % top_n
    )
    build_table(med_pairs, "median_baseline")
    rep.p("")
    rep.p("  NOTE: no automatic classification -- a human reads the two lists and")
    rep.p("  decides whether the top-decile tail is dominated by person-centric packs.")


# ===========================================================================
# §7  Degree-stratified null (OPTIONAL, behind --run-degree-null)
# ===========================================================================
def section7(
    rep, df, cg_col, packs_path, null_curves_path, seed, sentinel, n_deciles=10
):
    rep.h("7. DEGREE-STRATIFIED NULL (optional)")
    if not packs_path or not Path(packs_path).exists():
        rep.p("  SKIPPED: needs --packs-path for pack-degree; not found.")
        return
    rep.p("  Conditions on pack-degree instead of rewiring: for each observed pair")
    rep.p("  at shared_packs=s, draws a matched zero-overlap pair (shared_packs==0)")
    rep.p("  from the SAME (degree_decile_u, degree_decile_v) cell, using the bin-0")
    rep.p("  pairs already present in the pairs table (no re-scoring).")

    # pack-degree per user = number of packs the user belongs to (from membership).
    need = pd.unique(pd.concat([df["user_a"], df["user_b"]]))
    print("[§7] building membership for %d users ..." % len(need))
    packs, _ = load_starterpacks(packs_path)
    membership = build_user_pack_membership(packs, users=set(need.tolist()))
    deg = {u: len(membership.get(u, frozenset())) for u in need}

    a = df["user_a"].map(deg).to_numpy(float)
    b = df["user_b"].map(deg).to_numpy(float)
    s = df["shared_packs"].to_numpy(float)
    cosine = df["cosine"].to_numpy(float)

    # decile edges over the pooled per-user degree of endpoints.
    pooled = np.concatenate([a, b])
    edges = np.unique(np.percentile(pooled, np.linspace(0, 100, n_deciles + 1)))

    def dec(x):
        return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)

    da, db = dec(a), dec(b)
    # symmetric cell key (unordered decile pair)
    cell = np.minimum(da, db) * n_deciles + np.maximum(da, db)

    zero = s == 0
    # pool of bin-0 pairs indexed by cell
    from collections import defaultdict

    pool = defaultdict(list)
    zidx = np.flatnonzero(zero)
    for i in zidx:
        pool[cell[i]].append(i)

    rng = np.random.default_rng(seed)
    rows = []
    obs_idx = np.flatnonzero(~zero)
    matched_cos_by_s = defaultdict(list)
    obs_cos_by_s = defaultdict(list)
    n_unmatched = 0
    for i in obs_idx:
        obs_cos_by_s[int(s[i])].append(cosine[i])
        cand = pool.get(cell[i])
        if cand:
            j = cand[int(rng.integers(len(cand)))]
            matched_cos_by_s[int(s[i])].append(cosine[j])
        else:
            n_unmatched += 1

    for k in sorted(obs_cos_by_s):
        ov = np.array(obs_cos_by_s[k])
        mv = (
            np.array(matched_cos_by_s[k]) if matched_cos_by_s[k] else np.array([np.nan])
        )
        rows.append(
            {
                "shared_packs": k,
                "n_obs": len(ov),
                "obs_mean_cosine": float(np.nanmean(ov)),
                "matched_zero_mean_cosine": float(np.nanmean(mv)),
                "diff_obs_minus_matched": float(np.nanmean(ov) - np.nanmean(mv)),
            }
        )
    tbl = pd.DataFrame(rows)
    rep.table(tbl, "s7_degree_matched_curve", floatfmt="%.5f")
    if n_unmatched:
        rep.p(
            "  NOTE: %d observed pairs had no bin-0 pair in their degree cell "
            "(unmatched, dropped from the matched curve)." % n_unmatched
        )

    # crossing point: smallest s where observed >= matched
    diff = tbl["diff_obs_minus_matched"].to_numpy()
    ss = tbl["shared_packs"].to_numpy()
    cross = None
    for i in range(len(ss)):
        if diff[i] >= 0:
            cross = int(ss[i])
            break
    rep.p("")
    if cross is None:
        rep.p(
            "  Observed stays BELOW the degree-matched zero-overlap curve at every s."
        )
        rep.p("  -> The null crossing does NOT survive degree matching: the crossing")
        rep.p("     is (at least partly) a DEGREE-COMPOSITION ARTIFACT. The paper's")
        rep.p('     "The Null Crossing" section needs substantial rewriting.')
        rep.contradict(
            "§7",
            "null crossing does not survive degree matching -- "
            "likely a degree-composition artifact; revisit 'The Null Crossing'.",
        )
    else:
        rep.p(
            "  Observed crosses ABOVE the degree-matched zero-overlap curve at s=%d."
            % cross
        )
        rep.p("  -> The crossing SURVIVES degree matching: the 'homogeneous core'")
        rep.p("     reading stands (not explained by low-degree niche users).")

    # For context, echo the existing BiCM null crossing if available.
    if null_curves_path and Path(null_curves_path).exists():
        nc = pd.read_parquet(null_curves_path)
        sub = nc[nc["x_kind"] == "count"][["bin", "observed_mean", "null_mean"]]
        bcross = None
        for _, r in sub.sort_values("bin").iterrows():
            if r["observed_mean"] >= r["null_mean"]:
                bcross = int(r["bin"])
                break
        rep.p(
            "  (BiCM null, from null_curves): observed>=null first at count bin %s."
            % (bcross if bcross is not None else "never")
        )


# ===========================================================================
# CLI
# ===========================================================================
@click.command()
@click.option(
    "--pairs-path",
    default="/scratch/xee6vz/study_data/directed/"
    "pairs_with_hops_n_eff_seed16.parquet",
    show_default=True,
    help="Cached pairs_with_hops (user_a,user_b,cosine,shared_packs,n_eff*,hops).",
)
@click.option(
    "--null-curves",
    default="output/run_20260709_053144_directed/"
    "null_curves/null_curves_cluster_average_seed16.parquet",
    show_default=True,
    help="Existing BiCM null curves (for §7 context).",
)
@click.option(
    "--packs-path",
    default="/scratch/xee6vz/bluesky-graph/starterpacks.jsonl",
    show_default=True,
    help="Starterpack table (pack_id via line index, name, description).",
)
@click.option(
    "--emb-cache",
    default=None,
    help="Cached pack-embedding npz (pack_ids, embeddings) for §6 distances.",
)
@click.option(
    "--incidence",
    default=None,
    help="(Optional) precomputed incidence; unused -- §7 derives degree "
    "from --packs-path membership. Accepted for interface parity.",
)
@click.option("--out", "out_dir", default="output/paper_numbers/", show_default=True)
@click.option("--seed", default=16, show_default=True, type=int)
@click.option("--n-boot", default=500, show_default=True, type=int)
@click.option("--cg-col", default="n_eff", show_default=True)
@click.option(
    "--max-hops",
    default=3,
    show_default=True,
    type=int,
    help="Pipeline MAX_HOPS (sentinel = max_hops+1). This run used 3.",
)
@click.option(
    "--eps",
    default=1e-3,
    show_default=True,
    type=float,
    help="Primary offset for log(cosine+eps) in §4/§5.",
)
@click.option("--expect-rows", default=212514, show_default=True, type=int)
@click.option(
    "--expect-sentinel",
    default=11346,
    show_default=True,
    type=int,
    help="Expected count at hops==sentinel (task sanity check).",
)
@click.option(
    "--run-degree-null",
    is_flag=True,
    default=False,
    help="Run §7 (expensive; off by default).",
)
def main(
    pairs_path,
    null_curves,
    packs_path,
    emb_cache,
    incidence,
    out_dir,
    seed,
    n_boot,
    cg_col,
    max_hops,
    eps,
    expect_rows,
    expect_sentinel,
    run_degree_null,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sentinel = max_hops + 1
    eps_set = [1e-2, 1e-3, 1e-4]

    print("reading pairs: %s" % pairs_path)
    df = pd.read_parquet(pairs_path)
    for c in ("user_a", "user_b", "cosine", "shared_packs", cg_col, "hops"):
        if c not in df.columns:
            raise SystemExit(
                "pairs table missing required column '%s' (have %s)"
                % (c, list(df.columns))
            )

    if not _HAVE_SM:
        raise SystemExit("statsmodels is required (§4). It is already a dependency.")

    # ---- sanity checks: fail loudly -------------------------------------
    sanity_checks(df, cg_col, max_hops, expect_rows, expect_sentinel)

    rep = Report(out)
    rep.p("PAPER NUMBERS  --  read-only report from cached pipeline artifacts")
    rep.p("pairs: %s" % pairs_path)
    rep.p(
        "rows=%d  seed=%d  n_boot=%d  cg_col=%s  max_hops=%d (sentinel=%d)  eps=%g"
        % (len(df), seed, n_boot, cg_col, max_hops, sentinel, eps)
    )

    hops = df["hops"].to_numpy(int)
    levels = sorted(np.unique(hops).tolist())  # ints, sentinel last

    # ---- one shared cluster-bootstrap pass for every contrast CI --------
    stat_fn, meta = build_stat_fn(df, cg_col, levels, eps)
    print("running the shared user-level cluster bootstrap (%d reps) ..." % n_boot)
    boot = node_cluster_bootstrap(df, stat_fn, n_boot=n_boot, seed=seed)
    bt = _boot_lookup(boot)

    section1(rep, df, bt)
    section2(rep, df, cg_col)
    section3(rep, df, cg_col, levels, meta, bt, sentinel)
    section4(rep, df, cg_col, levels, bt, eps, eps_set, sentinel)
    section5(rep, df, levels, bt, sentinel)
    section6(rep, df, cg_col, packs_path, emb_cache)

    if run_degree_null:
        section7(rep, df, cg_col, packs_path, null_curves, seed, sentinel)
    else:
        rep.h("7. DEGREE-STRATIFIED NULL (optional)")
        rep.p("  NOT RUN. Re-run with --run-degree-null to compute it.")

    # ---- provenance / reuse notes ---------------------------------------
    rep.h("NOTE ON REUSE (awkward imports)")
    rep.p("  * run_network_analysis._wls_beta is private (leading underscore) but")
    rep.p("    imported directly per the task's constraint to reuse, not copy or")
    rep.p("    refactor existing code. If it should be public, promote it there.")
    rep.p("  * node_cluster_bootstrap and cluster_bootstrap_ci are the SAME dyadic")
    rep.p(
        "    resampler; every contrast CI here comes from one pass so --seed=%d" % seed
    )
    rep.p("    reproduces all tables exactly.")
    rep.p("  * §7 derives pack-degree from build_user_pack_membership rather than a")
    rep.p("    separate --incidence artifact (accepted but unused), to stay")
    rep.p("    self-contained; matched zero-overlap pairs are drawn from the bin-0")
    rep.p("    pairs ALREADY in the table -- no cosine is recomputed.")

    rep.flush()
    print("done. see %s/paper_numbers.txt" % out)


if __name__ == "__main__":
    main()
