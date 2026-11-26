"""Inferential engine validated against scipy and textbook / G*Power reference values."""
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analytics import inferential as inf
from analytics import power as pw

rng = np.random.default_rng(7)
A = rng.normal(10, 2, 30)
B = rng.normal(11.5, 3, 25)
C = rng.normal(12, 4, 35)


def test_fmt_p_apa():
    assert inf.fmt_p(0.0004) == "p < .001" and inf.fmt_p(0.0234) == "p = .023"


def test_p_adjust_reference():
    p = [0.01, 0.04, 0.03, 0.005]
    assert np.allclose(inf.p_adjust(p, "bonferroni"), [0.04, 0.16, 0.12, 0.02])
    assert np.allclose(inf.p_adjust(p, "holm"), [0.03, 0.06, 0.06, 0.02])
    assert np.allclose(inf.p_adjust(p, "fdr_bh"), stats.false_discovery_control(p))


def test_t_tests_match_scipy():
    r = inf.t_independent(A, B, equal_var=False)
    ref = stats.ttest_ind(A, B, equal_var=False)
    assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.p, ref.pvalue)
    assert np.isclose(r.df, ref.df)
    ci = ref.confidence_interval()
    assert np.allclose(r.ci, (ci.low, ci.high))
    rs = inf.t_independent(A, B, equal_var=True)
    ci = stats.ttest_ind(A, B).confidence_interval()
    assert np.allclose(rs.ci, (ci.low, ci.high))
    one = inf.one_sample_t(A, 10)
    assert np.isclose(one.p, stats.ttest_1samp(A, 10).pvalue)


def test_paired_and_nonparametric():
    x, y = A[:25], B
    r = inf.t_paired(x, y)
    assert np.isclose(r.p, stats.ttest_rel(x, y).pvalue)
    w = inf.wilcoxon_paired(x, y)
    assert np.isclose(w.p, stats.wilcoxon(x, y).pvalue)
    u = inf.mann_whitney(A, B)
    assert np.isclose(u.p, stats.mannwhitneyu(A, B).pvalue) and -1 <= u.effect <= 1


def test_welch_anova_equals_welch_t_for_two_groups():
    r = inf.welch_anova([A, B])
    t = stats.ttest_ind(A, B, equal_var=False)
    assert np.isclose(r.statistic, t.statistic ** 2)
    assert np.isclose(r.df[1], t.df) and np.isclose(r.p, t.pvalue)


def test_games_howell_two_groups_equals_welch():
    gh = inf.posthoc_games_howell([A, B], ["a", "b"])
    assert np.isclose(gh["p-adj"].iloc[0], stats.ttest_ind(A, B, equal_var=False).pvalue, atol=1e-6)


def test_anova_and_tukey():
    r = inf.anova_oneway([A, B, C])
    ref = stats.f_oneway(A, B, C)
    assert np.isclose(r.statistic, ref.statistic) and np.isclose(r.p, ref.pvalue)
    tk = inf.posthoc_tukey([A, B, C], ["a", "b", "c"])
    assert np.allclose(tk["p-adj"], [stats.tukey_hsd(A, B, C).pvalue[i, j] for i, j in [(0, 1), (0, 2), (1, 2)]])


def test_dunn_matches_reference_structure():
    d = inf.posthoc_dunn([A, B, C], ["a", "b", "c"], "none")
    assert len(d) == 3 and (d["p"].between(0, 1)).all()
    # two groups, no ties: Dunn z² ≈ Kruskal H
    d2 = inf.posthoc_dunn([A, B], ["a", "b"], "none")
    assert np.isclose(d2["z"].iloc[0] ** 2, stats.kruskal(A, B).statistic)


def test_two_way_anova():
    # balanced 2x3 design: Type II == Type III; single factor matches one-way ANOVA
    rows = []
    for a in ["x", "y"]:
        for b in ["p", "q", "r"]:
            for v in rng.normal({"x": 0, "y": 1}[a] + {"p": 0, "q": 0.5, "r": 1}[b], 1, 8):
                rows.append((a, b, v))
    df = pd.DataFrame(rows, columns=["A", "B", "y"])
    t3 = inf.anova_twoway(df, "y", "A", "B", ss_type=3)
    t2 = inf.anova_twoway(df, "y", "A", "B", ss_type=2)
    assert np.allclose(t3["SS"], t2["SS"])
    assert list(t3["df"]) == [1, 2, 2, 42]
    # SS components add up to total in a balanced design
    assert np.isclose(t3["SS"].sum(), ((df.y - df.y.mean()) ** 2).sum())
    # Sanity: main effect of B equals one-way ANOVA SS between
    ssb = sum(len(g) * (g.mean() - df.y.mean()) ** 2 for _, g in df.groupby("B")["y"])
    assert np.isclose(t3.loc[1, "SS"], ssb)


def test_rm_anova_and_friedman():
    base = rng.normal(0, 1, 20)
    data = pd.DataFrame({"t1": base, "t2": base + 0.5 + rng.normal(0, .3, 20), "t3": base + 1 + rng.normal(0, .3, 20)})
    r = inf.rm_anova(data)
    assert r.p < 0.001 and 0.5 <= r.extra["Greenhouse-Geisser ε"] <= 1
    f = inf.friedman(data)
    assert np.isclose(f.p, stats.friedmanchisquare(data.t1, data.t2, data.t3).pvalue)


def test_chi2_and_correlations():
    df = pd.DataFrame({"a": rng.choice(["m", "f"], 200), "b": rng.choice(["yes", "no"], 200)})
    res, ct, resid = inf.chi2_independence(df, "a", "b")
    assert "Fisher exact p" in res.extra and resid.shape == (2, 2)
    x = rng.normal(size=60); y = x * 0.5 + rng.normal(size=60)
    c = inf.correlation_test(x, y)
    assert np.isclose(c.statistic, stats.pearsonr(x, y)[0])
    lo, hi = stats.pearsonr(x, y).confidence_interval()
    assert np.allclose(c.ci, (lo, hi))
    z = rng.normal(size=60)
    pc = inf.partial_correlation(pd.DataFrame({"x": x, "y": y, "z": z}), "x", "y", ["z"])
    assert -1 <= pc.statistic <= 1 and pc.df == 57


def test_proportions_and_mcnemar():
    r = inf.one_proportion(60, 100, 0.5)
    assert np.isclose(r.p, stats.binomtest(60, 100, 0.5).pvalue)
    lo, hi = inf.proportion_ci(60, 100)
    assert np.isclose(lo, 0.502, atol=1e-3) and np.isclose(hi, 0.6906, atol=1e-3)  # Wilson reference
    m = inf.mcnemar(["y"] * 30 + ["n"] * 10 + ["y"] * 5, ["y"] * 30 + ["y"] * 10 + ["n"] * 5)
    assert m.extra["Discordant b"] + m.extra["Discordant c"] == 15


def test_compact_letters():
    names = ["a", "b", "c"]
    means = [A.mean(), B.mean(), C.mean()]
    tk = inf.posthoc_tukey([A, B, C], names)
    letters = inf.compact_letters(tk, names, means)
    for _, r in tk.iterrows():
        share = set(letters[r["Group 1"]]) & set(letters[r["Group 2"]])
        assert bool(share) == (r["p-adj"] >= 0.05)


def test_power_against_gpower():
    # G*Power: d = 0.5, α = .05 two-tailed, n = 64 per group → power ≈ 0.8015
    assert np.isclose(pw.power_t(0.5, 64), 0.8015, atol=1e-3)
    assert pw.solve_n(lambda n: pw.power_t(0.5, n), 0.80) == 64
    # paired / one-sample: d = 0.5 → n = 34 for 80%
    assert pw.solve_n(lambda n: pw.power_t(0.5, n, kind="one"), 0.80) == 34
    # ANOVA f = 0.25, k = 3 → 53 per group (G*Power total 159)
    assert pw.solve_n(lambda n: pw.power_anova(0.25, 3, n), 0.80) == 53
    # correlation r = .3 → n ≈ 85
    assert pw.solve_n(lambda n: pw.power_correlation(0.3, n), 0.80) in (84, 85)


def test_survey_sample_size():
    assert pw.cochran_sample_size()["Recommended n"] == 385
    assert pw.yamane_sample_size(1000, 0.05) == 286
    assert pw.cochran_sample_size(population=1000)["Recommended n"] == 278


def test_bootstrap_and_advisor():
    est, lo, hi, se = inf.bootstrap_ci(A, "median", n_resamples=2000)
    assert lo < est < hi
    assert inf.recommend("compare", 2, False, "numeric", True, False)[0] == "t_welch"
    assert inf.recommend("compare", 4, False, "numeric", False, True)[0] == "kruskal"
    assert inf.recommend("compare", 3, True, "numeric", True, True)[0] == "rm_anova"
    assert inf.recommend("relationship", outcome="numeric", normal=False)[0] == "spearman"


def test_q_critical_interpolation_accuracy():
    for k, df in [(3, 15), (5, 33.7), (8, 300)]:
        exact = stats.studentized_range.ppf(0.95, k, df)
        assert abs(inf.q_critical(k, df) - exact) < 0.01
