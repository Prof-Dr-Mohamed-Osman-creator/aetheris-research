# ==========================================
# Aetheris Research - Inferential Statistics Engine
# ------------------------------------------
# Hypothesis tests, post-hoc comparisons, effect sizes, confidence intervals
# and APA-style reporting. Pure functions (numpy / scipy only) so every result
# can be unit-tested against reference values.
# ==========================================

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


# ==========================================================
# Result container + reporting
# ==========================================================
@dataclass
class TestResult:
    test: str                      # English name
    test_ar: str                   # Arabic name
    stat_label: str                # t, F, U, W, H, χ², z …
    statistic: float
    p: float
    df: tuple | float | None = None
    effect_name: str | None = None
    effect: float | None = None
    effect_kind: str | None = None  # d, r, eta2, omega2, w, V, W, or
    ci: tuple | None = None         # CI of the main estimate (e.g. mean difference)
    ci_label: str | None = None
    estimate: float | None = None
    n: int | None = None
    alternative: str = "two-sided"
    extra: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    # -------------------------------------------------
    def magnitude(self, lang: str = "en") -> str:
        return effect_magnitude(self.effect, self.effect_kind, lang, self.extra.get("dof_v"))

    def df_text(self) -> str:
        if self.df is None:
            return ""
        if isinstance(self.df, tuple):
            return "(" + ", ".join(_num(d) for d in self.df) + ")"
        return f"({_num(self.df)})"

    def apa(self) -> str:
        """APA 7 style result string, e.g.  t(48) = 2.31, p = .025, d = 0.65, 95% CI [0.1, 1.2]"""
        s = f"{self.stat_label}{self.df_text()} = {self.statistic:.2f}, {fmt_p(self.p)}"
        if self.effect is not None and not pd.isna(self.effect):
            s += f", {self.effect_name} = {self.effect:.2f}"
        if self.ci is not None and not any(pd.isna(c) for c in self.ci):
            s += f", 95% CI [{self.ci[0]:.3g}, {self.ci[1]:.3g}]"
        return s

    def to_frame(self) -> pd.DataFrame:
        rows = [("Test", self.test), ("Statistic", f"{self.stat_label} = {self.statistic:.4f}"),
                ("df", self.df_text().strip("()") or "—"), ("p-value", f"{self.p:.4g}"),
                ("Alternative", self.alternative)]
        if self.estimate is not None:
            rows.append(("Estimate", f"{self.estimate:.4g}"))
        if self.ci is not None:
            rows.append((self.ci_label or "95% CI", f"[{self.ci[0]:.4g}, {self.ci[1]:.4g}]"))
        if self.effect is not None:
            rows.append((f"Effect size ({self.effect_name})", f"{self.effect:.4f} — {self.magnitude()}"))
        if self.n is not None:
            rows.append(("N", str(self.n)))
        for k, v in self.extra.items():
            if k != "dof_v":
                rows.append((k, f"{v:.4g}" if isinstance(v, (float, np.floating)) else str(v)))
        return pd.DataFrame(rows, columns=["Item", "Value"])

    def interpret(self, alpha: float = 0.05, lang: str = "en", context: str = "") -> str:
        sig = self.p < alpha
        mag = self.magnitude(lang)
        if lang == "ar":
            txt = (f"<b>{self.test_ar}</b>: {self.apa()}. "
                   + (f"النتيجة <b>دالة إحصائياً</b> عند مستوى معنوية {alpha}، فنرفض فرض العدم"
                      if sig else f"النتيجة <b>غير دالة إحصائياً</b> عند مستوى معنوية {alpha}، فلا نرفض فرض العدم"))
            if context:
                txt += f" ({context})"
            txt += "."
            if self.effect is not None and not pd.isna(self.effect):
                txt += f" حجم الأثر ({self.effect_name}) = {self.effect:.3f}، وهو أثر {mag}."
                if sig and mag in ("ضئيل", "صغير"):
                    txt += " الدلالة الإحصائية مع أثر صغير تعني أن الفرق حقيقي لكن أهميته العملية محدودة."
                if not sig and mag in ("متوسط", "كبير"):
                    txt += " الأثر ليس صغيراً رغم عدم الدلالة، وقد يكون حجم العينة غير كافٍ (راجع تبويب القوة وحجم العينة)."
        else:
            txt = (f"<b>{self.test}</b>: {self.apa()}. "
                   + (f"The result is <b>statistically significant</b> at α = {alpha}; the null hypothesis is rejected"
                      if sig else f"The result is <b>not significant</b> at α = {alpha}; the null hypothesis is not rejected"))
            if context:
                txt += f" ({context})"
            txt += "."
            if self.effect is not None and not pd.isna(self.effect):
                txt += f" Effect size ({self.effect_name}) = {self.effect:.3f}, a {mag} effect."
                if sig and mag in ("negligible", "small"):
                    txt += " Significant but small: the difference is real yet of limited practical importance."
                if not sig and mag in ("medium", "large"):
                    txt += " The effect is not trivial despite non-significance — the sample may be underpowered."
        for note in self.notes:
            txt += f"<br>⚠️ {note}"
        return txt


def _num(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"


def fmt_p(p: float) -> str:
    """APA: no leading zero, exact to 3 decimals, 'p < .001' below that."""
    if p is None or pd.isna(p):
        return "p = —"
    if p < 0.001:
        return "p < .001"
    return "p = " + f"{p:.3f}".lstrip("0")


def significance_marker(p: float) -> str:
    if p is None or pd.isna(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"


_THRESH = {  # Cohen (1988) conventions
    "d": (0.2, 0.5, 0.8), "r": (0.1, 0.3, 0.5), "eta2": (0.01, 0.06, 0.14), "omega2": (0.01, 0.06, 0.14),
    "w": (0.1, 0.3, 0.5), "W": (0.1, 0.3, 0.5), "f": (0.1, 0.25, 0.4), "h": (0.2, 0.5, 0.8),
}


def effect_magnitude(value, kind, lang="en", dof_v=None) -> str:
    if value is None or kind is None or pd.isna(value):
        return ""
    v = abs(value)
    if kind == "V":  # Cramér's V depends on min(r,c)-1 (Cohen 1988)
        k = max(1, dof_v or 1)
        thr = tuple(t / np.sqrt(k) for t in (0.1, 0.3, 0.5))
    elif kind == "or":
        v = max(value, 1 / value) if value > 0 else np.inf
        thr = (1.44, 2.48, 4.27)  # Chen, Cohen & Chen (2010)
    else:
        thr = _THRESH.get(kind, (0.2, 0.5, 0.8))
    labels = ["negligible", "small", "medium", "large"] if lang == "en" else ["ضئيل", "صغير", "متوسط", "كبير"]
    return labels[int(np.searchsorted(thr, v, side="right"))]


def p_adjust(pvals, method: str = "holm") -> np.ndarray:
    """Multiple-comparison adjustment: none, bonferroni, holm, fdr_bh (Benjamini-Hochberg)."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if method in (None, "none") or m == 0:
        return p
    if method == "bonferroni":
        return np.minimum(p * m, 1)
    order = np.argsort(p)
    if method == "holm":
        adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    elif method == "fdr_bh":
        adj = np.minimum.accumulate((m / np.arange(m, 0, -1)) * p[order][::-1])[::-1]
    else:
        raise ValueError(method)
    out = np.empty(m)
    out[order] = np.minimum(adj, 1)
    return out


ADJUST_METHODS = {"holm": ("Holm", "هولم"), "bonferroni": ("Bonferroni", "بونفيروني"),
                  "fdr_bh": ("Benjamini-Hochberg (FDR)", "بنجاميني-هوشبرغ (FDR)"), "none": ("None", "بدون")}


def _clean(x) -> np.ndarray:
    return pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(dtype=float)


def _alt_label(alternative: str) -> str:
    return {"two-sided": "two-sided", "greater": "one-sided (greater)", "less": "one-sided (less)"}[alternative]


# ==========================================================
# Assumption checks
# ==========================================================
def check_assumptions(groups: list, alpha: float = 0.05) -> dict:
    """Normality (Shapiro-Wilk per group) + homogeneity of variance (Levene, Brown-Forsythe)."""
    norm = []
    for g in groups:
        g = _clean(g)
        if 3 <= g.size <= 5000 and np.ptp(g) > 0:
            norm.append(stats.shapiro(g).pvalue)
        else:
            norm.append(np.nan)
    ok_norm = all((p > alpha) or pd.isna(p) for p in norm)
    # Central limit theorem relief for large groups
    large = all(_clean(g).size >= 30 for g in groups)
    lev = bf = np.nan
    if len(groups) >= 2 and all(_clean(g).size >= 2 for g in groups):
        lev = stats.levene(*[_clean(g) for g in groups], center="mean").pvalue
        bf = stats.levene(*[_clean(g) for g in groups], center="median").pvalue
    return {"normality_p": norm, "normal": ok_norm, "large_samples": large,
            "levene_p": lev, "brown_forsythe_p": bf,
            "equal_var": (bf > alpha) if not pd.isna(bf) else True}


# ==========================================================
# One-sample tests
# ==========================================================
def one_sample_t(x, mu: float = 0.0, alternative: str = "two-sided", conf: float = 0.95) -> TestResult:
    x = _clean(x)
    n = x.size
    res = stats.ttest_1samp(x, mu, alternative=alternative)
    se = x.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
    d = (x.mean() - mu) / x.std(ddof=1)
    return TestResult("One-sample t-test", "اختبار t لعينة واحدة", "t", res.statistic, res.pvalue, n - 1,
                      "Cohen's d", d, "d", (x.mean() - tcrit * se, x.mean() + tcrit * se),
                      f"{conf:.0%} CI of mean", x.mean(), n, _alt_label(alternative),
                      {"Test value (μ₀)": mu, "Mean": x.mean(), "SD": x.std(ddof=1)})


def _rank_biserial_signed(diff: np.ndarray) -> float:
    d = diff[diff != 0]
    if d.size == 0:
        return np.nan
    ranks = stats.rankdata(np.abs(d))
    w_pos, w_neg = ranks[d > 0].sum(), ranks[d < 0].sum()
    return (w_pos - w_neg) / (w_pos + w_neg)


def wilcoxon_one_sample(x, mu: float = 0.0, alternative: str = "two-sided") -> TestResult:
    x = _clean(x)
    diff = x - mu
    res = stats.wilcoxon(diff, alternative=alternative, zero_method="wilcox")
    return TestResult("Wilcoxon signed-rank test (one sample)", "اختبار ويلكوكسون للرتب ذات الإشارة (عينة واحدة)",
                      "W", res.statistic, res.pvalue, None, "rank-biserial r", _rank_biserial_signed(diff), "r",
                      None, None, float(np.median(x)), x.size, _alt_label(alternative),
                      {"Test value (median₀)": mu, "Sample median": float(np.median(x))})


def sign_test(x, mu: float = 0.0, alternative: str = "two-sided") -> TestResult:
    x = _clean(x)
    pos, neg = int((x > mu).sum()), int((x < mu).sum())
    res = stats.binomtest(pos, pos + neg, 0.5, alternative=alternative)
    return TestResult("Sign test", "اختبار الإشارة", "S", pos, res.pvalue, None, None, None, None,
                      None, None, float(np.median(x)), x.size, _alt_label(alternative),
                      {"Above μ₀": pos, "Below μ₀": neg, "Ties": int(x.size - pos - neg)})


def proportion_ci(k: int, n: int, conf: float = 0.95, method: str = "wilson") -> tuple:
    if n == 0:
        return (np.nan, np.nan)
    if method == "exact":
        ci = stats.binomtest(k, n).proportion_ci(confidence_level=conf, method="exact")
        return (ci.low, ci.high)
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    den = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (centre - half, centre + half)


def one_proportion(k: int, n: int, p0: float = 0.5, alternative: str = "two-sided") -> TestResult:
    exact = stats.binomtest(k, n, p0, alternative=alternative)
    p_hat = k / n
    z = (p_hat - p0) / np.sqrt(p0 * (1 - p0) / n)
    h = 2 * np.arcsin(np.sqrt(p_hat)) - 2 * np.arcsin(np.sqrt(p0))
    notes = [] if n * p0 >= 5 and n * (1 - p0) >= 5 else ["n·p₀ < 5: rely on the exact binomial p-value."]
    return TestResult("Binomial test for one proportion", "اختبار ذي الحدين لنسبة واحدة", "z", z, exact.pvalue,
                      None, "Cohen's h", h, "h", proportion_ci(k, n), "95% CI (Wilson)", p_hat, n,
                      _alt_label(alternative), {"Successes": k, "Hypothesised p₀": p0}, notes)


def chi2_goodness_of_fit(observed, expected_props=None, labels=None) -> TestResult:
    obs = np.asarray(observed, dtype=float)
    k = obs.size
    props = np.full(k, 1 / k) if expected_props is None else np.asarray(expected_props, float) / np.sum(expected_props)
    exp = props * obs.sum()
    chi2, p = stats.chisquare(obs, exp)
    w = np.sqrt(chi2 / obs.sum())
    notes = [] if (exp >= 5).all() else ["Some expected counts are < 5; merge categories."]
    return TestResult("χ² goodness-of-fit test", "اختبار كاي² لجودة التوفيق", "χ²", chi2, p, k - 1,
                      "Cohen's w", w, "w", None, None, None, int(obs.sum()), "two-sided",
                      {"Categories": k}, notes)


def variance_test_one(x, sigma2_0: float, alternative: str = "two-sided") -> TestResult:
    x = _clean(x)
    n = x.size
    s2 = x.var(ddof=1)
    chi2 = (n - 1) * s2 / sigma2_0
    cdf = stats.chi2.cdf(chi2, n - 1)
    p = {"two-sided": 2 * min(cdf, 1 - cdf), "greater": 1 - cdf, "less": cdf}[alternative]
    ci = ((n - 1) * s2 / stats.chi2.ppf(0.975, n - 1), (n - 1) * s2 / stats.chi2.ppf(0.025, n - 1))
    return TestResult("χ² test for one variance", "اختبار كاي² لتباين مجتمع واحد", "χ²", chi2, p, n - 1,
                      None, None, None, ci, "95% CI of variance", s2, n, _alt_label(alternative),
                      {"Hypothesised σ²": sigma2_0})


# ==========================================================
# Two independent groups
# ==========================================================
def _cohen_d(x, y) -> float:
    nx, ny = x.size, y.size
    sp = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp


def hedges_g(x, y) -> float:
    nx, ny = x.size, y.size
    return _cohen_d(x, y) * (1 - 3 / (4 * (nx + ny) - 9))


def t_independent(x, y, equal_var: bool = False, alternative: str = "two-sided", conf: float = 0.95) -> TestResult:
    x, y = _clean(x), _clean(y)
    nx, ny = x.size, y.size
    res = stats.ttest_ind(x, y, equal_var=equal_var, alternative=alternative)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    if equal_var:
        df = nx + ny - 2
        sp2 = ((nx - 1) * vx + (ny - 1) * vy) / df
        se = np.sqrt(sp2 * (1 / nx + 1 / ny))
    else:
        se = np.sqrt(vx / nx + vy / ny)
        df = (vx / nx + vy / ny) ** 2 / ((vx / nx) ** 2 / (nx - 1) + (vy / ny) ** 2 / (ny - 1))
    diff = x.mean() - y.mean()
    tcrit = stats.t.ppf(1 - (1 - conf) / 2, df)
    name = ("Independent-samples t-test (Student)", "اختبار t لعينتين مستقلتين (ستودنت)") if equal_var else \
           ("Welch's t-test (unequal variances)", "اختبار ويلش t (تباينات غير متساوية)")
    return TestResult(name[0], name[1], "t", res.statistic, res.pvalue, df, "Hedges' g", hedges_g(x, y), "d",
                      (diff - tcrit * se, diff + tcrit * se), f"{conf:.0%} CI of mean difference", diff, nx + ny,
                      _alt_label(alternative),
                      {"Mean 1": x.mean(), "Mean 2": y.mean(), "SD 1": np.sqrt(vx), "SD 2": np.sqrt(vy),
                       "n1": nx, "n2": ny, "Cohen's d": _cohen_d(x, y)})


def mann_whitney(x, y, alternative: str = "two-sided") -> TestResult:
    x, y = _clean(x), _clean(y)
    res = stats.mannwhitneyu(x, y, alternative=alternative, method="auto")
    n1n2 = x.size * y.size
    r_rb = 2 * res.statistic / n1n2 - 1          # rank-biserial (positive: x tends to be larger)
    # normal approximation z for reporting
    mu = n1n2 / 2
    sigma = np.sqrt(n1n2 * (x.size + y.size + 1) / 12)
    z = (res.statistic - mu) / sigma
    return TestResult("Mann-Whitney U test", "اختبار مان-ويتني U", "U", res.statistic, res.pvalue, None,
                      "rank-biserial r", r_rb, "r", None, None, float(np.median(x) - np.median(y)),
                      x.size + y.size, _alt_label(alternative),
                      {"Median 1": float(np.median(x)), "Median 2": float(np.median(y)), "z (approx.)": z,
                       "Prob. superiority P(X>Y)": res.statistic / n1n2})


def ks_two_sample(x, y) -> TestResult:
    x, y = _clean(x), _clean(y)
    res = stats.ks_2samp(x, y)
    return TestResult("Kolmogorov-Smirnov two-sample test", "اختبار كولموجوروف-سميرنوف لعينتين", "D",
                      res.statistic, res.pvalue, None, None, None, None, n=x.size + y.size,
                      extra={"Interpretation": "tests any difference in distribution (location, spread, shape)"})


def variance_tests(groups: list, names: list | None = None) -> pd.DataFrame:
    """F (2 groups), Levene, Brown-Forsythe, Bartlett, Fligner-Killeen."""
    g = [_clean(v) for v in groups]
    rows = []
    if len(g) == 2:
        f = g[0].var(ddof=1) / g[1].var(ddof=1)
        d1, d2 = g[0].size - 1, g[1].size - 1
        p = 2 * min(stats.f.cdf(f, d1, d2), stats.f.sf(f, d1, d2))
        rows.append(("F-test (ratio of variances)", "اختبار F لنسبة التباينين", f, p, "sensitive to non-normality"))
    rows.append(("Levene (mean)", "ليفين (بالوسط)", *stats.levene(*g, center="mean"), "classic"))
    rows.append(("Brown-Forsythe (median)", "براون-فورسايث (بالوسيط)", *stats.levene(*g, center="median"),
                 "robust — recommended"))
    rows.append(("Bartlett", "بارتليت", *stats.bartlett(*g), "requires normality"))
    rows.append(("Fligner-Killeen", "فلينجر-كيلين", *stats.fligner(*g), "non-parametric"))
    return pd.DataFrame(rows, columns=["Test", "Test (AR)", "Statistic", "p-value", "Note"])


def two_proportions(k1: int, n1: int, k2: int, n2: int, alternative: str = "two-sided") -> TestResult:
    p1, p2 = k1 / n1, k2 / n2
    pp = (k1 + k2) / (n1 + n2)
    se0 = np.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se0
    p = {"two-sided": 2 * stats.norm.sf(abs(z)), "greater": stats.norm.sf(z), "less": stats.norm.cdf(z)}[alternative]
    se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    diff = p1 - p2
    h = 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))
    extra = {"p1": p1, "p2": p2, "Risk ratio": p1 / p2 if p2 else np.nan}
    a, b, c, d = k1, n1 - k1, k2, n2 - k2
    if min(a, b, c, d) > 0:
        orr = a * d / (b * c)
        se_log = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        extra.update({"Odds ratio": orr, "OR 95% CI low": np.exp(np.log(orr) - 1.96 * se_log),
                      "OR 95% CI high": np.exp(np.log(orr) + 1.96 * se_log)})
    return TestResult("Two-proportion z-test", "اختبار z للفرق بين نسبتين", "z", z, p, None, "Cohen's h", h, "h",
                      (diff - 1.96 * se, diff + 1.96 * se), "95% CI of difference", diff, n1 + n2,
                      _alt_label(alternative), extra)


# ==========================================================
# Paired samples
# ==========================================================
def _paired(x, y):
    d = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"),
                      "y": pd.to_numeric(pd.Series(y), errors="coerce")}).dropna()
    return d["x"].to_numpy(float), d["y"].to_numpy(float)


def t_paired(x, y, alternative: str = "two-sided", conf: float = 0.95) -> TestResult:
    x, y = _paired(x, y)
    diff = x - y
    n = diff.size
    res = stats.ttest_rel(x, y, alternative=alternative)
    se = diff.std(ddof=1) / np.sqrt(n)
    tcrit = stats.t.ppf(1 - (1 - conf) / 2, n - 1)
    dz = diff.mean() / diff.std(ddof=1)
    return TestResult("Paired-samples t-test", "اختبار t للعينات المرتبطة (المزدوجة)", "t", res.statistic,
                      res.pvalue, n - 1, "Cohen's dz", dz, "d",
                      (diff.mean() - tcrit * se, diff.mean() + tcrit * se), f"{conf:.0%} CI of mean difference",
                      diff.mean(), n, _alt_label(alternative),
                      {"Mean 1": x.mean(), "Mean 2": y.mean(), "r (pre-post)": stats.pearsonr(x, y)[0]})


def wilcoxon_paired(x, y, alternative: str = "two-sided") -> TestResult:
    x, y = _paired(x, y)
    diff = x - y
    res = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
    return TestResult("Wilcoxon signed-rank test (paired)", "اختبار ويلكوكسون للرتب ذات الإشارة (عينات مرتبطة)",
                      "W", res.statistic, res.pvalue, None, "rank-biserial r", _rank_biserial_signed(diff), "r",
                      None, None, float(np.median(diff)), diff.size, _alt_label(alternative),
                      {"Median difference": float(np.median(diff))})


def mcnemar(x, y) -> TestResult:
    """Paired binary outcomes (e.g. adopted technology before / after)."""
    d = pd.DataFrame({"x": pd.Series(x), "y": pd.Series(y)}).dropna()
    levels = sorted(pd.unique(pd.concat([d["x"], d["y"]])), key=str)
    if len(levels) != 2:
        raise ValueError("McNemar's test needs exactly two categories")
    b = int(((d["x"] == levels[0]) & (d["y"] == levels[1])).sum())
    c = int(((d["x"] == levels[1]) & (d["y"] == levels[0])).sum())
    exact = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    return TestResult("McNemar test (exact)", "اختبار ماكنمار (الدقيق)", "χ²", chi2, exact, 1,
                      "odds ratio (b/c)", b / c if c else np.nan, "or", n=len(d),
                      extra={"Discordant b": b, "Discordant c": c, "Categories": f"{levels[0]} / {levels[1]}"})


# ==========================================================
# k independent groups
# ==========================================================
def anova_oneway(groups: list) -> TestResult:
    g = [_clean(v) for v in groups]
    k, N = len(g), sum(v.size for v in g)
    res = stats.f_oneway(*g)
    grand = np.concatenate(g).mean()
    ssb = sum(v.size * (v.mean() - grand) ** 2 for v in g)
    ssw = sum(((v - v.mean()) ** 2).sum() for v in g)
    sst = ssb + ssw
    msw = ssw / (N - k)
    eta2 = ssb / sst
    omega2 = (ssb - (k - 1) * msw) / (sst + msw)
    return TestResult("One-way ANOVA", "تحليل التباين الأحادي (ANOVA)", "F", res.statistic, res.pvalue,
                      (k - 1, N - k), "η²", eta2, "eta2", n=N,
                      extra={"ω²": max(omega2, 0.0), "SS between": ssb, "SS within": ssw, "MS within": msw,
                             "Cohen's f": np.sqrt(eta2 / (1 - eta2)) if eta2 < 1 else np.nan})


def welch_anova(groups: list) -> TestResult:
    """Welch's heteroscedastic one-way ANOVA (Welch, 1951)."""
    g = [_clean(v) for v in groups]
    k = len(g)
    n = np.array([v.size for v in g], float)
    m = np.array([v.mean() for v in g])
    s2 = np.array([v.var(ddof=1) for v in g])
    w = n / s2
    mw = (w * m).sum() / w.sum()
    a = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = ((1 - w / w.sum()) ** 2 / (n - 1)).sum()
    b = 1 + 2 * (k - 2) / (k ** 2 - 1) * lam
    F = a / b
    df2 = (k ** 2 - 1) / (3 * lam)
    p = stats.f.sf(F, k - 1, df2)
    # adjusted omega² (Carroll & Nordholm) — approximate effect size
    N = n.sum()
    om = (k - 1) * (F - 1) / ((k - 1) * (F - 1) + N)
    return TestResult("Welch's ANOVA (unequal variances)", "تحليل التباين لويلش (تباينات غير متساوية)", "F",
                      F, p, (k - 1, df2), "ω² (est.)", max(om, 0.0), "omega2", n=int(N))


def kruskal_wallis(groups: list) -> TestResult:
    g = [_clean(v) for v in groups]
    N, k = sum(v.size for v in g), len(g)
    res = stats.kruskal(*g)
    eps2 = res.statistic / ((N ** 2 - 1) / (N + 1))
    return TestResult("Kruskal-Wallis H test", "اختبار كروسكال-واليس H", "H", res.statistic, res.pvalue, k - 1,
                      "ε²", eps2, "eta2", n=N,
                      extra={"η²(H)": max((res.statistic - k + 1) / (N - k), 0.0)})


def mood_median(groups: list) -> TestResult:
    g = [_clean(v) for v in groups]
    stat, p, med, table = stats.median_test(*g)
    return TestResult("Mood's median test", "اختبار الوسيط لمود", "χ²", stat, p, len(g) - 1, n=sum(v.size for v in g),
                      extra={"Grand median": med})


# ---------------- post-hoc ----------------
def posthoc_tukey(groups: list, names: list) -> pd.DataFrame:
    g = [_clean(v) for v in groups]
    res = stats.tukey_hsd(*g)
    ci = res.confidence_interval(0.95)
    rows = []
    for i, j in combinations(range(len(g)), 2):
        rows.append({"Group 1": names[i], "Group 2": names[j], "Mean diff": g[i].mean() - g[j].mean(),
                     "95% CI low": ci.low[i, j], "95% CI high": ci.high[i, j], "p-adj": res.pvalue[i, j],
                     "Hedges' g": hedges_g(g[i], g[j])})
    return pd.DataFrame(rows)


@lru_cache(maxsize=64)
def _q_grid(k: int, conf: float = 0.95) -> tuple:
    dfs = (5.0, 10.0, 20.0, 40.0, 120.0, 1e6)
    return dfs, tuple(float(stats.studentized_range.ppf(conf, k, d)) for d in dfs)


def q_critical(k: int, df: float, conf: float = 0.95) -> float:
    """Studentized-range critical value. Exact for df < 5; otherwise interpolated linearly in 1/df
    between cached grid points (classical table interpolation, error < 0.01)."""
    if df < 5:
        return float(stats.studentized_range.ppf(conf, k, df))
    dfs, qs = _q_grid(k, conf)
    inv = 1 / np.asarray(dfs)
    return float(np.interp(1 / df, inv[::-1], np.asarray(qs)[::-1]))


def posthoc_games_howell(groups: list, names: list) -> pd.DataFrame:
    """Games-Howell: no equal-variance assumption (use after Welch's ANOVA)."""
    g = [_clean(v) for v in groups]
    k = len(g)
    rows = []
    for i, j in combinations(range(k), 2):
        a, b = g[i], g[j]
        va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
        diff = a.mean() - b.mean()
        se = np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
        q = abs(diff) / se * np.sqrt(2)
        p = stats.studentized_range.sf(q, k, df)
        qcrit = q_critical(k, df) / np.sqrt(2)
        rows.append({"Group 1": names[i], "Group 2": names[j], "Mean diff": diff,
                     "95% CI low": diff - qcrit * se, "95% CI high": diff + qcrit * se,
                     "p-adj": min(p, 1.0), "Hedges' g": hedges_g(a, b)})
    return pd.DataFrame(rows)


def posthoc_pairwise_t(groups: list, names: list, adjust: str = "holm", equal_var: bool = False) -> pd.DataFrame:
    g = [_clean(v) for v in groups]
    rows = []
    for i, j in combinations(range(len(g)), 2):
        r = stats.ttest_ind(g[i], g[j], equal_var=equal_var)
        rows.append({"Group 1": names[i], "Group 2": names[j], "Mean diff": g[i].mean() - g[j].mean(),
                     "t": r.statistic, "p": r.pvalue, "Hedges' g": hedges_g(g[i], g[j])})
    out = pd.DataFrame(rows)
    out["p-adj"] = p_adjust(out["p"], adjust)
    return out


def posthoc_dunn(groups: list, names: list, adjust: str = "holm") -> pd.DataFrame:
    """Dunn (1964) test after Kruskal-Wallis, with tie correction."""
    g = [_clean(v) for v in groups]
    allv = np.concatenate(g)
    N = allv.size
    ranks = stats.rankdata(allv)
    idx = np.cumsum([0] + [v.size for v in g])
    mean_r = [ranks[idx[i]:idx[i + 1]].mean() for i in range(len(g))]
    _, counts = np.unique(allv, return_counts=True)
    tie = (counts ** 3 - counts).sum() / (12 * (N - 1))
    rows = []
    for i, j in combinations(range(len(g)), 2):
        se = np.sqrt((N * (N + 1) / 12 - tie) * (1 / g[i].size + 1 / g[j].size))
        z = (mean_r[i] - mean_r[j]) / se
        rows.append({"Group 1": names[i], "Group 2": names[j], "Mean rank diff": mean_r[i] - mean_r[j],
                     "z": z, "p": 2 * stats.norm.sf(abs(z))})
    out = pd.DataFrame(rows)
    out["p-adj"] = p_adjust(out["p"], adjust)
    return out


def compact_letters(posthoc: pd.DataFrame, names: list, means: list, alpha: float = 0.05,
                    p_col: str = "p-adj") -> dict:
    """Compact letter display (a, b, ab …) — the format agricultural & biological journals expect.
    Groups sharing a letter are not significantly different."""
    order = list(np.argsort(means)[::-1])
    sig = {(r["Group 1"], r["Group 2"]): r[p_col] < alpha for _, r in posthoc.iterrows()}

    def different(a, b):
        return sig.get((a, b), sig.get((b, a), False))

    letters: list[set] = []
    for idx in order:
        name = names[idx]
        placed = False
        for s in letters:
            if all(not different(name, other) for other in s):
                s.add(name)
                placed = True
        if not placed:
            letters.append({name})
    # absorb sets fully contained in others (insertion-order algorithm can create redundancies)
    letters = [s for s in letters if not any(s < t for t in letters)]
    abc = "abcdefghijklmnopqrstuvwxyz"
    return {n: "".join(abc[i] for i, s in enumerate(letters) if n in s) for n in names}


# ==========================================================
# Repeated measures / blocks (wide data: rows = subjects, cols = conditions)
# ==========================================================
def rm_anova(data: pd.DataFrame) -> TestResult:
    """One-way repeated-measures ANOVA with Greenhouse-Geisser correction."""
    X = data.apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
    n, k = X.shape
    grand = X.mean()
    ss_cond = n * ((X.mean(axis=0) - grand) ** 2).sum()
    ss_subj = k * ((X.mean(axis=1) - grand) ** 2).sum()
    ss_total = ((X - grand) ** 2).sum()
    ss_err = ss_total - ss_cond - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    F = (ss_cond / df1) / (ss_err / df2)
    p = stats.f.sf(F, df1, df2)
    # Greenhouse-Geisser epsilon from double-centred covariance
    S = np.cov(X, rowvar=False)
    C = np.eye(k) - np.ones((k, k)) / k
    Sc = C @ S @ C
    eps = np.trace(Sc) ** 2 / ((k - 1) * (Sc ** 2).sum()) if k > 2 else 1.0
    eps = float(min(max(eps, 1 / (k - 1)), 1.0))
    p_gg = stats.f.sf(F, df1 * eps, df2 * eps)
    pes = ss_cond / (ss_cond + ss_err)
    res = TestResult("Repeated-measures ANOVA", "تحليل التباين للقياسات المتكررة", "F", F, p, (df1, df2),
                     "partial η²", pes, "eta2", n=n,
                     extra={"Greenhouse-Geisser ε": eps, "p (GG-corrected)": p_gg, "Conditions": k})
    if eps < 0.75:
        res.notes.append(f"Sphericity likely violated (ε = {eps:.2f}); report the Greenhouse-Geisser p = {p_gg:.4f}.")
    return res


def friedman(data: pd.DataFrame) -> TestResult:
    X = data.apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
    n, k = X.shape
    res = stats.friedmanchisquare(*[X[:, j] for j in range(k)])
    W = res.statistic / (n * (k - 1))
    return TestResult("Friedman test", "اختبار فريدمان", "χ²F", res.statistic, res.pvalue, k - 1,
                      "Kendall's W", W, "W", n=n)


def cochran_q(data: pd.DataFrame) -> TestResult:
    """Cochran's Q for k related binary (0/1) outcomes."""
    X = data.apply(pd.to_numeric, errors="coerce").dropna().to_numpy(float)
    n, k = X.shape
    col, row = X.sum(axis=0), X.sum(axis=1)
    T = X.sum()
    q = (k - 1) * (k * (col ** 2).sum() - T ** 2) / (k * T - (row ** 2).sum())
    p = stats.chi2.sf(q, k - 1)
    return TestResult("Cochran's Q test", "اختبار كوكران Q", "Q", q, p, k - 1, n=n)


# ==========================================================
# Two-way ANOVA (OLS with sum-to-zero coding)
# ==========================================================
def _design(df: pd.DataFrame, terms: list[str]) -> np.ndarray:
    cols = [np.ones(len(df))]
    for term in terms:
        factors = term.split(":")
        mats = []
        for f in factors:
            lv = sorted(df[f].astype(str).unique())
            m = np.zeros((len(df), len(lv) - 1))
            v = df[f].astype(str).to_numpy()
            for i, level in enumerate(lv[:-1]):
                m[:, i] = (v == level).astype(float)
            m[v == lv[-1], :] = -1.0            # effect (sum) coding
            mats.append(m)
        block = mats[0]
        for m in mats[1:]:
            block = np.einsum("ni,nj->nij", block, m).reshape(len(df), -1)
        cols.append(block)
    return np.column_stack(cols)


def _rss(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return float(r @ r)


def anova_twoway(df: pd.DataFrame, y: str, a: str, b: str, interaction: bool = True,
                 ss_type: int = 3) -> pd.DataFrame:
    """Two-way ANOVA table (Type II or Type III sums of squares, SPSS-compatible Type III by default)."""
    d = df[[y, a, b]].dropna().copy()
    d[a], d[b] = d[a].astype(str), d[b].astype(str)
    yv = pd.to_numeric(d[y], errors="coerce").to_numpy(float)
    full_terms = [a, b] + ([f"{a}:{b}"] if interaction else [])
    X_full = _design(d, full_terms)
    rss_full = _rss(X_full, yv)
    df_res = len(d) - np.linalg.matrix_rank(X_full)
    rows = []
    for term in full_terms:
        if ss_type == 3:
            rest = [t for t in full_terms if t != term]
        else:  # Type II: compare models containing all terms that do not contain this one
            rest = [t for t in full_terms if t != term and not set(term.split(":")) < set(t.split(":"))]
        with_term = rest + [term]
        ss = _rss(_design(d, rest), yv) - _rss(_design(d, with_term), yv) if ss_type == 2 else \
            _rss(_design(d, rest), yv) - rss_full
        dfe = np.linalg.matrix_rank(_design(d, with_term)) - np.linalg.matrix_rank(_design(d, rest))
        F = (ss / dfe) / (rss_full / df_res)
        rows.append({"Source": term.replace(":", " × "), "SS": ss, "df": dfe, "MS": ss / dfe, "F": F,
                     "p-value": stats.f.sf(F, dfe, df_res), "partial η²": ss / (ss + rss_full)})
    rows.append({"Source": "Residual", "SS": rss_full, "df": df_res, "MS": rss_full / df_res,
                 "F": np.nan, "p-value": np.nan, "partial η²": np.nan})
    return pd.DataFrame(rows)


# ==========================================================
# Association
# ==========================================================
def correlation_test(x, y, method: str = "pearson", alternative: str = "two-sided", conf: float = 0.95) -> TestResult:
    d = pd.DataFrame({"x": pd.to_numeric(pd.Series(x), errors="coerce"),
                      "y": pd.to_numeric(pd.Series(y), errors="coerce")}).dropna()
    n = len(d)
    if method == "pearson":
        res = stats.pearsonr(d["x"], d["y"], alternative=alternative)
        name = ("Pearson correlation", "معامل ارتباط بيرسون")
    elif method == "spearman":
        res = stats.spearmanr(d["x"], d["y"], alternative=alternative)
        name = ("Spearman rank correlation", "معامل ارتباط سبيرمان للرتب")
    else:
        res = stats.kendalltau(d["x"], d["y"], alternative=alternative)
        name = ("Kendall's tau-b", "معامل كندال تاو-ب")
    r = float(res[0])
    # Fisher z CI (Bonett-Wright SE for Spearman, Fieller for Kendall)
    se = {"pearson": 1 / np.sqrt(n - 3), "spearman": np.sqrt((1 + r ** 2 / 2) / (n - 3)),
          "kendall": np.sqrt(0.437 / (n - 4)) if n > 4 else np.nan}[method]
    zc = stats.norm.ppf(1 - (1 - conf) / 2)
    ci = (np.tanh(np.arctanh(r) - zc * se), np.tanh(np.arctanh(r) + zc * se)) if abs(r) < 1 else (r, r)
    tstat = r * np.sqrt((n - 2) / (1 - r ** 2)) if abs(r) < 1 else np.inf
    return TestResult(name[0], name[1], "r", r, float(res[1]), n - 2, "r", r, "r", ci, f"{conf:.0%} CI of r", r, n,
                      _alt_label(alternative), {"r²": r ** 2, "t": tstat})


def partial_correlation(df: pd.DataFrame, x: str, y: str, covars: list[str], method: str = "pearson") -> TestResult:
    covars = [c for c in dict.fromkeys(covars) if c not in (x, y)]
    if not covars:
        raise ValueError("Control variables must differ from X and Y")
    d = df[[x, y] + covars].apply(pd.to_numeric, errors="coerce").dropna()
    if method == "spearman":
        d = d.rank()
    Z = np.column_stack([np.ones(len(d)), d[covars].to_numpy(float)])

    def resid(v):
        beta, *_ = np.linalg.lstsq(Z, v, rcond=None)
        return v - Z @ beta

    rx, ry = resid(d[x].to_numpy(float)), resid(d[y].to_numpy(float))
    r = float(np.corrcoef(rx, ry)[0, 1])
    n, k = len(d), len(covars)
    dfree = n - 2 - k
    t = r * np.sqrt(dfree / (1 - r ** 2))
    p = 2 * stats.t.sf(abs(t), dfree)
    se = 1 / np.sqrt(n - 3 - k)
    ci = (np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se))
    return TestResult(f"Partial correlation (controlling for {', '.join(covars)})",
                      f"الارتباط الجزئي (بعد عزل أثر: {'، '.join(covars)})", "r", r, p, dfree, "partial r", r, "r",
                      ci, "95% CI", r, n, extra={"t": t})


def chi2_independence(df: pd.DataFrame, a: str, b: str) -> tuple[TestResult, pd.DataFrame, pd.DataFrame]:
    """Pearson χ² (Yates for 2×2), likelihood-ratio G, Fisher exact (2×2), Cramér's V,
    and adjusted standardised residuals (|r| > 1.96 marks the cells driving the association)."""
    ct = pd.crosstab(df[a], df[b])
    r, c = ct.shape
    if r < 2 or c < 2:
        raise ValueError("Both variables need at least two categories")
    chi2, p, dof, exp = stats.chi2_contingency(ct, correction=(r == 2 and c == 2))
    g, pg, _, _ = stats.chi2_contingency(ct, lambda_="log-likelihood", correction=False)
    n = ct.values.sum()
    chi2_raw = stats.chi2_contingency(ct, correction=False)[0]
    v = np.sqrt(chi2_raw / (n * (min(r, c) - 1)))
    extra = {"Likelihood-ratio G²": g, "G² p-value": pg, "Expected < 5 (%)": float((exp < 5).mean() * 100)}
    notes = []
    if r == 2 and c == 2:
        a_, b_, c_, d_ = ct.values.ravel()
        extra["Fisher exact p"] = stats.fisher_exact(ct.values)[1]
        extra["φ (phi)"] = (a_ * d_ - b_ * c_) / np.sqrt((a_ + b_) * (c_ + d_) * (a_ + c_) * (b_ + d_))
        if min(a_, b_, c_, d_) > 0:
            orr = a_ * d_ / (b_ * c_)
            se = np.sqrt(1 / a_ + 1 / b_ + 1 / c_ + 1 / d_)
            extra.update({"Odds ratio": orr, "OR 95% CI low": np.exp(np.log(orr) - 1.96 * se),
                          "OR 95% CI high": np.exp(np.log(orr) + 1.96 * se)})
    if (exp < 5).mean() > 0.2:
        notes.append("More than 20% of expected counts are < 5 — use the Fisher exact / G² result or merge categories.")
    res = TestResult("Pearson χ² test of independence" + (" (Yates)" if r == 2 and c == 2 else ""),
                     "اختبار كاي² للاستقلال" + (" (بتصحيح ييتس)" if r == 2 and c == 2 else ""),
                     "χ²", chi2, p, dof, "Cramér's V", v, "V", n=int(n), extra={**extra, "dof_v": min(r, c) - 1},
                     notes=notes)
    row_p = ct.sum(axis=1).to_numpy()[:, None] / n
    col_p = ct.sum(axis=0).to_numpy()[None, :] / n
    adj = (ct.values - exp) / np.sqrt(exp * (1 - row_p) * (1 - col_p))
    resid = pd.DataFrame(adj, index=ct.index, columns=ct.columns)
    return res, ct, resid


# ==========================================================
# Confidence intervals & bootstrap
# ==========================================================
def ci_mean(x, conf: float = 0.95) -> tuple:
    x = _clean(x)
    se = x.std(ddof=1) / np.sqrt(x.size)
    t = stats.t.ppf(1 - (1 - conf) / 2, x.size - 1)
    return x.mean(), x.mean() - t * se, x.mean() + t * se


def ci_variance(x, conf: float = 0.95) -> tuple:
    x = _clean(x)
    n, s2 = x.size, x.var(ddof=1)
    a = 1 - conf
    return s2, (n - 1) * s2 / stats.chi2.ppf(1 - a / 2, n - 1), (n - 1) * s2 / stats.chi2.ppf(a / 2, n - 1)


BOOT_STATS = {
    "mean": (np.mean, "Mean", "الوسط الحسابي"),
    "median": (np.median, "Median", "الوسيط"),
    "std": (lambda v, axis=-1: np.std(v, ddof=1, axis=axis), "Standard deviation", "الانحراف المعياري"),
    "cv": (lambda v, axis=-1: np.std(v, ddof=1, axis=axis) / np.mean(v, axis=axis) * 100, "CV (%)", "معامل الاختلاف (%)"),
    "trimmed": (lambda v, axis=-1: stats.trim_mean(v, 0.1, axis=axis), "10% trimmed mean", "الوسط المشذّب 10%"),
}


def bootstrap_ci(x, statistic: str = "median", conf: float = 0.95, n_resamples: int = 5000,
                 method: str = "BCa", seed: int = 2026) -> tuple:
    x = _clean(x)
    f = BOOT_STATS[statistic][0]
    res = stats.bootstrap((x,), f, confidence_level=conf, n_resamples=n_resamples, method=method,
                          random_state=np.random.default_rng(seed), vectorized=True)
    return float(f(x)), res.confidence_interval.low, res.confidence_interval.high, res.standard_error


# ==========================================================
# Test Advisor (decision logic, Field 2018; Sheskin 2011)
# ==========================================================
def recommend(goal: str, n_groups: int = 2, paired: bool = False, outcome: str = "numeric",
              normal: bool = True, equal_var: bool = True, large: bool = False) -> tuple[str, str, str]:
    """Return (test key, reason EN, reason AR)."""
    normal_ok = normal or large
    if goal == "one_sample":
        if outcome == "binary":
            return "one_prop", "Binary outcome vs a reference proportion.", "متغير ثنائي يُقارن بنسبة مرجعية."
        if normal_ok:
            return "t1", "Numeric outcome, approximately normal (or n ≥ 30).", "متغير رقمي توزيعه قريب من الطبيعي (أو n ≥ 30)."
        return "wilcoxon1", "Numeric outcome, not normal and small n → non-parametric.", \
            "متغير رقمي غير طبيعي وعينة صغيرة، لذا نستخدم اختباراً لامعلمياً."
    if goal == "compare":
        if outcome == "binary":
            if paired:
                return ("mcnemar", "Paired binary outcomes.", "متغير ثنائي في عينات مرتبطة.") if n_groups == 2 else \
                    ("cochran", "k related binary outcomes.", "عدة قياسات ثنائية مرتبطة.")
            return "chi2", "Binary/categorical outcome across independent groups.", "متغير نوعي عبر مجموعات مستقلة."
        if paired:
            if n_groups == 2:
                return ("tpaired", "Two related measurements, differences ≈ normal.", "قياسان مرتبطان والفروق قريبة من الطبيعي.") \
                    if normal_ok else ("wilcoxon_paired", "Two related measurements, non-normal differences.",
                                       "قياسان مرتبطان والفروق غير طبيعية.")
            return ("rm_anova", "≥ 3 related measurements, approximately normal.", "ثلاثة قياسات مرتبطة أو أكثر وتوزيعها قريب من الطبيعي.") \
                if normal_ok else ("friedman", "≥ 3 related measurements, non-normal.", "ثلاثة قياسات مرتبطة أو أكثر وتوزيعها غير طبيعي.")
        if n_groups == 2:
            if not normal_ok:
                return "mwu", "Two independent groups, non-normal and small samples.", "مجموعتان مستقلتان غير طبيعيتين وعينات صغيرة."
            return ("t_student", "Two independent groups, normal, equal variances.", "مجموعتان مستقلتان، توزيع طبيعي وتباينات متساوية.") \
                if equal_var else ("t_welch", "Two independent groups, unequal variances → Welch.",
                                   "مجموعتان مستقلتان وتباينات غير متساوية، لذا نستخدم اختبار ويلش.")
        if not normal_ok:
            return "kruskal", "≥ 3 independent groups, non-normal → Kruskal-Wallis + Dunn.", \
                "ثلاث مجموعات مستقلة أو أكثر غير طبيعية، لذا نستخدم كروسكال-واليس ثم دن."
        return ("anova", "≥ 3 groups, normal, equal variances → ANOVA + Tukey.", "ثلاث مجموعات أو أكثر، طبيعية ومتجانسة، لذا نستخدم ANOVA ثم توكي.") \
            if equal_var else ("welch_anova", "≥ 3 groups, unequal variances → Welch ANOVA + Games-Howell.",
                               "ثلاث مجموعات أو أكثر بتباينات غير متساوية، لذا نستخدم ANOVA لويلش ثم جيمس-هاول.")
    if goal == "relationship":
        if outcome == "categorical":
            return "chi2", "Two categorical variables → χ² test of independence.", "متغيران نوعيان، لذا نستخدم اختبار كاي² للاستقلال."
        return ("pearson", "Two numeric variables, approximately normal, linear.", "متغيران رقميان طبيعيان والعلاقة خطية.") \
            if normal_ok else ("spearman", "Numeric/ordinal variables, non-normal or monotonic.",
                               "متغيرات رقمية أو ترتيبية غير طبيعية أو علاقة رتيبة.")
    raise ValueError(goal)
