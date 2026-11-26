# ==========================================
# Aetheris Research - Descriptive Statistics Engine
# ------------------------------------------
# Pure functions (no Streamlit) so they can be unit-tested and reused
# by the report generator and the AI assistant.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


# ------------------------------------------
# Column typing
# ------------------------------------------
def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, max_unique: int = 50) -> list[str]:
    cats = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            # low-cardinality integer codes (e.g. Likert 1-5) are also categorical candidates
            if pd.api.types.is_integer_dtype(s) and 1 < s.nunique() <= 10:
                cats.append(c)
            continue
        if pd.api.types.is_datetime64_any_dtype(s):
            continue
        if s.nunique(dropna=True) <= max_unique:
            cats.append(c)
    return cats


def time_columns(df: pd.DataFrame) -> list[str]:
    out = []
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            out.append(c)
        elif str(c).lower() in {"year", "date", "time", "period", "السنة", "العام", "التاريخ"}:
            out.append(c)
        elif pd.api.types.is_integer_dtype(s) and s.dropna().between(1800, 2200).all() and s.nunique() > 2:
            out.append(c)
    return out


# ------------------------------------------
# Summary statistics
# ------------------------------------------
# Measure groups, in display order. Keys are the canonical (English) names.
MEASURE_GROUPS: dict[str, list[str]] = {
    "central": ["Mean", "Median", "Mode", "Geometric mean", "Harmonic mean", "Trimmed mean (5%)"],
    "dispersion": ["Std. Dev.", "Variance", "Range", "IQR", "Mean abs. deviation", "MAD (median)",
                   "CV (%)", "Quartile coeff. of dispersion", "SE Mean", "95% CI Lower", "95% CI Upper"],
    "shape": ["Skewness", "Kurtosis (excess)", "Pearson skewness (median)"],
    "position": ["Min", "P5", "P10", "Q1", "Q3", "P90", "P95", "Max"],
    "concentration": ["Gini", "Share top 10% (%)", "Share bottom 40% (%)", "Palma ratio"],
    "data": ["N", "Missing", "Outliers (1.5·IQR)"],
}
GROUP_LABELS = {
    "central": ("Central tendency", "مقاييس النزعة المركزية"),
    "dispersion": ("Dispersion", "مقاييس التشتت"),
    "shape": ("Shape of distribution", "مقاييس شكل التوزيع"),
    "position": ("Position (percentiles)", "مقاييس الموقع (المئينات)"),
    "concentration": ("Concentration & inequality", "مقاييس التركز وعدالة التوزيع"),
    "data": ("Sample information", "معلومات العينة"),
}
# Compact set used in journal tables (APA-style "Table 1: Descriptive statistics")
BRIEF_MEASURES = ["N", "Mean", "Std. Dev.", "Min", "Median", "Max", "Skewness", "Kurtosis (excess)"]

MEASURE_LABELS_AR = {
    "N": "عدد المشاهدات", "Missing": "القيم المفقودة", "Outliers (1.5·IQR)": "القيم الشاذة (1.5×المدى الربيعي)",
    "Mean": "الوسط الحسابي", "Median": "الوسيط", "Mode": "المنوال", "Geometric mean": "الوسط الهندسي",
    "Harmonic mean": "الوسط التوافقي", "Trimmed mean (5%)": "الوسط المشذّب (5%)",
    "Std. Dev.": "الانحراف المعياري", "Variance": "التباين", "Range": "المدى", "IQR": "المدى الربيعي",
    "Mean abs. deviation": "الانحراف المتوسط", "MAD (median)": "الانحراف المطلق الوسيط",
    "CV (%)": "معامل الاختلاف (%)", "Quartile coeff. of dispersion": "معامل الاختلاف الربيعي",
    "SE Mean": "الخطأ المعياري للوسط", "95% CI Lower": "الحد الأدنى لفترة الثقة 95%",
    "95% CI Upper": "الحد الأعلى لفترة الثقة 95%",
    "Skewness": "معامل الالتواء", "Kurtosis (excess)": "معامل التفرطح (الزائد)",
    "Pearson skewness (median)": "معامل التواء بيرسون (بالوسيط)",
    "Min": "أصغر قيمة", "P5": "المئين 5", "P10": "المئين 10 (العشير الأول)", "Q1": "الربيع الأدنى (Q1)",
    "Q3": "الربيع الأعلى (Q3)", "P90": "المئين 90 (العشير التاسع)", "P95": "المئين 95", "Max": "أكبر قيمة",
    "Gini": "معامل جيني", "Share top 10% (%)": "نصيب أعلى 10% (%)",
    "Share bottom 40% (%)": "نصيب أدنى 40% (%)", "Palma ratio": "نسبة بالما",
}


def measure_label(key: str, lang: str = "en") -> str:
    return MEASURE_LABELS_AR.get(key, key) if lang == "ar" else key


def gini(values) -> float:
    """Gini coefficient (population formula) for non-negative values; 0 = perfect equality."""
    x = np.sort(np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float))
    n = x.size
    if n < 2 or (x < 0).any() or x.sum() == 0:
        return np.nan
    i = np.arange(1, n + 1)
    return float((2 * np.sum(i * x)) / (n * x.sum()) - (n + 1) / n)


def lorenz_curve(values) -> pd.DataFrame:
    """Cumulative population share vs cumulative value share (both 0..1)."""
    x = np.sort(np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float))
    if x.size == 0 or x.sum() == 0:
        return pd.DataFrame({"population_share": [0.0, 1.0], "value_share": [0.0, 1.0]})
    cum = np.concatenate([[0.0], np.cumsum(x) / x.sum()])
    pop = np.linspace(0, 1, x.size + 1)
    return pd.DataFrame({"population_share": pop, "value_share": cum})


def income_shares(values) -> dict:
    """Shares held by the top 10% and bottom 40% of units, and the Palma ratio."""
    x = np.sort(np.asarray(pd.to_numeric(pd.Series(values), errors="coerce").dropna(), dtype=float))
    n = x.size
    if n < 10 or (x < 0).any() or x.sum() == 0:
        return {"Share top 10% (%)": np.nan, "Share bottom 40% (%)": np.nan, "Palma ratio": np.nan}
    lc = lorenz_curve(x)
    share_at = lambda p: float(np.interp(p, lc["population_share"], lc["value_share"]))  # noqa: E731
    top10 = 1 - share_at(0.9)
    bottom40 = share_at(0.4)
    return {"Share top 10% (%)": top10 * 100, "Share bottom 40% (%)": bottom40 * 100,
            "Palma ratio": top10 / bottom40 if bottom40 > 0 else np.nan}


def describe_series(s: pd.Series) -> dict:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float)
    n, missing = int(x.size), int(s.size - x.size)
    if n == 0:
        return {"N": 0, "Missing": missing}
    mean = x.mean()
    std = x.std(ddof=1) if n > 1 else np.nan
    p5, p10, q1, med, q3, p90, p95 = np.percentile(x, [5, 10, 25, 50, 75, 90, 95])
    iqr = q3 - q1
    se = std / np.sqrt(n) if n > 1 else np.nan
    tcrit = stats.t.ppf(0.975, n - 1) if n > 1 else np.nan
    outliers = int(((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).sum())
    counts = x.value_counts()
    mode = float(counts.index[counts == counts.max()].min()) if counts.max() > 1 else np.nan
    positive = bool((x > 0).all())
    out = {
        "N": n,
        "Missing": missing,
        # central tendency
        "Mean": mean,
        "Median": med,
        "Mode": mode,
        "Geometric mean": float(stats.gmean(x)) if positive else np.nan,
        "Harmonic mean": float(stats.hmean(x)) if positive else np.nan,
        "Trimmed mean (5%)": float(stats.trim_mean(x, 0.05)),
        # dispersion
        "Std. Dev.": std,
        "Variance": std ** 2 if n > 1 else np.nan,
        "Range": x.max() - x.min(),
        "IQR": iqr,
        "Mean abs. deviation": float((x - mean).abs().mean()),
        "MAD (median)": float((x - med).abs().median()),
        "CV (%)": (std / abs(mean) * 100) if n > 1 and mean != 0 else np.nan,
        "Quartile coeff. of dispersion": iqr / (q3 + q1) if (q3 + q1) != 0 else np.nan,
        "SE Mean": se,
        "95% CI Lower": mean - tcrit * se if n > 1 else np.nan,
        "95% CI Upper": mean + tcrit * se if n > 1 else np.nan,
        # shape
        "Skewness": stats.skew(x, bias=False) if n > 2 else np.nan,
        "Kurtosis (excess)": stats.kurtosis(x, bias=False) if n > 3 else np.nan,
        "Pearson skewness (median)": 3 * (mean - med) / std if n > 1 and std > 0 else np.nan,
        # position
        "Min": x.min(), "P5": p5, "P10": p10, "Q1": q1, "Q3": q3, "P90": p90, "P95": p95, "Max": x.max(),
        # concentration
        "Gini": gini(x),
        "Outliers (1.5·IQR)": outliers,
    }
    out.update(income_shares(x))
    return out


def summary_table(df: pd.DataFrame, cols: list[str] | None = None,
                  groups: list[str] | None = None, brief: bool = False) -> pd.DataFrame:
    """One row per variable. `groups` selects measure groups (see MEASURE_GROUPS);
    `brief=True` returns the compact journal-style set."""
    cols = cols or numeric_columns(df)
    table = pd.DataFrame({c: describe_series(df[c]) for c in cols}).T
    if brief:
        order = BRIEF_MEASURES
    else:
        groups = groups or list(MEASURE_GROUPS.keys())
        order = [m for g in MEASURE_GROUPS if g in groups for m in MEASURE_GROUPS[g]]
    table = table.reindex(columns=[m for m in order if m in table.columns])
    table.index.name = "Variable"
    return table


def percentile_table(df: pd.DataFrame, cols: list[str], kind: str = "deciles") -> pd.DataFrame:
    """Deciles (10..90), quartiles or ventiles (5..95) for each variable."""
    qs = {"quartiles": [25, 50, 75], "deciles": list(range(10, 100, 10)),
          "ventiles": list(range(5, 100, 5))}[kind]
    rows = {}
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce").dropna()
        rows[c] = np.percentile(x, qs) if len(x) else [np.nan] * len(qs)
    return pd.DataFrame(rows, index=[f"P{q}" for q in qs]).T


def gini_by_group(df: pd.DataFrame, value_col: str, group_col: str) -> pd.DataFrame:
    rows = []
    for g, sub in df.groupby(group_col):
        sh = income_shares(sub[value_col])
        rows.append({"Group": g, "N": int(sub[value_col].notna().sum()), "Gini": gini(sub[value_col]), **sh})
    return pd.DataFrame(rows)


def group_summary(df: pd.DataFrame, group_col: str, value_cols: list[str],
                  funcs: tuple = ("count", "mean", "std", "min", "median", "max")) -> pd.DataFrame:
    g = df.groupby(group_col, dropna=False)[value_cols].agg(list(funcs))
    g.columns = [f"{v} | {f}" for v, f in g.columns]
    return g.reset_index()


# ------------------------------------------
# Normality
# ------------------------------------------
def normality_tests(s: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    x = pd.to_numeric(s, errors="coerce").dropna().astype(float).values
    n = x.size
    rows = []
    if 3 <= n <= 5000:
        w, p = stats.shapiro(x)
        rows.append(("Shapiro-Wilk", w, p))
    if n >= 8:
        k2, p = stats.normaltest(x)
        rows.append(("D'Agostino-Pearson K²", k2, p))
    if n >= 3:
        jb = stats.jarque_bera(x)
        rows.append(("Jarque-Bera", jb.statistic, jb.pvalue))
    if n >= 5 and x.std(ddof=1) > 0:
        z = (x - x.mean()) / x.std(ddof=1)
        ks = stats.kstest(z, "norm")
        rows.append(("Kolmogorov-Smirnov (standardised)", ks.statistic, ks.pvalue))
    out = pd.DataFrame(rows, columns=["Test", "Statistic", "p-value"])
    out["Normal at α"] = out["p-value"] > alpha
    return out


def normality_overview(df: pd.DataFrame, cols: list[str], alpha: float = 0.05) -> pd.DataFrame:
    """Compact table: one row per variable, Shapiro (or JB for large n) + verdict."""
    rows = []
    for c in cols:
        t = normality_tests(df[c], alpha)
        if t.empty:
            continue
        pick = t[t["Test"] == "Shapiro-Wilk"]
        if pick.empty:
            pick = t[t["Test"] == "Jarque-Bera"]
        r = pick.iloc[0]
        rows.append({"Variable": c, "Test": r["Test"], "Statistic": r["Statistic"],
                     "p-value": r["p-value"], "Normal at α": bool(r["Normal at α"])})
    return pd.DataFrame(rows)


# ------------------------------------------
# Correlation
# ------------------------------------------
def correlation_matrix(df: pd.DataFrame, cols: list[str], method: str = "pearson"):
    """Return (r, p, n) square DataFrames using pairwise-complete observations."""
    funcs = {"pearson": stats.pearsonr, "spearman": stats.spearmanr, "kendall": stats.kendalltau}
    f = funcs[method]
    k = len(cols)
    r = pd.DataFrame(np.eye(k), index=cols, columns=cols)
    p = pd.DataFrame(np.zeros((k, k)), index=cols, columns=cols)
    n = pd.DataFrame(np.zeros((k, k), dtype=int), index=cols, columns=cols)
    for i in range(k):
        n.iloc[i, i] = int(df[cols[i]].notna().sum())
        for j in range(i + 1, k):
            pair = df[[cols[i], cols[j]]].apply(pd.to_numeric, errors="coerce").dropna()
            nn = len(pair)
            if nn >= 3 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                res = f(pair.iloc[:, 0], pair.iloc[:, 1])
                rv, pv = float(res[0]), float(res[1])
            else:
                rv, pv = np.nan, np.nan
            r.iloc[i, j] = r.iloc[j, i] = rv
            p.iloc[i, j] = p.iloc[j, i] = pv
            n.iloc[i, j] = n.iloc[j, i] = nn
    return r, p, n


def significance_stars(p: float) -> str:
    if pd.isna(p):
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def starred_matrix(r: pd.DataFrame, p: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    """Publication-style lower-triangular matrix: 0.452** etc."""
    out = r.copy().astype(object)
    for i in range(len(r)):
        for j in range(len(r)):
            if j > i:
                out.iloc[i, j] = ""
            elif i == j:
                out.iloc[i, j] = "1"
            else:
                v = r.iloc[i, j]
                out.iloc[i, j] = "" if pd.isna(v) else f"{v:.{decimals}f}{significance_stars(p.iloc[i, j])}"
    return out


def correlation_strength(r: float) -> tuple[str, str]:
    a = abs(r)
    if a >= 0.8:
        return "very strong", "قوية جداً"
    if a >= 0.6:
        return "strong", "قوية"
    if a >= 0.4:
        return "moderate", "متوسطة"
    if a >= 0.2:
        return "weak", "ضعيفة"
    return "very weak / none", "ضعيفة جداً أو منعدمة"


def top_correlations(r: pd.DataFrame, p: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    rows = []
    cols = list(r.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if not pd.isna(r.iloc[i, j]):
                rows.append((cols[i], cols[j], r.iloc[i, j], p.iloc[i, j]))
    out = pd.DataFrame(rows, columns=["Var 1", "Var 2", "r", "p-value"])
    if out.empty:
        return out
    return out.reindex(out["r"].abs().sort_values(ascending=False).index).head(k).reset_index(drop=True)


# ------------------------------------------
# Categorical
# ------------------------------------------
def frequency_table(s: pd.Series, include_missing: bool = True) -> pd.DataFrame:
    counts = s.value_counts(dropna=not include_missing)
    out = pd.DataFrame({"Category": counts.index.astype(str), "Frequency": counts.values})
    total = out["Frequency"].sum()
    out["Percent (%)"] = out["Frequency"] / total * 100
    out["Cumulative (%)"] = out["Percent (%)"].cumsum()
    return out


def crosstab_chi2(df: pd.DataFrame, a: str, b: str):
    """Contingency table + Pearson chi-square + Cramér's V."""
    ct = pd.crosstab(df[a], df[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return ct, None
    chi2, p, dof, expected = stats.chi2_contingency(ct)
    n = ct.values.sum()
    v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
    low_expected = float((expected < 5).mean() * 100)
    return ct, {"chi2": chi2, "dof": int(dof), "p": p, "cramers_v": v, "pct_expected_lt5": low_expected}


# ------------------------------------------
# Time series / growth
# ------------------------------------------
def growth_analysis(df: pd.DataFrame, time_col: str, value_col: str,
                    entity_col: str | None = None) -> pd.DataFrame:
    """Per entity: start/end values, CAGR, average annual % change, and
    log-linear trend growth rate with its significance (classic agri-econ measure)."""
    groups = df.groupby(entity_col) if entity_col else [("All", df)]
    rows = []
    for ent, g in groups:
        g = g[[time_col, value_col]].dropna().sort_values(time_col)
        if len(g) < 2:
            continue
        t = g[time_col]
        if pd.api.types.is_datetime64_any_dtype(t):
            t_years = (t - t.iloc[0]).dt.days / 365.25
        else:
            t_years = pd.to_numeric(t, errors="coerce") - pd.to_numeric(t, errors="coerce").iloc[0]
        y = g[value_col].astype(float)
        span = float(t_years.iloc[-1])
        start, end = y.iloc[0], y.iloc[-1]
        cagr = ((end / start) ** (1 / span) - 1) * 100 if span > 0 and start > 0 and end > 0 else np.nan
        pct = y.pct_change().replace([np.inf, -np.inf], np.nan) * 100
        trend, trend_p, r2 = np.nan, np.nan, np.nan
        if (y > 0).all() and len(y) >= 3 and t_years.nunique() > 1:
            lr = stats.linregress(t_years.astype(float), np.log(y))
            trend, trend_p, r2 = (np.exp(lr.slope) - 1) * 100, lr.pvalue, lr.rvalue ** 2
        rows.append({
            "Entity": ent, "Obs": len(y), "Start": start, "End": end,
            "Total change (%)": (end / start - 1) * 100 if start else np.nan,
            "CAGR (%)": cagr, "Avg. period change (%)": pct.mean(),
            "Trend growth (%/yr, log-lin)": trend, "Trend p-value": trend_p, "Trend R²": r2,
            "Instability (CV %)": y.std(ddof=1) / abs(y.mean()) * 100 if y.mean() else np.nan,
        })
    return pd.DataFrame(rows)


# ------------------------------------------
# Automatic academic interpretation
# ------------------------------------------
def _f(v, d=2):
    return "—" if v is None or pd.isna(v) else f"{v:,.{d}f}"


def interpret_variable(name: str, d: dict, lang: str = "en") -> str:
    if d.get("N", 0) < 2:
        return ""
    sk, ku, cv = d.get("Skewness", np.nan), d.get("Kurtosis (excess)", np.nan), d.get("CV (%)", np.nan)
    ar = lang == "ar"
    if pd.isna(sk):
        shape = ""
    elif abs(sk) < 0.5:
        shape = "متماثل تقريباً" if ar else "approximately symmetric"
    elif sk > 0:
        shape = "ملتوٍ نحو اليمين (التواء موجب)" if ar else "right-skewed (positive skew)"
    else:
        shape = "ملتوٍ نحو اليسار (التواء سالب)" if ar else "left-skewed (negative skew)"
    if pd.isna(ku):
        tail = ""
    elif ku > 1:
        tail = "مدبب بأطراف ثقيلة" if ar else "leptokurtic (heavy tails)"
    elif ku < -1:
        tail = "مفلطح بأطراف خفيفة" if ar else "platykurtic (light tails)"
    else:
        tail = "قريب من تفرطح التوزيع الطبيعي" if ar else "mesokurtic (near-normal tails)"
    if pd.isna(cv):
        disp = ""
    elif cv < 15:
        disp = "تشتت منخفض" if ar else "low dispersion"
    elif cv < 35:
        disp = "تشتت متوسط" if ar else "moderate dispersion"
    else:
        disp = "تشتت مرتفع (عدم تجانس واضح)" if ar else "high dispersion (marked heterogeneity)"
    out = d.get("Outliers (1.5·IQR)", 0)
    g = d.get("Gini", np.nan)
    if pd.isna(g):
        conc = ""
    elif g < 0.3:
        conc = " (توزيع متقارب نسبياً بين المشاهدات)" if ar else " (relatively even distribution across units)"
    elif g < 0.5:
        conc = " (تركز متوسط)" if ar else " (moderate concentration)"
    else:
        conc = " (تركز مرتفع في عدد محدود من المشاهدات)" if ar else " (high concentration in few units)"

    if ar:
        txt = (f"<b>{name}</b>: بلغ المتوسط {_f(d['Mean'])} بانحراف معياري {_f(d['Std. Dev.'])} "
               f"(فترة ثقة 95%: {_f(d['95% CI Lower'])} – {_f(d['95% CI Upper'])})، والوسيط {_f(d['Median'])}. "
               f"معامل الاختلاف {_f(cv, 1)}% أي {disp}. التوزيع {shape}، و{tail}.")
        if out:
            txt += f" توجد {out} قيمة شاذة وفق قاعدة 1.5×المدى الربيعي، ويُنصح بفحصها قبل التحليل القياسي."
        if not pd.isna(sk) and abs(sk) >= 1:
            txt += " يُفضَّل استخدام الوسيط في الوصف، والنظر في التحويل اللوغاريتمي أو الاختبارات اللامعلمية."
        if not pd.isna(g):
            txt += f" معامل جيني {_f(g, 3)}{conc}."
        if not pd.isna(d.get("Geometric mean", np.nan)):
            txt += f" الوسط الهندسي {_f(d['Geometric mean'])}، وهو الأنسب لمتوسطات معدلات النمو والأرقام القياسية."
    else:
        txt = (f"<b>{name}</b>: mean = {_f(d['Mean'])}, SD = {_f(d['Std. Dev.'])} "
               f"(95% CI {_f(d['95% CI Lower'])} – {_f(d['95% CI Upper'])}), median = {_f(d['Median'])}. "
               f"CV = {_f(cv, 1)}%, i.e. {disp}. The distribution is {shape} and {tail}.")
        if out:
            txt += f" {out} outlier(s) by the 1.5×IQR rule — inspect them before econometric modelling."
        if not pd.isna(sk) and abs(sk) >= 1:
            txt += " Report the median, and consider a log transformation or non-parametric tests."
        if not pd.isna(g):
            txt += f" Gini = {_f(g, 3)}{conc}."
        if not pd.isna(d.get("Geometric mean", np.nan)):
            txt += f" Geometric mean = {_f(d['Geometric mean'])} — preferred for averaging growth rates and index numbers."
    return txt


def interpret_correlation(a: str, b: str, r: float, p: float, method: str, lang: str = "en") -> str:
    en_s, ar_s = correlation_strength(r)
    sig = p < 0.05
    if lang == "ar":
        direction = "طردية" if r > 0 else "عكسية"
        s = "دالة إحصائياً عند 5%" if sig else "غير دالة إحصائياً عند 5%"
        return (f"العلاقة بين <b>{a}</b> و<b>{b}</b> {direction} {ar_s} (r = {r:.3f}، p = {p:.4f})، وهي {s}. "
                "تذكير: الارتباط لا يعني السببية.")
    direction = "positive" if r > 0 else "negative"
    s = "statistically significant at 5%" if sig else "not significant at 5%"
    return (f"<b>{a}</b> and <b>{b}</b> show a {en_s} {direction} {method} correlation "
            f"(r = {r:.3f}, p = {p:.4f}), {s}. Correlation does not imply causation.")
