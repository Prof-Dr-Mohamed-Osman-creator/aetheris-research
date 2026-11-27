# ==========================================
# Aetheris Research - Regression reporting layer
# ------------------------------------------
# Elasticities & marginal effects (delta-method CIs), turning points,
# journal-style multi-model tables (Excel / Word / LaTeX / HTML),
# bilingual academic interpretation, policy brief + what-if simulation,
# auto-generated methodology paragraph with references, and reproducible
# Python / R / Stata code. Pure functions — no Streamlit.
# ==========================================

from __future__ import annotations

import re
from io import BytesIO

import numpy as np
import pandas as pd
from scipy import stats

from analytics import diagnostics as dx
from analytics import regression as rg


def _q(r, level=0.95):
    return stats.t.ppf(0.5 + level / 2, r.df_resid) if r.dist == "t" else stats.norm.ppf(0.5 + level / 2)


def _p_from_z(r, z):
    return 2 * (stats.t.sf(abs(z), r.df_resid) if r.dist == "t" else stats.norm.sf(abs(z)))


# ==========================================================
# Elasticities, marginal effects, discrete effects, turning points
# ==========================================================
def effects_table(df: pd.DataFrame, r: rg.RegResult, base: pd.DataFrame | None = None) -> pd.DataFrame:
    """For every original regressor: marginal effect dY/dX and elasticity at the representative point
    (numeric variables), or the discrete effect relative to the reference level (categorical).
    Computed through the full design pipeline, so logs, polynomials and interactions are handled exactly;
    standard errors by the delta method with the model's (possibly robust) covariance."""
    spec = r.design.spec
    base = rg.baseline_values(df, r) if base is None else base
    b = r.params.to_numpy()
    V = r.cov.to_numpy()
    logy = r.design.info["ytx"] == "log"
    row0 = rg.design_rows(r, base).to_numpy(float)[0]
    eta0 = row0 @ b
    sf = rg.smearing_factor(r) if logy else 1.0
    y0 = np.exp(eta0) * sf if logy else eta0
    q = _q(r)
    rows = []
    for v in spec.x:
        x0 = float(base[v].iloc[0])
        h = 1e-5 * max(1.0, abs(x0))
        up, dn = base.copy(), base.copy()
        up[v], dn[v] = x0 + h, x0 - h
        drow = (rg.design_rows(r, up).to_numpy(float)[0] - rg.design_rows(r, dn).to_numpy(float)[0]) / (2 * h)
        deta = drow @ b                      # ∂η/∂x
        if logy:
            me, g_me = y0 * deta, y0 * (drow + deta * row0)
            el, g_el = x0 * deta, x0 * drow
        else:
            me, g_me = deta, drow
            el = deta * x0 / y0
            g_el = drow * x0 / y0 - deta * x0 / y0 ** 2 * row0
        se_me = float(np.sqrt(g_me @ V @ g_me))
        se_el = float(np.sqrt(g_el @ V @ g_el))
        rows.append({"Variable": v, "Type": "continuous", "At value": x0, "Marginal effect dY/dX": me,
                     "ME s.e.": se_me, "ME p-value": _p_from_z(r, me / se_me) if se_me > 0 else np.nan,
                     "Elasticity": el, "Elasticity s.e.": se_el, "El. CI low": el - q * se_el,
                     "El. CI high": el + q * se_el, "El. p-value": _p_from_z(r, el / se_el) if se_el > 0 else np.nan})
    for c in spec.categorical:
        levels = r.design.info["levels"][c]
        ref = base.copy()
        ref[c] = levels[0]
        rref = rg.design_rows(r, ref).to_numpy(float)[0]
        for lv in levels[1:]:
            alt = base.copy()
            alt[c] = lv
            d = rg.design_rows(r, alt).to_numpy(float)[0] - rref
            dEta = d @ b
            se = float(np.sqrt(d @ V @ d))
            if logy:
                eff = 100 * (np.exp(dEta - 0.5 * se ** 2) - 1)      # Kennedy (1981)
                se_eff = 100 * np.exp(dEta) * se
                lab = "% change in Y vs reference (Kennedy)"
            else:
                eff, se_eff, lab = dEta, se, "Change in Y vs reference"
            rows.append({"Variable": f"{c} = {lv} (vs {levels[0]})", "Type": "discrete", "At value": np.nan,
                         "Marginal effect dY/dX": eff, "ME s.e.": se_eff,
                         "ME p-value": _p_from_z(r, dEta / se) if se > 0 else np.nan, "Elasticity": np.nan,
                         "Elasticity s.e.": np.nan, "El. CI low": np.nan, "El. CI high": np.nan,
                         "El. p-value": np.nan, "Note": lab})
    return pd.DataFrame(rows)


def turning_points(r: rg.RegResult, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """x* = −β₁ / (2β₂) for quadratic terms, back-transformed to the units of X, with delta-method CI."""
    spec = r.design.spec
    rows = []
    for v, deg in spec.poly.items():
        if int(deg) != 2:
            continue
        how = r.design.info["hows"].get(v, "level")
        name = rg._tname(v, how)
        sq = f"{name}^2"
        if name not in r.names or sq not in r.names:
            continue
        b1, b2 = r.params[name], r.params[sq]
        if b2 == 0:
            continue
        xs = -b1 / (2 * b2)
        g = np.array([-1 / (2 * b2), b1 / (2 * b2 ** 2)])
        Vs = r.cov.loc[[name, sq], [name, sq]].to_numpy()
        se = float(np.sqrt(g @ Vs @ g))
        q = _q(r)
        lo, hi = xs - q * se, xs + q * se
        mean = r.design.info["means"].get(v, 0.0) if spec.center else 0.0
        tr = (lambda z: np.exp(z + mean)) if how == "log" else (lambda z: 1 / (z + mean)) if how == "recip" else (lambda z: z + mean)
        vals = sorted([tr(lo), tr(hi)])
        inside = None
        if df is not None:
            raw = pd.to_numeric(df.loc[r.design.index, v], errors="coerce")
            inside = bool(raw.min() <= tr(xs) <= raw.max())
        rows.append({"Variable": v, "Shape": "maximum (inverted U)" if b2 < 0 else "minimum (U-shape)",
                     "Turning point": tr(xs), "CI low": vals[0], "CI high": vals[1],
                     "Within data range": inside})
    return pd.DataFrame(rows)


def what_if(df: pd.DataFrame, r: rg.RegResult, changes: dict, mode: str = "pct",
            base: pd.DataFrame | None = None, level: float = 0.95) -> dict:
    """Scenario simulation: change one or more regressors (in % or absolute units, or set a category)
    from the representative point; returns baseline, scenario, change and its delta-method CI."""
    base = rg.baseline_values(df, r) if base is None else base
    scen = base.copy()
    for v, ch in changes.items():
        if v in r.design.spec.categorical:
            scen[v] = ch
        elif mode == "pct":
            scen[v] = base[v] * (1 + ch / 100)
        else:
            scen[v] = base[v] + ch
    b = r.params.to_numpy()
    V = r.cov.to_numpy()
    r0 = rg.design_rows(r, base).to_numpy(float)[0]
    r1 = rg.design_rows(r, scen).to_numpy(float)[0]
    d = r1 - r0
    dEta = float(d @ b)
    se = float(np.sqrt(d @ V @ d))
    q = _q(r, level)
    p0, p1 = rg.predict(r, base, level), rg.predict(r, scen, level)
    y0, y1 = float(p0["fit"].iloc[0]), float(p1["fit"].iloc[0])
    if r.design.info["ytx"] == "log":
        pct = 100 * (np.exp(dEta) - 1)
        lo, hi = 100 * (np.exp(dEta - q * se) - 1), 100 * (np.exp(dEta + q * se) - 1)
        abs_ch = y1 - y0
        abs_lo, abs_hi = y0 * lo / 100, y0 * hi / 100
    else:
        abs_ch, abs_lo, abs_hi = dEta, dEta - q * se, dEta + q * se
        pct = 100 * abs_ch / y0 if y0 else np.nan
        lo, hi = (100 * abs_lo / y0, 100 * abs_hi / y0) if y0 else (np.nan, np.nan)
    return {"baseline": base, "scenario": scen, "y0": y0, "y1": y1, "change": abs_ch, "change_ci": (abs_lo, abs_hi),
            "pct": pct, "pct_ci": (lo, hi), "p": _p_from_z(r, dEta / se) if se > 0 else np.nan,
            "y0_ci": (float(p0["ci_low"].iloc[0]), float(p0["ci_high"].iloc[0])),
            "y1_ci": (float(p1["ci_low"].iloc[0]), float(p1["ci_high"].iloc[0]))}


# ==========================================================
# Journal-style multi-model table
# ==========================================================
def _fmt(v, d):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:,.{d}f}" if abs(v) >= 1000 else f"{v:.{d}f}"


def journal_table(results: list, labels: list | None = None, inner: str = "se", decimals: int = 3,
                  star_levels=(0.01, 0.05, 0.10), stats_rows=("N", "R²", "Adj. R²", "F", "AIC", "BIC"),
                  keep: list | None = None, const_last: bool = True, lang: str = "en") -> tuple[pd.DataFrame, str]:
    """Coefficients with significance stars and SE / t / p in parentheses below — the layout of
    AER / Econometrica / most Arabic journals. Returns (table, footnote)."""
    labels = labels or [f"({i + 1})" for i in range(len(results))]
    order = []
    for r in results:
        for n in r.names:
            if n not in order:
                order.append(n)
    if const_last and "const" in order:
        order.remove("const")
        order.append("const")
    if keep:
        order = [o for o in order if o in keep]
    rows = []
    for n in order:
        coef_row, inner_row = {"": n}, {"": ""}
        for lab, r in zip(labels, results):
            if n in r.names:
                p = r.pvalues[n]
                coef_row[lab] = _fmt(r.params[n], decimals) + rg.stars(p, star_levels)
                val = {"se": r.bse[n], "t": r.tvalues[n], "p": p}[inner]
                inner_row[lab] = f"({_fmt(val, decimals)})" if inner != "ci" else ""
                if inner == "ci":
                    inner_row[lab] = f"[{_fmt(r.ci.loc[n, 'low'], decimals)}, {_fmt(r.ci.loc[n, 'high'], decimals)}]"
            else:
                coef_row[lab] = inner_row[lab] = ""
        rows += [coef_row, inner_row]
    lab_ar = {"N": "عدد المشاهدات", "R²": "R²", "Adj. R²": "R² المعدل", "F": "إحصاءة F", "AIC": "AIC", "BIC": "BIC",
              "Dependent variable": "المتغير التابع", "Estimator": "طريقة التقدير", "Std. errors": "الأخطاء المعيارية"}
    sep = {"": "—" * 3}
    sep.update({lab: "" for lab in labels})
    rows.append(sep)
    for s in stats_rows:
        row = {"": lab_ar.get(s, s) if lang == "ar" else ("Observations" if s == "N" else s)}
        for lab, r in zip(labels, results):
            v = r.fit_stats().get(s)
            if s == "N":
                row[lab] = f"{int(v):,}"
            elif s == "F":
                row[lab] = (_fmt(v, 2) + rg.stars(r.f_pvalue, star_levels)) if v is not None and np.isfinite(v) else ""
            else:
                row[lab] = _fmt(v, decimals if s in ("R²", "Adj. R²") else 2) if v is not None else ""
        rows.append(row)
    for key, getter in (("Dependent variable", lambda r: r.y_label),
                        ("Estimator", lambda r: r.method.upper() if r.method != "huber" else "Huber"),
                        ("Std. errors", lambda r: r.cov_type if r.cov_type != "nonrobust" else "classical")):
        row = {"": lab_ar[key] if lang == "ar" else key}
        for lab, r in zip(labels, results):
            row[lab] = getter(r)
        rows.append(row)
    inner_txt = {"se": ("Standard errors", "الأخطاء المعيارية"), "t": ("t-statistics", "قيم t"),
                 "p": ("p-values", "القيم الاحتمالية"), "ci": ("95% confidence intervals", "فترات الثقة 95%")}[inner]
    a, b_, c = star_levels
    if lang == "ar":
        note = (f"ملاحظة: {inner_txt[1]} بين الأقواس. *** p < {a}، ** p < {b_}، * p < {c}. "
                "المصدر: تقديرات الباحث باستخدام منصة Aetheris Research.")
    else:
        note = (f"Notes: {inner_txt[0]} in parentheses. *** p < {a}, ** p < {b_}, * p < {c}. "
                "Source: author's estimates using Aetheris Research.")
    return pd.DataFrame(rows), note


def _latex_escape(s: str) -> str:
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    s = s.replace("×", r"$\times$").replace("²", r"$^2$").replace("—", "--")
    s = re.sub(r"(\*+)$", lambda m: "$^{" + m.group(1) + "}$", s)
    return s


def to_latex(table: pd.DataFrame, note: str, caption: str = "Regression results", label: str = "tab:reg") -> str:
    cols = list(table.columns)
    lines = [r"\begin{table}[!htbp]\centering", rf"\caption{{{_latex_escape(caption)}}}\label{{{label}}}",
             r"\begin{threeparttable}", r"\begin{tabular}{l" + "c" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(_latex_escape(c) for c in cols) + r" \\", r"\midrule"]
    for _, row in table.iterrows():
        if str(row.iloc[0]).startswith("—"):
            lines.append(r"\midrule")
            continue
        lines.append(" & ".join(_latex_escape(v) for v in row.values) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\begin{tablenotes}\footnotesize",
              rf"\item {_latex_escape(note)}", r"\end{tablenotes}", r"\end{threeparttable}", r"\end{table}"]
    return "\n".join(lines) + "\n% requires \\usepackage{booktabs,threeparttable}\n"


def to_html(table: pd.DataFrame, note: str, caption: str = "") -> str:
    head = "".join(f"<th style='border-bottom:1px solid #000;padding:3px 10px'>{c}</th>" for c in table.columns)
    body = []
    for _, row in table.iterrows():
        if str(row.iloc[0]).startswith("—"):
            body.append(f"<tr><td colspan='{len(row)}' style='border-top:1px solid #000'></td></tr>")
            continue
        body.append("<tr>" + "".join(f"<td style='padding:1px 10px;text-align:{'left' if i == 0 else 'center'}'>{v}</td>"
                                     for i, v in enumerate(row.values)) + "</tr>")
    return (f"<table style='border-collapse:collapse;font-family:Times New Roman,serif;border-top:2px solid #000;"
            f"border-bottom:2px solid #000'><caption style='caption-side:top;font-weight:bold'>{caption}</caption>"
            f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
            f"<p style='font-size:0.85em;font-family:Times New Roman,serif'>{note}</p>")


# ==========================================================
# Word export
# ==========================================================
def _docx_table(doc, table: pd.DataFrame, rtl=False):
    from docx.shared import Pt
    t = doc.add_table(rows=1, cols=len(table.columns))
    t.style = "Table Grid" if False else t.style
    for i, c in enumerate(table.columns):
        cell = t.rows[0].cells[i]
        cell.text = str(c)
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True
    for _, row in table.iterrows():
        cells = t.add_row().cells
        for i, v in enumerate(row.values):
            cells[i].text = "" if (isinstance(v, float) and np.isnan(v)) else (str(v) if not str(v).startswith("—") else "")
    for row in t.rows:
        for cell in row.cells:
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(9.5)
                    run.font.name = "Times New Roman"
    _booktabs(t)
    return t


def _booktabs(t):
    """Three horizontal rules (top, below header, bottom) — journal style, no vertical lines."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    tbl = t._tbl
    tblPr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, val in (("top", "single"), ("bottom", "single"), ("left", "nil"), ("right", "nil"),
                      ("insideH", "nil"), ("insideV", "nil")):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), "12")
        borders.append(el)
    tblPr.append(borders)
    for cell in t.rows[0].cells:
        tcPr = cell._tc.get_or_add_tcPr()
        b = OxmlElement("w:tcBorders")
        el = OxmlElement("w:bottom")
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), "6")
        b.append(el)
        tcPr.append(b)


def _num_df(df: pd.DataFrame, d=4) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_float_dtype(out[c]):
            out[c] = out[c].map(lambda v: "" if pd.isna(v) else f"{v:.{d}f}")
    return out


def to_docx(sections: list, title: str, rtl: bool = False) -> bytes:
    """Build a Word report. sections = [("h", text) | ("p", text) | ("table", DataFrame) | ("note", text)]."""
    from docx import Document
    from docx.shared import Pt
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    doc.add_heading(title, level=1)
    for kind, content in sections:
        if kind == "h":
            doc.add_heading(content, level=2)
        elif kind == "p":
            doc.add_paragraph(_strip_html(content))
        elif kind == "note":
            p = doc.add_paragraph(_strip_html(content))
            for run in p.runs:
                run.font.size = Pt(9.5)
                run.italic = True
        elif kind == "table":
            _docx_table(doc, _num_df(content), rtl)
            doc.add_paragraph("")
        elif kind == "bullets":
            for item in content:
                doc.add_paragraph(_strip_html(item), style="List Bullet")
    if rtl:
        from docx.oxml import OxmlElement
        from docx.oxml.ns import qn
        for p in doc.paragraphs:
            pPr = p._p.get_or_add_pPr()
            bidi = OxmlElement("w:bidi")
            bidi.set(qn("w:val"), "1")
            pPr.append(bidi)
    buf = BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _strip_html(s: str) -> str:
    s = re.sub(r"<br\s*/?>", "\n", str(s))
    return re.sub(r"<[^>]+>", "", s)


# ==========================================================
# Interpretation (bilingual)
# ==========================================================
def _coef_sentence(r: rg.RegResult, name: str, lang: str) -> str:
    b = r.params[name]
    p = r.pvalues[name]
    kind = r.design.info["kinds"].get(name, "level")
    var = r.design.info["base"].get(name, name)
    logy = r.design.info["ytx"] == "log"
    y = r.design.spec.y
    pv = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    sign_en = "increase" if b > 0 else "decrease"
    sign_ar = "زيادة" if b > 0 else "انخفاض"
    if kind == "dummy":
        lvl = name[name.find("[") + 1:-1]
        ref = r.design.info["levels"][var][0]
        if logy:
            pct = 100 * (np.exp(b) - 1)
            return (f"«{lvl}» مقارنةً بالفئة المرجعية «{ref}»: {y} أعلى/أقل بنحو {pct:.2f}% ({pv})." if lang == "ar" else
                    f"'{lvl}' vs the reference '{ref}': {y} differs by about {pct:.2f}% ({pv}).")
        return (f"«{lvl}» مقارنةً بالفئة المرجعية «{ref}»: يختلف {y} بمقدار {b:.4g} وحدة ({pv})." if lang == "ar" else
                f"'{lvl}' vs the reference '{ref}': {y} differs by {b:.4g} units ({pv}).")
    if kind in ("poly", "inter"):
        return (f"الحد «{name}» معنوي ({pv})، فالأثر غير خطي أو يتوقف على متغير آخر؛ راجع جدول المرونات والآثار الحدية." if lang == "ar"
                else f"The term '{name}' is significant ({pv}): the effect is non-linear or conditional — see the elasticities / marginal-effects table.")
    has_poly = int(r.design.spec.poly.get(var, 1)) > 1 or any(var in pr for pr in r.design.spec.interactions)
    if has_poly:
        return (f"المعامل الرئيسي لـ {name} = {b:.4g} ({pv})، ويُقرأ مع حدوده التربيعية/التفاعلية في جدول الآثار." if lang == "ar"
                else f"The main coefficient of {name} is {b:.4g} ({pv}); read it together with its squared / interaction terms in the effects table.")
    if kind == "log" and logy:
        return (f"زيادة {var} بنسبة 1% ترتبط بـ{sign_ar} {y} بنحو {abs(b):.3f}% (مرونة ثابتة، {pv})." if lang == "ar" else
                f"A 1% increase in {var} is associated with a {abs(b):.3f}% {sign_en} in {y} (constant elasticity, {pv}).")
    if kind == "log" and not logy:
        return (f"زيادة {var} بنسبة 1% ترتبط بـ{sign_ar} {y} بنحو {abs(b) / 100:.4g} وحدة ({pv})." if lang == "ar" else
                f"A 1% increase in {var} is associated with a {abs(b) / 100:.4g}-unit {sign_en} in {y} ({pv}).")
    if kind == "recip":
        return (f"العلاقة مع {var} مقلوبة: مع تزايد {var} يقترب {y} من مستوى تقاربي، والمعامل {b:.4g} ({pv})." if lang == "ar" else
                f"The relationship with {var} is reciprocal: as {var} grows, {y} approaches an asymptote; coefficient {b:.4g} ({pv}).")
    if logy:
        pct = 100 * (np.exp(b) - 1)
        return (f"زيادة {var} بوحدة واحدة ترتبط بتغير {y} بنسبة {pct:.3f}% ({pv})." if lang == "ar" else
                f"A one-unit increase in {var} is associated with a {pct:.3f}% change in {y} ({pv}).")
    return (f"زيادة {var} بوحدة واحدة ترتبط بـ{sign_ar} {y} بمقدار {abs(b):.4g} وحدة، مع ثبات العوامل الأخرى ({pv})." if lang == "ar" else
            f"A one-unit increase in {var} is associated with a {abs(b):.4g}-unit {sign_en} in {y}, other things equal ({pv}).")


def interpret(r: rg.RegResult, diag: dict | None = None, alpha: float = 0.05, lang: str = "en") -> str:
    ar = lang == "ar"
    n, k = r.nobs, r.k
    parts = []
    if r.design is None:
        return ""
    method = r.method_label(lang)
    cov = r.cov_label(lang)
    if ar:
        parts.append(f"قُدِّر النموذج بطريقة <b>{method}</b> على {n} مشاهدة، مع أخطاء معيارية {cov}.")
    else:
        parts.append(f"The model was estimated by <b>{method}</b> on {n} observations with {cov} standard errors.")
    if np.isfinite(r.rsquared):
        r2 = r.rsquared
        q_en = "high" if r2 >= 0.7 else "moderate" if r2 >= 0.4 else "low"
        q_ar = "مرتفعة" if r2 >= 0.7 else "متوسطة" if r2 >= 0.4 else "منخفضة"
        if ar:
            parts.append(f"تفسّر المتغيرات المستقلة {r2:.1%} من التباين في {r.y_label} (R² المعدل = {r.rsquared_adj:.3f})، "
                         f"وهي قدرة تفسيرية {q_ar}.")
        else:
            parts.append(f"The regressors explain {r2:.1%} of the variation in {r.y_label} (adjusted R² = "
                         f"{r.rsquared_adj:.3f}), a {q_en} explanatory power.")
    if np.isfinite(r.fvalue):
        sig = r.f_pvalue < alpha
        if ar:
            parts.append(f"اختبار F للمعنوية الكلية = {r.fvalue:.3f} (p = {r.f_pvalue:.4f})، "
                         + ("فالنموذج <b>معنوي ككل</b>." if sig else "فلا يمكن رفض أن جميع المعاملات صفرية."))
        else:
            parts.append(f"The overall F-test = {r.fvalue:.3f} (p = {r.f_pvalue:.4f}): "
                         + ("the model is <b>jointly significant</b>." if sig else "we cannot reject that all slopes are zero."))
    sig_terms = [c for c in r.names if c != "const" and r.pvalues[c] < alpha]
    ns_terms = [c for c in r.names if c != "const" and r.pvalues[c] >= alpha]
    if sig_terms:
        parts.append("<b>" + ("المتغيرات المعنوية:" if ar else "Significant effects:") + "</b><br>• "
                     + "<br>• ".join(_coef_sentence(r, c, lang) for c in sig_terms))
    if ns_terms:
        parts.append(("غير معنوية عند α = " if ar else "Not significant at α = ") + f"{alpha}: " + ", ".join(ns_terms) + ".")
    if diag:
        vs = dx.verdicts(diag, lang)
        bad = [v for v in vs if v["status"] != "ok"]
        good = [v["label"] for v in vs if v["status"] == "ok"]
        if good:
            parts.append(("✅ الافتراضات المستوفاة: " if ar else "✅ Assumptions satisfied: ") + "، ".join(good) if ar else
                         "✅ Assumptions satisfied: " + ", ".join(good) + ".")
        for v in bad:
            flag = "⚠️" if v["status"] == "mixed" else "❌"
            parts.append(f"{flag} <b>{v['label']}</b>: " + ", ".join(v["failed"]) + ". " + v["remedy"])
        hetero = next((v for v in vs if v["group"] == "hetero" and v["status"] != "ok"), None)
        if hetero and r.cov_type == "nonrobust":
            parts.append("👉 " + ("أعد التقدير بأخطاء معيارية متينة (HC3/HC1) قبل الاستدلال." if ar else
                                  "Re-estimate with robust (HC3/HC1) standard errors before drawing inferences."))
    parts.append(("ملاحظة منهجية: معاملات الانحدار تقيس الارتباط الشرطي، ولا تُفسَّر سببياً إلا إذا دعمها تصميم "
                  "البحث (انظر أدوات الاستدلال السببي في الإصدارات القادمة)." if ar else
                  "Methodological note: regression coefficients measure conditional associations; a causal reading "
                  "requires an identification strategy (causal-inference tools arrive in later releases)."))
    for note in r.notes:
        parts.append("ℹ️ " + note)
    return "<br><br>".join(parts)


def policy_brief(df: pd.DataFrame, r: rg.RegResult, diag: dict | None, lang: str = "en", alpha: float = 0.05) -> dict:
    """Plain-language summary for decision-makers: key drivers ranked by elasticity, reliability rating."""
    ar = lang == "ar"
    eff = effects_table(df, r)
    cont = eff[eff["Type"] == "continuous"].copy()
    cont = cont[cont["El. p-value"] < alpha]
    cont["abs"] = cont["Elasticity"].abs()
    cont = cont.sort_values("abs", ascending=False)
    y = r.design.spec.y
    bullets = []
    for _, row in cont.head(4).iterrows():
        e = row["Elasticity"]
        if ar:
            bullets.append(f"<b>{row['Variable']}</b>: زيادته بنسبة 10% ترتبط بتغير {y} بنحو {10 * e:+.2f}% "
                           f"(فترة الثقة 95%: {10 * row['El. CI low']:+.2f}% إلى {10 * row['El. CI high']:+.2f}%).")
        else:
            bullets.append(f"<b>{row['Variable']}</b>: a 10% increase is associated with a {10 * e:+.2f}% change in {y} "
                           f"(95% CI {10 * row['El. CI low']:+.2f}% to {10 * row['El. CI high']:+.2f}%).")
    disc = eff[(eff["Type"] == "discrete") & (eff["ME p-value"] < alpha)]
    for _, row in disc.head(3).iterrows():
        unit = "%" if r.design.info["ytx"] == "log" else (" وحدة" if ar else " units")
        bullets.append((f"<b>{row['Variable']}</b>: فرق قدره {row['Marginal effect dY/dX']:+.2f}{unit} في {y}." if ar else
                        f"<b>{row['Variable']}</b>: a difference of {row['Marginal effect dY/dX']:+.2f}{unit} in {y}."))
    if not bullets:
        bullets.append("لا توجد محددات معنوية إحصائياً عند مستوى المعنوية المختار." if ar else
                       "No statistically significant driver at the chosen significance level.")
    score = 3
    issues = []
    if diag:
        for v in dx.verdicts(diag, lang):
            if (v["status"] == "fail" and v["group"] in ("spec", "hetero", "autocorr")) or \
                    (v["status"] == "mixed" and v["group"] == "spec"):
                if v["group"] == "hetero" and r.cov_type != "nonrobust":
                    continue
                if v["group"] == "autocorr" and r.cov_type == "HAC":
                    continue
                score -= 1
                issues.append(v["label"])
    rating = {3: ("High", "مرتفعة", "🟢"), 2: ("Moderate", "متوسطة", "🟡")}.get(max(score, 1), ("Low", "منخفضة", "🔴"))
    if ar:
        head = (f"يلخّص هذا الموجز نتائج نموذج {r.y_label} على {r.nobs} مشاهدة. أهم المحددات مرتبةً حسب قوة أثرها النسبي:")
        rel = (f"{rating[2]} درجة الاعتماد على النتائج: <b>{rating[1]}</b>"
               + (f" — مشكلات قائمة: {', '.join(issues)}." if issues else " — اجتاز النموذج الاختبارات التشخيصية الأساسية."))
        caution = ("النتائج ارتباطات إحصائية متوسطة على مستوى العينة؛ يُنصح باستخدامها مع الخبرة الميدانية وعدم "
                   "تعميمها خارج نطاق البيانات.")
    else:
        head = f"This brief summarises the model of {r.y_label} estimated on {r.nobs} observations. Key drivers, ranked by relative strength:"
        rel = (f"{rating[2]} Reliability of the evidence: <b>{rating[0]}</b>"
               + (f" — open issues: {', '.join(issues)}." if issues else " — the model passes the core diagnostic tests."))
        caution = ("These are average statistical associations in the sample; combine them with field expertise and "
                   "avoid extrapolating outside the range of the data.")
    return {"head": head, "bullets": bullets, "reliability": rel, "caution": caution, "rating": rating[0]}


# ==========================================================
# Methodology paragraph + references
# ==========================================================
REFERENCES = {
    "ols": "Wooldridge, J. M. (2020). Introductory econometrics: A modern approach (7th ed.). Cengage.",
    "gujarati": "Gujarati, D. N., & Porter, D. C. (2009). Basic econometrics (5th ed.). McGraw-Hill.",
    "greene": "Greene, W. H. (2018). Econometric analysis (8th ed.). Pearson.",
    "white": "White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. Econometrica, 48(4), 817–838.",
    "mackinnon": "MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. Journal of Econometrics, 29(3), 305–325.",
    "long": "Long, J. S., & Ervin, L. H. (2000). Using heteroscedasticity consistent standard errors in the linear regression model. The American Statistician, 54(3), 217–224.",
    "nw": "Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703–708.",
    "cgm": "Cameron, A. C., Gelbach, J. B., & Miller, D. L. (2008). Bootstrap-based improvements for inference with clustered errors. Review of Economics and Statistics, 90(3), 414–427.",
    "efron": "Efron, B., & Tibshirani, R. J. (1993). An introduction to the bootstrap. Chapman & Hall.",
    "bp": "Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. Econometrica, 47(5), 1287–1294.",
    "koenker": "Koenker, R. (1981). A note on studentizing a test for heteroscedasticity. Journal of Econometrics, 17(1), 107–112.",
    "glejser": "Glejser, H. (1969). A new test for heteroskedasticity. Journal of the American Statistical Association, 64(325), 316–323.",
    "park": "Park, R. E. (1966). Estimation with heteroscedastic error terms. Econometrica, 34(4), 888.",
    "gq": "Goldfeld, S. M., & Quandt, R. E. (1965). Some tests for homoscedasticity. Journal of the American Statistical Association, 60(310), 539–547.",
    "dw": "Durbin, J., & Watson, G. S. (1951). Testing for serial correlation in least squares regression II. Biometrika, 38(1/2), 159–177.",
    "imhof": "Imhof, J. P. (1961). Computing the distribution of quadratic forms in normal variables. Biometrika, 48(3/4), 419–426.",
    "bg": "Godfrey, L. G. (1978). Testing against general autoregressive and moving average error models when the regressors include lagged dependent variables. Econometrica, 46(6), 1293–1301.",
    "lb": "Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297–303.",
    "durbin": "Durbin, J. (1970). Testing for serial correlation in least-squares regression when some of the regressors are lagged dependent variables. Econometrica, 38(3), 410–421.",
    "reset": "Ramsey, J. B. (1969). Tests for specification errors in classical linear least-squares regression analysis. Journal of the Royal Statistical Society B, 31(2), 350–371.",
    "chow": "Chow, G. C. (1960). Tests of equality between sets of coefficients in two linear regressions. Econometrica, 28(3), 591–605.",
    "bde": "Brown, R. L., Durbin, J., & Evans, J. M. (1975). Techniques for testing the constancy of regression relationships over time. Journal of the Royal Statistical Society B, 37(2), 149–192.",
    "jb": "Jarque, C. M., & Bera, A. K. (1987). A test for normality of observations and regression residuals. International Statistical Review, 55(2), 163–172.",
    "bkw": "Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). Regression diagnostics: Identifying influential data and sources of collinearity. Wiley.",
    "cook": "Cook, R. D. (1977). Detection of influential observation in linear regression. Technometrics, 19(1), 15–18.",
    "boxcox": "Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. Journal of the Royal Statistical Society B, 26(2), 211–252.",
    "zarembka": "Zarembka, P. (1968). Functional form in the demand for money. Journal of the American Statistical Association, 63(322), 502–511.",
    "harvey": "Harvey, A. C. (1976). Estimating regression models with multiplicative heteroscedasticity. Econometrica, 44(3), 461–465.",
    "co": "Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression to relationships containing auto-correlated error terms. Journal of the American Statistical Association, 44(245), 32–61.",
    "huber": "Huber, P. J. (1964). Robust estimation of a location parameter. Annals of Mathematical Statistics, 35(1), 73–101.",
    "duan": "Duan, N. (1983). Smearing estimate: A nonparametric retransformation method. Journal of the American Statistical Association, 78(383), 605–610.",
    "kennedy": "Kennedy, P. E. (1981). Estimation with correctly interpreted dummy variables in semilogarithmic equations. American Economic Review, 71(4), 801.",
    "akaike": "Akaike, H. (1974). A new look at the statistical model identification. IEEE Transactions on Automatic Control, 19(6), 716–723.",
    "schwarz": "Schwarz, G. (1978). Estimating the dimension of a model. Annals of Statistics, 6(2), 461–464.",
    "theil": "Theil, H. (1971). Principles of econometrics. Wiley.",
    "baron": "Baron, R. M., & Kenny, D. A. (1986). The moderator–mediator variable distinction in social psychological research. Journal of Personality and Social Psychology, 51(6), 1173–1182.",
    "hayes": "Preacher, K. J., & Hayes, A. F. (2008). Asymptotic and resampling strategies for assessing and comparing indirect effects in multiple mediator models. Behavior Research Methods, 40(3), 879–891.",
    "jn": "Johnson, P. O., & Neyman, J. (1936). Tests of certain linear hypotheses and their application to some educational problems. Statistical Research Memoirs, 1, 57–93.",
    "mallows": "Mallows, C. L. (1973). Some comments on Cp. Technometrics, 15(4), 661–675.",
    "seber": "Seber, G. A. F., & Wild, C. J. (2003). Nonlinear regression. Wiley.",
}


def methodology(r: rg.RegResult, diag_run: bool = True, lang: str = "en") -> tuple[str, list]:
    """Ready-to-paste methodology paragraph + the references it cites (APA 7)."""
    ar = lang == "ar"
    keys = ["ols", "greene"]
    spec = r.design.spec if r.design else None
    form = rg.FORMS.get(r.form, ("NLS", "NLS", ""))
    eq = form[2]
    xs = ", ".join((spec.x + spec.categorical) if spec else [r.extra.get("x_name", "X")])
    method_txt = r.method_label(lang)
    cov = r.cov_type
    cov_ref = {"HC0": ["white"], "HC1": ["white", "mackinnon"], "HC2": ["mackinnon"], "HC3": ["mackinnon", "long"],
               "HAC": ["nw"], "cluster": ["cgm"], "bootstrap": ["efron"]}.get(cov, [])
    keys += cov_ref
    keys += {"fgls": ["harvey"], "glsar": ["co"], "huber": ["huber"], "bisquare": ["huber"], "rls": ["theil"],
             "nls": ["seber"]}.get(r.method, [])
    if spec and spec.form in ("loglog", "loglin"):
        keys += ["duan"]
        if spec.categorical:
            keys += ["kennedy"]
    if ar:
        txt = (f"لتقدير العلاقة بين {r.y_label} والمتغيرات المفسِّرة ({xs}) اعتمدت الدراسة الصيغة "
               f"«{form[1]}» ({eq})، وقُدِّرت بطريقة {method_txt} على عينة من {r.nobs} مشاهدة. ")
        if spec and spec.categorical:
            txt += ("أُدخلت المتغيرات النوعية في صورة متغيرات صورية مع استبعاد فئة مرجعية لتجنّب فخ المتغيرات الصورية. ")
        if spec and spec.poly:
            txt += "وأُضيفت حدود تربيعية لالتقاط الأثر غير الخطي وحساب نقطة التحول. "
        if spec and spec.interactions:
            txt += "وأُضيفت حدود تفاعلية لاختبار الأثر المعدِّل. "
        txt += f"وحُسبت الأخطاء المعيارية بطريقة {r.cov_label(lang)}. "
        if diag_run:
            txt += ("وقبل الاستدلال خضع النموذج لحزمة كاملة من الاختبارات التشخيصية: عدم تجانس التباين (بروش-باجان "
                    "بصيغة كوينكر، ووايت، وجليجسر، وبارك، وجولدفيلد-كواندت)، والارتباط الذاتي (دربن-واتسون بقيمة احتمالية "
                    "دقيقة بخوارزمية إمهوف، وبروش-جودفري، وليونج-بوكس)، والشكل الدالي (رامزي RESET) واستقرار المعاملات "
                    "(CUSUM وCUSUMSQ)، وطبيعية البواقي (جارك-بيرا وشابيرو-ويلك)، والازدواج الخطي (معامل تضخم التباين "
                    "وأدلة الحالة لبيلسلي)، والمشاهدات المؤثرة (مسافة كوك وDFBETAS). ")
            keys += ["bp", "koenker", "white", "glejser", "park", "gq", "dw", "imhof", "bg", "lb", "reset", "bde",
                     "jb", "bkw", "cook"]
        txt += "واستُخدم معيارا أكايكي وشوارتز للمفاضلة بين النماذج البديلة."
    else:
        txt = (f"To estimate the relationship between {r.y_label} and the explanatory variables ({xs}), the study adopts "
               f"a {form[0].lower()} specification ({eq}) estimated by {method_txt} on {r.nobs} observations. ")
        if spec and spec.categorical:
            txt += "Qualitative factors enter as dummy variables with an omitted reference category. "
        if spec and spec.poly:
            txt += "Quadratic terms capture non-linear effects and identify turning points. "
        if spec and spec.interactions:
            txt += "Interaction terms test for moderating effects. "
        txt += f"Standard errors are {r.cov_label(lang)}. "
        if diag_run:
            txt += ("Before inference the model was subjected to a full diagnostic battery: heteroskedasticity "
                    "(Koenker's studentized Breusch-Pagan, White, Glejser, Park and Goldfeld-Quandt tests), serial "
                    "correlation (Durbin-Watson with exact Imhof p-values, Breusch-Godfrey and Ljung-Box), functional form "
                    "(Ramsey RESET) and parameter stability (CUSUM and CUSUM of squares), residual normality "
                    "(Jarque-Bera, Shapiro-Wilk), multicollinearity (VIF and Belsley condition indices) and influential "
                    "observations (Cook's distance, DFBETAS). ")
            keys += ["bp", "koenker", "white", "glejser", "park", "gq", "dw", "imhof", "bg", "lb", "reset", "bde",
                     "jb", "bkw", "cook"]
        txt += "Akaike and Schwarz information criteria guide the choice among competing specifications."
    keys += ["akaike", "schwarz"]
    refs = sorted({REFERENCES[k] for k in dict.fromkeys(keys)})
    return txt, refs


# ==========================================================
# Reproducible code (Python / R / Stata)
# ==========================================================
def _stata_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    if not re.match(r"[A-Za-z_]", s):
        s = "v_" + s
    return s[:32]


def code_python(r: rg.RegResult, file_name: str = "your_data.csv") -> str:
    spec = r.design.spec
    L = ["import numpy as np", "import pandas as pd", "import statsmodels.api as sm",
         "from statsmodels.stats import diagnostic as dg", "from statsmodels.stats.stattools import durbin_watson",
         "from statsmodels.stats.outliers_influence import variance_inflation_factor", "",
         f'df = pd.read_csv("{file_name}")', ""]
    ytx = r.design.info["ytx"]
    yexpr = f'np.log(df["{spec.y}"])' if ytx == "log" else f'df["{spec.y}"]'
    L.append(f"y = {yexpr}")
    L.append("X = pd.DataFrame(index=df.index)")
    hows = r.design.info["hows"]
    means = r.design.info["means"]

    def term(v):
        how = hows.get(v, "level")
        e = {"log": f'np.log(df["{v}"])', "recip": f'1 / df["{v}"]'}.get(how, f'df["{v}"]')
        if v in means:
            e = f"({e} - {means[v]!r})"
        return e
    for v in spec.x:
        L.append(f'X["{rg._tname(v, hows.get(v, "level"))}"] = {term(v)}')
        for p in range(2, int(spec.poly.get(v, 1)) + 1):
            L.append(f'X["{rg._tname(v, hows.get(v, "level"))}^{p}"] = {term(v)} ** {p}')
    for c in spec.categorical:
        lv = r.design.info["levels"][c]
        L.append(f"for level in {lv[1:]!r}:  # reference level: {lv[0]!r}")
        L.append(f'    X[f"{c}[{{level}}]"] = (df["{c}"].astype(str) == level).astype(float)')
    def comp(col):
        if "[" in col and col.endswith("]") and col.split("[")[0] in spec.categorical:
            c, lvl = col.split("[", 1)
            return f'(df["{c}"].astype(str) == {lvl[:-1]!r}).astype(float)'
        for v in spec.all_vars():
            if rg._tname(v, hows.get(v, "level")) == col:
                return term(v)
        return f'X["{col}"]'
    for g, cols in r.design.info["groups"].items():
        if "×" not in g or g in spec.x:
            continue
        L.append(f"# interaction {g}")
        for iname in cols:
            a_, b_ = iname.split("×", 1)
            L.append(f'X["{iname}"] = {comp(a_)} * {comp(b_)}')
    L.append("X = sm.add_constant(X)" if spec.const else "")
    L.append("data = pd.concat([y.rename('y'), X], axis=1).replace([np.inf, -np.inf], np.nan).dropna()")
    L.append("y, X = data['y'], data.drop(columns='y')")
    L.append(f"X = X[{r.names!r}]  # final columns used by Aetheris (collinear / constant columns removed)")
    cov = r.cov_type
    fit = {"nonrobust": "fit()", "HC0": 'fit(cov_type="HC0", use_t=True)', "HC1": 'fit(cov_type="HC1", use_t=True)',
           "HC2": 'fit(cov_type="HC2", use_t=True)', "HC3": 'fit(cov_type="HC3", use_t=True)',
           "HAC": f'fit(cov_type="HAC", cov_kwds={{"maxlags": {r.cov_opts.get("maxlags", 1)}}}, use_t=True)',
           "cluster": f'fit(cov_type="cluster", cov_kwds={{"groups": pd.factorize(df.loc[X.index, "{r.cov_opts.get("groups_name", "cluster_var")}"])[0]}}, use_t=True)',
           "bootstrap": "fit()  # bootstrap SEs: resample rows with replacement and refit (see Aetheris output)"}[cov]
    model = {"ols": "sm.OLS(y, X)", "wls": "sm.WLS(y, X, weights=w)  # w = your weights column",
             "fgls": "sm.WLS(y, X, weights=1 / np.exp(sm.OLS(np.log(sm.OLS(y, X).fit().resid ** 2), X).fit().fittedvalues))",
             "glsar": "sm.GLSAR(y, X, rho=1)", "huber": "sm.RLM(y, X, M=sm.robust.norms.HuberT())",
             "bisquare": "sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight())", "rls": "sm.OLS(y, X)"}.get(r.method, "sm.OLS(y, X)")
    L.append(f"model = {model}")
    if r.method == "glsar":
        L.append("res = model.iterative_fit(maxiter=50)")
    elif r.method in ("huber", "bisquare"):
        L.append("res = model.fit()")
    else:
        L.append(f"res = model.{fit}")
    if r.method == "rls":
        L.append(f'print(res.f_test("{"; ".join(r.extra.get("restrictions", []))}"))  # restrictions (use the column names)')
    L += ["print(res.summary())", "", "# ---- diagnostics", "e, Z = res.resid, X",
          'print("Breusch-Pagan (Koenker):", dg.het_breuschpagan(e, Z, robust=True)[:2])',
          'print("White:", dg.het_white(e, Z)[:2])',
          'print("Durbin-Watson:", durbin_watson(e))',
          'print("Breusch-Godfrey (2 lags):", dg.acorr_breusch_godfrey(sm.OLS(y, X).fit(), nlags=2)[:2])',
          'print("RESET:", dg.linear_reset(sm.OLS(y, X).fit(), power=3, use_f=True))',
          'print("Jarque-Bera:", sm.stats.jarque_bera(e)[:2])',
          'print("VIF:", {c: variance_inflation_factor(X.values, i) for i, c in enumerate(X.columns) if c != "const"})']
    return "\n".join(x for x in L if x is not None) + "\n"


def code_r(r: rg.RegResult, file_name: str = "your_data.csv") -> str:
    spec = r.design.spec
    hows = r.design.info["hows"]
    means = r.design.info["means"]

    def q(v):
        return f"`{v}`"

    def term(v):
        how = hows.get(v, "level")
        e = {"log": f"log({q(v)})", "recip": f"1/{q(v)}"}.get(how, q(v))
        if v in means:
            e = f"({e} - {means[v]:.6g})"
        return e
    ytx = r.design.info["ytx"]
    lhs = f"log({q(spec.y)})" if ytx == "log" else q(spec.y)
    rhs = []
    for v in spec.x:
        t = term(v)
        rhs.append(f"I({t})" if t != q(v) else t)
        for p in range(2, int(spec.poly.get(v, 1)) + 1):
            rhs.append(f"I({t}^{p})")
    for c in spec.categorical:
        rhs.append(f"relevel(factor({q(c)}), ref = \"{r.design.info['levels'][c][0]}\")")
    for a, b in spec.interactions:
        ta = f"relevel(factor({q(a)}), ref = \"{r.design.info['levels'][a][0]}\")" if a in spec.categorical else f"I({term(a)})"
        tb = f"relevel(factor({q(b)}), ref = \"{r.design.info['levels'][b][0]}\")" if b in spec.categorical else f"I({term(b)})"
        rhs.append(f"{ta}:{tb}")
    f = f"{lhs} ~ " + (" + ".join(rhs) or "1") + ("" if spec.const else " - 1")
    L = ["# install.packages(c('lmtest', 'sandwich', 'car', 'tseries', 'MASS', 'strucchange'))",
         "library(lmtest); library(sandwich); library(car); library(tseries); library(strucchange)", "",
         f'df <- read.csv("{file_name}", check.names = FALSE)']
    call = {"ols": f"lm({f}, data = df)", "rls": f"lm({f}, data = df)",
            "wls": f"lm({f}, data = df, weights = w)  # w = your weights column",
            "fgls": f"lm({f}, data = df, weights = 1 / exp(fitted(lm(log(resid(lm({f}, data = df))^2) ~ ., data = model.frame(lm({f}, data = df))[-1]))))",
            "glsar": f"orcutt::cochrane.orcutt(lm({f}, data = df))  # install.packages('orcutt')",
            "huber": f"MASS::rlm({f}, data = df, psi = MASS::psi.huber)",
            "bisquare": f"MASS::rlm({f}, data = df, psi = MASS::psi.bisquare)"}.get(r.method, f"lm({f}, data = df)")
    L.append(f"m <- {call}")
    vc = {"nonrobust": "vcov(m)", "HC0": 'vcovHC(m, type = "HC0")', "HC1": 'vcovHC(m, type = "HC1")',
          "HC2": 'vcovHC(m, type = "HC2")', "HC3": 'vcovHC(m, type = "HC3")',
          "HAC": f"NeweyWest(m, lag = {r.cov_opts.get('maxlags', 1)}, prewhite = FALSE)",
          "cluster": f"vcovCL(m, cluster = ~ {q(r.cov_opts.get('groups_name', 'cluster_var'))})",
          "bootstrap": "vcovBS(m, R = 999)"}[r.cov_type]
    L += [f"V <- {vc}", "coeftest(m, vcov. = V)", "coefci(m, vcov. = V)", "summary(m)", "",
          "# ---- diagnostics", "bptest(m)                      # Breusch-Pagan (Koenker)",
          "bptest(m, ~ fitted(m) + I(fitted(m)^2))  # White (special form)", "gqtest(m, fraction = 0.2)",
          "dwtest(m)                      # exact Durbin-Watson", "bgtest(m, order = 2)",
          "Box.test(resid(m), lag = 10, type = 'Ljung-Box')", "resettest(m, power = 2:3, type = 'fitted')",
          "jarque.bera.test(resid(m)); shapiro.test(resid(m))", "vif(m)",
          "plot(efp(formula(m), data = df, type = 'Rec-CUSUM'))", "influence.measures(m)"]
    if r.method == "rls":
        L.append(f'linearHypothesis(m, c({", ".join(chr(34) + s + chr(34) for s in r.extra.get("restrictions", []))}))')
    return "\n".join(L) + "\n"


def code_stata(r: rg.RegResult, file_name: str = "your_data.csv") -> str:
    spec = r.design.spec
    hows = r.design.info["hows"]
    means = r.design.info["means"]
    S = _stata_name
    L = [f'import delimited "{file_name}", clear varnames(1)', "* Variable names are converted to Stata-compatible names"]
    ytx = r.design.info["ytx"]
    yv = S(spec.y)
    if ytx == "log":
        L.append(f"gen ln_{yv} = ln({yv})")
        yv = f"ln_{yv}"
    xs = []
    for v in spec.x:
        how = hows.get(v, "level")
        base = S(v)
        name = {"log": f"ln_{base}", "recip": f"inv_{base}"}.get(how, base)
        expr = {"log": f"ln({base})", "recip": f"1/{base}"}.get(how, base)
        if v in means:
            expr = f"({expr}) - {means[v]:.6g}"
            name = "c_" + name
        if name != base:
            L.append(f"gen {name[:32]} = {expr}")
        xs.append(name[:32])
        if int(spec.poly.get(v, 1)) >= 2:
            xs[-1] = f"c.{name[:32]}##c.{name[:32]}" if int(spec.poly[v]) == 2 else xs[-1]
    for c in spec.categorical:
        lv = r.design.info["levels"][c]
        ref_pos = sorted(lv, key=str).index(lv[0]) + 1
        L.append(f"encode {S(c)}, gen({S(c)[:28]}_n)   // reference level: {lv[0]}")
        xs.append(f"ib{ref_pos}.{S(c)[:28]}_n")
    for a, b in spec.interactions:
        ta = f"i.{S(a)[:28]}_n" if a in spec.categorical else f"c.{xs[spec.x.index(a)].split('##')[0].replace('c.', '')}" if a in spec.x else f"c.{S(a)}"
        tb = f"i.{S(b)[:28]}_n" if b in spec.categorical else f"c.{xs[spec.x.index(b)].split('##')[0].replace('c.', '')}" if b in spec.x else f"c.{S(b)}"
        xs.append(f"{ta}#{tb}")
    vce = {"nonrobust": "", "HC0": ", vce(robust)", "HC1": ", vce(robust)", "HC2": ", vce(hc2)", "HC3": ", vce(hc3)",
           "cluster": f", vce(cluster {S(r.cov_opts.get('groups_name', 'cluster_var'))})",
           "bootstrap": ", vce(bootstrap, reps(999) seed(2026))", "HAC": ""}[r.cov_type]
    rhs = " ".join(xs)
    if r.cov_type == "HAC":
        L += ["gen t = _n", "tsset t", f"newey {yv} {rhs}, lag({r.cov_opts.get('maxlags', 1)})"]
    elif r.method == "wls":
        L.append(f"regress {yv} {rhs} [aweight = w]{vce}  // w = your weights variable")
    elif r.method in ("huber", "bisquare"):
        L.append(f"rreg {yv} {rhs}")
    elif r.method == "glsar":
        L += ["gen t = _n", "tsset t", f"prais {yv} {rhs}, corc"]
    else:
        L.append(f"regress {yv} {rhs}{vce}")
    if r.method in ("ols", "rls", "fgls") and r.cov_type in ("nonrobust",):
        L += ["estat hettest            // Breusch-Pagan", "estat hettest, iid      // Koenker",
              "estat imtest, white      // White", "estat ovtest             // RESET", "estat vif",
              "predict e, resid", "sktest e", "swilk e", "predict cooksd, cooksd", "dfbeta",
              "gen t = _n", "tsset t", "estat dwatson", "estat bgodfrey, lags(1 2)"]
    else:
        L += ["* Run the diagnostics on the OLS fit: regress ... then estat hettest / estat imtest, white /",
              "* estat ovtest / estat vif / estat bgodfrey"]
    if r.method == "rls":
        L.append("* Restricted LS: constraint define 1 <expr> ; cnsreg y x..., constraints(1)")
    return "\n".join(L) + "\n"
