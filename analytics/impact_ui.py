# ==========================================
# Aetheris Research - Impact, Spatial & Climate Lab (UI) — v1.11, Catalogue §1.14, §1.16, §1.20, §1.26
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import causal as cz
from analytics import charts as ch
from analytics import climate as cl
from analytics import descriptive as ds
from analytics import impact_report as ir
from analytics import quantile as qn
from analytics import reg_report as rp
from analytics import spatial as sp
from analytics import ts_ui as tu
from core.memory import active_name, get_memory, has_data, set_memory
from core.utils import tr, is_ar, insight, excel_download

PAL = ch.PALETTES["academic"]
_disp, _code, _lang = tu._disp, tu._code, tu._lang
fp = ir.fmt_p
tu._AR_COLS.update({
    "τ": "τ", "Variable": "المتغير", "Coef.": "المعامل", "Std. Err.": "الخطأ المعياري", "CI low": "الحد الأدنى", "CI high": "الحد الأعلى",
    "pseudo R¹": "R¹ الكاذب", "Wald": "والد", "df": "د.ح", "F": "F", "Location β": "الموقع β", "Scale γ": "المقياس γ",
    "Std. Err. (bootstrap)": "الخطأ (بوتستراب)", "lag": "الفجوة", "ρ": "ρ", "significant": "معنوي", "Q(p)": "Q(p)", "p (Q)": "p (Q)",
    "Test": "الاختبار", "Statistic": "الإحصاءة", "Parameter": "المعلمة", "Impact": "الأثر", "Estimate": "التقدير",
    "Std. Err. (simulated)": "الخطأ (محاكاة)", "Model": "النموذج", "Log-lik.": "دالة الإمكان", "Mean": "المتوسط", "SD": "الانحراف",
    "Min": "الأدنى", "Median": "الوسيط", "Max": "الأعلى", "% |t| > 1.96": "% |t| > 1.96", "Term": "الحد", "t / z": "t / z",
    "Estimator": "المقدِّر", "Covariate": "المتغير المصاحب", "Mean treated": "متوسط المعالجين", "Mean control": "متوسط الضابطة",
    "Difference": "الفرق", "Std. mean diff.": "الفرق المعياري", "Variance ratio": "نسبة التباين", "p-value (Welch)": "p (ولش)",
    "Balanced (|SMD|<0.1)": "متوازن", "SMD before": "SMD قبل", "SMD after matching": "SMD بعد المطابقة", "SMD after weighting": "SMD بعد الترجيح",
    "Group": "المجموعة", "Time": "الزمن", "ATT(g,t)": "ATT(g,t)", "Post": "بعد المعالجة", "Aggregation": "التجميع", "ATT": "ATT",
    "Event time": "زمن الحدث", "Relative period": "الفترة النسبية", "Comparison": "المقارنة", "Treated": "المعالَج", "Control": "الضابطة",
    "Weight": "الوزن", "Predictor": "المحدِّد", "Synthetic": "التركيبي", "Donor average": "متوسط المانحين", "V weight": "وزن V",
    "post/pre RMSPE": "نسبة RMSPE بعد/قبل", "Method": "الطريقة", "Bandwidth × h*": "عرض النطاق × h*", "Robust CI low": "الحد الأدنى المتين",
    "Robust CI high": "الحد الأعلى المتين", "Robust p": "p المتين", "N eff": "N الفعّال", "Placebo cutoff": "نقطة القطع الوهمية", "Side": "الجانب",
    "Jump": "القفزة", "Warming (°C)": "الاحترار (°م)", "Δ outcome": "Δ النتيجة", "% change": "% التغير", "% CI low": "% الحد الأدنى",
    "% CI high": "% الحد الأعلى", "Climate variable": "المتغير المناخي", "Marginal effect at mean": "الأثر الحدي عند المتوسط",
    "Shape": "الشكل", "Turning point": "نقطة التحول", "Baseline T": "حرارة الأساس", "GDP/cap end (no warming)": "الناتج للفرد (دون احترار)",
    "GDP/cap end (warming)": "الناتج للفرد (مع الاحترار)", "Ridge elasticity": "مرونة الحافة", "λ (GCV)": "λ (GCV)", "VIF": "VIF",
    "Unit": "الوحدة", "Period": "الفترة", "%Δ emissions": "%Δ الانبعاثات", "%Δ GDP": "%Δ الناتج", "Elasticity": "المرونة", "State": "الحالة",
    "Factor": "العامل", "Effect": "الأثر", "Year": "السنة", "Peak": "الذروة", "Bin": "الفئة", "Effect of one day": "أثر يوم إضافي",
    "Temperature": "الحرارة", "Cohort": "الفوج", "Ii": "Ii", "cluster": "العنقود", "quadrant": "الربع", "p (permutation)": "p (تباديل)",
})
tu._AR_VALS.update({
    "Direct": "مباشر", "Indirect (spillover)": "غير مباشر (انتشاري)", "Total": "كلي", "HH": "مرتفع-مرتفع", "LL": "منخفض-منخفض",
    "HL": "مرتفع-منخفض", "LH": "منخفض-مرتفع", "not significant": "غير معنوي", "Conventional": "تقليدي", "Bias-corrected": "مصحح التحيز",
    "Robust": "متين", "left": "يسار", "right": "يمين", "Difference in means": "الفرق بين المتوسطات", "Strata fixed effects": "آثار ثابتة للطبقات",
    "ANCOVA (additive)": "ANCOVA (جمعي)", "Lin (interacted)": "لين (متفاعل)", "Simple (weighted by group size)": "بسيط (موزون بحجم المجموعة)",
    "Dynamic (average over e ≥ 0)": "ديناميكي (متوسط e ≥ 0)", "Group (selective timing)": "حسب المجموعة", "Calendar time": "الزمن التقويمي",
    "Treated vs never treated": "معالَج مقابل غير معالَج أبداً", "Earlier vs later treated (later as control)": "مبكر مقابل متأخر (المتأخر ضابطة)",
    "Later vs earlier treated (earlier as control — 'forbidden')": "متأخر مقابل مبكر (المبكر ضابطة — «ممنوعة»)",
    "hill (∩)": "قبّة (∩)", "U-shaped": "على شكل U", "Population": "السكان", "Affluence (GDP/capita)": "الرخاء (الناتج للفرد)",
    "Energy intensity (E/GDP)": "كثافة الطاقة (الطاقة/الناتج)", "Carbon intensity (CO2/E)": "كثافة الكربون (CO2/الطاقة)",
    "Total ΔCO2": "إجمالي ΔCO2", "Residual": "الباقي", "never": "لم يُعالَج أبداً",
})


def _t(pair):
    return pair[1] if is_ar() else pair[0]


def _chart(fig, key, name):
    st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, name), key=key)


def _table(df, digits=4):
    show = df.copy()
    for c in show.columns:
        if str(c) in ("p-value", "p", "Robust p", "p (Q)") or str(c).startswith("p ("):
            show[c] = show[c].map(lambda v: fp(v) if isinstance(v, (float, np.floating)) else v)
    kw = {"height": 35 * (len(show) + 1) + 3} if len(show) <= 18 else {}
    st.dataframe(_disp(show).round(digits), width="stretch", hide_index=True, **kw)


def _store(key, v, sig):
    st.session_state[f"_v111res_{key}"] = (sig, v)


def _load(key, sig):
    v = st.session_state.get(f"_v111res_{key}")
    return v[1] if isinstance(v, tuple) and v[0] == sig else None


def _finish(kind, key, sheets: dict, title, docx_tables=(), code_kw=None):
    txt, refs = ir.methodology(kind, _lang())
    with st.expander(tr("📝 Methodology paragraph & references", "📝 فقرة المنهجية والمراجع"), expanded=False):
        st.text_area("m", txt, height=150, key=f"{key}_meth", label_visibility="collapsed")
        st.markdown("\n".join(f"- {x}" for x in refs))
    c1, c2 = st.columns(2)
    with c1:
        excel_download({k: v for k, v in sheets.items() if isinstance(v, pd.DataFrame)}, f"{key}.xlsx", key=f"{key}_xl")
    with c2:
        items = [("h", tr("Methodology", "المنهجية")), ("p", txt)]
        for t_ in docx_tables:
            items.append(("table", t_.round(4).astype(str)))
        items.append(("bullets", refs))
        st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"), rp.to_docx(items, title, is_ar()), f"{key}.docx", key=f"{key}_docx")
    _code(ir.code(kind, **(code_kw or {})), f"{key}_code")


_IDLIKE = ("year", "round", "quarter", "month", "_id", "id_", "period", "first_treated", "cohort", "season")


def _num(df, exclude=()):
    tc = set(tu._time_cols(df))
    out = []
    for c in ds.numeric_columns(df):
        lc = c.lower()
        if c in exclude or c in tc or lc in ("id", "year") or any(k in lc for k in _IDLIKE):
            continue
        out.append(c)
    return out


def _prefer(cols, keys):
    for k in keys:
        for c in cols:
            if k in c.lower():
                return [c] + [x for x in cols if x != c]
    return list(cols)


def _binary(df):
    return [c for c in ds.numeric_columns(df) if set(pd.unique(df[c].dropna())) <= {0, 1} and df[c].nunique() == 2]


def _cats(df, maxlev=400):
    return [c for c in df.columns if 1 < df[c].nunique() <= maxlev and (df[c].dtype == object or str(df[c].dtype) == "category" or "id" in c.lower()
                                                                         or c.lower() in ("district", "country", "region", "village", "station", "unit"))]


def _coords(df):
    cols = list(df.columns)
    lon = next((c for c in cols if c.lower() in ("longitude", "lon", "long", "x_coord", "x")), None)
    lat = next((c for c in cols if c.lower() in ("latitude", "lat", "y_coord", "y")), None)
    return lon, lat


DEMOS = [("spatial_districts_demo", ("🗺️ Districts: land values (spatial)", "🗺️ المراكز: قيم الأراضي (مكاني)"), "Demo – spatial districts (synthetic)"),
         ("spatial_panel_demo", ("🧱 Districts × years: yields (spatial panel)", "🧱 مراكز × سنوات: الإنتاجية (بانل مكاني)"), "Demo – spatial panel (synthetic)"),
         ("rct_extension_demo", ("🎲 Extension RCT", "🎲 تجربة عشوائية للإرشاد"), "Demo – extension RCT (synthetic)"),
         ("staggered_did_demo", ("⏱️ Staggered irrigation reform (DiD)", "⏱️ إصلاح ري متدرج (فرق الفروق)"), "Demo – staggered DiD (synthetic)"),
         ("scm_regions_demo", ("🧬 Regional policy (synthetic control)", "🧬 سياسة إقليمية (الضابط التركيبي)"), "Demo – synthetic control (synthetic)"),
         ("rdd_scholarship_demo", ("✂️ Scholarship cutoff (RDD)", "✂️ حد المنحة (انقطاع الانحدار)"), "Demo – RDD scholarship (synthetic)"),
         ("credit_program_demo", ("⚖️ Micro-credit programme (matching)", "⚖️ برنامج تمويل أصغر (مطابقة)"), "Demo – credit programme (synthetic)"),
         ("crop_climate_panel_demo", ("🌡️ Crop yields & weather panel", "🌡️ بانل الإنتاجية والطقس"), "Demo – crop-climate panel (synthetic)"),
         ("daily_weather_demo", ("☀️ Daily weather stations", "☀️ محطات الطقس اليومية"), "Demo – daily weather (synthetic)"),
         ("ricardian_demo", ("🏞️ Farm land values (Ricardian)", "🏞️ قيم الأراضي الزراعية (ريكاردي)"), "Demo – Ricardian farms (synthetic)"),
         ("country_emissions_demo", ("🏭 Countries: GDP, energy & CO2", "🏭 الدول: الناتج والطاقة وCO2"), "Demo – country emissions (synthetic)")]


def _demo_buttons(key):
    import data.samples as smp
    cols = st.columns(3)
    for i, (fn, lab, nm) in enumerate(DEMOS):
        with cols[i % 3]:
            if st.button(_t(lab), key=f"{key}_{fn}", width="stretch"):
                set_memory(getattr(smp, fn)(), source="demo (synthetic)", name=nm)
                st.rerun()
    st.caption(tr("⚠️ Simulated data for training only — never cite as real statistics.",
                  "⚠️ بيانات محاكاة للتدريب فقط — لا تُستشهد بها كإحصاءات حقيقية."))


# ==========================================================
def render_impact():
    st.caption("💡 " + tr("Distributional, spatial and causal evidence for policy: who gains at each quantile, how effects spill over "
                         "space, what a programme really caused, and what climate change costs.",
                         "أدلة توزيعية ومكانية وسببية لصنع السياسات: من يستفيد عند كل كمية، وكيف تنتشر الآثار مكانياً، وما الأثر الحقيقي "
                         "للبرنامج، وكم يكلّف تغير المناخ."))
    if not has_data():
        st.info(tr("Load data first (Data Hub), or try a demo dataset:", "حمّل بيانات أولاً (مركز البيانات)، أو جرّب بيانات تجريبية:"))
        _demo_buttons("im0")
        return
    df = get_memory()
    st.caption(tr(f"Active dataset: **{active_name()}** — {df.shape[0]} rows × {df.shape[1]} columns",
                  f"البيانات النشطة: **{active_name()}** — {df.shape[0]} صف × {df.shape[1]} عمود"))
    with st.expander(tr("🧪 Demo datasets for training", "🧪 بيانات تجريبية للتدريب"), expanded=False):
        _demo_buttons("im1")
    tabs = st.tabs([tr("📐 Quantile regression", "📐 الانحدار الكمّي"), tr("🗺️ Spatial autocorrelation", "🗺️ الارتباط المكاني"),
                    tr("🧮 Spatial regression", "🧮 الانحدار المكاني"), tr("🎲 RCT & IV", "🎲 التجارب والمتغيرات الأداتية"),
                    tr("⏱️ DiD & event studies", "⏱️ فرق الفروق ودراسات الحدث"), tr("🧬 Synthetic control", "🧬 الضابط التركيبي"),
                    tr("✂️ Regression discontinuity", "✂️ انقطاع الانحدار"), tr("⚖️ Matching & weighting", "⚖️ المطابقة والترجيح"),
                    tr("🌡️ Climate impacts", "🌡️ آثار المناخ"), tr("🏭 Emissions & damages", "🏭 الانبعاثات والأضرار")])
    fns = [_quantile, _spatial_auto, _spatial_reg, _rct_iv, _did, _synth, _rdd, _matching, _climate, _emissions]
    for tab, fn in zip(tabs, fns):
        with tab:
            try:
                fn(df)
            except Exception as e:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                st.warning(tr(f"This analysis could not run with the current data / selection ({e}). Check the variables, or load the matching demo dataset.",
                              f"تعذّر تشغيل هذا التحليل بالبيانات أو الاختيارات الحالية ({e}). راجع المتغيرات، أو حمّل البيانات التجريبية المناسبة."))


# ---------------------------------------------------------- 1. quantile regression
def _quantile(df):
    num = _num(df)
    mode = st.radio(tr("Analysis", "التحليل"), ["qr", "mmqr", "cq"], horizontal=True, key="qn_mode",
                    format_func=lambda k: {"qr": tr("Quantile process", "عملية الكميات"), "mmqr": tr("Panel MMQR", "MMQR للبانل"),
                                           "cq": tr("Cross-quantilogram", "مخطط الكميات المتقاطع")}[k])
    if mode == "cq":
        return _cq(df, num)
    c1, c2, c3 = st.columns([1, 2, 1])
    y = c1.selectbox(tr("Dependent variable", "المتغير التابع"), _prefer(num, ("ln_", "yield", "income", "value", "gpa")), key="qn_y")
    xs = c2.multiselect(tr("Regressors", "المتغيرات المفسرة"), [c for c in num if c != y], [c for c in num if c != y][:2], key="qn_x",
                        placeholder=tr("Choose…", "اختر…"))
    if not xs:
        return
    if mode == "mmqr":
        units = _cats(df)
        if not units:
            st.info(tr("MMQR needs a unit identifier (panel data).", "تحتاج MMQR إلى معرّف للوحدات (بيانات بانل)."))
            return
        u = c3.selectbox(tr("Unit", "الوحدة"), units, key="qn_u")
        boot = st.slider(tr("Cluster-bootstrap replications", "تكرارات البوتستراب العنقودي"), 50, 400, 100, 50, key="qn_b")
        sig = (u, y, tuple(xs), boot, active_name())
        r = _load("mmqr", sig)
        if r is None:
            with st.spinner(tr("Bootstrapping…", "جارٍ البوتستراب…")):
                r = qn.mmqr(df, u, y, xs, boot=boot)
            _store("mmqr", r, sig)
        _table(r["location_scale"], 4)
        _table(r["table"], 4)
        T = r["table"]
        fig = go.Figure()
        for j, v in enumerate(xs):
            s = T[T.Variable == v]
            fig.add_trace(go.Scatter(x=s["τ"], y=s["Coef."], name=v, mode="lines+markers", line=dict(color=PAL[j % len(PAL)], width=2.5),
                                     error_y=dict(type="data", array=1.96 * s["Std. Err. (bootstrap)"], visible=True)))
        ch.style_academic(fig, height=380, x_title="τ", y_title=tr("Effect", "الأثر"), legend="bottom", rtl=is_ar())
        _chart(fig, "qn_mm", "mmqr")
        ls = r["location_scale"]
        top = ls.iloc[np.argmax(np.abs(ls["Scale γ"]))]
        insight(tr(f"Scale effect of {top.Variable} = {top['Scale γ']:.4f}: " + ("it widens the outcome distribution — effects grow at upper quantiles."
                   if top["Scale γ"] > 0 else "it compresses the distribution — effects are larger at lower quantiles.") +
                   f" Positive fitted scale for {r['positive scale %']:.0f}% of observations.",
                   f"أثر المقياس لـ {top.Variable} = {top['Scale γ']:.4f}: " + ("يوسّع توزيع النتيجة — تكبر الآثار عند الكميات العليا."
                   if top["Scale γ"] > 0 else "يضغط التوزيع — تكبر الآثار عند الكميات الدنيا.") + f" المقياس المقدر موجب في {r['positive scale %']:.0f}% من المشاهدات."))
        _finish("mmqr", "qn_mmqr", {"MMQR": T, "Location-scale": ls}, tr("Method of moments quantile regression", "انحدار الكميات بالعزوم"), [T],
                {"y": y, "x": xs, "unit": u})
        return
    se = c3.selectbox(tr("Standard errors", "الأخطاء المعيارية"), list(qn.SE_TYPES), key="qn_se", format_func=lambda k: _t(qn.SE_TYPES[k]))
    sig = (y, tuple(xs), se, active_name())
    r = _load("qp", sig)
    if r is None:
        with st.spinner(tr("Estimating the quantile process…", "جارٍ تقدير عملية الكميات…")):
            r = qn.qr_process(df, y, xs, se=se)
        _store("qp", r, sig)
    T, O = r["table"], r["ols"]
    pick = T[T["τ"].isin([0.1, 0.25, 0.5, 0.75, 0.9])]
    wide = pick.pivot(index="Variable", columns="τ", values="Coef.").reindex(["const"] + xs)
    wse = pick.pivot(index="Variable", columns="τ", values="Std. Err.").reindex(["const"] + xs)
    from scipy import stats as _st
    pv = pd.DataFrame(2 * _st.norm.sf(np.abs((wide / wse).to_numpy(float))), index=wide.index, columns=wide.columns)
    jt = wide.copy().astype(object)
    for i in wide.index:
        for c in wide.columns:
            p = pv.loc[i, c]
            jt.loc[i, c] = f"{wide.loc[i, c]:.4f}{'***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''}\n({wse.loc[i, c]:.4f})"
    jt.columns = [f"Q{int(round(100 * c))}" for c in jt.columns]
    jt.insert(0, "OLS", [f"{b:.4f}" for b in O["Coef."]])
    st.markdown("**" + tr("Journal table: coefficients (standard errors) by quantile", "جدول المجلات: المعاملات (الأخطاء المعيارية) حسب الكمية") + "**")
    st.dataframe(_disp(jt.reset_index()), width="stretch", hide_index=True)
    eq = qn.slope_equality(df, y, xs)
    st.markdown("**" + tr("Equality of slopes across τ = 0.25, 0.5, 0.75 (Koenker-Bassett)", "تساوي الميول عبر الكميات 0.25 و0.5 و0.75 (كوينكر-باسيت)") + "**")
    _table(eq, 4)
    cols = st.columns(min(len(xs), 3))
    for j, v in enumerate(xs):
        s = T[T.Variable == v]
        o = O[O.Variable == v].iloc[0]
        fig = go.Figure([go.Scatter(x=np.r_[s["τ"], s["τ"][::-1]], y=np.r_[s["CI high"], s["CI low"][::-1]], fill="toself", fillcolor="rgba(31,78,121,0.18)",
                                    line=dict(width=0), name="95% CI", hoverinfo="skip"),
                         go.Scatter(x=s["τ"], y=s["Coef."], mode="lines+markers", line=dict(color=PAL[0], width=2.5), name=tr("Quantile", "كمّي"))])
        fig.add_hline(y=o["Coef."], line=dict(color=PAL[1], dash="dash"), annotation_text="OLS")
        fig.add_hline(y=o["CI low"], line=dict(color=PAL[1], dash="dot", width=1))
        fig.add_hline(y=o["CI high"], line=dict(color=PAL[1], dash="dot", width=1))
        fig.add_hline(y=0, line=dict(color="#999", width=1))
        ch.style_academic(fig, height=330, x_title="τ", y_title=v, legend="none", rtl=is_ar())
        with cols[j % len(cols)]:
            _chart(fig, f"qn_pl_{j}", f"quantile_process_{v}")
    ej = eq.iloc[-1]
    v0 = xs[0]
    lo, hi = wide.loc[v0, 0.1], wide.loc[v0, 0.9]
    insight(tr(f"For {v0}, the effect is {lo:.4f} at the 10th and {hi:.4f} at the 90th percentile. Joint slope-equality test p = {fp(ej['p-value'])}: "
               + ("effects differ across the distribution — OLS hides heterogeneity." if ej["p-value"] < 0.05 else "no strong evidence that effects differ across quantiles."),
               f"أثر {v0} يساوي {lo:.4f} عند المئين العاشر و{hi:.4f} عند المئين التسعين. اختبار تساوي الميول المشترك p = {fp(ej['p-value'])}: "
               + ("تختلف الآثار عبر التوزيع — يُخفي OLS عدم التجانس." if ej["p-value"] < 0.05 else "لا دليل قوي على اختلاف الآثار عبر الكميات.")))
    with st.expander(tr("🎯 Policy simulation: who gains from a change?", "🎯 محاكاة السياسة: من يستفيد من التغيير؟"), expanded=True):
        a, b = st.columns(2)
        v = a.selectbox(tr("Variable to change", "المتغير المراد تغييره"), xs, key="qn_sv")
        dx = b.number_input(tr("Change (units)", "مقدار التغيير (بالوحدات)"), value=float(np.round(df[v].std(), 3)) or 1.0, key="qn_sd")
        s = T[T.Variable == v]
        sim = pd.DataFrame({"τ": s["τ"], tr("Predicted change in y", "التغير المتوقع في y"): s["Coef."] * dx, "CI low": s["CI low"] * dx, "CI high": s["CI high"] * dx})
        fig = go.Figure(go.Bar(x=sim["τ"], y=sim.iloc[:, 1], marker_color=[PAL[2] if q < 0.5 else PAL[0] for q in sim["τ"]]))
        ch.style_academic(fig, height=300, x_title="τ", y_title=tr("Δ y", "Δ y"), legend="none", rtl=is_ar())
        _chart(fig, "qn_sim", "quantile_policy_sim")
        st.caption(tr("Bars show the change in the conditional quantile of y — compare low (vulnerable) and high quantiles.",
                      "تُظهر الأعمدة التغير في الكمية الشرطية لـ y — قارن الكميات الدنيا (الفئات الهشة) بالعليا."))
    _finish("quantile", "qn_qr", {"Quantile process": T, "Journal": jt.reset_index(), "Slope equality": eq, "OLS": O},
            tr("Quantile regression", "الانحدار الكمّي"), [jt.reset_index(), eq], {"y": y, "x": xs})


def _cq(df, num):
    c1, c2 = st.columns(2)
    x = c1.selectbox(tr("Series predicted (x_t)", "السلسلة المتنبأ بها (x_t)"), num, key="cq_x")
    y = c2.selectbox(tr("Predictor (y_{t−k})", "المتنبئ (y_{t−k})"), [c for c in num if c != x], key="cq_y")
    a, b, c = st.columns(3)
    a1 = a.slider("α1", 0.05, 0.95, 0.1, 0.05, key="cq_a1")
    a2 = b.slider("α2", 0.05, 0.95, 0.1, 0.05, key="cq_a2")
    L = c.slider(tr("Max lag", "أقصى فجوة"), 2, 30, 10, key="cq_L")
    d = df[[x, y]].dropna()
    sig = (x, y, a1, a2, L, active_name())
    r = _load("cq", sig)
    if r is None:
        with st.spinner(tr("Stationary bootstrap…", "جارٍ البوتستراب الساكن…")):
            r = qn.cross_quantilogram(d[x].to_numpy(), d[y].to_numpy(), a1, a2, L, boot=300)
        _store("cq", r, sig)
    fig = go.Figure([go.Bar(x=r.lag, y=r["ρ"], marker_color=[PAL[2] if s else PAL[0] for s in r.significant], name="ρ"),
                     go.Scatter(x=r.lag, y=r["band low (H0)"], line=dict(color="#999", dash="dash"), name="95%"),
                     go.Scatter(x=r.lag, y=r["band high (H0)"], line=dict(color="#999", dash="dash"), showlegend=False)])
    ch.style_academic(fig, height=360, x_title=tr("Lag k", "الفجوة k"), y_title="ρα(k)", legend="bottom", rtl=is_ar())
    _chart(fig, "cq_pl", "cross_quantilogram")
    _table(r, 4)
    ns = int(r.significant.sum())
    insight(tr(f"{ns} of {L} lags are significant at 5%. Q({L}) p = {fp(r['p (Q)'].iloc[-1])}: " + ("tail events in the predictor help forecast tail events in x."
               if r["p (Q)"].iloc[-1] < 0.05 else "no directional tail predictability."),
               f"{ns} من {L} فجوات معنوية عند 5%. اختبار Q({L}) p = {fp(r['p (Q)'].iloc[-1])}: " + ("أحداث الأطراف في المتنبئ تساعد على التنبؤ بأحداث الأطراف في x."
               if r["p (Q)"].iloc[-1] < 0.05 else "لا توجد قدرة تنبؤية اتجاهية في الأطراف.")))
    _finish("cq", "qn_cq", {"Cross-quantilogram": r}, tr("Cross-quantilogram", "مخطط الكميات المتقاطع"), [r])


# ---------------------------------------------------------- 2. spatial autocorrelation
def _weights_ui(df, key):
    lon, lat = _coords(df)
    num = ds.numeric_columns(df)
    c1, c2, c3, c4 = st.columns(4)
    lon = c1.selectbox(tr("Longitude / x", "خط الطول / x"), num, num.index(lon) if lon in num else 0, key=f"{key}_lon")
    lat = c2.selectbox(tr("Latitude / y", "خط العرض / y"), num, num.index(lat) if lat in num else min(1, len(num) - 1), key=f"{key}_lat")
    kind = c3.selectbox(tr("Weights", "الأوزان"), list(sp.W_KINDS), key=f"{key}_wk", format_func=lambda k: _t(sp.W_KINDS[k]))
    k = c4.slider("k", 2, 15, 6, key=f"{key}_k") if kind == "knn" else None
    geo = df[lon].between(-180, 180).all() and df[lat].between(-90, 90).all()
    return lon, lat, kind, k or 6, geo


def _need_coords(df):
    lon, lat = _coords(df)
    if not (lon and lat):
        st.info(tr("Spatial analysis needs coordinate columns (longitude/latitude or x/y). Try the spatial districts demo.",
                   "يحتاج التحليل المكاني إلى أعمدة إحداثيات (خط الطول/العرض أو x/y). جرّب بيانات المراكز المكانية."))
        return False
    return True


def _spatial_auto(df):
    if not _need_coords(df):
        return
    if "year" in df.columns and df.duplicated([c for c in _coords(df) if c]).any():
        st.info(tr("Coordinates repeat (panel data): the latest year is used for cross-sectional maps; see the spatial panel option in the next tab.",
                   "الإحداثيات مكررة (بيانات بانل): تُستخدم آخر سنة للخرائط المقطعية؛ وانظر خيار البانل المكاني في التبويب التالي."))
        df = df[df.year == df.year.max()]
    lon, lat, kind, k, geo = _weights_ui(df, "sa")
    num = [c for c in _num(df) if c not in (lon, lat)]
    y = st.selectbox(tr("Variable", "المتغير"), _prefer(num, ("ln_", "value", "poverty", "yield")), key="sa_y")
    d = df.dropna(subset=[lon, lat, y]).reset_index(drop=True)
    W = sp.weights(d[[lon, lat]].to_numpy(), kind, k, geo=geo)
    sig = (lon, lat, kind, k, y, len(d), active_name())
    res = _load("sa", sig)
    if res is None:
        with st.spinner(tr("Permutation inference…", "جارٍ الاستدلال بالتباديل…")):
            res = (sp.moran(d[y], W), sp.geary(d[y], W), sp.lisa(d[y], W, perms=499))
        _store("sa", res, sig)
    m, g, L = res
    st.caption(" · ".join(f"{k_} = {v}" for k_, v in sp.describe_w(W).items()))
    tab = pd.DataFrame([{"Statistic": "Moran's I", "Estimate": m["I"], "Expected": m["E[I]"], "z": m["z (randomisation)"], "p-value": m["p (randomisation, two-sided)"],
                         "p (permutation)": m["p (permutation)"]},
                        {"Statistic": "Geary's C", "Estimate": g["C"], "Expected": 1.0, "z": g["z"], "p-value": 2 * min(g["p (greater)"], 1 - g["p (greater)"]), "p (permutation)": np.nan}])
    _table(tab, 4)
    c1, c2 = st.columns(2)
    with c1:
        z = L["z"]
        fig = go.Figure(go.Scatter(x=z, y=L["lag z"], mode="markers", marker=dict(color=PAL[0], size=7, opacity=0.7), name=""))
        bb = np.polyfit(z, L["lag z"], 1)
        gx = np.linspace(z.min(), z.max(), 10)
        fig.add_trace(go.Scatter(x=gx, y=np.polyval(bb, gx), line=dict(color=PAL[1], width=2.5), name=f"I = {m['I']:.3f}"))
        fig.add_hline(y=0, line=dict(color="#999", width=1))
        fig.add_vline(x=0, line=dict(color="#999", width=1))
        ch.style_academic(fig, height=400, x_title=tr(f"{y} (standardised)", f"{y} (معياري)"), y_title=tr("Spatial lag", "الإبطاء المكاني"), legend="bottom", rtl=is_ar())
        _chart(fig, "sa_ms", "moran_scatterplot")
    with c2:
        colors = {"HH": "#c0392b", "LL": "#2471a3", "HL": "#f1948a", "LH": "#85c1e9", "not significant": "#d5d8dc"}
        fig = go.Figure()
        for cname, col in colors.items():
            s = L.cluster == cname
            if s.any():
                fig.add_trace(go.Scatter(x=d[lon][s], y=d[lat][s], mode="markers", marker=dict(color=col, size=10, line=dict(color="#555", width=0.5)),
                                         name=_disp(pd.DataFrame({"c": [cname]})).iloc[0, 0]))
        ch.style_academic(fig, height=400, x_title=lon, y_title=lat, legend="bottom", rtl=is_ar())
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        _chart(fig, "sa_lisa", "lisa_cluster_map")
    cnt = L.cluster.value_counts()
    insight(tr(f"Moran's I = {m['I']:.3f} (p {fp(m['p (permutation)'])}): " + ("strong positive spatial clustering — similar values sit next to each other."
               if m["I"] > 0.3 and m["p (permutation)"] < 0.05 else "positive spatial dependence." if m["p (permutation)"] < 0.05 and m["I"] > 0 else
               "negative dependence (checkerboard)." if m["p (permutation)"] < 0.05 else "no significant spatial autocorrelation.") +
               f" LISA: {cnt.get('HH', 0)} hot spots (HH), {cnt.get('LL', 0)} cold spots (LL), {cnt.get('HL', 0) + cnt.get('LH', 0)} spatial outliers.",
               f"معامل موران I = {m['I']:.3f} (p {fp(m['p (permutation)'])}): " + ("تجمع مكاني موجب قوي — القيم المتشابهة متجاورة."
               if m["I"] > 0.3 and m["p (permutation)"] < 0.05 else "اعتماد مكاني موجب." if m["p (permutation)"] < 0.05 and m["I"] > 0 else
               "اعتماد سالب (نمط رقعة الشطرنج)." if m["p (permutation)"] < 0.05 else "لا ارتباط مكاني معنوي.") +
               f" تحليل LISA: {cnt.get('HH', 0)} بؤر ساخنة (مرتفع-مرتفع)، و{cnt.get('LL', 0)} بؤر باردة (منخفض-منخفض)، و{cnt.get('HL', 0) + cnt.get('LH', 0)} قيم شاذة مكانياً."))
    with st.expander(tr("🎯 Policy targeting: priority areas", "🎯 استهداف السياسات: المناطق ذات الأولوية"), expanded=False):
        idc = next((c for c in d.columns if d[c].dtype == object), None)
        tgt = d.assign(cluster=L.cluster, Ii=L.Ii, p=L["p (permutation)"])
        st.caption(tr("Cold spots (LL) are spatially concentrated lows — candidates for area-based programmes; hot spots may serve as growth poles.",
                      "البؤر الباردة (منخفض-منخفض) تجمعات للقيم المنخفضة — مرشحة لبرامج موجهة جغرافياً؛ والبؤر الساخنة قد تكون أقطاب نمو."))
        _table(tgt[tgt.cluster.isin(["LL", "HH"])][[c for c in [idc, y, "cluster", "Ii", "p"] if c]].sort_values("cluster"), 3)
    _finish("spatial_auto", "sp_auto", {"Global": tab, "LISA": d.assign(**L.to_dict("series"))}, tr("Spatial autocorrelation", "الارتباط الذاتي المكاني"),
            [tab], {"y": y, "lon": lon, "lat": lat, "k": k})


# ---------------------------------------------------------- 3. spatial regression
def _spatial_reg(df):
    mode = st.radio(tr("Model family", "عائلة النماذج"), ["cs", "gwr", "panel"], horizontal=True, key="sr_mode",
                    format_func=lambda k: {"cs": tr("Cross-section ML (SAR/SEM/SDM…)", "مقطعي بالإمكان الأعظم (SAR/SEM/SDM…)"),
                                           "gwr": tr("Geographically weighted regression", "الانحدار الموزون جغرافياً"),
                                           "panel": tr("Spatial panel (fixed effects)", "بانل مكاني (آثار ثابتة)")}[k])
    if mode == "panel":
        return _spatial_panel(df)
    if not _need_coords(df):
        return
    if "year" in df.columns and df.duplicated([c for c in _coords(df) if c]).any():
        df = df[df.year == df.year.max()]
    lon, lat, kind, k, geo = _weights_ui(df, "sr")
    num = [c for c in _num(df) if c not in (lon, lat)]
    c1, c2 = st.columns([1, 2])
    y = c1.selectbox(tr("Dependent variable", "المتغير التابع"), _prefer(num, ("ln_", "value", "yield")), key="sr_y")
    xs = c2.multiselect(tr("Regressors", "المتغيرات المفسرة"), [c for c in num if c != y], [c for c in num if c != y][:4], key="sr_x", placeholder=tr("Choose…", "اختر…"))
    if not xs:
        return
    d = df.dropna(subset=[lon, lat, y] + xs).reset_index(drop=True)
    if mode == "gwr":
        if len(d) > 1500:
            st.warning(tr("GWR is O(n²): use at most 1,500 locations.", "الانحدار الموزون جغرافياً مكلف حسابياً: استخدم 1500 موقعاً على الأكثر."))
            return
        sig = (lon, lat, y, tuple(xs), len(d), active_name())
        g = _load("gwr", sig)
        if g is None:
            with st.spinner(tr("Selecting the bandwidth (AICc)…", "جارٍ اختيار عرض النطاق (AICc)…")):
                g = sp.gwr(d, y, xs, d[[lon, lat]].to_numpy())
            _store("gwr", g, sig)
        st.caption(tr(f"Adaptive bisquare kernel · bandwidth = {g['bandwidth']} neighbours · AICc = {g['AICc']:.2f} · R² = {g['R²']:.3f} · effective parameters = {g['effective parameters']:.1f}",
                      f"نواة ثنائية التربيع تكيفية · عرض النطاق = {g['bandwidth']} جاراً · AICc = {g['AICc']:.2f} · R² = {g['R²']:.3f} · المعلمات الفعّالة = {g['effective parameters']:.1f}"))
        _table(g["summary"], 4)
        v = st.selectbox(tr("Map the local coefficient of", "خريطة المعامل المحلي لـ"), xs, key="gwr_v")
        loc = g["local"]
        fig = go.Figure(go.Scatter(x=d[lon], y=d[lat], mode="markers", marker=dict(color=loc[v], colorscale="RdBu_r", size=11, showscale=True,
                                   colorbar=dict(title=v), line=dict(color=np.where(np.abs(loc[f"t {v}"]) > 1.96, "#000", "#ccc"), width=1.2)),
                                   text=[f"t = {t:.2f}" for t in loc[f"t {v}"]]))
        ch.style_academic(fig, height=440, x_title=lon, y_title=lat, legend="none", rtl=is_ar())
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
        _chart(fig, "gwr_map", f"gwr_{v}")
        sv = g["summary"].set_index("Variable").loc[v]
        insight(tr(f"The local effect of {v} ranges from {sv['Min']:.4f} to {sv['Max']:.4f} (median {sv['Median']:.4f}); it is significant in {sv['% |t| > 1.96']:.0f}% of locations "
                   "(black outlines) — one national coefficient would mislead place-based policy.",
                   f"يتراوح الأثر المحلي لـ {v} بين {sv['Min']:.4f} و{sv['Max']:.4f} (الوسيط {sv['Median']:.4f})، وهو معنوي في {sv['% |t| > 1.96']:.0f}% من المواقع "
                   "(الحدود السوداء) — معامل وطني واحد قد يضلل السياسات المكانية."))
        _finish("gwr", "sp_gwr", {"GWR summary": g["summary"], "Local estimates": pd.concat([d[[lon, lat]], loc], axis=1)},
                tr("Geographically weighted regression", "الانحدار الموزون جغرافياً"), [g["summary"]], {"y": y, "x": xs, "lon": lon, "lat": lat})
        return
    W = sp.weights(d[[lon, lat]].to_numpy(), kind, k, geo=geo)
    lm = sp.lm_tests(d[y], np.column_stack([np.ones(len(d)), d[xs]]), W)
    st.markdown("**" + tr("Lagrange multiplier diagnostics on OLS residuals", "اختبارات مضاعف لاغرانج على بواقي OLS") + "**")
    _table(lm, 4)
    p = lm.set_index("Test")["p-value"]
    if p["Robust LM lag"] < 0.05 and p["Robust LM error"] >= 0.05:
        rec = "sar"
    elif p["Robust LM error"] < 0.05 and p["Robust LM lag"] >= 0.05:
        rec = "sem"
    elif p["Robust LM error"] < 0.05 and p["Robust LM lag"] < 0.05:
        rec = "sdm"
    else:
        rec = "ols" if p["LM lag"] >= 0.05 and p["LM error"] >= 0.05 else "sar" if p["LM lag"] < p["LM error"] else "sem"
    st.caption(tr(f"Anselin's decision rule suggests: **{_t(sp.MODELS[rec])}**.", f"تقترح قاعدة أنسلين: **{_t(sp.MODELS[rec])}**."))
    models = st.multiselect(tr("Models to estimate", "النماذج المراد تقديرها"), list(sp.MODELS), list(dict.fromkeys(["ols", "sar", "sem", "sdm", rec])), key="sr_m",
                            format_func=lambda k_: _t(sp.MODELS[k_]))
    sig = (lon, lat, kind, k, y, tuple(xs), tuple(models), len(d), active_name())
    fits = _load("sr", sig)
    if fits is None:
        with st.spinner(tr("Maximum likelihood…", "جارٍ التقدير بالإمكان الأعظم…")):
            fits = {m_: sp.spatial_model(d, y, xs, W, m_, impacts_draws=500) for m_ in models}
        _store("sr", fits, sig)
    comp = pd.DataFrame([{"Model": _t(sp.MODELS[m_]), "Log-lik.": f["loglik"], "AIC": f["AIC"], "BIC": f["BIC"], "ρ": f["rho"], "λ": f["lambda"]} for m_, f in fits.items()])
    _table(comp, 4)
    best = min(fits, key=lambda m_: fits[m_]["AIC"])
    show = st.selectbox(tr("Show model", "عرض النموذج"), list(fits), list(fits).index(best), key="sr_show", format_func=lambda k_: _t(sp.MODELS[k_]))
    f = fits[show]
    _table(f["table"], 4)
    if "impacts" in f:
        st.markdown("**" + tr("Direct, indirect (spillover) and total impacts (LeSage-Pace)", "الآثار المباشرة وغير المباشرة (الانتشارية) والكلية (ليساج-بيس)") + "**")
        _table(f["impacts"], 4)
        im = f["impacts"]
        fig = go.Figure()
        for j, typ in enumerate(["Direct", "Indirect (spillover)"]):
            s = im[im.Impact == typ]
            fig.add_trace(go.Bar(x=s.Variable, y=s.Estimate, name=_disp(pd.DataFrame({"c": [typ]})).iloc[0, 0], marker_color=PAL[j],
                                 error_y=dict(type="data", array=1.96 * s["Std. Err. (simulated)"])))
        fig.update_layout(barmode="group")
        ch.style_academic(fig, height=360, y_title=tr("Impact", "الأثر"), legend="bottom", rtl=is_ar())
        _chart(fig, "sr_imp", "spatial_impacts")
    msg = tr(f"Best model by AIC: {_t(sp.MODELS[best])}.", f"أفضل نموذج وفق AIC: {_t(sp.MODELS[best])}.")
    if f["rho"]:
        msg += tr(f" ρ = {f['rho']:.3f}: a 1-unit rise in neighbours' outcome raises the local outcome by {f['rho']:.3f}; the spatial multiplier is {1 / (1 - f['rho']):.2f}.",
                  f" ρ = {f['rho']:.3f}: ارتفاع نتيجة الجيران وحدة يرفع النتيجة المحلية بمقدار {f['rho']:.3f}، والمضاعف المكاني {1 / (1 - f['rho']):.2f}.")
    if "impacts" in f:
        im = f["impacts"]
        tot = im[im.Impact == "Total"].set_index("Variable")
        ind = im[im.Impact == "Indirect (spillover)"].set_index("Variable")
        v0 = tot.Estimate.abs().idxmax()
        share = 100 * ind.loc[v0, "Estimate"] / tot.loc[v0, "Estimate"] if tot.loc[v0, "Estimate"] else np.nan
        msg += tr(f" For {v0}, {share:.0f}% of the total effect is spillover to other districts — ignoring it understates policy returns.",
                  f" بالنسبة لـ {v0} فإن {share:.0f}% من الأثر الكلي ينتقل إلى المراكز الأخرى — وإهماله يبخس عائد السياسة.")
    if "residual Moran" in f:
        rm = f["residual Moran"]
        msg += tr(f" Residual Moran's I = {rm['I']:.3f} (p {fp(rm['p (randomisation, two-sided)'])}).", f" معامل موران للبواقي = {rm['I']:.3f} (p {fp(rm['p (randomisation, two-sided)'])}).")
    insight(msg)
    if "impacts" in f:
        with st.expander(tr("🎯 Policy simulation: a local investment and its spillovers", "🎯 محاكاة السياسة: استثمار محلي وآثاره الانتشارية"), expanded=False):
            a, b, c = st.columns(3)
            idc = next((cc for cc in d.columns if d[cc].dtype == object), None)
            labels = d[idc].astype(str).tolist() if idc else [str(i) for i in range(len(d))]
            who = a.selectbox(tr("Location receiving the investment", "الموقع المستفيد من الاستثمار"), range(len(d)), format_func=lambda i: labels[i], key="sr_who")
            v = b.selectbox(tr("Variable increased", "المتغير الذي يزداد"), xs, key="sr_v")
            dx = c.number_input(tr("Increase", "مقدار الزيادة"), value=float(np.round(d[v].std(), 3)) or 1.0, key="sr_dx")
            j = xs.index(v)
            beta = f["est"][1 + j]
            theta = f["est"][1 + len(xs) + j] if show in ("sdm", "slx") else 0.0
            rho = f["rho"]
            e = np.zeros(len(d))
            e[who] = dx
            eff = np.linalg.solve(np.eye(len(d)) - rho * W, beta * e + theta * (W @ e))
            fig = go.Figure(go.Scatter(x=d[lon], y=d[lat], mode="markers", marker=dict(color=eff, colorscale="Viridis", size=11, showscale=True, colorbar=dict(title="Δ y"))))
            fig.add_trace(go.Scatter(x=[d[lon][who]], y=[d[lat][who]], mode="markers", marker=dict(symbol="star", size=20, color="#e74c3c"), name=labels[who]))
            ch.style_academic(fig, height=420, x_title=lon, y_title=lat, legend="none", rtl=is_ar())
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            _chart(fig, "sr_diff", "spillover_simulation")
            st.caption(tr(f"Own effect = {eff[who]:.4f}; spillover to all other locations = {eff.sum() - eff[who]:.4f} (total {eff.sum():.4f}).",
                          f"الأثر الذاتي = {eff[who]:.4f}؛ والانتشار إلى بقية المواقع = {eff.sum() - eff[who]:.4f} (الإجمالي {eff.sum():.4f})."))
    sheets = {"LM tests": lm, "Comparison": comp, **{f"{m_} coefficients": fits[m_]["table"] for m_ in fits}}
    sheets.update({f"{m_} impacts": fits[m_]["impacts"] for m_ in fits if "impacts" in fits[m_]})
    _finish("spatial_reg", "sp_reg", sheets, tr("Spatial regression", "الانحدار المكاني"), [lm, comp, f["table"]] + ([f["impacts"]] if "impacts" in f else []),
            {"y": y, "x": xs, "lon": lon, "lat": lat, "k": k})


def _spatial_panel(df):
    lon, lat = _coords(df)
    cats = _cats(df)
    tcols = [c for c in df.columns if c.lower() in ("year", "period", "time")]
    if not (lon and lat and cats and tcols):
        st.info(tr("A spatial panel needs a unit identifier, a time variable and coordinates (load the spatial panel demo).",
                   "يحتاج البانل المكاني إلى معرّف للوحدات ومتغير زمني وإحداثيات (حمّل البيانات التجريبية للبانل المكاني)."))
        return
    c1, c2, c3, c4 = st.columns(4)
    u = c1.selectbox(tr("Unit", "الوحدة"), cats, key="spp_u")
    t = c2.selectbox(tr("Time", "الزمن"), tcols, key="spp_t")
    model = c3.selectbox(tr("Model", "النموذج"), ["sar", "sem"], key="spp_m", format_func=lambda k_: _t(sp.MODELS[k_]))
    eff = c4.selectbox(tr("Fixed effects", "الآثار الثابتة"), ["unit", "twoway"], key="spp_e", format_func=lambda k_: {"unit": tr("Unit", "الوحدة"), "twoway": tr("Two-way", "ثنائية")}[k_])
    num = [c for c in _num(df) if c not in (lon, lat)]
    a, b = st.columns([1, 2])
    y = a.selectbox(tr("Dependent variable", "المتغير التابع"), _prefer(num, ("ln_y", "yield")), key="spp_y")
    xs = b.multiselect(tr("Regressors", "المتغيرات المفسرة"), [c for c in num if c != y], [c for c in num if c != y][:2], key="spp_x", placeholder=tr("Choose…", "اختر…"))
    k = st.slider("k", 2, 15, 6, key="spp_k")
    if not xs:
        return
    first = df.drop_duplicates(u)
    W = sp.weights(first[[lon, lat]].to_numpy(), "knn", k)
    sig = (u, t, model, eff, y, tuple(xs), k, active_name())
    r = _load("spp", sig)
    if r is None:
        with st.spinner(tr("Concentrated ML…", "جارٍ التقدير…")):
            r = sp.spatial_panel(df, u, t, y, xs, W, model, eff)
        _store("spp", r, sig)
    st.caption(f"N = {r['N']} · T = {r['T']} · log-lik = {r['loglik']:.3f} · σ² = {r['sigma2']:.5f}")
    _table(r["table"], 4)
    if "impacts" in r:
        _table(r["impacts"], 4)
    insight(tr(f"Spatial parameter = {r['param']:.3f} after removing {'unit' if eff == 'unit' else 'unit and year'} fixed effects: "
               + ("yields co-move with neighbours beyond common shocks." if model == "sar" else "unobserved shocks are spatially correlated."),
               f"المعلمة المكانية = {r['param']:.3f} بعد إزالة الآثار الثابتة " + ("للوحدات" if eff == "unit" else "للوحدات والسنوات") + ": "
               + ("تتحرك الإنتاجية مع الجيران بما يتجاوز الصدمات المشتركة." if model == "sar" else "الصدمات غير المشاهدة مترابطة مكانياً.")))
    _finish("spatial_panel", "sp_pan", {"Coefficients": r["table"], **({"Impacts": r["impacts"]} if "impacts" in r else {})},
            tr("Spatial panel", "البانل المكاني"), [r["table"]], {"y": y, "x": xs, "unit": u, "time": t, "lon": lon, "lat": lat, "k": k})


# ---------------------------------------------------------- 4. RCT & IV
def _rct_iv(df):
    bins = _binary(df)
    num = _num(df)
    if not bins:
        st.info(tr("A binary (0/1) treatment variable is required.", "يلزم متغير معالجة ثنائي (0/1)."))
        return
    mode = st.radio(tr("Design", "التصميم"), ["rct", "iv"], horizontal=True, key="rc_mode",
                    format_func=lambda k: {"rct": tr("Randomised trial", "تجربة عشوائية"), "iv": tr("IV / LATE (imperfect compliance)", "متغيرات أداتية / LATE (عدم امتثال)")}[k])
    c1, c2, c3 = st.columns(3)
    y = c1.selectbox(tr("Outcome", "النتيجة"), _prefer([c for c in num if c not in bins], ("ln_y", "yield", "income", "outcome")), key="rc_y")
    if mode == "rct":
        t = c2.selectbox(tr("Treatment (random assignment)", "المعالجة (التخصيص العشوائي)"), _prefer(bins, ("offer", "treat", "assign")), key="rc_t")
        cats = _cats(df, 50)
        strata = c3.selectbox(tr("Strata (optional)", "الطبقات (اختياري)"), [None] + cats, key="rc_s", format_func=lambda v: tr("None", "بدون") if v is None else v)
        covs = st.multiselect(tr("Baseline covariates", "المتغيرات الأساسية"), [c for c in num if c not in (y, t) and c not in bins] + [b for b in bins if b != t],
                              [c for c in num if c not in (y, t) and c not in bins][:3], key="rc_c", placeholder=tr("Choose…", "اختر…"))
        r = cz.rct(df, y, t, covs, strata)
        _table(r["table"].drop(columns=[""]), 4)
        if "balance" in r:
            st.markdown("**" + tr("Baseline balance", "التوازن عند خط الأساس") + "**")
            _table(r["balance"], 3)
            jf = r["joint F"]
            st.caption(tr(f"Joint orthogonality F({jf['df1']}, {jf['df2']}) = {jf['F']:.3f}, p = {fp(jf['p-value'])}",
                          f"اختبار التعامد المشترك F({jf['df1']}, {jf['df2']}) = {jf['F']:.3f}، p = {fp(jf['p-value'])}"))
        T = r["table"]
        fig = go.Figure(go.Scatter(x=T.Estimate, y=[_disp(pd.DataFrame({"c": [e]})).iloc[0, 0] for e in T.Estimator], mode="markers",
                                   marker=dict(size=12, color=PAL[0]), error_x=dict(type="data", symmetric=False, array=T["CI high"] - T.Estimate, arrayminus=T.Estimate - T["CI low"])))
        fig.add_vline(x=0, line=dict(color="#999"))
        ch.style_academic(fig, height=300, x_title=tr("Treatment effect", "أثر المعالجة"), legend="none", rtl=is_ar())
        _chart(fig, "rc_forest", "rct_estimates")
        best = T.iloc[-1]
        gain = 100 * (1 - T["Std. Err."].iloc[-1] / T["Std. Err."].iloc[0])
        insight(tr(f"Average effect = {best.Estimate:.4f} (95% CI {best['CI low']:.4f} to {best['CI high']:.4f}, p {fp(best['p-value'])}). "
                   f"Covariate adjustment cut the standard error by {gain:.0f}%." + (f" Balance: joint p = {fp(r['joint F']['p-value'])} — randomisation looks successful."
                   if "joint F" in r and r["joint F"]["p-value"] > 0.05 else ""),
                   f"متوسط الأثر = {best.Estimate:.4f} (فترة ثقة 95%: {best['CI low']:.4f} إلى {best['CI high']:.4f}، p {fp(best['p-value'])}). "
                   f"خفّض التعديل بالمتغيرات الخطأ المعياري بنسبة {gain:.0f}%." + (f" التوازن: p المشترك = {fp(r['joint F']['p-value'])} — يبدو التوزيع العشوائي ناجحاً."
                   if "joint F" in r and r["joint F"]["p-value"] > 0.05 else "")))
        het = st.selectbox(tr("Heterogeneity by", "عدم التجانس حسب"), [None] + _cats(df, 20) + [b for b in bins if b != t], key="rc_h",
                           format_func=lambda v: tr("None", "بدون") if v is None else v)
        sheets = {"Estimates": T, **({"Balance": r["balance"]} if "balance" in r else {})}
        if het:
            H = cz.rct_heterogeneity(df, y, t, het)
            _table(H.drop(columns=[""]), 4)
            it = H.attrs["interaction test"]
            st.caption(tr(f"Interaction Wald test: χ²({it['df']}) = {it['Wald']:.3f}, p = {fp(it['p-value'])}", f"اختبار والد للتفاعل: χ²({it['df']}) = {it['Wald']:.3f}، p = {fp(it['p-value'])}"))
            sheets["Heterogeneity"] = H
        _cba(best.Estimate, best["Std. Err."], "rc")
        _finish("rct", "pe_rct", sheets, tr("Randomised controlled trial", "التجربة العشوائية المضبوطة"), [T] + ([r["balance"]] if "balance" in r else []),
                {"y": y, "x": covs or ["x1"], "treat": t})
        return
    d_ = c2.selectbox(tr("Treatment received (endogenous)", "المعالجة الفعلية (داخلية)"), _prefer(bins, ("particip", "take", "received")), key="iv_d")
    zs = c3.multiselect(tr("Instrument(s)", "الأداة/الأدوات"), [b for b in bins + num if b not in (y, d_)], [b for b in _prefer(bins, ("offer", "assign", "elig")) if b != d_][:1], key="iv_z",
                        placeholder=tr("Choose…", "اختر…"))
    ex = st.multiselect(tr("Exogenous controls", "ضوابط خارجية"), [c for c in num if c not in (y, d_) and c not in zs], [], key="iv_x", placeholder=tr("Optional", "اختياري"))
    if not zs:
        return
    r = cz.iv_2sls(df, y, d_, zs, ex)
    _table(r["table"].drop(columns=[""]), 4)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**" + tr("First stage", "المرحلة الأولى") + "**")
        _table(r["first stage"].drop(columns=[""]), 4)
        st.caption(tr(f"First-stage F = {r['first-stage F']:.1f} " + ("(strong instrument, F > 10)" if r["first-stage F"] > 10 else "(⚠️ weak instrument)"),
                      f"إحصاءة F للمرحلة الأولى = {r['first-stage F']:.1f} " + ("(أداة قوية، F > 10)" if r["first-stage F"] > 10 else "(⚠️ أداة ضعيفة)")))
    with c2:
        st.markdown("**" + tr("Reduced form (ITT)", "المعادلة المختزلة (أثر النية)") + "**")
        _table(r["reduced form"].drop(columns=[""]), 4)
        if "Wald" in r:
            st.json({k: round(v, 4) for k, v in r["Wald"].items()})
    o = r["OLS"]
    msg = tr(f"LATE = {r['late']:.4f} (SE {r['late se']:.4f}) versus naive OLS {o['Estimate']:.4f}.", f"متوسط الأثر المحلي LATE = {r['late']:.4f} (خطأ {r['late se']:.4f}) مقابل OLS الساذج {o['Estimate']:.4f}.")
    if "Wald" in r:
        cs_ = r["Wald"]["Complier share (first stage)"]
        msg += tr(f" {100 * cs_:.0f}% of those offered complied; the effect applies to these compliers.", f" امتثل {100 * cs_:.0f}% ممن عُرضت عليهم المعالجة؛ ويخص الأثر هؤلاء الممتثلين.")
    if "endogeneity test" in r:
        msg += tr(f" Endogeneity test p = {fp(r['endogeneity test']['p-value'])}.", f" اختبار الداخلية p = {fp(r['endogeneity test']['p-value'])}.")
    insight(msg)
    _cba(r["late"], r["late se"], "iv")
    _finish("iv", "pe_iv", {"2SLS": r["table"], "First stage": r["first stage"], "Reduced form": r["reduced form"]}, tr("Instrumental variables", "المتغيرات الأداتية"),
            [r["table"], r["first stage"]], {"y": y, "treat": d_, "z": zs})


def _cba(effect, se, key):
    with st.expander(tr("🎯 Policy simulation: scale-up cost-benefit", "🎯 محاكاة السياسة: التكلفة والعائد عند التوسع"), expanded=False):
        a, b, c, d = st.columns(4)
        n = a.number_input(tr("Beneficiaries", "عدد المستفيدين"), 1000, 10_000_000, 100_000, 1000, key=f"{key}_cba_n")
        base = b.number_input(tr("Baseline outcome value per person (EGP)", "قيمة النتيجة الأساسية للفرد (جنيه)"), 0.0, 1e7, 20000.0, 500.0, key=f"{key}_cba_b")
        cost = c.number_input(tr("Cost per beneficiary (EGP)", "التكلفة لكل مستفيد (جنيه)"), 0.0, 1e7, 600.0, 50.0, key=f"{key}_cba_c")
        logy = d.checkbox(tr("Outcome is in logs", "النتيجة لوغاريتمية"), True, key=f"{key}_cba_l")
        per = base * (np.exp(effect) - 1) if logy else effect
        lo = base * (np.exp(effect - 1.96 * se) - 1) if logy else effect - 1.96 * se
        hi = base * (np.exp(effect + 1.96 * se) - 1) if logy else effect + 1.96 * se
        tot, tc = per * n, cost * n
        st.dataframe(_disp(pd.DataFrame([{tr("Gain per beneficiary", "العائد لكل مستفيد"): per, tr("95% range", "المدى 95%"): f"{lo:,.0f} – {hi:,.0f}",
                                           tr("Total gain", "إجمالي العائد"): tot, tr("Total cost", "إجمالي التكلفة"): tc,
                                           tr("Benefit / cost", "العائد / التكلفة"): tot / tc if tc else np.nan}])).round(2), width="stretch", hide_index=True)
        st.caption(tr("One-year benefits only; add multi-year flows and discounting for a full appraisal.", "عوائد سنة واحدة فقط؛ أضف التدفقات متعددة السنوات والخصم لتقييم كامل."))


# ---------------------------------------------------------- 5. DiD
def _did(df):
    cats = _cats(df)
    tcols = [c for c in df.columns if c.lower() in ("year", "period", "time", "month")]
    num = _num(df)
    gcols = [c for c in ds.numeric_columns(df) if "first" in c.lower() or "cohort" in c.lower() or c.lower() in ("g", "gvar", "treat_year")]
    mode = st.radio(tr("Design", "التصميم"), ["stag", "2x2"], horizontal=True, key="dd_mode",
                    format_func=lambda k: {"stag": tr("Staggered adoption (event study)", "تطبيق متدرج (دراسة حدث)"), "2x2": tr("Classic 2×2", "الكلاسيكي 2×2")}[k])
    if mode == "2x2":
        bins = _binary(df)
        if len(bins) < 2:
            st.info(tr("Needs binary treated-group and post-period indicators.", "يحتاج مؤشرين ثنائيين للمجموعة المعالجة وفترة ما بعد."))
            return
        c1, c2, c3, c4 = st.columns(4)
        y = c1.selectbox(tr("Outcome", "النتيجة"), num, key="d2_y")
        g = c2.selectbox(tr("Treated group", "المجموعة المعالجة"), bins, key="d2_g")
        p = c3.selectbox(tr("Post period", "فترة ما بعد"), [b for b in bins if b != g], key="d2_p")
        clu = c4.selectbox(tr("Cluster", "العنقود"), [None] + cats, key="d2_c", format_func=lambda v: tr("None (HC1)", "بدون (HC1)") if v is None else v)
        r = cz.did_2x2(df, y, g, p, clu)
        _table(r["table"].drop(columns=[""]), 4)
        st.dataframe(r["cells"].round(4), width="stretch")
        insight(tr(f"DiD effect = {r['did']:.4f} (SE {r['se']:.4f}).", f"أثر فرق الفروق = {r['did']:.4f} (خطأ {r['se']:.4f})."))
        _finish("did22", "pe_d2", {"DiD": r["table"], "Cell means": r["cells"].reset_index()}, tr("Difference-in-differences", "فرق الفروق"), [r["table"]], {"y": y})
        return
    if not (cats and tcols and gcols):
        st.info(tr("Needs a unit id, a time variable and the first treatment period (0 = never treated). Try the staggered DiD demo.",
                   "يحتاج معرّف وحدة ومتغيراً زمنياً وأول فترة معالجة (0 = لم يُعالَج أبداً). جرّب بيانات فرق الفروق المتدرج."))
        return
    c1, c2, c3, c4 = st.columns(4)
    u = c1.selectbox(tr("Unit", "الوحدة"), cats, key="dd_u")
    t = c2.selectbox(tr("Time", "الزمن"), tcols, key="dd_t")
    g = c3.selectbox(tr("First treated period (0 = never)", "أول فترة معالجة (0 = أبداً)"), gcols, key="dd_g")
    y = c4.selectbox(tr("Outcome", "النتيجة"), _prefer([c for c in num if c != g], ("ln_", "productiv", "yield")), key="dd_y")
    ctl = st.radio(tr("Comparison group", "مجموعة المقارنة"), ["never", "notyet"], horizontal=True, key="dd_c",
                   format_func=lambda k: {"never": tr("Never treated", "لم تُعالَج أبداً"), "notyet": tr("Not yet treated", "لم تُعالَج بعد")}[k])
    sig = (u, t, g, y, ctl, active_name())
    res = _load("did", sig)
    if res is None:
        with st.spinner(tr("Estimating ATT(g,t), Sun-Abraham and TWFE…", "جارٍ تقدير ATT(g,t) وسن-أبراهام وTWFE…")):
            res = (cz.callaway_santanna(df, u, t, y, g, ctl), cz.sun_abraham(df, u, t, y, g), cz.event_study_twfe(df, u, t, y, g),
                   cz.bacon_weights(df, u, t, y, g))
        _store("did", res, sig)
    cs, sa, es, bw = res
    _table(cs["summary"], 4)
    dyn = cs["dynamic"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dyn["Event time"], y=dyn.ATT, mode="markers+lines", name="Callaway-Sant'Anna", line=dict(color=PAL[0], width=2.5),
                             error_y=dict(type="data", array=1.96 * dyn["Std. Err."])))
    ev = sa["event"]
    fig.add_trace(go.Scatter(x=ev["Relative period"] + 0.15, y=ev.Estimate, mode="markers", name="Sun-Abraham", marker=dict(color=PAL[1], symbol="diamond", size=8),
                             error_y=dict(type="data", array=1.96 * ev["Std. Err."])))
    fig.add_trace(go.Scatter(x=es["Relative period"] - 0.15, y=es.Estimate, mode="markers", name=tr("TWFE (binned)", "TWFE (مجمّع)"), marker=dict(color=PAL[3], symbol="square", size=7)))
    fig.add_hline(y=0, line=dict(color="#999"))
    fig.add_vline(x=-0.5, line=dict(color="#999", dash="dot"))
    ch.style_academic(fig, height=420, x_title=tr("Periods relative to treatment", "الفترات نسبةً إلى المعالجة"), y_title="ATT", legend="bottom", rtl=is_ar())
    _chart(fig, "dd_ev", "event_study")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**" + tr("ATT by cohort", "الأثر حسب الفوج") + "**")
        _table(cs["group"], 4)
    with c2:
        st.markdown("**" + tr("Goodman-Bacon decomposition of static TWFE", "تفكيك جودمان-بيكون لنموذج TWFE الساكن") + "**")
        _table(bw, 4)
    with st.expander(tr("All ATT(g,t)", "كل قيم ATT(g,t)"), expanded=False):
        _table(cs["att_gt"], 4)
    s0 = cs["summary"].iloc[0]
    pt = cs["pretrend"]
    forb = bw[bw.Comparison.str.startswith("Later")].Weight.sum()
    insight(tr(f"Overall ATT = {s0.ATT:.4f} (SE {s0['Std. Err.']:.4f}, p {fp(s0['p-value'])})." +
               (f" Pre-trend Wald test p = {fp(pt['p-value'])}: " + ("parallel trends are not rejected." if pt["p-value"] > 0.05 else "⚠️ parallel trends are doubtful.") if pt else "") +
               f" Sun-Abraham ATT = {sa['att']['Estimate']:.4f}. 'Forbidden' comparisons carry {100 * forb:.0f}% of the static TWFE weight (TWFE estimate {bw.attrs['weighted sum']:.4f}).",
               f"متوسط الأثر الكلي = {s0.ATT:.4f} (خطأ {s0['Std. Err.']:.4f}، p {fp(s0['p-value'])})." +
               (f" اختبار والد للاتجاهات السابقة p = {fp(pt['p-value'])}: " + ("لا يُرفض فرض الاتجاهات المتوازية." if pt["p-value"] > 0.05 else "⚠️ فرض الاتجاهات المتوازية موضع شك.") if pt else "") +
               f" تقدير سن-أبراهام = {sa['att']['Estimate']:.4f}. تحمل المقارنات «الممنوعة» {100 * forb:.0f}% من وزن TWFE الساكن (تقديره {bw.attrs['weighted sum']:.4f})."))
    with st.expander(tr("🎯 Policy simulation: extending the reform", "🎯 محاكاة السياسة: تعميم الإصلاح"), expanded=False):
        nu = int((df.groupby(u)[g].first() == 0).sum())
        a, b = st.columns(2)
        e_ = a.slider(tr("Years after adoption", "السنوات بعد التطبيق"), 0, int(dyn["Event time"].max()), min(3, int(dyn["Event time"].max())), key="dd_e")
        row = dyn[dyn["Event time"] == e_].iloc[0]
        b.metric(tr(f"Expected effect in the {nu} untreated units", f"الأثر المتوقع في {nu} وحدة غير معالجة"), f"{row.ATT:.4f}", f"± {1.96 * row['Std. Err.']:.4f}")
        st.caption(tr("Assumes the untreated units would respond like adopters — check comparability before generalising.",
                      "يفترض أن الوحدات غير المعالجة ستستجيب مثل المطبِّقين — تحقق من قابلية المقارنة قبل التعميم."))
    _finish("did", "pe_did", {"Summary": cs["summary"], "Dynamic": dyn, "ATT(g,t)": cs["att_gt"], "By cohort": cs["group"], "Calendar": cs["calendar"],
                              "Sun-Abraham": sa["event"], "TWFE": es, "Bacon": bw}, tr("Staggered difference-in-differences", "فرق الفروق المتدرج"),
            [cs["summary"], dyn, sa["event"]], {"y": y, "unit": u, "time": t, "first_treated": g})


# ---------------------------------------------------------- 6. synthetic control
def _synth(df):
    cats = _cats(df)
    tcols = [c for c in df.columns if c.lower() in ("year", "period", "time")]
    if not (cats and tcols):
        st.info(tr("Needs a unit id and a time variable (try the synthetic-control demo).", "يحتاج معرّف وحدة ومتغيراً زمنياً (جرّب بيانات الضابط التركيبي)."))
        return
    num = _num(df)
    c1, c2, c3 = st.columns(3)
    u = c1.selectbox(tr("Unit", "الوحدة"), cats, key="sc_u")
    t = c2.selectbox(tr("Time", "الزمن"), tcols, key="sc_t")
    y = c3.selectbox(tr("Outcome", "النتيجة"), _prefer(num, ("ln_gdp", "gdp", "ln_")), key="sc_y")
    units = list(pd.unique(df[u]))
    bins = _binary(df)
    flag = next((b for b in bins if "treat" in b.lower()), None)
    default_u = df.loc[df[flag] == 1, u].iloc[0] if flag else units[0]
    tl = sorted(df[t].unique())
    a, b, c = st.columns(3)
    tu_ = a.selectbox(tr("Treated unit", "الوحدة المعالجة"), units, units.index(default_u), key="sc_tu")
    tt = b.selectbox(tr("First treated period", "أول فترة معالجة"), tl[2:], min(len(tl) - 3, max(0, len(tl) // 2)), key="sc_tt")
    preds = c.multiselect(tr("Predictors (pre-period means)", "المحددات (متوسطات ما قبل)"), [x for x in num if x != y and x not in bins], [x for x in num if x != y and x not in bins][:2], key="sc_p",
                          placeholder=tr("Optional", "اختياري"))
    many = len(units) > 40
    placebo = st.checkbox(tr("Run in-space placebos (slower)", "تشغيل اختبارات الوهم المكانية (أبطأ)"), not many, key="sc_pl")
    sig = (u, t, y, tu_, tt, tuple(preds), placebo, active_name())
    r = _load("sc", sig)
    if r is None and many and not st.button(tr(f"▶️ Estimate ({len(units)} units)", f"▶️ تقدير ({len(units)} وحدة)"), key="sc_run"):
        st.info(tr("Synthetic control is designed for one treated unit and a few dozen donors; press the button to estimate.",
                   "صُمم الضابط التركيبي لوحدة معالجة واحدة وعشرات قليلة من المانحين؛ اضغط الزر للتقدير."))
        return
    if r is None:
        with st.spinner(tr("Optimising weights and placebos…", "جارٍ تحسين الأوزان واختبارات الوهم…")):
            r = cz.synth_control(df, u, t, y, tu_, tt, preds, placebo=placebo)
        _store("sc", r, sig)
    P = r["path"]
    fig = go.Figure([go.Scatter(x=P[t], y=P.Treated, name=str(tu_), line=dict(color=PAL[0], width=3)),
                     go.Scatter(x=P[t], y=P.Synthetic, name=tr("Synthetic", "التركيبي"), line=dict(color=PAL[1], width=2.5, dash="dash"))])
    fig.add_vline(x=tt - 0.5, line=dict(color="#999", dash="dot"))
    ch.style_academic(fig, height=380, x_title=t, y_title=y, legend="bottom", rtl=is_ar())
    c1, c2 = st.columns(2)
    with c1:
        _chart(fig, "sc_path", "synthetic_control")
    with c2:
        fig = go.Figure()
        if "placebo gaps" in r:
            G = r["placebo gaps"]
            pre_ok = {k_: v for k_, v in r["pre RMSPE (all)"].items() if v <= 5 * r["pre RMSPE"]}
            for col in G.columns:
                if col != tu_ and col in pre_ok:
                    fig.add_trace(go.Scatter(x=G.index, y=G[col], line=dict(color="#ccc", width=1), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=P[t], y=P.Gap, name=str(tu_), line=dict(color=PAL[0], width=3)))
        fig.add_hline(y=0, line=dict(color="#999"))
        fig.add_vline(x=tt - 0.5, line=dict(color="#999", dash="dot"))
        ch.style_academic(fig, height=380, x_title=t, y_title=tr("Gap (treated − synthetic)", "الفجوة (المعالج − التركيبي)"), legend="bottom", rtl=is_ar())
        _chart(fig, "sc_gap", "synthetic_gaps_placebos")
    c1, c2 = st.columns(2)
    with c1:
        _table(r["weights"][r["weights"].Weight > 0.001], 4)
    with c2:
        _table(r["balance"], 4)
    msg = tr(f"Average post-treatment gap = {r['ATT (post average gap)']:.4f} (pre-treatment RMSPE {r['pre RMSPE']:.4f}).",
             f"متوسط الفجوة بعد المعالجة = {r['ATT (post average gap)']:.4f} (RMSPE قبل المعالجة {r['pre RMSPE']:.4f}).")
    if "p-value (rank)" in r:
        msg += tr(f" Placebo rank p-value = {r['p-value (rank)']:.3f} ({len(r['ratios'])} units).", f" قيمة p الترتيبية لاختبارات الوهم = {r['p-value (rank)']:.3f} ({len(r['ratios'])} وحدة).")
    dmax = r["weights"].iloc[0]
    msg += tr(f" The largest donor is {dmax.iloc[0]} ({dmax.Weight:.2f}).", f" أكبر مانح هو {dmax.iloc[0]} ({dmax.Weight:.2f}).")
    insight(msg)
    sheets = {"Path": P, "Weights": r["weights"], "Balance": r["balance"], **({"Placebo ratios": r["ratios"]} if "ratios" in r else {})}
    if st.checkbox(tr("Also estimate synthetic difference-in-differences", "تقدير فرق الفروق التركيبي أيضاً"), False, key="sc_sdid"):
        d = df[[u, t, y]].copy()
        d["_D"] = ((d[u] == tu_) & (d[t] >= tt)).astype(int)
        sd = _load("sdid", sig)
        if sd is None:
            with st.spinner("SDiD…"):
                sd = cz.synth_did(d, u, t, y, "_D", reps=100)
            _store("sdid", sd, sig)
        st.dataframe(_disp(pd.DataFrame([{"Estimator": "SDiD", "Estimate": sd["ATT"], "Std. Err.": sd["se"], "CI low": sd["CI"][0], "CI high": sd["CI"][1], "p-value": sd["p-value"]},
                                          {"Estimator": "DiD", "Estimate": sd["DiD"]}, {"Estimator": "SC (post average gap)", "Estimate": r["ATT (post average gap)"]}])).round(4),
                     width="stretch", hide_index=True)
        sheets["SDiD unit weights"] = sd["unit weights"].rename("ω").reset_index()
        sheets["SDiD time weights"] = sd["time weights"].rename("λ").reset_index()
    with st.expander(tr("🎯 Cumulative policy effect", "🎯 الأثر التراكمي للسياسة"), expanded=False):
        post = P[P[t] >= tt]
        st.metric(tr("Cumulative gap over the post period (log points)", "الفجوة التراكمية بعد المعالجة (نقاط لوغاريتمية)"), f"{post.Gap.sum():.3f}")
        st.caption(tr("For a log outcome, the average % effect ≈ 100·(exp(mean gap) − 1) = ", "لنتيجة لوغاريتمية، متوسط الأثر % ≈ 100·(exp(متوسط الفجوة) − 1) = ")
                   + f"{100 * (np.exp(post.Gap.mean()) - 1):.2f}%")
    _finish("synth", "pe_sc", sheets, tr("Synthetic control", "الضابط التركيبي"), [r["weights"].head(10), r["balance"]],
            {"y": y, "x": preds or ["x1"], "unit": u, "time": t, "treated": tu_, "treat_time": tt})


# ---------------------------------------------------------- 7. RDD
def _rdd(df):
    num = _num(df)
    bins = _binary(df)
    c1, c2, c3 = st.columns(3)
    y = c1.selectbox(tr("Outcome", "النتيجة"), _prefer([c for c in num if c not in bins], ("gpa", "outcome", "ln_")), key="rd_y")
    x = c2.selectbox(tr("Running variable", "المتغير الجاري"), _prefer([c for c in num if c != y and c not in bins], ("score", "running", "age", "distance")), key="rd_x")
    cut = c3.number_input(tr("Cutoff", "نقطة القطع"), value=float(np.round(df[x].median(), 2)) if "score" not in x else 60.0, key="rd_c")
    a, b, c = st.columns(3)
    fuzzy = a.selectbox(tr("Treatment take-up (fuzzy RD)", "تلقي المعالجة (تصميم غامض)"), [None] + bins, key="rd_f", format_func=lambda v: tr("None (sharp)", "بدون (حاد)") if v is None else v)
    p = b.selectbox(tr("Polynomial order", "درجة كثيرة الحدود"), [1, 2], key="rd_p")
    kern = c.selectbox(tr("Kernel", "النواة"), ["triangular", "epanechnikov", "uniform"], key="rd_k")
    covs = st.multiselect(tr("Covariates for continuity checks", "متغيرات لفحص الاستمرارية"), [v for v in num + bins if v not in (x, y, fuzzy)], [v for v in bins if v not in (fuzzy,) and "elig" not in v][:1],
                          key="rd_cv", placeholder=tr("Optional", "اختياري"))
    sig = (y, x, cut, fuzzy, p, kern, tuple(covs), active_name())
    r = _load("rd", sig)
    if r is None:
        with st.spinner("rdrobust…"):
            r = cz.rdd(df, y, x, cut, fuzzy, p, kern, covs=covs)
        _store("rd", r, sig)
    _table(r["table"], 4)
    st.caption(tr(f"MSE-optimal bandwidth h = {r['h']:.3f} (bias b = {r['b']:.3f}); effective N = {r['N eff left']} left / {r['N eff right']} right.",
                  f"عرض النطاق الأمثل h = {r['h']:.3f} (التحيز b = {r['b']:.3f})؛ N الفعّال = {r['N eff left']} يسار / {r['N eff right']} يمين."))
    B = r["binned"]
    fig = go.Figure()
    for j, side in enumerate(("left", "right")):
        s = B["bins"][B["bins"].side == side]
        f = B["fit"][B["fit"].side == side]
        fig.add_trace(go.Scatter(x=s.x, y=s.y, mode="markers", marker=dict(color=PAL[j], size=8), name=_disp(pd.DataFrame({"c": [side]})).iloc[0, 0]))
        fig.add_trace(go.Scatter(x=f.x, y=f.fit, line=dict(color=PAL[j], width=2.5), showlegend=False))
    fig.add_vline(x=cut, line=dict(color="#c0392b", dash="dash"))
    ch.style_academic(fig, height=400, x_title=x, y_title=y, legend="bottom", rtl=is_ar())
    c1, c2 = st.columns(2)
    with c1:
        _chart(fig, "rd_plot", "rd_plot")
    with c2:
        S = r["sensitivity"]
        fig = go.Figure(go.Scatter(x=S.h, y=S.Estimate, mode="markers+lines", marker=dict(color=PAL[0], size=9),
                                   error_y=dict(type="data", symmetric=False, array=S["Robust CI high"] - S.Estimate, arrayminus=S.Estimate - S["Robust CI low"])))
        fig.add_hline(y=0, line=dict(color="#999"))
        fig.add_vline(x=r["h"], line=dict(color="#999", dash="dot"))
        ch.style_academic(fig, height=400, x_title=tr("Bandwidth", "عرض النطاق"), y_title=tr("RD estimate", "تقدير الانقطاع"), legend="none", rtl=is_ar())
        _chart(fig, "rd_sens", "rd_bandwidth_sensitivity")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**" + tr("Placebo cutoffs", "نقاط القطع الوهمية") + "**")
        _table(r["placebo"], 4)
    with c2:
        if r.get("covariates") is not None:
            st.markdown("**" + tr("Covariate continuity", "استمرارية المتغيرات المصاحبة") + "**")
            _table(r["covariates"], 4)
        if r["density test"]:
            dt = r["density test"]
            st.caption(tr(f"Manipulation test (rddensity): T = {dt['T (jackknife)']:.3f}, p = {fp(dt['p-value'])}",
                          f"اختبار التلاعب (rddensity): T = {dt['T (jackknife)']:.3f}، p = {fp(dt['p-value'])}"))
    rb = r["table"].iloc[2]
    cv = r["table"].iloc[0]
    msg = tr(f"Jump at the cutoff = {cv.Estimate:.4f}; robust 95% CI {rb['CI low']:.4f} to {rb['CI high']:.4f} (p {fp(rb['p-value'])}).",
             f"القفزة عند نقطة القطع = {cv.Estimate:.4f}؛ فترة الثقة المتينة 95%: {rb['CI low']:.4f} إلى {rb['CI high']:.4f} (p {fp(rb['p-value'])}).")
    if fuzzy and "first stage" in r:
        msg += tr(f" Take-up jumps by {r['first stage']['Jump in treatment']:.3f}; the estimate is the effect for compliers at the cutoff.",
                  f" يقفز التلقي بمقدار {r['first stage']['Jump in treatment']:.3f}؛ والتقدير هو الأثر للممتثلين عند نقطة القطع.")
    if r["density test"]:
        msg += tr(" No sign of sorting around the cutoff." if r["density test"]["p-value"] > 0.05 else " ⚠️ The density jumps at the cutoff — possible manipulation.",
                  " لا دليل على الفرز حول نقطة القطع." if r["density test"]["p-value"] > 0.05 else " ⚠️ تقفز الكثافة عند نقطة القطع — احتمال تلاعب.")
    insight(msg)
    with st.expander(tr("🎯 Policy simulation: moving the cutoff", "🎯 محاكاة السياسة: تحريك نقطة القطع"), expanded=False):
        newc = st.slider(tr("New cutoff", "نقطة القطع الجديدة"), float(df[x].quantile(0.05)), float(df[x].quantile(0.95)), float(cut), key="rd_nc")
        n_new = int(((df[x] >= newc) & (df[x] < cut)).sum()) if newc < cut else -int(((df[x] >= cut) & (df[x] < newc)).sum())
        st.metric(tr("Change in the number of eligible units", "التغير في عدد المؤهلين"), f"{n_new:+,d}")
        st.caption(tr(f"Aggregate effect ≈ {n_new:+,d} × {cv.Estimate:.4f} = {n_new * cv.Estimate:+.2f} (assumes a constant effect near the cutoff — extrapolation).",
                      f"الأثر الإجمالي ≈ {n_new:+,d} × {cv.Estimate:.4f} = {n_new * cv.Estimate:+.2f} (بافتراض ثبات الأثر قرب نقطة القطع — استقراء)."))
    _finish("rdd", "pe_rd", {"Estimates": r["table"], "Sensitivity": r["sensitivity"], "Placebo": r["placebo"], "Bins": B["bins"],
                             **({"Covariates": r["covariates"]} if r.get("covariates") is not None else {})},
            tr("Regression discontinuity", "انقطاع الانحدار"), [r["table"], r["sensitivity"]], {"y": y, "running": x, "cutoff": cut})


# ---------------------------------------------------------- 8. matching & weighting
def _matching(df):
    bins = _binary(df)
    num = _num(df)
    if not bins:
        st.info(tr("A binary (0/1) treatment variable is required.", "يلزم متغير معالجة ثنائي (0/1)."))
        return
    c1, c2, c3 = st.columns(3)
    t = c1.selectbox(tr("Treatment", "المعالجة"), _prefer(bins, ("particip", "treat", "program")), key="mt_t")
    y = c2.selectbox(tr("Outcome", "النتيجة"), _prefer([c for c in num if c not in bins], ("ln_inc", "income", "ln_")), key="mt_y")
    est = c3.selectbox(tr("Estimand", "الكمية المقدرة"), ["ATT", "ATE"], key="mt_e")
    dflt = [c for c in num if c not in (t, y) and c not in bins][:5] + [b for b in bins if b != t and any(k in b for k in ("female", "head", "urban", "male"))]
    covs = st.multiselect(tr("Confounders", "المتغيرات المربكة"), [c for c in num + bins if c not in (t, y)], dflt, key="mt_c",
                          placeholder=tr("Choose…", "اختر…"))
    if not covs:
        return
    a, b = st.columns(2)
    cal = a.select_slider(tr("Caliper (SD of logit score)", "المعايرة (انحراف لوجيت الدرجة)"), [None, 0.05, 0.1, 0.2, 0.25, 0.5], None, key="mt_cal",
                          format_func=lambda v: tr("none", "بدون") if v is None else str(v))
    trim = b.select_slider(tr("Trim propensity scores", "تشذيب درجات الميل"), [None, 0.01, 0.05, 0.1], None, key="mt_tr", format_func=lambda v: tr("none", "بدون") if v is None else f"[{v}, {1 - v}]")
    m = cz.psm(df, y, t, covs, 1, cal, est)
    w = cz.ipw_aipw(df, y, t, covs, est, trim)
    tab = pd.concat([pd.DataFrame([{"Estimator": f"PSM 1:1 ({est})", **{k: v for k, v in m["estimate"].items() if k not in ("Term", "")}}]),
                     w["table"].drop(columns=["Term", ""])], ignore_index=True)
    naive = df[[y, t]].dropna().groupby(t)[y].mean()
    tab = pd.concat([pd.DataFrame([{"Estimator": tr("Naive difference", "الفرق الساذج"), "Estimate": naive.get(1, np.nan) - naive.get(0, np.nan)}]), tab], ignore_index=True)
    _table(tab, 4)
    bal = m["balance"].merge(w["balance"][["Covariate", "SMD after weighting"]], on="Covariate")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for j, col in enumerate(["SMD before", "SMD after matching", "SMD after weighting"]):
            fig.add_trace(go.Scatter(x=bal[col].abs(), y=bal.Covariate, mode="markers", name=_disp(pd.DataFrame(columns=[col])).columns[0],
                                     marker=dict(size=11, color=PAL[j], symbol=["circle", "diamond", "square"][j])))
        fig.add_vline(x=0.1, line=dict(color="#c0392b", dash="dash"))
        ch.style_academic(fig, height=380, x_title="|SMD|", legend="bottom", rtl=is_ar())
        _chart(fig, "mt_love", "love_plot")
    with c2:
        O = cz.overlap(w["ps"], w["D"])
        fig = go.Figure([go.Bar(x=O.ps, y=O.Treated, name=tr("Treated", "المعالَجون"), marker_color=PAL[0], opacity=0.7),
                         go.Bar(x=O.ps, y=-O.Control, name=tr("Control", "الضابطة"), marker_color=PAL[1], opacity=0.7)])
        fig.update_layout(barmode="overlay")
        ch.style_academic(fig, height=380, x_title=tr("Propensity score", "درجة الميل"), y_title=tr("Density", "الكثافة"), legend="bottom", rtl=is_ar())
        _chart(fig, "mt_ov", "overlap")
    _table(bal, 3)
    lo, hi = w["ps range"]
    insight(tr(f"Naive gap {tab.Estimate[0]:.4f} vs adjusted {est} ≈ {tab.Estimate[1:].mean():.4f} — selection bias of about {tab.Estimate[0] - tab.Estimate[1:].mean():.4f}. "
               f"Max |SMD| after weighting = {bal['SMD after weighting'].abs().max():.3f}; scores range {lo:.3f}–{hi:.3f}" + (" (⚠️ near 0/1: consider trimming)." if lo < 0.02 or hi > 0.98 else "."),
               f"الفرق الساذج {tab.Estimate[0]:.4f} مقابل {est} المعدَّل ≈ {tab.Estimate[1:].mean():.4f} — أي تحيز انتقاء بنحو {tab.Estimate[0] - tab.Estimate[1:].mean():.4f}. "
               f"أقصى |SMD| بعد الترجيح = {bal['SMD after weighting'].abs().max():.3f}؛ وتتراوح الدرجات بين {lo:.3f} و{hi:.3f}" + (" (⚠️ قريبة من 0/1: فكّر في التشذيب)." if lo < 0.02 or hi > 0.98 else ".")))
    _cba(float(tab.Estimate.iloc[-1]), float(tab["Std. Err."].iloc[-1]), "mt")
    _finish("matching", "pe_mt", {"Estimates": tab, "Balance": bal, "Overlap": cz.overlap(w["ps"], w["D"])}, tr("Matching and weighting", "المطابقة والترجيح"),
            [tab, bal], {"y": y, "x": covs, "treat": t})


# ---------------------------------------------------------- 9. climate impacts
def _climate(df):
    cols = [c.lower() for c in df.columns]
    default = "dd" if "tmin" in cols and "tmax" in cols else "ric" if any("land_value" in c for c in cols) else "panel"
    mode = st.radio(tr("Analysis", "التحليل"), ["dd", "panel", "ric"], ["dd", "panel", "ric"].index(default), horizontal=True, key="cl_mode",
                    format_func=lambda k: {"dd": tr("Degree days from daily weather", "درجات الأيام من الطقس اليومي"), "panel": tr("Temperature-yield panel", "بانل الحرارة والإنتاجية"),
                                           "ric": tr("Ricardian land values", "قيم الأراضي (ريكاردي)")}[k])
    num = _num(df)
    if mode == "dd":
        if not ("tmin" in cols and "tmax" in cols):
            st.info(tr("Needs daily tmin and tmax columns (load the daily weather demo).", "يحتاج عمودَي tmin وtmax اليوميين (حمّل بيانات الطقس اليومي)."))
            return
        tmin, tmax = df.columns[cols.index("tmin")], df.columns[cols.index("tmax")]
        a, b, c, d = st.columns(4)
        grp = a.multiselect(tr("Group by", "التجميع حسب"), [x for x in df.columns if x not in (tmin, tmax) and df[x].nunique() < 500], [x for x in ("station", "season") if x in df], key="dd_g")
        base = b.number_input(tr("Base (°C)", "حد الأساس (°م)"), value=8.0, key="dd_b")
        upper = c.number_input(tr("Upper threshold (°C)", "العتبة العليا (°م)"), value=29.0, key="dd_up")
        meth = d.selectbox(tr("Method", "الطريقة"), ["sine", "average"], key="dd_m", format_func=lambda k: {"sine": tr("Single sine", "الجيب المفرد"), "average": tr("Daily average", "المتوسط اليومي")}[k])
        rain = next((x for x in df.columns if "rain" in x.lower() or "precip" in x.lower()), None)
        A = cl.degree_days(df, tmin, tmax, grp, base, upper, meth, rain)
        _table(A, 2)
        gcol, kcol = f"GDD {base:g}–{upper:g}°C", f"KDD > {upper:g}°C"
        fig = go.Figure()
        lab = A[grp].astype(str).agg(" · ".join, axis=1) if grp else pd.Series(["all"])
        fig.add_trace(go.Bar(x=lab, y=A[gcol], name="GDD", marker_color=PAL[0]))
        fig.add_trace(go.Bar(x=lab, y=A[kcol], name="KDD", marker_color=PAL[2], yaxis="y2"))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False), barmode="group")
        ch.style_academic(fig, height=380, legend="bottom", rtl=is_ar())
        _chart(fig, "dd_bar", "degree_days")
        with st.expander(tr("🎯 Warming scenario: degree days under +1 … +4 °C", "🎯 سيناريو الاحترار: درجات الأيام عند +1 … +4 °م"), expanded=True):
            Wd = cl.warming_degree_days(df, tmin, tmax, grp, base, upper)
            avg = Wd.groupby("warming (°C)")[[gcol, kcol]].mean().reset_index()
            _table(avg, 1)
            k0, k4 = avg[kcol].iloc[0], avg[kcol].iloc[-1]
            insight(tr(f"Killing degree days rise from {k0:.0f} to {k4:.0f} with +4 °C ({k4 / max(k0, 1e-9):.1f}×) — heat damage grows far faster than average temperature.",
                       f"ترتفع درجات الأيام الضارة من {k0:.0f} إلى {k4:.0f} مع +4 °م ({k4 / max(k0, 1e-9):.1f} مرة) — يتسارع الضرر الحراري أسرع كثيراً من متوسط الحرارة."))
        _finish("degree_days", "cl_dd", {"Degree days": A, "Warming": Wd}, tr("Degree days", "درجات الأيام"), [A.head(30)])
        return
    if mode == "ric":
        a, b = st.columns([1, 2])
        y = a.selectbox(tr("Land value", "قيمة الأرض"), _prefer(num, ("land_value", "value", "rent")), key="ri_y")
        clim = b.multiselect(tr("Climate normals", "المتوسطات المناخية"), [c for c in num if c != y], [c for c in num if any(k in c for k in ("temp", "precip", "rain"))], key="ri_c",
                             placeholder=tr("Choose…", "اختر…"))
        ctrl = st.multiselect(tr("Controls", "الضوابط"), [c for c in num + _binary(df) if c not in clim and c != y], [c for c in num + _binary(df) if c not in clim and c != y][:3], key="ri_x",
                              placeholder=tr("Optional", "اختياري"))
        if not clim:
            return
        st.markdown("**" + tr("🎯 Climate scenario", "🎯 السيناريو المناخي") + "**")
        sc = st.columns(len(clim))
        deltas = {}
        for j, c in enumerate(clim):
            if "temp" in c:
                deltas[c] = ("abs", sc[j].slider(f"Δ {c} (°C)", -2.0, 5.0, 2.0, 0.5, key=f"ri_d{j}"))
            else:
                deltas[c] = ("pct", sc[j].slider(f"Δ {c} (%)", -50, 50, -10, 5, key=f"ri_d{j}"))
        r = cl.ricardian(df, y, clim, ctrl, True, deltas)
        _table(r["table"].drop(columns=[""]), 5)
        _table(r["marginal"], 4)
        s = r["scenario"]
        st.dataframe(_disp(pd.DataFrame([s])).round(3), width="stretch", hide_index=True)
        v = st.selectbox(tr("Response curve for", "منحنى الاستجابة لـ"), clim, key="ri_v")
        j = clim.index(v)
        grid = np.linspace(df[v].min(), df[v].max(), 60)
        b1, b2 = r["b"][1 + 2 * j], r["b"][2 + 2 * j]
        xm = df[v].mean()
        fig = go.Figure(go.Scatter(x=grid, y=100 * (np.exp(b1 * (grid - xm) + b2 * (grid ** 2 - xm ** 2)) - 1), line=dict(color=PAL[0], width=3)))
        fig.add_vline(x=xm, line=dict(color="#999", dash="dot"))
        ch.style_academic(fig, height=340, x_title=v, y_title=tr("% land value vs mean climate", "% قيمة الأرض مقارنة بالمتوسط"), legend="none", rtl=is_ar())
        _chart(fig, "ri_curve", f"ricardian_{v}")
        insight(tr(f"Under the scenario, land values change by {s['% change']:.2f}% on average (95% CI {s['% CI low']:.2f}% to {s['% CI high']:.2f}%); {100 * s['Share of farms losing']:.0f}% of farms lose. "
                   "Because land values capitalise adapted farm profits, this already includes farmers' adaptation.",
                   f"وفق السيناريو تتغير قيم الأراضي بنحو {s['% change']:.2f}% في المتوسط (فترة ثقة 95%: {s['% CI low']:.2f}% إلى {s['% CI high']:.2f}%)، ويخسر {100 * s['Share of farms losing']:.0f}% من المزارع. "
                   "ولأن قيمة الأرض ترسمل الأرباح بعد التكيف، فهذا يتضمن تكيّف المزارعين بالفعل."))
        _finish("ricardian", "cl_ric", {"Coefficients": r["table"], "Marginal effects": r["marginal"], "Scenario": pd.DataFrame([s])}, tr("Ricardian analysis", "التحليل الريكاردي"),
                [r["table"], r["marginal"]], {"y": y, "x": clim})
        return
    cats = _cats(df)
    tcols = [c for c in df.columns if c.lower() in ("year", "season", "time")]
    if not (cats and tcols):
        st.info(tr("Needs a location id and year (load the crop-weather panel demo).", "يحتاج معرّف موقع وسنة (حمّل بانل الإنتاجية والطقس)."))
        return
    has = {"gddkdd": any("gdd" in x for x in num) and any("kdd" in x for x in num), "bins": any(x.startswith("days") for x in df.columns),
           "quad": any("temp" in x or "tmean" in x for x in num)}
    specs = [k_ for k_ in cl.SPECS if has[k_]]
    if not specs:
        st.info(tr("A temperature-response panel needs weather columns: GDD and KDD, 'days_…' temperature bins, or a season temperature. "
                   "Compute them from daily data in the degree-days option, or load the crop-weather panel demo.",
                   "يحتاج بانل استجابة الإنتاجية للحرارة إلى أعمدة طقس: درجات أيام النمو والضرر، أو فئات الحرارة 'days_…'، أو حرارة الموسم. "
                   "احسبها من البيانات اليومية في خيار درجات الأيام، أو حمّل بانل الإنتاجية والطقس."))
        return
    a, b, c, d = st.columns(4)
    u = a.selectbox(tr("Location", "الموقع"), cats, key="tp_u")
    t = b.selectbox(tr("Year", "السنة"), tcols, key="tp_t")
    y = c.selectbox(tr("Outcome (log yield)", "النتيجة (لوغاريتم الإنتاجية)"), _prefer(num, ("ln_yield", "yield")), key="tp_y")
    spec = d.selectbox(tr("Temperature specification", "صيغة الحرارة"), specs, key="tp_s", format_func=lambda k: _t(cl.SPECS[k]))
    e1, e2, e3 = st.columns(3)
    pcol = next((x for x in num if "precip" in x or "rain" in x), None)
    pr_ = e1.selectbox(tr("Precipitation", "الأمطار"), [None] + num, (num.index(pcol) + 1) if pcol else 0, key="tp_p",
                       format_func=lambda v: tr("None", "بدون") if v is None else v)
    trend = e2.checkbox(tr("Quadratic trend instead of year FE", "اتجاه تربيعي بدل آثار السنوات"), False, key="tp_tr")
    tm = next((x for x in num if "tmean" in x or "temp" in x), None)
    kw = {}
    if spec == "gddkdd":
        kw = dict(gdd=next((x for x in num if "gdd" in x), num[0]), kdd=next((x for x in num if "kdd" in x), num[0]), tmean=tm)
    elif spec == "bins":
        bl = [x for x in df.columns if x.startswith("days")]
        if not bl:
            st.info(tr("No 'days_…' bin columns found.", "لا توجد أعمدة فئات 'days_…'."))
            return
        ref = e3.selectbox(tr("Reference bin", "الفئة المرجعية"), bl, min(3, len(bl) - 1), key="tp_ref")
        kw = dict(bins=bl, ref_bin=ref)
    else:
        kw = dict(tmean=e3.selectbox(tr("Season mean temperature", "متوسط حرارة الموسم"), num, num.index(tm) if tm in num else 0, key="tp_tm"))
    r = cl.temp_response(df, u, t, y, spec, precip=pr_, trend=trend, **kw)
    _table(r["table"].drop(columns=[""]), 6)
    st.caption(f"N = {r['N']} · {tr('locations', 'مواقع')} = {r['units']} · " + tr("SEs clustered by location", "أخطاء عنقودية حسب الموقع"))
    if spec == "bins":
        rc = cl.response_curve(r)
        fig = go.Figure(go.Bar(x=rc.Bin, y=rc["Effect of one day"], marker_color=[PAL[2] if v < 0 else PAL[0] for v in rc["Effect of one day"]],
                               error_y=dict(type="data", array=1.96 * rc["Std. Err."])))
        ch.style_academic(fig, height=360, y_title=tr("Effect of one more day vs reference", "أثر يوم إضافي مقارنة بالمرجع"), legend="none", rtl=is_ar())
        _chart(fig, "tp_bins", "temperature_bins")
    elif spec == "quad":
        rc = cl.response_curve(r, np.linspace(df[kw["tmean"]].min(), df[kw["tmean"]].max(), 60))
        fig = go.Figure([go.Scatter(x=np.r_[rc.Temperature, rc.Temperature[::-1]], y=np.r_[rc["CI high"], rc["CI low"][::-1]], fill="toself", fillcolor="rgba(31,78,121,0.18)", line=dict(width=0), showlegend=False),
                         go.Scatter(x=rc.Temperature, y=rc.Effect, line=dict(color=PAL[0], width=3), showlegend=False)])
        ch.style_academic(fig, height=360, x_title=kw["tmean"], y_title=tr("Δ log yield vs optimum", "Δ لوغاريتم الإنتاجية مقارنة بالأمثل"), legend="none", rtl=is_ar())
        _chart(fig, "tp_quad", "temperature_response")
    st.markdown("**" + tr("🎯 Warming scenarios", "🎯 سيناريوهات الاحترار") + "**")
    if spec == "gddkdd":
        a, b = st.columns(2)
        dg = a.number_input(tr("ΔGDD per +1 °C", "ΔGDD لكل +1 °م"), value=float(np.round(r.get("dGDD/dT", 100.0), 1)), key="tp_dg")
        dk = b.number_input(tr("ΔKDD per +1 °C", "ΔKDD لكل +1 °م"), value=float(np.round(r.get("dKDD/dT", 20.0), 1)), key="tp_dk")
        st.caption(tr("Defaults are the empirical within-location slopes of degree days on season temperature; replace them with values from daily data (Degree-days tab).",
                      "القيم الافتراضية هي الميول التجريبية لدرجات الأيام على حرارة الموسم داخل المواقع؛ ويمكن استبدالها بقيم من البيانات اليومية (تبويب درجات الأيام)."))
        Wi = cl.warming_impact(r, dgdd=dg, dkdd=dk)
    elif spec == "quad":
        Wi = cl.warming_impact(r, base_tmean=float(df[kw["tmean"]].mean()))
    else:
        bl = [b_ for b_ in r["b"] if b_.startswith("days")]
        hot = [b_ for b_ in bl if any(k in b_ for k in ("30", "35", "gt"))]
        cool = [b_ for b_ in bl if b_ not in hot]
        Wi = cl.warming_impact(r, bin_shift={**{h: 6.0 / max(len(hot), 1) for h in hot}, **{c: -6.0 / max(len(cool), 1) for c in cool}})
        st.caption(tr("Approximation: each +1 °C moves 6 season days from cooler bins into hot bins.", "تقريب: كل +1 °م ينقل 6 أيام من الفئات الأبرد إلى الفئات الحارة."))
    _table(Wi, 3)
    fig = go.Figure(go.Bar(x=Wi["Warming (°C)"], y=Wi["% change"], marker_color=PAL[2], error_y=dict(type="data", symmetric=False, array=Wi["% CI high"] - Wi["% change"],
                                                                                                   arrayminus=Wi["% change"] - Wi["% CI low"])))
    ch.style_academic(fig, height=320, x_title=tr("Warming (°C)", "الاحترار (°م)"), y_title=tr("% yield change", "% تغير الإنتاجية"), legend="none", rtl=is_ar())
    _chart(fig, "tp_warm", "warming_impacts")
    w2 = Wi[Wi["Warming (°C)"] == 2].iloc[0]
    msg = tr(f"A 2 °C warming changes yields by {w2['% change']:.1f}% (95% CI {w2['% CI low']:.1f}% to {w2['% CI high']:.1f}%) without adaptation.",
             f"احترار 2 °م يغيّر الإنتاجية بنحو {w2['% change']:.1f}% (فترة ثقة 95%: {w2['% CI low']:.1f}% إلى {w2['% CI high']:.1f}%) دون تكيف.")
    if spec == "gddkdd":
        bb = r["b"]
        g_, k_ = list(bb)[:2]
        msg += tr(f" One extra killing degree day costs {abs(bb[k_]) / max(abs(bb[g_]), 1e-12):.1f}× what an extra growing degree day adds.",
                  f" يوم درجة ضار إضافي يكلّف {abs(bb[k_]) / max(abs(bb[g_]), 1e-12):.1f} مرة ما يضيفه يوم درجة نمو إضافي.")
    if spec == "quad" and np.isfinite(r.get("optimum", np.nan)):
        msg += tr(f" Optimal season temperature ≈ {r['optimum']:.1f} °C.", f" حرارة الموسم المثلى ≈ {r['optimum']:.1f} °م.")
    insight(msg)
    _finish("temp_response", "cl_tp", {"Coefficients": r["table"], "Warming impacts": Wi}, tr("Temperature response", "استجابة الإنتاجية للحرارة"), [r["table"], Wi],
            {"y": y, "x": [kw.get("gdd", "gdd"), kw.get("kdd", "kdd")] if spec == "gddkdd" else ["tmean_season"], "unit": u, "time": t})


# ---------------------------------------------------------- 10. emissions & damages
def _emissions(df):
    mode = st.radio(tr("Analysis", "التحليل"), ["bhm", "scc", "stirpat", "ekc", "tapio"], horizontal=True, key="em_mode",
                    format_func=lambda k: {"bhm": tr("Growth damages (BHM)", "أضرار النمو (BHM)"), "scc": tr("Social cost of carbon", "تكلفة الكربون الاجتماعية"),
                                           "stirpat": "STIRPAT", "ekc": tr("Kuznets curve (EKC)", "منحنى كوزنتس البيئي"), "tapio": tr("Decoupling & LMDI", "فك الارتباط وLMDI")}[k])
    if mode == "scc":
        return _scc()
    cats = _cats(df)
    tcols = [c for c in df.columns if c.lower() in ("year", "time", "period")]
    num = _num(df)
    if not (cats and tcols):
        st.info(tr("Needs a country/unit id and year (load the country emissions demo).", "يحتاج معرّف دولة/وحدة وسنة (حمّل بيانات الانبعاثات الدولية)."))
        return
    need = {"bhm": (("growth",), ("temp",)), "stirpat": (("co2", "emission", "energy"),), "ekc": (("co2", "emission"), ("gdp", "income")),
            "tapio": (("co2", "emission"), ("gdp",))}[mode]
    if not all(any(k_ in x.lower() for x in num for k_ in grp) for grp in need):
        st.info(tr("This analysis needs columns such as " + " + ".join("/".join(g) for g in need) + " (load the country emissions demo).",
                   "يحتاج هذا التحليل إلى أعمدة مثل " + " + ".join("/".join(g) for g in need) + " (حمّل بيانات الانبعاثات الدولية)."))
        return
    a, b = st.columns(2)
    u = a.selectbox(tr("Unit", "الوحدة"), cats, key="em_u")
    t = b.selectbox(tr("Year", "السنة"), tcols, key="em_t")
    if mode == "bhm":
        a, b, c = st.columns(3)
        g = a.selectbox(tr("GDP per capita growth", "نمو نصيب الفرد من الناتج"), _prefer(num, ("growth",)), key="bh_g")
        T = b.selectbox(tr("Temperature", "الحرارة"), _prefer(num, ("temp",)), key="bh_T")
        tr_ = c.selectbox(tr("Unit-specific trends", "اتجاهات خاصة بكل وحدة"), ["linear", "quadratic", "none"], key="bh_tr")
        r = cl.bhm(df, u, t, g, T, trends=tr_)
        _table(r["table"].drop(columns=[""]), 6)
        C = r["curve"]
        fig = go.Figure([go.Scatter(x=np.r_[C.Temperature, C.Temperature[::-1]], y=100 * np.r_[C["CI high"], C["CI low"][::-1]], fill="toself", fillcolor="rgba(31,78,121,0.18)", line=dict(width=0), showlegend=False),
                         go.Scatter(x=C.Temperature, y=100 * C["Growth effect vs optimum"], line=dict(color=PAL[0], width=3), showlegend=False)])
        fig.add_vline(x=r["optimum"], line=dict(color=PAL[2], dash="dash"), annotation_text=f"{r['optimum']:.1f} °C")
        ch.style_academic(fig, height=380, x_title=tr("Annual average temperature (°C)", "متوسط الحرارة السنوي (°م)"), y_title=tr("Growth effect (pp)", "أثر النمو (نقطة مئوية)"), legend="none", rtl=is_ar())
        _chart(fig, "bh_curve", "bhm_response")
        st.markdown("**" + tr("🎯 Projection under warming", "🎯 الإسقاط تحت الاحترار") + "**")
        a, b, c = st.columns(3)
        warm = a.slider(tr("Warming by the horizon (°C)", "الاحترار حتى نهاية الأفق (°م)"), 0.5, 5.0, 3.0, 0.5, key="bh_w")
        yrs = b.slider(tr("Horizon (years)", "الأفق (سنوات)"), 20, 100, 80, 10, key="bh_y")
        g0 = c.slider(tr("Baseline growth (%/yr)", "النمو الأساسي (%/سنة)"), 0.0, 5.0, 2.0, 0.25, key="bh_g0") / 100
        gdp = next((x for x in num if "gdp_pc" in x), None)
        popc = next((x for x in num if "pop" in x), None)
        if gdp:
            base = df[df[t] == df[t].max()]
            pj = cl.bhm_projection(r, base, u, T, gdp, g0, warm, yrs, popc)
            U = pj["units"].sort_values("Baseline T")
            fig = go.Figure(go.Scatter(x=U["Baseline T"], y=U["% change"], mode="markers", marker=dict(color=np.where(U["% change"] < 0, "#c0392b", "#27ae60"), size=9), text=U[u]))
            fig.add_hline(y=0, line=dict(color="#999"))
            ch.style_academic(fig, height=360, x_title=tr("Baseline temperature (°C)", "حرارة الأساس (°م)"), y_title=tr("% GDP per capita at horizon", "% الناتج للفرد في نهاية الأفق"), legend="none", rtl=is_ar())
            _chart(fig, "bh_proj", "bhm_projection")
            msg = tr(f" With +{warm} °C over {yrs} years, aggregate GDP per capita is {pj['aggregate % change']:.1f}% lower" + (" (population-weighted)" if popc else "") +
                     f"; {100 * pj['share losing']:.0f}% of units lose — hot countries lose most.",
                     f" مع +{warm} °م خلال {yrs} سنة ينخفض نصيب الفرد الإجمالي من الناتج بنحو {abs(pj['aggregate % change']):.1f}%" + (" (موزون بالسكان)" if popc else "") +
                     f"؛ ويخسر {100 * pj['share losing']:.0f}% من الوحدات — وتخسر الدول الحارة أكثر.")
        else:
            pj, msg = None, ""
        lo, hi = r["optimum CI"]
        insight(tr(f"Growth peaks at {r['optimum']:.1f} °C (95% CI {lo:.1f}–{hi:.1f} °C); warming hurts units above it and helps colder ones.",
                   f"يبلغ النمو ذروته عند {r['optimum']:.1f} °م (فترة ثقة 95%: {lo:.1f}–{hi:.1f} °م)؛ يضر الاحترار الوحدات الأدفأ من ذلك ويفيد الأبرد.") + msg)
        _finish("bhm", "em_bhm", {"Coefficients": r["table"], "Curve": C, **({"Projection": pj["units"]} if pj else {})}, tr("Climate damages (BHM)", "أضرار المناخ (BHM)"),
                [r["table"]] + ([pj["units"].head(20)] if pj else []), {"y": g, "unit": u, "time": t})
        return
    if mode == "stirpat":
        a, b = st.columns([1, 2])
        I = a.selectbox(tr("Impact (e.g. CO2)", "الأثر (مثل CO2)"), _prefer(num, ("co2", "emission")), key="stp_i")
        drv = b.multiselect(tr("Drivers (P, A, T)", "المحددات (السكان، الرخاء، التقنية)"), [c for c in num if c != I], [c for c in num if any(k in c for k in ("pop", "gdp_pc", "energy"))][:3], key="stp_d",
                            placeholder=tr("Choose…", "اختر…"))
        fe = st.radio(tr("Fixed effects", "الآثار الثابتة"), ["twoway", "unit", "none"], horizontal=True, key="stp_fe")
        if not drv:
            return
        r = cl.stirpat(df, u, t, I, drv, fe, ridge=True)
        _table(r["table"].drop(columns=[""]), 4)
        _table(r["ridge"], 4)
        st.markdown("**" + tr("🎯 Emission projection", "🎯 إسقاط الانبعاثات") + "**")
        cols = st.columns(len(drv))
        growth = {v: cols[j].slider(f"{v} (%/yr)", -5.0, 8.0, 1.5 if "pop" in v else 3.0 if "gdp" in v else -1.0, 0.25, key=f"stp_g{j}") for j, v in enumerate(drv)}
        tot = df[df[t] == df[t].max()][I].sum()
        P = cl.emission_projection(r["b"], {"impact": tot}, growth, 30, int(df[t].max()))
        fig = go.Figure(go.Scatter(x=P.Year, y=P.Impact, line=dict(color=PAL[0], width=3)))
        pk = P[P.Peak].iloc[0]
        fig.add_trace(go.Scatter(x=[pk.Year], y=[pk.Impact], mode="markers", marker=dict(size=14, color=PAL[2], symbol="star"), name=tr("Peak", "الذروة")))
        ch.style_academic(fig, height=340, x_title=tr("Year", "السنة"), y_title=I, legend="none", rtl=is_ar())
        _chart(fig, "stp_proj", "stirpat_projection")
        el = r["table"].set_index("Term").Estimate
        insight(tr("Elasticities: " + ", ".join(f"{k} {v:.3f}" for k, v in el.items()) + f". Under the scenario the impact peaks in {int(pk.Year)}"
                   + (" (still rising at the horizon)." if pk.Year == P.Year.max() else "."),
                   "المرونات: " + "، ".join(f"{k} {v:.3f}" for k, v in el.items()) + f". وفق السيناريو تبلغ الانبعاثات ذروتها في {int(pk.Year)}"
                   + (" (ما زالت ترتفع حتى نهاية الأفق)." if pk.Year == P.Year.max() else ".")))
        _finish("stirpat", "em_stp", {"STIRPAT": r["table"], "Ridge": r["ridge"], "Projection": P}, "STIRPAT", [r["table"], r["ridge"]], {"y": I, "x": drv, "unit": u, "time": t})
        return
    if mode == "ekc":
        a, b, c = st.columns(3)
        E = a.selectbox(tr("Emissions", "الانبعاثات"), _prefer(num, ("co2", "emission")), key="ek_e")
        Y = b.selectbox(tr("Income per capita", "الدخل للفرد"), _prefer([x for x in num if x != E], ("gdp_pc", "income")), key="ek_y")
        pc_ = next((x for x in num if "pop" in x), None)
        P_ = c.selectbox(tr("Divide emissions by (per capita)", "قسمة الانبعاثات على (للفرد)"), [None] + num, (num.index(pc_) + 1) if pc_ else 0,
                         key="ek_p", format_func=lambda v: tr("None", "بدون") if v is None else v)
        r = cl.ekc(df, u, t, E, Y, per_capita=P_)
        _table(r["table"].drop(columns=[""]), 5)
        if "curve" in r:
            C = r["curve"]
            fig = go.Figure([go.Scatter(x=np.r_[C.income, C.income[::-1]], y=np.r_[C["CI high"], C["CI low"][::-1]], fill="toself", fillcolor="rgba(31,78,121,0.18)", line=dict(width=0), showlegend=False),
                             go.Scatter(x=C.income, y=C["ln E relative to mean income"], line=dict(color=PAL[0], width=3), showlegend=False)])
            if np.isfinite(r["turning point"]) and C.income.min() < r["turning point"] < C.income.max():
                fig.add_vline(x=r["turning point"], line=dict(color=PAL[2], dash="dash"), annotation_text=f"{r['turning point']:,.0f}")
            fig.update_xaxes(type="log")
            ch.style_academic(fig, height=380, x_title=Y, y_title=tr("ln emissions (relative)", "لوغاريتم الانبعاثات (نسبي)"), legend="none", rtl=is_ar())
            _chart(fig, "ek_curve", "ekc")
            lo, hi = r["turning point CI (delta)"]
            fl, fh = r["turning point CI (Fieller)"]
            lm = r["Lind-Mehlum slopes"]
            insight(tr(f"Shape: {r['shape']}. Turning point ≈ {r['turning point']:,.0f} (delta 95% CI {lo:,.0f}–{hi:,.0f}; Fieller {fl:,.0f}–{fh:,.0f}). "
                       f"{100 * r['share beyond turning point']:.0f}% of observations lie beyond it. Slopes at the data ends: {lm['slope at min']:.3f} / {lm['slope at max']:.3f}"
                       + (" — both signs present, consistent with a genuine inverted U." if lm["slope at min"] > 0 > lm["slope at max"] else " — the turn is not supported inside the data range."),
                       f"الشكل: {_t(('inverted U (EKC)', 'U مقلوب (منحنى كوزنتس)')) if r['shape'].startswith('inverted') else r['shape']}. نقطة التحول ≈ {r['turning point']:,.0f} (فترة دلتا 95%: {lo:,.0f}–{hi:,.0f}؛ فيلر {fl:,.0f}–{fh:,.0f}). "
                       f"تقع {100 * r['share beyond turning point']:.0f}% من المشاهدات بعدها. الميل عند طرفي البيانات: {lm['slope at min']:.3f} / {lm['slope at max']:.3f}"
                       + (" — إشارتان متعاكستان تدعمان شكل U المقلوب فعلاً." if lm["slope at min"] > 0 > lm["slope at max"] else " — نقطة التحول غير مدعومة داخل مدى البيانات.")))
            with st.expander(tr("🎯 Years to reach the turning point", "🎯 السنوات اللازمة لبلوغ نقطة التحول"), expanded=False):
                gg = st.slider(tr("Income growth (%/yr)", "نمو الدخل (%/سنة)"), 0.5, 8.0, 3.0, 0.5, key="ek_g") / 100
                last = df[df[t] == df[t].max()][[u, Y]].copy()
                last["years to turning point"] = np.where(last[Y] >= r["turning point"], 0, np.log(r["turning point"] / last[Y]) / np.log(1 + gg))
                _table(last.sort_values("years to turning point", ascending=False).head(15), 1)
        _finish("ekc", "em_ekc", {"EKC": r["table"], **({"Curve": r["curve"]} if "curve" in r else {})}, tr("Environmental Kuznets curve", "منحنى كوزنتس البيئي"), [r["table"]],
                {"y": E, "x": [Y], "unit": u, "time": t})
        return
    a, b, c = st.columns(3)
    E = a.selectbox(tr("CO2", "CO2"), _prefer(num, ("co2",)), key="tp_e")
    G = b.selectbox(tr("GDP (total or per capita)", "الناتج (إجمالي أو للفرد)"), _prefer([x for x in num if x != E], ("gdp",)), key="tp_g")
    yrs = sorted(df[t].unique())
    p0, p1 = c.select_slider(tr("Period", "الفترة"), yrs, (yrs[0], yrs[-1]), key="tp_per")
    mid = yrs[len(yrs) // 2]
    T_ = cl.tapio(df, u, t, E, G, [(p0, mid), (mid, p1)] if p0 < mid < p1 else [(p0, p1)])
    cnt = T_.groupby(["Period", "State"]).size().unstack(fill_value=0)
    if is_ar():
        cnt.columns = [dict(cl.TAPIO).get(c_, c_) for c_ in cnt.columns]
    st.dataframe(cnt, width="stretch")
    fig = go.Figure()
    for j, (per, s) in enumerate(T_.groupby("Period")):
        fig.add_trace(go.Scatter(x=s["%Δ GDP"], y=s["%Δ emissions"], mode="markers", name=per, marker=dict(size=9, color=PAL[j]), text=s.Unit))
    mx = np.nanmax(np.abs(T_[["%Δ GDP", "%Δ emissions"]].to_numpy())) * 1.05
    for el_, ds_ in ((0.8, "dot"), (1.2, "dash")):
        fig.add_trace(go.Scatter(x=[-mx, mx], y=[-el_ * mx, el_ * mx], line=dict(color="#999", dash=ds_, width=1), name=f"e = {el_}"))
    fig.add_hline(y=0, line=dict(color="#666", width=1))
    fig.add_vline(x=0, line=dict(color="#666", width=1))
    ch.style_academic(fig, height=420, x_title="%Δ GDP", y_title="%Δ CO2", legend="bottom", rtl=is_ar())
    _chart(fig, "tp_sc", "tapio_decoupling")
    disp = T_.drop(columns=["code", "State_ar"] if not is_ar() else ["code", "State"])
    if is_ar():
        disp = disp.rename(columns={"State_ar": "State"})
    _table(disp, 2)
    pop = next((x for x in num if "pop" in x), None)
    en = next((x for x in num if "energy" in x), None)
    sheets = {"Tapio": T_}
    if pop and en:
        d = df.copy()
        Gt = G
        if "pc" in G or "per" in G:
            d["_gdp_total"] = d[G] * d[pop]
            Gt = "_gdp_total"
        L = cl.lmdi_kaya(d, u, t, E, pop, Gt, en, p0, p1)
        A = L["aggregate"]
        A = A[A.Factor != "Residual"]
        eff = A[A.Factor != "Total ΔCO2"]
        fig = go.Figure(go.Waterfall(x=[_disp(pd.DataFrame({"c": [f]})).iloc[0, 0] for f in eff.Factor] + [_disp(pd.DataFrame({"c": ["Total ΔCO2"]})).iloc[0, 0]],
                                     measure=["relative"] * len(eff) + ["total"], y=list(eff.Effect) + [A[A.Factor == "Total ΔCO2"].Effect.iloc[0]],
                                     decreasing=dict(marker=dict(color="#27ae60")), increasing=dict(marker=dict(color="#c0392b")), totals=dict(marker=dict(color=PAL[0]))))
        ch.style_academic(fig, height=380, y_title="Δ CO2", legend="none", rtl=is_ar())
        _chart(fig, "lmdi_wf", "lmdi_kaya")
        _table(L["aggregate"], 2)
        mlt = pd.DataFrame([L["multiplicative"]])
        st.dataframe(_disp(mlt).round(4), width="stretch", hide_index=True)
        big = eff.iloc[np.argmax(eff.Effect.abs())]
        neg = eff[eff.Effect < 0]
        insight(tr(f"{p0}–{p1}: emissions changed by {A[A.Factor == 'Total ΔCO2'].Effect.iloc[0]:,.0f}; the largest driver is {big.Factor} ({big.Effect:+,.0f})"
                   + (f", partly offset by {', '.join(neg.Factor)}." if len(neg) else "."),
                   f"{p0}–{p1}: تغيرت الانبعاثات بمقدار {A[A.Factor == 'Total ΔCO2'].Effect.iloc[0]:,.0f}؛ وأكبر محرك هو {_disp(pd.DataFrame({'c': [big.Factor]})).iloc[0, 0]} ({big.Effect:+,.0f})"
                   + (f"، ويعوّضه جزئياً {'، '.join(_disp(pd.DataFrame({'c': list(neg.Factor)})).c)}." if len(neg) else ".")))
        sheets.update({"LMDI aggregate": L["aggregate"], "LMDI by unit": L["by unit"]})
    _finish("tapio", "em_tap", sheets, tr("Decoupling and decomposition", "فك الارتباط والتفكيك"), [cnt.reset_index()] + ([L["aggregate"]] if pop and en else []))


def _scc():
    a, b, c = st.columns(3)
    r = a.slider(tr("Discount rate (%)", "معدل الخصم (%)"), 1.0, 7.0, 3.0, 0.25, key="scc_r") / 100
    dmg = b.selectbox(tr("Damage coefficient a (D = aT²)", "معامل الضرر a (D = aT²)"), [0.00236, 0.005, 0.007438], key="scc_a",
                      format_func=lambda v: {0.00236: "0.00236 (DICE-2016R)", 0.005: "0.005", 0.007438: "0.007438 (Howard-Sterner)"}[v])
    tcre = c.slider(tr("TCRE (°C per 1000 GtCO2)", "TCRE (°م لكل 1000 جيجا طن CO2)"), 0.27, 0.63, 0.45, 0.01, key="scc_t")
    a, b, c = st.columns(3)
    g = a.slider(tr("World GDP growth (%/yr)", "نمو الناتج العالمي (%/سنة)"), 0.5, 4.0, 2.0, 0.25, key="scc_g") / 100
    wr = b.slider(tr("Warming rate (°C/decade)", "معدل الاحترار (°م/عقد)"), 0.1, 0.5, 0.25, 0.05, key="scc_w") / 10
    gdp0 = c.number_input(tr("World GDP today (trillion USD)", "الناتج العالمي حالياً (تريليون دولار)"), 50.0, 300.0, 105.0, 5.0, key="scc_y") * 1e12
    out = cl.scc_simple(gdp0, g, 1.2, wr, dmg, tcre, r)
    st.metric(tr("Social cost of carbon (USD per tCO2, illustrative)", "تكلفة الكربون الاجتماعية (دولار/طن CO2، توضيحية)"), f"{out['SCC (USD/tCO2)']:.1f}")
    sens = pd.DataFrame([{"Discount rate (%)": 100 * rr, "SCC (USD/tCO2)": cl.scc_simple(gdp0, g, 1.2, wr, dmg, tcre, rr)["SCC (USD/tCO2)"]} for rr in (0.015, 0.02, 0.025, 0.03, 0.04, 0.05)])
    fig = go.Figure(go.Bar(x=sens["Discount rate (%)"], y=sens["SCC (USD/tCO2)"], marker_color=PAL[0]))
    ch.style_academic(fig, height=320, x_title=tr("Discount rate (%)", "معدل الخصم (%)"), y_title="USD / tCO2", legend="none", rtl=is_ar())
    _chart(fig, "scc_s", "scc_discount")
    dc = cl.damage_curve(dmg)
    fig = go.Figure(go.Scatter(x=dc["Warming (°C)"], y=dc["Quadratic damages (% GDP)"], line=dict(color=PAL[2], width=3)))
    ch.style_academic(fig, height=300, x_title=tr("Warming (°C)", "الاحترار (°م)"), y_title=tr("Damages (% GDP)", "الأضرار (% من الناتج)"), legend="none", rtl=is_ar())
    _chart(fig, "scc_dc", "damage_function")
    insight(tr(f"At a {100 * r:.2f}% discount rate the marginal tonne of CO2 causes about USD {out['SCC (USD/tCO2)']:.0f} of discounted damages; halving the discount rate "
               "roughly doubles to triples the SCC — the discount rate is the key ethical choice. This is a transparent teaching calculator, not an integrated assessment model.",
               f"عند معدل خصم {100 * r:.2f}% يسبب الطن الحدي من CO2 أضراراً مخصومة بنحو {out['SCC (USD/tCO2)']:.0f} دولاراً؛ وتنصيف معدل الخصم يضاعف التكلفة مرتين إلى ثلاث تقريباً — "
               "فمعدل الخصم هو الاختيار الأخلاقي الحاسم. هذه حاسبة تعليمية شفافة وليست نموذج تقييم متكاملاً."))
    _finish("bhm", "em_scc", {"SCC sensitivity": sens, "Path": out["path"], "Damage curve": dc}, tr("Social cost of carbon", "تكلفة الكربون الاجتماعية"), [sens])
