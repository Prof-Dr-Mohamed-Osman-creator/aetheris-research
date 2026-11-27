# ==========================================
# Aetheris Research - Econometrics Lab (UI) — v1.4
# ------------------------------------------
# Catalogue §1.1 classical & cross-sectional regression + §1.3 full diagnostics
# + journal tables, delivered with the 10-element standard of docs/MODEL_CATALOG.md:
# guided/expert modes, pre-checks, journal tables, diagnostics, elasticities,
# model charts, bilingual interpretation, policy brief & what-if, methodology +
# references, and Excel / Word / LaTeX / Python / R / Stata export.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from analytics import charts as ch
from analytics import descriptive as ds
from analytics import diagnostics as dx
from analytics import reg_charts as rc
from analytics import reg_report as rp
from analytics import regression as rg
from core.memory import get_memory, has_data, active_name, set_memory
from core.utils import tr, is_ar, insight, csv_download, excel_download

MODELS_KEY = "eco_models"        # {label: {"r": RegResult, "diag": dict, "dataset": str}}
CURRENT_KEY = "eco_current"

FLAG_AR = {"leverage": "رفع مرتفع", "outlier": "قيمة شاذة", "Cook": "كوك", "DFFITS": "DFFITS", "DFBETAS": "DFBETAS"}
CONCERN_AR = {"none": "لا يوجد", "moderate": "متوسط", "serious": "خطير"}
COL_AR = {"Variable": "المتغير", "Coef.": "المعامل", "Std. Err.": "الخطأ المعياري", "p-value": "القيمة الاحتمالية",
          "CI low": "حد الثقة الأدنى", "CI high": "حد الثقة الأعلى", "Sig.": "المعنوية", "Std. beta": "المعامل المعياري",
          "Test": "الاختبار", "Statistic": "الإحصاءة", "df": "درجات الحرية", "Verdict": "الحكم", "Note": "ملاحظة",
          "Type": "النوع", "At value": "عند القيمة", "Marginal effect dY/dX": "الأثر الحدي dY/dX", "ME s.e.": "خ.م للأثر",
          "ME p-value": "p للأثر", "Elasticity": "المرونة", "Elasticity s.e.": "خ.م للمرونة",
          "El. CI low": "حد أدنى للمرونة", "El. CI high": "حد أعلى للمرونة", "El. p-value": "p للمرونة",
          "VIF": "VIF", "Tolerance (1/VIF)": "التسامح (1/VIF)", "Concern": "درجة القلق", "Dimension": "البُعد",
          "Eigenvalue": "الجذر المميز", "Condition index": "دليل الحالة", "obs": "المشاهدة", "Leverage": "الرفع",
          "Std. residual": "البواقي المعيارية", "Stud. residual": "البواقي المعيارية الخارجية", "Cook's D": "مسافة كوك",
          "Flags": "التنبيهات", "Shape": "الشكل", "Turning point": "نقطة التحول", "Within data range": "داخل مدى البيانات",
          "Term": "الحد", "Partial SS": "مجموع المربعات الجزئي", "F": "F", "Partial η²": "η² الجزئي",
          "Level": "الفئة", "Raw mean": "المتوسط الخام", "Adjusted mean": "المتوسط المعدل",
          "Size": "الحجم", "Variables": "المتغيرات", "Best by": "الأفضل وفق", "Block": "الكتلة", "Added": "المضاف",
          "Form": "الصيغة", "Equation": "المعادلة", "Action": "الإجراء", "Path": "المسار",
          "Moderator value": "قيمة المعدِّل", "Slope": "الميل"}


def _ms(label, options, **kw):
    """Multiselect with a translated placeholder."""
    return st.multiselect(label, options, placeholder=tr("Choose…", "اختر…"), **kw)


def _lang():
    return "ar" if is_ar() else "en"


def _ar_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COL_AR) if is_ar() else df


def _chart(fig, key, name="figure"):
    st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, name), key=key)


def _models() -> dict:
    return st.session_state.setdefault(MODELS_KEY, {})


def _explain(guided: bool, en: str, ar: str):
    if guided:
        st.caption("💡 " + tr(en, ar))


# ==========================================================
# Entry
# ==========================================================
def render_econometrics():
    if not has_data():
        st.info(tr("Load data first (Data Hub), or try the built-in demo datasets:",
                   "حمّل بيانات أولاً (مركز البيانات)، أو جرّب البيانات التجريبية المدمجة:"))
        _demo_buttons()
        return
    df = get_memory()
    st.caption(tr(f"Active dataset: **{active_name()}** — {df.shape[0]} rows × {df.shape[1]} columns",
                  f"البيانات النشطة: **{active_name()}** — {df.shape[0]} صف × {df.shape[1]} عمود"))
    with st.expander(tr("🧪 Demo datasets for training", "🧪 بيانات تجريبية للتدريب"), expanded=False):
        _demo_buttons()
    labels = [tr("🛠️ Regression builder", "🛠️ بناء نموذج الانحدار"),
              tr("📑 Journal table", "📑 جدول بنمط المجلات"),
              tr("🔎 Model selection & form", "🔎 اختيار النموذج والصورة الدالية"),
              tr("🔀 Moderation & mediation", "🔀 التأثير المعدِّل والوسيط"),
              tr("🧮 Hypothesis tests & RLS", "🧮 اختبار الفروض والمربعات المقيدة"),
              tr("〰️ Nonlinear LS", "〰️ المربعات الصغرى غير الخطية")]
    tabs = st.tabs(labels)
    with tabs[0]:
        _builder_tab(df)
    with tabs[1]:
        _journal_tab()
    with tabs[2]:
        _selection_tab(df)
    with tabs[3]:
        _moderation_tab(df)
    with tabs[4]:
        _tests_tab(df)
    with tabs[5]:
        _nls_tab(df)


def _demo_buttons():
    from data.samples import climate_crop_demo, farm_survey_demo
    c1, c2 = st.columns(2)
    with c1:
        if st.button(tr("🌾 Farm survey (cross-section, 320 farms)", "🌾 مسح المزارع (بيانات مقطعية، 320 مزرعة)"),
                     key="eco_demo_farm"):
            set_memory(farm_survey_demo(), source="demo (synthetic)", name="Demo – farm survey (synthetic)")
            st.rerun()
    with c2:
        if st.button(tr("🌡️ Climate & crops (panel, 768 obs.)", "🌡️ المناخ والمحاصيل (بانل، 768 مشاهدة)"),
                     key="eco_demo_climate"):
            set_memory(climate_crop_demo(), source="demo (synthetic)", name="Demo – climate & crops (synthetic)")
            st.rerun()
    st.caption(tr("⚠️ Simulated data for training only — never cite as real statistics.",
                  "⚠️ بيانات مُحاكاة للتدريب فقط — لا يجوز الاستشهاد بها كإحصاءات حقيقية."))


# ==========================================================
# 1. Regression builder
# ==========================================================
Y_HINTS = ("yield", "output", "production", "gdp", "income", "revenue", "sales", "consumption", "demand", "export",
           "import", "profit", "wage", "price", "growth", "y")


def _id_like(df, c) -> bool:
    s = pd.to_numeric(df[c], errors="coerce")
    name = str(c).lower()
    return name in ("id", "index") or name.endswith("_id") or name.startswith("id_") or (
        s.notna().all() and s.is_unique and s.is_monotonic_increasing and np.allclose(np.diff(s.to_numpy(float)), 1)
        and len(s) > 20)


def _model_vars(df):
    num = ds.analysis_columns(df) or ds.numeric_columns(df)
    num = [c for c in num if not _id_like(df, c)] or num
    ranked = sorted(num, key=lambda c: next((i for i, h in enumerate(Y_HINTS) if h in str(c).lower()), len(Y_HINTS)))
    return num, ranked[0] if ranked else None


def _spec_widgets(df, key, guided, allow_poly=True, allow_inter=True):
    num, y_default = _model_vars(df)
    cat = ds.categorical_columns(df)
    if guided:
        st.markdown("#### " + tr("Step 1 — Variables", "الخطوة 1 — المتغيرات"))
        _explain(guided, "Choose the dependent variable (what you want to explain) and the explanatory variables. "
                 "Qualitative factors (region, crop, gender…) enter as dummy variables against a reference category.",
                 "اختر المتغير التابع (ما تريد تفسيره) والمتغيرات المفسِّرة. وتدخل العوامل النوعية (الإقليم، المحصول، "
                 "النوع…) في صورة متغيرات صورية مقارنةً بفئة مرجعية.")
    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        y = st.selectbox(tr("Dependent variable Y", "المتغير التابع Y"), num, index=num.index(y_default) if y_default in num else 0,
                         key=f"{key}_y")
    with c2:
        tcols = set(ds.time_columns(df))
        x = _ms(tr("Numeric regressors X", "المتغيرات المستقلة الرقمية X"), [c for c in num if c != y],
                           default=[c for c in num if c != y and c not in tcols][:2], key=f"{key}_x")
    with c3:
        cats = _ms(tr("Categorical factors (dummies)", "العوامل النوعية (متغيرات صورية)"),
                              [c for c in cat if c != y], key=f"{key}_cat")
    reference = {}
    if cats:
        with st.expander(tr("Reference categories", "الفئات المرجعية"), expanded=False):
            cols = st.columns(min(3, len(cats)))
            for i, c in enumerate(cats):
                lv = sorted(df[c].dropna().astype(str).unique().tolist())
                with cols[i % len(cols)]:
                    reference[c] = st.selectbox(c, lv, key=f"{key}_ref_{c}")

    if guided:
        st.markdown("#### " + tr("Step 2 — Functional form", "الخطوة 2 — الصورة الدالية"))
        _explain(guided, "Log-log gives constant elasticities (coefficients = % effects); log-lin gives growth / semi-"
                 "elasticities; lin-log suits diminishing returns; reciprocal suits asymptotic curves. Use the "
                 "'Model selection & form' tab (Box-Cox) to let the data choose.",
                 "الصيغة اللوغاريتمية المزدوجة تعطي مرونات ثابتة (المعاملات آثار نسبية)؛ ولوغاريتم التابع يعطي معدلات النمو "
                 "وأشباه المرونات؛ ولوغاريتم المستقل يناسب تناقص العائد؛ والمقلوب يناسب المنحنيات التقاربية. واستخدم تبويب "
                 "«اختيار النموذج والصورة الدالية» (بوكس-كوكس) لتختار البيانات الصيغة المناسبة.")
    c1, c2 = st.columns([2, 3])
    with c1:
        form = st.selectbox(tr("Functional form", "الصورة الدالية"), list(rg.FORMS), key=f"{key}_form",
                            format_func=lambda f: f"{rg.FORMS[f][1 if is_ar() else 0]}   [{rg.FORMS[f][2]}]")
    nolog = []
    with c2:
        if form in ("loglog", "linlog", "reciprocal") and x:
            nolog = _ms(tr("Keep in levels (not transformed): trend, shares, scores…",
                                      "تبقى بقيمها دون تحويل (الاتجاه الزمني، النسب، الدرجات…)"), x, key=f"{key}_nolog")
    poly, inter, center = {}, [], False
    if allow_poly or allow_inter:
        c1, c2, c3 = st.columns([2, 3, 1])
        if allow_poly:
            with c1:
                psel = _ms(tr("Add squared terms (U-shape / turning point)", "إضافة حدود تربيعية (شكل U / نقطة التحول)"),
                                      x, key=f"{key}_poly")
                deg = 2
                if psel and not guided:
                    deg = st.radio(tr("Degree", "الدرجة"), [2, 3], horizontal=True, key=f"{key}_deg")
                poly = {v: deg for v in psel}
        if allow_inter:
            with c2:
                pool = x + cats
                pairs = [f"{a} × {b}" for i, a in enumerate(pool) for b in pool[i + 1:]]
                isel = _ms(tr("Interaction terms (moderation)", "حدود التفاعل (التأثير المعدِّل)"), pairs,
                                      key=f"{key}_inter")
                inter = [tuple(s.split(" × ")) for s in isel]
            with c3:
                center = st.checkbox(tr("Mean-centre", "طرح المتوسط"), value=bool(poly or inter), key=f"{key}_center",
                                     help=tr("Centring the variables in squared / interaction terms reduces "
                                             "collinearity and makes main effects interpretable at the mean.",
                                             "طرح المتوسط من المتغيرات الداخلة في الحدود التربيعية والتفاعلية يقلل "
                                             "الازدواج الخطي ويجعل الآثار الرئيسية مفسَّرة عند المتوسط."))
    const = True
    if not guided:
        const = st.checkbox(tr("Include intercept", "تضمين الحد الثابت"), value=True, key=f"{key}_const")
    if not x and not cats:
        return None
    return rg.ModelSpec(y=y, x=x, categorical=cats, form=form, poly=poly, interactions=inter, nolog=nolog,
                        center=center, const=const, reference=reference)


def _estimator_widgets(df, key, guided, n_hint):
    if guided:
        st.markdown("#### " + tr("Step 3 — Estimator & standard errors", "الخطوة 3 — طريقة التقدير والأخطاء المعيارية"))
        _explain(guided, "OLS is the default. If you expect the error variance to differ across observations "
                 "(typical in cross-sections of farms, firms or households) choose robust HC3/HC1 errors; for time "
                 "series choose Newey-West; for grouped data choose cluster-robust errors.",
                 "المربعات الصغرى العادية هي الأساس. وإذا توقعت اختلاف تباين الخطأ بين المشاهدات (وهو الشائع في بيانات "
                 "المزارع والمنشآت والأسر) فاختر الأخطاء المتينة HC3/HC1؛ وللسلاسل الزمنية اختر نيوي-ويست؛ وللبيانات "
                 "المجمعة في مجموعات اختر الأخطاء العنقودية.")
    methods = ["ols", "wls", "fgls", "glsar", "huber", "bisquare"]
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox(tr("Estimator", "طريقة التقدير"), methods, key=f"{key}_method",
                              format_func=lambda m: rg.METHODS[m][1 if is_ar() else 0])
    weights = None
    cov_opts = {}
    num = ds.numeric_columns(df)
    with c2:
        if method in ("huber", "bisquare"):
            st.caption(tr("Robust M-estimators report their own (asymptotic) standard errors.",
                          "مقدّرات M المتينة تعرض أخطاءها المعيارية التقاربية الخاصة."))
            cov = "nonrobust"
        else:
            covs = ["nonrobust", "HC1", "HC3", "HC0", "HC2", "HAC", "cluster"]
            if method in ("ols", "wls"):
                covs.append("bootstrap")
            default = covs.index("HC3") if n_hint < 250 else covs.index("HC1")
            cov = st.selectbox(tr("Standard errors", "الأخطاء المعيارية"), covs, index=default if guided else 0,
                               key=f"{key}_cov", format_func=lambda c: rg.COV_TYPES[c][1 if is_ar() else 0])
    with c3:
        if method == "wls":
            weights = st.selectbox(tr("Weights variable (∝ 1/variance)", "متغير الأوزان (يتناسب مع 1/التباين)"), num,
                                   key=f"{key}_w")
        if cov == "HAC":
            cov_opts["maxlags"] = st.number_input(tr("Newey-West lags", "إبطاءات نيوي-ويست"), 0, 48,
                                                  rg.newey_west_lags(n_hint), key=f"{key}_lags")
        if cov == "cluster":
            cat = ds.categorical_columns(df, max_unique=10_000)
            cov_opts["groups"] = st.selectbox(tr("Cluster variable", "متغير العناقيد"), cat or list(df.columns),
                                              key=f"{key}_clu")
        if cov == "bootstrap":
            cov_opts["reps"] = st.select_slider(tr("Bootstrap replications", "عدد تكرارات البوتستراب"),
                                                [199, 499, 999, 1999], 999, key=f"{key}_reps")
    return method, cov, cov_opts, weights


def _diag_widgets(df, key, guided, spec):
    time_cols = ds.time_columns(df)
    if guided:
        st.markdown("#### " + tr("Step 4 — Diagnostics settings", "الخطوة 4 — إعدادات التشخيص"))
        _explain(guided, "Autocorrelation and stability tests (Durbin-Watson, Breusch-Godfrey, CUSUM, Chow) only make "
                 "sense when the rows have a natural (time) order. For a survey cross-section, untick the box.",
                 "اختبارات الارتباط الذاتي والاستقرار (دربن-واتسون، بروش-جودفري، CUSUM، شاو) لا معنى لها إلا إذا كان "
                 "للصفوف ترتيب طبيعي (زمني). وفي البيانات المقطعية (المسوح) ألغِ تحديد الخيار.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        ordered = st.checkbox(tr("Rows are time-ordered", "الصفوف مرتبة زمنياً"), value=bool(time_cols),
                              key=f"{key}_ordered")
    with c2:
        bg = st.number_input(tr("Breusch-Godfrey lags", "إبطاءات بروش-جودفري"), 1, 24, 2, key=f"{key}_bg")
    with c3:
        chow = st.number_input(tr("Chow break at row (0 = none)", "انكسار شاو عند الصف (0 = بدون)"), 0, len(df), 0,
                               key=f"{key}_chow")
    with c4:
        lagged = st.selectbox(tr("Lagged Y regressor (Durbin h)", "المتغير التابع المُبطأ (دربن h)"),
                              ["—"] + list(spec.x if spec else []), key=f"{key}_lagy")
    return {"time_ordered": ordered, "bg_lags": int(bg), "chow_at": int(chow) or None,
            "lagged_y": None if lagged == "—" else lagged}


def _builder_tab(df):
    mode = st.radio(tr("Mode", "الوضع"), ["guided", "expert"], horizontal=True, key="eco_mode",
                    format_func=lambda m: tr("🧭 Guided (step by step)", "🧭 موجَّه (خطوة بخطوة)") if m == "guided"
                    else tr("⚡ Expert (compact)", "⚡ خبير (مختصر)"))
    guided = mode == "guided"
    spec = _spec_widgets(df, "eco", guided)
    if spec is None:
        st.info(tr("Select at least one regressor.", "اختر متغيراً مستقلاً واحداً على الأقل."))
        return
    method, cov, cov_opts, weights = _estimator_widgets(df, "eco", guided, len(df))
    dopts = _diag_widgets(df, "eco", guided, spec)
    if dopts["lagged_y"]:
        dopts["lagged_y"] = rg._tname(dopts["lagged_y"], "level" if dopts["lagged_y"] in spec.nolog else rg.x_transform(spec.form))
    alpha = st.select_slider(tr("Significance level α", "مستوى المعنوية α"), [0.01, 0.05, 0.10], 0.05, key="eco_alpha")

    try:
        design = rg.build_design(df, spec)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return
    with st.expander(tr("🩺 Pre-estimation checks", "🩺 فحص الافتراضات قبل التقدير"), expanded=guided):
        for chk in dx.pre_checks(df, spec, design):
            (st.success if chk["ok"] else st.warning)(chk["ar"] if is_ar() else chk["en"])
        st.caption(tr(f"Design matrix: {design.X.shape[0]} × {design.X.shape[1]} — columns: ",
                      f"مصفوفة التصميم: {design.X.shape[0]} × {design.X.shape[1]} — الأعمدة: ")
                   + ", ".join(design.X.columns))

    if st.button(tr("▶️ Estimate model", "▶️ تقدير النموذج"), type="primary", key="eco_run"):
        try:
            with st.spinner(tr("Estimating and running diagnostics…", "جارٍ التقدير وإجراء الاختبارات التشخيصية…")):
                r = rg.fit(df, spec, method=method, cov_type=cov, cov_opts=cov_opts, weights=weights)
                diag = dx.run_all(r, bg_lags=dopts["bg_lags"], chow_at=dopts["chow_at"], lagged_y=dopts["lagged_y"],
                                  time_ordered=dopts["time_ordered"])
            st.session_state[CURRENT_KEY] = {"r": r, "diag": diag, "alpha": alpha, "dataset": active_name()}
        except Exception as e:  # noqa: BLE001
            st.error(tr(f"Estimation failed: {e}", f"تعذّر التقدير: {e}"))
            return
    cur = st.session_state.get(CURRENT_KEY)
    if cur and cur.get("dataset") == active_name():
        st.markdown("---")
        show_model(df, cur["r"], cur["diag"], cur.get("alpha", 0.05), key="cur")


# ==========================================================
# Model output (used by the builder and saved models)
# ==========================================================
def _fmt_p(p):
    return "" if p is None or pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.4f}")


def _coef_display(r: rg.RegResult) -> pd.DataFrame:
    t = r.coef_table().round(4)
    t["p-value"] = r.pvalues.map(_fmt_p).values
    if "Std. beta" in t:
        t["Std. beta"] = t["Std. beta"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    return _ar_cols(t)


def show_model(df, r: rg.RegResult, diag: dict, alpha: float, key: str):
    lang = _lang()
    st.subheader(tr(f"Results — {r.method_label()}", f"النتائج — {r.method_label('ar')}"))
    st.caption(f"{r.y_label} · N = {r.nobs} · " + tr("SE: ", "الأخطاء المعيارية: ") + r.cov_label(lang))
    sub = st.tabs([tr("📊 Estimates", "📊 التقديرات"), tr("🩺 Diagnostics", "🩺 التشخيص"),
                   tr("📐 Elasticities & effects", "📐 المرونات والآثار"), tr("📈 Charts", "📈 الرسوم"),
                   tr("🏛️ Policy brief & what-if", "🏛️ ملخص صانع القرار وماذا لو"),
                   tr("📝 Methodology & export", "📝 المنهجية والتصدير")])
    with sub[0]:
        _estimates(df, r, diag, alpha, key)
    with sub[1]:
        _diagnostics(r, diag, key)
    with sub[2]:
        _effects(df, r, key)
    with sub[3]:
        _charts(df, r, diag, key)
    with sub[4]:
        _policy(df, r, diag, alpha, key)
    with sub[5]:
        _export(df, r, diag, key)


def _estimates(df, r, diag, alpha, key):
    st.dataframe(_coef_display(r), width="stretch", hide_index=True)
    fs = r.fit_stats()
    m = st.columns(6)
    for col, (lab, v) in zip(m, [("R²", fs["R²"]), (tr("Adj. R²", "R² المعدل"), fs["Adj. R²"]),
                                 ("F", fs["F"]), ("Prob(F)", fs["Prob(F)"]), ("AIC", fs["AIC"]), ("BIC", fs["BIC"])]):
        col.metric(lab, "—" if v is None or not np.isfinite(v) else (f"{v:.4f}" if abs(v) < 100 else f"{v:,.1f}"))
    with st.expander(tr("All fit statistics", "جميع إحصاءات جودة التوفيق"), expanded=False):
        st.dataframe(pd.DataFrame({"": list(fs), tr("Value", "القيمة"): [
            "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else v for v in fs.values()]}).astype(str),
                     hide_index=True, width="stretch")
    insight(rp.interpret(r, diag, alpha, _lang()))
    if r.design.spec.categorical:
        with st.expander(tr("ANCOVA / term tests & adjusted means", "تحليل التغاير: اختبار الحدود والمتوسطات المعدلة"),
                         expanded=False):
            st.dataframe(_ar_cols(rg.term_tests(r).round(4)), width="stretch", hide_index=True)
            st.caption(tr("Joint Wald/F test of each term (all dummies of a factor together), each added last; "
                          "partial η² = share of remaining variance explained by the term.",
                          "اختبار والد/F المشترك لكل حد (جميع المتغيرات الصورية للعامل معاً) بعد إدخال بقية الحدود؛ "
                          "وη² الجزئي هو نسبة التباين المتبقي الذي يفسّره الحد."))
            fac = st.selectbox(tr("Adjusted means for", "المتوسطات المعدلة لـ"), r.design.spec.categorical, key=f"{key}_am")
            st.dataframe(_ar_cols(rg.adjusted_means(df, r, fac).round(4)), width="stretch", hide_index=True)
    c1, c2 = st.columns([3, 1])
    with c1:
        default = f"({len(_models()) + 1}) {r.y_label} — {r.method.upper()}"
        label = st.text_input(tr("Model name for the journal table", "اسم النموذج في جدول المجلات"), default,
                              key=f"{key}_save_name")
    with c2:
        st.write("")
        if st.button(tr("💾 Save model", "💾 حفظ النموذج"), key=f"{key}_save"):
            models = _models()
            if len(models) >= 10:
                st.warning(tr("Up to 10 saved models — remove one in the Journal table tab.",
                              "الحد الأقصى 10 نماذج محفوظة — احذف أحدها من تبويب جدول المجلات."))
            else:
                r.name = label
                models[label] = {"r": r, "diag": diag, "dataset": active_name()}
                st.success(tr("Saved. Open the 'Journal table' tab to compare models side by side.",
                              "تم الحفظ. افتح تبويب «جدول بنمط المجلات» لمقارنة النماذج جنباً إلى جنب."))


def _status_badge(s):
    return {"ok": "🟢", "mixed": "🟡", "fail": "🔴"}[s]


def _diagnostics(r, diag, key):
    lang = _lang()
    vs = dx.verdicts(diag, lang)
    cols = st.columns(max(1, len(vs)))
    for c, v in zip(cols, vs):
        c.markdown(f"**{_status_badge(v['status'])} {v['label']}**")
    for v in vs:
        if v["status"] != "ok":
            st.warning(f"{_status_badge(v['status'])} **{v['label']}** — {v['remedy']}")
    tab = diag["table"].copy()
    tab["Verdict"] = tab["ok"].map({True: tr("✅ holds", "✅ مستوفى"), False: tr("❌ violated", "❌ غير مستوفى")}).fillna("ℹ️")
    tab["Group"] = tab["group"].map(lambda g: dx.GROUPS[g][1 if is_ar() else 0])
    show = tab[["Group", "Test_ar" if is_ar() else "Test", "Statistic", "df", "p-value", "Verdict",
                "note_ar" if is_ar() else "note"]].copy()
    show.columns = [tr("Assumption", "الافتراض"), tr("Test", "الاختبار"), tr("Statistic", "الإحصاءة"),
                    tr("df", "درجات الحرية"), tr("p-value", "القيمة الاحتمالية"), tr("Verdict", "الحكم"), tr("Note", "ملاحظة")]
    show[show.columns[3]] = show[show.columns[3]].map(lambda v: "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    show[show.columns[4]] = show[show.columns[4]].map(_fmt_p)
    show[show.columns[2]] = show[show.columns[2]].map(lambda v: "" if v is None or pd.isna(v) else f"{v:.4f}")
    st.dataframe(show.round(4), width="stretch", hide_index=True, height=min(760, 38 * len(show) + 40))
    st.caption(tr("Tests use the whitened residuals of the estimated model (weighted for WLS/FGLS, quasi-differenced "
                  "for GLS-AR(1)). Durbin-Watson p-values are exact (Imhof algorithm) for the actual regressor matrix.",
                  "تُجرى الاختبارات على البواقي المحوَّلة للنموذج المقدَّر (الموزونة في WLS/FGLS، والمفروقة جزئياً في GLS-AR(1)). "
                  "والقيم الاحتمالية لدربن-واتسون دقيقة (خوارزمية إمهوف) لمصفوفة المتغيرات الفعلية."))
    d = diag["detail"]
    c1, c2 = st.columns(2)
    with c1:
        if "vif" in d and not d["vif"].empty:
            st.markdown("**" + tr("Variance inflation factors", "معاملات تضخم التباين") + "**")
            vt = d["vif"].round(3)
            if is_ar():
                vt["Concern"] = vt["Concern"].map(CONCERN_AR)
            st.dataframe(_ar_cols(vt), width="stretch", hide_index=True)
    with c2:
        if "cond" in d:
            st.markdown("**" + tr("Belsley condition indices & variance proportions",
                                  "أدلة الحالة لبيلسلي ونسب تجزئة التباين") + "**")
            st.dataframe(_ar_cols(d["cond"].round(3)), width="stretch", hide_index=True)
    if "influence" in d:
        inf = d["influence"]
        flagged = inf[inf["Flags"] != ""].sort_values("Cook's D", ascending=False)
        cut = inf.attrs.get("cutoffs", {})
        st.markdown("**" + tr(f"Influential / unusual observations ({len(flagged)} flagged)",
                              f"المشاهدات المؤثرة أو غير المعتادة ({len(flagged)} مشاهدة)") + "**")
        st.caption(tr(f"Cut-offs: leverage > {cut.get('leverage', 0):.3f}, |studentized| > 3, Cook's D > {cut.get('cook', 0):.4f}, "
                      f"|DFFITS| > {cut.get('dffits', 0):.3f}, |DFBETAS| > {cut.get('dfbetas', 0):.3f}.",
                      f"حدود القطع: الرفع > {cut.get('leverage', 0):.3f}، |البواقي المعيارية الخارجية| > 3، مسافة كوك > {cut.get('cook', 0):.4f}، "
                      f"|DFFITS| > {cut.get('dffits', 0):.3f}، |DFBETAS| > {cut.get('dfbetas', 0):.3f}."))
        base_cols = ["obs", "Leverage", "Std. residual", "Stud. residual", "Cook's D", "DFFITS", "Flags"]
        fl = flagged[base_cols].head(50).round(4)
        if is_ar():
            fl["Flags"] = fl["Flags"].map(lambda f: "، ".join(FLAG_AR.get(x.strip(), x) for x in f.split(",")))
        st.dataframe(_ar_cols(fl), width="stretch", hide_index=True)
        csv_download(inf.round(6), "influence_measures.csv", tr("⬇️ All influence measures (CSV)", "⬇️ جميع مقاييس التأثير (CSV)"),
                     key=f"{key}_infl_dl")


def _effects(df, r, key):
    st.markdown("**" + tr("Marginal effects and elasticities at the representative point",
                          "الآثار الحدية والمرونات عند النقطة الممثِّلة") + "**")
    st.caption(tr("Numeric variables at their mean (geometric mean for logged variables), factors at the reference level. "
                  "Delta-method standard errors use the model's covariance matrix (robust if selected). "
                  "Discrete effects in log-Y models use Kennedy's (1981) correction.",
                  "المتغيرات الرقمية عند متوسطها (المتوسط الهندسي للمتغيرات اللوغاريتمية)، والعوامل عند الفئة المرجعية. "
                  "والأخطاء المعيارية بطريقة دلتا من مصفوفة التغاير للنموذج (المتينة إن اختيرت). والآثار المتقطعة في "
                  "نماذج لوغاريتم Y مصححة بطريقة كينيدي (1981)."))
    try:
        eff = rp.effects_table(df, r)
        show = eff.round(4)
        for c in ("ME p-value", "El. p-value"):
            show[c] = eff[c].map(_fmt_p)
        if is_ar():
            show["Type"] = show["Type"].map({"continuous": "متصل", "discrete": "متقطع (مقارنة بالفئة المرجعية)"})
            if "Note" in show:
                show["Note"] = show["Note"].map(lambda n: "نسبة التغير في Y مقارنة بالفئة المرجعية (تصحيح كينيدي)" if isinstance(n, str) and "Kennedy" in n
                                                else "التغير في Y مقارنة بالفئة المرجعية" if isinstance(n, str) else "")
        st.dataframe(_ar_cols(show), width="stretch", hide_index=True)
        for _, row in eff[eff["Type"] == "continuous"].iterrows():
            e = row["Elasticity"]
            kind = tr("elastic (|e| > 1)", "مرن (|e| > 1)") if abs(e) > 1 else tr("inelastic (|e| < 1)", "غير مرن (|e| < 1)")
            st.markdown(f"- **{row['Variable']}**: " + tr(f"elasticity {e:.3f} — {kind}; a 1% rise changes {r.design.spec.y} by {e:.3f}%.",
                                                          f"المرونة {e:.3f} — {kind}؛ زيادة 1% تغيّر {r.design.spec.y} بنسبة {e:.3f}%."))
    except Exception as ex:  # noqa: BLE001
        st.info(tr(f"Effects not available: {ex}", f"الآثار غير متاحة: {ex}"))
        eff = None
    tp = rp.turning_points(r, df)
    if not tp.empty:
        st.markdown("**" + tr("Turning points of quadratic terms", "نقاط التحول للحدود التربيعية") + "**")
        st.dataframe(_ar_cols(tp.round(4)), width="stretch", hide_index=True)
        for _, row in tp.iterrows():
            st.caption(tr(f"{row['Variable']}: {row['Shape']} at {row['Turning point']:.4g} "
                          f"(95% CI {row['CI low']:.4g} – {row['CI high']:.4g})"
                          + ("" if row["Within data range"] in (True, None) else " — outside the observed range: interpret with caution."),
                          f"{row['Variable']}: {'قيمة عظمى (منحنى U مقلوب)' if 'max' in row['Shape'] else 'قيمة صغرى (منحنى U)'} عند "
                          f"{row['Turning point']:.4g} (فترة الثقة 95%: {row['CI low']:.4g} – {row['CI high']:.4g})"
                          + ("" if row["Within data range"] in (True, None) else " — خارج المدى المشاهد للبيانات: تُفسَّر بحذر.")))


def _charts(df, r, diag, key):
    rtl = is_ar()
    spec = r.design.spec
    st.markdown("**" + tr("Residual diagnostics", "تشخيص البواقي") + "**")
    _chart(rc.residual_panel(r, rtl, [tr("Residuals vs fitted", "البواقي مقابل القيم المقدَّرة"),
                                       tr("Normal Q-Q", "شكل الاحتمال الطبيعي Q-Q"),
                                       tr("Scale-location", "الموقع والانتشار"),
                                       tr("Residuals vs leverage (Cook's D)", "البواقي مقابل الرفع (مسافة كوك)")]),
           f"{key}_rpanel", "residual_diagnostics")
    c1, c2 = st.columns(2)
    with c1:
        _chart(rc.residual_hist(r, rtl), f"{key}_rhist", "residual_histogram")
    with c2:
        _chart(rc.observed_vs_predicted(r, rtl), f"{key}_ovp", "observed_vs_predicted")
    if "acf" in diag["detail"]:
        c1, c2 = st.columns(2)
        with c1:
            _chart(rc.acf_plot(diag["detail"]["acf"], r.nobs, rtl), f"{key}_acf", "residual_acf")
        with c2:
            _chart(rc.actual_vs_fitted(r, rtl=rtl, names=(tr("Actual", "الفعلي"), tr("Fitted", "المقدَّر"))),
                   f"{key}_avf", "actual_vs_fitted")
    if "cusum" in diag["detail"]:
        st.markdown("**" + tr("Parameter stability (Brown-Durbin-Evans)", "استقرار المعاملات (براون-دربن-إيفانز)") + "**")
        _chart(rc.cusum_plots(diag["detail"]["cusum"], rtl, (tr("CUSUM (5% bounds)", "CUSUM (حدود 5%)"),
                                                             tr("CUSUM of squares (5% bounds)", "CUSUM للمربعات (حدود 5%)"))),
               f"{key}_cusum", "cusum")
    st.markdown("**" + tr("Coefficients with 95% confidence intervals", "المعاملات مع فترات الثقة 95%") + "**")
    _chart(rc.coef_plot([r], [r.name], rtl=rtl), f"{key}_coef", "coefficients")
    if spec.x:
        st.markdown("**" + tr("Predicted relationship (others at representative values)",
                              "العلاقة المقدَّرة (بقية المتغيرات عند قيمها الممثِّلة)") + "**")
        c1, c2 = st.columns(2)
        with c1:
            var = st.selectbox(tr("Variable", "المتغير"), spec.x, key=f"{key}_ecv")
        with c2:
            by = st.selectbox(tr("Separate curves by", "منحنيات منفصلة حسب"), ["—"] + spec.categorical, key=f"{key}_ecb")
        if len(spec.x) == 1 and not spec.categorical and not spec.poly and spec.form == "linear":
            _chart(rc.slr_bands(df, r, var, rtl), f"{key}_slr", "fitted_line_bands")
        else:
            _chart(rc.effect_curve(df, r, var, None if by == "—" else by, rtl), f"{key}_ec", "effect_curve")


def _policy(df, r, diag, alpha, key):
    lang = _lang()
    pb = rp.policy_brief(df, r, diag, lang, alpha)
    st.markdown("### " + tr("🏛️ Brief for decision-makers", "🏛️ موجز لصانع القرار"))
    st.write(pb["head"])
    for b in pb["bullets"]:
        st.markdown("- " + b, unsafe_allow_html=True)
    st.markdown(pb["reliability"], unsafe_allow_html=True)
    st.caption(pb["caution"])
    st.markdown("---")
    st.markdown("### " + tr("🔮 What-if simulation", "🔮 محاكاة «ماذا لو»"))
    spec = r.design.spec
    st.caption(tr("Start from the representative observation and change one or more drivers; the model predicts the new "
                  "level of Y with a 95% confidence interval for the change.",
                  "ابدأ من المشاهدة الممثِّلة وغيّر محدداً أو أكثر؛ يتنبأ النموذج بالمستوى الجديد لـ Y مع فترة ثقة 95% للتغير."))
    mode = st.radio(tr("Change numeric drivers by", "تغيير المحددات الرقمية بـ"), ["pct", "abs"], horizontal=True,
                    key=f"{key}_wim", format_func=lambda m: tr("percent (%)", "نسبة مئوية (%)") if m == "pct" else tr("absolute units", "وحدات مطلقة"))
    changes = {}
    cols = st.columns(min(3, max(1, len(spec.x) + len(spec.categorical))))
    base = rg.baseline_values(df, r)
    for i, v in enumerate(spec.x):
        with cols[i % len(cols)]:
            if mode == "pct":
                val = st.slider(f"{v} (%)", -50, 100, 0, 5, key=f"{key}_wi_{v}")
            else:
                sd = float(pd.to_numeric(df[v], errors="coerce").std())
                val = st.number_input(f"{v} (Δ, base {base[v].iloc[0]:.3g})", value=0.0, step=round(sd / 10, 4) or 0.1,
                                      key=f"{key}_wia_{v}")
            if val:
                changes[v] = val
    for j, c in enumerate(spec.categorical):
        with cols[(len(spec.x) + j) % len(cols)]:
            lv = r.design.info["levels"][c]
            sel = st.selectbox(c, lv, key=f"{key}_wic_{c}")
            if sel != lv[0]:
                changes[c] = sel
    if changes:
        w = rp.what_if(df, r, changes, mode)
        m = st.columns(3)
        m[0].metric(tr("Baseline Y", "Y في الوضع الأساسي"), f"{w['y0']:.4g}")
        m[1].metric(tr("Scenario Y", "Y في السيناريو"), f"{w['y1']:.4g}", f"{w['pct']:+.2f}%")
        m[2].metric(tr("Change (95% CI)", "التغير (فترة ثقة 95%)"), f"{w['change']:+.4g}",
                    f"[{w['change_ci'][0]:+.3g}, {w['change_ci'][1]:+.3g}]", delta_color="off")
        sig = w["p"] < alpha
        insight(tr(f"Under this scenario {spec.y} moves from {w['y0']:.4g} to {w['y1']:.4g} ({w['pct']:+.2f}%; 95% CI "
                   f"{w['pct_ci'][0]:+.2f}% to {w['pct_ci'][1]:+.2f}%). The change is "
                   + ("statistically significant." if sig else "not statistically distinguishable from zero."),
                   f"في هذا السيناريو ينتقل {spec.y} من {w['y0']:.4g} إلى {w['y1']:.4g} ({w['pct']:+.2f}%؛ فترة الثقة 95%: "
                   f"{w['pct_ci'][0]:+.2f}% إلى {w['pct_ci'][1]:+.2f}%). والتغير "
                   + ("معنوي إحصائياً." if sig else "غير مختلف معنوياً عن الصفر.")))
        st.caption(tr("Predictions in log-Y models are retransformed with Duan's (1983) smearing estimator.",
                      "التنبؤات في نماذج لوغاريتم Y مُعادة إلى وحداتها الأصلية بمقدّر دوان (1983)."))


def _export(df, r, diag, key):
    lang = _lang()
    txt, refs = rp.methodology(r, True, lang)
    st.markdown("**" + tr("Methodology paragraph (ready to paste)", "فقرة المنهجية (جاهزة للنسخ)") + "**")
    st.text_area("methodology", txt, height=220, key=f"{key}_meth", label_visibility="collapsed")
    with st.expander(tr(f"References ({len(refs)}, APA 7)", f"المراجع ({len(refs)}، بنمط APA 7)"), expanded=False):
        st.markdown("\n".join(f"- {x}" for x in refs))
    coef = r.coef_table().round(6)
    diag_tab = diag["table"].drop(columns=["Test_ar", "H0_ar"]).copy()
    diag_tab["df"] = diag_tab["df"].astype(str).replace({"None": "", "nan": ""})
    diag_tab["ok"] = diag_tab["ok"].astype(str)
    try:
        eff = rp.effects_table(df, r).round(6)
    except Exception:  # noqa: BLE001
        eff = pd.DataFrame()
    stats_df = pd.DataFrame({"Statistic": list(r.fit_stats()), "Value": [str(v) for v in r.fit_stats().values()]})
    jt, note = rp.journal_table([r], [r.name or "(1)"], lang=lang)
    st.markdown("**" + tr("Downloads", "التحميلات") + "**")
    c1, c2, c3 = st.columns(3)
    with c1:
        excel_download({"Coefficients": coef, "Fit": stats_df, "Diagnostics": diag_tab, "Effects": eff,
                        "Journal table": jt}, "aetheris_regression.xlsx", key=f"{key}_xlsx")
    with c2:
        sections = [("h", tr("Methodology", "المنهجية")), ("p", txt),
                    ("h", tr("Estimation results", "نتائج التقدير")), ("table", jt), ("note", note),
                    ("h", tr("Coefficients", "المعاملات")), ("table", coef),
                    ("h", tr("Diagnostic tests", "الاختبارات التشخيصية")),
                    ("table", diag_tab[["Test", "Statistic", "df", "p-value", "ok", "note"]].astype(str)),
                    ("h", tr("Interpretation", "التفسير")), ("p", rp.interpret(r, diag, 0.05, lang)),
                    ("h", tr("Policy brief", "موجز لصانع القرار"))]
        pb = rp.policy_brief(df, r, diag, lang)
        sections += [("p", pb["head"]), ("bullets", pb["bullets"]), ("p", pb["reliability"]),
                     ("h", tr("References", "المراجع")), ("bullets", refs)]
        if not eff.empty:
            sections.insert(6, ("h", tr("Elasticities & marginal effects", "المرونات والآثار الحدية")))
            sections.insert(7, ("table", eff.drop(columns=[c for c in ("Note",) if c in eff.columns])))
        st.download_button(tr("⬇️ Word report (.docx)", "⬇️ تقرير Word (.docx)"),
                           rp.to_docx(sections, tr("Regression analysis — Aetheris Research", "تحليل الانحدار — Aetheris Research"),
                                      rtl=is_ar()),
                           "aetheris_regression_report.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"{key}_docx")
    with c3:
        st.download_button(tr("⬇️ LaTeX table (.tex)", "⬇️ جدول LaTeX (.tex)"), rp.to_latex(jt, note), "regression_table.tex",
                           "text/x-tex", key=f"{key}_tex")
    st.markdown("**" + tr("Reproducible code", "الكود القابل لإعادة الإنتاج") + "**")
    st.caption(tr("Re-run the same model in Python, R or Stata (replace the file name with your data).",
                  "أعد تشغيل النموذج نفسه في Python أو R أو Stata (استبدل اسم الملف ببياناتك)."))
    t1, t2, t3 = st.tabs(["Python", "R", "Stata"])
    with t1:
        st.code(rp.code_python(r), language="python")
    with t2:
        st.code(rp.code_r(r), language="r")
    with t3:
        st.code(rp.code_stata(r), language="stata")


# ==========================================================
# 2. Journal table
# ==========================================================
def _journal_tab():
    models = _models()
    if not models:
        st.info(tr("No saved models yet. Estimate a model in the builder and press 'Save model' — "
                   "save several specifications to compare them side by side.",
                   "لا توجد نماذج محفوظة بعد. قدّر نموذجاً في تبويب البناء واضغط «حفظ النموذج» — واحفظ عدة "
                   "صيغ لمقارنتها جنباً إلى جنب."))
        return
    names = list(models)
    sel = _ms(tr("Models (columns)", "النماذج (الأعمدة)"), names, default=names[-4:], key="eco_jt_sel")
    if not sel:
        return
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        inner = st.selectbox(tr("In parentheses", "بين الأقواس"), ["se", "t", "p", "ci"], key="eco_jt_inner",
                             format_func=lambda s: {"se": tr("Std. errors", "الأخطاء المعيارية"), "t": tr("t-statistics", "قيم t"),
                                                    "p": tr("p-values", "القيم الاحتمالية"), "ci": tr("95% CI", "فترة الثقة 95%")}[s])
    with c2:
        dec = st.number_input(tr("Decimals", "المنازل العشرية"), 2, 6, 3, key="eco_jt_dec")
    with c3:
        lv = st.selectbox(tr("Star levels", "مستويات النجوم"), ["0.01/0.05/0.10", "0.001/0.01/0.05"], key="eco_jt_lv")
    with c4:
        caption = st.text_input(tr("Table caption", "عنوان الجدول"), tr("Table 1. Regression results", "جدول 1. نتائج تقدير الانحدار"),
                                key="eco_jt_cap")
    levels = (0.01, 0.05, 0.10) if lv.startswith("0.01") else (0.001, 0.01, 0.05)
    res = [models[n]["r"] for n in sel]
    jt, note = rp.journal_table(res, sel, inner=inner, decimals=int(dec), star_levels=levels, lang=_lang())
    st.markdown(rp.to_html(jt, note, caption), unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        excel_download({"Journal table": jt}, "journal_table.xlsx", key="eco_jt_xlsx")
    with c2:
        st.download_button(tr("⬇️ Word (.docx)", "⬇️ Word (.docx)"),
                           rp.to_docx([("table", jt), ("note", note)], caption, rtl=is_ar()), "journal_table.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="eco_jt_docx")
    with c3:
        st.download_button("⬇️ LaTeX (.tex)", rp.to_latex(jt, note, caption), "journal_table.tex", "text/x-tex", key="eco_jt_tex")
    with c4:
        st.download_button("⬇️ HTML", rp.to_html(jt, note, caption), "journal_table.html", "text/html", key="eco_jt_html")
    st.markdown("**" + tr("Coefficient comparison", "مقارنة المعاملات") + "**")
    _chart(rc.coef_plot(res, sel, rtl=is_ar()), "eco_jt_coef", "coefficient_comparison")
    with st.expander(tr("🗑️ Manage saved models", "🗑️ إدارة النماذج المحفوظة"), expanded=False):
        rm = _ms(tr("Remove", "حذف"), names, key="eco_jt_rm")
        if rm and st.button(tr("Remove selected", "حذف المحدد"), key="eco_jt_rm_btn"):
            for n in rm:
                models.pop(n, None)
            st.rerun()
        pick = st.selectbox(tr("Open full output of", "عرض المخرجات الكاملة لـ"), names, key="eco_jt_open")
        if st.checkbox(tr("Show", "عرض"), key="eco_jt_show"):
            m = models[pick]
            df = get_memory()
            if m["dataset"] == active_name():
                show_model(df, m["r"], m["diag"], 0.05, key="saved")
            else:
                st.info(tr(f"This model was estimated on '{m['dataset']}'. Activate that dataset to open its full output.",
                           f"قُدِّر هذا النموذج على «{m['dataset']}». فعِّل تلك البيانات لعرض مخرجاته الكاملة."))


# ==========================================================
# 3. Model selection & functional form
# ==========================================================
def _selection_tab(df):
    tool = st.radio(tr("Tool", "الأداة"), ["stepwise", "best", "hier", "forms", "boxcox"], horizontal=True, key="eco_sel_tool",
                    format_func=lambda t: {"stepwise": tr("Stepwise", "التدريجي"), "best": tr("Best subsets", "أفضل مجموعة"),
                                           "hier": tr("Hierarchical", "الهرمي"), "forms": tr("Compare functional forms", "مقارنة الصور الدالية"),
                                           "boxcox": tr("Box-Cox", "بوكس-كوكس")}[t])
    spec = _spec_widgets(df, f"eco_s_{tool}", False, allow_poly=tool in ("stepwise", "best", "hier"),
                         allow_inter=tool in ("stepwise", "best", "hier"))
    if spec is None:
        st.info(tr("Select at least one regressor.", "اختر متغيراً مستقلاً واحداً على الأقل."))
        return
    try:
        design = rg.build_design(df, spec)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return
    cand = [c for c in design.X.columns if c != "const"]
    if tool == "stepwise":
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            direction = st.selectbox(tr("Direction", "الاتجاه"), ["both", "forward", "backward"], key="eco_sw_dir",
                                     format_func=lambda d: {"both": tr("Bidirectional", "ثنائي الاتجاه"), "forward": tr("Forward", "أمامي"),
                                                            "backward": tr("Backward", "خلفي")}[d])
        with c2:
            crit = st.selectbox(tr("Criterion", "المعيار"), ["p", "aic", "bic"], key="eco_sw_crit",
                                format_func=lambda c: {"p": tr("p-values (SPSS)", "القيم الاحتمالية (SPSS)"), "aic": "AIC", "bic": "BIC"}[c])
        with c3:
            pe = st.number_input(tr("p to enter", "p للإدخال"), 0.001, 0.5, 0.05, key="eco_sw_pe")
        with c4:
            pr = st.number_input(tr("p to remove", "p للحذف"), 0.001, 0.5, 0.10, key="eco_sw_pr")
        force = _ms(tr("Always keep", "متغيرات تبقى دائماً"), cand, key="eco_sw_force")
        if st.button(tr("Run stepwise", "تشغيل الانحدار التدريجي"), key="eco_sw_run"):
            out = rg.stepwise(design, direction, crit, pe, pr, force)
            st.dataframe(out["steps"].round(5), width="stretch", hide_index=True)
            sel = [c for c in out["selected"] if c != "const"]
            st.success(tr("Selected: ", "المتغيرات المختارة: ") + (", ".join(sel) or "—"))
            st.warning(tr("Stepwise p-values after selection are biased downward (Harrell 2015). Treat the result as "
                          "exploratory and justify the final model with theory.",
                          "القيم الاحتمالية بعد الاختيار التدريجي متحيزة إلى الأسفل (Harrell 2015)، فتعامل مع النتيجة "
                          "باعتبارها استكشافية وبرّر النموذج النهائي نظرياً."))
            import statsmodels.api as sm
            fin = sm.OLS(design.y, design.X[out["selected"]]).fit()
            st.dataframe(pd.DataFrame({"Coef.": fin.params, "Std. Err.": fin.bse, "t": fin.tvalues, "p-value": fin.pvalues}).round(4),
                         width="stretch")
    elif tool == "best":
        c1, c2 = st.columns(2)
        with c1:
            mx = st.number_input(tr("Max model size", "أقصى حجم للنموذج"), 1, max(1, len(cand)), min(len(cand), 8), key="eco_bs_mx")
        with c2:
            force = _ms(tr("Always keep", "متغيرات تبقى دائماً"), cand, key="eco_bs_force")
        if st.button(tr("Run best subsets", "تشغيل أفضل مجموعة"), key="eco_bs_run"):
            try:
                out = rg.best_subsets(design, int(mx), force)
            except ValueError as e:
                st.error(str(e))
                return
            st.dataframe(_ar_cols(out.drop(columns="_cols").round(4)), width="stretch", hide_index=True)
            import plotly.graph_objects as go
            fig = go.Figure()
            for j, c in enumerate(["Adj. R²", "Mallows Cp", "BIC"]):
                fig.add_trace(go.Scatter(x=out["Size"], y=out[c], mode="lines+markers", name=c, yaxis="y" if j == 0 else "y2"))
            fig.update_layout(yaxis2=dict(overlaying="y", side="right"))
            ch.style_academic(fig, x_title=tr("Number of regressors", "عدد المتغيرات"), height=380, rtl=is_ar())
            _chart(fig, "eco_bs_fig", "best_subsets")
    elif tool == "hier":
        st.caption(tr("Assign regressors to blocks entered in sequence (e.g. block 1 = controls, block 2 = variables of interest).",
                      "وزّع المتغيرات على كتل تُدخل بالتتابع (مثلاً الكتلة 1 = المتغيرات الضابطة، والكتلة 2 = المتغيرات محل الاهتمام)."))
        nb = st.number_input(tr("Number of blocks", "عدد الكتل"), 2, 6, 2, key="eco_h_nb")
        blocks, used = [], set()
        cols = st.columns(int(nb))
        for i in range(int(nb)):
            with cols[i]:
                b = _ms(tr(f"Block {i + 1}", f"الكتلة {i + 1}"), [c for c in cand if c not in used],
                                   key=f"eco_h_b{i}")
                used |= set(b)
                blocks.append(b)
        blocks = [b for b in blocks if b]
        if blocks and st.button(tr("Run hierarchical regression", "تشغيل الانحدار الهرمي"), key="eco_h_run"):
            out = rg.hierarchical(design, blocks)
            st.dataframe(out.round(4), width="stretch", hide_index=True)
            for _, row in out.iterrows():
                if row["Block"] > 1:
                    sig = row["p (F change)"] < 0.05
                    st.markdown(tr(f"- Block {int(row['Block'])} adds ΔR² = {row['ΔR²']:.4f} (F change = {row['F change']:.3f}, "
                                   f"p = {row['p (F change)']:.4f}) — " + ("a significant improvement." if sig else "no significant improvement."),
                                   f"- الكتلة {int(row['Block'])} تضيف ΔR² = {row['ΔR²']:.4f} (F للتغير = {row['F change']:.3f}، "
                                   f"p = {row['p (F change)']:.4f}) — " + ("تحسن معنوي." if sig else "دون تحسن معنوي.")))
    elif tool == "forms":
        if st.button(tr("Compare all functional forms", "مقارنة جميع الصور الدالية"), key="eco_f_run"):
            out = rg.compare_forms(df, spec)
            st.dataframe(out.round(4), width="stretch", hide_index=True)
            st.caption(tr("R² is not comparable between Y and ln Y models. The Y-scale log-likelihood adds the Jacobian "
                          "(−Σ ln Y) to ln-Y models (Zarembka 1968), so the 'comparable' AIC/BIC rank all forms on the same footing.",
                          "معامل التحديد R² غير قابل للمقارنة بين نماذج Y ونماذج ln Y. ويضيف لوغاريتم الإمكان على مقياس Y حدَّ "
                          "جاكوبي (−Σ ln Y) لنماذج ln Y (زارمبكا 1968)، فتصبح قيم AIC/BIC «القابلة للمقارنة» صالحة للمفاضلة."))
            if "AIC (comparable)" in out:
                best = out.loc[out["AIC (comparable)"].idxmin()]
                insight(tr(f"Best form by comparable AIC: <b>{rg.FORMS[best['Form']][0]}</b> ({best['Equation']}).",
                           f"أفضل صيغة وفق معيار AIC القابل للمقارنة: <b>{rg.FORMS[best['Form']][1]}</b> ({best['Equation']})."))
    elif tool == "boxcox":
        if st.button(tr("Estimate Box-Cox λ", "تقدير معلمة بوكس-كوكس λ"), key="eco_bc_run"):
            try:
                bc = rg.box_cox(df, spec)
            except ValueError as e:
                st.error(str(e))
                return
            m = st.columns(3)
            m[0].metric("λ̂", f"{bc['lambda']:.4f}")
            m[1].metric(tr("95% CI", "فترة الثقة 95%"), f"[{bc['ci'][0]:.3f}, {bc['ci'][1]:.3f}]")
            m[2].metric("N", bc["n"])
            st.dataframe(pd.DataFrame([
                {"H0": "λ = 1 (linear Y)", "LR χ²(1)": bc["LR_lambda1"], "p-value": bc["p_lambda1"]},
                {"H0": "λ = 0 (ln Y)", "LR χ²(1)": bc["LR_lambda0"], "p-value": bc["p_lambda0"]}]).round(4),
                width="stretch", hide_index=True)
            rec = {"linear": tr("keep Y in levels", "إبقاء Y بقيمه الأصلية"), "log": tr("use ln(Y)", "استخدام ln(Y)"),
                   "boxcox": tr("neither level nor log is supported — consider the Box-Cox transformed Y or a different form",
                                "لا تدعم البيانات الصيغة الخطية ولا اللوغاريتمية — فكّر في Y المحوَّل ببوكس-كوكس أو صيغة أخرى")}[bc["recommend"]]
            insight(tr(f"Box-Cox (1964) profile likelihood: λ̂ = {bc['lambda']:.3f}. Recommendation: <b>{rec}</b>.",
                       f"دالة الإمكان الجانبية لبوكس-كوكس (1964): λ̂ = {bc['lambda']:.3f}. التوصية: <b>{rec}</b>."))
            _chart(rc.boxcox_profile(bc, is_ar()), "eco_bc_fig", "box_cox_profile")


# ==========================================================
# 4. Moderation & mediation
# ==========================================================
def _moderation_tab(df):
    num = ds.analysis_columns(df) or ds.numeric_columns(df)
    kind = st.radio(tr("Analysis", "التحليل"), ["mod", "med"], horizontal=True, key="eco_mm_kind",
                    format_func=lambda k: tr("Moderation (interaction)", "التأثير المعدِّل (التفاعل)") if k == "mod"
                    else tr("Mediation (indirect effect)", "التأثير الوسيط (الأثر غير المباشر)"))
    if len(num) < 3:
        st.info(tr("Needs at least three numeric variables.", "يتطلب ثلاثة متغيرات رقمية على الأقل."))
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        y = st.selectbox("Y", num, key="eco_mm_y")
    with c2:
        x = st.selectbox(tr("X (focal / cause)", "X (المتغير الرئيسي / السبب)"), [c for c in num if c != y], key="eco_mm_x")
    with c3:
        m = st.selectbox(tr("Moderator W" if kind == "mod" else "Mediator M", "المتغير المعدِّل W" if kind == "mod" else "المتغير الوسيط M"),
                         [c for c in num if c not in (y, x)], key="eco_mm_m")
    controls = _ms(tr("Controls", "متغيرات ضابطة"), [c for c in num if c not in (y, x, m)], key="eco_mm_c")
    if kind == "mod":
        center = st.checkbox(tr("Mean-centre X and W", "طرح المتوسط من X وW"), True, key="eco_mm_center")
        cov = st.selectbox(tr("Standard errors", "الأخطاء المعيارية"), ["nonrobust", "HC3", "HC1"], key="eco_mm_cov",
                           format_func=lambda c: rg.COV_TYPES[c][1 if is_ar() else 0])
        if st.button(tr("Estimate moderation model", "تقدير نموذج التأثير المعدِّل"), key="eco_mm_run"):
            spec = rg.ModelSpec(y=y, x=[x, m] + controls, interactions=[(x, m)], center=center)
            r = rg.fit(df, spec, cov_type=cov)
            st.dataframe(_coef_display(r), width="stretch", hide_index=True)
            ss = rg.simple_slopes(r, x, m)
            st.markdown("**" + tr("Simple slopes of X at values of W", "الميول البسيطة لـ X عند قيم W") + "**")
            st.dataframe(_ar_cols(ss.round(4)), width="stretch", hide_index=True)
            jn = rg.johnson_neyman(r, x, m)
            inter = f"{x}×{m}"
            p = r.pvalues[inter]
            roots = [z for z in jn["roots"] if jn["range"][0] <= z <= jn["range"][1]]
            txt = tr(f"The interaction {inter} has coefficient {r.params[inter]:.4g} (p = {p:.4f}): "
                     + ("W <b>significantly moderates</b> the effect of X." if p < 0.05 else "no significant moderation.")
                     + (f" Johnson-Neyman boundaries (centred W units): {', '.join(f'{z:.4g}' for z in roots)}." if roots else
                        " The slope's significance does not change within the observed range of W."),
                     f"معامل التفاعل {inter} = {r.params[inter]:.4g} (p = {p:.4f}): "
                     + ("المتغير W <b>يعدّل معنوياً</b> أثر X." if p < 0.05 else "لا يوجد تأثير معدِّل معنوي.")
                     + (f" حدود جونسون-نيمان (بوحدات W بعد طرح المتوسط): {', '.join(f'{z:.4g}' for z in roots)}." if roots else
                        " لا تتغير معنوية الميل داخل المدى المشاهد لـ W."))
            insight(txt)
            c1, c2 = st.columns(2)
            with c1:
                _chart(rc.simple_slopes_plot(r, x, m, is_ar()), "eco_mm_ss", "simple_slopes")
            with c2:
                _chart(rc.jn_plot(jn, m, x, is_ar()), "eco_mm_jn", "johnson_neyman")
    else:
        reps = st.select_slider(tr("Bootstrap replications", "عدد تكرارات البوتستراب"), [500, 1000, 2000, 5000], 2000, key="eco_med_reps")
        if st.button(tr("Estimate mediation", "تقدير التأثير الوسيط"), key="eco_med_run"):
            with st.spinner(tr("Bootstrapping the indirect effect…", "جارٍ حساب البوتستراب للأثر غير المباشر…")):
                out = rg.mediation(df, x, m, y, controls, reps=int(reps))
            st.dataframe(_ar_cols(out["table"].round(4)), width="stretch", hide_index=True)
            lo, hi = out["boot_ci"]
            kind_txt = {"full": tr("full mediation", "وساطة كاملة"), "partial": tr("partial mediation", "وساطة جزئية"),
                        "none": tr("no mediation", "لا توجد وساطة")}[out["type"]]
            insight(tr(f"Indirect effect a·b = {out['indirect']:.4g}; percentile bootstrap 95% CI [{lo:.4g}, {hi:.4g}] "
                       f"({out['reps']} resamples); Sobel z = {out['sobel_z']:.3f} (p = {out['sobel_p']:.4f}). "
                       f"Proportion mediated ≈ {out['prop_mediated']:.1%}. Conclusion: <b>{kind_txt}</b> "
                       "(Preacher & Hayes 2008 criterion: the bootstrap CI excludes zero).",
                       f"الأثر غير المباشر a·b = {out['indirect']:.4g}؛ فترة الثقة 95% بالبوتستراب [{lo:.4g}، {hi:.4g}] "
                       f"({out['reps']} عينة)؛ اختبار سوبل z = {out['sobel_z']:.3f} (p = {out['sobel_p']:.4f}). "
                       f"نسبة الأثر المنقول عبر الوسيط ≈ {out['prop_mediated']:.1%}. الخلاصة: <b>{kind_txt}</b> "
                       "(معيار بريتشر وهايز 2008: فترة ثقة البوتستراب لا تتضمن الصفر)."))
            import plotly.graph_objects as go
            fig = go.Figure(go.Histogram(x=out["boot"], marker_color=ch.PALETTES["academic"][0], opacity=0.75))
            for v in (lo, hi):
                fig.add_vline(x=v, line=dict(color=ch.PALETTES["academic"][1], dash="dash"))
            fig.add_vline(x=0, line=dict(color="#777"))
            ch.style_academic(fig, x_title="a·b", y_title=tr("Frequency", "التكرار"), height=360, legend="none", rtl=is_ar())
            _chart(fig, "eco_med_fig", "indirect_effect_bootstrap")


# ==========================================================
# 5. Hypothesis tests & restricted LS
# ==========================================================
def _tests_tab(df):
    cur = st.session_state.get(CURRENT_KEY)
    models = _models()
    pool = {}
    if cur and cur.get("dataset") == active_name():
        pool[tr("Current model (builder)", "النموذج الحالي (تبويب البناء)")] = cur["r"]
    for k, v in models.items():
        if v["dataset"] == active_name():
            pool[k] = v["r"]
    if not pool:
        st.info(tr("Estimate a model in the builder first.", "قدّر نموذجاً في تبويب البناء أولاً."))
        return
    name = st.selectbox(tr("Model", "النموذج"), list(pool), key="eco_t_model")
    r = pool[name]
    if r.design is None:
        return
    st.caption(tr("Coefficient names: ", "أسماء المعاملات: ") + " · ".join(f"`{n}`" for n in r.names))
    st.markdown("**" + tr("Linear restrictions (Wald test & restricted least squares)", "القيود الخطية (اختبار والد والمربعات الصغرى المقيدة)") + "**")
    example = " + ".join(r.names[1:3]) + " = 1" if len(r.names) > 2 else f"{r.names[-1]} = 0"
    text = st.text_area(tr("One restriction per line, e.g. constant returns to scale in a Cobb-Douglas model",
                           "قيد واحد في كل سطر، مثل ثبات العائد للسعة في دالة كوب-دوجلاس"), example, key="eco_t_text", height=90)
    c1, c2 = st.columns(2)
    with c1:
        run_w = st.button(tr("Wald test", "اختبار والد"), key="eco_t_wald")
    with c2:
        run_r = st.button(tr("Estimate restricted LS", "تقدير المربعات الصغرى المقيدة"), key="eco_t_rls")
    if run_w or run_r:
        try:
            R, q, labels = rg.parse_restrictions(text, r.names)
        except ValueError as e:
            st.error(str(e))
            return
        w = rg.wald_test(r, R, q)
        st.dataframe(pd.DataFrame([{"Restrictions": "; ".join(labels), "F": w["F"], "p (F)": w["p (F)"],
                                    "Wald χ²": w["Wald χ²"], "df": w["df"], "p (χ²)": w["p (χ²)"]}]).round(4),
                     width="stretch", hide_index=True)
        rej = (w["p (F)"] if np.isfinite(w["p (F)"]) else w["p (χ²)"]) < 0.05
        insight(tr(f"Using the model's {r.cov_label()} covariance, the restrictions are " + ("<b>rejected</b>." if rej else "<b>not rejected</b>."),
                   f"باستخدام مصفوفة التغاير ({r.cov_label('ar')}) " + ("<b>تُرفض</b> القيود." if rej else "<b>لا تُرفض</b> القيود.")))
        if run_r:
            rr = rg.restricted_ls(df, r.design.spec, R, q, labels, name=f"RLS: {'; '.join(labels)}")
            comp = pd.DataFrame({"Unrestricted": rr.extra["unrestricted_params"], "Restricted": rr.params,
                                 "Restricted s.e.": rr.bse})
            st.session_state["eco_rls"] = {"r": rr, "comp": comp, "dataset": active_name()}
    rls = st.session_state.get("eco_rls")
    if rls and rls["dataset"] == active_name():
        rr = rls["r"]
        st.markdown("**" + tr("Restricted least squares", "المربعات الصغرى المقيدة") + "**")
        st.dataframe(rls["comp"].round(5), width="stretch")
        st.info(rr.notes[-1])
        if st.button(tr("💾 Save RLS model to the journal table", "💾 حفظ نموذج RLS في جدول المجلات"), key="eco_t_save"):
            _models()[rr.name] = {"r": rr, "diag": dx.run_all(rr), "dataset": active_name()}
            st.success(tr("Saved.", "تم الحفظ."))
    st.markdown("---")
    st.markdown("**" + tr("Exclusion restrictions: F, Wald, LR and LM tests", "قيود الاستبعاد: اختبارات F ووالد ونسبة الإمكان ومضاعف لاجرانج") + "**")
    drop = _ms(tr("Coefficients jointly equal to zero", "معاملات تساوي الصفر معاً"), [n for n in r.names if n != "const"],
                          key="eco_t_drop")
    if drop and st.button(tr("Run tests", "إجراء الاختبارات"), key="eco_t_nest"):
        out = rg.nested_tests(df, r, drop)
        st.dataframe(out.round(5), width="stretch", hide_index=True)
        st.caption(tr("Classical versions satisfy W ≥ LR ≥ LM (Berndt-Savin); the robust Wald row uses the model's covariance.",
                      "تحقق الصيغ الكلاسيكية المتباينة W ≥ LR ≥ LM (بيرندت-سافين)؛ وصف والد المتين يستخدم مصفوفة تغاير النموذج."))


# ==========================================================
# 6. Nonlinear least squares
# ==========================================================
def _nls_tab(df):
    num = ds.analysis_columns(df) or ds.numeric_columns(df)
    c1, c2, c3 = st.columns(3)
    with c1:
        y = st.selectbox("Y", num, key="eco_n_y")
    with c2:
        x = st.selectbox("X", [c for c in num if c != y], key="eco_n_x")
    with c3:
        kind = st.selectbox(tr("Model", "النموذج"), list(rg.NLS_FORMS), key="eco_n_kind",
                            format_func=lambda k: rg.NLS_FORMS[k][1 if is_ar() else 0])
    st.caption(tr("Starting values come from linearised regressions; adjust them if the algorithm does not converge.",
                  "القيم الابتدائية مستمدة من انحدارات مُخطَّطة؛ عدّلها إذا لم تتقارب الخوارزمية."))
    if st.button(tr("Estimate NLS", "تقدير NLS"), key="eco_n_run"):
        try:
            r = rg.fit_nls(df, y, x, kind, name=f"NLS {kind}")
        except Exception as e:  # noqa: BLE001
            st.error(tr(f"Did not converge: {e}", f"لم تتقارب الخوارزمية: {e}"))
            return
        st.dataframe(_coef_display(r), width="stretch", hide_index=True)
        m = st.columns(4)
        m[0].metric("R²", f"{r.rsquared:.4f}")
        m[1].metric("σ̂", f"{r.sigma:.4g}")
        m[2].metric("AIC", f"{r.aic:.2f}")
        m[3].metric("N", r.nobs)
        _chart(rc.nls_curve(r, is_ar()), "eco_n_fig", "nls_fit")
        rows = []
        for t in dx.normality(r.resid):
            rows.append({"Test": dx.TEST_LABELS[t["key"]][1 if is_ar() else 0], "Statistic": t["stat"], "p-value": t["p"]})
        bp = dx.breusch_pagan(r.resid, r.wexog)
        rows.append({"Test": tr("Breusch-Pagan (on the Jacobian)", "بروش-باجان (على مصفوفة جاكوبي)"), "Statistic": bp["stat"], "p-value": bp["p"]})
        st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
        p = r.params
        if kind == "logistic":
            insight(tr(f"Saturation level K = {p['K']:.4g}; growth rate r = {p['r']:.4g}; inflection point x₀ = {p['x0']:.4g} "
                       f"(fastest growth, Y = K/2).",
                       f"مستوى التشبع K = {p['K']:.4g}؛ ومعدل النمو r = {p['r']:.4g}؛ ونقطة الانقلاب x₀ = {p['x0']:.4g} "
                       f"(أسرع نمو عندما Y = K/2)."))
        elif kind == "mitscherlich":
            insight(tr(f"Asymptotic maximum A = {p['A']:.4g}; response rate c = {p['c']:.4g} — each extra unit of X closes "
                       f"{(1 - np.exp(-p['c'])):.1%} of the remaining gap to the maximum.",
                       f"الحد الأقصى التقاربي A = {p['A']:.4g}؛ ومعدل الاستجابة c = {p['c']:.4g} — كل وحدة إضافية من X تسدّ "
                       f"{(1 - np.exp(-p['c'])):.1%} من الفجوة المتبقية حتى الحد الأقصى."))
        elif kind == "power":
            insight(tr(f"Constant elasticity b = {p['b']:.4g}: a 1% rise in {x} changes {y} by {p['b']:.4g}%.",
                       f"مرونة ثابتة b = {p['b']:.4g}: زيادة {x} بنسبة 1% تغيّر {y} بنسبة {p['b']:.4g}%."))
        elif kind == "exponential":
            insight(tr(f"Instantaneous growth rate b = {p['b']:.4g} per unit of {x} (compound {100 * (np.exp(p['b']) - 1):.3f}% per unit).",
                       f"معدل النمو اللحظي b = {p['b']:.4g} لكل وحدة من {x} (مركب {100 * (np.exp(p['b']) - 1):.3f}% لكل وحدة)."))
        csv_download(r.coef_table(std_beta=False).round(6), "nls_coefficients.csv", key="eco_n_dl")
