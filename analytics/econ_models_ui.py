# ==========================================
# Aetheris Research - Econometrics Lab, v1.5 tabs (UI)
# ------------------------------------------
# §1.2 IV & simultaneous equations · §1.4 qualitative response, censored & selection models ·
# §1.5 count & survival models — each delivered with the 10-element model standard.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from analytics import count_surv as cs
from analytics import descriptive as ds
from analytics import iv as ivm
from analytics import ldv
from analytics import margins as mg
from analytics import model_charts as mch
from analytics import model_code as mc
from analytics import model_report as mr
from analytics import reg_charts as rc
from analytics import reg_report as rp
from analytics import regression as rg
from core.memory import active_name
from core.utils import tr, is_ar, insight, excel_download

MODELS_KEY = "eco_models"


def _lang():
    return "ar" if is_ar() else "en"


def _ms(label, options, **kw):
    return st.multiselect(label, options, placeholder=tr("Choose…", "اختر…"), **kw)


def _chart(fig, key, name="figure"):
    from analytics import charts as ch
    st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, name), key=key)


def _explain(en, ar):
    st.caption("💡 " + tr(en, ar))


def _fmt_p(p):
    return "" if p is None or pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.4f}")


def _num(df):
    from analytics.econometrics_ui import _id_like
    num = [c for c in (ds.analysis_columns(df) or ds.numeric_columns(df))]
    return [c for c in num if not _id_like(df, c)] or num


def _ar(t: pd.DataFrame) -> pd.DataFrame:
    from analytics.econometrics_ui import COL_AR
    extra = {"Odds ratio": "نسبة الأرجحية", "RRR": "نسبة المخاطر النسبية", "IRR": "نسبة معدل الحدوث",
             "Hazard ratio": "نسبة الخطر", "Time ratio exp(β)": "نسبة الزمن exp(β)", "Outcome": "الفئة / النتيجة",
             "Effect": "الأثر", "z": "z", "Reading": "القراءة", "Term": "الحد", "Endogenous": "المتغير الداخلي",
             "Baseline": "الوضع الأساسي", "Scenario": "السيناريو", "Change": "التغير"}
    return t.rename(columns={**COL_AR, **extra}) if is_ar() else t


def _binary_cols(df):
    cols = [c for c in ds.numeric_columns(df) if set(pd.to_numeric(df[c], errors="coerce").dropna().unique()) <= {0, 1}
            and df[c].nunique() == 2]
    hints = ("adopt", "particip", "works", "default", "event", "select", "choice", "use", "access")
    return sorted(cols, key=lambda c: next((i for i, h in enumerate(hints) if h in c.lower()), len(hints)))


def _xvars(df, key, y, default_n=4, allow_cat=True, exclude=()):
    num = [c for c in _num(df) if c != y and c not in exclude]
    cat = [c for c in ds.categorical_columns(df) if c != y and c not in exclude]
    c1, c2, c3 = st.columns([3, 2, 2])
    with c1:
        x = _ms(tr("Explanatory variables X", "المتغيرات المفسِّرة X"), num, default=num[:default_n], key=f"{key}_x")
    cats = []
    with c2:
        if allow_cat:
            cats = _ms(tr("Categorical factors", "العوامل النوعية"), cat, key=f"{key}_cat")
    with c3:
        logs = _ms(tr("Use ln(X) for", "استخدام ln(X) لـ"), [v for v in x if (pd.to_numeric(df[v], errors="coerce") > 0).all()],
                   key=f"{key}_log")
    return rg.ModelSpec(y=y, x=x, categorical=cats, form="linlog" if logs else "linear",
                        nolog=[v for v in x if v not in logs])


def _save(r, key, extra=None):
    c1, c2 = st.columns([3, 1])
    models = st.session_state.setdefault(MODELS_KEY, {})
    with c1:
        label = st.text_input(tr("Model name for the journal table", "اسم النموذج في جدول المجلات"),
                              f"({len(models) + 1}) {r.y_label} — {r.name}", key=f"{key}_sname")
    with c2:
        st.write("")
        if st.button(tr("💾 Save model", "💾 حفظ النموذج"), key=f"{key}_save"):
            r.name = label
            models[label] = {"r": r, "diag": {"table": pd.DataFrame(), "problems": {}, "detail": {}}, "dataset": active_name(),
                             **(extra or {})}
            st.success(tr("Saved — compare it in the 'Journal table' tab.", "تم الحفظ — قارنه في تبويب «جدول بنمط المجلات»."))


def _coef_table(r) -> pd.DataFrame:
    t = pd.DataFrame({"Variable": r.names, "Coef.": r.params.values, "Std. Err.": r.bse.values,
                      ("t" if r.dist == "t" else "z"): r.tvalues.values, "p-value": [_fmt_p(p) for p in r.pvalues.values],
                      "CI low": r.ci["low"].values, "CI high": r.ci["high"].values,
                      "Sig.": [rg.stars(p) for p in r.pvalues.values]})
    return t.round(4)


def _stats_row(r):
    st_ = {"N": r.nobs, **{k: v for k, v in r.extra.get("stats", {}).items() if not isinstance(v, str)}}
    if np.isfinite(r.aic):
        st_.update({"AIC": r.aic, "BIC": r.bic})
    cols = st.columns(min(6, len(st_)))
    for c, (k, v) in zip(cols * 3, list(st_.items())[:12]):
        txt = f"{v:.4f}" if isinstance(v, float) and abs(v) < 100 else (f"{v:,.2f}" if isinstance(v, float) else f"{v}")
        c.metric(k, "\u200e" + txt)   # LRM keeps the minus sign on the left in RTL layouts


def _effects(df, r, key, at="average"):
    cache = r.extra.setdefault("_ame", {})
    if at not in cache:
        pred = r.extra.get("pred")
        if pred is None:
            cache[at] = pd.DataFrame()
        else:
            try:
                cache[at] = mg.marginal_effects(df, r, pred, outputs=r.extra.get("outputs"), at=at)
            except Exception as e:  # noqa: BLE001
                st.info(tr(f"Marginal effects not available: {e}", f"الآثار الحدية غير متاحة: {e}"))
                cache[at] = pd.DataFrame()
    return cache[at]


# ==========================================================
# Generic output for every v1.5 model
# ==========================================================
def show_generic(df, r, key, charts=None, extra_tables=None, alpha=0.05):
    lang = _lang()
    df = r.extra.get("person_period", df)   # discrete-time hazard models work on the person-period data
    st.markdown("---")
    st.subheader(tr("Results — ", "النتائج — ") + r.name)
    tabs = st.tabs([tr("📊 Estimates", "📊 التقديرات"), tr("🩺 Diagnostics", "🩺 التشخيص"),
                    tr("📐 Marginal effects", "📐 الآثار الحدية"), tr("📈 Charts", "📈 الرسوم"),
                    tr("🏛️ Policy & what-if", "🏛️ صانع القرار وماذا لو"), tr("📝 Methodology & export", "📝 المنهجية والتصدير")])
    is_iv = "Z" in r.extra
    eff = None
    with tabs[0]:
        st.dataframe(_ar(_coef_table(r)), width="stretch", hide_index=True)
        _stats_row(r)
        for title, tab in (extra_tables or []):
            st.markdown(f"**{title}**")
            tab = tab.round(4)
            if "p-value" in tab:
                tab["p-value"] = tab["p-value"].map(_fmt_p)
            st.dataframe(_ar(tab), width="stretch", hide_index=True)
        if is_iv:
            eff = rp.effects_table(df, r)
        else:
            eff = _effects(df, r, key)
        insight(mr.interpret(df, r, None if is_iv else eff, lang, alpha))
        _save(r, key)
    with tabs[1]:
        d = mr.diagnostics(df, r, lang)
        if not d.empty:
            d["p-value"] = d["p-value"].map(_fmt_p)
            d["df"] = d["df"].astype(str)
            st.dataframe(_ar(d.round(4)), width="stretch", hide_index=True)
        if is_iv:
            st.markdown("**" + tr("First-stage regressions", "انحدارات المرحلة الأولى") + "**")
            st.dataframe(ivm.first_stage(r).round(4), width="stretch", hide_index=True)
            st.caption(tr("Stock-Yogo (2005) critical values for one endogenous regressor are shown in the Cragg-Donald row; "
                          "the rule of thumb F ≥ 10 (Staiger & Stock 1997) applies otherwise.",
                          "القيم الحرجة لستوك-يوغو (2005) لمتغير داخلي واحد معروضة في صف كراج-دونالد؛ وفي غير ذلك تُطبَّق "
                          "القاعدة التقريبية F ≥ 10 (ستايجر وستوك 1997)."))
        cl = r.extra.get("classification")
        if isinstance(cl, dict):
            st.markdown("**" + tr("Classification table (cut-off 0.5)", "جدول التصنيف (حد القطع 0.5)") + "**")
            st.dataframe(cl["table"], width="stretch")
        elif isinstance(cl, pd.DataFrame):
            st.dataframe(cl, width="stretch")
        if "freq" in r.extra and not r.extra["freq"].empty:
            st.markdown("**" + tr("Observed vs predicted count frequencies", "التكرارات المشاهدة مقابل المتنبأ بها") + "**")
            st.dataframe(r.extra["freq"].round(4), width="stretch", hide_index=True)
    with tabs[2]:
        if is_iv:
            st.dataframe(_ar(eff.round(4)), width="stretch", hide_index=True)
        else:
            at = st.radio(tr("Evaluated", "تُحسب"), ["average", "mean"], horizontal=True, key=f"{key}_at",
                          format_func=lambda a: tr("Average marginal effects (AME)", "متوسط الآثار الحدية (AME)") if a == "average"
                          else tr("At the means (MEM)", "عند المتوسطات (MEM)"))
            e = _effects(df, r, key, at)
            if not e.empty:
                show = e.copy()
                show["p-value"] = show["p-value"].map(_fmt_p)
                st.dataframe(_ar(show.round(5)), width="stretch", hide_index=True)
                st.caption(tr("Continuous variables: derivative in the units of the original variable; binary variables and "
                              "factor levels: discrete change from 0 / the reference level. Delta-method standard errors.",
                              "المتغيرات المتصلة: المشتقة بوحدات المتغير الأصلي؛ والمتغيرات الثنائية ومستويات العوامل: التغير "
                              "المتقطع من 0 أو من الفئة المرجعية. الأخطاء المعيارية بطريقة دلتا."))
    with tabs[3]:
        if charts:
            charts()
        else:
            st.info(tr("No specific chart for this model.", "لا يوجد رسم خاص بهذا النموذج."))
    with tabs[4]:
        _policy(df, r, key, eff, alpha)
    with tabs[5]:
        _export(df, r, key, eff)


def _policy(df, r, key, eff, alpha):
    lang = _lang()
    st.markdown("### " + tr("🏛️ Key messages for decision-makers", "🏛️ رسائل أساسية لصانع القرار"))
    if "Z" in r.extra:
        pb = rp.policy_brief(df, r, None, lang, alpha)
        for b in pb["bullets"]:
            st.markdown("- " + b, unsafe_allow_html=True)
    else:
        bullets = mr.policy_brief(df, r, eff, lang, alpha)
        for b in bullets:
            st.markdown("- " + b, unsafe_allow_html=True)
        if not bullets:
            st.info(tr("No statistically significant driver at the chosen level — the model gives no actionable lever.",
                       "لا يوجد محدد معنوي إحصائياً عند المستوى المختار — لا يقدّم النموذج أداة سياسة قابلة للتطبيق."))
    pred = r.extra.get("pred")
    spec = r.design.spec if r.design is not None else None
    if spec is None or (pred is None and "Z" not in r.extra):
        return
    st.markdown("### " + tr("🔮 Policy scenario", "🔮 سيناريو السياسة"))
    st.caption(tr("Apply a change to every observation and compare the average predicted outcome before and after "
                  "(e.g. extend credit to all households, add 2 extension visits, raise schooling by 10%).",
                  "طبّق تغييراً على جميع المشاهدات وقارن متوسط النتيجة المتنبأ بها قبل التغيير وبعده (مثل إتاحة الائتمان "
                  "لكل الأسر، أو إضافة زيارتين إرشاديتين، أو رفع التعليم 10%)."))
    vars_ = list(spec.x) + list(spec.categorical)
    c1, c2, c3 = st.columns(3)
    with c1:
        v = st.selectbox(tr("Variable", "المتغير"), vars_, key=f"{key}_pv")
    if v in spec.categorical:
        with c2:
            val = st.selectbox(tr("Set every unit to", "اجعل جميع الوحدات"), r.design.info["levels"][v], key=f"{key}_pl")
        mode = "set"
    else:
        with c2:
            mode = st.selectbox(tr("Change", "نوع التغيير"), ["pct", "abs", "set"], key=f"{key}_pm",
                                format_func=lambda m: {"pct": tr("by percent", "بنسبة مئوية"), "abs": tr("by units", "بوحدات"),
                                                       "set": tr("set to value", "تثبيت عند قيمة")}[m])
        with c3:
            base = float(pd.to_numeric(df.loc[r.design.index, v], errors="coerce").mean())
            val = st.number_input(tr("Amount", "المقدار"), value=10.0 if mode == "pct" else (1.0 if mode == "abs" else round(base, 3)),
                                  key=f"{key}_pa")
    if "Z" in r.extra:
        w = rp.what_if(df, r, {v: val}, "pct" if mode == "pct" else "abs")
        insight(tr(f"Predicted {spec.y}: {w['y0']:.4g} → {w['y1']:.4g} ({w['pct']:+.2f}%; 95% CI {w['pct_ci'][0]:+.2f}% to {w['pct_ci'][1]:+.2f}%).",
                   f"قيمة {spec.y} المتنبأ بها: {w['y0']:.4g} ← {w['y1']:.4g} ({w['pct']:+.2f}%؛ فترة الثقة 95%: {w['pct_ci'][0]:+.2f}% إلى {w['pct_ci'][1]:+.2f}%)."))
        return
    outs = r.extra.get("outputs") or {"binary": ["P(y = 1)"], "discrete_hazard": ["P(event in period)"],
                                      "fractional": ["E[share]"], "count": ["E[count]"],
                                      "survival": ["relative hazard" if r.method == "cox" else "mean survival time"]}.get(
        r.extra.get("family"), ["E[y]"])
    sc = mg.scenario(df, r, pred, {v: val}, mode=mode, outputs=outs)
    show = sc.copy()
    show["p-value"] = show["p-value"].map(_fmt_p)
    st.dataframe(_ar(show.round(4)), width="stretch", hide_index=True)
    row = sc.iloc[-1] if r.extra.get("family") in ("censored",) else sc.iloc[0]
    insight(tr(f"Average {row['Outcome']}: {row['Baseline']:.4g} → {row['Scenario']:.4g} (change {row['Change']:+.4g}; 95% CI "
               f"{row['CI low']:+.4g} to {row['CI high']:+.4g}).",
               f"متوسط {row['Outcome']}: {row['Baseline']:.4g} ← {row['Scenario']:.4g} (التغير {row['Change']:+.4g}؛ فترة الثقة 95%: "
               f"{row['CI low']:+.4g} إلى {row['CI high']:+.4g})."))


def _export(df, r, key, eff):
    lang = _lang()
    if "Z" in r.extra:
        txt, refs = mr.methodology(r, lang)
    else:
        txt, refs = mr.methodology(r, lang)
    st.markdown("**" + tr("Methodology paragraph", "فقرة المنهجية") + "**")
    st.text_area("m", txt, height=170, key=f"{key}_meth", label_visibility="collapsed")
    with st.expander(tr(f"References ({len(refs)}, APA 7)", f"المراجع ({len(refs)}، بنمط APA 7)")):
        st.markdown("\n".join(f"- {x}" for x in refs))
    coef = _coef_table(r)
    diag = mr.diagnostics(df, r, lang)
    jt, note = rp.journal_table([r], [r.name], stats_rows=("N", "Pseudo R²", "Log-likelihood", "AIC", "BIC"), lang=lang)
    effx = eff if isinstance(eff, pd.DataFrame) else pd.DataFrame()
    c1, c2, c3 = st.columns(3)
    with c1:
        excel_download({"Coefficients": coef, "Diagnostics": diag.astype(str), "Marginal effects": effx, "Journal table": jt},
                       "aetheris_model.xlsx", key=f"{key}_xl")
    with c2:
        secs = [("h", tr("Methodology", "المنهجية")), ("p", txt), ("h", tr("Estimates", "التقديرات")), ("table", jt), ("note", note),
                ("h", tr("Diagnostics", "التشخيص")), ("table", diag.astype(str))]
        if not effx.empty:
            secs += [("h", tr("Marginal effects", "الآثار الحدية")), ("table", effx)]
        secs += [("h", tr("Interpretation", "التفسير")), ("p", mr.interpret(df, r, effx if "Z" not in r.extra else None, lang)),
                 ("h", tr("References", "المراجع")), ("bullets", refs)]
        st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"), rp.to_docx(secs, r.name, rtl=is_ar()), "aetheris_model.docx",
                           "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"{key}_docx")
    with c3:
        st.download_button("⬇️ LaTeX", rp.to_latex(jt, note, r.name), "model_table.tex", "text/x-tex", key=f"{key}_tex")
    t1, t2, t3 = st.tabs(["Python", "R", "Stata"])
    with t1:
        st.code(mc.code_python(r), language="python")
    with t2:
        st.code(mc.code_r(r), language="r")
    with t3:
        st.code(mc.code_stata(r), language="stata")


# ==========================================================
# §1.2 IV & simultaneous equations
# ==========================================================
def render_iv(df):
    mode = st.radio(tr("Type", "النوع"), ["single", "system"], horizontal=True, key="iv_mode",
                    format_func=lambda m: tr("Single equation (IV / 2SLS / LIML / GMM)", "معادلة واحدة (IV / 2SLS / LIML / GMM)")
                    if m == "single" else tr("System of equations (ILS / 2SLS / 3SLS / SUR)", "نظام معادلات (ILS / 2SLS / 3SLS / SUR)"))
    if mode == "single":
        _iv_single(df)
    else:
        _iv_system(df)


def _iv_single(df):
    _explain("An endogenous regressor is correlated with the error (simultaneity, omitted ability, measurement error). An "
             "instrument must be relevant (correlated with the endogenous regressor) and excluded (no direct effect on Y).",
             "المتغير الداخلي مرتبط بحد الخطأ (التحديد الآني، أو القدرة المحذوفة، أو خطأ القياس). ويجب أن تكون الأداة ذات صلة "
             "(مرتبطة بالمتغير الداخلي) ومستبعَدة (لا أثر مباشر لها على Y).")
    num = _num(df)
    cat = ds.categorical_columns(df)
    c1, c2 = st.columns(2)
    with c1:
        y = st.selectbox(tr("Dependent variable Y", "المتغير التابع Y"), num, key="iv_y")
        endog = _ms(tr("Endogenous regressor(s)", "المتغيرات الداخلية"), [c for c in num if c != y], key="iv_en")
        inst = _ms(tr("Excluded instruments", "الأدوات المستبعَدة"), [c for c in num if c != y and c not in endog], key="iv_z")
    with c2:
        exog = _ms(tr("Exogenous controls", "المتغيرات الخارجية الضابطة"), [c for c in num if c not in [y] + endog + inst], key="iv_ex")
        cats = _ms(tr("Categorical controls", "الضوابط النوعية"), [c for c in cat if c != y], key="iv_cat")
        form = st.selectbox(tr("Functional form", "الصورة الدالية"), ["linear", "loglin", "loglog", "linlog"], key="iv_form",
                            format_func=lambda f: f"{rg.FORMS[f][1 if is_ar() else 0]}   [{rg.FORMS[f][2]}]")
    nolog = []
    if form in ("loglog", "linlog"):
        nolog = _ms(tr("Keep in levels", "تبقى بقيمها"), exog + endog + inst, key="iv_nolog")
    c1, c2, c3 = st.columns(3)
    with c1:
        method = st.selectbox(tr("Estimator", "طريقة التقدير"), list(ivm.IV_METHODS), index=1, key="iv_m",
                              format_func=lambda m: ivm.IV_METHODS[m][1 if is_ar() else 0])
    with c2:
        cov = st.selectbox(tr("Standard errors", "الأخطاء المعيارية"), ["nonrobust", "HC1", "HC0", "HC3", "cluster", "HAC"], key="iv_cov",
                           format_func=lambda c: rg.COV_TYPES[c][1 if is_ar() else 0])
    with c3:
        small = st.checkbox(tr("Small-sample (t / F, n − K)", "تصحيح العينة الصغيرة (t / F، n − K)"), True, key="iv_small")
    opts = {}
    if cov == "cluster":
        opts["groups"] = st.selectbox(tr("Cluster variable", "متغير العناقيد"), ds.categorical_columns(df, 10_000) or list(df.columns), key="iv_clu")
    if cov == "HAC":
        opts["maxlags"] = st.number_input(tr("Lags", "الإبطاءات"), 0, 48, rg.newey_west_lags(len(df)), key="iv_lags")
    if not (endog and inst):
        st.info(tr("Choose at least one endogenous regressor and one excluded instrument.",
                   "اختر متغيراً داخلياً واحداً وأداة مستبعَدة واحدة على الأقل."))
        return
    spec = ivm.IVSpec(y=y, exog=exog, endog=endog, instruments=inst, categorical=cats, form=form, nolog=nolog)
    st.caption(tr(f"Order condition: {len(inst)} instrument(s) for {len(endog)} endogenous regressor(s) — ",
                  f"شرط الرتبة: {len(inst)} أداة لـ {len(endog)} متغير داخلي — ")
               + (tr("exactly identified", "تام التعرّف") if len(inst) == len(endog) else
                  tr("over-identified", "فائق التعرّف") if len(inst) > len(endog) else tr("UNDER-identified", "ناقص التعرّف")))
    if st.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="iv_run"):
        try:
            st.session_state["iv_res"] = (ivm.iv_fit(df, spec, method, cov, opts, small), active_name())
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    res = st.session_state.get("iv_res")
    if res and res[1] == active_name():
        r = res[0]

        def charts():
            ro = ivm.iv_fit(df, r.extra["spec"], "ols", small=r.extra["small"])
            st.markdown("**" + tr("OLS vs IV coefficients (95% CI)", "معاملات OLS مقابل IV (فترة ثقة 95%)") + "**")
            _chart(rc.coef_plot([ro, r], ["OLS", r.name], rtl=is_ar()), "iv_coefplot", "ols_vs_iv")
            _chart(rc.residual_hist(r, is_ar()), "iv_rh", "iv_residuals")
        show_generic(df, r, "ivr", charts=charts)


def _iv_system(df):
    _explain("Write each structural equation (e.g. demand and supply). The order condition requires each equation to exclude "
             "at least as many exogenous variables of the system as it includes endogenous regressors.",
             "اكتب كل معادلة هيكلية (مثل الطلب والعرض). ويتطلب شرط الرتبة أن تستبعد كل معادلة من متغيرات النظام الخارجية "
             "عدداً لا يقل عن عدد المتغيرات الداخلية التي تتضمنها.")
    num = _num(df)
    G = st.number_input(tr("Number of equations", "عدد المعادلات"), 2, 5, 2, key="sys_G")
    eqs = []
    for g in range(int(G)):
        with st.container(border=True):
            c0, c1, c2, c3 = st.columns([1, 1.2, 2, 2])
            with c0:
                nm = st.text_input(tr("Name", "الاسم"), ["Demand", "Supply", "Eq3", "Eq4", "Eq5"][g], key=f"sys_n{g}")
            with c1:
                y = st.selectbox("Y", num, index=min(g, len(num) - 1), key=f"sys_y{g}")
            with c2:
                en = _ms(tr("RHS endogenous", "متغيرات داخلية في الطرف الأيمن"), [c for c in num if c != y], key=f"sys_en{g}")
            with c3:
                ex = _ms(tr("Included exogenous", "متغيرات خارجية مضمَّنة"), [c for c in num if c != y and c not in en], key=f"sys_ex{g}")
            eqs.append(ivm.Equation(nm, y, en, ex))
    endog_all = {e.y for e in eqs} | {v for e in eqs for v in e.endog}
    exog_default = sorted({v for e in eqs for v in e.exog} - endog_all)
    extra = _ms(tr("Additional exogenous instruments for the system", "أدوات خارجية إضافية للنظام"),
                [c for c in num if c not in endog_all and c not in exog_default], key="sys_extra")
    exog_all = exog_default + extra
    method = st.selectbox(tr("Estimator", "طريقة التقدير"), list(ivm.SYS_METHODS), index=3, key="sys_m",
                          format_func=lambda m: ivm.SYS_METHODS[m][1 if is_ar() else 0])
    if exog_all and all(e.exog or e.endog for e in eqs):
        st.markdown("**" + tr("Identification", "التعرّف") + "**")
        try:
            st.dataframe(ivm.identification(df, eqs, exog_all).round(3), width="stretch", hide_index=True)
        except Exception as e:  # noqa: BLE001
            st.warning(str(e))
    if st.button(tr("▶️ Estimate system", "▶️ تقدير النظام"), type="primary", key="sys_run"):
        try:
            st.session_state["sys_res"] = (ivm.system_fit(df, eqs, exog_all, method), eqs, exog_all, active_name())
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    res = st.session_state.get("sys_res")
    if not res or res[3] != active_name():
        return
    s, eqs, exog_all = res[0], res[1], res[2]
    st.subheader(ivm.SYS_METHODS[s.method][1 if is_ar() else 0])
    jt, note = rp.journal_table(s.equations, [f"{e.name}: {e.y}" for e in eqs], stats_rows=("N", "R²"), lang=_lang())
    st.markdown(rp.to_html(jt, note, tr("System estimates", "تقديرات النظام")), unsafe_allow_html=True)
    for n in s.notes:
        st.info(n)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**" + tr("Residual covariance Σ̂", "مصفوفة تغاير البواقي Σ̂") + "**")
        st.dataframe(s.sigma.round(4), width="stretch")
    with c2:
        st.markdown("**" + tr("Reduced form (OLS on all exogenous variables)", "الصيغة المختزلة (OLS على كل المتغيرات الخارجية)") + "**")
        st.dataframe(ivm.reduced_form(df, eqs, exog_all).round(4), width="stretch")
    lines = []
    for e, r in zip(eqs, s.equations):
        sig = [c for c in r.names if c != "const" and r.pvalues[c] < 0.05]
        lines.append(f"<b>{e.name}</b> ({e.y}): " + ", ".join(f"{c} = {r.params[c]:.4g}{rg.stars(r.pvalues[c])}" for c in sig))
    insight("<br>".join(lines))
    c1, c2, c3 = st.columns(3)
    with c1:
        excel_download({"System": jt, "Sigma": s.sigma.reset_index(), "Identification": s.identification}, "system.xlsx", key="sys_xl")
    with c2:
        st.download_button("⬇️ LaTeX", rp.to_latex(jt, note, "System estimates"), "system.tex", key="sys_tex")
    with c3:
        st.download_button(tr("⬇️ Word", "⬇️ Word"), rp.to_docx([("table", jt), ("note", note)], "System estimates", is_ar()),
                           "system.docx", key="sys_docx")
    code = mc.system_code(s, eqs, exog_all)
    t = st.tabs(list(code))
    for tab, (k, v) in zip(t, code.items()):
        with tab:
            st.code(v, language=k.lower())
    st.caption(tr("References: ", "المراجع: ") + " · ".join(mr.REF[k].split(" (")[0] for k in ("theil53", "zt", "zellner")))


# ==========================================================
# §1.4 Qualitative response (binary, multinomial, ordered)
# ==========================================================
def render_choice(df):
    kind = st.radio(tr("Outcome type", "نوع المتغير التابع"), ["binary", "multi", "ordered"], horizontal=True, key="ch_kind",
                    format_func=lambda k: {"binary": tr("Binary (0/1)", "ثنائي (0/1)"),
                                           "multi": tr("Unordered categories", "فئات غير مرتبة"),
                                           "ordered": tr("Ordered categories", "فئات مرتبة")}[k])
    if kind == "binary":
        ys = _binary_cols(df)
        if not ys:
            st.info(tr("No 0/1 variable in the active dataset.", "لا يوجد متغير 0/1 في البيانات النشطة."))
            return
        _explain("Logit and probit give almost the same marginal effects; logit adds odds ratios. Complementary log-log suits "
                 "rare events. The LPM is easy to read but can predict outside [0, 1].",
                 "يعطي اللوجيت والبروبيت آثاراً حدية متقاربة جداً، ويضيف اللوجيت نسب الأرجحية. ويناسب cloglog الأحداث النادرة. "
                 "ونموذج الاحتمال الخطي سهل القراءة لكنه قد يتنبأ خارج المدى [0، 1].")
        y = st.selectbox(tr("Binary outcome", "المتغير الثنائي"), ys, key="bn_y")
        spec = _xvars(df, "bn", y)
        c1, c2 = st.columns(2)
        with c1:
            link = st.selectbox(tr("Model", "النموذج"), list(ldv.BINARY), index=1, key="bn_link",
                                format_func=lambda k: ldv.BINARY[k][1 if is_ar() else 0])
        with c2:
            cov = st.selectbox(tr("Standard errors", "الأخطاء المعيارية"), ["nonrobust", "HC0", "cluster"], key="bn_cov",
                               format_func=lambda c: {"nonrobust": tr("Observed information (ML)", "مصفوفة المعلومات المشاهدة (ML)"),
                                                      "HC0": tr("Robust (sandwich)", "متينة (ساندويتش)"),
                                                      "cluster": tr("Cluster-robust", "عنقودية متينة")}[c])
        opts = {"groups": st.selectbox(tr("Cluster variable", "متغير العناقيد"), ds.categorical_columns(df, 10_000), key="bn_clu")} if cov == "cluster" else {}
        if spec.x or spec.categorical:
            if st.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="bn_run"):
                try:
                    st.session_state["bn_res"] = (ldv.fit_binary(df, spec, link, cov, opts), active_name())
                except Exception as e:  # noqa: BLE001
                    st.error(str(e))
        res = st.session_state.get("bn_res")
        if res and res[1] == active_name():
            r = res[0]
            tables = []
            if "odds_ratios" in r.extra:
                tables.append((tr("Odds ratios", "نسب الأرجحية"), r.extra["odds_ratios"]))

            def charts():
                if r.method != "lpm":
                    _chart(mch.roc_plot(ldv.roc(r.y, r.fitted), is_ar()), "bn_roc", "roc")
                _chart(mch.prob_hist(r.y, np.clip(r.fitted, 0, 1), is_ar()), "bn_ph", "predicted_probabilities")
                v = st.selectbox(tr("Probability curve for", "منحنى الاحتمال لـ"), r.design.spec.x, key="bn_cv")
                if v:
                    _chart(mch.response_curve(df, r, v, r.extra["pred"], rtl=is_ar()), "bn_curve", "probability_curve")
            show_generic(df, r, "bnr", charts=charts, extra_tables=tables)
    elif kind == "multi":
        cats = [c for c in ds.categorical_columns(df) if 3 <= df[c].nunique() <= 12]
        if not cats:
            st.info(tr("Needs a categorical variable with 3–12 categories.", "يتطلب متغيراً نوعياً من 3 إلى 12 فئة."))
            return
        _explain("Multinomial logit compares every category with a base category; coefficients are log relative-risk ratios. "
                 "The IIA test checks whether removing one alternative changes the others' odds.",
                 "يقارن اللوجيت متعدد الحدود كل فئة بفئة أساس؛ والمعاملات لوغاريتمات نسب المخاطر النسبية. ويختبر IIA ما إذا كان "
                 "حذف بديل يغيّر أرجحية البدائل الأخرى.")
        y = st.selectbox(tr("Outcome", "المتغير التابع"), cats, key="mn_y")
        base = st.selectbox(tr("Base category", "فئة الأساس"), sorted(df[y].dropna().astype(str).unique()), key="mn_base")
        spec = _xvars(df, "mn", y)
        mm = st.radio(tr("Model", "النموذج"), ["mnlogit", "mnp_iid", "mnp_corr"], horizontal=True, key="mn_model",
                      format_func=lambda k: {"mnlogit": tr("Multinomial logit", "اللوجيت متعدد الحدود"),
                                             "mnp_iid": tr("Multinomial probit — independent errors", "البروبيت متعدد الحدود — أخطاء مستقلة"),
                                             "mnp_corr": tr("Multinomial probit — correlated errors (GHK)", "البروبيت متعدد الحدود — أخطاء مترابطة (GHK)")}[k])
        draws = 200
        if mm == "mnp_corr":
            draws = st.select_slider(tr("Halton draws (GHK)", "عدد سحبات هالتون (GHK)"), [100, 200, 400, 800], 200, key="mn_draws")
            st.caption(tr("Simulated ML can take 10–60 seconds. Correlated errors relax IIA but are weakly identified "
                          "with only case-specific regressors.",
                          "قد يستغرق التقدير بالمحاكاة من 10 إلى 60 ثانية. الأخطاء المترابطة تتجاوز قيد IIA لكنها ضعيفة "
                          "التعرّف عند الاقتصار على متغيرات خاصة بالمشاهدة."))
        if (spec.x or spec.categorical) and st.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="mn_run"):
            try:
                with st.spinner(tr("Estimating…", "جارٍ التقدير…")):
                    r_ = ldv.fit_mnlogit(df, spec, base) if mm == "mnlogit" else \
                        ldv.fit_mnprobit(df, spec, base, correlated=(mm == "mnp_corr"), draws=draws)
                st.session_state["mn_res"] = (r_, active_name())
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
        res = st.session_state.get("mn_res")
        if res and res[1] == active_name():
            r = res[0]

            def charts():
                v = st.selectbox(tr("Probabilities across", "الاحتمالات عبر"), r.design.spec.x, key="mn_cv")
                if v:
                    _chart(mch.response_curve(df, r, v, r.extra["pred"], r.extra["outputs"], rtl=is_ar()), "mn_curve", "mnl_probabilities")
            tables = [(tr("Relative-risk ratios", "نسب المخاطر النسبية"), r.extra["rrr"])] if "rrr" in r.extra else []
            if "omega_corr" in r.extra:
                tables.append((tr("Correlation of utility differences (Ω)", "ارتباط فروق المنفعة (Ω)"), r.extra["omega_corr"].reset_index()))
            show_generic(df, r, "mnr", charts=charts, extra_tables=tables)
    else:
        cand = [c for c in df.columns if 3 <= df[c].nunique() <= 10]
        if not cand:
            st.info(tr("Needs an ordinal variable with 3–10 levels.", "يتطلب متغيراً مرتباً من 3 إلى 10 مستويات."))
            return
        _explain("Ordered models assume one latent index and thresholds (cut-points). The Brant test checks that each "
                 "regressor has the same effect at every threshold (parallel lines).",
                 "تفترض النماذج المرتبة مؤشراً كامناً واحداً ونقاط قطع. ويختبر برانت أن لكل متغير الأثر نفسه عند كل نقطة قطع "
                 "(الخطوط المتوازية).")
        y = st.selectbox(tr("Ordered outcome", "المتغير المرتب"), cand, key="or_y")
        numeric = pd.to_numeric(df[y], errors="coerce").notna().all()
        if numeric:
            levels = order = None
            st.caption(tr("Numeric codes: categories are ordered from the lowest to the highest value.",
                          "رموز رقمية: تُرتَّب الفئات من القيمة الأدنى إلى الأعلى."))
        else:
            levels = sorted(df[y].dropna().astype(str).unique().tolist())
            order = _ms(tr("Order of categories (low → high)", "ترتيب الفئات (من الأدنى إلى الأعلى)"), levels, default=levels, key="or_order")
        spec = _xvars(df, "or", y)
        distr = st.radio(tr("Link", "دالة الربط"), ["logit", "probit"], horizontal=True, key="or_d")
        if (spec.x or spec.categorical) and (order is None or len(order) == len(levels)) and st.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="or_run"):
            try:
                st.session_state["or_res"] = (ldv.fit_ordered(df, spec, distr, order), active_name())
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
        res = st.session_state.get("or_res")
        if res and res[1] == active_name():
            r = res[0]
            tables = [(tr("Odds ratios", "نسب الأرجحية"), r.extra["odds_ratios"])] if "odds_ratios" in r.extra else []

            def charts():
                v = st.selectbox(tr("Probabilities across", "الاحتمالات عبر"), r.design.spec.x, key="or_cv")
                if v:
                    _chart(mch.response_curve(df, r, v, r.extra["pred"], r.extra["outputs"], rtl=is_ar()), "or_curve", "ordered_probabilities")
            show_generic(df, r, "orr", charts=charts, extra_tables=tables)


# ==========================================================
# §1.4 Censored, truncated, selection, hurdle, fractional
# ==========================================================
def render_censored(df):
    model = st.selectbox(tr("Model", "النموذج"), list(ldv.CENSORED), key="ce_model",
                         format_func=lambda k: ldv.CENSORED[k][1 if is_ar() else 0])
    _explain({"tobit": "Tobit: one decision generates both the zeros and the positive amounts (e.g. spending with many zeros).",
              "truncreg": "Truncated regression: units below / above the limit are absent from the sample altogether.",
              "heckman2": "Heckman: the outcome is observed only for participants (wages for workers). Needs a selection variable, "
                          "ideally with an exclusion restriction (a variable affecting participation but not the outcome).",
              "heckman_ml": "Heckman ML: joint estimation of selection and outcome — more efficient than two-step under normality.",
              "cragg": "Cragg hurdle: separate decisions to participate and how much (zeros only from the first hurdle).",
              "dhurdle": "Double hurdle: zeros may come from non-participation OR a corner solution of the amount decision.",
              "frac": "Fractional response: shares / rates in [0, 1] including exact 0s and 1s."}[model],
             {"tobit": "توبيت: قرار واحد يولّد الأصفار والقيم الموجبة معاً (مثل الإنفاق مع أصفار كثيرة).",
              "truncreg": "الانحدار المبتور: الوحدات خارج الحد غير موجودة في العينة أصلاً.",
              "heckman2": "هيكمان: النتيجة مشاهَدة للمشاركين فقط (الأجر للعاملين). يحتاج متغير اختيار، ويُفضَّل وجود قيد استبعاد "
                          "(متغير يؤثر في المشاركة دون النتيجة).",
              "heckman_ml": "هيكمان بالإمكان الأعظم: تقدير مشترك للاختيار والنتيجة — أكفأ من الخطوتين في ظل الطبيعية.",
              "cragg": "عقبة كراج: قرار المشاركة منفصل عن قرار الكمية (الأصفار من العقبة الأولى فقط).",
              "dhurdle": "العقبة المزدوجة: قد تأتي الأصفار من عدم المشاركة أو من حل زاوي في قرار الكمية.",
              "frac": "الاستجابة الكسرية: النسب والمعدلات في [0، 1] بما فيها القيم 0 و1 بالضبط."}[model])
    num = _num(df)
    y = st.selectbox(tr("Outcome Y", "المتغير التابع Y"), num, key="ce_y")
    if model in ("heckman2", "heckman_ml"):
        form = st.selectbox(tr("Outcome form", "صيغة معادلة النتيجة"), ["linear", "loglin"], key="ce_form",
                            format_func=lambda f: rg.FORMS[f][1 if is_ar() else 0])
        spec = _xvars(df, "ce", y)
        spec.form = "loglin" if form == "loglin" and spec.form == "linear" else ("loglog" if form == "loglin" else spec.form)
        st.markdown("**" + tr("Selection equation", "معادلة الاختيار") + "**")
        sel_y = st.selectbox(tr("Selection indicator (1 = observed)", "مؤشر الاختيار (1 = مشاهَد)"), _binary_cols(df), key="ce_sel")
        zopts = [c for c in num if c not in (y, sel_y)]
        zx = _ms(tr("Selection regressors (include an exclusion restriction)", "متغيرات معادلة الاختيار (ضمّن قيد استبعاد)"),
                 zopts, default=[v for v in spec.x if v in zopts], key="ce_z")
        sel = rg.ModelSpec(y=sel_y, x=zx, categorical=spec.categorical)
    else:
        spec = _xvars(df, "ce", y)
    c1, c2 = st.columns(2)
    left = right = None
    if model == "tobit":
        with c1:
            left = st.number_input(tr("Lower limit (censoring)", "الحد الأدنى (المراقبة)"), value=float(pd.to_numeric(df[y], errors="coerce").min()), key="ce_l")
        with c2:
            ur = st.checkbox(tr("Upper limit", "حد أعلى"), key="ce_ur")
            right = st.number_input(tr("Upper limit", "الحد الأعلى"), value=float(pd.to_numeric(df[y], errors="coerce").max()), key="ce_r") if ur else None
    if model == "truncreg":
        with c1:
            left = st.number_input(tr("Truncation point (y > c)", "نقطة البتر (y > c)"), value=0.0, key="ce_t")
    link = "logit"
    if model == "frac":
        with c1:
            link = st.radio(tr("Link", "دالة الربط"), ["logit", "probit"], horizontal=True, key="ce_link")
    if not (spec.x or spec.categorical):
        return
    if st.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="ce_run"):
        try:
            if model == "tobit":
                r = ldv.fit_tobit(df, spec, left=left, right=right)
            elif model == "truncreg":
                r = ldv.fit_truncreg(df, spec, lower=left)
            elif model in ("heckman2", "heckman_ml"):
                r = ldv.fit_heckman(df, spec, sel, "twostep" if model == "heckman2" else "ml")
            elif model in ("cragg", "dhurdle"):
                r = ldv.fit_hurdle(df, spec, kind=model)
            else:
                r = ldv.fit_fractional(df, spec, link)
            st.session_state["ce_res"] = (r, active_name())
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    res = st.session_state.get("ce_res")
    if res and res[1] == active_name():
        r = res[0]

        def charts():
            if r.extra.get("pred") is not None and r.design.spec.x:
                v = st.selectbox(tr("Prediction curve for", "منحنى التنبؤ لـ"), r.design.spec.x, key="ce_cv")
                outs = r.extra.get("outputs")
                _chart(mch.response_curve(df, r, v, r.extra["pred"], outs, rtl=is_ar()), "ce_curve", "prediction_curve")
            yy = pd.to_numeric(df[r.design.spec.y], errors="coerce").dropna()
            import plotly.graph_objects as go
            from analytics import charts as ch
            fig = go.Figure(go.Histogram(x=yy, nbinsx=50, marker_color=ch.PALETTES["academic"][0]))
            ch.style_academic(fig, x_title=r.design.spec.y, y_title=tr("Frequency", "التكرار"), height=340, legend="none", rtl=is_ar())
            _chart(fig, "ce_hist", "outcome_distribution")
        show_generic(df, r, "cer", charts=charts)


# ==========================================================
# §1.5 Count models
# ==========================================================
def render_count(df):
    num = [c for c in _num(df) if (pd.to_numeric(df[c], errors="coerce").dropna() >= 0).all()]
    if not num:
        st.info(tr("No non-negative numeric variable.", "لا يوجد متغير رقمي غير سالب."))
        return
    _explain("Start with Poisson; if the overdispersion test rejects, use the negative binomial. Many zeros → compare "
             "zero-inflated (two kinds of zeros) and hurdle (zero vs positive decision) models with Vuong / AIC. PPML suits "
             "non-negative continuous outcomes such as trade flows.",
             "ابدأ ببواسون؛ وإذا رفض اختبار فرط التشتت فاستخدم ذا الحدين السالب. ومع كثرة الأصفار قارن نماذج التضخم بالأصفار "
             "(نوعان من الأصفار) ونماذج العقبة (قرار الصفر مقابل الموجب) باختبار فوونج وAIC. ويناسب PPML المتغيرات الموجبة "
             "المتصلة مثل تدفقات التجارة.")
    c1, c2 = st.columns(2)
    with c1:
        y = st.selectbox(tr("Count outcome", "المتغير العدّي"), num, key="cn_y")
    with c2:
        model = st.selectbox(tr("Model", "النموذج"), list(cs.COUNT), key="cn_m", format_func=lambda k: cs.COUNT[k][1 if is_ar() else 0])
    spec = _xvars(df, "cn", y)
    inflate, opts, cov = None, {}, "nonrobust"
    if model in ("zip", "zinb"):
        inflate = _ms(tr("Inflation (excess-zero) regressors", "متغيرات معادلة تضخم الأصفار"), [c for c in _num(df) if c != y], key="cn_inf")
    if model in ("hurdle_p", "hurdle_nb"):
        opts["zero_dist"] = st.radio(tr("Zero hurdle", "عقبة الصفر"), ["logit", "poisson"], horizontal=True, key="cn_zd",
                                     format_func=lambda z: tr("Logit (binomial)", "لوجيت (ثنائي)") if z == "logit" else tr("Poisson-type (cloglog)", "من نوع بواسون (cloglog)"))
    if model in ("poisson", "negbin"):
        cov = st.selectbox(tr("Standard errors", "الأخطاء المعيارية"), ["nonrobust", "HC0"], key="cn_cov",
                           format_func=lambda c: tr("ML", "الإمكان الأعظم") if c == "nonrobust" else tr("Robust (sandwich)", "متينة"))
    if (spec.x or spec.categorical) and st.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="cn_run"):
        try:
            st.session_state["cn_res"] = (cs.fit_count(df, spec, model, inflate=inflate, cov_type=cov, cov_opts=opts), active_name())
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    res = st.session_state.get("cn_res")
    if res and res[1] == active_name():
        r = res[0]
        tables = [(tr("Incidence-rate ratios", "نسب معدلات الحدوث (IRR)"), r.extra["irr"])] if "irr" in r.extra else []

        def charts():
            if "freq" in r.extra and not r.extra["freq"].empty:
                _chart(mch.count_freq_plot(r.extra["freq"], is_ar()), "cn_freq", "count_frequencies")
            if r.extra.get("pred") is not None and r.design.spec.x:
                v = st.selectbox(tr("Expected count across", "العدد المتوقع عبر"), r.design.spec.x, key="cn_cv")
                _chart(mch.response_curve(df, r, v, r.extra["pred"], rtl=is_ar()), "cn_curve", "expected_count")
        show_generic(df, r, "cnr", charts=charts, extra_tables=tables)


# ==========================================================
# §1.5 Survival / duration
# ==========================================================
def render_survival(df):
    num = _num(df)
    bins = _binary_cols(df)
    if not bins:
        st.info(tr("Needs an event indicator (1 = event, 0 = censored).", "يتطلب مؤشر حدث (1 = وقع الحدث، 0 = مراقَب)."))
        return
    _explain("Duration data: time until an event (adoption, exit, default) with right-censoring for units still at risk at the "
             "end of observation. Kaplan-Meier describes, Cox estimates hazard ratios without assuming a baseline shape, AFT "
             "models estimate how covariates stretch or shorten time.",
             "بيانات المدة: الزمن حتى وقوع حدث (التبني، الخروج، التعثر) مع مراقبة يمينية للوحدات التي لم يقع لها الحدث حتى "
             "نهاية المشاهدة. يصف كابلان-ماير البيانات، ويقدّر كوكس نسب المخاطر دون افتراض شكل للخطر الأساسي، وتقدّر نماذج "
             "الزمن المتسارع كيف تطيل المتغيرات الزمن أو تقصّره.")
    c1, c2, c3 = st.columns(3)
    with c1:
        t = st.selectbox(tr("Duration / time", "المدة / الزمن"), [c for c in num if (pd.to_numeric(df[c], errors="coerce").dropna() > 0).all()], key="sv_t")
    with c2:
        e = st.selectbox(tr("Event indicator", "مؤشر الحدث"), bins, key="sv_e")
    with c3:
        model = st.selectbox(tr("Analysis", "التحليل"), ["km", "cox", "weibull", "exponential", "lognormal", "loglogistic", "discrete"], key="sv_m",
                             format_func=lambda m: {"km": tr("Kaplan-Meier & log-rank", "كابلان-ماير واختبار log-rank"),
                                                    "cox": tr("Cox proportional hazards", "المخاطر النسبية لكوكس"),
                                                    "discrete": tr("Discrete-time hazard", "المخاطر في الزمن المتقطع")}[m]
                             if m in ("km", "cox", "discrete") else f"AFT {cs.AFT[m][1 if is_ar() else 0]}")
    if model == "km":
        grp = st.selectbox(tr("Compare groups", "مقارنة المجموعات"), ["—"] + ds.categorical_columns(df) + bins, key="sv_g")
        d = df[[t, e] + ([grp] if grp != "—" else [])].dropna()
        km = cs.kaplan_meier(d[t], d[e])
        med = cs.km_median(km)
        m = st.columns(4)
        m[0].metric("N", len(d))
        m[1].metric(tr("Events", "الأحداث"), int(d[e].sum()))
        m[2].metric(tr("Median time", "الزمن الوسيط"), "—" if not np.isfinite(med["median"]) else f"{med['median']:.3g}")
        m[3].metric(tr("95% CI of median", "فترة الثقة للوسيط"), f"[{med['ci_low']:.3g}, {med['ci_high']:.3g}]")
        _chart(mch.km_plot(d[t], d[e], None if grp == "—" else d[grp], is_ar()), "sv_km", "kaplan_meier")
        if grp != "—":
            lr = cs.logrank(d[t], d[e], d[grp])
            st.dataframe(lr["table"].round(3), width="stretch", hide_index=True)
            insight(tr(f"Log-rank test: χ²({lr['df']}) = {lr['chi2']:.3f}, p = {lr['p']:.4f} — survivor functions "
                       + ("differ significantly across groups." if lr["p"] < 0.05 else "do not differ significantly."),
                       f"اختبار log-rank: χ²({lr['df']}) = {lr['chi2']:.3f}، p = {lr['p']:.4f} — دوال البقاء "
                       + ("تختلف معنوياً بين المجموعات." if lr["p"] < 0.05 else "لا تختلف معنوياً.")))
        with st.expander(tr("Life table", "جدول الحياة")):
            st.dataframe(km.round(4), width="stretch", hide_index=True)
        st.caption(tr("Greenwood standard errors, log(−log) confidence limits (as R survfit conf.type = 'log-log' and Stata sts).",
                      "أخطاء جرينوود المعيارية وحدود ثقة log(−log) (كما في R وStata)."))
        return
    spec = _xvars(df, "sv", t, exclude=(e,))
    ties = "efron"
    if model == "cox":
        ties = st.radio(tr("Ties", "معالجة القيم المتساوية"), ["efron", "breslow"], horizontal=True, key="sv_ties")
    if (spec.x or spec.categorical) and st.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="sv_run"):
        try:
            if model == "cox":
                r = cs.fit_cox(df, spec, t, e, ties)
            elif model == "discrete":
                r = cs.fit_discrete_hazard(df, spec, t, e)
            else:
                r = cs.fit_aft(df, spec, t, e, model)
            st.session_state["sv_res"] = (r, active_name())
        except Exception as ex:  # noqa: BLE001
            st.error(str(ex))
    res = st.session_state.get("sv_res")
    if res and res[1] == active_name():
        r = res[0]
        tables = []
        if "hr" in r.extra:
            tables.append((tr("Hazard ratios", "نسب المخاطر"), r.extra["hr"]))
        if "time_ratios" in r.extra:
            tables.append((tr("Time ratios", "نسب الزمن"), r.extra["time_ratios"]))
        if "ph_form" in r.extra:
            tables.append((tr("Equivalent proportional-hazards form", "الصيغة المكافئة للمخاطر النسبية"), r.extra["ph_form"]))

        def charts():
            if r.design.spec.x:
                v = st.selectbox(tr("Survival curves across", "منحنيات البقاء عبر"), r.design.spec.x, key="sv_cv")
                if r.method == "cox":
                    _chart(mch.cox_curves(df, r, v, is_ar()), "sv_cox", "cox_survival")
                elif r.method.startswith("aft_"):
                    _chart(mch.aft_curves(df, r, v, is_ar()), "sv_aft", "aft_survival")
            _chart(mch.km_plot(df[t], df[e], rtl=is_ar()), "sv_km2", "kaplan_meier")
        show_generic(df, r, "svr", charts=charts, extra_tables=tables)
