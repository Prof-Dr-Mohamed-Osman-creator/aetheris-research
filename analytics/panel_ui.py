# ==========================================
# Aetheris Research - Panel data tab (UI) — v1.8, Catalogue §1.15
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics import charts as ch
from analytics import descriptive as ds
from analytics import panel as pn
from analytics import panel_adv as pa
from analytics import panel_gmm as pg
from analytics import panel_report as pr
from analytics import reg_report as rp
from analytics import ts_ui as tu
from core.memory import active_name
from core.utils import tr, is_ar, insight, excel_download

PAL = ch.PALETTES["academic"]
_disp, _chart, _code, _lang = tu._disp, tu._chart, tu._code, tu._lang
fp = pr.fmt_p
tu._AR_COLS.update({"SD overall": "الانحراف الكلي", "SD between": "الانحراف البيني", "SD within": "الانحراف الداخلي",
                    "Min": "الأدنى", "Max": "الأعلى", "Mean": "المتوسط", "Share of within variance": "نصيب التباين الداخلي",
                    "Distribution": "التوزيع", "Standardised": "المعيارية", "Long-run effect": "الأثر طويل الأجل",
                    "Std. Err. (cluster)": "الخطأ المعياري (عنقودي)", "Factors": "عدد العوامل", "Units": "الوحدات",
                    "Lags": "الإبطاءات", "H0": "فرض العدم"})


def _sig(P) -> tuple:
    return (active_name(), P.y, tuple(P.x), P.O, P.N)


def _store(key, v, P):
    st.session_state[f"_v18res_{key}"] = (_sig(P), v)


def _load(key, P):
    v = st.session_state.get(f"_v18res_{key}")
    return v[1] if isinstance(v, tuple) and v[0] == _sig(P) else None


def _method(kind, key, **kw):
    txt, refs = pr.methodology(kind, _lang(), **kw)
    with st.expander(tr("📝 Methodology paragraph & references", "📝 فقرة المنهجية والمراجع"), expanded=False):
        st.text_area("m", txt, height=160, key=f"{key}_meth", label_visibility="collapsed")
        st.markdown("\n".join(f"- {x}" for x in refs))
    return txt, refs


def _table(df, digits=4):
    show = df.copy()
    for c in show.columns:
        if c.startswith("p-value") or c in ("p", "p (boot)"):
            show[c] = show[c].map(lambda v: fp(v) if isinstance(v, (float, np.floating)) else v)
    h = 35 * (len(show) + 1) + 3 if len(show) <= 20 else None
    st.dataframe(_disp(show).round(digits), width="stretch", hide_index=True, **({"height": h} if h else {}))


# ==========================================================
def render_panel(df):
    st.caption("💡 " + tr("Choose the unit identifier(s) and the time variable. Each unit must have one observation per period; "
                         "add a second identifier (e.g. governorate + crop) or a filter if needed.",
                         "اختر معرّف الوحدة والمتغير الزمني. يجب أن يكون لكل وحدة مشاهدة واحدة في كل فترة؛ أضف معرّفاً ثانياً "
                         "(مثل المحافظة + المحصول) أو مرشحاً عند الحاجة."))
    cols = list(df.columns)
    tcols = tu._time_cols(df) + [c for c in ds.time_columns(df) if c not in tu._time_cols(df)]
    num = ds.numeric_columns(df)
    cats = [c for c in df.columns if c not in num or df[c].nunique() < 0.5 * len(df)]
    c1, c2 = st.columns(2)
    with c1:
        ent = st.multiselect(tr("Unit identifier(s)", "معرّف الوحدة"), [c for c in cols if c not in tcols],
                             default=[c for c in cols if c not in tcols and df[c].dtype == object][:1], key="pn_ent",
                             placeholder=tr("Choose…", "اختر…"))
    with c2:
        tvar = st.selectbox(tr("Time variable", "المتغير الزمني"), tcols or cols, key="pn_t")
    xs_all = [c for c in num if c not in ent and c != tvar]
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        y = st.selectbox(tr("Dependent variable", "المتغير التابع"), xs_all, key="pn_y")
    with c2:
        x = st.multiselect(tr("Regressors", "المتغيرات المفسرة"), [c for c in xs_all if c != y],
                           default=[c for c in xs_all if c != y][:2], key="pn_x", placeholder=tr("Choose…", "اختر…"))
    with c3:
        logs = st.multiselect(tr("Take logs of", "أخذ اللوغاريتم لـ"), [y] + x, key="pn_log", placeholder=tr("None", "لا شيء"))
    filters = {}
    others = [c for c in df.columns if df[c].dtype == object and c not in ent]
    if others:
        with st.expander(tr("Filters", "المرشحات"), expanded=False):
            fc = st.columns(min(3, len(others)))
            for i, c in enumerate(others):
                with fc[i % len(fc)]:
                    v = st.selectbox(c, ["—"] + sorted(df[c].dropna().astype(str).unique().tolist()), key=f"pn_f_{c}")
                    if v != "—":
                        filters[c] = v
    if not ent or not x:
        st.info(tr("Select the unit identifier and at least one regressor.", "اختر معرّف الوحدة ومتغيراً مفسراً واحداً على الأقل."))
        return
    d = df.copy()
    for v in logs:
        if (pd.to_numeric(d[v], errors="coerce") <= 0).any():
            st.error(tr(f"{v}: logs need positive values.", f"{v}: اللوغاريتم يحتاج قيماً موجبة."))
            return
        d[v] = np.log(pd.to_numeric(d[v], errors="coerce"))
    ren = {v: f"ln_{v}" for v in logs}
    d = d.rename(columns=ren)
    y2, x2 = ren.get(y, y), [ren.get(v, v) for v in x]
    try:
        P = pn.prepare(d, ent, tvar, y2, x2, filters)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return
    desc = pn.describe(P)
    m = st.columns(5)
    for col, (k_, v_) in zip(m, [(tr("Units", "الوحدات"), desc["N"]), (tr("Periods", "الفترات"), desc["T"]),
                                 (tr("Observations", "المشاهدات"), desc["obs"]), (tr("Balanced", "متوازن"), tr("yes", "نعم") if desc["balanced"] else tr("no", "لا")),
                                 ("T̄", f"{desc['Ti mean']:.1f}")]):
        col.metric(k_, str(v_))
    tabs = st.tabs([tr("📋 Static models & tests", "📋 النماذج الساكنة والاختبارات"), tr("📉 Unit roots & cointegration", "📉 جذور الوحدة والتكامل المشترك"),
                    tr("🔁 Dynamic GMM", "🔁 GMM الديناميكي"), tr("📐 Panel ARDL (PMG / MG / DFE)", "📐 ARDL للبانل (PMG / MG / DFE)"),
                    tr("🧠 Advanced (CCE, IFE, threshold, PVAR)", "🧠 المتطورة (CCE، IFE، العتبة، PVAR)"), tr("➡️ Causality", "➡️ السببية")])
    with tabs[0]:
        _static(P, desc)
    with tabs[1]:
        _unitroots(P)
    with tabs[2]:
        _gmm(P, d, ent, tvar)
    with tabs[3]:
        _pardl(P)
    with tabs[4]:
        _advanced(P)
    with tabs[5]:
        _causality(P)


def _static(P, desc):
    lang = _lang()
    with st.expander(tr("Within / between variation (xtsum)", "التباين الداخلي والبيني (xtsum)"), expanded=False):
        _table(desc["xtsum"])
        st.caption(tr("FE identifies effects only from within variation; variables with little within variance are poorly "
                      "estimated by FE.", "تعتمد الآثار الثابتة على التباين الداخلي فقط؛ والمتغيرات قليلة التباين الداخلي تُقدَّر بدقة ضعيفة."))
    c1, c2 = st.columns(2)
    cov = c1.selectbox(tr("Standard errors", "الأخطاء المعيارية"), list(pn.COVS), key="pn_cov", format_func=lambda k: pn.COVS[k][1 if is_ar() else 0])
    td = c2.checkbox(tr("Time dummies in pooled / FE / RE / FD", "متغيرات زمنية وهمية في التجميعي/FE/RE/FD"), key="pn_td")
    res = {}
    for m_ in ("pooled", "fe", "fe2", "re", "be", "fd"):
        try:
            res[m_] = pn.fit(P, m_, cov if m_ != "be" else "classic", td)
        except Exception as e:  # noqa: BLE001
            st.warning(f"{m_}: {e}")
    try:
        res["fgls"] = pn.fgls(P, "hetero", td)
    except Exception as e:  # noqa: BLE001
        st.warning(f"FGLS: {e}")
    labs = [pn.MODELS[k][1 if is_ar() else 0] for k in res]
    regs = [r.to_reg(lab) for r, lab in zip(res.values(), labs)]
    keep = list(P.x) + ["const"]
    jt, note = rp.journal_table(regs, labs, stats_rows=("N", "R²"), lang=lang, keep=keep)
    est_lab, se_lab = ("طريقة التقدير", "الأخطاء المعيارية") if lang == "ar" else ("Estimator", "Std. errors")
    jt = jt[jt[""] != est_lab].reset_index(drop=True)
    i_se = jt.index[jt[""] == se_lab]
    if len(i_se):
        for k, lab in zip(res, labs):
            jt.loc[i_se[0], lab] = ("GLS" if k == "fgls" else pn.COVS["classic" if k == "be" else cov][1 if lang == "ar" else 0])
    st.markdown(rp.to_html(jt, note, tr("Table. Static panel estimates", "جدول. تقديرات النماذج الساكنة")), unsafe_allow_html=True)
    st.caption(tr(f"Standard errors: {pn.COVS[cov][0]} (between: classical; FGLS: GLS covariance as Stata xtgls, panels(hetero)).",
                  f"الأخطاء المعيارية: {pn.COVS[cov][1]} (البيني: تقليدية؛ FGLS: تباين GLS كما في xtgls بخيار panels(hetero))."))
    fe, re, po = res["fe"], res["re"], res["pooled"]
    pcl = pn.fit(P, "pooled", time_dummies=td)
    fcl, rcl = pn.fit(P, "fe", time_dummies=td), pn.fit(P, "re", time_dummies=td)
    bp = pn.bp_lm(pcl)
    hs = pn.hausman(fcl, rcl)
    try:
        hr = pn.hausman_robust(P, td)
    except Exception:  # noqa: BLE001
        hr = None
    cd = pn.cross_section_dependence(fcl)
    war = pn.wooldridge_ar(P, td)
    wfd = pn.wooldridge_fd(P)
    gh = pn.groupwise_het(fcl)
    rows = [{"Test": tr("F test (unit effects = 0)", "اختبار F (الآثار الفردية = 0)"), "H0": tr("pooled OLS", "التجميعي"), "Statistic": fcl.stats["F (effects = 0)"], "p-value": fcl.stats["p (F effects)"]},
            {"Test": "Breusch-Pagan LM", "H0": tr("no random effect", "لا أثر عشوائي"), "Statistic": bp["LM"], "p-value": bp["p"]},
            {"Test": "Honda", "H0": tr("no random effect", "لا أثر عشوائي"), "Statistic": bp["Honda"], "p-value": bp["p_honda"]},
            {"Test": "Hausman", "H0": tr("RE consistent", "RE متسق"), "Statistic": hs["stat"], "p-value": hs["p"]}]
    if hr:
        rows.append({"Test": tr("Hausman robust (Mundlak)", "هاوسمان المتين (موندلاك)"), "H0": tr("RE consistent", "RE متسق"), "Statistic": hr["stat"], "p-value": hr["p"]})
    rows += [{"Test": r_["Test"], "H0": tr("no cross-sectional dependence", "لا اعتماد مقطعي"), "Statistic": r_["Statistic"], "p-value": r_["p-value"]} for _, r_ in cd.iterrows()]
    rows += [{"Test": tr("Wooldridge AR (FE residuals)", "وولدريدج للارتباط الذاتي (بواقي FE)"), "H0": tr("no serial correlation", "لا ارتباط ذاتي"), "Statistic": war["stat"], "p-value": war["p"]},
             {"Test": tr("Wooldridge / Drukker (xtserial)", "وولدريدج / دروكر (xtserial)"), "H0": tr("no serial correlation", "لا ارتباط ذاتي"), "Statistic": wfd["stat"], "p-value": wfd["p"]},
             {"Test": tr("Modified Wald (groupwise heteroskedasticity)", "والد المعدل (عدم تجانس التباين بين الوحدات)"), "H0": tr("homoskedastic", "تجانس التباين"), "Statistic": gh["stat"], "p-value": gh["p"]}]
    try:
        sh = pn.slope_homogeneity(P)
        rows += [{"Test": "Pesaran-Yamagata Δ", "H0": tr("homogeneous slopes", "تجانس الميول"), "Statistic": sh["Delta"], "p-value": sh["p"]},
                 {"Test": "Pesaran-Yamagata Δ_adj", "H0": tr("homogeneous slopes", "تجانس الميول"), "Statistic": sh["Delta_adj"], "p-value": sh["p_adj"]}]
    except Exception:  # noqa: BLE001
        sh = None
    tt = pd.DataFrame(rows)
    _table(tt)
    if hs.get("note"):
        st.caption("⚠️ " + hs["note"])
    insight(pr.advice({"F": fcl.stats["p (F effects)"], "LM": bp["p"], "H": (hr or hs)["p"], "CD": float(cd["p-value"].iloc[0]),
                       "AR": war["p"], "HET": gh["p"]}, lang))
    if sh and sh["p_adj"] < 0.05:
        st.caption("⚠️ " + tr("Slopes are heterogeneous: consider the mean-group / CCE-MG estimators (Advanced tab).",
                               "الميول غير متجانسة: جرّب مقدّرات المتوسط الجماعي أو CCE-MG (تبويب المتطورة)."))
    if "re" in res:
        st.caption(tr(f"RE: θ̄ = {re.stats['θ (mean)']:.3f}, share of unit-effect variance ρ = {re.stats['ρ (share of individual variance)']:.3f}.",
                      f"الآثار العشوائية: θ̄ = {re.stats['θ (mean)']:.3f}، ونصيب تباين الأثر الفردي ρ = {re.stats['ρ (share of individual variance)']:.3f}."))
    txt, refs = _method("static", "pns")
    c1, c2, c3 = st.columns(3)
    with c1:
        excel_download({**{k: r.table() for k, r in res.items()}, "Tests": tt}, "panel_static.xlsx", key="pn_xl")
    with c2:
        st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"),
                           rp.to_docx([("h", tr("Methodology", "المنهجية")), ("p", txt), ("table", jt), ("note", note),
                                       ("table", tt.round(4).astype(str)), ("bullets", refs)], tr("Static panel models", "نماذج البانل الساكنة"), is_ar()),
                           "panel_static.docx", key="pn_docx")
    with c3:
        if st.button(tr("💾 Save to journal table", "💾 حفظ في جدول المجلات"), key="pn_save"):
            for k, r in res.items():
                st.session_state.setdefault("eco_models", {})[f"{pn.MODELS[k][0]} — {P.y}"] = {
                    "r": r.to_reg(pn.MODELS[k][0]), "diag": {"table": pd.DataFrame(), "problems": {}, "detail": {}}, "dataset": active_name()}
            st.success(tr("Saved.", "تم الحفظ."))
    _code(pr.code("static", unit=P.entity, time=P.time, y=P.y, x=P.x), "pns_code")


def _unitroots(P):
    lang = _lang()
    v = st.selectbox(tr("Variable", "المتغير"), [P.y] + P.x, key="pur_v_" + "_".join([P.y] + P.x))
    c1, c2, c3 = st.columns(3)
    exo = c1.selectbox(tr("Deterministics", "الحدود المحددة"), ["intercept", "trend"], key="pur_exo",
                       format_func=lambda e: tr("Individual intercepts", "ثوابت فردية") if e == "intercept" else tr("Intercepts and trends", "ثوابت واتجاهات"))
    lagm = c2.selectbox(tr("Lag selection", "اختيار الإبطاء"), ["SIC", "AIC", "Hall", 0, 1, 2], key="pur_lag")
    pmax = c3.number_input("pmax", 1, 8, 4, key="pur_pmax")
    S = pn.series_by_unit(P, v)
    Sd = {k: np.diff(a) for k, a in S.items()}
    rows = []
    for lab, ser in ((tr("level", "المستوى"), S), ("Δ", Sd)):
        for t in ("levinlin", "ips", "madwu", "Pm", "invnormal", "hadri"):
            try:
                r = pn.unit_root(ser, t, exo, lagm, int(pmax))
                rows.append({"Series": lab, "Test": r["name"], "H0": tr("unit root", "جذر وحدة") if t != "hadri" else tr("stationary", "سكون"),
                             "Statistic": r["stat"], "p-value": r["p"]})
            except Exception as e:  # noqa: BLE001
                rows.append({"Series": lab, "Test": t, "H0": "", "Statistic": np.nan, "p-value": np.nan})
                del e
        try:
            cp = pn.cips(ser, 1, "trend" if exo == "trend" else "drift")
            rows.append({"Series": lab, "Test": "CIPS (Pesaran 2007)", "H0": tr("unit root", "جذر وحدة"), "Statistic": cp["stat"],
                         "p-value": cp["p"], "5% cv": cp["cv"]["5%"]})
        except Exception:  # noqa: BLE001
            pass
    tab = pd.DataFrame(rows)
    _table(tab)
    st.caption(tr("Fisher / Choi p-values use the MacKinnon (1994) approximation (identical to R plm with p.approx = 'MacKinnon1994'). "
                  "Prefer CIPS when the CD test shows cross-sectional dependence.",
                  "قيم فيشر وتشوي الاحتمالية بتقريب ماكينون (1994) (مطابقة لحزمة plm بالخيار MacKinnon1994). وفضّل CIPS عند وجود اعتماد مقطعي."))
    lev = tab[tab["Series"] == tab["Series"].iloc[0]]
    rej = int((lev[lev["Test"].str.contains("Levin|Im-|Maddala|Choi")]["p-value"].astype(float) < 0.05).sum())
    insight(tr(f"In levels, {rej} of 5 first-generation tests reject the unit root.", f"في المستوى، رفضت {rej} من 5 اختبارات من الجيل الأول جذر الوحدة."))
    st.markdown("#### " + tr("Panel cointegration", "التكامل المشترك للبانل"))
    if not P.balanced:
        st.info(tr("Kao, Pedroni and Westerlund need a balanced panel.", "تحتاج اختبارات كاو وبيدروني وويسترلوند إلى بانل متوازن."))
    elif st.button(tr("▶️ Kao · Pedroni · Westerlund", "▶️ كاو · بيدروني · ويسترلوند"), key="pco_run"):
        with st.spinner(tr("Simulating the null distributions…", "جارٍ محاكاة توزيعات فرض العدم…")):
            k_ = pa.kao(P)
            pdn = pa.pedroni(P, exo == "trend")
            ws = pa.westerlund(P, 1, 1, exo == "trend")
        _store("pco", (k_, pdn, ws), P)
    co = _load("pco", P)
    if co is not None:
        k_, pdn, ws = co
        st.markdown("**Kao (1999)**"); _table(k_)
        st.markdown("**Pedroni (1999, 2004)**"); _table(pdn)
        st.markdown("**Westerlund (2007)**"); _table(ws)
        st.caption(tr("Pedroni and Westerlund: standardisation and p-values from Monte Carlo simulation of the null (no "
                      "cointegration) for this N, T and number of regressors.",
                      "بيدروني وويسترلوند: التوحيد المعياري والقيم الاحتمالية من محاكاة مونت كارلو لفرض العدم (لا تكامل) لقيم N وT وعدد المتغيرات الحالية."))
        nrej = int((pd.concat([k_, pdn, ws])["p-value"] < 0.05).sum())
        insight(tr(f"{nrej} of {len(k_) + len(pdn) + len(ws)} statistics reject 'no cointegration' at 5%.",
                   f"رفضت {nrej} من {len(k_) + len(pdn) + len(ws)} إحصاءة فرض «لا تكامل مشترك» عند 5%."))
    _method("unitroot", "pnu")
    _code(pr.code("unitroot", unit=P.entity, time=P.time, y=P.y, x=P.x, var=v), "pnu_code")


def _gmm(P, d, ent, tvar):
    lang = _lang()
    st.caption("💡 " + tr("Dynamic panels with many units and few periods (N large, T small). Endogenous regressors are "
                         "instrumented with lags ≥ 2, predetermined ones with lags ≥ 1, exogenous ones by themselves.",
                         "للبانل الديناميكي بوحدات كثيرة وفترات قليلة. تُستخدم للمتغيرات الداخلية إبطاءات ≥ 2، وللمحددة مسبقاً إبطاءات ≥ 1، "
                         "وللخارجية نفسها كأدوات."))
    roles = {}
    cols = st.columns(min(4, len(P.x)))
    for i, v in enumerate(P.x):
        roles[v] = cols[i % len(cols)].selectbox(v, ["endo", "pre", "exo"], key=f"gmm_role_{v}",
                                                 format_func=lambda r: {"endo": tr("endogenous", "داخلي"), "pre": tr("predetermined", "محدد مسبقاً"),
                                                                        "exo": tr("exogenous", "خارجي")}[r])
    c1, c2, c3, c4, c5 = st.columns(5)
    tr_ = c1.selectbox(tr("Estimator", "المقدّر"), ["d", "ld"], key="gmm_tr",
                       format_func=lambda t: tr("Difference GMM (Arellano-Bond)", "GMM الفروق (أريلانو-بوند)") if t == "d" else tr("System GMM (Blundell-Bond)", "GMM النظامي (بلونديل-بوند)"))
    mo = c2.selectbox(tr("Steps", "الخطوات"), ["twosteps", "onestep"], key="gmm_mo", format_func=lambda m: tr("two-step (Windmeijer)", "خطوتان (ويندماير)") if m == "twosteps" else tr("one-step", "خطوة واحدة"))
    ef = c3.selectbox(tr("Effects", "الآثار"), ["twoways", "individual"], key="gmm_ef", format_func=lambda e: tr("unit + time", "الوحدة + الزمن") if e == "twoways" else tr("unit", "الوحدة"))
    co = c4.checkbox(tr("Collapse instruments", "تجميع الأدوات"), True, key="gmm_co")
    ny = c5.selectbox(tr("Lags of y", "إبطاءات y"), [1, 2], key="gmm_ny")
    if st.button(tr("▶️ Estimate GMM", "▶️ تقدير GMM"), type="primary", key="gmm_run"):
        gmm_ins = {P.y: (2, 99)}
        for v, r_ in roles.items():
            if r_ == "endo":
                gmm_ins[v] = (2, 99)
            elif r_ == "pre":
                gmm_ins[v] = (1, 99)
        spec = pg.GmmSpec(P.y, list(range(1, ny + 1)), {v: [0] for v in P.x}, gmm_ins,
                          [v for v, r_ in roles.items() if r_ == "exo"])
        dd = P.df.rename(columns={"_unit": "__unit"})
        dd = dd[["__unit", P.time, P.y] + P.x]
        try:
            with st.spinner(tr("Estimating…", "جارٍ التقدير…")):
                _store("gmm", pg.fit(dd, "__unit", P.time, spec, mo, tr_, ef, co), P)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    r = _load("gmm", P)
    if r is None:
        return
    st.subheader(("Arellano-Bond" if r.transformation == "d" else "Blundell-Bond") + f" — {r.model}")
    _table(r.table())
    t = r.tests
    rows = [{"Test": k, "Statistic": v[0], "df": "" if v[1] is None else str(v[1]), "p-value": v[2]} for k, v in t.items()]
    _table(pd.DataFrame(rows))
    m = st.columns(3)
    m[0].metric(tr("Units", "الوحدات"), r.n_units); m[1].metric(tr("Instruments", "الأدوات"), r.n_instr); m[2].metric(tr("Observations", "المشاهدات"), r.n_obs)
    insight(pr.interpret_gmm(r, lang))
    ols, fe = pn.fit(_lag_pdata(P, len(r.spec.ylags)), "pooled"), pn.fit(_lag_pdata(P, len(r.spec.ylags)), "fe")
    rho = float(r.coef[0])
    lo, hi = float(fe.b[0]), float(ols.b[1])
    insight(tr(f"Bond (2002) check: the GMM autoregressive coefficient ({rho:.3f}) should lie between FE ({lo:.3f}, biased down) and pooled OLS ({hi:.3f}, biased up) — "
               + ("✓ it does." if min(lo, hi) <= rho <= max(lo, hi) else "⚠️ it does not: suspect weak or invalid instruments."),
               f"فحص بوند (2002): يجب أن يقع معامل الانحدار الذاتي في GMM ({rho:.3f}) بين FE ({lo:.3f}، متحيز لأسفل) والتجميعي ({hi:.3f}، متحيز لأعلى) — "
               + ("✓ وهو كذلك." if min(lo, hi) <= rho <= max(lo, hi) else "⚠️ ليس كذلك: يُشتبه في ضعف الأدوات أو عدم صلاحيتها.")))
    lr = pg.long_run(r)
    st.markdown("**" + tr("Long-run effects β / (1 − ρ)", "الآثار طويلة الأجل β / (1 − ρ)") + "**")
    _table(lr)
    _method("gmm", "png", transformation=r.transformation, model=r.model)
    excel_download({"GMM": r.table(), "Tests": pd.DataFrame(rows), "Long run": lr}, "panel_gmm.xlsx", key="gmm_xl")
    _code(pr.code("gmm", unit=P.entity, time=P.time, y=P.y, x=P.x, transformation=r.transformation, model=r.model, effect=r.effect,
                  collapse=r.collapse, endo=[v for v, k in roles.items() if k == "endo"], pre=[v for v, k in roles.items() if k == "pre"],
                  exo=[v for v, k in roles.items() if k == "exo"]), "png_code")


def _lag_pdata(P, ny):
    d = P.df.copy()
    for j in range(1, ny + 1):
        d[f"L{j}.{P.y}"] = d.groupby("_id")[P.y].shift(j)
    d = d.dropna()
    return pn.PanelData(d.reset_index(drop=True), P.y, [f"L{j}.{P.y}" for j in range(1, ny + 1)] + P.x, P.units, P.times, P.entity, P.time)


def _pardl(P):
    lang = _lang()
    st.caption("💡 " + tr("For panels with long T (≈ 20+). PMG: common long run, heterogeneous short run; MG: everything "
                         "heterogeneous; DFE: everything common except intercepts.",
                         "للبانل ذي الفترات الطويلة (≈ 20 فأكثر). PMG: أجل طويل مشترك وأجل قصير مختلف؛ MG: كل شيء مختلف؛ DFE: كل شيء "
                         "مشترك عدا الثوابت."))
    c1, c2 = st.columns(2)
    p = c1.number_input(tr("Lags of y (p)", "إبطاءات y (p)"), 1, 4, 1, key="pardl_p")
    q = c2.number_input(tr("Lags of x (q)", "إبطاءات x (q)"), 1, 4, 1, key="pardl_q")
    try:
        r = pa.panel_ardl(P, int(p), int(q))
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return
    c = st.columns(3)
    for col, (lab, k_) in zip(c, [("PMG", "pmg"), ("MG", "mg"), ("DFE", "dfe")]):
        with col:
            st.markdown(f"**{lab} — " + tr("long run", "الأجل الطويل") + "**")
            _table(r[k_]["long_run"])
            st.markdown("**" + tr("short run", "الأجل القصير") + "**")
            _table(r[k_]["short_run"])
    sr = r["dfe"]["short_run"].set_index("Variable")["Coef."]
    phi_dfe = float(sr.iloc[0])
    if phi_dfe > -0.02:
        st.warning(tr(f"DFE adjustment coefficient φ = {phi_dfe:.4f} is (close to) zero: there is no error correction, so long-run "
                      "estimates are unreliable. Check the specification (logs, choice of dependent variable) and the cointegration tests.",
                      f"معامل التعديل في DFE φ = {phi_dfe:.4f} صفر أو قريب منه: لا يوجد تصحيح للخطأ، فتقديرات الأجل الطويل غير موثوقة. "
                      "راجع توصيف النموذج (اللوغاريتمات واختيار المتغير التابع) واختبارات التكامل المشترك."))
    h1, h2 = r["hausman_mg_pmg"], r["hausman_pmg_dfe"]
    _table(pd.DataFrame([{"Test": "Hausman MG vs PMG", "Statistic": h1["stat"], "df": str(h1["df"]), "p-value": h1["p"]},
                         {"Test": "Hausman PMG vs DFE", "Statistic": h2["stat"], "df": str(h2["df"]), "p-value": h2["p"]}]))
    insight(tr("Hausman MG vs PMG: " + ("long-run homogeneity not rejected → <b>PMG</b> is consistent and efficient." if h1["p"] >= 0.05 else "long-run homogeneity rejected → use <b>MG</b>."),
               "هاوسمان MG مقابل PMG: " + ("لا يُرفض تجانس الأجل الطويل ← <b>PMG</b> متسق وأكفأ." if h1["p"] >= 0.05 else "يُرفض تجانس الأجل الطويل ← استخدم <b>MG</b>.")))
    phi = r["pmg"]["phi_i"]
    fig = go.Figure(go.Bar(x=[P.units[i] for i in range(len(phi))], y=phi, marker_color=[PAL[0] if v < 0 else PAL[1] for v in phi]))
    ch.style_academic(fig, height=360, y_title=tr("Adjustment speed φᵢ", "سرعة التعديل φᵢ"), legend="none", rtl=is_ar())
    _chart(fig, "pardl_phi", "pmg_adjustment_speeds")
    bad = int((phi >= 0).sum())
    insight(tr(f"Average φ = {phi.mean():.3f}: about {abs(phi.mean()) * 100:.0f}% of a deviation from the long-run equilibrium is closed each period. "
               + (f"⚠️ {bad} unit(s) show no error correction (φ ≥ 0)." if bad else "All units error-correct (φ < 0)."),
               f"متوسط φ = {phi.mean():.3f}: يُصحَّح نحو {abs(phi.mean()) * 100:.0f}% من الانحراف عن التوازن طويل الأجل في كل فترة. "
               + (f"⚠️ {bad} وحدة لا تُظهر تصحيحاً للخطأ (φ ≥ 0)." if bad else "كل الوحدات تصحح الخطأ (φ < 0).")))
    _method("pardl", "pnp")
    excel_download({"PMG long run": r["pmg"]["long_run"], "PMG short run": r["pmg"]["short_run"], "MG long run": r["mg"]["long_run"],
                    "DFE long run": r["dfe"]["long_run"]}, "panel_ardl.xlsx", key="pardl_xl")
    _code(pr.code("pardl", unit=P.entity, time=P.time, y=P.y, x=P.x, p=int(p), q=int(q)), "pnp_code")
    del lang


def _advanced(P):
    st.markdown("#### " + tr("Mean group and common correlated effects", "المتوسط الجماعي والآثار المرتبطة الشائعة"))
    try:
        mg, cmg, ccp = pn.mean_group(P), pn.mean_group(P, cce=True), pn.cce_pooled(P)
        c = st.columns(3)
        for col, (lab, t_) in zip(c, [("MG (Pesaran-Smith)", mg["table"]), ("CCE-MG", cmg["table"]), ("CCE pooled", ccp["table"])]):
            with col:
                st.markdown(f"**{lab}**")
                _table(t_)
    except Exception as e:  # noqa: BLE001
        st.info(str(e))
    st.markdown("#### " + tr("Interactive fixed effects (Bai 2009)", "الآثار الثابتة التفاعلية (باي 2009)"))
    if not P.balanced:
        st.info(tr("Needs a balanced panel.", "يحتاج بانلاً متوازناً."))
    else:
        r_ = st.number_input(tr("Number of factors r", "عدد العوامل r"), 1, 6, 1, key="ife_r")
        try:
            ife = pa.interactive_fe(P, int(r_))
            c1, c2 = st.columns([1.2, 1])
            with c1:
                _table(ife["table"])
            with c2:
                _table(ife["ic"])
                best = int(ife["ic"].loc[ife["ic"]["IC_p2"].idxmin(), "Factors"])
                st.caption(tr(f"Bai-Ng IC_p2 suggests r = {best}.", f"يقترح معيار باي-نج IC_p2 عدد عوامل r = {best}."))
        except Exception as e:  # noqa: BLE001
            st.info(str(e))
    st.markdown("#### " + tr("Panel threshold regression (Hansen 1999)", "انحدار العتبة للبانل (هانسن 1999)"))
    c1, c2, c3 = st.columns(3)
    rv = c1.multiselect(tr("Regime-dependent regressors", "متغيرات تتغير بحسب النظام"), P.x, P.x[:1], key="thr_rv_" + "_".join(P.x))
    qv = c2.selectbox(tr("Threshold variable q", "متغير العتبة q"), P.x, index=len(P.x) - 1, key="thr_q_" + "_".join(P.x))
    bs = c3.select_slider(tr("Bootstrap replications", "تكرارات البوتستراب"), [100, 200, 300, 500], 200, key="thr_bs")
    if rv and st.button(tr("▶️ Estimate threshold model", "▶️ تقدير نموذج العتبة"), key="thr_run"):
        try:
            with st.spinner(tr("Grid search and bootstrap…", "جارٍ البحث الشبكي والبوتستراب…")):
                _store("thr", pa.panel_threshold(P, rv, qv, boot=int(bs)), P)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    th = _load("thr", P)
    if th is not None:
        _table(th["table"])
        insight(tr(f"Threshold γ̂ = {th['gamma']:.4g} (95% CI {th['ci'][0]:.4g} – {th['ci'][1]:.4g}); {th['share_low'] * 100:.0f}% of observations are in the low regime. "
                   f"F = {th['F']:.2f}, bootstrap p = {th['p_boot']:.3f} — " + ("the effect changes significantly at the threshold." if th["p_boot"] < 0.05 else "no significant threshold."),
                   f"العتبة γ̂ = {th['gamma']:.4g} (فترة الثقة 95%: {th['ci'][0]:.4g} – {th['ci'][1]:.4g})؛ {th['share_low'] * 100:.0f}% من المشاهدات في النظام المنخفض. "
                   f"F = {th['F']:.2f}، وقيمة p بالبوتستراب = {th['p_boot']:.3f} — " + ("يتغير الأثر معنوياً عند العتبة." if th["p_boot"] < 0.05 else "لا عتبة معنوية.")))
        lrp = th["LR"]
        fig = go.Figure([go.Scatter(x=lrp["gamma"], y=lrp["LR"], mode="lines", line=dict(color=PAL[0]))])
        fig.add_hline(y=th["c95"], line=dict(color=PAL[1], dash="dash"))
        ch.style_academic(fig, height=320, x_title="γ", y_title="LR(γ)", legend="none", rtl=is_ar())
        _chart(fig, "thr_lr", "threshold_lr")
    st.markdown("#### " + tr("Panel VAR (GMM, forward orthogonal deviations)", "VAR للبانل (GMM، الانحرافات الأمامية المتعامدة)"))
    c1, c2, c3 = st.columns(3)
    vv = c1.multiselect(tr("Variables", "المتغيرات"), [P.y] + P.x, [P.y] + P.x[:1], key="pvar_v_" + "_".join([P.y] + P.x))
    pl = c2.number_input("p", 1, 3, 1, key="pvar_p")
    il = c3.number_input(tr("Instrument lags", "إبطاءات الأدوات"), int(pl), 4, int(pl) + 1, key="pvar_il")
    if len(vv) >= 2 and st.button(tr("▶️ Estimate panel VAR", "▶️ تقدير VAR للبانل"), key="pvar_run"):
        try:
            _store("pvar", pa.panel_var(P, vv, int(pl), int(il)), P)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    pv = _load("pvar", P)
    if pv is not None:
        _table(pv["table"])
        st.caption(tr(f"Largest companion root = {pv['roots'].max():.3f} " + ("(stable)." if pv["roots"].max() < 1 else "(not stable)."),
                      f"أكبر جذر = {pv['roots'].max():.3f} " + ("(مستقر)." if pv["roots"].max() < 1 else "(غير مستقر).")))
        _table(pv["granger"])
        base, lo, hi = pa.pvar_irf(pv, 10, 200)
        from analytics.mts_ui import _irf_grid
        _irf_grid(base, lo, hi, pv["names"], lambda a, b: f"{a} → {b}", "pvar_irf", "panel_var_irf")
    _method("advanced", "pna")
    _code(pr.code("advanced", unit=P.entity, time=P.time, y=P.y, x=P.x), "pna_code")


def _causality(P):
    c1, c2 = st.columns(2)
    order = c1.number_input(tr("Lag order", "رتبة الإبطاء"), 1, 4, 1, key="dh_o")
    vars_ = [P.y] + P.x
    rows = []
    for a in vars_:
        for b in vars_:
            if a == b:
                continue
            try:
                r = pn.dumitrescu_hurlin(P, a, b, int(order))
                rows.append({"Cause → Effect": f"{a} → {b}", "W̄": r["Wbar"], "Z̄": r["Zbar"], "p (Z̄)": r["p_Zbar"],
                             "Z̃": r["Ztilde"], "p-value (Z̃)": r["p_Ztilde"]})
            except Exception:  # noqa: BLE001
                pass
    tab = pd.DataFrame(rows)
    show = tab.copy()
    for c in ("p (Z̄)", "p-value (Z̃)"):
        show[c] = show[c].map(fp)
    st.dataframe(show.round(3), width="stretch", hide_index=True)
    pz = tab["p-value (Z̃)"].where(tab["p-value (Z̃)"].notna(), tab["p (Z̄)"])
    sig = tab[pz < 0.05]["Cause → Effect"].tolist()
    insight(tr("Dumitrescu-Hurlin (H0: no Granger causality in any unit): significant at 5% for " + (", ".join(sig) if sig else "no pair") +
               ". Z̃ is the small-T corrected statistic.",
               "دوميترسكو-هورلن (فرض العدم: لا سببية في أي وحدة): معنوية عند 5% لـ " + ("، ".join(sig) if sig else "لا يوجد زوج") + ". وZ̃ هي الإحصاءة المصححة لصغر T."))
    del c2
