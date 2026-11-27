# ==========================================
# Aetheris Research - Multivariate time series, ARDL and volatility tabs (UI) — v1.7, Catalogue §1.8–1.9
# ==========================================

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics import ardl as ad
from analytics import charts as ch
from analytics import descriptive as ds
from analytics import diagnostics as dx
from analytics import garch as gm
from analytics import mts_report as mr
from analytics import reg_charts as rc
from analytics import reg_report as rp
from analytics import regression as rg
from analytics import ts
from analytics import ts_ui as tu
from analytics import var as vr
from core.memory import active_name
from core.utils import tr, is_ar, insight, excel_download

PAL = ch.PALETTES["academic"]
tu._AR_COLS.update({
    "Equation": "المعادلة", "Regressor": "المتغير المفسر", "t": "t", "Variable": "المتغير", "Order": "درجة التكامل",
    "Dependent": "المتغير التابع", "Excluded": "المتغير المستبعد", "Chi-sq": "كاي²", "df": "د.ح", "LogL": "لوغاريتم الإمكان",
    "Lag": "الإبطاء", "H0: rank ≤": "فرض العدم: الرتبة ≤", "Eigenvalue": "القيمة الذاتية", "Trace": "الأثر",
    "Trace 5% cv": "القيمة الحرجة 5% (الأثر)", "Trace p-value": "p (الأثر)", "Max-eigen": "القيمة الذاتية العظمى",
    "Max 5% cv": "القيمة الحرجة 5% (العظمى)", "Max p-value": "p (العظمى)", "Relation": "العلاقة", "β": "β", "LR": "LR",
    "Break": "تاريخ الانكسار", "1% cv": "ق.ح 1%", "5% cv": "ق.ح 5%", "10% cv": "ق.ح 10%", "Long-run coef.": "معامل الأجل الطويل",
    "Significance": "مستوى المعنوية", "τ": "τ", "Long-run β(τ)": "معامل الأجل الطويل β(τ)", "θ⁺ / Σω⁺": "θ⁺ / Σω⁺",
    "θ⁻ / Σω⁻": "θ⁻ / Σω⁻", "Wald F": "والد F", "R²": "R²", "Adj. R²": "R² المعدل", "S.E. of regression": "الخطأ المعياري للانحدار",
    "F": "F", "p(F)": "p(F)", "Modulus": "المقياس", "Model": "النموذج", "AIC": "AIC", "BIC": "BIC", "z": "z",
    "variance": "التباين", "volatility": "التقلب", "unconditional": "التباين غير الشرطي", "Estimate": "التقدير"})
tu._AR_VALS.update({"no residual autocorrelation": "لا ارتباط ذاتي في البواقي", "normal residuals": "بواقٍ طبيعية",
                    "no ARCH effects": "لا آثار ARCH", "no autocorrelation in the mean": "لا ارتباط ذاتي في المتوسط",
                    "no remaining volatility clustering": "لا تجمع متبقٍ للتقلب", "no remaining ARCH": "لا ARCH متبقٍ",
                    "normal standardised residuals": "بواقٍ معيارية طبيعية", "no asymmetry missed by the model": "لا عدم تماثل مُهمَل",
                    "ALL": "الكل", "cointegration": "تكامل مشترك", "no cointegration": "لا تكامل", "inconclusive": "غير حاسم"})
_disp, _chart, _fp, _code, _lang = tu._disp, tu._chart, tu._fp, tu._code, tu._lang
TRANSF = {"level": ("Level", "المستوى"), "log": ("ln", "ln"), "diff": ("Δ", "Δ"), "dlog": ("Δln ×100 (growth %)", "Δln ×100 (نمو %)")}


def _t(pair):
    return pair[1] if is_ar() else pair[0]


def _method(kind, key, **kw):
    txt, refs = mr.methodology(kind, _lang(), **kw)
    with st.expander(tr("📝 Methodology paragraph & references", "📝 فقرة المنهجية والمراجع"), expanded=False):
        st.text_area("m", txt, height=160, key=f"{key}_meth", label_visibility="collapsed")
        st.markdown("\n".join(f"- {x}" for x in refs))
    return txt, refs


def _sig(Y) -> tuple:
    return (active_name(), tuple(Y.columns), len(Y), float(np.nansum(Y.to_numpy())))


def _store(key, value, Y):
    st.session_state[f"_v17res_{key}"] = (_sig(Y), value)


def _load(key, Y):
    v = st.session_state.get(f"_v17res_{key}")
    return v[1] if isinstance(v, tuple) and v[0] == _sig(Y) else None


# ==========================================================
# Data selection (several series, per-series transformation)
# ==========================================================
def _multi(df, key: str, min_vars: int = 2, default=None):
    tcols = tu._time_cols(df) + [c for c in ds.time_columns(df) if c not in tu._time_cols(df)]
    if not tcols:
        st.info(tr("No date / year column found. Time-series tools need a time variable.",
                   "لا يوجد عمود للتاريخ أو السنة. تحتاج أدوات السلاسل الزمنية إلى متغير زمني."))
        return None
    num = [c for c in ds.numeric_columns(df) if c not in tcols]
    c1, c2 = st.columns([1, 3])
    with c1:
        t = st.selectbox(tr("Time variable", "المتغير الزمني"), tcols, key=f"{key}_t")
    with c2:
        dflt = [v for v in (default or []) if v in num] or num[:min_vars]
        vs = st.multiselect(tr("Series", "السلاسل"), num, default=dflt, key=f"{key}_vars", placeholder=tr("Choose series…", "اختر السلاسل…"))
    if len(vs) < min_vars:
        st.info(tr(f"Select at least {min_vars} series.", f"اختر {min_vars} سلاسل على الأقل."))
        return None
    cols = st.columns(min(4, len(vs)))
    trans = {}
    for i, v in enumerate(vs):
        with cols[i % len(cols)]:
            trans[v] = st.selectbox(v, list(TRANSF), key=f"{key}_tr_{v}", format_func=lambda k: _t(TRANSF[k]))
    d = df.copy()
    cats = [c for c in ds.categorical_columns(df) if c != t]
    if cats and d[t].duplicated().any():
        with st.expander(tr("Select one unit from a panel (filters)", "اختيار وحدة واحدة من بيانات بانل (مرشحات)"), expanded=True):
            fc = st.columns(min(3, len(cats)))
            for i, c in enumerate(cats):
                with fc[i % len(fc)]:
                    val = st.selectbox(c, ["—"] + sorted(d[c].dropna().astype(str).unique().tolist()), key=f"{key}_f_{c}")
                    if val != "—":
                        d = d[d[c].astype(str) == val]
    tt = d[t]
    idx = tt.astype(int) if pd.api.types.is_numeric_dtype(tt) and tt.dropna().between(1000, 3000).all() else pd.to_datetime(tt, errors="coerce")
    raw = d[vs].apply(pd.to_numeric, errors="coerce")
    raw.index = idx
    raw = raw[raw.index.notna()].groupby(level=0).mean().sort_index()
    out = {}
    for v in vs:
        x = raw[v]
        k = trans[v]
        if k in ("log", "dlog") and (x <= 0).any():
            st.error(tr(f"{v}: the log needs positive values.", f"{v}: اللوغاريتم يحتاج قيماً موجبة."))
            return None
        if k == "level":
            out[v] = x
        elif k == "log":
            out[f"ln_{v}"] = np.log(x)
        elif k == "diff":
            out[f"d_{v}"] = x.diff()
        else:
            out[f"dln_{v}"] = 100 * np.log(x).diff()
    Y = pd.DataFrame(out).dropna()
    if len(Y) < 20:
        st.error(tr("At least 20 complete observations are needed.", "يلزم 20 مشاهدة مكتملة على الأقل."))
        return None
    s = ts.detect_period(Y.index) if isinstance(Y.index, pd.DatetimeIndex) else 1
    a, b = tu._tlabel(Y.index[0]), tu._tlabel(Y.index[-1])
    st.caption(tr(f"{len(Y)} observations · {Y.shape[1]} series · period s = {s} · {a} → {b}",
                  f"{len(Y)} مشاهدة · {Y.shape[1]} سلاسل · طول الموسم s = {s} · من ⁦{a}⁩ إلى ⁦{b}⁩"))
    return Y, t, s, trans


# ==========================================================
# VAR / SVAR / cointegration / VECM
# ==========================================================
def render_mts(df):
    st.caption("💡 " + tr("Workflow: (1) test the integration order of every series, (2) all I(0) → VAR in levels; all I(1) → "
                         "Johansen: cointegrated → VECM, otherwise VAR in differences; a mix → ARDL tab.",
                         "خطوات العمل: (1) اختبر درجة تكامل كل سلسلة، (2) كلها I(0) ← VAR في المستويات؛ كلها I(1) ← جوهانسن: "
                         "إن وُجد تكامل مشترك ← VECM وإلا ← VAR في الفروق؛ خليط ← تبويب ARDL."))
    out = _multi(df, "mv", 2)
    if out is None:
        return
    Y, tcol, s, trans = out
    tabs = st.tabs([tr("🧭 Pre-tests & model choice", "🧭 الاختبارات القبلية واختيار النموذج"), tr("📊 VAR", "📊 VAR"),
                    tr("🏛️ Structural VAR", "🏛️ VAR الهيكلي"), tr("🔗 Cointegration", "🔗 التكامل المشترك"),
                    tr("⚖️ VECM", "⚖️ VECM")])
    with tabs[0]:
        _pretests(Y)
    with tabs[1]:
        _var_tab(Y, s, trans)
    with tabs[2]:
        _svar_tab(Y)
    with tabs[3]:
        _coint_tab(Y)
    with tabs[4]:
        _vecm_tab(Y)


def _pretests(Y):
    rows, orders = [], {}
    for c in Y.columns:
        io = ts.integration_order(Y[c].to_numpy(), "ct")
        orders[c] = io["order"]
        lvl = io["table"][(io["table"]["Series"] == "level") & (io["table"]["Test"] == "ADF")]
        kp = io["table"][(io["table"]["Series"] == "level") & (io["table"]["Test"] == "KPSS")]
        rows.append({"Variable": c, "Order": "—" if io["order"] is None else f"I({io['order']})",
                     "ADF p (level)": float(lvl["p-value"].iloc[0]), "KPSS (level)": float(kp["Statistic"].iloc[0]),
                     "Verdict": io["verdict_last"]})
    tab = pd.DataFrame(rows)
    show = tab.copy()
    show["ADF p (level)"] = show["ADF p (level)"].map(_fp)
    st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
    st.caption(tr("Strategy of the Unit-roots tab (ADF, PP, DF-GLS vs KPSS; level with constant + trend, then differences).",
                  "استراتيجية تبويب جذور الوحدة (ADF وPP وDF-GLS مقابل KPSS؛ المستوى بثابت واتجاه ثم الفروق)."))
    insight(mr.model_advice(orders, _lang()))
    fig = make_subplots(rows=len(Y.columns), cols=1, shared_xaxes=True, vertical_spacing=0.03, subplot_titles=list(Y.columns))
    for i, c in enumerate(Y.columns, 1):
        fig.add_trace(go.Scatter(x=Y.index, y=Y[c], mode="lines", line=dict(color=PAL[(i - 1) % len(PAL)], width=1.6)), i, 1)
    ch.style_academic(fig, height=170 * len(Y.columns) + 80, legend="none", rtl=is_ar())
    _chart(fig, "mv_series", "series")


def _eq_results(r: vr.VarResult) -> list:
    res = []
    tq = r.table()
    for k, eq in enumerate(r.names):
        sub = tq[tq["Equation"] == eq]
        names = sub["Regressor"].tolist()
        rr = rg.RegResult(name=eq, method="var", cov_type="nonrobust", design=None,
                          params=pd.Series(sub["Coef."].to_numpy(), index=names), bse=pd.Series(sub["Std. Err."].to_numpy(), index=names),
                          tvalues=pd.Series(sub["t"].to_numpy(), index=names), pvalues=pd.Series(sub["p-value"].to_numpy(), index=names),
                          ci=pd.DataFrame(sub[["CI low", "CI high"]].to_numpy(), index=names, columns=["low", "high"]),
                          cov=pd.DataFrame(np.diag(sub["Std. Err."].to_numpy() ** 2), index=names, columns=names), dist="t",
                          df_resid=r.df, nobs=r.obs, y=np.zeros(1), fitted=np.zeros(1), resid=np.zeros(1), wresid=np.zeros(1),
                          wexog=np.zeros((1, 1)))
        es = r.stats["equations"].iloc[k]
        rr.rsquared, rr.rsquared_adj = float(es["R²"]), float(es["Adj. R²"])
        rr.fvalue, rr.f_pvalue = float(es["F"]), float(es["p(F)"])
        rr.extra.update({"y_label": eq, "family": "var"})
        res.append(rr)
    return res


def _irf_grid(I, lo, hi, names, title_fmt, key, name):
    K = len(names)
    fig = make_subplots(rows=K, cols=K, subplot_titles=[title_fmt(names[j], names[i]) for i in range(K) for j in range(K)],
                        vertical_spacing=0.08, horizontal_spacing=0.06)
    h = np.arange(I.shape[0])
    for i in range(K):
        for j in range(K):
            if lo is not None:
                fig.add_trace(go.Scatter(x=np.r_[h, h[::-1]], y=np.r_[hi[:, i, j], lo[:, i, j][::-1]], fill="toself",
                                         fillcolor="rgba(31,78,121,0.15)", line=dict(width=0), hoverinfo="skip"), i + 1, j + 1)
            fig.add_trace(go.Scatter(x=h, y=I[:, i, j], mode="lines", line=dict(color=PAL[0], width=2)), i + 1, j + 1)
            fig.add_hline(y=0, line=dict(color="#999", width=1), row=i + 1, col=j + 1)
    ch.style_academic(fig, height=240 * K + 60, legend="none", rtl=is_ar())
    fig.update_annotations(font_size=11)
    _chart(fig, key, name)


def _var_tab(Y, s, trans):
    lang = _lang()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        typ = st.selectbox(tr("Deterministic terms", "الحدود المحددة"), list(vr.VTYPES), key="var_type",
                           format_func=lambda k: _t(vr.VTYPES[k]))
    with c2:
        lag_max = st.number_input(tr("Max lag for selection", "أقصى إبطاء للاختيار"), 1, 16,
                                  int(min(8, max(2, len(Y) // 12))), key="var_lmax")
    with c3:
        season = st.checkbox(tr(f"Seasonal dummies (s = {s})", f"متغيرات موسمية (s = {s})"), False, key="var_seas",
                             disabled=s < 2) if s > 1 else False
    sel = vr.var_select(Y, int(lag_max), typ, s if season else None)
    show = sel.copy()
    for c in ("AIC", "HQ", "SC", "FPE"):
        best = int(sel.attrs["selection"][c])
        show[c] = [f"{v:.4f}{' *' if int(l) == best else ''}" for v, l in zip(sel[c], sel["Lag"])]
    if "p(LR)" in show:
        show["p(LR)"] = show["p(LR)"].map(_fp)
    with st.expander(tr("Lag-order selection (* = selected)", "اختيار فترة الإبطاء (* = المختار)"), expanded=True):
        st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
    with c4:
        p = st.number_input(tr("Lag order p", "فترة الإبطاء p"), 1, 16, max(1, int(sel.attrs["selection"]["AIC"])), key="var_p")
    if st.button(tr("▶️ Estimate VAR", "▶️ تقدير VAR"), type="primary", key="var_run"):
        try:
            r = vr.fit_var(Y, int(p), typ, s if season else None, names=list(Y.columns), index=Y.index)
            _store("mv_var", r, Y)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    r = _load("mv_var", Y)
    if r is None:
        return
    st.subheader(f"VAR({r.p}) — " + ", ".join(r.names))
    m = st.columns(4)
    for col, (k_, v_) in zip(m, [("N", r.obs), ("Log-likelihood", r.stats["Log-likelihood"]), ("AIC", r.stats["AIC"]), ("SC (BIC)", r.stats["SC (BIC)"])]):
        col.metric(k_, "‎" + (f"{v_:,.3f}" if isinstance(v_, float) else str(v_)))
    for n in r.notes:
        st.warning(n)
    st.dataframe(_disp(r.stats["equations"]).round(4), width="stretch", hide_index=True)
    eqs = _eq_results(r)
    jt, note = rp.journal_table(eqs, r.names, stats_rows=("N", "R²", "Adj. R²", "F"), lang=lang, const_last=True)
    with st.expander(tr("Journal-style table (columns = equations)", "جدول بنمط المجلات (الأعمدة = المعادلات)"), expanded=False):
        st.markdown(rp.to_html(jt, note, tr("Table. VAR estimates", "جدول. تقديرات VAR")), unsafe_allow_html=True)
    sub = st.tabs([tr("🩺 Stability & diagnostics", "🩺 الاستقرار والتشخيص"), tr("➡️ Granger causality", "➡️ سببية جرانجر"),
                   tr("📈 Impulse responses", "📈 دوال الاستجابة"), tr("🧩 Variance decomposition", "🧩 تجزئة التباين"),
                   tr("🔮 Forecast", "🔮 التنبؤ"), tr("📝 Methodology & export", "📝 المنهجية والتصدير")])
    with sub[0]:
        mod = vr.roots(r)
        c1, c2 = st.columns([1, 1.3])
        with c1:
            th = np.linspace(0, 2 * np.pi, 200)
            ev = np.linalg.eigvals(vr.companion(r))
            fig = go.Figure([go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(color="#999")),
                             go.Scatter(x=ev.real, y=ev.imag, mode="markers", marker=dict(size=9, color=PAL[0]))])
            fig.update_xaxes(range=[-1.2, 1.2]); fig.update_yaxes(range=[-1.2, 1.2], scaleanchor="x")
            ch.style_academic(fig, height=360, legend="none", rtl=is_ar(), title=tr("Companion roots", "جذور مصفوفة المرافقة"))
            _chart(fig, "var_roots", "var_roots")
        with c2:
            st.dataframe(pd.DataFrame({"Modulus": mod}).round(4).T, width="stretch")
            insight(tr(f"Largest root modulus = {mod.max():.4f} " + ("< 1: the VAR is stable." if mod.max() < 1 else "≥ 1: not stable."),
                       f"أكبر مقياس للجذور = {mod.max():.4f} " + ("< 1: النموذج مستقر." if mod.max() < 1 else "≥ 1: النموذج غير مستقر.")))
        dg = vr.diagnostics(r)
        show = dg.copy()
        show["p-value"] = show["p-value"].map(_fp)
        st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
        bad = dg[dg["p-value"] < 0.05]["Test"].tolist()
        insight(tr("All residual tests pass at 5%." if not bad else "Rejected at 5%: " + "; ".join(bad) +
                   ". Autocorrelation → add lags; non-normality / ARCH → use bootstrap bands and interpret tests cautiously.",
                   "اجتازت البواقي كل الاختبارات عند 5%." if not bad else "مرفوض عند 5%: " + "؛ ".join(bad) +
                   ". الارتباط الذاتي ← زد الإبطاءات؛ عدم الطبيعية أو ARCH ← استخدم فترات البوتستراب وفسّر بحذر."))
        nu = vr.normality(r)["univariate"]
        st.dataframe(_disp(nu).round(4), width="stretch", hide_index=True)
    with sub[1]:
        ty = st.checkbox(tr("Toda-Yamamoto (estimate VAR(p + d_max) in levels; valid for I(1) / cointegrated data)",
                            "تودا-ياماموتو (تقدير VAR(p + d_max) في المستويات؛ صالح للبيانات I(1) أو المتكاملة)"), key="var_ty")
        if ty:
            dmax = st.selectbox("d_max", [1, 2], key="var_dmax")
            gp, _ = vr.toda_yamamoto(Y, r.p, int(dmax), r.type, r.season, names=r.names)
        else:
            gp = vr.granger_pairs(r)
        show = gp.copy()
        show["p-value"] = show["p-value"].map(_fp)
        st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
        sig = gp[(gp["p-value"] < 0.05) & (gp["Excluded"] != "ALL")]
        if sig.empty:
            insight(tr("No significant Granger-causal link at 5%.", "لا توجد علاقة سببية معنوية بمفهوم جرانجر عند 5%."))
        else:
            insight(tr("Significant Granger causality (5%): ", "سببية جرانجر معنوية (5%): ") + "; ".join(
                f"{a} → {b} ({_fp(p_)})" for a, b, p_ in zip(sig["Excluded"], sig["Dependent"], sig["p-value"])) +
                tr(". Granger causality means predictive precedence, not structural causation.",
                   ". وسببية جرانجر تعني الأسبقية التنبؤية لا السببية الهيكلية."))
        cause = st.selectbox(tr("System test — cause variable(s)", "الاختبار على مستوى النظام — المتغير السببي"), r.names, key="var_cause")
        g = vr.granger(r, [cause])
        st.dataframe(_disp(pd.DataFrame([
            {"Test": f"Granger: {cause} ↛ {', '.join(g['effect'])}", "Statistic": g["F"], "df": f"{g['df1']}, {g['df2']}", "p-value": _fp(g["p"])},
            {"Test": tr("Instantaneous causality", "السببية اللحظية"), "Statistic": g["inst"], "df": str(g["inst_df"]), "p-value": _fp(g["inst_p"])}])).round(4),
            width="stretch", hide_index=True)
    with sub[2]:
        c1, c2, c3, c4 = st.columns(4)
        kind = c1.selectbox(tr("Identification", "طريقة التحديد"), ["orth", "generalized", "simple"], key="var_irfk",
                            format_func=lambda k: {"orth": tr("Orthogonalised (Cholesky)", "متعامدة (تشوليسكي)"),
                                                   "generalized": tr("Generalised (Pesaran-Shin)", "معممة (بيساران-شين)"),
                                                   "simple": tr("Forecast-error (non-orthogonal)", "خطأ التنبؤ (غير متعامدة)")}[k])
        h = c2.number_input(tr("Horizon", "الأفق"), 4, 60, 10, key="var_h")
        runs = c3.select_slider(tr("Bootstrap runs", "تكرارات البوتستراب"), [0, 100, 200, 500, 1000], 200, key="var_runs")
        cum = c4.checkbox(tr("Cumulative", "تراكمية"), key="var_cum")
        if kind == "orth":
            st.caption(tr("Cholesky ordering = the order of the series above (first = most exogenous). Use the generalised "
                          "responses for order-invariant results.", "ترتيب تشوليسكي = ترتيب السلاسل أعلاه (الأول = الأكثر خارجية). "
                          "استخدم الاستجابات المعممة لنتائج لا تتأثر بالترتيب."))
        I = vr.irf(r, int(h), kind, cum)
        lo = hi = None
        if runs:
            with st.spinner(tr("Bootstrapping…", "جارٍ البوتستراب…")):
                lo, hi = vr.irf_bootstrap(r, int(h), kind, cum, int(runs))
        _irf_grid(I, lo, hi, r.names, lambda a, b: f"{a} → {b}", "var_irf", "impulse_responses")
        flat = [{"step": k_, "impulse": r.names[j], "response": r.names[i], "irf": I[k_, i, j],
                 **({"lower": lo[k_, i, j], "upper": hi[k_, i, j]} if lo is not None else {})}
                for k_ in range(I.shape[0]) for i in range(r.K) for j in range(r.K)]
        st.session_state["var_irf_tab"] = pd.DataFrame(flat)
    with sub[3]:
        hh = st.number_input(tr("Horizon", "الأفق"), 2, 60, 10, key="var_fh")
        F = vr.fevd(r, int(hh))
        for i, nm in enumerate(r.names):
            fig = go.Figure([go.Bar(x=np.arange(1, F.shape[0] + 1), y=100 * F[:, i, j], name=r.names[j], marker_color=PAL[j % len(PAL)])
                             for j in range(r.K)])
            fig.update_layout(barmode="stack")
            ch.style_academic(fig, height=320, y_title="%", title=tr(f"FEVD of {nm}", f"تجزئة تباين {nm}"), rtl=is_ar())
            _chart(fig, f"var_fevd_{i}", f"fevd_{nm}")
        last = pd.DataFrame(100 * F[-1], index=r.names, columns=r.names).round(2)
        st.dataframe(last, width="stretch")
        own = {nm: last.loc[nm, nm] for nm in r.names}
        insight(tr(f"At horizon {int(hh)}, the share of each variable's forecast-error variance due to its own shocks: ",
                   f"عند الأفق {int(hh)}، نصيب الصدمات الذاتية من تباين خطأ التنبؤ لكل متغير: ") +
                ", ".join(f"{k_} {v_:.1f}%" for k_, v_ in own.items()))
    with sub[4]:
        hf = st.number_input(tr("Forecast horizon", "أفق التنبؤ"), 1, 48, 8, key="var_fch")
        if r.exog_names:
            st.info(tr("Exogenous regressors are held at their last value.", "تُثبَّت المتغيرات الخارجية عند آخر قيمة."))
        fc = vr.forecast(r, int(hf))
        vsel = st.selectbox(tr("Variable", "المتغير"), r.names, key="var_fcv")
        j = r.names.index(vsel)
        idx = Y.index
        try:
            fidx = pd.date_range(idx[-1], periods=int(hf) + 1, freq=pd.infer_freq(idx) or "MS")[1:] if isinstance(idx, pd.DatetimeIndex) else np.arange(1, int(hf) + 1) + idx[-1]
        except Exception:  # noqa: BLE001
            fidx = np.arange(1, int(hf) + 1)
        fig = go.Figure()
        hist = Y[vsel].iloc[-min(len(Y), 8 * int(hf)):]
        fig.add_trace(go.Scatter(x=hist.index, y=hist, mode="lines", name=tr("Observed", "المشاهدة"), line=dict(color="#555")))
        fig.add_trace(go.Scatter(x=np.r_[fidx, fidx[::-1]], y=np.r_[fc["upper"][:, j], fc["lower"][:, j][::-1]], fill="toself",
                                 fillcolor="rgba(31,78,121,0.18)", line=dict(width=0), name="95% CI"))
        fig.add_trace(go.Scatter(x=fidx, y=fc["mean"][:, j], mode="lines+markers", name=tr("Forecast", "التنبؤ"), line=dict(color=PAL[0], width=2.5)))
        ch.style_academic(fig, height=400, y_title=vsel, rtl=is_ar())
        _chart(fig, "var_fc", "var_forecast")
        fct = pd.DataFrame({"step": np.arange(1, int(hf) + 1), "forecast": fc["mean"][:, j], "lower": fc["lower"][:, j], "upper": fc["upper"][:, j]})
        st.dataframe(_disp(fct).round(4), width="stretch", hide_index=True)
    with sub[5]:
        txt, refs = _method("var", "mvv")
        c1, c2 = st.columns(2)
        with c1:
            excel_download({"Coefficients": r.table(), "Equations": r.stats["equations"], "Lag selection": sel,
                            "Diagnostics": vr.diagnostics(r), "Granger": vr.granger_pairs(r),
                            **({"IRF": st.session_state["var_irf_tab"]} if "var_irf_tab" in st.session_state else {})}, "var.xlsx", key="var_xl")
        with c2:
            st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"),
                               rp.to_docx([("h", tr("Methodology", "المنهجية")), ("p", txt), ("table", jt), ("note", note),
                                           ("table", vr.diagnostics(r).astype(str)), ("table", vr.granger_pairs(r).round(4).astype(str)),
                                           ("bullets", refs)], f"VAR({r.p})", is_ar()), "var.docx", key="var_docx")
        if st.button(tr("💾 Save equations to journal table", "💾 حفظ المعادلات في جدول المجلات"), key="var_save"):
            for e in eqs:
                st.session_state.setdefault("eco_models", {})[f"VAR({r.p}) — {e.name}"] = {
                    "r": e, "diag": {"table": pd.DataFrame(), "problems": {}, "detail": {}}, "dataset": active_name()}
            st.success(tr("Saved.", "تم الحفظ."))
        _code(mr.code("var", vars=r.names, p=r.p, type=r.type), "var_code")


def _parse_matrix(dfm: pd.DataFrame) -> np.ndarray:
    M = np.zeros(dfm.shape)
    for i in range(dfm.shape[0]):
        for j in range(dfm.shape[1]):
            v = str(dfm.iat[i, j]).strip().lower()
            M[i, j] = np.nan if v in ("free", "*", "na", "nan", "", "حر") else float(v)
    return M


def _svar_tab(Y):
    r = _load("mv_var", Y)
    if r is None:
        st.info(tr("Estimate the VAR first (VAR tab).", "قدّر نموذج VAR أولاً (تبويب VAR)."))
        return
    K = r.K
    mode = st.radio(tr("Identification", "طريقة التحديد"), ["chol", "A", "B", "bq"], horizontal=True, key="sv_mode",
                    format_func=lambda m: {"chol": tr("Recursive A-model (Cholesky)", "نموذج A تكراري (تشوليسكي)"),
                                           "A": tr("Custom A-model", "نموذج A مخصص"), "B": tr("Custom B-model", "نموذج B مخصص"),
                                           "bq": tr("Long-run (Blanchard-Quah)", "طويل الأجل (بلانشارد-كواه)")}[m])
    st.caption(tr("A·u_t = B·ε_t. Write 'free' for an estimated element, or a number for a fixed one. You need at least "
                  f"K(K−1)/2 = {K * (K - 1) // 2} restrictions beyond the normalisation.",
                  "A·u_t = B·ε_t. اكتب 'free' للعنصر المقدَّر أو رقماً للعنصر المثبت. تحتاج على الأقل "
                  f"K(K−1)/2 = {K * (K - 1) // 2} قيداً إضافياً."))
    Amat = Bmat = None
    if mode in ("chol", "A"):
        base = pd.DataFrame([["1" if i == j else ("free" if i > j else "0") for j in range(K)] for i in range(K)], index=r.names, columns=r.names)
        A_ed = st.data_editor(base, key="sv_A", disabled=mode == "chol")
        Amat = _parse_matrix(A_ed)
    elif mode == "B":
        base = pd.DataFrame([["free" if i >= j else "0" for j in range(K)] for i in range(K)], index=r.names, columns=r.names)
        B_ed = st.data_editor(base, key="sv_B")
        Bmat = _parse_matrix(B_ed)
    if st.button(tr("▶️ Estimate structural VAR", "▶️ تقدير VAR الهيكلي"), type="primary", key="sv_run"):
        try:
            sv = vr.blanchard_quah(r) if mode == "bq" else vr.svar_fit(r, Amat, Bmat)
            _store("mv_svar", sv, Y)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    sv = _load("mv_svar", Y)
    if sv is None:
        return
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**A**")
        st.dataframe(pd.DataFrame(sv["A"], index=r.names, columns=r.names).round(4), width="stretch")
        if sv.get("Ase") is not None:
            st.caption(tr("Std. errors", "الأخطاء المعيارية"))
            st.dataframe(pd.DataFrame(sv["Ase"], index=r.names, columns=r.names).round(4), width="stretch")
    with c2:
        st.markdown("**B**")
        st.dataframe(pd.DataFrame(sv["B"], index=r.names, columns=r.names).round(4), width="stretch")
        if sv.get("Bse") is not None:
            st.caption(tr("Std. errors", "الأخطاء المعيارية"))
            st.dataframe(pd.DataFrame(sv["Bse"], index=r.names, columns=r.names).round(4), width="stretch")
    if sv.get("LR_df", 0) and sv["LR_df"] > 0:
        insight(tr(f"LR test of over-identifying restrictions: χ²({sv['LR_df']}) = {sv['LR']:.3f}, {mr._p(sv['LR_p'])} — "
                   + ("restrictions rejected." if sv["LR_p"] < 0.05 else "restrictions not rejected."),
                   f"اختبار LR للقيود الزائدة: χ²({sv['LR_df']}) = {sv['LR']:.3f}، {mr._p(sv['LR_p'])} — "
                   + ("تُرفض القيود." if sv["LR_p"] < 0.05 else "لا تُرفض القيود.")))
    if "LRIM" in sv:
        st.markdown("**" + tr("Long-run impact matrix", "مصفوفة الأثر طويل الأجل") + "**")
        st.dataframe(pd.DataFrame(sv["LRIM"], index=r.names, columns=r.names).round(4), width="stretch")
    h = st.number_input(tr("Horizon", "الأفق"), 4, 60, 10, key="sv_h")
    runs = st.select_slider(tr("Bootstrap runs", "تكرارات البوتستراب"), [0, 100, 200, 500], 100, key="sv_runs")
    I = vr.irf(r, int(h), svar=sv)
    lo = hi = None
    if runs:
        if mode == "bq":
            fun = vr.blanchard_quah
        else:
            fun = lambda rb: vr.svar_fit(rb, Amat, Bmat)  # noqa: E731
        with st.spinner(tr("Bootstrapping…", "جارٍ البوتستراب…")):
            lo, hi = vr.irf_bootstrap(r, int(h), runs=int(runs), svar_fun=fun)
    _irf_grid(I, lo, hi, r.names, lambda a, b: f"ε_{a} → {b}", "sv_irf", "structural_irf")
    _method("svar", "mvs")
    _code(mr.code("svar", vars=r.names, p=r.p), "sv_code")


def _coint_tab(Y):
    lang = _lang()
    names = list(Y.columns)
    st.markdown("#### " + tr("Johansen system test", "اختبار جوهانسن على مستوى النظام"))
    c1, c2, c3 = st.columns(3)
    with c1:
        case = st.selectbox(tr("Deterministic case", "حالة الحدود المحددة"), [1, 2, 3, 4, 5], index=2, key="jo_case",
                            format_func=lambda c: f"{c} — {_t(vr.CASES[c])}")
    try:
        k_def = max(2, int(vr.var_select(Y, int(min(8, max(2, len(Y) // 12))), "const").attrs["selection"]["AIC"]))
    except Exception:  # noqa: BLE001
        k_def = 2
    with c2:
        K = st.number_input(tr("Lags in the levels VAR (K)", "الإبطاءات في VAR المستويات (K)"), 1, 12, k_def, key="jo_K")
    with c3:
        ss = st.checkbox(tr("Small-sample correction (Reinsel-Ahn)", "تصحيح العينة الصغيرة (رينسل-آن)"), key="jo_ss")
    st.caption(tr("Case 3 (unrestricted constant) is the usual choice for trending data without a trend in the cointegrating "
                  "relation; case 2 for non-trending data; case 4 if the equilibrium itself trends.",
                  "الحالة 3 (ثابت غير مقيد) هي المعتادة للبيانات ذات الاتجاه دون اتجاه في علاقة التكامل؛ والحالة 2 للبيانات "
                  "بلا اتجاه؛ والحالة 4 إذا كان التوازن نفسه ذا اتجاه."))
    with st.spinner(tr("Simulating critical values (first time only)…", "جارٍ محاكاة القيم الحرجة (أول مرة فقط)…")):
        jo = vr.johansen(Y, int(K), int(case), small_sample=ss)
    _store("mv_jo", jo, Y)
    show = jo["table"].copy()
    for c in ("Trace p-value", "Max p-value"):
        show[c] = show[c].map(lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}")
    st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
    st.caption(tr("Critical values and p-values: Monte Carlo of the asymptotic distributions (8 000 replications, T = 800), "
                  "as MacKinnon-Haug-Michelis (1999).", "القيم الحرجة والاحتمالية: محاكاة مونت كارلو للتوزيعات التقاربية "
                  "(8000 تكرار، T = 800)، على نهج ماكينون وهوغ وميكيليس (1999)."))
    insight(mr.interpret_johansen(jo, lang))
    st.markdown("#### " + tr("Single-equation tests and long-run estimators", "اختبارات المعادلة الواحدة ومقدرات الأجل الطويل"))
    c1, c2 = st.columns(2)
    with c1:
        dep = st.selectbox(tr("Dependent variable", "المتغير التابع"), names, key="eg_y")
    with c2:
        trd = st.selectbox(tr("Deterministics", "الحدود المحددة"), ["c", "ct"], key="eg_tr",
                           format_func=lambda k: tr("Constant", "ثابت") if k == "c" else tr("Constant + trend", "ثابت واتجاه"))
    xs = [n for n in names if n != dep]
    yv, Xv = Y[dep].to_numpy(), Y[xs].to_numpy()
    eg = vr.engle_granger(yv, Xv, trd, names=xs)
    st.dataframe(_disp(pd.DataFrame([{"Test": "Engle-Granger (ADF on residuals)", "Statistic": eg["stat"], "1% cv": eg["crit"]["1%"],
                                      "5% cv": eg["crit"]["5%"], "10% cv": eg["crit"]["10%"], "p-value": _fp(eg["p"]),
                                      "Lags": eg["lags"]}])).round(4), width="stretch", hide_index=True)
    insight(tr(f"Engle-Granger: {mr._p(eg['p'])} — " + ("the residuals are stationary: cointegration." if eg["p"] < 0.05 else "no cointegration at 5%."),
               f"إنجل-جرانجر: {mr._p(eg['p'])} — " + ("البواقي مستقرة: يوجد تكامل مشترك." if eg["p"] < 0.05 else "لا تكامل مشترك عند 5%.")))
    with st.expander(tr("Error-correction model (second step)", "نموذج تصحيح الخطأ (الخطوة الثانية)")):
        e_t = vr.ecm(yv, Xv, eg["resid"], 1, xs)
        show = e_t.copy()
        show["p-value"] = show["p-value"].map(_fp)
        st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
    lr = {"OLS (static)": (eg["coef"], None), "FMOLS": None, "DOLS": None, "CCR": None}
    fm, do, cc = vr.fmols(yv, Xv, trd, names=xs), vr.dols(yv, Xv, trd, 2, 2, names=xs), vr.ccr(yv, Xv, trd, names=xs)
    rows = []
    for lab, res in (("FMOLS", fm), ("DOLS (2 leads, 2 lags)", do), ("CCR", cc)):
        for _, row in res["table"].iterrows():
            rows.append({"Estimator": lab, "Variable": row["Variable"], "Coef.": row["Coef."], "Std. Err.": row["Std. Err."],
                         "t": row["t"], "p-value": row["p-value"]})
    del lr
    lrt = pd.DataFrame(rows)
    show = lrt.copy()
    show["p-value"] = show["p-value"].map(_fp)
    st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
    st.caption(tr("FMOLS / CCR: Bartlett kernel with Andrews (1991) bandwidth; DOLS: long-run variance standard errors (R cointReg conventions).",
                  "FMOLS وCCR: نواة بارتليت وعرض نطاق أندروز (1991)؛ DOLS: أخطاء معيارية من التباين طويل الأجل (اصطلاحات حزمة cointReg)."))
    st.markdown("#### " + tr("Gregory-Hansen test with a structural break", "اختبار جريجوري-هانسن مع انكسار هيكلي"))
    gmod = st.radio(tr("Break type", "نوع الانكسار"), list(vr.GH_MODELS), horizontal=True, key="gh_m",
                    format_func=lambda k: _t(vr.GH_MODELS[k]))
    if st.button(tr("▶️ Run Gregory-Hansen", "▶️ تشغيل جريجوري-هانسن"), key="gh_run"):
        with st.spinner(tr("Searching break dates and simulating critical values…", "جارٍ البحث عن تاريخ الانكسار ومحاكاة القيم الحرجة…")):
            _store("mv_gh", vr.gregory_hansen(yv, Xv, gmod, index=Y.index), Y)
    ghr = _load("mv_gh", Y)
    if ghr is not None:
        show = ghr["table"].copy()
        show["p-value"] = show["p-value"].map(lambda v: "< 0.001" if v < 0.001 else f"{v:.3f}")
        show["Break"] = show["Break"].map(tu._tlabel)
        show["Lags"] = show["Lags"].astype(str)
        st.dataframe(_disp(show).round(3), width="stretch", hide_index=True)
        b = ghr["table"].iloc[0]
        insight(tr(f"ADF* = {b['Statistic']:.3f} (break {tu._tlabel(b['Break'])}): " + ("cointegration with a regime shift." if b["Statistic"] < b["5% cv"] else "no cointegration even allowing for a break."),
                   f"ADF* = {b['Statistic']:.3f} (الانكسار {tu._tlabel(b['Break'])}): " + ("تكامل مشترك مع تحول في النظام." if b["Statistic"] < b["5% cv"] else "لا تكامل حتى مع السماح بانكسار.")))
    txt, refs = _method("coint", "mvc")
    c1, c2 = st.columns(2)
    with c1:
        excel_download({"Johansen": jo["table"], "Long-run estimators": lrt}, "cointegration.xlsx", key="co_xl")
    with c2:
        st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"),
                           rp.to_docx([("h", tr("Methodology", "المنهجية")), ("p", txt), ("table", jo["table"].round(4).astype(str)),
                                       ("p", mr.interpret_johansen(jo, lang)), ("table", lrt.round(4).astype(str)), ("bullets", refs)],
                                      tr("Cointegration analysis", "تحليل التكامل المشترك"), is_ar()), "cointegration.docx", key="co_docx")
    _code(mr.code("coint", vars=names, case=int(case), K=int(K)), "co_code")


def _vecm_tab(Y):
    lang = _lang()
    jo = _load("mv_jo", Y)
    if jo is None:
        st.info(tr("Open the Cointegration tab first (Johansen settings).", "افتح تبويب التكامل المشترك أولاً (إعدادات جوهانسن)."))
        return
    n = len(jo["names"])
    rdef = int(min(max(1, jo["rank_trace"]), n - 1))
    r_ = st.number_input(tr("Cointegration rank r", "رتبة التكامل المشترك r"), 1, n - 1, rdef, key="vecm_r")
    st.caption(tr(f"Johansen case {jo['case']}, K = {jo['K']} (change them in the Cointegration tab).",
                  f"حالة جوهانسن {jo['case']}، K = {jo['K']} (غيّرهما في تبويب التكامل المشترك)."))
    vm = vr.vecm(jo, int(r_))
    st.markdown("**" + tr("Cointegrating vectors β (normalised)", "متجهات التكامل المشترك β (مطبّعة)") + "**")
    bt = vm["beta_table"].copy()
    st.dataframe(_disp(bt).round(4), width="stretch", hide_index=True)
    tab = vm["table"]
    al = tab[tab["Regressor"].str.startswith("ECT")].copy()
    show = al.copy()
    show["p-value"] = show["p-value"].map(_fp)
    st.markdown("**" + tr("Adjustment coefficients α", "معاملات التعديل α") + "**")
    st.dataframe(_disp(show[["Equation", "Regressor", "Coef.", "Std. Err.", "t", "p-value"]]).round(4), width="stretch", hide_index=True)
    lines = []
    for _, row in al.iterrows():
        if row["p-value"] < 0.05:
            lines.append(tr(f"{row['Equation']} adjusts to {row['Regressor']} (α = {row['Coef.']:.3f}, {mr._p(row['p-value'])})",
                            f"{row['Equation']} يتعدل نحو {row['Regressor']} (α = {row['Coef.']:.3f}، {mr._p(row['p-value'])})"))
    insight((tr("Significant error correction: ", "تصحيح خطأ معنوي: ") + "; ".join(lines)) if lines else
            tr("No equation shows significant error correction at 5%.", "لا توجد معادلة بتصحيح خطأ معنوي عند 5%."))
    we = vr.weak_exogeneity(jo, int(r_))
    show = we.copy()
    show["p-value"] = show["p-value"].map(_fp)
    st.markdown("**" + tr("Weak exogeneity (LR test that the variable's α row = 0)", "الضعف الخارجي (اختبار LR أن صف α للمتغير = 0)") + "**")
    st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
    if int(r_) == 1:
        h = st.text_input(tr("Test a long-run restriction: β proportional to (comma-separated, one value per row of β)",
                             "اختبار قيد طويل الأجل: β متناسب مع (قيم مفصولة بفواصل، قيمة لكل صف من β)"),
                          ",".join(["1", "-1"] + ["0"] * (len(jo["z1_names"]) - 2)), key="vecm_h")
        try:
            hv = np.array([float(x) for x in h.split(",")])
            if len(hv) == len(jo["z1_names"]):
                rt = vr.restriction_test(jo, 1, H=hv[:, None])
                insight(tr(f"LR = {rt['LR']:.3f}, χ²({rt['df']}), {mr._p(rt['p'])} — " + ("restriction rejected." if rt["p"] < 0.05 else "restriction accepted."),
                           f"LR = {rt['LR']:.3f}، χ²({rt['df']})، {mr._p(rt['p'])} — " + ("يُرفض القيد." if rt["p"] < 0.05 else "يُقبل القيد.")))
        except ValueError:
            pass
    with st.expander(tr("Short-run coefficients Γ", "معاملات الأجل القصير Γ")):
        show = tab.copy()
        show["p-value"] = show["p-value"].map(_fp)
        st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
    lv = vr.vecm_levels(vm)
    shim = SimpleNamespace(resid=vm["resid"], p=jo["K"], names=vm["names"], K=n)
    pt, nm_ = vr.serial_pt(shim, max(jo["K"] + 4, 12), vec_rank=int(r_)), vr.normality(shim)
    ar = vr.arch_multi(shim, 5)
    st.dataframe(_disp(pd.DataFrame([
        {"Test": f"Portmanteau Q({pt['h']})", "Statistic": pt["Q"], "df": str(pt["df"]), "p-value": _fp(pt["p"])},
        {"Test": "Jarque-Bera (multivariate)", "Statistic": nm_["JB"], "df": str(nm_["df"]), "p-value": _fp(nm_["p"])},
        {"Test": "ARCH-LM multivariate (5)", "Statistic": ar["stat"], "df": str(int(ar["df"])), "p-value": _fp(ar["p"])}])).round(4),
        width="stretch", hide_index=True)
    hz = st.number_input(tr("IRF horizon", "أفق دوال الاستجابة"), 4, 80, 20, key="vecm_h_irf")
    I = vr.irf(lv, int(hz), "orth")
    st.caption(tr("Responses of the levels (vec2var): with cointegration, effects of shocks to the common trends are permanent.",
                  "استجابات المستويات (vec2var): مع التكامل المشترك تكون آثار صدمات الاتجاهات المشتركة دائمة."))
    _irf_grid(I, None, None, vm["names"], lambda a, b: f"{a} → {b}", "vecm_irf", "vecm_irf")
    txt, refs = _method("vecm", "mvm", r=int(r_))
    excel_download({"beta": bt, "alpha+Gamma": tab, "Weak exogeneity": we}, "vecm.xlsx", key="vecm_xl")
    _code(mr.code("vecm", vars=vm["names"], case=jo["case"], K=jo["K"], r=int(r_)), "vecm_code")


# ==========================================================
# ARDL / NARDL / QARDL
# ==========================================================
def render_ardl(df):
    lang = _lang()
    st.caption("💡 " + tr("ARDL suits regressors that are I(0), I(1) or a mix (never I(2)). Choose the dependent variable, the "
                         "long-run regressors, optional fixed I(0) regressors (dummies, weather) and — for NARDL — which "
                         "regressors have asymmetric effects.",
                         "يناسب ARDL متغيرات I(0) أو I(1) أو خليطاً منهما (وليس I(2)). اختر المتغير التابع ومتغيرات الأجل الطويل "
                         "ومتغيرات ثابتة I(0) اختيارية (متغيرات وهمية، طقس) — وفي NARDL المتغيرات ذات الأثر غير المتماثل."))
    out = _multi(df, "ar", 2)
    if out is None:
        return
    Y, tcol, s, trans = out
    names = list(Y.columns)
    c1, c2, c3 = st.columns(3)
    with c1:
        y = st.selectbox(tr("Dependent variable", "المتغير التابع"), names, key="ardl_y")
    rest = [n for n in names if n != y]
    with c2:
        fixed = st.multiselect(tr("Fixed regressors (no lags, short-run only)", "متغيرات ثابتة (بلا إبطاء)"), rest, key="ardl_fixed", placeholder=tr("None", "لا شيء"))
    xs = [n for n in rest if n not in fixed]
    with c3:
        asym = st.multiselect(tr("Asymmetric regressors (NARDL)", "متغيرات غير متماثلة (NARDL)"), xs, key="ardl_asym", placeholder=tr("None (linear ARDL)", "لا شيء (ARDL خطي)"))
    if not xs:
        st.info(tr("Keep at least one long-run regressor.", "أبقِ متغيراً واحداً على الأقل للأجل الطويل."))
        return
    D = Y.copy()
    pairs = {}
    if asym:
        D, pairs = ad.nardl_data(D, asym)
    xlist = []
    for x in xs:
        xlist += list(pairs[x]) if x in pairs else [x]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        case = st.selectbox(tr("PSS case", "حالة PSS"), [1, 2, 3, 4, 5], index=2, key="ardl_case", format_func=lambda c: _t(ad.CASES[c]))
    with c2:
        my = st.number_input(tr("Max lag of y", "أقصى إبطاء لـ y"), 1, 8, 4 if len(D) > 60 else 2, key="ardl_my")
    with c3:
        mx = st.number_input(tr("Max lag of x", "أقصى إبطاء لـ x"), 0, 8, 4 if len(D) > 60 else 2, key="ardl_mx")
    with c4:
        crit = st.selectbox(tr("Criterion", "المعيار"), ["AIC", "BIC"], key="ardl_crit")
    n_models = int(my) * (int(mx) + 1) ** len(xlist)
    if n_models > 20000:
        st.warning(tr(f"{n_models:,} candidate models — reduce the maximum lags.", f"{n_models:,} نموذجاً مرشحاً — قلّل أقصى الإبطاءات."))
        return
    with st.spinner(tr(f"Comparing {n_models:,} ARDL specifications…", f"جارٍ مقارنة {n_models:,} صيغة ARDL…")):
        sel = ad.auto_ardl(D, y, xlist, [int(my)] + [int(mx)] * len(xlist), int(case), fixed, crit)
    with st.expander(tr("Lag selection (top 20, common sample)", "اختيار الإبطاءات (أفضل 20، عينة مشتركة)"), expanded=False):
        st.dataframe(_disp(sel.drop(columns="order")).round(4), width="stretch", hide_index=True)
    best = sel["order"].iloc[0]
    oc = st.columns(len(best))
    order = []
    for i, (col, o) in enumerate(zip(oc, best)):
        lab = f"p ({y})" if i == 0 else f"q ({xlist[i - 1]})"
        order.append(int(col.number_input(lab, 1 if i == 0 else 0, 12, int(o), key=f"ardl_o{i}_{'_'.join(xlist)}")))
    r = ad.fit_ardl(D, y, xlist, tuple(order), int(case), fixed)
    title = ("NARDL" if pairs else "ARDL") + "(" + ", ".join(map(str, order)) + ")"
    st.subheader(title)
    m = st.columns(4)
    for col, (k_, v_) in zip(m, [("N", r.stats["N"]), ("Adj. R²", r.stats["Adj. R²"]), ("AIC", r.stats["AIC"]), ("σ", r.stats["σ"])]):
        col.metric(k_, "‎" + (f"{v_:,.4f}" if isinstance(v_, float) else str(v_)))
    sub = st.tabs([tr("🧾 Estimates", "🧾 التقديرات"), tr("📏 Bounds test", "📏 اختبار الحدود"), tr("⚖️ Long run & ECM", "⚖️ الأجل الطويل وتصحيح الخطأ"),
                   tr("🩺 Diagnostics", "🩺 التشخيص"), tr("↔️ Asymmetry (NARDL)", "↔️ عدم التماثل (NARDL)"),
                   tr("📊 Quantile ARDL", "📊 ARDL الكمي"), tr("📝 Methodology & export", "📝 المنهجية والتصدير")])
    with sub[0]:
        show = r.table()
        show["p-value"] = show["p-value"].map(_fp)
        st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
        u = ad._coef_table(r.uecm["names"], r.uecm["b"], r.uecm["V"], r.uecm["df"])
        with st.expander(tr("Unrestricted ECM (conditional, PSS eq. 6)", "نموذج تصحيح الخطأ غير المقيد (الشرطي)")):
            u2 = u.copy()
            u2["p-value"] = u2["p-value"].map(_fp)
            st.dataframe(_disp(u2).round(4), width="stretch", hide_index=True)
    with sub[1]:
        ex = st.checkbox(tr(f"Finite-sample critical values for T = {r.nobs} (simulated, Narayan-type)",
                            f"قيم حرجة للعينة الفعلية T = {r.nobs} (بالمحاكاة، على نهج ناريان)"), key="ardl_exact")
        kk = None
        if pairs:
            kopt = st.radio(tr("k for the bounds", "قيمة k للحدود"), ["dec", "orig"], horizontal=True, key="ardl_k",
                            format_func=lambda o: tr(f"decomposed (k = {len(xlist)})", f"بعد التفكيك (k = {len(xlist)})") if o == "dec"
                            else tr(f"original (k = {len(xs)}, more conservative)", f"الأصلي (k = {len(xs)}، أكثر تحفظاً)"))
            kk = len(xlist) if kopt == "dec" else len(xs)
        with st.spinner(tr("Simulating…", "جارٍ المحاكاة…")):
            bt = ad.bounds_test(r, exact=ex, k=kk)
        rows = [{"Test": "F (Wald)", "Statistic": bt["F"], "p-value I(0)": _fp(bt["pF_I0"]), "p-value I(1)": _fp(bt["pF_I1"]),
                 "Verdict": bt["verdict_F"]}]
        if "pt_I1" in bt:
            rows.append({"Test": "t (on y₍t−1₎)", "Statistic": bt["t"], "p-value I(0)": _fp(bt["pt_I0"]), "p-value I(1)": _fp(bt["pt_I1"]),
                         "Verdict": bt["verdict_t"]})
        st.dataframe(_disp(pd.DataFrame(rows)).round(4), width="stretch", hide_index=True)
        st.dataframe(_disp(bt["critical"]).round(3), width="stretch", hide_index=True)
        st.caption(tr("Critical values simulated with the PSS (2001) design (asymptotic: T = 1000, 40 000 replications).",
                      "القيم الحرجة بالمحاكاة وفق تصميم PSS (2001) (التقاربية: T = 1000، 40 ألف تكرار)."))
        insight(mr.interpret_bounds(bt, lang))
        c5 = bt["critical"].iloc[1]
        fig = go.Figure()
        fig.add_vrect(x0=c5["F I(0)"], x1=c5["F I(1)"], fillcolor="rgba(200,160,0,0.18)", line_width=0)
        fig.add_vline(x=bt["F"], line=dict(color=PAL[0], width=3))
        fig.add_annotation(x=bt["F"], y=1, yref="paper", text=f"F = {bt['F']:.2f}", showarrow=False, yanchor="bottom")
        fig.update_xaxes(range=[0, max(bt["F"], c5["F I(1)"]) * 1.3])
        fig.update_yaxes(visible=False)
        ch.style_academic(fig, height=180, x_title=tr("F statistic vs 5% bounds", "إحصاءة F مقابل حدود 5%"), legend="none", rtl=is_ar())
        _chart(fig, "ardl_bounds", "bounds")
    with sub[2]:
        lr = ad.long_run(r)
        show = lr.copy()
        show["p-value"] = show["p-value"].map(_fp)
        st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
        rcm = ad.recm(r)
        show = rcm["table"].copy()
        show["p-value"] = show["p-value"].map(_fp)
        st.markdown("**" + tr("Restricted error-correction model", "نموذج تصحيح الخطأ المقيد") + "**")
        st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
        per = {12: tr("month", "شهر"), 4: tr("quarter", "ربع"), 1: tr("year", "سنة")}.get(s, tr("period", "فترة"))
        insight(mr.interpret_ecm(rcm["speed"], rcm["half_life"], lang, per))
        sig = lr[(lr["p-value"] < 0.05) & (~lr["Variable"].isin(["(Intercept)", "trend"]))]
        if not sig.empty:
            st.markdown("### " + tr("🏛️ For decision-makers", "🏛️ لصانع القرار"))
            insight("<br>".join(
                (tr(f"A permanent 1-unit (1% if both in logs) rise in {row['Variable']} changes {y} by {row['Long-run coef.']:.3f} in the long run (95% CI {row['CI low']:.3f} – {row['CI high']:.3f}).",
                    f"ارتفاع دائم بوحدة واحدة (1% إذا كان المتغيران لوغاريتميين) في {row['Variable']} يغيّر {y} بمقدار {row['Long-run coef.']:.3f} في الأجل الطويل (فترة الثقة 95%: {row['CI low']:.3f} – {row['CI high']:.3f})."))
                for _, row in sig.iterrows()))
    with sub[3]:
        e = r.resid
        X = r.X
        bg = dx.breusch_godfrey(e, X, 2)
        bp = dx.breusch_pagan(e, X[:, 1:] if case >= 2 else X, True)
        jb = dx.normality(e)[0]
        rs = dx.reset_test(r.yv, X, X @ r.b, 2)
        rows = [{"Test": "Breusch-Godfrey LM (2)", "Statistic": bg["stat"], "p-value": _fp(bg["p"])},
                {"Test": "Breusch-Pagan-Koenker", "Statistic": bp["stat"], "p-value": _fp(bp["p"])},
                {"Test": "Jarque-Bera", "Statistic": jb["stat"], "p-value": _fp(jb["p"])},
                {"Test": "Ramsey RESET (ŷ²)", "Statistic": rs["stat"], "p-value": _fp(rs["p"])}]
        st.dataframe(_disp(pd.DataFrame(rows)).round(4), width="stretch", hide_index=True)
        try:
            cs = dx.cusum(r.yv, X)
            _chart(rc.cusum_plots(cs, rtl=is_ar()), "ardl_cusum", "cusum")
            insight(tr("CUSUM " + ("stays within" if cs["cusum_ok"] else "crosses") + " the 5% bands and CUSUMSQ " +
                       ("stays within" if cs["cusumsq_ok"] else "crosses") + " them: " +
                       ("the coefficients are stable." if cs["cusum_ok"] and cs["cusumsq_ok"] else "evidence of parameter instability."),
                       "CUSUM " + ("داخل" if cs["cusum_ok"] else "يخرج عن") + " حدود 5%، وCUSUMSQ " + ("داخلها" if cs["cusumsq_ok"] else "يخرج عنها") +
                       ": " + ("المعاملات مستقرة." if cs["cusum_ok"] and cs["cusumsq_ok"] else "توجد دلائل على عدم استقرار المعاملات.")))
        except Exception as ex_:  # noqa: BLE001
            st.info(str(ex_))
    with sub[4]:
        if not pairs:
            st.info(tr("Select asymmetric regressors above to estimate a NARDL.", "اختر متغيرات غير متماثلة أعلاه لتقدير NARDL."))
        else:
            at = ad.asymmetry_tests(r, pairs)
            show = at.copy()
            show["p-value"] = show["p-value"].map(_fp)
            st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
            for _, row in at.iterrows():
                insight(tr(f"{row['Test']} for {row['Variable']}: {mr._p(row['p-value'])} — " + ("asymmetry confirmed." if row["p-value"] < 0.05 else "symmetry not rejected."),
                           f"{row['Test']} لـ {row['Variable']}: {mr._p(row['p-value'])} — " + ("يثبت عدم التماثل." if row["p-value"] < 0.05 else "لا يُرفض التماثل.")))
            hz = st.number_input(tr("Multiplier horizon", "أفق المضاعفات"), 4, 120, 24, key="nardl_h")
            mb = ad.multiplier_bands(r, D, pairs, int(hz), 300)
            for a_, (pn, nn) in pairs.items():
                mp, mn = ad.dynamic_multipliers(r, pn, int(hz)), ad.dynamic_multipliers(r, nn, int(hz))
                hh = np.arange(int(hz) + 1)
                fig = go.Figure([
                    go.Scatter(x=np.r_[hh, hh[::-1]], y=np.r_[mb[a_]["upper"], mb[a_]["lower"][::-1]], fill="toself",
                               fillcolor="rgba(120,120,120,0.18)", line=dict(width=0), name=tr("95% band (asymmetry)", "نطاق 95% (عدم التماثل)")),
                    go.Scatter(x=hh, y=mp, name=tr("positive change", "تغير موجب"), line=dict(color=PAL[0], width=2.5)),
                    go.Scatter(x=hh, y=-mn, name=tr("negative change (sign flipped)", "تغير سالب (بإشارة معكوسة)"), line=dict(color=PAL[1], width=2.5, dash="dash")),
                    go.Scatter(x=hh, y=mb[a_]["asym"], name=tr("asymmetry", "عدم التماثل"), line=dict(color="#333", width=1.5))])
                fig.add_hline(y=0, line=dict(color="#999", width=1))
                ch.style_academic(fig, height=420, x_title=tr("Periods after the shock", "الفترات بعد الصدمة"),
                                  title=tr(f"Cumulative dynamic multipliers — {a_}", f"المضاعفات الديناميكية التراكمية — {a_}"), rtl=is_ar())
                _chart(fig, f"nardl_mult_{a_}", f"multipliers_{a_}")
                insight(tr(f"After {int(hz)} periods a 1-unit rise in {a_} raises {y} by {mp[-1]:.3f}, while a 1-unit fall lowers it by {abs(mn[-1]):.3f}.",
                           f"بعد {int(hz)} فترة، ارتفاع {a_} بوحدة يرفع {y} بمقدار {mp[-1]:.3f}، بينما انخفاضه بوحدة يخفضه بمقدار {abs(mn[-1]):.3f}."))
    with sub[5]:
        taus = st.multiselect("τ", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], [0.1, 0.25, 0.5, 0.75, 0.9], key="qardl_tau")
        if taus and st.button(tr("▶️ Estimate QARDL", "▶️ تقدير QARDL"), key="qardl_run"):
            with st.spinner(tr("Quantile regressions…", "جارٍ الانحدار الكمي…")):
                _store("ar_q", ad.qardl(D, y, xlist, tuple(order), tuple(sorted(taus)), int(case)), Y)
        q = _load("ar_q", Y)
        if q is not None:
            show = q["long_run"].copy()
            show["p-value"] = show["p-value"].map(_fp)
            st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
            lrq = q["long_run"]
            fig = go.Figure()
            for i, vname in enumerate(lrq["Variable"].unique()):
                sdf = lrq[lrq["Variable"] == vname]
                fig.add_trace(go.Scatter(x=sdf["τ"], y=sdf["Long-run β(τ)"], mode="lines+markers", name=vname, line=dict(color=PAL[i % len(PAL)]),
                                         error_y=dict(type="data", array=1.96 * sdf["Std. Err."], visible=True)))
            ch.style_academic(fig, height=400, x_title="τ", y_title="β(τ)", rtl=is_ar())
            _chart(fig, "qardl_fig", "qardl")
            insight(tr("Long-run effects that change across quantiles indicate that the relation differs between low and high "
                       "outcomes of the dependent variable (e.g. in crises vs booms).",
                       "تغير آثار الأجل الطويل عبر الكميات يعني أن العلاقة تختلف بين القيم المنخفضة والمرتفعة للمتغير التابع "
                       "(مثل فترات الأزمات مقابل الرواج)."))
    with sub[6]:
        txt, refs = _method("nardl" if pairs else "ardl", "ard")
        if pairs:
            t2, r2 = mr.methodology("ardl", lang)
            txt, refs = t2 + " " + txt, sorted(set(refs + r2))
        c1, c2 = st.columns(2)
        bt0 = ad.bounds_test(r)
        with c1:
            excel_download({"ARDL": r.table(), "Long run": ad.long_run(r), "ECM": ad.recm(r)["table"], "Bounds": bt0["critical"],
                            "Lag selection": sel.drop(columns="order")}, "ardl.xlsx", key="ardl_xl")
        with c2:
            st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"),
                               rp.to_docx([("h", tr("Methodology", "المنهجية")), ("p", txt), ("table", r.table().round(4).astype(str)),
                                           ("p", mr.interpret_bounds(bt0, lang)), ("table", ad.long_run(r).round(4).astype(str)),
                                           ("table", ad.recm(r)["table"].round(4).astype(str)), ("bullets", refs)], title, is_ar()),
                               "ardl.docx", key="ardl_docx")
        _code(mr.code("nardl" if pairs else "ardl", y=y, x=xs if pairs else xlist, order=tuple(order), case=int(case), fixed=fixed,
                      asym=asym[0] if asym else None), "ardl_code")


# ==========================================================
# Volatility
# ==========================================================
def render_garch(df):
    lang = _lang()
    st.caption("💡 " + tr("Use returns, not prices: choose 'Δln ×100' for price series (percentage log returns). GARCH needs "
                         "a few hundred observations at least (daily or weekly data).",
                         "استخدم العوائد لا الأسعار: اختر «Δln ×100» لسلاسل الأسعار (عوائد لوغاريتمية مئوية). يحتاج GARCH "
                         "بضع مئات من المشاهدات على الأقل (بيانات يومية أو أسبوعية)."))
    out = _multi(df, "ga", 1)
    if out is None:
        return
    Y, tcol, s, trans = out
    tabs = st.tabs([tr("🌪️ Univariate GARCH", "🌪️ GARCH أحادي"), tr("🕸️ Multivariate (CCC / DCC / BEKK)", "🕸️ متعدد المتغيرات (CCC / DCC / BEKK)")])
    with tabs[0]:
        _uni_garch(Y, lang)
    with tabs[1]:
        if Y.shape[1] < 2:
            st.info(tr("Select at least two return series.", "اختر سلسلتي عوائد على الأقل."))
        else:
            _multi_garch(Y, lang)


def _uni_garch(Y, lang):
    v = st.selectbox(tr("Return series", "سلسلة العوائد"), list(Y.columns), key="g_v")
    y = Y[v].to_numpy()
    al = gm.arch_lm(y, 5)
    from scipy import stats as sst
    d = pd.DataFrame([{"N": len(y), "Mean": y.mean(), "Std. dev.": y.std(ddof=1), "Skewness": sst.skew(y), "Excess kurtosis": sst.kurtosis(y),
                       "ARCH-LM(5)": al["LM"], "p (ARCH)": _fp(al["p"])}])
    st.dataframe(d.round(4), width="stretch", hide_index=True)
    insight(tr(f"ARCH-LM pre-test: {mr._p(al['p'])} — " + ("volatility clustering is present: a GARCH model is warranted." if al["p"] < 0.05 else "no significant ARCH effects; GARCH may be unnecessary."),
               f"اختبار ARCH-LM القبلي: {mr._p(al['p'])} — " + ("يوجد تجمع للتقلب: نموذج GARCH مبرر." if al["p"] < 0.05 else "لا آثار ARCH معنوية؛ قد لا يلزم GARCH.")))
    c1, c2, c3, c4, c5 = st.columns(5)
    model = c1.selectbox(tr("Model", "النموذج"), list(gm.MODELS), index=1, key="g_model", format_func=lambda k: _t(gm.MODELS[k]))
    dist = c2.selectbox(tr("Distribution", "التوزيع"), list(gm.DISTS), index=1, key="g_dist", format_func=lambda k: _t(gm.DISTS[k]))
    mean = c3.selectbox(tr("Mean equation", "معادلة المتوسط"), ["const", "zero", "ar1", "m_sd", "m_var"], key="g_mean",
                        format_func=lambda k: {"const": tr("Constant", "ثابت"), "zero": tr("Zero", "صفر"), "ar1": "AR(1)",
                                               "m_sd": tr("GARCH-M (σ)", "GARCH-M (σ)"), "m_var": tr("GARCH-M (σ²)", "GARCH-M (σ²)")}[k])
    p = c4.number_input("p (GARCH)", 0 if model == "arch" else 1, 2, 0 if model == "arch" else 1, key="g_p", disabled=model == "arch")
    q = c5.number_input("q (ARCH)", 1, 3, 1, key="g_q")
    c1, c2 = st.columns(2)
    if c1.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="g_run"):
        with st.spinner(tr("Maximum likelihood…", "جارٍ التقدير بالإمكان الأعظم…")):
            try:
                _store("g_res", gm.fit_garch(y, model, int(p), int(q), dist, mean, Y.index, v), Y)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    if c2.button(tr("🏁 Compare GARCH / GJR / EGARCH × distributions", "🏁 مقارنة GARCH وGJR وEGARCH × التوزيعات"), key="g_cmp"):
        rows = []
        bar = st.progress(0.0)
        combos = [(m_, d_) for m_ in ("garch", "gjr", "egarch") for d_ in ("norm", "std", "ged", "sstd")]
        for i, (m_, d_) in enumerate(combos):
            try:
                rr = gm.fit_garch(y, m_, 1, 1, d_, "const")
                dg = gm.garch_diagnostics(rr).set_index("Test")["p-value"]
                rows.append({"Model": rr.title, "Log-likelihood": rr.llf, "AIC": rr.stats["AIC"], "BIC": rr.stats["BIC"],
                             "Persistence": rr.stats["Persistence"], "LB(z²) p": float(dg.iloc[1])})
            except Exception:  # noqa: BLE001
                pass
            bar.progress((i + 1) / len(combos))
        _store("g_cmp", pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True), Y)
    cmpt = _load("g_cmp", Y)
    if cmpt is not None:
        st.dataframe(_disp(cmpt).round(4), width="stretch", hide_index=True)
        insight(tr(f"Lowest AIC: <b>{cmpt['Model'].iloc[0]}</b>; lowest BIC: <b>{cmpt.sort_values('BIC')['Model'].iloc[0]}</b>.",
                   f"أقل AIC: <b>{cmpt['Model'].iloc[0]}</b>؛ أقل BIC: <b>{cmpt.sort_values('BIC')['Model'].iloc[0]}</b>."))
    r = _load("g_res", Y)
    if r is None or r.label != v:
        return
    st.subheader(r.title + f" — {v}")
    rob = st.toggle(tr("Robust (Bollerslev-Wooldridge) standard errors", "أخطاء معيارية متينة (بولرسليف-وولدريدج)"), True, key="g_rob")
    tab = r.table(rob)
    show = tab.copy()
    show["p-value"] = show["p-value"].map(_fp)
    st.dataframe(_disp(show).round(5), width="stretch", hide_index=True)
    m = st.columns(5)
    for col, (k_, v_) in zip(m, [("Log-likelihood", r.llf), ("AIC", r.stats["AIC"]), ("BIC", r.stats["BIC"]),
                                 (tr("Persistence", "الاستمرارية"), r.stats["Persistence"]), (tr("Half-life", "نصف العمر"), r.stats["Half-life"])]):
        col.metric(k_, "‎" + (f"{v_:,.3f}" if np.isfinite(v_) else "∞"))
    dg = gm.garch_diagnostics(r)
    insight(mr.interpret_garch(r, dg, lang))
    sub = st.tabs([tr("📈 Conditional volatility", "📈 التقلب الشرطي"), tr("📰 News impact", "📰 أثر الأخبار"), tr("🩺 Diagnostics", "🩺 التشخيص"),
                   tr("🔮 Forecast", "🔮 التنبؤ"), tr("🛡️ Value-at-Risk", "🛡️ القيمة المعرضة للخطر"), tr("📝 Methodology & export", "📝 المنهجية والتصدير")])
    idx = r.index if r.index is not None else np.arange(len(r.y))
    with sub[0]:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                            subplot_titles=[tr("Returns", "العوائد"), tr("Conditional volatility σ_t", "التقلب الشرطي σ_t")])
        fig.add_trace(go.Scatter(x=idx, y=r.y, mode="lines", line=dict(color="#888", width=0.8)), 1, 1)
        fig.add_trace(go.Scatter(x=idx, y=np.sqrt(r.sigma2), mode="lines", line=dict(color=PAL[0], width=1.6)), 2, 1)
        ch.style_academic(fig, height=560, legend="none", rtl=is_ar())
        _chart(fig, "g_vol", "conditional_volatility")
        top = pd.Series(np.sqrt(r.sigma2), index=idx).nlargest(3)
        insight(tr("Highest-volatility dates: ", "أعلى تواريخ التقلب: ") + ", ".join(f"{tu._tlabel(k_)} (σ = {v_:.2f})" for k_, v_ in top.items()))
    with sub[1]:
        ni = gm.news_impact(r)
        fig = go.Figure(go.Scatter(x=ni["shock"], y=ni["sigma2"], mode="lines", line=dict(color=PAL[0], width=2.5)))
        ch.style_academic(fig, height=380, x_title=tr("Shock ε_{t−1}", "الصدمة ε_{t−1}"), y_title="σ²_t", legend="none", rtl=is_ar())
        _chart(fig, "g_nic", "news_impact")
        st.caption(tr("A steeper left branch = bad news raises volatility more (leverage).", "فرع أيسر أشد انحداراً = الأخبار السيئة ترفع التقلب أكثر (الرافعة)."))
    with sub[2]:
        show = dg.copy()
        show["p-value"] = show["p-value"].map(_fp)
        st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
        z = r.z
        from scipy import stats as sst
        qq = np.sort(z)
        th = sst.norm.ppf((np.arange(1, len(z) + 1) - 0.5) / len(z))
        fig = go.Figure([go.Scatter(x=th, y=qq, mode="markers", marker=dict(size=3, color=PAL[0])),
                         go.Scatter(x=[th.min(), th.max()], y=[th.min(), th.max()], mode="lines", line=dict(color=PAL[1], dash="dash"))])
        ch.style_academic(fig, height=360, x_title=tr("Normal quantiles", "كميات التوزيع الطبيعي"), y_title=tr("Standardised residuals", "البواقي المعيارية"), legend="none", rtl=is_ar())
        _chart(fig, "g_qq", "qq_standardised")
    with sub[3]:
        h = st.number_input(tr("Horizon", "الأفق"), 1, 250, 20, key="g_h")
        with st.spinner(tr("Forecasting…", "جارٍ التنبؤ…")):
            fv = gm.forecast_variance(r, int(h))
        fig = go.Figure([go.Scatter(x=fv["step"], y=fv["volatility"], mode="lines+markers", name=tr("σ forecast", "التنبؤ بـ σ"), line=dict(color=PAL[0])),
                         go.Scatter(x=fv["step"], y=np.sqrt(fv["unconditional"]), mode="lines", name=tr("long-run σ", "σ طويل الأجل"), line=dict(color=PAL[1], dash="dash"))])
        ch.style_academic(fig, height=360, x_title=tr("Steps ahead", "خطوات للأمام"), y_title="σ", rtl=is_ar())
        _chart(fig, "g_fc", "volatility_forecast")
        st.dataframe(_disp(fv).round(4), width="stretch", hide_index=True)
    with sub[4]:
        lvl = st.select_slider(tr("VaR level", "مستوى VaR"), [0.01, 0.025, 0.05, 0.10], 0.01, key="g_var", format_func=lambda a: f"{a:.1%}")
        vv = gm.var_es(r, lvl)
        c1, c2, c3 = st.columns(3)
        c1.metric(tr("1-step VaR", "VaR لخطوة واحدة"), f"{vv['VaR']:.3f}")
        c2.metric(tr("1-step ES", "ES لخطوة واحدة"), f"{vv['ES']:.3f}")
        c3.metric(tr("Violations / expected", "التجاوزات / المتوقع"), f"{vv['violations']} / {vv['expected']:.1f}")
        st.dataframe(_disp(pd.DataFrame([
            {"Test": "Kupiec POF (unconditional coverage)", "Statistic": vv["kupiec"], "p-value": _fp(vv["p_kupiec"])},
            {"Test": "Christoffersen independence", "Statistic": vv["christoffersen"], "p-value": _fp(vv["p_christoffersen"])},
            {"Test": "Conditional coverage (joint)", "Statistic": vv["kupiec"] + vv["christoffersen"], "p-value": _fp(vv["p_cc"])}])).round(4),
            width="stretch", hide_index=True)
        fig = go.Figure([go.Scatter(x=idx, y=r.y, mode="lines", line=dict(color="#999", width=0.8), name=tr("Returns", "العوائد")),
                         go.Scatter(x=idx, y=vv["series"], mode="lines", line=dict(color=PAL[1], width=1.4), name=f"VaR {lvl:.1%}"),
                         go.Scatter(x=np.asarray(idx)[vv["hits"] == 1], y=r.y[vv["hits"] == 1], mode="markers",
                                    marker=dict(color="#c0392b", size=6), name=tr("violations", "التجاوزات"))])
        ch.style_academic(fig, height=420, rtl=is_ar())
        _chart(fig, "g_varfig", "var_backtest")
        st.markdown("### " + tr("🏛️ For decision-makers", "🏛️ لصانع القرار"))
        insight(tr(f"With {1 - lvl:.1%} confidence, tomorrow's return should not fall below {vv['VaR']:.2f}%; if it does, the expected loss is {vv['ES']:.2f}% (expected shortfall). "
                   + ("The backtest supports the model." if vv["p_cc"] >= 0.05 else "⚠️ The backtest rejects the model's risk forecasts — consider fatter tails or an asymmetric model."),
                   f"بثقة {1 - lvl:.1%} لا يُتوقع أن ينخفض عائد الغد عن {vv['VaR']:.2f}%؛ وإن حدث فالخسارة المتوقعة {vv['ES']:.2f}% (العجز المتوقع). "
                   + ("ويدعم الاختبار الرجعي النموذج." if vv["p_cc"] >= 0.05 else "⚠️ يرفض الاختبار الرجعي تنبؤات المخاطر — جرّب ذيولاً أثقل أو نموذجاً غير متماثل.")))
    with sub[5]:
        txt, refs = _method("garch", "gu")
        c1, c2 = st.columns(2)
        with c1:
            excel_download({"Estimates": tab, "Diagnostics": dg, "Volatility": pd.DataFrame({"time": idx, "return": r.y, "sigma": np.sqrt(r.sigma2)})},
                           "garch.xlsx", key="g_xl")
        with c2:
            st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"),
                               rp.to_docx([("h", tr("Methodology", "المنهجية")), ("p", txt), ("table", tab.round(4).astype(str)),
                                           ("p", mr.interpret_garch(r, dg, lang)), ("table", dg.round(4).astype(str)), ("bullets", refs)],
                                          r.title, is_ar()), "garch.docx", key="g_docx")
        orig = v.split("_", 1)[1] if v.startswith("dln_") else v
        _code(mr.code("garch", y=orig, model=r.model, dist=r.dist, mean=r.mean), "g_code")


def _multi_garch(Y, lang):
    names = st.multiselect(tr("Return series (2–4)", "سلاسل العوائد (2–4)"), list(Y.columns), list(Y.columns)[:min(3, Y.shape[1])], key="mg_v_" + "_".join(Y.columns))
    if not 2 <= len(names) <= 4:
        st.info(tr("Choose 2 to 4 series.", "اختر من 2 إلى 4 سلاسل."))
        return
    c1, c2, c3 = st.columns(3)
    kind = c1.selectbox(tr("Model", "النموذج"), ["DCC", "CCC", "BEKK-diagonal", "BEKK-full"], key="mg_kind")
    um = c2.selectbox(tr("Univariate margins", "النماذج الأحادية"), ["garch", "gjr", "egarch"], key="mg_um",
                      format_func=lambda k: _t(gm.MODELS[k]), disabled=kind.startswith("BEKK"))
    udist = c3.selectbox(tr("Margin distribution", "توزيع النماذج الأحادية"), list(gm.DISTS), index=1, key="mg_ud",
                         format_func=lambda k: _t(gm.DISTS[k]), disabled=kind.startswith("BEKK"))
    if st.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="mg_run"):
        Yv = Y[names].to_numpy()
        with st.spinner(tr("Estimating (this can take a minute for BEKK)…", "جارٍ التقدير (قد يستغرق BEKK دقيقة)…")):
            try:
                if kind == "DCC":
                    m = gm.fit_dcc(Yv, names, um, udist, index=Y.index)
                elif kind == "CCC":
                    m = gm.fit_ccc(Yv, names, um, udist, index=Y.index)
                else:
                    m = gm.fit_bekk(Yv, names, "diagonal" if kind.endswith("diagonal") else "full", index=Y.index)
                _store("mg_res", m, Y)
            except Exception as e:  # noqa: BLE001
                st.error(str(e))
    m = _load("mg_res", Y)
    if m is None or list(m.names) != list(names):
        return
    st.subheader(m.model + " — " + ", ".join(m.names))
    st.metric("Log-likelihood", f"{m.llf:,.2f}")
    if m.model == "DCC":
        st.dataframe(_disp(pd.DataFrame([{"Parameter": k_, "Coef.": m.params[k_], "Std. Err.": m.se[k_], "z": m.params[k_] / m.se[k_]} for k_ in ("a", "b")])).round(4),
                     width="stretch", hide_index=True)
        insight(tr(f"a + b = {m.params['a'] + m.params['b']:.3f}: correlations are " + ("highly persistent." if m.params['a'] + m.params['b'] > 0.95 else "moderately persistent.") +
                   f" a = {m.params['a']:.3f} measures how strongly correlations react to new joint shocks.",
                   f"a + b = {m.params['a'] + m.params['b']:.3f}: الارتباطات " + ("شديدة الاستمرار." if m.params['a'] + m.params['b'] > 0.95 else "متوسطة الاستمرار.") +
                   f" ويقيس a = {m.params['a']:.3f} قوة استجابة الارتباطات للصدمات المشتركة الجديدة."))
    if m.model in ("DCC", "CCC"):
        es = gm.engle_sheppard(np.column_stack([u.z for u in m.uni]), 5)
        insight(tr(f"Engle-Sheppard test of constant correlation: χ²({es['df']}) = {es['stat']:.2f}, {mr._p(es['p'])} — " +
                   ("correlations are time-varying (prefer DCC)." if es["p"] < 0.05 else "constant correlation is not rejected (CCC is adequate)."),
                   f"اختبار إنجل-شيبارد لثبات الارتباط: χ²({es['df']}) = {es['stat']:.2f}، {mr._p(es['p'])} — " +
                   ("الارتباطات متغيرة عبر الزمن (يُفضَّل DCC)." if es["p"] < 0.05 else "لا يُرفض ثبات الارتباط (CCC كافٍ).")))
        ut = pd.concat([u.table().assign(Series=u.label) for u in m.uni])
        with st.expander(tr("Univariate GARCH margins", "النماذج الأحادية")):
            show = ut.copy()
            show["p-value"] = show["p-value"].map(_fp)
            st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
    if m.model.startswith("BEKK"):
        pt = pd.DataFrame([{"Parameter": k_, "Coef.": v_, "Std. Err.": m.se[k_], "z": v_ / m.se[k_]} for k_, v_ in m.params.items()])
        st.dataframe(_disp(pt).round(4), width="stretch", hide_index=True)
        for n_ in m.notes:
            st.caption(n_)
    cf = m.corr_frame()
    fig = go.Figure([go.Scatter(x=cf.index, y=cf[c], mode="lines", name=c, line=dict(color=PAL[i % len(PAL)], width=1.3)) for i, c in enumerate(cf.columns)])
    ch.style_academic(fig, height=420, y_title=tr("Conditional correlation", "الارتباط الشرطي"), rtl=is_ar())
    _chart(fig, "mg_corr", "conditional_correlations")
    st.dataframe(_disp(cf.describe().T[["mean", "min", "max"]].reset_index().rename(columns={"index": "Series"})).round(3), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    i_ = c1.selectbox(tr("Hedge asset", "الأصل المراد تحوطه"), m.names, key="mg_hi")
    j_ = c2.selectbox(tr("with", "باستخدام"), [n_ for n_ in m.names if n_ != i_], key="mg_hj")
    hr = gm.hedge_ratio(m, m.names.index(i_), m.names.index(j_))
    fig = go.Figure(go.Scatter(x=cf.index, y=hr, mode="lines", line=dict(color=PAL[0], width=1.3)))
    ch.style_academic(fig, height=320, y_title=tr("Hedge ratio", "نسبة التحوط"), legend="none", rtl=is_ar())
    _chart(fig, "mg_hr", "hedge_ratio")
    insight(tr(f"Average minimum-variance hedge: {hr.mean():.3f} units of {j_} per unit of {i_} (range {hr.min():.2f} – {hr.max():.2f}).",
               f"متوسط نسبة التحوط ذات التباين الأدنى: {hr.mean():.3f} وحدة من {j_} لكل وحدة من {i_} (المدى {hr.min():.2f} – {hr.max():.2f})."))
    txt, refs = _method("mgarch", "mg")
    excel_download({"Correlations": cf.reset_index()}, "mgarch.xlsx", key="mg_xl")
    origs = [n_.split("_", 1)[1] if n_.startswith("dln_") else n_ for n_ in m.names]
    _code(mr.code("mgarch", vars=origs, model=m.model), "mg_code")
