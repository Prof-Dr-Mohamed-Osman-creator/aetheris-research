# ==========================================
# Aetheris Research - Univariate time series tab (UI) — v1.6, Catalogue §1.7
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from analytics import charts as ch
from analytics import descriptive as ds
from analytics import reg_report as rp
from analytics import regression as rg
from analytics import ts
from analytics import ts_report as tsr
from core.memory import active_name
from core.utils import tr, is_ar, insight, excel_download

PAL = ch.PALETTES["academic"]


_AR_COLS = {"Parameter": "المعلمة", "Coef.": "المعامل", "Std. Err.": "الخطأ المعياري", "p-value": "القيمة الاحتمالية",
            "CI low": "الحد الأدنى للثقة", "CI high": "الحد الأعلى للثقة", "Statistic": "الإحصاءة", "Value": "القيمة",
            "5% critical": "القيمة الحرجة 5%", "Lags": "الإبطاءات", "Test": "الاختبار", "Model": "النموذج", "Estimator": "المقدِّر",
            "step": "الخطوة", "forecast": "التنبؤ", "lower": "الحد الأدنى", "upper": "الحد الأعلى", "se": "الخطأ المعياري",
            "time": "الزمن", "forecast (level, median)": "التنبؤ بالمستوى (الوسيط)", "forecast (level, mean)": "التنبؤ بالمستوى (المتوسط)",
            "lower (level)": "الحد الأدنى (المستوى)", "upper (level)": "الحد الأعلى (المستوى)", "Valid": "صالح",
            "Ljung-Box p": "p ليونج-بوكس", "Frequency": "التكرار", "Unit root at this frequency": "جذر الوحدة عند هذا التكرار",
            "Metric": "المقياس", "Series": "السلسلة", "Deterministics": "الحدود المحددة", "H0": "فرض العدم",
            "Reject H0 (5%)": "رفض فرض العدم (5%)"}
_AR_VALS = {"level": "المستوى", "Δ (first difference)": "الفرق الأول Δ", "Δ^2": "الفرق الثاني Δ²", "constant": "ثابت",
            "constant + trend": "ثابت واتجاه", "none": "بدون", "unit root": "جذر وحدة", "stationary": "سكون",
            "not rejected": "لا يُرفض", "rejected": "يُرفض"}


def _disp(df: pd.DataFrame) -> pd.DataFrame:
    """Arabic column headers and categorical labels for on-screen tables (exports keep English headers)."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d")
    if not is_ar():
        return out
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].map(lambda v: _AR_VALS.get(v, v) if isinstance(v, str) else v)
    return out.rename(columns=_AR_COLS)


def _tlabel(t) -> str:
    return str(t)[:10] if isinstance(t, pd.Timestamp) else str(t)


def _lang():
    return "ar" if is_ar() else "en"


def _chart(fig, key, name="figure"):
    st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, name), key=key)


def _fp(p):
    return "" if p is None or pd.isna(p) else ("< 0.001" if p < 0.001 else f"{p:.4f}")


def _code(codes: dict, key: str):
    t = st.tabs(list(codes))
    for tab, (k, v) in zip(t, codes.items()):
        with tab:
            st.code(v, language={"Python": "python", "R": "r", "Stata": "stata"}[k])


def _method_block(kind, key, **kw):
    txt, refs = tsr.methodology(kind, _lang(), **kw)
    with st.expander(tr("📝 Methodology paragraph & references", "📝 فقرة المنهجية والمراجع"), expanded=False):
        st.text_area("m", txt, height=150, key=f"{key}_meth", label_visibility="collapsed")
        st.markdown("\n".join(f"- {x}" for x in refs))
    return txt, refs


def _time_cols(df):
    out = []
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            out.append(c)
        elif pd.api.types.is_numeric_dtype(df[c]) and df[c].dropna().between(1000, 3000).all() and df[c].nunique() > 5:
            out.append(c)
        elif df[c].dtype == object and pd.to_datetime(df[c].head(20), errors="coerce").notna().mean() > 0.9:
            out.append(c)
    return out


def _series(df):
    tcols = _time_cols(df) + [c for c in ds.time_columns(df) if c not in _time_cols(df)]
    if not tcols:
        st.info(tr("No date / year column found. Time-series tools need a time variable.",
                   "لا يوجد عمود للتاريخ أو السنة. تحتاج أدوات السلاسل الزمنية إلى متغير زمني."))
        return None
    num = [c for c in ds.numeric_columns(df) if c not in tcols]
    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1, 1])
    with c1:
        t = st.selectbox(tr("Time variable", "المتغير الزمني"), tcols, key="ts_t")
    with c2:
        v = st.selectbox(tr("Series", "السلسلة"), num, key="ts_v")
    with c3:
        transform = st.radio(tr("Transform", "التحويل"), ["level", "log"], horizontal=True, key="ts_tr",
                             format_func=lambda x: tr("Level", "المستوى") if x == "level" else "ln")
    with c4:
        per = st.selectbox(tr("Seasonal period", "طول الموسم"), ["auto", 1, 4, 12, 52, 7], key="ts_per",
                           format_func=lambda p: tr("auto-detect", "تلقائي") if p == "auto" else f"{p} — {ts.FREQS[p][1 if is_ar() else 0]}")
    filters = {}
    cats = [c for c in ds.categorical_columns(df) if c != t]
    if cats:
        with st.expander(tr("Select one series from a panel (filters)", "اختيار سلسلة واحدة من بيانات بانل (مرشحات)"),
                         expanded=bool(df[t].duplicated().any())):
            cols = st.columns(min(3, len(cats)))
            for i, c in enumerate(cats):
                with cols[i % len(cols)]:
                    val = st.selectbox(c, ["—"] + sorted(df[c].dropna().astype(str).unique().tolist()), key=f"ts_f_{c}")
                    if val != "—":
                        filters[c] = val
    try:
        s = ts.prepare(df, t, v, filters, transform, None if per == "auto" else int(per))
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return None
    for n in s.notes:
        st.warning(n)
    a, b = _tlabel(s.y.index[0]), _tlabel(s.y.index[-1])
    st.caption(tr(f"{len(s.y)} observations · period s = {s.s} · {a} → {b}",
                  f"{len(s.y)} مشاهدة · طول الموسم s = {s.s} · من \u2066{a}\u2069 إلى \u2066{b}\u2069"))
    return s, t, v, transform


# ==========================================================
def render_ts(df):
    st.caption("💡 " + tr("Choose a time variable and a series. For panel data (several units) use the filters to isolate "
                         "one unit. Work in logs for prices / indices so that differences are growth rates.",
                         "اختر المتغير الزمني والسلسلة. وفي بيانات البانل (عدة وحدات) استخدم المرشحات لعزل وحدة واحدة. "
                         "واستخدم اللوغاريتم للأسعار والأرقام القياسية لتصبح الفروق معدلات نمو."))
    out = _series(df)
    if out is None:
        return
    s, tcol, vcol, transform = out
    tabs = st.tabs([tr("📈 Explore & decompose", "📈 الاستكشاف والتفكيك"), tr("🧪 Unit roots", "🧪 جذور الوحدة"),
                    tr("🔁 Seasonal unit roots", "🔁 الجذور الموسمية"), tr("🧭 Box-Jenkins ARIMA", "🧭 منهجية بوكس-جنكنز ARIMA"),
                    tr("🌊 Long memory", "🌊 الذاكرة الطويلة")])
    with tabs[0]:
        _explore(s, tcol, vcol, transform)
    with tabs[1]:
        _unit_roots(s, tcol, vcol, transform)
    with tabs[2]:
        _hegy(s, tcol, vcol, transform)
    with tabs[3]:
        _arima(s, tcol, vcol, transform)
    with tabs[4]:
        _long_memory(s, tcol, vcol, transform)


def _explore(s, tcol, vcol, transform):
    y = s.y
    fig = go.Figure(go.Scatter(x=y.index, y=y.values, mode="lines", line=dict(color=PAL[0], width=2)))
    ch.style_academic(fig, y_title=s.name, height=360, legend="none", rtl=is_ar())
    _chart(fig, "ts_line", "series")
    g = y.diff().dropna()
    stt = pd.DataFrame([{tr("Mean", "المتوسط"): y.mean(), tr("Std. dev.", "الانحراف المعياري"): y.std(),
                         tr("Mean change", "متوسط التغير"): g.mean(), tr("Std. of change", "انحراف التغير"): g.std(),
                         tr("First-order autocorrelation", "الارتباط الذاتي من الرتبة الأولى"): y.autocorr(1)}])
    st.dataframe(stt.round(4), width="stretch", hide_index=True)
    if transform == "log":
        k = {12: 12, 4: 4}.get(s.s, 1)
        cagr = (np.exp((y.iloc[-1] - y.iloc[0]) / (len(y) - 1) * (k if k else 1)) - 1) * 100
        insight(tr(f"Average compound growth: {cagr:.2f}% per year (log-linear endpoints).",
                   f"متوسط معدل النمو المركب: {cagr:.2f}% سنوياً (من طرفي السلسلة اللوغاريتمية)."))
    c1, c2 = st.columns(2)
    with c1:
        method = st.radio(tr("Decomposition", "طريقة التفكيك"), ["stl", "classical"], horizontal=True, key="ts_dm",
                          format_func=lambda m: "STL" if m == "stl" else tr("Classical (moving averages)", "كلاسيكية (متوسطات متحركة)"))
    with c2:
        model = st.radio(tr("Model", "النموذج"), ["additive", "multiplicative"], horizontal=True, key="ts_dmod",
                         format_func=lambda m: tr("Additive", "تجميعي") if m == "additive" else tr("Multiplicative", "ضربي"),
                         disabled=method == "stl")
    if s.s > 1:
        try:
            dec = ts.decompose(y, s.s, method, model)
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                subplot_titles=[tr("Observed", "المشاهدة"), tr("Trend", "الاتجاه"), tr("Seasonal", "الموسمية"),
                                                tr("Remainder", "البواقي")])
            for i, c in enumerate(["observed", "trend", "seasonal", "remainder"], 1):
                fig.add_trace(go.Scatter(x=dec.index, y=dec[c], mode="lines", line=dict(color=PAL[i - 1], width=1.6)), i, 1)
            ch.style_academic(fig, height=720, legend="none", rtl=is_ar())
            _chart(fig, "ts_dec", "decomposition")
            insight(tr(f"Strength of trend = {dec.attrs['strength_trend']:.2f}, strength of seasonality = "
                       f"{dec.attrs['strength_seasonal']:.2f} (0 = none, 1 = dominant; Wang, Smith & Hyndman 2006).",
                       f"قوة الاتجاه = {dec.attrs['strength_trend']:.2f}، وقوة الموسمية = {dec.attrs['strength_seasonal']:.2f} "
                       "(0 = منعدمة، 1 = مهيمنة)."))
        except Exception as e:  # noqa: BLE001
            st.info(str(e))
    st.markdown("**" + tr("Trend–cycle filters", "مرشحات فصل الاتجاه عن الدورة") + "**")
    lam = st.number_input(tr("HP smoothing λ", "معامل التمهيد λ لمرشح HP"), value=float(ts.HP_LAMBDA.get(s.s, 1600.0)),
                          key="ts_lam", help=tr("Defaults: 100 annual, 1600 quarterly, 14400 monthly (Ravn-Uhlig: 6.25 / 1600 / 129600).",
                                                "القيم الافتراضية: 100 سنوي، 1600 ربع سنوي، 14400 شهري (رافن-أوليج: 6.25 / 1600 / 129600)."))
    fl = ts.filters(y, s.s, lam)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[tr("Series and HP trend", "السلسلة واتجاه HP"), tr("Cycles", "الدورات")])
    fig.add_trace(go.Scatter(x=fl.index, y=fl["observed"], name=tr("Observed", "المشاهدة"), line=dict(color="#999")), 1, 1)
    fig.add_trace(go.Scatter(x=fl.index, y=fl["HP trend"], name="HP trend", line=dict(color=PAL[0], width=2.5)), 1, 1)
    for j, c in enumerate([c for c in fl.columns if "cycle" in c]):
        fig.add_trace(go.Scatter(x=fl.index, y=fl[c], name=c, line=dict(color=PAL[j + 1], width=1.8)), 2, 1)
    ch.style_academic(fig, height=600, rtl=is_ar())
    _chart(fig, "ts_filt", "filters")
    _method_block("filters", "tsf")
    excel_download({"filters": fl.reset_index()}, "filters.xlsx", key="ts_fxl")
    _code(tsr.code("filters", vcol, tcol, s=s.s, log=transform == "log"), "tsf_code")


def _unit_roots(s, tcol, vcol, transform):
    lang = _lang()
    st.caption("💡 " + tr("ADF, PP and DF-GLS test H₀: unit root; KPSS tests H₀: stationarity. Use 'constant + trend' for "
                         "trending levels and 'constant' for differences / growth rates.",
                         "اختبارات ADF وPP وDF-GLS فرضها العدم وجود جذر الوحدة؛ أما KPSS ففرضه العدم السكون. استخدم «ثابت "
                         "واتجاه» للمستويات ذات الاتجاه، و«ثابت» للفروق ومعدلات النمو."))
    reg = st.radio(tr("Deterministic terms for the level", "الحدود المحددة لاختبار المستوى"), ["ct", "c", "n"], horizontal=True,
                   key="ur_reg", format_func=lambda r: ts.REG[r][1 if is_ar() else 0])
    io = ts.integration_order(s.y.values, reg)
    tab = io["table"].copy()
    show = tab.copy()
    show["p-value"] = show["p-value"].map(_fp)
    show["Reject H0 (5%)"] = show["Reject H0 (5%)"].map({True: tr("✅ reject", "✅ يُرفض"), False: tr("— not rejected", "— لا يُرفض")})
    st.dataframe(_disp(show).round(4), width="stretch", hide_index=True)
    st.caption(tr("p-values: MacKinnon (2010) response surfaces (ADF, PP); KPSS interpolated in the KPSS (1992) table; DF-GLS "
                  "critical values from Elliott, Rothenberg & Stock (1996).",
                  "القيم الاحتمالية: أسطح استجابة ماكينون (2010) لاختباري ADF وPP؛ ومُستكمَلة من جدول KPSS (1992)؛ "
                  "والقيم الحرجة لاختبار DF-GLS من إليوت وروثنبرج وستوك (1996)."))
    insight(tsr.interpret_unit_roots(io, s.name, lang))
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**" + tr("Ng-Perron modified tests (level)", "اختبارات نج-بيرون المعدلة (المستوى)") + "**")
        npr = ts.ng_perron(s.y.values, "ct" if reg == "ct" else "c")
        rows = [{"Statistic": k, "Value": v, "5% critical": npr["crit"][k][1],
                 tr("Reject unit root", "رفض جذر الوحدة"): "✅" if npr["reject_5"][k] else "—"} for k, v in npr["values"].items()]
        st.dataframe(_disp(pd.DataFrame(rows)).round(4), width="stretch", hide_index=True)
        st.caption(tr(f"GLS detrending, lags = {npr['lags']} (MAIC). Asymptotic critical values, Ng & Perron (2001).",
                      f"إزالة الاتجاه بطريقة GLS، والإبطاءات = {npr['lags']} (MAIC). القيم الحرجة التقاربية لنج وبيرون (2001)."))
    with c2:
        st.markdown("**" + tr("Zivot-Andrews (one endogenous break)", "زيفوت-أندروز (انكسار داخلي واحد)") + "**")
        zr = st.radio(tr("Break in", "الانكسار في"), ["c", "t", "ct"], horizontal=True, key="za_r",
                      format_func=lambda r: {"c": tr("intercept", "الثابت"), "t": tr("trend", "الاتجاه"), "ct": tr("both", "كليهما")}[r])
        z = ts.zivot_andrews(s.y.values, zr)
        bdate = s.y.index[z["break_pos"] - 1] if 0 < z["break_pos"] <= len(s.y) else "—"
        st.dataframe(_disp(pd.DataFrame([{"Statistic": z["stat"], "p-value": _fp(z["p"]), "5% critical": z["crit"]["5%"],
                                          tr("Break date", "تاريخ الانكسار"): str(bdate)[:10], "Lags": z["lags"]}])).round(4),
                     width="stretch", hide_index=True)
        insight(tr(f"Break at {str(bdate)[:10]}: unit root " + ("rejected — the series is stationary around a broken trend."
                                                                 if z["p"] < 0.05 else "not rejected even allowing for a break."),
                   f"الانكسار عند {str(bdate)[:10]}: " + ("يُرفض جذر الوحدة — السلسلة مستقرة حول اتجاه منكسر."
                                                         if z["p"] < 0.05 else "لا يُرفض جذر الوحدة حتى مع السماح بانكسار.")))
    txt, refs = _method_block("unitroot", "tsu")
    c1, c2 = st.columns(2)
    with c1:
        excel_download({"Unit roots": tab, "Ng-Perron": pd.DataFrame(rows)}, "unit_roots.xlsx", key="ur_xl")
    with c2:
        st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"),
                           rp.to_docx([("h", tr("Methodology", "المنهجية")), ("p", txt), ("table", tab.astype(str)),
                                       ("p", tsr.interpret_unit_roots(io, s.name, lang)), ("bullets", refs)],
                                      tr(f"Unit-root analysis — {s.name}", f"تحليل جذور الوحدة — {s.name}"), is_ar()),
                           "unit_roots.docx", key="ur_docx")
    _code(tsr.code("unitroot", vcol, tcol, s=s.s, log=transform == "log", regression=reg), "tsu_code")


def _hegy(s, tcol, vcol, transform):
    if s.s != 4:
        st.info(tr("HEGY is implemented for quarterly data (seasonal period 4). Aggregate monthly data to quarters or "
                   "set the period to 4.", "اختبار HEGY مُنفَّذ للبيانات الربع سنوية (طول الموسم 4). اجمع البيانات الشهرية "
                   "إلى أرباع أو اضبط طول الموسم على 4."))
        return
    c1, c2 = st.columns(2)
    with c1:
        det = st.selectbox(tr("Deterministic terms", "الحدود المحددة"), ["cs", "cst", "c", "ct"], key="hg_det",
                           format_func=lambda d: {"cs": tr("constant + seasonal dummies", "ثابت ومتغيرات موسمية"),
                                                  "cst": tr("constant + seasonal dummies + trend", "ثابت وموسمية واتجاه"),
                                                  "c": tr("constant", "ثابت"), "ct": tr("constant + trend", "ثابت واتجاه")}[d])
    with c2:
        lags = st.number_input(tr("Augmentation lags", "إبطاءات التعزيز"), 0, 8, 0, key="hg_lags")
    if st.button(tr("▶️ Run HEGY (Monte Carlo critical values)", "▶️ تشغيل HEGY (قيم حرجة بالمحاكاة)"), key="hg_run"):
        with st.spinner(tr("Simulating the null distribution…", "جارٍ محاكاة توزيع فرض العدم…")):
            h = ts.hegy(s.y.values, det, int(lags))
        st.dataframe(_disp(h).round(4), width="stretch", hide_index=True)
        nr = h[h["Unit root at this frequency"] == "not rejected"]["Test"].tolist()
        insight(tr("Unit roots not rejected at: " + (", ".join(nr) if nr else "none") +
                   ". A zero-frequency root calls for Δ; seasonal roots call for the seasonal difference Δ₄ (or SARIMA with D = 1).",
                   "لا تُرفض جذور الوحدة عند: " + ("، ".join(nr) if nr else "لا شيء") +
                   ". الجذر عند التكرار الصفري يستلزم الفرق الأول Δ، والجذور الموسمية تستلزم الفرق الموسمي Δ₄ (أو SARIMA مع D = 1)."))
    _method_block("seasonal", "tsh")
    _code(tsr.code("seasonal", vcol, tcol, s=4, log=transform == "log"), "tsh_code")


def _arima_to_reg(r: ts.ArimaResult, label: str = "y") -> rg.RegResult:
    res = r.res
    names = list(res.param_names)
    ci = res.conf_int()
    rr = rg.RegResult(name=r.name, method="arima", cov_type="nonrobust", design=None,
                      params=pd.Series(np.asarray(res.params), index=names), bse=pd.Series(np.asarray(res.bse), index=names),
                      tvalues=pd.Series(np.asarray(res.tvalues), index=names), pvalues=pd.Series(np.asarray(res.pvalues), index=names),
                      ci=pd.DataFrame(np.asarray(ci), index=names, columns=["low", "high"]),
                      cov=pd.DataFrame(np.asarray(res.cov_params()), index=names, columns=names), dist="z",
                      df_resid=r.stats["N"] - len(names), nobs=r.stats["N"], y=np.asarray(r.y), fitted=np.zeros(1),
                      resid=np.zeros(1), wresid=np.zeros(1), wexog=np.zeros((1, 1)))
    rr.llf, rr.aic, rr.bic = r.stats["Log-likelihood"], r.stats["AIC"], r.stats["BIC"]
    rr.extra.update({"y_label": label, "stats": {"Log-likelihood": rr.llf, "AICc": r.stats["AICc"]}, "family": "arima"})
    return rr


def _arima(s, tcol, vcol, transform):
    lang = _lang()
    y = s.y
    st.markdown("#### " + tr("1 — Identification", "1 — التعرف على النموذج"))
    d_id = st.radio(tr("Show ACF / PACF of", "عرض دالتي الارتباط الذاتي والجزئي لـ"), [0, 1, 2], horizontal=True, key="ar_did",
                    format_func=lambda d: {0: tr("level", "المستوى"), 1: "Δ", 2: "Δ²"}[d])
    x = y.copy()
    for _ in range(d_id):
        x = x.diff().dropna()
    if s.s > 1 and st.checkbox(tr(f"Also seasonal difference Δ{s.s}", f"مع الفرق الموسمي Δ{s.s}"), key="ar_sd"):
        x = x.diff(s.s).dropna()
    af = ts.acf_pacf(x.values, min(36, len(x) // 3))
    fig = make_subplots(rows=1, cols=2, subplot_titles=["ACF", "PACF"])
    for j, c in enumerate(["ACF", "PACF"], 1):
        fig.add_trace(go.Bar(x=af["lag"], y=af[c], marker_color=PAL[0], width=0.3), 1, j)
        band = af[f"{c} band"]
        fig.add_trace(go.Scatter(x=af["lag"], y=band, mode="lines", line=dict(color=PAL[1], dash="dash")), 1, j)
        fig.add_trace(go.Scatter(x=af["lag"], y=-band, mode="lines", line=dict(color=PAL[1], dash="dash")), 1, j)
    ch.style_academic(fig, height=360, legend="none", rtl=is_ar())
    _chart(fig, "ar_acf", "acf_pacf")
    st.caption(tr("Cut-off in the PACF after lag p and tailing ACF → AR(p); cut-off in the ACF after lag q → MA(q); both tailing → ARMA. "
                  "Ljung-Box Q at lag 12: ", "انقطاع PACF بعد الإبطاء p وتلاشي ACF ← AR(p)؛ وانقطاع ACF بعد q ← MA(q)؛ وتلاشيهما معاً ← ARMA. "
                  "إحصاءة ليونج-بوكس عند الإبطاء 12: ")
               + f"{af['Ljung-Box Q'].iloc[min(11, len(af) - 1)]:.2f} ({tsr._p(af['p-value'].iloc[min(11, len(af) - 1)])})")
    st.markdown("#### " + tr("2 — Automatic order search", "2 — البحث التلقائي عن الرتب"))
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        d_auto = st.selectbox("d", ["auto", 0, 1, 2], index=0, key="ar_d_auto",
                              format_func=lambda d: tr("from unit-root tests", "من اختبارات جذر الوحدة") if d == "auto" else str(d))
    with c2:
        D_auto = st.selectbox("D", [0, 1], key="ar_D_auto") if s.s > 1 else 0
    with c3:
        mp = st.number_input("max p = max q", 1, 5, 3, key="ar_max")
    with c4:
        crit = st.selectbox(tr("Criterion", "المعيار"), ["AICc", "AIC", "BIC"], key="ar_crit")
    if st.button(tr("🔎 Search", "🔎 بحث"), key="ar_auto"):
        with st.spinner(tr("Estimating candidate models…", "جارٍ تقدير النماذج المرشحة…")):
            a = ts.auto_arima(y, s.s, None if d_auto == "auto" else int(d_auto), int(D_auto), int(mp), int(mp),
                              1 if s.s > 1 else 0, 1 if s.s > 1 else 0, crit)
        st.session_state["ar_search"] = (a, active_name(), s.name)
    srch = st.session_state.get("ar_search")
    if srch and srch[1] == active_name() and srch[2] == s.name:
        st.dataframe(_disp(srch[0].head(12)).round(3), width="stretch", hide_index=True)
        best = srch[0].iloc[0]
        st.caption(tr(f"Best by {crit}: {best['Model']} — copy its orders into step 3.", f"الأفضل وفق {crit}: {best['Model']} — انقل رتبه إلى الخطوة 3."))
    st.markdown("#### " + tr("3 — Estimation, diagnostics and forecasting", "3 — التقدير والتشخيص والتنبؤ"))
    c = st.columns(7)
    p = c[0].number_input("p", 0, 6, 1, key="ar_p")
    d = c[1].number_input("d", 0, 2, 1, key="ar_d")
    q = c[2].number_input("q", 0, 6, 1, key="ar_q")
    P = c[3].number_input("P", 0, 2, 0, key="ar_P", disabled=s.s < 2)
    D = c[4].number_input("D", 0, 1, 0, key="ar_D", disabled=s.s < 2)
    Q = c[5].number_input("Q", 0, 2, 0, key="ar_Q", disabled=s.s < 2)
    trend = c[6].selectbox(tr("Constant / drift", "ثابت / انجراف"), ["auto", "n", "c", "t", "ct"], key="ar_tr",
                           format_func=lambda t: {"auto": tr("auto", "تلقائي"), "n": tr("none", "بدون"), "c": tr("constant", "ثابت"),
                                                  "t": tr("drift (d=1)", "انجراف (d=1)"), "ct": tr("constant + trend", "ثابت واتجاه")}[t])
    h = st.slider(tr("Forecast horizon", "أفق التنبؤ"), 1, 60, 12 if s.s >= 4 else 5, key="ar_h")
    if st.button(tr("▶️ Estimate", "▶️ تقدير"), type="primary", key="ar_run"):
        try:
            with st.spinner(tr("Estimating…", "جارٍ التقدير…")):
                r = ts.fit_arima(y, (int(p), int(d), int(q)), (int(P), int(D), int(Q)), s.s,
                                 None if trend == "auto" else trend, s.name, transform)
            st.session_state["ar_res"] = (r, active_name(), s.name)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
    res = st.session_state.get("ar_res")
    if not (res and res[1] == active_name() and res[2] == s.name):
        return
    r = res[0]
    diag = ts.residual_tests(r)
    st.subheader(r.name)
    tab = r.table.copy()
    tab["p-value"] = tab["p-value"].map(_fp)
    st.dataframe(_disp(tab).round(4), width="stretch", hide_index=True)
    m = st.columns(5)
    for col, (k, v) in zip(m, [(k, r.stats[k]) for k in ("N", "Log-likelihood", "AIC", "AICc", "BIC")]):
        col.metric(k, "‎" + (f"{v:,.2f}" if isinstance(v, float) else str(v)))
    insight(tsr.interpret_arima(r, diag, lang))
    sub = st.tabs([tr("🩺 Diagnostics", "🩺 التشخيص"), tr("🔮 Forecast", "🔮 التنبؤ"), tr("🎯 Out-of-sample test", "🎯 الاختبار خارج العينة"),
                   tr("📝 Methodology & export", "📝 المنهجية والتصدير")])
    with sub[0]:
        dt = pd.DataFrame([
            {"Test": f"Ljung-Box Q({diag['lb_lags']})", "Statistic": diag["lb_stat"], "df": str(diag["lb_df"]), "p-value": _fp(diag["lb_p"])},
            {"Test": "Jarque-Bera", "Statistic": diag["jb"], "df": "2", "p-value": _fp(diag["jb_p"])},
            {"Test": "ARCH-LM (Engle)", "Statistic": diag["arch"], "df": "", "p-value": _fp(diag["arch_p"])}])
        st.dataframe(_disp(dt).round(4), width="stretch", hide_index=True)
        e = diag["resid"]
        ra = ts.acf_pacf(e, min(24, len(e) // 3))
        fig = make_subplots(rows=1, cols=3, subplot_titles=[tr("Residuals", "البواقي"), tr("Residual ACF", "دالة الارتباط الذاتي للبواقي"),
                                                            tr("Inverse roots", "الجذور العكسية")])
        fig.add_trace(go.Scatter(y=e, mode="lines", line=dict(color=PAL[0], width=1)), 1, 1)
        fig.add_trace(go.Bar(x=ra["lag"], y=ra["ACF"], marker_color=PAL[0], width=0.3), 1, 2)
        fig.add_trace(go.Scatter(x=ra["lag"], y=ra["ACF band"], line=dict(color=PAL[1], dash="dash")), 1, 2)
        fig.add_trace(go.Scatter(x=ra["lag"], y=-ra["ACF band"], line=dict(color=PAL[1], dash="dash")), 1, 2)
        th = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(go.Scatter(x=np.cos(th), y=np.sin(th), mode="lines", line=dict(color="#999")), 1, 3)
        arr, mar = ts.roots(r.res)
        for rts, sym, col_, lab in ((arr, "circle", PAL[0], "AR"), (mar, "x", PAL[1], "MA")):
            if len(rts):
                inv = 1 / rts
                fig.add_trace(go.Scatter(x=inv.real, y=inv.imag, mode="markers", marker=dict(symbol=sym, size=10, color=col_), name=lab), 1, 3)
        fig.update_xaxes(range=[-1.2, 1.2], row=1, col=3)
        fig.update_yaxes(range=[-1.2, 1.2], row=1, col=3)
        ch.style_academic(fig, height=380, legend="none", rtl=is_ar())
        _chart(fig, "ar_diag", "arima_diagnostics")
        st.caption(tr("Inverse AR roots inside the unit circle → stationary; inverse MA roots inside → invertible.",
                      "الجذور العكسية لحدود AR داخل دائرة الوحدة ← نموذج مستقر؛ ولحدود MA داخلها ← قابل للانعكاس."))
    with sub[1]:
        fc = ts.forecast(r, int(h))
        level = transform == "log" and st.checkbox(tr("Show forecasts in original units (exp)", "عرض التنبؤ بالوحدات الأصلية (exp)"),
                                                   True, key="ar_lev")
        hist = np.exp(y) if level else y
        fcol, lo, hi = ("forecast (level, mean)", "lower (level)", "upper (level)") if level else ("forecast", "lower", "upper")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist.index[-min(len(hist), 10 * max(h, 4)):], y=hist.values[-min(len(hist), 10 * max(h, 4)):],
                                 mode="lines", name=tr("Observed", "المشاهدة"), line=dict(color="#555")))
        fig.add_trace(go.Scatter(x=np.r_[fc["time"], fc["time"][::-1]], y=np.r_[fc[hi], fc[lo][::-1]], fill="toself",
                                 fillcolor=PAL[0], opacity=0.18, line=dict(width=0), name="95% CI"))
        fig.add_trace(go.Scatter(x=fc["time"], y=fc[fcol], mode="lines+markers", name=tr("Forecast", "التنبؤ"),
                                 line=dict(color=PAL[0], width=2.5)))
        ch.style_academic(fig, y_title=vcol if level else s.name, height=420, rtl=is_ar())
        _chart(fig, "ar_fc", "forecast")
        st.dataframe(_disp(fc).round(4), width="stretch", hide_index=True)
        last = float(hist.iloc[-1])
        f1, fh = float(fc[fcol].iloc[0]), float(fc[fcol].iloc[-1])
        st.markdown("### " + tr("🏛️ For decision-makers", "🏛️ لصانع القرار"))
        insight(tr(f"Next period: {f1:,.4g} (95% CI {fc[lo].iloc[0]:,.4g} – {fc[hi].iloc[0]:,.4g}), versus {last:,.4g} last observed. "
                   f"At the end of the horizon ({h} periods): {fh:,.4g}, a change of {100 * (fh / last - 1):+.2f}%. "
                   "Uncertainty widens with the horizon; update the forecast as new data arrive.",
                   f"الفترة القادمة: {f1:,.4g} (فترة الثقة 95%: {fc[lo].iloc[0]:,.4g} – {fc[hi].iloc[0]:,.4g})، مقابل {last:,.4g} آخر قيمة. "
                   f"وفي نهاية الأفق ({h} فترات): {fh:,.4g}، بتغير {100 * (fh / last - 1):+.2f}%. "
                   "ويتسع عدم اليقين مع طول الأفق؛ حدّث التنبؤ مع ورود بيانات جديدة."))
        if level:
            st.caption(tr("Level forecasts: mean = exp(μ + σ²/2) (bias-corrected); intervals = exp of the log-scale limits.",
                          "تنبؤات المستوى: المتوسط = exp(μ + σ²/2) (مصحح التحيز)، والحدود = exp لحدود المقياس اللوغاريتمي."))
    with sub[2]:
        test = st.slider(tr("Hold-out size (last observations)", "حجم عينة الاختبار (آخر المشاهدات)"), 4, max(5, len(y) // 4),
                         min(24, max(5, len(y) // 6)), key="ar_ho")
        try:
            ho = ts.holdout(y, r.order, r.seasonal, s.s, int(test), r.trend)
            st.dataframe(_disp(ho["table"]).round(4), width="stretch", hide_index=True)
            dm = ho["dm"]
            insight(tr(f"Diebold-Mariano (HLN) test vs naive: statistic = {dm['stat']:.3f}, p = {dm['p']:.4f} — "
                       + ("the model forecasts significantly better than the random walk." if dm["p"] < 0.05 and dm["stat"] < 0 else
                          "no significant difference from the random-walk benchmark."),
                       f"اختبار ديبولد-ماريانو (HLN) مقابل النموذج الساذج: الإحصاءة = {dm['stat']:.3f}، p = {dm['p']:.4f} — "
                       + ("النموذج يتنبأ أفضل معنوياً من السير العشوائي." if dm["p"] < 0.05 and dm["stat"] < 0 else
                          "لا فرق معنوي عن السير العشوائي.")))
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ho["time"], y=ho["actual"], mode="lines+markers", name=tr("Actual", "الفعلي"), line=dict(color="#333")))
            fig.add_trace(go.Scatter(x=ho["time"], y=ho["forecast"], mode="lines+markers", name=r.name, line=dict(color=PAL[0])))
            ch.style_academic(fig, height=360, rtl=is_ar())
            _chart(fig, "ar_ho_fig", "holdout")
        except Exception as e:  # noqa: BLE001
            st.info(str(e))
    with sub[3]:
        txt, refs = _method_block("arima", "tsa", name=r.name)
        rr = _arima_to_reg(r, s.name)
        jt, note = rp.journal_table([rr], [r.name], stats_rows=("N", "Log-likelihood", "AIC", "BIC"), lang=lang)
        c1, c2, c3 = st.columns(3)
        with c1:
            excel_download({"ARIMA": r.table, "Forecast": ts.forecast(r, int(h)), "Diagnostics": pd.DataFrame([{k: v for k, v in diag.items() if k != "resid"}])},
                           "arima.xlsx", key="ar_xl")
        with c2:
            st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"),
                               rp.to_docx([("h", tr("Methodology", "المنهجية")), ("p", txt), ("table", jt), ("note", note),
                                           ("p", tsr.interpret_arima(r, diag, lang)), ("table", ts.forecast(r, int(h)).astype(str)),
                                           ("bullets", refs)], r.name, is_ar()), "arima.docx", key="ar_docx")
        with c3:
            if st.button(tr("💾 Save to journal table", "💾 حفظ في جدول المجلات"), key="ar_save"):
                st.session_state.setdefault("eco_models", {})[f"{r.name} — {s.name}"] = {
                    "r": rr, "diag": {"table": pd.DataFrame(), "problems": {}, "detail": {}}, "dataset": active_name()}
                st.success(tr("Saved.", "تم الحفظ."))
        _code(tsr.code("arima", vcol, tcol, s=s.s, log=transform == "log", order=r.order, seasonal=r.seasonal), "tsa_code")


def _long_memory(s, tcol, vcol, transform):
    y = s.y.values
    st.caption("💡 " + tr("Long memory: autocorrelations decay hyperbolically. d = 0 short memory, 0 < d < 0.5 stationary long "
                         "memory, 0.5 ≤ d < 1 non-stationary but mean-reverting, d = 1 unit root.",
                         "الذاكرة الطويلة: تتلاشى الارتباطات الذاتية ببطء (قطعياً). d = 0 ذاكرة قصيرة، و0 < d < 0.5 ذاكرة طويلة "
                         "مستقرة، و0.5 ≤ d < 1 غير مستقرة لكنها تعود للمتوسط، وd = 1 جذر وحدة."))
    c1, c2, c3 = st.columns(3)
    with c1:
        bw = st.slider(tr("GPH bandwidth m = n^a", "عرض النطاق لـ GPH: m = n^a"), 0.4, 0.8, 0.5, 0.05, key="lm_bw")
    with c2:
        p = st.number_input("ARFIMA p", 0, 3, 0, key="lm_p")
    with c3:
        q = st.number_input("ARFIMA q", 0, 3, 0, key="lm_q")
    g = ts.gph(y, bw)
    lw = ts.local_whittle(y, max(bw, 0.6))
    af = ts.arfima_whittle(y, int(p), int(q))
    rows = [{"Estimator": "GPH (Geweke & Porter-Hudak)", "d": g["d"], "Std. Err.": g["se"], "m": str(g["m"]), "p (d = 0)": _fp(g["p (d = 0)"])},
            {"Estimator": "Local Whittle (Robinson)", "d": lw["d"], "Std. Err.": lw["se"], "m": str(lw["m"]), "p (d = 0)": _fp(lw["p (d = 0)"])},
            {"Estimator": f"ARFIMA({p},d,{q}) — Whittle", "d": af["d"], "Std. Err.": float(af["table"]["Std. Err."].iloc[0]),
             "m": "", "p (d = 0)": _fp(float(af["table"]["p-value"].iloc[0]))}]
    st.dataframe(_disp(pd.DataFrame(rows)).round(4), width="stretch", hide_index=True)
    if p or q:
        st.dataframe(_disp(af["table"]).round(4), width="stretch", hide_index=True)
    dd = af["d"]
    kind = (tr("short memory", "ذاكرة قصيرة") if abs(dd) < 0.1 else tr("stationary long memory", "ذاكرة طويلة مستقرة") if dd < 0.5
            else tr("non-stationary, mean-reverting", "غير مستقرة لكنها تعود للمتوسط") if dd < 1 else tr("unit root / explosive", "جذر وحدة أو انفجارية"))
    insight(tr(f"Estimated d ≈ {dd:.3f} → <b>{kind}</b>. Shocks die out at a hyperbolic rate; ARMA models understate persistence "
               "and ARIMA (d = 1) overstates it.",
               f"قيمة d المقدرة ≈ {dd:.3f} ← <b>{kind}</b>. تتلاشى الصدمات بمعدل قطعي بطيء؛ فنماذج ARMA تقلل من تقدير الاستمرارية "
               "ونماذج ARIMA (d = 1) تبالغ فيها."))
    ac = ts.acf_pacf(y, min(60, len(y) // 3))
    fig = go.Figure(go.Bar(x=ac["lag"], y=ac["ACF"], marker_color=PAL[0], width=0.4))
    fig.add_trace(go.Scatter(x=ac["lag"], y=ac["ACF band"], line=dict(color=PAL[1], dash="dash")))
    ch.style_academic(fig, x_title="Lag", y_title="ACF", height=340, legend="none", rtl=is_ar())
    _chart(fig, "lm_acf", "long_memory_acf")
    _method_block("longmemory", "tsl")
    _code(tsr.code("longmemory", vcol, tcol, s=s.s, log=transform == "log"), "tsl_code")
