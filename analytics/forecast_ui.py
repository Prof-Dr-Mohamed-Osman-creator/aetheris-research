# ==========================================
# Aetheris Research - Forecasting Studio (UI) — v1.9, Catalogue §3.1–3.4, §3.6
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import charts as ch
from analytics import descriptive as ds
from analytics import fcst as F
from analytics import fcst_report as fr
from analytics import nowcast as nc
from analytics import reg_report as rp
from analytics import ts
from analytics import ts_ui as tu
from core.memory import active_name, get_memory, has_data, set_memory
from core.utils import tr, is_ar, insight, excel_download

PAL = ch.PALETTES["academic"]
_disp, _code, _lang = tu._disp, tu._code, tu._lang
fp = fr.fmt_p
tu._AR_COLS.update({"Model": "النموذج", "Rank": "الترتيب", "failed origins": "أصول فاشلة", "Nowcast": "التنبؤ الآني",
                    "Quarter": "الربع", "forecast": "التنبؤ", "Weight": "الوزن", "Method": "الطريقة", "violations": "الانتهاكات",
                    "expected": "المتوقع", "rate %": "النسبة %", "p Kupiec": "p كوبيك", "p independence": "p الاستقلال",
                    "p conditional coverage": "p التغطية المشروطة", "Tick loss": "خسارة المئين", "Feature": "الخاصية",
                    "Importance": "الأهمية", "Actual": "الفعلي", "Bridge": "الجسر", "Indicator": "المؤشر",
                    "Months forecast (AR)": "الأشهر المتنبأ بها (AR)", "Posterior mean": "المتوسط البعدي", "Posterior sd": "الانحراف البعدي",
                    "Estimate": "التقدير", "Scenario": "السيناريو", "Monthly lag": "الإبطاء الشهري", "Weight × β": "الوزن × β",
                    "Months observed (average)": "الأشهر المشاهدة (متوسط)", "R² on factor(s)": "R² على العامل", "Series": "السلسلة",
                    "lower 80%": "الحد الأدنى 80%", "upper 80%": "الحد الأعلى 80%", "lower 95%": "الحد الأدنى 95%", "upper 95%": "الحد الأعلى 95%",
                    "lower 68%": "الحد الأدنى 68%", "upper 68%": "الحد الأعلى 68%", "lower 90%": "الحد الأدنى 90%", "upper 90%": "الحد الأعلى 90%",
                    "median": "الوسيط", "mean": "المتوسط", "IRF": "الاستجابة", "lower": "الحد الأدنى", "upper": "الحد الأعلى"})
tu._AR_VALS.update({"Naive": "الساذج", "Seasonal naive": "الساذج الموسمي", "Random walk with drift": "السير العشوائي بانجراف",
                    "Mean": "المتوسط", "Theta": "ثيتا", "ETS (automatic)": "ETS (تلقائي)", "ARIMA (automatic)": "ARIMA (تلقائي)",
                    "Gradient boosting": "تعزيز التدرج", "Random forest": "الغابات العشوائية", "Prophet-style": "على نمط Prophet",
                    "State space (local linear trend)": "الفضاء الحالتي (اتجاه خطي محلي)",
                    "Combination (mean of the top 3)": "التجميع (متوسط أفضل 3)"})


def _t(pair):
    return pair[1] if is_ar() else pair[0]


_NAMES_AR = {"Mean": "المتوسط", "Naive": "الساذج", "Seasonal naive": "الساذج الموسمي", "Random walk with drift": "السير العشوائي بانجراف",
             "Theta": "ثيتا", "Simple exponential smoothing": "التمهيد الأسي البسيط", "Holt": "هولت", "Holt (damped)": "هولت (مخمَّد)",
             "Holt-Winters (multiplicative)": "هولت-وينترز (ضربي)", "Holt-Winters (additive)": "هولت-وينترز (جمعي)",
             "Prophet-style (piecewise trend + Fourier)": "على نمط Prophet (اتجاه متقطع + فورييه)"}


def _nm(name: str) -> str:
    """Arabic display name of a forecast method."""
    if not is_ar():
        return name
    if name in _NAMES_AR:
        return _NAMES_AR[name]
    if name.startswith("Moving average"):
        return name.replace("Moving average", "المتوسط المتحرك")
    for k, (en, ar) in {**F.CURVES, **F.ML_MODELS}.items():
        if name == en:
            return ar
    if name.startswith("State space"):
        return name.replace("State space — ", "الفضاء الحالتي — ").replace("Local linear trend", "اتجاه خطي محلي").replace(
            "Local level", "مستوى محلي").replace("Smooth trend", "اتجاه ممهد").replace(" + seasonal", " + موسمية")
    if name.startswith("BSTS"):
        return name.replace("local linear trend", "اتجاه خطي محلي").replace("local level", "مستوى محلي").replace(
            " + seasonal", " + موسمية").replace(" + regression", " + انحدار")
    return name.replace("Regression with", "انحدار بأخطاء").replace(" errors", "").replace(" with drift", " مع انجراف")


def _sig(key):
    return (active_name(), key)


def _store(key, v, sig):
    st.session_state[f"_v19res_{key}"] = (sig, v)


def _load(key, sig):
    v = st.session_state.get(f"_v19res_{key}")
    return v[1] if isinstance(v, tuple) and v[0] == sig else None


def _chart(fig, key, name):
    st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, name), key=key)


def _table(df, digits=4, height=None):
    show = df.copy()
    for c in show.columns:
        if str(c).startswith("p") and ("p-value" in str(c) or str(c).startswith("p ") or c == "p"):
            show[c] = show[c].map(lambda v: fp(v) if isinstance(v, (float, np.floating)) else v)
    kw = {"height": height} if height else ({"height": 35 * (len(show) + 1) + 3} if len(show) <= 18 else {})
    st.dataframe(_disp(show).round(digits), width="stretch", hide_index=True, **kw)


def _method(kind, key, **kw):
    txt, refs = fr.methodology(kind, _lang(), **kw)
    with st.expander(tr("📝 Methodology paragraph & references", "📝 فقرة المنهجية والمراجع"), expanded=False):
        st.text_area("m", txt, height=160, key=f"{key}_meth", label_visibility="collapsed")
        st.markdown("\n".join(f"- {x}" for x in refs))
    return txt, refs


def fan_chart(y: pd.Series, fcs: list, h: int, key: str, name: str, tail: int = 120, fan: bool = True, title_y: str = ""):
    """History + forecasts; the first forecast gets a fan of quantile bands (or 80/95% bands)."""
    fig = go.Figure()
    yy = y.iloc[-tail:]
    fig.add_trace(go.Scatter(x=yy.index, y=yy.values, mode="lines", name=tr("Observed", "المشاهد"), line=dict(color="#222", width=1.6)))
    fidx = F.future_index(y.index, h)
    for i, f in enumerate(fcs):
        col = PAL[i % len(PAL)]
        if i == 0 and fan:
            if f.sims is not None or f.pi:
                if f.sims is not None:
                    Q = f.quantiles()
                    bands = [(0.05, 0.95), (0.1, 0.9), (0.2, 0.8), (0.3, 0.7), (0.4, 0.6)]
                    for k, (a, b) in enumerate(bands):
                        fig.add_trace(go.Scatter(x=np.r_[fidx, fidx[::-1]], y=np.r_[Q[b], Q[a][::-1]], fill="toself", line=dict(width=0),
                                                 fillcolor=f"rgba(31,78,121,{0.10 + 0.05 * k})", hoverinfo="skip", showlegend=k == 0,
                                                 name=tr("Fan (5–95%)", "المروحة (5–95%)")))
                else:
                    for lv, a in ((95, 0.15), (80, 0.25)):
                        if lv in f.pi:
                            lo, hi = f.pi[lv]
                            fig.add_trace(go.Scatter(x=np.r_[fidx, fidx[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", line=dict(width=0),
                                                     fillcolor=f"rgba(31,78,121,{a})", hoverinfo="skip", name=f"{lv}%"))
        fig.add_trace(go.Scatter(x=fidx, y=f.mean, mode="lines+markers" if h <= 24 else "lines", name=_nm(f.name),
                                 line=dict(color=col, width=2.4 if i == 0 else 1.6, dash=None if i == 0 else "dot"), marker=dict(size=4)))
    ch.style_academic(fig, height=460, y_title=title_y, legend="bottom", rtl=is_ar())
    _chart(fig, key, name)


def _demo_buttons(key):
    c = st.columns(4)
    demos = [("monthly_forecast_demo", tr("📈 Monthly demand & tourism (2008–2025)", "📈 الطلب الشهري والسياحة (2008–2025)"), "Demo – monthly forecasting (synthetic)"),
             ("nowcast_demo", tr("🛰️ Nowcasting GDP (monthly indicators)", "🛰️ التنبؤ الآني بالناتج (مؤشرات شهرية)"), "Demo – nowcasting (synthetic)"),
             ("realized_vol_demo", tr("🌪️ Daily returns & realized variance", "🌪️ عوائد يومية وتباين محقق"), "Demo – realized volatility (synthetic)"),
             ("macro_var_demo", tr("🏦 Quarterly macro (BVAR, LP)", "🏦 كلي ربع سنوي (BVAR وLP)"), "Demo – macro VAR (synthetic)")]
    import data.samples as smp
    for col, (fn, lab, nm) in zip(c, demos):
        with col:
            if st.button(lab, key=f"{key}_{fn}", width="stretch"):
                set_memory(getattr(smp, fn)(), source="demo (synthetic)", name=nm)
                st.rerun()
    st.caption(tr("⚠️ Simulated data for training only — never cite as real statistics.",
                  "⚠️ بيانات محاكاة للتدريب فقط — لا تُستشهد بها كإحصاءات حقيقية."))


def _pick(df, key):
    tcols = tu._time_cols(df) + [c for c in ds.time_columns(df) if c not in tu._time_cols(df)]
    if not tcols:
        st.info(tr("No date / year column found.", "لا يوجد عمود للتاريخ أو السنة."))
        return None
    num = [c for c in ds.numeric_columns(df) if c not in tcols]
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.6, 0.9, 1, 0.8])
    t = c1.selectbox(tr("Time variable", "المتغير الزمني"), tcols, key=f"{key}_t")
    v = c2.selectbox(tr("Series to forecast", "السلسلة المراد التنبؤ بها"), num, key=f"{key}_v")
    transform = c3.radio(tr("Transform", "التحويل"), ["level", "log"], horizontal=True, key=f"{key}_tr",
                         format_func=lambda x: tr("Level", "المستوى") if x == "level" else "ln")
    per = c4.selectbox(tr("Seasonal period", "طول الموسم"), ["auto", 1, 4, 12, 52, 7, 5], key=f"{key}_per",
                       format_func=lambda p: tr("auto-detect", "تلقائي") if p == "auto" else str(p))
    h = c5.number_input(tr("Horizon h", "الأفق h"), 1, 120, 12, key=f"{key}_h")
    try:
        s = ts.prepare(df, t, v, {}, transform, None if per == "auto" else int(per))
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return None
    if transform == "log":
        st.caption(tr("Forecasts are made for ln(series); exp() gives the median forecast in levels.",
                      "التنبؤ للوغاريتم السلسلة؛ والدالة الأسية تعطي التنبؤ الوسيط بالمستوى."))
    st.caption(tr(f"{len(s.y)} observations · s = {s.s} · {tu._tlabel(s.y.index[0])} → {tu._tlabel(s.y.index[-1])}",
                  f"{len(s.y)} مشاهدة · s = {s.s} · من ⁦{tu._tlabel(s.y.index[0])}⁩ إلى ⁦{tu._tlabel(s.y.index[-1])}⁩"))
    return s, t, v, int(h)


# ==========================================================
def render_forecasting():
    st.caption("💡 " + tr("Forecasting studio: let the platform race the models on your series, or build a specific model. "
                         "Every model is evaluated out of sample before it is trusted.",
                         "استوديو التنبؤ: دع المنصة تُجري مسابقة بين النماذج على سلسلتك، أو ابنِ نموذجاً بعينه. ويُقيَّم كل نموذج "
                         "خارج العينة قبل الوثوق به."))
    if not has_data():
        st.info(tr("Load data first (Data Hub), or try a demo dataset:", "حمّل بيانات أولاً (مركز البيانات)، أو جرّب بيانات تجريبية:"))
        _demo_buttons("fcd0")
        return
    df = get_memory()
    st.caption(tr(f"Active dataset: **{active_name()}** — {df.shape[0]} rows × {df.shape[1]} columns",
                  f"البيانات النشطة: **{active_name()}** — {df.shape[0]} صف × {df.shape[1]} عمود"))
    with st.expander(tr("🧪 Demo datasets for training", "🧪 بيانات تجريبية للتدريب"), expanded=False):
        _demo_buttons("fcd1")
    tabs = st.tabs([tr("🏁 Model horse-race", "🏁 مسابقة النماذج"), tr("📈 Statistical models", "📈 النماذج الإحصائية"),
                    tr("🛰️ Nowcasting & macro", "🛰️ التنبؤ الآني والكلي"), tr("🌪️ Volatility & VaR", "🌪️ التقلب والقيمة المعرضة للخطر"),
                    tr("🤖 Machine learning", "🤖 التعلم الآلي"), tr("🧮 Evaluation & combination", "🧮 التقييم والتجميع"),
                    tr("🎯 Scenarios", "🎯 السيناريوهات")])
    with tabs[0]:
        _race(df)
    with tabs[1]:
        _classical(df)
    with tabs[2]:
        _nowcast(df)
    with tabs[3]:
        _volatility(df)
    with tabs[4]:
        _ml(df)
    with tabs[5]:
        _evaluation(df)
    with tabs[6]:
        _scenarios(df)


def _align(df, t, cols, index):
    tc = pd.to_datetime(df[t], errors="coerce") if not pd.api.types.is_numeric_dtype(df[t]) else df[t]
    d = df.assign(_t=tc).dropna(subset=["_t"]).groupby("_t")[cols].mean().reindex(index)
    return None if d.isna().any().any() else d


# ---------------------------------------------------------- horse race
def _race(df):
    p = _pick(df, "fr")
    if p is None:
        return
    s, t, v, h = p
    y = s.y
    c1, c2, c3 = st.columns([2.4, 1, 1])
    avail = [k for k in F.RACE if k != "combo" and (s.s > 1 or k != "snaive")]
    models = c1.multiselect(tr("Models", "النماذج"), avail, avail, key="fr_models", format_func=lambda k: _t(F.RACE[k]))
    origins = c2.number_input(tr("Forecast origins", "أصول التنبؤ"), 3, 30, 8, key="fr_orig")
    metric = c3.selectbox(tr("Rank by", "الترتيب حسب"), ["MASE", "RMSE", "MAE", "MAPE", "RMSSE"], key="fr_metric")
    st.caption(tr("Each model is re-estimated at every origin and forecasts h steps ahead; ETS / ARIMA specifications are "
                  "selected on the first training window. Machine-learning models make the race slower (≈ 1–2 minutes).",
                  "يُعاد تقدير كل نموذج عند كل أصل ويتنبأ بـ h خطوة؛ وتُختار مواصفة ETS وARIMA على أول نافذة تدريب. ونماذج التعلم الآلي "
                  "تجعل المسابقة أبطأ (≈ دقيقة إلى دقيقتين)."))
    sig = _sig(("race", v, s.transform, s.s, h, tuple(models), origins, metric))
    if st.button(tr("🏁 Run the horse-race", "🏁 تشغيل المسابقة"), type="primary", key="fr_run") and models:
        bar = st.progress(0.0)
        try:
            r = F.horse_race(y.to_numpy(), h, s.s, models, int(origins), metric=metric,
                             progress=lambda f, m: bar.progress(min(1.0, f), text=f"{tr('Running', 'جارٍ تشغيل')}: {m}"))
            _store("race", r, sig)
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
        bar.empty()
    r = _load("race", sig)
    if r is None:
        return
    r = dict(r)
    loc = r["table"].copy()
    top_names = " + ".join(_t(F.RACE[k]) for k in r["top3"])
    loc["Model"] = [(_t(F.RACE["combo"]) + ": " + top_names) if k == "combo" else _t(F.RACE[k]) for k in loc.key]
    r["table"] = loc
    if len(r["dm"]):
        r["dm"] = r["dm"].assign(Model=[_t(F.RACE[k]) if k in F.RACE else m for k, m in zip(
            [next((kk for kk in F.RACE if F.RACE[kk][0] == m or m.startswith(F.RACE[kk][0])), None) for m in r["dm"].Model], r["dm"].Model)])
    tab = loc.drop(columns=["key"])
    _table(tab)
    insight(fr.interpret_race(r, _lang()))
    fc_all = {k: f for k, f in r["forecasts"].items()}
    for k, f in fc_all.items():
        f.name = _t(F.RACE[k]) if k in F.RACE else f.name
    best = fc_all.get(r["best"]) or fc_all["combo"]
    fcs = [best] + [fc_all[k] for k in r["top3"] if k != r["best"]] + ([fc_all["combo"]] if r["best"] != "combo" else [])
    fan_chart(y, fcs, h, "fr_fan", "horse_race_forecasts", title_y=s.name)
    E = r["errors"]
    fig = go.Figure()
    for i, k in enumerate(r["table"]["key"][:6]):
        e = E[k]
        fig.add_trace(go.Scatter(x=np.arange(1, h + 1), y=np.sqrt(np.nanmean(e ** 2, 0)), mode="lines+markers",
                                 name=_t(F.RACE[k]), line=dict(color=PAL[i % len(PAL)])))
    ch.style_academic(fig, height=360, x_title=tr("Horizon", "الأفق"), y_title="RMSE", legend="bottom", rtl=is_ar())
    _chart(fig, "fr_hor", "rmse_by_horizon")
    if len(r["dm"]):
        st.markdown("**" + tr("Diebold-Mariano tests against the best model (h = 1, squared-error loss)",
                              "اختبارات ديبولد-ماريانو مقابل أفضل نموذج (h = 1، خسارة مربع الخطأ)") + "**")
        _table(r["dm"][["Model", "DM", "p", "n"]].rename(columns={"p": "p-value"}))
    out = best.frame(F.future_index(y.index, h))
    st.markdown("**" + tr("Forecasts of the best model", "تنبؤات أفضل نموذج") + f" — {best.name}**")
    _table(out, 3)
    txt, refs = _method("race", "frm", origins=r["origins"], h=h)
    c1, c2 = st.columns(2)
    with c1:
        excel_download({"Ranking": tab, "Forecast (best)": out, "Combination": r["forecasts"]["combo"].frame(F.future_index(y.index, h)),
                        "DM tests": r["dm"]}, "forecast_horse_race.xlsx", key="fr_xl")
    with c2:
        st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"),
                           rp.to_docx([("h", tr("Methodology", "المنهجية")), ("p", txt), ("table", tab.round(4).astype(str)),
                                       ("p", fr.interpret_race(r, _lang()).replace("<b>", "").replace("</b>", "")),
                                       ("table", out.round(3).astype(str)), ("bullets", refs)],
                                      tr("Forecast competition", "مسابقة نماذج التنبؤ"), is_ar()), "forecast_horse_race.docx", key="fr_docx")
    _code(fr.code("race", time=t, y=v, h=h, s=s.s), "fr_code")


# ---------------------------------------------------------- statistical models
CLASSICAL = {"bench": ("Benchmarks (mean, naive, seasonal naive, drift, MA)", "النماذج المرجعية (المتوسط، الساذج، الموسمي، الانجراف، المتوسط المتحرك)"),
             "ets": ("Exponential smoothing (ETS)", "التمهيد الأسي (ETS)"), "theta": ("Theta", "ثيتا"),
             "arima": ("ARIMA / SARIMA", "ARIMA / SARIMA"), "ss": ("State space (structural, Kalman)", "الفضاء الحالتي (هيكلي، كالمان)"),
             "bsts": ("Bayesian structural (BSTS)", "الهيكلي البايزي (BSTS)"), "growth": ("Trend & growth curves", "الاتجاه ومنحنيات النمو")}


def _classical(df):
    p = _pick(df, "fc")
    if p is None:
        return
    s, t, v, h = p
    y = s.y
    ya = y.to_numpy()
    kind = st.selectbox(tr("Model family", "عائلة النموذج"), list(CLASSICAL), key="fc_kind", format_func=lambda k: _t(CLASSICAL[k]))
    fcs = []
    try:
        if kind == "bench":
            k = st.slider(tr("Moving-average window", "نافذة المتوسط المتحرك"), 2, 24, 3, key="fc_ma")
            fcs = ([F.snaive(ya, h, s.s)] if s.s > 1 else []) + [F.drift(ya, h), F.naive(ya, h), F.moving_average(ya, h, k), F.mean_fc(ya, h)]
        elif kind == "ets":
            mode = st.radio(tr("Specification", "التوصيف"), ["auto", "manual"], horizontal=True, key="fc_ets_mode",
                            format_func=lambda m: tr("Automatic (AICc)", "تلقائي (AICc)") if m == "auto" else tr("Manual", "يدوي"))
            if mode == "auto":
                sig = _sig(("ets_auto", v, s.transform, s.s))
                tab = _load("ets_auto", sig)
                if tab is None:
                    with st.spinner(tr("Estimating the ETS family…", "جارٍ تقدير عائلة ETS…")):
                        tab = F.ets_auto(ya, s.s)
                    _store("ets_auto", tab, sig)
                with st.expander(tr("All ETS models (ranked by AICc)", "كل نماذج ETS (مرتبة بـ AICc)"), expanded=False):
                    _table(tab.drop(columns=["_theta", "error", "trend", "damped", "seasonal"]), 3)
                row = tab.iloc[0]
                spec = {"error": row.error, "trend": row.trend, "damped": bool(row.damped), "seasonal": row.seasonal, "start": row._theta}
                spec = {k_: (None if isinstance(v_, float) and np.isnan(v_) else v_) for k_, v_ in spec.items()}
                spec["effort"] = "quick"
            else:
                c = st.columns(4)
                e = c[0].selectbox(tr("Error", "الخطأ"), ["add", "mul"], key="fc_ets_e")
                tr_ = c[1].selectbox(tr("Trend", "الاتجاه"), [None, "add"], key="fc_ets_t", format_func=lambda x: "N" if x is None else "A")
                dmp = c[2].checkbox(tr("Damped", "مُخمَّد"), key="fc_ets_d", disabled=tr_ is None)
                sn = c[3].selectbox(tr("Seasonal", "الموسمية"), [None, "add", "mul"], key="fc_ets_s", disabled=s.s < 2,
                                    format_func=lambda x: {None: "N", "add": "A", "mul": "M"}[x])
                spec = {"error": e, "trend": tr_, "damped": dmp, "seasonal": sn if s.s > 1 else None}
            f = F.ets(ya, h, s.s, spec)
            fcs = [f]
            pr_ = pd.DataFrame({"Parameter": list(f.info["params"]), "Estimate": list(f.info["params"].values())})
            st.markdown("**" + tr("Smoothing parameters and initial states", "معلمات التمهيد والحالات الأولية") + "**")
            _table(pr_.head(6), 4)
            st.caption(f"AICc = {f.info['AICc']:.2f} · AIC = {f.info['AIC']:.2f} · BIC = {f.info['BIC']:.2f} · log L = {f.info['loglik']:.2f}")
        elif kind == "theta":
            f = F.theta(ya, h, s.s)
            fcs = [f, F.ses(ya, h)]
            st.caption(tr(f"α = {f.info['alpha']:.3f}; drift (b/2) = {f.info['drift (b/2)']:.4g}; seasonally adjusted: {f.info['seasonally adjusted']}",
                          f"α = {f.info['alpha']:.3f}؛ الانجراف (b/2) = {f.info['drift (b/2)']:.4g}؛ معدَّلة موسمياً: {'نعم' if f.info['seasonally adjusted'] else 'لا'}"))
        elif kind == "arima":
            mode = st.radio(tr("Specification", "التوصيف"), ["auto", "manual"], horizontal=True, key="fc_ar_mode",
                            format_func=lambda m: tr("Automatic (KPSS + AICc)", "تلقائي (KPSS + AICc)") if m == "auto" else tr("Manual", "يدوي"))
            if mode == "auto":
                sig = _sig(("arima_auto", v, s.transform, s.s, h))
                f = _load("arima_auto", sig)
                if f is None:
                    with st.spinner(tr("Searching ARIMA orders…", "جارٍ البحث في رتب ARIMA…")):
                        f = F.arima_auto(ya, h, s.s, max_p=2, max_q=2)
                    _store("arima_auto", f, sig)
            else:
                c = st.columns(7)
                o = [c[i].number_input(lab, 0, 5 if i != 1 else 2, dv, key=f"fc_ar_{lab}") for i, (lab, dv) in enumerate((("p", 1), ("d", 1), ("q", 1)))]
                so = [c[3 + i].number_input(lab, 0, 2 if i != 1 else 1, 0, key=f"fc_ar_{lab}", disabled=s.s < 2) for i, lab in enumerate(("P", "D", "Q"))]
                trd = c[6].selectbox(tr("Constant / drift", "الثابت / الانجراف"), ["auto", "c", "t", "n"], key="fc_ar_tr")
                f = F.sarimax(ya, h, tuple(int(x) for x in o), tuple(int(x) for x in so) if s.s > 1 else (0, 0, 0), s.s,
                              trend=None if trd == "auto" else trd)
            fcs = [f]
            _table(f.info["table"], 4)
            st.caption(f"AICc = {f.info['AICc']:.2f} · BIC = {f.info['BIC']:.2f} · log L = {f.info['loglik']:.2f} · σ² = {f.info['sigma2']:.4g}")
        elif kind == "ss":
            c1, c2, c3 = st.columns(3)
            lv = c1.selectbox(tr("Trend component", "مكون الاتجاه"), list(F.UC_LEVELS), 1, key="fc_ss_l", format_func=lambda k: _t(F.UC_LEVELS[k]))
            seas = c2.checkbox(tr("Stochastic seasonal", "موسمية عشوائية"), s.s > 1, key="fc_ss_s", disabled=s.s < 2)
            cyc = c3.checkbox(tr("Stochastic cycle", "دورة عشوائية"), False, key="fc_ss_c")
            f = F.state_space(ya, h, lv, s.s if seas else 1, cycle=cyc)
            fcs = [f]
            _table(f.info["table"], 5)
            comps = f.info["components"]
            if comps:
                fig = go.Figure()
                for i, (nm, c) in enumerate(comps.items()):
                    fig.add_trace(go.Scatter(x=y.index, y=c, name=nm, line=dict(color=PAL[i % len(PAL)])))
                ch.style_academic(fig, height=360, y_title=tr("Smoothed components", "المكونات الممهدة"), legend="bottom", rtl=is_ar())
                _chart(fig, "fc_ss_comp", "state_space_components")
        elif kind == "bsts":
            c1, c2, c3 = st.columns(3)
            lv = c1.selectbox(tr("Trend component", "مكون الاتجاه"), ["llevel", "lltrend"], 1, key="fc_bs_l", format_func=lambda k: _t(F.UC_LEVELS[k]))
            nd = c2.select_slider(tr("MCMC draws", "سحوبات MCMC"), [500, 1000, 2000], 1000, key="fc_bs_n")
            xs = c3.multiselect(tr("Regressors (need future values)", "متغيرات مفسرة (تحتاج قيماً مستقبلية)"),
                                [c for c in ds.numeric_columns(df) if c not in (v, t)], key="fc_bs_x", placeholder=tr("None", "لا شيء"))
            sig = _sig(("bsts", v, s.transform, s.s, h, lv, nd, tuple(xs)))
            if st.button(tr("▶️ Run the Gibbs sampler", "▶️ تشغيل معاين جيبس"), key="fc_bs_run"):
                ex, exf = None, None
                if xs:
                    X = _align(df, t, xs, y.index)
                    if X is None:
                        st.error(tr("Regressors have gaps over the sample.", "المتغيرات المفسرة بها فجوات في فترة العينة."))
                        return
                    X = X.to_numpy(float)
                    ex = X
                    exf = np.vstack([F.drift(X[:, j], h).mean for j in range(X.shape[1])]).T
                with st.spinner(tr("Sampling…", "جارٍ المعاينة…")):
                    _store("bsts", F.bsts(ya, h, lv, s.s, ex, exf, int(nd), int(nd) // 3), sig)
            f = _load("bsts", sig)
            if f is None:
                return
            fcs = [f]
            _table(f.info["table"], 5)
            if xs:
                st.caption(tr("Future regressor values: random-walk-with-drift projections.", "القيم المستقبلية للمتغيرات المفسرة: إسقاط بالسير العشوائي بانجراف."))
        else:
            kd = st.selectbox(tr("Curve", "المنحنى"), list(F.CURVES), 3, key="fc_gc", format_func=lambda k: _t(F.CURVES[k]))
            f = F.growth_curve(ya, h, kd)
            fcs = [f] + [F.growth_curve(ya, h, k_) for k_ in F.CURVES if k_ != kd and k_ in ("linear", "exponential")]
            _table(f.info["table"], 5)
            st.caption(f"R² = {f.info['R²']:.4f} · AIC = {f.info['AIC']:.2f} · σ = {f.info['sigma']:.4g}")
            if kd in ("logistic", "gompertz"):
                K = f.info["params"][0]
                insight(tr(f"Estimated saturation level (asymptote) K = {K:,.4g}; the last observation is at {ya[-1] / K * 100:.0f}% of it.",
                           f"مستوى التشبع المقدَّر (الخط المقارب) K = {K:,.4g}؛ وآخر مشاهدة عند {ya[-1] / K * 100:.0f}% منه."))
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return
    if not fcs:
        return
    fan_chart(y, fcs, h, f"fc_fan_{kind}", f"forecast_{kind}", title_y=s.name)
    f0 = fcs[0]
    insight(fr.interpret_fc(f0, float(ya[-1]), _lang(), name=_nm(f0.name)))
    out = f0.frame(F.future_index(y.index, h))
    if s.transform == "log":
        for c in [c for c in out.columns if c not in ("time", "step")]:
            out[f"{c} (level)"] = np.exp(out[c])
    _table(out, 3)
    with st.expander(tr("Residual diagnostics and hold-out accuracy", "تشخيص البواقي ودقة العينة المحجوزة"), expanded=False):
        if f0.resid is not None:
            e = np.asarray(f0.resid, float)
            e = e[np.isfinite(e)]
            from statsmodels.stats.diagnostic import acorr_ljungbox
            L = min(2 * s.s if s.s > 1 else 10, len(e) // 5)
            lb = acorr_ljungbox(e, lags=[L], return_df=True)
            st.caption(tr(f"Ljung-Box Q({L}) = {lb.lb_stat.iloc[0]:.2f}, p = {fp(lb.lb_pvalue.iloc[0])}",
                          f"ليونج-بوكس Q({L}) = {lb.lb_stat.iloc[0]:.2f}، p = {fp(lb.lb_pvalue.iloc[0])}"))
        test = st.number_input(tr("Hold-out size", "حجم العينة المحجوزة"), h, max(h, len(ya) // 3), min(max(h, 12), len(ya) // 3), key=f"fc_ho_{kind}")
        if st.button(tr("Evaluate on the hold-out", "التقييم على العينة المحجوزة"), key=f"fc_ho_run_{kind}"):
            tr_, te = ya[:-int(test)], ya[-int(test):]
            rows = []
            fn = {"bench": lambda yy, hh: F.drift(yy, hh), "ets": lambda yy, hh: F.ets(yy, hh, s.s, {k_: f0.info["spec"][k_] for k_ in ("error", "trend", "damped", "seasonal")}),
                  "theta": lambda yy, hh: F.theta(yy, hh, s.s), "arima": lambda yy, hh: F.sarimax(yy, hh, f0.info["order"], f0.info["seasonal"], s.s, trend=f0.info["trend"]),
                  "ss": lambda yy, hh: F.state_space(yy, hh, "lltrend", s.s), "bsts": lambda yy, hh: F.bsts(yy, hh, "lltrend", s.s, draws=400, burn=150),
                  "growth": lambda yy, hh: F.growth_curve(yy, hh, st.session_state.get("fc_gc", "logistic"))}[kind]
            for nm, fun in ((f0.name, fn), (tr("Seasonal naive", "الساذج الموسمي") if s.s > 1 else tr("Naive", "الساذج"),
                                            (lambda yy, hh: F.snaive(yy, hh, s.s)) if s.s > 1 else (lambda yy, hh: F.naive(yy, hh)))):
                try:
                    ff = fun(tr_, int(test))
                    rows.append({"Model": _nm(nm), **F.accuracy(te, ff.mean, tr_, s.s)})
                except Exception as e:  # noqa: BLE001
                    st.warning(f"{nm}: {e}")
            _table(pd.DataFrame(rows), 4)
    _method("classical", "fcm")
    excel_download({"Forecast": out, **({"Parameters": f0.info["table"]} if "table" in f0.info else {})}, f"forecast_{kind}.xlsx", key=f"fc_xl_{kind}")
    _code(fr.code("classical", time=t, y=v, h=h, s=s.s), "fc_code")


# ---------------------------------------------------------- nowcasting & macro
def _nowcast(df):
    mode = st.radio(tr("Method", "الطريقة"), ["bridge", "midas", "dfm", "bvar", "lp"], horizontal=True, key="nc_mode",
                    format_func=lambda m: {"bridge": tr("Bridge equations", "معادلات الجسر"), "midas": "MIDAS",
                                           "dfm": tr("Dynamic factor model", "نموذج العوامل الديناميكية"), "bvar": tr("Bayesian VAR", "VAR بايزي"),
                                           "lp": tr("Local projections", "الإسقاطات المحلية")}[m])
    tcols = tu._time_cols(df) + [c for c in ds.time_columns(df) if c not in tu._time_cols(df)]
    num = [c for c in ds.numeric_columns(df) if c not in tcols]
    if not tcols:
        st.info(tr("A date column is needed.", "يلزم عمود للتاريخ."))
        return
    if mode in ("bridge", "midas", "dfm"):
        st.caption("💡 " + tr("Monthly data: the quarterly target is stored in one month of each quarter (others empty); unpublished "
                             "quarters are left empty and will be nowcast.",
                             "بيانات شهرية: يُسجَّل المتغير ربع السنوي في شهر واحد من كل ربع (والبقية فارغة)؛ وتُترك الأرباع غير المنشورة "
                             "فارغة ليُتنبأ بها آنياً."))
        sparse = [c for c in num if df[c].notna().mean() < 0.5]
        c1, c2, c3 = st.columns([1, 1.2, 2.4])
        t = c1.selectbox(tr("Date", "التاريخ"), tcols, key="nc_t")
        y = c2.selectbox(tr("Quarterly target", "المتغير ربع السنوي"), sparse or num, key="nc_y")
        rest = [c for c in num if c != y]
        ind = c3.multiselect(tr("Monthly indicators", "المؤشرات الشهرية"), rest, rest[:4], key="nc_x", placeholder=tr("Choose…", "اختر…"))
        if not ind:
            return
        try:
            if mode == "bridge":
                yl = st.checkbox(tr("Include the lagged target", "إدراج المتغير التابع المبطأ"), False, key="nc_yl")
                r = nc.bridge(df, t, y, ind, yl)
                _table(r["table"], 4)
                st.caption(f"R² = {r['R²']:.3f} · N = {r['N']}")
                if len(r["filled"]):
                    st.markdown("**" + tr("Ragged edge: months filled by AR forecasts", "الحافة غير المكتملة: أشهر استُكملت بتنبؤات AR") + "**")
                    _table(r["filled"])
                st.markdown("**" + tr("Nowcasts", "التنبؤات الآنية") + "**")
                _table(r["nowcasts"], 3)
                cc = [c for c in r["nowcasts"].columns if c.startswith("contrib")]
                if cc and len(r["nowcasts"]):
                    fig = go.Figure()
                    for i, c in enumerate(cc):
                        fig.add_trace(go.Bar(x=r["nowcasts"]["Quarter"], y=r["nowcasts"][c], name=c.replace("contrib: ", ""), marker_color=PAL[i % len(PAL)]))
                    fig.update_layout(barmode="relative")
                    ch.style_academic(fig, height=340, y_title=tr("Contribution vs the sample mean", "المساهمة مقارنة بمتوسط العينة"), legend="bottom", rtl=is_ar())
                    _chart(fig, "nc_contrib", "bridge_contributions")
                es = r["eval_summary"]
                if es:
                    insight(tr(f"Pseudo out-of-sample ({len(r['evaluation'])} quarters): RMSE bridge = {es['RMSE bridge']:.3f} vs AR(1) = {es['RMSE AR(1)']:.3f} "
                               f"(ratio {es['relative RMSE']:.2f}); DM p = {fp(es['DM p'])}.",
                               f"تقييم شبه حقيقي خارج العينة ({len(r['evaluation'])} ربعاً): RMSE للجسر = {es['RMSE bridge']:.3f} مقابل AR(1) = {es['RMSE AR(1)']:.3f} "
                               f"(النسبة {es['relative RMSE']:.2f})؛ وقيمة p لديبولد-ماريانو = {fp(es['DM p'])}."))
                    ev = r["evaluation"]
                    fig = go.Figure([go.Scatter(x=ev.Quarter, y=ev[c], name=c, mode="lines+markers", line=dict(color=PAL[i]))
                                     for i, c in enumerate(["Actual", "Bridge", "AR(1)"])])
                    ch.style_academic(fig, height=320, legend="bottom", rtl=is_ar())
                    _chart(fig, "nc_ev", "bridge_evaluation")
                excel_download({"Coefficients": r["table"], "Nowcasts": r["nowcasts"], "Evaluation": r["evaluation"]}, "bridge_nowcast.xlsx", key="nc_xl1")
            elif mode == "midas":
                c1, c2, c3 = st.columns(3)
                x1 = c1.selectbox(tr("High-frequency indicator", "المؤشر عالي التكرار"), ind, key="nc_mx")
                K = c2.number_input(tr("Monthly lags K", "الإبطاءات الشهرية K"), 3, 24, 6, key="nc_K")
                wk = c3.selectbox(tr("Lag weights", "أوزان الإبطاء"), list(nc.WEIGHTS), 1, key="nc_w", format_func=lambda k: _t(nc.WEIGHTS[k]))
                r = nc.midas(df, t, y, x1, int(K), wk, True)
                _table(r["table"], 4)
                st.caption(f"R² = {r['R²']:.3f} · RMSE = {r['RMSE']:.4f} · N = {r['N']} · AIC = {r['AIC']:.2f}")
                if r["restriction_test"]:
                    rt = r["restriction_test"]
                    st.caption(tr(f"Test of the weight restriction vs U-MIDAS: F({rt['df'][0]}, {rt['df'][1]}) = {rt['F']:.3f}, p = {fp(rt['p'])} — "
                                  + ("restriction not rejected." if rt["p"] >= 0.05 else "restriction rejected: prefer U-MIDAS or another weighting."),
                                  f"اختبار قيد الأوزان مقابل U-MIDAS: F({rt['df'][0]}، {rt['df'][1]}) = {rt['F']:.3f}، p = {fp(rt['p'])} — "
                                  + ("لا يُرفض القيد." if rt["p"] >= 0.05 else "يُرفض القيد: فضّل U-MIDAS أو صيغة أوزان أخرى.")))
                lw = r["lag_weights"]
                fig = go.Figure([go.Bar(x=lw["Monthly lag"], y=lw["U-MIDAS"], name="U-MIDAS", marker_color="#bbb"),
                                 go.Scatter(x=lw["Monthly lag"], y=lw["Weight × β"], name=_t(nc.WEIGHTS[wk]), mode="lines+markers", line=dict(color=PAL[0], width=2.5))])
                ch.style_academic(fig, height=320, x_title=tr("Monthly lag", "الإبطاء الشهري"), y_title=tr("Effect", "الأثر"), legend="bottom", rtl=is_ar())
                _chart(fig, "nc_mw", "midas_weights")
                _table(r["nowcasts"], 3)
                excel_download({"MIDAS": r["table"], "Weights": lw, "Nowcasts": r["nowcasts"]}, "midas.xlsx", key="nc_xl2")
            else:
                c1, c2 = st.columns(2)
                nf = c1.number_input(tr("Factors", "عدد العوامل"), 1, 3, 1, key="nc_nf")
                fo = c2.number_input(tr("Factor AR order", "رتبة AR للعامل"), 1, 4, 2, key="nc_fo")
                sig = _sig(("dfm", t, y, tuple(ind), nf, fo))
                if st.button(tr("▶️ Estimate the factor model (EM)", "▶️ تقدير نموذج العوامل (EM)"), key="nc_dfm_run"):
                    with st.spinner(tr("EM iterations…", "تكرارات EM…")):
                        _store("dfm", nc.dfm_nowcast(df, t, y, ind, int(nf), int(fo)), sig)
                r = _load("dfm", sig)
                if r is None:
                    return
                _table(r["nowcasts"], 3)
                _table(r["loadings"], 3)
                fg = r["factor"]
                fig = go.Figure(go.Scatter(x=fg.index.to_timestamp(), y=fg.values, line=dict(color=PAL[0])))
                ch.style_academic(fig, height=300, y_title=tr("Common factor (smoothed)", "العامل المشترك (ممهد)"), legend="none", rtl=is_ar())
                _chart(fig, "nc_fac", "dfm_factor")
                st.caption(tr(f"In-sample RMSE of the quarterly target: {r['RMSE in-sample']:.3f}", f"RMSE داخل العينة للمتغير ربع السنوي: {r['RMSE in-sample']:.3f}"))
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            return
        _method("nowcast", "ncm")
        _code(fr.code("nowcast", time=t, y=y, target=y, indicators=ind), "nc_code")
        return
    # BVAR / LP
    c1, c2 = st.columns([1, 3])
    t = c1.selectbox(tr("Time", "الزمن"), tcols, key="bv_t")
    vs = c2.multiselect(tr("Variables (ordering = Cholesky ordering)", "المتغيرات (الترتيب = ترتيب تشولسكي)"), num, num[:3], key="bv_v",
                        placeholder=tr("Choose…", "اختر…"))
    if len(vs) < 2:
        return
    Y = df.sort_values(t)[vs].dropna().reset_index(drop=True)
    try:
        if mode == "bvar":
            c = st.columns(4)
            p = c[0].number_input(tr("Lags p", "الإبطاءات p"), 1, 8, 2, key="bv_p")
            own = c[1].selectbox(tr("Prior mean of own first lag", "المتوسط القبلي للإبطاء الذاتي الأول"), [1.0, 0.0], key="bv_own",
                                 format_func=lambda x: tr("1 — levels (random walk)", "1 — مستويات (سير عشوائي)") if x == 1 else tr("0 — growth rates / stationary", "0 — معدلات نمو / ساكنة"))
            lm = c[2].selectbox("λ", ["ML", 0.05, 0.1, 0.2, 0.5, 1.0], key="bv_lam", format_func=lambda x: tr("by marginal likelihood", "بالإمكان الهامشي") if x == "ML" else str(x))
            hh = c[3].number_input(tr("Horizon", "الأفق"), 1, 40, 8, key="bv_h")
            r = nc.bvar(Y, int(p), None if lm == "ML" else float(lm), b_own=float(own), h=int(hh), draws=1500)
            st.caption(tr(f"λ = {r['lambda']:.3f} · log marginal likelihood = {r['logml']:.2f}", f"λ = {r['lambda']:.3f} · لوغاريتم الإمكان الهامشي = {r['logml']:.2f}"))
            if r["grid"] is not None:
                g = r["grid"]
                fig = go.Figure(go.Scatter(x=g["λ"], y=g["log ML"], line=dict(color=PAL[0])))
                fig.add_vline(x=r["lambda"], line=dict(color=PAL[1], dash="dash"))
                fig.update_xaxes(type="log")
                ch.style_academic(fig, height=280, x_title="λ", y_title="log ML", legend="none", rtl=is_ar())
                _chart(fig, "bv_grid", "bvar_lambda")
            cols = st.columns(min(3, len(vs)))
            for i, vn in enumerate(vs):
                fr_ = r["forecasts"][vn]
                with cols[i % len(cols)]:
                    fig = go.Figure()
                    hist = Y[vn].iloc[-24:]
                    xh = np.arange(-len(hist) + 1, 1)
                    fig.add_trace(go.Scatter(x=xh, y=hist.values, line=dict(color="#222"), name=tr("Observed", "المشاهد")))
                    xs_ = fr_["step"].to_numpy()
                    for lv, a in ((90, 0.15), (68, 0.3)):
                        fig.add_trace(go.Scatter(x=np.r_[xs_, xs_[::-1]], y=np.r_[fr_[f"upper {lv}%"], fr_[f"lower {lv}%"][::-1]], fill="toself",
                                                 fillcolor=f"rgba(31,78,121,{a})", line=dict(width=0), name=f"{lv}%"))
                    fig.add_trace(go.Scatter(x=xs_, y=fr_["median"], line=dict(color=PAL[0], width=2.5), name=tr("Median", "الوسيط")))
                    ch.style_academic(fig, title=vn, height=320, legend="none", rtl=is_ar())
                    _chart(fig, f"bv_fc_{i}", f"bvar_{vn}")
            with st.expander(tr("Posterior mean coefficients vs OLS", "المعاملات البعدية مقارنة بـ OLS"), expanded=False):
                st.dataframe(pd.concat({tr("BVAR posterior mean", "المتوسط البعدي"): r["coef"], "OLS": r["ols"]}, axis=1).round(4), width="stretch")
            excel_download({vn: r["forecasts"][vn] for vn in vs} | {"Coefficients": r["coef"].reset_index()}, "bvar.xlsx", key="bv_xl")
        else:
            c = st.columns(5)
            sh = c[0].selectbox(tr("Shock", "الصدمة"), vs, key="lp_s")
            rs = c[1].selectbox(tr("Response", "الاستجابة"), vs, 1 if len(vs) > 1 else 0, key="lp_r")
            H = c[2].number_input(tr("Horizons", "الآفاق"), 4, 40, 12, key="lp_H")
            p = c[3].number_input(tr("Lags", "الإبطاءات"), 1, 8, 2, key="lp_p")
            cum = c[4].checkbox(tr("Cumulative", "تراكمي"), key="lp_c")
            lp = nc.local_projection(Y, sh, rs, int(H), int(p), cumulative=cum)
            from analytics import var as vr
            vrr = vr.fit_var(Y, int(p), "const")
            I = np.asarray(vr.irf(vrr, int(H), "orth"))
            virf = I[:, vs.index(rs), vs.index(sh)]
            if cum:
                virf = np.cumsum(virf)
            fig = go.Figure([go.Scatter(x=np.r_[lp.h, lp.h[::-1]], y=np.r_[lp.upper, lp.lower[::-1]], fill="toself", fillcolor="rgba(31,78,121,0.18)",
                                        line=dict(width=0), name="90%"),
                             go.Scatter(x=lp.h, y=lp.IRF, line=dict(color=PAL[0], width=2.5), name=tr("Local projection", "الإسقاط المحلي")),
                             go.Scatter(x=np.arange(len(virf)), y=virf, line=dict(color=PAL[1], dash="dash"), name="VAR")])
            fig.add_hline(y=0, line=dict(color="#999"))
            ch.style_academic(fig, height=400, x_title=tr("Horizon", "الأفق"), y_title=f"{sh} → {rs}", legend="bottom", rtl=is_ar())
            _chart(fig, "lp_fig", "local_projection")
            _table(lp, 4)
            insight(tr("Local projections estimate each horizon by a separate regression: more robust to misspecification than the VAR "
                       "(dashed) but less precise; the two coincide at h = 1 by construction.",
                       "تقدّر الإسقاطات المحلية كل أفق بانحدار منفصل: أكثر متانة من VAR (المتقطع) أمام سوء التوصيف وأقل دقة؛ ويتطابقان "
                       "عند h = 1 بالبناء."))
            st.markdown("**" + tr("Direct multi-step forecasts", "التنبؤات المباشرة متعددة الخطوات") + f" — {rs}**")
            _table(nc.lp_direct_forecast(Y, rs, 8, int(p)), 3)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return
    _method("macro", "bvm")
    _code(fr.code("macro", time=t, variables=vs), "bv_code")


# ---------------------------------------------------------- volatility & VaR
def _volatility(df):
    num = ds.numeric_columns(df)
    c1, c2, c3 = st.columns(3)
    src = c1.selectbox(tr("Returns (%) or prices", "العوائد (%) أو الأسعار"), num, key="vf_r")
    is_price = c2.checkbox(tr("Column holds prices → log returns × 100", "العمود أسعار ← عوائد لوغاريتمية × 100"),
                           bool((df[src].dropna() > 0).all() and df[src].dropna().mean() > 20), key=f"vf_p_{src}")
    rvc = c3.selectbox(tr("Realized variance (optional)", "التباين المحقق (اختياري)"), ["—"] + [c for c in num if c != src], key="vf_rv")
    x = df[src].dropna().to_numpy(float)
    if is_price and np.any(x <= 0):
        st.error(tr("Prices must be positive to compute log returns.", "يجب أن تكون الأسعار موجبة لحساب العوائد اللوغاريتمية."))
        return
    ret = 100 * np.diff(np.log(x)) if is_price else x
    rv = None
    if rvc != "—":
        rv = df[rvc].to_numpy(float)
        rv = rv[1:] if is_price else rv
        ok = np.isfinite(rv) & np.isfinite(ret[:len(rv)]) if len(rv) == len(ret) else None
        if ok is None or not ok.all():
            st.warning(tr("Returns and realized variance must align without gaps.", "يجب أن تتطابق العوائد والتباين المحقق دون فجوات."))
            rv = None
    if len(ret) < 600:
        st.info(tr("At least ~600 daily observations are needed for rolling volatility forecasts.", "يلزم نحو 600 مشاهدة يومية على الأقل لتنبؤات التقلب المتدحرجة."))
        return
    if rv is not None:
        st.markdown("#### HAR-RV (Corsi 2009)")
        c1, c2 = st.columns(2)
        lg = c1.checkbox(tr("Log HAR", "HAR لوغاريتمي"), True, key="vf_log")
        hz = c2.number_input(tr("Forecast days", "أيام التنبؤ"), 1, 66, 22, key="vf_h")
        hr = F.har(rv, 1, lg, int(hz))
        _table(hr["table"], 4)
        st.caption(tr(f"R² = {hr['R²']:.3f}; persistence b_d + b_w + b_m = {hr['persistence']:.3f}", f"R² = {hr['R²']:.3f}؛ الاستمرارية b_d + b_w + b_m = {hr['persistence']:.3f}"))
        tail = 250
        fig = go.Figure([go.Scatter(x=np.arange(-tail + 1, 1), y=rv[-tail:], name=tr("Realized variance", "التباين المحقق"), line=dict(color="#999")),
                         go.Scatter(x=np.arange(1, int(hz) + 1), y=hr["forecast"], name=tr("HAR forecast", "تنبؤ HAR"), line=dict(color=PAL[0], width=2.5))])
        ch.style_academic(fig, height=320, x_title=tr("Days", "الأيام"), legend="bottom", rtl=is_ar())
        _chart(fig, "vf_har", "har_forecast")
        b = hr["b"]
        insight(tr(f"Daily, weekly and monthly components weigh {b[1]:.2f}, {b[2]:.2f} and {b[3]:.2f}: the slow-decaying (long-memory) "
                   "pattern of volatility is captured by the cascade of horizons.",
                   f"أوزان المكونات اليومية والأسبوعية والشهرية {b[1]:.2f} و{b[2]:.2f} و{b[3]:.2f}: يلتقط تتالي الآفاق نمط التضاؤل البطيء "
                   "(الذاكرة الطويلة) في التقلب."))
    c1, c2, c3 = st.columns(3)
    lvl = c1.selectbox(tr("VaR level", "مستوى VaR"), [0.01, 0.025, 0.05], key="vf_lvl", format_func=lambda a: f"{a * 100:g}%")
    win = c2.number_input(tr("Estimation window", "نافذة التقدير"), 250, 3000, min(1000, len(ret) // 2), key="vf_win")
    ref = c3.number_input(tr("Re-estimate every (days)", "إعادة التقدير كل (يوم)"), 1, 250, 20, key="vf_ref")
    sig = _sig(("vol", src, rvc, is_price, lvl, win, ref))
    st.caption(tr("GARCH models are re-estimated on a rolling window every few days and filtered daily in between.", "تُعاد تقدير نماذج GARCH على نافذة متحركة كل بضعة أيام وتُرشَّح يومياً بين ذلك."))
    if st.button(tr("▶️ Backtest VaR and volatility forecasts", "▶️ الاختبار التاريخي لتنبؤات VaR والتقلب"), type="primary", key="vf_run"):
        with st.spinner(tr("Rolling estimation…", "جارٍ التقدير المتدحرج…")):
            vb = F.var_backtest(ret, float(lvl), int(win), int(ref), rv)
            vc = F.vol_forecast_compare(ret, rv, int(win), int(ref)) if rv is not None else None
        _store("vol", (vb, vc), sig)
    res = _load("vol", sig)
    if res is None:
        return
    vb, vc = res
    st.markdown("**" + tr("Value-at-Risk backtest (one day ahead)", "الاختبار التاريخي للقيمة المعرضة للخطر (يوم واحد)") + "**")
    _table(vb["table"].drop(columns=["key"]), 3)
    best = vb["table"].iloc[0]
    insight(tr(f"Lowest quantile loss: <b>{best.Method}</b> with {best.violations} violations vs {best.expected:.1f} expected "
               f"(Kupiec p = {fp(best['p Kupiec'])}, conditional coverage p = {fp(best['p conditional coverage'])}).",
               f"أقل خسارة مئين: <b>{best.Method}</b> بعدد {best.violations} انتهاكاً مقابل {best.expected:.1f} متوقعاً "
               f"(p كوبيك = {fp(best['p Kupiec'])}، وp التغطية المشروطة = {fp(best['p conditional coverage'])})."))
    s0 = vb["start"]
    xx = np.arange(s0, len(ret))
    fig = go.Figure([go.Scatter(x=xx, y=ret[s0:], mode="markers", marker=dict(size=3, color="#999"), name=tr("Return", "العائد"))])
    for i, (k, path) in enumerate(vb["paths"].items()):
        fig.add_trace(go.Scatter(x=xx, y=path[s0:], name=_t(F.VAR_METHODS[k]), line=dict(color=PAL[i % len(PAL)], width=1.4)))
    ch.style_academic(fig, height=420, y_title=tr("Return and VaR", "العائد وVaR"), legend="bottom", rtl=is_ar())
    _chart(fig, "vf_var", "var_backtest")
    if vc is not None:
        st.markdown("**" + tr("Variance forecasts vs realized variance", "تنبؤات التباين مقابل التباين المحقق") + "**")
        _table(vc["table"], 4)
        _table(vc["dm"].rename(columns={"p": "p-value"}), 3)
    _method("volatility", "vfm")
    excel_download({"VaR backtest": vb["table"]} | ({"Volatility": vc["table"]} if vc is not None else {}), "volatility_forecasts.xlsx", key="vf_xl")
    _code(fr.code("volatility", ret=src, rv=rvc), "vf_code")


# ---------------------------------------------------------- machine learning
def _ml(df):
    p = _pick(df, "fm")
    if p is None:
        return
    s, t, v, h = p
    y = s.y
    ya = y.to_numpy()
    av = F.ml_available()
    keys = [k for k in F.ML_MODELS if av.get(k)]
    if not keys:
        st.error(tr("scikit-learn is not installed.", "مكتبة scikit-learn غير مثبتة."))
        return
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    ks = c1.multiselect(tr("Models", "النماذج"), keys + ["prophet"], [k for k in ("gbm", "rf", "lgbm") if k in keys] + ["prophet"], key="fm_k",
                        format_func=lambda k: tr("Prophet-style", "على نمط Prophet") if k == "prophet" else _t(F.ML_MODELS[k]))
    lags = c2.number_input(tr("Lags", "الإبطاءات"), 1, 36, max(2, min(s.s, 12)) if s.s > 1 else 6, key="fm_l")
    dif = c3.selectbox(tr("Differencing", "الفروق"), ["auto", "yes", "no"], key="fm_d",
                       format_func=lambda d: {"auto": tr("auto (KPSS)", "تلقائي (KPSS)"), "yes": tr("first differences", "فروق أولى"), "no": tr("levels", "مستويات")}[d])
    test = c4.number_input(tr("Hold-out", "العينة المحجوزة"), h, max(h, len(ya) // 3), min(max(h, 12), len(ya) // 3), key="fm_test")
    missing = [k for k in ("xgb", "cat") if not av.get(k)]
    if missing:
        st.caption(tr("XGBoost / CatBoost are used when installed (pip install xgboost catboost); LightGBM-style histogram boosting is always available.",
                      "يُستخدم XGBoost وCatBoost عند تثبيتهما (pip install xgboost catboost)؛ ويتوفر دائماً التعزيز بالهيستوجرام على نمط LightGBM."))
    sig = _sig(("ml", v, s.transform, s.s, h, tuple(ks), lags, dif, test))
    if st.button(tr("▶️ Train and evaluate", "▶️ التدريب والتقييم"), type="primary", key="fm_run") and ks:
        dd = None if dif == "auto" else dif == "yes"
        out, acc = {}, []
        bar = st.progress(0.0)
        tr_, te = ya[:-int(test)], ya[-int(test):]
        for i, k in enumerate(ks):
            bar.progress(i / len(ks), text=k)
            try:
                fn = (lambda yy, hh: F.prophet_style(yy, hh, s.s)) if k == "prophet" else (lambda yy, hh, k=k: F.ml_forecast(yy, hh, k, s.s, int(lags), dd))
                out[k] = fn(ya, h)
                ho = fn(tr_, int(test))
                acc.append({"Model": _nm(out[k].name), **F.accuracy(te, ho.mean, tr_, s.s)})
            except Exception as e:  # noqa: BLE001
                st.warning(f"{k}: {e}")
        sn = F.snaive(tr_, int(test), s.s) if s.s > 1 else F.naive(tr_, int(test))
        acc.append({"Model": _nm(sn.name) + " (" + tr("benchmark", "مرجعي") + ")", **F.accuracy(te, sn.mean, tr_, s.s)})
        bar.empty()
        _store("ml", (out, pd.DataFrame(acc)), sig)
    res = _load("ml", sig)
    if res is None:
        return
    out, acc = res
    st.markdown("**" + tr("Hold-out accuracy", "دقة العينة المحجوزة") + "**")
    _table(acc.sort_values("MASE"), 4)
    best = acc.sort_values("MASE").iloc[0]
    insight(tr(f"Most accurate on the hold-out: <b>{best.Model}</b> (MASE = {best.MASE:.3f}).", f"الأدق على العينة المحجوزة: <b>{best.Model}</b> (MASE = {best.MASE:.3f})."))
    fan_chart(y, list(out.values()), h, "fm_fan", "ml_forecasts", title_y=s.name)
    for k, f in out.items():
        imp = f.info.get("importance")
        if imp is not None and len(imp):
            with st.expander(tr(f"Permutation importance — {f.name}", f"الأهمية بالتبديل — {_nm(f.name)}"), expanded=False):
                im = imp.head(12)
                fig = go.Figure(go.Bar(x=im["Importance"][::-1], y=im["Feature"][::-1], orientation="h", marker_color=PAL[0]))
                ch.style_academic(fig, height=360, legend="none", rtl=is_ar())
                _chart(fig, f"fm_imp_{k}", f"importance_{k}")
        if k == "prophet" and len(f.info["changepoints"]):
            with st.expander(tr("Prophet-style trend and changepoints", "اتجاه نمط Prophet ونقاط التغيّر"), expanded=False):
                fig = go.Figure([go.Scatter(x=y.index, y=ya, name=tr("Observed", "المشاهد"), line=dict(color="#bbb")),
                                 go.Scatter(x=y.index, y=f.info["trend"], name=tr("Trend", "الاتجاه"), line=dict(color=PAL[0], width=2.5))])
                cp = f.info["changepoints"]
                for _, rw in cp[cp["rate change"].abs() > 1e-6 * max(1, np.abs(ya).max())].iterrows():
                    fig.add_vline(x=y.index[int(rw.iloc[0] * (len(ya) - 1))], line=dict(color=PAL[1], dash="dot", width=1))
                ch.style_academic(fig, height=340, legend="bottom", rtl=is_ar())
                _chart(fig, "fm_cp", "prophet_trend")
    _method("ml", "fmm")
    excel_download({"Hold-out accuracy": acc} | {k[:28]: f.frame(F.future_index(y.index, h)) for k, f in out.items()}, "ml_forecasts.xlsx", key="fm_xl")
    _code(fr.code("ml", time=t, y=v, h=h, s=s.s), "fm_code")


# ---------------------------------------------------------- evaluation & combination
def _evaluation(df):
    p = _pick(df, "fe")
    if p is None:
        return
    s, t, v, h = p
    y = s.y
    ya = y.to_numpy()
    sig_r = _sig(("race", v, s.transform, s.s, h, tuple(st.session_state.get("fr_models", [])), st.session_state.get("fr_orig", 8),
                  st.session_state.get("fr_metric", "MASE")))
    r = _load("race", sig_r)
    fast = [k for k in ("naive", "snaive", "drift", "ets", "theta", "arima", "ss") if s.s > 1 or k != "snaive"]
    if r is None:
        st.caption(tr("Uses the rolling-origin errors of the horse-race. Run it there, or run a fast race of the statistical models here.",
                      "يستخدم أخطاء الأصل المتدحرج من مسابقة النماذج. شغّلها هناك، أو شغّل هنا مسابقة سريعة للنماذج الإحصائية."))
        sig_f = _sig(("race_fast", v, s.transform, s.s, h))
        if st.button(tr("▶️ Fast race (statistical models)", "▶️ مسابقة سريعة (النماذج الإحصائية)"), key="fe_run"):
            with st.spinner(tr("Rolling-origin evaluation…", "جارٍ التقييم بالأصل المتدحرج…")):
                _store("race_fast", F.horse_race(ya, h, s.s, fast, 8), sig_f)
        r = _load("race_fast", sig_f)
    if r is None:
        return
    keys = [k for k in r["table"]["key"] if k != "combo"]
    E = {k: r["errors"][k] for k in keys}
    st.markdown("**" + tr("Accuracy by horizon (rolling origin)", "الدقة حسب الأفق (الأصل المتدحرج)") + "**")
    rows = []
    for k in keys:
        e = E[k]
        rows.append({"Model": r["table"].set_index("key").loc[k, "Model"], **{f"RMSE h={j + 1}": float(np.sqrt(np.nanmean(e[:, j] ** 2))) for j in range(min(h, 6))}})
    _table(pd.DataFrame(rows), 4)
    st.markdown("**" + tr("Pairwise Diebold-Mariano p-values (h = 1)", "قيم p لاختبار ديبولد-ماريانو الثنائي (h = 1)") + "**")
    M = pd.DataFrame(index=[_t(F.RACE[k]) for k in keys], columns=[_t(F.RACE[k]) for k in keys], dtype=float)
    for a in keys:
        for b in keys:
            if a != b:
                e1, e2 = E[a][:, 0], E[b][:, 0]
                ok = np.isfinite(e1) & np.isfinite(e2)
                if ok.sum() >= 4:
                    M.loc[_t(F.RACE[a]), _t(F.RACE[b])] = F.dm_test(e1[ok], e2[ok])["p"]
    st.dataframe(M.round(3), width="stretch")
    st.markdown("#### " + tr("Forecast combination", "تجميع التنبؤات"))
    sel = st.multiselect(tr("Models to combine", "النماذج المراد تجميعها"), keys, keys[:4], key="fe_sel", format_func=lambda k: _t(F.RACE[k]))
    if len(sel) < 2:
        return
    Em = np.column_stack([E[k][:, 0] for k in sel])
    ok = np.isfinite(Em).all(1)
    init, step = r["initial"], r["step"]
    act = np.array([ya[init + o * step] for o in range(Em.shape[0])])
    Fh = act[:, None] - Em
    rows, combos = [], {}
    fcs = []
    for k in sel:
        if k in r["forecasts"]:
            fcs.append(r["forecasts"][k])
        else:
            fcs.append(F.race_model(k, s.s)(ya, h))
    for m in F.COMBOS:
        if m == "aic":
            continue
        try:
            w = F.combination_weights(Em[ok], m, Fh[ok], act[ok])
            c = F.combine(fcs, m, w)
            combos[m] = c
            rows.append({"Method": _t(F.COMBOS[m]), **{_t(F.RACE[k]): wi for k, wi in zip(sel, w)}})
        except Exception as e:  # noqa: BLE001
            st.warning(f"{m}: {e}")
    _table(pd.DataFrame(rows), 3)
    st.caption(tr("Weights from the h = 1 backtest errors (median and trimmed mean use equal treatment). Combination often beats the "
                  "individual models because their errors are imperfectly correlated (Bates & Granger 1969).",
                  "الأوزان من أخطاء الاختبار التاريخي عند h = 1 (والوسيط والمتوسط المشذّب يعاملان النماذج بالتساوي). وكثيراً ما يتفوق "
                  "التجميع على النماذج المنفردة لأن أخطاءها غير تامة الارتباط (بيتس وجرينجر 1969)."))
    m_ = st.selectbox(tr("Combination to display", "التجميع المعروض"), list(combos), key="fe_cm", format_func=lambda k: _t(F.COMBOS[k]))
    fan_chart(y, [combos[m_]] + fcs, h, "fe_fan", "combination", title_y=s.name)
    _table(combos[m_].frame(F.future_index(y.index, h)), 3)
    _method("eval", "fem")
    excel_download({"Weights": pd.DataFrame(rows), "Combined forecast": combos[m_].frame(F.future_index(y.index, h)), "DM p-values": M.reset_index()},
                   "forecast_combination.xlsx", key="fe_xl")


# ---------------------------------------------------------- scenarios
def _scenarios(df):
    p = _pick(df, "fs")
    if p is None:
        return
    s, t, v, h = p
    y = s.y
    num = [c for c in ds.numeric_columns(df) if c not in (v, t)]
    xs = st.multiselect(tr("Drivers (explanatory variables)", "المحركات (المتغيرات المفسرة)"), num, num[:1], key="fs_x", placeholder=tr("Choose…", "اختر…"))
    if not xs:
        return
    d = _align(df, t, xs, y.index)
    if d is None:
        st.warning(tr("Drivers have gaps over the sample.", "المحركات بها فجوات في فترة العينة."))
        return
    c = st.columns(7)
    o = tuple(int(c[i].number_input(lab, 0, 3 if lab != "d" else 2, dv, key=f"fs_{lab}")) for i, (lab, dv) in enumerate((("p", 1), ("d", 1), ("q", 0))))
    so = tuple(int(c[3 + i].number_input(lab, 0, 1, dv, key=f"fs_{lab}", disabled=s.s < 2)) for i, (lab, dv) in enumerate((("P", 0), ("D", 1 if s.s > 1 else 0), ("Q", 1 if s.s > 1 else 0))))
    st.markdown("**" + tr("Scenario paths for the drivers (growth per period, %)", "مسارات المحركات في السيناريوهات (النمو لكل فترة، %)") + "**")
    spec = {"Optimistic": {}, "Pessimistic": {}}
    for x in xs:
        cc = st.columns(3)
        cc[0].markdown(f"`{x}` — {tr('baseline: trend projection', 'الأساسي: إسقاط الاتجاه')}")
        g1 = cc[1].number_input(tr(f"{x}: optimistic %", f"{x}: متفائل %"), -50.0, 50.0, 1.0, 0.1, key=f"fs_o_{x}")
        g2 = cc[2].number_input(tr(f"{x}: pessimistic %", f"{x}: متشائم %"), -50.0, 50.0, -1.0, 0.1, key=f"fs_p_{x}")
        spec["Optimistic"][x] = ("growth", g1)
        spec["Pessimistic"][x] = ("growth", g2)
    try:
        paths = F.scenario_paths(d, h, spec)
        res = F.scenario_forecast(y.to_numpy(), d, paths, h, o, so if s.s > 1 else (0, 0, 0), s.s)
    except Exception as e:  # noqa: BLE001
        st.error(str(e))
        return
    names = {"Baseline": tr("Baseline", "الأساسي"), "Optimistic": tr("Optimistic", "المتفائل"), "Pessimistic": tr("Pessimistic", "المتشائم")}
    fcs = []
    for k, f in res.items():
        f.name = names[k]
        fcs.append(f)
    fan_chart(y, fcs, h, "fs_fan", "scenario_forecasts", title_y=s.name)
    b = res["Baseline"]
    _table(b.info["table"], 4)
    fidx = F.future_index(y.index, h)
    tab = pd.DataFrame({"time": fidx, **{names[k]: f.mean for k, f in res.items()}})
    _table(tab, 3)
    last = float(y.iloc[-1])
    insight(tr(f"After {h} periods: baseline {b.mean[-1]:,.4g}; optimistic {res['Optimistic'].mean[-1]:,.4g} "
               f"({(res['Optimistic'].mean[-1] / b.mean[-1] - 1) * 100:+.1f}% vs baseline); pessimistic {res['Pessimistic'].mean[-1]:,.4g} "
               f"({(res['Pessimistic'].mean[-1] / b.mean[-1] - 1) * 100:+.1f}%). Intervals are conditional on the driver paths.",
               f"بعد {h} فترة: الأساسي {b.mean[-1]:,.4g}؛ والمتفائل {res['Optimistic'].mean[-1]:,.4g} "
               f"({(res['Optimistic'].mean[-1] / b.mean[-1] - 1) * 100:+.1f}% عن الأساسي)؛ والمتشائم {res['Pessimistic'].mean[-1]:,.4g} "
               f"({(res['Pessimistic'].mean[-1] / b.mean[-1] - 1) * 100:+.1f}%). والفترات مشروطة بمسارات المحركات."))
    del last
    txt, refs = _method("scenario", "fsm")
    c1, c2 = st.columns(2)
    with c1:
        excel_download({"Scenarios": tab, "Model": b.info["table"], **{f"Drivers {k}": pth for k, pth in paths.items()}}, "scenario_forecasts.xlsx", key="fs_xl")
    with c2:
        st.download_button(tr("⬇️ Policy brief (Word)", "⬇️ موجز سياسات (Word)"),
                           rp.to_docx([("h", tr("Scenario forecasts", "تنبؤات السيناريوهات")), ("table", tab.round(3).astype(str)),
                                       ("h", tr("Methodology", "المنهجية")), ("p", txt), ("bullets", refs)],
                                      tr("Scenario forecasts for decision-makers", "تنبؤات السيناريوهات لصانع القرار"), is_ar()),
                           "scenario_brief.docx", key="fs_docx")
    _code(fr.code("scenario", time=t, y=v, h=h, drivers=xs), "fs_code")
