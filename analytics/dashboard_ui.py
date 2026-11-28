# ==========================================
# Aetheris Research - Decision-maker dashboard (UI) — v1.12, Catalogue §8
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import ai_assistant as ai
from analytics import charts as ch
from analytics import dashboard as db
from analytics import report_studio as rs
from analytics import ts_ui as tu
from core import report_store as store
from core.memory import active_name, get_memory, has_data, set_memory
from core.utils import tr, is_ar, insight

PAL = ch.PALETTES["academic"]
COL = {"green": "#2e7d32", "amber": "#f9a825", "red": "#c62828", "na": "#9e9e9e"}
ICON = {"green": "🟢", "amber": "🟠", "red": "🔴", "na": "⚪"}


def _t(p):
    return p[1] if is_ar() else p[0]


def _lang():
    return "ar" if is_ar() else "en"


def _chart(fig, key, name):
    st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, name), key=key)


def _fmt(v, d=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "—"
    return f"{v:,.{d}f}"


def render_dashboard():
    st.caption("💡 " + tr("For decision makers: the headline numbers, early-warning lights, what-if scenarios and the key findings of your analyses — "
                         "in plain language.",
                         "لصانع القرار: الأرقام الرئيسية، وأضواء الإنذار المبكر، وسيناريوهات «ماذا لو»، وأهم نتائج تحليلاتك — بلغة مبسطة."))
    tabs = st.tabs([tr("📌 Key findings", "📌 أهم النتائج"), tr("📊 Indicators & early warning", "📊 المؤشرات والإنذار المبكر"),
                    tr("🎛️ What-if scenarios", "🎛️ سيناريوهات ماذا لو")])
    with tabs[0]:
        _findings()
    with tabs[1]:
        _ews()
    with tabs[2]:
        _scenarios()


# ---------------------------------------------------------- key findings (non-technical)
def _findings():
    res = store.results()
    if not res:
        st.info(tr("Key findings appear here once you run analyses (each analysis's interpretation is summarised for non-specialists).",
                   "تظهر أهم النتائج هنا بعد تشغيل التحليلات (يُلخَّص تفسير كل تحليل لغير المتخصصين)."))
        return
    R = sorted(res.values(), key=lambda r: -r["time"])
    for r in R[:8]:
        with st.container(border=True):
            st.markdown(f"**{r['title']}**")
            if r["insights"]:
                st.markdown(rs.strip_html(r["insights"][0]))
            else:
                st.markdown(rs.results_paragraph(r, _lang())[:600] or tr("(no interpretation recorded)", "(لا يوجد تفسير مسجل)"))
    from analytics.ai_settings_ui import compact_status
    llm = compact_status()
    if st.button(tr("🗞️ Write a one-page policy brief from these results", "🗞️ اكتب موجز سياسات من صفحة واحدة من هذه النتائج"), key="db_brief", type="primary"):
        try:
            with st.spinner("…"):
                out = ai.write("policy", R[:6], _lang(), llm if llm.available else None)
        except Exception as e:  # noqa: BLE001
            st.warning(tr(f"AI unavailable ({e}); offline summary.", f"الذكاء الاصطناعي غير متاح ({e})؛ ملخص دون اتصال."))
            out = ai.write("policy", R[:6], _lang(), None)
        st.session_state["_db_brief"] = out
    b = st.session_state.get("_db_brief")
    if b:
        st.text_area(tr("Policy brief", "موجز السياسات"), b["text"], height=280, key="db_brief_txt")
        if b["unverified"]:
            st.warning(tr("Check these numbers (not found in the results): ", "راجع هذه الأرقام (غير موجودة في النتائج): ") + ", ".join(b["unverified"][:10]))
        from analytics.reg_report import to_docx
        st.download_button(tr("⬇️ Brief (Word)", "⬇️ الموجز (Word)"), to_docx([("p", x) for x in b["text"].split("\n") if x.strip()], tr("Policy brief", "موجز السياسات"), is_ar()),
                           "policy_brief.docx", key="db_brief_dl")


# ---------------------------------------------------------- indicators & early warning
def _ews():
    if not has_data():
        st.info(tr("Load indicator data, or try the macro demo:", "حمّل بيانات المؤشرات، أو جرّب البيانات الكلية التجريبية:"))
    if st.button(tr("🏛️ Load macro early-warning demo (monthly, synthetic)", "🏛️ تحميل بيانات الإنذار المبكر الكلية (شهرية، محاكاة)"), key="db_demo"):
        from data.samples import macro_dashboard_demo
        set_memory(macro_dashboard_demo(), source="demo (synthetic)", name="Demo – macro early warning (synthetic)")
        st.rerun()
    if not has_data():
        return
    df = get_memory()
    times = [c for c in df.columns if c.lower() in ("date", "year", "month", "quarter", "time", "period") or pd.api.types.is_datetime64_any_dtype(df[c])]
    num = [c for c in df.select_dtypes("number").columns if c not in times and not (set(pd.unique(df[c].dropna())) <= {0, 1})]
    if not times or not num:
        st.info(tr("Needs a time column and numeric indicators.", "يحتاج إلى عمود زمني ومؤشرات رقمية."))
        return
    st.caption(tr(f"Dataset: **{active_name()}**", f"البيانات: **{active_name()}**"))
    c1, c2, c3 = st.columns([1, 3, 1])
    t = c1.selectbox(tr("Time", "الزمن"), times, key="db_t")
    inds = c2.multiselect(tr("Indicators to monitor", "المؤشرات المراد مراقبتها"), num, num[:6], key="db_i", placeholder=tr("Choose…", "اختر…"))
    unit = None
    cats = [c for c in df.columns if df[c].dtype == object and 1 < df[c].nunique() <= 300]
    if cats:
        u = c3.selectbox(tr("Unit (optional)", "الوحدة (اختياري)"), [None] + cats, key="db_u", format_func=lambda v: tr("All (mean)", "الكل (متوسط)") if v is None else v)
        if u:
            unit = (u, st.selectbox(tr("Which", "أيّها"), sorted(df[u].dropna().unique()), key="db_uv"))
    if not inds:
        return
    cfg = st.session_state.setdefault("_db_cfg", {})
    with st.expander(tr("⚙️ Warning rules per indicator", "⚙️ قواعد الإنذار لكل مؤشر"), expanded=False):
        st.caption(tr("Direction: which movement is dangerous. Method: z-score vs history (1 / 2 SD), historical percentile (80 / 95), gap from a real-time HP "
                      "trend (2 / 10 %) or your own fixed thresholds.",
                      "الاتجاه: أي حركة خطرة. الطريقة: الانحراف المعياري عن التاريخ (1 / 2)، أو المئين التاريخي (80 / 95)، أو الفجوة عن اتجاه HP الآني "
                      "(2 / 10 %)، أو عتبات ثابتة تحددها."))
        for v in inds:
            lv = v.lower()
            dflt_bad = "low" if any(k in lv for k in ("reserve", "growth_gdp", "export", "account", "income")) and "credit" not in lv else "high"
            a, b, c, d = st.columns(4)
            bad = a.selectbox(v, ["high", "low", "both"], ["high", "low", "both"].index(cfg.get(v, {}).get("bad", dflt_bad)), key=f"db_bad_{v}",
                              format_func=lambda k: {"high": tr("rise is bad", "الارتفاع خطر"), "low": tr("fall is bad", "الانخفاض خطر"), "both": tr("both", "كلاهما")}[k])
            meth = b.selectbox(tr("Method", "الطريقة"), list(db.METHODS), list(db.METHODS).index(cfg.get(v, {}).get("method", "z")), key=f"db_m_{v}",
                               format_func=lambda k: _t(db.METHODS[k]))
            am = rd = None
            if meth == "fixed":
                s0 = df[v].dropna()
                am = c.number_input(tr("Amber", "أصفر"), value=float(np.round(s0.quantile(0.8 if bad != "low" else 0.2), 3)), key=f"db_a_{v}")
                rd = d.number_input(tr("Red", "أحمر"), value=float(np.round(s0.quantile(0.95 if bad != "low" else 0.05), 3)), key=f"db_r_{v}")
            cfg[v] = {"bad": bad, "method": meth, "amber": am, "red": rd}
    sigs, kp = {}, {}
    for v in inds:
        s = db.series(df, t, v, unit[0] if unit else None, unit[1] if unit else None)
        c = cfg.get(v, {"bad": "high", "method": "z"})
        try:
            sigs[v] = db.signal(s, c["method"], c["bad"], c.get("amber"), c.get("red"), lamb=129600.0 if len(s) > 120 else 1600.0)
        except Exception as e:  # noqa: BLE001
            st.warning(f"{v}: {e}")
            continue
        kp[v] = db.kpi(s)
    if not sigs:
        return
    comp = db.composite(sigs)
    last = comp.dropna(subset=["index"]).iloc[-1] if comp["index"].notna().any() else None
    if last is not None:
        stt = last["status"]
        st.markdown(f"### {ICON[stt]} " + tr(f"Composite early-warning index: {last['index']:.0f}/100 — {_t(db.STATUS[stt])}",
                                              f"المؤشر المركب للإنذار المبكر: {last['index']:.0f}/100 — {_t(db.STATUS[stt])}"))
    cols = st.columns(min(4, len(sigs)))
    for i, (v, sg) in enumerate(sigs.items()):
        k = kp[v]
        stt = sg.status.iloc[-1]
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"{ICON[stt]} **{v}**")
                st.metric(tr("Latest", "الأحدث"), _fmt(k["latest"]), None if not np.isfinite(k.get("pct", np.nan)) else f"{k['pct']:+.1f}%",
                          delta_color="inverse" if cfg[v]["bad"] == "high" else "normal")
                st.caption(f"{_t(db.STATUS[stt])} · " + tr(f"signal {sg.signal.iloc[-1]:.2f}", f"الإشارة {sg.signal.iloc[-1]:.2f}") if np.isfinite(sg.signal.iloc[-1]) else _t(db.STATUS[stt]))
                fig = go.Figure(go.Scatter(x=sg.index, y=sg.value, line=dict(color=COL[stt], width=2), hoverinfo="x+y"))
                fig.update_layout(height=110, margin=dict(l=0, r=0, t=0, b=0), xaxis=dict(visible=False), yaxis=dict(visible=False), showlegend=False)
                st.plotly_chart(fig, config={"displayModeBar": False}, key=f"db_sp_{v}")
    v = st.selectbox(tr("Indicator detail", "تفاصيل المؤشر"), list(sigs), key="db_det")
    sg = sigs[v]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sg.index, y=sg.value, line=dict(color=PAL[0], width=2.2), name=v))
    for stt in ("amber", "red"):
        m = sg.status == stt
        fig.add_trace(go.Scatter(x=sg.index[m], y=sg.value[m], mode="markers", marker=dict(color=COL[stt], size=7), name=_t(db.STATUS[stt])))
    crisis = next((c for c in df.columns if c.lower() in ("crisis", "recession", "crisis_dummy")), None)
    if crisis:
        cs = db.series(df, t, crisis, unit[0] if unit else None, unit[1] if unit else None, "max")
        for x0 in cs.index[cs > 0]:
            fig.add_vrect(x0=x0, x1=x0, line_width=6, line_color="rgba(198,40,40,0.12)")
    ch.style_academic(fig, height=360, x_title=t, y_title=v, legend="bottom", rtl=is_ar())
    _chart(fig, "db_det_ch", f"ews_{v}")
    a, b = st.columns(2)
    with a:
        comp_d = comp.dropna(subset=["index"])
        fig = go.Figure(go.Scatter(x=comp_d.index, y=comp_d["index"], line=dict(color=PAL[1], width=2.4), fill="tozeroy", fillcolor="rgba(200,120,30,0.12)"))
        fig.add_hline(y=25, line=dict(color=COL["amber"], dash="dot"))
        fig.add_hline(y=50, line=dict(color=COL["red"], dash="dot"))
        ch.style_academic(fig, height=300, x_title=t, y_title=tr("Composite index (0–100)", "المؤشر المركب (0–100)"), legend="none", rtl=is_ar())
        _chart(fig, "db_comp", "ews_composite")
    with b:
        h = st.slider(tr("Forecast horizon", "أفق التنبؤ"), 1, 24, 6, key="db_h")
        s = db.series(df, t, v, unit[0] if unit else None, unit[1] if unit else None)
        c = cfg[v]
        thr = c.get("red") if c["method"] == "fixed" else (float(np.nanpercentile(s, 95 if c["bad"] != "low" else 5)))
        try:
            fw = db.forecast_warning(s, h, thr, "low" if c["bad"] == "low" else "high")
            T = fw["table"]
            fig = go.Figure([go.Scatter(x=np.r_[T.step, T.step[::-1]], y=np.r_[T["upper 95"], T["lower 95"][::-1]], fill="toself", fillcolor="rgba(31,78,121,0.15)",
                                        line=dict(width=0), showlegend=False), go.Scatter(x=T.step, y=T.forecast, line=dict(color=PAL[0], width=2.4), showlegend=False)])
            fig.add_hline(y=thr, line=dict(color=COL["red"], dash="dash"), annotation_text=tr("red threshold", "عتبة الإنذار"))
            ch.style_academic(fig, height=300, x_title=tr("Periods ahead", "الفترات القادمة"), y_title=v, legend="none", rtl=is_ar())
            _chart(fig, "db_fc", f"ews_forecast_{v}")
            pmax = float(T["P(breach)"].max())
            st.caption(tr(f"{fw['model']}: highest probability of crossing {thr:,.2f} within {h} periods = {100 * pmax:.0f}%.",
                          f"{fw['model']}: أعلى احتمال لتجاوز {thr:,.2f} خلال {h} فترة = {100 * pmax:.0f}%."))
        except Exception as e:  # noqa: BLE001
            st.caption(tr(f"Forecast unavailable: {e}", f"التنبؤ غير متاح: {e}"))
    red = [x for x, sg_ in sigs.items() if sg_.status.iloc[-1] == "red"]
    amb = [x for x, sg_ in sigs.items() if sg_.status.iloc[-1] == "amber"]
    insight(tr(f"Latest period: {len(red)} indicator(s) in alert ({', '.join(red) or 'none'}) and {len(amb)} on watch ({', '.join(amb) or 'none'})."
               + (f" Composite index {last['index']:.0f}/100." if last is not None else ""),
               f"آخر فترة: {len(red)} مؤشر في حالة إنذار ({'، '.join(red) or 'لا يوجد'})، و{len(amb)} تحت المراقبة ({'، '.join(amb) or 'لا يوجد'})."
               + (f" المؤشر المركب {last['index']:.0f}/100." if last is not None else "")))
    if crisis:
        with st.expander(tr("🎯 Which indicators predicted past crises? (signals approach, KLR 1998)", "🎯 أي المؤشرات تنبأت بالأزمات السابقة؟ (منهج الإشارات، KLR 1998)"), expanded=False):
            hz = st.slider(tr("Warning window (periods before a crisis)", "نافذة الإنذار (فترات قبل الأزمة)"), 3, 24, 12, key="db_hz")
            cs = db.series(df, t, crisis, unit[0] if unit else None, unit[1] if unit else None, "max")
            rows = []
            for x in sigs:
                r = db.noise_to_signal(db.series(df, t, x, unit[0] if unit else None, unit[1] if unit else None), cs, hz, "low" if cfg[x]["bad"] == "low" else "high")
                b_ = r["best"]
                if b_:
                    rows.append({"Indicator": x, "Optimal threshold": b_["threshold"], "Percentile": b_["percentile"], "NtS": b_["NtS"],
                                 "Share of crises signalled": b_["share of crises signalled"], "P(crisis | signal)": b_["P(crisis | signal)"]})
            if rows:
                T = pd.DataFrame(rows).sort_values("NtS")
                st.dataframe(tu._disp(T).round(3), width="stretch", hide_index=True)
                st.caption(tr("Lower noise-to-signal (NtS < 1) = more reliable indicator. Thresholds can be set as fixed rules above.",
                              "كلما قلت نسبة الضوضاء إلى الإشارة (NtS < 1) كان المؤشر أكثر موثوقية. ويمكن اعتماد العتبات كقواعد ثابتة في الأعلى."))
    tab = pd.DataFrame([{"Indicator": x, "Latest": kp[x]["latest"], "Change %": kp[x]["pct"], "Signal": sg_.signal.iloc[-1], "Status": _t(db.STATUS[sg_.status.iloc[-1]])}
                        for x, sg_ in sigs.items()])
    from analytics.reg_report import to_docx
    st.download_button(tr("⬇️ Dashboard snapshot (Word)", "⬇️ لقطة اللوحة (Word)"),
                       to_docx([("p", tr("Early-warning snapshot", "لقطة الإنذار المبكر")), ("table", tab)], tr("Decision dashboard", "لوحة صانع القرار"), is_ar()),
                       "dashboard.docx", key="db_dl")


# ---------------------------------------------------------- what-if
def _scenarios():
    if not has_data():
        st.info(tr("Load data first.", "حمّل البيانات أولاً."))
        return
    df = get_memory()
    num = [c for c in df.select_dtypes("number").columns if c.lower() not in ("year", "month", "crisis") and not (set(pd.unique(df[c].dropna())) <= {0, 1})]
    c1, c2, c3 = st.columns([1, 2, 1])
    y = c1.selectbox(tr("Target indicator", "المؤشر المستهدف"), num, key="sc2_y")
    xs = c2.multiselect(tr("Drivers / policy levers", "المحركات / أدوات السياسة"), [c for c in num if c != y], [c for c in num if c != y][:2], key="sc2_x",
                        placeholder=tr("Choose…", "اختر…"))
    form = c3.selectbox(tr("Model", "النموذج"), ["loglog", "linear"], key="sc2_f",
                        format_func=lambda k: {"loglog": tr("% changes (elasticities)", "تغيرات % (مرونات)"), "linear": tr("Levels", "مستويات")}[k])
    if not xs:
        return
    try:
        m = db.scenario_model(df, y, xs, form)
    except Exception as e:  # noqa: BLE001
        st.warning(tr(f"Model could not be estimated: {e}", f"تعذّر تقدير النموذج: {e}"))
        return
    st.caption(tr(f"OLS with robust SEs · N = {m['N']} · R² = {m['R2']:.3f}. Associations, not necessarily causal — for causal effects use the Impact lab.",
                  f"مربعات صغرى بأخطاء متينة · N = {m['N']} · R² = {m['R2']:.3f}. علاقات ارتباطية وليست بالضرورة سببية — للآثار السببية استخدم مختبر الأثر."))
    ch_ = {}
    cols = st.columns(len(xs))
    for j, v in enumerate(xs):
        if form == "loglog":
            ch_[v] = cols[j].slider(f"{v} (%)", -50, 50, 0, 1, key=f"sc2_s_{v}")
        else:
            sd = float(df[v].std())
            ch_[v] = cols[j].slider(f"Δ {v}", -2 * sd, 2 * sd, 0.0, sd / 20 if sd > 0 else 0.1, key=f"sc2_s_{v}")
    out = db.scenario(m, ch_)
    a, b, c = st.columns(3)
    a.metric(tr(f"Expected change in {y}", f"التغير المتوقع في {y}"), f"{out['change %']:+.2f}%")
    if form == "loglog":
        b.metric(tr("95% range", "المدى 95%"), f"{out['low %']:+.1f}% … {out['high %']:+.1f}%")
    else:
        b.metric(tr("95% range", "المدى 95%"), f"{out['low']:+.3g} … {out['high']:+.3g}")
    c.metric(tr("New level (at the mean)", "المستوى الجديد (عند المتوسط)"), _fmt(out["new level"]))
    grid = np.linspace(-30, 30, 13) if form == "loglog" else np.linspace(-2, 2, 13) * float(df[xs[0]].std())
    path = [db.scenario(m, {**ch_, xs[0]: g}) for g in grid]
    fig = go.Figure([go.Scatter(x=grid, y=[p["high %"] if form == "loglog" else 100 * p["high"] / m["mean_y"] for p in path], line=dict(width=0), showlegend=False),
                     go.Scatter(x=grid, y=[p["low %"] if form == "loglog" else 100 * p["low"] / m["mean_y"] for p in path], fill="tonexty", fillcolor="rgba(31,78,121,0.15)",
                                line=dict(width=0), showlegend=False),
                     go.Scatter(x=grid, y=[p["change %"] for p in path], line=dict(color=PAL[0], width=3), showlegend=False)])
    fig.add_hline(y=0, line=dict(color="#999"))
    ch.style_academic(fig, height=330, x_title=(f"% Δ {xs[0]}" if form == "loglog" else f"Δ {xs[0]}"), y_title=f"% Δ {y}", legend="none", rtl=is_ar())
    _chart(fig, "sc2_curve", "scenario_curve")
    el = m["model"].params.drop("const")
    insight(tr("Sensitivities: " + "; ".join(f"1% higher {k} ↔ {v:+.3f}% {y}" if form == "loglog" else f"+1 {k} ↔ {v:+.4g} {y}" for k, v in el.items()) + ".",
               "الحساسيات: " + "؛ ".join(f"ارتفاع {k} بنسبة 1% ↔ {v:+.3f}% في {y}" if form == "loglog" else f"زيادة {k} وحدة ↔ {v:+.4g} في {y}" for k, v in el.items()) + "."))
