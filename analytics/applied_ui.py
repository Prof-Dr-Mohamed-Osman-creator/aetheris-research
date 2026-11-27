# ==========================================
# Aetheris Research - Applied Economics Lab (UI) — v1.10, Catalogue §2.7–2.8
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import applied_report as ar_
from analytics import charts as ch
from analytics import demand as dm
from analytics import descriptive as ds
from analytics import production as pr
from analytics import reg_report as rp
from analytics import ts_ui as tu
from core.memory import active_name, get_memory, has_data, set_memory
from core.utils import tr, is_ar, insight, excel_download

PAL = ch.PALETTES["academic"]
_disp, _code, _lang = tu._disp, tu._code, tu._lang
fp = ar_.fmt_p
tu._AR_COLS.update({"Input": "المدخل", "Elasticity": "المرونة", "Marginal product": "الناتج الحدي", "Average product": "الناتج المتوسط",
                    "MP / AP": "الحدي / المتوسط", "VMP": "قيمة الناتج الحدي", "Price": "السعر", "VMP / price": "قيمة الناتج الحدي / السعر",
                    "Recommendation": "التوصية", "Optimal quantity (CD)": "الكمية المثلى (كوب-دوجلاس)", "Mean use": "متوسط الاستخدام",
                    "Group": "المجموعة", "Mean TE": "متوسط الكفاءة", "TE (group frontier)": "الكفاءة (حد المجموعة)", "TGR": "نسبة الفجوة التقنية",
                    "MTE": "الكفاءة الفوقية", "DMU": "الوحدة", "CRS (CCR)": "ثبات العائد (CCR)", "VRS (BCC)": "تغير العائد (BCC)",
                    "Scale efficiency": "كفاءة الحجم", "Returns to scale": "العائد للسعة", "Malmquist": "مالمكويست",
                    "Efficiency change": "تغير الكفاءة", "Technical change": "التغير التقني", "Period": "الفترة", "Good": "السلعة",
                    "Mean share": "متوسط الحصة", "Expenditure elasticity": "المرونة الإنفاقية", "Class": "التصنيف", "Type": "النوع",
                    "Short-run elasticity": "المرونة قصيرة الأجل", "Long-run elasticity": "المرونة طويلة الأجل", "SE (SR)": "الخطأ (قصير)",
                    "SE (LR)": "الخطأ (طويل)", "Coef.": "المعامل", "Std. Err. (bootstrap)": "الخطأ المعياري (بوتستراب)", "TE": "الكفاءة"})
tu._AR_VALS.update({"increase": "زيادة الاستخدام", "reduce": "تخفيض الاستخدام", "near optimum": "قريب من الأمثل", "necessity": "ضرورية",
                    "luxury": "كمالية", "inferior": "دنيا", "own price": "السعر الذاتي", "cross price": "سعر تقاطعي", "income": "الدخل",
                    "CRS": "ثابت", "IRS": "متزايد", "DRS": "متناقص"})


def _t(pair):
    return pair[1] if is_ar() else pair[0]


def _chart(fig, key, name):
    st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, name), key=key)


def _table(df, digits=4):
    show = df.copy()
    for c in show.columns:
        if str(c) in ("p-value", "p") or str(c).startswith("p ("):
            show[c] = show[c].map(lambda v: fp(v) if isinstance(v, (float, np.floating)) else v)
    kw = {"height": 35 * (len(show) + 1) + 3} if len(show) <= 18 else {}
    st.dataframe(_disp(show).round(digits), width="stretch", hide_index=True, **kw)


def _method(kind, key):
    txt, refs = ar_.methodology(kind, _lang())
    with st.expander(tr("📝 Methodology paragraph & references", "📝 فقرة المنهجية والمراجع"), expanded=False):
        st.text_area("m", txt, height=150, key=f"{key}_meth", label_visibility="collapsed")
        st.markdown("\n".join(f"- {x}" for x in refs))
    return txt, refs


def _store(key, v, sig):
    st.session_state[f"_v110res_{key}"] = (sig, v)


def _load(key, sig):
    v = st.session_state.get(f"_v110res_{key}")
    return v[1] if isinstance(v, tuple) and v[0] == sig else None


_IDLIKE = ("year", "round", "quarter", "month", "replicate", "_id", "id_", "period")


def _num(df, exclude=()):
    """Numeric analysis variables: time variables and identifiers are left out."""
    tc = set(tu._time_cols(df))
    out = []
    for c in ds.numeric_columns(df):
        lc = c.lower()
        if c in exclude or c in tc or lc in ("id", "year") or any(k in lc for k in _IDLIKE):
            continue
        out.append(c)
    return out


def _prefer(cols, keys):
    """Move the first column whose name contains one of `keys` to the front (sensible defaults)."""
    for k in keys:
        for c in cols:
            if k in c.lower():
                return [c] + [x for x in cols if x != c]
    return cols


def _pos(df, cols):
    return [c for c in cols if (pd.to_numeric(df[c], errors="coerce").dropna() > 0).all()]


DEMOS = [("farm_efficiency_demo", ("🌾 Farms: production & efficiency", "🌾 المزارع: الإنتاج والكفاءة"), "Demo – farm efficiency panel (synthetic)"),
         ("fertilizer_trial_demo", ("🧪 Nitrogen trial", "🧪 تجربة السماد النيتروجيني"), "Demo – fertilizer trial (synthetic)"),
         ("firm_cost_demo", ("🏭 Flour mills: costs", "🏭 المطاحن: التكاليف"), "Demo – flour mill costs (synthetic)"),
         ("firm_tfp_demo", ("📈 Firm panel: TFP", "📈 بانل المنشآت: الإنتاجية"), "Demo – firm TFP panel (synthetic)"),
         ("household_budget_demo", ("🛒 Household food budgets", "🛒 ميزانيات الغذاء للأسر"), "Demo – household budgets (synthetic)"),
         ("supply_response_demo", ("🌱 Wheat supply response", "🌱 استجابة عرض القمح"), "Demo – wheat supply (synthetic)")]


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
def render_applied():
    st.caption("💡 " + tr("Applied production, cost, efficiency, supply and demand models with economic interpretation and policy "
                         "simulations for decision-makers.",
                         "نماذج تطبيقية للإنتاج والتكاليف والكفاءة والعرض والطلب، مع التفسير الاقتصادي ومحاكاة السياسات لصانع القرار."))
    if not has_data():
        st.info(tr("Load data first (Data Hub), or try a demo dataset:", "حمّل بيانات أولاً (مركز البيانات)، أو جرّب بيانات تجريبية:"))
        _demo_buttons("ad0")
        return
    df = get_memory()
    st.caption(tr(f"Active dataset: **{active_name()}** — {df.shape[0]} rows × {df.shape[1]} columns",
                  f"البيانات النشطة: **{active_name()}** — {df.shape[0]} صف × {df.shape[1]} عمود"))
    with st.expander(tr("🧪 Demo datasets for training", "🧪 بيانات تجريبية للتدريب"), expanded=False):
        _demo_buttons("ad1")
    tabs = st.tabs([tr("🏭 Production functions", "🏭 دوال الإنتاج"), tr("🌱 Yield response & input use", "🌱 استجابة المحصول واستخدام الموارد"),
                    tr("💰 Costs, scale & scope", "💰 التكاليف والسعة والنطاق"), tr("🎯 Efficiency (SFA, DEA)", "🎯 الكفاءة (SFA وDEA)"),
                    tr("📈 Productivity (TFP)", "📈 الإنتاجية الكلية"), tr("🔄 Supply response", "🔄 استجابة العرض"),
                    tr("🛒 Demand & Engel", "🛒 الطلب ومنحنيات إنجل"), tr("🧺 Demand systems & welfare", "🧺 أنظمة الطلب والرفاه"),
                    tr("💵 Consumption", "💵 الاستهلاك")])
    fns = [_production, _yield, _cost, _efficiency, _tfp, _supply, _demand, _systems, _consumption]
    for tb, fn in zip(tabs, fns):
        with tb:
            try:
                fn(df)
            except Exception as e:  # noqa: BLE001
                st.error(f"{type(e).__name__}: {e}")


# ---------------------------------------------------------- production
def _production(df):
    num = _num(df)
    pos = _pos(df, num)
    c1, c2, c3 = st.columns([1, 2, 1])
    y = c1.selectbox(tr("Output", "الناتج"), _prefer(pos or num, ("output", "yield", "production", "value_added", "sales")), key="ap_y")
    xs = c2.multiselect(tr("Inputs", "المدخلات"), [c for c in pos if c != y], [c for c in pos if c != y][:3], key="ap_x", placeholder=tr("Choose…", "اختر…"))
    form = c3.selectbox(tr("Functional form", "الصورة الدالية"), list(pr.FORMS), key="ap_f", format_func=lambda k: _t(pr.FORMS[k]))
    if len(xs) < 1:
        return
    if form == "ces" and len(xs) < 2:
        st.info(tr("CES needs at least two inputs.", "تحتاج CES مدخلين على الأقل."))
        return
    r = pr.production_function(df, y, xs, form)
    lang = _lang()
    if form == "leontief":
        a = r.extra
        _table(pd.DataFrame({"Input": xs, tr("Technical coefficient (median)", "المعامل الفني (الوسيط)"): a["a_median"],
                             tr("Frontier coefficient (min)", "معامل الحد (الأدنى)"): a["a_frontier"],
                             tr("Share of obs. where limiting", "نسبة المشاهدات التي يكون فيها مقيِّداً"): [a["limiting share"][v] for v in xs]}), 4)
        insight(tr("Fixed proportions: output is limited by the scarcest input; there is no substitution and marginal products are zero for "
                   "non-limiting inputs.", "نسب ثابتة: يتحدد الناتج بالمدخل الأندر؛ لا إحلال بين المدخلات، والناتج الحدي للمدخلات غير المقيِّدة صفر."))
        _method("production", "apm")
        return
    _table(r.table(), 4)
    stt = {k: v for k, v in r.stats.items() if isinstance(v, (int, float, np.floating)) and k != "N"}
    st.caption(" · ".join(f"{k} = {v:.4g}" for k, v in stt.items()) + f" · N = {r.stats.get('N', len(r.y))}")
    e = pr.elasticities(r)
    st.markdown("**" + tr("Elasticities and products at the sample mean", "المرونات والنواتج عند متوسط العينة") + "**")
    _table(e["table"], 4)
    rts = e["RTS"]
    msg = tr(f"Returns to scale = {rts:.3f}: " + ("increasing" if rts > 1.02 else ("decreasing" if rts < 0.98 else "about constant")) +
             f" — a 10% rise in all inputs raises output by about {10 * rts:.1f}%.",
             f"العائد للسعة = {rts:.3f}: " + ("متزايد" if rts > 1.02 else ("متناقص" if rts < 0.98 else "ثابت تقريباً")) +
             f" — زيادة كل المدخلات 10% ترفع الناتج نحو {10 * rts:.1f}%.")
    if form == "cd":
        msg += tr(f" Test RTS = 1: p = {fp(r.stats['p (RTS = 1)'])}.", f" اختبار ثبات العائد: p = {fp(r.stats['p (RTS = 1)'])}.")
    if form == "translog":
        msg += tr(f" Translog vs Cobb-Douglas: F p = {fp(r.stats['p (translog vs CD)'])} — " + ("Cobb-Douglas is adequate." if r.stats["p (translog vs CD)"] >= 0.05 else "prefer the translog."),
                  f" اختبار الترانسلوج مقابل كوب-دوجلاس: p = {fp(r.stats['p (translog vs CD)'])} — " + ("كوب-دوجلاس كافية." if r.stats["p (translog vs CD)"] >= 0.05 else "تُفضَّل الترانسلوج."))
    top = e["table"].sort_values("Elasticity", ascending=False).iloc[0]
    msg += tr(f" The largest output elasticity is for {top.Input} ({top.Elasticity:.3f}).", f" أكبر مرونة إنتاجية لـ {top.Input} ({top.Elasticity:.3f}).")
    insight(msg)
    if len(xs) >= 2:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**" + tr("Allen-Uzawa elasticities of substitution", "مرونات الإحلال لألن-أوزاوا") + "**")
            st.dataframe(e["AES"].round(3), width="stretch")
        with c2:
            st.markdown("**" + tr("Morishima elasticities (row i → column j)", "مرونات موريشيما (الصف i ← العمود j)") + "**")
            st.dataframe(e["MES"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        j = st.selectbox(tr("Input for the stages of production", "المدخل لمراحل الإنتاج"), xs, key="ap_stage")
        sgs = pr.stages(r, xs.index(j))
        fig = go.Figure([go.Scatter(x=sgs["x"], y=sgs["TP"], name=tr("TP", "الناتج الكلي"), line=dict(color=PAL[0], width=2.5)),
                         go.Scatter(x=sgs["x"], y=sgs["AP"], name=tr("AP", "الناتج المتوسط"), line=dict(color=PAL[1]), yaxis="y2"),
                         go.Scatter(x=sgs["x"], y=sgs["MP"], name=tr("MP", "الناتج الحدي"), line=dict(color=PAL[2], dash="dash"), yaxis="y2")])
        for xv, lab in ((sgs["end stage I"], "I | II"), (sgs["end stage II"], "II | III")):
            if np.isfinite(xv):
                fig.add_vline(x=xv, line=dict(color="#999", dash="dot"), annotation_text=lab)
        fig.add_vline(x=sgs["mean"], line=dict(color=PAL[3], width=1))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))
        ch.style_academic(fig, height=380, x_title=j, legend="bottom", rtl=is_ar())
        _chart(fig, "ap_stages", "stages_of_production")
        st.caption(tr(f"Mean use of {j} lies in stage {sgs['stage at mean']} " + ("(rational stage)." if sgs["stage at mean"] == "II" else "(irrational stage)."),
                      f"يقع متوسط استخدام {j} في المرحلة {sgs['stage at mean']} " + ("(المرحلة الرشيدة)." if sgs["stage at mean"] == "II" else "(مرحلة غير رشيدة).")))
    with c2:
        if len(xs) >= 2:
            a_, b_ = st.columns(2)
            i1 = a_.selectbox(tr("Isoquant: x-axis", "منحنى الناتج المتساوي: المحور الأفقي"), xs, 0, key="ap_iq1")
            i2 = b_.selectbox(tr("y-axis", "المحور الرأسي"), xs, 1, key="ap_iq2")
            if i1 != i2:
                a, b, Z, lv = pr.isoquants(r, xs.index(i1), xs.index(i2))
                fig = go.Figure(go.Contour(x=a, y=b, z=Z, contours=dict(coloring="lines", showlabels=True, start=float(lv[0]), end=float(lv[-1]),
                                                                          size=float((lv[-1] - lv[0]) / 2) or 1), line=dict(width=2), showscale=False))
                ch.style_academic(fig, height=380, x_title=i1, y_title=i2, legend="none", rtl=is_ar())
                _chart(fig, "ap_iso", "isoquants")
    oe = pr.obs_elasticities(r)
    with st.expander(tr("Elasticities for every observation", "المرونات لكل مشاهدة"), expanded=False):
        _table(oe.describe().T.reset_index().rename(columns={"index": "Input"}), 3)
        st.caption(tr(f"Monotonicity (all elasticities > 0) holds for {100 * (oe[xs] > 0).all(1).mean():.0f}% of observations.",
                      f"يتحقق شرط الرتابة (كل المرونات > 0) في {100 * (oe[xs] > 0).all(1).mean():.0f}% من المشاهدات."))
    txt, refs = _method("production", "apm")
    c1, c2 = st.columns(2)
    with c1:
        excel_download({"Coefficients": r.table(), "Elasticities": e["table"], "AES": e["AES"].reset_index(), "Obs elasticities": oe}, "production_function.xlsx", key="ap_xl")
    with c2:
        st.download_button(tr("⬇️ Word report", "⬇️ تقرير Word"), rp.to_docx([("h", tr("Methodology", "المنهجية")), ("p", txt),
                           ("table", r.table().round(4).astype(str)), ("table", e["table"].round(4).astype(str)), ("bullets", refs)],
                           tr("Production function", "دالة الإنتاج"), is_ar()), "production_function.docx", key="ap_docx")
    _code(ar_.code("production", y=y, x=xs), "ap_code")
    del lang


# ---------------------------------------------------------- yield response & optimal inputs
def _yield(df):
    num = _num(df)
    st.markdown("#### " + tr("Yield response and the economic optimum rate", "استجابة المحصول والمعدل الاقتصادي الأمثل"))
    c1, c2, c3, c4 = st.columns(4)
    x = c1.selectbox(tr("Input rate (e.g. kg N/ha)", "معدل المدخل (مثل كجم ن/هكتار)"), _prefer(num, ("n_kg", "rate", "fert", "dose")), key="ay_x")
    y = c2.selectbox(tr("Yield", "الإنتاجية"), _prefer([c for c in num if c != x], ("yield", "output")), key="ay_y")
    pdef = float(df["wheat_price_egp_t"].iloc[0]) if "wheat_price_egp_t" in df else 1000.0
    ndef = float(df["n_price_egp_kg"].iloc[0]) if "n_price_egp_kg" in df else 1.0
    po = c3.number_input(tr("Output price (per yield unit)", "سعر الناتج (لكل وحدة إنتاجية)"), 0.0, 1e9, pdef, key="ay_po")
    pi = c4.number_input(tr("Input price (per input unit)", "سعر المدخل (لكل وحدة)"), 0.0, 1e9, ndef, key="ay_pi")
    yld = None
    if not (df[x].nunique() < 3):
        rows, fits = [], {}
        for k in pr.YR_FORMS:
            try:
                f = pr.yield_response(df[x].to_numpy(float), df[y].to_numpy(float), k, po, pi)
                fits[k] = f
                rows.append({"Model": _t(pr.YR_FORMS[k]), "R²": f["R²"], "AIC": f["AIC"], tr("Agronomic max", "الحد الفني الأقصى"): f["agronomic max"],
                             tr("Economic optimum (EOR)", "المعدل الاقتصادي الأمثل"): f["EOR"], "SE (EOR)": f["EOR se"]})
            except Exception:  # noqa: BLE001
                continue
        tab = pd.DataFrame(rows)
        _table(tab, 3)
        best = min(fits, key=lambda k: fits[k]["AIC"])
        sel = st.selectbox(tr("Model to display", "النموذج المعروض"), list(fits), list(fits).index(best), key="ay_sel", format_func=lambda k: _t(pr.YR_FORMS[k]))
        yld = fits[sel]
        _table(yld["table"], 5)
        xs = np.linspace(0, df[x].max() * 1.1, 200)
        fig = go.Figure([go.Scatter(x=df[x], y=df[y], mode="markers", marker=dict(color="#999", size=6), name=tr("Observed", "المشاهد"))] +
                        [go.Scatter(x=xs, y=fits[k]["f"](xs), name=_t(pr.YR_FORMS[k]), line=dict(color=PAL[i % len(PAL)], width=2.8 if k == sel else 1.2,
                                                                                                 dash=None if k == sel else "dot")) for i, k in enumerate(fits)])
        if np.isfinite(yld["EOR"]):
            fig.add_vline(x=yld["EOR"], line=dict(color=PAL[0], dash="dash"), annotation_text=tr("EOR", "الأمثل اقتصادياً"))
        if np.isfinite(yld["agronomic max"]) and yld["agronomic max"] < xs.max():
            fig.add_vline(x=yld["agronomic max"], line=dict(color="#999", dash="dot"), annotation_text=tr("max", "الأقصى"))
        ch.style_academic(fig, height=420, x_title=x, y_title=y, legend="bottom", rtl=is_ar())
        _chart(fig, "ay_fig", "yield_response")
        if np.isfinite(yld["EOR"]):
            gain = yld.get("profit gain vs max")
            insight(tr(f"With the price ratio {pi:g}/{po:g} = {pi / po:.4f}, the economic optimum is {yld['EOR']:.1f} (± {1.96 * yld['EOR se']:.1f}) units of "
                       f"{x}, giving a yield of {yld.get('yield at EOR', np.nan):.3f}" +
                       (f"; compared with the agronomic maximum ({yld['agronomic max']:.1f}) it saves {yld['agronomic max'] - yld['EOR']:.1f} input units and adds "
                        f"{gain:,.0f} to profit per unit area." if gain is not None and np.isfinite(gain) else "."),
                       f"عند نسبة الأسعار {pi:g}/{po:g} = {pi / po:.4f} يكون المعدل الاقتصادي الأمثل {yld['EOR']:.1f} (± {1.96 * yld['EOR se']:.1f}) وحدة من "
                       f"{x}، بإنتاجية {yld.get('yield at EOR', np.nan):.3f}" +
                       (f"؛ ومقارنة بالحد الفني الأقصى ({yld['agronomic max']:.1f}) يوفر {yld['agronomic max'] - yld['EOR']:.1f} وحدة من المدخل ويضيف "
                        f"{gain:,.0f} إلى الربح لكل وحدة مساحة." if gain is not None and np.isfinite(gain) else ".")))
        with st.expander(tr("Policy simulation: optimum rate as prices change", "محاكاة السياسات: المعدل الأمثل مع تغير الأسعار"), expanded=False):
            ch_ = st.slider(tr("Input price change (%)", "تغير سعر المدخل (%)"), -50, 100, 0, 5, key="ay_sim")
            rows = []
            for dp in sorted({-50, -25, 0, 25, 50, 100, ch_}):
                f2 = pr.yield_response(df[x].to_numpy(float), df[y].to_numpy(float), sel, po, pi * (1 + dp / 100))
                rows.append({tr("Input price change %", "تغير سعر المدخل %"): dp, "EOR": f2["EOR"], tr("Yield at EOR", "الإنتاجية عند الأمثل"): f2.get("yield at EOR", np.nan)})
            _table(pd.DataFrame(rows), 3)
    else:
        st.info(tr("The input needs several distinct rates.", "يحتاج المدخل إلى عدة معدلات مختلفة."))
    st.markdown("#### " + tr("Optimal resource use (VMP = price)", "الاستخدام الأمثل للموارد (قيمة الناتج الحدي = السعر)"))
    pos = _pos(df, num)
    c1, c2 = st.columns([1, 2])
    yo = c1.selectbox(tr("Output", "الناتج"), _prefer(pos, ("output", "yield", "production")), key="ao_y")
    xo = c2.multiselect(tr("Inputs", "المدخلات"), [c for c in pos if c != yo], [c for c in pos if c != yo][:3], key="ao_x", placeholder=tr("Choose…", "اختر…"))
    if xo:
        r = pr.production_function(df, yo, xo, "cd")
        cc = st.columns(len(xo) + 1)
        pout = cc[0].number_input(tr("Output price", "سعر الناتج"), 0.0, 1e9, float(df["price_output"].mean()) if "price_output" in df else 1.0, key="ao_po")
        defaults = {"labour_days": "wage_day", "fertilizer_kg": "fert_price_kg", "land_feddan": "land_rent"}
        prices = {}
        for i, v in enumerate(xo):
            dv = float(df[defaults[v]].mean()) if v in defaults and defaults[v] in df else 1.0
            prices[v] = cc[i + 1].number_input(tr(f"Price of {v}", f"سعر {v}"), 0.0, 1e9, dv, key=f"ao_p_{v}")
        oi = pr.optimal_inputs(r, pout, prices)
        _table(oi, 3)
        insight(tr("VMP/price > 1: the input is under-used and more of it raises profit; < 1: over-used. Values are at the sample geometric mean "
                   "of a Cobb-Douglas technology.", "قيمة الناتج الحدي/السعر > 1: المدخل مستخدم بأقل من الأمثل وزيادته ترفع الربح؛ < 1: مستخدم بإفراط. "
                   "القيم عند المتوسط الهندسي للعينة لتقنية كوب-دوجلاس."))
    _method("yield", "aym")
    _code(ar_.code("yield", x=x, y=y), "ay_code")


# ---------------------------------------------------------- costs
def _cost(df):
    num = _num(df)
    pos = _pos(df, num)
    mode = st.radio(tr("Analysis", "التحليل"), ["cubic", "longrun", "scope", "system", "profit"], horizontal=True, key="ac_mode",
                    format_func=lambda m: {"cubic": tr("Cubic cost curves", "منحنيات التكاليف التكعيبية"), "longrun": tr("Long-run cost & scale", "التكاليف طويلة الأجل والسعة"),
                                           "scope": tr("Economies of scope", "وفورات النطاق"), "system": tr("Translog / CD cost system", "نظام تكاليف ترانسلوج / كوب-دوجلاس"),
                                           "profit": tr("Translog profit function", "دالة الربح ترانسلوج")}[m])
    if mode == "cubic":
        c1, c2 = st.columns(2)
        q = c1.selectbox(tr("Output Q", "الناتج Q"), _prefer(pos, ("output", "q")), key="ac_q")
        c = c2.selectbox(tr("Total cost", "التكاليف الكلية"), _prefer([v for v in pos if v != q], ("cost",)), key="ac_c")
        r = pr.cubic_cost(df[q], df[c])
        _table(r["table"], 5)
        qs = np.linspace(df[q].min() * 0.5, df[q].max() * 1.2, 200)
        fig = go.Figure([go.Scatter(x=qs, y=r["ATC"](qs), name=tr("ATC", "متوسط التكاليف الكلية"), line=dict(color=PAL[0], width=2.5)),
                         go.Scatter(x=qs, y=r["AVC"](qs), name=tr("AVC", "متوسط التكاليف المتغيرة"), line=dict(color=PAL[1])),
                         go.Scatter(x=qs, y=r["MC"](qs), name=tr("MC", "التكاليف الحدية"), line=dict(color=PAL[2], dash="dash"))])
        ch.style_academic(fig, height=420, x_title=q, y_title=tr("Cost per unit", "التكلفة للوحدة"), legend="bottom", rtl=is_ar())
        _chart(fig, "ac_cub", "cost_curves")
        msg = tr(f"Efficient scale (minimum ATC) at Q = {r['q efficient']:,.4g}, break-even price {r['break-even price']:,.4g}. ",
                 f"الحجم الكفء (أدنى متوسط تكاليف كلية) عند Q = {r['q efficient']:,.4g}، وسعر التعادل {r['break-even price']:,.4g}. ")
        if np.isfinite(r["q shutdown"]):
            msg += tr(f"Shutdown point (minimum AVC) at Q = {r['q shutdown']:,.4g}, price {r['shutdown price']:,.4g}. ",
                      f"نقطة الإغلاق (أدنى متوسط تكاليف متغيرة) عند Q = {r['q shutdown']:,.4g}، والسعر {r['shutdown price']:,.4g}. ")
        else:
            msg += tr("AVC has no interior minimum (cubic term ≤ 0). ", "لا يوجد حد أدنى داخلي لمتوسط التكاليف المتغيرة (الحد التكعيبي ≤ 0). ")
        msg += tr(f"At the mean output, scale economies 1 − MC/AC = {r['scale economies at mean']:.3f}.", f"عند متوسط الناتج، وفورات الحجم 1 − MC/AC = {r['scale economies at mean']:.3f}.")
        insight(msg)
    elif mode == "longrun":
        c1, c2, c3 = st.columns([1, 1, 2])
        q = c1.selectbox(tr("Output Q", "الناتج Q"), _prefer(pos, ("output", "q")), key="acl_q")
        c = c2.selectbox(tr("Total cost", "التكاليف الكلية"), _prefer([v for v in pos if v != q], ("cost",)), key="acl_c")
        ps = c3.multiselect(tr("Input prices (optional)", "أسعار المدخلات (اختياري)"), [v for v in pos if v not in (q, c)], key="acl_p", placeholder=tr("None", "لا شيء"))
        r = pr.long_run_cost(df, c, q, ps)
        _table(r["table"], 4)
        fig = go.Figure(go.Scatter(x=np.exp(r["lq"]), y=r["elasticity"], mode="markers", marker=dict(color=PAL[0], size=5)))
        fig.add_hline(y=1, line=dict(color="#999", dash="dash"))
        fig.update_xaxes(type="log")
        ch.style_academic(fig, height=360, x_title=q, y_title=tr("Cost elasticity ε = ∂lnC/∂lnQ", "مرونة التكاليف"), legend="none", rtl=is_ar())
        _chart(fig, "acl_fig", "cost_elasticity")
        insight(tr(f"Cost elasticity at the mean = {r['elasticity at mean']:.3f} (returns to scale {r['RTS at mean']:.2f}); {100 * r['share with scale economies']:.0f}% of "
                   f"units operate with scale economies (ε < 1). Minimum efficient scale ≈ {r['minimum efficient scale']:,.4g}.",
                   f"مرونة التكاليف عند المتوسط = {r['elasticity at mean']:.3f} (العائد للسعة {r['RTS at mean']:.2f})؛ و{100 * r['share with scale economies']:.0f}% من الوحدات "
                   f"تعمل بوفورات حجم (ε < 1). والحجم الأدنى الكفء ≈ {r['minimum efficient scale']:,.4g}."))
    elif mode == "scope":
        c1, c2, c3 = st.columns(3)
        c = c1.selectbox(tr("Total cost", "التكاليف الكلية"), _prefer(pos, ("cost",)), key="acs_c")
        nn = [v for v in num if v != c]
        q1 = c2.selectbox(tr("Product 1", "المنتج 1"), nn, key="acs_q1")
        q2 = c3.selectbox(tr("Product 2 (zeros allowed)", "المنتج 2 (يُسمح بالأصفار)"), [v for v in nn if v != q1], key="acs_q2")
        r = pr.scope_economies(df, c, q1, q2)
        _table(r["table"], 5)
        insight(tr(f"Economies of scope SC = {r['SC']:.3f}: producing both products together is {abs(r['SC']) * 100:.1f}% "
                   + ("cheaper" if r["SC"] > 0 else "more expensive") + " than separate production. Cost complementarity (b12 < 0): "
                   + ("yes" if r["complementarity"] else "no") + f" (p = {fp(r['p (b12)'])}).",
                   f"وفورات النطاق SC = {r['SC']:.3f}: إنتاج المنتجين معاً " + ("أرخص" if r["SC"] > 0 else "أغلى") + f" بنحو {abs(r['SC']) * 100:.1f}% من إنتاجهما منفصلين. "
                   "التكامل في التكاليف (b12 < 0): " + ("نعم" if r["complementarity"] else "لا") + f" (p = {fp(r['p (b12)'])})."))
    else:
        c1, c2 = st.columns(2)
        val = c1.selectbox(tr("Total cost" if mode == "system" else "Profit", "التكاليف الكلية" if mode == "system" else "الربح"),
                           _prefer(pos, ("cost",) if mode == "system" else ("profit",)), key=f"acy_v_{mode}")
        q = c2.selectbox(tr("Output (cost) / fixed factor (profit)", "الناتج (التكاليف) / عامل ثابت (الربح)"), ["—"] + [v for v in pos if v != val], key=f"acy_q_{mode}")
        c1, c2 = st.columns(2)
        ps = c1.multiselect(tr("Prices (last = numeraire)", "الأسعار (الأخير = المقياس)"), [v for v in pos if v not in (val, q)], key=f"acy_p_{mode}", placeholder=tr("Choose…", "اختر…"))
        ss = c2.multiselect(tr("Matching shares (same order)", "الحصص المقابلة (بالترتيب نفسه)"), [v for v in num if v not in (val, q) + tuple(ps)], key=f"acy_s_{mode}",
                            placeholder=tr("Choose…", "اختر…"))
        if len(ps) < 2 or len(ss) != len(ps):
            st.info(tr("Choose at least two prices and one share per price.", "اختر سعرين على الأقل وحصة لكل سعر."))
            return
        r = pr.translog_system(df, "cost" if mode == "system" else "profit", val, ps, ss, None if q == "—" else (q if mode == "system" else None),
                               None if mode == "system" or q == "—" else [q])
        _table(r["table"], 4)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**" + tr("Allen elasticities of substitution", "مرونات الإحلال لألن") + "**")
            st.dataframe(r["AES"].round(3), width="stretch")
            st.markdown("**" + tr("Morishima elasticities", "مرونات موريشيما") + "**")
            st.dataframe(r["MES"].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}"), width="stretch")
        with c2:
            st.markdown("**" + tr("Price elasticities (row quantity, column price)", "المرونات السعرية (الصف: الكمية، العمود: السعر)") + "**")
            st.dataframe(r["elasticities"].round(3), width="stretch")
        msg = tr(f"Curvature ({'concavity' if mode == 'system' else 'convexity'} in prices) holds at {r['curvature %']:.0f}% of observations",
                 f"يتحقق شرط التقعر/التحدب في الأسعار في {r['curvature %']:.0f}% من المشاهدات")
        if mode == "system" and np.isfinite(r.get("monotonicity %", np.nan)):
            msg += tr(f"; positive fitted shares at {r['monotonicity %']:.0f}%", f"؛ والحصص المقدرة موجبة في {r['monotonicity %']:.0f}%")
        if "RTS" in r:
            msg += tr(f". Cost elasticity {r['cost elasticity']:.3f} → returns to scale {r['RTS']:.2f}.", f". مرونة التكاليف {r['cost elasticity']:.3f} ← العائد للسعة {r['RTS']:.2f}.")
        insight(msg + ".")
        if mode == "system" and q != "—":
            with st.expander(tr("Cobb-Douglas cost function (for comparison)", "دالة تكاليف كوب-دوجلاس (للمقارنة)"), expanded=False):
                cd = pr.cd_cost(df, val, q, ps)
                _table(cd["table"], 4)
                st.caption(f"RTS = {cd['RTS']:.3f} · R² = {cd['R²']:.4f}")
    _method("cost", "acm")
    _code(ar_.code("cost", cost=st.session_state.get("acy_v_system", "c"), q=st.session_state.get("acy_q_system", "q"),
                   prices=st.session_state.get("acy_p_system", ["w1", "w2", "w3"]) or ["w1", "w2", "w3"],
                   shares=st.session_state.get("acy_s_system", ["s1", "s2", "s3"]) or ["s1", "s2", "s3"]), "ac_code")


# ---------------------------------------------------------- efficiency
def _efficiency(df):
    num = _num(df)
    pos = _pos(df, num)
    mode = st.radio(tr("Method", "الطريقة"), ["sfa", "panel", "meta", "dea", "malmquist"], horizontal=True, key="ae_mode",
                    format_func=lambda m: {"sfa": tr("Stochastic frontier", "الحد العشوائي"), "panel": tr("Panel SFA (BC92)", "SFA للبانل (BC92)"),
                                           "meta": tr("Meta-frontier", "الحد الفوقي"), "dea": "DEA", "malmquist": tr("Malmquist", "مالمكويست")}[m])
    c1, c2 = st.columns([1, 2])
    y = c1.selectbox(tr("Output (or cost)", "الناتج (أو التكاليف)"), _prefer(pos, ("output", "yield", "production", "cost")), key="ae_y")
    xs = c2.multiselect(tr("Inputs (or output and input prices for a cost frontier)", "المدخلات (أو الناتج وأسعار المدخلات لحد التكاليف)"),
                        [c for c in pos if c != y], [c for c in pos if c != y][:3], key="ae_x", placeholder=tr("Choose…", "اختر…"))
    if not xs:
        return
    cats = [c for c in df.columns if df[c].dtype == object]
    tcols = tu._time_cols(df)
    if mode == "sfa":
        c1, c2, c3 = st.columns(3)
        dist = c1.selectbox(tr("Inefficiency distribution", "توزيع عدم الكفاءة"), list(pr.SFA_DIST), key="ae_d", format_func=lambda k: _t(pr.SFA_DIST[k]))
        cost = c2.checkbox(tr("Cost frontier", "حد التكاليف"), key="ae_c")
        z = c3.multiselect(tr("Inefficiency determinants (BC95)", "محددات عدم الكفاءة (BC95)"), [c for c in num if c not in [y] + xs], key="ae_z",
                           disabled=dist != "tnormal", placeholder=tr("None", "لا شيء"))
        r = pr.sfa(df, y, xs, dist, cost, z if dist == "tnormal" else None)
        _table(r["table"], 4)
        st.caption(f"log L = {r['loglik']:.3f} · σv = {r['sigma_v']:.4f} · σu = {r['sigma_u']:.4f} · λ = {r['lambda']:.3f} · γ = {r['gamma']:.3f} · N = {r['N']}")
        if not np.all(np.isfinite(r["table"]["Std. Err."])) or r["sigma_v"] < 1e-4 or (r["table"]["Std. Err."] == 0).any():
            st.warning(tr("The likelihood is degenerate (a variance at its boundary): check the choice of output and inputs.",
                          "دالة الإمكان متدهورة (تباين عند حده): راجع اختيار الناتج والمدخلات."))
        if r["wrong skew"]:
            st.warning(tr("OLS residuals have the 'wrong' skewness: the data show little evidence of inefficiency.",
                          "بواقي OLS ذات التواء «خاطئ»: البيانات لا تُظهر دليلاً يُذكر على عدم الكفاءة."))
        te = r["efficiency"]
        fig = go.Figure(go.Histogram(x=te, nbinsx=30, marker_color=PAL[0]))
        ch.style_academic(fig, height=320, x_title=tr("Technical efficiency", "الكفاءة الفنية") if not cost else tr("Cost efficiency", "كفاءة التكاليف"), legend="none", rtl=is_ar())
        _chart(fig, "ae_hist", "efficiency_distribution")
        msg = tr(f"Mean efficiency = {r['mean efficiency']:.3f}: on average units " + (f"could raise output by {100 * (1 / r['mean efficiency'] - 1):.1f}% with the same inputs"
                 if not cost else f"spend {100 * (r['mean efficiency'] - 1):.1f}% more than the minimum cost") + f". LR test of no inefficiency = {r['LR (no inefficiency)']:.2f} "
                 f"(mixed χ², p = {fp(r['p LR'])}).",
                 f"متوسط الكفاءة = {r['mean efficiency']:.3f}: " + (f"يمكن للوحدات في المتوسط رفع الناتج {100 * (1 / r['mean efficiency'] - 1):.1f}% بالمدخلات نفسها"
                 if not cost else f"تنفق الوحدات {100 * (r['mean efficiency'] - 1):.1f}% أكثر من أدنى تكلفة") + f". اختبار LR لعدم وجود عدم كفاءة = {r['LR (no inefficiency)']:.2f} "
                 f"(χ² مختلط، p = {fp(r['p LR'])}).")
        if z:
            dz = r["table"][r["table"].Parameter.str.startswith("δ (")]
            msg += " " + tr("Inefficiency determinants (negative δ = raises efficiency): ", "محددات عدم الكفاءة (δ السالبة ترفع الكفاءة): ") + \
                   "، ".join(f"{p_.replace('δ (', '').rstrip(')')} {c_:+.3f} (p {fp(pv)})" for p_, c_, pv in zip(dz.Parameter, dz["Coef."], dz["p-value"]))
        insight(msg)
        out = df.loc[r["index"]].assign(TE=te)
        excel_download({"Frontier": r["table"], "Efficiency": out}, "stochastic_frontier.xlsx", key="ae_xl")
    elif mode == "panel":
        if not cats or not tcols:
            st.info(tr("Needs a unit identifier and a time variable.", "يحتاج معرّف وحدة ومتغيراً زمنياً."))
            return
        c1, c2, c3 = st.columns(3)
        u = c1.selectbox(tr("Unit", "الوحدة"), cats, key="aep_u")
        t = c2.selectbox(tr("Time", "الزمن"), tcols, key="aep_t")
        mdl = c3.selectbox(tr("Model", "النموذج"), ["tvd", "ti"], key="aep_m",
                           format_func=lambda m: tr("Time-varying decay (BC92)", "متغير زمنياً (BC92)") if m == "tvd" else tr("Time-invariant (Pitt-Lee)", "ثابت زمنياً (بيت-لي)"))
        r = pr.sfa_panel(df, u, t, y, xs, mdl)
        _table(r["table"], 4)
        st.caption(f"log L = {r['loglik']:.3f} · σ² = {r['sigma2']:.4f} · γ = {r['gamma']:.3f} · μ = {r['mu']:.4f} · η = {r['eta']:.4f} · N = {r['N']} · {tr('units', 'وحدات')} = {r['units']}")
        mp = r["mean efficiency by period"]
        fig = go.Figure(go.Scatter(x=mp.index.astype(str), y=mp.values, mode="lines+markers", line=dict(color=PAL[0], width=2.5)))
        ch.style_academic(fig, height=320, y_title=tr("Mean technical efficiency", "متوسط الكفاءة الفنية"), legend="none", rtl=is_ar())
        _chart(fig, "aep_fig", "efficiency_over_time")
        insight(tr(f"η = {r['eta']:.4f}: efficiency " + ("improves" if r["eta"] > 0 else "declines") + " over time; γ = σu²/σ² = " + f"{r['gamma']:.3f} of the composed-error variance is due to inefficiency.",
                   f"η = {r['eta']:.4f}: الكفاءة " + ("تتحسن" if r["eta"] > 0 else "تتراجع") + f" عبر الزمن؛ وγ = σu²/σ² = {r['gamma']:.3f} من تباين الخطأ المركب يعود إلى عدم الكفاءة."))
        excel_download({"Frontier": r["table"], "Efficiency": r["efficiency"]}, "panel_frontier.xlsx", key="aep_xl")
    elif mode == "meta":
        if not cats:
            st.info(tr("Needs a group variable.", "يحتاج متغيراً للمجموعات."))
            return
        few = [c for c in cats if 2 <= df[c].nunique() <= 12] or cats
        g = st.selectbox(tr("Technology groups", "مجموعات التقنية"), few, key="aem_g")
        small = df.groupby(g).size()
        if (small < 10 * (len(xs) + 2)).any():
            st.warning(tr("Each group needs enough observations for its own frontier; small groups were dropped: ", "تحتاج كل مجموعة مشاهدات كافية لحدها الخاص؛ حُذفت المجموعات الصغيرة: ") +
                       ", ".join(map(str, small[small < 10 * (len(xs) + 2)].index)))
            df = df[df[g].isin(small[small >= 10 * (len(xs) + 2)].index)]
            if df[g].nunique() < 2:
                return
        r = pr.metafrontier(df, g, y, xs)
        _table(r["groups"], 4)
        st.markdown("**" + tr("Meta-frontier coefficients", "معاملات الحد الفوقي") + "**")
        st.dataframe(r["meta beta"].round(4).to_frame("β*"), width="stretch")
        _table(r["summary"], 3)
        worst = r["summary"].sort_values("TGR").iloc[0]
        insight(tr(f"Technology gap: {worst.Group} has the lowest technology-gap ratio ({worst.TGR:.3f}) — its frontier produces {100 * (1 - worst.TGR):.0f}% less than the "
                   "meta-frontier with the same inputs; meta-technical efficiency = group efficiency × TGR.",
                   f"الفجوة التقنية: لدى {worst.Group} أدنى نسبة فجوة تقنية ({worst.TGR:.3f}) — حدها ينتج أقل بنحو {100 * (1 - worst.TGR):.0f}% من الحد الفوقي بالمدخلات نفسها؛ "
                   "والكفاءة الفوقية = كفاءة المجموعة × نسبة الفجوة."))
        excel_download({"Groups": r["groups"], "Summary": r["summary"], "Observations": r["obs"]}, "metafrontier.xlsx", key="aem_xl")
    elif mode == "dea":
        c1, c2, c3 = st.columns(3)
        ori = c1.selectbox(tr("Orientation", "الاتجاه"), ["input", "output"], key="aed_o", format_func=lambda o: tr("Input", "المدخلات") if o == "input" else tr("Output", "المخرجات"))
        sub = df
        if tcols:
            tsel = c2.selectbox(tr("Period (one cross-section)", "الفترة (مقطع واحد)"), ["—"] + sorted(df[tcols[0]].dropna().unique().tolist()), key="aed_t")
            if tsel != "—":
                sub = df[df[tcols[0]] == tsel]
        idc = c3.selectbox(tr("DMU label", "تسمية الوحدة"), ["—"] + cats, key="aed_id")
        sub = sub.dropna(subset=[y] + xs)
        if len(sub) > 800:
            st.warning(tr("Large problem: using the first 800 DMUs.", "مسألة كبيرة: تُستخدم أول 800 وحدة."))
            sub = sub.head(800)
        r = pr.dea_table(sub, xs, [y], ori, sub[idc].to_numpy() if idc != "—" else None)
        _table(r.describe().T.reset_index().rename(columns={"index": "DMU"}).iloc[1:], 3)
        fig = go.Figure([go.Histogram(x=r["CRS (CCR)"], name="CRS", marker_color=PAL[0], opacity=0.7), go.Histogram(x=r["VRS (BCC)"], name="VRS", marker_color=PAL[1], opacity=0.7)])
        fig.update_layout(barmode="overlay")
        ch.style_academic(fig, height=320, x_title=tr("Efficiency", "الكفاءة"), legend="bottom", rtl=is_ar())
        _chart(fig, "aed_fig", "dea_scores")
        nat = r["Returns to scale"].value_counts()
        insight(tr(f"{int((r['CRS (CCR)'] > 0.9999).sum())} of {len(r)} units are on the CRS frontier; mean pure technical efficiency (VRS) {r['VRS (BCC)'].mean():.3f}, "
                   f"mean scale efficiency {r['Scale efficiency'].mean():.3f}. Returns to scale: " + ", ".join(f"{k} {v}" for k, v in nat.items()) + ".",
                   f"{int((r['CRS (CCR)'] > 0.9999).sum())} من {len(r)} وحدة على حد ثبات العائد؛ ومتوسط الكفاءة الفنية البحتة (VRS) {r['VRS (BCC)'].mean():.3f}، "
                   f"ومتوسط كفاءة الحجم {r['Scale efficiency'].mean():.3f}. العائد للسعة: " + "، ".join(f"{tu._AR_VALS.get(k, k)} {v}" for k, v in nat.items()) + "."))
        with st.expander(tr("Scores for every unit", "الدرجات لكل وحدة"), expanded=False):
            _table(r, 3)
        excel_download({"DEA": r}, "dea.xlsx", key="aed_xl")
    else:
        if not cats or not tcols:
            st.info(tr("Needs a unit identifier and a time variable.", "يحتاج معرّف وحدة ومتغيراً زمنياً."))
            return
        c1, c2 = st.columns(2)
        u = c1.selectbox(tr("Unit", "الوحدة"), cats, key="aeM_u")
        t = c2.selectbox(tr("Time", "الزمن"), tcols, key="aeM_t")
        sig = (active_name(), u, t, y, tuple(xs))
        if st.button(tr("▶️ Compute Malmquist indices", "▶️ حساب مؤشرات مالمكويست"), key="aeM_run"):
            with st.spinner(tr("Solving DEA programmes…", "جارٍ حل برامج DEA…")):
                _store("malm", pr.malmquist(df, u, t, xs, [y]), sig)
        M = _load("malm", sig)
        if M is None:
            return
        g = M.groupby("to")[["Malmquist", "Efficiency change", "Technical change"]].apply(lambda x: np.exp(np.log(x).mean())).reset_index().rename(columns={"to": "Period"})
        _table(g, 4)
        fig = go.Figure([go.Scatter(x=g.Period.astype(str), y=g[c], name=_disp(pd.DataFrame(columns=[c])).columns[0], mode="lines+markers", line=dict(color=PAL[i]))
                         for i, c in enumerate(["Malmquist", "Efficiency change", "Technical change"])])
        fig.add_hline(y=1, line=dict(color="#999", dash="dash"))
        ch.style_academic(fig, height=360, legend="bottom", rtl=is_ar())
        _chart(fig, "aeM_fig", "malmquist")
        cum = float(np.prod(g["Malmquist"]))
        insight(tr(f"Cumulative productivity change over the period: {100 * (cum - 1):+.1f}% (geometric means across units); technical change "
                   f"{100 * (np.prod(g['Technical change']) - 1):+.1f}%, efficiency change {100 * (np.prod(g['Efficiency change']) - 1):+.1f}%.",
                   f"التغير التراكمي في الإنتاجية خلال الفترة: {100 * (cum - 1):+.1f}% (متوسطات هندسية عبر الوحدات)؛ التغير التقني "
                   f"{100 * (np.prod(g['Technical change']) - 1):+.1f}%، وتغير الكفاءة {100 * (np.prod(g['Efficiency change']) - 1):+.1f}%."))
        excel_download({"Malmquist (means)": g, "By unit": M}, "malmquist.xlsx", key="aeM_xl")
    _method("efficiency", "aem")
    _code(ar_.code("efficiency", y=y, x=xs), "ae_code")


# ---------------------------------------------------------- TFP
def _tfp(df):
    num = _num(df)
    cats = [c for c in df.columns if df[c].dtype == object]
    tcols = tu._time_cols(df)
    st.markdown("#### " + tr("Production function by control-function methods", "دالة الإنتاج بطرق الدالة الضابطة"))
    if not cats or not tcols:
        st.info(tr("Needs a firm identifier and a time variable.", "يحتاج معرّف منشأة ومتغيراً زمنياً."))
    else:
        c = st.columns(6)
        u = c[0].selectbox(tr("Firm", "المنشأة"), cats, key="at_u")
        t = c[1].selectbox(tr("Time", "الزمن"), tcols, key="at_t")
        y = c[2].selectbox(tr("ln output (value added)", "ln الناتج (القيمة المضافة)"), _prefer(num, ("value_added", "output", "sales")), key="at_y")
        fr_ = c[3].multiselect(tr("Free inputs (ln)", "المدخلات الحرة (ln)"), [v for v in num if v != y], _prefer([v for v in num if v != y], ("labour", "labor"))[:1], key="at_f", placeholder=tr("Choose…", "اختر…"))
        sv = c[4].selectbox(tr("State (ln capital)", "متغير الحالة (ln رأس المال)"), _prefer([v for v in num if v not in [y] + fr_], ("capital",)), key="at_s")
        px = c[5].selectbox(tr("Proxy", "المتغير البديل"), _prefer([v for v in num if v not in [y, sv] + fr_], ("material", "invest")), key="at_p")
        c1, c2 = st.columns(2)
        meths = c1.multiselect(tr("Estimators", "المقدرات"), ["op", "lp", "acf"], ["lp", "acf"], key="at_m",
                               format_func=lambda m: {"op": "Olley-Pakes", "lp": "Levinsohn-Petrin", "acf": "Ackerberg-Caves-Frazer"}[m])
        boot = c2.select_slider(tr("Bootstrap replications", "تكرارات البوتستراب"), [0, 50, 100, 200], 50, key="at_b")
        sig = (active_name(), u, t, y, tuple(fr_), sv, px, tuple(meths), boot)
        if fr_ and st.button(tr("▶️ Estimate", "▶️ التقدير"), type="primary", key="at_run"):
            res = {}
            with st.spinner(tr("Estimating (bootstrap)…", "جارٍ التقدير (بوتستراب)…")):
                for m in meths:
                    if m == "op" and not (df[px] > 0).all():
                        st.warning(tr("Olley-Pakes needs positive investment as proxy (log taken).", "يحتاج أولي-باكس استثماراً موجباً متغيراً بديلاً (يؤخذ لوغاريتمه)."))
                        continue
                    res[m] = pr.prodest(df, u, t, y, fr_, sv, px, m, boot=int(boot))
            _store("tfp", res, sig)
        res = _load("tfp", sig)
        if res:
            rows = []
            for m, r in res.items():
                for _, rw in r["table"].iterrows():
                    rows.append({"Method": {"op": "Olley-Pakes", "lp": "Levinsohn-Petrin", "acf": "ACF"}[m], "Input": rw.Input, "Coef.": rw["Coef."],
                                 "Std. Err. (bootstrap)": rw["Std. Err. (bootstrap)"], "OLS": rw["OLS"]})
            _table(pd.DataFrame(rows), 4)
            insight(tr("OLS over-states the coefficients of inputs that respond to productivity (simultaneity bias); the control-function "
                       "estimates correct it. Productivity (ln TFP) = ln output − β'x.",
                       "تبالغ OLS في معاملات المدخلات التي تستجيب للإنتاجية (تحيز التزامن)؛ وتصحح طرق الدالة الضابطة ذلك. والإنتاجية (ln TFP) = ln الناتج − β'x."))
            m0 = list(res)[-1]
            om = res[m0]["omega"]
            g = om.groupby(t)["ln TFP"].mean()
            fig = go.Figure(go.Scatter(x=g.index.astype(str), y=g.values, mode="lines+markers", line=dict(color=PAL[0], width=2.5)))
            ch.style_academic(fig, height=320, y_title=tr("Mean ln TFP", "متوسط ln الإنتاجية الكلية"), legend="none", rtl=is_ar())
            _chart(fig, "at_fig", "tfp_path")
            excel_download({m: r["table"] for m, r in res.items()} | {"ln TFP": om}, "tfp_estimates.xlsx", key="at_xl")
    st.markdown("#### " + tr("TFP index numbers (Törnqvist, Fisher)", "الأرقام القياسية للإنتاجية الكلية (تورنكفيست، فيشر)"))
    if tcols:
        c = st.columns(5)
        t = c[0].selectbox(tr("Time", "الزمن"), tcols, key="ati_t")
        outs = c[1].multiselect(tr("Outputs", "المخرجات"), num, key="ati_o", placeholder=tr("Choose…", "اختر…"))
        ops = c[2].multiselect(tr("Output prices", "أسعار المخرجات"), num, key="ati_op", placeholder=tr("Choose…", "اختر…"))
        ins = c[3].multiselect(tr("Inputs", "المدخلات"), num, key="ati_i", placeholder=tr("Choose…", "اختر…"))
        ips = c[4].multiselect(tr("Input prices", "أسعار المدخلات"), num, key="ati_ip", placeholder=tr("Choose…", "اختر…"))
        if outs and len(outs) == len(ops) and ins and len(ins) == len(ips):
            ti = pr.tfp_index(df, t, outs, ops, ins, ips)
            _table(ti, 4)
            fig = go.Figure([go.Scatter(x=ti.Period.astype(str), y=ti[c_], name=c_, mode="lines+markers", line=dict(color=PAL[i]))
                             for i, c_ in enumerate(["Törnqvist TFP", "Fisher TFP"])])
            ch.style_academic(fig, height=320, legend="bottom", rtl=is_ar())
            _chart(fig, "ati_fig", "tfp_index")
            g_ = (ti["Törnqvist TFP"].iloc[-1]) ** (1 / max(len(ti) - 1, 1)) - 1
            insight(tr(f"Average annual TFP growth (Törnqvist): {100 * g_:.2f}%.", f"متوسط النمو السنوي للإنتاجية الكلية (تورنكفيست): {100 * g_:.2f}%."))
    _method("tfp", "atm")
    _code(ar_.code("tfp"), "at_code")


# ---------------------------------------------------------- supply response
def _supply(df):
    num = _num(df)
    tcols = tu._time_cols(df) or [c for c in df.columns if "year" in c.lower()]
    if not tcols:
        st.info(tr("Needs a time variable (annual data).", "يحتاج متغيراً زمنياً (بيانات سنوية)."))
        return
    c = st.columns(5)
    t = c[0].selectbox(tr("Year", "السنة"), tcols, key="as_t")
    q = c[1].selectbox(tr("Area / output", "المساحة / الإنتاج"), _prefer([v for v in num if v != t], ("area", "production", "output")), key="as_q")
    p = c[2].selectbox(tr("Own (real) price", "السعر الذاتي (الحقيقي)"), _prefer([v for v in num if v not in (t, q)], ("price",)), key="as_p")
    oth = c[3].multiselect(tr("Other (lagged) drivers", "محددات أخرى (مبطأة)"), [v for v in num if v not in (t, q, p)], key="as_o", placeholder=tr("None", "لا شيء"))
    mdl = c[4].selectbox(tr("Model", "النموذج"), ["pa", "paae"], key="as_m",
                         format_func=lambda m: tr("Partial adjustment / adaptive expectations", "التعديل الجزئي / التوقعات المتكيفة") if m == "pa" else tr("Combined (PA + AE)", "المركب (PA + AE)"))
    r = dm.nerlove(df, t, q, p, oth, mdl)
    _table(r["table"], 4)
    sr, lr = r["SR"], r["LR"]
    msg = tr(f"Short-run price elasticity {sr[0]:.3f} (SE {sr[1]:.3f}); long-run {lr[0]:.3f} (SE {lr[1]:.3f}). ",
             f"المرونة السعرية قصيرة الأجل {sr[0]:.3f} (الخطأ {sr[1]:.3f})؛ وطويلة الأجل {lr[0]:.3f} (الخطأ {lr[1]:.3f}). ")
    if mdl == "pa":
        a = r["adjustment coefficient"][0]
        msg += tr(f"Adjustment coefficient δ = {a:.3f}: producers close {100 * a:.0f}% of the gap to the desired level each year (mean lag {r['mean lag']:.2f} years, "
                  f"half-life {r['half-life']:.2f}).", f"معامل التعديل δ = {a:.3f}: يسد المنتجون {100 * a:.0f}% من الفجوة نحو المستوى المرغوب سنوياً (متوسط الإبطاء "
                  f"{r['mean lag']:.2f} سنة، ونصف العمر {r['half-life']:.2f}).")
    else:
        d_, g_ = r["δ, γ (unordered)"]
        msg += tr(f"Adjustment and expectation coefficients (unordered): {d_:.3f}, {g_:.3f}.", f"معاملا التعديل والتوقع (بلا ترتيب): {d_:.3f}، {g_:.3f}.")
    insight(msg)
    ft = r["fitted"]
    fig = go.Figure([go.Scatter(x=ft[t], y=ft.actual, name=tr("Actual (ln)", "الفعلي (ln)"), line=dict(color="#222")),
                     go.Scatter(x=ft[t], y=ft.fitted, name=tr("Fitted", "المقدر"), line=dict(color=PAL[0], dash="dash"))])
    ch.style_academic(fig, height=320, legend="bottom", rtl=is_ar())
    _chart(fig, "as_fit", "supply_fit")
    with st.expander(tr("Policy simulation: a permanent price change", "محاكاة السياسات: تغير دائم في السعر"), expanded=False):
        dp = st.slider(tr("Price change (%)", "تغير السعر (%)"), -40, 60, 10, 5, key="as_dp")
        if mdl == "pa":
            b, c_ = r["SR"][0], 1 - r["adjustment coefficient"][0]
            path = [b * dp]
            for _ in range(14):
                path.append(b * dp + c_ * path[-1])
            fig = go.Figure(go.Bar(x=np.arange(1, 16), y=path, marker_color=PAL[0]))
            fig.add_hline(y=lr[0] * dp, line=dict(color=PAL[1], dash="dash"), annotation_text=tr("long run", "الأجل الطويل"))
            ch.style_academic(fig, height=300, x_title=tr("Years after the change", "السنوات بعد التغير"), y_title=tr(f"% change in {q}", f"% التغير في {q}"), legend="none", rtl=is_ar())
            _chart(fig, "as_sim", "supply_simulation")
    st.markdown("#### " + tr("Cobweb dynamics", "ديناميكية بيت العنكبوت"))
    c = st.columns(4)
    es = c[0].number_input(tr("Supply elasticity", "مرونة العرض"), 0.01, 5.0, float(round(max(abs(sr[0]), 0.05), 3)), key="as_es")
    ed = c[1].number_input(tr("Demand elasticity", "مرونة الطلب"), -5.0, -0.01, -0.3, key="as_ed")
    shock = c[2].number_input(tr("Initial price shock (%)", "صدمة السعر الأولية (%)"), -80.0, 200.0, 20.0, key="as_sh")
    nper = c[3].number_input(tr("Periods", "الفترات"), 5, 60, 15, key="as_np")
    pm, qm = float(df[p].mean()), float(df[q].mean())
    ad, bd, as_, bs = dm.cobweb_from_elasticities(pm, qm, es, ed)
    cw = dm.cobweb(ad, bd, as_, bs, pm * (1 + shock / 100), int(nper))
    xs_ = []
    ys_ = []
    for i in range(len(cw["Q"])):
        xs_ += [cw["Q"][i], cw["Q"][i]]
        ys_ += [cw["P"][i], cw["P"][i + 1]]
    qq = np.linspace(qm * 0.5, qm * 1.5, 50)
    fig = go.Figure([go.Scatter(x=qq, y=(ad - qq) / bd, name=tr("Demand", "الطلب"), line=dict(color=PAL[0])),
                     go.Scatter(x=qq, y=(qq - as_) / bs, name=tr("Supply", "العرض"), line=dict(color=PAL[1])),
                     go.Scatter(x=xs_, y=ys_, name=tr("Cobweb path", "مسار بيت العنكبوت"), line=dict(color="#888", width=1), mode="lines+markers", marker=dict(size=3))])
    ch.style_academic(fig, height=400, x_title=q, y_title=p, legend="bottom", rtl=is_ar())
    _chart(fig, "as_cw", "cobweb")
    insight(tr(f"Slope ratio b_s/b_d = |e_s/e_d| = {cw['ratio']:.3f} → {cw['verdict']} oscillations around the equilibrium price {cw['P*']:.4g}.",
               f"نسبة الميلين b_s/b_d = |e_s/e_d| = {cw['ratio']:.3f} ← تذبذبات " + {"convergent": "متقاربة", "continuous": "مستمرة", "explosive": "متباعدة"}[cw["verdict"]] +
               f" حول سعر التوازن {cw['P*']:.4g}."))
    _method("supply", "asm")
    excel_download({"Nerlove": r["table"]}, "supply_response.xlsx", key="as_xl")
    _code(ar_.code("supply", q=q, p=p), "as_code")


# ---------------------------------------------------------- demand & Engel
def _demand(df):
    num = _num(df)
    pos = _pos(df, num)
    st.markdown("#### " + tr("Single-equation demand", "دالة الطلب المفردة"))
    c = st.columns(5)
    q = c[0].selectbox(tr("Quantity", "الكمية"), _prefer(pos, ("consum", "quantity", "demand", "q_")), key="ad_q")
    own = c[1].selectbox(tr("Own price", "السعر الذاتي"), _prefer([v for v in pos if v != q], ("price", "p_")), key="ad_p")
    cross = c[2].multiselect(tr("Other prices", "أسعار أخرى"), [v for v in pos if v not in (q, own)], key="ad_c", placeholder=tr("None", "لا شيء"))
    inc = c[3].selectbox(tr("Income", "الدخل"), ["—"] + [v for v in pos if v not in (q, own) + tuple(cross)], key="ad_i")
    form = c[4].selectbox(tr("Form", "الصورة"), ["loglog", "linear"], key="ad_f", format_func=lambda f: tr("Log-log", "لوغاريتمية مزدوجة") if f == "loglog" else tr("Linear", "خطية"))
    tcols = tu._time_cols(df)
    dyn = st.checkbox(tr("Dynamic (lagged quantity, time-series data)", "ديناميكي (الكمية المبطأة، بيانات زمنية)"), key="ad_dyn", disabled=not tcols)
    r = dm.demand_equation(df, q, own, cross, None if inc == "—" else inc, form, dyn, tcols[0] if dyn and tcols else None)
    _table(r["table"], 4)
    _table(r["elasticities"], 3)
    e0 = r["elasticities"].iloc[0]
    msg = tr(f"Own-price elasticity {e0['Short-run elasticity']:.3f}: demand is " + ("elastic" if abs(e0["Short-run elasticity"]) > 1 else "inelastic") + ". ",
             f"المرونة السعرية الذاتية {e0['Short-run elasticity']:.3f}: الطلب " + ("مرن" if abs(e0["Short-run elasticity"]) > 1 else "غير مرن") + ". ")
    if "homogeneity" in r:
        h = r["homogeneity"]
        msg += tr(f"Homogeneity (sum of elasticities = 0): sum = {h['sum']:.3f}, p = {fp(h['p'])}.", f"التجانس (مجموع المرونات = 0): المجموع = {h['sum']:.3f}، p = {fp(h['p'])}.")
    insight(msg)
    st.markdown("#### " + tr("Engel curves", "منحنيات إنجل"))
    c = st.columns(4)
    tot = c[0].selectbox(tr("Total expenditure / income", "الإنفاق الكلي / الدخل"), _prefer(pos, ("expend", "income", "total")), key="ae2_x")
    sh = c[1].multiselect(tr("Budget shares", "حصص الإنفاق"), [v for v in num if v != tot and df[v].between(0, 1).all()],
                          [v for v in num if v != tot and df[v].between(0, 1).all()][:6], key="ae2_s", placeholder=tr("Choose…", "اختر…"))
    ef = c[2].selectbox(tr("Form", "الصورة"), list(dm.ENGEL_FORMS), key="ae2_f", format_func=lambda k: _t(dm.ENGEL_FORMS[k]))
    dg = c[3].multiselect(tr("Demographics", "المتغيرات الديموغرافية"), [v for v in num if v not in [tot] + sh], key="ae2_d", placeholder=tr("None", "لا شيء"))
    if sh:
        en = dm.engel(df, tot, sh, ef, dg)
        _table(en, 3)
        fig = go.Figure(go.Bar(x=en.Good, y=en["Expenditure elasticity"], marker_color=[PAL[0] if v < 1 else PAL[1] for v in en["Expenditure elasticity"]],
                               error_y=dict(type="data", array=1.96 * en["Std. Err."])))
        fig.add_hline(y=1, line=dict(color="#999", dash="dash"))
        ch.style_academic(fig, height=340, y_title=tr("Expenditure elasticity", "المرونة الإنفاقية"), legend="none", rtl=is_ar())
        _chart(fig, "ae2_fig", "engel_elasticities")
        insight(tr("Necessities (elasticity < 1) take a falling share of the budget as households get richer; luxuries (> 1) a rising share.",
                   "تأخذ السلع الضرورية (مرونة < 1) حصة متناقصة من الميزانية مع ارتفاع دخل الأسرة، والكمالية (> 1) حصة متزايدة."))
    _method("demand", "adm")
    _code(ar_.code("demand", shares=sh if sh else ["w1", "w2"], prices=["p1", "p2"], total=tot), "ad_code")


# ---------------------------------------------------------- demand systems & welfare
def _systems(df):
    num = _num(df)
    sh_c = [v for v in num if df[v].between(0, 1).all()]
    pos = _pos(df, num)
    c = st.columns([2, 2, 1, 1])
    sh = c[0].multiselect(tr("Budget shares (sum to 1)", "حصص الإنفاق (مجموعها 1)"), sh_c, sh_c[:6], key="sy_s", placeholder=tr("Choose…", "اختر…"))
    guess = [v for v in pos if v.startswith("p_")][:len(sh)]
    ps = c[1].multiselect(tr("Prices (same order)", "الأسعار (بالترتيب نفسه)"), pos, guess, key="sy_p", placeholder=tr("Choose…", "اختر…"))
    tot = c[2].selectbox(tr("Total expenditure", "الإنفاق الكلي"), _prefer([v for v in pos if v not in ps], ("expend", "total")), key="sy_x")
    mdl = c[3].selectbox(tr("System", "النظام"), list(dm.SYSTEMS), 2, key="sy_m", format_func=lambda k: _t(dm.SYSTEMS[k]))
    if len(sh) < 2 or len(ps) != len(sh):
        st.info(tr("Choose at least two shares and one price per share.", "اختر حصتين على الأقل وسعراً لكل حصة."))
        return
    c = st.columns(3)
    dg = c[0].multiselect(tr("Demographics (intercept shifters)", "متغيرات ديموغرافية (تزيح الثابت)"), [v for v in num if v not in sh + ps + [tot]], key="sy_d",
                          disabled=mdl in ("les", "rotterdam"), placeholder=tr("None", "لا شيء"))
    lx = np.log(df[tot].dropna())
    a0 = c[1].number_input("α₀", value=float(round(lx.min() - 0.1, 2)), key="sy_a0", disabled=mdl in ("laaids", "les", "rotterdam"),
                           help=tr("Fixed constant of the translog price index (Banks et al.: just below min ln x).", "ثابت مؤشر الأسعار ترانسلوج (بانكس وآخرون: أقل قليلاً من أدنى ln x)."))
    cats = [v for v in df.columns if df[v].dtype == object]
    tcols = [v for v in df.columns if v in tu._time_cols(df) or v.lower() in ("round", "year", "quarter", "period")]
    if mdl == "rotterdam":
        if not tcols:
            st.info(tr("Rotterdam needs a time / survey-round variable to take differences.", "يحتاج روتردام متغيراً زمنياً أو جولة مسح لأخذ الفروق."))
            return
        few = [v for v in cats if 2 <= df[v].nunique() <= 50]
        uu = c[1].selectbox(tr("Group (differences within)", "المجموعة (الفروق داخلها)"), few + ["—"], key="sy_u")
        tt = c[2].selectbox(tr("Time / round", "الزمن / الجولة"), tcols, key="sy_t")
        r = dm.rotterdam(df, None if uu == "—" else uu, tt, sh, ps, tot)
        st.caption(f"N = {r['N']}")
        el = r
    elif mdl == "les":
        r = dm.les(df, sh, ps, tot)
        _table(r["table"], 4)
        st.caption(tr(f"Frisch parameter = {r['frisch']:.3f}; subsistence expenditure = {100 * r['subsistence share']:.1f}% of the mean budget.",
                      f"معامل فريش = {r['frisch']:.3f}؛ والإنفاق الكفافي = {100 * r['subsistence share']:.1f}% من متوسط الميزانية."))
        el = r
    else:
        r = dm.aids(df, sh, ps, tot, mdl, dg, None if mdl == "laaids" else float(a0))
        with st.expander(tr("Parameters", "المعلمات"), expanded=False):
            _table(r["table"], 5)
        st.caption(tr(f"Iterations: {r['iterations']} · log L = {r['loglik']:.2f} · N = {r['N']} · homogeneity and symmetry imposed.",
                      f"التكرارات: {r['iterations']} · log L = {r['loglik']:.2f} · N = {r['N']} · مع فرض التجانس والتماثل."))
        el = r["elasticities"]
    st.markdown("**" + tr("Expenditure elasticities", "المرونات الإنفاقية") + "**")
    st.dataframe(el["expenditure"].round(3).to_frame(tr("elasticity", "المرونة")).T, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**" + tr("Marshallian (uncompensated) price elasticities", "مرونات مارشال السعرية (غير المعوَّضة)") + "**")
        st.dataframe(el["marshallian"].round(3), width="stretch")
    with c2:
        st.markdown("**" + tr("Hicksian (compensated) price elasticities", "مرونات هيكس السعرية (المعوَّضة)") + "**")
        st.dataframe(el["hicksian"].round(3), width="stretch")
    ex = el["expenditure"]
    mar = np.diag(el["marshallian"].to_numpy())
    msg = tr("Luxuries: " + (", ".join(ex[ex > 1].index) or "none") + "; necessities: " + (", ".join(ex[(ex >= 0) & (ex <= 1)].index) or "none") +
             f". Own-price elasticities range from {mar.min():.2f} to {mar.max():.2f}.",
             "الكمالية: " + ("، ".join(ex[ex > 1].index) or "لا يوجد") + "؛ والضرورية: " + ("، ".join(ex[(ex >= 0) & (ex <= 1)].index) or "لا يوجد") +
             f". وتتراوح المرونات السعرية الذاتية بين {mar.min():.2f} و{mar.max():.2f}.")
    if "negativity" in el:
        msg += tr(" Slutsky matrix negative semidefinite: " + ("yes." if el["negativity"] else "no (check specification)."),
                  " مصفوفة سلوتسكي سالبة شبه محددة: " + ("نعم." if el["negativity"] else "لا (راجع التوصيف)."))
    insight(msg)
    if mdl in ("aids", "quaids"):
        st.markdown("#### " + tr("Welfare effect of price changes", "أثر تغير الأسعار على الرفاه"))
        cc = st.columns(len(sh))
        dl = np.array([cc[i].number_input(tr(f"{ps[i]} change %", f"تغير {ps[i]} %"), -80.0, 300.0, 20.0 if i == 0 else 0.0, key=f"sy_dp_{i}") for i in range(len(sh))]) / 100
        w = dm.welfare_quaids(r, np.log1p(dl))
        _table(w.describe().T[["mean", "25%", "50%", "75%"]].reset_index().rename(columns={"index": tr("Measure", "المقياس")}), 3)
        q5 = pd.qcut(w["expenditure"], 5, labels=[f"Q{i}" for i in range(1, 6)])
        g = w.groupby(q5, observed=True)[["CV %", "EV %"]].mean().reset_index()
        fig = go.Figure([go.Bar(x=g.iloc[:, 0].astype(str), y=g["CV %"], name="CV %", marker_color=PAL[0]),
                         go.Bar(x=g.iloc[:, 0].astype(str), y=g["EV %"], name="EV %", marker_color=PAL[1])])
        ch.style_academic(fig, height=340, x_title=tr("Expenditure quintile (Q1 = poorest)", "خُمس الإنفاق (Q1 = الأفقر)"), y_title=tr("% of expenditure", "% من الإنفاق"),
                          legend="bottom", rtl=is_ar())
        _chart(fig, "sy_w", "welfare_by_quintile")
        insight(tr(f"Average compensating variation {w['CV %'].mean():.2f}% of the food budget ({w['CV'].mean():,.0f} per household): the poorest quintile loses "
                   f"{g['CV %'].iloc[0]:.2f}% vs {g['CV %'].iloc[-1]:.2f}% for the richest — " + ("regressive" if g["CV %"].iloc[0] > g["CV %"].iloc[-1] else "progressive") + " impact.",
                   f"متوسط التغير التعويضي {w['CV %'].mean():.2f}% من ميزانية الغذاء ({w['CV'].mean():,.0f} لكل أسرة): يخسر الخُمس الأفقر "
                   f"{g['CV %'].iloc[0]:.2f}% مقابل {g['CV %'].iloc[-1]:.2f}% للأغنى — أثر " + ("تنازلي (يضر الفقراء أكثر)" if g["CV %"].iloc[0] > g["CV %"].iloc[-1] else "تصاعدي") + "."))
        excel_download({"Welfare": w, "By quintile": g}, "welfare.xlsx", key="sy_wxl")
    _method("system", "sym")
    exl = {"Expenditure": el["expenditure"].to_frame().reset_index(), "Marshallian": el["marshallian"].reset_index(), "Hicksian": el["hicksian"].reset_index()}
    if mdl not in ("rotterdam",):
        exl["Parameters"] = r["table"]
    excel_download(exl, "demand_system.xlsx", key="sy_xl")
    _code(ar_.code("system", shares=sh, prices=ps, total=tot), "sy_code")


# ---------------------------------------------------------- consumption
def _consumption(df):
    num = _num(df)
    tcols = tu._time_cols(df) or [c for c in df.columns if c.lower() in ("year", "quarter", "date")]
    if not tcols:
        st.info(tr("Needs a time variable.", "يحتاج متغيراً زمنياً."))
        return
    c = st.columns(3)
    t = c[0].selectbox(tr("Time", "الزمن"), tcols, key="aq_t")
    cv = c[1].selectbox(tr("Consumption", "الاستهلاك"), _prefer([v for v in num if v != t], ("consum",)), key="aq_c")
    yv = c[2].selectbox(tr("Income", "الدخل"), _prefer([v for v in num if v not in (t, cv)], ("income", "gdp")), key="aq_y")
    r = dm.consumption(df, t, cv, yv)
    k, p, h, cm = r["keynes"], r["pih"], r["hall"], r["cm"]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**" + tr("Keynesian consumption function", "دالة الاستهلاك الكينزية") + "**")
        _table(k["table"], 4)
    with c2:
        st.markdown("**" + tr("Permanent income (Koyck)", "الدخل الدائم (كويك)") + "**")
        _table(p["table"], 4)
    lr = p["MPC long run"]
    insight(tr(f"MPC = {k['MPC']:.3f} and APC = {k['APC']:.3f} (Keynes). Permanent-income form: short-run MPC {p['MPC short run']:.3f}, long-run {lr[0]:.3f} (SE {lr[1]:.3f}). "
               f"Hall's test (lags of ΔC and ΔY do not predict ΔC): F p = {fp(h['p'])} → " + ("random walk rejected" if h["p"] < 0.05 else "random walk not rejected") +
               f". Campbell-Mankiw λ = {cm['λ']:.3f} (SE {cm['se']:.3f}): share of income going to rule-of-thumb consumers" +
               (" (weak instruments: first-stage R² small)." if cm["first-stage R²"] < 0.1 else "."),
               f"الميل الحدي للاستهلاك = {k['MPC']:.3f} والمتوسط = {k['APC']:.3f} (كينز). صيغة الدخل الدائم: الميل الحدي قصير الأجل {p['MPC short run']:.3f} وطويل الأجل {lr[0]:.3f} "
               f"(الخطأ {lr[1]:.3f}). اختبار هول (إبطاءات ΔC وΔY لا تتنبأ بـ ΔC): p = {fp(h['p'])} ← " + ("يُرفض السير العشوائي" if h["p"] < 0.05 else "لا يُرفض السير العشوائي") +
               f". λ لكامبل-مانكيو = {cm['λ']:.3f} (الخطأ {cm['se']:.3f}): نصيب الدخل المتجه لمستهلكي قاعدة الإبهام" +
               (" (أدوات ضعيفة: R² المرحلة الأولى صغير)." if cm["first-stage R²"] < 0.1 else ".")))
    _method("consumption", "aqm")
    excel_download({"Keynes": k["table"], "PIH": p["table"], "Hall": h["table"]}, "consumption.xlsx", key="aq_xl")
