# ==========================================
# Aetheris Research - Chart Builder (UI)
# ==========================================

from __future__ import annotations

import pandas as pd
import streamlit as st

from analytics import charts as ch
from analytics import descriptive as ds
from core.memory import active_name
from core.utils import tr, is_ar

# Which inputs each chart needs: role -> "num" | "cat" | "any" | "time"; "?" suffix = optional
ROLES: dict[str, dict[str, str]] = {
    "bar": {"x": "cat", "y": "num", "color": "cat?"},
    "hbar": {"x": "cat", "y": "num", "color": "cat?"},
    "pie": {"x": "cat", "y": "num"},
    "line": {"x": "time", "y": "num", "color": "cat?"},
    "area": {"x": "time", "y": "num", "color": "cat?"},
    "dual": {"x": "time", "y": "num", "y2": "num"},
    "scatter": {"x": "num", "y": "num", "color": "cat?"},
    "bubble": {"x": "num", "y": "num", "size": "num", "color": "cat?"},
    "histogram": {"x": "num", "color": "cat?"},
    "polygon": {"x": "num", "color": "cat?"},
    "ecdf": {"x": "num", "color": "cat?"},
    "box": {"y": "num", "x": "cat?"},
    "violin": {"y": "num", "x": "cat?"},
    "pareto": {"x": "cat", "y": "num"},
    "lorenz": {"y": "num", "color": "cat?"},
    "heatmap": {"x": "cat", "color": "cat", "y": "num"},
    "map": {"x": "any", "y": "num"},
}
ROLE_LABELS = {
    "x": ("X axis / category", "المحور الأفقي / الفئة"),
    "y": ("Value (Y)", "القيمة (المحور الرأسي)"),
    "y2": ("Second value (right axis)", "القيمة الثانية (المحور الأيمن)"),
    "color": ("Group / series", "المجموعة / السلسلة"),
    "size": ("Bubble size", "حجم الفقاعة"),
}
SPECIAL_LABELS = {
    ("map", "x"): ("Country code column (ISO3)", "عمود رمز الدولة (ISO3)"),
    ("heatmap", "color"): ("Rows (category)", "الصفوف (فئة)"),
    ("pie", "x"): ("Slices (category)", "الشرائح (فئة)"),
    ("box", "x"): ("Compare groups (optional)", "المقارنة بين مجموعات (اختياري)"),
    ("violin", "x"): ("Compare groups (optional)", "المقارنة بين مجموعات (اختياري)"),
}


def _options(df: pd.DataFrame, kind: str) -> list[str]:
    num, cat, time = ds.numeric_columns(df), ds.categorical_columns(df), ds.time_columns(df)
    if kind == "num":
        return num
    if kind == "cat":
        return cat + [c for c in time if c not in cat]
    if kind == "time":
        return time + [c for c in df.columns if c not in time]
    return list(df.columns)


def _default_index(opts: list[str], df: pd.DataFrame, role: str, kind: str, used: set) -> int:
    if kind == "map":
        for i, c in enumerate(opts):
            if str(c).lower() in ("iso3", "iso", "code", "countrycode"):
                return i
    for i, c in enumerate(opts):
        if c not in used:
            return i
    return 0


def render_chart_builder(df: pd.DataFrame):
    ar = is_ar()
    lang = "ar" if ar else "en"

    # ---------------- optional row filter ----------------
    with st.expander(tr("🔍 Filter rows (optional)", "🔍 تصفية الصفوف (اختياري)")):
        cats = ds.categorical_columns(df)
        fcol = st.selectbox(tr("Filter by", "التصفية حسب"), [None] + cats, key="cb_fcol",
                            format_func=lambda c: tr("— none —", "— بدون —") if c is None else c)
        if fcol:
            vals = sorted(df[fcol].dropna().astype(str).unique().tolist())
            keep = st.multiselect(tr("Keep values", "القيم المطلوبة"), vals, default=vals[:1], key="cb_fvals")
            if keep:
                df = df[df[fcol].astype(str).isin(keep)]
        st.caption(tr(f"{len(df)} rows in chart", f"عدد الصفوف في الرسم: {len(df)}"))

    left, right = st.columns([1, 2.2], gap="large")

    # ---------------- settings ----------------
    with left:
        kind = st.selectbox(tr("Chart type", "نوع الرسم البياني"), list(ch.CHART_TYPES.keys()),
                            format_func=lambda k: ch.CHART_TYPES[k][1 if ar else 0], key="cb_kind")
        params, used = {}, set()
        for role, spec in ROLES[kind].items():
            optional = spec.endswith("?")
            opts = _options(df, spec.rstrip("?"))
            label = SPECIAL_LABELS.get((kind, role), ROLE_LABELS[role])[1 if ar else 0]
            if optional:
                choice = st.selectbox(label, [None] + opts, key=f"cb_{kind}_{role}",
                                      format_func=lambda c: tr("— none —", "— بدون —") if c is None else c)
            else:
                if not opts:
                    st.warning(tr(f"No suitable column for '{label}'.", f"لا يوجد عمود مناسب لـ «{label}»."))
                    return
                choice = st.selectbox(label, opts, index=_default_index(opts, df, role, kind, used),
                                      key=f"cb_{kind}_{role}")
            params[role] = choice
            if choice:
                used.add(choice)

        if kind in ("bar", "hbar", "pie", "line", "area", "dual", "pareto", "heatmap", "map"):
            params["agg"] = st.selectbox(
                tr("Aggregate values by", "تجميع القيم باستخدام"), ch.AGG_FUNCS, key="cb_agg",
                index=1 if kind in ("line", "area", "dual", "map") else 0,
                format_func=lambda a: {"sum": tr("Sum", "المجموع"), "mean": tr("Mean", "المتوسط"),
                                       "median": tr("Median", "الوسيط"), "count": tr("Count", "العدد"),
                                       "min": tr("Minimum", "أصغر قيمة"), "max": tr("Maximum", "أكبر قيمة")}[a])
        if kind in ("bar", "hbar") and params.get("color"):
            params["barmode"] = st.radio(tr("Bars", "الأعمدة"), ["group", "stack", "relative"], horizontal=True,
                                         key="cb_barmode",
                                         format_func=lambda m: {"group": tr("Grouped", "مجمّعة"),
                                                                "stack": tr("Stacked", "مكدّسة"),
                                                                "relative": tr("Relative", "نسبية")}[m])
        if kind == "pie":
            params["donut"] = st.checkbox(tr("Donut", "حلقية"), key="cb_donut")
        if kind in ("histogram", "polygon"):
            params["bins"] = st.slider(tr("Number of classes", "عدد الفئات"), 5, 60, 15, key="cb_bins")
        if kind == "scatter":
            params["trendline"] = st.checkbox(tr("Regression line + equation", "خط الانحدار ومعادلته"),
                                              True, key="cb_trend")
        if kind in ("bar", "hbar"):
            params["show_values"] = st.checkbox(tr("Show values on bars", "إظهار القيم على الأعمدة"), key="cb_vals")

        st.markdown("---")
        st.markdown("**" + tr("Academic formatting", "التنسيق الأكاديمي") + "**")
        fig_no = st.number_input(tr("Figure number", "رقم الشكل"), 1, 200, 1, key="cb_figno")
        default_title = {"x": params.get("x"), "y": params.get("y")}
        auto_title = " — ".join(str(v) for v in default_title.values() if v)
        title = st.text_input(tr("Title", "العنوان"), auto_title, key=f"cb_title_{kind}")
        numbered = st.checkbox(tr("Prefix 'Figure n:'", "إضافة «شكل (ن):» قبل العنوان"), True, key="cb_num")
        x_title = st.text_input(tr("X-axis title", "عنوان المحور الأفقي"),
                                str(params.get("x") or ""), key=f"cb_xt_{kind}")
        y_title = st.text_input(tr("Y-axis title", "عنوان المحور الرأسي"),
                                str(params.get("y") or ""), key=f"cb_yt_{kind}")
        src_default = tr(f"Source: prepared by the researcher using {active_name()}.",
                         f"المصدر: إعداد الباحث اعتماداً على بيانات {active_name()}.")
        source = st.text_input(tr("Source note", "ملاحظة المصدر"), src_default, key="cb_src")
        palette = st.selectbox(tr("Colour scheme", "نظام الألوان"), list(ch.PALETTES.keys()), key="cb_pal",
                               format_func=lambda p: ch.PALETTE_LABELS[p][1 if ar else 0])
        c1, c2 = st.columns(2)
        with c1:
            font_size = st.slider(tr("Font size", "حجم الخط"), 10, 22, 14, key="cb_font")
        with c2:
            height = st.slider(tr("Height (px)", "الارتفاع (بكسل)"), 350, 900, 520, step=10, key="cb_h")
        legend = st.selectbox(tr("Legend", "مفتاح الرسم"), ["bottom", "right", "top", "none"], key="cb_leg",
                              format_func=lambda v: {"bottom": tr("Bottom", "أسفل"), "right": tr("Right", "يمين"),
                                                     "top": tr("Top", "أعلى"), "none": tr("Hidden", "مخفي")}[v])
        grid = st.checkbox(tr("Gridlines", "خطوط الشبكة"), True, key="cb_grid")

    # ---------------- chart ----------------
    with right:
        try:
            fig, info = ch.build_chart(kind, df, palette=palette, lang=lang, **params)
        except Exception as e:  # noqa: BLE001 - surface plotting problems to the user
            st.error(tr(f"Could not draw this chart with the selected columns: {e}",
                        f"تعذّر رسم هذا الشكل بالأعمدة المختارة: {e}"))
            return
        prefix = (f"شكل ({fig_no}): " if ar else f"Figure {fig_no}: ") if numbered else ""
        ch.style_academic(fig, title=prefix + title if title else "",
                          x_title=x_title if kind not in ("pie", "map") else None,
                          y_title=y_title if kind not in ("pie", "map", "heatmap") else None,
                          source=source, font_size=font_size, height=height, legend=legend,
                          gridlines=grid, rtl=ar)

        c1, c2 = st.columns(2)
        with c1:
            fmt = st.radio(tr("Camera-button format", "صيغة زر الكاميرا"), ["png", "svg"], horizontal=True,
                           key="cb_fmt", format_func=lambda f: {"png": "PNG", "svg": tr("SVG (vector)", "SVG (متجه)")}[f])
        with c2:
            scale = st.select_slider(tr("PNG resolution", "دقة PNG"), [1, 2, 3, 4], 3, key="cb_scale",
                                     format_func=lambda s: f"×{s} (~{s * 100} dpi)")
        cfg = ch.plotly_export_config(fmt, scale, filename=f"figure_{fig_no}")
        st.plotly_chart(fig, config=cfg, key="cb_chart")
        st.caption(tr("📷 Use the camera icon above the chart to save it for Word / LaTeX "
                      "(×3 PNG ≈ 300 dpi, journal quality; SVG stays sharp at any size).",
                      "📷 استخدم أيقونة الكاميرا أعلى الرسم لحفظه وإدراجه في Word أو LaTeX "
                      "(PNG بدقة ×3 ≈ 300 نقطة/بوصة، وهي الجودة المطلوبة في المجلات؛ وصيغة SVG تبقى واضحة بأي حجم)."))
        st.download_button(tr("⬇️ Download interactive chart (HTML)", "⬇️ تحميل الرسم التفاعلي (HTML)"),
                           fig.to_html(include_plotlyjs="cdn", config=cfg).encode("utf-8"),
                           file_name=f"figure_{fig_no}.html", mime="text/html", key="cb_html")

        if kind == "scatter" and info:
            sig = info["p"] < 0.05
            st.markdown(tr(
                f"**Simple regression:** ŷ = {info['intercept']:.4g} + {info['slope']:.4g}·x, "
                f"R² = {info['r2']:.3f}, p = {info['p']:.4f}, n = {info['n']} — "
                f"{'significant' if sig else 'not significant'} at 5%.",
                f"**الانحدار البسيط:** ŷ = {info['intercept']:.4g} + {info['slope']:.4g}·x، "
                f"R² = {info['r2']:.3f}، p = {info['p']:.4f}، n = {info['n']} — "
                f"{'دال' if sig else 'غير دال'} إحصائياً عند 5%."))
            st.caption(tr("Descriptive only — full regression diagnostics will be in the Econometrics Lab.",
                          "هذا لغرض الوصف فقط، وسيتوفر الانحدار بتشخيصه الكامل في مختبر الاقتصاد القياسي."))
        if kind == "lorenz" and info.get("gini"):
            st.markdown(" · ".join(f"**{k}**: G = {v:.3f}" for k, v in info["gini"].items()))
