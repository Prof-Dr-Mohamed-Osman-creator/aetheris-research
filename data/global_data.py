# ==========================================
# Aetheris Research - Global Data (World Bank + FRED)
# ==========================================

from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from core.memory import set_memory
from core.utils import tr, is_ar, csv_download
from data.engines import worldbank as wb
from data.engines import fred


# ------------------------------------------
# Cached fetchers (1 day) so repeated requests are instant
# ------------------------------------------
@st.cache_data(ttl=86400, show_spinner=False)
def _wb_panel(indicators: tuple, countries: tuple, start: int, end: int, labels: tuple):
    return wb.fetch_panel(list(indicators), list(countries), start, end, dict(labels))


@st.cache_data(ttl=86400, show_spinner=False)
def _wb_search(query: str):
    return wb.search_indicators(query)


@st.cache_data(ttl=86400, show_spinner=False)
def _fred_many(ids: tuple, start: str, end: str, freq: str, how: str):
    return fred.fetch_many(list(ids), start, end, freq, how)


def render_global_data():
    source = st.radio(
        tr("Data source", "مصدر البيانات"),
        ["worldbank", "fred"],
        format_func=lambda s: {"worldbank": tr("🌍 World Bank (WDI) — 200+ economies",
                                               "🌍 البنك الدولي (مؤشرات التنمية) — أكثر من 200 دولة"),
                               "fred": tr("🇺🇸 FRED — US & global markets (St. Louis Fed)",
                                          "🇺🇸 FRED — الاقتصاد الأمريكي والأسواق العالمية")}[s],
        horizontal=True,
        key="gd_source",
    )
    st.markdown("---")
    if source == "worldbank":
        _render_worldbank()
    else:
        _render_fred()


# ------------------------------------------
# World Bank
# ------------------------------------------
def _ind_label(code: str) -> str:
    if code in wb.INDICATORS:
        en, ar, _ = wb.INDICATORS[code]
        return ar if is_ar() else en
    return code


def _render_worldbank():
    ar = is_ar()
    st.subheader(tr("World Development Indicators", "مؤشرات التنمية العالمية"))

    # --- Indicators ---
    themes = list(wb.THEMES.keys())
    theme_sel = st.multiselect(
        tr("Filter indicators by theme", "تصفية المؤشرات حسب المحور"),
        themes,
        default=themes,
        format_func=lambda k: wb.THEMES[k][1 if ar else 0],
        key="wb_themes",
    )
    catalog = [c for c, v in wb.INDICATORS.items() if v[2] in theme_sel]
    indicators = st.multiselect(
        tr("Indicators", "المؤشرات"),
        catalog,
        default=[c for c in ["NY.GDP.MKTP.KD.ZG", "FP.CPI.TOTL.ZG"] if c in catalog],
        format_func=lambda c: f"{_ind_label(c)}  ·  {c}",
        key="wb_indicators",
    )

    with st.expander(tr("🔎 Search the full World Bank catalog (1,400+ indicators)",
                        "🔎 البحث في كامل قاعدة البنك الدولي (أكثر من 1400 مؤشر)")):
        q = st.text_input(tr("Keyword in English (e.g. wheat, tariff, electricity)",
                             "كلمة بحث بالإنجليزية (مثل wheat أو tariff أو electricity)"), key="wb_q")
        if q:
            try:
                with st.spinner(tr("Searching…", "جارٍ البحث…")):
                    hits = _wb_search(q)
                st.dataframe(hits, width="stretch", hide_index=True)
            except wb.WorldBankError as e:
                st.error(str(e))
        extra = st.text_input(tr("Add indicator codes (comma-separated)",
                                 "أضف رموز مؤشرات (مفصولة بفاصلة)"), key="wb_extra",
                              placeholder="AG.PRD.CREL.MT, EG.ELC.ACCS.ZS")
    extra_codes = [c.strip() for c in (extra or "").split(",") if c.strip()]
    all_codes = list(dict.fromkeys(indicators + extra_codes))

    # --- Countries ---
    c1, c2 = st.columns([3, 2])
    with c1:
        countries = st.multiselect(
            tr("Countries / regions", "الدول / الأقاليم"),
            list(wb.COUNTRIES.keys()),
            default=["EGY", "SAU", "MAR", "ARB", "WLD"],
            format_func=lambda c: f"{wb.COUNTRIES[c][1 if ar else 0]} ({c})",
            key="wb_countries",
        )
        extra_c = st.text_input(tr("Other ISO3 codes (comma-separated)", "رموز ISO3 أخرى (مفصولة بفاصلة)"),
                                key="wb_extra_c", placeholder="VNM, THA, CHL")
    with c2:
        this_year = dt.date.today().year
        start, end = st.slider(tr("Years", "السنوات"), 1960, this_year, (2000, this_year - 1), key="wb_years")
    all_countries = list(dict.fromkeys(countries + [c.strip().upper() for c in (extra_c or "").split(",") if c.strip()]))

    use_labels = st.checkbox(tr("Use readable column names instead of codes",
                                "استخدام أسماء مقروءة للأعمدة بدل الرموز"), value=True, key="wb_labels")

    if st.button(tr("📥 Fetch data", "📥 جلب البيانات"), type="primary", key="wb_fetch",
                 disabled=not (all_codes and all_countries)):
        labels = {c: _ind_label(c) for c in all_codes} if use_labels else {}
        try:
            with st.spinner(tr("Contacting the World Bank…", "جارٍ الاتصال بالبنك الدولي…")):
                panel, empty = _wb_panel(tuple(all_codes), tuple(all_countries), start, end,
                                         tuple(sorted(labels.items())))
        except wb.WorldBankError as e:
            st.error(tr(f"Could not fetch data: {e}", f"تعذّر جلب البيانات: {e}"))
            return
        if empty:
            st.warning(tr("No data returned for: ", "لم تُرجع بيانات للمؤشرات: ") + ", ".join(empty))
        if panel.empty:
            st.error(tr("No observations for this selection.", "لا توجد مشاهدات لهذا الاختيار."))
            return
        st.session_state["wb_result"] = panel

    panel = st.session_state.get("wb_result")
    if panel is not None:
        _preview_and_store(panel, default_name=f"WorldBank_{start}_{end}",
                           source="World Bank WDI", key="wb")
        st.caption(tr("Source: World Bank, World Development Indicators (api.worldbank.org). "
                      "Cite the indicator codes in your methodology.",
                      "المصدر: البنك الدولي، مؤشرات التنمية العالمية (api.worldbank.org). "
                      "اذكر رموز المؤشرات في منهجية البحث."))


# ------------------------------------------
# FRED
# ------------------------------------------
def _render_fred():
    ar = is_ar()
    st.subheader(tr("FRED — Federal Reserve Economic Data", "FRED — البيانات الاقتصادية للاحتياطي الفيدرالي"))
    ids = st.multiselect(
        tr("Series", "السلاسل"),
        list(fred.SERIES.keys()),
        default=["CPIAUCSL", "UNRATE", "FEDFUNDS"],
        format_func=lambda s: f"{fred.SERIES[s][1 if ar else 0]}  ·  {s}",
        key="fred_ids",
    )
    extra = st.text_input(tr("Other FRED series IDs (comma-separated, from fred.stlouisfed.org)",
                             "رموز سلاسل FRED أخرى (مفصولة بفاصلة، من موقع fred.stlouisfed.org)"),
                          key="fred_extra", placeholder="EGYCPIALLMINMEI, DEXJPUS")
    all_ids = list(dict.fromkeys(ids + [s.strip().upper() for s in (extra or "").split(",") if s.strip()]))

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start = st.date_input(tr("From", "من"), dt.date(2000, 1, 1), key="fred_start")
    with c2:
        end = st.date_input(tr("To", "إلى"), dt.date.today(), key="fred_end")
    with c3:
        freq = st.selectbox(tr("Frequency", "التكرار"), list(fred.FREQ_RULES.keys()), index=1,
                            format_func=lambda f: {"Original": tr("Original", "الأصلي"),
                                                   "Monthly": tr("Monthly", "شهري"),
                                                   "Quarterly": tr("Quarterly", "ربع سنوي"),
                                                   "Annual": tr("Annual", "سنوي")}[f],
                            key="fred_freq")
    with c4:
        how = st.selectbox(tr("Aggregation", "طريقة التجميع"), ["mean", "last"],
                           format_func=lambda h: tr("Average", "المتوسط") if h == "mean"
                           else tr("End of period", "آخر الفترة"), key="fred_how")
    st.caption(tr("Mixing daily, monthly and quarterly series? Choose a common frequency so rows align.",
                  "عند خلط سلاسل يومية وشهرية وربع سنوية، اختر تكراراً موحداً حتى تتطابق الصفوف."))

    if st.button(tr("📥 Fetch data", "📥 جلب البيانات"), type="primary", key="fred_fetch", disabled=not all_ids):
        with st.spinner(tr("Contacting FRED…", "جارٍ الاتصال بـ FRED…")):
            data, failed = _fred_many(tuple(all_ids), start.isoformat(), end.isoformat(), freq, how)
        if failed:
            st.warning(tr("Not found / failed: ", "غير موجودة أو تعذّر جلبها: ") + ", ".join(failed))
        if data.empty:
            st.error(tr("No observations returned.", "لم تُرجع أي مشاهدات."))
            return
        st.session_state["fred_result"] = data

    data = st.session_state.get("fred_result")
    if data is not None:
        _preview_and_store(data, default_name=f"FRED_{freq}", source="FRED", key="fred")
        st.caption(tr("Source: FRED®, Federal Reserve Bank of St. Louis.",
                      "المصدر: FRED®، بنك الاحتياطي الفيدرالي في سانت لويس."))


# ------------------------------------------
# Shared preview + save to memory
# ------------------------------------------
def _preview_and_store(df: pd.DataFrame, default_name: str, source: str, key: str):
    st.success(tr(f"{df.shape[0]} rows × {df.shape[1]} columns ready.",
                  f"البيانات جاهزة: {df.shape[0]} صف × {df.shape[1]} عمود."))
    st.dataframe(df.head(50), width="stretch", hide_index=True)

    num = df.select_dtypes("number")
    coverage = (num.notna().mean() * 100).round(1)
    with st.expander(tr("Data coverage (% non-missing per variable)", "اكتمال البيانات (% غير المفقود لكل متغير)")):
        st.dataframe(coverage.rename(tr("coverage %", "نسبة الاكتمال %")).to_frame(), width="stretch")

    c1, c2 = st.columns([3, 1])
    with c1:
        name = st.text_input(tr("Dataset name", "اسم مجموعة البيانات"), default_name, key=f"{key}_name")
    with c2:
        st.write("")
        st.write("")
        if st.button(tr("💾 Save to Smart Memory", "💾 حفظ في الذاكرة الذكية"), key=f"{key}_save"):
            set_memory(df, source=source, name=name)
            st.success(tr("Saved — now active for all analysis wings.",
                          "تم الحفظ — أصبحت البيانات النشطة لجميع أجنحة التحليل."))
    csv_download(df, f"{name}.csv", key=f"{key}_dl")
