# ==========================================
# Aetheris Research - Descriptive Statistics Wing (UI)
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from analytics import descriptive as ds
from core.memory import get_memory, active_name
from core.utils import tr, is_ar, insight, csv_download

PALETTE = px.colors.qualitative.Safe


def render_descriptive():
    df = get_memory()
    lang = "ar" if is_ar() else "en"
    st.caption(tr(f"Active dataset: **{active_name()}** — {df.shape[0]} rows × {df.shape[1]} columns",
                  f"البيانات النشطة: **{active_name()}** — {df.shape[0]} صف × {df.shape[1]} عمود"))

    num_cols = ds.numeric_columns(df)
    cat_cols = ds.categorical_columns(df)
    time_cols = ds.time_columns(df)

    if not num_cols:
        st.warning(tr("No numeric variables found in the active dataset.",
                      "لا توجد متغيرات رقمية في البيانات النشطة."))
        return

    tabs = st.tabs([
        tr("📋 Summary", "📋 الملخص الإحصائي"),
        tr("📈 Distribution & normality", "📈 التوزيع والطبيعية"),
        tr("🔗 Correlation", "🔗 الارتباط"),
        tr("🧮 Categorical", "🧮 المتغيرات النوعية"),
        tr("👥 Group comparison", "👥 المقارنة بين المجموعات"),
        tr("⏱ Trends & growth", "⏱ الاتجاه العام ومعدلات النمو"),
    ])
    with tabs[0]:
        _summary_tab(df, num_cols, lang)
    with tabs[1]:
        _distribution_tab(df, num_cols)
    with tabs[2]:
        _correlation_tab(df, num_cols, lang)
    with tabs[3]:
        _categorical_tab(df, cat_cols)
    with tabs[4]:
        _group_tab(df, num_cols, cat_cols)
    with tabs[5]:
        _trend_tab(df, num_cols, cat_cols, time_cols)


# ------------------------------------------
def _summary_tab(df, num_cols, lang):
    # sensible default: exclude year/id-like columns
    default = [c for c in num_cols if c not in ds.time_columns(df)][:8] or num_cols[:8]
    cols = st.multiselect(tr("Variables", "المتغيرات"), num_cols, default=default, key="ds_sum_cols")
    if not cols:
        return
    c1, c2 = st.columns(2)
    with c1:
        decimals = st.slider(tr("Decimals", "عدد المنازل العشرية"), 0, 6, 3, key="ds_dec")
    with c2:
        layout = st.radio(tr("Layout", "شكل الجدول"), ["rows", "cols"], horizontal=True, key="ds_layout",
                          format_func=lambda x: tr("Variables as rows", "المتغيرات صفوف") if x == "rows"
                          else tr("Variables as columns (SPSS style)", "المتغيرات أعمدة (نمط SPSS)"))
    table = ds.summary_table(df, cols)
    shown = table if layout == "rows" else table.T
    st.dataframe(shown.astype(float).round(decimals), width="stretch")
    csv_download(table.round(decimals).reset_index(), "descriptive_statistics.csv", key="ds_sum_dl")

    st.markdown("#### " + tr("Automatic interpretation", "القراءة التحليلية التلقائية"))
    for c in cols:
        txt = ds.interpret_variable(c, table.loc[c].to_dict(), lang)
        if txt:
            insight(txt)

    miss = df[cols].isna().mean() * 100
    if miss.any():
        st.markdown("#### " + tr("Missing data", "البيانات المفقودة"))
        mdf = miss.sort_values().rename_axis("variable").reset_index(name="pct")
        fig = px.bar(mdf, x="pct", y="variable", orientation="h", labels={"pct": "%", "variable": ""},
                     color_discrete_sequence=[PALETTE[1]])
        fig.update_layout(showlegend=False, height=max(250, 28 * len(cols)), margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)


# ------------------------------------------
def _distribution_tab(df, num_cols):
    c1, c2 = st.columns([2, 1])
    with c1:
        var = st.selectbox(tr("Variable", "المتغير"), num_cols, key="ds_dist_var")
    with c2:
        log = st.checkbox(tr("Log transform (ln)", "تحويل لوغاريتمي (ln)"), key="ds_dist_log")
    x = pd.to_numeric(df[var], errors="coerce").dropna()
    if log:
        x = np.log(x[x > 0])
    if len(x) < 3:
        st.info(tr("Not enough observations.", "عدد المشاهدات غير كافٍ."))
        return

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(x=x.values, nbins=min(50, max(10, int(np.sqrt(len(x))))), histnorm="probability density",
                           marginal="box", color_discrete_sequence=[PALETTE[0]], opacity=0.75)
        grid = np.linspace(x.min(), x.max(), 200)
        fig.add_trace(go.Scatter(x=grid, y=stats.norm.pdf(grid, x.mean(), x.std(ddof=1)), mode="lines",
                                 name=tr("Normal curve", "المنحنى الطبيعي"), line=dict(color="#c0392b")))
        fig.update_layout(title=tr("Histogram + normal curve", "المدرج التكراري والمنحنى الطبيعي"),
                          showlegend=False, height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig)
    with c2:
        (osm, osr), (slope, intercept, _) = stats.probplot(x, dist="norm")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=dict(color=PALETTE[0], size=6)))
        fig.add_trace(go.Scatter(x=osm, y=slope * np.asarray(osm) + intercept, mode="lines",
                                 line=dict(color="#c0392b")))
        fig.update_layout(title="Q-Q plot", showlegend=False, height=420,
                          xaxis_title=tr("Theoretical quantiles", "الكميات النظرية"),
                          yaxis_title=tr("Sample quantiles", "كميات العينة"),
                          margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig)

    alpha = st.select_slider(tr("Significance level α", "مستوى المعنوية α"), [0.01, 0.05, 0.10], 0.05, key="ds_alpha")
    tests = ds.normality_tests(x, alpha)
    tests["Normal at α"] = tests["Normal at α"].map({True: tr("Yes", "نعم"), False: tr("No", "لا")})
    st.dataframe(tests.round(4), width="stretch", hide_index=True)
    rejected = (ds.normality_tests(x, alpha)["p-value"] <= alpha).sum()
    if rejected == 0:
        insight(tr(f"No test rejects normality at α = {alpha}: parametric methods (t-test, ANOVA, OLS inference) are reasonable.",
                   f"لا يرفض أي اختبار فرض الطبيعية عند α = {alpha}، لذا يمكن استخدام الأساليب المعلمية (اختبار t وتحليل التباين وانحدار المربعات الصغرى)."))
    else:
        insight(tr(f"{rejected} test(s) reject normality at α = {alpha}. Consider a log/Box-Cox transform, "
                   "non-parametric tests (Mann-Whitney, Kruskal-Wallis, Spearman), or robust standard errors.",
                   f"{rejected} من الاختبارات ترفض فرض الطبيعية عند α = {alpha}. يُنصح بالتحويل اللوغاريتمي أو Box-Cox، "
                   "أو الاختبارات اللامعلمية (مان-ويتني، كروسكال-واليس، سبيرمان)، أو الأخطاء المعيارية المتينة."))

    st.markdown("#### " + tr("Normality overview — all variables", "ملخص الطبيعية لجميع المتغيرات"))
    ov = ds.normality_overview(df, num_cols, alpha)
    if not ov.empty:
        ov["Normal at α"] = ov["Normal at α"].map({True: "✅", False: "❌"})
        st.dataframe(ov.round(4), width="stretch", hide_index=True)


# ------------------------------------------
def _correlation_tab(df, num_cols, lang):
    if len(num_cols) < 2:
        st.info(tr("At least two numeric variables are needed.", "يلزم متغيران رقميان على الأقل."))
        return
    default = [c for c in num_cols if c not in ds.time_columns(df)][:8]
    c1, c2 = st.columns([3, 1])
    with c1:
        cols = st.multiselect(tr("Variables", "المتغيرات"), num_cols, default=default, key="ds_cor_cols")
    with c2:
        method = st.selectbox(tr("Method", "الطريقة"), ["pearson", "spearman", "kendall"], key="ds_cor_m",
                              format_func=lambda m: {"pearson": "Pearson", "spearman": "Spearman ρ",
                                                     "kendall": "Kendall τ"}[m])
    if len(cols) < 2:
        return
    r, p, n = ds.correlation_matrix(df, cols, method)

    fig = px.imshow(r.round(2), text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r", aspect="auto")
    fig.update_layout(height=max(380, 45 * len(cols)), margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)

    st.markdown("#### " + tr("Publication-ready matrix", "مصفوفة جاهزة للنشر"))
    pub = ds.starred_matrix(r, p)
    st.dataframe(pub, width="stretch")
    st.caption("*** p<0.01, ** p<0.05, * p<0.10 — " +
               tr("pairwise-complete observations", "باستخدام المشاهدات المكتملة لكل زوج"))
    csv_download(pub.reset_index().rename(columns={"index": "Variable"}), "correlation_matrix.csv", key="ds_cor_dl")

    with st.expander(tr("p-values and N", "قيم p وعدد المشاهدات")):
        st.dataframe(p.round(4), width="stretch")
        st.dataframe(n, width="stretch")

    st.markdown("#### " + tr("Strongest relationships", "أقوى العلاقات"))
    top = ds.top_correlations(r, p, 5)
    for _, row in top.iterrows():
        insight(ds.interpret_correlation(row["Var 1"], row["Var 2"], row["r"], row["p-value"], method, lang))

    if st.checkbox(tr("Show scatter-plot matrix", "عرض مصفوفة أشكال الانتشار"), key="ds_spm") and len(cols) <= 6:
        fig = px.scatter_matrix(df, dimensions=cols, color_discrete_sequence=[PALETTE[0]])
        fig.update_traces(diagonal_visible=False, marker=dict(size=4, opacity=0.6))
        fig.update_layout(height=700)
        st.plotly_chart(fig)


# ------------------------------------------
def _categorical_tab(df, cat_cols):
    if not cat_cols:
        st.info(tr("No categorical variables detected.", "لم يتم العثور على متغيرات نوعية."))
        return
    var = st.selectbox(tr("Variable", "المتغير"), cat_cols, key="ds_cat_var")
    ft = ds.frequency_table(df[var])
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(ft.round(2), width="stretch", hide_index=True)
        csv_download(ft.round(2), f"frequency_{var}.csv", key="ds_cat_dl")
    with c2:
        fig = px.bar(ft.head(25), x="Category", y="Percent (%)", text_auto=".1f",
                     color_discrete_sequence=[PALETTE[0]])
        fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig)

    if len(cat_cols) >= 2:
        st.markdown("#### " + tr("Cross-tabulation & χ² test of independence", "الجداول المتقاطعة واختبار كاي² للاستقلال"))
        c1, c2 = st.columns(2)
        with c1:
            a = st.selectbox(tr("Rows", "الصفوف"), cat_cols, key="ds_ct_a")
        with c2:
            b = st.selectbox(tr("Columns", "الأعمدة"), [c for c in cat_cols if c != a], key="ds_ct_b")
        ct, res = ds.crosstab_chi2(df, a, b)
        st.dataframe(ct, width="stretch")
        if res:
            st.write(f"χ² = {res['chi2']:.3f}, df = {res['dof']}, p = {res['p']:.4f}, Cramér's V = {res['cramers_v']:.3f}")
            sig = res["p"] < 0.05
            insight(tr(
                f"The association between <b>{a}</b> and <b>{b}</b> is "
                f"{'statistically significant' if sig else 'not significant'} at 5% (Cramér's V = {res['cramers_v']:.3f}).",
                f"العلاقة بين <b>{a}</b> و<b>{b}</b> {'دالة' if sig else 'غير دالة'} إحصائياً عند 5% "
                f"(معامل كرامر V = {res['cramers_v']:.3f})."))
            if res["pct_expected_lt5"] > 20:
                st.warning(tr(f"{res['pct_expected_lt5']:.0f}% of expected counts are < 5 — χ² may be unreliable; "
                              "merge categories or use Fisher's exact test.",
                              f"{res['pct_expected_lt5']:.0f}% من التكرارات المتوقعة أقل من 5، لذا قد لا يكون اختبار كاي² دقيقاً. "
                              "ادمج بعض الفئات أو استخدم اختبار فيشر الدقيق."))


# ------------------------------------------
def _group_tab(df, num_cols, cat_cols):
    if not cat_cols:
        st.info(tr("No grouping variable detected.", "لم يتم العثور على متغير للتجميع."))
        return
    c1, c2 = st.columns(2)
    with c1:
        g = st.selectbox(tr("Group by", "التجميع حسب"), cat_cols, key="ds_grp_g")
    with c2:
        vals = st.multiselect(tr("Variables", "المتغيرات"), [c for c in num_cols if c != g],
                              default=[c for c in num_cols if c != g][:2], key="ds_grp_v")
    if not vals:
        return
    table = ds.group_summary(df, g, vals)
    st.dataframe(table.round(3), width="stretch", hide_index=True)
    csv_download(table.round(3), "group_summary.csv", key="ds_grp_dl")
    plot_df = df.assign(**{g: df[g].astype(str)})  # numeric codes (e.g. 1-5) -> discrete groups
    for v in vals:
        fig = px.box(plot_df, x=g, y=v, points="outliers", color=g, color_discrete_sequence=PALETTE)
        fig.update_layout(showlegend=False, height=380, margin=dict(l=10, r=10, t=30, b=10), title=v)
        st.plotly_chart(fig)


# ------------------------------------------
def _trend_tab(df, num_cols, cat_cols, time_cols):
    if not time_cols:
        st.info(tr("No time column (year/date) detected.", "لم يتم العثور على عمود زمني (سنة/تاريخ)."))
        return
    c1, c2, c3 = st.columns(3)
    with c1:
        tcol = st.selectbox(tr("Time", "الزمن"), time_cols, key="ds_tr_t")
    with c2:
        vcol = st.selectbox(tr("Variable", "المتغير"), [c for c in num_cols if c != tcol], key="ds_tr_v")
    with c3:
        ent_opts = [None] + [c for c in cat_cols if c != tcol]
        ecol = st.selectbox(tr("Entity (country, governorate…)", "الوحدة (دولة، محافظة…)"), ent_opts, key="ds_tr_e",
                            format_func=lambda x: tr("— none —", "— بدون —") if x is None else x)
    if vcol is None:
        return
    plot_df = df[[c for c in [tcol, vcol, ecol] if c]].dropna().sort_values(tcol)
    fig = px.line(plot_df, x=tcol, y=vcol, color=ecol, markers=True, color_discrete_sequence=PALETTE)
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig)

    gr = ds.growth_analysis(df, tcol, vcol, ecol)
    if gr.empty:
        return
    st.markdown("#### " + tr("Growth rates", "معدلات النمو"))
    st.dataframe(gr.round(3), width="stretch", hide_index=True)
    csv_download(gr.round(4), f"growth_{vcol}.csv", key="ds_tr_dl")
    st.caption(tr("Trend growth = (e^b − 1)×100 from ln(Y) = a + b·t, the standard compound growth estimate "
                  "in agricultural and development economics; CV measures instability.",
                  "معدل النمو من الاتجاه العام = (e^b − 1)×100 من معادلة ln(Y) = a + b·t، وهو المقياس المعتاد "
                  "لمعدل النمو المركب في الاقتصاد الزراعي واقتصاديات التنمية. ويقيس معامل الاختلاف درجة عدم الاستقرار."))
    for _, row in gr.iterrows():
        if pd.isna(row["Trend growth (%/yr, log-lin)"]):
            continue
        sig = row["Trend p-value"] < 0.05
        insight(tr(
            f"<b>{row['Entity']}</b>: {vcol} grew at {row['Trend growth (%/yr, log-lin)']:.2f}% per year on trend "
            f"({'significant' if sig else 'not significant'} at 5%, R² = {row['Trend R²']:.2f}); instability CV = {row['Instability (CV %)']:.1f}%.",
            f"<b>{row['Entity']}</b>: نما {vcol} بمعدل اتجاه عام {row['Trend growth (%/yr, log-lin)']:.2f}% سنوياً "
            f"({'دال' if sig else 'غير دال'} إحصائياً عند 5%، R² = {row['Trend R²']:.2f})، ومعامل عدم الاستقرار {row['Instability (CV %)']:.1f}%."))
