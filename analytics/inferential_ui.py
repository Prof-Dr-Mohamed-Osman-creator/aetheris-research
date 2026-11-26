# ==========================================
# Aetheris Research - Inferential Statistics Wing (UI)
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics import charts as ch
from analytics import descriptive as ds
from analytics import inferential as inf
from analytics import power as pw
from core.memory import get_memory, has_data, active_name
from core.utils import tr, is_ar, insight, csv_download

ITEM_AR = {"Test": "الاختبار", "Statistic": "الإحصاءة", "df": "درجات الحرية", "p-value": "القيمة الاحتمالية p",
           "Alternative": "الفرض البديل", "Estimate": "التقدير", "N": "حجم العينة"}
ALT = {"two-sided": ("Two-sided (≠)", "ذو طرفين (≠)"), "greater": ("One-sided (>)", "طرف واحد (>)"),
       "less": ("One-sided (<)", "طرف واحد (<)")}


# ==========================================================
# Shared helpers
# ==========================================================
def _lang():
    return "ar" if is_ar() else "en"


def _settings(key: str, alternative: bool = True):
    c1, c2 = st.columns(2)
    with c1:
        alpha = st.select_slider(tr("Significance level α", "مستوى المعنوية α"), [0.01, 0.05, 0.10], 0.05,
                                 key=f"{key}_alpha")
    alt = "two-sided"
    if alternative:
        with c2:
            alt = st.selectbox(tr("Alternative hypothesis", "الفرض البديل"), list(ALT), key=f"{key}_alt",
                               format_func=lambda a: ALT[a][1 if is_ar() else 0])
    return alpha, alt


def show_result(res: inf.TestResult, alpha: float, key: str, context: str = ""):
    lang = _lang()
    frame = res.to_frame()
    if lang == "ar":
        frame["Item"] = frame["Item"].map(lambda i: ITEM_AR.get(i, i))
        frame.loc[0, "Value"] = res.test_ar
        frame.columns = ["البند", "القيمة"]
    st.dataframe(frame, width="stretch", hide_index=True)
    insight(res.interpret(alpha, lang, context))
    st.caption(tr("APA-style report (copy into your paper):", "صيغة التقرير بنمط APA (انسخها إلى بحثك):"))
    st.code(res.apa(), language=None)
    csv_download(frame, f"{key}_result.csv", key=f"{key}_dl")


def _group_data(df, key, min_levels=2, max_levels=None):
    """Pick a numeric outcome and a grouping factor; return (outcome, factor, levels, list of arrays)."""
    num, cat = ds.analysis_columns(df), ds.categorical_columns(df)
    if not num or not cat:
        st.info(tr("Needs a numeric outcome and a categorical grouping variable.",
                   "يتطلب متغيراً تابعاً رقمياً ومتغيراً نوعياً للتجميع."))
        return None
    c1, c2 = st.columns(2)
    with c1:
        y = st.selectbox(tr("Outcome (numeric)", "المتغير التابع (رقمي)"), num, key=f"{key}_y")
    with c2:
        g = st.selectbox(tr("Grouping factor", "متغير التجميع (العامل)"), [c for c in cat if c != y], key=f"{key}_g")
    if g is None:
        return None
    levels = sorted(df[g].dropna().astype(str).unique().tolist())
    default = levels[:max_levels] if max_levels else levels
    chosen = st.multiselect(tr("Groups to compare", "المجموعات المراد مقارنتها"), levels, default=default,
                            key=f"{key}_lv", max_selections=max_levels)
    if len(chosen) < min_levels:
        st.warning(tr(f"Select at least {min_levels} groups.", f"اختر {min_levels} مجموعات على الأقل."))
        return None
    gs = df[g].astype(str)
    arrays = [pd.to_numeric(df.loc[gs == lv, y], errors="coerce").dropna().to_numpy(float) for lv in chosen]
    if any(a.size < 2 for a in arrays):
        st.warning(tr("Each group needs at least 2 observations.", "كل مجموعة تحتاج مشاهدتين على الأقل."))
        return None
    return y, g, chosen, arrays


def _group_summary(names, arrays):
    rows = []
    for n, a in zip(names, arrays):
        m, lo, hi = inf.ci_mean(a)
        rows.append({tr("Group", "المجموعة"): n, "n": a.size, tr("Mean", "المتوسط"): m,
                     tr("SD", "الانحراف المعياري"): a.std(ddof=1), tr("Median", "الوسيط"): np.median(a),
                     tr("95% CI low", "حد الثقة الأدنى"): lo, tr("95% CI high", "حد الثقة الأعلى"): hi})
    st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)


def _assumptions(arrays, names, alpha):
    a = inf.check_assumptions(arrays, alpha)
    rows = [{tr("Group", "المجموعة"): n, "Shapiro-Wilk p": p,
             tr("Normal?", "طبيعي؟"): "—" if pd.isna(p) else ("✅" if p > alpha else "❌")}
            for n, p in zip(names, a["normality_p"])]
    with st.expander(tr("🩺 Assumption checks", "🩺 فحص الافتراضات"), expanded=False):
        st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
        st.write(f"Levene p = {a['levene_p']:.4f} · Brown-Forsythe p = {a['brown_forsythe_p']:.4f}")
        st.caption(tr("Normality matters less when every group has n ≥ 30 (central limit theorem). "
                      "Brown-Forsythe is the robust test of equal variances.",
                      "تقل أهمية شرط الطبيعية عندما يكون حجم كل مجموعة 30 أو أكثر (نظرية النهاية المركزية). "
                      "واختبار براون-فورسايث هو الاختبار المتين لتجانس التباين."))
    return a


def _means_plot(names, arrays, title, letters=None):
    """Mean ± 95% CI with raw points — the figure reviewers expect next to ANOVA/t-test results."""
    fig = go.Figure()
    pal = ch.PALETTES["academic"]
    for i, (n, a) in enumerate(zip(names, arrays)):
        jitter = np.random.default_rng(i).uniform(-0.12, 0.12, a.size)
        fig.add_trace(go.Scatter(x=np.full(a.size, i) + jitter, y=a, mode="markers", showlegend=False,
                                 marker=dict(color=pal[i % len(pal)], size=5, opacity=0.35)))
        m, lo, hi = inf.ci_mean(a)
        fig.add_trace(go.Scatter(x=[i], y=[m], mode="markers", showlegend=False,
                                 marker=dict(color="#111", size=11, symbol="diamond"),
                                 error_y=dict(type="data", symmetric=False, array=[hi - m], arrayminus=[m - lo],
                                              thickness=2, width=10, color="#111")))
        if letters:
            fig.add_annotation(x=i, y=max(a.max(), hi), text=f"<b>{letters.get(n, '')}</b>", showarrow=False,
                               yshift=16, font=dict(size=15))
    fig.update_xaxes(tickvals=list(range(len(names))), ticktext=names)
    ch.style_academic(fig, title=title, y_title=None, height=440, legend="none", rtl=is_ar())
    st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, "means_ci"))
    st.caption(tr("◆ mean with 95% confidence interval; dots = observations.",
                  "◆ المتوسط مع فترة الثقة 95%؛ والنقاط هي المشاهدات الفعلية."))


# ==========================================================
# Main entry
# ==========================================================
def render_inferential():
    data_ok = has_data()
    df = get_memory() if data_ok else None
    if data_ok:
        st.caption(tr(f"Active dataset: **{active_name()}** — {df.shape[0]} rows",
                      f"البيانات النشطة: **{active_name()}** — {df.shape[0]} صف"))
    else:
        st.info(tr("No data loaded — the Power & sample-size tools work without data. "
                   "Load data (or the demo dataset on Home) for the other tools.",
                   "لا توجد بيانات محمّلة، لكن أدوات القوة وحجم العينة تعمل بدونها. "
                   "حمّل بيانات (أو البيانات التجريبية من الصفحة الرئيسية) لاستخدام بقية الأدوات."))

    labels = [tr("🧭 Test Advisor", "🧭 مستشار الاختبارات"), tr("📏 One sample", "📏 عينة واحدة"),
              tr("⚖️ Two groups", "⚖️ مجموعتان"), tr("📊 Several groups (ANOVA)", "📊 عدة مجموعات (ANOVA)"),
              tr("🔗 Association", "🔗 العلاقة والاستقلال"), tr("🎯 Confidence intervals", "🎯 فترات الثقة"),
              tr("⚡ Power & sample size", "⚡ القوة وحجم العينة")]
    tabs = st.tabs(labels)
    with tabs[6]:
        _power_tab()
    if not data_ok:
        return
    with tabs[0]:
        _advisor_tab(df)
    with tabs[1]:
        _one_sample_tab(df)
    with tabs[2]:
        _two_groups_tab(df)
    with tabs[3]:
        _k_groups_tab(df)
    with tabs[4]:
        _association_tab(df)
    with tabs[5]:
        _ci_tab(df)


# ==========================================================
# 0. Test Advisor
# ==========================================================
def _advisor_tab(df):
    lang = _lang()
    st.markdown(tr("Answer three questions; the advisor checks the assumptions on **your data** and runs the right test.",
                   "أجب عن ثلاثة أسئلة، وسيفحص المستشار الافتراضات على **بياناتك** ثم يطبّق الاختبار المناسب."))
    goal = st.radio(tr("1. What is your research question?", "1. ما سؤالك البحثي؟"),
                    ["compare", "relationship", "one_sample"], key="adv_goal",
                    format_func=lambda g: {
                        "compare": tr("Are there differences between groups / conditions?", "هل توجد فروق بين مجموعات أو حالات؟"),
                        "relationship": tr("Is there a relationship between two variables?", "هل توجد علاقة بين متغيرين؟"),
                        "one_sample": tr("Does my sample differ from a known / target value?", "هل تختلف عينتي عن قيمة معلومة أو مستهدفة؟"),
                    }[g])
    alpha = st.select_slider(tr("Significance level α", "مستوى المعنوية α"), [0.01, 0.05, 0.10], 0.05, key="adv_alpha")
    num, cat = ds.analysis_columns(df), ds.categorical_columns(df)

    if goal == "one_sample":
        y = st.selectbox(tr("2. Variable", "2. المتغير"), num, key="adv_os_y")
        mu = st.number_input(tr("3. Test value (μ₀)", "3. القيمة المرجعية (μ₀)"), value=0.0, key="adv_os_mu", format="%.4f")
        x = pd.to_numeric(df[y], errors="coerce").dropna()
        a = inf.check_assumptions([x], alpha)
        key, en, ar = inf.recommend("one_sample", normal=a["normal"], large=a["large_samples"])
        _advisor_verdict(en, ar, a)
        res = inf.one_sample_t(x, mu) if key == "t1" else inf.wilcoxon_one_sample(x, mu)
        show_result(res, alpha, "adv")
        return

    if goal == "relationship":
        kind = st.radio(tr("2. Type of variables", "2. نوع المتغيرين"), ["numeric", "categorical"], horizontal=True,
                        key="adv_rel_kind", format_func=lambda k: tr("Both numeric / ordinal", "رقميان أو ترتيبيان")
                        if k == "numeric" else tr("Both categorical", "نوعيان"))
        if kind == "categorical":
            if len(cat) < 2:
                st.info(tr("Need two categorical variables.", "يلزم متغيران نوعيان."))
                return
            c1, c2 = st.columns(2)
            a_ = c1.selectbox(tr("3. Variable A", "3. المتغير الأول"), cat, key="adv_ca")
            b_ = c2.selectbox(tr("Variable B", "المتغير الثاني"), [c for c in cat if c != a_], key="adv_cb")
            _advisor_verdict("Two categorical variables → χ² test of independence.",
                             "متغيران نوعيان، لذا نستخدم اختبار كاي² للاستقلال.", None)
            res, ct, resid = inf.chi2_independence(df, a_, b_)
            st.dataframe(ct, width="stretch")
            show_result(res, alpha, "adv")
            return
        c1, c2 = st.columns(2)
        x_ = c1.selectbox(tr("3. Variable X", "3. المتغير X"), num, key="adv_rx")
        y_ = c2.selectbox(tr("Variable Y", "المتغير Y"), [c for c in num if c != x_], key="adv_ry")
        pair = df[[x_, y_]].apply(pd.to_numeric, errors="coerce").dropna()
        a = inf.check_assumptions([pair[x_], pair[y_]], alpha)
        key, en, ar = inf.recommend("relationship", outcome="numeric", normal=a["normal"], large=a["large_samples"])
        _advisor_verdict(en, ar, a)
        res = inf.correlation_test(pair[x_], pair[y_], "pearson" if key == "pearson" else "spearman")
        show_result(res, alpha, "adv")
        return

    # compare
    design = st.radio(tr("2. Design", "2. تصميم الدراسة"), ["independent", "paired"], horizontal=True, key="adv_design",
                      format_func=lambda d: tr("Independent groups (different units)", "مجموعات مستقلة (وحدات مختلفة)")
                      if d == "independent" else tr("Repeated / paired (same units measured more than once)",
                                                     "قياسات متكررة أو مرتبطة (نفس الوحدات قيست أكثر من مرة)"))
    if design == "paired":
        cols = st.multiselect(tr("3. Columns holding the repeated measurements (2 or more)",
                                 "3. الأعمدة التي تحتوي القياسات المتكررة (2 أو أكثر)"), num, default=num[:2], key="adv_pcols")
        if len(cols) < 2:
            return
        wide = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
        diffs = [wide[cols[0]] - wide[cols[1]]] if len(cols) == 2 else [wide[c] for c in cols]
        a = inf.check_assumptions(diffs, alpha)
        key, en, ar = inf.recommend("compare", len(cols), True, "numeric", a["normal"], True, a["large_samples"])
        _advisor_verdict(en, ar, a)
        res = {"tpaired": lambda: inf.t_paired(wide[cols[0]], wide[cols[1]]),
               "wilcoxon_paired": lambda: inf.wilcoxon_paired(wide[cols[0]], wide[cols[1]]),
               "rm_anova": lambda: inf.rm_anova(wide), "friedman": lambda: inf.friedman(wide)}[key]()
        show_result(res, alpha, "adv")
        return

    picked = _group_data(df, "adv")
    if not picked:
        return
    y, g, names, arrays = picked
    a = _assumptions(arrays, names, alpha)
    key, en, ar = inf.recommend("compare", len(arrays), False, "numeric", a["normal"], a["equal_var"], a["large_samples"])
    _advisor_verdict(en, ar, a)
    _group_summary(names, arrays)
    runners = {
        "t_student": lambda: inf.t_independent(*arrays, equal_var=True),
        "t_welch": lambda: inf.t_independent(*arrays, equal_var=False),
        "mwu": lambda: inf.mann_whitney(*arrays),
        "anova": lambda: inf.anova_oneway(arrays),
        "welch_anova": lambda: inf.welch_anova(arrays),
        "kruskal": lambda: inf.kruskal_wallis(arrays),
    }
    res = runners[key]()
    show_result(res, alpha, "adv", context=f"{y} ~ {g}")
    if len(arrays) > 2 and res.p < alpha:
        st.markdown("**" + tr("Post-hoc comparisons", "المقارنات البعدية") + "**")
        ph = {"anova": lambda: inf.posthoc_tukey(arrays, names),
              "welch_anova": lambda: inf.posthoc_games_howell(arrays, names),
              "kruskal": lambda: inf.posthoc_dunn(arrays, names, "holm")}[key]()
        st.dataframe(ph.round(4), width="stretch", hide_index=True)
        letters = inf.compact_letters(ph, names, [np.median(x) if key == "kruskal" else x.mean() for x in arrays], alpha)
        _means_plot(names, arrays, f"{y} — {g}", letters)
    else:
        _means_plot(names, arrays, f"{y} — {g}")


def _advisor_verdict(en, ar, a):
    msg = tr(f"✅ Recommended: {en}", f"✅ الاختبار المقترح: {ar}")
    if a is not None:
        norm = tr("normality satisfied", "شرط الطبيعية متحقق") if a["normal"] else (
            tr("normality not satisfied but n ≥ 30 in every group", "شرط الطبيعية غير متحقق لكن n ≥ 30 في كل مجموعة")
            if a["large_samples"] else tr("normality not satisfied", "شرط الطبيعية غير متحقق"))
        msg += f"  \n<small>{norm}" + (
            "" if pd.isna(a.get("brown_forsythe_p", np.nan)) else
            f" · {tr('equal variances', 'تجانس التباين')}: {'✅' if a['equal_var'] else '❌'} "
            f"(Brown-Forsythe p = {a['brown_forsythe_p']:.3f})") + "</small>"
    insight(msg)


# ==========================================================
# 1. One sample
# ==========================================================
def _one_sample_tab(df):
    test = st.selectbox(tr("Test", "الاختبار"), ["t", "wilcoxon", "sign", "prop", "gof", "var"], key="os_test",
                        format_func=lambda k: {
                            "t": tr("One-sample t-test (mean)", "اختبار t لعينة واحدة (المتوسط)"),
                            "wilcoxon": tr("Wilcoxon signed-rank (median, non-parametric)", "ويلكوكسون للرتب (الوسيط، لامعلمي)"),
                            "sign": tr("Sign test (median, minimal assumptions)", "اختبار الإشارة (الوسيط، أقل الافتراضات)"),
                            "prop": tr("Binomial test for a proportion", "اختبار ذي الحدين لنسبة"),
                            "gof": tr("χ² goodness of fit (distribution of categories)", "كاي² لجودة التوفيق (توزيع الفئات)"),
                            "var": tr("χ² test for a variance", "اختبار كاي² للتباين"),
                        }[k])
    alpha, alt = _settings("os", alternative=test not in ("gof",))
    num, cat = ds.analysis_columns(df), ds.categorical_columns(df)
    if test in ("t", "wilcoxon", "sign", "var"):
        y = st.selectbox(tr("Variable", "المتغير"), num, key="os_y")
        label = tr("Hypothesised variance σ₀²", "التباين المفترض σ₀²") if test == "var" else tr("Test value (μ₀)", "القيمة المرجعية (μ₀)")
        mu = st.number_input(label, value=1.0 if test == "var" else 0.0, key=f"os_mu_{test}", format="%.4f")
        x = df[y]
        res = {"t": lambda: inf.one_sample_t(x, mu, alt), "wilcoxon": lambda: inf.wilcoxon_one_sample(x, mu, alt),
               "sign": lambda: inf.sign_test(x, mu, alt), "var": lambda: inf.variance_test_one(x, mu, alt)}[test]()
        show_result(res, alpha, "os")
    elif test == "prop":
        if not cat:
            st.info(tr("No categorical variable.", "لا يوجد متغير نوعي."))
            return
        v = st.selectbox(tr("Variable", "المتغير"), cat, key="os_pv")
        lv = st.selectbox(tr("'Success' category", "فئة «النجاح»"), sorted(df[v].dropna().astype(str).unique()), key="os_plv")
        p0 = st.number_input(tr("Hypothesised proportion p₀", "النسبة المفترضة p₀"), 0.0, 1.0, 0.5, key="os_p0")
        s = df[v].dropna().astype(str)
        show_result(inf.one_proportion(int((s == lv).sum()), int(s.size), p0, alt), alpha, "os")
    else:
        if not cat:
            st.info(tr("No categorical variable.", "لا يوجد متغير نوعي."))
            return
        v = st.selectbox(tr("Variable", "المتغير"), cat, key="os_gv")
        counts = df[v].dropna().astype(str).value_counts().sort_index()
        st.caption(tr("Expected proportions (default: equal). Edit the 'expected' column:",
                      "النسب المتوقعة (الافتراضي: متساوية). عدّل عمود «المتوقع»:"))
        exp_df = pd.DataFrame({"category": counts.index, "observed": counts.values,
                               "expected": np.round(1 / len(counts), 4)})
        edited = st.data_editor(exp_df, disabled=["category", "observed"], key="os_gof_ed", hide_index=True) \
            if hasattr(st, "data_editor") else exp_df
        show_result(inf.chi2_goodness_of_fit(edited["observed"], edited["expected"]), alpha, "os")


# ==========================================================
# 2. Two groups
# ==========================================================
def _two_groups_tab(df):
    design = st.radio(tr("Design", "التصميم"), ["independent", "paired"], horizontal=True, key="tg_design",
                      format_func=lambda d: tr("Independent samples", "عينتان مستقلتان") if d == "independent"
                      else tr("Paired / related samples", "عينتان مرتبطتان (مزدوجتان)"))
    alpha, alt = _settings("tg")
    if design == "independent":
        picked = _group_data(df, "tg", 2, 2)
        if not picked:
            return
        y, g, names, (x1, x2) = picked
        _assumptions([x1, x2], names, alpha)
        _group_summary(names, [x1, x2])
        test = st.selectbox(tr("Test", "الاختبار"), ["welch", "student", "mwu", "ks", "var"], key="tg_test",
                            format_func=lambda k: {
                                "welch": tr("Welch's t-test (recommended default)", "اختبار ويلش t (الخيار الافتراضي الموصى به)"),
                                "student": tr("Student's t-test (equal variances)", "اختبار ستودنت t (تباينات متساوية)"),
                                "mwu": tr("Mann-Whitney U (non-parametric)", "مان-ويتني U (لامعلمي)"),
                                "ks": tr("Kolmogorov-Smirnov (whole distribution)", "كولموجوروف-سميرنوف (كامل التوزيع)"),
                                "var": tr("Tests of equal variances", "اختبارات تجانس التباين"),
                            }[k])
        if test == "var":
            vt = inf.variance_tests([x1, x2])
            vt = vt.drop(columns="Test") if is_ar() else vt.drop(columns="Test (AR)")
            st.dataframe(vt.round(4), width="stretch", hide_index=True)
        else:
            res = {"welch": lambda: inf.t_independent(x1, x2, False, alt),
                   "student": lambda: inf.t_independent(x1, x2, True, alt),
                   "mwu": lambda: inf.mann_whitney(x1, x2, alt), "ks": lambda: inf.ks_two_sample(x1, x2)}[test]()
            show_result(res, alpha, "tg", context=f"{names[0]} vs {names[1]}")
        _means_plot(names, [x1, x2], f"{y} — {g}")
        return

    num, cat = ds.analysis_columns(df), ds.categorical_columns(df)
    kind = st.radio(tr("Measurement", "نوع القياس"), ["numeric", "binary"], horizontal=True, key="tg_pkind",
                    format_func=lambda k: tr("Numeric (before / after)", "رقمي (قبل / بعد)") if k == "numeric"
                    else tr("Binary (yes / no)", "ثنائي (نعم / لا)"))
    pool = num if kind == "numeric" else cat
    if len(pool) < 2:
        st.info(tr("Need two columns of this type.", "يلزم عمودان من هذا النوع."))
        return
    c1, c2 = st.columns(2)
    a = c1.selectbox(tr("Measurement 1 (e.g. before)", "القياس الأول (مثلاً: قبل)"), pool, key="tg_pa")
    b = c2.selectbox(tr("Measurement 2 (e.g. after)", "القياس الثاني (مثلاً: بعد)"), [c for c in pool if c != a], key="tg_pb")
    if kind == "binary":
        show_result(inf.mcnemar(df[a], df[b]), alpha, "tg")
        return
    test = st.selectbox(tr("Test", "الاختبار"), ["t", "wilcoxon", "sign"], key="tg_ptest",
                        format_func=lambda k: {"t": tr("Paired t-test", "اختبار t للعينات المرتبطة"),
                                               "wilcoxon": tr("Wilcoxon signed-rank", "ويلكوكسون للرتب ذات الإشارة"),
                                               "sign": tr("Sign test", "اختبار الإشارة")}[k])
    pair = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna()
    _assumptions([pair[a] - pair[b]], [tr("Differences", "الفروق")], alpha)
    res = {"t": lambda: inf.t_paired(pair[a], pair[b], alt),
           "wilcoxon": lambda: inf.wilcoxon_paired(pair[a], pair[b], alt),
           "sign": lambda: inf.sign_test(pair[a] - pair[b], 0.0, alt)}[test]()
    show_result(res, alpha, "tg", context=f"{a} vs {b}")
    _means_plot([a, b], [pair[a].to_numpy(), pair[b].to_numpy()], f"{a} vs {b}")


# ==========================================================
# 3. Several groups
# ==========================================================
def _k_groups_tab(df):
    design = st.radio(tr("Design", "التصميم"), ["oneway", "twoway", "repeated"], horizontal=True, key="kg_design",
                      format_func=lambda d: {"oneway": tr("One factor", "عامل واحد"),
                                             "twoway": tr("Two factors (factorial)", "عاملان (تجربة عاملية)"),
                                             "repeated": tr("Repeated measures / blocks", "قياسات متكررة / قطاعات")}[d])
    alpha = st.select_slider(tr("Significance level α", "مستوى المعنوية α"), [0.01, 0.05, 0.10], 0.05, key="kg_alpha")

    if design == "oneway":
        picked = _group_data(df, "kg", 2)
        if not picked:
            return
        y, g, names, arrays = picked
        a = _assumptions(arrays, names, alpha)
        _group_summary(names, arrays)
        test = st.selectbox(tr("Omnibus test", "الاختبار الكلي"), ["anova", "welch", "kruskal", "median"],
                            index=0 if a["equal_var"] else 1, key="kg_test",
                            format_func=lambda k: {
                                "anova": tr("One-way ANOVA (equal variances)", "ANOVA أحادي (تباينات متساوية)"),
                                "welch": tr("Welch's ANOVA (unequal variances)", "ANOVA لويلش (تباينات غير متساوية)"),
                                "kruskal": tr("Kruskal-Wallis (non-parametric)", "كروسكال-واليس (لامعلمي)"),
                                "median": tr("Mood's median test", "اختبار الوسيط لمود")}[k])
        res = {"anova": lambda: inf.anova_oneway(arrays), "welch": lambda: inf.welch_anova(arrays),
               "kruskal": lambda: inf.kruskal_wallis(arrays), "median": lambda: inf.mood_median(arrays)}[test]()
        show_result(res, alpha, "kg", context=f"{y} ~ {g}")

        st.markdown("#### " + tr("Post-hoc multiple comparisons", "المقارنات البعدية المتعددة"))
        default_ph = {"anova": "tukey", "welch": "games", "kruskal": "dunn", "median": "dunn"}[test]
        opts = ["tukey", "games", "bonf_t", "dunn"]
        c1, c2 = st.columns(2)
        ph_kind = c1.selectbox(tr("Method", "الطريقة"), opts, index=opts.index(default_ph), key="kg_ph",
                               format_func=lambda k: {"tukey": "Tukey HSD", "games": "Games-Howell",
                                                      "bonf_t": tr("Pairwise t-tests (adjusted)", "اختبارات t ثنائية (مصححة)"),
                                                      "dunn": tr("Dunn (after Kruskal-Wallis)", "دن (بعد كروسكال-واليس)")}[k])
        adj = c2.selectbox(tr("p-value adjustment", "تصحيح القيم الاحتمالية"), list(inf.ADJUST_METHODS), key="kg_adj",
                           format_func=lambda m: inf.ADJUST_METHODS[m][1 if is_ar() else 0],
                           disabled=ph_kind in ("tukey", "games"))
        ph = {"tukey": lambda: inf.posthoc_tukey(arrays, names), "games": lambda: inf.posthoc_games_howell(arrays, names),
              "bonf_t": lambda: inf.posthoc_pairwise_t(arrays, names, adj),
              "dunn": lambda: inf.posthoc_dunn(arrays, names, adj)}[ph_kind]()
        ph["Sig."] = ph["p-adj"].map(inf.significance_marker)
        st.dataframe(ph.round(4), width="stretch", hide_index=True)
        csv_download(ph.round(4), "posthoc.csv", key="kg_ph_dl")
        centre = [np.median(x) for x in arrays] if ph_kind == "dunn" else [x.mean() for x in arrays]
        letters = inf.compact_letters(ph, names, centre, alpha)
        st.markdown(tr("**Compact letter display** (groups sharing a letter do not differ significantly): ",
                       "**الحروف الدالة** (المجموعات التي تشترك في حرف لا تختلف معنوياً): ")
                    + " · ".join(f"{n}: **{letters[n]}**" for n in names))
        _means_plot(names, arrays, f"{y} — {g}", letters)
        return

    if design == "twoway":
        num, cat = ds.analysis_columns(df), ds.categorical_columns(df)
        if len(cat) < 2:
            st.info(tr("Needs two categorical factors.", "يتطلب عاملين نوعيين."))
            return
        c1, c2, c3 = st.columns(3)
        y = c1.selectbox(tr("Outcome", "المتغير التابع"), num, key="kg2_y")
        fa = c2.selectbox(tr("Factor A", "العامل A"), cat, key="kg2_a")
        fb = c3.selectbox(tr("Factor B", "العامل B"), [c for c in cat if c != fa], key="kg2_b")
        c1, c2 = st.columns(2)
        inter = c1.checkbox(tr("Include interaction A × B", "تضمين التفاعل A × B"), True, key="kg2_int")
        sst = c2.radio(tr("Sums of squares", "نوع مجموع المربعات"), [3, 2], horizontal=True, key="kg2_ss",
                       format_func=lambda t: tr("Type III (SPSS default)", "النوع الثالث (افتراضي SPSS)") if t == 3
                       else tr("Type II", "النوع الثاني"))
        table = inf.anova_twoway(df, y, fa, fb, inter, sst)
        st.dataframe(table.round(4), width="stretch", hide_index=True)
        csv_download(table.round(5), "two_way_anova.csv", key="kg2_dl")
        for _, r in table.dropna(subset=["F"]).iterrows():
            sig = r["p-value"] < alpha
            mag = inf.effect_magnitude(r["partial η²"], "eta2", _lang())
            insight(tr(f"<b>{r['Source']}</b>: F({r['df']:.0f}, {table.iloc[-1]['df']:.0f}) = {r['F']:.2f}, "
                       f"{inf.fmt_p(r['p-value'])}, partial η² = {r['partial η²']:.3f} ({mag}) — "
                       f"{'significant' if sig else 'not significant'}.",
                       f"<b>{r['Source']}</b>: F({r['df']:.0f}, {table.iloc[-1]['df']:.0f}) = {r['F']:.2f}، "
                       f"{inf.fmt_p(r['p-value'])}، مربع إيتا الجزئي = {r['partial η²']:.3f} (أثر {mag}) — "
                       f"{'دال' if sig else 'غير دال'} إحصائياً."))
        if inter and table.iloc[2]["p-value"] < alpha:
            st.warning(tr("The interaction is significant: interpret main effects with care, and compare simple effects.",
                          "التفاعل دال إحصائياً، لذا يجب تفسير الآثار الرئيسية بحذر ودراسة الآثار البسيطة."))
        means = df.groupby([fa, fb])[y].mean().reset_index()
        fig, _ = ch.build_chart("line", means.assign(**{fa: means[fa].astype(str)}), x=fa, y=y, color=fb, agg="mean")
        ch.style_academic(fig, title=tr("Interaction plot (cell means)", "رسم التفاعل (متوسطات الخلايا)"),
                          x_title=fa, y_title=y, height=430, rtl=is_ar())
        st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, "interaction_plot"))
        return

    # repeated measures
    num = ds.analysis_columns(df)
    kind = st.radio(tr("Outcome type", "نوع المتغير"), ["numeric", "binary"], horizontal=True, key="kg_rm_kind",
                    format_func=lambda k: tr("Numeric", "رقمي") if k == "numeric" else tr("Binary 0/1", "ثنائي 0/1"))
    cols = st.multiselect(tr("Columns = conditions / time points (each row = one subject)",
                             "الأعمدة = الحالات أو الفترات الزمنية (كل صف = وحدة واحدة)"), num, default=num[:3], key="kg_rm_cols")
    if len(cols) < 2:
        return
    wide = df[cols]
    if kind == "binary":
        show_result(inf.cochran_q(wide), alpha, "kg_rm")
        return
    test = st.radio(tr("Test", "الاختبار"), ["rm", "friedman"], horizontal=True, key="kg_rm_test",
                    format_func=lambda k: tr("Repeated-measures ANOVA", "ANOVA للقياسات المتكررة") if k == "rm"
                    else tr("Friedman (non-parametric)", "فريدمان (لامعلمي)"))
    show_result(inf.rm_anova(wide) if test == "rm" else inf.friedman(wide), alpha, "kg_rm")
    clean = wide.apply(pd.to_numeric, errors="coerce").dropna()
    _means_plot(cols, [clean[c].to_numpy() for c in cols], tr("Means across conditions", "المتوسطات عبر الحالات"))


# ==========================================================
# 4. Association
# ==========================================================
def _association_tab(df):
    kind = st.radio(tr("Analysis", "التحليل"), ["corr", "partial", "chi2"], horizontal=True, key="as_kind",
                    format_func=lambda k: {"corr": tr("Correlation test", "اختبار الارتباط"),
                                           "partial": tr("Partial correlation", "الارتباط الجزئي"),
                                           "chi2": tr("χ² independence / Fisher", "كاي² للاستقلال / فيشر")}[k])
    alpha, alt = _settings("as", alternative=kind == "corr")
    num, cat = ds.analysis_columns(df), ds.categorical_columns(df)
    if kind in ("corr", "partial"):
        c1, c2, c3 = st.columns(3)
        x = c1.selectbox("X", num, key="as_x")
        y = c2.selectbox("Y", [c for c in num if c != x], key="as_y")
        method = c3.selectbox(tr("Method", "الطريقة"), ["pearson", "spearman", "kendall"] if kind == "corr"
                              else ["pearson", "spearman"], key=f"as_m_{kind}")
        if kind == "corr":
            res = inf.correlation_test(df[x], df[y], method, alt)
        else:
            cov = st.multiselect(tr("Control variables", "متغيرات الضبط (المراد عزل أثرها)"),
                                 [c for c in num if c not in (x, y)], key="as_cov")
            if not cov:
                st.info(tr("Choose at least one control variable.", "اختر متغير ضبط واحداً على الأقل."))
                return
            try:
                res = inf.partial_correlation(df, x, y, cov, method)
            except ValueError as e:
                st.error(str(e))
                return
        show_result(res, alpha, "as", context=f"{x} — {y}")
        fig, _ = ch.build_chart("scatter", df, x=x, y=y)
        ch.style_academic(fig, title=f"{y} vs {x}", x_title=x, y_title=y, height=420, legend="none", rtl=is_ar())
        st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, "scatter"))
        return
    if len(cat) < 2:
        st.info(tr("Need two categorical variables.", "يلزم متغيران نوعيان."))
        return
    c1, c2 = st.columns(2)
    a = c1.selectbox(tr("Rows", "الصفوف"), cat, key="as_a")
    b = c2.selectbox(tr("Columns", "الأعمدة"), [c for c in cat if c != a], key="as_b")
    try:
        res, ct, resid = inf.chi2_independence(df, a, b)
    except ValueError as e:
        st.error(str(e))
        return
    st.markdown("**" + tr("Contingency table", "جدول التوافق") + "**")
    st.dataframe(ct, width="stretch")
    show_result(res, alpha, "as")
    st.markdown("**" + tr("Adjusted standardised residuals", "البواقي المعيارية المعدّلة") + "**")
    st.dataframe(resid.round(2).style.map(lambda v: "background-color:#fde2e1" if v > 1.96 else
                                          ("background-color:#dfe9f6" if v < -1.96 else "")), width="stretch")
    st.caption(tr("Cells with |residual| > 1.96 (shaded) drive the association: red = more than expected, blue = fewer.",
                  "الخلايا التي تزيد قيمة باقيها المطلق على 1.96 (المظللة) هي مصدر العلاقة: الأحمر أكثر من المتوقع والأزرق أقل."))


# ==========================================================
# 5. Confidence intervals
# ==========================================================
def _ci_tab(df):
    num, cat = ds.analysis_columns(df), ds.categorical_columns(df)
    conf = st.select_slider(tr("Confidence level", "مستوى الثقة"), [0.90, 0.95, 0.99], 0.95, key="ci_conf",
                            format_func=lambda c: f"{c:.0%}")
    kind = st.radio(tr("Parameter", "المعلمة"), ["mean", "variance", "prop", "boot"], horizontal=True, key="ci_kind",
                    format_func=lambda k: {"mean": tr("Mean (t)", "المتوسط (t)"), "variance": tr("Variance (χ²)", "التباين (كاي²)"),
                                           "prop": tr("Proportion", "النسبة"),
                                           "boot": tr("Bootstrap (any statistic)", "البوتستراب (لأي إحصاءة)")}[k])
    if kind == "prop":
        if not cat:
            st.info(tr("No categorical variable.", "لا يوجد متغير نوعي."))
            return
        v = st.selectbox(tr("Variable", "المتغير"), cat, key="ci_pv")
        s = df[v].dropna().astype(str)
        rows = []
        for lv, k in s.value_counts().items():
            w = inf.proportion_ci(int(k), s.size, conf, "wilson")
            e = inf.proportion_ci(int(k), s.size, conf, "exact")
            rows.append({tr("Category", "الفئة"): lv, "n": int(k), tr("Proportion", "النسبة"): k / s.size,
                         "Wilson low": w[0], "Wilson high": w[1], "Clopper-Pearson low": e[0], "Clopper-Pearson high": e[1]})
        st.dataframe(pd.DataFrame(rows).round(4), width="stretch", hide_index=True)
        st.caption(tr("Wilson interval is recommended; Clopper-Pearson is the exact (conservative) interval.",
                      "يُوصى بفترة ويلسون، أما كلوبر-بيرسون فهي الفترة الدقيقة (المتحفظة)."))
        return
    cols = st.multiselect(tr("Variables", "المتغيرات"), num, default=[c for c in num if c not in ds.time_columns(df)][:4],
                          key="ci_cols")
    rows = []
    if kind == "boot":
        c1, c2 = st.columns(2)
        stat = c1.selectbox(tr("Statistic", "الإحصاءة"), list(inf.BOOT_STATS), key="ci_bstat",
                            format_func=lambda k: inf.BOOT_STATS[k][2 if is_ar() else 1])
        B = c2.select_slider(tr("Resamples", "عدد إعادات المعاينة"), [1000, 2000, 5000, 10000], 2000, key="ci_B")
    for c in cols:
        x = df[c]
        if kind == "mean":
            est, lo, hi = inf.ci_mean(x, conf)
        elif kind == "variance":
            est, lo, hi = inf.ci_variance(x, conf)
        else:
            est, lo, hi, se = inf.bootstrap_ci(x, stat, conf, B)
        rows.append({tr("Variable", "المتغير"): c, tr("Estimate", "التقدير"): est,
                     tr("Lower", "الحد الأدنى"): lo, tr("Upper", "الحد الأعلى"): hi})
    if rows:
        out = pd.DataFrame(rows).round(4)
        st.dataframe(out, width="stretch", hide_index=True)
        csv_download(out, "confidence_intervals.csv", key="ci_dl")
    if kind == "boot":
        st.caption(tr("BCa bootstrap (bias-corrected & accelerated, Efron 1987) — valid without normality.",
                      "بوتستراب BCa (مصحح التحيز والمعجّل، Efron 1987)، وهو صالح دون افتراض الطبيعية."))


# ==========================================================
# 6. Power & sample size (no data needed)
# ==========================================================
def _power_tab():
    st.markdown(tr("Plan your study **before** collecting data, or justify your sample size to reviewers.",
                   "خطّط لدراستك **قبل** جمع البيانات، أو برّر حجم عينتك أمام المحكّمين."))
    design = st.selectbox(tr("Design", "التصميم"), ["t2", "t1", "anova", "corr", "prop2", "chi2", "survey"], key="pw_design",
                          format_func=lambda k: {
                              "t2": tr("Two independent groups (t-test)", "مجموعتان مستقلتان (اختبار t)"),
                              "t1": tr("Paired / one-sample (t-test)", "عينات مرتبطة / عينة واحدة (اختبار t)"),
                              "anova": tr("Several groups (one-way ANOVA)", "عدة مجموعات (ANOVA أحادي)"),
                              "corr": tr("Correlation", "الارتباط"),
                              "prop2": tr("Two proportions", "نسبتان"),
                              "chi2": tr("χ² test (contingency table)", "اختبار كاي² (جدول توافق)"),
                              "survey": tr("Survey / questionnaire sample size", "حجم عينة الاستبيان أو المسح الميداني"),
                          }[k])
    if design == "survey":
        _survey_size()
        return
    mode = st.radio(tr("Calculate", "المطلوب حسابه"), ["n", "power"], horizontal=True, key="pw_mode",
                    format_func=lambda m: tr("Required sample size", "حجم العينة اللازم") if m == "n"
                    else tr("Power of a given sample", "قوة الاختبار لعينة معلومة"))
    c1, c2, c3 = st.columns(3)
    alpha = c1.select_slider("α", [0.01, 0.05, 0.10], 0.05, key="pw_alpha")
    target = c2.select_slider(tr("Target power", "القوة المستهدفة"), [0.80, 0.85, 0.90, 0.95], 0.80, key="pw_target")
    alt = c3.selectbox(tr("Tails", "الأطراف"), ["two-sided", "greater"], key="pw_alt",
                       format_func=lambda a: tr("Two-sided", "طرفان") if a == "two-sided" else tr("One-sided", "طرف واحد"))

    if design in ("t2", "t1"):
        eff = st.select_slider(tr("Effect size d (Cohen: 0.2 small, 0.5 medium, 0.8 large)",
                                  "حجم الأثر d (كوهين: 0.2 صغير، 0.5 متوسط، 0.8 كبير)"),
                               [round(v, 2) for v in np.arange(0.1, 1.55, 0.05)], 0.5, key="pw_d")
        kind = "two" if design == "t2" else "one"
        fn = lambda n: pw.power_t(eff, n, alpha, kind, alt)  # noqa: E731
        unit = tr("per group", "لكل مجموعة") if design == "t2" else tr("pairs / subjects", "زوجاً / وحدة")
        curve_fn = lambda n, e: pw.power_t(e, n, alpha, kind, alt)  # noqa: E731
        effects = [0.2, 0.5, 0.8]
    elif design == "anova":
        k = st.number_input(tr("Number of groups k", "عدد المجموعات k"), 2, 20, 3, key="pw_k")
        eff = st.select_slider(tr("Effect size f (0.10 small, 0.25 medium, 0.40 large)",
                                  "حجم الأثر f (0.10 صغير، 0.25 متوسط، 0.40 كبير)"),
                               [round(v, 2) for v in np.arange(0.05, 0.81, 0.05)], 0.25, key="pw_f")
        fn = lambda n: pw.power_anova(eff, k, n, alpha)  # noqa: E731
        unit = tr("per group", "لكل مجموعة")
        curve_fn = lambda n, e: pw.power_anova(e, k, n, alpha)  # noqa: E731
        effects = [0.10, 0.25, 0.40]
    elif design == "corr":
        eff = st.select_slider(tr("Expected correlation r (0.1 small, 0.3 medium, 0.5 large)",
                                  "معامل الارتباط المتوقع r (0.1 صغير، 0.3 متوسط، 0.5 كبير)"),
                               [round(v, 2) for v in np.arange(0.05, 0.91, 0.05)], 0.3, key="pw_r")
        fn = lambda n: pw.power_correlation(eff, n, alpha, alt)  # noqa: E731
        unit = tr("observations", "مشاهدة")
        curve_fn = lambda n, e: pw.power_correlation(e, n, alpha, alt)  # noqa: E731
        effects = [0.1, 0.3, 0.5]
    elif design == "prop2":
        c1, c2 = st.columns(2)
        p1 = c1.number_input(tr("Expected proportion, group 1", "النسبة المتوقعة في المجموعة 1"), 0.01, 0.99, 0.30, key="pw_p1")
        p2 = c2.number_input(tr("Expected proportion, group 2", "النسبة المتوقعة في المجموعة 2"), 0.01, 0.99, 0.45, key="pw_p2")
        eff = abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
        fn = lambda n: pw.power_two_proportions(p1, p2, n, alpha, alt)  # noqa: E731
        unit = tr("per group", "لكل مجموعة")
        curve_fn = None
        effects = []
    else:
        c1, c2 = st.columns(2)
        eff = c1.select_slider(tr("Effect size w (0.1 small, 0.3 medium, 0.5 large)",
                                  "حجم الأثر w (0.1 صغير، 0.3 متوسط، 0.5 كبير)"),
                               [round(v, 2) for v in np.arange(0.05, 0.81, 0.05)], 0.3, key="pw_w")
        dfc = c2.number_input(tr("Degrees of freedom (r−1)(c−1)", "درجات الحرية (r−1)(c−1)"), 1, 50, 1, key="pw_df")
        fn = lambda n: pw.power_chi2(eff, n, dfc, alpha)  # noqa: E731
        unit = tr("total observations", "مشاهدة إجمالاً")
        curve_fn = lambda n, e: pw.power_chi2(e, n, dfc, alpha)  # noqa: E731
        effects = [0.1, 0.3, 0.5]

    if mode == "n":
        n = pw.solve_n(fn, target)
        st.metric(tr("Required sample size", "حجم العينة اللازم"), f"{n:,} {unit}")
        insight(tr(f"To detect an effect of {eff:.2f} with {target:.0%} power at α = {alpha} ({alt}), "
                   f"you need <b>{n:,} {unit}</b>. Add 10–20% for expected non-response or attrition.",
                   f"لاكتشاف أثر قدره {eff:.2f} بقوة {target:.0%} عند مستوى معنوية {alpha}، "
                   f"تحتاج إلى <b>{n:,} {unit}</b>. أضف 10–20% تحسباً لعدم الاستجابة أو التسرب."))
    else:
        n_in = st.number_input(tr("Sample size", "حجم العينة") + f" ({unit})", 3, 1_000_000, 30, key="pw_nin")
        power = fn(n_in)
        st.metric(tr("Statistical power", "قوة الاختبار"), f"{power:.1%}")
        insight(tr(f"With n = {n_in} {unit}, the probability of detecting the effect is {power:.1%}"
                   + (" — adequate (≥ 80%)." if power >= 0.8 else " — below the conventional 80%; consider a larger sample."),
                   f"مع n = {n_in} {unit}، احتمال اكتشاف الأثر {power:.1%}"
                   + (" وهي قوة كافية (≥ 80%)." if power >= 0.8 else " وهي أقل من 80% المتعارف عليها، لذا يُنصح بزيادة حجم العينة.")))

    if curve_fn:
        ns = np.unique(np.round(np.geomspace(5, 1000, 60)).astype(int))
        fig = go.Figure()
        for i, e in enumerate(effects):
            fig.add_trace(go.Scatter(x=ns, y=[curve_fn(n, e) for n in ns], mode="lines", name=f"{e}",
                                     line=dict(color=ch.PALETTES["academic"][i], width=2.5)))
        fig.add_hline(y=target, line_dash="dot", line_color="#888")
        fig.update_xaxes(type="log")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        ch.style_academic(fig, title=tr("Power curves by effect size", "منحنيات القوة حسب حجم الأثر"),
                          x_title=tr("Sample size", "حجم العينة") + f" ({unit})", y_title=tr("Power", "القوة"),
                          height=420, rtl=is_ar())
        st.plotly_chart(fig, config=ch.plotly_export_config("png", 3, "power_curve"))
    st.caption(tr("Exact non-central t / F / χ² computations (results match G*Power).",
                  "الحسابات دقيقة باستخدام توزيعات t وF وكاي² غير المركزية، والنتائج مطابقة لبرنامج G*Power."))


def _survey_size():
    method = st.radio(tr("Method", "الطريقة"), ["cochran", "yamane", "mean"], horizontal=True, key="sv_m",
                      format_func=lambda m: {"cochran": tr("Cochran (proportion)", "كوكران (للنسب)"),
                                             "yamane": tr("Yamane / Slovin", "يامان / سلوفين"),
                                             "mean": tr("Estimate a mean", "لتقدير متوسط")}[m])
    c1, c2, c3 = st.columns(3)
    margin = c1.number_input(tr("Margin of error e", "هامش الخطأ e"), 0.005, 0.5, 0.05, step=0.005, format="%.3f", key="sv_e")
    conf = c2.select_slider(tr("Confidence", "مستوى الثقة"), [0.90, 0.95, 0.99], 0.95, key="sv_conf",
                            format_func=lambda c: f"{c:.0%}")
    N = c3.number_input(tr("Population size N (0 = unknown / very large)", "حجم المجتمع N (صفر = غير معروف أو كبير جداً)"),
                        0, 100_000_000, 0, key="sv_N")
    if method == "cochran":
        c1, c2, c3 = st.columns(3)
        p = c1.number_input(tr("Expected proportion p (0.5 = most conservative)", "النسبة المتوقعة p (0.5 = الأكثر تحفظاً)"),
                            0.01, 0.99, 0.5, key="sv_p")
        deff = c2.number_input(tr("Design effect (cluster sampling > 1)", "أثر التصميم (العينات العنقودية > 1)"),
                               1.0, 5.0, 1.0, step=0.1, key="sv_deff")
        rr = c3.number_input(tr("Expected response rate", "معدل الاستجابة المتوقع"), 0.1, 1.0, 1.0, step=0.05, key="sv_rr")
        res = pw.cochran_sample_size(p, margin, conf, N or None, deff, rr)
        st.metric(tr("Recommended sample size", "حجم العينة الموصى به"), f"{res['Recommended n']:,}")
        st.write(tr(f"Questionnaires to distribute (allowing for non-response): **{res['n to contact (non-response)']:,}**",
                    f"عدد الاستمارات الواجب توزيعها (مع مراعاة عدم الاستجابة): **{res['n to contact (non-response)']:,}**"))
        st.latex(r"n_0=\frac{z^2\,p(1-p)}{e^2},\qquad n=\frac{n_0}{1+\frac{n_0-1}{N}}")
    elif method == "yamane":
        if not N:
            st.warning(tr("Yamane's formula needs the population size N.", "تتطلب معادلة يامان معرفة حجم المجتمع N."))
            return
        st.metric(tr("Sample size", "حجم العينة"), f"{pw.yamane_sample_size(int(N), margin):,}")
        st.latex(r"n=\frac{N}{1+N e^2}")
        st.caption(tr("Yamane assumes p = 0.5 and 95% confidence; Cochran is more general and preferred by reviewers.",
                      "تفترض معادلة يامان أن p = 0.5 ومستوى ثقة 95%، أما معادلة كوكران فأعم ويفضلها المحكّمون."))
    else:
        sd = st.number_input(tr("Expected standard deviation (pilot study / literature)",
                                "الانحراف المعياري المتوقع (من دراسة استطلاعية أو دراسات سابقة)"), 0.0001, 1e9, 10.0, key="sv_sd")
        e_abs = st.number_input(tr("Margin of error (same units as the variable)", "هامش الخطأ (بنفس وحدة المتغير)"),
                                0.0001, 1e9, 2.0, key="sv_eabs")
        st.metric(tr("Sample size", "حجم العينة"), f"{pw.mean_sample_size(sd, e_abs, conf, N or None):,}")
        st.latex(r"n=\left(\frac{z\,\sigma}{e}\right)^2")
