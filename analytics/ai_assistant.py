# ==========================================
# Aetheris Research - AI research assistant (v1.12, Catalogue §8)
# 1) Model advisor: profiles the active data and the research goal and recommends models and where to run them;
# 2) Writer: methodology, results, abstract and plain-language summaries written ONLY from the computed results (the
#    numbers in the output are checked against the results and any untraceable figure is flagged);
# 3) Q&A in Arabic or English, grounded in the platform's methodology library and the session's results.
# Works with Claude, ChatGPT, Gemini, Groq, OpenRouter or any OpenAI-compatible model (analytics/ai_providers.py), with
# the researcher's own key or the platform key under access controls; otherwise a deterministic offline mode answers from
# the built-in library and templates.
# ==========================================

from __future__ import annotations

import inspect
import json
import re

import numpy as np
import pandas as pd

from analytics import report_studio as rs

from analytics.ai_providers import LLM, PROVIDERS, resolve  # noqa: E402,F401  (multi-provider layer, v1.12.1)


def session_llm() -> LLM:
    """The LLM for the current Streamlit session (user key first, else the governed platform key)."""
    try:
        import streamlit as st
        return resolve(st.session_state)
    except Exception:  # noqa: BLE001
        return LLM("anthropic", None, None)


SYSTEM = {
    "en": ("You are the research assistant of Aetheris Research, an econometrics platform for economists and policy makers. "
           "Write in clear academic English. NEVER invent numbers: every number you write must appear in the RESULTS provided "
           "(you may round to fewer decimals or express a proportion as a percentage). If something is not in the results, say it "
           "was not computed. Do not claim causality the design does not support. Return plain text paragraphs without markdown "
           "headings."),
    "ar": ("أنت المساعد البحثي لمنصة Aetheris Research للاقتصاد القياسي، ومستخدموها باحثون اقتصاديون وصناع قرار. اكتب بالعربية "
           "الفصحى الأكاديمية الواضحة. لا تخترع أي رقم مطلقاً: كل رقم تكتبه يجب أن يكون موجوداً في النتائج المرفقة (يجوز التقريب "
           "إلى عدد أقل من المنازل العشرية أو التعبير عن النسبة كنسبة مئوية). وإذا لم يكن الشيء محسوباً فقل إنه لم يُحسب. لا تدّعِ "
           "سببية لا يدعمها التصميم. أعد فقرات نصية دون عناوين Markdown، واكتب أسماء المتغيرات والنماذج كما هي."),
}

TASKS = {
    "results": ("Write the Results paragraph(s) for the analysis below, in journal style: report the key estimates with standard "
                "errors / confidence intervals and significance, the diagnostics, and what they mean economically.",
                "اكتب فقرة (فقرات) النتائج للتحليل التالي بأسلوب المجلات العلمية: اعرض أهم التقديرات مع أخطائها المعيارية أو فترات "
                "الثقة ومعنويتها، ونتائج الاختبارات التشخيصية، ودلالتها الاقتصادية."),
    "methodology": ("Write the Methodology paragraph for the analysis below (model, estimator, inference, diagnostics), citing the "
                    "references given.", "اكتب فقرة المنهجية للتحليل التالي (النموذج والمقدِّر والاستدلال والاختبارات التشخيصية) مع "
                    "الإشارة إلى المراجع المرفقة."),
    "discussion": ("Write a short Discussion paragraph: interpret the findings, their policy implications and the main limitations.",
                   "اكتب فقرة مناقشة قصيرة: فسّر النتائج ودلالاتها للسياسات وأهم القيود."),
    "policy": ("Write a one-page policy brief for decision makers in plain, non-technical language: 3–5 key messages with the "
               "numbers that support them, recommended actions, and caveats. No jargon, no formulas.",
               "اكتب موجز سياسات من صفحة واحدة لصناع القرار بلغة مبسطة غير فنية: من 3 إلى 5 رسائل رئيسية مع الأرقام الداعمة لها، "
               "والإجراءات الموصى بها، والتحفظات. دون مصطلحات فنية أو معادلات."),
    "abstract": ("Write an abstract (150–220 words) for the whole report: objective, data, methods, main quantitative findings, "
                 "conclusion.", "اكتب ملخصاً (150–220 كلمة) للتقرير كله: الهدف، والبيانات، والمنهجية، وأهم النتائج الكمية، والخلاصة."),
}


def _result_context(results: list) -> str:
    out = []
    for r in results:
        d = {"analysis": r.get("title"), "page": r.get("page"), "dataset": r.get("dataset"), "interpretations": [rs.strip_html(x) for x in r.get("insights", [])],
             "content": []}
        for k, c in r.get("sections", []):
            if isinstance(c, pd.DataFrame):
                df = c.reset_index() if not isinstance(c.index, pd.RangeIndex) else c
                d["content"].append({"table": df.head(40).to_dict("records")})
            elif k == "bullets":
                d["content"].append({"references": [rs.strip_html(x) for x in c][:12]})
            else:
                d["content"].append({k: rs.strip_html(c)[:2500]})
        out.append(d)
    return json.dumps(out, ensure_ascii=False, default=lambda v: float(v) if isinstance(v, (np.floating, np.integer)) else str(v))[:90000]


def write(task: str, results: list, lang: str = "en", llm: LLM | None = None, extra: str = "") -> dict:
    """Draft text for `task` from the results. Returns {"text", "source": "ai" | "template", "unverified": [numbers]}."""
    if llm is not None and llm.available:
        prompt = (TASKS[task][1] if lang == "ar" else TASKS[task][0]) + ("\n" + extra if extra else "") + "\n\nRESULTS (JSON):\n" + _result_context(results)
        txt = llm.complete(SYSTEM[lang], [{"role": "user", "content": prompt}], max_tokens=1800)
        return {"text": txt, "source": "ai", "unverified": rs.check_numbers(txt, results)}
    txt = template(task, results, lang)
    return {"text": txt, "source": "template", "unverified": rs.check_numbers(txt, results)}


def template(task: str, results: list, lang: str = "en") -> str:
    ar = lang == "ar"
    if task == "results":
        return "\n\n".join(rs.results_paragraph(r, lang) for r in results if r)
    if task == "methodology":
        out = []
        for r in results:
            for k, c in r.get("sections", []):
                if k == "p":
                    out.append(rs.strip_html(c))
                    break
        return "\n\n".join(out)
    if task in ("policy", "discussion"):
        lines = []
        for r in results:
            ins = [rs.strip_html(x) for x in r.get("insights", [])]
            if ins:
                lines.append(f"• {r['title']}: {ins[0]}")
        head = ("رسائل رئيسية لصانع القرار (من النتائج المحسوبة):" if ar else "Key messages for decision makers (from the computed results):")
        tail = ("تنبيه: هذه صياغة آلية من نتائج التحليل؛ راجعها قبل النشر، وفعّل مساعد الذكاء الاصطناعي لصياغة أكثر سلاسة."
                if ar else "Note: assembled automatically from the analysis outputs — review before publishing; enable the AI assistant for fluent drafting.")
        return "\n".join([head] + lines + ["", tail])
    if task == "abstract":
        titles = ", ".join(r["title"] for r in results)
        first = [rs.strip_html(r["insights"][0]) for r in results if r.get("insights")][:3]
        return ((f"يعرض هذا التقرير نتائج التحليلات التالية: {titles}. " + " ".join(first)) if ar else
                (f"This report presents the following analyses: {titles}. " + " ".join(first)))
    return ""


# ======================================================================================================================
# Model advisor
# ======================================================================================================================
GOALS = {
    "effect": ("Estimate the effect of a policy / programme / treatment", "تقدير أثر سياسة أو برنامج أو معالجة"),
    "relation": ("Explain a variable / estimate elasticities", "تفسير متغير / تقدير المرونات"),
    "forecast": ("Forecast a series", "التنبؤ بسلسلة"),
    "distribution": ("Effects across the distribution (poor vs rich, low vs high yields)", "الآثار عبر التوزيع (الفقراء والأغنياء، الإنتاجية المنخفضة والمرتفعة)"),
    "efficiency": ("Production, costs, efficiency, productivity", "الإنتاج والتكاليف والكفاءة والإنتاجية"),
    "demand": ("Demand, consumption and welfare", "الطلب والاستهلاك والرفاه"),
    "spatial": ("Spatial patterns and spillovers between regions", "الأنماط المكانية والانتشار بين المناطق"),
    "climate": ("Climate, weather and emissions", "المناخ والطقس والانبعاثات"),
    "dynamics": ("Long-run relations and dynamics between series", "العلاقات طويلة الأجل والديناميكيات بين السلاسل"),
    "compare": ("Compare groups / test hypotheses", "مقارنة المجموعات واختبار الفروض"),
}


def profile(df: pd.DataFrame) -> dict:
    """Structure of a dataset: types, identifiers, time, panel, binary treatments, coordinates, cohorts."""
    cols = list(df.columns)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in cols if c not in num]
    lc = {c: c.lower() for c in cols}
    timec = [c for c in cols if lc[c] in ("year", "date", "month", "quarter", "time", "period", "season") or pd.api.types.is_datetime64_any_dtype(df[c])]
    ids = [c for c in cols if c not in timec and (lc[c].endswith("_id") or lc[c] in ("id", "country", "district", "region", "village", "firm", "station", "unit", "state"))
           and 1 < df[c].nunique() < len(df)]
    binary = [c for c in num if set(pd.unique(df[c].dropna())) <= {0, 1} and df[c].nunique() == 2]
    panel = None
    if ids and timec:
        u, t = ids[0], timec[0]
        n_u, n_t = df[u].nunique(), df[t].nunique()
        if n_u > 1 and n_t > 1 and len(df) <= n_u * n_t * 1.01:
            panel = {"unit": u, "time": t, "N": int(n_u), "T": int(n_t), "balanced": bool(len(df.drop_duplicates([u, t])) == n_u * n_t)}
    ts = bool(timec) and not panel and df[timec[0]].is_unique
    coords = [c for c in cols if lc[c] in ("longitude", "latitude", "lon", "lat")]
    cohort = [c for c in num if "first" in lc[c] or "cohort" in lc[c] or lc[c] in ("gvar", "treat_year")]
    daily = any(lc[c] in ("tmin", "tmax") for c in cols)
    shares = [c for c in num if lc[c].startswith("w_") or "share" in lc[c]]
    return {"rows": len(df), "columns": len(cols), "numeric": num, "categorical": cat, "time": timec, "ids": ids, "binary": binary, "panel": panel,
            "time_series": ts, "coordinates": coords, "cohort": cohort, "daily_weather": daily, "shares": shares,
            "missing_%": float(100 * df.isna().mean().mean()) if len(cols) else 0.0}


def advise(df: pd.DataFrame, goal: str, outcome: str | None = None, lang: str = "en") -> list:
    """Rule-based recommendations: [{model, why, where, checks}] ordered by suitability."""
    P = profile(df)
    ar = lang == "ar"
    y = outcome
    yb = y in P["binary"] if y else False
    ycount = bool(y) and y in P["numeric"] and (df[y].dropna() >= 0).all() and np.allclose(df[y].dropna() % 1, 0) and df[y].nunique() > 2 if y else False
    W = {"econ": ("📐 Econometrics Lab", "📐 مختبر الاقتصاد القياسي"), "fc": ("🔮 Forecasting Studio", "🔮 استوديو التنبؤ"),
         "ap": ("🌾 Applied Economics Lab", "🌾 مختبر الاقتصاد التطبيقي"), "im": ("🧭 Impact, Spatial & Climate Lab", "🧭 مختبر الأثر والمكان والمناخ"),
         "inf": ("🧪 Inferential Statistics", "🧪 الإحصاء الاستدلالي")}
    recs = []

    def add(model, why, wing, tab, checks, score):
        recs.append({"model": model[1] if ar else model[0], "why": why[1] if ar else why[0], "where": f"{W[wing][1 if ar else 0]} → {tab[1] if ar else tab[0]}",
                     "checks": checks[1] if ar else checks[0], "score": score})
    pan, ts = P["panel"], P["time_series"]
    if goal == "effect":
        if P["cohort"] and pan:
            add(("Staggered DiD — Callaway-Sant'Anna / Sun-Abraham", "فرق الفروق المتدرج — كالاواي-سانت آنا / سن-أبراهام"),
                ("Units adopt at different dates (a first-treatment column was found); TWFE would be biased.", "الوحدات تبدأ المعالجة في تواريخ مختلفة (وُجد عمود لأول فترة معالجة)، وTWFE متحيز في هذه الحالة."),
                "im", ("DiD & event studies", "فرق الفروق ودراسات الحدث"), ("Pre-trend test, event-study plot, Bacon decomposition.", "اختبار الاتجاهات السابقة، ورسم دراسة الحدث، وتفكيك بيكون."), 10)
        if pan and P["panel"]["N"] <= 60:
            add(("Synthetic control / synthetic DiD", "الضابط التركيبي / فرق الفروق التركيبي"),
                ("Few aggregate units (regions, countries) and one treated unit.", "عدد قليل من الوحدات المجمّعة (أقاليم، دول) ووحدة معالجة واحدة."),
                "im", ("Synthetic control", "الضابط التركيبي"), ("Pre-treatment fit (RMSPE), placebo rank p-value.", "جودة التوفيق قبل المعالجة (RMSPE)، وقيمة p الترتيبية لاختبارات الوهم."), 7)
        if P["binary"] and not pan:
            add(("Randomised trial estimators (difference in means, Lin)", "مقدرات التجربة العشوائية (فرق المتوسطات، لين)"),
                ("A binary treatment in cross-section data; if assignment was random this is the cleanest design.", "معالجة ثنائية في بيانات مقطعية؛ وإن كان التخصيص عشوائياً فهذا أنظف تصميم."),
                "im", ("RCT & IV", "التجارب والمتغيرات الأداتية"), ("Baseline balance and joint orthogonality F test.", "توازن خط الأساس واختبار F للتعامد."), 8)
            add(("Matching / IPW / AIPW", "المطابقة / IPW / AIPW"),
                ("Non-random participation explained by observed characteristics.", "مشاركة غير عشوائية تفسرها خصائص مشاهدة."),
                "im", ("Matching & weighting", "المطابقة والترجيح"), ("Overlap of propensity scores, |SMD| < 0.1 after weighting.", "تداخل درجات الميل، و|SMD| < 0.1 بعد الترجيح."), 7)
            if len(P["binary"]) >= 2:
                add(("Instrumental variables / LATE", "المتغيرات الأداتية / LATE"),
                    ("Two binary variables (e.g. offer and take-up): imperfect compliance.", "متغيران ثنائيان (مثل العرض والمشاركة الفعلية): عدم امتثال كامل."),
                    "im", ("RCT & IV", "التجارب والمتغيرات الأداتية"), ("First-stage F > 10, exclusion restriction argument.", "F للمرحلة الأولى > 10، وتبرير شرط الاستبعاد."), 6)
        add(("Regression discontinuity", "انقطاع الانحدار"),
            ("If eligibility depends on a score cutoff (age, test score, poverty index).", "إذا كانت الأهلية تتحدد بنقطة قطع على درجة (عمر، اختبار، مؤشر فقر)."),
            "im", ("Regression discontinuity", "انقطاع الانحدار"), ("Density (manipulation) test, covariate continuity, bandwidth sensitivity.", "اختبار التلاعب، واستمرارية المتغيرات، وحساسية عرض النطاق."), 4)
    elif goal == "relation":
        if yb:
            add(("Logit / probit with marginal effects", "لوجيت / بروبيت مع الآثار الحدية"), ("The outcome is binary.", "المتغير التابع ثنائي."),
                "econ", ("Qualitative response", "الاستجابة النوعية"), ("Link test, classification table, AME.", "اختبار الربط، وجدول التصنيف، ومتوسط الآثار الحدية."), 10)
        elif ycount:
            add(("Poisson / negative binomial / PPML", "بواسون / ذو الحدين السالب / PPML"), ("The outcome is a non-negative count.", "المتغير التابع عدّ غير سالب."),
                "econ", ("Count & survival", "العدّ والبقاء"), ("Overdispersion test, zero inflation (Vuong).", "اختبار فرط التشتت وتضخم الأصفار (فونج)."), 10)
        if pan:
            add(("Panel FE / RE with Hausman test", "البانل بالآثار الثابتة والعشوائية مع اختبار هاوسمان"),
                (f"Panel structure detected ({pan['N']} units × {pan['T']} periods).", f"بنية بانل ({pan['N']} وحدة × {pan['T']} فترة)."),
                "econ", ("Panel data", "البيانات الطولية"), ("Hausman, Pesaran CD, Wooldridge serial correlation; cluster / Driscoll-Kraay SEs.", "هاوسمان، وCD لبيساران، ووولدريدج؛ وأخطاء عنقودية أو دريسكول-كراي."), 9)
        if ts:
            add(("ARDL bounds test / VECM", "اختبار الحدود ARDL / VECM"), ("Time series: check unit roots and cointegration before regressing levels.", "سلاسل زمنية: افحص جذور الوحدة والتكامل المشترك قبل الانحدار بالمستويات."),
                "econ", ("ARDL", "ARDL"), ("ADF / KPSS, bounds F test, CUSUM.", "ADF وKPSS، واختبار F للحدود، وCUSUM."), 8)
        add(("OLS with log-log form (elasticities) and full diagnostics", "OLS بالصيغة اللوغاريتمية المزدوجة (مرونات) مع التشخيص الكامل"),
            ("Coefficients in log-log form are elasticities, easy to communicate.", "المعاملات في الصيغة اللوغاريتمية المزدوجة مرونات سهلة التوصيل."),
            "econ", ("Regression", "الانحدار"), ("Heteroskedasticity, RESET, VIF, normality; robust SEs.", "عدم تجانس التباين، وRESET، وVIF، والطبيعية؛ وأخطاء متينة."), 6)
    elif goal == "forecast":
        add(("Automatic horse race (ETS, ARIMA, theta, ML) with backtests", "مسابقة تلقائية بين النماذج (ETS وARIMA وثيتا والتعلم الآلي) مع اختبار تاريخي"),
            ("Ranks models by out-of-sample accuracy instead of assuming one.", "ترتّب النماذج حسب الدقة خارج العينة بدلاً من افتراض نموذج واحد."),
            "fc", ("Horse race", "مسابقة النماذج"), ("MASE / RMSE ranking, Diebold-Mariano tests, fan charts.", "ترتيب MASE وRMSE، واختبارات ديبولد-ماريانو، ورسوم المروحة."), 10)
        if P["numeric"] and len(P["numeric"]) > 3:
            add(("Bayesian VAR / nowcasting (bridge, MIDAS)", "VAR البايزي / التنبؤ الآني (الجسر وMIDAS)"), ("Several related indicators are available.", "تتوفر عدة مؤشرات مترابطة."),
                "fc", ("Nowcasting & macro", "التنبؤ الآني والكلي"), ("Pseudo-real-time evaluation.", "التقييم في زمن شبه حقيقي."), 6)
    elif goal == "distribution":
        add(("Quantile regression (process plot + slope-equality test)", "الانحدار الكمّي (منحنى المعاملات + اختبار تساوي الميول)"),
            ("Shows whether effects differ for low and high values of the outcome.", "يبيّن هل تختلف الآثار عند القيم المنخفضة والمرتفعة للنتيجة."),
            "im", ("Quantile regression", "الانحدار الكمّي"), ("Koenker-Bassett slope equality; compare with OLS.", "اختبار كوينكر-باسيت لتساوي الميول؛ والمقارنة مع OLS."), 10)
        if pan:
            add(("MMQR for panels", "MMQR للبانل"), ("Quantile effects with unit fixed effects.", "آثار كمية مع آثار ثابتة للوحدات."), "im", ("Quantile regression", "الانحدار الكمّي"),
                ("Location and scale effects; cluster bootstrap.", "آثار الموقع والمقياس؛ بوتستراب عنقودي."), 8)
    elif goal == "efficiency":
        add(("Production function (Cobb-Douglas / translog) + stochastic frontier", "دالة الإنتاج (كوب-دوجلاس / ترانسلوج) + الحد العشوائي"),
            ("Elasticities, returns to scale and technical efficiency scores.", "المرونات والعائد للسعة ودرجات الكفاءة الفنية."), "ap", ("Production functions / Efficiency", "دوال الإنتاج / الكفاءة"),
            ("Translog vs CD F test; skewness of OLS residuals before SFA.", "اختبار F للترانسلوج مقابل كوب-دوجلاس؛ والتواء بواقي OLS قبل SFA."), 10)
        add(("DEA and Malmquist productivity", "DEA ومؤشر مالمكويست للإنتاجية"), ("Non-parametric efficiency; productivity change over time for panels.", "كفاءة لامعلمية؛ وتغير الإنتاجية عبر الزمن للبانل."),
            "ap", ("Efficiency (SFA, DEA)", "الكفاءة (SFA وDEA)"), ("Returns-to-scale assumption (CRS / VRS).", "فرض العائد للسعة (ثابت / متغير)."), 7)
    elif goal == "demand":
        add(("AIDS / QUAIDS demand system with welfare analysis", "نظام الطلب AIDS / QUAIDS مع تحليل الرفاه"),
            ("Budget shares were found." if P["shares"] else "Budget shares and prices are needed.", "وُجدت حصص إنفاق." if P["shares"] else "يلزم حصص إنفاق وأسعار."),
            "ap", ("Demand systems & welfare", "أنظمة الطلب والرفاه"), ("Homogeneity / symmetry tests, Slutsky negativity.", "اختبارات التجانس والتماثل، وسالبية سلوتسكي."), 10)
    elif goal == "spatial":
        add(("Moran's I / LISA, then SAR / SEM / SDM with spillover impacts", "موران وLISA ثم SAR / SEM / SDM مع الآثار الانتشارية"),
            ("Coordinates found." if P["coordinates"] else "Needs coordinates (longitude / latitude).", "وُجدت إحداثيات." if P["coordinates"] else "يحتاج إلى إحداثيات (خط الطول والعرض)."),
            "im", ("Spatial autocorrelation / regression", "الارتباط / الانحدار المكاني"), ("LM tests choose between lag and error models; report direct and indirect effects.", "اختبارات LM للاختيار بين نموذجي الإبطاء والخطأ؛ واعرض الآثار المباشرة وغير المباشرة."),
            10 if P["coordinates"] else 4)
    elif goal == "climate":
        if P["daily_weather"]:
            add(("Degree days from daily weather", "درجات الأيام من الطقس اليومي"), ("Daily Tmin / Tmax found.", "وُجدت درجات الحرارة اليومية الصغرى والعظمى."), "im",
                ("Climate impacts", "آثار المناخ"), ("Thresholds (8 / 29 °C) suited to the crop.", "عتبات (8 / 29 °م) مناسبة للمحصول."), 10)
        add(("Temperature-response panel (GDD / KDD) with warming scenarios", "بانل استجابة الإنتاجية للحرارة (GDD / KDD) مع سيناريوهات الاحترار"),
            ("Identifies weather effects from year-to-year variation within locations.", "يحدد آثار الطقس من التغيرات السنوية داخل المواقع."), "im", ("Climate impacts", "آثار المناخ"),
            ("Location and year fixed effects, clustered SEs.", "آثار ثابتة للموقع والسنة، وأخطاء عنقودية."), 9 if pan else 5)
        add(("EKC / STIRPAT / decoupling & LMDI", "منحنى كوزنتس / STIRPAT / فك الارتباط وLMDI"), ("For emissions and energy data.", "لبيانات الانبعاثات والطاقة."),
            "im", ("Emissions & damages", "الانبعاثات والأضرار"), ("Turning point inside the data range (Lind-Mehlum).", "وقوع نقطة التحول داخل مدى البيانات (ليند-ميلوم)."), 6)
    elif goal == "dynamics":
        add(("Unit roots → VAR / VECM / ARDL", "جذور الوحدة ← VAR / VECM / ARDL"), ("Determine integration order first, then model short- and long-run dynamics.", "حدد درجة التكامل أولاً ثم نمذج الديناميكيات قصيرة وطويلة الأجل."),
            "econ", ("VAR / VECM / ARDL", "VAR / VECM / ARDL"), ("ADF / KPSS; Johansen or bounds test; impulse responses.", "ADF وKPSS؛ جوهانسن أو اختبار الحدود؛ دوال الاستجابة."), 10 if ts else 5)
    elif goal == "compare":
        add(("Test Advisor (t / ANOVA / non-parametric) with effect sizes", "مستشار الاختبارات (t وANOVA واللامعلمية) مع أحجام الأثر"),
            ("Chooses the test from the data's type and assumptions.", "يختار الاختبار حسب نوع البيانات وفروضها."), "inf", ("Test Advisor", "مستشار الاختبارات"),
            ("Normality and equal-variance checks; post-hoc comparisons.", "فحص الطبيعية وتجانس التباين؛ والمقارنات البعدية."), 10)
    return sorted(recs, key=lambda r: -r["score"])


# ======================================================================================================================
# Offline knowledge base (methodology library of all wings) and Q&A
# ======================================================================================================================
_KB = {}


def knowledge_base(lang: str = "en") -> list:
    if lang in _KB:
        return _KB[lang]
    docs = []
    import importlib
    for modname in ("applied_report", "impact_report", "panel_report", "fcst_report", "mts_report", "ts_report"):
        try:
            mod = importlib.import_module(f"analytics.{modname}")
        except Exception:  # noqa: BLE001
            continue
        kinds = list(getattr(mod, "TEXT", {}).keys())
        if not kinds:
            try:
                src = inspect.getsource(mod.methodology)
                kinds = sorted(set(re.findall(r'kind (?:==|in) \(?"([a-z_0-9]+)"', src)))
            except Exception:  # noqa: BLE001
                kinds = []
        for k in kinds:
            try:
                txt, refs = mod.methodology(k, lang)
            except Exception:  # noqa: BLE001
                continue
            if txt:
                docs.append({"id": f"{modname}:{k}", "text": rs.strip_html(txt), "refs": refs})
    _KB[lang] = docs
    return docs


def _norm_ar(s: str) -> str:
    s = re.sub("[إأآا]", "ا", s)
    s = s.replace("ة", "ه").replace("ى", "ي")
    return re.sub(r"[ً-ْ]", "", s)


def retrieve(question: str, lang: str = "en", k: int = 3) -> list:
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = knowledge_base(lang) + (knowledge_base("en") if lang == "ar" else [])
    if not docs:
        return []
    corpus = [_norm_ar(d["text"]) for d in docs]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
    M = vec.fit_transform(corpus)
    q = vec.transform([_norm_ar(question)])
    sims = (M @ q.T).toarray().ravel()
    idx = np.argsort(-sims)[:k]
    return [dict(docs[i], score=float(sims[i])) for i in idx if sims[i] > 0.02]


def answer(question: str, lang: str = "en", llm: LLM | None = None, history: list | None = None, results: list | None = None,
           data_profile: dict | None = None) -> dict:
    """Answer a researcher's question. With an LLM: grounded on the retrieved library passages, the data profile and the
    session results. Offline: the best-matching methodology passages with their references."""
    hits = retrieve(question, lang)
    if llm is not None and llm.available:
        ctx = {"library": [{"id": h["id"], "text": h["text"][:1800], "refs": h["refs"][:6]} for h in hits],
               "data_profile": {k: v for k, v in (data_profile or {}).items() if k not in ("numeric", "categorical")} if data_profile else None,
               "session_results": json.loads(_result_context(results))[:6] if results else []}
        sysm = SYSTEM[lang] + (" أجب عن سؤال الباحث مستعيناً بالسياق المرفق، واذكر أين ينفّذ ذلك في المنصة إن أمكن، وأشر إلى المراجع."
                               if lang == "ar" else " Answer the researcher's question using the context; say where in the platform to run it when relevant, and cite references.")
        msgs = list(history or [])[-8:] + [{"role": "user", "content": f"CONTEXT (JSON): {json.dumps(ctx, ensure_ascii=False, default=str)[:60000]}\n\nQUESTION: {question}"}]
        txt = llm.complete(sysm, msgs, max_tokens=1500, temperature=0.3)
        return {"text": txt, "source": "ai", "hits": hits}
    if not hits:
        msg = ("لم أجد في مكتبة المنصة فقرة تطابق سؤالك. جرّب صياغة أخرى (مثل: «متى أستخدم نموذج الآثار الثابتة؟»)، أو فعّل مساعد الذكاء الاصطناعي بمفتاح API."
               if lang == "ar" else "No passage in the platform library matches your question. Try rephrasing, or enable the AI assistant with an API key.")
        return {"text": msg, "source": "library", "hits": []}
    head = "من مكتبة المنهجيات في المنصة:" if lang == "ar" else "From the platform's methodology library:"
    body = "\n\n".join(f"{h['text']}" for h in hits[:2])
    refs = sorted({r for h in hits[:2] for r in h["refs"]})[:6]
    return {"text": head + "\n\n" + body + ("\n\n" + ("المراجع:" if lang == "ar" else "References:") + "\n" + "\n".join("• " + r for r in refs) if refs else ""),
            "source": "library", "hits": hits}
