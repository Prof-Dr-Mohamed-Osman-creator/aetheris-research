# ==========================================
# Aetheris Research
# Global Research Platform for Economists & Decision-Makers
# ==========================================

import streamlit as st

from config.settings import SYSTEM_VERSION, LANGUAGE_KEY, DEFAULT_LANGUAGE
from core.memory import init_memory, has_data, memory_status_text, clear_memory, set_memory
from core.utils import get_current_language, set_language, t, tr, inject_custom_css

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(
    page_title="Aetheris Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------
# Initialize (language first, so the whole page renders in one language)
# ------------------------------------------
if LANGUAGE_KEY not in st.session_state:
    st.session_state[LANGUAGE_KEY] = DEFAULT_LANGUAGE
init_memory()
from core import report_store as _rs  # noqa: E402  (v1.12 result capture for the Report Studio)
_rs.install_hooks()
_rs.new_run()
try:  # v1.12.3: unlock the platform AI key on a remembered device
    from analytics.ai_settings_ui import restore_device_code as _restore_ai, after_unlock as _after_ai
    _restore_ai()
    _after_ai()
except Exception:  # noqa: BLE001
    pass

LANG_OPTIONS = {"en": "English", "ar": "العربية"}


def _on_lang_change():
    set_language(st.session_state["lang_select"])


WINGS = ["home", "data_hub", "descriptive", "inferential", "econometrics", "forecasting", "applied", "impact",
         "reports", "assistant", "dashboard", "operations", "academy", "community"]
WING_ICONS = {"home": "🏠", "data_hub": "📥", "descriptive": "📋", "inferential": "🧪",
              "econometrics": "📐", "forecasting": "🔮", "applied": "🌾", "impact": "🧭", "assistant": "🤖", "dashboard": "🏛️", "operations": "⚙️", "reports": "📄", "academy": "🎓", "community": "💬"}

# ------------------------------------------
# Sidebar - Language + Navigation
# ------------------------------------------
with st.sidebar:
    st.selectbox(
        "Language / اللغة",
        options=list(LANG_OPTIONS.keys()),
        format_func=lambda c: LANG_OPTIONS[c],
        index=list(LANG_OPTIONS.keys()).index(get_current_language()),
        key="lang_select",
        on_change=_on_lang_change,
    )
    inject_custom_css()

    st.markdown(f"### {t('sidebar_header')}")
    # Navigation keys are language-independent, so switching language keeps the current page
    page = st.radio(t("menu_title"), WINGS, key="nav",
                    format_func=lambda w: f"{WING_ICONS[w]}  {t(w)}",
                    label_visibility="collapsed")

    st.markdown("---")
    st.caption(f"**{t('smart_memory')}**")
    st.caption(memory_status_text(get_current_language()))
    if has_data():
        if st.button(t("clear_memory"), width="stretch"):
            clear_memory()
            st.rerun()

    st.markdown("---")
    st.caption(t("designed_by"))
    st.caption(f"v{SYSTEM_VERSION}")

# ------------------------------------------
# Main Header
# ------------------------------------------
st.markdown(f"<h1 class='main-title'>◈ {t('sys_title')}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='main-subtitle'>{t('sys_subtitle')}</p>", unsafe_allow_html=True)
st.markdown("---")


def _need_data() -> bool:
    if not has_data():
        st.warning(t("no_data"))
        return True
    return False


def _coming_soon(en: str, ar: str):
    st.info(tr(en, ar))
    st.caption(tr("See ROADMAP.md in the repository for the full development plan.",
                  "راجع ملف ROADMAP.md في المستودع للاطلاع على خطة التطوير الكاملة."))


# ------------------------------------------
# Wings Routing
# ------------------------------------------
if page == "home":
    st.info(f"👈 {t('welcome_msg')}")
    cards = [
        ("📥", tr("Data Hub", "مركز البيانات"),
         tr("Upload your files or pull World Bank & FRED data, then merge them in the Data Basket.",
            "ارفع ملفاتك أو اجلب بيانات البنك الدولي وFRED، ثم ادمجها في سلة البيانات.")),
        ("📋", tr("Descriptive Statistics", "الإحصاء الوصفي"),
         tr("Full summary tables, normality tests, correlation matrices, growth rates — with automatic interpretation.",
            "جداول وصفية كاملة، واختبارات الطبيعية، ومصفوفات الارتباط، ومعدلات النمو، مع قراءة تحليلية تلقائية.")),
        ("🧪", tr("Inferential Statistics", "الإحصاء الاستدلالي"),
         tr("Test Advisor, 30+ parametric & non-parametric tests, post-hoc, effect sizes, power & sample size.",
            "مستشار الاختبارات، وأكثر من 30 اختباراً معلمياً ولامعلمياً، والمقارنات البعدية، وأحجام الأثر، والقوة وحجم العينة.")),
        ("📐", tr("Econometrics Lab", "مختبر الاقتصاد القياسي"),
         tr("Regression with 5 functional forms, OLS/WLS/FGLS/GLS/robust/RLS/NLS, robust SEs, 25+ diagnostics, "
            "IV / 2SLS / GMM / 3SLS / SUR, logit-probit-multinomial-ordered, Tobit / Heckman / hurdles, count and survival "
            "models, unit roots and Box-Jenkins ARIMA / SARIMA / ARFIMA forecasting, elasticities, journal tables, policy what-if "
            "VAR / VECM, ARDL / NARDL, GARCH / DCC, panel data (FE / RE, panel unit roots and cointegration, GMM, PMG, CCE) and Python/R/Stata code.",
            "الانحدار بخمس صور دالية، وOLS وWLS وFGLS وGLS والمتين والمقيد وغير الخطي، والأخطاء المتينة، وأكثر من 25 "
            "اختباراً تشخيصياً، والمتغيرات الأداتية والأنظمة، ونماذج الاستجابة النوعية والمراقَبة والاختيار، والعدّ والبقاء، "
            "وجذور الوحدة ومنهجية بوكس-جنكنز للتنبؤ (ARIMA وSARIMA وARFIMA)، والمرونات، وجداول المجلات، ومحاكاة «ماذا لو»، "
            "ونماذج VAR وVECM وARDL وNARDL وGARCH وDCC، والبيانات الطولية (الآثار الثابتة والعشوائية وجذور الوحدة والتكامل وGMM وPMG وCCE)، وكود Python وR وStata.")),
        ("🔮", tr("Forecasting Studio", "استوديو التنبؤ"),
         tr("Automatic model horse-race with rolling-origin backtests and Diebold-Mariano tests; ETS, theta, ARIMA / SARIMAX, state "
            "space, BSTS and growth curves; nowcasting (bridge, MIDAS, dynamic factors), Bayesian VAR and local projections; HAR-RV, "
            "GARCH and VaR backtests; machine learning and Prophet-style models; combinations, fan charts and scenarios.",
            "مسابقة تلقائية بين النماذج مع اختبار تاريخي بالأصل المتدحرج واختبارات ديبولد-ماريانو؛ وETS وثيتا وARIMA/SARIMAX والفضاء "
            "الحالتي وBSTS ومنحنيات النمو؛ والتنبؤ الآني (الجسر وMIDAS والعوامل الديناميكية)، وVAR البايزي والإسقاطات المحلية؛ وHAR-RV "
            "وGARCH واختبار VaR؛ والتعلم الآلي ونماذج نمط Prophet؛ والتجميع ورسوم المروحة والسيناريوهات.")),
        ("🌾", tr("Applied Economics Lab", "مختبر الاقتصاد التطبيقي"),
         tr("Production functions (Cobb-Douglas, translog, CES, transcendental…) with elasticities, substitution, stages and isoquants; "
            "yield response and economic optimum; cost curves, scale and scope; translog cost / profit systems; stochastic frontiers, "
            "meta-frontier, DEA and Malmquist; Olley-Pakes / Levinsohn-Petrin / ACF productivity; Nerlove supply response and cobweb; "
            "demand, Engel curves, LES / AIDS / QUAIDS / Rotterdam with welfare analysis; consumption functions.",
            "دوال الإنتاج (كوب-دوجلاس، والترانسلوج، وCES، والترانسندنتالية…) مع المرونات والإحلال ومراحل الإنتاج والناتج المتساوي؛ واستجابة "
            "المحصول والمعدل الاقتصادي الأمثل؛ ومنحنيات التكاليف والسعة والنطاق؛ وأنظمة التكاليف والربح ترانسلوج؛ والحدود العشوائية والحد "
            "الفوقي وDEA ومالمكويست؛ والإنتاجية بطرق أولي-باكس وليفينسون-بترين وACF؛ واستجابة العرض لنيرلوف وبيت العنكبوت؛ والطلب "
            "ومنحنيات إنجل وأنظمة LES وAIDS وQUAIDS وروتردام مع تحليل الرفاه؛ ودوال الاستهلاك.")),
        ("🧭", tr("Impact, Spatial & Climate Lab", "مختبر الأثر والمكان والمناخ"),
         tr("Quantile regression (process plots, slope tests, MMQR, cross-quantilogram); spatial econometrics (weights, Moran / LISA "
            "maps, LM tests, SAR / SEM / SDM / SAC with spillover impacts, GWR, spatial panels); policy evaluation (RCTs, IV / LATE, "
            "staggered DiD with Callaway-Sant'Anna and Sun-Abraham, synthetic control and SDiD, regression discontinuity, matching, "
            "IPW and AIPW); climate economics (degree days, temperature-yield panels, Ricardian, BHM damages and SCC, STIRPAT, EKC, "
            "decoupling and LMDI).",
            "الانحدار الكمّي (منحنيات المعاملات واختبارات الميول وMMQR ومخطط الكميات المتقاطع)؛ والاقتصاد القياسي المكاني (الأوزان وخرائط "
            "موران وLISA واختبارات LM ونماذج SAR وSEM وSDM وSAC مع الآثار الانتشارية والانحدار الموزون جغرافياً والبانل المكاني)؛ وتقييم "
            "السياسات (التجارب العشوائية والمتغيرات الأداتية وفرق الفروق المتدرج لكالاواي-سانت آنا وسن-أبراهام والضابط التركيبي وSDiD "
            "وانقطاع الانحدار والمطابقة والترجيح)؛ واقتصاد المناخ (درجات الأيام ونماذج الحرارة والإنتاجية والمنهج الريكاردي ودالة الضرر "
            "وتكلفة الكربون وSTIRPAT ومنحنى كوزنتس البيئي وفك الارتباط وتفكيك LMDI).")),
        ("⚙️", tr("Operations Research", "بحوث العمليات"),
         tr("Linear programming, transportation, efficiency (DEA) — in development.",
            "البرمجة الخطية، ومسائل النقل، وتحليل الكفاءة (DEA) — قيد التطوير.")),
        ("📄", tr("Report Studio", "استوديو التقارير"),
         tr("Every analysis is captured with its tables, charts, interpretation and references; arrange them into one report with numbered APA tables "
            "and figures and export to Word, PDF, LaTeX, HTML or Markdown — in Arabic (RTL) or English.",
            "كل تحليل يُلتقط بجداوله ورسومه وتفسيره ومراجعه؛ رتّبها في تقرير واحد بجداول وأشكال مرقمة بنمط APA وصدّره إلى Word وPDF وLaTeX "
            "وHTML وMarkdown — بالعربية (من اليمين لليسار) أو الإنجليزية.")),
        ("🤖", tr("AI Research Assistant", "المساعد البحثي الذكي"),
         tr("A research-writing workspace (introduction, literature review, theoretical framework, hypotheses …) grounded only in your own sources, "
            "model advice, results and policy briefs written only from the computed numbers (every number and citation is checked), and Arabic Q&A — "
            "with Claude, ChatGPT, Gemini or free models, or offline.",
            "مساحة للكتابة البحثية (المقدمة، والدراسات السابقة، والإطار النظري، والفروض …) تستند إلى مصادرك فقط، ونصائح النماذج، وكتابة النتائج "
            "وموجز السياسات من الأرقام المحسوبة فقط (مع فحص كل رقم واستشهاد)، والإجابة بالعربية — مع Claude أو ChatGPT أو Gemini أو نماذج مجانية، أو دون اتصال.")),
        ("🏛️", tr("Decision Dashboard", "لوحة صانع القرار"),
         tr("Key findings in plain language, KPI cards, early-warning traffic lights with a composite index, the signals approach for crisis prediction, "
            "forecast-based breach probabilities and what-if scenarios.",
            "أهم النتائج بلغة مبسطة، وبطاقات المؤشرات، وأضواء الإنذار المبكر مع مؤشر مركب، ومنهج الإشارات للتنبؤ بالأزمات، واحتمالات تجاوز العتبات "
            "من التنبؤ، وسيناريوهات «ماذا لو».")),
        ("🎓", tr("Academy", "الأكاديمية"),
         tr("Courses, guided exercises and certificates in statistics & econometrics (planned).",
            "دورات وتمارين موجّهة وشهادات في الإحصاء والاقتصاد القياسي (مخطط لها).")),
    ]
    for row in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (icon, title, body) in zip(cols, cards[row:row + 3]):
            with col:
                st.markdown(f"#### {icon} {title}")
                st.write(body)

    st.markdown("---")
    st.markdown("#### " + tr("🚀 Try it now", "🚀 جرّب المنصة الآن"))
    st.write(tr("Load a synthetic demo dataset (climate & crop yields in 8 Egyptian desert governorates, 2000–2023) "
                "and explore the Descriptive Statistics wing.",
                "حمّل بيانات تجريبية مُولَّدة (المناخ وإنتاجية المحاصيل في 8 محافظات صحراوية مصرية، 2000–2023) "
                "واستكشف جناح الإحصاء الوصفي."))
    if st.button(tr("Load demo dataset", "تحميل البيانات التجريبية"), key="home_demo"):
        from data.samples import climate_crop_demo
        set_memory(climate_crop_demo(), source="demo (synthetic)", name="Demo – climate & crops (synthetic)")
        st.success(tr("Demo data loaded. Open 'Descriptive Statistics' from the sidebar.",
                      "تم تحميل البيانات التجريبية. افتح «الإحصاء الوصفي» من القائمة الجانبية."))
    st.caption(tr("⚠️ Demo data is simulated for training purposes and must not be cited as real statistics.",
                  "⚠️ البيانات التجريبية مُحاكاة لأغراض التدريب، ولا يجوز الاستشهاد بها كإحصاءات حقيقية."))

elif page == "data_hub":
    from data.upload import render_upload_section
    from data.global_data import render_global_data
    from data.basket import render_basket

    st.header(t("data_hub"))
    tab1, tab2, tab3 = st.tabs([
        tr("📁 My Data", "📁 بياناتي"),
        tr("🌍 Global Data", "🌍 بيانات دولية"),
        tr("🛒 Data Basket", "🛒 سلة البيانات"),
    ])
    with tab1:
        render_upload_section()
    with tab2:
        render_global_data()
    with tab3:
        render_basket()

elif page == "descriptive":
    st.header(t("descriptive"))
    if not _need_data():
        from analytics.descriptive_ui import render_descriptive
        render_descriptive()

elif page == "inferential":
    st.header(t("inferential"))
    from analytics.inferential_ui import render_inferential
    render_inferential()

elif page == "econometrics":
    st.header(t("econometrics"))
    from analytics.econometrics_ui import render_econometrics
    render_econometrics()

elif page == "forecasting":
    st.header(t("forecasting"))
    from analytics.forecast_ui import render_forecasting
    render_forecasting()

elif page == "applied":
    st.header(t("applied"))
    from analytics.applied_ui import render_applied
    render_applied()

elif page == "impact":
    st.header(t("impact"))
    from analytics.impact_ui import render_impact
    render_impact()

elif page == "operations":
    st.header(t("operations"))
    _coming_soon("Linear & integer programming, transportation/assignment, DEA efficiency — in development.",
                 "البرمجة الخطية والصحيحة، ومسائل النقل والتخصيص، وتحليل كفاءة DEA — قيد التطوير.")

elif page == "reports":
    st.header(t("reports"))
    from analytics.reports_ui import render_reports
    render_reports()

elif page == "assistant":
    st.header(t("assistant"))
    from analytics.assistant_ui import render_assistant
    render_assistant()

elif page == "dashboard":
    st.header(t("dashboard"))
    from analytics.dashboard_ui import render_dashboard
    render_dashboard()

elif page == "academy":
    st.header(t("academy"))
    _coming_soon("Courses, training paths and certificates — planned.",
                 "الدورات والمسارات التدريبية والشهادات — مخطط لها.")

elif page == "community":
    st.header(t("community"))
    _coming_soon("Research community, Q&A and collaboration space — planned.",
                 "مجتمع الباحثين، والأسئلة والأجوبة، ومساحة التعاون — مخطط لها.")
