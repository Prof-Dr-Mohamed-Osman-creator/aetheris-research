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

LANG_OPTIONS = {"en": "English", "ar": "العربية"}


def _on_lang_change():
    set_language(st.session_state["lang_select"])


WINGS = ["home", "data_hub", "descriptive", "inferential", "econometrics",
         "operations", "reports", "academy", "community"]
WING_ICONS = {"home": "🏠", "data_hub": "📥", "descriptive": "📋", "inferential": "🧪",
              "econometrics": "📐", "operations": "⚙️", "reports": "📄", "academy": "🎓", "community": "💬"}

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
        ("⚙️", tr("Operations Research", "بحوث العمليات"),
         tr("Linear programming, transportation, efficiency (DEA) — in development.",
            "البرمجة الخطية، ومسائل النقل، وتحليل الكفاءة (DEA) — قيد التطوير.")),
        ("📄", tr("Report Studio + AI", "استوديو التقارير + الذكاء الاصطناعي"),
         tr("AI-assisted academic reports in Arabic & English (in development).",
            "تقارير أكاديمية بالعربية والإنجليزية بمساعدة الذكاء الاصطناعي (قيد التطوير).")),
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

elif page == "operations":
    st.header(t("operations"))
    _coming_soon("Linear & integer programming, transportation/assignment, DEA efficiency — in development.",
                 "البرمجة الخطية والصحيحة، ومسائل النقل والتخصيص، وتحليل كفاءة DEA — قيد التطوير.")

elif page == "reports":
    st.header(t("reports"))
    _coming_soon("Report Studio with AI writing assistant (Word/PDF export, APA tables) — in development.",
                 "استوديو التقارير مع مساعد كتابة بالذكاء الاصطناعي (تصدير Word وPDF وجداول بنمط APA) — قيد التطوير.")

elif page == "academy":
    st.header(t("academy"))
    _coming_soon("Courses, training paths and certificates — planned.",
                 "الدورات والمسارات التدريبية والشهادات — مخطط لها.")

elif page == "community":
    st.header(t("community"))
    _coming_soon("Research community, Q&A and collaboration space — planned.",
                 "مجتمع الباحثين، والأسئلة والأجوبة، ومساحة التعاون — مخطط لها.")
