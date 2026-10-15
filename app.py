# ==========================================
# Aetheris Research
# Global Research Platform for Economists & Decision-Makers
# ==========================================

import streamlit as st
import pandas as pd

from config.settings import SYSTEM_NAME, SYSTEM_VERSION, LANGUAGE_KEY, DEFAULT_LANGUAGE
from config.languages import get_text
from core.memory import init_memory, has_data, memory_status_text, clear_memory
from core.utils import get_current_language, set_language, t, inject_custom_css
from data.upload import render_upload_section

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(
    page_title="Aetheris Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------
# Initialize
# ------------------------------------------
init_memory()
inject_custom_css()

if LANGUAGE_KEY not in st.session_state:
    st.session_state[LANGUAGE_KEY] = DEFAULT_LANGUAGE

# ------------------------------------------
# Sidebar - Language + Navigation
# ------------------------------------------
with st.sidebar:
    st.markdown(f"### {t('sidebar_header')}")

    # Language Switcher
    lang_options = {"English": "en", "العربية": "ar"}
    current_lang_label = "English" if get_current_language() == "en" else "العربية"
    selected_label = st.selectbox(
        t("language"),
        options=list(lang_options.keys()),
        index=0 if current_lang_label == "English" else 1
    )
    set_language(lang_options[selected_label])

    st.markdown("---")

    # Wings Menu
    wings = [
        t("home"),
        t("data_hub"),
        t("descriptive"),
        t("inferential"),
        t("econometrics"),
        t("operations"),
        t("reports"),
        t("academy"),
        t("community"),
    ]

    page = st.radio(t("menu_title"), wings, label_visibility="collapsed")

    st.markdown("---")

    # Smart Memory Status
    st.caption(f"**{t('smart_memory')}**")
    st.caption(memory_status_text(get_current_language()))

    if has_data():
        if st.button(t("clear_memory"), use_container_width=True):
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

# ------------------------------------------
# Wings Routing
# ------------------------------------------
if page == t("home"):
    st.info(f"👈 {t('welcome_msg')}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 📥 Data Hub")
        st.write("Upload your files or fetch international data." if get_current_language() == "en"
                 else "ارفع ملفاتك أو اجلب بيانات دولية.")
    with col2:
        st.markdown("#### 📊 Analytics")
        st.write("Descriptive, Inferential, Econometrics & Operations Research." if get_current_language() == "en"
                 else "إحصاء وصفي واستدلالي، اقتصاد قياسي، وبحوث عمليات.")
    with col3:
        st.markdown("#### 📄 Reports")
        st.write("Generate professional reports with AI assistance." if get_current_language() == "en"
                 else "توليد تقارير احترافية بمساعدة الذكاء الاصطناعي.")

elif page == t("data_hub"):
    st.header(t("data_hub"))

    # Tabs for future expansion (My Data / Global Data / Data Basket)
    tab1, tab2, tab3 = st.tabs([
        "📁 My Data" if get_current_language() == "en" else "📁 بياناتي",
        "🌍 Global Data" if get_current_language() == "en" else "🌍 بيانات دولية",
        "🛒 Data Basket" if get_current_language() == "en" else "🛒 سلة البيانات"
    ])

    with tab1:
        render_upload_section()

    with tab2:
        st.info(
            "Global Data section will be implemented next. You will be able to select indicators, countries, and years from international sources."
            if get_current_language() == "en" else
            "قسم البيانات الدولية سيتم تنفيذه لاحقاً. ستتمكن من اختيار المؤشرات والدول والسنوات من المصادر الدولية."
        )

    with tab3:
        st.info(
            "Data Basket will allow combining your uploaded data with international sources."
            if get_current_language() == "en" else
            "سلة البيانات ستتيح دمج بياناتك المرفوعة مع المصادر الدولية."
        )

elif page == t("descriptive"):
    st.header(t("descriptive"))
    if not has_data():
        st.warning(t("no_data"))
    else:
        st.success("Data is available in memory. Descriptive tools will be implemented here."
                   if get_current_language() == "en" else
                   "البيانات متوفرة في الذاكرة. سيتم تنفيذ أدوات الإحصاء الوصفي هنا.")

elif page == t("inferential"):
    st.header(t("inferential"))
    if not has_data():
        st.warning(t("no_data"))
    else:
        st.info("Inferential statistics module coming soon."
                if get_current_language() == "en" else
                "وحدة الإحصاء الاستدلالي قريباً.")

elif page == t("econometrics"):
    st.header(t("econometrics"))
    if not has_data():
        st.warning(t("no_data"))
    else:
        st.info("Econometrics Lab coming soon."
                if get_current_language() == "en" else
                "مختبر الاقتصاد القياسي قريباً.")

elif page == t("operations"):
    st.header(t("operations"))
    st.info("Operations Research module coming soon."
            if get_current_language() == "en" else
            "وحدة بحوث العمليات قريباً.")

elif page == t("reports"):
    st.header(t("reports"))
    st.info("Report Studio coming soon."
            if get_current_language() == "en" else
            "استوديو التقارير قريباً.")

elif page == t("academy"):
    st.header(t("academy"))
    st.info("Academy module coming soon."
            if get_current_language() == "en" else
            "وحدة الأكاديمية قريباً.")

elif page == t("community"):
    st.header(t("community"))
    st.info("Community space coming soon."
            if get_current_language() == "en" else
            "مساحة المجتمع قريباً.")
