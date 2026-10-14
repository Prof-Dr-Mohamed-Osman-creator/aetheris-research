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
    st.info("Data Hub is under construction. Upload & international data engines will be added next."
            if get_current_language() == "en"
            else "مركز البيانات قيد الإنشاء. سيتم إضافة رفع الملفات ومحركات البيانات الدولية تالياً.")

elif page == t("descriptive"):
    st.header(t("descriptive"))
    if not has_data():
        st.warning(t("no_data"))
    else:
        st.success("Data is available in memory. Descriptive tools will be implemented here.")

elif page == t("inferential"):
    st.header(t("inferential"))
    if not has_data():
        st.warning(t("no_data"))
    else:
        st.info("Inferential statistics module coming soon.")

elif page == t("econometrics"):
    st.header(t("econometrics"))
    if not has_data():
        st.warning(t("no_data"))
    else:
        st.info("Econometrics Lab coming soon.")

elif page == t("operations"):
    st.header(t("operations"))
    st.info("Operations Research module coming soon.")

elif page == t("reports"):
    st.header(t("reports"))
    st.info("Report Studio coming soon.")

elif page == t("academy"):
    st.header(t("academy"))
    st.info("Academy module coming soon.")

elif page == t("community"):
    st.header(t("community"))
    st.info("Community space coming soon.")
