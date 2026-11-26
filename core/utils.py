# ==========================================
# Aetheris Research - Utility Functions
# ==========================================

import streamlit as st
from config.settings import LANGUAGE_KEY, DEFAULT_LANGUAGE
from config.languages import get_text


def get_current_language() -> str:
    """Return the currently selected language code."""
    return st.session_state.get(LANGUAGE_KEY, DEFAULT_LANGUAGE)


def set_language(lang_code: str):
    """Set the current language."""
    st.session_state[LANGUAGE_KEY] = lang_code


def is_ar() -> bool:
    return get_current_language() == "ar"


def t(key: str) -> str:
    """Shortcut for getting translated text based on current language."""
    lang = get_current_language()
    return get_text(lang, key)


def tr(en: str, ar: str) -> str:
    """Inline bilingual text: returns the string for the active language."""
    return ar if is_ar() else en


def inject_custom_css():
    """Inject custom CSS for a clean professional look (+ RTL for Arabic)."""
    rtl = """
        [data-testid="stMain"] .block-container,
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            direction: rtl;
            text-align: right;
        }
        [data-testid="stMain"] [data-testid="stDataFrame"],
        [data-testid="stMain"] .js-plotly-plot,
        [data-testid="stMain"] code, [data-testid="stMain"] pre {
            direction: ltr;
            text-align: left;
        }
    """ if is_ar() else ""

    st.markdown(f"""
        <style>
        .main-title {{
            text-align: center;
            color: #1a365d;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .main-subtitle {{
            text-align: center;
            color: #4a5568;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }}
        .stButton>button {{
            border-radius: 6px;
        }}
        .insight-box {{
            background: #f0f5fb;
            border-left: 4px solid #1a365d;
            border-radius: 6px;
            padding: 0.8rem 1rem;
            margin: 0.4rem 0 1rem 0;
            line-height: 1.7;
        }}
        {rtl}
        </style>
    """, unsafe_allow_html=True)


def insight(text: str):
    """Render an interpretation box (academic-style reading of results)."""
    st.markdown(f"<div class='insight-box'>{text}</div>", unsafe_allow_html=True)


def csv_download(df, file_name: str, label: str | None = None, key: str | None = None):
    """Download button for a DataFrame as UTF-8 CSV (Excel-friendly with BOM)."""
    st.download_button(
        label or tr("⬇️ Download CSV", "⬇️ تحميل CSV"),
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=file_name,
        mime="text/csv",
        key=key,
    )
