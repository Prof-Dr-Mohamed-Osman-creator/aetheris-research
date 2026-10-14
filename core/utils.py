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


def t(key: str) -> str:
    """Shortcut for getting translated text based on current language."""
    lang = get_current_language()
    return get_text(lang, key)


def inject_custom_css():
    """Inject basic custom CSS for a clean professional look."""
    st.markdown("""
        <style>
        .main-title {
            text-align: center;
            color: #1a365d;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }
        .main-subtitle {
            text-align: center;
            color: #4a5568;
            font-size: 1.1rem;
            margin-bottom: 1.5rem;
        }
        .stButton>button {
            border-radius: 6px;
        }
        </style>
    """, unsafe_allow_html=True)
