# ==========================================
# Aetheris Research - Smart Memory System
# ==========================================

import streamlit as st
import pandas as pd
from config.settings import MEMORY_KEY, PROJECTS_KEY, CURRENT_PROJECT_KEY


def init_memory():
    """Initialize session state for smart memory and projects."""
    if MEMORY_KEY not in st.session_state:
        st.session_state[MEMORY_KEY] = None
    if PROJECTS_KEY not in st.session_state:
        st.session_state[PROJECTS_KEY] = {}
    if CURRENT_PROJECT_KEY not in st.session_state:
        st.session_state[CURRENT_PROJECT_KEY] = None


def get_memory() -> pd.DataFrame | None:
    """Return the current data stored in smart memory."""
    return st.session_state.get(MEMORY_KEY)


def set_memory(df: pd.DataFrame, source: str = "upload"):
    """Store a DataFrame in smart memory with metadata."""
    st.session_state[MEMORY_KEY] = df
    st.session_state["memory_source"] = source
    st.session_state["memory_shape"] = df.shape
    st.session_state["memory_columns"] = list(df.columns)


def clear_memory():
    """Clear smart memory."""
    st.session_state[MEMORY_KEY] = None
    st.session_state["memory_source"] = None
    st.session_state["memory_shape"] = None
    st.session_state["memory_columns"] = None


def has_data() -> bool:
    """Check if smart memory contains data."""
    mem = get_memory()
    return mem is not None and isinstance(mem, pd.DataFrame) and not mem.empty


def memory_status_text(lang: str = "en") -> str:
    """Return a short status text about current memory."""
    if has_data():
        shape = st.session_state.get("memory_shape", (0, 0))
        source = st.session_state.get("memory_source", "unknown")
        if lang == "ar":
            return f"الذاكرة تحتوي على بيانات: {shape[0]} صف × {shape[1]} عمود (المصدر: {source})"
        return f"Memory loaded: {shape[0]} rows × {shape[1]} columns (source: {source})"
    if lang == "ar":
        return "الذاكرة فارغة"
    return "Memory is empty"
