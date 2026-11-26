# ==========================================
# Aetheris Research - Smart Memory System
# ------------------------------------------
# Holds every dataset the researcher loads (uploads, World Bank,
# FRED, merged baskets) and one "active" dataset that all analysis
# wings read from.
# ==========================================

from __future__ import annotations

import streamlit as st
import pandas as pd
from config.settings import MEMORY_KEY, PROJECTS_KEY, CURRENT_PROJECT_KEY

DATASETS_KEY = "aetheris_datasets"       # {name: {"df": DataFrame, "source": str}}
ACTIVE_NAME_KEY = "aetheris_active_name"


def init_memory():
    """Initialize session state for smart memory and projects."""
    defaults = {
        MEMORY_KEY: None,
        PROJECTS_KEY: {},
        CURRENT_PROJECT_KEY: None,
        DATASETS_KEY: {},
        ACTIVE_NAME_KEY: None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ------------------------------------------
# Active dataset (used by every analysis wing)
# ------------------------------------------
def get_memory() -> pd.DataFrame | None:
    """Return the active dataset."""
    return st.session_state.get(MEMORY_KEY)


def set_memory(df: pd.DataFrame, source: str = "upload", name: str | None = None):
    """Store a DataFrame as the active dataset and register it in the library."""
    name = name or source
    st.session_state.setdefault(DATASETS_KEY, {})[name] = {"df": df, "source": source}
    _activate(name)


def _activate(name: str):
    entry = st.session_state[DATASETS_KEY][name]
    df = entry["df"]
    st.session_state[MEMORY_KEY] = df
    st.session_state[ACTIVE_NAME_KEY] = name
    st.session_state["memory_source"] = entry["source"]
    st.session_state["memory_shape"] = df.shape
    st.session_state["memory_columns"] = list(df.columns)


def clear_memory():
    """Clear all datasets."""
    st.session_state[MEMORY_KEY] = None
    st.session_state[DATASETS_KEY] = {}
    st.session_state[ACTIVE_NAME_KEY] = None
    st.session_state["memory_source"] = None
    st.session_state["memory_shape"] = None
    st.session_state["memory_columns"] = None


def has_data() -> bool:
    """Check if smart memory contains an active dataset."""
    mem = get_memory()
    return mem is not None and isinstance(mem, pd.DataFrame) and not mem.empty


# ------------------------------------------
# Dataset library
# ------------------------------------------
def list_datasets() -> list[str]:
    return list(st.session_state.get(DATASETS_KEY, {}).keys())


def get_dataset(name: str) -> pd.DataFrame | None:
    entry = st.session_state.get(DATASETS_KEY, {}).get(name)
    return entry["df"] if entry else None


def active_name() -> str | None:
    return st.session_state.get(ACTIVE_NAME_KEY)


def activate_dataset(name: str):
    if name in st.session_state.get(DATASETS_KEY, {}):
        _activate(name)


def remove_dataset(name: str):
    datasets = st.session_state.get(DATASETS_KEY, {})
    datasets.pop(name, None)
    if active_name() == name:
        remaining = list(datasets.keys())
        if remaining:
            _activate(remaining[-1])
        else:
            clear_memory()


def memory_status_text(lang: str = "en") -> str:
    """Return a short status text about current memory."""
    if has_data():
        shape = st.session_state.get("memory_shape", (0, 0))
        name = active_name() or st.session_state.get("memory_source", "unknown")
        n = len(list_datasets())
        if lang == "ar":
            return (f"النشطة: {name} — {shape[0]} صف × {shape[1]} عمود"
                    f" | عدد البيانات في الذاكرة: {n}")
        return f"Active: {name} — {shape[0]} rows × {shape[1]} cols | datasets in memory: {n}"
    if lang == "ar":
        return "الذاكرة فارغة"
    return "Memory is empty"
