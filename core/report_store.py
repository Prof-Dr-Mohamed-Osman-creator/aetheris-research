# ==========================================
# Aetheris Research - Report store (v1.12, Catalogue §8)
# Every analysis that builds a Word report (reg_report.to_docx) is captured automatically as a "result" together with
# the interpretation boxes (core.utils.insight) and the charts (st.plotly_chart) produced in the same script run on the
# same page. The Report Studio then lets the researcher pick results, order them and export one report; the AI
# assistant and the decision dashboard read the same store.
# ==========================================

from __future__ import annotations

import time
import uuid

import streamlit as st

RESULTS = "_rs_results"          # {key: result}  (latest version of each analysis)
ITEMS = "_rs_items"              # ordered list of report items chosen by the user
PENDING = "_rs_pending"          # {"run": id, "insights": [...], "figures": [...]}
RUN = "_rs_run"
SKIP_PAGES = {"reports", "assistant", "dashboard", "home", "data_hub"}
MAX_RESULTS = 60
MAX_FIGS = 6


def _ss():
    try:
        return st.session_state
    except Exception:  # noqa: BLE001  (bare mode / tests without a script run context)
        return None


def new_run():
    """Called once at the top of every script run (app.py)."""
    ss = _ss()
    if ss is None:
        return
    ss[RUN] = ss.get(RUN, 0) + 1
    ss[PENDING] = {"run": ss[RUN], "insights": [], "figures": []}


def _page():
    ss = _ss()
    return (ss.get("nav") if ss is not None else None) or "analysis"


def _active():
    ss = _ss()
    return ss is not None and _page() not in SKIP_PAGES and RUN in ss


def _pending():
    ss = _ss()
    p = ss.get(PENDING)
    if not p or p.get("run") != ss.get(RUN):
        p = ss[PENDING] = {"run": ss.get(RUN), "insights": [], "figures": []}
    return p


def capture_insight(text: str):
    if not _active():
        return
    p = _pending()
    if text and text not in p["insights"]:
        p["insights"].append(str(text))


def capture_figure(fig, name: str | None = None):
    if not _active():
        return
    try:
        js = fig.to_json()
    except Exception:  # noqa: BLE001
        return
    p = _pending()
    title = name or (fig.layout.title.text if getattr(fig.layout, "title", None) and fig.layout.title.text else None)
    p["figures"].append({"json": js, "name": title or f"figure {len(p['figures']) + 1}"})


def capture_sections(title: str, sections: list):
    """Register (or refresh) a result: the sections of a Word report plus the insights / figures pending in this run."""
    if not _active():
        return
    ss = _ss()
    p = _pending()
    page = _page()
    key = f"{page}|{title}"
    res = ss.setdefault(RESULTS, {})
    figs = p["figures"][-MAX_FIGS:]
    res[key] = {"key": key, "page": page, "title": title, "sections": list(sections), "insights": list(p["insights"]), "figures": figs,
                "time": time.time(), "lang": ss.get("aetheris_language", "en"), "dataset": ss.get("aetheris_active_name")}
    p["insights"], p["figures"] = [], []
    if len(res) > MAX_RESULTS:
        for k in sorted(res, key=lambda k_: res[k_]["time"])[:len(res) - MAX_RESULTS]:
            del res[k]


def put_result(r: dict):
    ss = _ss()
    if ss is not None:
        ss.setdefault(RESULTS, {})[r["key"]] = r


def results() -> dict:
    ss = _ss()
    return dict(ss.get(RESULTS, {})) if ss is not None else {}


def items() -> list:
    ss = _ss()
    return ss.setdefault(ITEMS, []) if ss is not None else []


def add_item(kind: str, title: str, **payload) -> dict:
    it = {"id": uuid.uuid4().hex[:8], "kind": kind, "title": title, **payload}
    items().append(it)
    return it


def add_result(key: str) -> dict | None:
    r = results().get(key)
    if r is None:
        return None
    return add_item("result", r["title"], result=r, include_figures=True, include_insights=True, include_tables=True, text="")


def move(i: int, d: int):
    L = items()
    j = i + d
    if 0 <= i < len(L) and 0 <= j < len(L):
        L[i], L[j] = L[j], L[i]


def remove(i: int):
    L = items()
    if 0 <= i < len(L):
        L.pop(i)


def install_hooks():
    """Wrap st.plotly_chart once so charts drawn by any wing are captured with the analysis of the same run."""
    if getattr(st.plotly_chart, "_aetheris_wrapped", False):
        return
    orig = st.plotly_chart

    def wrapped(figure_or_data, *a, **k):
        try:
            name = None
            cfg = k.get("config") or {}
            name = (cfg.get("toImageButtonOptions") or {}).get("filename")
            capture_figure(figure_or_data, name)
        except Exception:  # noqa: BLE001
            pass
        return orig(figure_or_data, *a, **k)
    wrapped._aetheris_wrapped = True
    st.plotly_chart = wrapped
