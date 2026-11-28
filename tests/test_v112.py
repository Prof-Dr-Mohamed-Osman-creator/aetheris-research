"""v1.12 — Report Studio, AI assistant and decision dashboard: document builders, the number checker, grounded writing,
the model advisor, the offline Q&A library, early-warning signals (checked against direct computations) and the
automatic capture of results from the analysis wings."""
import io
import json
import warnings
import zipfile

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from analytics import ai_assistant as A
from analytics import dashboard as D
from analytics import report_studio as R
from data import samples as smp

warnings.filterwarnings("ignore")

REF = "Cobb, C. W. & Douglas, P. H. (1928). A theory of production. American Economic Review, 18(1), 139–165."


def _result(title="Production function", lang="en"):
    tab = pd.DataFrame({"Term": ["const", "ln_land", "ln_labour"], "Estimate": [1.2, 0.5158, 0.1903], "Std. Err.": [0.19, 0.0517, 0.0321],
                        "p-value": [1e-9, 2e-12, 0.08]})
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 3, 2], mode="lines+markers", name="a"))
    return {"key": f"econometrics|{title}", "page": "econometrics", "title": title, "lang": lang,
            "sections": [("h", "Methodology"), ("p", "OLS on logs."), ("table", tab), ("bullets", [REF, "Not a reference"])],
            "insights": ["<b>Returns to scale</b> = 0.706 (decreasing)."], "figures": [{"json": fig.to_json(), "name": "residual_plot"}]}


def _items(lang="en"):
    r1, r2 = _result("Production function"), _result("Second model")
    r2["sections"][3] = ("bullets", [REF])
    return [{"kind": "heading", "title": "Introduction"}, {"kind": "text", "title": "", "text": "Intro text."},
            {"kind": "result", "title": r1["title"], "result": r1, "include_tables": True, "include_figures": True, "include_insights": True, "text": ""},
            {"kind": "result", "title": r2["title"], "result": r2, "include_tables": True, "include_figures": False, "include_insights": False, "text": ""}]


def test_assemble_numbers_tables_figures_and_dedupes_references():
    m = R.assemble(_items(), {"title": "T"}, "en")
    caps = [b[1] for b in m["blocks"] if b[0] in ("table", "figure")]
    assert caps == ["Table 1. Production function", "Figure 1. residual plot", "Table 2. Second model"]
    assert m["references"] == [REF]                                    # merged across results
    assert ("bullets", ["Not a reference"]) in m["blocks"]
    assert [b[1] for b in m["blocks"] if b[0] == "h1"] == ["1. Introduction", "2. Production function", "3. Second model"]
    t1 = next(b[2] for b in m["blocks"] if b[0] == "table")
    assert t1["p-value"].tolist() == ["< 0.001", "< 0.001", "0.080"]


def test_exports_all_formats_english_and_arabic():
    from docx import Document
    import pdfplumber
    for lang in ("en", "ar"):
        m = R.assemble(_items(lang), {"title": "تقرير" if lang == "ar" else "Report", "author": "A. Author", "abstract": "Short abstract."}, lang)
        doc = Document(io.BytesIO(R.to_docx(m)))
        assert len(doc.tables) == 2 and len(doc.inline_shapes) == 1
        heads = [p.text for p in doc.paragraphs if p.style.name.startswith("Heading")]
        assert ("References" if lang == "en" else "المراجع") in heads
        if lang == "ar":
            assert doc.paragraphs[0]._p.pPr.xml.count("bidi") >= 1
        pdf = R.to_pdf(m)
        assert pdf[:4] == b"%PDF"
        txt = "".join(pg.extract_text() or "" for pg in pdfplumber.open(io.BytesIO(pdf)).pages)
        assert "Cobb" in txt
        z = zipfile.ZipFile(io.BytesIO(R.to_latex_zip(m)))
        tex = z.read("main.tex").decode()
        assert r"\toprule" in tex and r"\includegraphics" in tex and "figures/fig1.png" in z.namelist()
        assert ("polyglossia" in tex) == (lang == "ar")
        html = R.to_html(m)
        assert "Plotly.newPlot" in html and (f"dir='{'rtl' if lang == 'ar' else 'ltr'}'" in html)
        assert "| " in R.to_markdown(m)


def test_matplotlib_figure_fallback(monkeypatch):
    monkeypatch.setenv("AETHERIS_NO_KALEIDO", "1")
    R._PNG_CACHE.clear()
    fig = go.Figure([go.Bar(x=["a", "b"], y=[1, 2]), go.Scatter(x=["a", "b"], y=[2, 1], mode="lines")])
    png = R.fig_png(fig.to_json())
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_number_checker():
    res = [_result()]
    assert R.check_numbers("The land elasticity is 0.516 (SE 0.052), i.e. 51.6 %; RTS 0.706; p < 0.001; 3 inputs; year 2020.", res) == []
    assert R.check_numbers("The elasticity is 0.61 and RTS 1.25.", res) == ["0.61", "1.25"]
    neg = _result()
    neg["sections"][2][1].loc[1, "Estimate"] = -0.5158
    assert R.check_numbers("Estimate −0.5158 reported; it fell by 0.5158.", [neg]) == []     # unicode minus; magnitude in words
    assert R.check_numbers("Estimate −0.5158 reported.", [_result()]) == ["-0.5158"]          # a sign error is flagged


def test_results_paragraph_uses_only_computed_numbers():
    r = _result()
    for lang in ("en", "ar"):
        txt = R.results_paragraph(r, lang)
        assert "0.5158" in txt and R.check_numbers(txt, [r]) == []
    assert "ln_labour" in R.results_paragraph(r, "en").split("not statistically significant")[0]


class _FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    class _M:
        def __init__(self, outer):
            self.o = outer

        def create(self, **kw):
            self.o.calls.append(kw)

            class B:
                text = self.o.reply

            class Resp:
                content = [B()]
            return Resp()

    @property
    def messages(self):
        return _FakeLLM._M(self)


def test_ai_write_is_grounded_and_checked():
    fake = _FakeLLM("Land has an elasticity of 0.516 (SE 0.0517), while labour's 0.33 is invented.")
    out = A.write("results", [_result()], "en", A.LLM(client=fake))
    assert out["source"] == "ai" and out["unverified"] == ["0.33"]
    kw = fake.calls[0]
    assert "NEVER invent numbers" in kw["system"] and "0.5158" in kw["messages"][0]["content"]
    off = A.write("results", [_result()], "ar")
    assert off["source"] == "template" and off["unverified"] == []
    for t in A.TASKS:
        assert isinstance(A.template(t, [_result()], "ar"), str)


def test_model_advisor():
    first = lambda df, g: A.advise(df, g, lang="en")[0]["model"]  # noqa: E731
    assert "Callaway" in first(smp.staggered_did_demo(), "effect")
    assert "Synthetic" in first(smp.scm_regions_demo(), "effect")
    assert "horse race" in first(smp.monthly_forecast_demo(), "forecast")
    assert "Moran" in first(smp.spatial_districts_demo(), "spatial")
    assert "Degree days" in first(smp.daily_weather_demo(), "climate")
    rec = A.advise(smp.credit_program_demo(), "relation", "participant", "ar")[0]
    assert "لوجيت" in rec["model"]
    P = A.profile(smp.farm_efficiency_demo())
    assert P["panel"] == {"unit": "farm_id", "time": "year", "N": 150, "T": 6, "balanced": True}


def test_offline_qa_library():
    hit = lambda q, l: [h["id"] for h in A.retrieve(q, l)]  # noqa: E731
    assert "panel_report:static" in hit("متى أستخدم اختبار هاوسمان بين الآثار الثابتة والعشوائية؟", "ar")
    assert hit("how is the synthetic control estimated", "en")[0] == "impact_report:synth"
    ans = A.answer("ما هي درجات أيام النمو والضرر؟", "ar")
    assert ans["source"] == "library" and "المراجع" in ans["text"]


# ---------------------------------------------------------------- dashboard
def test_zscore_signal_matches_direct_computation():
    rng = np.random.default_rng(0)
    s = pd.Series(np.r_[rng.normal(0, 1, 60), 3.5], index=pd.RangeIndex(61))
    sg = D.signal(s, "z", "high")
    hist = s.iloc[:-1]
    assert np.isclose(sg.signal.iloc[-1], (3.5 - hist.mean()) / hist.std(ddof=1))
    assert sg.status.iloc[-1] == "red" and sg.status.iloc[:8].eq("na").all()
    low = D.signal(-s, "z", "low")
    assert low.status.iloc[-1] == "red"
    fx = D.signal(s, "fixed", amber=1.0, red=2.0)
    assert fx.status.iloc[-1] == "red" and (fx.level[s < 1] == 0).all()


def test_composite_and_noise_to_signal_counts():
    idx = pd.RangeIndex(10)
    sig = {"a": pd.DataFrame({"level": [0, 1, 2, 2, np.nan, 0, 0, 1, 2, 0]}, index=idx), "b": pd.DataFrame({"level": [0, 0, 2, 1, 1, 0, 0, 0, 0, 0]}, index=idx)}
    c = D.composite(sig)
    assert np.isclose(c["index"].iloc[2], 100) and np.isclose(c["index"].iloc[4], 50) and c.status.iloc[2] == "red"
    # toy: crisis starts at t = 10; x is high at t = 7, 8 (inside the 3-period window) and at t = 2 (false alarm)
    x = pd.Series([0, 0, 5, 0, 0, 0, 0, 5, 5, 0, 0, 0], dtype=float)
    cr = pd.Series([0] * 10 + [1, 1])
    r = D.noise_to_signal(x, cr, horizon=3, grid=[80.0])
    row = r["table"].iloc[0]
    assert (row.A, row.B, row.C, row.D) == (2, 1, 1, 6)
    assert np.isclose(row.NtS, (1 / 7) / (2 / 3)) and r["crises"] == 1


def test_macro_demo_signals_and_scenario():
    df = smp.macro_dashboard_demo()
    res = D.noise_to_signal(D.series(df, "date", "reserves_months_imports"), D.series(df, "date", "crisis"), 12, "low")
    assert res["crises"] == 3 and res["best"]["NtS"] < 0.5
    k = D.kpi(D.series(df, "date", "inflation_pct"))
    assert np.isclose(k["pct"], 100 * (k["latest"] / k["previous"] - 1))
    m = D.scenario_model(df, "inflation_pct", ["exchange_rate"], "loglog")
    sc = D.scenario(m, {"exchange_rate": 10})
    assert np.isclose(sc["change %"], 100 * (1.1 ** m["model"].params["exchange_rate"] - 1))
    fw = D.forecast_warning(D.series(df, "date", "inflation_pct"), 6, threshold=15, bad="high")["table"]
    assert fw["P(breach)"].between(0, 1).all() and (fw["upper 95"] >= fw["forecast"]).all()


# ---------------------------------------------------------------- automatic capture from the wings
def test_results_are_captured_from_analysis_wings():
    from streamlit.testing.v1 import AppTest
    df = smp.rdd_scholarship_demo()
    from pathlib import Path
    at = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=600)
    at.session_state["aetheris_language"] = "en"
    at.session_state["aetheris_datasets"] = {"d": {"df": df, "source": "demo"}}
    at.session_state["aetheris_memory"] = df
    at.session_state["aetheris_active_name"] = "d"
    at.session_state["memory_shape"] = df.shape
    at.session_state["memory_source"] = "demo"
    at.session_state["nav"] = "impact"
    at.run()
    assert not at.exception
    res = at.session_state["_rs_results"]
    rd = res["impact|Regression discontinuity"]
    assert rd["insights"] and rd["figures"] and any(isinstance(c, pd.DataFrame) for _, c in rd["sections"])
    at.session_state["nav"] = "reports"
    at.run()
    assert not at.exception and any(b.key == "rs_add_impact|Regression discontinuity" for b in at.button)
    json.dumps([k for k in res])
