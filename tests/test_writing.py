"""v1.12.2 — research writing workspace: source parsing (APA lines, Arabic references, BibTeX, files), citation checking
against the supplied sources, grounded prompts, the offline outline and the report integration."""
import io

from analytics import report_studio as rs
from analytics import writing as W

SRC = """1. Schlenker, W. & Roberts, M. J. (2009). Nonlinear temperature effects indicate severe damages. PNAS, 106(37), 15594–15598.
سعيد، أحمد (2018). أثر التغيرات المناخية على إنتاجية القمح في مصر. المجلة المصرية للاقتصاد الزراعي، 28(3)، 1-20.
@article{burke2015, author = {Burke, Marshall and Hsiang, Solomon and Miguel, Edward}, title = {Global non-linear effect of temperature}, journal = {Nature}, year = {2015},
}
Some note without a year that is long enough to look like a line of text."""


def test_reference_detection():
    refs = W.reference_list(SRC)
    assert len(refs) == 3
    assert any(r.startswith("Burke, Marshall") and "(2015)" in r for r in refs)
    assert any(r.startswith("سعيد") for r in refs)


def test_citation_check_flags_only_unsupported():
    refs = W.reference_list(SRC)
    draft = ("Heat harms yields (Schlenker & Roberts, 2009); Burke et al. (2015) agree. Others disagree (Smith, 2020; Lee 2017). "
             "ووجد سعيد (2018) أثراً سالباً، بينما أشار (حسن، 2019) إلى غير ذلك. Water matters [reference needed]. See Table 2.")
    c = W.check_citations(draft, refs)
    assert sorted(c["unmatched"]) == sorted(["Smith, 2020", "Lee 2017", "حسن، 2019"])
    assert c["missing_markers"] == 1 and c["uncited_sources"] == []
    c2 = W.check_citations("Only (Schlenker & Roberts, 2009).", refs)
    assert len(c2["uncited_sources"]) == 2


def test_prompts_are_grounded():
    for lang in ("ar", "en"):
        sysm = W.system_prompt(lang)
        assert W.MISSING[lang] in sysm
        req = W.build_request("litreview", lang, "T", "Q", "Egypt", SRC, None, 500, "group by region")
        assert "Schlenker" in req and "group by region" in req and "500" in req
        empty = W.build_request("theory", lang, "", "", "", "", None, 300, "")
        assert W.MISSING[lang] in empty


def test_outline_offline_uses_sources_only():
    o = W.outline("litreview", "ar", "المناخ والقمح", "ما أثر الحرارة؟", W.reference_list(SRC))
    assert "المحور الأول" in o and "Schlenker" in o and "[مرجع مطلوب]" not in o
    assert W.outline("theory", "en", "", "", []).count("## ") >= 4


def test_split_references_and_file_extraction():
    body, refs = W.split_references("Para one.\n\nPara two.\n\nReferences used:\n- A, B. (2001). X.\n- C. (2002). Y.")
    assert body == "Para one.\n\nPara two." and refs == ["A, B. (2001). X.", "C. (2002). Y."]
    from docx import Document
    d = Document()
    d.add_paragraph("Said, A. (2018). Climate change and wheat productivity in Egypt. EJAE, 28(3), 1-20.")
    buf = io.BytesIO()
    d.save(buf)
    assert "Said, A. (2018)" in W.extract_text("study.docx", buf.getvalue())
    assert W.extract_text("refs.bib", b"@book{x, title={T}, year={2001}}").startswith("@book")


def test_report_integration_merges_references_and_subheadings():
    item = {"kind": "text", "title": "Literature review", "text": "## Theme 1\n\nFirst paragraph (Schlenker & Roberts, 2009).\n\nSecond paragraph.",
            "references": ["Schlenker, W. & Roberts, M. J. (2009). Nonlinear temperature effects. PNAS."]}
    m = rs.assemble([item], {"title": "T"}, "en")
    assert [b[0] for b in m["blocks"]] == ["h2", "h2", "p", "p"]
    assert m["references"] == item["references"]
