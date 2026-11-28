# ==========================================
# Aetheris Research - Free AI conversation (v1.12.3)
# An open conversation with the configured AI model (Gemini, Groq, OpenRouter, ChatGPT, Claude …) that answers from its
# own knowledge — explanations, critiques, ideas, drafting, translation — optionally given the dataset profile, the
# session's results or the writing-workspace draft as context. Because answers from a model's memory can contain
# plausible but non-existent references, any reference in an answer can be checked against Crossref (the DOI registry).
# ==========================================

from __future__ import annotations

import difflib
import json
import re

SYSTEM = {
    "ar": ("أنت مساعد بحثي خبير في الاقتصاد والاقتصاد القياسي والإحصاء والاقتصاد الزراعي والسياسات العامة داخل منصة Aetheris Research. "
           "أجب بالعربية الفصحى الواضحة ما لم يطلب الباحث لغة أخرى، وبعمق أكاديمي مع أمثلة عملية عند الحاجة. يمكنك الاعتماد على معرفتك العامة. "
           "قواعد الأمانة العلمية: ميّز بين ما هو راسخ وما هو اجتهاد أو غير مؤكد؛ لا تخترع أرقاماً أو إحصاءات أو اقتباسات؛ وإذا ذكرت مراجع فاذكر "
           "فقط ما أنت واثق من وجوده، مع المؤلف والسنة والعنوان والمجلة، ونبّه إلى ضرورة التحقق منها؛ وإن لم تكن متأكداً فقل ذلك صراحة. "
           "وإذا كان السؤال يتطلب تحليلاً للبيانات فاقترح أين ينفّذ في المنصة (الأجنحة والتبويبات)."),
    "en": ("You are an expert research assistant in economics, econometrics, statistics, agricultural economics and public policy inside the "
           "Aetheris Research platform. Answer in clear English unless asked otherwise, with academic depth and practical examples where useful. "
           "You may rely on your general knowledge. Scientific-integrity rules: distinguish established knowledge from judgement or uncertainty; "
           "never invent numbers, statistics or quotations; if you cite references, give only ones you are confident exist (author, year, title, "
           "journal) and remind the user to verify them; say plainly when you are not sure. If the question needs data analysis, suggest where "
           "to run it in the platform (wings and tabs)."),
}


def build_messages(history: list, question: str, context: dict | None, lang: str, max_turns: int = 12) -> list:
    """Chat messages for the model: the last turns plus the new question, with optional context attached to it."""
    msgs = [m for m in history if m.get("role") in ("user", "assistant")][-2 * max_turns:]
    q = question
    if context:
        label = "السياق الذي أرفقه الباحث (JSON)" if lang == "ar" else "CONTEXT ATTACHED BY THE RESEARCHER (JSON)"
        q = f"{label}:\n{json.dumps(context, ensure_ascii=False, default=str)[:60000]}\n\n{question}"
    return msgs + [{"role": "user", "content": q}]


# ======================================================================================================================
# Reference verification against Crossref
# ======================================================================================================================
_YEAR = re.compile(r"\(?((?:1[89]|20)\d\d)[a-z]?\)?")


def find_references(text: str) -> list:
    """Reference-like lines in an answer: author(s) + year + a title-like part (APA-ish, numbered or bulleted lists)."""
    out = []
    for line in text.splitlines():
        s = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", line).strip().strip("*_")
        if len(s) < 30 or len(s) > 500 or not _YEAR.search(s):
            continue
        if re.search(r"[A-Za-z؀-ۿ]{3,}.*\(?(?:1[89]|20)\d\d[a-z]?\)?[.,:]?\s+\S+.{12,}", s):
            out.append(s)
    return list(dict.fromkeys(out))[:20]


def _title_guess(ref: str) -> str:
    m = re.search(r"\(?(?:1[89]|20)\d\d[a-z]?\)?[.,:]?\s*(.+)", ref)
    t = m.group(1) if m else ref
    t = re.split(r"\.\s+[A-Z][A-Za-z&\s]+,?\s*\d|\.\s+(?:In|Journal|Review|Quarterly|Economic|American)\b", t)[0]
    return t.strip(" .\"'“”")[:250]


def verify_reference(ref: str, session=None, mailto: str = "aetheris-research@users.noreply.github.com") -> dict:
    """Query Crossref's bibliographic search and compare the best hit's title and year with the reference.
    status: 'verified' (title similarity ≥ 0.85 and year matches), 'partial' (≥ 0.6), 'not found'."""
    import requests
    sess = session or requests
    try:
        r = sess.get("https://api.crossref.org/works", params={"query.bibliographic": ref[:400], "rows": 3, "mailto": mailto,
                                                               "select": "DOI,title,author,issued,container-title"}, timeout=20)
        if r.status_code >= 400:
            return {"reference": ref, "status": "error", "detail": f"HTTP {r.status_code}"}
        items = r.json().get("message", {}).get("items", [])
    except Exception as e:  # noqa: BLE001
        return {"reference": ref, "status": "error", "detail": str(e)[:120]}
    want_t = _title_guess(ref).lower()
    ym = _YEAR.search(ref)
    want_y = ym.group(1) if ym else None
    best = None
    for it in items:
        title = (it.get("title") or [""])[0]
        sim = difflib.SequenceMatcher(None, want_t, title.lower()).ratio() if title else 0.0
        if want_t and title and (want_t in title.lower() or title.lower() in want_t):
            sim = max(sim, 0.9)
        y = None
        try:
            y = str(it.get("issued", {}).get("date-parts", [[None]])[0][0])
        except (IndexError, TypeError):
            pass
        if best is None or sim > best["similarity"]:
            authors = ", ".join(a.get("family", "") for a in (it.get("author") or [])[:3])
            best = {"similarity": round(sim, 2), "doi": it.get("DOI"), "title": title, "year": y, "authors": authors,
                    "journal": (it.get("container-title") or [""])[0]}
    if not best:
        return {"reference": ref, "status": "not found"}
    year_ok = want_y is None or best["year"] == want_y
    status = "verified" if best["similarity"] >= 0.85 and year_ok else ("partial" if best["similarity"] >= 0.6 else "not found")
    return {"reference": ref, "status": status, **best}


def to_markdown(history: list, lang: str = "ar") -> str:
    out = ["# " + ("محادثة مع المساعد البحثي" if lang == "ar" else "Conversation with the research assistant"), ""]
    for m in history:
        who = ("الباحث" if lang == "ar" else "Researcher") if m["role"] == "user" else ("المساعد" if lang == "ar" else "Assistant")
        out += [f"**{who}:**", "", m.get("display", m["content"]), ""]
    return "\n".join(out)
