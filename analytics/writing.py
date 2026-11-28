# ==========================================
# Aetheris Research - Research writing workspace (v1.12.2)
# Drafting of the narrative parts of a thesis / paper — introduction, problem and objectives, literature review,
# theoretical framework, conceptual model and hypotheses, methodology, discussion, conclusions — in a conversation with
# the AI, grounded ONLY in the sources the researcher supplies (pasted notes, abstracts, references, or uploaded PDF /
# Word / BibTeX / text files) and, optionally, the results computed in the platform. The model must never invent a
# reference: where support is missing it writes a "[reference needed]" marker. Every in-text citation of a draft is
# checked against the supplied source list.
# ==========================================

from __future__ import annotations

import io
import re

SECTIONS = {
    "intro": ("Introduction & background", "المقدمة والخلفية"),
    "problem": ("Research problem, questions & objectives", "مشكلة البحث وأسئلته وأهدافه"),
    "litreview": ("Literature review (previous studies)", "الدراسات السابقة (مراجعة الأدبيات)"),
    "theory": ("Theoretical framework", "الإطار النظري"),
    "hypotheses": ("Conceptual model & hypotheses", "النموذج المفاهيمي والفروض"),
    "method": ("Data & methodology", "البيانات والمنهجية"),
    "discussion": ("Discussion of results", "مناقشة النتائج"),
    "conclusion": ("Conclusions & recommendations", "الخلاصة والتوصيات"),
    "free": ("Free request", "طلب حر"),
}

GUIDE = {
    "intro": ("Context and importance of the topic; the gap; the aim; the structure of the study.", "سياق الموضوع وأهميته؛ والفجوة البحثية؛ والهدف؛ وتنظيم الدراسة."),
    "problem": ("A precise problem statement, 2–4 research questions, matching objectives and the study's contribution.",
                "صياغة دقيقة لمشكلة البحث، و2–4 أسئلة بحثية، وأهداف مقابلة لها، وإسهام الدراسة."),
    "litreview": ("Thematic synthesis (not a list of summaries): group studies by theme / method / region, compare findings, show "
                  "agreements and contradictions, and end with the gap this study fills.",
                  "تركيب موضوعي (لا قائمة ملخصات): صنّف الدراسات حسب الموضوع أو المنهج أو المنطقة، وقارن النتائج، وبيّن الاتفاق والتعارض، "
                  "واختم بالفجوة التي تسدها الدراسة."),
    "theory": ("The theories behind the study, their core propositions, how they link the variables, and why they fit the context.",
               "النظريات التي تستند إليها الدراسة، وفرضياتها الأساسية، وكيف تربط المتغيرات، ولماذا تناسب السياق."),
    "hypotheses": ("A conceptual model linking the variables and testable hypotheses (H1, H2 …) with their theoretical justification.",
                   "نموذج مفاهيمي يربط المتغيرات، وفروض قابلة للاختبار (ف1، ف2 …) مع تبريرها النظري."),
    "method": ("Data sources, variables and measurement, the econometric model and estimator, diagnostics and inference.",
               "مصادر البيانات، والمتغيرات وقياسها، والنموذج القياسي والمقدِّر، والاختبارات التشخيصية والاستدلال."),
    "discussion": ("Interpret the computed results, compare them with the previous studies supplied, explain differences.",
                   "فسّر النتائج المحسوبة، وقارنها بالدراسات السابقة المقدمة، واشرح أسباب الاختلاف."),
    "conclusion": ("Main findings, policy recommendations tied to the findings, limitations and future research.",
                   "أهم النتائج، والتوصيات المرتبطة بها، وحدود الدراسة، والبحوث المستقبلية."),
    "free": ("Anything you need: a paragraph, an abstract, a cover letter, a rewrite …", "أي شيء تحتاجه: فقرة، ملخص، خطاب للمجلة، إعادة صياغة …"),
}

MISSING = {"en": "[reference needed]", "ar": "[مرجع مطلوب]"}


# ======================================================================================================================
# Sources: text extraction and reference parsing
# ======================================================================================================================
def extract_text(name: str, data: bytes, max_chars: int = 60000) -> str:
    n = name.lower()
    try:
        if n.endswith(".pdf"):
            from pypdf import PdfReader
            r = PdfReader(io.BytesIO(data))
            txt = "\n".join((p.extract_text() or "") for p in r.pages[:60])
        elif n.endswith(".docx"):
            from docx import Document
            txt = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
        else:
            txt = data.decode("utf-8", errors="ignore")
    except Exception as e:  # noqa: BLE001
        return f"[could not read {name}: {e}]"
    txt = re.sub(r"[ \t]+", " ", txt)
    return txt[:max_chars]


def parse_bibtex(txt: str) -> list:
    out = []
    for m in re.finditer(r"@\w+\s*\{[^,]*,(.*?)(?:\n\s*\}|\}\s*(?:\n|$))", txt, re.S):
        body = m.group(1)

        def f(k):
            mm = re.search(rf"\b{k}\s*=\s*[{{\"](.+?)[}}\"]\s*(?:,|\n|$)", body, re.S | re.I)
            return re.sub(r"[{}]", "", mm.group(1)).strip() if mm else ""
        au, yr, ti, jo = f("author"), f("year"), f("title"), f("journal") or f("booktitle") or f("publisher")
        if au or ti:
            authors = [a.strip() for a in au.split(" and ") if a.strip()]
            line = (f"{', '.join(authors)} ({yr}). {ti}. {jo}." if authors else f"{ti} ({yr}). {jo}.").replace("..", ".").replace(" . ", " ")
            out.append(line.strip())
    return out


def reference_list(sources_text: str) -> list:
    """Reference-like lines from the supplied sources (APA-style lines, numbered lists, BibTeX)."""
    refs = []
    if "@" in sources_text and "{" in sources_text:
        refs += parse_bibtex(sources_text)
    for line in sources_text.splitlines():
        s = line.strip().lstrip("-•*0123456789.[]) ").strip()
        if 25 <= len(s) <= 600 and re.search(r"\((?:1[89]|20)\d\d[a-z]?\)|\b(?:1[89]|20)\d\d[a-z]?\.", s):
            refs.append(s)
    return list(dict.fromkeys(refs))


def _key(author: str, year: str) -> tuple:
    a = re.sub(r"[^A-Za-z0-9ء-ي ]", " ", author).strip().split()   # Arabic letters only (drops "،")
    return (a[0].lower() if a else "", year)


def source_keys(refs: list) -> set:
    keys = set()
    for r in refs:
        m = re.match(r"\s*([^,(]+)", r)
        y = re.search(r"((?:1[89]|20)\d\d)", r)
        if m and y:
            keys.add(_key(m.group(1), y.group(1)))
            # also accept the second author's surname for "A & B" style citations
    return keys


CITE_RX = re.compile(r"\(([^()]*?(?:1[89]|20)\d\d[a-z]?[^()]*?)\)|([A-Z][A-Za-z'\-]+|[؀-ۿ]{2,})(?:\s+(?:et al\.|وآخرون|and|&|و)\s*[A-Za-z؀-ۿ'\-]*)?\s*\((\d{4})[a-z]?\)")


def citations(text: str) -> list:
    """In-text author–year citations: (Smith, 2019; Lee & Kim 2020), Smith (2019), (سعيد، 2018), سعيد (2018)."""
    out = []
    for m in CITE_RX.finditer(text):
        if m.group(1):
            for part in re.split(r";|؛", m.group(1)):
                mm = re.search(r"([A-Za-z؀-ۿ][A-Za-z؀-ۿ'\-]+)[^0-9]*?((?:1[89]|20)\d\d)", part)
                if mm:
                    out.append((mm.group(1), mm.group(2), part.strip()))
        elif m.group(2):
            out.append((m.group(2), m.group(3), m.group(0).strip()))
    return out


def check_citations(text: str, refs: list) -> dict:
    """Citations in the draft that do not match any supplied source (first author surname + year), the number of
    '[reference needed]' markers, and supplied sources never cited."""
    keys = source_keys(refs)
    found = citations(text)
    stop = {"table", "figure", "جدول", "شكل", "section", "equation", "in", "since", "from", "عام", "سنة", "منذ"}
    unmatched, matched = [], set()
    for a, y, raw in found:
        if a.lower() in stop:
            continue
        k = _key(a, y)
        if k in keys:
            matched.add(k)
        else:
            unmatched.append(raw)
    uncited = [r for r in refs if not any(k in matched for k in source_keys([r]))]
    return {"unmatched": list(dict.fromkeys(unmatched)), "n_citations": len(found), "missing_markers": text.count(MISSING["en"]) + text.count(MISSING["ar"]),
            "uncited_sources": uncited}


# ======================================================================================================================
# Prompting
# ======================================================================================================================
def system_prompt(lang: str, style: str = "APA 7") -> str:
    if lang == "ar":
        return ("أنت مساعد كتابة أكاديمية لباحث اقتصادي في منصة Aetheris Research. اكتب بالعربية الفصحى الأكاديمية الرصينة. القواعد الصارمة: "
                "(1) لا تذكر أي مرجع أو دراسة أو مؤلف أو رقم غير موجود في «المصادر» أو «النتائج» المرفقة؛ "
                f"(2) استشهد داخل النص بأسلوب {style} (المؤلف، السنة) من المصادر المرفقة فقط؛ "
                f"(3) إذا احتاجت فكرة إلى سند غير موجود في المصادر فاكتب {MISSING['ar']} بدلاً من اختراع مرجع؛ "
                "(4) لا تنسخ نصوص المصادر حرفياً بل أعد صياغتها وركّبها؛ (5) أعد النص فقط بفقرات وعناوين فرعية عند الحاجة دون تعليقات جانبية، "
                "ثم سطر «المراجع المستخدمة:» يليه مراجع المصادر المرفقة التي استشهدت بها فعلاً فقط.")
    return ("You are an academic writing assistant for an economics researcher on the Aetheris Research platform. Write rigorous academic "
            "English. Strict rules: (1) never mention any reference, study, author or number that is not in the SOURCES or RESULTS provided; "
            f"(2) cite in the text in {style} author–year style using only the supplied sources; (3) where a claim needs support that the "
            f"sources do not give, write {MISSING['en']} instead of inventing a reference; (4) paraphrase and synthesise — do not copy "
            "the sources verbatim; (5) return only the text (paragraphs, sub-headings if useful), then a line 'References used:' followed by "
            "the supplied references you actually cited.")


def build_request(section: str, lang: str, topic: str, question: str, context: str, sources: str, results_json: str | None,
                  words: int, instructions: str) -> str:
    sec = SECTIONS[section][1 if lang == "ar" else 0]
    guide = GUIDE[section][1 if lang == "ar" else 0]
    parts = [("المطلوب" if lang == "ar" else "TASK") + f": {sec} — {guide}",
             ("عنوان الدراسة" if lang == "ar" else "STUDY TITLE") + f": {topic or '—'}",
             ("سؤال البحث" if lang == "ar" else "RESEARCH QUESTION") + f": {question or '—'}",
             ("السياق (الدولة/القطاع/الفترة)" if lang == "ar" else "CONTEXT (country / sector / period)") + f": {context or '—'}",
             ("الطول التقريبي" if lang == "ar" else "APPROXIMATE LENGTH") + f": {words} " + ("كلمة" if lang == "ar" else "words")]
    if instructions:
        parts.append(("تعليمات الباحث" if lang == "ar" else "RESEARCHER'S INSTRUCTIONS") + f": {instructions}")
    parts.append(("المصادر (النص الوحيد المسموح الاستشهاد به)" if lang == "ar" else "SOURCES (the only material you may cite)") + ":\n" +
                 (sources.strip()[:70000] if sources.strip() else ("لا توجد مصادر مرفقة — لا تستشهد بأي مرجع، وضع " + MISSING["ar"] if lang == "ar"
                                                                   else "No sources supplied — cite nothing; use " + MISSING["en"]) + (" حيث يلزم سند." if lang == "ar" else " where support is needed.")))
    if results_json:
        parts.append(("النتائج المحسوبة في المنصة (JSON)" if lang == "ar" else "RESULTS COMPUTED IN THE PLATFORM (JSON)") + ":\n" + results_json[:40000])
    return "\n\n".join(parts)


def outline(section: str, lang: str, topic: str, question: str, refs: list) -> str:
    """Offline mode: a structured skeleton with guidance and the supplied sources mapped in — no invented content."""
    ar = lang == "ar"
    T = {
        "intro": [("الخلفية والسياق", "Background and context"), ("أهمية الموضوع", "Why the topic matters"), ("الفجوة البحثية", "Research gap"),
                  ("هدف الدراسة", "Aim of the study"), ("تنظيم الدراسة", "Structure of the study")],
        "problem": [("مشكلة البحث", "Problem statement"), ("أسئلة البحث", "Research questions"), ("أهداف البحث", "Objectives"), ("إسهام الدراسة", "Contribution")],
        "litreview": [("المحور الأول: …", "Theme 1: …"), ("المحور الثاني: …", "Theme 2: …"), ("المحور الثالث: المنهجيات المستخدمة", "Theme 3: methods used"),
                      ("التعليق على الدراسات السابقة والفجوة البحثية", "Synthesis and research gap")],
        "theory": [("النظرية الأساسية", "Core theory"), ("الفرضيات الرئيسة للنظرية", "Key propositions"), ("ربط النظرية بمتغيرات الدراسة", "Linking theory to the variables"),
                   ("ملاءمة النظرية للسياق", "Fit to the context")],
        "hypotheses": [("النموذج المفاهيمي", "Conceptual model"), ("ف1: …", "H1: …"), ("ف2: …", "H2: …"), ("ف3: …", "H3: …")],
        "method": [("البيانات ومصادرها", "Data and sources"), ("المتغيرات وقياسها", "Variables and measurement"), ("النموذج القياسي", "Econometric model"),
                   ("طريقة التقدير والاختبارات", "Estimation and tests")],
        "discussion": [("النتائج الرئيسة", "Main results"), ("المقارنة بالدراسات السابقة", "Comparison with previous studies"), ("تفسير الاختلافات", "Explaining differences")],
        "conclusion": [("الخلاصة", "Conclusions"), ("التوصيات", "Recommendations"), ("حدود الدراسة", "Limitations"), ("بحوث مستقبلية", "Future research")],
        "free": [("النص", "Text")],
    }[section]
    lines = [f"# {SECTIONS[section][1 if ar else 0]}" + (f" — {topic}" if topic else ""), ""]
    if question:
        lines += [("سؤال البحث: " if ar else "Research question: ") + question, ""]
    lines += [("إرشاد: " if ar else "Guidance: ") + GUIDE[section][1 if ar else 0], ""]
    for h in T:
        lines += [f"## {h[0] if ar else h[1]}", ("[اكتب هنا — استشهد بالمصادر أدناه بصيغة (المؤلف، السنة)]" if ar else "[write here — cite the sources below as (Author, Year)]"), ""]
    if refs:
        lines += [("## المصادر المتاحة للاستشهاد" if ar else "## Sources available to cite")] + [f"- {r}" for r in refs]
    else:
        lines += [("لم تُرفق مصادر بعد: الصق المراجع أو ارفع ملفات الدراسات في خانة المصادر." if ar else "No sources yet: paste references or upload the studies in the Sources box.")]
    return "\n".join(lines)


def split_references(text: str) -> tuple:
    """Separate the draft body from a trailing 'References used:' block."""
    m = re.search(r"\n\s*(References used:|المراجع المستخدمة:)\s*\n?", text)
    if not m:
        return text.strip(), []
    body = text[:m.start()].strip()
    refs = [r.strip().lstrip("-•* ").strip() for r in text[m.end():].splitlines() if r.strip()]
    return body, refs
