# ==========================================
# Aetheris Research - AI research assistant (UI) — v1.12, Catalogue §8
# ==========================================

from __future__ import annotations

import streamlit as st

from analytics import ai_assistant as ai
from core import report_store as store
from core.memory import active_name, get_memory, has_data
from core.utils import tr, is_ar


def _t(p):
    return p[1] if is_ar() else p[0]


def _lang():
    return "ar" if is_ar() else "en"


def _llm():
    return ai.session_llm()


def render_assistant():
    st.caption("💡 " + tr("Suggests the right model for your question and data, writes academic text only from numbers the platform actually computed, "
                         "and answers your methodological questions — in Arabic or English.",
                         "يقترح النموذج المناسب لسؤالك وبياناتك، ويكتب النصوص الأكاديمية من الأرقام التي حسبتها المنصة فعلاً فقط، ويجيب عن أسئلتك "
                         "المنهجية — بالعربية أو الإنجليزية."))
    from analytics.ai_settings_ui import settings_panel
    llm = settings_panel("ai")
    tabs = st.tabs([tr("📝 Research writing", "📝 الكتابة البحثية"), tr("🧭 Model advisor", "🧭 مستشار النماذج"), tr("💬 Ask", "💬 اسأل")])
    with tabs[0]:
        _writing(llm)
    with tabs[1]:
        _advisor(llm)
    with tabs[2]:
        _chat(llm)


# ---------------------------------------------------------- research writing workspace
def _writing(llm):
    from analytics import report_studio as rs
    from analytics import writing as wr
    ss = st.session_state
    W = ss.setdefault("_wr", {"draft": "", "history": [], "refs_used": [], "section": None})
    lang = _lang()
    st.caption(tr("Write the narrative parts of your thesis or paper in a conversation with the AI. It cites ONLY the sources you give it (paste or upload "
                  "studies, abstracts, references or your notes) and writes “[reference needed]” instead of ever inventing a reference; every citation in "
                  "the draft is checked against your sources.",
                  "اكتب الأجزاء السردية لرسالتك أو بحثك في حوار مع الذكاء الاصطناعي. لا يستشهد إلا بالمصادر التي تقدمها (الصق أو ارفع الدراسات أو "
                  "ملخصاتها أو المراجع أو ملاحظاتك)، ويكتب «[مرجع مطلوب]» بدلاً من اختراع أي مرجع، ويُفحص كل استشهاد في المسودة مقابل مصادرك."))
    c1, c2 = st.columns([1.2, 1])
    sec = c1.selectbox(tr("What do you want to write?", "ماذا تريد أن تكتب؟"), list(wr.SECTIONS), key="wr_sec", format_func=lambda k: _t(wr.SECTIONS[k]))
    words = c2.slider(tr("Approximate length (words)", "الطول التقريبي (كلمات)"), 150, 2500, 700, 50, key="wr_words")
    st.caption("💡 " + _t(wr.GUIDE[sec]))
    a, b = st.columns(2)
    topic = a.text_input(tr("Study title", "عنوان الدراسة"), key="wr_topic")
    context = b.text_input(tr("Context (country, sector, period)", "السياق (الدولة، القطاع، الفترة)"), key="wr_ctx")
    question = st.text_input(tr("Research question / aim", "سؤال البحث / الهدف"), key="wr_q")
    with st.expander(tr("📚 Sources — studies, references, notes (the only material the AI may cite)", "📚 المصادر — الدراسات والمراجع والملاحظات (المادة الوحيدة التي يجوز الاستشهاد بها)"),
                     expanded=not W["draft"]):
        pasted = st.text_area(tr("Paste references, abstracts, summaries of previous studies or your notes", "الصق المراجع أو ملخصات الدراسات السابقة أو ملاحظاتك"),
                              height=180, key="wr_src", placeholder=tr("e.g. Schlenker, W. & Roberts, M. J. (2009). Nonlinear temperature effects … — then a few lines on its findings",
                                                                     "مثال: سعيد، أحمد (2018). أثر التغيرات المناخية على إنتاجية القمح في مصر … — ثم سطور عن نتائجها"))
        files = st.file_uploader(tr("…or upload studies (PDF, Word, BibTeX, text)", "…أو ارفع الدراسات (PDF أو Word أو BibTeX أو نص)"), type=["pdf", "docx", "txt", "md", "bib"],
                                 accept_multiple_files=True, key="wr_files")
        texts = [pasted or ""]
        for f in files or []:
            texts.append(f"=== {f.name} ===\n" + wr.extract_text(f.name, f.getvalue()))
        sources = "\n\n".join(t for t in texts if t.strip())
        refs = wr.reference_list(sources)
        st.caption(tr(f"{len(sources):,} characters of source material · {len(refs)} reference(s) detected", f"{len(sources):,} حرفاً من المصادر · {len(refs)} مرجعاً مكتشفاً"))
        if refs:
            st.markdown("\n".join(f"- {r[:220]}" for r in refs[:40]))
    res = list(store.results().values())
    use_res = st.checkbox(tr(f"Use the results computed in this session ({len(res)})", f"استخدم النتائج المحسوبة في هذه الجلسة ({len(res)})"), bool(res) and sec in ("method", "discussion", "conclusion"),
                          key="wr_useres", disabled=not res)
    instr = st.text_input(tr("Extra instructions (optional)", "تعليمات إضافية (اختياري)"), key="wr_instr",
                          placeholder=tr("e.g. group the studies by region; compare Ricardian and panel approaches", "مثال: صنّف الدراسات حسب الإقليم؛ قارن بين المنهج الريكاردي ونماذج البانل"))
    results_json = ai._result_context(res) if use_res and res else None
    go = st.button((tr("✍️ Draft with AI", "✍️ اكتب المسودة بالذكاء الاصطناعي") if llm.available else tr("📋 Create a structured outline (offline)", "📋 أنشئ هيكلاً منظماً (دون اتصال)")),
                   key="wr_go", type="primary")
    if go:
        if llm.available:
            req = wr.build_request(sec, lang, topic, question, context, sources, results_json, words, instr)
            try:
                with st.spinner(tr("Writing from your sources…", "جارٍ الكتابة من مصادرك…")):
                    out = llm.complete(wr.system_prompt(lang), [{"role": "user", "content": req}], max_tokens=min(8000, int(words * 3.2) + 600), temperature=0.4)
                W.update(history=[{"role": "user", "content": req}, {"role": "assistant", "content": out}], section=sec, model=llm.label)
                body, used = wr.split_references(out)
                W["draft"], W["refs_used"] = body, used
            except Exception as e:  # noqa: BLE001
                st.error(tr(f"The AI service did not respond: {e}", f"لم تستجب خدمة الذكاء الاصطناعي: {e}"))
        else:
            W.update(draft=wr.outline(sec, lang, topic, question, refs), history=[], refs_used=[], section=sec, model="")
        ss["wr_draft_txt"] = W["draft"]
    if not W["draft"]:
        return
    st.divider()
    if W.get("pending") is not None:
        ss["wr_draft_txt"] = W.pop("pending")
    if "wr_draft_txt" not in ss:
        ss["wr_draft_txt"] = W["draft"]
    txt = st.text_area(tr("Draft (editable)", "المسودة (قابلة للتعديل)"), height=420, key="wr_draft_txt")
    W["draft"] = txt
    chk = wr.check_citations(txt, refs)
    c1, c2, c3 = st.columns(3)
    c1.metric(tr("Citations in the draft", "الاستشهادات في المسودة"), chk["n_citations"])
    c2.metric(tr("Not in your sources", "غير موجودة في مصادرك"), len(chk["unmatched"]))
    c3.metric(tr("“Reference needed” markers", "مواضع «مرجع مطلوب»"), chk["missing_markers"])
    if chk["unmatched"]:
        st.warning(tr("⚠️ These citations do not match any source you supplied — verify them or remove them: ", "⚠️ هذه الاستشهادات لا تطابق أي مصدر قدمته — تحقق منها أو احذفها: ")
                   + " · ".join(chk["unmatched"][:15]))
    elif chk["n_citations"]:
        st.success(tr("✔️ Every citation matches a supplied source.", "✔️ كل استشهاد يطابق مصدراً قدمته."))
    if chk["missing_markers"]:
        st.info(tr("Add sources for the “[reference needed]” places (paste them above and revise), or rewrite those sentences.",
                   "أضف مصادر لمواضع «[مرجع مطلوب]» (الصقها أعلاه ثم اطلب المراجعة)، أو أعد صياغة تلك الجمل."))
    if results_json:
        bad = rs.check_numbers(txt, res)
        if bad:
            st.warning(tr("Numbers not found in the computed results: ", "أرقام غير موجودة في النتائج المحسوبة: ") + ", ".join(bad[:12]))
    if llm.available and W.get("history"):
        r1, r2 = st.columns([4, 1])
        rev = r1.text_input(tr("Ask for a revision", "اطلب تعديلاً"), key="wr_rev",
                            placeholder=tr("e.g. shorten to 400 words · add a paragraph on adaptation · more critical tone · use the new sources I added",
                                           "مثال: اختصر إلى 400 كلمة · أضف فقرة عن التكيف · نبرة أكثر نقداً · استخدم المصادر الجديدة التي أضفتها"))
        if r2.button(tr("↻ Revise", "↻ عدّل"), key="wr_rev_go") and rev:
            msg = rev + ("\n\n" + ("المصادر المحدّثة:" if lang == "ar" else "UPDATED SOURCES:") + "\n" + sources[:60000] if sources else "") + \
                ("\n\n" + ("المسودة الحالية بعد تعديلات الباحث:" if lang == "ar" else "CURRENT DRAFT (with the researcher's edits):") + "\n" + txt)
            hist = W["history"][-6:] + [{"role": "user", "content": msg}]
            try:
                with st.spinner("…"):
                    out = llm.complete(wr.system_prompt(lang), hist, max_tokens=min(8000, int(words * 3.2) + 600), temperature=0.4)
                body, used = wr.split_references(out)
                W["history"] = hist + [{"role": "assistant", "content": out}]
                W["draft"], W["refs_used"] = body, used or W["refs_used"]
                W["pending"] = body
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(tr(f"The AI service did not respond: {e}", f"لم تستجب خدمة الذكاء الاصطناعي: {e}"))
    used_refs = [r for r in refs if any(k in wr.source_keys([r]) for k in {wr._key(a, y) for a, y, _ in wr.citations(txt)})]
    if used_refs:
        with st.expander(tr(f"References cited ({len(used_refs)})", f"المراجع المستشهد بها ({len(used_refs)})"), expanded=False):
            st.markdown("\n".join(f"- {r}" for r in sorted(used_refs, key=str.lower)))
    a, b = st.columns(2)
    if a.button(tr("➕ Add to the report (Report Studio)", "➕ أضف إلى التقرير (استوديو التقارير)"), key="wr_add"):
        store.add_item("text", _t(wr.SECTIONS[W.get("section") or sec]), text=txt, references=sorted(used_refs, key=str.lower))
        st.success(tr("Added — open 📄 Report Studio → ② Report outline.", "تمت الإضافة — افتح 📄 استوديو التقارير ← ② هيكل التقرير."))
    from analytics.reg_report import to_docx
    b.download_button(tr("⬇️ Draft (Word)", "⬇️ المسودة (Word)"),
                      to_docx([("p", p) for p in txt.split("\n\n") if p.strip()] + ([("h", tr("References", "المراجع")), ("bullets", used_refs)] if used_refs else []),
                              _t(wr.SECTIONS[W.get("section") or sec]), is_ar()), "draft.docx", key="wr_dl")


def _advisor(llm):
    if not has_data():
        st.info(tr("Load a dataset (Data Hub or any demo) so the advisor can read its structure.", "حمّل بيانات (مركز البيانات أو أي بيانات تجريبية) ليقرأ المستشار بنيتها."))
        return
    df = get_memory()
    P = ai.profile(df)
    c1, c2 = st.columns([2, 1])
    goal = c1.selectbox(tr("What do you want to find out?", "ماذا تريد أن تعرف؟"), list(ai.GOALS), key="ai_goal", format_func=lambda k: _t(ai.GOALS[k]))
    y = c2.selectbox(tr("Main outcome (optional)", "المتغير الرئيسي (اختياري)"), [None] + P["numeric"], key="ai_y",
                     format_func=lambda v: tr("—", "—") if v is None else v)
    with st.expander(tr(f"Data profile — {active_name()}", f"وصف البيانات — {active_name()}"), expanded=False):
        pr = P["panel"]
        st.markdown("\n".join([
            tr(f"- {P['rows']} rows × {P['columns']} columns; missing {P['missing_%']:.1f}%", f"- {P['rows']} صف × {P['columns']} عمود؛ القيم المفقودة {P['missing_%']:.1f}%"),
            tr(f"- Panel: {pr['N']} units × {pr['T']} periods ({'balanced' if pr['balanced'] else 'unbalanced'})", f"- بانل: {pr['N']} وحدة × {pr['T']} فترة ({'متوازن' if pr['balanced'] else 'غير متوازن'})") if pr else
            (tr("- Time series", "- سلسلة زمنية") if P["time_series"] else tr("- Cross-section", "- بيانات مقطعية")),
            tr(f"- Binary variables: {', '.join(P['binary']) or '—'}", f"- متغيرات ثنائية: {', '.join(P['binary']) or '—'}"),
            tr(f"- Coordinates: {', '.join(P['coordinates']) or '—'} · cohorts: {', '.join(P['cohort']) or '—'}", f"- إحداثيات: {', '.join(P['coordinates']) or '—'} · أفواج: {', '.join(P['cohort']) or '—'}")]))
    recs = ai.advise(df, goal, y, _lang())
    if not recs:
        st.info(tr("No specific recommendation for this goal and data structure.", "لا توجد توصية محددة لهذا الهدف وهذه البنية."))
        return
    for i, r in enumerate(recs):
        with st.container(border=True):
            st.markdown(("⭐ " if i == 0 else "") + f"**{r['model']}**")
            st.markdown(r["why"])
            st.caption(tr("Where: ", "أين: ") + r["where"] + " · " + tr("Check: ", "تحقق من: ") + r["checks"])
    if llm.available and st.button(tr("🤖 Ask the AI to explain the choice for my data", "🤖 اطلب من الذكاء الاصطناعي شرح الاختيار لبياناتي"), key="ai_adv_llm"):
        q = (tr("Explain which of these models suits my data and goal, the identifying assumptions and the pitfalls: ", "اشرح أي هذه النماذج يناسب بياناتي وهدفي، وفروض التعريف، والمزالق: ")
             + "; ".join(r["model"] for r in recs) + f" · goal: {_t(ai.GOALS[goal])} · outcome: {y}")
        try:
            with st.spinner("…"):
                out = ai.answer(q, _lang(), llm, data_profile=P)
            st.markdown(out["text"])
        except Exception as e:  # noqa: BLE001
            st.error(tr(f"The AI service did not respond: {e}", f"لم تستجب خدمة الذكاء الاصطناعي: {e}"))


def _chat(llm):
    hist = st.session_state.setdefault("_ai_chat", [])
    st.caption(tr("Examples: “When should I use fixed rather than random effects?” · “What does a negative spillover in SDM mean?” · “How do I test for a unit root?”",
                  "أمثلة: «متى أستخدم الآثار الثابتة بدلاً من العشوائية؟» · «ماذا يعني الأثر الانتشاري السالب في نموذج SDM؟» · «كيف أختبر جذر الوحدة؟»"))
    for m in hist:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    q = st.chat_input(tr("Ask a methodological question…", "اطرح سؤالاً منهجياً…"), key="ai_chat_in")
    if q:
        hist.append({"role": "user", "content": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("…"):
                    out = ai.answer(q, _lang(), llm if llm.available else None, [m for m in hist[:-1]], list(store.results().values())[-4:],
                                    ai.profile(get_memory()) if has_data() else None)
            except Exception as e:  # noqa: BLE001
                st.warning(tr(f"The AI service did not respond ({e}); answering from the platform library.", f"لم تستجب خدمة الذكاء الاصطناعي ({e})؛ الإجابة من مكتبة المنصة."))
                out = ai.answer(q, _lang(), None)
            st.markdown(out["text"])
            if out.get("hits") and out["source"] == "ai":
                st.caption(tr("Grounded on: ", "استناداً إلى: ") + ", ".join(h["id"] for h in out["hits"]))
        hist.append({"role": "assistant", "content": out["text"]})
    if hist and st.button(tr("🧹 Clear conversation", "🧹 مسح المحادثة"), key="ai_clear"):
        st.session_state["_ai_chat"] = []
        st.rerun()
