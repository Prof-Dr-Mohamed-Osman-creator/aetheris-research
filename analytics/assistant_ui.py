# ==========================================
# Aetheris Research - AI research assistant (UI) — v1.12, Catalogue §8
# ==========================================

from __future__ import annotations

import json

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
    tabs = st.tabs([tr("🗨️ Free conversation", "🗨️ محادثة حرة"), tr("📝 Research writing", "📝 الكتابة البحثية"), tr("🧭 Model advisor", "🧭 مستشار النماذج"),
                    tr("💬 Methodology library", "💬 مكتبة المنهجيات")])
    with tabs[0]:
        _free_chat(llm)
    with tabs[1]:
        _writing(llm)
    with tabs[2]:
        _advisor(llm)
    with tabs[3]:
        _chat(llm)


# ---------------------------------------------------------- free conversation (the model's own knowledge)
def _free_chat(llm):
    from analytics import free_chat as fc
    ss = st.session_state
    hist = ss.setdefault("_fc_hist", [])
    lang = _lang()
    st.caption(tr("Talk freely with the AI about anything in your research — concepts, critiques of a model, ideas for a thesis, drafting, translation. "
                  "It answers from its own knowledge (no sources needed). Treat facts and references from a model's memory with care: use 🔎 to check "
                  "references against Crossref.",
                  "تحدّث بحرية مع الذكاء الاصطناعي في أي شأن بحثي — المفاهيم، ونقد نموذج، وأفكار لرسالة، والصياغة، والترجمة. يجيب من معرفته "
                  "(دون حاجة إلى مصادر). تعامل بحذر مع الحقائق والمراجع الصادرة من ذاكرة النموذج: استخدم 🔎 للتحقق من المراجع عبر Crossref."))
    from analytics.ai_settings_ui import gate
    llm = gate(llm, "fc")
    with st.expander(tr("📎 Attach context to my next questions (optional)", "📎 أرفق سياقاً بأسئلتي القادمة (اختياري)"), expanded=False):
        a, b, c = st.columns(3)
        use_data = a.checkbox(tr("Structure of the active dataset", "بنية البيانات النشطة"), False, key="fc_ctx_data", disabled=not has_data())
        res = list(store.results().values())
        use_res = b.checkbox(tr(f"Results of this session ({len(res)})", f"نتائج هذه الجلسة ({len(res)})"), False, key="fc_ctx_res", disabled=not res)
        draft = (ss.get("_wr") or {}).get("draft", "")
        use_draft = c.checkbox(tr("My writing-workspace draft", "مسودتي في مساحة الكتابة"), False, key="fc_ctx_draft", disabled=not draft)
        st.caption(tr("Only what you tick is sent — never the raw data rows.", "لا يُرسل إلا ما تختاره — ولا تُرسل صفوف البيانات الخام أبداً."))
    for i, m in enumerate(hist):
        with st.chat_message(m["role"]):
            st.markdown(m.get("display", m["content"]))
            if m["role"] == "assistant" and m.get("checks"):
                for r in m["checks"]:
                    icon = {"verified": "✅", "partial": "🟡", "not found": "❌", "error": "⚪"}[r["status"]]
                    line = f"{icon} {r['reference'][:160]}"
                    if r.get("doi") and r["status"] != "not found":
                        line += f"  \n↳ {r.get('title', '')[:120]} ({r.get('year')}) · https://doi.org/{r['doi']}"
                    st.caption(line)
    q = st.chat_input(tr("Write your question or request…", "اكتب سؤالك أو طلبك…"), key="fc_in", disabled=not llm.available)
    if q:
        ctx = {}
        if use_data and has_data():
            ctx["dataset_profile"] = {k: v for k, v in ai.profile(get_memory()).items()}
            ctx["dataset_name"] = active_name()
        if use_res and res:
            ctx["session_results"] = json.loads(ai._result_context(res[-4:]))
        if use_draft and draft:
            ctx["writing_draft"] = draft[:20000]
        msgs = fc.build_messages([{"role": m["role"], "content": m["content"]} for m in hist], q, ctx or None, lang)
        hist.append({"role": "user", "content": msgs[-1]["content"], "display": q})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("…"):
                    out = llm.complete(fc.SYSTEM[lang], msgs, max_tokens=3000, temperature=0.5)
            except Exception as e:  # noqa: BLE001
                out = None
                st.error(tr(f"The AI service did not respond: {e}", f"لم تستجب خدمة الذكاء الاصطناعي: {e}"))
            if out:
                st.markdown(out)
                hist.append({"role": "assistant", "content": out, "model": llm.label})
        if not out:
            hist.pop()
    last = next((m for m in reversed(hist) if m["role"] == "assistant"), None)
    if last:
        refs = fc.find_references(last["content"])
        c1, c2, c3 = st.columns(3)
        if refs and c1.button(tr(f"🔎 Verify {len(refs)} reference(s) (Crossref)", f"🔎 تحقق من {len(refs)} مرجعاً (Crossref)"), key="fc_verify"):
            with st.spinner(tr("Checking the DOI registry…", "جارٍ الفحص في سجل DOI…")):
                last["checks"] = [fc.verify_reference(r) for r in refs]
            st.rerun()
        if c2.button(tr("➕ Add last answer to the report", "➕ أضف آخر إجابة إلى التقرير"), key="fc_add"):
            store.add_item("text", tr("Notes from the AI conversation", "ملاحظات من المحادثة"), text=last["content"])
            st.success(tr("Added — see 📄 Report Studio → ② Report outline.", "تمت الإضافة — راجع 📄 استوديو التقارير ← ② هيكل التقرير."))
        if c3.button(tr("📝 Send to the writing workspace", "📝 أرسل إلى مساحة الكتابة"), key="fc_to_wr"):
            W = ss.setdefault("_wr", {"draft": "", "history": [], "refs_used": [], "section": "free"})
            W["draft"] = ((W.get("draft") or "") + "\n\n" + last["content"]).strip()
            W["pending"] = W["draft"]
            st.success(tr("Appended to the draft in 📝 Research writing.", "أُلحقت بالمسودة في 📝 الكتابة البحثية."))
        if last.get("checks"):
            bad = sum(r["status"] == "not found" for r in last["checks"])
            if bad:
                st.warning(tr(f"{bad} reference(s) could not be found in Crossref — they may not exist; do not cite them without checking the original.",
                              f"{bad} مرجعاً لم يُعثر عليه في Crossref — قد لا يكون موجوداً؛ لا تستشهد به قبل الرجوع إلى الأصل."))
    if hist:
        from analytics.reg_report import to_docx
        md = fc.to_markdown(hist, lang)
        d1, d2, d3 = st.columns(3)
        d1.download_button(tr("⬇️ Conversation (Markdown)", "⬇️ المحادثة (Markdown)"), md.encode("utf-8"), "conversation.md", key="fc_md")
        d2.download_button(tr("⬇️ Conversation (Word)", "⬇️ المحادثة (Word)"),
                           to_docx([("p", x) for x in md.split("\n\n") if x.strip() and not x.startswith("# ")], tr("AI conversation", "محادثة الذكاء الاصطناعي"), is_ar()),
                           "conversation.docx", key="fc_docx")
        if d3.button(tr("🧹 New conversation", "🧹 محادثة جديدة"), key="fc_clear"):
            ss["_fc_hist"] = []
            st.rerun()


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
