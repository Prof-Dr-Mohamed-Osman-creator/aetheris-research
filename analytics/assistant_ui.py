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
    tabs = st.tabs([tr("🧭 Model advisor", "🧭 مستشار النماذج"), tr("💬 Ask", "💬 اسأل"), tr("✍️ Write", "✍️ اكتب")])
    with tabs[0]:
        _advisor(llm)
    with tabs[1]:
        _chat(llm)
    with tabs[2]:
        st.info(tr("Drafting of methodology, results, discussion, abstract and policy briefs from your computed results is in 📄 Report Studio → ③ Write "
                   "with the assistant; every number in the draft is checked against the results.",
                   "صياغة المنهجية والنتائج والمناقشة والملخص وموجز السياسات من نتائجك المحسوبة موجودة في 📄 استوديو التقارير ← ③ الكتابة بالمساعد؛ "
                   "ويُفحص كل رقم في المسودة مقابل النتائج."))
        n = len(store.results())
        st.metric(tr("Results captured in this session", "النتائج الملتقطة في هذه الجلسة"), n)


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
