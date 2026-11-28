# ==========================================
# Aetheris Research - Report Studio (UI) — v1.12, Catalogue §8
# ==========================================

from __future__ import annotations

import datetime as _dt

import pandas as pd
import streamlit as st

from analytics import ai_assistant as ai
from analytics import report_studio as rs
from core import report_store as store
from core.utils import tr, is_ar

PAGES = {"econometrics": ("📐 Econometrics", "📐 الاقتصاد القياسي"), "forecasting": ("🔮 Forecasting", "🔮 التنبؤ"), "applied": ("🌾 Applied economics", "🌾 الاقتصاد التطبيقي"),
         "impact": ("🧭 Impact, spatial & climate", "🧭 الأثر والمكان والمناخ"), "descriptive": ("📋 Descriptive", "📋 الوصفي"), "inferential": ("🧪 Inferential", "🧪 الاستدلالي")}


def _t(p):
    return p[1] if is_ar() else p[0]


def _lang():
    return "ar" if is_ar() else "en"


def llm_from_session():
    return ai.session_llm()


def render_reports():
    st.caption("💡 " + tr("Every analysis you run in the platform is captured here with its tables, charts, interpretation and references. "
                         "Pick results, arrange them, let the assistant draft the text from the computed numbers, and export one report.",
                         "كل تحليل تجريه في المنصة يُلتقط هنا بجداوله ورسومه وتفسيره ومراجعه. اختر النتائج ورتّبها، ودع المساعد يصوغ النص من "
                         "الأرقام المحسوبة فعلاً، ثم صدّر تقريراً واحداً."))
    res = store.results()
    tabs = st.tabs([tr("① Results", "① النتائج"), tr("② Report outline", "② هيكل التقرير"), tr("③ Write with the assistant", "③ الكتابة بالمساعد"),
                    tr("④ Export", "④ التصدير")])
    with tabs[0]:
        _results_tab(res)
    with tabs[1]:
        _outline_tab()
    with tabs[2]:
        _write_tab(res)
    with tabs[3]:
        _export_tab()


def _results_tab(res):
    if not res:
        st.info(tr("No results yet. Run any analysis (e.g. Econometrics → Regression, or a demo in the Impact lab); each analysis that offers a Word "
                   "report appears here automatically.",
                   "لا توجد نتائج بعد. شغّل أي تحليل (مثل: الاقتصاد القياسي ← الانحدار، أو بيانات تجريبية في مختبر الأثر)؛ وكل تحليل يقدّم تقرير Word "
                   "يظهر هنا تلقائياً."))
        if st.button(tr("🧪 Load a sample result to try the studio", "🧪 تحميل نتيجة نموذجية لتجربة الاستوديو"), key="rs_demo"):
            _sample_result()
            st.rerun()
        return
    in_report = {it["result"]["key"] for it in store.items() if it["kind"] == "result"}
    st.caption(tr("Note: every tab of a wing is computed when the page opens, so results from tabs you did not inspect (or that do not suit your data) "
                  "may also be listed — add only what you checked.",
                  "ملاحظة: تُحسب كل تبويبات الجناح عند فتح الصفحة، لذا قد تظهر نتائج لتبويبات لم تراجعها (أو لا تناسب بياناتك) — أضف فقط ما راجعته."))
    c1, c2 = st.columns([3, 1])
    dsets = sorted({r.get("dataset") or "—" for r in res.values()})
    pick = c1.multiselect(tr("Datasets", "البيانات"), dsets, dsets, key="rs_ds")
    if c2.button(tr("🧹 Clear captured results", "🧹 مسح النتائج الملتقطة"), key="rs_clear"):
        st.session_state[store.RESULTS] = {}
        st.rerun()
    for key, r in sorted(res.items(), key=lambda kv: -kv[1]["time"]):
        if (r.get("dataset") or "—") not in pick:
            continue
        ntab = sum(1 for k, c in r["sections"] if isinstance(c, pd.DataFrame))
        with st.container(border=True):
            c1, c2 = st.columns([5, 1.4])
            with c1:
                st.markdown(f"**{r['title']}** · {_t(PAGES.get(r['page'], (r['page'], r['page'])))}")
                st.caption(tr(f"{ntab} table(s) · {len(r['figures'])} chart(s) · {len(r['insights'])} interpretation(s) · data: {r.get('dataset') or '—'} · "
                              f"{_dt.datetime.fromtimestamp(r['time']).strftime('%H:%M')}",
                              f"{ntab} جدول · {len(r['figures'])} رسم · {len(r['insights'])} تفسير · البيانات: {r.get('dataset') or '—'} · "
                              f"{_dt.datetime.fromtimestamp(r['time']).strftime('%H:%M')}"))
                if r["insights"]:
                    st.markdown("> " + rs.strip_html(r["insights"][0])[:400])
            with c2:
                if key in in_report:
                    st.success(tr("In report", "في التقرير"), icon="✅")
                elif st.button(tr("➕ Add to report", "➕ أضف للتقرير"), key=f"rs_add_{key}", width="stretch"):
                    store.add_result(key)
                    st.rerun()


def _sample_result():
    """A small regression result so the studio can be tried without running an analysis first."""
    import numpy as np
    import plotly.graph_objects as go
    from analytics import regression as rg
    from analytics import reg_report as rp
    from data.samples import farm_survey_demo
    df = farm_survey_demo()
    r = rg.fit(df, rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days", "fertilizer_kg"], form="loglog"), cov_type="HC1")
    txt, refs = rp.methodology(r, True, _lang())
    tab = pd.DataFrame({"Term": r.params.index, "Estimate": r.params.values, "Std. Err.": r.bse.values, "p-value": r.pvalues.values})
    import time as _time
    rts = float(r.params.iloc[1:].sum())
    ins = tr(f"Output elasticities: land {r.params.iloc[1]:.3f}, labour {r.params.iloc[2]:.3f}, fertilizer {r.params.iloc[3]:.3f}; returns to scale {rts:.3f}.",
             f"المرونات الإنتاجية: الأرض {r.params.iloc[1]:.3f}، والعمل {r.params.iloc[2]:.3f}، والسماد {r.params.iloc[3]:.3f}؛ والعائد للسعة {rts:.3f}.")
    fig = go.Figure(go.Scatter(x=np.asarray(r.fitted), y=np.asarray(r.resid), mode="markers"))
    fig.update_layout(xaxis_title="Fitted", yaxis_title="Residual")
    title = tr("Cobb-Douglas production function (sample)", "دالة إنتاج كوب-دوجلاس (نموذج)")
    store.put_result({"key": f"econometrics|{title}", "page": "econometrics", "title": title,
                      "sections": [("h", tr("Methodology", "المنهجية")), ("p", txt), ("table", tab), ("bullets", refs)], "insights": [ins],
                      "figures": [{"json": fig.to_json(), "name": "residuals_vs_fitted"}], "time": _time.time(), "lang": _lang(), "dataset": "farm_survey_demo"})


def _outline_tab():
    meta = st.session_state.setdefault("_rs_meta", {"title": tr("Research report", "تقرير بحثي"), "author": "", "affiliation": "", "abstract": "", "keywords": ""})
    c1, c2 = st.columns(2)
    meta["title"] = c1.text_input(tr("Title", "العنوان"), meta["title"], key="rs_title")
    meta["author"] = c2.text_input(tr("Author(s)", "المؤلف/المؤلفون"), meta["author"], key="rs_author")
    meta["affiliation"] = c1.text_input(tr("Affiliation", "جهة الانتماء"), meta["affiliation"], key="rs_aff")
    meta["keywords"] = c2.text_input(tr("Keywords", "الكلمات المفتاحية"), meta["keywords"], key="rs_kw")
    meta["abstract"] = st.text_area(tr("Abstract (or draft it in tab ③)", "الملخص (أو اكتبه بالمساعد في التبويب ③)"), meta["abstract"], height=110, key="rs_abs")
    st.divider()
    a, b, _ = st.columns([1, 1, 3])
    if a.button(tr("➕ Section heading", "➕ عنوان قسم"), key="rs_addh"):
        store.add_item("heading", tr("New section", "قسم جديد"))
    if b.button(tr("➕ Text block", "➕ فقرة نصية"), key="rs_addt"):
        store.add_item("text", "", text="")
    L = store.items()
    if not L:
        st.info(tr("The report is empty — add results from tab ①, or headings / text blocks above.", "التقرير فارغ — أضف نتائج من التبويب ①، أو عناوين وفقرات من الأعلى."))
        return
    for i, it in enumerate(L):
        with st.container(border=True):
            c = st.columns([0.5, 0.5, 6, 0.6])
            if c[0].button("⬆️", key=f"rs_up_{it['id']}", disabled=i == 0):
                store.move(i, -1)
                st.rerun()
            if c[1].button("⬇️", key=f"rs_dn_{it['id']}", disabled=i == len(L) - 1):
                store.move(i, 1)
                st.rerun()
            if c[3].button("🗑️", key=f"rs_rm_{it['id']}"):
                store.remove(i)
                st.rerun()
            with c[2]:
                if it["kind"] == "heading":
                    it["title"] = st.text_input(tr("Section heading", "عنوان القسم"), it["title"], key=f"rs_h_{it['id']}")
                elif it["kind"] == "text":
                    it["title"] = st.text_input(tr("Sub-heading (optional)", "عنوان فرعي (اختياري)"), it.get("title", ""), key=f"rs_tt_{it['id']}")
                    it["text"] = st.text_area(tr("Text", "النص"), it.get("text", ""), height=120, key=f"rs_tx_{it['id']}")
                    _check_box(it["text"])
                else:
                    r = it["result"]
                    it["title"] = st.text_input(tr("Section title", "عنوان القسم"), it["title"], key=f"rs_rt_{it['id']}")
                    x1, x2, x3 = st.columns(3)
                    it["include_tables"] = x1.checkbox(tr("Tables", "الجداول"), it.get("include_tables", True), key=f"rs_it_{it['id']}")
                    it["include_figures"] = x2.checkbox(tr(f"Charts ({len(r['figures'])})", f"الرسوم ({len(r['figures'])})"), it.get("include_figures", True), key=f"rs_if_{it['id']}")
                    it["include_insights"] = x3.checkbox(tr("Interpretation", "التفسير"), it.get("include_insights", True), key=f"rs_ii_{it['id']}")
                    it["text"] = st.text_area(tr("Results text (written by you or the assistant)", "نص النتائج (بقلمك أو بالمساعد)"), it.get("text", ""), height=110,
                                              key=f"rs_rx_{it['id']}")
                    _check_box(it["text"], [r])


def _check_box(text, results=None):
    if not text:
        return
    res = results or [it["result"] for it in store.items() if it["kind"] == "result"]
    if not res:
        return
    bad = rs.check_numbers(text, res)
    if bad:
        st.warning(tr("⚠️ These numbers were not found in the computed results — check them: ", "⚠️ هذه الأرقام غير موجودة في النتائج المحسوبة — راجعها: ")
                   + ", ".join(bad[:12]))
    else:
        st.caption(tr("✔️ Every number in this text matches the computed results.", "✔️ كل رقم في هذا النص يطابق النتائج المحسوبة."))


def _write_tab(res):
    L = [it for it in store.items() if it["kind"] == "result"]
    from analytics.ai_settings_ui import compact_status
    llm = compact_status()
    if not L:
        st.info(tr("Add results to the report first (tab ①).", "أضف نتائج إلى التقرير أولاً (التبويب ①)."))
        return
    labels = {it["id"]: it["title"] for it in L}
    pick = st.multiselect(tr("Results to write about", "النتائج المراد الكتابة عنها"), list(labels), list(labels), format_func=labels.get, key="rs_w_pick")
    task = st.radio(tr("Draft", "المسودة"), list(ai.TASKS), horizontal=True, key="rs_w_task",
                    format_func=lambda k: {"results": tr("Results", "النتائج"), "methodology": tr("Methodology", "المنهجية"), "discussion": tr("Discussion", "المناقشة"),
                                           "policy": tr("Policy brief", "موجز السياسات"), "abstract": tr("Abstract", "الملخص")}[k])
    extra = st.text_input(tr("Extra instructions (optional)", "تعليمات إضافية (اختياري)"), "", key="rs_w_extra",
                          placeholder=tr("e.g. focus on the elasticity of fertilizer; 150 words", "مثال: ركّز على مرونة السماد؛ 150 كلمة"))
    if st.button(tr("✍️ Draft", "✍️ اكتب المسودة"), key="rs_w_go", type="primary") and pick:
        chosen = [it for it in L if it["id"] in pick]
        try:
            with st.spinner(tr("Writing from the computed results…", "جارٍ الكتابة من النتائج المحسوبة…")):
                out = ai.write(task, [it["result"] for it in chosen], _lang(), llm if llm.available else None, extra)
        except Exception as e:  # noqa: BLE001
            st.error(tr(f"The AI service did not respond ({e}); using the offline template.", f"لم تستجب خدمة الذكاء الاصطناعي ({e})؛ سيُستخدم القالب دون اتصال."))
            out = ai.write(task, [it["result"] for it in chosen], _lang(), None, extra)
        st.session_state["_rs_draft"] = {"task": task, "ids": pick, "model": llm.label if llm.available else "", **out}
    d = st.session_state.get("_rs_draft")
    if d:
        txt = st.text_area(tr("Draft (editable)", "المسودة (قابلة للتعديل)"), d["text"], height=260, key="rs_w_txt")
        chosen = [it for it in L if it["id"] in d["ids"]]
        _check_box(txt, [it["result"] for it in chosen])
        st.caption(tr("Source: ", "المصدر: ") + (d.get("model", "AI") if d["source"] == "ai" else tr("offline template", "قالب دون اتصال")))
        c1, c2 = st.columns(2)
        if d["task"] == "abstract":
            if c1.button(tr("Use as the report abstract", "استخدامه ملخصاً للتقرير"), key="rs_w_abs"):
                st.session_state.setdefault("_rs_meta", {})["abstract"] = txt
                st.success(tr("Abstract updated.", "تم تحديث الملخص."))
        else:
            target = c1.selectbox(tr("Insert into", "إدراج في"), ["new"] + [it["id"] for it in chosen], key="rs_w_tgt",
                                  format_func=lambda k: tr("a new text block", "فقرة نصية جديدة") if k == "new" else labels.get(k, k))
            if c2.button(tr("➕ Insert", "➕ إدراج"), key="rs_w_ins"):
                if target == "new":
                    names = {"results": tr("Results", "النتائج"), "methodology": tr("Methodology", "المنهجية"), "discussion": tr("Discussion", "المناقشة"),
                             "policy": tr("Policy implications", "الدلالات للسياسات")}
                    store.add_item("text", names.get(d["task"], ""), text=txt)
                else:
                    for it in store.items():
                        if it["id"] == target:
                            it["text"] = (it.get("text", "") + "\n\n" + txt).strip()
                st.success(tr("Inserted — see tab ②.", "تم الإدراج — راجع التبويب ②."))


def _export_tab():
    L = store.items()
    if not L:
        st.info(tr("The report is empty.", "التقرير فارغ."))
        return
    meta = dict(st.session_state.get("_rs_meta", {}))
    meta["note"] = tr("Prepared with Aetheris Research. Results computed in the platform; numbers in the text were checked against them.",
                      "أُعدّ باستخدام منصة Aetheris Research. النتائج محسوبة في المنصة، وفُحصت الأرقام الواردة في النص مقابلها.")
    c1, c2 = st.columns(2)
    lang = c1.radio(tr("Report language / direction", "لغة التقرير واتجاهه"), ["ar", "en"], 0 if is_ar() else 1, horizontal=True, key="rs_lang",
                    format_func=lambda k: {"ar": "العربية (RTL)", "en": "English (LTR)"}[k])
    apa = c2.checkbox(tr("APA style (italic table titles, indented paragraphs)", "نمط APA (عناوين جداول مائلة، ومسافة بادئة للفقرات)"), True, key="rs_apa")
    model = rs.assemble(L, meta, lang)
    st.caption(tr(f"{len([b for b in model['blocks'] if b[0] == 'h1'])} sections · {model['n_tables']} tables · {model['n_figures']} figures · {len(model['references'])} references",
                  f"{len([b for b in model['blocks'] if b[0] == 'h1'])} أقسام · {model['n_tables']} جداول · {model['n_figures']} أشكال · {len(model['references'])} مرجعاً"))
    fname = "".join(ch for ch in (meta.get("title") or "report") if ch.isalnum() or ch in " _-").strip().replace(" ", "_")[:60] or "report"
    cols = st.columns(5)
    fmts = [("docx", "📝 Word", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"), ("pdf", "📄 PDF", "application/pdf"),
            ("zip", "🧾 LaTeX (.zip)", "application/zip"), ("html", "🌐 HTML", "text/html"), ("md", "Ⓜ️ Markdown", "text/markdown")]
    for col, (ext, lab, mime) in zip(cols, fmts):
        with col:
            if st.button(lab, key=f"rs_mk_{ext}", width="stretch"):
                with st.spinner(tr("Building…", "جارٍ الإنشاء…")):
                    data = {"docx": lambda: rs.to_docx(model, apa), "pdf": lambda: rs.to_pdf(model), "zip": lambda: rs.to_latex_zip(model),
                            "html": lambda: rs.to_html(model).encode("utf-8"), "md": lambda: rs.to_markdown(model).encode("utf-8")}[ext]()
                st.session_state[f"_rs_file_{ext}"] = data
            if st.session_state.get(f"_rs_file_{ext}"):
                st.download_button(tr("⬇️ Download", "⬇️ تحميل"), st.session_state[f"_rs_file_{ext}"], f"{fname}.{ext}", mime, key=f"rs_dl_{ext}", width="stretch")
    with st.expander(tr("👁️ Preview", "👁️ معاينة"), expanded=True):
        st.markdown(rs.to_markdown(model)[:20000])
    texts = " ".join([meta.get("abstract", "")] + [it.get("text", "") for it in L])
    if texts.strip():
        bad = rs.check_numbers(texts, [it["result"] for it in L if it["kind"] == "result"])
        if bad:
            st.warning(tr("Numbers in your texts not found in the results: ", "أرقام في نصوصك غير موجودة في النتائج: ") + ", ".join(bad[:15]))
