# ==========================================
# Aetheris Research - AI settings panel (v1.12.1): provider, model, the researcher's own key, platform-key access code
# and usage. Shared by the AI assistant, the Report Studio and the decision dashboard.
# ==========================================

from __future__ import annotations

import streamlit as st

from analytics import ai_providers as ap
from core.utils import tr, is_ar

REASON = {"no_key": ("No key for this provider: enter your own key below (or ask the administrator to add a platform key).",
                     "لا يوجد مفتاح لهذا المزوّد: أدخل مفتاحك أدناه (أو اطلب من المسؤول إضافة مفتاح للمنصة)."),
          "code": ("The platform key is protected: enter the access code, or use your own key.", "مفتاح المنصة محمي: أدخل رمز الدخول، أو استخدم مفتاحك الخاص."),
          "limit": ("Today's limit for the platform key is used up: use your own key or come back tomorrow.", "استُنفد حد اليوم لمفتاح المنصة: استخدم مفتاحك الخاص أو عد غداً.")}


def _t(p):
    return p[1] if is_ar() else p[0]


def status_line(llm) -> str:
    if llm.available:
        src = tr("your key", "مفتاحك") if llm.source == "user" else tr("platform key", "مفتاح المنصة")
        extra = ""
        if llm.source == "server":
            u, L = ap.usage(st.session_state), ap.limits()
            extra = tr(f" · {u['session']}/{L['session']} requests used today", f" · استُخدم {u['session']}/{L['session']} طلباً اليوم")
        return f"🟢 {llm.label} · {src}{extra}"
    return "⚪ " + tr("AI off — offline mode (templates and the methodology library). ", "الذكاء الاصطناعي غير مفعّل — الوضع دون اتصال (القوالب ومكتبة المنهجيات). ") + \
        (_t(REASON[llm.reason]) if llm.reason in REASON else "")


def settings_panel(key: str = "ai", expanded: bool | None = None):
    """Full settings: provider, model, own key, access code, test. Returns the resolved LLM."""
    ss = st.session_state
    llm = ap.resolve(ss)
    with st.expander(tr("⚙️ AI model & key", "⚙️ نموذج الذكاء الاصطناعي والمفتاح"), expanded=(not llm.available) if expanded is None else expanded):
        provs = list(ap.PROVIDERS)
        srv = set(ap.server_providers())
        cur = ss.get("_ai_provider") or ap.default_provider()

        def fmt(p):
            cfg = ap.PROVIDERS[p]
            tags = []
            if cfg["free"]:
                tags.append(tr("free tier", "شريحة مجانية"))
            if p in srv:
                tags.append(tr("platform key", "مفتاح المنصة"))
            return cfg["label"] + (f"  — {' · '.join(tags)}" if tags else "")
        c1, c2 = st.columns([1.3, 1])
        prov = c1.selectbox(tr("Provider", "المزوّد"), provs, provs.index(cur) if cur in provs else 0, key=f"{key}_prov", format_func=fmt)
        if prov != ss.get("_ai_provider"):
            ss["_ai_provider"] = prov
        cfg = ap.PROVIDERS[prov]
        st.caption("ℹ️ " + _t(cfg["note"]))
        if prov == "custom":
            ss["_ai_base_custom"] = st.text_input(tr("Base URL (…/v1)", "العنوان الأساسي (…/v1)"), ss.get("_ai_base_custom", ""), key=f"{key}_base",
                                                  placeholder="https://api.deepseek.com/v1  ·  http://localhost:11434/v1")
        keys = ss.setdefault("_ai_keys", {})
        k = c2.text_input(tr("Your API key (session only, never stored)", "مفتاح API الخاص بك (للجلسة فقط، لا يُخزَّن)"), value=keys.get(prov, ""), type="password",
                          key=f"{key}_key_{prov}")
        if k.strip() != keys.get(prov, ""):
            keys[prov] = k.strip()
        if cfg["keys_url"]:
            c2.markdown(f"[{tr('🔑 Get a key', '🔑 احصل على مفتاح')}]({cfg['keys_url']})")
        fetched = ss.get(f"_ai_models_{prov}") or []
        options = list(dict.fromkeys([m for m in cfg["models"]] + fetched))
        using_server = not keys.get(prov) and prov in srv
        if using_server and not ap.limits()["all_models"] and cfg["cheap"]:
            options = [m for m in options if m in cfg["cheap"]] or cfg["cheap"]
            st.caption(tr("With the platform key only low-cost models are available; with your own key you can choose any model.",
                          "مع مفتاح المنصة تتاح النماذج منخفضة التكلفة فقط؛ وبمفتاحك يمكنك اختيار أي نموذج."))
        m1, m2, m3 = st.columns([1.4, 1, 0.8])
        cur_m = ss.get(f"_ai_model_{prov}")
        choice = m1.selectbox(tr("Model", "النموذج"), options + ["✏️ " + tr("other…", "آخر…")] if options else ["✏️ " + tr("other…", "آخر…")],
                              (options.index(cur_m) if cur_m in options else 0), key=f"{key}_model_{prov}")
        if choice.startswith("✏️"):
            typed = m2.text_input(tr("Model ID", "معرّف النموذج"), cur_m or "", key=f"{key}_mtyped_{prov}")
            if typed:
                ss[f"_ai_model_{prov}"] = typed.strip()
        else:
            ss[f"_ai_model_{prov}"] = choice
        if m3.button(tr("🔄 Fetch models", "🔄 جلب النماذج"), key=f"{key}_fetch_{prov}", disabled=not (keys.get(prov) or prov in srv)):
            try:
                kk = keys.get(prov) or ap._secret(cfg["secret"])
                ss[f"_ai_models_{prov}"] = ap.list_models(prov, kk, ss.get("_ai_base_custom") if prov == "custom" else None)[:200]
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.warning(tr(f"Could not fetch the model list: {e}", f"تعذّر جلب قائمة النماذج: {e}"))
        if ap.access_code() and not keys.get(prov) and prov in srv and ss.get("_ai_code") != ap.access_code():
            code = st.text_input(tr("Access code for the platform key", "رمز الدخول لمفتاح المنصة"), type="password", key=f"{key}_code")
            if code:
                ss["_ai_code"] = code.strip()
                if code.strip() != ap.access_code():
                    st.error(tr("Wrong code.", "الرمز غير صحيح."))
        llm = ap.resolve(ss)
        st.markdown(status_line(llm))
        if llm.available and st.button(tr("🧪 Test connection", "🧪 اختبار الاتصال"), key=f"{key}_test"):
            try:
                with st.spinner("…"):
                    out = llm.complete("Reply with one short sentence.", [{"role": "user", "content": "مرحباً، هل تعمل؟" if is_ar() else "Hello, are you working?"}], max_tokens=60)
                st.success(f"✅ {out[:200]}")
            except Exception as e:  # noqa: BLE001
                st.error(tr(f"Connection failed: {e}", f"فشل الاتصال: {e}"))
        st.caption(tr("Only result tables, interpretations and a short data profile are sent — never your raw data. Your key stays in this browser session.",
                      "لا يُرسل إلا جداول النتائج وتفسيراتها ووصف مختصر للبيانات — ولا تُرسل بياناتك الخام أبداً. ويبقى مفتاحك في جلسة المتصفح هذه فقط."))
    return llm


def compact_status():
    """One-line status for pages that use the AI (Report Studio, dashboard), pointing to the settings."""
    llm = ap.resolve(st.session_state)
    st.caption(status_line(llm) + tr("  ·  change provider / model / key in 🤖 AI Research Assistant", "  ·  غيّر المزوّد أو النموذج أو المفتاح من 🤖 المساعد البحثي الذكي"))
    return llm


def gate(llm, key: str = "gate"):
    """Explain why the AI is unavailable and fix it in place: an access-code field when the platform key is protected,
    the daily-limit notice, or the free-key links. Returns the (possibly now available) LLM."""
    if llm.available:
        return llm
    ss = st.session_state
    if llm.reason == "code":
        st.info(tr("🔐 The platform's AI key is protected. Enter the access code once per session (each time the page is opened or refreshed).",
                   "🔐 مفتاح الذكاء الاصطناعي للمنصة محمي. أدخل رمز الدخول مرة واحدة في كل جلسة (عند كل فتح أو تحديث للصفحة)."))
        code = st.text_input(tr("Access code", "رمز الدخول"), type="password", key=f"{key}_code")
        if code:
            if code.strip() == (ap.access_code() or ""):
                ss["_ai_code"] = code.strip()
                st.rerun()
            st.error(tr("Wrong code.", "الرمز غير صحيح."))
        return llm
    if llm.reason == "limit":
        st.warning(tr("Today's limit for the platform's AI key is used up. Add your own free key in “⚙️ AI model & key”, or come back tomorrow.",
                      "استُنفد حد اليوم لمفتاح المنصة. أضف مفتاحك المجاني في «⚙️ نموذج الذكاء الاصطناعي والمفتاح»، أو عد غداً."))
        return llm
    st.warning(tr("No AI key for the selected provider. A free key is enough — Gemini (Google) or Groq: open “⚙️ AI model & key” above, choose the "
                  "provider and paste the key (or ask the administrator to add a platform key).",
                  "لا يوجد مفتاح للمزوّد المختار. يكفي مفتاح مجاني — Gemini من Google أو Groq: افتح «⚙️ نموذج الذكاء الاصطناعي والمفتاح» في الأعلى، "
                  "واختر المزوّد، والصق المفتاح (أو اطلب من المسؤول إضافة مفتاح للمنصة)."))
    c1, c2 = st.columns(2)
    c1.markdown(f"[{tr('🔑 Free Gemini key', '🔑 مفتاح Gemini مجاني')}](https://aistudio.google.com/apikey)")
    c2.markdown(f"[{tr('🔑 Free Groq key', '🔑 مفتاح Groq مجاني')}](https://console.groq.com/keys)")
    return llm
