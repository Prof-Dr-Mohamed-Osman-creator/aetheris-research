# ==========================================
# Aetheris Research - AI providers and access governance (v1.12.1)
# Several AI providers behind one interface: Claude (Anthropic), ChatGPT (OpenAI), Gemini (Google, free tier), Groq and
# OpenRouter (free open models) and any OpenAI-compatible endpoint (DeepSeek, Mistral, a local Ollama …).
# Key resolution: the researcher's own key (kept only in the session) comes first; otherwise the platform's key from
# the app secrets is used under controls set by the administrator: an optional access code, a daily request limit
# per session and for the whole platform, and (by default) only low-cost models.
#
# Secrets (Streamlit → App settings → Secrets, or environment variables), all optional:
#   ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY
#   AETHERIS_AI_ACCESS_CODE          code users must enter to use the platform keys (empty = no code)
#   AETHERIS_AI_SESSION_DAILY        requests per session per day with platform keys (default 20)
#   AETHERIS_AI_GLOBAL_DAILY         requests per day for the whole platform with platform keys (default 300)
#   AETHERIS_AI_SERVER_ALL_MODELS    "1" lets platform keys use every model (default: low-cost models only)
#   AETHERIS_AI_DEFAULT_PROVIDER     default provider shown to users (default: first provider with a platform key)
# ==========================================

from __future__ import annotations

import datetime as _dt
import os

PROVIDERS = {
    "anthropic": {"label": "Claude (Anthropic)", "kind": "anthropic", "secret": "ANTHROPIC_API_KEY", "free": False,
                  "models": ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"], "cheap": ["claude-haiku-5-5"],
                  "keys_url": "https://platform.claude.com/settings/keys",
                  "note": ("Paid per use (no free API tier). Haiku is the low-cost option.", "مدفوع حسب الاستخدام (لا توجد شريحة مجانية للـ API). Haiku هو الخيار منخفض التكلفة.")},
    "openai": {"label": "ChatGPT (OpenAI)", "kind": "openai", "base": "https://api.openai.com/v1", "secret": "OPENAI_API_KEY", "free": False,
               "models": ["gpt-5.6-luna", "gpt-5.6-terra", "gpt-6.2-sol"], "cheap": ["gpt-5.6-luna", "gpt-5-mini", "gpt-5-nano"],
               "keys_url": "https://platform.openai.com/api-keys",
               "note": ("Paid per use (no free API tier). Luna / mini models are the low-cost options.", "مدفوع حسب الاستخدام (لا توجد شريحة مجانية للـ API). نماذج Luna وmini منخفضة التكلفة.")},
    "gemini": {"label": "Gemini (Google)", "kind": "openai", "base": "https://generativelanguage.googleapis.com/v1beta/openai", "secret": "GEMINI_API_KEY",
               "free": True, "models": ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-pro"], "cheap": ["gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash"],
               "keys_url": "https://aistudio.google.com/apikey",
               "note": ("Free tier available (with daily limits). On the free tier Google may use the content to improve its products — do not send confidential results.",
                        "تتوفر شريحة مجانية (بحدود يومية). في الشريحة المجانية قد تستخدم Google المحتوى لتحسين منتجاتها — لا ترسل نتائج سرية.")},
    "groq": {"label": "Groq (open models)", "kind": "openai", "base": "https://api.groq.com/openai/v1", "secret": "GROQ_API_KEY", "free": True,
             "models": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b"], "cheap": ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b"],
             "keys_url": "https://console.groq.com/keys",
             "note": ("Free plan with rate limits; very fast open models (Llama, gpt-oss).", "خطة مجانية بحدود للمعدل؛ نماذج مفتوحة سريعة جداً (Llama وgpt-oss).")},
    "openrouter": {"label": "OpenRouter (many models; free ':free' models)", "kind": "openai", "base": "https://openrouter.ai/api/v1", "secret": "OPENROUTER_API_KEY",
                   "free": True, "models": ["openrouter/free", "deepseek/deepseek-r1:free", "qwen/qwen3-next-80b-a3b-instruct:free"], "cheap": ["openrouter/free"],
                   "keys_url": "https://openrouter.ai/keys",
                   "note": ("Free models end in ':free' (about 50 requests/day without credit); 'openrouter/free' picks one automatically. Paid models are also available.",
                            "النماذج المجانية تنتهي بـ ':free' (نحو 50 طلباً يومياً دون رصيد)؛ و'openrouter/free' يختار أحدها تلقائياً. وتتوفر نماذج مدفوعة أيضاً.")},
    "custom": {"label": "Other (OpenAI-compatible: DeepSeek, Mistral, Ollama …)", "kind": "openai", "base": "", "secret": None, "free": False,
               "models": [], "cheap": [], "keys_url": "",
               "note": ("Any service with an OpenAI-compatible /chat/completions endpoint: enter its base URL, model ID and key.",
                        "أي خدمة توفر واجهة /chat/completions متوافقة مع OpenAI: أدخل عنوانها الأساسي ومعرّف النموذج والمفتاح.")},
}


def _secret(name: str | None):
    if not name:
        return None
    try:
        import streamlit as st
        v = st.secrets.get(name)
        if v:
            return str(v)
    except Exception:  # noqa: BLE001
        pass
    return os.environ.get(name) or None


def _int_secret(name, default):
    try:
        return int(_secret(name) or default)
    except ValueError:
        return default


def server_providers() -> list:
    return [p for p, cfg in PROVIDERS.items() if _secret(cfg["secret"])]


def default_provider() -> str:
    d = _secret("AETHERIS_AI_DEFAULT_PROVIDER")
    if d in PROVIDERS:
        return d
    sp = server_providers()
    return sp[0] if sp else "gemini"


def access_code() -> str | None:
    return _secret("AETHERIS_AI_ACCESS_CODE")


def limits() -> dict:
    return {"session": _int_secret("AETHERIS_AI_SESSION_DAILY", 20), "global": _int_secret("AETHERIS_AI_GLOBAL_DAILY", 300),
            "all_models": (_secret("AETHERIS_AI_SERVER_ALL_MODELS") or "0") == "1"}


# ----------------------------------------------------------------------------------------------------------------------
# usage counters (per session in session_state; platform-wide in a process-wide store shared by all sessions)
# ----------------------------------------------------------------------------------------------------------------------
_GLOBAL = {"day": None, "count": 0}


def _global_store():
    try:
        import streamlit as st

        @st.cache_resource
        def _store():
            return {"day": None, "count": 0}
        return _store()
    except Exception:  # noqa: BLE001
        return _GLOBAL


def _today():
    return _dt.date.today().isoformat()


def usage(session: dict) -> dict:
    g = _global_store()
    if g["day"] != _today():
        g["day"], g["count"] = _today(), 0
    s = session.setdefault("_ai_usage", {"day": _today(), "count": 0})
    if s["day"] != _today():
        s["day"], s["count"] = _today(), 0
    return {"session": s["count"], "global": g["count"]}


def _count(session: dict):
    usage(session)
    session["_ai_usage"]["count"] += 1
    _global_store()["count"] += 1


# ----------------------------------------------------------------------------------------------------------------------
# clients
# ----------------------------------------------------------------------------------------------------------------------
class AIError(RuntimeError):
    pass


def _openai_compat(base, key, model, system, messages, max_tokens, temperature, provider):
    import requests
    url = base.rstrip("/") + "/chat/completions"
    body = {"model": model, "messages": [{"role": "system", "content": system}] + list(messages)}
    if provider == "openai":                      # current OpenAI models: max_completion_tokens, default temperature only
        body["max_completion_tokens"] = max_tokens
    else:
        body["max_tokens"] = max_tokens
        body["temperature"] = temperature
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    if provider == "openrouter":
        headers.update({"HTTP-Referer": "https://aetheris-research.streamlit.app", "X-Title": "Aetheris Research"})
    r = requests.post(url, json=body, headers=headers, timeout=180)
    if r.status_code >= 400:
        try:
            msg = r.json().get("error", {})
            msg = msg.get("message") if isinstance(msg, dict) else str(msg)
        except Exception:  # noqa: BLE001
            msg = r.text[:300]
        raise AIError(f"{provider} HTTP {r.status_code}: {msg}")
    data = r.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise AIError(f"{provider}: unexpected response") from e
    if isinstance(content, list):
        content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return (content or "").strip()


def list_models(provider: str, key: str, base: str | None = None) -> list:
    """Live model list from the provider (OpenAI-compatible /models, or Anthropic /v1/models)."""
    import requests
    cfg = PROVIDERS[provider]
    if cfg["kind"] == "anthropic":
        r = requests.get("https://api.anthropic.com/v1/models", headers={"x-api-key": key, "anthropic-version": "2023-06-01"}, timeout=30)
    else:
        b = (base or cfg.get("base") or "").rstrip("/")
        r = requests.get(b + "/models", headers={"Authorization": f"Bearer {key}"}, timeout=30)
    if r.status_code >= 400:
        raise AIError(f"HTTP {r.status_code}")
    ids = [m.get("id") for m in r.json().get("data", []) if isinstance(m, dict) and m.get("id")]
    ids = [i.split("/", 1)[1] if i.startswith("models/") else i for i in ids]
    if provider == "openrouter":
        ids = sorted(ids, key=lambda i: (not i.endswith(":free"), i))
    return ids


class LLM:
    """One chat model. `source` is 'user' (the researcher's key), 'server' (the platform key, counted against the daily
    limits) or None (unavailable). `client` lets tests inject a fake Anthropic-style client."""

    def __init__(self, provider: str = "anthropic", model: str | None = None, key: str | None = None, source: str | None = None, base: str | None = None,
                 client=None, session: dict | None = None, reason: str | None = None):
        self.provider = provider
        cfg = PROVIDERS.get(provider, PROVIDERS["custom"])
        self.model = model or (cfg["models"][0] if cfg["models"] else "")
        self.key = key
        self.source = source if (key or client is not None) else None
        self.base = base or cfg.get("base")
        self._client = client
        self._session = session
        self.reason = reason

    @property
    def available(self) -> bool:
        return self._client is not None or (bool(self.key) and bool(self.model))

    @property
    def label(self) -> str:
        return f"{PROVIDERS.get(self.provider, {}).get('label', self.provider)} · {self.model}"

    def complete(self, system: str, messages: list, max_tokens: int = 1500, temperature: float = 0.2) -> str:
        if self.source == "server" and self._session is not None:
            u, L = usage(self._session), limits()
            if u["session"] >= L["session"] or u["global"] >= L["global"]:
                raise AIError("daily limit for the platform AI key reached — add your own key or try tomorrow")
        if self._client is not None or PROVIDERS.get(self.provider, {}).get("kind") == "anthropic":
            if self._client is None:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.key)
            resp = self._client.messages.create(model=self.model, system=system, messages=messages, max_tokens=max_tokens, temperature=temperature)
            out = "".join(getattr(b, "text", "") for b in resp.content).strip()
        else:
            if not self.base:
                raise AIError("no base URL for this provider")
            out = _openai_compat(self.base, self.key, self.model, system, messages, max_tokens, temperature, self.provider)
        if self.source == "server" and self._session is not None:
            _count(self._session)
        return out


def resolve(session: dict, provider: str | None = None, model: str | None = None) -> LLM:
    """Build the LLM for this session: the user's own key first, else the platform key under the controls."""
    provider = provider or session.get("_ai_provider") or default_provider()
    cfg = PROVIDERS.get(provider, PROVIDERS["custom"])
    model = model or session.get(f"_ai_model_{provider}") or (cfg["models"][0] if cfg["models"] else "")
    base = session.get("_ai_base_custom") if provider == "custom" else cfg.get("base")
    user_key = (session.get("_ai_keys") or {}).get(provider)
    if user_key:
        return LLM(provider, model, user_key, "user", base, session=session)
    srv = _secret(cfg.get("secret"))
    if not srv:
        return LLM(provider, model, None, None, base, reason="no_key")
    code = access_code()
    if code and session.get("_ai_code") != code:
        return LLM(provider, model, None, None, base, reason="code")
    L = limits()
    if not L["all_models"] and cfg["cheap"] and model not in cfg["cheap"]:
        model = cfg["cheap"][0]                      # platform keys: low-cost models only
    u = usage(session)
    if u["session"] >= L["session"] or u["global"] >= L["global"]:
        return LLM(provider, model, None, None, base, reason="limit")
    return LLM(provider, model, srv, "server", base, session=session)
