"""v1.12.1 — multi-provider AI layer: key resolution (researcher key first, then the platform key), access code, daily
limits, low-cost-model enforcement for platform keys, the OpenAI-compatible request format and model listing.
All network calls are mocked."""
import pytest

from analytics import ai_assistant as A
from analytics import ai_providers as ap


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for v in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "AETHERIS_AI_ACCESS_CODE",
              "AETHERIS_AI_SESSION_DAILY", "AETHERIS_AI_GLOBAL_DAILY", "AETHERIS_AI_SERVER_ALL_MODELS", "AETHERIS_AI_DEFAULT_PROVIDER"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(ap, "_global_store", lambda: ap._GLOBAL)
    ap._GLOBAL.update(day=None, count=0)


class _Resp:
    def __init__(self, code, data):
        self.status_code, self._d, self.text = code, data, str(data)

    def json(self):
        return self._d


def test_no_key_means_offline():
    llm = ap.resolve({}, "gemini")
    assert not llm.available and llm.reason == "no_key"
    assert ap.default_provider() == "gemini"


def test_user_key_has_priority_and_any_model(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-key")
    ss = {"_ai_keys": {"openai": "user-key"}, "_ai_model_openai": "gpt-6.2-sol"}
    llm = ap.resolve(ss, "openai")
    assert (llm.key, llm.source, llm.model) == ("user-key", "user", "gpt-6.2-sol")


def test_platform_key_cheap_models_code_and_limits(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "srv")
    ss = {"_ai_model_anthropic": "claude-opus-5-5"}
    llm = ap.resolve(ss, "anthropic")
    assert llm.source == "server" and llm.model == "claude-haiku-5-5"          # forced to the low-cost model
    monkeypatch.setenv("AETHERIS_AI_SERVER_ALL_MODELS", "1")
    assert ap.resolve(ss, "anthropic").model == "claude-opus-5-5"
    monkeypatch.setenv("AETHERIS_AI_ACCESS_CODE", "1234")
    assert ap.resolve(ss, "anthropic").reason == "code"
    ss["_ai_code"] = "1234"
    assert ap.resolve(ss, "anthropic").available
    monkeypatch.setenv("AETHERIS_AI_SESSION_DAILY", "2")

    class Fake:
        class messages:
            @staticmethod
            def create(**kw):
                class B:
                    text = "ok"

                class R:
                    content = [B()]
                return R()
    for _ in range(2):
        llm = ap.resolve(ss, "anthropic")
        llm._client = Fake()
        assert llm.complete("s", [{"role": "user", "content": "x"}]) == "ok"
    assert ss["_ai_usage"]["count"] == 2 and ap._GLOBAL["count"] == 2
    assert ap.resolve(ss, "anthropic").reason == "limit"
    ss["_ai_keys"] = {"anthropic": "mine"}                                        # own key is not limited
    assert ap.resolve(ss, "anthropic").source == "user"


def test_global_limit(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "srv")
    monkeypatch.setenv("AETHERIS_AI_GLOBAL_DAILY", "1")
    ap.usage({})
    ap._GLOBAL["count"] = 1
    assert ap.resolve({}, "gemini").reason == "limit"


def test_openai_compatible_request(monkeypatch):
    import requests
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append((url, json, headers))
        return _Resp(200, {"choices": [{"message": {"content": " مرحباً "}}]})
    monkeypatch.setattr(requests, "post", fake_post)
    for prov in ("gemini", "groq", "openrouter", "openai"):
        llm = ap.LLM(prov, None, "k", "user")
        assert llm.complete("SYS", [{"role": "user", "content": "q"}], max_tokens=50) == "مرحباً"
    urls = [c[0] for c in calls]
    assert urls[0] == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert urls[1].startswith("https://api.groq.com/openai/v1") and urls[2].startswith("https://openrouter.ai/api/v1")
    assert calls[0][1]["messages"][0] == {"role": "system", "content": "SYS"} and calls[0][1]["max_tokens"] == 50
    assert "max_completion_tokens" in calls[3][1] and "temperature" not in calls[3][1]       # OpenAI current models
    assert calls[2][2]["X-Title"] == "Aetheris Research" and calls[0][2]["Authorization"] == "Bearer k"
    llm = ap.LLM("custom", "deepseek-chat", "k", "user", base="https://api.deepseek.com/v1")
    llm.complete("s", [{"role": "user", "content": "q"}])
    assert calls[-1][0] == "https://api.deepseek.com/v1/chat/completions"


def test_errors_and_model_listing(monkeypatch):
    import requests
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(429, {"error": {"message": "quota exceeded"}}))
    with pytest.raises(ap.AIError, match="429.*quota"):
        ap.LLM("gemini", None, "k", "user").complete("s", [])
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(200, {"data": [{"id": "a/b"}, {"id": "x:free"}, {"id": "models/gemini-3.5-flash"}]}))
    assert ap.list_models("openrouter", "k") == ["x:free", "a/b", "gemini-3.5-flash"]   # free first, "models/" prefix stripped
    assert "gemini-3.5-flash" in ap.list_models("gemini", "k")


def test_assistant_uses_any_provider(monkeypatch):
    import requests
    import pandas as pd
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent["body"] = json
        return _Resp(200, {"choices": [{"message": {"content": "The ATT is 0.1746 (SE 0.0105)."}}]})
    monkeypatch.setattr(requests, "post", fake_post)
    res = {"title": "DiD", "sections": [("table", pd.DataFrame({"Term": ["ATT"], "Estimate": [0.17459], "Std. Err.": [0.01049]}))], "insights": []}
    out = A.write("results", [res], "en", ap.LLM("groq", None, "k", "user"))
    assert out["source"] == "ai" and out["unverified"] == [] and "NEVER invent" in sent["body"]["messages"][0]["content"]
