"""v1.12.3 — free AI conversation: message building with optional context, detection of references in answers and
their verification against Crossref (mocked)."""
from analytics import free_chat as fc

ANSWER = """There are three key studies:
1. Schlenker, W. & Roberts, M. J. (2009). Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change. PNAS, 106(37).
2. Mendelsohn, R., Nordhaus, W. D. & Shaw, D. (1994). The impact of global warming on agriculture: a Ricardian analysis. American Economic Review, 84(4).
- Doe, J. (2021). A completely invented study on imaginary wheat dynamics in Atlantis. Journal of Nothing, 1(1).
Prices rose in 2020 by a lot."""


class _R:
    def __init__(self, d, code=200):
        self._d, self.status_code = d, code

    def json(self):
        return self._d


class _Sess:
    DB = [{"DOI": "10.1073/pnas.0906865106", "title": ["Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change"],
           "issued": {"date-parts": [[2009, 9, 15]]}, "author": [{"family": "Schlenker"}, {"family": "Roberts"}], "container-title": ["PNAS"]},
          {"DOI": "10.2307/2118029", "title": ["The Impact of Global Warming on Agriculture: A Ricardian Analysis"], "issued": {"date-parts": [[1994]]},
           "author": [{"family": "Mendelsohn"}], "container-title": ["American Economic Review"]}]

    def get(self, url, params=None, timeout=None):
        q = params["query.bibliographic"].lower()
        hits = [d for d in self.DB if d["author"][0]["family"].lower() in q] or [{"DOI": "10.1/zzz", "title": ["Unrelated paper about fish markets"],
                                                                                   "issued": {"date-parts": [[2003]]}, "author": [], "container-title": [""]}]
        return _R({"message": {"items": hits}})


def test_find_references():
    refs = fc.find_references(ANSWER)
    assert len(refs) == 3 and refs[0].startswith("Schlenker") and refs[2].startswith("Doe")


def test_verify_against_crossref():
    s = _Sess()
    r = [fc.verify_reference(x, session=s) for x in fc.find_references(ANSWER)]
    assert [x["status"] for x in r] == ["verified", "verified", "not found"]
    assert r[0]["doi"] == "10.1073/pnas.0906865106"
    wrong_year = fc.verify_reference("Schlenker, W. & Roberts, M. J. (2011). Nonlinear temperature effects indicate severe damages to U.S. crop yields "
                                     "under climate change. PNAS.", session=s)
    assert wrong_year["status"] == "partial"
    err = fc.verify_reference("X (2000). Title here long enough.", session=type("S", (), {"get": lambda *a, **k: _R({}, 503)})())
    assert err["status"] == "error"


def test_messages_and_export():
    hist = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    m = fc.build_messages(hist, "q2", {"dataset_profile": {"rows": 10}}, "ar")
    assert m[:2] == hist and m[-1]["content"].endswith("q2") and "السياق" in m[-1]["content"]
    assert fc.build_messages(hist * 20, "q", None, "en", max_turns=3)[-1] == {"role": "user", "content": "q"}
    assert len(fc.build_messages(hist * 20, "q", None, "en", max_turns=3)) == 7
    for lang in ("ar", "en"):
        assert "never invent" in fc.SYSTEM["en"] and "لا تخترع" in fc.SYSTEM["ar"]
        assert "**" in fc.to_markdown(hist, lang)
