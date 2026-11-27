"""v1.8 panel data: properties, simulations and reporting (engines with an R counterpart are validated in
test_r_crosscheck_v18.py)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import panel as pn
from analytics import panel_adv as pa
from analytics import panel_gmm as pg
from analytics import panel_report as pr
from data.samples import firm_panel_demo, macro_panel_demo

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def macro():
    M = macro_panel_demo()
    M = M.assign(lc=np.log(M.co2_t_pc), ly=np.log(M.gdp_pc_usd), le=np.log(M.energy_kgoe_pc))
    return pn.prepare(M, "country", "year", "lc", ["ly", "le"])


def _sim_panel(N, T, beta, seed, factor=0.0, gamma=None):
    rng = np.random.default_rng(seed)
    f = rng.normal(size=T)
    rows = []
    for i in range(N):
        a, lam = rng.normal(), rng.normal()
        x = rng.normal(size=T) + factor * lam * f + a
        q = rng.uniform(0, 1, T)
        b = beta if gamma is None else np.where(q <= gamma, beta, beta + 1.0)
        y = a + b * x + factor * lam * f + rng.normal(scale=0.5, size=T)
        rows.append(pd.DataFrame({"id": f"u{i}", "t": np.arange(T), "y": y, "x": x, "q": q}))
    return pd.concat(rows, ignore_index=True)


def test_demos_have_panel_structure():
    F, M = firm_panel_demo(), macro_panel_demo()
    assert F.groupby("firm_id").size().nunique() == 1 and F.firm_id.nunique() == 400
    assert M.groupby("country").size().eq(40).all() and M.country.nunique() == 20
    P = pn.prepare(F, "firm_id", "year", "ln_sales", ["ln_capital", "ln_labour"])
    d = pn.describe(P)
    assert d["balanced"] and d["N"] == 400 and d["T"] == 8 and set(d["xtsum"].columns) >= {"SD between", "SD within"}


def test_prepare_rejects_duplicates_and_accepts_two_ids():
    df = pd.DataFrame({"g": ["a", "a", "b", "b"] * 3, "c": ["x", "y"] * 6, "t": np.repeat([1, 2, 3], 4), "y": np.arange(12.0), "x": np.arange(12.0) ** 0.5})
    with pytest.raises(ValueError):
        pn.prepare(df, "g", "t", "y", ["x"])
    P = pn.prepare(df, ["g", "c"], "t", "y", ["x"])
    assert len(P.units) == 4


def test_fe_wipes_out_correlated_effects():
    D = _sim_panel(60, 10, 1.5, 3)
    P = pn.prepare(D, "id", "t", "y", ["x"])
    fe, po = pn.fit(P, "fe"), pn.fit(P, "pooled")
    assert abs(fe.b[0] - 1.5) < 4 * fe.se[0]
    assert abs(po.b[1] - 1.5) > abs(fe.b[0] - 1.5)
    assert pn.hausman(fe, pn.fit(P, "re"))["p"] < 0.05


def test_pmg_recovers_common_long_run(macro):
    r = pa.panel_ardl(macro, 1, 1)
    lr = r["pmg"]["long_run"].set_index("Variable")["Coef."]
    assert abs(lr["ly"] - 0.30) < 0.06 and abs(lr["le"] - 0.80) < 0.08
    assert (r["pmg"]["phi_i"] < 0).all()
    assert r["hausman_mg_pmg"]["p"] > 0.05


def test_cointegration_tests_reject_on_cointegrated_demo(macro):
    assert (pa.kao(macro)["p-value"] < 0.05).all()
    assert (pa.pedroni(macro, reps=150)["p-value"] < 0.05).sum() >= 4
    assert (pa.westerlund(macro, reps=150)["p-value"] < 0.05).sum() >= 2


def test_interactive_fe_recovers_beta_under_common_factor():
    D = _sim_panel(40, 30, 1.5, 5, factor=2.0)
    P = pn.prepare(D, "id", "t", "y", ["x"])
    r = pa.interactive_fe(P, 1)
    b = r["table"].set_index("Variable").loc["x", "Coef."]
    assert abs(b - 1.5) < 0.05
    assert abs(pn.fit(P, "fe").b[0] - 1.5) > abs(b - 1.5)


def test_threshold_recovers_gamma():
    D = _sim_panel(50, 10, 1.0, 8, gamma=0.5)
    P = pn.prepare(D, "id", "t", "y", ["x", "q"])
    r = pa.panel_threshold(P, ["x"], "q", boot=50)
    assert abs(r["gamma"] - 0.5) < 0.05 and r["ci"][0] <= r["gamma"] <= r["ci"][1] and r["p_boot"] < 0.05


def test_panel_var_recovers_coefficients():
    rng = np.random.default_rng(4)
    A = np.array([[0.5, 0.1], [0.2, 0.3]])
    rows = []
    for i in range(300):
        mu = rng.normal(size=2); z = np.zeros((12, 2))
        for t in range(1, 12):
            z[t] = mu + A @ z[t - 1] + rng.normal(size=2)
        rows.append(pd.DataFrame({"id": i, "t": range(12), "a": z[:, 0], "b": z[:, 1]}))
    P = pn.prepare(pd.concat(rows), "id", "t", "a", ["b"])
    r = pa.panel_var(P, ["a", "b"], 1, 3)
    assert np.max(np.abs(r["A"][0] - A)) < 0.1 and r["roots"].max() < 1
    base, lo, hi = pa.pvar_irf(r, 6, 50)
    assert base.shape == lo.shape == hi.shape and np.all(lo <= hi + 1e-12)


def test_gmm_long_run_and_report():
    F = firm_panel_demo()
    spec = pg.GmmSpec("ln_sales", [1], {"ln_capital": [0], "ln_labour": [0]}, {"ln_sales": (2, 99), "ln_labour": (2, 99), "ln_capital": (1, 99)})
    r = pg.fit(F, "firm_id", "year", spec, "twosteps", "ld", "twoways", True)
    assert 0.3 < r.coef[0] < 0.7 and r.tests["AR(2)"][2] > 0.05
    lr = pg.long_run(r)
    assert len(lr) == 2 and np.isfinite(lr["Long-run effect" if "Long-run effect" in lr else lr.columns[1]]).all()
    for lang in ("en", "ar"):
        assert len(pr.interpret_gmm(r, lang)) > 50


def test_cips_and_dh_guards(macro):
    S = pn.series_by_unit(macro, "lc")
    assert np.isfinite(pn.cips(S, 1, "trend")["stat"])
    with pytest.raises(ValueError):
        pn.cips({k: v[:8] for k, v in S.items()}, 1, "drift")
    dh = pn.dumitrescu_hurlin(macro, "le", "lc", 1)
    assert dh["Wbar"] > 0 and len(dh["W_i"]) == 20


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_report_texts_and_code(lang):
    a = pr.advice({"F": 0.001, "LM": 0.001, "H": 0.01, "CD": 0.2, "AR": 0.01, "HET": 0.02}, lang)
    assert ("FE" in a) or ("الثابتة" in a)
    for kind in ("static", "unitroot", "gmm", "pardl", "advanced"):
        txt, refs = pr.methodology(kind, lang)
        assert len(txt) > 100 and refs
        c = pr.code(kind, unit="id", time="t", y="y", x=["x1", "x2"])
        assert set(c) == {"Python", "R", "Stata"} and all(len(v) > 20 for v in c.values())


def test_fgls_matches_explicit_gls(macro):
    from scipy.linalg import block_diag
    y = macro.df[macro.y].to_numpy(); X = np.column_stack([np.ones(len(y)), macro.df[macro.x].to_numpy()])
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    E = e.reshape(macro.N, macro.T)
    for panels, Om in (("hetero", np.diag(np.repeat((E ** 2).mean(1), macro.T))), ("correlated", np.kron(E @ E.T / macro.T, np.eye(macro.T)))):
        Oi = np.linalg.inv(Om)
        V = np.linalg.inv(X.T @ Oi @ X); b = V @ X.T @ Oi @ y
        r = pn.fgls(macro, panels)
        np.testing.assert_allclose(r.b, b, rtol=1e-8); np.testing.assert_allclose(r.V, V, rtol=1e-7)
    del block_diag
