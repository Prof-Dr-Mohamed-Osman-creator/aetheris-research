"""v1.7 — multivariate time series, cointegration, ARDL / NARDL / QARDL and volatility models (Catalogue §1.8–1.9):
statistical properties, recovery of simulated parameters, reporting and exported Python code."""
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import ardl as ad
from analytics import garch as gm
from analytics import mts_report as mr
from analytics import var as vr
from data.samples import financial_returns_demo, macro_var_demo, price_transmission_demo

warnings.filterwarnings("ignore")
M = macro_var_demo()
L = pd.DataFrame({"y": np.log(M.gdp_bn), "c": np.log(M.consumption_bn), "i": np.log(M.investment_bn)})
S = pd.DataFrame({"dy": 100 * np.diff(np.log(M.gdp_bn)), "infl": M.inflation_pct[1:].values, "rate": M.policy_rate_pct[1:].values})
PT = price_transmission_demo()
D = pd.DataFrame({"d": np.log(PT.domestic_wheat_egp_t), "x": np.log(PT.import_parity_egp_t), "rain": PT.rainfall_anomaly})
F = financial_returns_demo()
RET = np.column_stack([gm.returns(F[c].values) for c in ("equity_index", "bank_stock_egp", "gold_usd_oz")])


# ---------------- VAR ----------------
def test_var_stability_and_lag_choice():
    sel = vr.var_select(S, 6, "const")
    assert set(sel.attrs["selection"]) == {"AIC", "HQ", "SC", "FPE"}
    r = vr.fit_var(S, 2, "const")
    assert vr.roots(r).max() < 1 and not r.notes
    rs = vr.fit_var(vr.fit_var(L, 1, "none").y, 1, "none")
    assert vr.roots(rs).max() > 0.98


def test_generalised_irf_properties():
    r = vr.fit_var(S, 2, "const")
    o, g = vr.irf(r, 6, "orth"), vr.irf(r, 6, "generalized")
    assert np.allclose(o[:, :, 0], g[:, :, 0])          # first variable: GIRF = Cholesky
    assert np.allclose(vr.fevd(r, 6).sum(2), 1)
    lo, hi = vr.irf_bootstrap(r, 6, runs=60)
    inside = ((o >= lo - 1e-9) & (o <= hi + 1e-9)).mean()
    assert inside > 0.9


def test_toda_yamamoto_and_granger():
    gp, r = vr.toda_yamamoto(S, 2, 1)
    assert r.p == 3 and (gp["df"].isin([2, 4])).all()
    gr = vr.granger_pairs(vr.fit_var(S, 2))
    row = gr[(gr["Dependent"] == "rate") & (gr["Excluded"] == "infl")]
    assert row["p-value"].iloc[0] < 0.01                 # the policy rate reacts to inflation in the DGP


def test_seasonal_and_exogenous_var():
    Q = S.copy()
    r = vr.fit_var(Q, 1, "both", season=4, exog=np.arange(len(Q))[:, None] % 3, exog_names=["ex"])
    assert {"const", "trend", "sd1", "sd2", "sd3", "ex"} <= set(r.reg_names)
    f = vr.forecast(r, 3)
    assert f["mean"].shape == (3, 3)


# ---------------- cointegration ----------------
def test_johansen_null_matches_mhm():
    ref = {(3, 1): 3.84, (3, 2): 15.49, (2, 1): 9.16, (4, 1): 12.52, (1, 2): 12.32}
    for (case, m), v in ref.items():
        assert np.quantile(vr._johansen_null(m, case)[0], 0.95) == pytest.approx(v, abs=0.45)


def test_johansen_rank_and_vecm_beta():
    jo = vr.johansen(L, 3, 3)
    assert jo["rank_trace"] == 2
    vm = vr.vecm(jo, 2)
    b = vm["beta"].to_numpy()
    assert np.allclose(b[:2], np.eye(2))
    assert b[2, 0] == pytest.approx(-1, abs=0.15) and b[2, 1] == pytest.approx(-1, abs=0.15)
    we = vr.weak_exogeneity(jo, 2)
    assert len(we) == 3 and we["df"].eq(2).all()
    assert vr.restriction_test(jo, 1, H=np.array([[1.0], [-1.0], [0.0]]))["df"] == 2


def test_long_run_estimators_recover_slope():
    rng = np.random.default_rng(5)
    T = 400
    x = np.cumsum(rng.normal(size=T))
    v = rng.normal(size=T)
    u = 0.6 * np.r_[0, v[:-1]] + rng.normal(size=T)     # endogenous, autocorrelated errors
    x = x + v
    y = 1 + 2 * x + u
    for est in (vr.fmols(y, x), vr.dols(y, x, "c", 2, 2), vr.ccr(y, x)):
        assert est["coef"]["x1"] == pytest.approx(2, abs=0.05)
    eg = vr.engle_granger(y, x)
    assert eg["p"] < 0.01


def test_gregory_hansen_finds_break():
    rng = np.random.default_rng(9)
    T = 160
    x = np.cumsum(rng.normal(size=T))
    y = 1 + 0.8 * x + 3 * (np.arange(T) >= 90) + rng.normal(0, 0.5, T)
    g = vr.gregory_hansen(y, x, "C")
    t = g["table"].set_index("Test")
    assert t.loc["ADF*", "p-value"] < 0.05 and abs(int(t.loc["ADF*", "Break"]) - 91) <= 4
    assert t.loc["ADF*", "5% cv"] == pytest.approx(-4.61, abs=0.2)        # Gregory-Hansen (1996) Table 1


# ---------------- ARDL ----------------
def test_bounds_table_matches_pss():
    for (case, k), (i0, i1) in {(3, 2): (3.79, 4.85), (2, 1): (3.62, 4.16), (5, 4): (3.47, 4.57), (4, 2): (3.88, 4.61)}.items():
        cv = ad._asym_quantiles(case, k)
        j = int(np.argmin(np.abs(ad.ALPHAS - 0.05)))
        assert cv["F0"][j] == pytest.approx(i0, abs=0.1) and cv["F1"][j] == pytest.approx(i1, abs=0.1)


def test_ardl_selection_bounds_ecm():
    sel = ad.auto_ardl(D, "d", ["x"], (3, 3), 3, ["rain"])
    assert sel["AIC"].is_monotonic_increasing
    r = ad.fit_ardl(D, "d", ["x"], sel["order"].iloc[0], 3, ["rain"])
    b = ad.bounds_test(r)
    assert b["verdict_F"] == "cointegration"
    rc = ad.recm(r)
    assert -1 < rc["speed"] < 0 and rc["half_life"] > 0
    m = ad.dynamic_multipliers(r, "x", 400)
    assert m[-1] == pytest.approx(ad.long_run(r).set_index("Variable").loc["x", "Long-run coef."], rel=1e-3)
    ex = ad.bounds_test(r, exact=True)
    assert ex["critical"]["F I(1)"].iloc[1] > b["critical"]["F I(1)"].iloc[1] - 0.3


def test_nardl_asymmetry():
    N, pairs = ad.nardl_data(D, ["x"])
    assert np.allclose(N["x_pos"] + N["x_neg"], D["x"] - D["x"].iloc[0])
    r = ad.fit_ardl(N, "d", ["x_pos", "x_neg"], (2, 2, 2), 3, ["rain"])
    lr = ad.long_run(r).set_index("Variable")["Long-run coef."]
    assert lr["x_pos"] > lr["x_neg"]                     # DGP: 0.85 vs 0.60
    at = ad.asymmetry_tests(r, pairs)
    assert at.iloc[0]["p-value"] < 0.05
    mb = ad.multiplier_bands(r, N, pairs, 12, 50)
    assert mb["x"]["lower"].shape == (13,)


def test_qardl():
    q = ad.qardl(D, "d", ["x"], (2, 1), (0.25, 0.5, 0.75))
    lr = q["long_run"]
    assert set(lr["τ"]) == {0.25, 0.5, 0.75}
    assert (lr[lr["Variable"] == "x"]["Long-run β(τ)"] > 0).all()


# ---------------- GARCH ----------------
def _sim_garch(n=3000, w=0.05, a=0.08, b=0.9, seed=3):
    rng = np.random.default_rng(seed)
    h, e = w / (1 - a - b), 0.0
    out = np.empty(n)
    for t in range(n):
        h = w + a * e ** 2 + b * h
        e = np.sqrt(h) * rng.standard_normal()
        out[t] = e
    return out


def test_garch_recovers_parameters():
    g = gm.fit_garch(_sim_garch(), "garch", 1, 1, "norm", "zero")
    assert g.p_("alpha1") == pytest.approx(0.08, abs=0.03) and g.p_("beta1") == pytest.approx(0.9, abs=0.04)
    fv = gm.forecast_variance(g, 500)
    assert fv["variance"].iloc[-1] == pytest.approx(g.stats["Unconditional variance"], rel=0.02)


def test_asymmetric_models_on_demo():
    gj = gm.fit_garch(RET[:, 0], "gjr", 1, 1, "std")
    eg = gm.fit_garch(RET[:, 0], "egarch", 1, 1, "norm")
    assert gj.p_("gamma1") > 0 and eg.p_("gamma1") < 0
    ni = gm.news_impact(gj)
    assert ni["sigma2"].iloc[0] > ni["sigma2"].iloc[-1]   # bad news raises volatility more
    dg = gm.garch_diagnostics(gj)
    assert dg.set_index("Test").loc["Ljung-Box Q(10) on z²", "p-value"] > 0.01
    v = gm.var_es(gj, 0.05)
    assert v["ES"] < v["VaR"] < 0 and abs(v["violations"] / len(v["hits"]) - 0.05) < 0.02


def test_garch_m_and_ar_mean():
    gmm = gm.fit_garch(RET[:800, 0], "garch", 1, 1, "norm", "m_sd")
    assert "lambda" in gmm.names and np.isfinite(gmm.llf)
    ga = gm.fit_garch(RET[:800, 0], "garch", 1, 1, "norm", "ar1")
    assert len(ga.resid) == 799


def test_dcc_ccc_bekk():
    d = gm.fit_dcc(RET[:, :2], ["eq", "bank"], "garch", "norm")
    assert d.params["a"] == pytest.approx(0.04, abs=0.03) and d.params["b"] == pytest.approx(0.93, abs=0.06)
    c = gm.fit_ccc(RET[:, :2], ["eq", "bank"])
    assert c.params["R"][0, 1] == pytest.approx(0.62, abs=0.08) and d.llf > c.llf
    c3 = gm.fit_ccc(RET, ["eq", "bank", "gold"])
    es = gm.engle_sheppard(np.column_stack([u.z for u in c3.uni]), 5)
    assert es["p"] < 0.01                                 # correlations are dynamic in the DGP
    b = gm.fit_bekk(RET[:700, :2], ["eq", "bank"], "scalar")
    assert 0 < b.params["a"] ** 2 + b.params["b"] ** 2 < 1
    assert gm.hedge_ratio(d, 1, 0).shape == (len(RET),)


# ---------------- reporting ----------------
@pytest.mark.parametrize("lang", ["en", "ar"])
def test_reporting(lang):
    for k in ("var", "svar", "coint", "vecm", "ardl", "nardl", "qardl", "garch", "mgarch"):
        txt, refs = mr.methodology(k, lang, r=1)
        assert len(txt) > 100 and refs
    assert "VECM" in mr.model_advice({"a": 1, "b": 1}, lang)
    assert "ARDL" in mr.model_advice({"a": 0, "b": 1}, lang)
    jo = vr.johansen(L, 3, 3)
    assert len(mr.interpret_johansen(jo, lang)) > 40
    r = ad.fit_ardl(D, "d", ["x"], (2, 1), 3)
    assert len(mr.interpret_bounds(ad.bounds_test(r), lang)) > 40


@pytest.mark.parametrize("kind", ["var", "coint", "vecm", "ardl", "garch"])
def test_exported_python_runs(kind, tmp_path):
    df = pd.DataFrame({"y": L.y, "c": L.c, "i": L.i})
    kw = {"vars": ["y", "c", "i"], "p": 2, "case": 3, "K": 2, "r": 2, "y": "y", "x": ["c", "i"], "order": (2, 1, 1)}
    if kind == "garch":
        df = pd.DataFrame({"price": F.equity_index[:600]})
        kw = {"y": "price", "model": "gjr", "dist": "std"}
    df.to_csv(tmp_path / "d.csv", index=False)
    code = mr.code(kind, "d.csv", **kw)
    assert set(code) == {"Python", "R", "Stata"}
    (tmp_path / "m.py").write_text(code["Python"].replace("res.irf(10).plot(orth=True); ", "").replace("m.irf(10).plot(orth=True)", ""))
    p = subprocess.run([sys.executable, "m.py"], cwd=tmp_path, capture_output=True, text=True,
                       env={"MPLBACKEND": "Agg", "PATH": "/usr/bin:/bin"})
    assert p.returncode == 0, p.stderr[-1500:]
