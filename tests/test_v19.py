"""v1.9 forecasting: properties and simulations for the engines without an R counterpart in this environment
(BSTS, state space, ML, Prophet-style, combination, horse-race, bridge / MIDAS / DFM, BVAR, local projections, HAR, VaR),
plus reporting. R-validated engines are in test_r_crosscheck_v19.py."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from analytics import fcst as F
from analytics import fcst_report as fr
from analytics import nowcast as nc
from analytics import var as vr
from data.samples import macro_var_demo, monthly_forecast_demo, nowcast_demo, realized_vol_demo

warnings.filterwarnings("ignore")
D = monthly_forecast_demo()
Y = D.electricity_gwh.values


def test_demos():
    assert D.shape == (216, 6) and (D.drop(columns="date") > 0).all().all()
    N = nowcast_demo()
    assert N.gdp_growth_q.notna().sum() == 82 and N.exports_mom.isna().sum() == 1
    R = realized_vol_demo()
    assert (R.realized_variance > 0).all() and len(R) > 3000


def test_fc_quantiles_and_frame():
    f = F.naive(Y, 5)
    q = f.quantiles((0.025, 0.975))
    np.testing.assert_allclose(q[0.975], f.pi[95][1], rtol=1e-10)
    np.testing.assert_allclose(q[0.025], f.pi[95][0], rtol=1e-10)
    fr_ = f.frame(F.future_index(pd.to_datetime(D.date), 5))
    assert list(fr_.columns[:3]) == ["time", "step", "forecast"] and fr_.time.iloc[0] == pd.Timestamp("2026-01-01")


def test_state_space_and_bsts_agree():
    rng = np.random.default_rng(3)
    lvl = np.cumsum(rng.normal(0, 0.5, 200))
    y = 10 + lvl + rng.normal(0, 1.0, 200)
    ss = F.state_space(y, 6, "llevel")
    b = F.bsts(y, 6, "llevel", draws=600, burn=200, seed=2)
    tab = b.info["table"].set_index("Parameter")
    assert 0.5 < tab.loc["sigma2.irregular", "Posterior mean"] < 1.6
    assert 0.08 < tab.loc["sigma2.level", "Posterior mean"] < 0.6
    assert abs(b.mean[0] - ss.mean[0]) < 0.5 and np.all(b.pi[95][0] < b.mean) and np.all(b.pi[95][1] > b.mean)


def test_growth_curve_recovers_logistic():
    t = np.arange(1, 81)
    y = 500 / (1 + np.exp((40 - t) / 8)) + np.random.default_rng(1).normal(0, 3, 80)
    g = F.growth_curve(y, 10, "logistic")
    K, xmid, scal = g.info["params"]
    assert abs(K - 500) < 15 and abs(xmid - 40) < 1.5 and abs(scal - 8) < 0.8


def test_ml_and_prophet_style():
    f = F.ml_forecast(Y, 12, "gbm", 12, calib=24)
    assert f.mean.shape == (12,) and f.info["differenced"] and 95 in f.pi
    assert np.all(f.pi[95][1] >= f.pi[80][1] - 1e-9)
    assert f.info["importance"] is not None and "lag12" in set(f.info["importance"].Feature)
    p = F.prophet_style(Y, 12, 12, sims=200)
    assert p.sims.shape == (200, 12) and abs(p.mean.mean() / Y[-12:].mean() - 1) < 0.15


def test_combination_weights():
    rng = np.random.default_rng(0)
    E = np.column_stack([rng.normal(0, 1, 40), rng.normal(0, 2, 40), rng.normal(0, 4, 40)])
    w = F.combination_weights(E, "inv_mse")
    assert np.isclose(w.sum(), 1) and w[0] > w[1] > w[2]
    act = rng.normal(10, 1, 40)
    Fh = act[:, None] - E
    g = F.combination_weights(E, "granger_ramanathan", Fh, act)
    assert np.all(g >= 0) and np.isclose(g.sum(), 1) and g[0] == g.max()
    a = F.combination_weights(E, "aic", ic=[100, 102, 110])
    np.testing.assert_allclose(a, np.exp(-np.array([0, 1, 5])) / np.exp(-np.array([0, 1, 5])).sum())
    c = F.combine([F.naive(Y, 3), F.drift(Y, 3)], "mean")
    np.testing.assert_allclose(c.mean, (F.naive(Y, 3).mean + F.drift(Y, 3).mean) / 2)


def test_horse_race_structure():
    r = F.horse_race(Y, 6, 12, ["naive", "snaive", "theta", "ets"], origins=4)
    t = r["table"]
    assert t.Rank.tolist() == list(range(1, len(t) + 1)) and "combo" in set(t.key)
    assert t.MASE.is_monotonic_increasing
    assert set(r["errors"]) >= {"naive", "snaive", "theta", "ets", "combo"} and r["errors"]["naive"].shape == (4, 6)
    assert t.set_index("key").loc["snaive", "MASE"] < t.set_index("key").loc["naive", "MASE"]
    for lang in ("en", "ar"):
        assert len(fr.interpret_race(r, lang)) > 60


def test_bridge_and_midas():
    N = nowcast_demo()
    ind = ["industrial_production_mom", "pmi", "exports_mom", "retail_sales_mom"]
    b = nc.bridge(N, "date", "gdp_growth_q", ind)
    assert list(b["nowcasts"].Quarter) == ["2025Q3", "2025Q4"] and b["R²"] > 0.5
    assert b["eval_summary"]["relative RMSE"] < 1
    m = nc.midas(N, "date", "gdp_growth_q", "industrial_production_mom", 6, "umidas")
    import statsmodels.api as sm
    q = m["table"]
    assert len(q) == 8 and m["N"] > 70
    mr = nc.midas(N, "date", "gdp_growth_q", "industrial_production_mom", 6, "nealmon")
    assert mr["restriction_test"]["p"] > 0.05 and mr["RMSE"] >= m["RMSE"] - 1e-9
    w = nc.midas_weights([0.3, -0.1], 6, "nealmon")
    assert np.isclose(w.sum(), 1) and np.all(w > 0)
    del sm


def test_dfm_nowcast():
    N = nowcast_demo()
    r = nc.dfm_nowcast(N, "date", "gdp_growth_q", ["industrial_production_mom", "pmi", "exports_mom", "retail_sales_mom"], maxiter=100)
    assert len(r["nowcasts"]) == 2 and r["RMSE in-sample"] < N.gdp_growth_q.std()
    assert abs(np.corrcoef(r["factor"].to_numpy(), N.pmi.to_numpy())[0, 1]) > 0.7


def test_bvar_marginal_likelihood_by_integration():
    """n = 1, p = 1: integrate the closed-form posterior kernel numerically over (b0, b1, σ²)."""
    rng = np.random.default_rng(5)
    y = np.zeros(30)
    for t in range(1, 30):
        y[t] = 0.5 * y[t - 1] + rng.normal()
    Y = y.reshape(-1, 1)
    lam, psi = 0.4, np.array([1.2])
    post = nc.bvar_posterior(Y, 1, lam, psi=psi, b_own=0.0, const_var=4.0)
    yy, X = Y[1:, 0], np.column_stack([np.ones(29), Y[:-1, 0]])
    om = np.array([4.0, lam ** 2 / psi[0] * (3 - 1 - 1)])
    d = 3

    def integrand(s2):
        # ∫ p(y | b, σ²) p(b | σ²) db analytically (Gaussian), then the IW(ψ, d) density on σ²
        V = s2 * np.diag(om)
        C = s2 * np.eye(29) + X @ V @ X.T
        _, ld = np.linalg.slogdet(C)
        lik = -0.5 * (29 * np.log(2 * np.pi) + ld + yy @ np.linalg.solve(C, yy))
        lp = stats.invgamma.logpdf(s2, d / 2, scale=psi[0] / 2)
        return np.exp(lik + lp)
    val, _ = integrate.quad(integrand, 1e-8, 50, limit=1000, points=[0.5, 1, 2])
    assert abs(np.log(val) - post["logml"]) < 1e-4


def test_bvar_limits_and_forecasts():
    M = macro_var_demo()
    Yd = pd.DataFrame({"dy": 100 * np.diff(np.log(M.gdp_bn)), "infl": M.inflation_pct[1:].values, "rate": M.policy_rate_pct[1:].values})
    big = nc.bvar_posterior(Yd.to_numpy(), 2, 1e4)
    ols = np.linalg.lstsq(big["X"], big["y"], rcond=None)[0]
    np.testing.assert_allclose(big["B"], ols, atol=1e-5)
    r = nc.bvar(Yd, 2, b_own=0.0, h=4, draws=300)
    assert 0.01 < r["lambda"] < 5 and set(r["forecasts"]) == {"dy", "infl", "rate"}
    fr_ = r["forecasts"]["infl"]
    assert np.all(fr_["lower 90%"] <= fr_["lower 68%"]) and np.all(fr_["upper 68%"] <= fr_["upper 90%"])


def test_local_projection_equals_var_at_h1():
    M = macro_var_demo()
    Yd = pd.DataFrame({"dy": 100 * np.diff(np.log(M.gdp_bn)), "infl": M.inflation_pct[1:].values, "rate": M.policy_rate_pct[1:].values})
    lp = nc.local_projection(Yd, "rate", "dy", 6, 2)
    r = vr.fit_var(Yd, 2, "const")
    irf = np.asarray(vr.irf(r, 6, "orth"))
    np.testing.assert_allclose(lp.IRF.iloc[:2].to_numpy(), irf[:2, 0, 2], rtol=1e-6, atol=1e-10)
    df_ = nc.lp_direct_forecast(Yd, "infl", 4, 2)
    assert len(df_) == 4 and (df_["upper 95%"] > df_["lower 95%"]).all()


def test_har_and_var_backtest():
    R = realized_vol_demo()
    rv = R.realized_variance.values
    h = F.har(rv, 1, log=False, horizon=5)
    X, yv = F.har_design(rv, 1)
    ok = X.notna().all(1) & yv.notna()
    np.testing.assert_allclose(h["b"], np.linalg.lstsq(X[ok].to_numpy(), yv[ok].to_numpy(), rcond=None)[0], rtol=1e-8)
    hl = F.har(rv, 1, log=True)
    assert 0.8 < hl["persistence"] < 1.0
    vb = F.var_backtest(R.return_pct.values, 0.01, 1000, 50, rv, methods=("garch_n", "hs", "har"))
    assert set(vb["table"].key) == {"garch_n", "hs", "har"} and vb["table"]["Tick loss"].is_monotonic_increasing
    c = F.coverage_tests(np.r_[np.zeros(95, int), np.ones(5, int)], 0.05)
    assert abs(c["Kupiec LR"]) < 1e-10 and c["p Kupiec"] > 0.99


def test_fast_garch_matches_full_engine():
    from analytics import garch as gm
    y = realized_vol_demo().return_pct.values[:1000]
    a = F._FastGarch(y, gjr=True, t=True)
    b = gm.fit_garch(y, "gjr", 1, 1, "std")
    np.testing.assert_allclose(a.x, b.params, rtol=5e-3, atol=2e-3)
    assert a.llf >= b.llf - 1e-2


def test_scenarios_order():
    ex = D[["temperature_c"]]
    paths = F.scenario_paths(ex, 6, {"Hot": {"temperature_c": ("shift", 10)}, "Cool": {"temperature_c": ("shift", -10)}})
    res = F.scenario_forecast(Y, ex, paths, 6, (1, 0, 1), (0, 1, 1), 12)
    assert np.all(res["Hot"].mean > res["Baseline"].mean) and np.all(res["Cool"].mean < res["Baseline"].mean)


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_report_texts_and_code(lang):
    for kind in ("race", "classical", "nowcast", "macro", "volatility", "ml", "eval", "scenario"):
        txt, refs = fr.methodology(kind, lang)
        assert len(txt) > 150 and len(refs) >= 2
        c = fr.code(kind, y="y", indicators=["a", "b"], variables=["a", "b"], drivers=["x"])
        assert set(c) == {"Python", "R", "Stata"}
    assert "%" in fr.interpret_fc(F.naive(Y, 6), Y[-1], lang)
