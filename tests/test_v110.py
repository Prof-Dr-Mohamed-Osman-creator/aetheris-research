"""v1.10 applied economics (§2.7–2.8): analytic identities, likelihood checks by numerical integration and recovery of
known data-generating parameters. R-validated engines are in test_r_crosscheck_v110.py."""
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import integrate, stats

from analytics import applied_report as ar_
from analytics import demand as dm
from analytics import production as pr
from data.samples import (farm_efficiency_demo, fertilizer_trial_demo, firm_cost_demo, firm_tfp_demo, household_budget_demo,
                          supply_response_demo)

warnings.filterwarnings("ignore")
FE = farm_efficiency_demo()
X3 = ["land_feddan", "labour_days", "fertilizer_kg"]
G5 = ["cereals", "meat_fish", "dairy_eggs", "fruit_veg", "other_food"]
SH, PR = [f"w_{g}" for g in G5], [f"p_{g}" for g in G5]


def test_demos_shapes():
    assert FE.shape == (900, 15) and FE.region.nunique() == 3
    assert fertilizer_trial_demo().shape == (112, 6)
    assert np.allclose(firm_cost_demo()[["share_labour", "share_capital", "share_energy"]].sum(1), 1, atol=1e-3)
    hb = household_budget_demo()
    assert np.allclose(hb[SH].sum(1), 1, atol=1e-4)
    assert firm_tfp_demo().firm_id.nunique() == 300 and supply_response_demo().shape[0] == 46


def test_production_forms_and_elasticities():
    cd = pr.production_function(FE, "output_tonnes", X3, "cd")
    e = pr.elasticities(cd)
    np.testing.assert_allclose(e["table"]["Elasticity"], cd.b[1:4])
    assert np.allclose(e["AES"].to_numpy(), 1)
    tl = pr.production_function(FE, "output_tonnes", X3, "translog")
    et = pr.elasticities(tl)
    # centred translog: elasticities at the geometric mean are the first-order coefficients
    eg = pr.elasticities(tl, "geo")
    np.testing.assert_allclose(eg["table"]["Elasticity"], tl.b[1:4], atol=1e-5)
    assert abs(et["RTS"] - cd.stats["RTS"]) < 0.03
    ces = pr.production_function(FE, "output_tonnes", ["land_feddan", "labour_days"], "ces")
    ec = pr.elasticities(ces)
    assert np.isclose(ec["AES"].iloc[0, 1], ces.stats["σ (elasticity of substitution)"])
    # numerical AES of the fitted CES equals the analytic σ
    ces.form = "_numeric"
    en = pr.elasticities(ces)
    assert abs(en["AES"].iloc[0, 1] - ec["AES"].iloc[0, 1]) < 1e-3 * max(1, abs(ec["AES"].iloc[0, 1]))


def test_stages_quadratic_analytic():
    rng = np.random.default_rng(2)
    x = rng.uniform(1, 40, 300)
    y = 5 + 3 * x - 0.06 * x ** 2 + rng.normal(0, 0.5, 300)
    r = pr.production_function(pd.DataFrame({"y": y, "x": x}), "y", ["x"], "quadratic")
    s = pr.stages(r, 0, grid=4000, upto=2.0)
    b, c = r.b[1], r.b[2] / 2
    assert abs(s["end stage II"] - (-b / (2 * c))) < 0.1
    assert not np.isfinite(s["end stage I"])          # a > 0: AP = a/x + b + cx falls everywhere (no stage I)


def test_yield_response_eor_quadratic():
    t = fertilizer_trial_demo()
    f = pr.yield_response(t.n_kg_ha, t.yield_t_ha, "quadratic", 12000, 35)
    a, b, c = f["params"]
    assert np.isclose(f["EOR"], (35 / 12000 - b) / (2 * c))
    qp = pr.yield_response(t.n_kg_ha, t.yield_t_ha, "quad_plateau", 12000, 35)
    assert abs(qp["EOR"] - 177) < 20 and qp["EOR"] < qp["agronomic max"]


def test_cubic_cost_minima():
    q = np.linspace(1, 100, 80)
    c = 500 + 20 * q - 0.4 * q ** 2 + 0.004 * q ** 3
    r = pr.cubic_cost(q, c + np.random.default_rng(0).normal(0, 0.01, 80))
    assert abs(r["q shutdown"] - 50) < 0.5
    atc = lambda x: (500 + 20 * x - 0.4 * x ** 2 + 0.004 * x ** 3) / x  # noqa: E731
    from scipy.optimize import minimize_scalar
    assert abs(r["q efficient"] - minimize_scalar(atc, bounds=(1, 300), method="bounded").x) < 0.5


def test_translog_cost_recovers_dgp():
    C = firm_cost_demo()
    r = pr.translog_system(C, "cost", "total_cost_k", ["wage", "capital_price", "energy_price"], ["share_labour", "share_capital", "share_energy"], "output_t")
    b = r["b"]
    np.testing.assert_allclose(b[1:3], [0.45, 0.35], atol=0.01)
    np.testing.assert_allclose(b[3:6], [0.10, -0.06, 0.09], atol=0.015)
    assert abs(b[6] - 0.75) < 0.02 and r["curvature %"] > 90
    assert np.allclose(r["gamma"].sum(1), 0)
    assert np.all(np.diag(r["elasticities"]) < 0)


def _sfa_ll_numeric(e, sv, su, dist, mu=0.0, s=1.0):
    def dens(ei):
        if dist == "exponential":
            f = lambda u: stats.norm.pdf(ei + s * u, 0, sv) * stats.expon.pdf(u, scale=su)  # noqa: E731
        else:
            a = (0 - mu) / su
            f = lambda u: stats.norm.pdf(ei + s * u, 0, sv) * stats.truncnorm.pdf(u, a, np.inf, loc=mu, scale=su)  # noqa: E731
        return integrate.quad(f, 0, np.inf, limit=200)[0]
    return np.sum([np.log(dens(v)) for v in e])


@pytest.mark.parametrize("dist", ["hnormal", "tnormal", "exponential"])
def test_sfa_loglik_by_integration(dist):
    d = FE[FE.year == 2024]
    r = pr.sfa(d, "output_tonnes", X3, dist)
    X = np.column_stack([np.ones(len(d)), np.log(d[X3].to_numpy())])
    e = np.log(d.output_tonnes.to_numpy()) - X @ r["beta"]
    mu = r["theta"][-1] if dist == "tnormal" else 0.0
    ll = _sfa_ll_numeric(e[:40], r["sigma_v"], r["sigma_u"], dist, mu)
    np.testing.assert_allclose(ll, np.sum(pr._sfa_ll(r["theta"], np.log(d.output_tonnes.to_numpy())[:40], X[:40],
                                                      np.ones((40, 1)) if dist == "tnormal" else np.zeros((40, 0)), dist, 1.0, X.shape[1])), rtol=1e-6)


def test_bc95_recovers_determinants():
    r = pr.sfa(FE, "output_tonnes", X3, "tnormal", z=["education_years", "extension_visits", "credit_access"])
    d = r["table"].set_index("Parameter")["Coef."]
    assert abs(d["δ (education_years)"] + 0.035) < 0.015 and abs(d["δ (credit_access)"] + 0.15) < 0.08
    assert abs(r["sigma_v"] - 0.15) < 0.03 and 0.6 < r["mean efficiency"] < 0.9


def test_bc92_with_one_period_equals_stevenson():
    d = FE[FE.year == 2024]
    p = pr.sfa_panel(d, "farm_id", "year", "output_tonnes", X3, "ti", mu=True)
    s = pr.sfa(d, "output_tonnes", X3, "tnormal")
    assert abs(p["loglik"] - s["loglik"]) < 1e-3


def test_metafrontier_and_dea_properties():
    m = pr.metafrontier(FE, "region", "output_tonnes", X3)
    assert (m["obs"]["TGR"] <= 1 + 1e-9).all() and np.isclose(m["obs"]["TGR"].max(), 1, atol=1e-6)
    d = FE[FE.year == 2024]
    t = pr.dea_table(d, X3, ["output_tonnes"])
    assert (t["Scale efficiency"] <= 1 + 1e-9).all() and (t["VRS (BCC)"] >= t["CRS (CCR)"] - 1e-9).all()
    oo = pr.dea(d[X3].to_numpy(), d[["output_tonnes"]].to_numpy(), "crs", "output")
    np.testing.assert_allclose(oo, t["CRS (CCR)"], rtol=1e-6)            # CRS: input and output orientations coincide


def test_malmquist_identity():
    a = FE[FE.year == 2023].copy()
    b = a.copy()
    b["year"] = 2024
    b["output_tonnes"] = b["output_tonnes"] * 1.1
    M = pr.malmquist(pd.concat([a, b]), "farm_id", "year", X3, ["output_tonnes"])
    np.testing.assert_allclose(M["Malmquist"], 1.1, rtol=1e-6)
    np.testing.assert_allclose(M["Efficiency change"], 1.0, rtol=1e-6)


def test_tfp_index_manual():
    d = pd.DataFrame({"t": [1, 1, 2, 2], "y": [10, 20, 12, 22], "py": [1, 1, 1.1, 1.1], "x": [5, 5, 5, 6], "px": [2, 2, 2, 2]})
    r = pr.tfp_index(d, "t", ["y"], ["py"], ["x"], ["px"])
    assert np.isclose(r["Törnqvist TFP"].iloc[1], (34 / 30) / (11 / 10))
    assert np.isclose(r["Fisher TFP"].iloc[1], (34 / 30) / (11 / 10))


def test_prodest_recovers_parameters():
    T = firm_tfp_demo()
    for m, px in (("lp", "ln_materials"), ("acf", "ln_materials"), ("op", "investment")):
        r = pr.prodest(T, "firm_id", "year", "ln_value_added", ["ln_labour"], "ln_capital", px, m)
        assert abs(r["beta"][0] - 0.6) < 0.06 and abs(r["beta"][1] - 0.3) < 0.08, (m, r["beta"])
    assert r["table"]["OLS"].iloc[0] > 1.0                               # OLS is badly biased upward for labour


def test_nerlove_and_cobweb():
    s = supply_response_demo()
    r = dm.nerlove(s, "year", "wheat_area_kfed", "real_wheat_price", ["real_berseem_price"])
    assert abs(r["SR"][0] - 0.35) < 0.12 and abs(r["adjustment coefficient"][0] - 0.4) < 0.15
    b = r["model"].params
    assert np.isclose(r["LR"][0], b[1] / (1 - b[2]))
    cw = dm.cobweb(100, 2, 10, 1.5, 40, 50)
    assert cw["verdict"] == "convergent" and abs(cw["P"][-1] - cw["P*"]) < 1e-3
    assert dm.cobweb(100, 1, 10, 1.5, 40)["verdict"] == "explosive"


def test_engel_and_demand():
    hb = household_budget_demo()
    en = dm.engel(hb, "food_expenditure", SH, "wl").set_index("Good")
    assert en.loc["w_cereals", "Class"] == "necessity" and en.loc["w_meat_fish", "Class"] == "luxury"
    assert abs(en.loc["w_fruit_veg", "Expenditure elasticity"] - 1) < 0.05
    rng = np.random.default_rng(1)
    p = np.exp(rng.normal(0, 0.3, 300))
    y = np.exp(rng.normal(5, 0.3, 300))
    q = np.exp(2 - 0.8 * np.log(p) + 0.5 * np.log(y) + rng.normal(0, 0.05, 300))
    r = dm.demand_equation(pd.DataFrame({"q": q, "p": p, "y": y}), "q", "p", None, "y")
    np.testing.assert_allclose(r["elasticities"]["Short-run elasticity"], [-0.8, 0.5], atol=0.03)


def test_quaids_recovers_dgp_and_properties():
    hb = household_budget_demo()
    r = dm.aids(hb, SH, PR, "food_expenditure", "quaids", alpha0=7.8)
    np.testing.assert_allclose(r["alpha"], [0.34, 0.22, 0.15, 0.17, 0.12], atol=0.01)
    np.testing.assert_allclose(r["beta"], [-0.09, 0.06, 0.02, 0.0, 0.01], atol=0.012)
    np.testing.assert_allclose(r["lambda"], [0.012, -0.008, -0.002, 0.0, -0.002], atol=0.004)
    assert np.allclose(r["gamma"], r["gamma"].T) and np.allclose(r["gamma"].sum(1), 0, atol=1e-10)
    e = r["elasticities"]
    w = e["shares"]
    assert abs(w @ e["expenditure"] - 1) < 1e-8                         # Engel aggregation
    np.testing.assert_allclose(e["hicksian"].to_numpy().sum(1), 0, atol=1e-8)   # homogeneity of Hicksian demands
    assert e["negativity"]


def test_welfare_cobb_douglas_exact():
    """AIDS with γ = β = λ = 0 is Cobb-Douglas: CV = x(Π(p1/p0)^α − 1) exactly."""
    hb = household_budget_demo().head(50)
    a = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    res = {"alpha": a, "gamma": np.zeros((5, 5)), "beta": np.zeros(5), "lambda": np.zeros(5), "alpha0": 0.0, "model": "aids",
           "lp": np.zeros((50, 5)), "lx": np.log(hb.food_expenditure.to_numpy()), "W": np.tile(a, (50, 1)),
           "elasticities": {"hicksian": pd.DataFrame(np.tile(a, (5, 1)) - np.eye(5))}}
    dl = np.log(np.array([1.2, 1.0, 0.9, 1.0, 1.05]))
    w = dm.welfare_quaids(res, dl)
    x = hb.food_expenditure.to_numpy()
    np.testing.assert_allclose(w["CV"], x * (np.exp(a @ dl) - 1), rtol=1e-10)
    np.testing.assert_allclose(w["EV"], x * (1 - np.exp(-a @ dl)), rtol=1e-10)


def test_les_and_rotterdam_and_consumption():
    hb = household_budget_demo()
    le = dm.les(hb, SH, PR, "food_expenditure")
    assert np.isclose(le["beta"].sum(), 1) and le["frisch"] < -1
    ro = dm.rotterdam(hb, "region", "round", SH, PR, "food_expenditure")
    assert np.isclose(ro["theta"].sum(), 1) and np.allclose(ro["pi"].to_numpy().sum(1), 0, atol=1e-10)
    from data.samples import macro_var_demo
    c = dm.consumption(macro_var_demo(), "quarter", "consumption_bn", "gdp_bn")
    assert 0 < c["keynes"]["MPC"] < 1.2 and "λ" in c["cm"]
    assert dm.consumer_surplus_loglinear(100, 10, 11, -1) == pytest.approx(100 * 10 * np.log(1.1))


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_reports(lang):
    for k in ar_.TEXT:
        txt, refs = ar_.methodology(k, lang)
        assert len(txt) > 150 and refs
    for k in ("production", "yield", "cost", "efficiency", "tfp", "supply", "demand", "system", "consumption"):
        assert set(ar_.code(k)) == {"Python", "R", "Stata"}
