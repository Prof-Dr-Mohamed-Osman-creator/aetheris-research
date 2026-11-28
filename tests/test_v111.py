"""v1.11 — quantile regression, spatial econometrics, policy evaluation and climate & energy economics: recovery of known
data-generating parameters, internal identities and report completeness (the R / PySAL cross-checks live in
test_r_crosscheck_v111.py)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import causal as C
from analytics import climate as K
from analytics import impact_report as ir
from analytics import quantile as Q
from analytics import spatial as S
from data import samples as smp

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------- quantile
def test_mmqr_recovers_location_scale():
    rng = np.random.default_rng(3)
    n_i, T = 200, 6
    a = np.repeat(rng.normal(0, 1, n_i), T)
    x = rng.uniform(0, 2, n_i * T)
    u = rng.normal(size=n_i * T)
    y = a + 1.0 * x + (1 + 0.5 * x) * u
    d = pd.DataFrame({"id": np.repeat(np.arange(n_i), T), "y": y, "x": x})
    r = Q.mmqr(d, "id", "y", ["x"], taus=(0.1, 0.5, 0.9), boot=0)
    ls = r["location_scale"].iloc[0]
    assert abs(ls["Location β"] - 1.0) < 0.08
    assert ls["Scale γ"] > 0.2          # scale rises with x
    b = r["table"].set_index("τ")["Coef."]
    assert b[0.1] < b[0.5] < b[0.9]


def test_qr_process_and_slope_equality_detect_heteroskedasticity():
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 3, 1500)
    y = 1 + x + (0.5 + x) * rng.normal(size=1500)
    d = pd.DataFrame({"y": y, "x": x})
    p = Q.qr_process(d, "y", ["x"], taus=[0.1, 0.5, 0.9])
    s = p["table"][p["table"].Variable == "x"].set_index("τ")["Coef."]
    assert s[0.1] < 0.2 < 1.5 < s[0.9]
    assert Q.slope_equality(d, "y", ["x"]).iloc[-1]["p-value"] < 1e-6


def test_cross_quantilogram_detects_lagged_tail_dependence():
    rng = np.random.default_rng(1)
    T = 600
    yv = rng.normal(size=T)
    xv = np.r_[0, -1.5 * yv[:-1]] + rng.normal(size=T)
    r = Q.cross_quantilogram(xv, yv, 0.1, 0.9, max_lag=3, boot=200)
    assert r.loc[0, "ρ"] < -0.1 and bool(r.loc[0, "significant"])   # high y_{t-1} (not below q90) → x in its lower tail
    indep = Q.cross_quantilogram(rng.normal(size=T), rng.normal(size=T), 0.1, 0.1, max_lag=5, boot=200)
    assert indep["p (Q)"].iloc[-1] > 0.01


# ---------------------------------------------------------------- spatial
def test_weights_and_moran_basics():
    D = smp.spatial_districts_demo()
    W = S.weights(D[["longitude", "latitude"]].values, "knn", 5)
    assert np.allclose(W.sum(1), 1) and (W > 0).sum(1).max() == 5 and np.all(np.diag(W) == 0)
    Wb = S.weights(D[["longitude", "latitude"]].values, "band", geo=True)
    assert S.describe_w(Wb)["islands"] == 0
    rng = np.random.default_rng(2)
    m = S.moran(rng.normal(size=len(D)), W, perms=199)
    assert abs(m["z (randomisation)"]) < 3
    assert S.moran(D.ln_land_value, W, perms=99)["I"] > 0.3


def test_spatial_models_recover_dgp_and_impacts_identities():
    D = smp.spatial_districts_demo()
    W = S.weights(D[["longitude", "latitude"]].values, "knn", 6)
    D["ln_dist"] = np.log(D.distance_market_km)
    x = ["irrigation_share", "ln_dist", "soil_index", "avg_temp_c"]
    r = S.spatial_model(D, "ln_land_value", x, W, "sdm", impacts_draws=200)
    assert 0.2 < r["rho"] < 0.7                                # DGP ρ = 0.45
    im = r["impacts"].pivot(index="Variable", columns="Impact", values="Estimate")
    assert np.allclose(im["Direct"] + im["Indirect (spillover)"], im["Total"])
    # total impact of SAR equals β / (1 − ρ)
    s = S.spatial_model(D, "ln_land_value", x, W, "sar", impacts_draws=50)
    tot = s["impacts"][s["impacts"].Impact == "Total"].Estimate.to_numpy()
    assert np.allclose(tot, s["est"][1:1 + len(x)] / (1 - s["rho"]))
    # AIC prefers a spatial model over OLS
    o = S.spatial_model(D, "ln_land_value", x, W, "ols")
    assert s["AIC"] < o["AIC"]


def test_gwr_and_spatial_panel_run():
    D = smp.spatial_districts_demo()
    g = S.gwr(D, "ln_land_value", ["irrigation_share", "soil_index"], D[["longitude", "latitude"]].values, bw=60)
    assert g["local"].shape[0] == len(D) and 0 < g["R²"] < 1
    P = smp.spatial_panel_demo()
    W = S.weights(P.drop_duplicates("district")[["longitude", "latitude"]].values, "knn", 6)
    r = S.spatial_panel(P, "district", "year", "ln_yield", ["ln_fertilizer", "rainfall_anomaly"], W, "sar")
    assert abs(r["param"] - 0.35) < 0.1                       # DGP ρ = 0.35
    with pytest.raises(ValueError):
        S.spatial_panel(P.iloc[:-1], "district", "year", "ln_yield", ["ln_fertilizer"], W)


# ---------------------------------------------------------------- policy evaluation
def test_rct_and_iv_recover_effects():
    d = smp.rct_extension_demo()
    r = C.rct(d, "ln_yield", "treatment_offered", ["land_feddan", "education_years", "baseline_ln_yield"], "region")
    t = r["table"].set_index("Estimator")
    assert t.loc["Lin (interacted)", "Std. Err."] < t.loc["Difference in means", "Std. Err."]
    assert r["joint F"]["p-value"] > 0.01
    iv = C.iv_2sls(d, "ln_yield", "participated", ["treatment_offered"])
    assert abs(iv["Wald"]["LATE = ITT / first stage"] - iv["late"]) < 1e-10
    assert iv["first-stage F"] > 100
    assert 0.03 < iv["late"] < 0.2                            # true effect 0.08 (+0.04 smallholders)


def test_staggered_did_estimators_agree_and_recover_effect():
    d = smp.staggered_did_demo()
    cs = C.callaway_santanna(d, "village", "year", "ln_water_productivity", "first_treated")
    sa = C.sun_abraham(d, "village", "year", "ln_water_productivity", "first_treated")
    assert cs["pretrend"]["p-value"] > 0.01
    pre = cs["dynamic"][cs["dynamic"]["Event time"] < 0]
    assert (pre.ATT.abs() < 4 * pre["Std. Err."]).all()
    assert abs(cs["summary"].ATT[0] - sa["att"]["Estimate"]) < 0.05
    bw = C.bacon_weights(d, "village", "year", "ln_water_productivity", "first_treated")
    assert abs(bw.Weight.sum() - 1) < 1e-10
    # Bacon weighted sum reproduces the static TWFE coefficient
    D = (d.year >= d.first_treated) & (d.first_treated > 0)
    b, V, keep, _ = C._fe_cluster(d.ln_water_productivity.to_numpy(), D.to_numpy(float)[:, None], pd.factorize(d.village)[0],
                                  pd.factorize(d.year)[0], d.village.to_numpy())
    assert abs(bw.attrs["weighted sum"] - b[0]) < 1e-8


def test_did_2x2_equals_cell_means():
    d = smp.staggered_did_demo()
    d = d[d.first_treated.isin([0, 2017])].copy()
    d["g"] = (d.first_treated > 0).astype(int)
    d["post"] = (d.year >= 2017).astype(int)
    r = C.did_2x2(d, "ln_water_productivity", "g", "post", "village")
    c = r["cells"]
    assert abs(r["did"] - ((c.loc[1, 1] - c.loc[1, 0]) - (c.loc[0, 1] - c.loc[0, 0]))) < 1e-10


def test_synthetic_control_and_sdid():
    d = smp.scm_regions_demo()
    r = C.synth_control(d, "region", "year", "ln_gdp_pc", "Sinai Coast", 2012, ["investment_share", "schooling_years"], placebo=True)
    w = r["weights"].Weight
    assert abs(w.sum() - 1) < 1e-8 and (w >= 0).all()
    assert r["ATT (post average gap)"] > 0.04 and r["p-value (rank)"] <= 0.1
    d["D"] = ((d.region == "Sinai Coast") & (d.year >= 2012)).astype(int)
    s = C.synth_did(d, "region", "year", "ln_gdp_pc", "D", reps=30)
    assert abs(s["unit weights"].sum() - 1) < 1e-8 and abs(s["time weights"].sum() - 1) < 1e-8
    assert s["ATT"] > 0.04


def test_sdid_reduces_to_did_with_uniform_weights():
    # with parallel, noise-free outcomes every simplex weighting gives the DiD effect
    N0, N1, T0, T1, tau = 12, 3, 8, 4, 0.5
    a = np.arange(N0 + N1)[:, None] * 0.3
    b = np.arange(T0 + T1)[None, :] * 0.1
    Y = a + b
    Y[N0:, T0:] += tau
    est, om, lam = C._sdid_core(Y + 1e-9 * np.random.default_rng(0).normal(size=Y.shape), N0, T0)
    assert abs(est - tau) < 1e-6


def test_rdd_recovers_jump():
    d = smp.rdd_scholarship_demo()
    r = C.rdd(d, "gpa", "entrance_score", 60, covs=["female"])
    assert abs(r["table"].Estimate[0] - 0.30) < 0.12
    assert r["covariates"]["Robust p"].iloc[0] > 0.01
    f = C.rdd(d, "gpa", "entrance_score", 60, treat="scholarship")
    assert f["first stage"]["Jump in treatment"] > 0.5


def test_matching_weighting_recover_att():
    d = smp.credit_program_demo()
    covs = ["age", "education_years", "land_feddan", "female_head", "distance_branch_km"]
    naive = d.groupby("participant").ln_income.mean().diff().iloc[-1]
    m = C.psm(d, "ln_income", "participant", covs)
    w = C.ipw_aipw(d, "ln_income", "participant", covs, "ATT")
    for est in (m["estimate"]["Estimate"], *w["table"].Estimate):
        assert abs(est - 0.12) < 0.05                          # true ATT ≈ 0.12
    assert abs(naive - 0.12) > abs(w["table"].Estimate[1] - 0.12)
    assert w["balance"]["SMD after weighting"].abs().max() < 0.1


# ---------------------------------------------------------------- climate
def test_degree_days_sine_matches_numerical_integration():
    lo, hi = np.array([2.0, 12, 25, -3]), np.array([18.0, 33, 41, 6])
    tt = np.linspace(0, 2 * np.pi, 400001)[:-1]
    for base in (8, 29):
        num = [np.mean(np.clip((l + h) / 2 + (h - l) / 2 * np.sin(tt) - base, 0, None)) for l, h in zip(lo, hi)]
        assert np.allclose(K._sine_dd(lo, hi, base), num, atol=1e-6)
    w = smp.daily_weather_demo()
    a = K.degree_days(w, "tmin", "tmax", ["station", "season"])
    wa = K.warming_degree_days(w, "tmin", "tmax", ["station", "season"], deltas=(0, 2))
    assert (wa[wa["warming (°C)"] == 2]["KDD > 29°C"].to_numpy() > wa[wa["warming (°C)"] == 0]["KDD > 29°C"].to_numpy()).all()
    assert len(a) == w.groupby(["station", "season"]).ngroups


def test_temperature_response_recovers_dgp():
    cp = smp.crop_climate_panel_demo()
    r = K.temp_response(cp, "district", "year", "ln_yield", "gddkdd", gdd="gdd", kdd="kdd", precip="precip_mm", tmean="tmean_season")
    assert abs(r["b"]["gdd"] - 0.0004) < 0.0001 and abs(r["b"]["kdd"] + 0.0025) < 0.0003
    w = K.warming_impact(r, deltas=(1, 2))
    assert (w["% change"] < 0).all() and w["% change"].iloc[1] < w["% change"].iloc[0]


def test_ricardian_bhm_ekc():
    rd = smp.ricardian_demo()
    r = K.ricardian(rd, "land_value_k_egp", ["temp_winter", "temp_summer", "precip_winter", "precip_summer"], ["soil_quality", "irrigated"],
                    deltas={"temp_summer": ("abs", 2)})
    assert r["scenario"]["% change"] < 0
    ce = smp.country_emissions_demo()
    b = K.bhm(ce, "country", "year", "gdp_growth", "temperature_c")
    assert b["optimum CI"][0] < 12.7 < b["optimum CI"][1]      # DGP optimum 0.0127 / 0.001 = 12.7 °C
    e = K.ekc(ce, "country", "year", "co2_mt", "gdp_pc_usd", per_capita="population")
    assert e["shape"].startswith("inverted") and e["turning point CI (delta)"][0] < 10400 < e["turning point CI (delta)"][1]
    s = K.scc_simple()
    assert 20 < s["SCC (USD/tCO2)"] < 200 and K.scc_simple(discount=0.02)["SCC (USD/tCO2)"] > s["SCC (USD/tCO2)"]


def test_stirpat_ridge_and_projection():
    ce = smp.country_emissions_demo()
    r = K.stirpat(ce, "country", "year", "co2_mt", ["population", "gdp_pc_usd"], ridge=True)
    assert r["ridge"]["λ (GCV)"].iloc[0] > 0
    p = K.emission_projection({"population": 1.0, "gdp_pc_usd": 0.5}, {"impact": 100.0}, {"population": 1.0, "gdp_pc_usd": -1.0}, 10)
    assert np.isclose(p.Impact.iloc[-1], 100 * (1.01 ** 10) * (0.99 ** 5))


def test_tapio_and_lmdi_identities():
    assert K.TAPIO[K.tapio_state(-0.1, 0.2)[0]][0] == "Strong decoupling"
    assert K.TAPIO[K.tapio_state(0.05, 0.2)[0]][0] == "Weak decoupling"
    assert K.TAPIO[K.tapio_state(0.2, 0.2)[0]][0] == "Expansive coupling"
    assert K.TAPIO[K.tapio_state(0.4, 0.2)[0]][0] == "Expansive negative decoupling"
    assert K.TAPIO[K.tapio_state(0.1, -0.1)[0]][0] == "Strong negative decoupling"
    assert K.TAPIO[K.tapio_state(-0.3, -0.1)[0]][0] == "Recessive decoupling"
    ce = smp.country_emissions_demo()
    ce["gdp"] = ce.gdp_pc_usd * ce.population
    L = K.lmdi_kaya(ce, "country", "year", "co2_mt", "population", "gdp", "energy_pj", 1990, 2023)
    agg = L["aggregate"].set_index("Factor").Effect
    assert abs(agg["Residual"]) < 1e-6 * abs(agg["Total ΔCO2"])
    mult = L["multiplicative"]
    assert np.isclose(np.prod([v for k, v in mult.items() if not k.startswith("Total")]), mult["Total (C_T / C_0)"])


# ---------------------------------------------------------------- reports
@pytest.mark.parametrize("kind", sorted(ir.TEXT))
def test_reports_complete(kind):
    for lang in ("en", "ar"):
        txt, refs = ir.methodology(kind, lang)
        assert len(txt) > 100 and refs
    if kind in ("did22",):
        return
    code = ir.code(kind, y="y", x=["x1", "x2"], treat="d", z=["z"])
    assert set(code) == {"Python", "R", "Stata"} and all(len(v) > 20 for v in code.values())
