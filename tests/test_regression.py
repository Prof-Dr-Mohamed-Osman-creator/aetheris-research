"""Regression engine (§1.1), diagnostics (§1.3) and reporting validated against statsmodels / scipy
reference implementations, closed-form results and Monte Carlo."""
import warnings
import zipfile
from io import BytesIO

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats import diagnostic as dg
from statsmodels.stats.outliers_influence import OLSInfluence, variance_inflation_factor

from analytics import diagnostics as dx
from analytics import reg_report as rp
from analytics import regression as rg
from data.samples import farm_survey_demo, climate_crop_demo

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def farm():
    return farm_survey_demo()


@pytest.fixture(scope="module")
def cd(farm):
    spec = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days", "fertilizer_kg", "education_years"],
                        categorical=["irrigation"], form="loglog", nolog=["education_years"])
    return rg.fit(farm, spec)


def _sm_design(farm):
    d = pd.DataFrame({"const": 1.0, "ln(land_feddan)": np.log(farm.land_feddan), "ln(labour_days)": np.log(farm.labour_days),
                      "ln(fertilizer_kg)": np.log(farm.fertilizer_kg), "education_years": farm.education_years.astype(float),
                      "irrigation[Drip]": (farm.irrigation == "Drip").astype(float),
                      "irrigation[Groundwater]": (farm.irrigation == "Groundwater").astype(float)})
    return np.log(farm.output_tonnes), d


# ---------------------------------------------------------------- design & estimation
def test_design_matches_manual_and_ols_matches_statsmodels(farm, cd):
    y, X = _sm_design(farm)
    assert list(cd.names) == list(X.columns)
    ref = sm.OLS(y, X).fit()
    assert np.allclose(cd.params.values, ref.params.values)
    assert np.allclose(cd.bse.values, ref.bse.values)
    assert np.isclose(cd.rsquared, ref.rsquared) and np.isclose(cd.fvalue, ref.fvalue)
    assert np.isclose(cd.aic, ref.aic) and np.isclose(cd.bic, ref.bic) and np.isclose(cd.llf, ref.llf)


@pytest.mark.parametrize("cov,kw", [("HC0", {}), ("HC1", {}), ("HC2", {}), ("HC3", {}), ("HAC", {"maxlags": 4})])
def test_robust_covariances_match_statsmodels(farm, cov, kw):
    spec = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days", "fertilizer_kg", "education_years"],
                        categorical=["irrigation"], form="loglog", nolog=["education_years"])
    r = rg.fit(farm, spec, cov_type=cov, cov_opts=dict(kw))
    y, X = _sm_design(farm)
    ref = sm.OLS(y, X).fit(cov_type=cov, use_t=True, **({"cov_kwds": {"maxlags": 4, "use_correction": True}} if cov == "HAC" else {}))
    assert np.allclose(r.bse.values, ref.bse.values) and np.allclose(r.pvalues.values, ref.pvalues.values)


def test_cluster_se_and_notes(farm):
    spec = rg.ModelSpec(y="output_tonnes", x=["land_feddan"], form="loglog")
    r = rg.fit(farm, spec, cov_type="cluster", cov_opts={"groups": "region"})
    ref = sm.OLS(np.log(farm.output_tonnes), sm.add_constant(np.log(farm.land_feddan))).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(farm.region)[0]}, use_t=True)
    assert np.allclose(r.bse.values, ref.bse.values)
    assert any("clusters" in n for n in r.notes)


def test_bootstrap_se_close_to_hc(farm):
    spec = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days"], form="loglog")
    rb = rg.fit(farm, spec, cov_type="bootstrap", cov_opts={"reps": 600})
    rh = rg.fit(farm, spec, cov_type="HC1")
    assert np.allclose(rb.bse.values, rh.bse.values, rtol=0.2)
    assert rb.dist == "z" and (rb.ci["low"] < rb.params).all()


def test_reference_level_and_collinearity_drop(farm):
    d = farm.assign(dup=farm.land_feddan * 2)
    r = rg.fit(d, rg.ModelSpec(y="output_tonnes", x=["land_feddan", "dup"], categorical=["irrigation"],
                               reference={"irrigation": "Drip"}))
    assert "dup" not in r.names and any("collinear" in n for n in r.notes)
    assert "irrigation[Canal]" in r.names and "irrigation[Drip]" not in r.names


def test_wls_fgls_glsar_robust_run(farm):
    spec = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days"])
    w = rg.fit(farm, spec, method="wls", weights="land_feddan")
    ref = sm.WLS(farm.output_tonnes, sm.add_constant(farm[["land_feddan", "labour_days"]]), weights=farm.land_feddan).fit()
    assert np.allclose(w.params.values, ref.params.values)
    for m in ("fgls", "glsar", "huber", "bisquare"):
        r = rg.fit(farm, spec, method=m)
        assert np.isfinite(r.params).all() and r.nobs == len(farm)
    rl = rg.fit(farm, spec, method="huber")
    ref = sm.RLM(farm.output_tonnes, sm.add_constant(farm[["land_feddan", "labour_days"]]), M=sm.robust.norms.HuberT()).fit()
    assert np.allclose(rl.params.values, ref.params.values)


def test_polynomial_interaction_centering(farm):
    spec = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "experience_years"], categorical=["credit_access"],
                        poly={"experience_years": 2}, interactions=[("land_feddan", "credit_access")], center=True)
    r = rg.fit(farm, spec)
    assert {"experience_years^2", "land_feddan×credit_access[Yes]"} <= set(r.names)
    assert abs(r.design.X["experience_years"].mean()) < 1e-9


# ---------------------------------------------------------------- restrictions & tests
def test_parse_restrictions():
    names = ["const", "ln(land)", "ln(labour)", "crop[Faba bean]"]
    R, q, lab = rg.parse_restrictions("ln(land) + ln(labour) = 1\n2*crop[Faba bean] - ln(land) = 0.5", names)
    assert np.allclose(R, [[0, 1, 1, 0], [0, -1, 0, 2]]) and np.allclose(q, [1, 0.5])
    with pytest.raises(ValueError):
        rg.parse_restrictions("ln(capital) = 1", names)


def test_rls_satisfies_restriction_and_f_matches(farm, cd):
    text = "ln(land_feddan) + ln(labour_days) + ln(fertilizer_kg) = 1"
    R, q, lab = rg.parse_restrictions(text, cd.names)
    rr = rg.restricted_ls(farm, cd.design.spec, R, q, lab)
    assert np.allclose(R @ rr.params.values, q)
    y, X = _sm_design(farm)
    ft = sm.OLS(y, X).fit().f_test((R, q))
    assert np.isclose(rr.extra["F_restr"], float(np.squeeze(ft.fvalue)))
    w = rg.wald_test(cd, R, q)
    assert np.isclose(w["F"], float(np.squeeze(ft.fvalue)))


def test_robust_wald_matches_statsmodels(farm):
    spec = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days", "fertilizer_kg"], form="loglog")
    r = rg.fit(farm, spec, cov_type="HC3")
    R = np.array([[0, 1, -1, 0.0]])
    ref = r.sm.wald_test((R, np.zeros(1)), use_f=True, scalar=True)
    assert np.isclose(rg.wald_test(r, R, np.zeros(1))["F"], float(ref.statistic))


def test_trinity_ordering(farm, cd):
    out = rg.nested_tests(farm, cd, ["education_years", "irrigation[Drip]"])
    W, LR, LM = out.set_index("Test")["Statistic"].loc[["Wald χ² (classical)", "Likelihood ratio χ²",
                                                          "Lagrange multiplier χ² (nR²)"]]
    assert W >= LR >= LM > 0


def test_term_tests_equal_nested_F(farm, cd):
    tt = rg.term_tests(cd).set_index("Term")
    nt = rg.nested_tests(farm, cd, ["irrigation[Drip]", "irrigation[Groundwater]"])
    assert np.isclose(tt.loc["irrigation", "F"], nt.loc[0, "Statistic"])


# ---------------------------------------------------------------- model selection & form
def test_stepwise_and_best_subsets_find_true_model():
    rng = np.random.default_rng(1)
    n = 300
    d = pd.DataFrame(rng.normal(size=(n, 6)), columns=list("abcdef"))
    d["y"] = 1 + 2 * d.a - 1.5 * d.c + 0.8 * d.e + rng.normal(size=n)
    des = rg.build_design(d, rg.ModelSpec(y="y", x=list("abcdef")))
    for crit in ("p", "aic", "bic"):
        for direc in ("forward", "backward", "both"):
            sel = set(rg.stepwise(des, direc, crit)["selected"]) - {"const"}
            assert {"a", "c", "e"} <= sel
    bs = rg.best_subsets(des)
    assert set(bs.loc[bs["Size"] == 3, "Variables"].iloc[0].split(", ")) == {"a", "c", "e"}
    assert "BIC" in bs.loc[bs["Size"] == 3, "Best by"].iloc[0]


def test_hierarchical_f_change(farm):
    des = rg.build_design(farm, rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days", "education_years"]))
    h = rg.hierarchical(des, [["land_feddan"], ["labour_days", "education_years"]])
    full = sm.OLS(des.y, des.X).fit()
    small = sm.OLS(des.y, des.X[["const", "land_feddan"]]).fit()
    F = full.compare_f_test(small)[0]
    assert np.isclose(h.loc[1, "F change"], F)


def test_box_cox_matches_scipy_and_recovers_log():
    rng = np.random.default_rng(3)
    y = np.exp(rng.normal(1, 0.5, 400))
    d = pd.DataFrame({"y": y, "x": rng.normal(size=400)})
    bc = rg.box_cox(d, rg.ModelSpec(y="y", x=["x"]))
    assert abs(bc["lambda"]) < 0.2 and bc["recommend"] == "log"
    # intercept-only model: profile λ equals scipy's MLE
    sp = rg.ModelSpec(y="y", x=["x"])
    d0 = d.assign(x=0.0 + 1e-12 * np.arange(400))
    lam_sc = stats.boxcox(y)[1]
    assert abs(rg.box_cox(d0, sp)["lambda"] - lam_sc) < 0.02


def test_compare_forms_jacobian_prefers_true_loglog(farm):
    out = rg.compare_forms(farm, rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days", "fertilizer_kg"]))
    assert out.loc[out["AIC (comparable)"].idxmin(), "Form"] == "loglog"


# ---------------------------------------------------------------- moderation & mediation
def test_simple_slopes_and_mediation():
    rng = np.random.default_rng(5)
    n = 500
    x, w = rng.normal(size=n), rng.normal(size=n)
    m = 0.6 * x + rng.normal(size=n)
    y = 0.5 * x + 0.7 * m + 0.4 * x * w + 0.3 * w + rng.normal(size=n)
    d = pd.DataFrame({"x": x, "w": w, "m": m, "y": y})
    r = rg.fit(d, rg.ModelSpec(y="y", x=["x", "w"], interactions=[("x", "w")]))
    ss = rg.simple_slopes(r, "x", "w", {"w=1": 1.0})
    assert np.isclose(ss["Slope"].iloc[0], r.params["x"] + r.params["x×w"])
    jn = rg.johnson_neyman(r, "x", "w")
    for root in jn["roots"]:
        s = r.params["x"] + r.params["x×w"] * root
        se = np.sqrt(r.cov.loc["x", "x"] + root ** 2 * r.cov.loc["x×w", "x×w"] + 2 * root * r.cov.loc["x", "x×w"])
        assert np.isclose(abs(s / se), stats.t.ppf(0.975, r.df_resid))
    med = rg.mediation(d, "x", "m", "y", reps=400)
    tab = med["table"].set_index("Path")
    assert np.isclose(med["indirect"], tab.iloc[0]["Coef."] * tab.iloc[1]["Coef."])
    assert med["boot_ci"][0] > 0 and med["type"] == "partial"
    assert np.isclose(tab.iloc[3]["Coef."], tab.iloc[2]["Coef."] + med["indirect"])  # c = c' + ab (OLS identity)


# ---------------------------------------------------------------- diagnostics vs statsmodels
@pytest.fixture(scope="module")
def ols_climate():
    df = climate_crop_demo()
    r = rg.fit(df, rg.ModelSpec(y="yield_t_per_feddan", x=["avg_temp_c", "rainfall_mm", "year"], categorical=["crop"]))
    return r


def test_heteroskedasticity_tests_match_statsmodels(ols_climate):
    e, X = ols_climate.wresid, ols_climate.wexog
    for koenker in (True, False):
        mine = dx.breusch_pagan(e, X, koenker)
        ref = dg.het_breuschpagan(e, X, robust=koenker)
        assert np.isclose(mine["stat"], ref[0]) and np.isclose(mine["p"], ref[1])
    w = dx.white_test(e, X)
    ref = dg.het_white(e, X)
    assert np.isclose(w["stat"], ref[0]) and np.isclose(w["p"], ref[1])


def test_autocorrelation_tests_match_statsmodels(ols_climate):
    e, X = ols_climate.wresid, ols_climate.wexog
    for L in (1, 2, 4):
        mine = dx.breusch_godfrey(e, X, L)
        ref = dg.acorr_breusch_godfrey(ols_climate.sm, nlags=L)
        assert np.isclose(mine["stat"], ref[0]) and np.isclose(mine["F"], ref[2])
    lb = dg.acorr_ljungbox(e, lags=[10])
    assert np.isclose(dx.ljung_box(e, 10)["stat"], lb["lb_stat"].iloc[0])
    from statsmodels.stats.stattools import durbin_watson
    assert np.isclose(dx.durbin_watson(e), durbin_watson(e))


def test_exact_durbin_watson_against_monte_carlo():
    rng = np.random.default_rng(11)
    n = 30
    X = np.c_[np.ones(n), rng.normal(size=(n, 2)), np.arange(n)]
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.3 * u[t - 1] + rng.normal()
    y = X @ [1, 1, -1, 0.1] + u
    e = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    d = dx.durbin_watson_exact(e, X)
    M = np.eye(n) - X @ np.linalg.pinv(X)
    sims = np.array([dx.durbin_watson(M @ rng.normal(size=n)) for _ in range(20000)])
    assert abs(d["p"] - np.mean(sims < d["stat"])) < 0.012


def test_reset_vif_influence_match_statsmodels(ols_climate):
    r = ols_climate
    ref = dg.linear_reset(r.sm, power=3, use_f=True)
    assert np.isclose(dx.reset_test(r.y, r.wexog, r.fitted, 3)["stat"], float(np.squeeze(ref.fvalue)))
    vt = dx.vif_table(r.design.X)
    for i, c in enumerate(r.design.X.columns):
        if c != "const":
            assert np.isclose(vt.set_index("Variable").loc[c, "VIF"], variance_inflation_factor(r.design.X.values, i))
    inf = OLSInfluence(r.sm)
    mine = dx.influence(r.wresid, r.wexog, r.names)
    assert np.allclose(mine["Cook's D"], inf.cooks_distance[0])
    assert np.allclose(mine["Leverage"], inf.hat_matrix_diag)
    assert np.allclose(mine["DFFITS"], inf.dffits[0])
    assert np.allclose(mine[[c for c in mine.columns if c.startswith("DFBETAS")]].values, inf.dfbetas)


def test_chow_and_cusum_detect_break():
    rng = np.random.default_rng(2)
    n = 200
    x = rng.normal(size=n)
    y = np.where(np.arange(n) < 100, 1 + x, 3 + 2 * x) + rng.normal(scale=0.5, size=n)
    X = np.c_[np.ones(n), x]
    assert dx.chow_test(y, X, 100)["p"] < 1e-6
    cs = dx.cusum(y, X)
    assert not (cs["cusum_ok"] and cs["cusumsq_ok"])
    y2 = 1 + x + rng.normal(size=n)
    assert dx.chow_test(y2, X, 100)["p"] > 0.01
    assert dx.chow_test(y2, X, 199)["kind"] == "predictive"


def test_cusumsq_critical_value_approaches_kolmogorov_limit():
    # Durbin (1969): S_r − r/m behaves like √(2/m) × Brownian bridge, so c0 → 1.358·√(2/m) (5%, two-sided)
    for m, tol in ((60, 0.10), (600, 0.04)):
        lim = 1.358 * np.sqrt(2 / m)
        assert lim * (1 - tol) < dx._cusumsq_c0(m) <= lim * 1.01


def test_goldfeld_quandt_detects_increasing_variance():
    rng = np.random.default_rng(4)
    x = np.sort(rng.uniform(1, 10, 300))
    y = 1 + x + rng.normal(scale=0.3 * x)
    X = np.c_[np.ones(300), x]
    assert dx.goldfeld_quandt(y, X, x)["p"] < 1e-6
    assert dx.park(y - X @ np.linalg.lstsq(X, y, rcond=None)[0], x)["p"] < 1e-4


def test_run_all_and_time_order_flag(cd):
    full = dx.run_all(cd, time_ordered=True)
    cross = dx.run_all(cd, time_ordered=False)
    assert {"hetero", "autocorr", "spec", "normality", "multicol", "influence"} <= set(full["problems"])
    assert "autocorr" not in cross["problems"]
    assert len(full["table"]) >= 18


# ---------------------------------------------------------------- effects, prediction, what-if
def test_elasticities(farm, cd):
    eff = rp.effects_table(farm, cd).set_index("Variable")
    assert np.isclose(eff.loc["land_feddan", "Elasticity"], cd.params["ln(land_feddan)"], atol=1e-5)
    assert np.isclose(eff.loc["land_feddan", "Elasticity s.e."], cd.bse["ln(land_feddan)"], rtol=1e-4)
    lin = rg.fit(farm, rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days"]))
    e2 = rp.effects_table(farm, lin).set_index("Variable")
    xbar = farm.land_feddan.mean()
    ybar = lin.params["const"] + lin.params["land_feddan"] * xbar + lin.params["labour_days"] * farm.labour_days.mean()
    assert np.isclose(e2.loc["land_feddan", "Elasticity"], lin.params["land_feddan"] * xbar / ybar, rtol=1e-6)
    assert np.isclose(e2.loc["land_feddan", "Marginal effect dY/dX"], lin.params["land_feddan"], rtol=1e-6)


def test_turning_point(farm):
    r = rg.fit(farm, rg.ModelSpec(y="output_tonnes", x=["experience_years"], poly={"experience_years": 2}))
    tp = rp.turning_points(r, farm).iloc[0]
    assert np.isclose(tp["Turning point"], -r.params["experience_years"] / (2 * r.params["experience_years^2"]))
    rc_ = rg.fit(farm, rg.ModelSpec(y="output_tonnes", x=["experience_years"], poly={"experience_years": 2}, center=True))
    assert np.isclose(rp.turning_points(rc_, farm).iloc[0]["Turning point"], tp["Turning point"])


def test_what_if_and_predict(farm, cd):
    lin = rg.fit(farm, rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days"]))
    w = rp.what_if(farm, lin, {"land_feddan": 2.0}, mode="abs")
    assert np.isclose(w["change"], 2 * lin.params["land_feddan"])
    w2 = rp.what_if(farm, cd, {"land_feddan": 10.0}, mode="pct")
    assert np.isclose(w2["pct"], 100 * (1.1 ** cd.params["ln(land_feddan)"] - 1))
    p = rg.predict(cd, rg.baseline_values(farm, cd))
    assert p["ci_low"].iloc[0] < p["fit"].iloc[0] < p["ci_high"].iloc[0]
    assert p["pi_low"].iloc[0] < p["ci_low"].iloc[0]


def test_nls_recovers_parameters():
    rng = np.random.default_rng(8)
    x = rng.uniform(1, 20, 300)
    d = pd.DataFrame({"x": x, "y": 3 * x ** 0.6 * np.exp(rng.normal(0, 0.03, 300))})
    r = rg.fit_nls(d, "y", "x", "power")
    assert np.allclose(r.params.values, [3, 0.6], rtol=0.03) and r.rsquared > 0.95
    d2 = pd.DataFrame({"x": x, "y": 10 * (1 - np.exp(-0.2 * (x + 1))) + rng.normal(0, 0.1, 300)})
    r2 = rg.fit_nls(d2, "y", "x", "mitscherlich")
    assert np.allclose(r2.params.values, [10, 0.2, 1], rtol=0.1, atol=0.15)


# ---------------------------------------------------------------- tables, exports, code
def test_journal_table_and_exports(farm, cd):
    lin = rg.fit(farm, rg.ModelSpec(y="output_tonnes", x=["land_feddan"]))
    jt, note = rp.journal_table([lin, cd], ["(1)", "(2)"])
    assert jt.iloc[-1, 0] == "Std. errors" and jt.columns.tolist() == ["", "(1)", "(2)"]
    assert jt[jt[""] == "const"].index[0] > jt[jt[""] == "ln(land_feddan)"].index[0]  # constant last
    assert "***" in jt.loc[jt[""] == "ln(land_feddan)", "(2)"].iloc[0]
    tex = rp.to_latex(jt, note)
    assert "\\toprule" in tex and "\\bottomrule" in tex and "threeparttable" in tex
    docx = rp.to_docx([("h", "x"), ("table", jt), ("p", "<b>t</b>"), ("bullets", ["a"])], "T", rtl=True)
    assert zipfile.ZipFile(BytesIO(docx)).testzip() is None
    txt, refs = rp.methodology(cd, True, "ar")
    assert "المربعات الصغرى" in txt and any("Breusch" in x for x in refs)
    assert "Significant" in rp.interpret(cd, dx.run_all(cd), lang="en")
    assert rp.policy_brief(farm, cd, dx.run_all(cd))["bullets"]


def test_generated_python_code_reproduces_estimates(farm, tmp_path):
    spec = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "experience_years"], categorical=["irrigation"],
                        form="loglog", nolog=["experience_years"], poly={"experience_years": 2},
                        interactions=[("land_feddan", "irrigation")], center=True)
    r = rg.fit(farm, spec, cov_type="HC3")
    f = tmp_path / "d.csv"
    farm.to_csv(f, index=False)
    code = rp.code_python(r, str(f)).replace("print(res.summary())", "").split("# ---- diagnostics")[0]
    ns = {}
    exec(code, ns)  # noqa: S102 — generated by our own template
    assert np.allclose(ns["res"].params.values, r.params.values) and np.allclose(ns["res"].bse.values, r.bse.values)
    assert "vcovHC(m, type = \"HC3\")" in rp.code_r(r) and "vce(hc3)" in rp.code_stata(r)
