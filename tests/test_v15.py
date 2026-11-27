"""v1.5 engines validated against linearmodels, statsmodels and lifelines, closed-form identities and
simulation recovery (IV / SEM §1.2, limited dependent variables §1.4, count & survival §1.5)."""
import warnings
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from analytics import count_surv as cs
from analytics import iv
from analytics import ldv
from analytics import margins as mg
from analytics import model_code as mc
from analytics import model_report as mr
from analytics import regression as rg
from data.samples import household_survey_demo, labour_iv_demo, market_sem_demo

warnings.filterwarnings("ignore")
H = household_survey_demo()
L = labour_iv_demo()
M = market_sem_demo()
XB = ["education_years", "farm_size_feddan", "distance_market_km", "credit_access"]
IVS = iv.IVSpec(y="wage_egp_hour", exog=["experience_years", "female", "urban"], endog=["education_years"],
                instruments=["distance_college_km", "father_education", "mother_education"], form="loglin")


def _lm_data():
    d = L.assign(lw=np.log(L.wage_egp_hour), const=1.0)
    return d, d[["const", "experience_years", "female", "urban"]], d[["distance_college_km", "father_education", "mother_education"]]


# ---------------------------------------------------------------- IV
@pytest.mark.parametrize("method,cls", [("2sls", "IV2SLS"), ("liml", "IVLIML")])
@pytest.mark.parametrize("cov,lcov,small", [("nonrobust", "unadjusted", True), ("HC1", "robust", True),
                                            ("nonrobust", "unadjusted", False)])
def test_iv_matches_linearmodels(method, cls, cov, lcov, small):
    lm = pytest.importorskip("linearmodels.iv")
    d, ex, Z = _lm_data()
    r = iv.iv_fit(L, IVS, method, cov, small=small)
    ref = getattr(lm, cls)(d.lw, ex, d.education_years, Z).fit(cov_type=lcov, debiased=small)
    assert np.allclose(r.params.values, ref.params.values)
    # robust LIML: Aetheris uses the k-class sandwich (X̃′X)⁻¹X̃′ΩX̃(X′X̃)⁻¹ as Stata; linearmodels differs at the 4th digit
    tol = 1e-8 if not (method == "liml" and cov != "nonrobust") else 5e-3
    assert np.allclose(r.bse.values, ref.std_errors.values, rtol=tol)


def test_gmm_fuller_cluster_and_tests_match_linearmodels():
    lm = pytest.importorskip("linearmodels.iv")
    d, ex, Z = _lm_data()
    g = iv.iv_fit(L, IVS, "gmm", "HC0", small=False)
    ref = lm.IVGMM(d.lw, ex, d.education_years, Z).fit(cov_type="robust", debiased=False)
    assert np.allclose(g.params.values, ref.params.values) and np.allclose(g.bse.values, ref.std_errors.values)
    assert np.isclose(g.extra["J"], ref.j_stat.stat, rtol=1e-3)
    f = iv.iv_fit(L, IVS, "fuller")
    ref = lm.IVLIML(d.lw, ex, d.education_years, Z, fuller=1).fit(cov_type="unadjusted", debiased=True)
    assert np.allclose(f.params.values, ref.params.values)
    c = iv.iv_fit(L, IVS, "2sls", "cluster", {"groups": "region"})
    ref = lm.IV2SLS(d.lw, ex, d.education_years, Z).fit(cov_type="clustered", clusters=pd.factorize(d.region)[0], debiased=True)
    assert np.allclose(c.bse.values, ref.std_errors.values)
    r = iv.iv_fit(L, IVS, "2sls")
    ref = lm.IV2SLS(d.lw, ex, d.education_years, Z).fit(cov_type="unadjusted")
    t = iv.iv_tests(L, r).set_index("Test")
    assert np.isclose(t.loc["Sargan χ² (homoskedastic)", "Statistic"], ref.sargan.stat)
    assert np.isclose(t.loc["Basmann χ²", "Statistic"], ref.basmann.stat)
    assert np.isclose(t.loc["Wu-Hausman F", "Statistic"], ref.wu_hausman().stat)
    assert np.isclose(t.loc["Durbin χ²", "Statistic"], ref.durbin().stat)
    fs = iv.first_stage(r).iloc[0]
    assert np.isclose(fs["Partial R²"], ref.first_stage.diagnostics.loc["education_years", "partial.rsquared"])
    assert np.isclose(t.loc["Cragg-Donald Wald F (min. eigenvalue)", "Statistic"], fs["F (excluded IVs)"])


def test_iv_recovers_true_return_to_schooling():
    r = iv.iv_fit(L, IVS, "2sls")
    o = iv.iv_fit(L, IVS, "ols")
    assert abs(r.params["education_years"] - 0.075) < 2.5 * r.bse["education_years"]
    assert o.params["education_years"] > r.params["education_years"]     # ability bias inflates OLS


def test_underidentified_raises():
    with pytest.raises(ValueError):
        iv.iv_fit(L, iv.IVSpec(y="wage_egp_hour", endog=["education_years", "experience_years"],
                               instruments=["distance_college_km"]), "2sls")


# ---------------------------------------------------------------- systems
EQS = [iv.Equation("Demand", "quantity", ["price"], ["income"]), iv.Equation("Supply", "quantity", ["price"], ["rainfall", "input_cost"])]
EX = ["income", "rainfall", "input_cost"]


def test_identification_and_ils_equals_2sls():
    idt = iv.identification(M, EQS, EX)
    assert list(idt["Order condition"]) == ["over-identified", "exactly identified"]
    assert (idt["Rank condition"] == "satisfied").all()
    eqs = [iv.Equation("Supply", "quantity", ["price"], ["rainfall", "input_cost"]),
           iv.Equation("Demand", "quantity", ["price"], ["income", "rainfall"])]
    s_ils = iv.system_fit(M, eqs, EX, "ils")
    s_2 = iv.system_fit(M, eqs, EX, "2sls")
    for a, b in zip(s_ils.equations, s_2.equations):
        assert np.allclose(a.params.values, b.params.values)


def test_3sls_sur_match_linearmodels():
    sysm = pytest.importorskip("linearmodels.system")
    mm = M.assign(const=1.0)
    eqd = OrderedDict(Demand={"dependent": mm.quantity, "exog": mm[["const", "income"]], "endog": mm[["price"]],
                              "instruments": mm[["rainfall", "input_cost"]]},
                      Supply={"dependent": mm.quantity, "exog": mm[["const", "rainfall", "input_cost"]], "endog": mm[["price"]],
                              "instruments": mm[["income"]]})
    ref = sysm.IV3SLS(eqd).fit(cov_type="unadjusted", debiased=False)
    s = iv.system_fit(M, EQS, EX, "3sls")
    mine = pd.concat([pd.Series(e.params.values, index=[f"{n}_{c}" for c in e.names]) for n, e in zip(["Demand", "Supply"], s.equations)])
    mse = pd.concat([pd.Series(e.bse.values, index=[f"{n}_{c}" for c in e.names]) for n, e in zip(["Demand", "Supply"], s.equations)])
    assert np.allclose(mine[ref.params.index], ref.params.values) and np.allclose(mse[ref.std_errors.index], ref.std_errors.values)
    eqs2 = [iv.Equation("Demand", "quantity", [], ["price", "income"]), iv.Equation("Price", "price", [], ["rainfall", "input_cost"])]
    s = iv.system_fit(M, eqs2, EX, "sur")
    ref = sysm.SUR(OrderedDict(Demand={"dependent": mm.quantity, "exog": mm[["const", "price", "income"]]},
                               Price={"dependent": mm.price, "exog": mm[["const", "rainfall", "input_cost"]]})).fit(cov_type="unadjusted", method="gls")
    assert np.allclose(np.concatenate([e.params.values for e in s.equations]), ref.params.values)
    assert np.allclose(np.concatenate([e.bse.values for e in s.equations]), ref.std_errors.values)


# ---------------------------------------------------------------- binary / multinomial / ordered
def test_binary_ame_matches_statsmodels_and_fit_measures():
    spec = rg.ModelSpec(y="adopt_technology", x=XB)
    r = ldv.fit_binary(H, spec, "logit")
    me = mg.marginal_effects(H, r, r.extra["pred"]).set_index("Variable")
    ref = r.extra["sm"].get_margeff(at="overall", dummy=True).summary_frame()
    for v in ["education_years", "farm_size_feddan", "distance_market_km"]:
        assert np.isclose(me.loc[v, "Effect"], ref.loc[v, "dy/dx"], rtol=1e-5)
        assert np.isclose(me.loc[v, "Std. Err."], ref.loc[v, "Std. Err."], rtol=1e-3)
    assert np.isclose(me.loc["credit_access (0 → 1)", "Effect"], ref.loc["credit_access", "dy/dx"], rtol=1e-5)


def test_roc_auc_equals_concordance():
    r = ldv.fit_binary(H, rg.ModelSpec(y="adopt_technology", x=XB), "probit")
    a = ldv.roc(r.y, r.fitted)["auc"]
    y, p = r.y.astype(int), r.fitted
    pos, neg = p[y == 1], p[y == 0]
    brute = (np.sum(pos[:, None] > neg[None, :]) + 0.5 * np.sum(pos[:, None] == neg[None, :])) / (len(pos) * len(neg))
    assert np.isclose(a, brute)
    hl = ldv.hosmer_lemeshow(r.y, r.fitted)
    assert hl["df"] == 8 and 0 <= hl["p"] <= 1


def test_mnlogit_ame_matches_statsmodels_and_iia():
    r = ldv.fit_mnlogit(H, rg.ModelSpec(y="technology_choice", x=XB), base="Traditional")
    me = mg.marginal_effects(H, r, r.extra["pred"], outputs=r.extra["outputs"])
    ref = r.extra["sm"].get_margeff().summary_frame()
    e = me[me["Variable"] == "education_years"]["Effect"].values
    assert np.allclose(e, ref.xs("education_years", level=1)["dy/dx"].values, rtol=1e-4)
    t = ldv.iia_test(H, r)
    assert len(t) == 2


def test_ordered_cutpoints_and_brant():
    r = ldv.fit_ordered(H, rg.ModelSpec(y="satisfaction_1to5", x=["extension_visits", "education_years", "distance_market_km"]))
    cuts = r.params.values[3:]
    assert np.all(np.diff(cuts) > 0)
    P = r.extra["probs"]
    assert np.allclose(P.sum(axis=1), 1)
    b = ldv.brant_test(r)
    assert b.iloc[0]["df"] == 3 * 3 and b.iloc[0]["p-value"] > 0.01    # data generated with parallel lines


# ---------------------------------------------------------------- censored / selection
def test_tobit_recovers_truth_and_mcdonald_moffitt():
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(size=n)
    ystar = 1 + 2 * x + rng.normal(0, 1.5, n)
    d = pd.DataFrame({"x": x, "y": np.maximum(ystar, 0)})
    r = ldv.fit_tobit(d, rg.ModelSpec(y="y", x=["x"]))
    assert np.allclose(r.params.values, [1, 2, 1.5], atol=0.08)
    me = mg.marginal_effects(d, r, r.extra["pred"], outputs=r.extra["outputs"]).set_index("Outcome")
    p = r.extra["pred"](r.design.X.to_numpy(float), r.params.values)
    assert np.isclose(me.loc["E[y] (unconditional)", "Effect"], r.params["x"] * p[:, 1].mean(), rtol=1e-3)


def test_cragg_equals_probit_plus_truncreg():
    import statsmodels.api as sm
    spec = rg.ModelSpec(y="fertilizer_spend_egp", x=["farm_size_feddan", "education_years", "credit_access"])
    c = ldv.fit_hurdle(H, spec, kind="cragg")
    X = c.design.X.to_numpy(float)
    pos = c.y > 0
    pr = sm.Probit(pos.astype(float), X).fit(disp=0)
    tr_ = ldv.fit_truncreg(H, spec, lower=0.0)
    assert np.isclose(c.llf, pr.llf + tr_.llf, rtol=1e-6)
    assert np.allclose(c.params.values[:X.shape[1]], pr.params, atol=1e-4)


def test_heckman_two_step_and_ml_recover_truth():
    rng = np.random.default_rng(3)
    n = 6000
    z, x, w = rng.normal(size=(3, n))
    u, v = rng.multivariate_normal([0, 0], [[1, 0.6], [0.6, 1]], n).T
    s = (0.3 + 0.8 * z + 0.4 * x + v > 0).astype(int)
    y = np.where(s == 1, 1 + 0.5 * x + 0.8 * u, np.nan)
    d = pd.DataFrame({"y": y, "x": x, "z": z, "s": s})
    out, sel = rg.ModelSpec(y="y", x=["x"]), rg.ModelSpec(y="s", x=["x", "z"])
    two = ldv.fit_heckman(d, out, sel, "twostep")
    ml = ldv.fit_heckman(d, out, sel, "ml")
    for r in (two, ml):
        assert abs(r.params["x"] - 0.5) < 0.06 and abs(r.extra["rho"] - 0.6) < 0.15 and abs(r.extra["sigma"] - 0.8) < 0.08
    assert ml.extra["p_LR_rho"] < 0.01


def test_fractional_and_truncreg():
    f = ldv.fit_fractional(H, rg.ModelSpec(y="share_sold", x=XB))
    assert np.all((f.fitted > 0) & (f.fitted < 1))
    rng = np.random.default_rng(5)
    x = rng.normal(size=20000)
    y = 1 + x + rng.normal(size=20000)
    d = pd.DataFrame({"x": x, "y": y})[y > 0.5]
    t = ldv.fit_truncreg(d, rg.ModelSpec(y="y", x=["x"]), lower=0.5)
    assert np.allclose(t.params.values, [1, 1, 1], atol=0.06)


# ---------------------------------------------------------------- count
def test_count_models_match_statsmodels_and_tests():
    import statsmodels.api as sm
    spec = rg.ModelSpec(y="extension_visits", x=XB)
    p = cs.fit_count(H, spec, "poisson")
    ref = sm.Poisson(p.y, p.design.X).fit(disp=0)
    assert np.allclose(p.params.values, ref.params.values) and np.isclose(p.llf, ref.llf)
    assert p.extra["overdispersion"]["p"] < 0.01                       # NB data → overdispersion detected
    nb = cs.fit_count(H, spec, "negbin")
    assert nb.extra["lr_alpha"]["p"] < 0.01 and nb.llf > p.llf
    z = cs.fit_count(H, spec, "zip", inflate=["distance_market_km"])
    assert z.extra["vuong"]["p"] < 0.05
    for m in ("hurdle_p", "hurdle_nb"):
        h = cs.fit_count(H, spec, m)
        pr = h.extra["pred"](h.design.X.to_numpy(float), h.params.values)
        assert abs(pr.mean() - h.y.mean()) / h.y.mean() < 0.05       # hurdle reproduces the mean closely
        fr = h.extra["freq"]
        assert abs(fr.loc[0, "Observed share"] - fr.loc[0, "Predicted share"]) < 0.01   # zeros fitted exactly in-sample


# ---------------------------------------------------------------- survival
def test_km_cox_logrank_match_lifelines():
    lf = pytest.importorskip("lifelines")
    from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
    km = cs.kaplan_meier(H.years_to_adoption, H.adopted_by_end)
    kmf = lf.KaplanMeierFitter().fit(H.years_to_adoption, H.adopted_by_end)
    sf = kmf.survival_function_.iloc[:, 0]
    assert np.allclose(km["survival"].values, sf.loc[km["time"].values].values)
    assert np.isclose(cs.logrank(H.years_to_adoption, H.adopted_by_end, H.region)["chi2"],
                      multivariate_logrank_test(H.years_to_adoption, H.region, H.adopted_by_end).test_statistic)
    spec = rg.ModelSpec(y="years_to_adoption", x=XB)
    c = cs.fit_cox(H, spec, "years_to_adoption", "adopted_by_end")
    d = H[["years_to_adoption", "adopted_by_end"] + XB]
    cph = lf.CoxPHFitter().fit(d, "years_to_adoption", "adopted_by_end")
    assert np.allclose(c.params.values, cph.params_[XB].values, atol=1e-5)
    assert np.isclose(c.extra["stats"]["Concordance (Harrell C)"], cph.concordance_index_, atol=1e-4)
    ph = proportional_hazard_test(cph, d, time_transform="rank").summary
    mine = c.extra["ph_test"].set_index("Variable")
    for v in XB:
        assert np.isclose(mine.loc[v, "χ²"], ph.loc[v, "test_statistic"], rtol=0.02)


def test_weibull_aft_recovers_shape_and_discrete_hazard():
    spec = rg.ModelSpec(y="years_to_adoption", x=XB, form="linlog", nolog=["education_years", "distance_market_km", "credit_access"])
    w = cs.fit_aft(H, spec, "years_to_adoption", "adopted_by_end", "weibull")
    assert abs(w.extra["stats"]["Weibull shape p = 1/σ"] - 1.3) < 0.12
    hr = w.extra["ph_form"].set_index("Variable")["Hazard ratio exp(−β/σ)"]
    assert abs(np.log(hr["credit_access"]) - 0.45) < 0.15
    dh = cs.fit_discrete_hazard(H, spec, "years_to_adoption", "adopted_by_end", link="cloglog")
    assert abs(dh.params["credit_access"] - 0.45) < 0.15            # cloglog ≈ continuous-time PH coefficient
    pp = dh.extra["person_period"]
    assert pp["event_now"].sum() == H["adopted_by_end"].sum()


# ---------------------------------------------------------------- reporting & code
@pytest.mark.parametrize("builder", [
    lambda: ldv.fit_binary(H, rg.ModelSpec(y="adopt_technology", x=XB), "logit"),
    lambda: ldv.fit_mnlogit(H, rg.ModelSpec(y="technology_choice", x=XB)),
    lambda: ldv.fit_ordered(H, rg.ModelSpec(y="satisfaction_1to5", x=XB)),
    lambda: ldv.fit_tobit(H, rg.ModelSpec(y="fertilizer_spend_egp", x=XB)),
    lambda: cs.fit_count(H, rg.ModelSpec(y="extension_visits", x=XB), "negbin"),
    lambda: cs.fit_cox(H, rg.ModelSpec(y="years_to_adoption", x=XB), "years_to_adoption", "adopted_by_end"),
    lambda: iv.iv_fit(L, IVS, "2sls"),
])
def test_reporting_layer(builder):
    r = builder()
    df = L if "Z" in r.extra else H
    for lang in ("en", "ar"):
        d = mr.diagnostics(df, r, lang)
        assert not d.empty
        txt, refs = mr.methodology(r, lang)
        assert len(txt) > 80 and refs
        eff = mg.marginal_effects(df, r, r.extra["pred"], outputs=r.extra.get("outputs")) if r.extra.get("pred") and "Z" not in r.extra else None
        assert len(mr.interpret(df, r, eff, lang)) > 50
    for f in (mc.code_python, mc.code_r, mc.code_stata):
        assert len(f(r)) > 50


def test_generated_python_code_runs_for_logit_and_count(tmp_path):
    f = tmp_path / "h.csv"
    H.to_csv(f, index=False)
    for r in (ldv.fit_binary(H, rg.ModelSpec(y="adopt_technology", x=XB, categorical=["region"]), "logit"),
              cs.fit_count(H, rg.ModelSpec(y="extension_visits", x=XB), "poisson")):
        ns = {}
        exec(mc.code_python(r, str(f)).replace("print(", "(lambda *a, **k: None)("), ns)  # noqa: S102
        assert np.allclose(np.asarray(ns["res"].params), r.params.values, rtol=1e-6)


# ---------------------------------------------------------------- multinomial probit
def test_mnprobit_ghk_equals_quadrature_and_recovers_truth():
    rng = np.random.default_rng(12)
    n = 3000
    x = rng.normal(size=n)
    e = rng.normal(size=(n, 3))
    U = np.c_[np.zeros(n), 0.3 + 0.8 * x, -0.2 - 0.6 * x] + e
    d = pd.DataFrame({"x": x, "y": np.array(["A", "B", "C"])[U.argmax(axis=1)]})
    r = ldv.fit_mnprobit(d, rg.ModelSpec(y="y", x=["x"]), base="A")
    truth = np.array([0.3, 0.8, -0.2, -0.6])
    assert np.all(np.abs(r.params.values - truth) < 3.5 * r.bse.values)
    # GHK with Ω = I + 11ʹ (independent errors) reproduces the exact quadrature probabilities
    X = r.design.X.to_numpy(float)
    V = X @ r.params.values.reshape(2, 2).T
    Ug = ldv._halton(2000, 2)
    for c in range(3):
        assert np.abs(ldv._ghk(V, np.full(n, c), np.eye(2) + 1, Ug) - r.extra["probs"][:, c]).max() < 5e-4
    assert np.allclose(r.extra["probs"].sum(axis=1), 1, atol=1e-8)
    me = mg.marginal_effects(d, r, r.extra["pred"], outputs=r.extra["outputs"])
    assert abs(me["Effect"].sum()) < 1e-6          # effects on the three probabilities sum to zero


def test_mnprobit_correlated_runs_and_reports_lr():
    r = ldv.fit_mnprobit(H, rg.ModelSpec(y="technology_choice", x=XB[:3]), base="Traditional", correlated=True, draws=100)
    assert "omega_corr" in r.extra and r.extra["LR_iid"][1] == 2
    assert np.isclose(r.extra["omega"].iloc[0, 0], 2.0)
    d = mr.diagnostics(H, r, "en")
    assert d["Test"].str.contains("independent vs correlated").any()
