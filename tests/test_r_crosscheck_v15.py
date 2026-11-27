"""v1.5 cross-validation against R: the R code exported by Aetheris (model_code.code_r) is executed and must
reproduce the platform's estimates and standard errors (AER, nnet, MASS, pscl, survival, systemfit).
Skipped when R or a package is missing."""
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import count_surv as cs
from analytics import iv
from analytics import ldv
from analytics import model_code as mc
from analytics import regression as rg
from data.samples import household_survey_demo, labour_iv_demo, market_sem_demo

warnings.filterwarnings("ignore")
PKGS = ["AER", "nnet", "MASS", "pscl", "survival", "systemfit", "sandwich", "lmtest"]
R_OK = bool(shutil.which("Rscript")) and subprocess.run(
    ["Rscript", "-e", f"for (p in c({', '.join(repr(p) for p in PKGS)})) library(p, character.only = TRUE)"],
    capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not R_OK, reason="R packages for the v1.5 cross-check are not installed")

H = household_survey_demo()
XB = ["education_years", "farm_size_feddan", "distance_market_km", "credit_access"]


def run(code: str, tmp_path, tail: str) -> pd.DataFrame:
    (tmp_path / "m.R").write_text(code + "\n" + tail)
    p = subprocess.run(["Rscript", "m.R"], cwd=tmp_path, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    return pd.read_csv(tmp_path / "out.csv")


def data(tmp_path, df):
    f = tmp_path / "d.csv"
    df.to_csv(f, index=False)
    return str(f)


OUT = 'write.csv(data.frame(b = B, se = SE), "out.csv", row.names = FALSE)\n'


@pytest.mark.parametrize("link", ["logit", "probit", "cloglog"])
def test_binary(link, tmp_path):
    spec = rg.ModelSpec(y="adopt_technology", x=XB, categorical=["region"], form="linlog",
                        nolog=["education_years", "distance_market_km", "credit_access"])
    r = ldv.fit_binary(H, spec, link)
    o = run(mc.code_r(r, data(tmp_path, H)), tmp_path, "B <- coef(m); SE <- sqrt(diag(vcov(m)))\n" + OUT)
    # logit: observed = expected information. probit / cloglog: Aetheris (like Stata) uses the observed
    # information, R's glm the expected (Fisher) information — identical estimates, SEs differ slightly.
    assert np.allclose(o.b, r.params.values, rtol=1e-5, atol=1e-6)
    assert np.allclose(o.se, r.bse.values, rtol=1e-4 if link == "logit" else 1e-2)


def test_mnlogit(tmp_path):
    r = ldv.fit_mnlogit(H, rg.ModelSpec(y="technology_choice", x=XB), base="Traditional")
    o = run(mc.code_r(r, data(tmp_path, H)), tmp_path,
            "B <- as.vector(t(coef(m))); SE <- as.vector(t(summary(m)$standard.errors))\n" + OUT)
    assert np.allclose(o.b, r.params.values, atol=1e-4) and np.allclose(o.se, r.bse.values, rtol=2e-3)


@pytest.mark.parametrize("distr", ["logit", "probit"])
def test_ordered(distr, tmp_path):
    r = ldv.fit_ordered(H, rg.ModelSpec(y="satisfaction_1to5", x=["extension_visits", "education_years",
                                                                     "distance_market_km", "adopt_technology"]), distr)
    o = run(mc.code_r(r, data(tmp_path, H)), tmp_path,
            "B <- c(coef(m), m$zeta); SE <- sqrt(diag(vcov(m)))\n" + OUT)
    assert np.allclose(o.b, r.params.values, atol=1e-4) and np.allclose(o.se, r.bse.values, rtol=2e-3)


def test_tobit(tmp_path):
    spec = rg.ModelSpec(y="fertilizer_spend_egp", x=XB + ["adopt_technology"])
    r = ldv.fit_tobit(H, spec)
    o = run(mc.code_r(r, data(tmp_path, H)), tmp_path,
            "B <- c(coef(m), m$scale); SE <- sqrt(diag(vcov(m)))\nSE[length(SE)] <- SE[length(SE)] * m$scale\n" + OUT)
    assert np.allclose(o.b, r.params.values, rtol=1e-4) and np.allclose(o.se, r.bse.values, rtol=1e-3)


def test_fractional(tmp_path):
    r = ldv.fit_fractional(H, rg.ModelSpec(y="share_sold", x=XB))
    o = run(mc.code_r(r, data(tmp_path, H)), tmp_path, "B <- coef(m); SE <- sqrt(diag(sandwich(m)))\n" + OUT)
    assert np.allclose(o.b, r.params.values, rtol=1e-6) and np.allclose(o.se, r.bse.values, rtol=1e-5)


@pytest.mark.parametrize("model", ["poisson", "ppml", "negbin"])
def test_count_glm(model, tmp_path):
    r = cs.fit_count(H, rg.ModelSpec(y="extension_visits", x=XB, categorical=["region"]), model)
    tail = ("B <- coef(m); SE <- sqrt(diag(" + ("sandwich(m)" if model == "ppml" else "vcov(m)") + "))\n"
            + ("B <- c(B, 1 / m$theta); SE <- c(SE, m$SE.theta / m$theta^2)\n" if model == "negbin" else "") + OUT)
    o = run(mc.code_r(r, data(tmp_path, H)), tmp_path, tail)
    # NB2: R's glm.nb reports β SEs conditional on θ; Aetheris / Stata nbreg use the joint observed information
    assert np.allclose(o.b, r.params.values, rtol=1e-4, atol=1e-6)
    assert np.allclose(o.se, r.bse.values, rtol=5e-3 if model != "negbin" else 8e-2)


@pytest.mark.parametrize("model", ["zip", "zinb", "hurdle_p", "hurdle_nb"])
def test_zero_models(model, tmp_path):
    r = cs.fit_count(H, rg.ModelSpec(y="extension_visits", x=XB), model,
                     inflate=["distance_market_km"] if model in ("zip", "zinb") else None)
    o = run(mc.code_r(r, data(tmp_path, H)), tmp_path,
            "B <- coef(m); SE <- sqrt(diag(vcov(m)))\n" + OUT)
    k = len(XB) + 1
    b, se = r.params.values, r.bse.values
    if model in ("zip", "zinb"):        # Aetheris: inflate, count (, alpha) — R: count, zero
        kz = 2
        mine_b, mine_se = np.r_[b[kz:kz + k], b[:kz]], np.r_[se[kz:kz + k], se[:kz]]
    else:                                # Aetheris: zero, count — R: count, zero
        mine_b, mine_se = np.r_[b[k:2 * k], b[:k]], np.r_[se[k:2 * k], se[:k]]
    assert np.allclose(o.b.values[:2 * k if model.startswith("hurdle") else k + 2], mine_b, atol=2e-3)
    # pscl uses optim's numerical BFGS Hessian; Aetheris refines with Newton steps — SEs agree to a few %
    assert np.allclose(o.se.values[:len(mine_se)], mine_se, rtol=3e-2)


def test_cox_and_aft(tmp_path):
    spec = rg.ModelSpec(y="years_to_adoption", x=XB, form="linlog", nolog=["education_years", "distance_market_km", "credit_access"])
    r = cs.fit_cox(H, spec, "years_to_adoption", "adopted_by_end")
    o = run(mc.code_r(r, data(tmp_path, H)), tmp_path, "B <- coef(m); SE <- sqrt(diag(vcov(m)))\n" + OUT)
    assert np.allclose(o.b, r.params.values, rtol=1e-6) and np.allclose(o.se, r.bse.values, rtol=1e-5)
    for dist in ("weibull", "exponential", "lognormal", "loglogistic"):
        a = cs.fit_aft(H, spec, "years_to_adoption", "adopted_by_end", dist)
        o = run(mc.code_r(a, data(tmp_path, H)), tmp_path,
                "B <- coef(m); SE <- sqrt(diag(vcov(m)))[seq_along(B)]\n" + OUT)
        k = len(o)
        assert np.allclose(o.b, a.params.values[:k], rtol=1e-4, atol=1e-6), dist
        assert np.allclose(o.se, a.bse.values[:k], rtol=2e-3), dist


def test_km_logrank(tmp_path):
    km = cs.kaplan_meier(H.years_to_adoption, H.adopted_by_end)
    lr = cs.logrank(H.years_to_adoption, H.adopted_by_end, H.region)
    code = (f'library(survival); df <- read.csv("{data(tmp_path, H)}")\n'
            "f <- survfit(Surv(years_to_adoption, adopted_by_end) ~ 1, data = df, conf.type = 'log-log')\n"
            "s <- summary(f); lr <- survdiff(Surv(years_to_adoption, adopted_by_end) ~ region, data = df)\n"
            "B <- c(s$surv, lr$chisq); SE <- c(s$lower, NA)\n")
    o = run(code, tmp_path, OUT)
    n = len(km)
    assert np.allclose(o.b.values[:n], km["survival"].values) and np.allclose(o.se.values[:n], km["CI low"].values, atol=1e-8)
    assert np.isclose(o.b.values[n], lr["chi2"])


def test_iv_2sls(tmp_path):
    df = labour_iv_demo()
    sp = iv.IVSpec(y="wage_egp_hour", exog=["experience_years", "female", "urban"], endog=["education_years"],
                   instruments=["distance_college_km", "father_education", "mother_education"], form="loglin")
    for ct in ("nonrobust", "HC1"):
        r = iv.iv_fit(df, sp, "2sls", ct)
        tail = ("B <- coef(m); SE <- sqrt(diag(" + ("vcov(m)" if ct == "nonrobust" else 'vcovHC(m, type = "HC1")') + "))\n"
                "d <- summary(m, diagnostics = TRUE)$diagnostics\n" + OUT +
                'write.csv(d, "diag.csv")\n')
        o = run(mc.code_r(r, data(tmp_path, df)), tmp_path, tail)
        assert np.allclose(o.b, r.params.values) and np.allclose(o.se, r.bse.values)
    d = pd.read_csv(tmp_path / "diag.csv", index_col=0)
    t = iv.iv_tests(df, iv.iv_fit(df, sp, "2sls")).set_index("Test")
    assert np.isclose(d.loc["Weak instruments", "statistic"], t.loc["Cragg-Donald Wald F (min. eigenvalue)", "Statistic"])
    assert np.isclose(d.loc["Wu-Hausman", "statistic"], t.loc["Wu-Hausman F", "Statistic"])
    assert np.isclose(d.loc["Sargan", "statistic"], t.loc["Sargan χ² (homoskedastic)", "Statistic"])


@pytest.mark.parametrize("meth", ["2sls", "3sls", "sur", "itsur"])
def test_systems(meth, tmp_path):
    m = market_sem_demo()
    if meth in ("sur", "itsur"):
        eqs = [iv.Equation("Demand", "quantity", [], ["price", "income"]), iv.Equation("Price", "price", [], ["rainfall", "input_cost"])]
    else:
        eqs = [iv.Equation("Demand", "quantity", ["price"], ["income"]), iv.Equation("Supply", "quantity", ["price"], ["rainfall", "input_cost"])]
    ex = ["income", "rainfall", "input_cost"]
    s = iv.system_fit(m, eqs, ex, meth)
    code = mc.system_code(s, eqs, ex, data(tmp_path, m))["R"]
    o = run(code, tmp_path, "B <- coef(m); SE <- sqrt(diag(vcov(m)))\n" + OUT)
    mine_b = np.concatenate([e.params.values for e in s.equations])
    mine_se = np.concatenate([e.bse.values for e in s.equations])
    assert np.allclose(o.b, mine_b, rtol=1e-6, atol=1e-8)
    assert np.allclose(o.se, mine_se, rtol=1e-4 if meth != "itsur" else 1e-3)
