"""v1.10 cross-validation against R: stats::nls (CES, yield-response functions), systemfit (iterated SUR with symmetry
and homogeneity restrictions: translog cost system and LA-AIDS), lpSolve (DEA linear programmes) and lm (Cobb-Douglas
returns-to-scale test). Skipped when R or a package is missing."""
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import demand as dm
from analytics import production as pr
from data.samples import farm_efficiency_demo, fertilizer_trial_demo, firm_cost_demo, household_budget_demo

warnings.filterwarnings("ignore")
PKGS = ["systemfit", "lpSolve", "car"]
R_OK = bool(shutil.which("Rscript")) and subprocess.run(
    ["Rscript", "-e", f"for (p in c({', '.join(repr(p) for p in PKGS)})) library(p, character.only = TRUE)"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not R_OK, reason="R packages systemfit / lpSolve / car are not installed")

FE = farm_efficiency_demo()
F24 = FE[FE.year == 2024].reset_index(drop=True)
TR = fertilizer_trial_demo()
FC = firm_cost_demo()
HB = household_budget_demo()
G5 = ["cereals", "meat_fish", "dairy_eggs", "fruit_veg", "other_food"]
SH, PR = [f"w_{g}" for g in G5], [f"p_{g}" for g in G5]


def _run(code, tmp):
    (tmp / "s.R").write_text(code + '\nfor (n in names(o)) { cat("##", n, "\\n"); cat(format(as.vector(as.matrix(o[[n]])), digits = 15), "\\n") }\n')
    out = subprocess.run(["Rscript", "s.R"], capture_output=True, text=True, cwd=tmp)
    assert out.returncode == 0, out.stderr[-2500:]
    B, cur = {}, None
    for line in out.stdout.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip(); B[cur] = []
        elif cur and line.strip():
            B[cur] += [float(v) for v in line.split()]
    return {k: np.array(v) for k, v in B.items()}


def test_nls_ces_and_yield_response(tmp_path):
    F24.to_csv(tmp_path / "f.csv", index=False)
    TR.to_csv(tmp_path / "t.csv", index=False)
    B = _run(r'''
suppressMessages(library(car)); f <- read.csv("f.csv"); t <- read.csv("t.csv"); o <- list()
cd <- lm(log(output_tonnes) ~ log(land_feddan) + log(labour_days) + log(fertilizer_kg), data = f)
o$cd <- c(coef(cd), sqrt(diag(vcov(cd)))); o$rts <- linearHypothesis(cd, "log(land_feddan) + log(labour_days) + log(fertilizer_kg) = 1")$F[2]
ces <- nls(log(output_tonnes) ~ lnA - nu/rho * log(d * land_feddan^(-rho) + (1 - d) * labour_days^(-rho)), data = f,
           start = list(lnA = 0, nu = 0.8, rho = 0.3, d = 0.5), algorithm = "port", control = nls.control(maxiter = 500, tol = 1e-10))
o$ces <- c(coef(ces), sqrt(diag(vcov(ces))))
x <- t$n_kg_ha; y <- t$yield_t_ha
q <- nls(y ~ a + b*x + c*x^2, start = list(a = 3, b = 0.04, c = -1e-4)); o$quad <- c(coef(q), sqrt(diag(vcov(q))))
qp <- nls(y ~ ifelse(x < -b/(2*c), a + b*x + c*x^2, a - b^2/(4*c)), start = list(a = 2.9, b = 0.045, c = -1.2e-4)); o$qp <- c(coef(qp), sqrt(diag(vcov(qp))))
lp <- nls(y ~ a + b * pmin(x, x0), start = list(a = 3, b = 0.03, x0 = 130)); o$lp <- c(coef(lp), sqrt(diag(vcov(lp))))
mi <- nls(y ~ A - B * R^x, start = list(A = 7.3, B = 4.5, R = 0.987)); o$mi <- c(coef(mi), sqrt(diag(vcov(mi))))
''', tmp_path)
    r = pr.production_function(F24, "output_tonnes", ["land_feddan", "labour_days", "fertilizer_kg"], "cd")
    np.testing.assert_allclose(np.r_[r.b, r.se], B["cd"], rtol=1e-8)
    assert np.isclose(r.stats["t (RTS = 1)"] ** 2, B["rts"][0], rtol=1e-8)
    c = pr.production_function(F24, "output_tonnes", ["land_feddan", "labour_days"], "ces")
    p_r = B["ces"]
    got = np.r_[c.b[:3], c.b[3], c.se[:4]]
    np.testing.assert_allclose(got[:4], p_r[:4], rtol=2e-3, atol=2e-4)
    np.testing.assert_allclose(got[4:], p_r[4:], rtol=2e-2)
    x, y = TR.n_kg_ha.to_numpy(float), TR.yield_t_ha.to_numpy(float)
    for kind, key in (("quadratic", "quad"), ("quad_plateau", "qp"), ("linear_plateau", "lp"), ("mitscherlich", "mi")):
        f = pr.yield_response(x, y, kind, 12000, 35)
        np.testing.assert_allclose(f["params"], B[key][:3], rtol=1e-4, err_msg=kind)
        np.testing.assert_allclose(f["table"]["Std. Err."].to_numpy(), B[key][3:], rtol=2e-3, err_msg=kind)


def test_translog_cost_system_matches_systemfit(tmp_path):
    d = FC.copy()
    lw = np.log(d[["wage", "capital_price", "energy_price"]].to_numpy())
    lw = lw - lw.mean(0)
    lq = np.log(d.output_t.to_numpy()); lq -= lq.mean()
    X = pd.DataFrame({"lv": np.log(d.total_cost_k.to_numpy()) - lw[:, 2], "lw1": lw[:, 0] - lw[:, 2], "lw2": lw[:, 1] - lw[:, 2]})
    X["lw1sq"], X["lw12"], X["lw2sq"] = 0.5 * X.lw1 ** 2, X.lw1 * X.lw2, 0.5 * X.lw2 ** 2
    X["lq"], X["lqsq"] = lq, 0.5 * lq ** 2
    X["lw1q"], X["lw2q"] = X.lw1 * lq, X.lw2 * lq
    X["s1"], X["s2"] = d.share_labour, d.share_capital
    X.to_csv(tmp_path / "c.csv", index=False)
    B = _run(r'''
suppressMessages(library(systemfit)); d <- read.csv("c.csv"); o <- list()
eqs <- list(cost = lv ~ lw1 + lw2 + lw1sq + lw12 + lw2sq + lq + lqsq + lw1q + lw2q, s1 = s1 ~ lw1 + lw2 + lq, s2 = s2 ~ lw1 + lw2 + lq)
R <- c("s1_(Intercept) - cost_lw1 = 0", "s2_(Intercept) - cost_lw2 = 0", "s1_lw1 - cost_lw1sq = 0", "s1_lw2 - cost_lw12 = 0",
       "s2_lw1 - cost_lw12 = 0", "s2_lw2 - cost_lw2sq = 0", "s1_lq - cost_lw1q = 0", "s2_lq - cost_lw2q = 0")
fit <- systemfit(eqs, method = "SUR", data = d, restrict.matrix = R,
                 control = systemfit.control(methodResidCov = "noDfCor", maxiter = 1000, tol = 1e-12))
o$b <- coef(fit)[1:10]
''', tmp_path)
    r = pr.translog_system(FC, "cost", "total_cost_k", ["wage", "capital_price", "energy_price"], ["share_labour", "share_capital", "share_energy"], "output_t")
    # ours: α0, α1, α2, γ11, γ12, γ22, αq, αqq, γ1q, γ2q — R: const, lw1, lw2, lw1sq, lw12, lw2sq, lq, lqsq, lw1q, lw2q
    np.testing.assert_allclose(r["b"], B["b"], rtol=1e-6, atol=1e-8)


def test_laaids_matches_systemfit(tmp_path):
    d, W, lp, lx, _ = dm._prep_system(HB, SH, PR, "food_expenditure", None, True)
    z = lx - lp @ W.mean(0)
    X = pd.DataFrame(lp, columns=[f"lp{j + 1}" for j in range(5)])
    X["z"] = z
    for i in range(4):
        X[f"w{i + 1}"] = W[:, i]
    X.to_csv(tmp_path / "a.csv", index=False)
    rhs = "lp1 + lp2 + lp3 + lp4 + lp5 + z"
    B = _run(r'''
suppressMessages(library(systemfit)); d <- read.csv("a.csv"); o <- list()
eqs <- list(e1 = w1 ~ ''' + rhs + ", e2 = w2 ~ " + rhs + ", e3 = w3 ~ " + rhs + ", e4 = w4 ~ " + rhs + r''')
R <- c()
for (i in 1:4) R <- c(R, paste0(paste0("e", i, "_lp", 1:5, collapse = " + "), " = 0"))
for (i in 1:3) for (j in (i + 1):4) R <- c(R, paste0("e", i, "_lp", j, " - e", j, "_lp", i, " = 0"))
fit <- systemfit(eqs, method = "SUR", data = d, restrict.matrix = R,
                 control = systemfit.control(methodResidCov = "noDfCor", maxiter = 1000, tol = 1e-12))
o$b <- coef(fit)
''', tmp_path)
    r = dm.aids(HB, SH, PR, "food_expenditure", "laaids")
    np.testing.assert_allclose(r["table"]["Coef."].to_numpy(), B["b"], rtol=1e-6, atol=1e-9)


def test_dea_matches_lpsolve(tmp_path):
    X = F24[["land_feddan", "labour_days", "fertilizer_kg"]].to_numpy()
    Y = F24[["output_tonnes"]].to_numpy()
    pd.DataFrame(np.c_[X, Y], columns=["x1", "x2", "x3", "y"]).to_csv(tmp_path / "d.csv", index=False)
    B = _run(r'''
suppressMessages(library(lpSolve)); d <- read.csv("d.csv"); X <- as.matrix(d[, 1:3]); Y <- as.matrix(d[, 4, drop = FALSE]); n <- nrow(d); o <- list()
eff <- function(vrs) sapply(1:n, function(k) {
  A <- rbind(cbind(-X[k, ], t(X)), cbind(0, t(Y)))
  dirs <- c(rep("<=", 3), ">="); rhs <- c(rep(0, 3), Y[k, ])
  if (vrs) { A <- rbind(A, c(0, rep(1, n))); dirs <- c(dirs, "="); rhs <- c(rhs, 1) }
  lp("min", c(1, rep(0, n)), A, dirs, rhs)$solution[1] })
o$crs <- eff(FALSE); o$vrs <- eff(TRUE)
''', tmp_path)
    np.testing.assert_allclose(pr.dea(X, Y, "crs", "input"), B["crs"], rtol=1e-6)
    np.testing.assert_allclose(pr.dea(X, Y, "vrs", "input"), B["vrs"], rtol=1e-6)
