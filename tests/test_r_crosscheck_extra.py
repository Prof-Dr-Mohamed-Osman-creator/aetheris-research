"""Extra cross-validation for v1.9 and v1.10 engines whose dedicated R packages are not on the system R:
- Bayesian VAR against the R package BVAR (Kuschnig & Vashold) — marginal likelihood and posterior mean at given
  hyperparameters; uses a conda-forge R (set AETHERIS_RSCRIPT_BVAR or install to /opt/mm/renv).
- MIDAS (U-MIDAS by lm, exponential Almon by nls), bridge equations (lm) and local projections (lm + sandwich
  Newey-West, lpirfs::lp_lin conventions) re-implemented independently in base R.
- Stochastic frontiers (Aigner-Lovell-Schmidt half-normal, exponential, Battese-Coelli 1995, Battese-Coelli 1992
  panel) by independently coded likelihoods maximised with R optim.
"""
import os
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import nowcast as nc
from analytics import production as pr
from data.samples import farm_efficiency_demo, macro_var_demo, nowcast_demo

warnings.filterwarnings("ignore")
R_SYS = shutil.which("Rscript")
R_BVAR = os.environ.get("AETHERIS_RSCRIPT_BVAR", "/opt/mm/renv/bin/Rscript")


def _ok(rs, pkgs):
    return bool(rs) and os.path.exists(rs) and subprocess.run([rs, "-e", ";".join(f"library({p})" for p in pkgs)], capture_output=True).returncode == 0


SYS_OK = _ok(R_SYS, ["sandwich"])
BVAR_OK = _ok(R_BVAR, ["BVAR"])


def _run(rs, code, tmp):
    (tmp / "s.R").write_text(code + '\nfor (n in names(o)) { cat("##", n, "\\n"); cat(format(as.vector(as.matrix(o[[n]])), digits = 15), "\\n") }\n')
    out = subprocess.run([rs, "s.R"], capture_output=True, text=True, cwd=tmp)
    assert out.returncode == 0, out.stderr[-2500:]
    B, cur = {}, None
    for line in out.stdout.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip(); B[cur] = []
        elif cur and line.strip():
            B[cur] += [float(v) for v in line.split()]
    return {k: np.array(v) for k, v in B.items()}


def _macro():
    M = macro_var_demo()
    return pd.DataFrame({"dy": 100 * np.diff(np.log(M.gdp_bn)), "infl": M.inflation_pct[1:].values, "rate": M.policy_rate_pct[1:].values})


@pytest.mark.skipif(not BVAR_OK, reason="R package BVAR not available")
def test_bvar_matches_r_bvar(tmp_path):
    Y = _macro()
    Ya = Y.to_numpy()
    p = 2
    y, X = nc._lagmat(Ya, p)
    psi = np.array([nc._ar_resvar(Ya[:, j], p) for j in range(3)])
    pd.DataFrame(y).to_csv(tmp_path / "y.csv", index=False)
    pd.DataFrame(X).to_csv(tmp_path / "x.csv", index=False)
    pd.DataFrame({"psi": psi}).to_csv(tmp_path / "psi.csv", index=False)
    B = _run(R_BVAR, r'''
suppressMessages(library(BVAR)); o <- list()
Y <- as.matrix(read.csv("y.csv")); X <- as.matrix(read.csv("x.csv")); psi <- read.csv("psi.csv")$psi
M <- ncol(Y); K <- ncol(X); N <- nrow(Y)
for (own in c(0, 1)) for (lam in c(0.1, 0.3, 1.0)) {
  b <- matrix(0, K, M); b[2:(M + 1), ] <- diag(M) * own
  pars <- c(lambda = lam, alpha = 2, setNames(psi, paste0("psi", 1:M)))
  r <- BVAR:::bv_ml(hyper = c(lambda = lam), pars = pars, priors = list(var = 1e7, b = b, hyper = character(0), dummy = NULL),
                    Y = Y, X = X, XX = crossprod(X), K = K, M = M, N = N, lags = 2)
  o[[paste0("ml_", own, "_", lam)]] <- r$log_ml; o[[paste0("b_", own, "_", lam)]] <- r$beta_hat }
''', tmp_path)
    for own in (0, 1):
        for lam in (0.1, 0.3, 1.0):
            post = nc.bvar_posterior(Ya, p, lam, 2.0, psi=psi, b_own=float(own))
            key = f"{own}_{lam if lam != 1.0 else 1}"
            np.testing.assert_allclose(post["logml"], B[f"ml_{key}"][0], rtol=1e-10)
            np.testing.assert_allclose(post["B"].ravel(order="F"), B[f"b_{key}"], rtol=1e-8, atol=1e-10)


@pytest.mark.skipif(not SYS_OK, reason="R with sandwich not available")
def test_midas_bridge_lp_base_r(tmp_path):
    N = nowcast_demo()
    N.to_csv(tmp_path / "n.csv", index=False)
    Y = _macro()
    Y.to_csv(tmp_path / "m.csv", index=False)
    B = _run(R_SYS, r'''
suppressMessages(library(sandwich)); o <- list()
d <- read.csv("n.csv"); x <- d$industrial_production_mom; y <- d$gdp_growth_q
last <- which(!is.na(y)); K <- 6
rows <- last[last >= K]
Xm <- t(sapply(rows, function(i) x[i - 0:(K - 1)])); yy <- y[rows]
ylag <- c(NA, yy[-length(yy)]); ok <- !is.na(ylag)
um <- lm(yy[ok] ~ ylag[ok] + Xm[ok, ]); o$umidas <- coef(um)
nealmon <- function(t1, t2) { j <- 1:K; w <- exp(t1 * j + t2 * j^2); w / sum(w) }
dd <- data.frame(yy = yy[ok], yl = ylag[ok]); XX <- Xm[ok, ]
f <- function(p) dd$yy - (p[1] + p[2] * dd$yl + p[3] * as.vector(XX %*% nealmon(p[4], p[5])))
best <- NULL
for (st in list(c(coef(um)[1:2], sum(coef(um)[-(1:2)]), 0, 0), c(coef(um)[1:2], sum(coef(um)[-(1:2)]), 0.5, -0.1),
                c(coef(um)[1:2], sum(coef(um)[-(1:2)]), -0.5, 0), c(coef(um)[1:2], sum(coef(um)[-(1:2)]), 1, -0.2))) {
  r <- optim(st, function(p) sum(f(p)^2), method = "BFGS", control = list(maxit = 5000, reltol = 1e-14))
  if (is.null(best) || r$value < best$value) best <- r }
o$nealmon_ssr <- best$value
# bridge: quarterly means of complete quarters
q <- (seq_along(x) - 1) %/% 3
ind <- c("industrial_production_mom", "pmi", "exports_mom", "retail_sales_mom")
Q <- aggregate(d[, ind], list(q = q), function(v) if (sum(!is.na(v)) == 3) mean(v) else NA)
Q$y <- tapply(y, q, function(v) if (any(!is.na(v))) v[!is.na(v)][1] else NA)
br <- lm(y ~ ., data = Q[complete.cases(Q), c(ind, "y")]); o$bridge <- c(coef(br), sqrt(diag(vcov(br))))
# local projection (lpirfs::lp_lin conventions): shock = rate (3rd), response = dy (1st), p = 2, H = 6
m <- as.matrix(read.csv("m.csv")); Tn <- nrow(m); p <- 2
Xv <- cbind(1, m[(p):(Tn - 1), ], m[(p - 1):(Tn - 2), ]); Yv <- m[(p + 1):Tn, ]
U <- Yv - Xv %*% solve(crossprod(Xv), crossprod(Xv, Yv)); S <- crossprod(U) / (nrow(U) - ncol(Xv)); dsh <- t(chol(S))[, 3]
irf <- dsh[1]; se <- 0
for (h in 1:6) {
  tt <- (p + 1):(Tn - h + 1)
  yy2 <- m[tt + h - 1, 1]; XX2 <- cbind(1, m[tt - 1, ], m[tt - 2, ])
  fit <- lm(yy2 ~ XX2 - 1); V <- NeweyWest(fit, lag = h + 1, prewhite = FALSE, adjust = FALSE)
  irf <- c(irf, sum(coef(fit)[2:4] * dsh)); se <- c(se, sqrt(drop(t(dsh) %*% V[2:4, 2:4] %*% dsh))) }
o$lp <- c(irf, se)
''', tmp_path)
    mu = nc.midas(N, "date", "gdp_growth_q", "industrial_production_mom", 6, "umidas")
    np.testing.assert_allclose(mu["table"]["Coef."].to_numpy(), B["umidas"], rtol=1e-8)
    mn = nc.midas(N, "date", "gdp_growth_q", "industrial_production_mom", 6, "nealmon")
    assert mn["RMSE"] ** 2 * mn["N"] <= B["nealmon_ssr"][0] * (1 + 1e-6)
    np.testing.assert_allclose(mn["RMSE"] ** 2 * mn["N"], B["nealmon_ssr"][0], rtol=1e-4)
    b = nc.bridge(N, "date", "gdp_growth_q", ["industrial_production_mom", "pmi", "exports_mom", "retail_sales_mom"])
    np.testing.assert_allclose(np.r_[b["table"]["Coef."], b["table"]["Std. Err."]], B["bridge"], rtol=1e-8)
    lp = nc.local_projection(_macro(), "rate", "dy", 6, 2)
    np.testing.assert_allclose(np.r_[lp.IRF, lp.se], B["lp"], rtol=1e-6, atol=1e-10)


@pytest.mark.skipif(not SYS_OK, reason="R not available")
def test_sfa_independent_r_likelihoods(tmp_path):
    F = farm_efficiency_demo()
    F.to_csv(tmp_path / "f.csv", index=False)
    B = _run(R_SYS, r'''
d <- read.csv("f.csv"); o <- list()
y <- log(d$output_tonnes); X <- cbind(1, log(d$land_feddan), log(d$labour_days), log(d$fertilizer_kg)); k <- ncol(X)
Z <- cbind(1, d$education_years, d$extension_visits, d$credit_access)
ols <- lm.fit(X, y)$coefficients
ll_hn <- function(t) { b <- t[1:k]; sv <- exp(t[k + 1] / 2); su <- exp(t[k + 2] / 2); e <- y - X %*% b; s <- sqrt(sv^2 + su^2)
  sum(log(2) - log(s) + dnorm(e / s, log = TRUE) + pnorm(-(su / sv) * e / s, log.p = TRUE)) }
ll_ex <- function(t) { b <- t[1:k]; sv <- exp(t[k + 1] / 2); su <- exp(t[k + 2] / 2); e <- y - X %*% b
  sum(-log(su) + sv^2 / (2 * su^2) + e / su + pnorm(-e / sv - sv / su, log.p = TRUE)) }
ll_bc95 <- function(t) { b <- t[1:k]; sv <- exp(t[k + 1] / 2); su <- exp(t[k + 2] / 2); dl <- t[(k + 3):(k + 6)]
  e <- y - X %*% b; mu <- Z %*% dl; s2 <- sv^2 + su^2; mus <- (sv^2 * mu - su^2 * e) / s2; ss <- su * sv / sqrt(s2)
  sum(-0.5 * log(2 * pi * s2) - 0.5 * (e + mu)^2 / s2 + pnorm(mus / ss, log.p = TRUE) - pnorm(mu / su, log.p = TRUE)) }
fit <- function(f, st) { best <- NULL
  lo <- rep(-Inf, length(st)); lo[k + 1:2] <- -12          # keep both variances away from the degenerate boundary
  g <- function(t) { v <- -f(t); if (!is.finite(v)) 1e10 else v }
  for (m in c("BFGS", "Nelder-Mead")) { r <- optim(st, g, method = m, control = list(maxit = 50000, reltol = 1e-14))
    r <- optim(pmax(r$par, lo + 1e-6), g, method = "L-BFGS-B", lower = lo, control = list(maxit = 10000, factr = 1e2))
    if (is.null(best) || r$value < best$value) best <- r }
  best }
r1 <- fit(ll_hn, c(ols + c(0.2, 0, 0, 0), -3, -2)); o$hn <- c(-r1$value, r1$par)
r2 <- fit(ll_ex, c(ols + c(0.2, 0, 0, 0), -3, -3)); o$ex <- c(-r2$value, r2$par)
r3 <- fit(ll_bc95, c(r1$par, 0.5, 0, 0, 0)); o$bc95 <- c(-r3$value, r3$par)
# Battese-Coelli (1992) time-varying decay, mu free; parameters as Stata xtfrontier (ln sigma2, logit gamma, mu, eta)
id <- as.integer(factor(d$farm_id)); tt <- d$year - min(d$year); Ti <- tapply(tt, id, max)[id]
ll_bc92 <- function(t) { b <- t[1:k]; s2 <- exp(t[k + 1]); g <- 1 / (1 + exp(-t[k + 2])); mu <- t[k + 3]; eta <- t[k + 4]
  su2 <- g * s2; sv2 <- (1 - g) * s2; e <- as.vector(y - X %*% b); et <- exp(-eta * (tt - Ti))
  tot <- 0
  for (i in unique(id)) { w <- id == i; ee <- sum(et[w]^2); n <- sum(w); den <- sv2 + su2 * ee
    ms <- (mu * sv2 - su2 * sum(et[w] * e[w])) / den; ss2 <- su2 * sv2 / den
    tot <- tot - n / 2 * log(2 * pi) - (n - 1) / 2 * log(sv2) - 0.5 * log(den) - 0.5 * sum(e[w]^2) / sv2 +
           0.5 * ms^2 / ss2 - 0.5 * mu^2 / su2 + pnorm(ms / sqrt(ss2), log.p = TRUE) - pnorm(mu / sqrt(su2), log.p = TRUE) }
  tot }
r4 <- fit(ll_bc92, c(ols + c(0.2, 0, 0, 0), log(0.08), 0, 0.1, 0)); o$bc92 <- c(-r4$value, r4$par)
''', tmp_path)
    X3 = ["land_feddan", "labour_days", "fertilizer_kg"]
    for dist, key, z in (("hnormal", "hn", None), ("exponential", "ex", None), ("tnormal", "bc95", ["education_years", "extension_visits", "credit_access"])):
        r = pr.sfa(F, "output_tonnes", X3, dist, z=z)
        assert r["loglik"] >= B[key][0] - 1e-4, (key, r["loglik"], B[key][0])
        np.testing.assert_allclose(r["loglik"], B[key][0], atol=1e-2)        # ours is never below R optim's maximum
        np.testing.assert_allclose(r["theta"], B[key][1:], atol=2e-2, err_msg=key)
    p = pr.sfa_panel(F, "farm_id", "year", "output_tonnes", X3, "tvd")
    assert p["loglik"] >= B["bc92"][0] - 1e-4
    np.testing.assert_allclose(p["loglik"], B["bc92"][0], atol=1e-2)
    np.testing.assert_allclose(p["theta"], B["bc92"][1:], atol=2e-2)
