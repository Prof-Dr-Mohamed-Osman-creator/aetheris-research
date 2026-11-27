"""v1.7 cross-validation against R: vars (VAR, diagnostics, causality, IRF, FEVD, forecasts, SVAR, BQ), urca (Johansen,
cajorls, blrtest, alrtest, Engle-Granger ADF), cointReg (FMOLS, DOLS), ARDL (ARDL, UECM, multipliers, bounds F/t
tests, RECM), quantreg (rq + nid SE) and fGarch (GARCH(1,1) with normal, t, GED and skewed-t errors).
Skipped when R or a package is missing."""
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import ardl as ad
from analytics import garch as gm
from analytics import var as vr
from data.samples import financial_returns_demo, macro_var_demo, price_transmission_demo

warnings.filterwarnings("ignore")
PKGS = ["vars", "urca", "cointReg", "ARDL", "quantreg", "fGarch"]
R_OK = bool(shutil.which("Rscript")) and subprocess.run(
    ["Rscript", "-e", f"for (p in c({', '.join(repr(p) for p in PKGS)})) library(p, character.only = TRUE)"],
    capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not R_OK, reason="R packages for the v1.7 cross-check are not installed")

M = macro_var_demo()
Y = pd.DataFrame({"dy": 100 * np.diff(np.log(M.gdp_bn)), "infl": M.inflation_pct[1:].values, "rate": M.policy_rate_pct[1:].values})
L = pd.DataFrame({"y": np.log(M.gdp_bn), "c": np.log(M.consumption_bn), "i": np.log(M.investment_bn)})
PT = price_transmission_demo()
P = pd.DataFrame({"d": np.log(PT.domestic_wheat_egp_t), "x": np.log(PT.import_parity_egp_t)})
D = pd.DataFrame({"d": np.log(PT.domestic_wheat_egp_t), "x": np.log(PT.import_parity_egp_t),
                  "e": np.log(PT.exchange_rate_egp_usd), "rain": PT.rainfall_anomaly})
RET = gm.returns(financial_returns_demo().equity_index.values)

R_SCRIPT = r'''
suppressMessages({library(vars); library(urca); library(cointReg); library(ARDL); library(quantreg); library(fGarch)})
Y <- read.csv("y.csv"); L <- read.csv("l.csv"); P <- read.csv("p.csv"); D <- read.csv("d.csv"); r <- read.csv("r.csv")$r
o <- list()
o$sel <- VARselect(Y, lag.max = 6, type = "const")$criteria
v <- VAR(Y, p = 2, type = "const"); o$coef <- sapply(v$varresult, coef); o$se <- sapply(v$varresult, function(e) sqrt(diag(vcov(e))))
o$pt <- serial.test(v, lags.pt = 12)$serial$statistic; o$pta <- serial.test(v, lags.pt = 12, type = "PT.adjusted")$serial$statistic
o$bg <- serial.test(v, lags.bg = 4, type = "BG")$serial$statistic; o$es <- serial.test(v, lags.bg = 4, type = "ES")$serial$statistic
o$jb <- normality.test(v)$jb.mul$JB$statistic; o$arch <- arch.test(v, lags.multi = 3)$arch.mul$statistic
cz <- causality(v, cause = "rate"); o$gr <- c(cz$Granger$statistic, cz$Instant$statistic)
o$irf <- irf(v, n.ahead = 8, boot = FALSE)$irf$rate[, "dy"]; o$cirf <- irf(v, n.ahead = 8, boot = FALSE, cumulative = TRUE)$irf$rate[, "dy"]
o$gir <- irf(v, n.ahead = 8, boot = FALSE, ortho = FALSE)$irf$infl[, "rate"]
o$fevd <- fevd(v, n.ahead = 8)$dy[8, ]
pr <- predict(v, n.ahead = 4); o$fc <- pr$fcst$dy[, 1]; o$fclo <- pr$fcst$dy[, 2]
Am <- diag(3); Am[lower.tri(Am)] <- NA; sv <- SVAR(v, Amat = Am, estmethod = "scoring"); o$svA <- sv$A; o$svAse <- sv$Ase
Am2 <- Am; Am2[3, 1] <- 0; sv2 <- SVAR(v, Amat = Am2, estmethod = "scoring"); o$sv2LR <- sv2$LR$statistic
Bm <- diag(3); Bm[lower.tri(Bm, diag = TRUE)] <- NA; o$svB <- SVAR(v, Bmat = Bm, estmethod = "scoring")$B
o$bq <- BQ(v)$B
for (ec in c("none", "const", "trend")) { j <- ca.jo(L, type = "trace", ecdet = ec, K = 3, spec = "transitory"); o[[paste0("tr_", ec)]] <- rev(j@teststat) }
o$mx <- rev(ca.jo(L, type = "eigen", ecdet = "const", K = 2)@teststat)
j <- ca.jo(L, type = "trace", ecdet = "none", K = 3, spec = "transitory"); cr <- cajorls(j, r = 2)
o$beta <- cr$beta; o$vcoef <- coef(cr$rlm); o$vse <- sapply(summary(cr$rlm), function(s) s$coefficients[, 2])
j1 <- ca.jo(L, type = "trace", ecdet = "none", K = 3, spec = "longrun")
o$blr <- blrtest(j1, H = matrix(c(1, -1, 0, 1, 0, -1), 3, 2), r = 2)@teststat
o$alr <- alrtest(j1, A = matrix(c(1, 0, 0, 0, 1, 0), 3, 2), r = 2)@teststat
v2 <- vec2var(j, r = 2); o$vA1 <- v2$A$A1; o$virf <- irf(v2, n.ahead = 6, boot = FALSE)$irf$y[, "c"]
o$eg <- ur.df(resid(lm(d ~ x, data = P)), type = "none", lags = 2)@teststat
fm <- cointRegFM(x = as.matrix(P$x), y = P$d, deter = matrix(1, nrow(P), 1), kernel = "ba", bandwidth = "and")
o$fm <- c(fm$theta, fm$sd.theta, fm$bandwidth$number)
dl <- cointRegD(x = as.matrix(P$x), y = P$d, deter = matrix(1, nrow(P), 1), n.lag = 2, n.lead = 2, kernel = "ba", bandwidth = "and")
o$dols <- c(dl$theta, dl$sd.theta)
m <- ardl(d ~ x + e | rain, data = D, order = c(2, 2, 1)); o$acoef <- coef(m); o$ase <- sqrt(diag(vcov(m))); o$aaic <- AIC(m)
o$ucoef <- coef(uecm(m)); mu <- multipliers(m); o$lr <- mu$Estimate; o$lrse <- mu$`Std. Error`
m0 <- ardl(d ~ x + e - 1 | rain, data = D, order = c(2, 2, 1)); mt <- ardl(d ~ x + e | rain + trend(d), data = D, order = c(2, 2, 1))
mods <- list(m0, m, m, mt, mt)
for (cs in 1:5) { b <- bounds_f_test(mods[[cs]], case = cs); o[[paste0("F", cs)]] <- c(b$statistic, b$p.value) }
for (cs in c(1, 3, 5)) { b <- bounds_t_test(mods[[cs]], case = cs); o[[paste0("t", cs)]] <- c(b$statistic, b$p.value) }
rc <- recm(m, case = 2); o$recm <- coef(rc); o$recmse <- sqrt(diag(vcov(rc)))
q <- rq(d ~ x + e, data = D, tau = 0.25); o$rq <- coef(q); o$rqse <- summary(q, se = "nid")$coefficients[, 2]
for (dd in c("norm", "std", "ged", "sstd")) { g <- garchFit(~garch(1, 1), data = r, cond.dist = dd, trace = FALSE)
  o[[paste0("g_", dd)]] <- c(coef(g), -g@fit$llh); o[[paste0("gse_", dd)]] <- g@fit$matcoef[, 2] }
for (n in names(o)) { cat("##", n, "\n"); write.table(as.matrix(o[[n]]), col.names = FALSE, row.names = FALSE) }
'''


@pytest.fixture(scope="module")
def R(tmp_path_factory):
    d = tmp_path_factory.mktemp("v17")
    Y.to_csv(d / "y.csv", index=False); L.to_csv(d / "l.csv", index=False); P.to_csv(d / "p.csv", index=False)
    D.to_csv(d / "d.csv", index=False); pd.DataFrame({"r": RET}).to_csv(d / "r.csv", index=False)
    (d / "m.R").write_text(R_SCRIPT)
    p = subprocess.run(["Rscript", "m.R"], cwd=d, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-2000:]
    blocks, cur = {}, None
    for line in p.stdout.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip(); blocks[cur] = []
        elif cur:
            blocks[cur].append([float(x) for x in line.split()])
    return {k: np.array(v) for k, v in blocks.items()}


def close(a, b, tol):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    assert a.shape == b.shape
    assert np.max(np.abs(a - b)) < tol, (a, b)


@pytest.fixture(scope="module")
def var2():
    return vr.fit_var(Y, 2, "const")


def test_var_selection_and_estimates(R, var2):
    close(vr.var_select(Y, 6, "const")[["AIC", "HQ", "SC", "FPE"]].iloc[1:].T.values, R["sel"], 1e-7)
    close(var2.B, R["coef"], 1e-7)
    close(var2.se, R["se"], 1e-7)


def test_var_diagnostics(R, var2):
    pt, bg = vr.serial_pt(var2, 12), vr.serial_bg(var2, 4)
    close([pt["Q"], pt["Q_adj"], bg["LM"], bg["F_ES"]], np.r_[R["pt"].ravel(), R["pta"].ravel(), R["bg"].ravel(), R["es"].ravel()], 1e-6)
    close([vr.normality(var2)["JB"], vr.arch_multi(var2, 3)["stat"]], np.r_[R["jb"].ravel(), R["arch"].ravel()], 1e-6)
    g = vr.granger(var2, ["rate"])
    close([g["F"], g["inst"]], R["gr"], 1e-7)


def test_irf_fevd_forecast(R, var2):
    close(vr.irf(var2, 8)[:, 0, 2], R["irf"], 1e-7)
    close(vr.irf(var2, 8, cumulative=True)[:, 0, 2], R["cirf"], 1e-7)
    close(vr.irf(var2, 8, "simple")[:, 2, 1], R["gir"], 1e-7)
    close(vr.fevd(var2, 8)[7, 0], R["fevd"], 1e-8)
    f = vr.forecast(var2, 4)
    close(f["mean"][:, 0], R["fc"], 1e-7)
    close(f["lower"][:, 0], R["fclo"], 1e-7)


def test_svar(R, var2):
    Am = np.eye(3); Am[np.tril_indices(3, -1)] = np.nan
    s = vr.svar_fit(var2, Amat=Am)
    close(s["A"], R["svA"], 1e-6); close(s["Ase"], R["svAse"], 1e-6)
    Am2 = Am.copy(); Am2[2, 0] = 0
    close([vr.svar_fit(var2, Amat=Am2)["LR"]], R["sv2LR"], 1e-6)
    Bm = np.eye(3); Bm[np.tril_indices(3)] = np.nan
    close(vr.svar_fit(var2, Bmat=Bm)["B"], R["svB"], 1e-6)
    close(vr.blanchard_quah(var2)["B"], R["bq"], 1e-7)


def test_johansen_and_vecm(R):
    for ec, case in (("none", 3), ("const", 2), ("trend", 4)):
        close(vr.johansen(L, 3, case)["table"]["Trace"], R[f"tr_{ec}"], 1e-6)
    close(vr.johansen(L, 2, 2)["table"]["Max-eigen"], R["mx"], 1e-6)
    jo = vr.johansen(L, 3, 3)
    vm = vr.vecm(jo, 2)
    close(vm["beta"].values, R["beta"], 1e-7)
    close(vm["Gamma"], R["vcoef"], 1e-7)
    close(vm["table"]["Std. Err."].values.reshape(3, -1).T, R["vse"], 1e-7)
    close([vr.restriction_test(jo, 2, H=np.array([[1, 1], [-1, 0], [0, -1]]))["LR"]], R["blr"], 1e-6)
    close([vr.restriction_test(jo, 2, A=np.array([[1, 0], [0, 1], [0, 0]]))["LR"]], R["alr"], 1e-6)
    lv = vr.vecm_levels(vm)
    close(lv.A(1), R["vA1"], 1e-7)
    close(vr.irf(lv, 6)[:, 1, 0], R["virf"], 1e-7)


def test_single_equation_cointegration(R):
    close([vr.engle_granger(P.d, P.x, "c", maxlag=2, autolag=None)["stat"]], R["eg"], 1e-7)
    fm = vr.fmols(P.d, P.x, "c")
    close(np.r_[fm["table"]["Coef."], fm["table"]["Std. Err."], fm["info"]["bandwidth"]], R["fm"], 1e-7)
    dl = vr.dols(P.d, P.x, "c", 2, 2)
    close(np.r_[dl["table"]["Coef."], dl["table"]["Std. Err."]], R["dols"], 1e-7)


def test_ardl(R):
    r = ad.fit_ardl(D, "d", ["x", "e"], (2, 2, 1), 3, ["rain"])
    close(r.b, R["acoef"], 1e-7); close(np.sqrt(np.diag(r.V)), R["ase"], 1e-7); close([r.stats["AIC"]], R["aaic"], 1e-6)
    close(r.uecm["b"], R["ucoef"], 1e-7)
    lr = ad.long_run(r).iloc[:3]
    close(lr["Long-run coef."], R["lr"], 1e-7); close(lr["Std. Err."], R["lrse"], 1e-7)
    for cs in range(1, 6):
        b = ad.bounds_test(ad.fit_ardl(D, "d", ["x", "e"], (2, 2, 1), cs, ["rain"]))
        ref = R[f"F{cs}"].ravel()
        assert b["F"] == pytest.approx(ref[0], abs=1e-6)
        assert b["pF_I1"] == pytest.approx(max(ref[1], 0.001), abs=0.004)
        if cs in (1, 3, 5):
            ref = R[f"t{cs}"].ravel()
            assert b["t"] == pytest.approx(ref[0], abs=1e-6)
            assert b["pt_I1"] == pytest.approx(max(ref[1], 0.001), abs=0.004)
    rc = ad.recm(ad.fit_ardl(D, "d", ["x", "e"], (2, 2, 1), 2, ["rain"]))
    close(rc["table"]["Coef."], R["recm"], 1e-7); close(rc["table"]["Std. Err."], R["recmse"], 1e-7)


def test_quantile_regression(R):
    b, V = ad.rq_nid(D.d.values, np.column_stack([np.ones(len(D)), D.x, D.e]), 0.25)
    close(b, R["rq"], 1e-7)
    close(np.sqrt(np.diag(V)), R["rqse"], 1e-5)


@pytest.mark.parametrize("dist", ["norm", "std", "ged", "sstd"])
def test_garch_vs_fgarch(R, dist):
    g = gm.fit_garch(RET, "garch", 1, 1, dist)
    ref = R[f"g_{dist}"].ravel()
    close(g.params, ref[:-1], 2e-4)
    assert g.llf == pytest.approx(ref[-1], abs=1e-3)
    assert np.allclose(g.se, R[f"gse_{dist}"].ravel(), rtol=0.04)


@pytest.mark.parametrize("kind", ["var", "svar", "coint", "vecm", "ardl", "nardl", "qardl", "garch"])
def test_exported_r_code_runs(kind, tmp_path):
    from analytics import mts_report as mr
    df = pd.DataFrame({"y": L.y, "c": L.c, "i": L.i})
    kw = {"vars": ["y", "c", "i"], "p": 2, "case": 3, "K": 2, "r": 2, "y": "y", "x": ["c", "i"], "order": (2, 1, 1), "asym": "c"}
    if kind == "garch":
        df = pd.DataFrame({"price": financial_returns_demo().equity_index[:800]})
        kw = {"y": "price", "model": "garch", "dist": "std"}
    df.to_csv(tmp_path / "d.csv", index=False)
    code = mr.code(kind, "d.csv", **kw)["R"]
    code = "\n".join(ln for ln in code.splitlines() if "wald.test" not in ln and not ln.startswith("plot(") and "plot(irf" not in ln)
    code = code.replace("; plot(fit, which = 4)", "")
    (tmp_path / "m.R").write_text("pdf(NULL)\n" + code)
    p = subprocess.run(["Rscript", "m.R"], cwd=tmp_path, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
