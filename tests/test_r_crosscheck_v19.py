"""v1.9 cross-validation against R: forecast (meanf, naive, snaive, rwf with drift, thetaf, ets likelihood / information
criteria / forecasts / intervals at given parameters and estimation quality, accuracy, dm.test, tsCV, Arima with xreg,
drift and mean) and stats::nls (SSlogis, SSgompertz). Skipped when R or the forecast package is missing."""
import shutil
import subprocess
import warnings

import numpy as np
import pytest

from analytics import fcst as F
from analytics import fcst_report as fr
from data.samples import monthly_forecast_demo

warnings.filterwarnings("ignore")
R_OK = bool(shutil.which("Rscript")) and subprocess.run(["Rscript", "-e", "library(forecast)"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not R_OK, reason="R package forecast is not installed")

D = monthly_forecast_demo()
Y, FP, TA = D.electricity_gwh.values, D.food_price_index.values, D.tourist_arrivals_k.values
SPECS = {"MAM": ("mul", "add", "mul"), "AAN": ("add", "add", None), "MNM": ("mul", None, "mul"), "ANN": ("add", None, None), "AAA": ("add", "add", "add")}

R_SCRIPT = r'''
suppressMessages(library(forecast)); d <- read.csv("m.csv"); o <- list()
y <- ts(d$electricity_gwh, frequency = 12, start = c(2008, 1)); f <- ts(d$food_price_index, frequency = 12, start = c(2008, 1))
f1 <- ts(d$food_price_index)
pk <- function(fc) c(fc$mean, fc$lower[, 1], fc$upper[, 1], fc$lower[, 2], fc$upper[, 2])
o$meanf <- pk(meanf(f, 10)); o$naive <- pk(naive(f, 10)); o$snaive <- pk(snaive(y, 30)); o$drift <- pk(rwf(f, 10, drift = TRUE))
o$theta_s <- pk(thetaf(y, 18)); o$theta_s_par <- ses(seasadj(decompose(y, type = "multiplicative")), 18)$model$par
o$theta_n <- pk(thetaf(f1, 10)); o$theta_n_par <- ses(f1, 10)$model$par
for (mm in c("MAM", "AAN", "MNM", "ANN", "AAA")) for (dd in c(FALSE, TRUE)) {
  if (dd && substr(mm, 2, 2) == "N") next
  e <- ets(if (substr(mm, 3, 3) == "N") f else y, model = mm, damped = dd)
  k <- paste0(mm, dd); o[[paste0("par_", k)]] <- c(e$par[names(e$par) %in% c("alpha", "beta", "gamma", "phi")], e$initstate)
  o[[paste0("ic_", k)]] <- c(e$loglik, e$aic, e$aicc, e$bic); o[[paste0("fc_", k)]] <- pk(forecast(e, 12)) }
o$ets_auto <- ets(y)$aicc
tr <- window(f, end = c(2023, 12)); te <- window(f, start = c(2024, 1)); fc <- rwf(tr, 24, drift = TRUE)
o$acc <- accuracy(fc, te)[2, c("ME", "RMSE", "MAE", "MPE", "MAPE", "MASE", "Theil's U")]
e1 <- te - fc$mean; e2 <- te - naive(tr, 24)$mean
o$dm <- c(dm.test(e1, e2, h = 1)$statistic, dm.test(e1, e2, h = 1)$p.value, dm.test(e1, e2, h = 4)$statistic, dm.test(e1, e2, h = 4)$p.value,
          dm.test(e1, e2, h = 4, power = 1)$statistic, dm.test(e1, e2, h = 3, varestimator = "bartlett")$statistic)
cv <- tsCV(f, rwf, drift = TRUE, h = 3, initial = 150); o$tscv <- as.vector(cv[150:215, ])
tt <- seq_along(d$tourist_arrivals_k); ta <- d$tourist_arrivals_k
m1 <- nls(ta ~ SSlogis(tt, Asym, xmid, scal)); o$logis <- c(coef(m1), sqrt(diag(vcov(m1))))
m2 <- nls(ta ~ SSgompertz(tt, Asym, b2, b3)); o$gomp <- c(coef(m2), sqrt(diag(vcov(m2))))
a <- Arima(y, order = c(1, 0, 1), seasonal = c(0, 1, 1), xreg = cbind(temp = d$temperature_c), method = "ML")
fq <- forecast(a, xreg = cbind(temp = rep(25, 6)), h = 6)
o$arimax <- c(coef(a), sqrt(diag(a$var.coef)), a$loglik, fq$mean, fq$lower[, 2], fq$upper[, 2])
a2 <- Arima(f, order = c(1, 1, 1), include.drift = TRUE, method = "ML"); a3 <- Arima(f, order = c(2, 0, 0), method = "ML")
o$ll23 <- c(a2$loglik, a3$loglik)
for (n in names(o)) { cat("##", n, "\n"); cat(format(as.vector(o[[n]]), digits = 15), "\n") }
'''


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v19")
    D.to_csv(tmp / "m.csv", index=False)
    (tmp / "s.R").write_text(R_SCRIPT)
    out = subprocess.run(["Rscript", "s.R"], capture_output=True, text=True, cwd=tmp)
    assert out.returncode == 0, out.stderr[-2000:]
    B, cur = {}, None
    for line in out.stdout.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip(); B[cur] = []
        elif cur and line.strip():
            B[cur] += [np.nan if v == "NA" else float(v) for v in line.split()]
    return {k: np.array(v) for k, v in B.items()}


def _pk(fc):
    return np.r_[fc.mean, fc.pi[80][0], fc.pi[80][1], fc.pi[95][0], fc.pi[95][1]]


def test_benchmarks_and_theta(ref):
    np.testing.assert_allclose(_pk(F.mean_fc(FP, 10)), ref["meanf"], rtol=1e-10)
    np.testing.assert_allclose(_pk(F.naive(FP, 10)), ref["naive"], rtol=1e-10)
    np.testing.assert_allclose(_pk(F.snaive(Y, 30, 12)), ref["snaive"], rtol=1e-10)
    np.testing.assert_allclose(_pk(F.drift(FP, 10)), ref["drift"], rtol=1e-10)
    np.testing.assert_allclose(_pk(F.theta(Y, 18, 12, params=ref["theta_s_par"])), ref["theta_s"], rtol=1e-9)
    np.testing.assert_allclose(_pk(F.theta(FP, 10, 1, params=ref["theta_n_par"])), ref["theta_n"], rtol=1e-9)
    th = F.theta(Y, 18, 12)
    np.testing.assert_allclose(th.mean, ref["theta_s"][:18], rtol=1e-4)


@pytest.mark.parametrize("mm", list(SPECS))
def test_ets_matches_r(ref, mm):
    e, t, sn = SPECS[mm]
    for dd in ([False, True] if t else [False]):
        k = f"{mm}{'TRUE' if dd else 'FALSE'}"
        data, s = (FP, 1) if sn is None else (Y, 12)
        par = ref[f"par_{k}"]
        nsm = 1 + (t is not None) + (sn is not None) + dd
        p = np.r_[par[:nsm], par[nsm:-1] if sn else par[nsm:]]
        r = F.ets_fit(data, e, t, dd, sn, s, params=p)
        np.testing.assert_allclose([r.llf, r.aic, r.aicc, r.bic], ref[f"ic_{k}"], rtol=1e-6, err_msg=k)
        fc = F.ets(data, 12, s, {"error": e, "trend": t, "damped": dd, "seasonal": sn, "params": p})
        np.testing.assert_allclose(fc.mean, ref[f"fc_{k}"][:12], rtol=1e-3, err_msg=k)
        tol = 1e-8 if e == "add" else 2e-2          # multiplicative-error intervals: simulation vs R's approximations
        np.testing.assert_allclose(np.r_[fc.pi[95][0], fc.pi[95][1]], ref[f"fc_{k}"][36:60], rtol=tol, err_msg=k)
        est = F.ets_fit(data, e, t, dd, sn, s)
        assert est.llf >= ref[f"ic_{k}"][0] - 0.5, (k, est.llf, ref[f"ic_{k}"][0])


def test_ets_auto_not_worse_than_r(ref):
    tab = F.ets_auto(Y, 12)
    assert tab.Model.iloc[0] == "ETS(M,A,M)" or tab.AICc.iloc[0] <= ref["ets_auto"][0] + 1e-6
    assert tab.AICc.iloc[0] <= ref["ets_auto"][0] + 0.5


def test_accuracy_dm_tscv(ref):
    tr, te = FP[:192], FP[192:]
    fc = F.drift(tr, 24)
    a = F.accuracy(te, fc.mean, tr, 12)
    np.testing.assert_allclose([a[k] for k in ("ME", "RMSE", "MAE", "MPE", "MAPE", "MASE", "Theil's U")], ref["acc"], rtol=1e-10)
    e1, e2 = te - fc.mean, te - F.naive(tr, 24).mean
    d = [F.dm_test(e1, e2, 1)["DM"], F.dm_test(e1, e2, 1)["p"], F.dm_test(e1, e2, 4)["DM"], F.dm_test(e1, e2, 4)["p"],
         F.dm_test(e1, e2, 4, power=1)["DM"], F.dm_test(e1, e2, 3, varestimator="bartlett")["DM"]]
    np.testing.assert_allclose(d, ref["dm"], rtol=1e-10)
    E = F.rolling_origin(FP, lambda yy, h: F.drift(yy, h), 3, 150)
    got, want = np.r_[[E[:, j] for j in range(3)]].ravel(), ref["tscv"]
    ok = np.isfinite(want)                       # tsCV leaves the first origin (t = initial) empty
    assert ok.sum() > 150
    np.testing.assert_allclose(got[ok], want[ok], rtol=1e-10)


def test_growth_curves_match_nls(ref):
    for kind, key in (("logistic", "logis"), ("gompertz", "gomp")):
        g = F.growth_curve(TA, 6, kind)
        np.testing.assert_allclose(np.r_[g.info["params"], g.info["table"]["Std. Err."]], ref[key], rtol=1e-3)


def test_arima_matches_r(ref):
    a = F.sarimax(Y, 6, (1, 0, 1), (0, 1, 1), 12, D[["temperature_c"]].values, np.full((6, 1), 25.0))
    tb = a.info["table"]
    got = np.r_[tb["Coef."].values[[1, 2, 3, 0]], tb["Std. Err."].values[[1, 2, 3, 0]], a.info["loglik"], a.mean, a.pi[95][0], a.pi[95][1]]
    np.testing.assert_allclose(got, ref["arimax"], rtol=2e-3)
    # near-unit-root models: the likelihood is flat; ours must be at least as high as R's
    a2 = F.sarimax(FP, 5, (1, 1, 1), trend="t")
    a3 = F.sarimax(FP, 5, (2, 0, 0))
    assert a2.info["loglik"] >= ref["ll23"][0] - 1e-3 and a3.info["loglik"] >= ref["ll23"][1] - 1e-3


def test_exported_r_code_runs(tmp_path):
    D.to_csv(tmp_path / "data.csv", index=False)
    code = fr.code("classical", file_name="data.csv", time="date", y="electricity_gwh", h=12, s=12)["R"]
    code = code.replace("length(y) - 36", "length(y) - 6")
    (tmp_path / "c.R").write_text(code)
    out = subprocess.run(["Rscript", "c.R"], capture_output=True, text=True, cwd=tmp_path, timeout=900)
    assert out.returncode == 0, out.stderr[-1500:]
