"""v1.6 cross-validation against R (urca, uroot, fracdiff, stats::arima, forecast). Skipped when R or a package is missing.
Tolerances: exact-formula tests 1e-6; PP with trend 2e-3 (urca uses the asymptotic form of the moment matrix);
ARIMA ML 5e-4 relative (different optimisers — Aetheris' log-likelihood is ≥ R's); fracdiff d vs Whittle d 0.02
(different estimators of the same parameter)."""
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import ts
from analytics import ts_report as tsr
from data.samples import macro_ts_demo, quarterly_demo

warnings.filterwarnings("ignore")
PKGS = ["urca", "uroot", "fracdiff", "forecast"]
R_OK = bool(shutil.which("Rscript")) and subprocess.run(
    ["Rscript", "-e", f"for (p in c({', '.join(repr(p) for p in PKGS)})) library(p, character.only = TRUE)"],
    capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not R_OK, reason="R packages for the v1.6 cross-check are not installed")

R_SCRIPT = r'''
suppressMessages({library(urca); library(uroot); library(fracdiff)})
m <- read.csv("ts.csv"); q <- read.csv("q.csv")
y <- log(m$cpi_index); w <- m$wheat_price_egp_t; f <- m$nile_flow_index; qa <- ts(q$ag_output_index, frequency = 4, start = c(1990, 1))
out <- c()
a <- ur.df(y, type = "trend", lags = 4); out["adf_ct4"] <- a@teststat[1]
a <- ur.df(diff(y), type = "drift", lags = 2); out["adf_c2"] <- a@teststat[1]
p <- ur.pp(y, type = "Z-tau", model = "trend", lags = "short"); out["pp_ct"] <- p@teststat
p <- ur.pp(y, type = "Z-alpha", model = "constant", lags = "short"); out["ppa_c"] <- p@teststat
p <- ur.pp(y, type = "Z-tau", model = "constant", lags = "short"); out["pp_c"] <- p@teststat
k <- ur.kpss(y, type = "tau", lags = "short"); out["kpss_ct"] <- k@teststat
k <- ur.kpss(diff(y), type = "mu", lags = "long"); out["kpss_c_long"] <- k@teststat
e <- ur.ers(y, type = "DF-GLS", model = "trend", lag.max = 4); out["dfgls_ct4"] <- e@teststat
e <- ur.ers(diff(y), type = "DF-GLS", model = "constant", lag.max = 3); out["dfgls_c3"] <- e@teststat
z <- ur.za(y, model = "intercept", lag = 4); out["za_c4"] <- z@teststat; out["za_bp"] <- z@bpoint
h <- hegy.test(qa, deterministic = c(1, 0, 1), lag.method = "fixed", maxlag = 0); out["hegy_t1"] <- h$statistics[1]; out["hegy_t2"] <- h$statistics[2]; out["hegy_F34"] <- h$statistics[3]
ar <- arima(w, order = c(1, 1, 1), method = "ML"); out["ar1"] <- coef(ar)[1]; out["ma1"] <- coef(ar)[2]; out["sig2"] <- ar$sigma2; out["ll"] <- ar$loglik
out["se_ar1"] <- sqrt(diag(ar$var.coef))[1]
fc <- predict(ar, n.ahead = 3); out["fc1"] <- fc$pred[1]; out["fcse3"] <- fc$se[3]
ar2 <- arima(y, order = c(0, 1, 1), seasonal = list(order = c(0, 1, 1), period = 12), method = "ML"); out["sma"] <- coef(ar2)[2]; out["ll_s"] <- ar2$loglik; out["ma_s"] <- coef(ar2)[1]
ar3 <- arima(m$unemployment_rate, order = c(2, 0, 0), method = "ML"); out["u_ar1"] <- coef(ar3)[1]; out["u_mean"] <- coef(ar3)[3]
g <- fdGPH(f, bandw.exp = 0.5); out["gph"] <- g$d
fd <- fracdiff(f - mean(f), nar = 0, nma = 0); out["fracdiff_d"] <- fd$d
write.csv(data.frame(k = names(out), v = unname(out)), "out.csv", row.names = FALSE)
'''


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    d = tmp_path_factory.mktemp("tsr")
    macro_ts_demo().to_csv(d / "ts.csv", index=False)
    quarterly_demo().to_csv(d / "q.csv", index=False)
    (d / "m.R").write_text(R_SCRIPT)
    p = subprocess.run(["Rscript", "m.R"], cwd=d, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    return pd.read_csv(d / "out.csv").set_index("k").v


M = macro_ts_demo()
Q = quarterly_demo()
Y = np.log(M.cpi_index.values)
W = M.wheat_price_egp_t.values
F = M.nile_flow_index.values


def test_unit_root_statistics(ref):
    mine = {"adf_ct4": ts.adf(Y, "ct", None, 4)["stat"], "adf_c2": ts.adf(np.diff(Y), "c", None, 2)["stat"],
            "ppa_c": ts.phillips_perron(Y, "c")["Z_alpha"], "pp_c": ts.phillips_perron(Y, "c")["stat"],
            "kpss_ct": ts.kpss(Y, "ct")["stat"], "kpss_c_long": ts.kpss(np.diff(Y), "c", kind="long")["stat"],
            "dfgls_ct4": ts.dfgls(Y, "ct", lags=4)["stat"], "dfgls_c3": ts.dfgls(np.diff(Y), "c", lags=3)["stat"],
            "za_c4": ts.zivot_andrews(Y, "c", lags=4)["stat"], "za_bp": ts.zivot_andrews(Y, "c", lags=4)["break_pos"]}
    for k, v in mine.items():
        assert v == pytest.approx(ref[k], abs=1e-4), k
    assert ts.phillips_perron(Y, "ct")["stat"] == pytest.approx(ref["pp_ct"], abs=2e-3)


def test_hegy(ref):
    h = ts.hegy(Q.ag_output_index.values, "cs", 0)
    assert h.iloc[0, 1] == pytest.approx(ref["hegy_t1"], abs=1e-6)
    assert h.iloc[1, 1] == pytest.approx(ref["hegy_t2"], abs=1e-6)
    assert h.iloc[2, 1] == pytest.approx(ref["hegy_F34"], abs=1e-6)


def test_arima_ml(ref):
    r = ts.fit_arima(pd.Series(W), (1, 1, 1))
    assert r.res.llf >= ref["ll"] - 1e-6
    for k, v in {"ar1": r.res.params["ar.L1"], "ma1": r.res.params["ma.L1"], "se_ar1": r.res.bse["ar.L1"]}.items():
        assert v == pytest.approx(ref[k], abs=5e-4), k
    assert r.res.params["sigma2"] == pytest.approx(ref["sig2"], rel=5e-4)
    fc = ts.forecast(r, 3)
    assert fc.forecast[0] == pytest.approx(ref["fc1"], rel=1e-5)
    assert fc.se[2] == pytest.approx(ref["fcse3"], rel=5e-4)
    r2 = ts.fit_arima(pd.Series(Y), (0, 1, 1), (0, 1, 1), 12)
    assert r2.res.llf >= ref["ll_s"] - 1e-3
    assert r2.res.params["ma.S.L12"] == pytest.approx(ref["sma"], abs=5e-4)
    r3 = ts.fit_arima(pd.Series(M.unemployment_rate.values), (2, 0, 0))
    assert r3.res.params["ar.L1"] == pytest.approx(ref["u_ar1"], abs=1e-4)
    assert r3.res.params["const"] == pytest.approx(ref["u_mean"], abs=1e-3)


def test_long_memory(ref):
    assert ts.gph(F)["d"] == pytest.approx(ref["gph"], abs=1e-8)
    assert ts.arfima_whittle(F)["d"] == pytest.approx(ref["fracdiff_d"], abs=0.02)


@pytest.mark.parametrize("kind", ["unitroot", "arima", "seasonal", "longmemory"])
def test_exported_r_code_runs(kind, tmp_path):
    df = quarterly_demo() if kind == "seasonal" else macro_ts_demo()
    v = "ag_output_index" if kind == "seasonal" else "cpi_index"
    t = "quarter" if kind == "seasonal" else "date"
    df.to_csv(tmp_path / "d.csv", index=False)
    code = tsr.code(kind, v, t, "d.csv", s=4 if kind == "seasonal" else 12, log=True, order=(0, 1, 1), seasonal=(0, 1, 1),
                    regression="ct")["R"].replace("checkresiduals(m)", "invisible(Box.test(residuals(m), 24, 'Ljung'))")
    code = code.replace("autoplot(forecast(m, h = 12))", "print(forecast(m, h = 12))").replace(
        "auto.arima(y, stepwise = FALSE, approximation = FALSE)", "auto.arima(y)")
    (tmp_path / "m.R").write_text(code)
    p = subprocess.run(["Rscript", "m.R"], cwd=tmp_path, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    if kind == "arima":
        r = ts.fit_arima(pd.Series(Y), (0, 1, 1), (0, 1, 1), 12)
        assert f"{r.res.params['ma.S.L12']:.3f}"[:5] in p.stdout or f"{r.res.params['ma.S.L12']:.4f}"[:6] in p.stdout
