"""v1.6 — univariate time series (Catalogue §1.7): unit-root tests, decomposition and filters, Box-Jenkins
ARIMA, forecasting / hold-out evaluation, long memory, reporting and exported Python code."""
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import ts
from analytics import ts_report as tsr
from data.samples import macro_ts_demo, quarterly_demo

warnings.filterwarnings("ignore")
M = macro_ts_demo()
Q = quarterly_demo()
RNG = np.random.default_rng(3)
_e = RNG.normal(size=300)
AR1 = np.zeros(300)
for t in range(1, 300):
    AR1[t] = 0.5 * AR1[t - 1] + _e[t]
RW = np.cumsum(RNG.normal(size=300))


def test_prepare_and_period():
    s = ts.prepare(M, "date", "cpi_index", transform="log")
    assert s.s == 12 and s.name == "ln(cpi_index)" and len(s.y) == len(M)
    assert ts.prepare(Q, "quarter", "ag_output_index").s == 4
    with pytest.raises(ValueError):
        ts.prepare(M.assign(cpi_index=M.cpi_index - 1000), "date", "cpi_index", transform="log")


def test_prepare_panel_duplicates_averaged():
    d = pd.concat([M.assign(region="A"), M.assign(region="B", cpi_index=M.cpi_index * 2)])
    s = ts.prepare(d, "date", "cpi_index")
    assert s.notes and len(s.y) == len(M)
    assert len(ts.prepare(d, "date", "cpi_index", {"region": "A"}).y) == len(M)


def test_decompose_and_filters():
    s = ts.prepare(M, "date", "rainfall_mm")
    dec = ts.decompose(s.y, 12, "stl")
    assert np.allclose(dec["trend"] + dec["seasonal"] + dec["remainder"], dec["observed"])
    assert dec.attrs["strength_seasonal"] > 0.5
    cl = ts.decompose(s.y, 12, "classical")
    assert {"trend", "seasonal"} <= set(cl.columns)
    f = ts.filters(s.y, 12)
    assert np.allclose(f["HP trend"] + f["HP cycle"], f["observed"])


def test_unit_root_power_and_size():
    assert ts.adf(RW, "c")["p"] > 0.05 and ts.adf(AR1, "c")["p"] < 0.01
    assert ts.phillips_perron(AR1, "c")["p"] < 0.01
    assert ts.kpss(AR1, "c")["p"] > 0.05 and ts.kpss(RW, "c")["p"] < 0.05
    assert ts.dfgls(AR1, "c")["stat"] < -3


def test_integration_order():
    assert ts.integration_order(RW, "c")["order"] == 1
    assert ts.integration_order(AR1, "c")["order"] == 0
    assert ts.integration_order(np.cumsum(RW), "c")["order"] == 2


def test_ng_perron_and_za():
    npr = ts.ng_perron(AR1, "c")
    assert set(npr["values"]) == {"MZa", "MZt", "MSB", "MPT"} and npr["reject_5"]["MZt"]
    y = np.r_[RNG.normal(size=150), 5 + RNG.normal(size=150)]
    z = ts.zivot_andrews(y, "c")
    assert abs(z["break_pos"] - 151) <= 3 and z["p"] < 0.05


def test_hegy_seasonal_roots():
    h = ts.hegy(Q.ag_output_index.values, "cs", 0, reps=500)
    assert len(h) == 3
    assert (h["Unit root at this frequency"] == "not rejected").any()


def test_arima_recovers_parameters():
    e = RNG.normal(size=600)
    y = np.zeros(600)
    for t in range(1, 600):
        y[t] = 0.6 * y[t - 1] + e[t] + 0.3 * e[t - 1]
    r = ts.fit_arima(pd.Series(y), (1, 0, 1))
    assert r.res.params["ar.L1"] == pytest.approx(0.6, abs=0.08)
    assert r.res.params["ma.L1"] == pytest.approx(0.3, abs=0.1)
    d = ts.residual_tests(r)
    assert d["lb_p"] > 0.05 and d["lb_df"] == d["lb_lags"] - 2
    ar, ma = ts.roots(r.res)
    assert np.all(np.abs(ar) > 1) and np.all(np.abs(ma) > 1)


def test_auto_arima_and_holdout():
    y = pd.Series(AR1)
    a = ts.auto_arima(y, 1, 0, 0, 2, 2, 0, 0)
    assert a["AICc"].is_monotonic_increasing and a.iloc[0]["Ljung-Box p"] > 0.05
    assert a.iloc[0]["p"] + a.iloc[0]["q"] >= 1
    ho = ts.holdout(pd.Series(np.log(M.cpi_index.values)), (0, 1, 1), (0, 1, 1), 12, 24)
    assert {"RMSE", "MAE"} <= set(ho["table"].columns) and len(ho["forecast"]) == 24
    assert np.isfinite(ho["dm"]["stat"])


def test_forecast_log_back_transform():
    s = ts.prepare(M, "date", "cpi_index", transform="log")
    r = ts.fit_arima(s.y, (0, 1, 1), (0, 1, 1), 12, name=s.name, transform="log")
    fc = ts.forecast(r, 6)
    assert np.allclose(fc["forecast (level, median)"], np.exp(fc["forecast"]))
    assert (fc["forecast (level, mean)"] > fc["forecast (level, median)"]).all()
    assert pd.api.types.is_datetime64_any_dtype(fc["time"]) and fc["time"].iloc[0] > s.y.index[-1]


def test_diebold_mariano_symmetry():
    e1, e2 = RNG.normal(size=50), 2 * RNG.normal(size=50)
    a, b = ts.diebold_mariano(e1, e2), ts.diebold_mariano(e2, e1)
    assert a["stat"] == pytest.approx(-b["stat"]) and a["stat"] < 0


def test_long_memory():
    x = ts.frac_diff(RNG.normal(size=2000), -0.3)          # fractionally integrated noise, d = 0.3
    assert ts.gph(x)["d"] == pytest.approx(0.3, abs=0.15)
    assert ts.local_whittle(x)["d"] == pytest.approx(0.3, abs=0.1)
    assert ts.arfima_whittle(x)["d"] == pytest.approx(0.3, abs=0.08)
    assert abs(ts.arfima_whittle(AR1 - AR1.mean(), 1, 0)["d"]) < 0.2


@pytest.mark.parametrize("lang", ["en", "ar"])
def test_reporting(lang):
    io = ts.integration_order(np.log(M.cpi_index.values), "ct")
    txt = tsr.interpret_unit_roots(io, "ln(cpi)", lang)
    assert len(txt) > 50
    r = ts.fit_arima(pd.Series(M.wheat_price_egp_t.values), (1, 1, 1))
    assert "ARIMA(1,1,1)" in tsr.interpret_arima(r, ts.residual_tests(r), lang)
    for kind in ("unitroot", "arima", "filters", "seasonal", "longmemory"):
        m, refs = tsr.methodology(kind, lang, name="ARIMA(1,1,1)")
        assert len(m) > 100 and refs


@pytest.mark.parametrize("kind", ["unitroot", "arima", "filters"])
def test_exported_python_runs(kind, tmp_path):
    M.to_csv(tmp_path / "d.csv", index=False)
    code = tsr.code(kind, "cpi_index", "date", "d.csv", s=12, log=True, order=(0, 1, 1), seasonal=(0, 1, 1), regression="ct")
    (tmp_path / "m.py").write_text(code["Python"])
    p = subprocess.run([sys.executable, "m.py"], cwd=tmp_path, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-1500:]
    assert all(k in code for k in ("Python", "R", "Stata"))
