# ==========================================
# Aetheris Research - Univariate time series (Catalogue §1.7)
# ------------------------------------------
# Preparation & decomposition (classical, STL), filters (Hodrick-Prescott, Baxter-King,
# Christiano-Fitzgerald); unit-root & stationarity tests (ADF, Phillips-Perron, KPSS, DF-GLS,
# Ng-Perron MZα / MZt / MSB / MPT, Zivot-Andrews, HEGY seasonal) and an integration-order
# decision; Box-Jenkins ARIMA / SARIMA (identification, automatic order search, exact ML,
# diagnostics, forecasting, hold-out evaluation); long memory (GPH, local Whittle, ARFIMA).
# Pure numpy / scipy / statsmodels — no Streamlit.
# ==========================================

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

warnings.filterwarnings("ignore")

FREQS = {12: ("Monthly", "شهرية"), 4: ("Quarterly", "ربع سنوية"), 1: ("Annual", "سنوية"),
         52: ("Weekly", "أسبوعية"), 7: ("Daily (weekly cycle)", "يومية (دورة أسبوعية)")}


# ==========================================================
# Preparation
# ==========================================================
@dataclass
class Series:
    y: pd.Series                  # values indexed by time (DatetimeIndex or integer years)
    name: str
    s: int                        # seasonal period
    transform: str = "level"      # level | log
    notes: list = field(default_factory=list)


def detect_period(idx) -> int:
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 3:
        f = pd.infer_freq(idx)
        if f:
            f = f.upper()
            if f.startswith(("M", "BM", "MS")):
                return 12
            if f.startswith(("Q", "BQ")):
                return 4
            if f.startswith(("A", "Y", "BA", "BY")):
                return 1
            if f.startswith("W"):
                return 52
            if f.startswith(("D", "B")):
                return 7
        d = np.median(np.diff(idx.values).astype("timedelta64[D]").astype(float))
        return 12 if 27 <= d <= 32 else 4 if 85 <= d <= 95 else 1 if d >= 360 else 52 if 6 <= d <= 8 else 7
    return 1


def prepare(df: pd.DataFrame, time: str, value: str, filters: dict | None = None, transform: str = "level",
            period: int | None = None) -> Series:
    d = df.copy()
    for c, v in (filters or {}).items():
        d = d[d[c].astype(str) == str(v)]
    t = d[time]
    notes = []
    if pd.api.types.is_numeric_dtype(t) and t.dropna().between(1000, 3000).all():
        idx = t.astype(int)
    else:
        idx = pd.to_datetime(t, errors="coerce")
    s = pd.Series(pd.to_numeric(d[value], errors="coerce").to_numpy(), index=idx).dropna()
    s = s[s.index.notna()]
    if s.index.duplicated().any():
        notes.append(f"{int(s.index.duplicated().sum())} duplicated time points were averaged "
                     "(select filters to isolate one series).")
        s = s.groupby(level=0).mean()
    s = s.sort_index()
    if transform == "log":
        if (s <= 0).any():
            raise ValueError("The log transform needs strictly positive values.")
        s = np.log(s)
    per = period or detect_period(s.index if isinstance(s.index, pd.DatetimeIndex) else pd.Index([]))
    if len(s) < 12:
        raise ValueError("At least 12 observations are needed for time-series analysis.")
    return Series(y=s.astype(float), name=(f"ln({value})" if transform == "log" else value), s=per,
                  transform=transform, notes=notes)


# ==========================================================
# Decomposition & filters
# ==========================================================
def decompose(y: pd.Series, s: int, method: str = "stl", model: str = "additive") -> pd.DataFrame:
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    if s < 2:
        raise ValueError("Seasonal decomposition needs a seasonal period ≥ 2 (monthly, quarterly…).")
    if method == "stl":
        r = STL(y.to_numpy(), period=s, robust=True).fit()
        out = pd.DataFrame({"observed": y.values, "trend": r.trend, "seasonal": r.seasonal, "remainder": r.resid},
                           index=y.index)
    else:
        r = seasonal_decompose(y.to_numpy(), model=model, period=s, two_sided=True)
        out = pd.DataFrame({"observed": y.values, "trend": r.trend, "seasonal": r.seasonal, "remainder": r.resid},
                           index=y.index)
    var_r = np.nanvar(out["remainder"])
    out.attrs["strength_trend"] = float(max(0, 1 - var_r / np.nanvar(out["trend"] + out["remainder"])))
    out.attrs["strength_seasonal"] = float(max(0, 1 - var_r / np.nanvar(out["seasonal"] + out["remainder"])))
    return out


HP_LAMBDA = {1: 100.0, 4: 1600.0, 12: 14400.0}
HP_LAMBDA_RU = {1: 6.25, 4: 1600.0, 12: 129600.0}          # Ravn & Uhlig (2002)
BK_PARAMS = {1: (2, 8, 3), 4: (6, 32, 12), 12: (18, 96, 36)}


def filters(y: pd.Series, s: int, hp_lambda: float | None = None) -> pd.DataFrame:
    from statsmodels.tsa.filters.bk_filter import bkfilter
    from statsmodels.tsa.filters.cf_filter import cffilter
    from statsmodels.tsa.filters.hp_filter import hpfilter
    x = y.to_numpy()
    lam = hp_lambda or HP_LAMBDA.get(s, 1600.0)
    hc, ht = hpfilter(x, lamb=lam)
    lo, hi, K = BK_PARAMS.get(s, (6, 32, 12))
    out = pd.DataFrame({"observed": x, "HP trend": ht, "HP cycle": hc}, index=y.index)
    if len(x) > 2 * K + 4:
        bk = np.full(len(x), np.nan)
        bk[K:len(x) - K] = bkfilter(x, lo, hi, K)
        out["BK cycle"] = bk
    cfc, cft = cffilter(x, lo, hi, drift=True)
    out["CF cycle"] = cfc
    out.attrs.update({"hp_lambda": lam, "band": (lo, hi), "K": K})
    return out


# ==========================================================
# Unit-root & stationarity tests
# ==========================================================
REG = {"n": ("none", "بدون ثابت أو اتجاه"), "c": ("constant", "ثابت"), "ct": ("constant + trend", "ثابت واتجاه")}


def _lags_rule(n: int, kind: str = "short") -> int:
    return int(np.trunc((4 if kind == "short" else 12) * (n / 100) ** 0.25))


def adf(y, regression: str = "c", autolag: str | None = "AIC", maxlag: int | None = None) -> dict:
    from statsmodels.tsa.stattools import adfuller
    x = np.asarray(y, float)
    if autolag is None:
        maxlag = _lags_rule(len(x), "long") if maxlag is None else maxlag
    r = adfuller(x, maxlag=maxlag, regression=regression, autolag=autolag, regresults=False)
    stat, p, lags, nobs, crit = r[0], r[1], r[2], r[3], r[4]
    return {"test": "ADF", "stat": float(stat), "p": float(p), "lags": int(lags), "nobs": int(nobs),
            "crit": {k: float(v) for k, v in crit.items()}, "H0": "unit root"}


def _nw_lrv(u, l):
    T = len(u)
    s = u @ u / T
    for j in range(1, l + 1):
        s += 2 * (1 - j / (l + 1)) * (u[j:] @ u[:-j]) / T
    return s


def phillips_perron(y, regression: str = "c", lags: int | None = None, kind: str = "short") -> dict:
    """Phillips-Perron (1988) Z(t) and Z(α) with a Bartlett / Newey-West long-run variance
    (lags = trunc(4 (T/100)^¼) by default, as R urca::ur.pp(lags = "short")); MacKinnon p-values."""
    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
    x = np.asarray(y, float)
    ylag, yt = x[:-1], x[1:]
    T = len(yt)
    cols = [ylag]
    if regression in ("c", "ct"):
        cols.append(np.ones(T))
    if regression == "ct":
        cols.append(np.arange(1, T + 1) - (T + 1) / 2 - 0.5)   # t − n/2 with n = T + 1 (as R urca::ur.pp)
    Z = np.column_stack(cols)
    b, *_ = np.linalg.lstsq(Z, yt, rcond=None)
    u = yt - Z @ b
    k = Z.shape[1]
    s2 = u @ u / (T - k)
    V = s2 * np.linalg.inv(Z.T @ Z)
    se = np.sqrt(V[0, 0])
    rho = b[0]
    t_rho = (rho - 1) / se
    l = _lags_rule(T + 1, kind) if lags is None else lags
    g0 = u @ u / T
    lam2 = _nw_lrv(u, l)
    Zt = np.sqrt(g0 / lam2) * t_rho - 0.5 * (lam2 - g0) / np.sqrt(lam2) * T * se / np.sqrt(s2)
    Za = T * (rho - 1) - 0.5 * T ** 2 * se ** 2 / s2 * (lam2 - g0)
    crit = mackinnoncrit(N=1, regression=regression, nobs=T)
    return {"test": "Phillips-Perron Z(t)", "stat": float(Zt), "p": float(mackinnonp(Zt, regression=regression, N=1)),
            "lags": l, "nobs": T, "crit": {"1%": crit[0], "5%": crit[1], "10%": crit[2]}, "Z_alpha": float(Za),
            "H0": "unit root"}


def kpss(y, regression: str = "c", lags: int | None = None, kind: str = "short") -> dict:
    """Kwiatkowski-Phillips-Schmidt-Shin (1992): H0 = (trend-)stationarity. Bartlett lags trunc(4 (T/100)^¼)
    ('short', R urca default) or trunc(12 (T/100)^¼) ('long')."""
    x = np.asarray(y, float)
    T = len(x)
    Z = np.ones((T, 1)) if regression == "c" else np.c_[np.ones(T), np.arange(1, T + 1)]
    e = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    l = _lags_rule(T, kind) if lags is None else lags
    S = np.cumsum(e)
    stat = float(S @ S / (T ** 2 * _nw_lrv(e, l)))
    cv = {"c": (0.347, 0.463, 0.574, 0.739), "ct": (0.119, 0.146, 0.176, 0.216)}[regression]
    pts = [0.10, 0.05, 0.025, 0.01]
    p = float(np.interp(stat, cv, pts)) if cv[0] <= stat <= cv[-1] else (0.10 if stat < cv[0] else 0.01)
    return {"test": "KPSS", "stat": stat, "p": p, "p_note": "interpolated (bounded to [0.01, 0.10])", "lags": l, "nobs": T,
            "crit": {"10%": cv[0], "5%": cv[1], "2.5%": cv[2], "1%": cv[3]}, "H0": "stationary"}


# ERS (1996) / urca::ur.ers critical values for DF-GLS, rows T = 50, 100, 200, ∞; columns 1%, 5%, 10%
_ERS_CV = {"c": np.array([[-2.62, -1.95, -1.61], [-2.60, -1.95, -1.61], [-2.58, -1.95, -1.62], [-2.58, -1.95, -1.62]]),
           "ct": np.array([[-3.77, -3.19, -2.89], [-3.58, -3.03, -2.74], [-3.46, -2.93, -2.64], [-3.48, -2.89, -2.57]])}
# Ng & Perron (2001), Table 1 asymptotic critical values (1%, 5%, 10%)
_NP_CV = {"c": {"MZa": (-13.8, -8.1, -5.7), "MZt": (-2.58, -1.98, -1.62), "MSB": (0.174, 0.233, 0.275),
                "MPT": (1.78, 3.17, 4.45)},
          "ct": {"MZa": (-23.8, -17.3, -14.2), "MZt": (-3.42, -2.91, -2.62), "MSB": (0.143, 0.168, 0.185),
                 "MPT": (4.03, 5.48, 6.67)}}


def gls_detrend(y, regression: str = "c"):
    x = np.asarray(y, float)
    T = len(x)
    cbar = -7.0 if regression == "c" else -13.5
    a = 1 + cbar / T
    Z = np.ones((T, 1)) if regression == "c" else np.c_[np.ones(T), np.arange(1, T + 1)]
    xq = np.r_[x[0], x[1:] - a * x[:-1]]
    Zq = np.vstack([Z[0], Z[1:] - a * Z[:-1]])
    b = np.linalg.lstsq(Zq, xq, rcond=None)[0]
    return x - Z @ b, cbar


def _dfgls_reg(yd, k):
    """Δyᵈₜ = π yᵈₜ₋₁ + Σⱼ γⱼ Δyᵈₜ₋ⱼ + e (no deterministic terms)."""
    dy = np.diff(yd)
    m = len(dy)
    Y = dy[k:]
    X = np.column_stack([yd[k:m]] + [dy[k - j:m - j] for j in range(1, k + 1)])
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ b
    s2 = e @ e / (len(Y) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    return b, e, s2, se, len(Y)


def dfgls(y, regression: str = "c", lags: int | None = None, maxlag: int | None = None, ic: str = "MAIC") -> dict:
    """Elliott, Rothenberg & Stock (1996) DF-GLS: GLS-detrended ADF regression without deterministics.
    Lag length fixed or chosen by the modified AIC of Ng & Perron (2001) / AIC / BIC."""
    yd, cbar = gls_detrend(y, regression)
    T = len(yd)
    kmax = int(np.floor(12 * (T / 100) ** 0.25)) if maxlag is None else maxlag
    if lags is None:
        best = None
        for k in range(0, kmax + 1):
            dy = np.diff(yd)
            Y = dy[kmax:]
            X = np.column_stack([yd[kmax:T - 1]] + [dy[kmax - j:len(dy) - j] for j in range(1, k + 1)])
            b = np.linalg.lstsq(X, Y, rcond=None)[0]
            e = Y - X @ b
            n = len(Y)
            s2 = e @ e / n
            if ic == "MAIC":
                tau = b[0] ** 2 * np.sum(yd[kmax:T - 1] ** 2) / s2
                crit = np.log(s2) + 2 * (tau + k) / n
            elif ic == "BIC":
                crit = np.log(s2) + k * np.log(n) / n
            else:
                crit = np.log(s2) + 2 * k / n
            if best is None or crit < best[0]:
                best = (crit, k)
        lags = best[1]
    b, e, s2, se, n = _dfgls_reg(yd, lags)
    stat = float(b[0] / se)
    tab = _ERS_CV[regression]
    Ts = np.array([50, 100, 200, 1e9])
    crit = [float(np.interp(T, Ts[:3], tab[:3, j])) if T <= 200 else float(tab[3, j]) for j in range(3)]
    return {"test": "DF-GLS (ERS)", "stat": stat, "p": None, "lags": int(lags), "nobs": int(n),
            "crit": {"1%": crit[0], "5%": crit[1], "10%": crit[2]}, "H0": "unit root",
            "reject_5": stat < crit[1], "yd": yd, "coef": b, "s2": s2}


def ng_perron(y, regression: str = "c", lags: int | None = None) -> dict:
    """Ng & Perron (2001) modified tests MZα, MZt, MSB, MPT on GLS-detrended data; spectral AR estimate
    of the long-run variance from the DF-GLS regression (lags by MAIC)."""
    d = dfgls(y, regression, lags=lags)
    yd = d["yd"]
    T = len(yd)
    k = d["lags"]
    b, s2 = d["coef"], d["s2"]
    s2ar = s2 / (1 - (np.sum(b[1:]) if k else 0.0)) ** 2
    kap = np.sum(yd[:-1] ** 2) / T ** 2
    MZa = (yd[-1] ** 2 / T - s2ar) / (2 * kap)
    MSB = np.sqrt(kap / s2ar)
    MZt = MZa * MSB
    cbar = -7.0 if regression == "c" else -13.5
    MPT = ((cbar ** 2 * kap - cbar * yd[-1] ** 2 / T) / s2ar) if regression == "c" else \
        ((cbar ** 2 * kap + (1 - cbar) * yd[-1] ** 2 / T) / s2ar)
    cv = _NP_CV[regression]
    vals = {"MZa": MZa, "MZt": MZt, "MSB": MSB, "MPT": MPT}
    rej = {m: (vals[m] < cv[m][1]) for m in vals}      # all four reject for small values
    return {"test": "Ng-Perron", "values": {m: float(v) for m, v in vals.items()}, "crit": cv, "lags": k,
            "reject_5": rej, "H0": "unit root"}


def zivot_andrews(y, regression: str = "c", lags: int | None = None, trim: float = 0.15) -> dict:
    from statsmodels.tsa.stattools import zivot_andrews as za
    x = np.asarray(y, float)
    r = za(x, trim=trim, maxlag=lags, regression=regression, autolag=None if lags is not None else "AIC")
    stat, p, crit, bl, bp = r
    return {"test": "Zivot-Andrews", "stat": float(stat), "p": float(p), "lags": int(bl), "break_pos": int(bp) + 1,
            "crit": {k: float(v) for k, v in crit.items()}, "H0": "unit root without break"}


@lru_cache(maxsize=32)
def _hegy_null(T: int, reps: int, det: str) -> np.ndarray:
    rng = np.random.default_rng(1990)
    out = np.empty((reps, 5))
    for r in range(reps):
        e = rng.standard_normal(T + 4)
        x = np.zeros(T + 4)
        for t in range(4, T + 4):
            x[t] = x[t - 4] + e[t]
        out[r] = _hegy_stats(x[4:], det, 0)[0]
    return out


def _hegy_stats(x, det, k):
    x = np.asarray(x, float)
    T = len(x)
    L = lambda v, j: np.r_[np.full(j, np.nan), v[:-j]]  # noqa: E731
    y1 = x + L(x, 1) + L(x, 2) + L(x, 3)
    y2 = -(x - L(x, 1) + L(x, 2) - L(x, 3))
    y3 = -(x - L(x, 2))
    y4 = x - L(x, 4)
    regs = [L(y1, 1), L(y2, 1), L(y3, 2), L(y3, 1)]
    for j in range(1, k + 1):
        regs.append(L(y4, j))
    D = []
    if det in ("c", "cs", "cst", "ct"):
        D.append(np.ones(T))
    if det in ("cs", "cst"):
        q = np.arange(T) % 4
        D += [(q == j).astype(float) for j in (1, 2, 3)]
    if det in ("ct", "cst"):
        D.append(np.arange(1, T + 1, dtype=float))
    X = np.column_stack(regs + D)
    ok = ~np.isnan(X).any(axis=1) & ~np.isnan(y4)
    X, Y = X[ok], y4[ok]
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    e = Y - X @ b
    s2 = e @ e / (len(Y) - X.shape[1])
    V = s2 * np.linalg.inv(X.T @ X)
    t = b[:4] / np.sqrt(np.diag(V)[:4])
    R = np.zeros((2, X.shape[1]))
    R[0, 2], R[1, 3] = 1, 1
    F34 = float((R @ b) @ np.linalg.solve(R @ V @ R.T, R @ b) / 2)
    return np.array([t[0], t[1], t[2], t[3], F34]), b, len(Y)


def hegy(y, det: str = "cs", lags: int = 0, reps: int = 2000) -> pd.DataFrame:
    """Hylleberg, Engle, Granger & Yoo (1990) test for quarterly data: unit roots at the zero (π₁),
    semi-annual (π₂) and annual (π₃, π₄) frequencies. Critical values / p-values by Monte Carlo under
    the seasonal random walk for the actual sample size (seeded, reproducible)."""
    x = np.asarray(y, float)
    stat, b, n = _hegy_stats(x, det, lags)
    null = _hegy_null(len(x), reps, det)
    names = ["t(π₁) — zero frequency", "t(π₂) — semi-annual (π)", "t(π₃)", "t(π₄)", "F(π₃ ∩ π₄) — annual (π/2)"]
    rows = []
    for j, nm in enumerate(names):
        if j in (2, 3):
            continue
        dist = null[:, j]
        if j == 4:
            p = float(np.mean(dist >= stat[j]))
            cv = np.quantile(dist, [0.99, 0.95, 0.90])
        else:
            p = float(np.mean(dist <= stat[j]))
            cv = np.quantile(dist, [0.01, 0.05, 0.10])
        rows.append({"Test": nm, "Statistic": float(stat[j]), "p-value (simulated)": p, "1%": cv[0], "5%": cv[1],
                     "10%": cv[2], "Unit root at this frequency": "not rejected" if p >= 0.05 else "rejected"})
    out = pd.DataFrame(rows)
    out.attrs.update({"det": det, "lags": lags, "nobs": n, "reps": reps})
    return out


def battery(y, regression: str = "c", kpss_reg: str | None = None) -> pd.DataFrame:
    """ADF, PP, DF-GLS and KPSS on one series with a common reading."""
    rows = []
    a = adf(y, regression)
    rows.append({"Test": "ADF", "H0": "unit root", "Statistic": a["stat"], "5% critical": a["crit"]["5%"],
                 "p-value": a["p"], "Lags": a["lags"], "Reject H0 (5%)": a["p"] < 0.05})
    pp = phillips_perron(y, regression)
    rows.append({"Test": "Phillips-Perron Z(t)", "H0": "unit root", "Statistic": pp["stat"], "5% critical": pp["crit"]["5%"],
                 "p-value": pp["p"], "Lags": pp["lags"], "Reject H0 (5%)": pp["p"] < 0.05})
    if regression in ("c", "ct"):
        g = dfgls(y, regression)
        rows.append({"Test": "DF-GLS (ERS)", "H0": "unit root", "Statistic": g["stat"], "5% critical": g["crit"]["5%"],
                     "p-value": np.nan, "Lags": g["lags"], "Reject H0 (5%)": g["reject_5"]})
        k = kpss(y, kpss_reg or regression)
        rows.append({"Test": "KPSS", "H0": "stationary", "Statistic": k["stat"], "5% critical": k["crit"]["5%"],
                     "p-value": k["p"], "Lags": k["lags"], "Reject H0 (5%)": k["stat"] > k["crit"]["5%"]})
    return pd.DataFrame(rows)


def _verdict(tab) -> str:
    ur = tab[tab["H0"] == "unit root"]["Reject H0 (5%)"]
    st = tab[tab["H0"] == "stationary"]["Reject H0 (5%)"]
    rej_ur = ur.mean() >= 0.5
    rej_st = bool(st.iloc[0]) if len(st) else None
    if rej_ur and rej_st is not True:
        return "stationary"
    if not rej_ur and rej_st is not False:
        return "unit root"
    return "conflicting"


def integration_order(y, regression: str = "ct", max_d: int = 2) -> dict:
    """Test the level, then successive differences, until the series is stationary (ADF/PP/DF-GLS vs KPSS)."""
    x = pd.Series(np.asarray(y, float))
    out, d = [], 0
    reg = regression
    while d <= max_d:
        tab = battery(np.diff(x.to_numpy(), n=d) if d else x.to_numpy(), reg)    # Δ^d (not x_t − x_{t−d})
        v = _verdict(tab)
        tab.insert(0, "Series", "level" if d == 0 else f"Δ^{d}" if d > 1 else "Δ (first difference)")
        tab.insert(1, "Deterministics", REG[reg][0])
        out.append(tab)
        if v == "stationary":
            break
        d += 1
        reg = "c"
    table = pd.concat(out, ignore_index=True)
    return {"order": d if d <= max_d else None, "table": table, "verdict_last": v}


# ==========================================================
# Box-Jenkins
# ==========================================================
def acf_pacf(y, nlags: int = 24) -> pd.DataFrame:
    from statsmodels.tsa.stattools import acf, pacf
    x = np.asarray(y, float)
    nlags = min(nlags, len(x) // 2 - 1)
    a, aci = acf(x, nlags=nlags, alpha=0.05, fft=False)
    p = pacf(x, nlags=nlags, method="ywadjusted")
    n = len(x)
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb = acorr_ljungbox(x, lags=list(range(1, nlags + 1)), return_df=True)
    return pd.DataFrame({"lag": np.arange(1, nlags + 1), "ACF": a[1:], "ACF band": aci[1:, 1] - a[1:],
                         "PACF": p[1:], "PACF band": np.full(nlags, 1.96 / np.sqrt(n)),
                         "Ljung-Box Q": lb["lb_stat"].to_numpy(), "p-value": lb["lb_pvalue"].to_numpy()})


@dataclass
class ArimaResult:
    order: tuple
    seasonal: tuple
    trend: str
    res: object
    y: pd.Series
    s: int
    name: str
    transform: str
    table: pd.DataFrame
    stats: dict
    notes: list = field(default_factory=list)


def fit_arima(y: pd.Series, order=(1, 0, 0), seasonal=(0, 0, 0), s: int = 1, trend: str | None = None,
              name: str = "y", transform: str = "level", refine: bool = True) -> ArimaResult:
    """Exact Gaussian maximum likelihood (Kalman filter, statsmodels ARIMA state-space). `trend`: 'n', 'c'
    (mean / drift handled automatically for d ≥ 1 with 't'), 'ct'."""
    from statsmodels.tsa.arima.model import ARIMA
    d, D = order[1], seasonal[1]
    if trend is None:
        trend = "c" if d + D == 0 else "n"
    sorder = (seasonal[0], seasonal[1], seasonal[2], s) if s > 1 and any(seasonal) else (0, 0, 0, 0)
    x = pd.Series(np.asarray(y, float))
    mod = ARIMA(x, order=order, seasonal_order=sorder, trend=trend)
    res = mod.fit(cov_type="approx")      # numerical-Hessian covariance (as R arima / Stata arima)
    # second start from a derivative-free search: guards against L-BFGS stopping early near MA unit roots
    try:
        if not refine:
            raise RuntimeError
        r_nm = mod.fit(method_kwargs={"method": "nm", "maxiter": 5000})
        r2 = mod.fit(start_params=r_nm.params, cov_type="approx")
        if np.isfinite(r2.llf) and r2.llf > res.llf + 1e-7:
            res = r2
    except Exception:  # noqa: BLE001
        pass
    names = list(res.param_names)
    tab = pd.DataFrame({"Parameter": names, "Coef.": res.params.values, "Std. Err.": res.bse.values,
                        "z": res.tvalues.values, "p-value": res.pvalues.values,
                        "CI low": res.conf_int().iloc[:, 0].values, "CI high": res.conf_int().iloc[:, 1].values})
    n = int(res.nobs)
    k = len(names)
    aicc = res.aic + 2 * k * (k + 1) / max(n - k - 1, 1)
    st = {"N": n, "Log-likelihood": float(res.llf), "AIC": float(res.aic), "AICc": float(aicc), "BIC": float(res.bic),
          "σ²": float(res.params.get("sigma2", np.nan))}
    lab = f"ARIMA({order[0]},{order[1]},{order[2]})" + (f"({seasonal[0]},{seasonal[1]},{seasonal[2]})[{s}]" if sorder[3] else "")
    out = ArimaResult(order=tuple(order), seasonal=tuple(seasonal), trend=trend, res=res, y=y, s=s, name=lab,
                      transform=transform, table=tab, stats=st)
    ar_r, ma_r = roots(res)
    if len(ar_r) and np.min(np.abs(ar_r)) <= 1.0001:
        out.notes.append("An AR root lies on / inside the unit circle: the model is not stationary — consider differencing.")
    if len(ma_r) and np.min(np.abs(ma_r)) <= 1.0001:
        out.notes.append("An MA root lies on / inside the unit circle: the model is not invertible (possible over-differencing).")
    return out


def roots(res):
    try:
        ar = np.asarray(res.arroots)
    except Exception:  # noqa: BLE001
        ar = np.array([])
    try:
        ma = np.asarray(res.maroots)
    except Exception:  # noqa: BLE001
        ma = np.array([])
    return ar, ma


def auto_arima(y: pd.Series, s: int = 1, d: int | None = None, D: int = 0, max_p: int = 3, max_q: int = 3,
               max_P: int = 1, max_Q: int = 1, criterion: str = "AICc", trend: str | None = None) -> pd.DataFrame:
    """Grid search over (p, q)(P, Q) for given d, D (d from the unit-root strategy when not supplied)."""
    if d is None:
        d = integration_order(y, "ct").get("order") or 0
    seas = [(P, D, Q) for P in range(max_P + 1) for Q in range(max_Q + 1)] if s > 1 else [(0, 0, 0)]
    rows = []
    for (p, q), sea in product(product(range(max_p + 1), range(max_q + 1)), seas):
        try:
            r = fit_arima(y, (p, d, q), sea, s, trend, refine=False)
            lb = residual_tests(r)["lb_p"]
            rows.append({"Model": r.name, "p": p, "d": d, "q": q, "P": sea[0], "D": sea[1], "Q": sea[2],
                         "AIC": r.stats["AIC"], "AICc": r.stats["AICc"], "BIC": r.stats["BIC"],
                         "Ljung-Box p": lb, "Valid": not r.notes})
        except Exception:  # noqa: BLE001
            continue
    out = pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)
    return out


def residual_tests(r: ArimaResult, lags: int | None = None) -> dict:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    e = np.asarray(r.res.resid)[max(r.order[1] + r.seasonal[1] * r.s, 0):]
    n = len(e)
    L = lags or (min(2 * r.s, n // 5) if r.s > 1 else min(10, n // 5))
    dfm = r.order[0] + r.order[2] + r.seasonal[0] + r.seasonal[2]
    L = max(L, dfm + 1)
    lb = acorr_ljungbox(e, lags=[L], model_df=dfm, return_df=True)
    jb = stats.jarque_bera(e)
    arch = het_arch(e, nlags=min(4, n // 10 or 1))
    return {"lb_stat": float(lb["lb_stat"].iloc[0]), "lb_p": float(lb["lb_pvalue"].iloc[0]), "lb_lags": L, "lb_df": L - dfm,
            "jb": float(jb.statistic), "jb_p": float(jb.pvalue), "arch": float(arch[0]), "arch_p": float(arch[1]),
            "resid": e}


def forecast(r: ArimaResult, h: int = 12, level: float = 0.95, back: bool = True) -> pd.DataFrame:
    f = r.res.get_forecast(h)
    mu = np.asarray(f.predicted_mean)
    ci = np.asarray(f.conf_int(alpha=1 - level))
    se = np.asarray(f.se_mean)
    out = pd.DataFrame({"step": np.arange(1, h + 1), "forecast": mu, "lower": ci[:, 0], "upper": ci[:, 1], "se": se})
    idx = r.y.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 2:
        freq = pd.infer_freq(idx) or {12: "MS", 4: "QS", 1: "YS"}.get(r.s, "D")
        out.insert(0, "time", pd.date_range(idx[-1], periods=h + 1, freq=freq)[1:])
    else:
        out.insert(0, "time", np.asarray(idx)[-1] + np.arange(1, h + 1))
    if back and r.transform == "log":
        out["forecast (level, median)"] = np.exp(mu)
        out["forecast (level, mean)"] = np.exp(mu + se ** 2 / 2)
        out["lower (level)"] = np.exp(ci[:, 0])
        out["upper (level)"] = np.exp(ci[:, 1])
    return out


def holdout(y: pd.Series, order, seasonal=(0, 0, 0), s: int = 1, test: int = 12, trend: str | None = None) -> dict:
    """Out-of-sample evaluation: fit on all but the last `test` points, forecast them, compare with the naive
    (random walk) and seasonal-naive benchmarks; Diebold-Mariano test vs naive."""
    tr, te = y.iloc[:-test], y.iloc[-test:]
    r = fit_arima(tr, order, seasonal, s, trend)
    fc = np.asarray(r.res.get_forecast(test).predicted_mean)
    actual = te.to_numpy()
    naive = np.full(test, tr.iloc[-1])
    snaive = np.array([tr.iloc[-s + (i % s)] for i in range(test)]) if s > 1 and len(tr) >= s else naive

    def m(f):
        e = actual - f
        return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
                "MAPE %": float(100 * np.mean(np.abs(e / actual))) if np.all(actual != 0) else np.nan}
    rows = [{"Method": r.name, **m(fc)}, {"Method": "Naive (random walk)", **m(naive)}]
    if s > 1:
        rows.append({"Method": "Seasonal naive", **m(snaive)})
    tab = pd.DataFrame(rows)
    tab["Theil U2 (vs naive)"] = tab["RMSE"] / tab.loc[1, "RMSE"]
    e1, e2 = actual - fc, actual - naive
    dm = diebold_mariano(e1, e2)
    return {"table": tab, "forecast": fc, "actual": actual, "time": te.index, "dm": dm, "train_model": r}


def diebold_mariano(e1, e2, h: int = 1) -> dict:
    """Diebold-Mariano (1995) test of equal squared-error accuracy with the Harvey-Leybourne-Newbold correction."""
    d = np.asarray(e1) ** 2 - np.asarray(e2) ** 2
    n = len(d)
    lrv = _nw_lrv(d - d.mean(), h - 1) if h > 1 else np.var(d)
    stat = d.mean() / np.sqrt(lrv / n)
    stat *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return {"stat": float(stat), "p": float(2 * stats.t.sf(abs(stat), n - 1))}


# ==========================================================
# Long memory
# ==========================================================
def _periodogram(x):
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.fft(x)
    m = (n - 1) // 2
    lam = 2 * np.pi * np.arange(1, m + 1) / n
    I = np.abs(f[1:m + 1]) ** 2 / (2 * np.pi * n)
    return lam, I


def gph(y, bandwidth: float = 0.5) -> dict:
    """Geweke & Porter-Hudak (1983) log-periodogram estimator of d with m = n^bandwidth frequencies
    (regressor −ln(4 sin²(λ/2)), as R fracdiff::fdGPH)."""
    lam, I = _periodogram(y)
    m = int(np.floor(len(np.asarray(y)) ** bandwidth))
    X = -np.log(4 * np.sin(lam[:m] / 2) ** 2)
    Yv = np.log(I[:m])
    Xc = np.c_[np.ones(m), X]
    b = np.linalg.lstsq(Xc, Yv, rcond=None)[0]
    e = Yv - Xc @ b
    se_reg = np.sqrt((e @ e / (m - 2)) / np.sum((X - X.mean()) ** 2))
    se_asy = np.pi / np.sqrt(6 * np.sum((X - X.mean()) ** 2))
    return {"d": float(b[1]), "se": float(se_asy), "se_reg": float(se_reg), "m": m,
            "p (d = 0)": float(2 * stats.norm.sf(abs(b[1] / se_asy)))}


def local_whittle(y, bandwidth: float = 0.65) -> dict:
    """Robinson (1995) Gaussian semiparametric (local Whittle) estimator of d; s.e. = 1 / (2√m)."""
    lam, I = _periodogram(y)
    m = int(np.floor(len(np.asarray(y)) ** bandwidth))
    lm, Im = lam[:m], I[:m]

    def R(d):
        return np.log(np.mean(lm ** (2 * d) * Im)) - 2 * d * np.mean(np.log(lm))
    o = optimize.minimize_scalar(R, bounds=(-0.49, 1.49), method="bounded")
    se = 1 / (2 * np.sqrt(m))
    return {"d": float(o.x), "se": float(se), "m": m, "p (d = 0)": float(2 * stats.norm.sf(abs(o.x / se)))}


def frac_diff(y, d: float) -> np.ndarray:
    x = np.asarray(y, float)
    n = len(x)
    w = np.empty(n)
    w[0] = 1.0
    for k in range(1, n):
        w[k] = -w[k - 1] * (d - k + 1) / k
    return np.array([w[:t + 1][::-1] @ x[:t + 1] for t in range(n)])


def arfima_whittle(y, p: int = 0, q: int = 0) -> dict:
    """ARFIMA(p, d, q) by the Whittle (frequency-domain) likelihood, σ² profiled out; standard errors from the
    numerical Hessian of the Whittle objective."""
    x = np.asarray(y, float)
    lam, I = _periodogram(x)
    n = len(x)
    z = np.exp(-1j * lam)

    def spec(th):
        d, ar, ma = th[0], th[1:1 + p], th[1 + p:1 + p + q]
        phi = 1 - sum(a * z ** (j + 1) for j, a in enumerate(ar)) if p else np.ones_like(z)
        theta = 1 + sum(m_ * z ** (j + 1) for j, m_ in enumerate(ma)) if q else np.ones_like(z)
        return np.abs(1 - z) ** (-2 * d) * np.abs(theta) ** 2 / np.abs(phi) ** 2

    def obj(th):
        g = spec(th)
        if np.any(~np.isfinite(g)) or np.any(g <= 0):
            return 1e10
        return np.log(np.mean(I / g)) + np.mean(np.log(g))
    best = None
    for d0 in (0.1, 0.3, 0.45):
        o = optimize.minimize(obj, np.r_[d0, np.zeros(p + q)], method="Nelder-Mead",
                              options={"xatol": 1e-8, "fatol": 1e-12, "maxiter": 20000})
        if best is None or o.fun < best.fun:
            best = o
    th = best.x
    from statsmodels.tools.numdiff import approx_hess
    m = len(lam)
    H = approx_hess(th, lambda t: m * obj(t))
    try:
        cov = np.linalg.inv(H)
        se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        se = np.full(len(th), np.nan)
    names = ["d"] + [f"ar.L{j + 1}" for j in range(p)] + [f"ma.L{j + 1}" for j in range(q)]
    sig2 = 2 * np.pi * np.mean(I / spec(th))
    tab = pd.DataFrame({"Parameter": names, "Coef.": th, "Std. Err.": se, "z": th / se,
                        "p-value": 2 * stats.norm.sf(np.abs(th / se))})
    return {"table": tab, "d": float(th[0]), "sigma2": float(sig2), "objective": float(best.fun), "n": n}


# ==========================================================
# Reporting helpers
# ==========================================================
REFS = {
    "boxjenkins": "Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). Time series analysis: Forecasting and control (5th ed.). Wiley.",
    "hamilton": "Hamilton, J. D. (1994). Time series analysis. Princeton University Press.",
    "df": "Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. Journal of the American Statistical Association, 74(366), 427–431.",
    "said": "Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in autoregressive-moving average models of unknown order. Biometrika, 71(3), 599–607.",
    "mackinnon10": "MacKinnon, J. G. (2010). Critical values for cointegration tests (Queen's Economics Department Working Paper No. 1227).",
    "pp": "Phillips, P. C. B., & Perron, P. (1988). Testing for a unit root in time series regression. Biometrika, 75(2), 335–346.",
    "kpss": "Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. Journal of Econometrics, 54(1–3), 159–178.",
    "ers": "Elliott, G., Rothenberg, T. J., & Stock, J. H. (1996). Efficient tests for an autoregressive unit root. Econometrica, 64(4), 813–836.",
    "ngperron": "Ng, S., & Perron, P. (2001). Lag length selection and the construction of unit root tests with good size and power. Econometrica, 69(6), 1519–1554.",
    "za": "Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis. Journal of Business & Economic Statistics, 10(3), 251–270.",
    "hegy": "Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S. (1990). Seasonal integration and cointegration. Journal of Econometrics, 44(1–2), 215–238.",
    "hp": "Hodrick, R. J., & Prescott, E. C. (1997). Postwar U.S. business cycles: An empirical investigation. Journal of Money, Credit and Banking, 29(1), 1–16.",
    "ravn": "Ravn, M. O., & Uhlig, H. (2002). On adjusting the Hodrick-Prescott filter for the frequency of observations. Review of Economics and Statistics, 84(2), 371–376.",
    "bk": "Baxter, M., & King, R. G. (1999). Measuring business cycles: Approximate band-pass filters for economic time series. Review of Economics and Statistics, 81(4), 575–593.",
    "cf": "Christiano, L. J., & Fitzgerald, T. J. (2003). The band pass filter. International Economic Review, 44(2), 435–465.",
    "stl": "Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. (1990). STL: A seasonal-trend decomposition procedure based on loess. Journal of Official Statistics, 6(1), 3–73.",
    "ljungbox": "Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in time series models. Biometrika, 65(2), 297–303.",
    "engle": "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987–1007.",
    "dm": "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.",
    "hln": "Harvey, D., Leybourne, S., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281–291.",
    "hyndman": "Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles and practice (3rd ed.). OTexts.",
    "gph": "Geweke, J., & Porter-Hudak, S. (1983). The estimation and application of long memory time series models. Journal of Time Series Analysis, 4(4), 221–238.",
    "robinson": "Robinson, P. M. (1995). Gaussian semiparametric estimation of long range dependence. Annals of Statistics, 23(5), 1630–1661.",
    "granger": "Granger, C. W. J., & Joyeux, R. (1980). An introduction to long-memory time series models and fractional differencing. Journal of Time Series Analysis, 1(1), 15–29.",
    "hosking": "Hosking, J. R. M. (1981). Fractional differencing. Biometrika, 68(1), 165–176.",
    "fox": "Fox, R., & Taqqu, M. S. (1986). Large-sample properties of parameter estimates for strongly dependent stationary Gaussian time series. Annals of Statistics, 14(2), 517–532.",
}
