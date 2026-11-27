# ==========================================
# Aetheris Research - Forecasting engines (v1.9, Catalogue §3.1, §3.3, §3.4, §3.6)
# Benchmarks, ETS, theta, state space, SARIMAX, BSTS, growth curves, HAR-RV, machine learning, Prophet-style,
# accuracy metrics, Diebold-Mariano, rolling-origin backtests, combinations, scenarios and the model horse-race.
# Conventions follow the R forecast package (Hyndman & Khandakar 2008; Hyndman & Athanasopoulos, FPP).
# ==========================================

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore")

LEVELS = (80, 95)


@dataclass
class Fc:
    """A forecast: point path, prediction intervals {level: (lower, upper)}, in-sample fitted values and residuals."""
    name: str
    mean: np.ndarray
    pi: dict
    fitted: np.ndarray | None = None
    resid: np.ndarray | None = None
    info: dict = field(default_factory=dict)
    sims: np.ndarray | None = None          # optional simulated paths (draws × h) for fan charts

    def frame(self, index=None) -> pd.DataFrame:
        h = len(self.mean)
        out = pd.DataFrame({"step": np.arange(1, h + 1), "forecast": self.mean})
        for lv in sorted(self.pi):
            lo, hi = self.pi[lv]
            out[f"lower {lv}%"] = lo
            out[f"upper {lv}%"] = hi
        if index is not None:
            out.insert(0, "time", list(index)[:h])
        return out

    def quantiles(self, qs=(0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95)) -> dict:
        """Fan-chart quantiles: from simulated paths when available, otherwise Gaussian from the 95% interval."""
        if self.sims is not None:
            return {q: np.quantile(self.sims, q, axis=0) for q in qs}
        lo, hi = self.pi[95] if 95 in self.pi else self.pi[max(self.pi)]
        se = (np.asarray(hi) - np.asarray(lo)) / (2 * stats.norm.ppf(0.975))
        return {q: self.mean + stats.norm.ppf(q) * se for q in qs}


def _gauss_pi(mean, se, levels=LEVELS, df=None):
    q = (lambda lv: stats.t.ppf(0.5 + lv / 200, df)) if df else (lambda lv: stats.norm.ppf(0.5 + lv / 200))
    return {lv: (mean - q(lv) * se, mean + q(lv) * se) for lv in levels}


def future_index(idx, h: int, s: int | None = None):
    idx = pd.Index(idx)
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 2:
        freq = idx.freq or pd.infer_freq(idx) or {12: "MS", 4: "QS", 1: "YS", 52: "W", 7: "D", 5: "B"}.get(s or 1, "D")
        return pd.date_range(idx[-1], periods=h + 1, freq=freq)[1:]
    try:
        return np.asarray(idx)[-1] + np.arange(1, h + 1)
    except Exception:  # noqa: BLE001
        return np.arange(len(idx) + 1, len(idx) + h + 1)


# ==========================================================
# §3.1 Benchmarks (forecast::meanf, naive, snaive, rwf(drift = TRUE)) and moving averages
# ==========================================================
def mean_fc(y, h: int, levels=LEVELS) -> Fc:
    y = np.asarray(y, float)
    n = len(y)
    m, s = y.mean(), y.std(ddof=1)
    se = np.full(h, s * np.sqrt(1 + 1 / n))
    return Fc("Mean", np.full(h, m), _gauss_pi(np.full(h, m), se, levels, df=n - 1), np.full(n, m), y - m)


def naive(y, h: int, levels=LEVELS) -> Fc:
    y = np.asarray(y, float)
    fit = np.r_[np.nan, y[:-1]]
    e = y - fit
    sig = np.sqrt(np.nanmean(e ** 2))
    mean = np.full(h, y[-1])
    return Fc("Naive", mean, _gauss_pi(mean, sig * np.sqrt(np.arange(1, h + 1)), levels), fit, e, {"sigma": sig})


def snaive(y, h: int, s: int, levels=LEVELS) -> Fc:
    y = np.asarray(y, float)
    if s < 2 or len(y) < s + 1:
        out = naive(y, h, levels)
        out.name = "Seasonal naive"
        return out
    fit = np.r_[[np.nan] * s, y[:-s]]
    e = y - fit
    sig = np.sqrt(np.nanmean(e ** 2))
    mean = np.array([y[len(y) - s + (i % s)] for i in range(h)])
    se = sig * np.sqrt(np.floor(np.arange(h) / s) + 1)
    return Fc("Seasonal naive", mean, _gauss_pi(mean, se, levels), fit, e, {"sigma": sig})


def drift(y, h: int, levels=LEVELS) -> Fc:
    y = np.asarray(y, float)
    n = len(y)
    dy = np.diff(y)
    b = dy.mean()
    e = dy - b
    sig = np.sqrt(np.sum(e ** 2) / (n - 2))
    bse = sig / np.sqrt(n - 1)
    k = np.arange(1, h + 1)
    mean = y[-1] + b * k
    se = np.sqrt(sig ** 2 * k + (k * bse) ** 2)
    return Fc("Random walk with drift", mean, _gauss_pi(mean, se, levels), np.r_[np.nan, y[:-1] + b], np.r_[np.nan, e],
              {"drift": b, "drift se": bse, "sigma": sig})


def moving_average(y, h: int, k: int = 3, levels=LEVELS) -> Fc:
    """Trailing k-period moving-average forecast; intervals from the one-step errors of the same rule (constant width)."""
    y = np.asarray(y, float)
    ma = pd.Series(y).rolling(k).mean().to_numpy()
    fit = np.r_[np.nan, ma[:-1]]
    e = y - fit
    sig = np.sqrt(np.nanmean(e ** 2))
    mean = np.full(h, ma[-1])
    return Fc(f"Moving average ({k})", mean, _gauss_pi(mean, np.full(h, sig), levels), fit, e, {"sigma": sig, "k": k})


# ==========================================================
# §3.1 Exponential smoothing (ETS state-space framework; statsmodels ETSModel, R forecast::ets likelihood)
# ==========================================================
ETS_LABEL = {"add": "A", "mul": "M", None: "N"}


class _EtsFit:
    """Wraps a statsmodels ETS results object with information criteria on the R forecast::ets scale."""

    def __init__(self, res, n_free: int, n: int, theta):
        self.res = res
        self.theta = theta
        self.params = np.asarray(res.params)
        self.model = res.model
        self.fittedvalues = np.asarray(res.fittedvalues)
        off = n / 2 * (np.log(n) - 1 - np.log(2 * np.pi))
        self.llf = float(res.llf - off)
        k = n_free + 1
        self.k = k
        self.aic = -2 * self.llf + 2 * k
        self.aicc = self.aic + 2 * k * (k + 1) / max(n - k - 1, 1)
        self.bic = -2 * self.llf + np.log(n) * k

    def forecast(self, h):
        return self.res.forecast(h)

    def get_prediction(self, **kw):
        return self.res.get_prediction(**kw)


def ets_fit(y, error="add", trend=None, damped=False, seasonal=None, s: int = 1, params=None, start=None, effort: str = "full"):
    """ETS by maximum likelihood with R forecast::ets conventions: 'usual' parameter region (1e-4 ≤ α ≤ 0.9999,
    1e-4 ≤ β ≤ α, 1e-4 ≤ γ ≤ 1 − α, 0.8 ≤ φ ≤ 0.98) and normalised initial seasonal states (Σs = m multiplicative,
    0 additive), so m − 1 seasonal states are free. Likelihood evaluated by statsmodels' ETS recursions; optimised by
    Powell from several starts with Nelder-Mead polishing. `params` (full statsmodels vector) evaluates a given model."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel
    y = np.asarray(y, float)
    n = len(y)
    m = s if seasonal else 1
    mod = ETSModel(pd.Series(y), error=error, trend=trend, damped_trend=bool(damped and trend), seasonal=seasonal,
                   seasonal_periods=s if seasonal else None, initialization_method="estimated")
    names = list(mod.param_names)
    ia = names.index("smoothing_level")
    ib = names.index("smoothing_trend") if "smoothing_trend" in names else None
    ig = names.index("smoothing_seasonal") if "smoothing_seasonal" in names else None
    ip = names.index("damping_trend") if "damping_trend" in names else None
    seas_idx = [i for i, nm in enumerate(names) if nm.startswith("initial_seasonal")]
    norm = (m if seasonal == "mul" else 0.0)

    def full(th):
        if not seasonal:
            return th
        return np.r_[th, norm - th[seas_idx[:-1]].sum()]

    n_free = len(names) - (1 if seasonal else 0)
    if params is not None:
        params = np.asarray(params, float)
        if seasonal and len(params) == len(names) - 1:
            params = full(params)
        return _EtsFit(mod.smooth(params), n_free, n, params[:n_free])

    def nll(th):
        a = th[ia]
        if not 1e-4 <= a <= 0.9999:
            return 1e12
        if ib is not None and not 1e-4 <= th[ib] <= a:
            return 1e12
        if ig is not None and not 1e-4 <= th[ig] <= 1 - a:
            return 1e12
        if ip is not None and not 0.8 <= th[ip] <= 0.98:
            return 1e12
        if seasonal == "mul" and np.any(full(th)[seas_idx] <= 0):
            return 1e12
        try:
            v = -mod.loglike(full(th))
        except Exception:  # noqa: BLE001
            return 1e12
        return v if np.isfinite(v) else 1e12

    sp = np.asarray(mod.start_params, float).copy()
    if seasonal:
        sv = sp[seas_idx]
        sp[seas_idx] = sv / sv.mean() if seasonal == "mul" else sv - sv.mean()
    sp = sp[:n_free]
    starts = []
    if start is not None:
        starts.append(np.asarray(start, float)[:n_free])
    for a0 in ((0.1, 0.4, 0.8) if effort == "full" else (0.3,)) if start is None or effort == "full" else ():
        st = sp.copy()
        st[ia] = a0
        if ib is not None:
            st[ib] = min(0.01, a0 / 2)
        if ig is not None:
            st[ig] = min(0.01, (1 - a0) / 2)
        if ip is not None:
            st[ip] = 0.97
        starts.append(st)
    best = None
    for st in starts:
        if nll(st) >= 1e12:
            continue
        r = optimize.minimize(nll, st, method="Powell", options={"maxiter": 20000, "maxfev": 20000, "xtol": 1e-6, "ftol": 1e-10})
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        raise ValueError("ETS: no admissible starting values")
    if effort == "full":
        for _ in range(3):
            r = optimize.minimize(nll, best.x, method="Nelder-Mead", options={"maxiter": 6000, "maxfev": 6000, "adaptive": True,
                                                                              "xatol": 1e-8, "fatol": 1e-10})
            r2 = optimize.minimize(nll, r.x, method="Powell", options={"maxiter": 20000, "xtol": 1e-8, "ftol": 1e-12})
            if r2.fun < best.fun - 1e-9:
                best = r2
            else:
                break
    return _EtsFit(mod.smooth(full(best.x)), n_free, n, best.x)


def ets_label(error, trend, damped, seasonal):
    return f"ETS({ETS_LABEL[error]},{ETS_LABEL[trend]}{'d' if damped and trend else ''},{ETS_LABEL[seasonal]})"


def ets_auto(y, s: int = 1, criterion: str = "aicc", allow_mult_trend: bool = False, damped=None, refine: int = 3) -> pd.DataFrame:
    """Fit the admissible ETS family (as forecast::ets: no additive-error × multiplicative-seasonal models;
    multiplicative components only for positive data) and rank by AICc."""
    y = np.asarray(y, float)
    pos = bool(np.all(y > 0))
    errs = ["add", "mul"] if pos else ["add"]
    trends = [None, "add"] + (["mul"] if allow_mult_trend and pos else [])
    seas = [None] + (["add", "mul"] if s > 1 and len(y) >= 2 * s + 2 else [])
    if not pos:
        seas = [v for v in seas if v != "mul"]
    rows = []
    for e, t, sn in product(errs, trends, seas):
        for d in ([False, True] if t else [False]):
            if damped is not None and t and d != damped:
                continue
            if e == "add" and (sn == "mul" or t == "mul"):
                continue
            try:
                r = ets_fit(y, e, t, d, sn, s, effort="quick")
                if len(y) <= r.k + 1:
                    continue
                rows.append({"Model": ets_label(e, t, d, sn), "error": e, "trend": t, "damped": d, "seasonal": sn,
                             "AIC": r.aic, "AICc": r.aicc, "BIC": r.bic, "loglik": r.llf, "k": r.k, "_theta": r.theta})
            except Exception:  # noqa: BLE001
                continue
    if not rows:
        raise ValueError("No ETS model could be estimated")
    crit = {"aicc": "AICc", "aic": "AIC", "bic": "BIC"}[criterion]
    tab = pd.DataFrame(rows).sort_values(crit).reset_index(drop=True)
    for i in range(min(refine, len(tab))):
        rw = tab.iloc[i]
        r = ets_fit(y, rw.error, rw.trend, rw.damped, rw.seasonal, s, start=rw._theta, effort="full")
        if r.llf > rw.loglik:
            tab.loc[i, ["AIC", "AICc", "BIC", "loglik"]] = [r.aic, r.aicc, r.bic, r.llf]
            tab.at[i, "_theta"] = r.theta
    return tab.sort_values(crit).reset_index(drop=True)


def ets(y, h: int, s: int = 1, spec: dict | None = None, levels=LEVELS) -> Fc:
    start = None
    if spec is None:
        tab = ets_auto(y, s)
        spec = tab.iloc[0][["error", "trend", "damped", "seasonal"]].to_dict()
        spec = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in spec.items()}
        start = tab.iloc[0]["_theta"]
    spec = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in spec.items()}
    r = ets_fit(y, spec["error"], spec["trend"], bool(spec.get("damped")), spec["seasonal"], s,
                params=spec.get("params"), start=start if start is not None else spec.get("start"),
                effort="quick" if start is not None else spec.get("effort", "full"))
    pr = r.get_prediction(start=len(y), end=len(y) + h - 1)
    mean = np.asarray(pr.predicted_mean)
    n = len(y)
    scale = np.sqrt(n / (n - (r.k - 1)))          # σ² = SSE / (n − #parameters), as forecast::ets
    pi = {}
    for lv in levels:
        sf = pr.summary_frame(alpha=1 - lv / 100)
        pi[lv] = (mean - scale * (mean - sf["pi_lower"].to_numpy()), mean + scale * (sf["pi_upper"].to_numpy() - mean))
    name = ets_label(spec["error"], spec["trend"], bool(spec.get("damped")), spec["seasonal"])
    par = dict(zip(r.model.param_names, np.asarray(r.params)))
    return Fc(name, mean, pi, np.asarray(r.fittedvalues), np.asarray(y) - np.asarray(r.fittedvalues),
              {"params": par, "AICc": float(r.aicc), "AIC": float(r.aic), "BIC": float(r.bic), "loglik": float(r.llf),
               "spec": {**spec, "start": r.theta}, "k": r.k})


def ses(y, h, levels=LEVELS):
    f = ets(y, h, 1, {"error": "add", "trend": None, "damped": False, "seasonal": None}, levels)
    f.name = "Simple exponential smoothing"
    return f


def holt(y, h, damped=False, levels=LEVELS):
    f = ets(y, h, 1, {"error": "add", "trend": "add", "damped": damped, "seasonal": None}, levels)
    f.name = "Holt" + (" (damped)" if damped else "")
    return f


def holt_winters(y, h, s, seasonal="mul", damped=False, levels=LEVELS):
    f = ets(y, h, s, {"error": "mul" if seasonal == "mul" else "add", "trend": "add", "damped": damped, "seasonal": seasonal}, levels)
    f.name = f"Holt-Winters ({'multiplicative' if seasonal == 'mul' else 'additive'})"
    return f


# ==========================================================
# §3.1 Theta (forecast::thetaf — Hyndman & Billah 2003)
# ==========================================================
def seasonality_test(y, s: int) -> bool:
    """forecast::thetaf test: |r_s| / sqrt((1 + 2 Σ_{k<s} r_k²) / n) > z(0.95)."""
    y = np.asarray(y, float)
    n = len(y)
    if s < 2 or n < 2 * s:
        return False
    yc = y - y.mean()
    den = yc @ yc
    r = np.array([yc[k:] @ yc[:-k] / den for k in range(1, s + 1)])
    stat = abs(r[s - 1]) / np.sqrt((1 + 2 * np.sum(r[:s - 1] ** 2)) / n)
    return bool(stat > stats.norm.ppf(0.95))


def classical_mult_index(y, s: int) -> np.ndarray:
    """Seasonal indices of R's decompose(type = 'multiplicative') (centred MA, mean-normalised), aligned to y[0]."""
    y = np.asarray(y, float)
    n = len(y)
    if s % 2 == 0:
        w = np.r_[0.5, np.ones(s - 1), 0.5] / s
    else:
        w = np.ones(s) / s
    tr = np.convolve(y, w, mode="same")
    k = len(w) // 2
    tr[:k] = np.nan
    tr[n - k:] = np.nan
    ratio = y / tr
    per = np.array([np.nanmean(ratio[i::s]) for i in range(s)])
    per = per / per.mean()
    return per


def theta(y, h: int, s: int = 1, levels=LEVELS, params=None) -> Fc:
    """forecast::thetaf: SES on the (multiplicatively) seasonally adjusted series plus half the OLS trend slope with
    Hyndman-Billah's correction; intervals σ·sqrt(jα² + 1) around the re-seasonalised mean (as forecast::thetaf).
    `params` = (alpha, l0) evaluates the model at given SES parameters (for cross-checks)."""
    y = np.asarray(y, float)
    n = len(y)
    seasonal = seasonality_test(y, s) if s > 1 else False
    if seasonal:
        idx = classical_mult_index(y, s)
        if np.any(np.abs(idx) < 1e-4):
            seasonal = False
    if seasonal:
        si = idx[np.arange(n) % s]
        x = y / si
    else:
        x = y
    r = ets_fit(x, "add", None, False, None, 1, None if params is None else list(params))
    alpha = max(1e-10, float(np.asarray(r.params)[0]))
    fit = np.asarray(r.fittedvalues)
    e = x - fit
    sig2 = float(e @ e / (n - 2))
    level_T = float(np.asarray(r.forecast(1))[0])
    b = np.polyfit(np.arange(n), x, 1)[0] / 2
    mean = level_T + b * (np.arange(h) + (1 - (1 - alpha) ** n) / alpha)
    fitted = fit
    if seasonal:
        fs = idx[(n + np.arange(h)) % s]
        mean = mean * fs
        fitted = fit * si
    se = np.sqrt(sig2) * np.sqrt(np.arange(h) * alpha ** 2 + 1)
    return Fc("Theta", mean, _gauss_pi(mean, se, levels), fitted, y - fitted,
              {"alpha": alpha, "drift (b/2)": b, "seasonally adjusted": seasonal, "sigma2": sig2})


# ==========================================================
# §3.1 State space (structural time-series models by Kalman filter / smoother)
# ==========================================================
UC_LEVELS = {"llevel": ("Local level", "المستوى المحلي"), "lltrend": ("Local linear trend", "الاتجاه الخطي المحلي"),
             "strend": ("Smooth trend", "الاتجاه الممهد"), "rwdrift": ("Random walk with drift", "السير العشوائي بانجراف")}


def state_space(y, h: int, level: str = "lltrend", s: int = 1, stochastic_seasonal: bool = True, exog=None,
                exog_future=None, cycle: bool = False, levels=LEVELS) -> Fc:
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    y = np.asarray(y, float)
    kw = {}
    if s > 1:
        kw.update(seasonal=s, stochastic_seasonal=stochastic_seasonal)
    if cycle:
        kw.update(cycle=True, stochastic_cycle=True, damped_cycle=True)
    mod = UnobservedComponents(y, level=level, exog=exog, **kw)
    r = mod.fit(disp=False, maxiter=1000)
    try:
        r2 = mod.fit(disp=False, method="nm", maxiter=4000)
        r = mod.fit(start_params=r2.params, disp=False) if r2.llf > r.llf else r
    except Exception:  # noqa: BLE001
        pass
    fc = r.get_forecast(h, exog=exog_future)
    mean = np.asarray(fc.predicted_mean)
    pi = {lv: tuple(np.asarray(fc.conf_int(alpha=1 - lv / 100)).T) for lv in levels}
    comps = {}
    for nm in ("level", "trend", "seasonal", "cycle"):
        c = getattr(r, nm, None)
        if c is not None and getattr(c, "smoothed", None) is not None:
            comps[nm] = np.asarray(c["smoothed"])
    fitted = np.asarray(r.fittedvalues)
    d = max(1, s if s > 1 else 1)
    tab = pd.DataFrame({"Parameter": r.model.param_names, "Estimate": np.asarray(r.params), "Std. Err.": np.asarray(r.bse)})
    return Fc(f"State space — {UC_LEVELS.get(level, (level,))[0]}" + (" + seasonal" if s > 1 else ""), mean, pi, fitted,
              y - fitted, {"table": tab, "components": comps, "loglik": float(r.llf), "AIC": float(r.aic), "burn": d})


# ==========================================================
# §3.1 ARIMA / SARIMA / SARIMAX (exact ML), automatic order selection
# ==========================================================
def sarimax(y, h: int, order=(1, 0, 0), seasonal=(0, 0, 0), s: int = 1, exog=None, exog_future=None, trend=None,
            levels=LEVELS, polish: bool = True) -> Fc:
    """Regression with SARIMA errors by exact ML on the differenced data (as R forecast::Arima, method = "ML"):
    the mean (d + D = 0) and the drift (d + D = 1) enter as regressors 1 and t, as in R. Forecasts and intervals come
    from the state-space model in levels at the estimated parameters; σ² = SSR / (n* − k), with n* = n − d − D·s and
    k parameters, as forecast::Arima."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    y = np.asarray(y, float)
    n = len(y)
    d, D = order[1], seasonal[1]
    so = (*seasonal, s) if s > 1 and any(seasonal) else (0, 0, 0, 0)
    if so[3] == 0:
        D = 0
    if trend is None:
        trend = "c" if d + D == 0 else "n"          # R Arima defaults: include.mean = TRUE, include.drift = FALSE
    cols, names = [], []
    if trend == "c" and d + D == 0:
        cols.append(np.ones(n)); names.append("intercept")
    if trend == "t" and d + D == 1:
        cols.append(np.arange(1, n + 1, dtype=float)); names.append("drift")
    X_user = None if exog is None else np.asarray(exog, float).reshape(n, -1)
    if X_user is not None:
        cols += list(X_user.T)
        names += [f"x{j + 1}" for j in range(X_user.shape[1])] if not hasattr(exog, "columns") else list(exog.columns)
    X = np.column_stack(cols) if cols else None
    mod = SARIMAX(y, exog=X, order=order, seasonal_order=so, simple_differencing=True)
    cands = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in ({}, {"method": "nm", "maxiter": 6000}, {"method": "powell", "maxiter": 6000}):
            try:
                r0 = mod.fit(disp=False, maxiter=kw.pop("maxiter", 1000), **kw)
                if np.isfinite(r0.llf):
                    cands.append(r0)
                    if not kw and not polish:
                        break
            except Exception:  # noqa: BLE001
                continue
        if polish and cands:
            try:
                b0 = max([c for c in cands if _admissible(c)] or cands, key=lambda c: c.llf)
                r2 = mod.fit(start_params=b0.params, disp=False, method="nm", maxiter=6000)
                cands.append(mod.fit(start_params=r2.params, disp=False, maxiter=1000))
            except Exception:  # noqa: BLE001
                pass
    if not cands:
        raise ValueError("ARIMA estimation failed")
    ok = [c for c in cands if _admissible(c)]
    best = max(ok or cands, key=lambda c: c.llf)
    admissible = bool(ok)
    par = np.asarray(best.params)
    try:
        best = mod.smooth(par, cov_type="approx")      # numerical-Hessian standard errors, as R arima
    except Exception:  # noqa: BLE001
        pass
    full = SARIMAX(y, exog=X, order=order, seasonal_order=so).smooth(par)
    cf = []
    if "intercept" in names:
        cf.append(np.ones(h))
    if "drift" in names:
        cf.append(np.arange(n + 1, n + h + 1, dtype=float))
    if X_user is not None:
        if exog_future is None:
            raise ValueError("Future values of the regressors are needed to forecast.")
        cf += list(np.asarray(exog_future, float).reshape(h, -1).T)
    Xf = np.column_stack(cf) if cf else None
    fc = full.get_forecast(h, exog=Xf)
    mean = np.asarray(fc.predicted_mean)
    e = np.asarray(best.resid)
    nstar = len(e)
    k = len(par)
    sig2_r = float(e @ e / (nstar - k + 1))
    scale = np.sqrt(sig2_r / par[-1])
    se = np.asarray(fc.se_mean) * scale
    pi = _gauss_pi(mean, se, levels)
    lab = f"ARIMA({order[0]},{order[1]},{order[2]})" + (f"({seasonal[0]},{seasonal[1]},{seasonal[2]})[{s}]" if so[3] else "")
    if "drift" in names:
        lab += " with drift"
    if X_user is not None:
        lab = "Regression with " + lab + " errors"
    pn = list(best.model.param_names)
    for j, nm in enumerate(names):
        pn[j] = nm
    tab = pd.DataFrame({"Parameter": pn, "Coef.": par, "Std. Err.": np.asarray(best.bse), "z": np.asarray(best.tvalues),
                        "p-value": np.asarray(best.pvalues)})
    fitted = np.asarray(full.fittedvalues)
    nb = d + D * s
    ll = float(best.llf)
    aic = -2 * ll + 2 * k
    return Fc(lab, mean, pi, np.r_[[np.nan] * nb, fitted[nb:]], np.r_[[np.nan] * nb, (y - fitted)[nb:]],
              {"table": tab, "AICc": aic + 2 * k * (k + 1) / max(nstar - k - 1, 1), "AIC": aic, "BIC": -2 * ll + np.log(nstar) * k,
               "loglik": ll, "order": tuple(order), "seasonal": tuple(seasonal), "trend": trend, "sigma2": sig2_r, "res": best,
               "admissible": admissible})


def _admissible(res, tol: float = 1.001) -> bool:
    """R auto.arima rule: reject fits with AR or MA roots (incl. seasonal) on or inside |z| = 1.001."""
    try:
        for r in (np.asarray(res.arroots), np.asarray(res.maroots)):
            if len(r) and np.min(np.abs(r)) < tol:
                return False
    except Exception:  # noqa: BLE001
        return True
    return True


def arima_auto(y, h: int, s: int = 1, exog=None, exog_future=None, max_p: int = 3, max_q: int = 3, levels=LEVELS) -> Fc:
    """Hyndman-Khandakar-style selection: d by repeated KPSS (α = 5%), D by the seasonal-strength rule
    (STL seasonal strength > 0.64, as forecast::nsdiffs), then (p, q)(P, Q) minimising AICc."""
    y = np.asarray(y, float)
    D = 0
    if s > 1 and len(y) >= 3 * s:
        D = int(seasonal_strength(y, s) > 0.64)
    x = np.diff(y, s) if D else y
    d = ndiffs(x)
    best = None
    seas = [(P, D, Q) for P in range(2) for Q in range(2)] if s > 1 else [(0, 0, 0)]
    trends = ["t", "n"] if d + D == 1 else (["c", "n"] if d + D == 0 else ["n"])
    for p, q in product(range(max_p + 1), range(max_q + 1)):
        for sea in seas:
            for tr in trends:
                try:
                    f = sarimax(y, h, (p, d, q), sea, s, exog, exog_future, trend=tr, levels=levels, polish=False)
                except Exception:  # noqa: BLE001
                    continue
                if not f.info["admissible"]:
                    continue
                if best is None or f.info["AICc"] < best.info["AICc"]:
                    best = f
    if best is None:
        raise ValueError("No ARIMA model could be estimated")
    best = sarimax(y, h, best.info["order"], best.info["seasonal"], s, exog, exog_future, trend=best.info["trend"], levels=levels)
    best.info["d"], best.info["D"] = d, D
    return best


def ndiffs(y, alpha: float = 0.05, max_d: int = 2) -> int:
    from statsmodels.tsa.stattools import kpss
    x = np.asarray(y, float)
    d = 0
    while d < max_d:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = kpss(x, "c", nlags="legacy")[1]
        except Exception:  # noqa: BLE001
            break
        if p >= alpha:
            break
        x = np.diff(x)
        d += 1
    return d


def seasonal_strength(y, s: int) -> float:
    from statsmodels.tsa.seasonal import STL
    r = STL(np.asarray(y, float), period=s, robust=True).fit()
    rem, sea = np.asarray(r.resid), np.asarray(r.seasonal)
    return float(max(0.0, 1 - np.var(rem) / np.var(rem + sea)))


# ==========================================================
# §3.1 Bayesian structural time series (Gibbs sampler with the Durbin-Koopman simulation smoother)
# ==========================================================
def bsts(y, h: int, level: str = "lltrend", s: int = 1, exog=None, exog_future=None, draws: int = 1000,
         burn: int = 300, seed: int = 1, levels=LEVELS) -> Fc:
    """Scott & Varian (2014) structural model: local level / local linear trend (+ seasonal, + regression with a
    Gaussian prior on β treated as constant states). Variances get inverse-gamma priors SdPrior(σ_guess, 0.01)
    (σ_guess = sd(y) for the observation, 0.01·sd(y) for states, as bsts). Returns posterior-predictive paths."""
    from statsmodels.tsa.statespace.structural import UnobservedComponents
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    n = len(y)
    kw = {"seasonal": s, "stochastic_seasonal": True} if s > 1 else {}
    X = None if exog is None else np.asarray(exog, float).reshape(n, -1)
    mod = UnobservedComponents(y, level=level, exog=X, mle_regression=False, **kw)
    names = mod.param_names
    sdy = np.std(y, ddof=1)
    nu0 = 0.01
    guess = {nm: (sdy if nm == "sigma2.irregular" else 0.01 * sdy) for nm in names}
    params = np.array([max(guess[nm] ** 2, 1e-8) for nm in names])
    sim = mod.simulation_smoother()
    keep = []
    st_end = []
    # map each variance to its state(s) through the selection matrix
    ssm = mod.ssm
    sel = np.asarray(ssm["selection"])
    if sel.ndim == 3:
        sel = sel[:, :, 0]
    var_names = [nm for nm in names if nm != "sigma2.irregular"]
    Z = np.asarray(ssm["design"])
    for it in range(draws + burn):
        mod.update(params)
        sim.simulate()
        a = sim.simulated_state                       # k_states × n
        eta = sim.simulated_state_disturbance         # k_posdef × n
        eps = sim.simulated_measurement_disturbance   # 1 × n
        newp = params.copy()
        ss = float(np.sum(eps ** 2))
        g = guess["sigma2.irregular"]
        newp[names.index("sigma2.irregular")] = 1 / rng.gamma((nu0 + n) / 2, 2 / (nu0 * g ** 2 + ss))
        for j, nm in enumerate(var_names):
            e = eta[j, :-1]
            g = guess[nm]
            newp[names.index(nm)] = 1 / rng.gamma((nu0 + len(e)) / 2, 2 / (nu0 * g ** 2 + float(e @ e)))
        params = newp
        if it >= burn:
            keep.append(params.copy())
            st_end.append(a[:, -1].copy())
    keep = np.array(keep)
    # posterior-predictive simulation
    T = np.asarray(ssm["transition"])[:, :, 0] if np.asarray(ssm["transition"]).ndim == 3 else np.asarray(ssm["transition"])
    k = T.shape[0]
    Zc = Z[:, :, -1] if Z.ndim == 3 else Z
    Xf = None if exog_future is None else np.asarray(exog_future, float).reshape(h, -1)
    kx = 0 if X is None else X.shape[1]
    paths = np.empty((len(keep), h))
    for i, (p, a0) in enumerate(zip(keep, st_end)):
        Q = np.diag([p[names.index(nm)] for nm in var_names])
        a = a0.copy()
        for t in range(h):
            a = T @ a + sel @ rng.multivariate_normal(np.zeros(len(var_names)), Q) if var_names else T @ a
            z = Zc[0].copy()
            if kx:
                z[k - kx:] = Xf[t]
            paths[i, t] = z @ a + rng.normal(0, np.sqrt(p[names.index("sigma2.irregular")]))
    mean = paths.mean(0)
    pi = {lv: (np.quantile(paths, 0.5 - lv / 200, 0), np.quantile(paths, 0.5 + lv / 200, 0)) for lv in levels}
    post = pd.DataFrame(keep, columns=names)
    summ = pd.DataFrame({"Parameter": names, "Posterior mean": post.mean().to_numpy(), "Posterior sd": post.std().to_numpy(),
                         "2.5%": post.quantile(0.025).to_numpy(), "97.5%": post.quantile(0.975).to_numpy()})
    mod.update(post.mean().to_numpy())
    sm = mod.smooth(post.mean().to_numpy())
    fitted = np.asarray(sm.fittedvalues)
    return Fc("BSTS (" + UC_LEVELS.get(level, (level,))[0].lower() + (" + seasonal" if s > 1 else "") +
              (" + regression" if kx else "") + ")", mean, pi, fitted, y - fitted, {"table": summ, "draws": post}, sims=paths)


# ==========================================================
# §3.1 Trend and growth curves
# ==========================================================
CURVES = {"linear": ("Linear trend", "اتجاه خطي"), "quadratic": ("Quadratic trend", "اتجاه تربيعي"),
          "exponential": ("Exponential growth", "نمو أسي"), "logistic": ("Logistic growth", "نمو لوجستي"),
          "gompertz": ("Gompertz growth", "نمو جومبرتز")}


def _curve_f(kind):
    return {"linear": lambda t, a, b: a + b * t,
            "quadratic": lambda t, a, b, c: a + b * t + c * t ** 2,
            "exponential": lambda t, a, b: a * np.exp(b * t),
            "logistic": lambda t, K, xmid, scal: K / (1 + np.exp((xmid - t) / scal)),
            "gompertz": lambda t, K, b2, b3: K * np.exp(-b2 * b3 ** t)}[kind]


def growth_curve(y, h: int, kind: str = "logistic", t0: float = 1.0, levels=LEVELS) -> Fc:
    """Least-squares trend / growth curve on t = t0, t0 + 1, … (R nls parameterisations: SSlogis K / (1 + exp((xmid −
    t) / scal)); SSgompertz K·exp(−b2·b3^t)); prediction intervals = delta-method curve variance + σ²."""
    y = np.asarray(y, float)
    n = len(y)
    t = t0 + np.arange(n)
    f = _curve_f(kind)
    if kind == "linear":
        p0 = np.polyfit(t, y, 1)[::-1]
    elif kind == "quadratic":
        p0 = np.polyfit(t, y, 2)[::-1]
    elif kind == "exponential":
        b = np.polyfit(t, np.log(np.maximum(y, 1e-9)), 1)
        p0 = [np.exp(b[1]), b[0]]
    elif kind == "logistic":
        K = y.max() * 1.1
        z = np.log(np.clip(y / K, 1e-6, 1 - 1e-6) / (1 - np.clip(y / K, 1e-6, 1 - 1e-6)))
        bb = np.polyfit(t, z, 1)
        p0 = [K, -bb[1] / bb[0], 1 / bb[0]]
    else:
        K = y.max() * 1.1
        z = np.log(-np.log(np.clip(y / K, 1e-6, 1 - 1e-6)))
        bb = np.polyfit(t, z, 1)
        p0 = [K, np.exp(bb[1]), np.exp(bb[0])]
    best = None
    for start in (p0, [p * 1.2 for p in p0], [p * 0.8 for p in p0]):
        try:
            r = optimize.least_squares(lambda p: f(t, *p) - y, start, method="lm", max_nfev=20000, x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
        except Exception:  # noqa: BLE001
            continue
    p = best.x
    J = best.jac
    k = len(p)
    rss = float(np.sum(best.fun ** 2))
    s2 = rss / (n - k)
    V = s2 * np.linalg.pinv(J.T @ J)
    tf = t0 + n + np.arange(h)
    mean = f(tf, *p)
    eps = 1e-6 * np.maximum(1, np.abs(p))
    G = np.column_stack([(f(tf, *(p + np.eye(k)[j] * eps[j])) - mean) / eps[j] for j in range(k)])
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, V, G) + s2)
    names = {"linear": ["a", "b"], "quadratic": ["a", "b", "c"], "exponential": ["a", "b"], "logistic": ["K (asymptote)", "xmid", "scal"],
             "gompertz": ["K (asymptote)", "b2", "b3"]}[kind]
    tab = pd.DataFrame({"Parameter": names, "Estimate": p, "Std. Err.": np.sqrt(np.diag(V))})
    tab["t"] = tab["Estimate"] / tab["Std. Err."]
    tab["p-value"] = 2 * stats.t.sf(np.abs(tab["t"]), n - k)
    fit = f(t, *p)
    ll = -n / 2 * (np.log(2 * np.pi * rss / n) + 1)
    return Fc(CURVES[kind][0], mean, _gauss_pi(mean, se, levels, df=n - k), fit, y - fit,
              {"table": tab, "sigma": np.sqrt(s2), "AIC": -2 * ll + 2 * (k + 1), "BIC": -2 * ll + np.log(n) * (k + 1),
               "R²": 1 - rss / np.sum((y - y.mean()) ** 2), "params": p})


# ==========================================================
# §3.3 Volatility: HAR-RV (Corsi 2009) and VaR forecasts
# ==========================================================
def har_design(rv, h: int = 1, log: bool = False):
    x = np.log(rv) if log else np.asarray(rv, float)
    s = pd.Series(x)
    d, w, m = s, s.rolling(5).mean(), s.rolling(22).mean()
    target = s[::-1].rolling(h).mean()[::-1].shift(-1)       # average of t+1 … t+h
    X = pd.DataFrame({"const": 1.0, "RV_d": d, "RV_w": w, "RV_m": m})
    return X, target


def har(rv, h: int = 1, log: bool = False, horizon: int = 22, nw_lags: int | None = None) -> dict:
    """HAR-RV: RV(t+1…t+h) = b0 + b_d RV_t + b_w RV_t^(w) + b_m RV_t^(m) + e, OLS with Newey-West SEs (lag = max(5, 2h)).
    Iterated daily forecasts for `horizon` days; log-HAR uses the log-normal mean correction exp(σ²/2)."""
    import statsmodels.api as sm
    X, yv = har_design(rv, h, log)
    ok = X.notna().all(1) & yv.notna()
    m = sm.OLS(yv[ok], X[ok]).fit(cov_type="HAC", cov_kwds={"maxlags": nw_lags or max(5, 2 * h)})
    tab = pd.DataFrame({"Parameter": X.columns, "Coef.": m.params.values, "Std. Err. (NW)": m.bse.values, "t": m.tvalues.values,
                        "p-value": m.pvalues.values})
    # iterated 1-day forecasts
    x = list(np.log(rv) if log else np.asarray(rv, float))
    b = m.params.values
    m1 = m if h == 1 else har(rv, 1, log, 1)["model"]
    b1 = m1.params.values
    s2 = m1.mse_resid
    out = []
    for _ in range(horizon):
        f = b1 @ [1, x[-1], np.mean(x[-5:]), np.mean(x[-22:])]
        x.append(f)
        out.append(np.exp(f + s2 / 2) if log else f)
    fit = m.fittedvalues
    return {"table": tab, "model": m, "R²": float(m.rsquared), "forecast": np.array(out), "fitted": fit, "b": b,
            "persistence": float(b[1:].sum())}


def mincer_zarnowitz(actual, fc) -> dict:
    import statsmodels.api as sm
    r = sm.OLS(np.asarray(actual, float), sm.add_constant(np.asarray(fc, float))).fit()
    w = r.f_test("const = 0, x1 = 1")
    return {"a": float(r.params[0]), "b": float(r.params[1]), "R²": float(r.rsquared), "F": float(w.fvalue), "p": float(w.pvalue)}


def qlike(actual, fc):
    a, f = np.asarray(actual, float), np.asarray(fc, float)
    return float(np.mean(a / f - np.log(a / f) - 1))


# ==========================================================
# §3.4 Machine-learning forecasting (direct / recursive with lag features; conformal intervals)
# ==========================================================
ML_MODELS = {"rf": ("Random forest", "الغابات العشوائية"), "gbm": ("Gradient boosting (histogram, LightGBM-style)", "تعزيز التدرج (هيستوجرام، بأسلوب LightGBM)"),
             "lgbm": ("LightGBM", "LightGBM"), "xgb": ("XGBoost", "XGBoost"), "cat": ("CatBoost", "CatBoost"),
             "svr": ("Support vector regression", "انحدار المتجهات الداعمة"), "ridge": ("Ridge (linear benchmark)", "انحدار ريدج (مرجع خطي)")}


def ml_available() -> dict:
    out = {"rf": True, "gbm": True, "svr": True, "ridge": True}
    for k, mod in (("lgbm", "lightgbm"), ("xgb", "xgboost"), ("cat", "catboost")):
        try:
            __import__(mod)
            out[k] = True
        except Exception:  # noqa: BLE001
            out[k] = False
    try:
        __import__("sklearn")
    except Exception:  # noqa: BLE001
        return {k: False for k in out}
    return out


def _ml_estimator(kind: str, seed: int = 1):
    from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    if kind == "rf":
        return RandomForestRegressor(n_estimators=150, min_samples_leaf=2, max_features=0.6, random_state=seed, n_jobs=-1)
    if kind == "gbm":
        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=5, random_state=seed)
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=5, subsample=0.9,
                                 subsample_freq=1, colsample_bytree=0.9, random_state=seed, verbose=-1)
    if kind == "xgb":
        import xgboost as xgb
        return xgb.XGBRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=seed)
    if kind == "cat":
        import catboost as cb
        return cb.CatBoostRegressor(iterations=500, learning_rate=0.05, depth=4, random_seed=seed, verbose=0, allow_writing_files=False)
    if kind == "svr":
        return make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05, gamma="scale"))
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 25)))


def _features(z: np.ndarray, t_index: np.ndarray, lags: int, s: int, period: np.ndarray, exog=None):
    """Row t uses information up to t-1: lags 1..p, seasonal lag s, rolling means, trend and Fourier seasonality."""
    cols = {f"lag{j}": np.r_[[np.nan] * j, z[:-j]] for j in range(1, lags + 1)}
    if s > 1 and s > lags:
        cols[f"lag{s}"] = np.r_[[np.nan] * s, z[:-s]]
    zs = pd.Series(z).shift(1)
    cols["roll_mean_3"] = zs.rolling(3).mean().to_numpy()
    if s > 1:
        cols[f"roll_mean_{s}"] = zs.rolling(s).mean().to_numpy()
    cols["trend"] = t_index.astype(float)
    if s > 1:
        for k in range(1, min(3, s // 2) + 1):
            cols[f"sin{k}"] = np.sin(2 * np.pi * k * period / s)
            cols[f"cos{k}"] = np.cos(2 * np.pi * k * period / s)
    F = pd.DataFrame(cols)
    if exog is not None:
        E = pd.DataFrame(np.asarray(exog, float).reshape(len(z), -1))
        E.columns = [f"x{j + 1}" for j in range(E.shape[1])]
        F = pd.concat([F, E], axis=1)
    return F


def ml_forecast(y, h: int, kind: str = "gbm", s: int = 1, lags: int | None = None, difference: bool | None = None,
                exog=None, exog_future=None, calib: int | None = None, seed: int = 1, levels=LEVELS) -> Fc:
    """Recursive one-step model on lag features. Trees cannot extrapolate a trend, so by default the model is fitted
    to first differences when the series is non-stationary (KPSS) and forecasts are re-integrated. Intervals are
    split-conformal: empirical quantiles of h-step errors from a rolling-origin calibration on the last `calib` points."""
    y = np.asarray(y, float)
    n = len(y)
    lags = lags or (max(2, min(s, 12)) if s > 1 else 6)
    if difference is None:
        difference = ndiffs(y, max_d=1) > 0
    exog_a = None if exog is None else np.asarray(exog, float).reshape(n, -1)

    def fit_predict(yy, hh, ex=None, exf=None):
        z = np.diff(yy) if difference else yy.copy()
        off = 1 if difference else 0
        nn = len(z)
        per = (np.arange(off, off + nn)) % max(s, 1)
        F = _features(z, np.arange(nn), lags, s, per, None if ex is None else ex[off:])
        ok = F.notna().all(1).to_numpy()
        est = _ml_estimator(kind, seed)
        est.fit(F[ok].to_numpy(), z[ok])
        zz = list(z)
        exx = None if ex is None else np.vstack([ex[off:], exf])
        for j in range(hh):
            m = len(zz)
            zz_a = np.r_[zz, 0.0]
            F2 = _features(zz_a, np.arange(m + 1), lags, s, np.arange(off, off + m + 1) % max(s, 1),
                           None if exx is None else exx[:m + 1])
            zz.append(float(est.predict(F2.iloc[[-1]].to_numpy())[0]))
        path = np.array(zz[nn:])
        if difference:
            path = yy[-1] + np.cumsum(path)
        return path, est, F[ok], z[ok]

    mean, est, Ftr, ztr = fit_predict(y, h, exog_a, None if exog_future is None else np.asarray(exog_future, float).reshape(h, -1))
    fitted_z = est.predict(Ftr.to_numpy())
    calib = min(max(2 * h, 24), n // 4) if calib is None else calib
    errs = []
    for o in range(n - calib, n - 1) if calib else ():
        hh = min(h, n - o)
        try:
            p, *_ = fit_predict(y[:o], hh, None if exog_a is None else exog_a[:o], None if exog_a is None else exog_a[o:o + hh])
            e = np.full(h, np.nan)
            e[:hh] = y[o:o + hh] - p
            errs.append(e)
        except Exception:  # noqa: BLE001
            continue
    E = np.array(errs) if errs else np.full((0, h), np.nan)
    pi = {}
    for lv in levels if len(E) >= 5 else ():
        q = np.array([np.nanquantile(np.abs(E[:, j]), lv / 100) if np.isfinite(E[:, j]).sum() >= 5 else np.nan for j in range(h)])
        q = pd.Series(q).ffill().to_numpy()
        pi[lv] = (mean - q, mean + q)
    imp = None
    try:
        from sklearn.inspection import permutation_importance
        pimp = permutation_importance(est, Ftr.to_numpy(), ztr, n_repeats=5, random_state=seed)
        imp = pd.DataFrame({"Feature": Ftr.columns, "Importance": pimp.importances_mean}).sort_values("Importance", ascending=False)
    except Exception:  # noqa: BLE001
        pass
    off = 1 if difference else 0
    fitted = np.full(n, np.nan)
    idx = np.where(np.r_[[False] * off, Ftr.index.isin(Ftr.index)])[0]
    if difference:
        fitted[Ftr.index.to_numpy() + 1] = y[Ftr.index.to_numpy()] + fitted_z
    else:
        fitted[Ftr.index.to_numpy()] = fitted_z
    del idx
    return Fc(ML_MODELS[kind][0], mean, pi, fitted, y - fitted, {"importance": imp, "lags": lags, "differenced": difference,
                                                                  "calibration errors": E})


# ==========================================================
# §3.4 Prophet-style decomposable model (Taylor & Letham 2018)
# ==========================================================
def prophet_style(y, h: int, s: int = 12, n_changepoints: int = 25, cp_range: float = 0.8, cp_scale: float = 0.05,
                  fourier: int | None = None, multiplicative: bool = False, sims: int = 500, seed: int = 1, levels=LEVELS) -> Fc:
    """Piecewise-linear trend with potential changepoints in the first 80% of the sample and a Laplace(0, τ) prior on
    the rate changes (MAP = L1-penalised least squares), plus Fourier seasonality. Prediction intervals simulate
    future trend changes (Poisson-rate changepoints with Laplace magnitudes, as Prophet) plus observation noise.
    The official `prophet` package is used instead when installed."""
    rng = np.random.default_rng(seed)
    y0 = np.asarray(y, float)
    yv = np.log(y0) if multiplicative else y0
    n = len(yv)
    scale = np.max(np.abs(yv))
    z = yv / scale
    t = np.arange(n) / (n - 1)
    ncp = min(n_changepoints, max(0, int(cp_range * n) - 1))
    cps = np.linspace(0, cp_range, ncp + 2)[1:-1] if ncp else np.array([])
    A = (t[:, None] >= cps[None, :]).astype(float)
    K = fourier if fourier is not None else (min(10, s // 2) if s > 1 else 0)
    per = np.arange(n)

    def season_mat(idx):
        if not K:
            return np.zeros((len(idx), 0))
        return np.column_stack([f(2 * np.pi * k * idx / s) for k in range(1, K + 1) for f in (np.sin, np.cos)])
    S = season_mat(per)
    X_trend = np.column_stack([np.ones(n), t, A * (t[:, None] - cps[None, :])])
    X = np.column_stack([X_trend, S])
    p_trend = X_trend.shape[1]

    def obj(b):
        r = z - X @ b
        lam = 1 / cp_scale
        return 0.5 * r @ r / np.var(z) + lam * np.sum(np.abs(b[2:p_trend])) + 0.5 * np.sum(b[p_trend:] ** 2) / 10.0 ** 2

    b0 = np.linalg.lstsq(X, z, rcond=None)[0]
    b0[2:p_trend] = 0
    res = optimize.minimize(obj, b0, method="Powell", options={"maxiter": 200000, "xtol": 1e-8, "ftol": 1e-12})
    b = res.x
    fit = X @ b
    sigma = np.std(z - fit, ddof=X.shape[1] - np.sum(np.abs(b[2:p_trend]) < 1e-6))
    tf = (n - 1 + np.arange(1, h + 1)) / (n - 1)
    k_end = b[1] + np.sum(b[2:p_trend])
    m_end = fit[-1] - (S[-1] @ b[p_trend:] if K else 0)
    Sf = season_mat(np.arange(n, n + h))
    trend_f = m_end + k_end * (tf - 1)
    mean_z = trend_f + (Sf @ b[p_trend:] if K else 0)
    deltas = b[2:p_trend]
    lam_l = np.mean(np.abs(deltas)) + 1e-8 if ncp else 1e-8
    rate = ncp / (cp_range if ncp else 1)
    paths = np.empty((sims, h))
    for i in range(sims):
        k = k_end
        tr_ = np.empty(h)
        last = m_end
        prev_t = 1.0
        for j in range(h):
            dt = tf[j] - prev_t
            if rng.random() < rate * dt:
                k += rng.laplace(0, lam_l)
            last = last + k * dt
            prev_t = tf[j]
            tr_[j] = last
        paths[i] = tr_ + (Sf @ b[p_trend:] if K else 0) + rng.normal(0, sigma, h)
    paths *= scale
    mean = mean_z * scale
    fitted = fit * scale
    if multiplicative:
        paths, mean, fitted = np.exp(paths), np.exp(mean), np.exp(fitted)
    pi = {lv: (np.quantile(paths, 0.5 - lv / 200, 0), np.quantile(paths, 0.5 + lv / 200, 0)) for lv in levels}
    cp_tab = pd.DataFrame({"changepoint (share of sample)": cps, "rate change": deltas * scale}) if ncp else pd.DataFrame()
    return Fc("Prophet-style (piecewise trend + Fourier)", mean, pi, fitted, y0 - fitted,
              {"changepoints": cp_tab, "trend": (X_trend @ b[:p_trend]) * scale, "fourier order": K}, sims=paths)


def prophet_available() -> bool:
    try:
        __import__("prophet")
        return True
    except Exception:  # noqa: BLE001
        return False


def prophet_official(dates, y, h: int, s: int = 12, levels=LEVELS, samples: int = 500) -> Fc:
    """Meta's Prophet (Taylor & Letham 2018) via the `prophet` package: piecewise-linear trend, yearly / weekly
    seasonality chosen from the frequency, intervals and fan from its posterior-predictive samples."""
    import logging
    for lg in ("cmdstanpy", "prophet"):
        logging.getLogger(lg).setLevel(logging.ERROR)
    from prophet import Prophet
    y = np.asarray(y, float)
    if dates is None:
        freq0 = {12: "MS", 4: "QS", 52: "W", 7: "D", 5: "B", 1: "YS"}.get(s, "MS")
        dates = pd.date_range("2000-01-01", periods=len(y), freq=freq0)
    ds_ = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    freq = pd.infer_freq(ds_) or {12: "MS", 4: "QS", 52: "W", 7: "D", 5: "B", 1: "YS"}.get(s, "MS")
    m = Prophet(interval_width=0.95, uncertainty_samples=samples, yearly_seasonality=s in (4, 12, 52, 7, 5),
                weekly_seasonality=s in (7, 5), daily_seasonality=False)
    m.fit(pd.DataFrame({"ds": ds_, "y": y}))
    fut = m.make_future_dataframe(h, freq=freq, include_history=False)
    p = m.predict(fut)
    sims = m.predictive_samples(fut)["yhat"].T
    mean = p["yhat"].to_numpy()
    pi = {lv: (np.quantile(sims, 0.5 - lv / 200, 0), np.quantile(sims, 0.5 + lv / 200, 0)) for lv in levels}
    fit = m.predict(pd.DataFrame({"ds": ds_}))["yhat"].to_numpy()
    return Fc("Prophet (Meta)", mean, pi, fit, y - fit, {"changepoints": pd.DataFrame({"changepoint": m.changepoints.to_numpy()})}, sims=sims)


# ==========================================================
# §3.6 Accuracy, Diebold-Mariano, rolling-origin evaluation
# ==========================================================
def accuracy(actual, fc, train=None, s: int = 1) -> dict:
    """forecast::accuracy definitions: ME, RMSE, MAE, MPE, MAPE, sMAPE (as in Hyndman), MASE (scaled by the in-sample
    seasonal-naive MAE), RMSSE, Theil's U (U2 of forecast::accuracy) and U1."""
    a, f = np.asarray(actual, float), np.asarray(fc, float)
    e = a - f
    out = {"ME": e.mean(), "RMSE": np.sqrt(np.mean(e ** 2)), "MAE": np.mean(np.abs(e)),
           "MPE": 100 * np.mean(e / a) if np.all(a != 0) else np.nan,
           "MAPE": 100 * np.mean(np.abs(e / a)) if np.all(a != 0) else np.nan,
           "sMAPE": 200 * np.mean(np.abs(e) / (np.abs(a) + np.abs(f)))}
    if train is not None:
        tr = np.asarray(train, float)
        m = s if s > 1 and len(tr) > s else 1
        sc = np.mean(np.abs(tr[m:] - tr[:-m]))
        sc2 = np.mean((tr[m:] - tr[:-m]) ** 2)
        out["MASE"] = out["MAE"] / sc
        out["RMSSE"] = out["RMSE"] / np.sqrt(sc2)
        prev = np.r_[tr[-1], a[:-1]]
    else:
        prev = np.r_[np.nan, a[:-1]]
    with np.errstate(divide="ignore", invalid="ignore"):
        fpe = (f[1:] - a[1:]) / a[:-1]
        ape = (a[1:] - a[:-1]) / a[:-1]
        out["Theil's U"] = float(np.sqrt(np.sum(fpe ** 2) / np.sum(ape ** 2))) if len(a) > 1 else np.nan
    out["Theil U1"] = float(np.sqrt(np.mean(e ** 2)) / (np.sqrt(np.mean(a ** 2)) + np.sqrt(np.mean(f ** 2))))
    del prev
    return {k: float(v) for k, v in out.items()}


def dm_test(e1, e2, h: int = 1, power: int = 2, alternative: str = "two.sided", varestimator: str = "acf") -> dict:
    """forecast::dm.test: d = |e1|^p − |e2|^p; long-run variance from autocovariances up to h−1 ('acf': rectangular,
    falls back to Bartlett if negative; 'bartlett'); Harvey-Leybourne-Newbold small-sample correction; t(n−1)."""
    e1, e2 = np.asarray(e1, float), np.asarray(e2, float)
    d = np.abs(e1) ** power - np.abs(e2) ** power
    n = len(d)
    dc = d - d.mean()
    gam = np.array([dc[k:] @ dc[:n - k] / n for k in range(h)])
    if varestimator == "bartlett":
        w = 1 - np.arange(h) / h
        lrv = gam[0] + 2 * np.sum(w[1:] * gam[1:])
    else:
        lrv = gam[0] + 2 * np.sum(gam[1:])
        if lrv <= 0:
            w = 1 - np.arange(h) / h
            lrv = gam[0] + 2 * np.sum(w[1:] * gam[1:])
    stat = d.mean() / np.sqrt(lrv / n)
    k = ((n + 1 - 2 * h + (h / n) * (h - 1)) / n) ** 0.5
    stat *= k
    if alternative == "two.sided":
        p = 2 * stats.t.cdf(-abs(stat), n - 1)
    elif alternative == "less":
        p = stats.t.cdf(stat, n - 1)
    else:
        p = stats.t.sf(stat, n - 1)
    return {"DM": float(stat), "p": float(p), "h": h, "n": n, "mean loss differential": float(d.mean())}


def rolling_origin(y, model, h: int = 1, initial: int | None = None, step: int = 1, window: int | None = None,
                   exog=None) -> np.ndarray:
    """Errors e[o, j] = y[o + j] − ŷ_{o+j|o−1} for forecast origins o = initial … n−1 (as forecast::tsCV, which puts
    e at row o−1 for origin t = o; here rows are the origins in the evaluation sample). `model(y_train, h, exog_train,
    exog_future)` must return an Fc. `window` gives a rolling instead of an expanding estimation window."""
    y = np.asarray(y, float)
    n = len(y)
    initial = initial or max(n // 2, 10)
    origins = list(range(initial, n, step))
    E = np.full((len(origins), h), np.nan)
    for i, o in enumerate(origins):
        lo = 0 if window is None else max(0, o - window)
        tr = y[lo:o]
        hh = min(h, n - o)
        try:
            ex_tr = None if exog is None else exog[lo:o]
            ex_f = None if exog is None else exog[o:o + h] if o + h <= n else None
            f = model(tr, h, ex_tr, ex_f) if exog is not None else model(tr, h)
            E[i, :hh] = y[o:o + hh] - f.mean[:hh]
        except Exception:  # noqa: BLE001
            continue
    return E


def backtest_summary(y, E, initial: int, s: int = 1, step: int = 1) -> pd.DataFrame:
    """Per-horizon RMSE, MAE, MAPE and MASE from a rolling-origin error matrix."""
    y = np.asarray(y, float)
    tr = y[:initial]
    m = s if s > 1 and len(tr) > s else 1
    sc = np.mean(np.abs(tr[m:] - tr[:-m]))
    rows = []
    for j in range(E.shape[1]):
        e = E[:, j]
        ok = np.isfinite(e)
        act = np.array([y[initial + i * step + j] if initial + i * step + j < len(y) else np.nan for i in range(E.shape[0])])
        rows.append({"h": j + 1, "N": int(ok.sum()), "RMSE": float(np.sqrt(np.mean(e[ok] ** 2))), "MAE": float(np.mean(np.abs(e[ok]))),
                     "MAPE": float(100 * np.mean(np.abs(e[ok] / act[ok]))) if np.all(act[ok] != 0) else np.nan,
                     "MASE": float(np.mean(np.abs(e[ok])) / sc)})
    return pd.DataFrame(rows)


# ==========================================================
# §3.2 Forecast combination
# ==========================================================
COMBOS = {"mean": ("Simple average", "المتوسط البسيط"), "median": ("Median", "الوسيط"), "trimmed": ("Trimmed mean (20%)", "المتوسط المشذّب (20%)"),
          "inv_mse": ("Inverse-MSE weights", "أوزان مقلوب MSE"), "bates_granger": ("Bates-Granger (discounted)", "بيتس-جرينجر (مخصوم)"),
          "granger_ramanathan": ("Granger-Ramanathan (constrained OLS)", "جرينجر-راماناثان (OLS مقيد)"),
          "aic": ("Akaike weights (Bayesian-style)", "أوزان أكايكي (بايزي التوجه)")}


def combination_weights(E: np.ndarray, method: str = "inv_mse", F_hist=None, actual_hist=None, ic=None, discount: float = 0.95) -> np.ndarray:
    """Weights from past errors E (T × M) of M models. Granger-Ramanathan uses past forecasts F_hist (T × M) and outcomes,
    with weights ≥ 0 summing to one (NNLS + normalisation). Akaike weights from information criteria (exp(−Δ/2))."""
    M = E.shape[1]
    if method in ("mean", "median", "trimmed"):
        return np.full(M, 1 / M)
    if method == "inv_mse":
        mse = np.nanmean(E ** 2, 0)
        w = 1 / mse
    elif method == "bates_granger":
        T = E.shape[0]
        disc = discount ** np.arange(T)[::-1]
        mse = np.nansum(disc[:, None] * E ** 2, 0) / np.sum(disc)
        w = 1 / mse
    elif method == "granger_ramanathan":
        ok = np.isfinite(F_hist).all(1) & np.isfinite(actual_hist)
        w, _ = optimize.nnls(F_hist[ok], actual_hist[ok])
        if w.sum() == 0:
            w = np.ones(M)
    elif method == "aic":
        a = np.asarray(ic, float)
        w = np.exp(-(a - np.nanmin(a)) / 2)
        w[~np.isfinite(w)] = 0
    else:
        raise ValueError(method)
    return w / w.sum()


def combine(fcs: list, method: str = "mean", weights=None, levels=LEVELS) -> Fc:
    P = np.array([f.mean for f in fcs])
    if method == "median":
        mean = np.median(P, 0)
    elif method == "trimmed":
        mean = stats.trim_mean(P, 0.2, axis=0)
    else:
        w = np.full(len(fcs), 1 / len(fcs)) if weights is None else np.asarray(weights)
        mean = w @ P
    # intervals: weighted average of component bounds (conservative, Hyndman fable-style approximation)
    w = np.full(len(fcs), 1 / len(fcs)) if weights is None or method in ("median", "trimmed") else np.asarray(weights)
    pi = {}
    for lv in levels:
        lo = np.array([f.pi[lv][0] for f in fcs if lv in f.pi])
        hi = np.array([f.pi[lv][1] for f in fcs if lv in f.pi])
        if len(lo) == len(fcs):
            pi[lv] = (mean + w @ (lo - P), mean + w @ (hi - P))
    return Fc(f"Combination — {COMBOS[method][0]}", mean, pi, info={"weights": dict(zip([f.name for f in fcs], w)), "method": method})


# ==========================================================
# §3.6 Scenario forecasting (dynamic regression with user paths for the drivers)
# ==========================================================
def scenario_paths(exog_hist: pd.DataFrame, h: int, scenarios: dict) -> dict:
    """scenarios = {name: {driver: ('growth', g% per period) | ('level', value) | ('shift', % vs baseline) | ('path', array)}}.
    Baseline drivers are projected with drift (random walk with drift, in logs when positive)."""
    base = {}
    for c in exog_hist.columns:
        x = exog_hist[c].to_numpy(float)
        if np.all(x > 0):
            base[c] = np.exp(drift(np.log(x), h).mean)
        else:
            base[c] = drift(x, h).mean
    out = {"Baseline": pd.DataFrame(base)}
    for name, spec in scenarios.items():
        df = pd.DataFrame(base).copy()
        for c, (kind, v) in spec.items():
            last = exog_hist[c].iloc[-1]
            if kind == "growth":
                df[c] = last * (1 + v / 100) ** np.arange(1, h + 1)
            elif kind == "level":
                df[c] = v
            elif kind == "shift":
                df[c] = df[c] * (1 + v / 100)
            elif kind == "path":
                df[c] = np.asarray(v, float)[:h]
        out[name] = df
    return out


def scenario_forecast(y, exog_hist: pd.DataFrame, paths: dict, h: int, order=(1, 1, 1), seasonal=(0, 0, 0), s: int = 1,
                      log: bool = False, levels=LEVELS) -> dict:
    yy = np.log(np.asarray(y, float)) if log else np.asarray(y, float)
    out = {}
    for name, X in paths.items():
        f = sarimax(yy, h, order, seasonal, s, exog_hist.to_numpy(float), X.to_numpy(float), levels=levels)
        if log:
            f.mean = np.exp(f.mean)
            f.pi = {lv: (np.exp(lo), np.exp(hi)) for lv, (lo, hi) in f.pi.items()}
        out[name] = f
    return out


# ==========================================================
# §3.6 Model horse-race
# ==========================================================
RACE = {"naive": ("Naive", "الساذج"), "snaive": ("Seasonal naive", "الساذج الموسمي"), "drift": ("Random walk with drift", "السير العشوائي بانجراف"),
        "mean": ("Mean", "المتوسط"), "ets": ("ETS (automatic)", "ETS (تلقائي)"), "arima": ("ARIMA (automatic)", "ARIMA (تلقائي)"),
        "theta": ("Theta", "ثيتا"), "ss": ("State space (local linear trend)", "الفضاء الحالتي (اتجاه خطي محلي)"),
        "ml_gbm": ("Gradient boosting", "تعزيز التدرج"), "ml_rf": ("Random forest", "الغابات العشوائية"),
        "prophet": ("Prophet-style", "على نمط Prophet"), "prophet_meta": ("Prophet (Meta)", "Prophet (ميتا)"),
        "ml_xgb": ("XGBoost", "XGBoost"), "ml_cat": ("CatBoost", "CatBoost"), "combo": ("Combination (mean of the top 3)", "التجميع (متوسط أفضل 3)")}


def race_keys(s: int) -> list:
    """Models available for the horse-race in this installation (optional packages only when installed)."""
    av = ml_available()
    out = []
    for m in RACE:
        if m == "combo" or (m == "snaive" and s < 2):
            continue
        if (m == "prophet_meta" and not prophet_available()) or (m == "ml_xgb" and not av.get("xgb")) or (m == "ml_cat" and not av.get("cat")):
            continue
        out.append(m)
    return out


def race_model(key: str, s: int, spec: dict | None = None, calib: int | None = None):
    """Factory: f(y, h) → Fc. ETS and ARIMA use the specification selected on the first training sample (`spec`),
    re-estimated at every origin; without `spec` they are selected automatically."""
    spec = spec or {}
    ets_f = (lambda y, h: ets(y, h, s, dict(spec["ets"], effort="quick"))) if "ets" in spec else (lambda y, h: ets(y, h, s))
    ar_f = ((lambda y, h: sarimax(y, h, spec["arima"]["order"], spec["arima"]["seasonal"], s, trend=spec["arima"]["trend"]))
            if "arima" in spec else (lambda y, h: arima_auto(y, h, s, max_p=2, max_q=2)))
    return {"naive": lambda y, h: naive(y, h), "snaive": lambda y, h: snaive(y, h, s), "drift": lambda y, h: drift(y, h),
            "mean": lambda y, h: mean_fc(y, h), "ets": ets_f, "arima": ar_f,
            "theta": lambda y, h: theta(y, h, s), "ss": lambda y, h: state_space(y, h, "lltrend", s),
            "ml_gbm": lambda y, h: ml_forecast(y, h, "gbm", s, calib=calib), "ml_rf": lambda y, h: ml_forecast(y, h, "rf", s, calib=calib),
            "prophet": lambda y, h: prophet_style(y, h, s, sims=200), "prophet_meta": lambda y, h: prophet_official(None, y, h, s, samples=200),
            "ml_xgb": lambda y, h: ml_forecast(y, h, "xgb", s, calib=calib), "ml_cat": lambda y, h: ml_forecast(y, h, "cat", s, calib=calib)}[key]


def horse_race(y, h: int, s: int = 1, models=None, origins: int = 8, step: int | None = None, metric: str = "MASE",
               progress=None) -> dict:
    """Rolling-origin competition: each model is re-estimated at `origins` forecast origins spread over the last part
    of the sample and evaluated on the next h observations; models are ranked by the average `metric` across all
    horizons; DM tests against the best model use the h = 1 errors; the final forecasts are re-estimated on the full
    sample. A combination of the top three models is added."""
    y = np.asarray(y, float)
    n = len(y)
    models = models or [m for m in race_keys(s) if m not in ("prophet_meta", "ml_xgb", "ml_cat")]
    step = step or max(1, h // 2)
    initial = n - h - (origins - 1) * step
    if initial < max(2 * s, 12):
        raise ValueError("The series is too short for this horizon and number of origins.")
    errs, fcs, rows = {}, {}, []
    spec = {}
    if "ets" in models:
        f0 = ets(y[:initial], 1, s)
        spec["ets"] = {k_: v for k_, v in f0.info["spec"].items() if k_ in ("error", "trend", "damped", "seasonal", "start")}
    if "arima" in models:
        f0 = arima_auto(y[:initial], 1, s, max_p=2, max_q=2)
        spec["arima"] = {"order": f0.info["order"], "seasonal": f0.info["seasonal"], "trend": f0.info["trend"]}
    for i, k in enumerate(models):
        if progress:
            progress(i / (len(models) + 1), RACE[k][0])
        E = rolling_origin(y[:n], race_model(k, s, spec, calib=0), h, initial, step)
        E = E[:origins]
        errs[k] = E
        tr = y[:initial]
        m = s if s > 1 and len(tr) > s else 1
        sc = np.mean(np.abs(tr[m:] - tr[:-m]))
        sc2 = np.mean((tr[m:] - tr[:-m]) ** 2)
        acts = np.array([[y[initial + o * step + j] for j in range(h)] for o in range(E.shape[0])])
        ok = np.isfinite(E)
        rows.append({"key": k, "Model": RACE[k][0], "RMSE": float(np.sqrt(np.mean(E[ok] ** 2))), "MAE": float(np.mean(np.abs(E[ok]))),
                     "MAPE": float(100 * np.mean(np.abs(E[ok] / acts[ok]))) if np.all(acts[ok] != 0) else np.nan,
                     "MASE": float(np.mean(np.abs(E[ok])) / sc), "RMSSE": float(np.sqrt(np.mean(E[ok] ** 2) / sc2)),
                     "failed origins": int((~np.isfinite(E[:, 0])).sum())})
    tab = pd.DataFrame(rows).sort_values(metric).reset_index(drop=True)
    top = tab["key"].tolist()[:3]
    Ec = np.nanmean(np.array([errs[k] for k in top]), 0)
    tr = y[:initial]
    m = s if s > 1 and len(tr) > s else 1
    sc = np.mean(np.abs(tr[m:] - tr[:-m]))
    sc2 = np.mean((tr[m:] - tr[:-m]) ** 2)
    acts = np.array([[y[initial + o * step + j] for j in range(h)] for o in range(Ec.shape[0])])
    errs["combo"] = Ec
    tab = pd.concat([tab, pd.DataFrame([{"key": "combo", "Model": RACE["combo"][0] + ": " + " + ".join(RACE[k][0] for k in top),
                                          "RMSE": float(np.sqrt(np.mean(Ec ** 2))), "MAE": float(np.mean(np.abs(Ec))),
                                          "MAPE": float(100 * np.mean(np.abs(Ec / acts))) if np.all(acts != 0) else np.nan,
                                          "MASE": float(np.mean(np.abs(Ec)) / sc), "RMSSE": float(np.sqrt(np.mean(Ec ** 2) / sc2)),
                                          "failed origins": 0}])]).sort_values(metric).reset_index(drop=True)
    tab.insert(0, "Rank", np.arange(1, len(tab) + 1))
    best = tab["key"].iloc[0]
    dm = []
    for k in tab["key"]:
        if k == best:
            continue
        e1, e2 = errs[k][:, 0], errs[best][:, 0]
        ok = np.isfinite(e1) & np.isfinite(e2)
        if ok.sum() >= 4:
            dm.append({"Model": tab.set_index("key").loc[k, "Model"], **dm_test(e1[ok], e2[ok], 1)})
    for i, k in enumerate(top + ([best] if best not in top and best != "combo" else [])):
        if progress:
            progress((len(models) + i / 4) / (len(models) + 1), RACE[k][0])
        fcs[k] = race_model(k, s)(y, h)
        if k in spec:
            fcs[k].info["backtest spec"] = spec[k]
    fcs["combo"] = combine([fcs[k] for k in top], "mean")
    fcs["combo"].name = RACE["combo"][0]
    if progress:
        progress(1.0, "")
    return {"table": tab, "errors": errs, "forecasts": fcs, "best": best, "top3": top, "dm": pd.DataFrame(dm),
            "initial": initial, "step": step, "origins": origins, "spec": spec}


# ==========================================================
# §3.3 Out-of-sample volatility and VaR forecasting
# ==========================================================
def coverage_tests(hits, level: float) -> dict:
    """Kupiec (1995) unconditional coverage, Christoffersen (1998) independence and conditional coverage."""
    hits = np.asarray(hits, int)
    n, x = len(hits), int(hits.sum())
    pi = x / n

    def ll(p, k0, k1):
        return (k0 * np.log(1 - p) if k0 and p < 1 else 0.0) + (k1 * np.log(p) if k1 and p > 0 else 0.0)
    lr_uc = -2 * (ll(level, n - x, x) - ll(pi, n - x, x))
    a, b = hits[:-1], hits[1:]
    n00, n01, n10, n11 = [int(np.sum((a == i) & (b == j))) for i, j in ((0, 0), (0, 1), (1, 0), (1, 1))]
    p01, p11 = n01 / max(n00 + n01, 1), n11 / max(n10 + n11, 1)
    p_ = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    lr_ind = -2 * (ll(p_, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return {"violations": x, "expected": level * n, "rate %": 100 * pi, "Kupiec LR": lr_uc, "p Kupiec": float(stats.chi2.sf(lr_uc, 1)),
            "Christoffersen LR": lr_ind, "p independence": float(stats.chi2.sf(lr_ind, 1)),
            "p conditional coverage": float(stats.chi2.sf(lr_uc + lr_ind, 2))}


class _FastGarch:
    """GARCH(1,1) / GJR-GARCH(1,1) with normal or standardised-t errors; the variance recursion runs as a linear filter
    (fast enough for rolling re-estimation). h_1 = mean squared residual (as fGarch)."""

    def __init__(self, y, gjr=False, t=False, start=None):
        from scipy.signal import lfilter
        from scipy.special import gammaln
        y = np.asarray(y, float)
        self.dist = "std" if t else "norm"
        k = 4 + gjr + t

        def sig2(p):
            mu, om, al = p[0], p[1], p[2]
            ga = p[3] if gjr else 0.0
            be = p[3 + gjr]
            e = y - mu
            x = om + (al + ga * (e < 0)) * e ** 2           # contribution to h_{t+1}
            h = np.empty(len(y))
            h[0] = np.mean(e ** 2)
            h[1:] = lfilter([1.0], [1.0, -be], x[:-1], zi=[be * h[0]])[0]
            return e, h

        def nll(p):
            al, be = p[2], p[3 + gjr]
            ga = p[3] if gjr else 0.0
            if p[1] <= 0 or al < 0 or be < 0 or al + ga < 0 or al + be + ga / 2 >= 0.9999:
                return 1e12
            e, h = sig2(p)
            if np.any(h <= 0):
                return 1e12
            if t:
                nu = p[-1]
                if nu <= 2.05 or nu > 200:
                    return 1e12
                ll = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(h)
                      - (nu + 1) / 2 * np.log1p(e ** 2 / (h * (nu - 2))))
                return -float(np.sum(ll))
            return float(0.5 * np.sum(np.log(2 * np.pi * h) + e ** 2 / h))
        v = np.var(y)
        st = (np.asarray(start, float) if start is not None and len(start) == k else
              np.r_[y.mean(), 0.05 * v, 0.08, [0.04] * gjr, 0.88, [8.0] * t])
        r = optimize.minimize(nll, st, method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9, "adaptive": True})
        self.x = r.x
        self.llf = -r.fun
        self.names = ["mu", "omega", "alpha1"] + ["gamma1"] * gjr + ["beta1"] + ["shape"] * t
        self.resid, self.sigma2 = sig2(r.x)

    def p_(self, name, default=0.0):
        return float(self.x[self.names.index(name)]) if name in self.names else default


def _garch_filter(r, y_new, h_last, e_last):
    om, al, ga, be, mu = r.p_("omega"), r.p_("alpha1"), r.p_("gamma1"), r.p_("beta1"), r.p_("mu")
    hs, h, e = [], h_last, e_last
    for v in y_new:
        h = om + (al + ga * (e < 0)) * e ** 2 + be * h
        hs.append(h)
        e = v - mu
    return np.array(hs)


def var_backtest(ret, level: float = 0.01, window: int = 1000, refit: int = 20, rv=None, start: int | None = None,
                 methods=("garch_n", "gjr_t", "hs", "ewma", "har")) -> dict:
    """Rolling one-day-ahead VaR forecasts on the last part of the sample: GARCH(1,1)-normal, GJR-GARCH(1,1)-t
    (re-estimated every `refit` days on a rolling `window`, filtered daily in between), historical simulation,
    RiskMetrics EWMA (λ = 0.94) and, when realized variance is supplied, HAR-RV with Student-t quantiles from the
    standardised returns. Coverage tests and the average quantile (tick) loss rank the methods."""
    from analytics import garch as gm
    y = np.asarray(ret, float)
    n = len(y)
    start = start or max(window, n - 750)
    res, paths = [], {}
    for m in methods:
        if m == "har" and rv is None:
            continue
        v = np.full(n, np.nan)
        if m in ("garch_n", "gjr_t"):
            prev = None
            for t0 in range(start, n, refit):
                tr = y[t0 - window:t0]
                r = _FastGarch(tr, gjr=m == "gjr_t", t=m == "gjr_t", start=prev)
                prev = r.x
                hs = _garch_filter(r, y[t0:min(n, t0 + refit)], r.sigma2[-1], r.resid[-1])
                qz = float(gm._ppf(level, r.dist, r.p_("shape", None), None))
                v[t0:t0 + len(hs)] = r.p_("mu") + np.sqrt(hs) * qz
        elif m == "hs":
            for t in range(start, n):
                v[t] = np.quantile(y[t - window:t], level)
        elif m == "ewma":
            s2 = np.var(y[:start - window + 50]) if start - window + 50 > 10 else np.var(y[:start])
            for t in range(1, n):
                s2 = 0.94 * s2 + 0.06 * y[t - 1] ** 2
                if t >= start:
                    v[t] = np.sqrt(s2) * stats.norm.ppf(level)
        elif m == "har":
            rvv = np.asarray(rv, float)
            for t0 in range(start, n, refit):
                h = har(rvv[t0 - window:t0], 1, log=True, horizon=1)
                b = h["b"]
                s2 = h["model"].mse_resid
                z = (y[t0 - window:t0] - y[t0 - window:t0].mean()) / np.sqrt(rvv[t0 - window:t0])
                nu = max(stats.t.fit(z, floc=0)[0], 2.5)
                sc = np.std(z)
                q = stats.t.ppf(level, nu) * np.sqrt((nu - 2) / nu) * sc
                for t in range(t0, min(n, t0 + refit)):
                    x = np.log(rvv[:t])
                    f = np.exp(b @ [1, x[-1], x[-5:].mean(), x[-22:].mean()] + s2 / 2)
                    v[t] = y[t0 - window:t0].mean() + np.sqrt(f) * q
        paths[m] = v
        ok = np.isfinite(v)
        if not ok.any():
            continue
        hits = (y[ok] < v[ok]).astype(int)
        tick = np.mean((level - hits) * (y[ok] - v[ok]))
        res.append({"Method": VAR_METHODS[m][0], "key": m, **coverage_tests(hits, level), "Tick loss": float(tick)})
    tab = pd.DataFrame(res).sort_values("Tick loss").reset_index(drop=True)
    return {"table": tab, "paths": paths, "start": start, "level": level}


VAR_METHODS = {"garch_n": ("GARCH(1,1) – normal", "GARCH(1,1) – طبيعي"), "gjr_t": ("GJR-GARCH(1,1) – Student t", "GJR-GARCH(1,1) – t"),
               "hs": ("Historical simulation", "المحاكاة التاريخية"), "ewma": ("RiskMetrics EWMA (λ = 0.94)", "EWMA لريسك متركس (λ = 0.94)"),
               "har": ("HAR-RV (log, t quantile)", "HAR-RV (لوغاريتمي، مئين t)")}


def vol_forecast_compare(ret, rv, window: int = 1000, refit: int = 20, start: int | None = None) -> dict:
    """One-day-ahead variance forecasts evaluated against realized variance: GARCH(1,1), GJR-GARCH(1,1), HAR-RV
    (levels and logs) and EWMA; MSE, QLIKE, Mincer-Zarnowitz regressions and DM tests (QLIKE loss) against the best."""
    y, rvv = np.asarray(ret, float), np.asarray(rv, float)
    n = len(y)
    start = start or max(window, n - 750)
    F_ = {k: np.full(n, np.nan) for k in ("GARCH(1,1)", "GJR-GARCH(1,1)", "HAR-RV", "log HAR-RV", "EWMA")}
    prev = {}
    for t0 in range(start, n, refit):
        tr = y[t0 - window:t0]
        for k, mdl in (("GARCH(1,1)", "garch"), ("GJR-GARCH(1,1)", "gjr")):
            r = _FastGarch(tr, gjr=mdl == "gjr", start=prev.get(k))
            prev[k] = r.x
            F_[k][t0:t0 + refit] = _garch_filter(r, y[t0:min(n, t0 + refit)], r.sigma2[-1], r.resid[-1])[:min(refit, n - t0)]
        for k, lg in (("HAR-RV", False), ("log HAR-RV", True)):
            h = har(rvv[t0 - window:t0], 1, log=lg, horizon=1)
            b, s2 = h["b"], h["model"].mse_resid
            for t in range(t0, min(n, t0 + refit)):
                x = np.log(rvv[:t]) if lg else rvv[:t]
                f = b @ [1, x[-1], x[-5:].mean(), x[-22:].mean()]
                F_[k][t] = np.exp(f + s2 / 2) if lg else max(f, 1e-8)
    s2 = np.var(y[:start])
    for t in range(1, n):
        s2 = 0.94 * s2 + 0.06 * y[t - 1] ** 2
        if t >= start:
            F_["EWMA"][t] = s2
    rows, L = [], {}
    a = rvv[start:]
    for k, f in F_.items():
        ff = f[start:]
        L[k] = a / ff - np.log(a / ff) - 1
        mz = mincer_zarnowitz(a, ff)
        rows.append({"Model": k, "MSE": float(np.mean((a - ff) ** 2)), "QLIKE": float(np.mean(L[k])), "MZ a": mz["a"], "MZ b": mz["b"],
                     "MZ R²": mz["R²"], "MZ p (a=0, b=1)": mz["p"]})
    tab = pd.DataFrame(rows).sort_values("QLIKE").reset_index(drop=True)
    best = tab.Model.iloc[0]
    dm = []
    for k in tab.Model[1:]:
        d = L[k] - L[best]
        r_ = dm_test(np.sqrt(np.abs(L[k])), np.sqrt(np.abs(L[best])), 1, power=2)
        dm.append({"Model": k, "vs": best, "DM (QLIKE)": r_["DM"], "p": r_["p"], "mean loss diff": float(d.mean())})
    return {"table": tab, "dm": pd.DataFrame(dm), "forecasts": {k: v[start:] for k, v in F_.items()}, "actual": a, "start": start}
