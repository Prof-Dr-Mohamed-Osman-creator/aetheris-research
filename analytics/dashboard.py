# ==========================================
# Aetheris Research - Decision-maker dashboard engine (v1.12, Catalogue §8)
# KPI cards, early-warning signals (z-score, percentile, HP-gap and fixed thresholds) with traffic lights, a composite
# early-warning index, the signals approach of Kaminsky, Lizondo & Reinhart (1998) (noise-to-signal ratio and optimal
# threshold), forecast-based warnings (probability of crossing a threshold) and a what-if scenario simulator.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

STATUS = {"green": ("Normal", "طبيعي"), "amber": ("Watch", "مراقبة"), "red": ("Alert", "إنذار"), "na": ("Insufficient data", "بيانات غير كافية")}
METHODS = {"z": ("z-score vs history", "الانحراف المعياري عن التاريخ"), "pct": ("Historical percentile", "المئين التاريخي"),
           "gap": ("Gap from HP trend (%)", "الفجوة عن اتجاه HP (%)"), "fixed": ("Fixed thresholds", "عتبات ثابتة")}


def series(df: pd.DataFrame, time: str, col: str, unit: str | None = None, unit_value=None, agg: str = "mean") -> pd.Series:
    d = df if unit is None or unit_value is None else df[df[unit] == unit_value]
    s = d.groupby(time)[col].agg(agg).sort_index()
    return s.dropna()


def kpi(s: pd.Series, window: int = 12) -> dict:
    s = s.dropna()
    if len(s) < 2:
        return {"latest": float(s.iloc[-1]) if len(s) else np.nan, "change": np.nan, "pct": np.nan, "z": np.nan, "period": s.index[-1] if len(s) else None}
    last, prev = float(s.iloc[-1]), float(s.iloc[-2])
    hist = s.iloc[:-1].tail(max(window, 3) * 5)
    z = (last - hist.mean()) / hist.std(ddof=1) if hist.std(ddof=1) > 0 else 0.0
    base = s.iloc[-1 - window] if len(s) > window else s.iloc[0]
    n = min(window, len(s) - 1)
    cagr = (last / base) ** (1 / n) - 1 if base > 0 and last > 0 and n > 0 else np.nan
    return {"latest": last, "previous": prev, "change": last - prev, "pct": 100 * (last / prev - 1) if prev else np.nan, "z": float(z),
            "window change %": 100 * (last / base - 1) if base else np.nan, "CAGR %": 100 * cagr if np.isfinite(cagr) else np.nan,
            "min": float(s.min()), "max": float(s.max()), "period": s.index[-1]}


def _hp_trend_onesided(x: np.ndarray, lamb: float) -> np.ndarray:
    """One-sided (real-time) HP trend: the end point of the two-sided filter applied recursively (Basel credit gap)."""
    from statsmodels.tsa.filters.hp_filter import hpfilter
    out = np.full(len(x), np.nan)
    for t in range(8, len(x) + 1):
        _, tr = hpfilter(x[:t], lamb=lamb)
        out[t - 1] = tr[-1]
    return out


def signal(s: pd.Series, method: str = "z", bad: str = "high", amber: float | None = None, red: float | None = None, lamb: float | None = None,
           window: int | None = None) -> pd.DataFrame:
    """Signal path and traffic-light status for one indicator. `bad` = 'high' (rises are dangerous), 'low' or 'both'.
    Default thresholds: z 1 / 2; percentile 80 / 95; HP gap 2 / 10 %; fixed thresholds must be given."""
    x = s.astype(float).to_numpy()
    idx = s.index
    if method == "z":
        a, r = amber or 1.0, red or 2.0
        sr = pd.Series(x, index=idx)
        hist = sr.shift(1)
        mu = hist.rolling(window, min_periods=8).mean() if window else hist.expanding(8).mean()
        sd = hist.rolling(window, min_periods=8).std() if window else hist.expanding(8).std()
        val = ((sr - mu) / sd).to_numpy()
    elif method == "pct":
        a, r = amber or 80.0, red or 95.0
        val = np.array([100 * stats.percentileofscore(x[:t], x[t], kind="weak") if t >= 8 else np.nan for t in range(len(x))])
        if bad == "low":
            val = 100 - val
    elif method == "gap":
        a, r = amber or 2.0, red or 10.0
        tr = _hp_trend_onesided(x, lamb or 1600.0)
        val = 100 * (x - tr) / np.abs(tr)
    else:
        if amber is None or red is None:
            raise ValueError("Fixed thresholds need amber and red levels.")
        a, r = amber, red
        val = x.copy()
    if method == "fixed":
        hi = a < r                              # thresholds increasing = high values are bad
        lvl = np.where(np.isnan(val), np.nan, np.where((val >= r) if hi else (val <= r), 2, np.where((val >= a) if hi else (val <= a), 1, 0)))
    else:
        v = val if bad in ("high",) or method == "pct" else (-val if bad == "low" else np.abs(val))
        lvl = np.where(np.isnan(v), np.nan, np.where(v >= r, 2, np.where(v >= a, 1, 0)))
    st_ = np.array(["na" if np.isnan(l_) else ["green", "amber", "red"][int(l_)] for l_ in lvl])
    return pd.DataFrame({"value": x, "signal": val, "level": lvl, "status": st_, "amber": a, "red": r}, index=idx)


def composite(signals: dict, weights: dict | None = None) -> pd.DataFrame:
    """Composite early-warning index: weighted mean of the indicators' alert levels (0 / 1 / 2) rescaled to 0–100."""
    L = pd.DataFrame({k: v["level"] for k, v in signals.items()})
    w = pd.Series({k: (weights or {}).get(k, 1.0) for k in L.columns})
    num = (L * w).sum(1, min_count=1)
    den = (L.notna() * w).sum(1)
    idx = 100 * num / (2 * den.replace(0, np.nan))
    st_ = np.where(idx.isna(), "na", np.where(idx >= 50, "red", np.where(idx >= 25, "amber", "green")))
    return pd.DataFrame({"index": idx, "status": st_, "n red": (L == 2).sum(1), "n amber": (L == 1).sum(1)})


def noise_to_signal(x: pd.Series, crisis: pd.Series, horizon: int = 12, bad: str = "high", grid=None) -> dict:
    """Kaminsky-Lizondo-Reinhart signals approach: a signal is issued when x crosses a percentile threshold; it is
    correct if a crisis starts within `horizon` periods. Noise-to-signal ratio NtS = [B/(B+D)] / [A/(A+C)]; the optimal
    threshold minimises NtS (requiring at least one correct signal)."""
    d = pd.concat([x.rename("x"), crisis.rename("c")], axis=1).dropna()
    c = d["c"].to_numpy().astype(int)
    starts = np.where((c == 1) & (np.r_[0, c[:-1]] == 0))[0]
    win = np.zeros(len(c), bool)
    for s0 in starts:
        win[max(0, s0 - horizon):s0] = True
    valid = c == 0                              # exclude crisis periods themselves
    v = d["x"].to_numpy()
    grid = grid if grid is not None else np.arange(70, 99, 1.0)
    rows = []
    for g in grid:
        thr = np.percentile(v, g if bad == "high" else 100 - g)
        sig = v >= thr if bad == "high" else v <= thr
        A = int(np.sum(sig & win & valid))
        B = int(np.sum(sig & ~win & valid))
        C = int(np.sum(~sig & win & valid))
        D = int(np.sum(~sig & ~win & valid))
        good = A / (A + C) if A + C else np.nan
        noise = B / (B + D) if B + D else np.nan
        nts = noise / good if good and good > 0 else np.nan
        pcond = A / (A + B) if A + B else np.nan
        rows.append({"percentile": g, "threshold": thr, "A": A, "B": B, "C": C, "D": D, "share of crises signalled": good, "false-alarm rate": noise,
                     "NtS": nts, "P(crisis | signal)": pcond})
    T = pd.DataFrame(rows)
    ok = T[T.A > 0].dropna(subset=["NtS"])
    best = ok.loc[ok.NtS.idxmin()].to_dict() if len(ok) else None
    return {"table": T, "best": best, "crises": int(len(starts)), "uncond P(crisis window)": float(win[valid].mean()) if valid.any() else np.nan}


def forecast_warning(s: pd.Series, h: int = 6, threshold: float | None = None, bad: str = "high", season: int = 1) -> dict:
    """ETS forecast (R forecast::ets conventions) and the probability that each future value is on the bad side of
    the threshold (normal approximation from the 95 % interval)."""
    from analytics import fcst
    y = s.astype(float).to_numpy()
    f = fcst.ets(y, h, s=season)
    lo, hi = f.pi[95] if 95 in f.pi else list(f.pi.values())[-1]
    sd = (np.asarray(hi) - np.asarray(lo)) / (2 * 1.959963984540054)
    mean = np.asarray(f.mean)
    out = pd.DataFrame({"step": np.arange(1, h + 1), "forecast": mean, "lower 95": lo, "upper 95": hi})
    if threshold is not None:
        z = (threshold - mean) / np.where(sd > 0, sd, np.nan)
        out["P(breach)"] = stats.norm.sf(z) if bad == "high" else stats.norm.cdf(z)
    return {"table": out, "model": f.name}


def scenario_model(df: pd.DataFrame, y: str, xs: list, form: str = "loglog") -> dict:
    """What-if model for decision makers: OLS (HC1) in log-log (elasticities) or linear form."""
    import statsmodels.api as sm
    d = df[[y] + xs].dropna()
    if form == "loglog":
        d = d[(d > 0).all(1)]
        Y, X = np.log(d[y]), np.log(d[xs])
    else:
        Y, X = d[y], d[xs]
    m = sm.OLS(Y, sm.add_constant(X)).fit(cov_type="HC1")
    return {"model": m, "form": form, "y": y, "xs": xs, "mean_y": float(d[y].mean()), "means": d[xs].mean().to_dict(), "N": len(d), "R2": float(m.rsquared)}


def scenario(sm_: dict, changes: dict) -> dict:
    """Predicted % (log-log) or level (linear) change of y for given % changes (log-log) / level changes of the drivers."""
    m = sm_["model"]
    names = list(m.params.index)
    g = np.zeros(len(names))
    for j, v in enumerate(sm_["xs"]):
        c = changes.get(v, 0.0)
        g[names.index(v)] = np.log1p(c / 100) if sm_["form"] == "loglog" else c
    est = float(g @ m.params.to_numpy())
    se = float(np.sqrt(g @ m.cov_params().to_numpy() @ g))
    if sm_["form"] == "loglog":
        f = lambda v: 100 * (np.exp(v) - 1)  # noqa: E731
        return {"change %": f(est), "low %": f(est - 1.96 * se), "high %": f(est + 1.96 * se), "new level": sm_["mean_y"] * np.exp(est)}
    return {"change": est, "low": est - 1.96 * se, "high": est + 1.96 * se, "change %": 100 * est / sm_["mean_y"], "new level": sm_["mean_y"] + est}
