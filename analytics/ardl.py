# ==========================================
# Aetheris Research - ARDL bounds testing, NARDL and QARDL (Catalogue §1.8)
# ARDL(p, q₁ … q_k) of Pesaran, Shin & Smith (2001) with the five deterministic cases, automatic order selection on a
# common sample, unrestricted / restricted ECM, long-run multipliers (delta method), bounds F and t tests with
# asymptotic and finite-sample critical values simulated exactly as in PSS (2001); NARDL of Shin, Yu &
# Greenwood-Nimmo (2014) with long- and short-run asymmetry tests and dynamic multipliers; QARDL of Cho, Kim &
# Shin (2015) by exact linear-programming quantile regression. Conventions follow R ARDL / dynlm / quantreg.
# ==========================================

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from scipy import stats

CASES = {1: ("I — no intercept, no trend", "I — بدون ثابت أو اتجاه"),
         2: ("II — restricted intercept, no trend", "II — ثابت مقيد، بدون اتجاه"),
         3: ("III — unrestricted intercept, no trend", "III — ثابت غير مقيد، بدون اتجاه"),
         4: ("IV — unrestricted intercept, restricted trend", "IV — ثابت غير مقيد واتجاه مقيد"),
         5: ("V — unrestricted intercept and trend", "V — ثابت واتجاه غير مقيدين")}
_TABLE = Path(__file__).resolve().parent.parent / "data" / "bounds_cv.csv"
ALPHAS = np.r_[np.arange(0.001, 0.1001, 0.001), np.arange(0.11, 1.0, 0.01)]


@dataclass
class ArdlResult:
    y: str
    x: list
    order: tuple
    case: int
    fixed: list
    data: pd.DataFrame           # aligned estimation sample (levels)
    X: np.ndarray                # ARDL regressors (levels form)
    names: list
    b: np.ndarray
    V: np.ndarray
    resid: np.ndarray
    yv: np.ndarray
    df: int
    uecm: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def nobs(self):
        return len(self.yv)

    def table(self) -> pd.DataFrame:
        return _coef_table(self.names, self.b, self.V, self.df)


def _coef_table(names, b, V, df) -> pd.DataFrame:
    se = np.sqrt(np.diag(V))
    t = b / se
    q = stats.t.ppf(0.975, df)
    return pd.DataFrame({"Regressor": names, "Coef.": b, "Std. Err.": se, "t": t, "p-value": 2 * stats.t.sf(np.abs(t), df),
                         "CI low": b - q * se, "CI high": b + q * se})


def _ols(Y, X):
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ Y
    e = Y - X @ b
    df = len(Y) - X.shape[1]
    s2 = e @ e / df
    return b, s2 * XtXi, e, df, s2


def _lag(v: np.ndarray, L: int) -> np.ndarray:
    out = np.full(len(v), np.nan)
    if L == 0:
        return v.astype(float).copy()
    out[L:] = v[:-L]
    return out


def _det(T: int, case: int, start: int):
    t = np.arange(1, T + 1, dtype=float)
    cols, names = [], []
    if case >= 2:
        cols.append(np.ones(T)); names.append("(Intercept)")
    if case >= 4:
        cols.append(t); names.append("trend")
    return cols, names


def fit_ardl(df: pd.DataFrame, y: str, x: list, order: tuple, case: int = 3, fixed: list | None = None,
             start: int | None = None) -> ArdlResult:
    """ARDL in levels by OLS. order = (p, q₁, …, q_k); `start` (0-based row) fixes a common estimation sample."""
    fixed = list(fixed or [])
    p, qs = order[0], list(order[1:])
    if len(qs) != len(x):
        raise ValueError("Give one lag order per regressor.")
    T = len(df)
    yv = df[y].to_numpy(float)
    cols, names = _det(T, case, 0)
    for i in range(1, p + 1):
        cols.append(_lag(yv, i)); names.append(f"L({y}, {i})")
    for xn, q in zip(x, qs):
        xv = df[xn].to_numpy(float)
        for L in range(0, q + 1):
            cols.append(_lag(xv, L)); names.append(xn if L == 0 else f"L({xn}, {L})")
    for f in fixed:
        cols.append(df[f].to_numpy(float)); names.append(f)
    X = np.column_stack(cols)
    s0 = max([p] + qs) if start is None else start
    keep = np.arange(s0, T)
    keep = keep[np.isfinite(X[keep]).all(1) & np.isfinite(yv[keep])]
    Xk, yk = X[keep], yv[keep]
    b, V, e, dfr, s2 = _ols(yk, Xk)
    r = ArdlResult(y, list(x), tuple(order), case, fixed, df.iloc[keep], Xk, names, b, V, e, yk, dfr)
    n, k = Xk.shape
    ll = -n / 2 * (np.log(2 * np.pi) + np.log(e @ e / n) + 1)
    tss = ((yk - yk.mean()) ** 2).sum() if case >= 2 else (yk ** 2).sum()
    r2 = 1 - e @ e / tss
    r.stats = {"N": n, "Log-likelihood": ll, "AIC": -2 * ll + 2 * (k + 1), "BIC": -2 * ll + np.log(n) * (k + 1),
               "R²": r2, "Adj. R²": 1 - (1 - r2) * (n - (1 if case >= 2 else 0)) / dfr, "σ": np.sqrt(s2)}
    r.uecm = uecm(df, y, x, order, case, fixed, keep)
    return r


def uecm(df, y, x, order, case, fixed, keep) -> dict:
    """Unrestricted error-correction form (conditional ECM, PSS 2001 eq. 6) on the same sample."""
    p, qs = order[0], list(order[1:])
    T = len(df)
    yv = df[y].to_numpy(float)
    dy = np.r_[np.nan, np.diff(yv)]
    cols, names = _det(T, case, 0)
    cols.append(_lag(yv, 1)); names.append(f"L({y}, 1)")
    for xn, q in zip(x, qs):
        xv = df[xn].to_numpy(float)
        cols.append(_lag(xv, 1) if q > 0 else xv); names.append(f"L({xn}, 1)" if q > 0 else xn)
    for i in range(1, p):
        cols.append(_lag(dy, i)); names.append(f"L(d({y}), {i})")
    for xn, q in zip(x, qs):
        dxv = np.r_[np.nan, np.diff(df[xn].to_numpy(float))]
        for L in range(0, q):
            cols.append(_lag(dxv, L)); names.append(f"d({xn})" if L == 0 else f"L(d({xn}), {L})")
    for f in fixed:
        cols.append(df[f].to_numpy(float)); names.append(f)
    X = np.column_stack(cols)[keep]
    Y = dy[keep]
    b, V, e, dfr, s2 = _ols(Y, X)
    return {"names": names, "b": b, "V": V, "resid": e, "df": dfr, "X": X, "Y": Y}


def long_run(r: ArdlResult) -> pd.DataFrame:
    """Long-run multipliers θ = −π_x / π_y (and −c/π_y for restricted deterministic terms) with delta-method SE."""
    u = r.uecm
    nm, b, V = u["names"], u["b"], u["V"]
    iy = nm.index(f"L({r.y}, 1)")
    targets = []
    if r.case in (2, 3, 4, 5):
        targets.append(("(Intercept)", nm.index("(Intercept)")))
    if r.case in (4, 5):
        targets.append(("trend", nm.index("trend")))
    for xn, q in zip(r.x, r.order[1:]):
        targets.append((xn, nm.index(f"L({xn}, 1)" if q > 0 else xn)))
    for f in r.fixed:
        targets.append((f, nm.index(f)))
    rows = []
    py = b[iy]
    for lab, j in targets:
        th = -b[j] / py
        g = np.zeros(len(b))
        g[j] = -1 / py
        g[iy] = b[j] / py ** 2
        se = float(np.sqrt(g @ V @ g))
        t = th / se
        rows.append({"Variable": lab, "Long-run coef.": th, "Std. Err.": se, "t": t, "p-value": 2 * stats.t.sf(abs(t), u["df"]),
                     "CI low": th - stats.t.ppf(0.975, u["df"]) * se, "CI high": th + stats.t.ppf(0.975, u["df"]) * se})
    return pd.DataFrame(rows)


def recm(r: ArdlResult) -> dict:
    """Restricted ECM: short-run dynamics with the error-correction term built from the long-run multipliers
    (R ARDL::recm). The ECT coefficient is the speed of adjustment."""
    lr = long_run(r).set_index("Variable")["Long-run coef."]
    u = r.uecm
    nm = u["names"]
    X = u["X"]
    iy = nm.index(f"L({r.y}, 1)")
    ect = X[:, iy].copy()
    restricted = {2: ["(Intercept)"], 4: ["trend"]}.get(r.case, [])
    for xn, q in zip(r.x, r.order[1:]):
        ect -= lr[xn] * X[:, nm.index(f"L({xn}, 1)" if q > 0 else xn)]
    for f in r.fixed:
        pass
    for d in restricted:
        ect -= lr[d] * X[:, nm.index(d)]
    keep_cols = [j for j, n in enumerate(nm) if n.startswith(("L(d(", "d(")) or n in r.fixed
                 or (n in ("(Intercept)", "trend") and n not in restricted)]
    Xr = np.column_stack([X[:, keep_cols], ect])
    names = [nm[j] for j in keep_cols] + ["ect"]
    b, V, e, dfr, s2 = _ols(u["Y"], Xr)
    return {"table": _coef_table(names, b, V, dfr), "ect": ect, "speed": b[-1],
            "half_life": np.log(0.5) / np.log(1 + b[-1]) if -2 < b[-1] < 0 else np.nan}


def auto_ardl(df, y, x, max_order, case: int = 3, fixed=None, criterion: str = "AIC", top: int = 20) -> pd.DataFrame:
    """Grid search over (p ≥ 1, q_j ≥ 0) on a common sample starting after the largest maximum lag."""
    max_order = list(max_order)
    start = max(max_order)
    grids = [range(1, max_order[0] + 1)] + [range(0, q + 1) for q in max_order[1:]]
    rows = []
    for o in itertools.product(*grids):
        try:
            r = fit_ardl(df, y, x, o, case, fixed, start=start)
            rows.append({"Order": "ARDL(" + ", ".join(map(str, o)) + ")", "order": o, "AIC": r.stats["AIC"],
                         "BIC": r.stats["BIC"], "Adj. R²": r.stats["Adj. R²"]})
        except Exception:  # noqa: BLE001
            continue
    out = pd.DataFrame(rows).sort_values(criterion).reset_index(drop=True)
    return out.head(top)


# ---------------- bounds tests ----------------
def _restricted_idx(r: ArdlResult):
    nm = r.uecm["names"]
    idx = [nm.index(f"L({r.y}, 1)")] + [nm.index(f"L({xn}, 1)" if q > 0 else xn) for xn, q in zip(r.x, r.order[1:])]
    if r.case == 2:
        idx.append(nm.index("(Intercept)"))
    elif r.case == 4:
        idx.append(nm.index("trend"))
    return idx


def bounds_test(r: ArdlResult, exact: bool = False, k: int | None = None) -> dict:
    """PSS (2001) bounds F test (all cases) and t test (cases I, III, V). Critical values / p-values: asymptotic
    (simulated with T = 1000, 40 000 replications, as in PSS) or, with `exact`, simulated for the actual sample size.
    `k` overrides the number of long-run forcing variables (NARDL conventions)."""
    u = r.uecm
    idx = _restricted_idx(r)
    b, V = u["b"], u["V"]
    W = float(b[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], b[idx]))
    F = W / len(idx)
    iy = u["names"].index(f"L({r.y}, 1)")
    t = float(b[iy] / np.sqrt(V[iy, iy]))
    kx = len(r.x) if k is None else k
    if exact:
        cv = _sim_quantiles(r.case, kx, r.nobs, 10000)
    else:
        cv = _asym_quantiles(r.case, kx)
    out = {"F": F, "t": t, "k": kx, "case": r.case, "n": r.nobs, "exact": exact}
    rows = []
    for a in (0.10, 0.05, 0.025, 0.01):
        j = int(np.argmin(np.abs(ALPHAS - a)))
        rows.append({"Significance": f"{int(a * 1000) / 10:g}%", "F I(0)": cv["F0"][j], "F I(1)": cv["F1"][j],
                     "t I(0)": cv["t0"][j] if r.case in (1, 3, 5) else np.nan,
                     "t I(1)": cv["t1"][j] if r.case in (1, 3, 5) else np.nan})
    out["critical"] = pd.DataFrame(rows)
    out["pF_I0"], out["pF_I1"] = _pval(F, cv["F0"], upper=True), _pval(F, cv["F1"], upper=True)
    if r.case in (1, 3, 5):
        out["pt_I0"], out["pt_I1"] = _pval(t, cv["t0"], upper=False), _pval(t, cv["t1"], upper=False)
    c5 = out["critical"].iloc[1]
    out["verdict_F"] = "cointegration" if F > c5["F I(1)"] else "no cointegration" if F < c5["F I(0)"] else "inconclusive"
    if r.case in (1, 3, 5):
        out["verdict_t"] = ("cointegration" if t < c5["t I(1)"] else "no cointegration" if t > c5["t I(0)"]
                            else "inconclusive")
    return out


def _pval(stat, q, upper=True):
    """Interpolate the p-value in the simulated quantile function (q[j] = critical value at ALPHAS[j])."""
    a = ALPHAS
    if upper:                       # reject for large values; q decreasing in α
        if stat >= q[0]:
            return float(a[0])
        if stat <= q[-1]:
            return float(a[-1])
        return float(np.interp(-stat, -q, a))
    if stat <= q[0]:
        return float(a[0])
    if stat >= q[-1]:
        return float(a[-1])
    return float(np.interp(stat, q, a))


def _bounds_stats(case: int, k: int, T: int, reps: int, seed: int) -> dict:
    """Simulate F (and t) under the null for I(0) and I(1) regressors — PSS (2001) / R ARDL::f_bounds_sim design."""
    rng = np.random.default_rng(seed)
    out = {"F0": [], "F1": [], "t0": [], "t1": []}
    tt = np.arange(1, T, dtype=float)
    for c0 in range(0, reps, 500):
        R = min(500, reps - c0)
        y = np.cumsum(rng.standard_normal((R, T)), 1)
        dy = np.diff(y, axis=1)
        ylag = y[:, :-1]
        x = rng.standard_normal((R, T, k)) if k else np.empty((R, T, 0))
        for integ in (0, 1):
            xx = (np.cumsum(x, 1) if integ else x)[:, :-1, :]
            parts, nrest_front = [], 0
            if case in (2, 3, 4, 5):
                parts.append(np.ones((R, T - 1, 1)))
            if case in (4, 5):
                parts.append(np.broadcast_to(tt[None, :, None], (R, T - 1, 1)))
            parts += [xx, ylag[:, :, None]]
            X = np.concatenate(parts, 2)
            XtX = np.einsum("rti,rtj->rij", X, X)
            Xty = np.einsum("rti,rt->ri", X, dy)
            bb = np.linalg.solve(XtX, Xty[:, :, None])[:, :, 0]
            e = dy - np.einsum("rti,ri->rt", X, bb)
            n, m = T - 1, X.shape[2]
            s2 = (e ** 2).sum(1) / (n - m)
            XtXi = np.linalg.inv(XtX)
            ndet = (1 if case >= 2 else 0) + (1 if case >= 4 else 0)
            if case in (1, 3, 5):
                idx = list(range(ndet, m))
            elif case == 2:
                idx = [0] + list(range(ndet, m))
            else:                                  # case 4: trend + level terms
                idx = [1] + list(range(ndet, m))
            Vr = XtXi[:, idx][:, :, idx] * s2[:, None, None]
            br = bb[:, idx]
            W = np.einsum("ri,ri->r", br, np.linalg.solve(Vr, br[:, :, None])[:, :, 0])
            out["F1" if integ else "F0"].append(W / len(idx))
            out["t1" if integ else "t0"].append(bb[:, -1] / np.sqrt(XtXi[:, -1, -1] * s2))
    return {k_: np.concatenate(v) for k_, v in out.items()}


def _quantiles(sims: dict) -> dict:
    return {"F0": np.quantile(sims["F0"], 1 - ALPHAS), "F1": np.quantile(sims["F1"], 1 - ALPHAS),
            "t0": np.quantile(sims["t0"], ALPHAS), "t1": np.quantile(sims["t1"], ALPHAS)}


@lru_cache(maxsize=64)
def _sim_quantiles(case: int, k: int, T: int, reps: int = 10000) -> dict:
    return _quantiles(_bounds_stats(case, k, T, reps, seed=2001 + 97 * case + k))


@lru_cache(maxsize=1)
def _table() -> pd.DataFrame | None:
    return pd.read_csv(_TABLE) if _TABLE.exists() else None


def _asym_quantiles(case: int, k: int) -> dict:
    tab = _table()
    if tab is not None and ((tab["case"] == case) & (tab["k"] == k)).any():
        s = tab[(tab["case"] == case) & (tab["k"] == k)].sort_values("alpha")
        return {c: s[c].to_numpy() for c in ("F0", "F1", "t0", "t1")}
    return _sim_quantiles(case, k, 1000, 10000)


# ---------------- NARDL ----------------
def partial_sums(x: np.ndarray, threshold: float = 0.0) -> tuple:
    """x⁺_t = Σ max(Δx_j − τ, 0), x⁻_t = Σ min(Δx_j − τ, 0) (Shin, Yu & Greenwood-Nimmo 2014)."""
    dx = np.r_[0.0, np.diff(np.asarray(x, float))]
    return np.cumsum(np.maximum(dx - threshold, 0)), np.cumsum(np.minimum(dx - threshold, 0))


def nardl_data(df: pd.DataFrame, asym: list, threshold: float = 0.0) -> tuple:
    d = df.copy()
    new = {}
    for a in asym:
        pos, neg = partial_sums(d[a].to_numpy(float), threshold)
        d[f"{a}_pos"], d[f"{a}_neg"] = pos, neg
        new[a] = (f"{a}_pos", f"{a}_neg")
    return d, new


def asymmetry_tests(r: ArdlResult, pairs: dict) -> pd.DataFrame:
    """Wald tests of long-run symmetry (θ⁺ = θ⁻) and short-run (additive) symmetry Σω⁺ = Σω⁻."""
    rows = []
    u = r.uecm
    nm, b, V = u["names"], u["b"], u["V"]
    iy = nm.index(f"L({r.y}, 1)")
    qd = dict(zip(r.x, r.order[1:]))
    for a, (pn, nn) in pairs.items():
        ip, in_ = nm.index(f"L({pn}, 1)" if qd[pn] else pn), nm.index(f"L({nn}, 1)" if qd[nn] else nn)
        th_p, th_n = -b[ip] / b[iy], -b[in_] / b[iy]
        g = np.zeros(len(b))
        g[ip], g[in_] = -1 / b[iy], 1 / b[iy]
        g[iy] = (b[ip] - b[in_]) / b[iy] ** 2
        W = (th_p - th_n) ** 2 / float(g @ V @ g)
        rows.append({"Variable": a, "Test": "Long-run symmetry θ⁺ = θ⁻", "θ⁺ / Σω⁺": th_p, "θ⁻ / Σω⁻": th_n,
                     "Wald F": W, "df": f"1, {u['df']}", "p-value": stats.f.sf(W, 1, u["df"])})
        sp = [j for j, n in enumerate(nm) if n in (f"d({pn})",) or n.startswith(f"L(d({pn}),")]
        sn = [j for j, n in enumerate(nm) if n in (f"d({nn})",) or n.startswith(f"L(d({nn}),")]
        if sp and sn:
            g = np.zeros(len(b))
            g[sp], g[sn] = 1, -1
            d_ = g @ b
            W = d_ ** 2 / float(g @ V @ g)
            rows.append({"Variable": a, "Test": "Short-run symmetry Σω⁺ = Σω⁻", "θ⁺ / Σω⁺": b[sp].sum(),
                         "θ⁻ / Σω⁻": b[sn].sum(), "Wald F": W, "df": f"1, {u['df']}", "p-value": stats.f.sf(W, 1, u["df"])})
    return pd.DataFrame(rows)


def dynamic_multipliers(r: ArdlResult, var: str, h: int = 24) -> np.ndarray:
    """Cumulative dynamic multiplier m_h = Σ_{j≤h} ∂y_{t+j}/∂x_t from the levels ARDL (converges to θ)."""
    nm = r.names
    p = r.order[0]
    phi = np.array([r.b[nm.index(f"L({r.y}, {i})")] for i in range(1, p + 1)])
    q = dict(zip(r.x, r.order[1:]))[var]
    beta = np.array([r.b[nm.index(var if L == 0 else f"L({var}, {L})")] for L in range(q + 1)])
    lam = np.zeros(h + 1)
    for j in range(h + 1):
        v = beta[j] if j <= q else 0.0
        for i in range(1, min(j, p) + 1):
            v += phi[i - 1] * lam[j - i]
        lam[j] = v
    return np.cumsum(lam)


def multiplier_bands(r: ArdlResult, df: pd.DataFrame, pairs: dict, h: int = 24, reps: int = 300, seed: int = 7,
                     level: float = 0.95) -> dict:
    """Bootstrap (residual resampling, fixed regressors) bands for the asymmetry m⁺_h − m⁻_h of each NARDL pair."""
    rng = np.random.default_rng(seed)
    out = {}
    base = {a: dynamic_multipliers(r, p, h) - dynamic_multipliers(r, n, h) for a, (p, n) in pairs.items()}
    draws = {a: [] for a in pairs}
    fitted = r.X @ r.b
    e = r.resid - r.resid.mean()
    ylag_cols = [r.names.index(f"L({r.y}, {i})") for i in range(1, r.order[0] + 1)]
    for _ in range(reps):
        eb = e[rng.integers(0, len(e), len(e))]
        X = r.X.copy()
        yb = np.zeros(len(e))
        # recursive regeneration of y with the estimated dynamics
        nonlag = [j for j in range(X.shape[1]) if j not in ylag_cols]
        hist = list(r.X[0, ylag_cols])
        for t in range(len(e)):
            X[t, ylag_cols] = hist[:len(ylag_cols)]
            yb[t] = X[t] @ r.b + eb[t]
            hist = [yb[t]] + hist[:-1]
        b = np.linalg.lstsq(X, yb, rcond=None)[0]
        rb = ArdlResult(r.y, r.x, r.order, r.case, r.fixed, r.data, X, r.names, b, r.V, eb, yb, r.df)
        for a, (p, n) in pairs.items():
            draws[a].append(dynamic_multipliers(rb, p, h) - dynamic_multipliers(rb, n, h))
    a_ = (1 - level) / 2
    for a in pairs:
        d = np.array(draws[a])
        out[a] = {"asym": base[a], "lower": np.quantile(d, a_, 0), "upper": np.quantile(d, 1 - a_, 0)}
    del fitted, nonlag
    return out


# ---------------- QARDL ----------------
def _rq_vertex(y, X, tau, b0, tol=1e-9):
    """Round an approximate solution to the vertex through the k smallest-residual observations and certify it with
    the KKT conditions (dual weights of the basic observations inside [τ−1, τ]); returns None if not certified."""
    n, k = X.shape
    order = np.argsort(np.abs(y - X @ b0), kind="stable")
    h = []
    for i in order[:min(n, 5 * k + 20)]:
        cand = h + [i]
        if np.linalg.matrix_rank(X[cand]) == len(cand):
            h = cand
        if len(h) == k:
            break
    if len(h) < k:
        return None
    h = np.array(h)
    b = np.linalg.solve(X[h], y[h])
    r = y - X @ b
    nb = np.ones(n, bool)
    nb[h] = False
    if np.any(np.abs(r[nb]) < 1e-12 * max(1.0, np.abs(y).max())):
        return None                                    # degenerate: let the LP decide
    psi = np.where(r[nb] > 0, tau, tau - 1)
    g = X[nb].T @ psi
    v = np.linalg.solve(X[h].T, -g)
    if np.all(v >= tau - 1 - tol) and np.all(v <= tau + tol):
        return b
    return None


def rq(y: np.ndarray, X: np.ndarray, tau: float) -> np.ndarray:
    """Exact quantile regression (Koenker-Bassett) — the R quantreg 'br' solution. A fast IRLS start is rounded to a
    vertex certified optimal by the KKT conditions; otherwise the LP is solved exactly by HiGHS (sparse)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    if len(y) > 300:
        try:
            import statsmodels.api as sm
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b0 = sm.QuantReg(y, X).fit(q=tau, max_iter=300, p_tol=1e-8).params
            b = _rq_vertex(y, X, tau, np.asarray(b0))
            if b is not None:
                return b
        except Exception:  # noqa: BLE001
            pass
    return _rq_lp(y, X, tau)


def _rq_lp(y: np.ndarray, X: np.ndarray, tau: float) -> np.ndarray:
    from scipy import sparse
    from scipy.optimize import linprog
    n, k = X.shape
    c = np.r_[np.zeros(k), tau * np.ones(n), (1 - tau) * np.ones(n)]
    I = sparse.identity(n, format="csc")
    A = sparse.hstack([sparse.csc_matrix(X), I, -I], format="csc")   # sparse: O(n) memory, fast HiGHS
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError("Quantile regression did not converge: " + res.message)
    return res.x[:k]


def _hs_bandwidth(n: int, tau: float, alpha: float = 0.05) -> float:
    x0 = stats.norm.ppf(tau)
    f0 = stats.norm.pdf(x0)
    return n ** (-1 / 3) * stats.norm.ppf(1 - alpha / 2) ** (2 / 3) * ((1.5 * f0 ** 2) / (2 * x0 ** 2 + 1)) ** (1 / 3)


def rq_nid(y, X, tau: float) -> tuple:
    """Coefficients and Hendricks-Koenker 'nid' sandwich covariance with the Hall-Sheather bandwidth
    (R quantreg::summary.rq(se = 'nid'))."""
    n = len(y)
    b = rq(y, X, tau)
    h = _hs_bandwidth(n, tau)
    if tau + h > 1 or tau - h < 0:
        h = min(tau, 1 - tau) * 0.99
    bu, bl = rq(y, X, tau + h), rq(y, X, tau - h)
    dyhat = X @ (bu - bl)
    eps = np.finfo(float).eps ** (2 / 3)
    f = np.maximum(0, (2 * h) / (dyhat - eps))
    fxxinv = np.linalg.inv((X * f[:, None]).T @ X)
    V = tau * (1 - tau) * fxxinv @ (X.T @ X) @ fxxinv
    return b, V


def qardl(df: pd.DataFrame, y: str, x: list, order: tuple, taus=(0.1, 0.25, 0.5, 0.75, 0.9), case: int = 3) -> dict:
    """QARDL (Cho, Kim & Shin 2015): the ARDL(p, q) in levels estimated at each quantile τ; long-run
    β(τ) = Σθ(τ)/(1 − Σφ(τ)) and speed ρ(τ) = Σφ(τ) − 1, delta-method SE from the nid covariance."""
    base = fit_ardl(df, y, x, order, case)
    X, yv, nm = base.X, base.yv, base.names
    p = order[0]
    iphi = [nm.index(f"L({y}, {i})") for i in range(1, p + 1)]
    rows, coefs = [], []
    for tau in taus:
        b, V = rq_nid(yv, X, tau)
        s = 1 - b[iphi].sum()
        for xn, q in zip(x, order[1:]):
            ith = [nm.index(xn if L == 0 else f"L({xn}, {L})") for L in range(q + 1)]
            th = b[ith].sum()
            lr = th / s
            g = np.zeros(len(b))
            g[ith] = 1 / s
            g[iphi] = th / s ** 2
            se = float(np.sqrt(g @ V @ g))
            rows.append({"τ": tau, "Variable": xn, "Long-run β(τ)": lr, "Std. Err.": se, "t": lr / se,
                         "p-value": 2 * stats.norm.sf(abs(lr / se))})
        g = np.zeros(len(b))
        g[iphi] = 1
        se_r = float(np.sqrt(g @ V @ g))
        rows.append({"τ": tau, "Variable": "ECM speed ρ(τ)", "Long-run β(τ)": -s, "Std. Err.": se_r, "t": -s / se_r,
                     "p-value": 2 * stats.norm.sf(abs(s / se_r))})
        coefs.append(pd.Series(b, index=nm, name=tau))
    return {"long_run": pd.DataFrame(rows), "coef": pd.concat(coefs, axis=1), "ols": base}
