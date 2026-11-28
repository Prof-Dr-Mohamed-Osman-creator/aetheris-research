# ==========================================
# Aetheris Research - Quantile methods (v1.11, Catalogue §1.14)
# Quantile regression (exact LP) with iid / nid / kernel / bootstrap standard errors, the quantile process and
# Wald tests of equal slopes across quantiles (Koenker 2005), panel quantile regression by the method of moments
# (Machado & Santos Silva 2019, MMQR) and the cross-quantilogram (Han, Linton, Oka & Whang 2016).
# ==========================================

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from analytics.ardl import _hs_bandwidth, rq

warnings.filterwarnings("ignore")

SE_TYPES = {"nid": ("Hendricks-Koenker sandwich (nid)", "ساندويتش هندريكس-كوينكر (nid)"), "iid": ("iid (Koenker-Bassett)", "مستقلة متماثلة (iid)"),
            "ker": ("Powell kernel", "نواة باول"), "boot": ("Bootstrap (xy-pairs)", "بوتستراب (أزواج)")}


def _bw(n, tau):
    h = _hs_bandwidth(n, tau)
    while tau - h < 0 or tau + h > 1:
        h /= 2
    return h


def _f_nid(y, X, tau, b):
    h = _bw(len(y), tau)
    bu, bl = rq(y, X, tau + h), rq(y, X, tau - h)
    dyhat = X @ (bu - bl)
    eps = np.finfo(float).eps ** (2 / 3)
    return np.maximum(0, (2 * h) / (dyhat - eps))


def qr_fit(y, X, tau: float, se: str = "nid", boot: int = 200, seed: int = 1) -> dict:
    """Quantile regression at τ with the covariance types of R quantreg::summary.rq: 'iid' (sparsity by the
    Siddiqui difference quotient), 'nid', 'ker' (Powell, Hall-Sheather bandwidth), 'boot' (xy-pairs bootstrap)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    b = rq(y, X, tau)
    r = y - X @ b
    if se == "nid":
        f = _f_nid(y, X, tau, b)
        A = np.linalg.pinv((X * f[:, None]).T @ X)
        V = tau * (1 - tau) * A @ (X.T @ X) @ A
    elif se == "iid":
        eps = np.finfo(float).eps ** (2 / 3)
        pz = int(np.sum(np.abs(r) < eps))
        h = int(max(k + 1, np.ceil(n * _hs_bandwidth(n, tau))))
        ir = np.arange(pz + 1, h + pz + 2)
        ordr = np.sort(r[np.argsort(np.abs(r), kind="stable")][ir - 1])
        xt = ir / (n - k)
        sp = rq(ordr, np.column_stack([np.ones(len(xt)), xt]), 0.5)[1]
        V = sp ** 2 * tau * (1 - tau) * np.linalg.inv(X.T @ X)
    elif se == "ker":
        h = _bw(n, tau)
        hk = (stats.norm.ppf(tau + h) - stats.norm.ppf(tau - h)) * min(np.sqrt(np.var(r, ddof=1)), (np.quantile(r, 0.75) - np.quantile(r, 0.25)) / 1.34)
        f = stats.norm.pdf(r / hk) / hk
        A = np.linalg.pinv((X * f[:, None]).T @ X)
        V = tau * (1 - tau) * A @ (X.T @ X) @ A
    else:
        rng = np.random.default_rng(seed)
        B = np.array([rq(y[i], X[i], tau) for i in (rng.integers(0, n, n) for _ in range(boot))])
        V = np.cov(B.T, ddof=1)
    return {"b": b, "V": V, "resid": r, "tau": tau}


def qr(df: pd.DataFrame, y: str, x: list, tau: float = 0.5, se: str = "nid", boot: int = 200) -> pd.DataFrame:
    d = df[[y] + x].dropna()
    X = np.column_stack([np.ones(len(d)), d[x].to_numpy(float)])
    r = qr_fit(d[y].to_numpy(float), X, tau, se, boot)
    s = np.sqrt(np.diag(r["V"]))
    t = r["b"] / s
    return pd.DataFrame({"Variable": ["const"] + x, "Coef.": r["b"], "Std. Err.": s, "t": t, "p-value": 2 * stats.t.sf(np.abs(t), len(d) - X.shape[1]),
                         "CI low": r["b"] - 1.96 * s, "CI high": r["b"] + 1.96 * s})


def qr_process(df: pd.DataFrame, y: str, x: list, taus=None, se: str = "nid") -> dict:
    """Coefficients across quantiles with 95% bands, OLS for comparison, and pseudo-R¹ (Koenker-Machado)."""
    import statsmodels.api as sm
    taus = np.round(np.arange(0.05, 0.96, 0.05), 2) if taus is None else np.asarray(taus)
    d = df[[y] + x].dropna()
    yv = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d[x].to_numpy(float)])
    rows = []
    for tau in taus:
        r = qr_fit(yv, X, float(tau), se)
        s = np.sqrt(np.diag(r["V"]))
        rho = lambda u, t=tau: np.sum(u * (t - (u < 0)))  # noqa: E731
        r0 = yv - np.quantile(yv, tau)
        r1 = rho(r["resid"]) / rho(r0) if rho(r0) > 0 else np.nan
        for j, nm in enumerate(["const"] + x):
            rows.append({"τ": float(tau), "Variable": nm, "Coef.": r["b"][j], "Std. Err.": s[j], "CI low": r["b"][j] - 1.96 * s[j],
                         "CI high": r["b"][j] + 1.96 * s[j], "pseudo R¹": 1 - r1})
    ols = sm.OLS(yv, X).fit()
    return {"table": pd.DataFrame(rows), "ols": pd.DataFrame({"Variable": ["const"] + x, "Coef.": ols.params, "CI low": ols.conf_int()[:, 0],
                                                              "CI high": ols.conf_int()[:, 1]})}


def slope_equality(df: pd.DataFrame, y: str, x: list, taus=(0.25, 0.5, 0.75)) -> pd.DataFrame:
    """Wald tests that each slope (and all slopes jointly) is equal across the quantiles (Koenker & Bassett 1982), with
    the joint nid covariance Cov(β̂τi, β̂τj) = (min(τi, τj) − τiτj) H_i⁻¹ J H_j⁻¹ (R quantreg::anova.rq, joint)."""
    d = df[[y] + x].dropna()
    yv = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d[x].to_numpy(float)])
    n, k = X.shape
    m = len(taus)
    B, Hi = [], []
    for tau in taus:
        b = rq(yv, X, tau)
        f = _f_nid(yv, X, tau, b)
        B.append(b)
        Hi.append(np.linalg.pinv((X * f[:, None]).T @ X / n))
    J = X.T @ X / n
    V = np.zeros((m * k, m * k))
    for i in range(m):
        for j in range(m):
            V[i * k:(i + 1) * k, j * k:(j + 1) * k] = (min(taus[i], taus[j]) - taus[i] * taus[j]) * Hi[i] @ J @ Hi[j] / n
    bvec = np.concatenate(B)
    rows = []

    def wald(slopes):
        R = []
        for s_ in slopes:
            for i in range(1, m):
                r = np.zeros(m * k)
                r[s_] = 1
                r[i * k + s_] = -1
                R.append(r)
        R = np.array(R)
        v = R @ bvec
        W = float(v @ np.linalg.pinv(R @ V @ R.T) @ v)
        q = R.shape[0]
        return W, q
    for j, nm in enumerate(x, start=1):
        W, q = wald([j])
        rows.append({"Variable": nm, "Wald": W, "df": q, "F": W / q, "p-value": float(stats.f.sf(W / q, q, n * m - k * m))})
    W, q = wald(list(range(1, k)))
    rows.append({"Variable": "all slopes (joint)", "Wald": W, "df": q, "F": W / q, "p-value": float(stats.f.sf(W / q, q, n * m - k * m))})
    return pd.DataFrame(rows)


# ==========================================================
# MMQR — panel quantile regression via moments (Machado & Santos Silva 2019)
# ==========================================================
def mmqr(df: pd.DataFrame, unit: str, y: str, x: list, taus=(0.1, 0.25, 0.5, 0.75, 0.9), boot: int = 200, seed: int = 1) -> dict:
    """Location-scale model Y = α_i + X'β + (δ_i + X'γ)U with individual effects. Location by within (FE) OLS; scale by
    within regression of |R| on X; q(τ) = τ-quantile of the standardised residual; β(τ) = β + q(τ)γ.
    Standard errors by the cluster (unit) bootstrap."""
    d = df[[unit, y] + x].dropna().reset_index(drop=True)
    ids = pd.factorize(d[unit])[0]

    def est(dd, ids_):
        Y = dd[y].to_numpy(float)
        X = dd[x].to_numpy(float)
        Yd = Y - pd.Series(Y).groupby(ids_).transform("mean").to_numpy()
        Xd = X - pd.DataFrame(X).groupby(ids_).transform("mean").to_numpy()
        beta = np.linalg.lstsq(Xd, Yd, rcond=None)[0]
        a_i = pd.Series(Y - X @ beta).groupby(ids_).transform("mean").to_numpy()
        R = Y - a_i - X @ beta
        A = np.abs(R)
        Ad = A - pd.Series(A).groupby(ids_).transform("mean").to_numpy()
        gamma = np.linalg.lstsq(Xd, Ad, rcond=None)[0]
        d_i = pd.Series(A - X @ gamma).groupby(ids_).transform("mean").to_numpy()
        sc = d_i + X @ gamma
        U = R / np.where(np.abs(sc) < 1e-12, np.nan, sc)
        q = np.nanquantile(U, list(taus))
        return beta, gamma, q, np.mean(sc > 0)
    beta, gamma, q, pos = est(d, ids)
    B = np.array([beta + qq * gamma for qq in q])
    se = np.full_like(B, np.nan)
    if boot:
        rng = np.random.default_rng(seed)
        units = np.unique(ids)
        draws = []
        for _ in range(boot):
            pick = rng.choice(units, len(units), replace=True)
            idx = np.concatenate([np.where(ids == u)[0] for u in pick])
            nid = np.concatenate([[j] * np.sum(ids == u) for j, u in enumerate(pick)])
            try:
                b_, g_, q_, _ = est(d.iloc[idx].reset_index(drop=True), nid)
                draws.append(np.array([b_ + qq * g_ for qq in q_]))
            except Exception:  # noqa: BLE001
                continue
        se = np.std(np.array(draws), axis=0, ddof=1)
    rows = []
    for i, tau in enumerate(taus):
        for j, v in enumerate(x):
            z = B[i, j] / se[i, j] if np.isfinite(se[i, j]) and se[i, j] > 0 else np.nan
            rows.append({"τ": tau, "Variable": v, "Coef.": B[i, j], "Std. Err. (bootstrap)": se[i, j], "z": z, "p-value": 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan})
    loc_scale = pd.DataFrame({"Variable": x, "Location β": beta, "Scale γ": gamma})
    return {"table": pd.DataFrame(rows), "location_scale": loc_scale, "q": dict(zip(taus, q)), "positive scale %": 100 * pos, "N": len(d)}


# ==========================================================
# Cross-quantilogram (Han, Linton, Oka & Whang 2016)
# ==========================================================
def cross_quantilogram(x, y, a1: float, a2: float, max_lag: int = 10, boot: int = 500, block: float | None = None,
                       seed: int = 1, alpha: float = 0.05) -> pd.DataFrame:
    """ρ_α(k) = corr(ψ_α1(x_t − q_x), ψ_α2(y_{t−k} − q_y)), ψ_a(u) = 1(u < 0) − a: does y being in its α2-quantile
    tail k periods earlier predict x being in its α1-quantile tail? Portmanteau Q(p) = T Σ ρ²(k) (Box-Ljung form);
    confidence bands and Q critical values by the stationary bootstrap (Politis-Romano), centred at the estimate."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    T = len(x)

    def cq(xx, yy):
        qx, qy = np.quantile(xx, a1), np.quantile(yy, a2)
        px = (xx < qx) - a1
        py = (yy < qy) - a2
        out = []
        for k in range(1, max_lag + 1):
            a, b = px[k:], py[:-k]
            out.append(np.sum(a * b) / np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)))
        return np.array(out)
    rho = cq(x, y)
    n_k = T - np.arange(1, max_lag + 1)
    Q = T * (T + 2) * np.cumsum(rho ** 2 / n_k)
    rng = np.random.default_rng(seed)
    gamma = 1 / (block or max(1.0, T ** (1 / 3)))
    R = np.zeros((boot, max_lag))
    QB = np.zeros((boot, max_lag))
    for b in range(boot):
        idx = np.empty(T, int)
        i = rng.integers(T)
        for t in range(T):
            if t > 0 and rng.random() > gamma:
                i = (i + 1) % T
            else:
                i = rng.integers(T) if t > 0 else i
            idx[t] = i
        rb = cq(x[idx], y[idx])
        R[b] = rb - rho
        QB[b] = T * (T + 2) * np.cumsum((rb - rho) ** 2 / n_k)
    lo = rho + np.quantile(R, alpha / 2, 0)
    hi = rho + np.quantile(R, 1 - alpha / 2, 0)
    band_lo, band_hi = np.quantile(R, alpha / 2, 0), np.quantile(R, 1 - alpha / 2, 0)
    qcrit = np.quantile(QB, 1 - alpha, 0)
    return pd.DataFrame({"lag": np.arange(1, max_lag + 1), "ρ": rho, "band low (H0)": band_lo, "band high (H0)": band_hi,
                         "CI low": lo, "CI high": hi, "significant": (rho < band_lo) | (rho > band_hi), "Q(p)": Q, "Q crit 5%": qcrit,
                         "p (Q)": [float(np.mean(QB[:, j] >= Q[j])) for j in range(max_lag)]})
