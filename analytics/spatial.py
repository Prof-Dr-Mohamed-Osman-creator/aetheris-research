# ==========================================
# Aetheris Research - Spatial econometrics (v1.11, Catalogue §1.16)
# Spatial weights, global and local autocorrelation (Moran's I, Geary's C, LISA), LM diagnostics, maximum-likelihood
# SAR / SEM / SDM / SAC / SLX with LeSage-Pace impacts, geographically weighted regression and fixed-effects spatial
# panels (Elhorst 2003). Conventions follow R spdep / spatialreg and PySAL.
# ==========================================

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore")

W_KINDS = {"knn": ("k nearest neighbours", "أقرب k جيران"), "band": ("Distance band", "نطاق المسافة"),
           "idw": ("Inverse distance (within a band)", "مقلوب المسافة (داخل نطاق)")}


def _dist(xy, geo: bool):
    xy = np.asarray(xy, float)
    if geo:          # great-circle distance in km, xy = (lon, lat)
        lon, lat = np.radians(xy[:, 0]), np.radians(xy[:, 1])
        a = (np.sin((lat[:, None] - lat[None, :]) / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin((lon[:, None] - lon[None, :]) / 2) ** 2)
        return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))


def weights(xy, kind: str = "knn", k: int = 6, band: float | None = None, power: float = 1.0, geo: bool = False,
            row_standardize: bool = True) -> np.ndarray:
    """Spatial weights matrix from coordinates. Distances in the coordinate units (or km with `geo`); the default band
    is the smallest distance that gives every unit at least one neighbour."""
    D = _dist(xy, geo)
    n = len(D)
    np.fill_diagonal(D, np.inf)
    if kind == "knn":
        W = np.zeros((n, n))
        idx = np.argsort(D, 1, kind="stable")[:, :k]
        W[np.arange(n)[:, None], idx] = 1.0
    else:
        b = band if band else float(np.max(np.min(D, 1))) * 1.0000001
        W = (D <= b).astype(float)
        if kind == "idw":
            with np.errstate(divide="ignore"):
                W = W / D ** power
    np.fill_diagonal(W, 0)
    if row_standardize:
        rs = W.sum(1, keepdims=True)
        W = np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)
    return W


def describe_w(W) -> dict:
    nn = (W > 0).sum(1)
    return {"N": len(W), "mean neighbours": float(nn.mean()), "min neighbours": int(nn.min()), "max neighbours": int(nn.max()),
            "islands": int((nn == 0).sum()), "% non-zero": float(100 * (W > 0).mean()), "symmetric": bool(np.allclose(W > 0, (W > 0).T))}


# ==========================================================
# Global and local autocorrelation
# ==========================================================
def moran(y, W, perms: int = 999, seed: int = 1) -> dict:
    """Global Moran's I with expectation and variance under normality and randomisation (Cliff & Ord), z-scores and a
    permutation pseudo p-value (as spdep::moran.test / moran.mc and esda.Moran)."""
    y = np.asarray(y, float)
    n = len(y)
    z = y - y.mean()
    S0 = W.sum()
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(1) + W.sum(0)) ** 2)
    I = n / S0 * (z @ W @ z) / (z @ z)
    EI = -1 / (n - 1)
    VN = (n ** 2 * S1 - n * S2 + 3 * S0 ** 2) / ((n ** 2 - 1) * S0 ** 2) - EI ** 2
    b2 = n * np.sum(z ** 4) / (np.sum(z ** 2) ** 2)
    VR = (n * ((n ** 2 - 3 * n + 3) * S1 - n * S2 + 3 * S0 ** 2) - b2 * ((n ** 2 - n) * S1 - 2 * n * S2 + 6 * S0 ** 2)) / ((n - 1) * (n - 2) * (n - 3) * S0 ** 2) - EI ** 2
    rng = np.random.default_rng(seed)
    sims = np.array([n / S0 * (zz @ W @ zz) / (zz @ zz) for zz in (rng.permutation(z) for _ in range(perms))]) if perms else np.array([])
    p_sim = (np.sum(sims >= I) + 1) / (perms + 1) if I >= np.median(sims if perms else [EI]) else (np.sum(sims <= I) + 1) / (perms + 1)
    zN, zR = (I - EI) / np.sqrt(VN), (I - EI) / np.sqrt(VR)
    return {"I": float(I), "E[I]": EI, "Var (normality)": float(VN), "Var (randomisation)": float(VR), "z (normality)": float(zN),
            "z (randomisation)": float(zR), "p (randomisation, two-sided)": float(2 * stats.norm.sf(abs(zR))),
            "p (randomisation, greater)": float(stats.norm.sf(zR)), "p (permutation)": float(p_sim) if perms else np.nan, "sims": sims}


def geary(y, W) -> dict:
    """Geary's C with its randomisation variance (spdep::geary.test)."""
    y = np.asarray(y, float)
    n = len(y)
    z = y - y.mean()
    S0 = W.sum()
    S1 = 0.5 * np.sum((W + W.T) ** 2)
    S2 = np.sum((W.sum(1) + W.sum(0)) ** 2)
    C = (n - 1) * np.sum(W * (y[:, None] - y[None, :]) ** 2) / (2 * S0 * (z @ z))
    b2 = n * np.sum(z ** 4) / (np.sum(z ** 2) ** 2)
    V = ((n - 1) * S1 * (n ** 2 - 3 * n + 3 - (n - 1) * b2) - 0.25 * (n - 1) * S2 * (n ** 2 + 3 * n - 6 - (n ** 2 - n + 2) * b2)
         + S0 ** 2 * (n ** 2 - 3 - (n - 1) ** 2 * b2)) / (n * (n - 2) * (n - 3) * S0 ** 2)
    zc = (1 - C) / np.sqrt(V)
    return {"C": float(C), "E[C]": 1.0, "Var": float(V), "z": float(zc), "p (greater)": float(stats.norm.sf(zc))}


def lisa(y, W, perms: int = 999, seed: int = 1, alpha: float = 0.05) -> pd.DataFrame:
    """Local Moran I_i = z_i Σ_j w_ij z_j / m2 (m2 = Σz²/n, as esda.Moran_Local) with conditional-permutation pseudo
    p-values (one-sided, folded as esda), and the HH / LL / HL / LH cluster classes at level α."""
    y = np.asarray(y, float)
    n = len(y)
    z = (y - y.mean()) / y.std()
    lag = W @ z
    m2 = np.sum(z ** 2) / n
    Ii = z * lag / m2
    rng = np.random.default_rng(seed)
    kmax = int((W > 0).sum(1).max())
    larger = np.zeros(n)
    for i in range(n):
        wi = W[i][W[i] > 0]
        others = np.delete(z, i)
        idx = np.array([rng.choice(n - 1, len(wi), replace=False) for _ in range(perms)])
        lag_sim = others[idx] @ wi
        Isim = z[i] * lag_sim / m2
        larger[i] = np.sum(Isim >= Ii[i])
    low = perms - larger
    extreme = np.minimum(larger, low)
    p = (extreme + 1) / (perms + 1)
    q = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    cl = np.where(p < alpha, q, "not significant")
    del kmax
    return pd.DataFrame({"Ii": Ii, "z": z, "lag z": lag, "p (permutation)": p, "quadrant": q, "cluster": cl})


# ==========================================================
# LM diagnostics on OLS residuals (Anselin 1988; Anselin et al. 1996)
# ==========================================================
def lm_tests(y, X, W) -> pd.DataFrame:
    y, X = np.asarray(y, float), np.asarray(X, float)
    n = len(y)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    s2 = e @ e / n
    T = np.trace(W.T @ W + W @ W)
    M = np.eye(n) - X @ np.linalg.pinv(X)
    WXb = W @ X @ b
    J = (WXb @ M @ WXb + T * s2) / (n * s2)
    lm_err = (e @ W @ e / s2) ** 2 / T
    lm_lag = (e @ W @ y / s2) ** 2 / (n * J)
    rlm_err = (e @ W @ e / s2 - T / (n * J) * (e @ W @ y / s2)) ** 2 / (T * (1 - T / (n * J)))
    rlm_lag = (e @ W @ y / s2 - e @ W @ e / s2) ** 2 / (n * J - T)
    sarma = (e @ W @ y / s2 - e @ W @ e / s2) ** 2 / (n * J - T) + lm_err
    rows = [("LM error", lm_err, 1), ("LM lag", lm_lag, 1), ("Robust LM error", rlm_err, 1), ("Robust LM lag", rlm_lag, 1), ("SARMA", sarma, 2)]
    return pd.DataFrame([{"Test": t, "Statistic": float(v), "df": d, "p-value": float(stats.chi2.sf(v, d))} for t, v, d in rows])


# ==========================================================
# Maximum-likelihood spatial regressions
# ==========================================================
MODELS = {"ols": ("OLS (no spatial terms)", "OLS (بلا حدود مكانية)"), "slx": ("SLX (spatially lagged X)", "SLX (المتغيرات المفسرة المبطأة مكانياً)"),
          "sar": ("SAR (spatial lag)", "SAR (الإبطاء المكاني)"), "sem": ("SEM (spatial error)", "SEM (الخطأ المكاني)"),
          "sdm": ("SDM (spatial Durbin)", "SDM (دوربن المكاني)"), "sdem": ("SDEM (spatial Durbin error)", "SDEM (دوربن للخطأ المكاني)"),
          "sac": ("SAC / SARAR (lag + error)", "SAC / SARAR (إبطاء + خطأ)")}


class _LogDet:
    def __init__(self, W):
        self.ev = np.linalg.eigvals(W)
        r = self.ev[np.abs(self.ev.imag) < 1e-10].real
        self.lo = 1 / r.min() if r.min() < 0 else -1.0
        self.hi = 1 / r.max()

    def __call__(self, rho):
        return float(np.sum(np.log(np.abs(1 - rho * self.ev))))


def _ols_conc(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return b, e


def spatial_model(df: pd.DataFrame, y: str, x: list, W, model: str = "sar", impacts_draws: int = 1000, seed: int = 1) -> dict:
    """Maximum likelihood (eigenvalue log-determinant, as spatialreg method = 'eigen') for SAR, SEM, SDM, SDEM and SAC;
    OLS / SLX by least squares. Standard errors from the numerical Hessian of the full log-likelihood. Direct,
    indirect and total impacts (LeSage & Pace 2009) for SAR / SDM / SAC with simulated standard errors."""
    d = df[[y] + x].dropna()
    if len(d) != len(W):
        raise ValueError("Missing values: the weights matrix must match the complete observations.")
    yv = d[y].to_numpy(float)
    Xr = d[x].to_numpy(float)
    n = len(yv)
    X = np.column_stack([np.ones(n), Xr])
    names = ["const"] + x
    if model in ("slx", "sdm", "sdem"):
        X = np.column_stack([X, W @ Xr])
        names += [f"W·{v}" for v in x]
    k = X.shape[1]
    ld = _LogDet(W)
    I = np.eye(n)
    Wy = W @ yv

    def ll_full(th):
        rho = th[k] if model in ("sar", "sdm", "sac") else 0.0
        lam = th[k + (model == "sac")] if model in ("sem", "sdem", "sac") else 0.0
        s2 = np.exp(th[-1])
        b = th[:k]
        u = yv - rho * Wy - X @ b
        e = u - lam * (W @ u)
        return -n / 2 * np.log(2 * np.pi * s2) - e @ e / (2 * s2) + ld(rho) + ld(lam)
    if model in ("ols", "slx"):
        b, e = _ols_conc(yv, X)
        s2 = e @ e / (n - k)
        V = s2 * np.linalg.inv(X.T @ X)
        ll = -n / 2 * (np.log(2 * np.pi * e @ e / n) + 1)
        theta = np.r_[b, np.log(e @ e / n)]
        par_names = names
        est = b
        se = np.sqrt(np.diag(V))
        rho = lam = 0.0
    else:
        if model in ("sar", "sdm"):
            def conc(rho):
                b, e = _ols_conc(yv - rho * Wy, X)
                return -(-n / 2 * np.log(e @ e / n) + ld(rho))
            rho = optimize.minimize_scalar(conc, bounds=(ld.lo + 1e-6, ld.hi - 1e-6), method="bounded", options={"xatol": 1e-10}).x
            b, e = _ols_conc(yv - rho * Wy, X)
            th0 = np.r_[b, rho, np.log(e @ e / n)]
        elif model in ("sem", "sdem"):
            def conc(lam):
                Xs, ys = X - lam * (W @ X), yv - lam * Wy
                b, e = _ols_conc(ys, Xs)
                return -(-n / 2 * np.log(e @ e / n) + ld(lam))
            lam = optimize.minimize_scalar(conc, bounds=(ld.lo + 1e-6, ld.hi - 1e-6), method="bounded", options={"xatol": 1e-10}).x
            b, e = _ols_conc(yv - lam * Wy, X - lam * (W @ X))
            th0 = np.r_[b, lam, np.log(e @ e / n)]
        else:
            def conc2(p):
                rho, lam = p
                if not (ld.lo < rho < ld.hi and ld.lo < lam < ld.hi):
                    return 1e10
                A = yv - rho * Wy
                Xs, ys = X - lam * (W @ X), A - lam * (W @ A)
                b, e = _ols_conc(ys, Xs)
                return -(-n / 2 * np.log(e @ e / n) + ld(rho) + ld(lam))
            best = None
            for st in ((0.2, 0.2), (0.5, 0.0), (0.0, 0.5), (0.4, -0.3)):
                r = optimize.minimize(conc2, st, method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 5000})
                if best is None or r.fun < best.fun:
                    best = r
            rho, lam = best.x
            A = yv - rho * Wy
            b, e = _ols_conc(A - lam * (W @ A), X - lam * (W @ X))
            th0 = np.r_[b, rho, lam, np.log(e @ e / n)]
        from statsmodels.tools.numdiff import approx_hess
        s2_ = float(np.exp(th0[-1]))
        if model in ("sar", "sdm"):
            # analytical asymptotic information (Anselin 1988), as spatialreg::lagsarlm
            rr = th0[k]
            WA = W @ np.linalg.inv(I - rr * W)
            WAXb = WA @ X @ th0[:k]
            Info = np.zeros((k + 2, k + 2))
            Info[:k, :k] = X.T @ X / s2_
            Info[:k, k] = Info[k, :k] = X.T @ WAXb / s2_
            Info[k, k] = np.trace(WA @ WA) + np.trace(WA.T @ WA) + WAXb @ WAXb / s2_
            Info[k, k + 1] = Info[k + 1, k] = np.trace(WA) / s2_
            Info[k + 1, k + 1] = n / (2 * s2_ ** 2)
            V = np.linalg.inv(Info)
            V[-1, -1] = V[-1, -1] / s2_ ** 2          # report on the ln σ² scale (not used for inference)
        elif model in ("sem", "sdem"):
            ll_ = th0[k]
            Bm = I - ll_ * W
            WB = W @ np.linalg.inv(Bm)
            Xs = Bm @ X
            V = np.zeros((k + 2, k + 2))
            V[:k, :k] = s2_ * np.linalg.inv(Xs.T @ Xs)
            Iw = np.array([[np.trace(WB @ WB) + np.trace(WB.T @ WB), np.trace(WB) / s2_], [np.trace(WB) / s2_, n / (2 * s2_ ** 2)]])
            Vw = np.linalg.inv(Iw)
            V[k, k] = Vw[0, 0]
            V[k + 1, k + 1] = Vw[1, 1] / s2_ ** 2
        else:
            H = approx_hess(th0, lambda t: -ll_full(t))      # SAC: numerical Hessian, as spatialreg::sacsarlm
            V = np.linalg.inv(H)
        theta = th0
        ll = ll_full(th0)
        sp = (["ρ"] if model in ("sar", "sdm", "sac") else []) + (["λ"] if model in ("sem", "sdem", "sac") else [])
        par_names = names + sp
        est = th0[:-1]
        se = np.sqrt(np.diag(V))[:-1]
        V = V[:-1, :-1]
        rho = th0[k] if model in ("sar", "sdm", "sac") else 0.0
        lam = th0[k + (model == "sac")] if model in ("sem", "sdem", "sac") else 0.0
    z = est / se
    tab = pd.DataFrame({"Parameter": par_names, "Coef.": est, "Std. Err.": se, "z": z, "p-value": 2 * stats.norm.sf(np.abs(z))})
    npar = len(est) + 1
    out = {"table": tab, "loglik": float(ll), "AIC": float(-2 * ll + 2 * npar), "BIC": float(-2 * ll + np.log(n) * npar), "rho": float(rho),
           "lambda": float(lam), "sigma2": float(np.exp(theta[-1])), "N": n, "model": model, "names": names, "V": V, "est": est, "k": k}
    if model in ("sar", "sdm", "sac", "slx"):
        out["impacts"] = impacts(est, V, W, x, model, k, impacts_draws, seed)
    # residual Moran
    if model in ("ols", "slx", "sar", "sdm"):
        res = yv - rho * Wy - X @ est[:k]
        out["residual Moran"] = moran(res, W, perms=0)
    return out


def impacts(est, V, W, x, model, k, draws=1000, seed=1) -> pd.DataFrame:
    """LeSage-Pace summary impacts: S_r(W) = (I − ρW)⁻¹(β_r I + θ_r W); direct = mean diagonal, total = mean row sum,
    indirect = total − direct. Standard errors by drawing (β, θ, ρ) from their asymptotic normal distribution."""
    n = len(W)
    nx = len(x)
    ev = np.linalg.eigvals(W)
    trW = [np.real(np.sum(ev ** p)) for p in range(0, 101)]

    def summ(par):
        rho = par[k] if model in ("sar", "sdm", "sac") else 0.0
        b = par[1:1 + nx]
        th = par[1 + nx:1 + 2 * nx] if model in ("sdm", "slx") else np.zeros(nx)
        if model == "slx":
            return b, b + th
        # traces of (I − ρW)⁻¹ and (I − ρW)⁻¹W by power series (spatialreg traces approach)
        rp = rho ** np.arange(101)
        tr0 = np.sum(rp * trW) / n
        tr1 = np.sum(rp[:100] * trW[1:101]) / n
        direct = b * tr0 + th * tr1
        total = (b + th) / (1 - rho)
        return direct, total
    d0, t0 = summ(est)
    rng = np.random.default_rng(seed)
    try:
        L = np.linalg.cholesky(V)
        sims = [summ(est + L @ rng.standard_normal(len(est))) for _ in range(draws)]
        D = np.array([s[0] for s in sims])
        Tt = np.array([s[1] for s in sims])
        sd, st_, si = D.std(0, ddof=1), Tt.std(0, ddof=1), (Tt - D).std(0, ddof=1)
    except Exception:  # noqa: BLE001
        sd = st_ = si = np.full(nx, np.nan)
    rows = []
    for j, v in enumerate(x):
        for typ, val, s in (("Direct", d0[j], sd[j]), ("Indirect (spillover)", t0[j] - d0[j], si[j]), ("Total", t0[j], st_[j])):
            zz = val / s if s and np.isfinite(s) and s > 0 else np.nan
            rows.append({"Variable": v, "Impact": typ, "Estimate": float(val), "Std. Err. (simulated)": float(s), "z": zz,
                         "p-value": float(2 * stats.norm.sf(abs(zz))) if np.isfinite(zz) else np.nan})
    return pd.DataFrame(rows)


# ==========================================================
# Geographically weighted regression
# ==========================================================
def gwr(df: pd.DataFrame, y: str, x: list, coords, kernel: str = "bisquare", bw: float | None = None, adaptive: bool = True,
        criterion: str = "AICc") -> dict:
    """GWR (Fotheringham, Brunsdon & Charlton 2002) with an adaptive (number of neighbours) or fixed (distance) Gaussian /
    bisquare kernel; bandwidth by golden-section search on AICc (or CV), as mgwr. Local coefficients, local SEs and R²."""
    d = df[[y] + x].dropna()
    yv = d[y].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d[x].to_numpy(float)])
    n, k = X.shape
    C = np.asarray(coords, float)
    D = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))

    def wts(b):
        if adaptive:
            bi = np.sort(D, 1)[:, int(b) - 1][:, None]
        else:
            bi = b
        u = D / bi
        if kernel == "gaussian":
            return np.exp(-0.5 * u ** 2)
        return np.where(u < 1, (1 - u ** 2) ** 2, 0.0)

    def fit(b, full=False):
        Wk = wts(b)
        B = np.zeros((n, k))
        S = np.zeros(n)
        CC = [] if full else None
        for i in range(n):
            w = Wk[i]
            XtW = X.T * w
            A = np.linalg.inv(XtW @ X)
            Ci = A @ XtW
            B[i] = Ci @ yv
            S[i] = X[i] @ Ci[:, i]
            if full:
                CC.append(Ci)
        yhat = np.sum(X * B, 1)
        e = yv - yhat
        rss = e @ e
        trS = S.sum()
        sig2 = rss / n
        aicc = 2 * n * np.log(np.sqrt(sig2)) + n * np.log(2 * np.pi) + n * (n + trS) / (n - 2 - trS)
        cv = np.sum((e / (1 - S)) ** 2)
        return {"B": B, "e": e, "trS": trS, "AICc": aicc, "CV": cv, "rss": rss, "C": CC, "yhat": yhat}
    if bw is None:
        if adaptive:
            lo, hi = k + 2, n - 1
            golden = (np.sqrt(5) - 1) / 2
            a, b_ = lo, hi
            c = int(round(b_ - golden * (b_ - a)))
            dd = int(round(a + golden * (b_ - a)))
            cache = {}

            def score(v):
                if v not in cache:
                    cache[v] = fit(v)[criterion]
                return cache[v]
            while b_ - a > 1:
                if score(c) <= score(dd):
                    b_ = dd
                else:
                    a = c
                c = int(round(b_ - golden * (b_ - a)))
                dd = int(round(a + golden * (b_ - a)))
                if c == dd:
                    break
            cand = range(max(lo, a - 1), min(hi, b_ + 1) + 1)
            bw = min(cand, key=score)
        else:
            r = optimize.minimize_scalar(lambda v: fit(v)[criterion], bounds=(np.percentile(D[D > 0], 1), D.max()), method="bounded")
            bw = float(r.x)
    f = fit(bw, full=True)
    sig2 = f["rss"] / (n - f["trS"])
    SE = np.array([np.sqrt(np.diag(Ci @ Ci.T) * sig2) for Ci in f["C"]])
    tss = np.sum((yv - yv.mean()) ** 2)
    cols = ["const"] + x
    local = pd.DataFrame(f["B"], columns=cols)
    for j, c in enumerate(cols):
        local[f"se {c}"] = SE[:, j]
        local[f"t {c}"] = f["B"][:, j] / SE[:, j]
    summ = pd.DataFrame({"Variable": cols, "Mean": f["B"].mean(0), "SD": f["B"].std(0, ddof=1), "Min": f["B"].min(0), "Median": np.median(f["B"], 0),
                         "Max": f["B"].max(0), "% |t| > 1.96": 100 * (np.abs(f["B"] / SE) > 1.96).mean(0)})
    return {"bandwidth": bw, "AICc": float(f["AICc"]), "R²": float(1 - f["rss"] / tss), "effective parameters": float(f["trS"]),
            "local": local, "summary": summ, "sigma2": float(sig2), "adaptive": adaptive, "kernel": kernel}


# ==========================================================
# Spatial panels with fixed effects (Elhorst 2003, 2014)
# ==========================================================
def spatial_panel(df: pd.DataFrame, unit: str, time: str, y: str, x: list, W, model: str = "sar", effects: str = "unit") -> dict:
    """Fixed-effects spatial-lag or spatial-error panel by concentrated ML after the within transformation (Elhorst
    2003): log-likelihood −NT/2 ln(2πσ²) + T ln|I − ρW| − e'e/(2σ²). The panel must be balanced and sorted consistently
    with W (units in the order of their first appearance). effects: 'unit' or 'twoway'."""
    d = df[[unit, time, y] + x].dropna().sort_values([time, unit])
    units = pd.unique(df[unit])
    N = len(units)
    T = d[time].nunique()
    if len(d) != N * T:
        raise ValueError("The spatial panel must be balanced.")
    d = d.set_index([time, unit]).loc[[(t, u) for t in sorted(d[time].unique()) for u in units]].reset_index()
    Y = d[y].to_numpy(float).reshape(T, N)
    Xa = d[x].to_numpy(float).reshape(T, N, len(x))

    def demean(A):
        A = A - A.mean(0, keepdims=True)
        if effects == "twoway":
            A = A - A.mean(1, keepdims=True)
        return A
    Yd, Xd = demean(Y), demean(Xa)
    WY = np.einsum("ij,tj->ti", W, Yd)
    yv, wy = Yd.ravel(), WY.ravel()
    X = Xd.reshape(N * T, len(x))
    WX = np.einsum("ij,tjk->tik", W, Xd).reshape(N * T, len(x))
    ld = _LogDet(W)
    NT = N * T
    if model == "sar":
        def conc(rho):
            b, e = _ols_conc(yv - rho * wy, X)
            return -(-NT / 2 * np.log(e @ e / NT) + T * ld(rho))
        par = optimize.minimize_scalar(conc, bounds=(ld.lo + 1e-6, ld.hi - 1e-6), method="bounded", options={"xatol": 1e-10}).x
        b, e = _ols_conc(yv - par * wy, X)
    else:
        def conc(lam):
            b, e = _ols_conc(yv - lam * wy, X - lam * WX)
            return -(-NT / 2 * np.log(e @ e / NT) + T * ld(lam))
        par = optimize.minimize_scalar(conc, bounds=(ld.lo + 1e-6, ld.hi - 1e-6), method="bounded", options={"xatol": 1e-10}).x
        b, e = _ols_conc(yv - par * wy, X - par * WX)
    s2 = e @ e / NT

    def llf(th):
        bb, p, ls2 = th[:-2], th[-2], th[-1]
        if model == "sar":
            ee = yv - p * wy - X @ bb
        else:
            ee = (yv - p * wy) - (X - p * WX) @ bb
        return -NT / 2 * np.log(2 * np.pi) - NT / 2 * ls2 - ee @ ee / (2 * np.exp(ls2)) + T * ld(p)
    th = np.r_[b, par, np.log(s2)]
    from statsmodels.tools.numdiff import approx_hess
    V = np.linalg.inv(approx_hess(th, lambda t: -llf(t)))
    est = th[:-1]
    se = np.sqrt(np.diag(V))[:-1]
    nm = list(x) + (["ρ"] if model == "sar" else ["λ"])
    tab = pd.DataFrame({"Parameter": nm, "Coef.": est, "Std. Err.": se, "z": est / se, "p-value": 2 * stats.norm.sf(np.abs(est / se))})
    out = {"table": tab, "loglik": float(llf(th)), "sigma2": float(s2), "N": N, "T": T, "param": float(par)}
    if model == "sar":
        m = len(est)
        Vf = np.zeros((m + 1, m + 1))
        Vf[1:, 1:] = V[:-1, :-1]
        Vf[0, 0] = 1e-12
        out["impacts"] = impacts(np.r_[0.0, est], Vf, W, x, "sar", 1 + len(x), 500)
    return out
