# ==========================================
# Aetheris Research - Multivariate time series (Catalogue §1.8)
# VAR (lag selection, estimation, stability, residual tests, Granger / Toda-Yamamoto causality, IRF, FEVD,
# forecasts), structural VAR (A / B models by scoring ML, Blanchard-Quah), cointegration (Engle-Granger,
# Johansen with the five deterministic cases, Gregory-Hansen), VECM (β normalisation, α, Γ, restriction tests,
# levels-VAR representation) and single-equation cointegrating regressions (FMOLS, DOLS, CCR).
# Conventions follow R vars / urca / cointReg so results can be reproduced line by line.
# ==========================================

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import warnings

import numpy as np
import pandas as pd
from scipy import stats

CASES = {1: ("No deterministic terms", "بدون حدود محددة"),
         2: ("Restricted constant (in the cointegrating relation)", "ثابت مقيد (داخل علاقة التكامل)"),
         3: ("Unrestricted constant", "ثابت غير مقيد"),
         4: ("Restricted trend + unrestricted constant", "اتجاه مقيد وثابت غير مقيد"),
         5: ("Unrestricted constant and trend", "ثابت واتجاه غير مقيدين")}
VTYPES = {"const": ("Constant", "ثابت"), "both": ("Constant + trend", "ثابت واتجاه"),
          "trend": ("Trend only", "اتجاه فقط"), "none": ("None", "بدون")}


# ==========================================================
# VAR
# ==========================================================
@dataclass
class VarResult:
    names: list
    p: int
    type: str
    y: np.ndarray            # full sample, T × K
    Z: np.ndarray            # regressors, obs × m
    reg_names: list
    B: np.ndarray            # m × K coefficients (column k = equation k)
    se: np.ndarray           # m × K
    resid: np.ndarray        # obs × K
    obs: int
    index: object = None
    season: int | None = None
    exog_names: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def K(self):
        return len(self.names)

    @property
    def df(self):
        return self.obs - self.Z.shape[1]

    @property
    def sigma_u(self):              # df-adjusted (vars: crossprod(resid) / (obs − m))
        return self.resid.T @ self.resid / self.df

    @property
    def sigma_ml(self):
        return self.resid.T @ self.resid / self.obs

    def A(self, i: int) -> np.ndarray:
        """Lag-i coefficient matrix (K × K), rows = equations."""
        K = self.K
        return self.B[(i - 1) * K:i * K, :].T

    @property
    def det_names(self):
        return self.reg_names[self.K * self.p:]

    def table(self) -> pd.DataFrame:
        rows = []
        tq = stats.t.ppf(0.975, self.df)
        for k, eq in enumerate(self.names):
            for j, rn in enumerate(self.reg_names):
                b, s = self.B[j, k], self.se[j, k]
                tv = b / s if s > 0 else np.nan
                rows.append({"Equation": eq, "Regressor": rn, "Coef.": b, "Std. Err.": s, "t": tv,
                             "p-value": 2 * stats.t.sf(abs(tv), self.df), "CI low": b - tq * s, "CI high": b + tq * s})
        return pd.DataFrame(rows)


def _season_dummies(T: int, s: int) -> np.ndarray:
    dum = (np.eye(s) - 1.0 / s)[:, :-1]
    reps = int(np.ceil(T / s))
    return np.tile(dum, (reps, 1))[:T]


def _det(T_full: int, p: int, type_: str, season: int | None, exog: np.ndarray | None, exog_names=None):
    obs = T_full - p
    cols, names = [], []
    if type_ in ("const", "both"):
        cols.append(np.ones(obs)); names.append("const")
    if type_ in ("trend", "both"):
        cols.append(np.arange(p + 1, T_full + 1, dtype=float)); names.append("trend")
    if season:
        d = _season_dummies(T_full, season)[p:]
        cols += list(d.T); names += [f"sd{i + 1}" for i in range(d.shape[1])]
    if exog is not None and exog.size:
        e = np.asarray(exog, float)[p:]
        e = e.reshape(obs, -1)
        cols += list(e.T); names += list(exog_names or [f"exo{i + 1}" for i in range(e.shape[1])])
    return (np.column_stack(cols) if cols else np.empty((obs, 0))), names


def _lags(y: np.ndarray, p: int) -> np.ndarray:
    T, K = y.shape
    return np.column_stack([y[p - i:T - i] for i in range(1, p + 1)]) if p else np.empty((T, 0))


def fit_var(Y, p: int = 1, type: str = "const", season: int | None = None, exog=None, exog_names=None,
            names=None, index=None) -> VarResult:
    """Equation-by-equation OLS of a VAR(p) — identical to R vars::VAR (regressor order y.l1 … y.lp, const, trend,
    seasonal dummies, exogenous)."""
    y = np.asarray(Y, float)
    T, K = y.shape
    if T - p <= K * p + 2:
        raise ValueError("Too few observations for this lag order.")
    names = list(names) if names is not None else (list(Y.columns) if hasattr(Y, "columns") else [f"y{i + 1}" for i in range(K)])
    X = _lags(y, p)
    D, dn = _det(T, p, type, season, exog, exog_names)
    Z = np.column_stack([X, D])
    reg = [f"{n}.l{i}" for i in range(1, p + 1) for n in names] + dn
    Yt = y[p:]
    ZtZi = np.linalg.inv(Z.T @ Z)
    B = ZtZi @ Z.T @ Yt
    U = Yt - Z @ B
    obs, m = Z.shape
    s2 = (U ** 2).sum(0) / (obs - m)
    se = np.sqrt(np.outer(np.diag(ZtZi), s2))
    r = VarResult(names, p, type, y, Z, reg, B, se, U, obs, index=index, season=season,
                  exog_names=list(exog_names or []))
    Sml = r.sigma_ml
    ld = np.linalg.slogdet(Sml)[1]
    r.stats["Log-likelihood"] = float(-(K * obs / 2) * np.log(2 * np.pi) - obs / 2 * ld - K * obs / 2)
    npar = K * m
    r.stats.update({"AIC": ld + 2 * npar / obs, "SC (BIC)": ld + np.log(obs) * npar / obs,
                    "HQ": ld + 2 * np.log(np.log(obs)) * npar / obs, "N": obs, "Parameters": npar})
    eq = []
    for k in range(K):
        yk = Yt[:, k]
        tss = ((yk - yk.mean()) ** 2).sum() if "const" in dn else (yk ** 2).sum()
        rss = (U[:, k] ** 2).sum()
        r2 = 1 - rss / tss
        dfm = m - (1 if "const" in dn else 0)
        F = (r2 / dfm) / ((1 - r2) / (obs - m)) if dfm > 0 else np.nan
        eq.append({"Equation": names[k], "R²": r2, "Adj. R²": 1 - (1 - r2) * (obs - (1 if "const" in dn else 0)) / (obs - m),
                   "S.E. of regression": np.sqrt(rss / (obs - m)), "F": F, "p(F)": stats.f.sf(F, dfm, obs - m)})
    r.stats["equations"] = pd.DataFrame(eq)
    mod = roots(r)
    if mod.max() >= 1:
        r.notes.append("The VAR is not stable (a companion root has modulus ≥ 1): impulse responses do not die out — "
                       "difference the data or estimate a VECM.")
    return r


def var_select(Y, lag_max: int = 8, type: str = "const", season: int | None = None, exog=None) -> pd.DataFrame:
    """Lag-order criteria on a common sample (R vars::VARselect): AIC, HQ, SC, FPE — plus the sequential LR test."""
    y = np.asarray(Y, float)
    T, K = y.shape
    lag_max = int(min(lag_max, (T - 10) // (K + 1)))
    Xall = _lags(y, lag_max)
    Yt = y[lag_max:]
    D, _ = _det(T, lag_max, type, season, exog)
    sample = Yt.shape[0]
    detint = D.shape[1]
    rows, prev = [], None
    for i in range(0, lag_max + 1):
        Z = np.column_stack([Xall[:, :K * i], D]) if (i or detint) else np.empty((sample, 0))
        U = Yt - Z @ np.linalg.lstsq(Z, Yt, rcond=None)[0] if Z.shape[1] else Yt
        sd = np.linalg.det(U.T @ U / sample)
        nstar = Z.shape[1]
        ld = np.log(sd)
        row = {"Lag": i, "LogL": -(K * sample / 2) * (1 + np.log(2 * np.pi)) - sample / 2 * ld,
               "AIC": ld + 2 / sample * (i * K ** 2 + K * detint),
               "HQ": ld + 2 * np.log(np.log(sample)) / sample * (i * K ** 2 + K * detint),
               "SC": ld + np.log(sample) / sample * (i * K ** 2 + K * detint),
               "FPE": ((sample + nstar) / (sample - nstar)) ** K * sd}
        if prev is not None:     # Sims' small-sample corrected LR, (T − m)(ln|Σ_{i−1}| − ln|Σ_i|)
            lr = (sample - nstar) * (prev - ld)
            row["LR"], row["p(LR)"] = lr, stats.chi2.sf(lr, K ** 2)
        rows.append(row)
        prev = ld
    out = pd.DataFrame(rows)
    sel = {c: int(out.loc[out[c].iloc[1:].idxmin() if c != "LR" else 0, "Lag"]) for c in ("AIC", "HQ", "SC", "FPE")}
    out.attrs["selection"] = sel
    return out


def companion(r: VarResult) -> np.ndarray:
    K, p = r.K, r.p
    C = np.zeros((K * p, K * p))
    C[:K] = np.hstack([r.A(i) for i in range(1, p + 1)])
    if p > 1:
        C[K:, :-K] = np.eye(K * (p - 1))
    return C


def roots(r) -> np.ndarray:
    return np.sort(np.abs(np.linalg.eigvals(companion(r))))[::-1]


# ---------------- residual diagnostics (R vars conventions) ----------------
def serial_pt(r: VarResult, h: int = 16, vec_rank: int | None = None) -> dict:
    U = r.resid
    obs, K = U.shape
    C0i = np.linalg.inv(U.T @ U / obs)
    tr = []
    for i in range(1, h + 1):
        Ci = U[i:].T @ U[:-i] / obs
        tr.append(np.trace(Ci.T @ C0i @ Ci @ C0i))
    tr = np.array(tr)
    Q = obs * tr.sum()
    Qs = obs ** 2 * (tr / (obs - np.arange(1, h + 1))).sum()
    df = K ** 2 * h - K ** 2 * r.p + (K if vec_rank is not None else 0)
    return {"Q": Q, "Q_adj": Qs, "df": df, "p": stats.chi2.sf(Q, df), "p_adj": stats.chi2.sf(Qs, df), "h": h}


def serial_bg(r: VarResult, h: int = 5) -> dict:
    U = r.resid
    obs, K = U.shape
    lagged = np.zeros((obs, K * h))
    for i in range(1, h + 1):
        lagged[i:, (i - 1) * K:i * K] = U[:-i]
    Z = r.Z
    X0 = np.column_stack([Z, lagged])
    e0 = U - X0 @ np.linalg.lstsq(X0, U, rcond=None)[0]
    e1 = U - Z @ np.linalg.lstsq(Z, U, rcond=None)[0]
    S0, S1 = e0.T @ e0 / obs, e1.T @ e1 / obs
    LM = obs * (K - np.trace(np.linalg.solve(S1, S0)))
    R2r = 1 - np.linalg.det(S0) / np.linalg.det(S1)
    m = K * h
    q = 0.5 * K * m - 1
    n = Z.shape[1]
    N = obs - n - m - 0.5 * (K - m + 1)
    rr = np.sqrt((K ** 2 * m ** 2 - 4) / (K ** 2 + m ** 2 - 5)) if (K ** 2 + m ** 2 - 5) > 0 else 1.0
    F = (1 - (1 - R2r) ** (1 / rr)) / (1 - R2r) ** (1 / rr) * (N * rr - q) / (K * m)
    df2 = np.floor(N * rr - q)
    return {"LM": LM, "df": h * K ** 2, "p": stats.chi2.sf(LM, h * K ** 2), "F_ES": F, "df1": h * K ** 2, "df2": df2,
            "p_ES": stats.f.sf(F, h * K ** 2, df2), "h": h}


def normality(r: VarResult) -> dict:
    U = r.resid - r.resid.mean(0)
    obs, K = U.shape
    P = np.linalg.cholesky(U.T @ U / obs).T          # upper (R chol)
    Zs = U @ np.linalg.inv(P)
    b1 = (Zs ** 3).sum(0) / obs
    b2 = (Zs ** 4).sum(0) / obs
    s3 = obs * b1 @ b1 / 6
    s4 = obs * (b2 - 3) @ (b2 - 3) / 24
    uni = []
    for k in range(K):
        x = U[:, k]
        m2 = (x ** 2).mean()
        sk = (x ** 3).mean() / m2 ** 1.5
        ku = (x ** 4).mean() / m2 ** 2
        jb = obs * (sk ** 2 / 6 + (ku - 3) ** 2 / 24)
        uni.append({"Equation": r.names[k], "Skewness": sk, "Kurtosis": ku, "JB": jb, "p-value": stats.chi2.sf(jb, 2)})
    return {"JB": s3 + s4, "df": 2 * K, "p": stats.chi2.sf(s3 + s4, 2 * K), "skew": s3, "p_skew": stats.chi2.sf(s3, K),
            "kurt": s4, "p_kurt": stats.chi2.sf(s4, K), "univariate": pd.DataFrame(uni)}


def arch_multi(r: VarResult, lags: int = 5) -> dict:
    U = r.resid
    U = (U - U.mean(0)) / U.std(0, ddof=1)
    obs, K = U.shape
    il = np.tril_indices(K)
    V = np.array([np.outer(u, u)[il] for u in U])
    c = V.shape[1]
    Y = V[lags:]
    X = np.column_stack([np.ones(len(Y))] + [V[lags - i:obs - i] for i in range(1, lags + 1)])
    e1 = Y - X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    e0 = Y - Y.mean(0)
    O1, O0 = np.cov(e1.T), np.cov(e0.T)
    R2 = 1 - 2 / (K * (K + 1)) * np.trace(O1 @ np.linalg.inv(O0))
    n = len(Y)
    st = 0.5 * n * K * (K + 1) * R2
    df = lags * K ** 2 * (K + 1) ** 2 / 4
    return {"stat": st, "df": df, "p": stats.chi2.sf(st, df), "lags": lags, "ncols": c}


def diagnostics(r: VarResult, h_pt: int | None = None, h_bg: int = 5, arch_lags: int = 5) -> pd.DataFrame:
    h_pt = h_pt or max(r.p + 4, min(16, r.obs // 5))
    pt, bg, nm, ar = serial_pt(r, h_pt), serial_bg(r, h_bg), normality(r), arch_multi(r, arch_lags)
    rows = [
        {"Test": f"Portmanteau Q({pt['h']})", "H0": "no residual autocorrelation", "Statistic": pt["Q"], "df": pt["df"], "p-value": pt["p"]},
        {"Test": f"Portmanteau adjusted Q*({pt['h']})", "H0": "no residual autocorrelation", "Statistic": pt["Q_adj"], "df": pt["df"], "p-value": pt["p_adj"]},
        {"Test": f"Breusch-Godfrey LM({bg['h']})", "H0": "no residual autocorrelation", "Statistic": bg["LM"], "df": bg["df"], "p-value": bg["p"]},
        {"Test": f"Edgerton-Shukur F({bg['h']})", "H0": "no residual autocorrelation", "Statistic": bg["F_ES"], "df": f"{bg['df1']}, {int(bg['df2'])}", "p-value": bg["p_ES"]},
        {"Test": "Jarque-Bera (multivariate)", "H0": "normal residuals", "Statistic": nm["JB"], "df": nm["df"], "p-value": nm["p"]},
        {"Test": "Skewness (multivariate)", "H0": "normal residuals", "Statistic": nm["skew"], "df": r.K, "p-value": nm["p_skew"]},
        {"Test": "Kurtosis (multivariate)", "H0": "normal residuals", "Statistic": nm["kurt"], "df": r.K, "p-value": nm["p_kurt"]},
        {"Test": f"ARCH-LM multivariate ({ar['lags']})", "H0": "no ARCH effects", "Statistic": ar["stat"], "df": ar["df"], "p-value": ar["p"]}]
    out = pd.DataFrame(rows)
    out["df"] = out["df"].astype(str)
    return out


# ---------------- causality ----------------
def _vcov_pi(r: VarResult) -> np.ndarray:
    """Covariance of vec(B) (columns stacked: equation 1 coefficients, equation 2, …) = Σ_u ⊗ (Z'Z)⁻¹."""
    return np.kron(r.sigma_u, np.linalg.inv(r.Z.T @ r.Z))


def granger(r: VarResult, cause: list) -> dict:
    """System Wald F test that `cause` does not Granger-cause the remaining variables + instantaneous causality
    (R vars::causality)."""
    K, p = r.K, r.p
    cause = [cause] if isinstance(cause, str) else list(cause)
    ci = [r.names.index(c) for c in cause]
    ei = [k for k in range(K) if k not in ci]
    m = r.Z.shape[1]
    R = []
    for k in ei:
        for lag in range(p):
            for c in ci:
                row = np.zeros(m * K)
                row[k * m + lag * K + c] = 1
                R.append(row)
    R = np.array(R)
    b = r.B.T.reshape(-1)       # vec(B) column-stacked
    V = _vcov_pi(r)
    Rb = R @ b
    N = len(R)
    F = float(Rb @ np.linalg.solve(R @ V @ R.T, Rb) / N)
    df1, df2 = p * len(ci) * len(ei), K * r.obs - m * K
    S = r.sigma_u
    # instantaneous causality: Wald on the off-diagonal block of Σ_u
    pairs = [(i, j) for j in range(K) for i in range(j, K)]          # column-major vech order
    vech = np.array([S[i, j] for i, j in pairs])
    sel = [n for n, (i, j) in enumerate(pairs) if (i in ei and j in ci) or (i in ci and j in ei)]
    C = np.zeros((len(sel), len(pairs)))
    for a, n in enumerate(sel):
        C[a, n] = 1
    D = _duplication(K)
    Dp = np.linalg.pinv(D)
    W = float(r.obs * (C @ vech) @ np.linalg.solve(2 * C @ Dp @ np.kron(S, S) @ Dp.T @ C.T, C @ vech))
    return {"F": F, "df1": df1, "df2": df2, "p": stats.f.sf(F, df1, df2),
            "inst": W, "inst_df": len(sel), "inst_p": stats.chi2.sf(W, len(sel)), "cause": cause,
            "effect": [r.names[k] for k in ei]}


def _duplication(n: int) -> np.ndarray:
    D = np.zeros((n * n, n * (n + 1) // 2))
    col = 0
    for j in range(n):
        for i in range(j, n):
            D[j * n + i, col] = 1
            D[i * n + j, col] = 1
            col += 1
    return D


def granger_pairs(r: VarResult, lags: list | None = None) -> pd.DataFrame:
    """Equation-by-equation block-exogeneity Wald χ² tests (EViews convention): for each equation, do the lags of
    each other variable — and of all others jointly — enter? `lags` restricts the tested lags (Toda-Yamamoto)."""
    K = r.K
    lags = lags or list(range(1, r.p + 1))
    ZtZi = np.linalg.inv(r.Z.T @ r.Z)
    rows = []
    for k in range(K):
        s2 = r.sigma_u[k, k]
        V = s2 * ZtZi
        b = r.B[:, k]
        groups = [[c] for c in range(K) if c != k] + [[c for c in range(K) if c != k]]
        for g in groups:
            idx = [(lag - 1) * K + c for lag in lags for c in g]
            bb = b[idx]
            W = float(bb @ np.linalg.solve(V[np.ix_(idx, idx)], bb))
            rows.append({"Dependent": r.names[k], "Excluded": "ALL" if len(g) > 1 else r.names[g[0]],
                         "Chi-sq": W, "df": len(idx), "p-value": stats.chi2.sf(W, len(idx))})
    return pd.DataFrame(rows)


def toda_yamamoto(Y, p: int, dmax: int = 1, type: str = "const", season=None, exog=None, names=None) -> tuple:
    """Toda & Yamamoto (1995): estimate a VAR(p + dmax) in levels and Wald-test only the first p lags (χ²(p)) —
    valid whatever the integration / cointegration properties."""
    r = fit_var(Y, p + dmax, type, season, exog, names=names)
    return granger_pairs(r, list(range(1, p + 1))), r


# ---------------- impulse responses, FEVD, forecasts ----------------
def ma_phi(Alist: list, h: int) -> np.ndarray:
    K = Alist[0].shape[0]
    p = len(Alist)
    Phi = np.zeros((h + 1, K, K))
    Phi[0] = np.eye(K)
    for i in range(1, h + 1):
        for j in range(1, min(i, p) + 1):
            Phi[i] += Phi[i - j] @ Alist[j - 1]
    return Phi


def impact_matrix(r, kind: str = "orth", svar=None) -> np.ndarray:
    S = r.sigma_u
    if svar is not None:
        return np.linalg.solve(svar["A"], svar["B"])
    if kind == "orth":
        return np.linalg.cholesky(S)
    if kind == "generalized":
        return S / np.sqrt(np.diag(S))[None, :]
    return np.eye(r.K)


def irf(r, h: int = 10, kind: str = "orth", cumulative: bool = False, svar=None) -> np.ndarray:
    """Responses [step, response, impulse]. kind: 'orth' (Cholesky, in the variable order), 'generalized'
    (Pesaran-Shin, order-invariant), 'simple' (forecast-error) — or a structural impact matrix via `svar`."""
    Al = [r.A(i) for i in range(1, r.p + 1)]
    Phi = ma_phi(Al, h)
    out = Phi @ impact_matrix(r, kind, svar)
    return np.cumsum(out, 0) if cumulative else out


def fevd(r, h: int = 10, svar=None) -> np.ndarray:
    """Forecast-error variance decomposition [step, variable, shock] (shares, rows sum to 1)."""
    th = irf(r, h - 1, "orth", svar=svar)
    num = np.cumsum(th ** 2, 0)
    return num / num.sum(2, keepdims=True)


def _simulate_var(r: VarResult, U: np.ndarray) -> np.ndarray:
    K, p, obs = r.K, r.p, r.obs
    y = np.zeros_like(r.y)
    y[:p] = r.y[:p]
    Bl = r.B[:K * p]
    D = r.Z[:, K * p:]
    Bd = r.B[K * p:]
    for t in range(obs):
        lag = np.concatenate([y[p + t - i] for i in range(1, p + 1)])
        y[p + t] = lag @ Bl + (D[t] @ Bd if D.shape[1] else 0) + U[t]
    return y


def irf_bootstrap(r: VarResult, h: int = 10, kind: str = "orth", cumulative: bool = False, runs: int = 500,
                  ci: float = 0.95, seed: int = 123, svar_fun=None) -> tuple:
    """Residual (Efron) bootstrap percentile bands, as R vars::irf(boot = TRUE)."""
    rng = np.random.default_rng(seed)
    U0 = r.resid - r.resid.mean(0)
    exog = r.Z[:, r.K * r.p:]
    keep = []
    for _ in range(runs):
        U = U0[rng.integers(0, r.obs, r.obs)]
        yb = _simulate_var(r, U)
        try:
            rb = _refit(r, yb)
            sv = svar_fun(rb) if svar_fun else None
            keep.append(irf(rb, h, kind, cumulative, sv))
        except Exception:  # noqa: BLE001
            continue
    keep = np.array(keep)
    a = (1 - ci) / 2
    return np.quantile(keep, a, axis=0), np.quantile(keep, 1 - a, axis=0)


def _refit(r: VarResult, y: np.ndarray) -> VarResult:
    exog = r.Z[:, r.K * r.p + len([d for d in r.det_names if d in ("const", "trend") or d.startswith("sd")]):]
    full_exog = None
    if exog.shape[1]:
        full_exog = np.vstack([np.full((r.p, exog.shape[1]), np.nan), exog])
    return fit_var(y, r.p, r.type, r.season, full_exog, r.exog_names, r.names)


def forecast(r: VarResult, h: int = 8, level: float = 0.95, exog_future=None) -> dict:
    K, p = r.K, r.p
    y = list(r.y[-p:][::-1])        # most recent first
    T = len(r.y)
    out = []
    Bl, Bd = r.B[:K * p], r.B[K * p:]
    dn = r.det_names
    for i in range(1, h + 1):
        lag = np.concatenate(y[:p])
        d = []
        for nm in dn:
            if nm == "const":
                d.append(1.0)
            elif nm == "trend":
                d.append(float(T + i))
            elif nm.startswith("sd"):
                d.append(_season_dummies(T + h, r.season)[T + i - 1][int(nm[2:]) - 1])
            else:
                j = r.exog_names.index(nm)
                d.append(float(np.asarray(exog_future)[i - 1, j]) if exog_future is not None else float(r.Z[-1, K * p + dn.index(nm)]))
        yh = lag @ Bl + (np.array(d) @ Bd if len(d) else 0)
        out.append(yh)
        y.insert(0, yh)
    out = np.array(out)
    Phi = ma_phi([r.A(i) for i in range(1, p + 1)], h)
    S = r.sigma_u
    var = np.cumsum(np.array([np.diag(Phi[j] @ S @ Phi[j].T) for j in range(h)]), 0)
    z = stats.norm.ppf(0.5 + level / 2)
    return {"mean": out, "lower": out - z * np.sqrt(var), "upper": out + z * np.sqrt(var), "se": np.sqrt(var)}


# ---------------- structural VAR ----------------
def svar_fit(r: VarResult, Amat=None, Bmat=None, max_iter: int = 200, tol: float = 1e-9) -> dict:
    """A-, B- or AB-model by scoring ML (Amisano & Giannini 1997; R vars::SVAR estmethod = 'scoring').
    Free elements are NaN; fixed elements hold their values."""
    K = r.K
    obs = r.obs
    S = r.sigma_u
    if Amat is None and Bmat is None:
        raise ValueError("Provide A and/or B with NaN for free elements.")
    typ = "AB-model" if Amat is not None and Bmat is not None else ("A-model" if Amat is not None else "B-model")
    Am = np.eye(K) if Amat is None else np.array(Amat, float)
    Bm = np.eye(K) if Bmat is None else np.array(Bmat, float)
    ra, rb = Am.flatten("F"), Bm.flatten("F")
    pa, pb = np.where(np.isnan(ra))[0], np.where(np.isnan(rb))[0]
    nA, nB = len(pa), len(pb)
    npar = nA + nB
    need = K * (K - 1) // 2 + (K * K if typ == "AB-model" else 0)
    if (2 * K * K if typ == "AB-model" else K * K) - npar < need:
        raise ValueError("The structural model is not identified: fix more elements (need at least "
                         f"{need} restrictions).")
    K2 = K * K
    R = np.zeros((2 * K2, npar))
    for i, pos in enumerate(pa):
        R[pos, i] = 1
    for i, pos in enumerate(pb):
        R[K2 + pos, nA + i] = 1
    ra = np.nan_to_num(ra)
    rb = np.nan_to_num(rb)
    rvec = np.r_[ra, rb]
    Kkk = np.eye(K2)[:, [i + K * j for i in range(K) for j in range(K)]]
    I2, I = np.eye(K2), np.eye(K)
    g = np.full(npar, 0.1)
    it = 0
    for it in range(1, max_iter + 1):
        vab = R @ g + rvec
        A = vab[:K2].reshape(K, K, order="F")
        B = vab[K2:].reshape(K, K, order="F")
        Bi = np.linalg.inv(B)
        Bti = np.linalg.inv(B.T)
        BiA = Bi @ A
        m1 = np.vstack([np.kron(np.linalg.inv(BiA), Bti), -np.kron(I, Bti)])
        m3 = np.hstack([np.kron(np.linalg.inv(BiA).T, Bi), -np.kron(I, Bi)])
        info = obs * (m1 @ (I2 + Kkk) @ m3)
        ig = R.T @ info @ R
        sc_b = obs * np.linalg.inv(BiA.T).flatten("F") - obs * (np.kron(S, I) @ BiA.flatten("F"))
        sc = np.vstack([np.kron(I, Bti), -np.kron(BiA, Bti)]) @ sc_b
        d = np.linalg.solve(ig, R.T @ sc)
        ln = np.max(np.abs(d))
        g = g + (d / ln if ln > 1 else d)
        if np.max(np.abs(d if ln <= 1 else d / ln)) < tol:
            break
    vab = R @ g + rvec
    A = vab[:K2].reshape(K, K, order="F")
    B = vab[K2:].reshape(K, K, order="F")
    se = np.sqrt(np.abs(np.diag(R @ np.linalg.inv(ig) @ R.T)))
    Ase = se[:K2].reshape(K, K, order="F")
    Bse = se[K2:].reshape(K, K, order="F")
    M = np.linalg.solve(A, B)
    neg = np.diag(M) < 0
    if neg.any():
        if typ == "A-model":
            A[:, neg] *= -1
        else:
            B[:, neg] *= -1
    Su = np.linalg.inv(A) @ B @ B.T @ np.linalg.inv(A).T
    dfo = K * (K + 1) // 2 - npar + (K * K if typ == "AB-model" else 0) - (K * K if typ == "AB-model" else 0)
    dfo = K * (K + 1) // 2 - npar
    lr = obs * (np.linalg.slogdet(Su)[1] - np.linalg.slogdet(S)[1]) if dfo > 0 else np.nan
    return {"A": A, "B": B, "Ase": Ase, "Bse": Bse, "type": typ, "iterations": it, "LR": lr, "LR_df": dfo,
            "LR_p": stats.chi2.sf(lr, dfo) if dfo > 0 else np.nan, "impact": np.linalg.solve(A, B)}


def blanchard_quah(r: VarResult) -> dict:
    """Long-run (Blanchard-Quah 1989) identification: the long-run impact matrix is lower triangular."""
    K = r.K
    A1 = np.eye(K) - sum(r.A(i) for i in range(1, r.p + 1))
    S = r.sigma_u
    lr = np.linalg.cholesky(np.linalg.inv(A1) @ S @ np.linalg.inv(A1).T)
    B = A1 @ lr
    return {"A": np.eye(K), "B": B, "LRIM": lr, "type": "Blanchard-Quah", "impact": B}


# ==========================================================
# Cointegration
# ==========================================================
def engle_granger(y, X, trend: str = "c", maxlag: int | None = None, autolag: str | None = "AIC", names=None) -> dict:
    """Engle & Granger (1987) two-step: static OLS in levels, then an ADF test (no deterministics) on the residuals
    with MacKinnon (2010) cointegration p-values and critical values for N = 1 + #regressors."""
    from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
    from statsmodels.tsa.stattools import adfuller
    yv = np.asarray(y, float)
    Xv = np.asarray(X, float).reshape(len(yv), -1)
    T = len(yv)
    det = {"n": [], "c": [np.ones(T)], "ct": [np.ones(T), np.arange(1, T + 1, dtype=float)]}[trend]
    W = np.column_stack(det + [Xv])
    b = np.linalg.lstsq(W, yv, rcond=None)[0]
    u = yv - W @ b
    names = names or [f"x{i + 1}" for i in range(Xv.shape[1])]
    labels = {"n": [], "c": ["const"], "ct": ["const", "trend"]}[trend] + list(names)
    maxlag = maxlag if maxlag is not None else int(np.floor(12 * (T / 100) ** 0.25))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = adfuller(u, maxlag=maxlag, regression="n", autolag=autolag)
    stat, lags = float(res[0]), int(res[2])
    N = Xv.shape[1] + 1
    p = float(mackinnonp(stat, regression=trend, N=N))
    cv = mackinnoncrit(N=N, regression=trend, nobs=T - lags - 1)
    return {"coef": pd.Series(b, index=labels), "resid": u, "stat": stat, "lags": lags, "p": p,
            "crit": {"1%": cv[0], "5%": cv[1], "10%": cv[2]}, "N": N, "nobs": T}


def ecm(y, X, u, lags: int = 1, names=None) -> pd.DataFrame:
    """Second-step error-correction model: Δy_t on Δx_t, lagged Δy and Δx, and u_{t−1} (OLS, classical SE)."""
    import statsmodels.api as sm
    yv = np.asarray(y, float)
    Xv = np.asarray(X, float).reshape(len(yv), -1)
    names = names or [f"x{i + 1}" for i in range(Xv.shape[1])]
    dy, dX = np.diff(yv), np.diff(Xv, axis=0)
    cols, labs = [dX[lags:]], [f"Δ{n}" for n in names]
    for L in range(1, lags + 1):
        cols += [dy[lags - L:-L or None][:, None], dX[lags - L:len(dX) - L]]
        labs += [f"Δy(−{L})"] + [f"Δ{n}(−{L})" for n in names]
    cols.append(u[lags:-1][:, None])
    labs.append("ECT(−1)")
    Xr = sm.add_constant(np.column_stack(cols))
    m = sm.OLS(dy[lags:], Xr).fit()
    return pd.DataFrame({"Regressor": ["const"] + labs, "Coef.": m.params, "Std. Err.": m.bse, "t": m.tvalues,
                         "p-value": m.pvalues})


# ---------------- Johansen ----------------
def _johansen_core(y: np.ndarray, K: int, case: int, season: int | None = None, exog=None):
    """Returns eigenvalues, eigenvectors V (for Z1, incl. restricted deterministic), S-matrices and residuals.
    K = lag order of the VAR in levels (Δ-lags = K − 1)."""
    T, n = y.shape
    dy = np.diff(y, axis=0)
    Z0 = dy[K - 1:]
    Teff = len(Z0)
    tt = np.arange(K + 1, T + 1, dtype=float)          # time index of the dependent observation (1-based)
    Z1 = y[K - 1:-1]
    if case == 2:
        Z1 = np.column_stack([Z1, np.ones(Teff)])
    elif case == 4:
        Z1 = np.column_stack([Z1, tt - 1])
    Z2 = []
    if case in (3, 4, 5):
        Z2.append(np.ones(Teff))
    if case == 5:
        Z2.append(tt)
    Z2 += [dy[K - 1 - i:len(dy) - i] for i in range(1, K)]
    if season:
        Z2 += list(_season_dummies(T, season)[K:].T)
    if exog is not None:
        e = np.asarray(exog, float).reshape(T, -1)[K:]
        Z2 += list(e.T)
    Z2 = np.column_stack(Z2) if Z2 else np.empty((Teff, 0))
    if Z2.shape[1]:
        M = lambda A_: A_ - Z2 @ np.linalg.lstsq(Z2, A_, rcond=None)[0]  # noqa: E731
        R0, R1 = M(Z0), M(Z1)
    else:
        R0, R1 = Z0, Z1
    S00, S01, S11 = R0.T @ R0 / Teff, R0.T @ R1 / Teff, R1.T @ R1 / Teff
    C = np.linalg.cholesky(S11)
    Ci = np.linalg.inv(C)
    Mx = Ci @ S01.T @ np.linalg.solve(S00, S01) @ Ci.T
    lam, E = np.linalg.eigh(Mx)
    order = np.argsort(lam)[::-1]
    lam, E = lam[order], E[:, order]
    V = Ci.T @ E                                       # V' S11 V = I
    return {"lam": lam[:n], "V": V, "S00": S00, "S01": S01, "S11": S11, "R0": R0, "R1": R1, "Z0": Z0, "Z1": Z1,
            "Z2": Z2, "T": Teff}


@lru_cache(maxsize=64)
def _johansen_null(m: int, case: int, reps: int = 8000, T: int = 800, seed: int = 2026) -> tuple:
    """Monte Carlo of the asymptotic trace / λ-max distributions with m common stochastic trends."""
    rng = np.random.default_rng(seed + 31 * m + case)
    tr_all, mx_all = [], []
    tt = np.arange(2, T + 1, dtype=float)
    for chunk in range(0, reps, 1000):
        R = min(1000, reps - chunk)
        e = rng.standard_normal((R, T, m))
        y = np.cumsum(e, axis=1)
        Z0 = np.diff(y, axis=1)
        Z1 = y[:, :-1].copy()
        Te = T - 1
        if case == 3:          # unrestricted constant ⇒ linear trend in the data: one common trend is deterministic
            Z1[:, :, 0] = tt - 1
        elif case == 5:        # unrestricted trend ⇒ quadratic trend in the data
            Z1[:, :, 0] = (tt - 1) ** 2
        if case == 2:
            Z1 = np.concatenate([Z1, np.ones((R, Te, 1))], 2)
        elif case == 4:
            Z1 = np.concatenate([Z1, np.broadcast_to((tt - 1)[None, :, None], (R, Te, 1))], 2)
        cols = []
        if case in (3, 4, 5):
            cols.append(np.ones(Te))
        if case == 5:
            cols.append(tt)
        if cols:
            Z2 = np.column_stack(cols)
            P = np.linalg.pinv(Z2)
            Z0 = Z0 - np.einsum("td,rdk->rtk", Z2, np.einsum("dt,rtk->rdk", P, Z0))
            Z1 = Z1 - np.einsum("td,rdk->rtk", Z2, np.einsum("dt,rtk->rdk", P, Z1))
        S00 = np.einsum("rti,rtj->rij", Z0, Z0) / Te
        S01 = np.einsum("rti,rtj->rij", Z0, Z1) / Te
        S11 = np.einsum("rti,rtj->rij", Z1, Z1) / Te
        M = np.linalg.solve(S11, np.transpose(S01, (0, 2, 1))) @ np.linalg.solve(S00, S01)
        lam = np.sort(np.clip(np.real(np.linalg.eigvals(M)), 0, 1 - 1e-12), axis=1)[:, ::-1][:, :m]
        tr_all.append(-Te * np.log(1 - lam).sum(1))
        mx_all.append(-Te * np.log(1 - lam[:, 0]))
    return np.sort(np.concatenate(tr_all)), np.sort(np.concatenate(mx_all))


def johansen(Y, K: int = 2, case: int = 3, season: int | None = None, exog=None, names=None,
             small_sample: bool = False) -> dict:
    """Johansen (1988, 1995) trace and maximum-eigenvalue tests. p-values and critical values from a Monte Carlo of
    the asymptotic distributions (cached); `small_sample` applies the Reinsel-Ahn factor (T − nK)/T."""
    y = np.asarray(Y, float)
    T, n = y.shape
    names = list(names) if names is not None else (list(Y.columns) if hasattr(Y, "columns") else [f"y{i + 1}" for i in range(n)])
    core = _johansen_core(y, K, case, season, exog)
    lam = core["lam"]
    Te = core["T"]
    fac = (Te - n * K) / Te if small_sample else 1.0
    rows = []
    for r in range(n):
        tr = -Te * np.log(1 - lam[r:]).sum() * fac
        mx = -Te * np.log(1 - lam[r]) * fac
        dtr, dmx = _johansen_null(n - r, case)
        rows.append({"H0: rank ≤": r, "Eigenvalue": lam[r], "Trace": tr, "Trace 5% cv": np.quantile(dtr, 0.95),
                     "Trace p-value": float((dtr >= tr).mean()), "Max-eigen": mx, "Max 5% cv": np.quantile(dmx, 0.95),
                     "Max p-value": float((dmx >= mx).mean())})
    tab = pd.DataFrame(rows)
    rank_tr = next((int(r["H0: rank ≤"]) for _, r in tab.iterrows() if r["Trace p-value"] >= 0.05), n)
    rank_mx = next((int(r["H0: rank ≤"]) for _, r in tab.iterrows() if r["Max p-value"] >= 0.05), n)
    zl = names + (["const"] if case == 2 else ["trend"] if case == 4 else [])
    return {"table": tab, "rank_trace": rank_tr, "rank_max": rank_mx, "core": core, "names": names, "z1_names": zl,
            "case": case, "K": K, "season": season, "exog": exog, "y": y, "small_sample": small_sample}


def vecm(jo: dict, r: int) -> dict:
    """VECM for cointegration rank r: β normalised so that its first r rows are the identity (Phillips
    triangular form), α and Γ by OLS given β (as R urca::cajorls), Johansen standard errors for the free β."""
    core = jo["core"]
    y, K, case, names = jo["y"], jo["K"], jo["case"], jo["names"]
    n = len(names)
    if not 1 <= r < n:
        raise ValueError("The cointegration rank must be between 1 and n − 1.")
    V = core["V"][:, :r]
    beta = V @ np.linalg.inv(V[:r, :r])
    Z0, Z1, Z2 = core["Z0"], core["Z1"], core["Z2"]
    Te = core["T"]
    ect = Z1 @ beta
    X = np.column_stack([ect, Z2]) if Z2.shape[1] else ect
    G = np.linalg.lstsq(X, Z0, rcond=None)[0]
    U = Z0 - X @ G
    m = X.shape[1]
    s2 = (U ** 2).sum(0) / (Te - m)
    XtXi = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.outer(np.diag(XtXi), s2))
    z2n = (["const"] if case in (3, 4, 5) else []) + (["trend"] if case == 5 else [])
    z2n += [f"Δ{nm}.l{i}" for i in range(1, K) for nm in names]
    if jo["season"]:
        z2n += [f"sd{i + 1}" for i in range(jo["season"] - 1)]
    z2n += [f"exo{i + 1}" for i in range(Z2.shape[1] - len(z2n))]
    regs = [f"ECT{i + 1}" for i in range(r)] + z2n
    alpha = G[:r].T
    # standard errors of the free part of β (rows r … end), Johansen (1995)
    Om = U.T @ U / Te
    R1 = core["R1"]
    R1b = R1[:, r:]
    Vb = np.kron(np.linalg.inv(alpha.T @ np.linalg.solve(Om, alpha)), np.linalg.inv(R1b.T @ R1b))
    bse = np.zeros_like(beta)
    bse[r:, :] = np.sqrt(np.diag(Vb)).reshape(r, -1).T
    tq = stats.t.ppf(0.975, Te - m)
    rows = []
    for k, eq in enumerate(names):
        for j, rn in enumerate(regs):
            b, s = G[j, k], se[j, k]
            rows.append({"Equation": f"Δ{eq}", "Regressor": rn, "Coef.": b, "Std. Err.": s, "t": b / s,
                         "p-value": 2 * stats.t.sf(abs(b / s), Te - m), "CI low": b - tq * s, "CI high": b + tq * s})
    btab = []
    for i in range(r):
        for j, zn in enumerate(jo["z1_names"]):
            st = bse[j, i]
            btab.append({"Relation": f"CE{i + 1}", "Variable": zn, "β": beta[j, i], "Std. Err.": st if j >= r else np.nan,
                         "t": beta[j, i] / st if j >= r and st > 0 else np.nan})
    llf = -Te * n / 2 * (1 + np.log(2 * np.pi)) - Te / 2 * (np.linalg.slogdet(core["S00"])[1] + np.log(1 - jo["core"]["lam"][:r]).sum())
    out = {"beta": pd.DataFrame(beta, index=jo["z1_names"], columns=[f"CE{i + 1}" for i in range(r)]),
           "beta_table": pd.DataFrame(btab), "alpha": pd.DataFrame(alpha, index=names, columns=[f"CE{i + 1}" for i in range(r)]),
           "table": pd.DataFrame(rows), "resid": U, "Gamma": G, "regs": regs, "r": r, "sigma": Om, "llf": llf,
           "ect": ect, "names": names, "case": case, "K": K, "T": Te}
    out["levels"] = vec2var(jo, out)
    return out


def vec2var(jo: dict, vm: dict) -> dict:
    """Levels-VAR representation of the VECM (R vars::vec2var): A₁ = Γ₁ + Π + I, A_i = Γ_i − Γ_{i−1}, A_K = −Γ_{K−1}."""
    n, K, r = len(vm["names"]), vm["K"], vm["r"]
    beta = vm["beta"].to_numpy()
    alpha = vm["alpha"].to_numpy()
    Pi = alpha @ beta[:n].T
    G = vm["Gamma"]
    off = r + (1 if vm["case"] in (3, 4, 5) else 0) + (1 if vm["case"] == 5 else 0)
    Gam = [G[off + (i - 1) * n:off + i * n].T for i in range(1, K)]
    A = []
    for i in range(1, K + 1):
        if K == 1:
            A.append(np.eye(n) + Pi)
        elif i == 1:
            A.append(Gam[0] + Pi + np.eye(n))
        elif i < K:
            A.append(Gam[i - 1] - Gam[i - 2])
        else:
            A.append(-Gam[K - 2])
    return {"A": A, "sigma": vm["resid"].T @ vm["resid"] / vm["T"], "obs": vm["T"]}


class _LevelsVar:
    """Minimal VAR-like object (A matrices, Σ, K, p) for IRF / FEVD of a VECM."""

    def __init__(self, lv: dict, names):
        self._A = lv["A"]
        self.p = len(self._A)
        self.K = len(names)
        self.names = names
        self.sigma_u = lv["sigma"]

    def A(self, i):
        return self._A[i - 1]


def vecm_levels(vm: dict) -> _LevelsVar:
    return _LevelsVar(vm["levels"], vm["names"])


def restriction_test(jo: dict, r: int, H=None, A=None) -> dict:
    """Likelihood-ratio tests of linear restrictions: β = Hφ (R urca::blrtest) or α = Aψ (urca::alrtest, e.g. weak
    exogeneity of a variable). LR = T Σᵢ ln[(1 − λ̃ᵢ)/(1 − λᵢ)], χ² with r·(rows − cols) degrees of freedom."""
    core = jo["core"]
    S00, S01, S11, Te = core["S00"], core["S01"], core["S11"], core["T"]
    lam = core["lam"]
    if H is not None:
        H = np.asarray(H, float)
        S11h = H.T @ S11 @ H
        S01h = S01 @ H
        C = np.linalg.cholesky(S11h)
        Ci = np.linalg.inv(C)
        lt = np.sort(np.linalg.eigvalsh(Ci @ S01h.T @ np.linalg.solve(S00, S01h) @ Ci.T))[::-1][:r]
        df = r * (H.shape[0] - H.shape[1])
    else:
        Am = np.asarray(A, float)
        n = Am.shape[0]
        Ab = Am @ np.linalg.inv(Am.T @ Am)
        Aperp = _perp(Am)
        R0, R1 = core["R0"], core["R1"]
        Ra = R0 @ Ab
        Rb = R0 @ Aperp
        Ma = lambda X: X - Rb @ np.linalg.lstsq(Rb, X, rcond=None)[0]  # noqa: E731
        Ra_, R1_ = Ma(Ra), Ma(R1)
        Saa, Sa1, S11_ = Ra_.T @ Ra_ / Te, Ra_.T @ R1_ / Te, R1_.T @ R1_ / Te
        C = np.linalg.cholesky(S11_)
        Ci = np.linalg.inv(C)
        lt = np.sort(np.linalg.eigvalsh(Ci @ Sa1.T @ np.linalg.solve(Saa, Sa1) @ Ci.T))[::-1][:r]
        df = r * (n - Am.shape[1])
    LR = Te * np.sum(np.log((1 - lt) / (1 - lam[:r])))
    return {"LR": float(LR), "df": int(df), "p": float(stats.chi2.sf(LR, df)), "eig_restricted": lt}


def _perp(A: np.ndarray) -> np.ndarray:
    q, _ = np.linalg.qr(A, mode="complete")
    return q[:, A.shape[1]:]


def weak_exogeneity(jo: dict, r: int) -> pd.DataFrame:
    """For each variable: LR test that its row of α is zero (it does not adjust to the long-run disequilibrium)."""
    n = len(jo["names"])
    rows = []
    for i, nm in enumerate(jo["names"]):
        A = np.delete(np.eye(n), i, axis=1)
        t = restriction_test(jo, r, A=A)
        rows.append({"Variable": nm, "LR": t["LR"], "df": t["df"], "p-value": t["p"]})
    return pd.DataFrame(rows)


# ---------------- Gregory-Hansen ----------------
GH_MODELS = {"C": ("Level shift", "تحول في المستوى"), "CT": ("Level shift with trend", "تحول في المستوى مع اتجاه"),
             "CS": ("Regime shift (intercept and slopes)", "تحول في النظام (الثابت والميول)")}


def _gh_design(X, T, tb, model):
    phi = (np.arange(T) >= tb).astype(float)
    cols = [np.ones(T), phi]
    if model == "CT":
        cols.append(np.arange(1, T + 1, dtype=float))
    cols.append(X)
    if model == "CS":
        cols.append(X * phi[:, None])
    return np.column_stack(cols)


def _za(u: np.ndarray, bw: int):
    """Phillips Z_t and Z_α on residuals (no constant), Bartlett long-run variance with bandwidth bw."""
    ul = u[:-1]
    rho = (ul @ u[1:]) / (ul @ ul)
    e = u[1:] - rho * ul
    n = len(e)
    g0 = e @ e / n
    lrv = g0 + 2 * sum((1 - j / (bw + 1)) * (e[j:] @ e[:-j] / n) for j in range(1, bw + 1))
    lam = (lrv - g0) / 2
    rs = (rho - 1) - lam * n / (ul @ ul)      # bias-corrected
    za = n * rs
    zt = rs * np.sqrt(ul @ ul) / np.sqrt(lrv)
    return za, zt


def gregory_hansen(y, X, model: str = "C", trim: float = 0.15, maxlag: int | None = None, index=None) -> dict:
    """Gregory & Hansen (1996) residual-based cointegration tests allowing one unknown break: ADF*, Z_t*, Z_α*
    (minimum over break dates). Critical values by Monte Carlo under no cointegration (cached)."""
    from statsmodels.tsa.stattools import adfuller
    yv = np.asarray(y, float)
    Xv = np.asarray(X, float).reshape(len(yv), -1)
    T, m = Xv.shape
    maxlag = maxlag if maxlag is not None else int(np.floor(12 * (T / 100) ** 0.25))
    bw = int(np.floor(4 * (T / 100) ** 0.25))
    lo, hi = int(np.floor(trim * T)), int(np.floor((1 - trim) * T))
    best = {"ADF": (np.inf, None, None), "Zt": (np.inf, None), "Za": (np.inf, None)}
    for tb in range(lo, hi + 1):
        W = _gh_design(Xv, T, tb, model)
        u = yv - W @ np.linalg.lstsq(W, yv, rcond=None)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = adfuller(u, maxlag=maxlag, regression="n", autolag="t-stat" if maxlag else None)
        if a[0] < best["ADF"][0]:
            best["ADF"] = (float(a[0]), tb, int(a[2]))
        za, zt = _za(u, bw)
        if zt < best["Zt"][0]:
            best["Zt"] = (float(zt), tb)
        if za < best["Za"][0]:
            best["Za"] = (float(za), tb)
    sim = _gh_null(m, model, trim)
    rows = []
    for k, lab in (("ADF", "ADF*"), ("Zt", "Zt*"), ("Za", "Zα*")):
        st, tb = best[k][0], best[k][1]
        dist = sim["ADF" if k == "ADF" else k]
        rows.append({"Test": lab, "Statistic": st, "Break": (index[tb] if index is not None else tb + 1),
                     "1% cv": np.quantile(dist, 0.01), "5% cv": np.quantile(dist, 0.05), "10% cv": np.quantile(dist, 0.10),
                     "p-value": float((dist <= st).mean()), "Lags": best["ADF"][2] if k == "ADF" else ""})
    return {"table": pd.DataFrame(rows), "model": model, "break_pos": best["ADF"][1]}


@lru_cache(maxsize=32)
def _gh_null(m: int, model: str, trim: float, reps: int = 1500, T: int = 250, seed: int = 1996) -> dict:
    rng = np.random.default_rng(seed + 7 * m + len(model))
    out = {"ADF": [], "Zt": [], "Za": []}
    bw = int(np.floor(4 * (T / 100) ** 0.25))
    lo, hi = int(np.floor(trim * T)), int(np.floor((1 - trim) * T))
    breaks = list(range(lo, hi + 1, 2))
    designs = [_gh_design(np.zeros((T, m)), T, tb, model) for tb in breaks]
    for _ in range(reps):
        w = np.cumsum(rng.standard_normal((T, m + 1)), axis=0)
        yv, Xv = w[:, 0], w[:, 1:]
        a_min = zt_min = za_min = np.inf
        for tb in breaks:
            W = _gh_design(Xv, T, tb, model)
            u = yv - W @ np.linalg.lstsq(W, yv, rcond=None)[0]
            du, ul = np.diff(u), u[:-1]
            rho = ul @ du / (ul @ ul)
            e = du - rho * ul
            t_adf = rho / np.sqrt((e @ e) / (len(e) - 1) / (ul @ ul))
            za, zt = _za(u, bw)
            a_min, zt_min, za_min = min(a_min, t_adf), min(zt_min, zt), min(za_min, za)
        out["ADF"].append(a_min); out["Zt"].append(zt_min); out["Za"].append(za_min)
    del designs
    return {k: np.array(v) for k, v in out.items()}


# ---------------- FMOLS / DOLS / CCR ----------------
def _lrv_parts(V: np.ndarray, kernel: str = "ba", bw: float | None = None):
    """Long-run covariance Ω, one-sided Δ (= Σ_{j≥0} Γ_j weighted) with Bartlett / Quadratic-Spectral kernel and
    Andrews (1991) AR(1) automatic bandwidth."""
    T, k = V.shape
    if bw is None:
        num = den = 0.0
        for i in range(k):
            v = V[:, i]
            rho = (v[1:] @ v[:-1]) / (v[:-1] @ v[:-1])
            s2 = np.sum((v[1:] - rho * v[:-1]) ** 2) / T
            if kernel == "ba":
                num += 4 * rho ** 2 * s2 ** 2 / ((1 - rho) ** 6 * (1 + rho) ** 2)
            else:
                num += 4 * rho ** 2 * s2 ** 2 / (1 - rho) ** 8
            den += s2 ** 2 / (1 - rho) ** 4
        a = num / den
        bw = 1.1447 * (a * T) ** (1 / 3) if kernel == "ba" else 1.3221 * (a * T) ** (1 / 5)
        bw = min(bw, T - 1)
    G = lambda j: V[j:].T @ V[:T - j] / T  # noqa: E731
    w = []
    for j in range(1, T):
        x = j / bw
        if kernel == "ba":
            wj = max(0.0, 1 - x) if x < 1 else 0.0
        else:
            z = 6 * np.pi * x / 5
            wj = 25 / (12 * np.pi ** 2 * x ** 2) * (np.sin(z) / z - np.cos(z))
        w.append(wj)
    Om = G(0).copy()
    Dl = G(0).copy()
    for j, wj in enumerate(w, start=1):
        if abs(wj) < 1e-12 and kernel == "ba":
            break
        Gj = G(j)
        Om += wj * (Gj + Gj.T)
        Dl += wj * Gj.T
    return Om, Dl, bw


def fmols(y, X, trend: str = "c", kernel: str = "ba", bw: float | None = None, names=None) -> dict:
    """Phillips & Hansen (1990) fully modified OLS."""
    yv = np.asarray(y, float)
    Xv = np.asarray(X, float).reshape(len(yv), -1)
    T, k = Xv.shape
    D = {"n": np.empty((T, 0)), "c": np.ones((T, 1)), "ct": np.column_stack([np.ones(T), np.arange(1, T + 1)])}[trend]
    W = np.column_stack([D, Xv])
    b0 = np.linalg.lstsq(W, yv, rcond=None)[0]
    u = yv - W @ b0
    dX = np.diff(Xv, axis=0)
    V = np.column_stack([u[1:], dX])
    Om, Dl, bw = _lrv_parts(V, kernel, bw)
    O12, O22 = Om[0, 1:], Om[1:, 1:]
    D21 = Dl[1:, 0]
    D22 = Dl[1:, 1:]
    th = np.linalg.solve(O22, O12)
    yp = yv[1:] - dX @ th
    Dp = D21 - D22 @ th
    Wt = W[1:]
    corr = np.r_[np.zeros(D.shape[1]), Dp] * T
    WtWi = np.linalg.inv(Wt.T @ Wt)
    b = WtWi @ (Wt.T @ yp - corr)
    o112 = Om[0, 0] - O12 @ th
    V_b = o112 * WtWi
    se = np.sqrt(np.diag(V_b))
    labs = {"n": [], "c": ["const"], "ct": ["const", "trend"]}[trend] + list(names or [f"x{i + 1}" for i in range(k)])
    return _coint_table(b, se, labs, "FMOLS", {"bandwidth": bw, "kernel": kernel, "omega_u.x": o112, "N": T - 1,
                                                    "df": T - k - D.shape[1]})


def dols(y, X, trend: str = "c", leads: int = 2, lags: int = 2, hac: bool = True, names=None) -> dict:
    """Stock & Watson (1993) dynamic OLS with leads and lags of Δx; long-run variance (Bartlett, Andrews bandwidth)
    standard errors."""
    yv = np.asarray(y, float)
    Xv = np.asarray(X, float).reshape(len(yv), -1)
    T, k = Xv.shape
    dX = np.vstack([np.full((1, k), np.nan), np.diff(Xv, axis=0)])
    D = {"n": np.empty((T, 0)), "c": np.ones((T, 1)), "ct": np.column_stack([np.ones(T), np.arange(1, T + 1)])}[trend]
    cols = [D, Xv]
    for j in range(-leads, lags + 1):
        cols.append(np.roll(dX, j, axis=0) if j else dX)
    W = np.column_stack(cols)
    idx = np.arange(1 + lags, T - leads)
    Wv, yy = W[idx], yv[idx]
    b = np.linalg.lstsq(Wv, yy, rcond=None)[0]
    e = yy - Wv @ b
    n, kk = Wv.shape
    WtWi = np.linalg.inv(Wv.T @ Wv)
    if hac:
        Om, _, bw = _lrv_parts(e[:, None], "ba", None)
        s2 = Om[0, 0]
    else:
        bw, s2 = None, e @ e / (n - kk)
    se = np.sqrt(np.diag(s2 * WtWi))
    nd = D.shape[1] + k
    labs = {"n": [], "c": ["const"], "ct": ["const", "trend"]}[trend] + list(names or [f"x{i + 1}" for i in range(k)])
    return _coint_table(b[:nd], se[:nd], labs, "DOLS", {"leads": leads, "lags": lags, "bandwidth": bw, "N": n, "df": T - k - D.shape[1],
                                                        "R²": 1 - e @ e / ((yy - yy.mean()) @ (yy - yy.mean()))})


def ccr(y, X, trend: str = "c", kernel: str = "ba", bw: float | None = None, names=None) -> dict:
    """Park (1992) canonical cointegrating regression."""
    yv = np.asarray(y, float)
    Xv = np.asarray(X, float).reshape(len(yv), -1)
    T, k = Xv.shape
    D = {"n": np.empty((T, 0)), "c": np.ones((T, 1)), "ct": np.column_stack([np.ones(T), np.arange(1, T + 1)])}[trend]
    W = np.column_stack([D, Xv])
    b0 = np.linalg.lstsq(W, yv, rcond=None)[0]
    u = yv - W @ b0
    dX = np.diff(Xv, axis=0)
    V = np.column_stack([u[1:], dX])
    Om, Dl, bw = _lrv_parts(V, kernel, bw)
    Sig = V.T @ V / len(V)
    L2 = Dl[:, 1:]                           # Λ₂ = (Δ₁₂', Δ₂₂')' column block for x
    O12, O22 = Om[0, 1:], Om[1:, 1:]
    th = np.linalg.solve(O22, O12)
    beta_x = b0[D.shape[1]:]
    Sinv = np.linalg.inv(Sig)
    adjX = (Sinv @ L2).T @ V.T               # k × (T−1)
    Xs = Xv[1:] - adjX.T
    ys = yv[1:] - (Sinv @ L2 @ beta_x + np.r_[0, th]) @ V.T
    Ws = np.column_stack([D[1:], Xs])
    WtWi = np.linalg.inv(Ws.T @ Ws)
    b = WtWi @ Ws.T @ ys
    o112 = Om[0, 0] - O12 @ th
    se = np.sqrt(np.diag(o112 * WtWi))
    labs = {"n": [], "c": ["const"], "ct": ["const", "trend"]}[trend] + list(names or [f"x{i + 1}" for i in range(k)])
    return _coint_table(b, se, labs, "CCR", {"bandwidth": bw, "kernel": kernel, "N": T - 1, "df": T - k - D.shape[1]})


def _coint_table(b, se, labs, method, info) -> dict:
    t = b / se
    df = info.get("df", 10 ** 6)
    p = 2 * stats.t.sf(np.abs(t), df)
    q = stats.t.ppf(0.975, df)
    tab = pd.DataFrame({"Variable": labs, "Coef.": b, "Std. Err.": se, "t": t, "p-value": p,
                        "CI low": b - q * se, "CI high": b + q * se})
    return {"table": tab, "method": method, "info": info, "coef": pd.Series(b, index=labs)}
