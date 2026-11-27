# ==========================================
# Aetheris Research - Panel data econometrics (Catalogue §1.15)
# Static models (pooled, one- and two-way FE, Swamy-Arora RE, between, first differences), robust covariances
# (cluster with Stata small-sample correction, Driscoll-Kraay, Beck-Katz PCSE), specification tests (F, Breusch-Pagan
# LM, Hausman classic and robust, Pesaran CD / LM / scaled LM, Wooldridge serial correlation, groupwise
# heteroskedasticity, Pesaran-Yamagata slope homogeneity), panel unit roots (LLC, IPS, Fisher-ADF, Choi, Hadri, CIPS),
# Dumitrescu-Hurlin causality, mean-group and CCE estimators. Conventions follow R plm so results reproduce exactly.
# ==========================================

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from analytics import regression as rg

MODELS = {"pooled": ("Pooled OLS", "التجميعي (OLS)"), "fe": ("Fixed effects (within)", "الآثار الثابتة (داخلي)"),
          "fe2": ("Two-way fixed effects", "الآثار الثابتة ثنائية الاتجاه"), "re": ("Random effects (Swamy-Arora)", "الآثار العشوائية (سوامي-أرورا)"),
          "be": ("Between", "البيني"), "fd": ("First differences", "الفروق الأولى"),
          "fgls": ("FGLS (heteroskedastic panels)", "FGLS (عدم تجانس بين الوحدات)")}
COVS = {"classic": ("Classical", "تقليدية"), "cluster": ("Clustered by unit (Stata small-sample)", "عنقودية حسب الوحدة (تصحيح Stata)"),
        "dk": ("Driscoll-Kraay (cross-sectional dependence)", "دريسكول-كراي (اعتماد مقطعي)"),
        "pcse": ("Beck-Katz PCSE", "PCSE لبيك-كاتز")}


@dataclass
class PanelData:
    df: pd.DataFrame          # sorted by unit, time; columns: _id (codes), _t (codes), y, x…
    y: str
    x: list
    units: list               # labels
    times: list               # labels (sorted)
    entity: str = "unit"
    time: str = "time"

    @property
    def N(self):
        return len(self.units)

    @property
    def T(self):
        return len(self.times)

    @property
    def O(self):
        return len(self.df)

    @property
    def Ti(self):
        return self.df.groupby("_id").size().to_numpy()

    @property
    def balanced(self):
        return bool((self.Ti == self.T).all())


def prepare(df: pd.DataFrame, entity, time: str, y: str, x: list, filters: dict | None = None) -> PanelData:
    d = df.copy()
    for c, v in (filters or {}).items():
        d = d[d[c].astype(str) == str(v)]
    ent = [entity] if isinstance(entity, str) else list(entity)
    lab = d[ent].astype(str).agg(" · ".join, axis=1) if len(ent) > 1 else d[ent[0]].astype(str)
    d = d.assign(_unit=lab)
    cols = [y] + list(x)
    d = d[["_unit", time] + cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    if d.duplicated(["_unit", time]).any():
        raise ValueError("Duplicate (unit, time) pairs: add another entity column or a filter so each unit has one "
                         "observation per period.")
    times = sorted(d[time].unique())
    units = sorted(d["_unit"].unique())
    d["_id"] = d["_unit"].map({u: i for i, u in enumerate(units)})
    d["_t"] = d[time].map({t: i for i, t in enumerate(times)})
    d = d.sort_values(["_id", "_t"]).reset_index(drop=True)
    if len(units) < 2 or len(times) < 2:
        raise ValueError("A panel needs at least two units and two periods.")
    return PanelData(d, y, list(x), units, times, " · ".join(ent), time)


def describe(pdata: PanelData) -> dict:
    """Panel structure and Stata-style xtsum (overall / between / within)."""
    d = pdata.df
    rows = []
    for v in [pdata.y] + pdata.x:
        s = d[v]
        m = d.groupby("_id")[v].transform("mean")
        bm = d.groupby("_id")[v].mean()
        w = s - m + s.mean()
        rows.append({"Variable": v, "Mean": s.mean(), "SD overall": s.std(), "SD between": bm.std(), "SD within": w.std(),
                     "Min": s.min(), "Max": s.max(), "Share of within variance": float(((s - m) ** 2).sum() / ((s - s.mean()) ** 2).sum())})
    Ti = pdata.Ti
    return {"N": pdata.N, "T": pdata.T, "obs": pdata.O, "balanced": pdata.balanced, "Ti min": int(Ti.min()),
            "Ti mean": float(Ti.mean()), "Ti max": int(Ti.max()), "xtsum": pd.DataFrame(rows)}


# ---------------- transformations ----------------
def _gmean(v: np.ndarray, g: np.ndarray, n: int) -> np.ndarray:
    """Group means broadcast back to observations (v: obs × k)."""
    v2 = v.reshape(len(v), -1)
    s = np.zeros((n, v2.shape[1]))
    np.add.at(s, g, v2)
    c = np.bincount(g, minlength=n)[:, None]
    return (s / c)[g].reshape(v.shape)


def _within(v, g, n):
    return v - _gmean(v, g, n)


# ---------------- estimation ----------------
@dataclass
class PanelResult:
    model: str
    names: list
    b: np.ndarray
    V: np.ndarray
    df_resid: int
    resid: np.ndarray            # residuals of the estimated (transformed) regression
    X: np.ndarray                # transformed regressors used for robust covariances
    ids: np.ndarray
    tcodes: np.ndarray
    pdata: PanelData
    cov: str = "classic"
    stats: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def se(self):
        return np.sqrt(np.diag(self.V))

    def table(self) -> pd.DataFrame:
        se = self.se
        t = self.b / se
        q = stats.t.ppf(0.975, self.df_resid)
        return pd.DataFrame({"Variable": self.names, "Coef.": self.b, "Std. Err.": se, "t": t,
                             "p-value": 2 * stats.t.sf(np.abs(t), self.df_resid), "CI low": self.b - q * se, "CI high": self.b + q * se})

    def to_reg(self, label: str | None = None) -> rg.RegResult:
        names = self.names
        se = self.se
        t = self.b / se
        q = stats.t.ppf(0.975, self.df_resid)
        rr = rg.RegResult(name=label or MODELS[self.model][0], method="panel", cov_type=self.cov, design=None,
                          params=pd.Series(self.b, index=names), bse=pd.Series(se, index=names), tvalues=pd.Series(t, index=names),
                          pvalues=pd.Series(2 * stats.t.sf(np.abs(t), self.df_resid), index=names),
                          ci=pd.DataFrame(np.c_[self.b - q * se, self.b + q * se], index=names, columns=["low", "high"]),
                          cov=pd.DataFrame(self.V, index=names, columns=names), dist="t", df_resid=self.df_resid,
                          nobs=len(self.resid), y=np.zeros(1), fitted=np.zeros(1), resid=self.resid, wresid=self.resid,
                          wexog=self.X)
        rr.rsquared = self.stats.get("R²", np.nan)
        rr.rsquared_adj = self.stats.get("Adj. R²", np.nan)
        rr.extra.update({"y_label": self.pdata.y, "family": "panel"})
        return rr


def _design(pdata: PanelData, time_dummies: bool = False):
    d = pdata.df
    y = d[pdata.y].to_numpy(float)
    X = d[pdata.x].to_numpy(float)
    names = list(pdata.x)
    if time_dummies:
        D = pd.get_dummies(d["_t"], drop_first=True).to_numpy(float)
        X = np.column_stack([X, D])
        names += [f"year={pdata.times[t]}" for t in sorted(d["_t"].unique())[1:]]
    return y, X, names


def _ols(y, X):
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    e = y - X @ b
    return b, e, XtXi


def fit(pdata: PanelData, model: str = "fe", cov: str = "classic", time_dummies: bool = False, dk_lag: int | None = None) -> PanelResult:
    """Static panel estimators with plm conventions."""
    d = pdata.df
    g, tc = d["_id"].to_numpy(), d["_t"].to_numpy()
    N, O = pdata.N, pdata.O
    y, X, names = _design(pdata, time_dummies and model not in ("fe2", "be"))
    st = {}
    if model == "pooled":
        Xc = np.column_stack([np.ones(O), X])
        nm = ["const"] + names
        b, e, XtXi = _ols(y, Xc)
        dfr = O - Xc.shape[1]
        Xt, yt = Xc, y
    elif model in ("fe", "fe2"):
        if model == "fe2":
            D = pd.get_dummies(tc, drop_first=True).to_numpy(float)
            Xa = np.column_stack([X, D])
        else:
            Xa = X
        Xw, yw = _within(Xa, g, N), _within(y, g, N)
        b_all, e, XtXi_all = _ols(yw, Xw)
        k = X.shape[1]
        if model == "fe2":
            # partial out the time dummies (FWL) so that the reported block uses the slope regressors only
            Dw = Xw[:, k:]
            P = lambda A: A - Dw @ np.linalg.lstsq(Dw, A, rcond=None)[0]  # noqa: E731
            Xt, yt = P(Xw[:, :k]), P(yw)
        else:
            Xt, yt = Xw, yw
        b, e, XtXi = _ols(yt, Xt)
        nm = names
        dfr = O - N - k - ((pdata.T - 1) if model == "fe2" else 0)
        fe_ = _gmean(y - X @ b, g, N)
        st["Mean of unit effects"] = float(np.mean(y - X @ b))
        del b_all, XtXi_all
    elif model == "be":
        ym = pd.Series(y).groupby(g).mean().to_numpy()
        Xm = pd.DataFrame(X).groupby(g).mean().to_numpy()
        Xt = np.column_stack([np.ones(N), Xm])
        yt = ym
        nm = ["const"] + names
        b, e, XtXi = _ols(yt, Xt)
        dfr = N - Xt.shape[1]
        g, tc = np.arange(N), np.zeros(N, int)
    elif model == "fd":
        prev_ok = np.r_[False, g[1:] == g[:-1]]     # differences of consecutive available observations (plm)
        idx = np.where(prev_ok)[0]
        yt = y[idx] - y[idx - 1]
        Xd = X[idx] - X[idx - 1]
        Xt = np.column_stack([np.ones(len(idx)), Xd])
        nm = ["const"] + names
        b, e, XtXi = _ols(yt, Xt)
        dfr = len(yt) - Xt.shape[1]
        g, tc = g[idx], tc[idx]
    elif model == "re":
        er = ercomp(pdata, X, y)
        th = er["theta_i"]
        yt = y - th * _gmean(y, g, N)
        Xc = np.column_stack([np.ones(O), X])
        Xt = Xc - th[:, None] * _gmean(Xc, g, N)
        nm = ["const"] + names
        b, e, XtXi = _ols(yt, Xt)
        dfr = O - Xt.shape[1]
        st.update({"σ²(idiosyncratic)": er["s2_e"], "σ²(individual)": er["s2_u"], "θ (mean)": float(np.mean(th)),
                   "ρ (share of individual variance)": er["s2_u"] / (er["s2_u"] + er["s2_e"])})
    else:
        raise ValueError(model)
    s2 = e @ e / dfr
    V = s2 * XtXi
    r = PanelResult(model, nm, b, V, dfr, e, Xt, g, tc, pdata, cov, st)
    tss = ((yt - yt.mean()) ** 2).sum() if model != "fe" and model != "fe2" else (yt ** 2).sum()
    r2 = 1 - e @ e / tss
    kk = Xt.shape[1]
    r.stats.update({"N (obs)": len(e), "Units": len(np.unique(g)), "R²": r2,
                    "Adj. R²": 1 - (1 - r2) * (len(e) - (1 if "const" in nm else 0)) / dfr, "σ": np.sqrt(s2), "SSR": float(e @ e)})
    if model in ("fe", "fe2"):
        # overall F test that all unit effects are zero (plm::pFtest)
        pool = fit(pdata, "pooled", time_dummies=model == "fe2")
        ssr_p, ssr_w = pool.stats["SSR"], float(e @ e)
        df1 = pool.df_resid - dfr
        F = ((ssr_p - ssr_w) / df1) / (ssr_w / dfr)
        r.stats.update({"F (effects = 0)": F, "p (F effects)": float(stats.f.sf(F, df1, dfr)), "df (F effects)": (df1, dfr)})
    if cov != "classic":
        r.V = robust_vcov(r, cov, dk_lag)
        r.cov = cov
    return r


def fgls(pdata: PanelData, panels: str = "hetero", time_dummies: bool = False) -> PanelResult:
    """Feasible GLS (Stata xtgls): two-step, from pooled-OLS residuals. panels = 'hetero' — unit-specific variances
    σ²ᵢ = e_i'e_i / T_i; 'correlated' — also contemporaneous cross-unit covariances Σ_ij = e_i'e_j / T (Parks; balanced
    panel with T ≥ N). V = (X'Ω⁻¹X)⁻¹ (large-sample, as xtgls)."""
    y, X, names = _design(pdata, time_dummies)
    X = np.column_stack([np.ones(len(y)), X])
    nm = ["const"] + names
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b0
    ids = pdata.df["_id"].to_numpy()
    tc = pdata.df["_t"].to_numpy()
    if panels == "hetero":
        s2 = pd.Series(e ** 2).groupby(ids).mean().to_numpy()
        w = 1 / s2[ids]
        XtW = X.T * w
        A = XtW @ X
        b = np.linalg.solve(A, XtW @ y)
        V = np.linalg.inv(A)
        extra = {"sigma2_i": s2}
    elif panels == "correlated":
        if not pdata.balanced or pdata.T < pdata.N:
            raise ValueError("Correlated-panels FGLS needs a balanced panel with at least as many periods as units (T ≥ N).")
        E = e.reshape(pdata.N, pdata.T)
        S = E @ E.T / pdata.T
        Si = np.linalg.inv(S)
        Xr = X.reshape(pdata.N, pdata.T, -1)
        yr = y.reshape(pdata.N, pdata.T)
        A = np.einsum("ij,itk,jtl->kl", Si, Xr, Xr)
        c = np.einsum("ij,itk,jt->k", Si, Xr, yr)
        b = np.linalg.solve(A, c)
        V = np.linalg.inv(A)
        extra = {"Sigma": S}
    else:
        raise ValueError(panels)
    r = y - X @ b
    res = PanelResult("fgls", nm, b, V, len(y) - len(b), r, X, ids, tc, pdata, "fgls", {}, extra)
    res.stats["panels"] = panels
    return res


def ercomp(pdata: PanelData, X: np.ndarray, y: np.ndarray) -> dict:
    """Swamy-Arora variance components (plm 'swar', general unbalanced formulas)."""
    d = pdata.df
    g = d["_id"].to_numpy()
    N, O = pdata.N, pdata.O
    K = X.shape[1]
    Ti = np.bincount(g, minlength=N).astype(float)
    # within residual quadratic form
    Xw, yw = _within(X, g, N), _within(y, g, N)
    bw, ew, _ = _ols(yw, Xw)
    q1 = ew @ ew
    s2_e = q1 / (O - N - K)
    # 'Between' (observation-weighted) regression on repeated means
    Z = np.column_stack([np.ones(O), X])
    ZB = _gmean(Z, g, N)
    yB = _gmean(y, g, N)
    bB = np.linalg.lstsq(ZB, yB, rcond=None)[0]
    eB = y - Z @ bB
    q2 = float((_gmean(eB, g, N) ** 2).sum())
    ZS = ZB * Ti[g][:, None]
    m_id_eta = O - np.trace(np.linalg.solve(ZB.T @ ZB, ZS.T @ Z))
    m_id_nu = N - K - 1
    s2_u = max((q2 - m_id_nu * s2_e) / m_id_eta, 0.0)
    theta = 1 - np.sqrt(s2_e / (Ti * s2_u + s2_e))
    return {"s2_e": float(s2_e), "s2_u": float(s2_u), "theta": theta, "theta_i": theta[g]}


def robust_vcov(r: PanelResult, kind: str, dk_lag: int | None = None) -> np.ndarray:
    X, u = r.X, r.resid
    n_obs, k = X.shape
    A = np.linalg.pinv(X.T @ X)
    g, tc = r.ids, r.tcodes
    if kind == "cluster":
        G = len(np.unique(g))
        H = np.zeros((G, k))
        np.add.at(H, np.searchsorted(np.unique(g), g), X * u[:, None])
        S = H.T @ H * (G / (G - 1)) * ((n_obs - 1) / (n_obs - k))
        return A @ S @ A
    if kind == "dk":
        tu = np.unique(tc)
        H = np.zeros((len(tu), k))
        np.add.at(H, np.searchsorted(tu, tc), X * u[:, None])
        Tmax = int(r.pdata.Ti.max()) - (1 if r.model == "fd" else 0)
        L = int(np.floor(Tmax ** 0.25)) if dk_lag is None else dk_lag
        S = H.T @ H
        for l in range(1, L + 1):
            w = 1 - l / (L + 1)
            Gl = H[l:].T @ H[:-l]
            S += w * (Gl + Gl.T)
        return A @ S @ A
    if kind == "pcse":
        units = np.unique(g)
        times = np.unique(tc)
        E = np.full((len(times), len(units)), np.nan)
        E[np.searchsorted(times, tc), np.searchsorted(units, g)] = u
        # contemporaneous covariance of units, averaged over periods (pairwise available)
        M = ~np.isnan(E)
        Ez = np.nan_to_num(E)
        cnt = M.T.astype(float) @ M.astype(float)
        Sig = (Ez.T @ Ez) / np.where(cnt > 0, cnt, 1)
        S = np.zeros((k, k))
        for ti, t in enumerate(times):
            sel = tc == t
            Xi = X[sel]
            ui = np.searchsorted(units, g[sel])
            S += Xi.T @ Sig[np.ix_(ui, ui)] @ Xi
        return A @ S @ A
    raise ValueError(kind)


# ---------------- specification tests ----------------
def hausman(fe: PanelResult, re: PanelResult) -> dict:
    """Classical Hausman (1978) test FE vs RE on the common slope coefficients (plm::phtest)."""
    c = [n for n in fe.names if n in re.names]
    i1, i2 = [fe.names.index(n) for n in c], [re.names.index(n) for n in c]
    d = fe.b[i1] - re.b[i2]
    Vd = fe.V[np.ix_(i1, i1)] - re.V[np.ix_(i2, i2)]
    H = abs(float(d @ np.linalg.solve(Vd, d)))
    out = {"stat": H, "df": len(c), "p": float(stats.chi2.sf(H, len(c)))}
    if np.any(np.linalg.eigvalsh((Vd + Vd.T) / 2) < 0):
        out["note"] = ("V(FE) − V(RE) is not positive definite; the absolute value is reported (as R plm). Prefer the "
                       "robust (Mundlak) version.")
    return out


def hausman_robust(pdata: PanelData, time_dummies: bool = False) -> dict:
    """Regression-based (Mundlak / Wooldridge 2010) robust Hausman test: RE-quasi-demeaned regression augmented
    with the within-demeaned regressors; cluster-robust Wald test that their coefficients are zero."""
    d = pdata.df
    g = d["_id"].to_numpy()
    N, O = pdata.N, pdata.O
    y, X, names = _design(pdata, time_dummies)
    er = ercomp(pdata, X, y)
    th = er["theta_i"]
    Xs = np.column_stack([np.ones(O), X - th[:, None] * _gmean(X, g, N)])   # plain intercept (plm 'aux')
    ys = y - th * _gmean(y, g, N)
    Xw = _within(X, g, N)
    keep = np.abs(Xw).sum(0) > 1e-12
    Z = np.column_stack([Xs, Xw[:, keep]])
    b, e, A = _ols(ys, Z)
    G = N
    H = np.zeros((G, Z.shape[1]))
    np.add.at(H, g, Z * e[:, None])
    V = A @ (H.T @ H) @ A
    idx = np.arange(Xs.shape[1], Z.shape[1])
    W = float(b[idx] @ np.linalg.solve(V[np.ix_(idx, idx)], b[idx]))
    return {"stat": W, "df": len(idx), "p": float(stats.chi2.sf(W, len(idx)))}


def bp_lm(pooled: PanelResult) -> dict:
    """Breusch-Pagan (1980) LM test for individual random effects on pooled OLS residuals (unbalanced Baltagi-Li form),
    plus Honda's one-sided version."""
    e = pooled.resid
    g = pooled.ids
    O = len(e)
    Ti = np.bincount(g)
    Ti = Ti[Ti > 0]
    s = pd.Series(e).groupby(g).sum().to_numpy()
    A = (s ** 2).sum() / (e @ e) - 1
    LM = O ** 2 / (2 * ((Ti ** 2).sum() - O)) * A ** 2
    honda = np.sqrt(O ** 2 / (2 * ((Ti ** 2).sum() - O))) * A
    return {"LM": float(LM), "p": float(stats.chi2.sf(LM, 1)), "Honda": float(honda), "p_honda": float(stats.norm.sf(honda))}


def _resid_matrix(r: PanelResult):
    units, times = np.unique(r.ids), np.unique(r.tcodes)
    E = np.full((len(times), len(units)), np.nan)
    E[np.searchsorted(times, r.tcodes), np.searchsorted(units, r.ids)] = r.resid
    return E


def cross_section_dependence(r: PanelResult) -> pd.DataFrame:
    """Pesaran (2004) CD, Breusch-Pagan LM, Pesaran scaled LM and Baltagi-Feng-Kao bias-corrected scaled LM on the
    model residuals (pairwise over common periods, plm::pcdtest)."""
    E = _resid_matrix(r)
    Tt, N = E.shape
    cd = lm = sclm = 0.0
    npair, tmax = 0, 0
    for i in range(N):
        for j in range(i + 1, N):
            m = ~np.isnan(E[:, i]) & ~np.isnan(E[:, j])
            Tij = int(m.sum())
            if Tij < 2:
                continue
            a, b_ = E[m, i], E[m, j]
            a, b_ = a - a.mean(), b_ - b_.mean()
            den = np.sqrt((a @ a) * (b_ @ b_))
            if den <= 0:
                continue
            rho = (a @ b_) / den
            cd += np.sqrt(Tij) * rho
            lm += Tij * rho ** 2
            sclm += Tij * rho ** 2 - 1
            npair += 1
            tmax = max(tmax, Tij)
    CD = cd / np.sqrt(npair)
    SC = sclm / np.sqrt(2 * npair)
    BC = SC - N / (2 * (tmax - 1))
    return pd.DataFrame([
        {"Test": "Pesaran CD", "Statistic": CD, "Distribution": "N(0,1)", "p-value": float(2 * stats.norm.sf(abs(CD)))},
        {"Test": "Breusch-Pagan LM", "Statistic": lm, "Distribution": f"χ²({int(npair)})", "p-value": float(stats.chi2.sf(lm, npair))},
        {"Test": "Pesaran scaled LM", "Statistic": SC, "Distribution": "N(0,1)", "p-value": float(2 * stats.norm.sf(abs(SC)))},
        {"Test": "Bias-corrected scaled LM (Baltagi-Feng-Kao)", "Statistic": BC, "Distribution": "N(0,1)",
         "p-value": float(2 * stats.norm.sf(abs(BC)))}])


def wooldridge_ar(pdata: PanelData, time_dummies: bool = False) -> dict:
    """Wooldridge (2002) test for serial correlation in FE models (plm::pwartest): regress the within residuals on
    their lag and test the coefficient = −1/(T − 1) with a cluster-robust covariance."""
    fe = fit(pdata, "fe", time_dummies=time_dummies)
    d = pdata.df
    g, tc = d["_id"].to_numpy(), d["_t"].to_numpy()
    e = fe.resid
    ok = np.r_[False, g[1:] == g[:-1]]                  # previous available observation of the same unit (plm)
    idx = np.where(ok)[0]
    yv, xv = e[idx], e[idx - 1]
    Xc = np.column_stack([np.ones(len(idx)), xv])
    b, u, A = _ols(yv, Xc)
    gg = g[idx]
    G = len(np.unique(gg))
    H = np.zeros((pdata.N, 2))
    np.add.at(H, gg, Xc * u[:, None])
    V = A @ (H.T @ H) @ A
    h0 = -1 / (pdata.T - 1)
    F = (b[1] - h0) ** 2 / V[1, 1]
    dfr = len(idx) - 2
    return {"stat": float(F), "df": (1, dfr), "p": float(stats.f.sf(F, 1, dfr)), "coef": float(b[1]), "h0": h0}


def wooldridge_fd(pdata: PanelData, h0: str = "fe") -> dict:
    """Wooldridge test on first-difference residuals (Drukker 2003; Stata xtserial, plm::pwfdtest): regress ê_FD,t on
    ê_FD,t−1. h0 = 'fe': no serial correlation in the level errors (slope −0.5, as xtserial); h0 = 'fd': no serial
    correlation in the differenced errors (slope 0)."""
    r = fit(pdata, "fd")
    e = r.resid
    g = r.ids
    ok = np.r_[False, g[1:] == g[:-1]]
    idx = np.where(ok)[0]
    yv, xv = e[idx], e[idx - 1]
    Xc = np.column_stack([np.ones(len(idx)), xv])
    b, u, A = _ols(yv, Xc)
    gg = g[idx]
    H = np.zeros((pdata.N, 2))
    np.add.at(H, gg, Xc * u[:, None])
    V = A @ (H.T @ H) @ A
    rho0 = -0.5 if h0 == "fe" else 0.0
    F = (b[1] - rho0) ** 2 / V[1, 1]
    dfr = len(idx) - 2
    return {"stat": float(F), "df": (1, dfr), "p": float(stats.f.sf(F, 1, dfr)), "coef": float(b[1]), "h0": rho0}


def groupwise_het(fe: PanelResult) -> dict:
    """Greene (2000) modified Wald test for groupwise heteroskedasticity in FE residuals (Stata xttest3)."""
    e, g = fe.resid, fe.ids
    W = 0.0
    s2 = e @ e / len(e)
    n = 0
    for i in np.unique(g):
        ei = e[g == i]
        Ti = len(ei)
        si = ei @ ei / Ti
        Vi = ((ei ** 2 - si) ** 2).sum() / (Ti * (Ti - 1))
        if Vi > 0:
            W += (si - s2) ** 2 / Vi
            n += 1
    return {"stat": float(W), "df": n, "p": float(stats.chi2.sf(W, n))}


def slope_homogeneity(pdata: PanelData) -> dict:
    """Pesaran & Yamagata (2008) Δ and Δ_adj tests of slope homogeneity (unit-specific intercepts, weighted FE)."""
    d = pdata.df
    k = len(pdata.x)
    Xs, ys, sig = [], [], []
    fe = fit(pdata, "fe")
    for i in range(pdata.N):
        di = d[d["_id"] == i]
        X = di[pdata.x].to_numpy(float)
        y = di[pdata.y].to_numpy(float)
        Xm, ym = X - X.mean(0), y - y.mean()
        Xs.append(Xm); ys.append(ym)
        u = ym - Xm @ fe.b
        sig.append(u @ u / (len(y) - 1))
    A = sum(X.T @ X / s for X, s in zip(Xs, sig))
    bwfe = np.linalg.solve(A, sum(X.T @ y / s for X, y, s in zip(Xs, ys, sig)))
    S = 0.0
    for X, y, s in zip(Xs, ys, sig):
        bi = np.linalg.lstsq(X, y, rcond=None)[0]
        dlt = bi - bwfe
        S += dlt @ (X.T @ X) @ dlt / s
    N = pdata.N
    T = float(np.mean(pdata.Ti))
    D = np.sqrt(N) * (S / N - k) / np.sqrt(2 * k)
    Da = np.sqrt(N) * (S / N - k) / np.sqrt(2 * k * (T - k - 1) / (T + 1))
    return {"Delta": float(D), "p": float(2 * stats.norm.sf(abs(D))), "Delta_adj": float(Da), "p_adj": float(2 * stats.norm.sf(abs(Da)))}


# ---------------- panel unit roots ----------------
@lru_cache(maxsize=1)
def _tables():
    return json.loads((Path(__file__).resolve().parent.parent / "data" / "panel_tables.json").read_text())


def _arr(t):
    a = np.array([np.nan if v is None else v for v in t["data"]], float)
    return a.reshape(t["dim"], order="F")


def _padf(t, exo):
    from statsmodels.tsa.adfvalues import mackinnonp
    return float(mackinnonp(t, regression={"none": "n", "intercept": "c", "trend": "ct"}[exo], N=1))


def _adf_unit(x: np.ndarray, exo: str, lags: int):
    T = len(x)
    dy = np.r_[np.nan, np.diff(x)]
    ly = np.r_[np.nan, x[:-1]]
    cols = [ly] + [np.r_[[np.nan] * j, dy[:-j]] for j in range(1, lags + 1)]
    if exo in ("intercept", "trend"):
        cols.append(np.ones(T))
    if exo == "trend":
        cols.append(np.arange(1, T + 1, dtype=float))
    X = np.column_stack(cols)[lags + 1:]
    y = dy[lags + 1:]
    b, e, XtXi = _ols(y, X)
    n = len(y)
    sigma = np.sqrt(e @ e / n)
    se = sigma * np.sqrt(np.diag(XtXi))
    out = {"rho": b[0], "se": se[0], "t": b[0] / se[0], "sigma": sigma, "T": T, "lags": lags}
    return out, X, y


def _lagsel(x, exo, method, pmax):
    T = len(x)
    dy = np.r_[np.nan, np.diff(x)]
    ly = np.r_[np.nan, x[:-1]]
    m = []
    if exo in ("intercept", "trend"):
        m.append(np.ones(T))
    if exo == "trend":
        m.append(np.arange(1, T + 1, dtype=float))
    LD = [np.r_[[np.nan] * j, dy[:-j]] for j in range(1, pmax + 1)]
    narow = pmax + 1
    crit = []
    for lags in range(pmax, -1, -1):
        X = np.column_stack([ly] + LD[:lags] + m)[narow:]
        y = dy[narow:]
        b, e, XtXi = _ols(y, X)
        n, K = len(y), X.shape[1]
        if method == "Hall":
            s = np.sqrt(e @ e / n)
            tl = b[lags] / (s * np.sqrt(XtXi[lags, lags])) if lags else 0
            if abs(tl) >= 1.96 or lags == 0:
                return lags
            continue
        crit.append(np.log(e @ e / n) + (2 * K / n if method == "AIC" else K * np.log(n) / n))
    return pmax - int(np.argmin(crit))


def _longrunvar(x, exo, q=None):
    T = len(x)
    q = round(3.21 * T ** (1 / 3)) if q is None else q
    dx = np.diff(x)
    if exo == "intercept":
        dx = dx - dx.mean()
    elif exo == "trend":
        Z = np.column_stack([np.ones(len(dx)), np.arange(1, len(dx) + 1)])
        dx = dx - Z @ np.linalg.lstsq(Z, dx, rcond=None)[0]
    res = (dx ** 2).sum() / (T - 1)
    for L in range(1, q + 1):
        res += 2 * (dx[:-L] @ dx[L:]) / (T - 1) * (1 - L / (q + 1))
    return res


def _interp_T(table_T, l):
    Ts = np.asarray(table_T, float)
    if l <= Ts[0]:
        return 0, 0, 0.0
    if l >= Ts[-1]:
        return len(Ts) - 1, len(Ts) - 1, 0.0
    if l in Ts:
        j = int(np.where(Ts == l)[0][0])
        return j, j, 0.0
    j = int(np.searchsorted(Ts, l))
    return j - 1, j, (l - Ts[j - 1]) / (Ts[j] - Ts[j - 1])


def unit_root(series: dict, test: str = "ips", exo: str = "intercept", lags="SIC", pmax: int = 4) -> dict:
    """Panel unit-root tests on a dict {unit: 1-D array} (plm::purtest): 'levinlin' (LLC), 'ips' (W̄), 'madwu'
    (Fisher-ADF), 'Pm', 'invnormal', 'logit' (Choi), 'hadri' (H0: stationarity)."""
    obj = [np.asarray(v, float) for v in series.values()]
    n = len(obj)
    if test == "hadri":
        L = {len(x) for x in obj}
        if len(L) > 1:
            raise ValueError("Hadri's test needs a balanced panel.")
        L = L.pop()
        res = []
        for x in obj:
            Z = np.ones((L, 1)) if exo == "intercept" else np.column_stack([np.ones(L), np.arange(1, L + 1)])
            res.append(x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0])
        adj = (1 / 6, 1 / 45) if exo == "intercept" else (1 / 15, 11 / 6300)
        s2i = np.array([np.mean(e ** 2) for e in res])
        LMi = np.array([np.sum(np.cumsum(e) ** 2) for e in res]) / L ** 2 / s2i
        z = np.sqrt(n) * (LMi.mean() - adj[0]) / np.sqrt(adj[1])
        return {"test": "hadri", "stat": float(z), "p": float(stats.norm.sf(z)), "name": "Hadri Z (heteroskedasticity-consistent)",
                "h0": "all series stationary"}
    if isinstance(lags, str):
        lagv = [_lagsel(x, exo, lags, pmax) for x in obj]
    else:
        lagv = [int(lags)] * n
    idres = [_adf_unit(x, exo, l) for x, l in zip(obj, lagv)]
    trho = np.array([r[0]["t"] for r in idres])
    pv = np.array([_padf(t, exo) for t in trho])
    out = {"test": test, "lags": lagv, "t_i": trho, "p_i": pv, "h0": "all series have a unit root"}
    if test == "levinlin":
        Ls = {len(x) for x in obj}
        if len(Ls) > 1:
            raise ValueError("LLC needs a balanced panel.")
        Tl = Ls.pop()
        t = _tables()["levinlin"]
        A = _arr(t)
        Ts = [float(v) for v in t["dimnames"][0]]
        ei = ["none", "intercept", "trend"].index(exo)
        j0, j1, w = _interp_T(Ts, Tl)
        mu = A[j0, 0, ei] + w * (A[j1, 0, ei] - A[j0, 0, ei])
        sg = A[j0, 1, ei] + w * (A[j1, 1, ei] - A[j0, 1, ei])
        sST = np.array([r[0]["sigma"] for r in idres])
        sLT = np.sqrt([_longrunvar(x, exo) for x in obj])
        sbar = np.mean(sLT / sST)
        lev, dif = [], []
        for (r, X, y) in idres:
            sig = r["sigma"]
            if r["lags"] == 0 and exo == "none":
                dif.append(y / sig); lev.append(X[:, 0] / sig)
            else:
                Z = X[:, 1:]
                dif.append((y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]) / sig)
                lv = X[:, 0]
                lev.append((lv - Z @ np.linalg.lstsq(Z, lv, rcond=None)[0]) / sig)
        lev, dif = np.concatenate(lev), np.concatenate(dif)
        rho = (lev @ dif) / (lev @ lev)
        ee = dif - rho * lev
        nn = len(dif)
        sd = np.sqrt(ee @ ee / nn) / np.sqrt(lev @ lev)
        tT = Tl - np.mean(lagv) - 1
        s2e = (ee @ ee) / (n * tT)
        stat = (rho / sd - n * tT * sbar / s2e * sd * mu) / sg
        out.update({"stat": float(stat), "p": float(stats.norm.cdf(stat)), "name": "Levin-Lin-Chu t*", "rho": float(rho)})
    elif test == "ips":
        if exo == "none":
            raise ValueError("IPS needs individual intercepts.")
        t = _tables()["wtbar"]
        A = _arr(t)                         # lags × T × (mean,var) × (intercept,trend)
        Ts = [float(v) for v in t["dimnames"][1]]
        ei = 0 if exo == "intercept" else 1
        m_, v_ = [], []
        for (r, X, y), l in zip(idres, lagv):
            Li = r["T"] - l - 1
            j0, j1, w = _interp_T(Ts, Li)
            ll = min(l, 8)
            m_.append(A[ll, j0, 0, ei] + w * (A[ll, j1, 0, ei] - A[ll, j0, 0, ei]))
            v_.append(A[ll, j0, 1, ei] + w * (A[ll, j1, 1, ei] - A[ll, j0, 1, ei]))
        tbar = trho.mean()
        stat = np.sqrt(n) * (tbar - np.mean(m_)) / np.sqrt(np.mean(v_))
        out.update({"stat": float(stat), "p": float(stats.norm.cdf(stat)), "name": "Im-Pesaran-Shin W̄", "tbar": float(tbar)})
    elif test == "madwu":
        stat = -2 * np.log(pv).sum()
        out.update({"stat": float(stat), "p": float(stats.chi2.sf(stat, 2 * n)), "name": f"Maddala-Wu Fisher χ²({2 * n})"})
    elif test == "Pm":
        stat = 1 / (2 * np.sqrt(n)) * np.sum(-2 * np.log(pv) - 2)
        out.update({"stat": float(stat), "p": float(stats.norm.sf(stat)), "name": "Choi modified P (Pm)"})
    elif test == "invnormal":
        stat = stats.norm.ppf(pv).sum() / np.sqrt(n)
        out.update({"stat": float(stat), "p": float(stats.norm.cdf(stat)), "name": "Choi inverse normal Z"})
    elif test == "logit":
        L_ = np.log(pv / (1 - pv)).sum()
        kk = 3 * (5 * n + 4) / (np.pi ** 2 * n * (5 * n + 2))
        stat = np.sqrt(kk) * L_
        out.update({"stat": float(stat), "p": float(stats.t.cdf(stat, 5 * n + 4)), "name": "Choi logit L*"})
    return out


def cips(series: dict, lags: int = 1, type: str = "trend") -> dict:
    """Pesaran (2007) CIPS test (cross-sectionally augmented IPS) robust to a common factor; critical values from
    Pesaran (2007) Tables II(a)-(c), interpolated in N and T."""
    units = list(series)
    M = pd.DataFrame({u: pd.Series(np.asarray(v, float)) for u, v in series.items()})
    T = len(M)
    if T < 10:
        raise ValueError("CIPS needs at least 10 periods (Pesaran 2007 tables start at T = 10).")
    ybar = M.mean(1).to_numpy()
    tstats = []
    for u in units:
        x = M[u].to_numpy()
        dx = np.r_[np.nan, np.diff(x)]
        dyb = np.r_[np.nan, np.diff(ybar)]
        cols = [np.r_[np.nan, x[:-1]]] + [np.r_[[np.nan] * j, dx[:-j]] for j in range(1, lags + 1)]
        cols += [np.r_[np.nan, ybar[:-1]], dyb] + [np.r_[[np.nan] * j, dyb[:-j]] for j in range(1, lags + 1)]
        if type in ("drift", "trend"):
            cols.append(np.ones(T))
        if type == "trend":
            cols.append(np.arange(1, T + 1, dtype=float))
        X = np.column_stack(cols)
        ok = np.isfinite(X).all(1) & np.isfinite(dx)
        X, y = X[ok], dx[ok]
        b, e, XtXi = _ols(y, X)
        s2 = e @ e / (len(y) - X.shape[1])
        tstats.append(b[0] / np.sqrt(s2 * XtXi[0, 0]))
    stat = float(np.mean(tstats))
    tab = _tables()["cips"][{"none": "none", "drift": "drift", "trend": "trend"}[type]]
    C = _arr(tab)                         # n × T × level(1,5,10)
    grid = np.array(_tables()["cips_n"], float)
    n = len(units)
    nn, tt = np.clip(n, grid[0], grid[-1]), np.clip(T, grid[0], grid[-1])
    cv = []
    for lv in range(3):
        rowvals = [np.interp(tt, grid, C[i, :, lv]) for i in range(len(grid))]
        cv.append(float(np.interp(nn, grid, rowvals)))
    if stat < cv[0]:
        p = "< 0.01"
    elif stat > cv[2]:
        p = "> 0.10"
    else:
        p = f"{np.interp(stat, cv, [0.01, 0.05, 0.10]):.3f}"
    return {"stat": stat, "cv": {"1%": cv[0], "5%": cv[1], "10%": cv[2]}, "p": p, "t_i": np.array(tstats), "lags": lags}


def series_by_unit(pdata: PanelData, var: str) -> dict:
    d = pdata.df
    return {pdata.units[i]: g[var].to_numpy(float) for i, g in d.groupby("_id")}


# ---------------- causality ----------------
def dumitrescu_hurlin(pdata: PanelData, x: str, y: str, order: int = 1) -> dict:
    """Dumitrescu & Hurlin (2012) panel Granger non-causality test (plm::pgrangertest): W̄, Z̄ and Z̃."""
    d = pdata.df
    W, Ts = [], []
    for _, g in d.groupby("_id"):
        yy, xx = g[y].to_numpy(float), g[x].to_numpy(float)
        T = len(yy)
        cols = [np.r_[[np.nan] * j, yy[:-j]] for j in range(1, order + 1)] + [np.r_[[np.nan] * j, xx[:-j]] for j in range(1, order + 1)]
        X = np.column_stack([np.ones(T)] + cols)[order:]
        Y = yy[order:]
        b, e, XtXi = _ols(Y, X)
        s2 = e @ e / (len(Y) - X.shape[1])
        idx = np.arange(1 + order, 1 + 2 * order)
        Wi = float(b[idx] @ np.linalg.solve(s2 * XtXi[np.ix_(idx, idx)], b[idx]))
        W.append(Wi)
        Ts.append(T)
    W = np.array(W)
    N = len(W)
    Wbar = W.mean()
    Zbar = np.sqrt(N / (2 * order)) * (Wbar - order)
    Tt = np.array(Ts, float) - order
    K = order
    if np.all(Tt > 5 + 3 * K):
        EW = K * (Tt - 2 * K - 1) / (Tt - 2 * K - 3)
        VW = 2 * K * (Tt - 2 * K - 1) ** 2 * (Tt - K - 3) / ((Tt - 2 * K - 3) ** 2 * (Tt - 2 * K - 5))
        Ztil = (np.sqrt(N) * (Wbar - EW.mean())) / np.sqrt(VW.mean())
    else:
        Ztil = np.nan
    return {"Wbar": float(Wbar), "Zbar": float(Zbar), "p_Zbar": float(2 * stats.norm.sf(abs(Zbar))), "Ztilde": float(Ztil),
            "p_Ztilde": float(2 * stats.norm.sf(abs(Ztil))) if np.isfinite(Ztil) else np.nan, "W_i": W, "order": order}


# ---------------- heterogeneous panels: MG and CCE ----------------
def mean_group(pdata: PanelData, cce: bool = False, trend: bool = False) -> dict:
    """Pesaran-Smith (1995) mean group; with `cce`, Pesaran (2006) CCE-MG (unit regressions augmented with cross-
    sectional averages of y and x). Non-parametric MG standard errors (plm::pmg 'mg' / 'cmg')."""
    d = pdata.df.copy()
    cols = [pdata.y] + pdata.x
    if cce:
        avg = d.groupby("_t")[cols].transform("mean")
        for c in cols:
            d[f"{c}_bar"] = avg[c]
    B = []
    for _, g in d.groupby("_id"):
        X = [np.ones(len(g)), *[g[c].to_numpy(float) for c in pdata.x]]
        if trend:
            X.append(g["_t"].to_numpy(float) + 1)
        if cce:
            X += [g[f"{c}_bar"].to_numpy(float) for c in cols]
        X = np.column_stack(X)
        B.append(np.linalg.lstsq(X, g[pdata.y].to_numpy(float), rcond=None)[0][: 1 + len(pdata.x) + (1 if trend else 0)])
    B = np.array(B)
    N = len(B)
    b = B.mean(0)
    V = (B - b).T @ (B - b) / (N * (N - 1))
    names = ["const"] + pdata.x + (["trend"] if trend else [])
    se = np.sqrt(np.diag(V))
    z = b / se
    tab = pd.DataFrame({"Variable": names, "Coef.": b, "Std. Err.": se, "z": z, "p-value": 2 * stats.norm.sf(np.abs(z))})
    return {"table": tab, "b": b, "V": V, "units": B, "names": names}


def cce_pooled(pdata: PanelData) -> dict:
    """Pesaran (2006) CCE pooled estimator: slopes common, unit-specific loadings on the cross-sectional averages;
    non-parametric (Pesaran 2006, eq. 67) covariance."""
    d = pdata.df.copy()
    cols = [pdata.y] + pdata.x
    avg = d.groupby("_t")[cols].transform("mean")
    k = len(pdata.x)
    Ms, Xs, ys = [], [], []
    for _, g in d.groupby("_id"):
        H = np.column_stack([np.ones(len(g))] + [avg.loc[g.index, c].to_numpy(float) for c in cols])
        M = np.eye(len(g)) - H @ np.linalg.pinv(H)
        X = g[pdata.x].to_numpy(float)
        y = g[pdata.y].to_numpy(float)
        Xs.append(M @ X); ys.append(M @ y)
    A = sum(X.T @ X for X in Xs)
    b = np.linalg.solve(A, sum(X.T @ y for X, y in zip(Xs, ys)))
    N = len(Xs)
    bi = np.array([np.linalg.lstsq(X, y, rcond=None)[0] for X, y in zip(Xs, ys)])
    bmg = bi.mean(0)
    Psi = A / N
    R = sum((X.T @ X / len(X)) @ np.outer(bb - bmg, bb - bmg) @ (X.T @ X / len(X)) for X, bb in zip(Xs, bi)) / (N - 1)
    Ti = np.mean([len(X) for X in Xs])
    Psi_t = Psi / Ti
    V = np.linalg.inv(Psi_t) @ R @ np.linalg.inv(Psi_t) / N
    se = np.sqrt(np.diag(V))
    z = b / se
    tab = pd.DataFrame({"Variable": pdata.x, "Coef.": b, "Std. Err.": se, "z": z, "p-value": 2 * stats.norm.sf(np.abs(z))})
    return {"table": tab, "b": b, "V": V}
