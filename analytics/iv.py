# ==========================================
# Aetheris Research - Instrumental variables & simultaneous equations (Catalogue §1.2)
# ------------------------------------------
# Single equation: OLS / 2SLS / LIML / Fuller / two-step efficient GMM (k-class family) with
#   classical, HC0–HC3, cluster and Newey-West covariances; first-stage diagnostics, Cragg-Donald,
#   Stock-Yogo critical values, Sargan / Basmann / Hansen J, Durbin and Wu-Hausman endogeneity tests,
#   robust control-function test, Anderson-Rubin weak-IV-robust test, Hausman OLS-vs-IV contrast.
# Systems: identification (order & rank conditions), ILS, equation-by-equation 2SLS, 3SLS, SUR /
#   iterated SUR — validated against linearmodels and R (AER, systemfit).
# Pure numpy / scipy — no Streamlit.
# ==========================================

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import linalg, stats

from analytics import regression as rg

IV_METHODS = {
    "ols": ("OLS (ignores endogeneity — benchmark)", "المربعات الصغرى العادية (تتجاهل الداخلية — للمقارنة)"),
    "2sls": ("Two-stage least squares (2SLS)", "المربعات الصغرى ذات المرحلتين (2SLS)"),
    "liml": ("Limited-information maximum likelihood (LIML)", "الإمكان الأعظم بالمعلومات المحدودة (LIML)"),
    "fuller": ("Fuller-modified LIML (α = 1)", "LIML المعدَّل لفولر (α = 1)"),
    "gmm": ("Two-step efficient GMM", "طريقة العزوم المعممة الكفؤة بخطوتين (GMM)"),
}
SYS_METHODS = {
    "ols": ("Equation-by-equation OLS", "OLS لكل معادلة على حدة"),
    "2sls": ("Equation-by-equation 2SLS", "2SLS لكل معادلة على حدة"),
    "ils": ("Indirect least squares (exactly identified equations)", "المربعات الصغرى غير المباشرة (للمعادلات تامة التعرّف)"),
    "3sls": ("Three-stage least squares (3SLS)", "المربعات الصغرى ذات الثلاث مراحل (3SLS)"),
    "sur": ("Seemingly unrelated regressions (SUR, FGLS)", "الانحدارات غير المرتبطة ظاهرياً (SUR)"),
    "itsur": ("Iterated SUR (= maximum likelihood)", "SUR التكراري (= الإمكان الأعظم)"),
}

# Stock & Yogo (2005) critical values for the Cragg-Donald F statistic, ONE endogenous regressor (K2 = 1).
# Keys: number of excluded instruments L2. Size: maximal size of a nominal 5% Wald test; bias: maximal
# relative bias of 2SLS vs OLS (defined for L2 ≥ 3).
STOCK_YOGO_SIZE = {1: (16.38, 8.96, 6.66, 5.53), 2: (19.93, 11.59, 8.75, 7.25), 3: (22.30, 12.83, 9.54, 7.80),
                   4: (24.58, 13.96, 10.26, 8.31), 5: (26.87, 15.09, 10.98, 8.84)}
STOCK_YOGO_BIAS = {3: (13.91, 9.08, 6.46, 5.39), 4: (16.85, 10.27, 6.71, 5.34), 5: (18.37, 10.83, 6.77, 5.25)}


# ==========================================================
# Specification & matrices
# ==========================================================
@dataclass
class IVSpec:
    y: str
    exog: list = field(default_factory=list)         # included exogenous regressors
    endog: list = field(default_factory=list)        # endogenous regressors
    instruments: list = field(default_factory=list)  # excluded instruments
    categorical: list = field(default_factory=list)  # exogenous factors (dummies)
    form: str = "linear"
    nolog: list = field(default_factory=list)
    const: bool = True

    def model_spec(self) -> rg.ModelSpec:
        return rg.ModelSpec(y=self.y, x=list(self.exog) + list(self.endog), categorical=list(self.categorical),
                            form=self.form, nolog=list(self.nolog), const=self.const)


def _matrices(df: pd.DataFrame, spec: IVSpec):
    """y, X (exog + endog), Z (exog + excluded instruments), design, names of endog / excluded columns."""
    if not spec.endog:
        raise ValueError("Select at least one endogenous regressor.")
    if not spec.instruments:
        raise ValueError("Select at least one excluded instrument.")
    overlap = set(spec.instruments) & (set(spec.exog) | set(spec.endog) | {spec.y})
    if overlap:
        raise ValueError(f"Instruments cannot also be regressors / the dependent variable: {sorted(overlap)}")
    ms = spec.model_spec()
    zspec = rg.ModelSpec(y=spec.y, x=list(spec.exog) + list(spec.instruments), categorical=list(spec.categorical),
                         form=spec.form, nolog=list(spec.nolog), const=spec.const)
    d = df.copy()
    keep = pd.Series(True, index=d.index)
    for v in [spec.y] + spec.exog + spec.endog + spec.instruments + spec.categorical:
        keep &= d[v].notna()
    d = d[keep]
    design = rg.build_design(d, ms)
    zd = rg.build_design(d.loc[design.index], zspec)
    idx = design.index.intersection(zd.index)
    if len(idx) < len(design.index):
        design = rg.build_design(d.loc[idx], ms)
        zd = rg.build_design(d.loc[design.index], zspec)
    X = design.X
    Z = zd.X.loc[X.index]
    endog_cols = [c for c in X.columns if design.info["base"].get(c) in spec.endog]
    excl_cols = [c for c in Z.columns if zd.info["base"].get(c) in spec.instruments]
    exog_cols = [c for c in X.columns if c not in endog_cols]
    if len(excl_cols) < len(endog_cols):
        raise ValueError(f"Under-identified: {len(excl_cols)} excluded instrument(s) for {len(endog_cols)} "
                         "endogenous regressor(s) — the order condition requires at least as many instruments.")
    return design, X, Z, exog_cols, endog_cols, excl_cols


def _proj(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(Z)
    return Q @ (Q.T @ A)


def _liml_kappa(y, X, Z, exog_cols_idx):
    """Smallest eigenvalue of (W'M_Z W)^-1 (W'M_X1 W), W = [y, Y_endog]."""
    endog_idx = [j for j in range(X.shape[1]) if j not in exog_cols_idx]
    W = np.c_[y, X[:, endog_idx]]
    X1 = X[:, exog_cols_idx]
    MzW = W - _proj(Z, W)
    M1W = W - (_proj(X1, W) if X1.shape[1] else 0)
    A = W.T @ M1W
    B = W.T @ MzW
    return float(np.min(np.real(linalg.eigvals(np.linalg.solve(B, A)))))


def _hac(scores: np.ndarray, lags: int) -> np.ndarray:
    S = scores.T @ scores
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = scores[L:].T @ scores[:-L]
        S += w * (G + G.T)
    return S


def _meat(scores, cov_type, cov_opts, n, k, small):
    if cov_type in ("HC0", "HC1", "HC2", "HC3", "robust"):
        S = scores.T @ scores
        if small or cov_type == "HC1":
            S *= n / (n - k)
        return S
    if cov_type == "cluster":
        g = pd.factorize(pd.Series(cov_opts["groups"]))[0]
        G = g.max() + 1
        sums = np.zeros((G, scores.shape[1]))
        np.add.at(sums, g, scores)
        S = sums.T @ sums
        adj = G / (G - 1) * ((n - 1) / (n - k) if small else 1.0)
        cov_opts["n_clusters"] = int(G)
        return S * adj
    if cov_type == "HAC":
        lags = cov_opts.get("maxlags")
        if lags is None:
            lags = cov_opts["maxlags"] = rg.newey_west_lags(n)
        S = _hac(scores, int(lags))
        return S * (n / (n - k) if small else 1.0)
    raise ValueError(cov_type)


def _hc23(Xt, X, e, cov_type):
    """HC2 / HC3 leverage adjustment for k-class estimators (leverage of the instrumented regressors)."""
    h = np.einsum("ij,jk,ik->i", Xt, np.linalg.pinv(Xt.T @ X), X)
    h = np.clip(h, 0, 1 - 1e-12)
    return e / np.sqrt(1 - h) if cov_type == "HC2" else e / (1 - h)


def iv_fit(df: pd.DataFrame, spec: IVSpec, method: str = "2sls", cov_type: str = "nonrobust",
           cov_opts: dict | None = None, small: bool = True, name: str = "") -> rg.RegResult:
    """k-class IV estimation. `small=True` → σ² with n−K and t / F tests (R ivreg, EViews, Stata `small`);
    `small=False` → σ² with n and z / χ² tests (Stata ivregress default)."""
    cov_opts = dict(cov_opts or {})
    design, Xdf, Zdf, exog_cols, endog_cols, excl_cols = _matrices(df, spec)
    X, Z = Xdf.to_numpy(float), Zdf.to_numpy(float)
    y = design.y.to_numpy(float)
    n, K = X.shape
    L = Z.shape[1]
    if cov_type == "cluster" and isinstance(cov_opts.get("groups"), str):
        cov_opts["groups_name"] = cov_opts["groups"]
        cov_opts["groups"] = df.loc[design.index, cov_opts["groups"]].astype(str).to_numpy()
    exog_idx = [list(Xdf.columns).index(c) for c in exog_cols]
    kappa = None
    if method == "ols":
        Xt = X
    elif method in ("2sls", "gmm"):
        Xt = _proj(Z, X)
    elif method in ("liml", "fuller"):
        lam = _liml_kappa(y, X, Z, exog_idx)
        kappa = lam - (1.0 / (n - L) if method == "fuller" else 0.0)
        Xt = X - kappa * (X - _proj(Z, X))     # (I − κ M_Z) X
    else:
        raise ValueError(method)
    b = np.linalg.solve(Xt.T @ X, Xt.T @ y)
    e = y - X @ b
    notes = list(design.notes)
    J = None
    if method == "gmm":
        # two-step efficient GMM with a heteroskedasticity / cluster / HAC-robust weight matrix
        ct = cov_type if cov_type != "nonrobust" else "HC0"
        S = _meat(Z * e[:, None], ct, dict(cov_opts), n, K, False) / n
        Wm = np.linalg.pinv(S)
        ZX, Zy = Z.T @ X, Z.T @ y
        A = ZX.T @ Wm @ ZX
        b = np.linalg.solve(A, ZX.T @ Wm @ Zy)
        e = y - X @ b
        S2 = _meat(Z * e[:, None], ct, cov_opts, n, K, small) / n
        Ai = np.linalg.inv(A)
        cov = n * Ai @ (ZX.T @ Wm @ S2 @ Wm @ ZX) @ Ai
        gbar = Z.T @ e / n
        J = float(n * gbar @ np.linalg.pinv(_meat(Z * e[:, None], ct, dict(cov_opts), n, K, False) / n) @ gbar)
        if cov_type == "nonrobust":
            cov_type = "HC0"
        notes.append("Two-step efficient GMM: weight matrix from first-step 2SLS residuals; robust covariance.")
    else:
        s2 = e @ e / ((n - K) if small else n)
        bread = np.linalg.inv(Xt.T @ X)
        if cov_type == "nonrobust":
            cov = s2 * np.linalg.inv(Xt.T @ X) if method != "ols" else s2 * np.linalg.inv(X.T @ X)
            if method in ("liml", "fuller"):
                cov = s2 * np.linalg.inv(Xt.T @ X)
        else:
            ee = _hc23(Xt, X, e, cov_type) if cov_type in ("HC2", "HC3") else e
            meat = _meat(Xt * ee[:, None], cov_type if cov_type not in ("HC2", "HC3") else "HC0", cov_opts, n, K,
                         small and cov_type not in ("HC0", "HC2", "HC3"))
            cov = bread @ meat @ bread.T
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    tv = b / se
    dfr = n - K
    if small:
        pv = 2 * stats.t.sf(np.abs(tv), dfr if cov_type != "cluster" else cov_opts.get("n_clusters", dfr + 1) - 1)
        q = stats.t.ppf(0.975, dfr if cov_type != "cluster" else cov_opts.get("n_clusters", dfr + 1) - 1)
    else:
        pv = 2 * stats.norm.sf(np.abs(tv))
        q = stats.norm.ppf(0.975)
    names = list(Xdf.columns)
    r = rg.RegResult(name=name or IV_METHODS[method][0].split(" (")[0], method=method, cov_type=cov_type, design=design,
                     params=pd.Series(b, index=names), bse=pd.Series(se, index=names), tvalues=pd.Series(tv, index=names),
                     pvalues=pd.Series(pv, index=names),
                     ci=pd.DataFrame(np.c_[b - q * se, b + q * se], index=names, columns=["low", "high"]),
                     cov=pd.DataFrame(cov, index=names, columns=names), dist="t" if small else "z", df_resid=dfr,
                     nobs=n, y=y, fitted=X @ b, resid=e, wresid=e, wexog=X, notes=notes, cov_opts=cov_opts)
    sst = np.sum((y - y.mean()) ** 2) if spec.const else np.sum(y ** 2)
    r.ssr = float(e @ e)
    r.sigma = float(np.sqrt(e @ e / (dfr if small else n)))
    r.rsquared = 1 - r.ssr / sst
    r.rsquared_adj = 1 - (1 - r.rsquared) * (n - 1) / dfr
    slopes = [j for j, c in enumerate(names) if c != "const"]
    if slopes:
        bs = b[slopes]
        W = float(bs @ np.linalg.solve(cov[np.ix_(slopes, slopes)], bs))
        r.fvalue = W / len(slopes) if small else W
        r.f_pvalue = float(stats.f.sf(W / len(slopes), len(slopes), dfr) if small else stats.chi2.sf(W, len(slopes)))
    r.df_model = len(slopes)
    r.extra.update({"Z": Z, "Z_names": list(Zdf.columns), "exog_cols": exog_cols, "endog_cols": endog_cols,
                    "excl_cols": excl_cols, "kappa": kappa, "J": J, "small": small, "spec": spec, "Xt": Xt,
                    "method_label": IV_METHODS[method],
                    "stats": {"Wald χ²" if not small else "Wald F": r.fvalue}})
    if method == "ols":
        r.extra["stats"] = {}
    if kappa is not None:
        r.notes.append(f"k-class parameter κ = {kappa:.5f}" + (" (LIML)" if method == "liml" else " (Fuller, α = 1)"))
    return r


# ==========================================================
# Diagnostics
# ==========================================================
def _ols_rss(y, X):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return float(e @ e), e, b


def first_stage(r: rg.RegResult) -> pd.DataFrame:
    """First-stage regressions of each endogenous regressor on all instruments: partial R², F-test of the
    excluded instruments (classical and HC1-robust), Shea's partial R² (several endogenous regressors)."""
    X = r.design.X
    Z = r.extra["Z"]
    zn = r.extra["Z_names"]
    excl = [zn.index(c) for c in r.extra["excl_cols"]]
    inc = [j for j in range(Z.shape[1]) if j not in excl]
    n, L = Z.shape
    L2 = len(excl)
    rows = []
    shea = shea_partial_r2(r) if len(r.extra["endog_cols"]) > 1 else {}
    for c in r.extra["endog_cols"]:
        x = X[c].to_numpy(float)
        rss_u, e_u, b_u = _ols_rss(x, Z)
        rss_r, _, _ = _ols_rss(x, Z[:, inc]) if inc else (float(np.sum((x - 0) ** 2)), None, None)
        F = ((rss_r - rss_u) / L2) / (rss_u / (n - L))
        pr2 = (rss_r - rss_u) / rss_r
        # HC1-robust Wald test of the excluded instruments
        ZtZi = np.linalg.inv(Z.T @ Z)
        meat = (Z * e_u[:, None]).T @ (Z * e_u[:, None]) * n / (n - L)
        V = ZtZi @ meat @ ZtZi
        bb = b_u[excl]
        Wr = float(bb @ np.linalg.solve(V[np.ix_(excl, excl)], bb))
        rows.append({"Endogenous": c, "Partial R²": pr2, "Shea partial R²": shea.get(c, pr2), "F (excluded IVs)": F,
                     "df": f"({L2}, {n - L})", "p-value": float(stats.f.sf(F, L2, n - L)),
                     "Robust F (HC1)": Wr / L2, "Robust p": float(stats.f.sf(Wr / L2, L2, n - L)),
                     "Weak (F < 10)?": F < 10})
    return pd.DataFrame(rows)


def shea_partial_r2(r: rg.RegResult) -> dict:
    """Shea (1997) partial R²: relevance of the instruments for each endogenous regressor net of the others."""
    X = r.design.X.to_numpy(float)
    names = list(r.design.X.columns)
    Xh = _proj(r.extra["Z"], X)
    out = {}
    for c in r.extra["endog_cols"]:
        j = names.index(c)
        others = [k for k in range(X.shape[1]) if k != j]
        xt = X[:, j] - _proj(X[:, others], X[:, [j]])[:, 0]
        xht = Xh[:, j] - _proj(Xh[:, others], Xh[:, [j]])[:, 0]
        out[c] = float((xt @ xht) ** 2 / ((xt @ xt) * (xht @ xht)))
    return out


def cragg_donald(r: rg.RegResult) -> float:
    """Cragg-Donald (1993) minimum-eigenvalue F statistic (= first-stage F with one endogenous regressor)."""
    X = r.design.X
    Z = r.extra["Z"]
    zn = r.extra["Z_names"]
    excl = [zn.index(c) for c in r.extra["excl_cols"]]
    inc = [j for j in range(Z.shape[1]) if j not in excl]
    n, L = Z.shape
    X2 = X[r.extra["endog_cols"]].to_numpy(float)
    Z1, Z2 = Z[:, inc], Z[:, excl]
    if inc:
        X2p = X2 - _proj(Z1, X2)
        Z2p = Z2 - _proj(Z1, Z2)
    else:
        X2p, Z2p = X2, Z2
    A = X2p.T @ _proj(Z2p, X2p)
    Mz = X2 - _proj(Z, X2)
    B = Mz.T @ Mz
    lam = float(np.min(np.real(linalg.eigvals(np.linalg.solve(B, A)))))
    return lam * (n - L) / len(excl)


def iv_tests(df: pd.DataFrame, r: rg.RegResult) -> pd.DataFrame:
    """Weak-instrument, over-identification and endogeneity tests for an IV fit."""
    X = r.design.X.to_numpy(float)
    names = list(r.design.X.columns)
    y = r.y
    Z = r.extra["Z"]
    zn = r.extra["Z_names"]
    n, K = X.shape
    L = Z.shape[1]
    K2 = len(r.extra["endog_cols"])
    L2 = len(r.extra["excl_cols"])
    rows = []

    def add(group, test, test_ar, stat, dfv, p, h0, ok, note=""):
        rows.append({"group": group, "Test": test, "Test_ar": test_ar, "Statistic": stat, "df": dfv, "p-value": p,
                     "H0": h0, "ok": ok, "note": note})
    # --- weak instruments
    cd = cragg_donald(r)
    sy = ""
    if K2 == 1 and L2 in STOCK_YOGO_SIZE:
        s10, s15, s20, s25 = STOCK_YOGO_SIZE[L2]
        sy = f"Stock-Yogo (size): 10% {s10}, 15% {s15}, 20% {s20}, 25% {s25}"
        if L2 in STOCK_YOGO_BIAS:
            b5, b10, b20, b30 = STOCK_YOGO_BIAS[L2]
            sy += f"; relative bias: 5% {b5}, 10% {b10}, 20% {b20}, 30% {b30}"
    add("weak", "Cragg-Donald Wald F (min. eigenvalue)", "إحصاءة كراج-دونالد F (أصغر جذر مميز)", cd, f"({L2}, {n - L})",
        None, "instruments are weak", cd >= 10, sy or "rule of thumb: F ≥ 10 (Staiger & Stock 1997)")
    # Anderson-Rubin test of H0: β_endog = 0 (robust to weak instruments)
    excl = [zn.index(c) for c in r.extra["excl_cols"]]
    inc = [j for j in range(L) if j not in excl]
    rss_u, _, _ = _ols_rss(y, Z)
    rss_r = _ols_rss(y, Z[:, inc])[0] if inc else float(y @ y)
    AR = ((rss_r - rss_u) / L2) / (rss_u / (n - L))
    pAR = float(stats.f.sf(AR, L2, n - L))
    add("weak", "Anderson-Rubin F (H₀: endogenous coefficients = 0)", "أندرسون-روبن F (فرض العدم: معاملات المتغيرات الداخلية = 0)",
        AR, f"({L2}, {n - L})", pAR, "β_endog = 0", None, "valid even with weak instruments")
    # --- over-identification
    e2sls = y - X @ np.linalg.solve(_proj(Z, X).T @ X, _proj(Z, X).T @ y) if r.method != "2sls" else r.resid
    if L > K:
        ePe = float(e2sls @ _proj(Z, e2sls[:, None])[:, 0])
        eMe = float(e2sls @ e2sls) - ePe
        sargan = n * ePe / (e2sls @ e2sls)
        basmann = ePe / (eMe / (n - L))
        q = L - K
        add("overid", "Sargan χ² (homoskedastic)", "سارجان χ² (بافتراض تجانس التباين)", sargan, q,
            float(stats.chi2.sf(sargan, q)), "all instruments valid", None)
        add("overid", "Basmann χ²", "باسمان χ²", basmann, q, float(stats.chi2.sf(basmann, q)), "all instruments valid", None)
        # Hansen J from two-step GMM (robust)
        Se = Z * e2sls[:, None]
        S = Se.T @ Se / n
        Wm = np.linalg.pinv(S)
        ZX, Zy = Z.T @ X, Z.T @ y
        bg = np.linalg.solve(ZX.T @ Wm @ ZX, ZX.T @ Wm @ Zy)
        eg = y - X @ bg
        gbar = Z.T @ eg / n
        Jst = float(n * gbar @ Wm @ gbar)
        add("overid", "Hansen J (heteroskedasticity-robust)", "هانسن J (متين لعدم تجانس التباين)", Jst, q,
            float(stats.chi2.sf(Jst, q)), "all instruments valid", None)
        for row in rows[-3:]:
            row["ok"] = row["p-value"] >= 0.05
    else:
        add("overid", "Over-identification", "اختبار فرط التحديد", np.nan, 0, np.nan, "—", None,
            "exactly identified: over-identifying restrictions cannot be tested")
    # --- endogeneity (control function)
    endog = r.extra["endog_cols"]
    V = np.column_stack([X[:, names.index(c)] - _proj(Z, X[:, [names.index(c)]])[:, 0] for c in endog])
    Xa = np.c_[X, V]
    rss_o, _, _ = _ols_rss(y, X)
    rss_a, ea, ba = _ols_rss(y, Xa)
    wu = ((rss_o - rss_a) / K2) / (rss_a / (n - K - K2))
    add("endog", "Wu-Hausman F", "وو-هاوسمان F", wu, f"({K2}, {n - K - K2})", float(stats.f.sf(wu, K2, n - K - K2)),
        "regressors are exogenous", None)
    durbin = n * (rss_o - rss_a) / rss_o
    add("endog", "Durbin χ²", "دربن χ²", durbin, K2, float(stats.chi2.sf(durbin, K2)), "regressors are exogenous", None)
    XtXi = np.linalg.inv(Xa.T @ Xa)
    meat = (Xa * ea[:, None]).T @ (Xa * ea[:, None]) * n / (n - K - K2)
    Va = XtXi @ meat @ XtXi
    bv = ba[K:]
    rob = float(bv @ np.linalg.solve(Va[K:, K:], bv)) / K2
    add("endog", "Robust control-function F (HC1)", "اختبار دالة التحكم المتين F (HC1)", rob, f"({K2}, {n - K - K2})",
        float(stats.f.sf(rob, K2, n - K - K2)), "regressors are exogenous", None,
        "Wooldridge (2010): valid under heteroskedasticity")
    for row in rows[-3:]:
        row["ok"] = None  # informative: rejection means IV is needed, not a model failure
    out = pd.DataFrame(rows)
    return out


def hausman_ols_iv(df: pd.DataFrame, r_iv: rg.RegResult) -> dict:
    """Classical Hausman (1978) contrast of IV and OLS coefficients on the endogenous regressors."""
    r_ols = iv_fit(df, r_iv.extra["spec"], "ols", small=r_iv.extra["small"])
    c = r_iv.extra["endog_cols"]
    d = r_iv.params[c].to_numpy() - r_ols.params[c].to_numpy()
    s2 = r_iv.ssr / (r_iv.nobs - r_iv.k)  # common σ² from the consistent estimator
    Xt = _proj(r_iv.extra["Z"], r_iv.wexog)
    Vi = s2 * np.linalg.inv(Xt.T @ r_iv.wexog)
    Vo = s2 * np.linalg.inv(r_iv.wexog.T @ r_iv.wexog)
    idx = [r_iv.names.index(x) for x in c]
    Vd = Vi[np.ix_(idx, idx)] - Vo[np.ix_(idx, idx)]
    H = float(d @ np.linalg.pinv(Vd) @ d)
    return {"H": H, "df": len(c), "p": float(stats.chi2.sf(H, len(c))), "ols": r_ols, "diff": pd.Series(d, index=c)}


# ==========================================================
# Systems of equations
# ==========================================================
@dataclass
class Equation:
    name: str
    y: str
    endog: list = field(default_factory=list)   # right-hand-side endogenous variables
    exog: list = field(default_factory=list)    # included exogenous variables
    const: bool = True


@dataclass
class SystemResult:
    method: str
    equations: list            # list[RegResult]
    sigma: pd.DataFrame
    cov: pd.DataFrame
    identification: pd.DataFrame
    nobs: int
    notes: list = field(default_factory=list)
    llf: float = np.nan
    iterations: int = 0


def _eq_mats(d: pd.DataFrame, eq: Equation):
    cols = (["const"] if eq.const else []) + eq.endog + eq.exog
    X = pd.DataFrame({c: (1.0 if c == "const" else pd.to_numeric(d[c], errors="coerce")) for c in cols}, index=d.index)
    return pd.to_numeric(d[eq.y], errors="coerce").to_numpy(float), X


def identification(df: pd.DataFrame, equations: list, exog_all: list) -> pd.DataFrame:
    """Order and rank conditions for each structural equation. The rank condition is checked on the
    estimated reduced form: rank of Π(excluded exogenous × included endogenous) must equal the number of
    included endogenous regressors (Cragg-Donald F indicates how far from rank deficiency)."""
    endog_all = sorted({e.y for e in equations} | {v for e in equations for v in e.endog})
    d = df[list(dict.fromkeys(endog_all + exog_all))].apply(pd.to_numeric, errors="coerce").dropna()
    Z = np.c_[np.ones(len(d)), d[exog_all].to_numpy(float)]
    rows = []
    for eq in equations:
        m = len(eq.endog)
        excluded = [v for v in exog_all if v not in eq.exog]
        k_ex = len(excluded)
        status = "exactly identified" if k_ex == m else "over-identified" if k_ex > m else "under-identified"
        rank_ok = None
        cdF = np.nan
        if m and k_ex >= m:
            Y = d[eq.endog].to_numpy(float)
            Pi = np.linalg.lstsq(Z, Y, rcond=None)[0]
            rows_ex = [1 + exog_all.index(v) for v in excluded]
            sub = Pi[rows_ex, :]
            sv = np.linalg.svd(sub / np.maximum(np.abs(sub).max(), 1e-300), compute_uv=False)
            rank_ok = bool(np.sum(sv > 1e-8) == m)
            inc_idx = [0] + [1 + exog_all.index(v) for v in eq.exog]
            Z1, Z2 = Z[:, inc_idx], Z[:, rows_ex]
            X2p = Y - _proj(Z1, Y)
            Z2p = Z2 - _proj(Z1, Z2)
            A = X2p.T @ _proj(Z2p, X2p)
            Mz = Y - _proj(Z, Y)
            cdF = float(np.min(np.real(linalg.eigvals(np.linalg.solve(Mz.T @ Mz, A))))) * (len(d) - Z.shape[1]) / k_ex
        elif m == 0:
            status, rank_ok = "no endogenous regressors (OLS is consistent)", True
        rows.append({"Equation": eq.name, "Dependent": eq.y, "Included endogenous (m)": m,
                     "Excluded exogenous (K − k)": k_ex, "Order condition": status,
                     "Rank condition": "—" if rank_ok is None else ("satisfied" if rank_ok else "FAILS"),
                     "Cragg-Donald F": cdF})
    return pd.DataFrame(rows)


def system_fit(df: pd.DataFrame, equations: list, exog_all: list, method: str = "3sls",
               small: bool = False, maxiter: int = 200, tol: float = 1e-10) -> SystemResult:
    """OLS / 2SLS / ILS / 3SLS / SUR / iterated SUR for a system. Σ̂ uses residuals divided by n
    (linearmodels default; R systemfit methodResidCov = "noDfCor")."""
    allv = list(dict.fromkeys([v for e in equations for v in [e.y] + e.endog + e.exog] + exog_all))
    d = df[allv].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)
    Z = np.c_[np.ones(n), d[exog_all].to_numpy(float)]
    G = len(equations)
    ys, Xs = [], []
    for eq in equations:
        y, X = _eq_mats(d, eq)
        ys.append(y)
        Xs.append(X)
    ident = identification(df, equations, exog_all)
    instr = method in ("2sls", "3sls", "ils")
    if instr:
        bad = ident[ident["Order condition"].str.startswith("under")]
        if not bad.empty:
            raise ValueError("Under-identified equation(s): " + ", ".join(bad["Equation"]))
    Xh = [(_proj(Z, X.to_numpy(float)) if instr and eq.endog else X.to_numpy(float)) for eq, X in zip(equations, Xs)]
    Xr = [X.to_numpy(float) for X in Xs]
    # first stage: equation by equation
    bs = [np.linalg.solve(xh.T @ xr, xh.T @ y) for xh, xr, y in zip(Xh, Xr, ys)]
    if method == "ils":
        for i, (eq, X) in enumerate(zip(equations, Xs)):
            if eq.endog and ident.iloc[i]["Order condition"] != "exactly identified":
                raise ValueError(f"ILS requires every equation with endogenous regressors to be exactly identified "
                                 f"('{eq.name}' is {ident.iloc[i]['Order condition']}). Use 2SLS or 3SLS.")
            if eq.endog:
                bs[i] = _ils(d, eq, exog_all)
    E = np.column_stack([y - xr @ b for y, xr, b in zip(ys, Xr, bs)])
    Sigma = E.T @ E / n
    it = 0
    if method in ("3sls", "sur", "itsur"):
        for it in range(1, (maxiter if method == "itsur" else 1) + 1):
            try:
                Si = np.linalg.inv(Sigma)
            except np.linalg.LinAlgError:
                raise ValueError("The residual covariance matrix Σ̂ is singular (equations are linearly dependent, e.g. the "
                                 "same dependent variable explained by the same regressors). Re-specify the system.") from None
            ks = [x.shape[1] for x in Xh]
            off = np.r_[0, np.cumsum(ks)]
            A = np.zeros((off[-1], off[-1]))
            c = np.zeros(off[-1])
            for i in range(G):
                for j in range(G):
                    A[off[i]:off[i + 1], off[j]:off[j + 1]] = Si[i, j] * Xh[i].T @ Xh[j]
                    c[off[i]:off[i + 1]] += Si[i, j] * Xh[i].T @ ys[j]
            bvec = np.linalg.solve(A, c)
            new = [bvec[off[i]:off[i + 1]] for i in range(G)]
            E = np.column_stack([y - xr @ b for y, xr, b in zip(ys, Xr, new)])
            Snew = E.T @ E / n
            delta = max(np.max(np.abs(b1 - b0)) for b1, b0 in zip(new, bs))
            bs = new
            if method != "itsur":
                break
            Sigma_old, Sigma = Sigma, Snew
            if delta < tol:
                break
        covall = np.linalg.inv(A)
        if method == "itsur":
            Sigma = Snew
        res_sigma = Snew
    else:
        res_sigma = Sigma
        covall = None
    # per-equation results
    results = []
    off = np.r_[0, np.cumsum([x.shape[1] for x in Xh])]
    for i, (eq, X, y, b) in enumerate(zip(equations, Xs, ys, bs)):
        names = list(X.columns)
        e = y - Xr[i] @ b
        k = len(b)
        if covall is not None:
            cov = covall[off[i]:off[i + 1], off[i]:off[i + 1]]
            if small:
                cov = cov * n / (n - k)
        else:
            s2 = e @ e / ((n - k) if small else n)
            cov = s2 * np.linalg.inv(Xh[i].T @ Xr[i])
        se = np.sqrt(np.diag(cov))
        tv = b / se
        if small:
            pv, q = 2 * stats.t.sf(np.abs(tv), n - k), stats.t.ppf(0.975, n - k)
        else:
            pv, q = 2 * stats.norm.sf(np.abs(tv)), stats.norm.ppf(0.975)
        r = rg.RegResult(name=eq.name, method=method, cov_type="nonrobust", design=None,
                         params=pd.Series(b, index=names), bse=pd.Series(se, index=names),
                         tvalues=pd.Series(tv, index=names), pvalues=pd.Series(pv, index=names),
                         ci=pd.DataFrame(np.c_[b - q * se, b + q * se], index=names, columns=["low", "high"]),
                         cov=pd.DataFrame(cov, index=names, columns=names), dist="t" if small else "z",
                         df_resid=n - k, nobs=n, y=y, fitted=Xr[i] @ b, resid=e, wresid=e, wexog=Xr[i])
        r.ssr = float(e @ e)
        r.sigma = float(np.sqrt(e @ e / (n - k)))
        r.rsquared = 1 - r.ssr / np.sum((y - y.mean()) ** 2)
        r.rsquared_adj = 1 - (1 - r.rsquared) * (n - 1) / (n - k)
        r.extra["y_label"] = eq.y
        results.append(r)
    names_all = [f"{eq.name}: {c}" for eq, X in zip(equations, Xs) for c in X.columns]
    if covall is not None:
        C = covall * (1.0 if not small else 1.0)
    else:
        C = linalg.block_diag(*[r.cov.to_numpy() for r in results])
    sres = SystemResult(method=method, equations=results,
                        sigma=pd.DataFrame(res_sigma, index=[e.name for e in equations], columns=[e.name for e in equations]),
                        cov=pd.DataFrame(C, index=names_all, columns=names_all), identification=ident, nobs=n,
                        iterations=it)
    if method in ("sur", "itsur"):
        sd = np.sqrt(np.diag(res_sigma))
        corr = res_sigma / np.outer(sd, sd)
        lm = n * np.sum(np.tril(corr, -1) ** 2)
        dfb = G * (G - 1) // 2
        sres.notes.append(f"Breusch-Pagan LM test of a diagonal Σ (no cross-equation correlation): "
                          f"χ²({dfb}) = {lm:.3f}, p = {stats.chi2.sf(lm, dfb):.4f}"
                          + (" → SUR gains efficiency over OLS." if stats.chi2.sf(lm, dfb) < 0.05 else
                             " → equation-by-equation OLS is adequate."))
        sres.extra_bp = (lm, dfb, float(stats.chi2.sf(lm, dfb)))
        sres.llf = float(-n / 2 * (G * np.log(2 * np.pi) + np.log(np.linalg.det(res_sigma)) + G))
    return sres


def _ils(d: pd.DataFrame, eq: Equation, exog_all: list) -> np.ndarray:
    """Indirect least squares for an exactly identified equation: solve the structural parameters from the
    OLS reduced-form coefficients (numerically identical to 2SLS)."""
    Z = np.c_[np.ones(len(d)), d[exog_all].to_numpy(float)]
    zn = ["const"] + exog_all
    Y = d[[eq.y] + eq.endog].to_numpy(float)
    Pi = np.linalg.lstsq(Z, Y, rcond=None)[0]
    excl = [zn.index(v) for v in exog_all if v not in eq.exog]
    inc = [zn.index(v) for v in (["const"] if eq.const else []) + eq.exog]
    beta = np.linalg.solve(Pi[excl, 1:], Pi[excl, 0])
    gamma = Pi[inc, 0] - Pi[inc, 1:] @ beta
    order = (["const"] if eq.const else []) + eq.endog + eq.exog
    vals = dict(zip([zn[j] for j in inc], gamma))
    vals.update(dict(zip(eq.endog, beta)))
    return np.array([vals[c] for c in order])


def reduced_form(df: pd.DataFrame, equations: list, exog_all: list) -> pd.DataFrame:
    endog_all = list(dict.fromkeys([e.y for e in equations]))
    d = df[endog_all + exog_all].apply(pd.to_numeric, errors="coerce").dropna()
    Z = np.c_[np.ones(len(d)), d[exog_all].to_numpy(float)]
    Pi = np.linalg.lstsq(Z, d[endog_all].to_numpy(float), rcond=None)[0]
    return pd.DataFrame(Pi, index=["const"] + exog_all, columns=endog_all)
