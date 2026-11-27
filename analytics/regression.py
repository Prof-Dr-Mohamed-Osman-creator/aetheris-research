# ==========================================
# Aetheris Research - Regression Engine (Catalogue §1.1)
# ------------------------------------------
# Classical & cross-sectional regression: design matrices with functional
# forms, dummies, polynomials and interactions; OLS / WLS / FGLS / GLS-AR(1)
# / robust M-estimation / restricted LS / nonlinear LS; classical, HC0–HC3,
# Newey-West, clustered and bootstrap standard errors; stepwise, best subsets,
# hierarchical regression, Box-Cox, ANCOVA, moderation and mediation.
#
# Pure functions (numpy / scipy / statsmodels, no Streamlit) so the same
# engine serves the UI, the report generator, the AI assistant and the API.
# ==========================================

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import optimize, stats

# ==========================================================
# Catalogues (English, Arabic, equation)
# ==========================================================
FORMS = {
    "linear": ("Linear", "خطي", "Y = β₀ + β₁X"),
    "loglog": ("Double-log (log-log)", "لوغاريتمي مزدوج", "ln Y = β₀ + β₁ ln X"),
    "loglin": ("Semi-log: log-lin (exponential / growth)", "نصف لوغاريتمي: لوغاريتم التابع (أسي / نمو)", "ln Y = β₀ + β₁X"),
    "linlog": ("Semi-log: lin-log", "نصف لوغاريتمي: لوغاريتم المستقل", "Y = β₀ + β₁ ln X"),
    "reciprocal": ("Reciprocal (inverse)", "مقلوب", "Y = β₀ + β₁ (1/X)"),
}

METHODS = {
    "ols": ("Ordinary least squares (OLS)", "المربعات الصغرى العادية (OLS)"),
    "wls": ("Weighted least squares (WLS)", "المربعات الصغرى الموزونة (WLS)"),
    "fgls": ("Feasible GLS — multiplicative heteroskedasticity (Harvey)", "المربعات الصغرى المعممة الممكنة (FGLS) — عدم تجانس ضربي (هارفي)"),
    "glsar": ("Feasible GLS — AR(1) errors (iterated Cochrane-Orcutt)", "المربعات الصغرى المعممة الممكنة — أخطاء AR(1) (كوكران-أوركت التكراري)"),
    "huber": ("Robust regression — Huber M-estimator", "الانحدار المتين — مقدّر هوبر"),
    "bisquare": ("Robust regression — Tukey bisquare M-estimator", "الانحدار المتين — مقدّر توكي ثنائي التربيع"),
    "rls": ("Restricted least squares (RLS)", "المربعات الصغرى المقيدة (RLS)"),
    "nls": ("Nonlinear least squares (NLS)", "المربعات الصغرى غير الخطية (NLS)"),
}

COV_TYPES = {
    "nonrobust": ("Classical (homoskedastic)", "كلاسيكية (بافتراض تجانس التباين)"),
    "HC0": ("White HC0", "وايت HC0"),
    "HC1": ("Robust HC1 (Stata 'robust')", "متينة HC1 (المعتمدة في Stata)"),
    "HC2": ("Robust HC2", "متينة HC2"),
    "HC3": ("Robust HC3 (recommended for small n)", "متينة HC3 (الأفضل للعينات الصغيرة)"),
    "HAC": ("Newey-West HAC", "نيوي-ويست HAC"),
    "cluster": ("Cluster-robust", "عنقودية متينة"),
    "bootstrap": ("Bootstrap (pairs)", "بوتستراب (أزواج المشاهدات)"),
}


# ==========================================================
# Model specification & design matrix
# ==========================================================
@dataclass
class ModelSpec:
    y: str
    x: list = field(default_factory=list)              # numeric regressors
    categorical: list = field(default_factory=list)    # dummy-coded (first level = reference)
    form: str = "linear"
    poly: dict = field(default_factory=dict)            # {var: degree ≥ 2}
    interactions: list = field(default_factory=list)    # [(a, b), ...] original variable names
    nolog: list = field(default_factory=list)           # numeric x kept in levels in log forms (trend, shares, dummies)
    center: bool = False                                # mean-center continuous terms used in poly / interactions
    const: bool = True
    reference: dict = field(default_factory=dict)       # {categorical: reference level}

    def all_vars(self) -> list:
        out = [self.y] + list(self.x) + list(self.categorical)
        for a, b in self.interactions:
            out += [a, b]
        return list(dict.fromkeys(out))


@dataclass
class Design:
    y: pd.Series
    X: pd.DataFrame
    y_label: str
    spec: ModelSpec
    index: pd.Index
    info: dict
    notes: list = field(default_factory=list)


def y_transform(form: str) -> str:
    return "log" if form in ("loglog", "loglin") else "level"


def x_transform(form: str) -> str:
    return {"loglog": "log", "linlog": "log", "reciprocal": "recip"}.get(form, "level")


def _tname(var: str, how: str) -> str:
    return {"log": f"ln({var})", "recip": f"1/{var}"}.get(how, var)


def _apply(s: pd.Series, how: str) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").astype(float)
    if how == "log":
        return np.log(s.where(s > 0))
    if how == "recip":
        return 1.0 / s.where(s != 0)
    return s


def build_design(df: pd.DataFrame, spec: ModelSpec, info: dict | None = None) -> Design:
    """Build (y, X) from the original data. If `info` from a fitted design is passed, the same
    transformations, reference levels and centring constants are reused (for predictions / what-if)."""
    fit_mode = info is None
    info = info or {}
    notes = []
    form = spec.form
    used = [v for v in spec.all_vars() if v in df.columns]
    missing = [v for v in spec.all_vars() if v not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    d = df[used].copy()

    ytx = y_transform(form)
    y = _apply(d[spec.y], ytx)
    y_label = _tname(spec.y, ytx)

    cols: dict[str, pd.Series] = {}
    groups: dict[str, list] = {}
    kinds: dict[str, str] = {}
    base: dict[str, str] = {}
    hows = info.get("hows", {})
    means = info.get("means", {})
    levels = info.get("levels", {})

    def num_term(var):
        how = hows.get(var) if not fit_mode else ("level" if var in spec.nolog else x_transform(form))
        hows[var] = how
        s = _apply(d[var], how)
        name = _tname(var, how)
        needs_center = spec.center and (var in spec.poly or any(var in pair for pair in spec.interactions))
        if needs_center:
            if fit_mode:
                means[var] = float(s.mean())
            s = s - means[var]
        return name, s

    def cat_terms(var):
        s = d[var].astype("string")
        if fit_mode:
            lv = sorted(s.dropna().unique().tolist(), key=str)
            ref = spec.reference.get(var)
            if ref in lv:
                lv.remove(ref)
                lv = [ref] + lv
            levels[var] = lv
        lv = levels[var]
        out = {}
        for level in lv[1:]:
            out[f"{var}[{level}]"] = (s == level).astype(float).where(s.notna())
        return out

    if spec.const:
        cols["const"] = pd.Series(1.0, index=d.index)
        kinds["const"] = "const"
    numeric_cache, cat_cache = {}, {}
    for v in spec.x:
        name, s = num_term(v)
        numeric_cache[v] = (name, s)
        cols[name] = s
        groups.setdefault(v, []).append(name)
        kinds[name], base[name] = hows[v], v
        deg = int(spec.poly.get(v, 1))
        for p in range(2, deg + 1):
            pname = f"{name}^{p}"
            cols[pname] = s ** p
            groups[v].append(pname)
            kinds[pname], base[pname] = "poly", v
    for v in spec.categorical:
        terms = cat_terms(v)
        cat_cache[v] = terms
        for name, s in terms.items():
            cols[name] = s
            kinds[name], base[name] = "dummy", v
        groups[v] = list(terms)
    for a, b in spec.interactions:
        def parts(v):
            if v in cat_cache:
                return cat_cache[v]
            if v in numeric_cache:
                n, s = numeric_cache[v]
                return {n: s}
            if v in spec.categorical:
                return cat_terms(v)
            n, s = num_term(v)
            numeric_cache[v] = (n, s)
            return {n: s}
        pa, pb = parts(a), parts(b)
        gname = f"{a}×{b}"
        groups[gname] = []
        for na, sa in pa.items():
            for nb, sb in pb.items():
                iname = f"{na}×{nb}"
                cols[iname] = sa * sb
                groups[gname].append(iname)
                kinds[iname], base[iname] = "inter", gname

    X = pd.DataFrame(cols, index=d.index)
    ok = y.notna() & X.notna().all(axis=1) & np.isfinite(y)
    ok &= np.isfinite(X).all(axis=1)
    n_drop = int((~ok).sum())
    if n_drop:
        nonpos = 0
        for v, how in [(spec.y, ytx)] + [(v, hows.get(v)) for v in spec.x]:
            if how == "log":
                nonpos += int((pd.to_numeric(d[v], errors="coerce") <= 0).sum())
        notes.append(f"{n_drop} observation(s) dropped (missing values"
                     + (f"; {nonpos} non-positive value(s) cannot be logged" if nonpos else "") + ").")
    y, X = y[ok], X[ok]
    # drop constant (all-zero) dummy / interaction columns and perfectly collinear columns
    if not fit_mode:  # prediction rows: keep every column, no sample checks
        return Design(y=y, X=X, y_label=y_label, spec=spec, index=y.index, info=info, notes=notes)
    keep = []
    for c in X.columns:
        if c != "const" and X[c].nunique() <= 1:
            notes.append(f"'{c}' dropped: no variation in the estimation sample.")
            continue
        keep.append(c)
    X = X[keep]
    if X.shape[1] and len(X):
        keep = []
        for c in X.columns:
            trial = X[keep + [c]].to_numpy()
            if np.linalg.matrix_rank(trial) == len(keep) + 1:
                keep.append(c)
            else:
                notes.append(f"'{c}' dropped: perfectly collinear with other regressors (dummy-variable trap?).")
        X = X[keep]
    for g in list(groups):
        groups[g] = [c for c in groups[g] if c in X.columns]
    new_info = {"hows": hows, "means": means, "levels": levels, "groups": groups, "kinds": kinds,
                "base": base, "ytx": ytx}
    return Design(y=y, X=X, y_label=y_label, spec=spec, index=y.index, info=new_info, notes=notes)


# ==========================================================
# Result container
# ==========================================================
@dataclass
class RegResult:
    name: str
    method: str
    cov_type: str
    design: Design | None
    params: pd.Series
    bse: pd.Series
    tvalues: pd.Series
    pvalues: pd.Series
    ci: pd.DataFrame
    cov: pd.DataFrame
    dist: str                    # "t" or "z"
    df_resid: float
    nobs: int
    y: np.ndarray
    fitted: np.ndarray
    resid: np.ndarray
    wresid: np.ndarray           # whitened residuals (= resid for OLS) — used by diagnostics
    wexog: np.ndarray            # whitened regressors
    rsquared: float = np.nan
    rsquared_adj: float = np.nan
    fvalue: float = np.nan
    f_pvalue: float = np.nan
    df_model: float = np.nan
    llf: float = np.nan
    aic: float = np.nan
    bic: float = np.nan
    hqic: float = np.nan
    ssr: float = np.nan
    sigma: float = np.nan
    sm: object | None = None
    notes: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    cov_opts: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.params)

    @property
    def names(self) -> list:
        return list(self.params.index)

    @property
    def y_label(self) -> str:
        return self.design.y_label if self.design else self.extra.get("y_label", "y")

    @property
    def form(self) -> str:
        return self.design.spec.form if self.design else "nls"

    def method_label(self, lang="en") -> str:
        return METHODS[self.method][1 if lang == "ar" else 0]

    def cov_label(self, lang="en") -> str:
        lab = COV_TYPES.get(self.cov_type, (self.cov_type, self.cov_type))[1 if lang == "ar" else 0]
        if self.cov_type == "HAC":
            lab += f" (lags = {self.cov_opts.get('maxlags')})"
        if self.cov_type == "bootstrap":
            lab += f" (B = {self.cov_opts.get('reps')})"
        return lab

    def coef_table(self, std_beta: bool = True) -> pd.DataFrame:
        crit = "t" if self.dist == "t" else "z"
        out = pd.DataFrame({"Variable": self.names, "Coef.": self.params.values, "Std. Err.": self.bse.values,
                            crit: self.tvalues.values, "p-value": self.pvalues.values,
                            "CI low": self.ci.iloc[:, 0].values, "CI high": self.ci.iloc[:, 1].values})
        out["Sig."] = [stars(p) for p in self.pvalues.values]
        if std_beta and self.design is not None:
            sy = np.std(self.y, ddof=1)
            sx = self.design.X.std(ddof=1)
            out["Std. beta"] = [np.nan if v == "const" else self.params[v] * sx[v] / sy if v in sx else np.nan
                                for v in self.names]
        return out

    def fit_stats(self) -> dict:
        return {"N": self.nobs, "R²": self.rsquared, "Adj. R²": self.rsquared_adj, "F": self.fvalue,
                "Prob(F)": self.f_pvalue, "Log-likelihood": self.llf, "AIC": self.aic, "BIC": self.bic,
                "HQIC": self.hqic, "Root MSE (σ̂)": self.sigma, "SSR": self.ssr, "df residual": self.df_resid}


def stars(p, levels=(0.01, 0.05, 0.10)) -> str:
    if p is None or pd.isna(p):
        return ""
    return "***" if p < levels[0] else "**" if p < levels[1] else "*" if p < levels[2] else ""


def _from_sm(res, design: Design, method: str, cov_type: str, name: str, cov_opts: dict,
             weights=None) -> RegResult:
    X = design.X
    names = list(X.columns)
    params = pd.Series(np.asarray(res.params), index=names)
    bse = pd.Series(np.asarray(res.bse), index=names)
    tv = pd.Series(np.asarray(res.tvalues), index=names)
    pv = pd.Series(np.asarray(res.pvalues), index=names)
    ci = pd.DataFrame(np.asarray(res.conf_int()), index=names, columns=["low", "high"])
    cov = pd.DataFrame(np.asarray(res.cov_params()), index=names, columns=names)
    y = design.y.to_numpy(float)
    fitted = X.to_numpy(float) @ params.to_numpy()
    resid = y - fitted
    n = len(y)
    k = len(names)
    use_t = bool(getattr(res, "use_t", True))
    wresid = np.asarray(getattr(res, "wresid", resid))
    wexog = np.asarray(res.model.wexog) if hasattr(res.model, "wexog") else X.to_numpy(float)
    ssr = float(np.sum(wresid ** 2))
    llf = float(getattr(res, "llf", np.nan))
    r = RegResult(name=name, method=method, cov_type=cov_type, design=design, params=params, bse=bse, tvalues=tv,
                  pvalues=pv, ci=ci, cov=cov, dist="t" if use_t else "z", df_resid=float(res.df_resid), nobs=n,
                  y=y, fitted=fitted, resid=resid, wresid=wresid, wexog=wexog, sm=res, cov_opts=cov_opts,
                  notes=list(design.notes))
    r.ssr = ssr
    r.sigma = float(np.sqrt(ssr / max(res.df_resid, 1)))
    r.rsquared = float(getattr(res, "rsquared", np.nan))
    r.rsquared_adj = float(getattr(res, "rsquared_adj", np.nan))
    r.df_model = float(getattr(res, "df_model", k - 1))
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.fvalue = float(np.squeeze(res.fvalue))
            r.f_pvalue = float(np.squeeze(res.f_pvalue))
    except Exception:  # noqa: BLE001 — no constant / robust estimator
        pass
    r.llf = llf
    r.aic = -2 * llf + 2 * k
    r.bic = -2 * llf + np.log(n) * k
    r.hqic = -2 * llf + 2 * k * np.log(np.log(n))
    return r


def newey_west_lags(n: int) -> int:
    """Newey-West (1994) plug-in rule used by EViews/Stata users: floor(4 (n/100)^(2/9))."""
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def _fit_cov(model, cov_type, cov_opts, n):
    if cov_type in ("nonrobust", "bootstrap"):
        return model.fit()
    if cov_type in ("HC0", "HC1", "HC2", "HC3"):
        return model.fit(cov_type=cov_type, use_t=True)
    if cov_type == "HAC":
        lags = cov_opts.get("maxlags")
        if lags is None:
            lags = cov_opts["maxlags"] = newey_west_lags(n)
        return model.fit(cov_type="HAC", cov_kwds={"maxlags": int(lags), "use_correction": True}, use_t=True)
    if cov_type == "cluster":
        g = pd.factorize(pd.Series(cov_opts["groups"]))[0]
        cov_opts["n_clusters"] = int(len(np.unique(g)))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return model.fit(cov_type="cluster", cov_kwds={"groups": g}, use_t=True)
    raise ValueError(cov_type)


def _bootstrap(r: RegResult, X: np.ndarray, y: np.ndarray, w: np.ndarray | None, reps: int, seed: int):
    rng = np.random.default_rng(seed)
    n, k = X.shape
    sw = np.sqrt(w) if w is not None else np.ones(n)
    draws = np.empty((reps, k))
    for b in range(reps):
        i = rng.integers(0, n, n)
        draws[b] = np.linalg.lstsq(X[i] * sw[i, None], y[i] * sw[i], rcond=None)[0]
    cov = np.cov(draws.T, ddof=1).reshape(k, k)
    se = np.sqrt(np.diag(cov))
    z = r.params.to_numpy() / se
    r.bse = pd.Series(se, index=r.names)
    r.tvalues = pd.Series(z, index=r.names)
    r.pvalues = pd.Series(2 * stats.norm.sf(np.abs(z)), index=r.names)
    r.ci = pd.DataFrame(np.percentile(draws, [2.5, 97.5], axis=0).T, index=r.names, columns=["low", "high"])
    r.cov = pd.DataFrame(cov, index=r.names, columns=r.names)
    r.dist = "z"
    slopes = [j for j, c in enumerate(r.names) if c != "const"]
    if slopes:
        b = r.params.to_numpy()[slopes]
        V = cov[np.ix_(slopes, slopes)]
        try:
            W = float(b @ np.linalg.solve(V, b))
            r.fvalue = W / len(slopes)
            r.f_pvalue = float(stats.chi2.sf(W, len(slopes)))
        except np.linalg.LinAlgError:
            pass
    r.notes.append(f"Bootstrap standard errors from {reps} pairs resamples (percentile 95% CIs, normal-based p-values).")
    r.extra["boot_draws"] = draws


def fit(df: pd.DataFrame, spec: ModelSpec, method: str = "ols", cov_type: str = "nonrobust",
        cov_opts: dict | None = None, weights: str | np.ndarray | None = None, name: str = "",
        robust_norm: str | None = None) -> RegResult:
    """Estimate a linear-in-parameters model. `weights` = column name or array (WLS)."""
    cov_opts = dict(cov_opts or {})
    design = build_design(df, spec)
    X, y = design.X, design.y
    n, k = X.shape
    if n <= k:
        raise ValueError(f"Not enough observations: n = {n} ≤ k = {k} parameters.")
    if cov_type == "cluster":
        g = cov_opts.get("groups")
        if isinstance(g, str):
            cov_opts["groups_name"] = g
            cov_opts["groups"] = df.loc[design.index, g].astype(str).to_numpy()
    w = None
    name = name or METHODS[method][0].split(" (")[0]
    if method == "ols":
        res = _fit_cov(sm.OLS(y, X), cov_type, cov_opts, n)
    elif method == "wls":
        if isinstance(weights, str):
            w = pd.to_numeric(df.loc[design.index, weights], errors="coerce").to_numpy(float)
        else:
            w = np.asarray(weights, float)[: n] if weights is not None else np.ones(n)
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("WLS weights must be positive and non-missing.")
        res = _fit_cov(sm.WLS(y, X, weights=w), cov_type, cov_opts, n)
    elif method == "fgls":
        # Harvey (1976): ln e² = Zγ → σ̂ᵢ² = exp(Zγ̂); weights 1/σ̂ᵢ²
        e = sm.OLS(y, X).fit().resid.to_numpy()
        aux = sm.OLS(np.log(e ** 2 + 1e-12), X).fit()
        w = 1.0 / np.exp(aux.fittedvalues.to_numpy())
        res = _fit_cov(sm.WLS(y, X, weights=w), cov_type, cov_opts, n)
    elif method == "glsar":
        model = sm.GLSAR(y, X, rho=1)
        res0 = model.iterative_fit(maxiter=50)
        if cov_type == "cluster":  # the AR(1) transformation drops the first observation
            cov_opts = {**cov_opts, "groups": np.asarray(cov_opts["groups"])[1:]}
        res = res0 if cov_type == "nonrobust" else _fit_cov(model, cov_type, cov_opts, n)
        rho = float(np.squeeze(model.rho))
    elif method in ("huber", "bisquare"):
        norm = sm.robust.norms.HuberT() if method == "huber" else sm.robust.norms.TukeyBiweight()
        res = sm.RLM(y, X, M=norm).fit()
        cov_type = "nonrobust"
    else:
        raise ValueError(method)

    if method in ("huber", "bisquare"):
        r = _from_rlm(res, design, method, name)
    else:
        r = _from_sm(res, design, method, cov_type, name, cov_opts, weights=w)
    if method == "glsar":
        r.extra["rho"] = rho
        r.notes.append(f"Estimated AR(1) coefficient ρ̂ = {rho:.4f}; the first observation is lost in the transformation.")
        sst = np.sum((r.y - r.y.mean()) ** 2)
        r.extra["raw_r2"] = 1 - np.sum(r.resid ** 2) / sst
    if w is not None:
        r.extra["weights"] = w
    if method == "fgls":
        r.notes.append("FGLS weights estimated from ln(ê²) regressed on all regressors (Harvey 1976).")
    if cov_type == "cluster":
        G = cov_opts.get("n_clusters", 0)
        if G < 30:
            r.notes.append(f"Only {G} clusters: cluster-robust inference is unreliable with few clusters "
                           "(Cameron, Gelbach & Miller 2008); consider a wild cluster bootstrap.")
        if G <= r.k - 1:
            r.fvalue = r.f_pvalue = np.nan
            r.notes.append("The overall F-test is not computable: clusters ≤ number of slope coefficients.")
    if cov_type == "bootstrap":
        _bootstrap(r, X.to_numpy(float), y.to_numpy(float), w, int(cov_opts.get("reps", 999)),
                   int(cov_opts.get("seed", 2026)))
    return r


def _from_rlm(res, design: Design, method: str, name: str) -> RegResult:
    X = design.X
    names = list(X.columns)
    y = design.y.to_numpy(float)
    params = pd.Series(np.asarray(res.params), index=names)
    fitted = X.to_numpy(float) @ params.to_numpy()
    resid = y - fitted
    n, k = X.shape
    r = RegResult(name=name, method=method, cov_type="nonrobust", design=design, params=params,
                  bse=pd.Series(np.asarray(res.bse), index=names), tvalues=pd.Series(np.asarray(res.tvalues), index=names),
                  pvalues=pd.Series(np.asarray(res.pvalues), index=names),
                  ci=pd.DataFrame(np.asarray(res.conf_int()), index=names, columns=["low", "high"]),
                  cov=pd.DataFrame(np.asarray(res.cov_params()), index=names, columns=names), dist="z",
                  df_resid=float(n - k), nobs=n, y=y, fitted=fitted, resid=resid, wresid=resid,
                  wexog=X.to_numpy(float), sm=res, notes=list(design.notes))
    sst = np.sum((y - y.mean()) ** 2)
    r.ssr = float(np.sum(resid ** 2))
    r.rsquared = 1 - r.ssr / sst
    r.rsquared_adj = 1 - (1 - r.rsquared) * (n - 1) / (n - k)
    r.sigma = float(res.scale)
    r.extra["weights_final"] = np.asarray(res.weights)
    r.notes.append(f"Robust M-estimation (IRLS); R² is a descriptive pseudo-R². "
                   f"{int(np.sum(np.asarray(res.weights) < 0.5))} observation(s) down-weighted below 0.5.")
    return r


# ==========================================================
# Restricted least squares  (R β = q)
# ==========================================================
def parse_restrictions(text: str, names: list) -> tuple[np.ndarray, np.ndarray, list]:
    """Parse linear restrictions such as  'ln(land) + ln(labour) = 1; x3 = 0; 2*x1 - x2 = 0'.
    Coefficient names are matched literally (longest first), so names with brackets / spaces work."""
    rows, rhs, labels = [], [], []
    order = sorted(range(len(names)), key=lambda i: -len(names[i]))
    for raw in re.split(r"[;\n]", text):
        eq = raw.strip()
        if not eq:
            continue
        if eq.count("=") != 1:
            raise ValueError(f"Each restriction needs exactly one '=': {eq}")
        expr = eq
        for i in order:
            expr = expr.replace(names[i], f" @{i}@ ")
        lhs, rhs_s = expr.split("=")
        vec = np.zeros(len(names))
        const = 0.0
        for side, sign in ((lhs, 1.0), (rhs_s, -1.0)):
            s = side.replace(" ", "")
            if not s:
                continue
            if s[0] not in "+-":
                s = "+" + s
            for sg, body in re.findall(r"([+-])([^+-]+)", s):
                f = sign * (1.0 if sg == "+" else -1.0)
                m = re.fullmatch(r"(?:(\d*\.?\d+(?:[eE][+-]?\d+)?)\*?)?@(\d+)@(?:\*(\d*\.?\d+))?", body)
                if m:
                    c = float(m.group(1)) if m.group(1) else 1.0
                    c *= float(m.group(3)) if m.group(3) else 1.0
                    vec[int(m.group(2))] += f * c
                elif re.fullmatch(r"\d*\.?\d+(?:[eE][+-]?\d+)?", body):
                    const -= f * float(body)
                else:
                    raise ValueError(f"Could not read '{body}' in: {eq}. Use coefficient names exactly as shown.")
        if not vec.any():
            raise ValueError(f"No coefficient found in: {eq}")
        rows.append(vec)
        rhs.append(const)
        labels.append(eq)
    if not rows:
        raise ValueError("No restriction entered.")
    return np.vstack(rows), np.asarray(rhs), labels


def restricted_ls(df: pd.DataFrame, spec: ModelSpec, R: np.ndarray, q: np.ndarray, labels=None,
                  name: str = "RLS") -> RegResult:
    """Restricted LS (Theil 1971; Greene §5.3) with the F test of the restrictions."""
    design = build_design(df, spec)
    X, y = design.X.to_numpy(float), design.y.to_numpy(float)
    names = list(design.X.columns)
    n, k = X.shape
    J = R.shape[0]
    XtXi = np.linalg.inv(X.T @ X)
    b = XtXi @ X.T @ y
    A = R @ XtXi @ R.T
    Ai = np.linalg.inv(A)
    br = b - XtXi @ R.T @ Ai @ (R @ b - q)
    er = y - X @ br
    eu = y - X @ b
    s2r = er @ er / (n - k + J)
    cov = s2r * (XtXi - XtXi @ R.T @ Ai @ R @ XtXi)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        tv = np.where(se > 1e-12, br / se, np.nan)
    dfr = n - k + J
    pv = 2 * stats.t.sf(np.abs(tv), dfr)
    tc = stats.t.ppf(0.975, dfr)
    F = ((er @ er - eu @ eu) / J) / (eu @ eu / (n - k))
    pF = stats.f.sf(F, J, n - k)
    sst = np.sum((y - y.mean()) ** 2)
    llf = -n / 2 * (np.log(2 * np.pi) + np.log(er @ er / n) + 1)
    r = RegResult(name=name, method="rls", cov_type="nonrobust", design=design, params=pd.Series(br, index=names),
                  bse=pd.Series(se, index=names), tvalues=pd.Series(tv, index=names), pvalues=pd.Series(pv, index=names),
                  ci=pd.DataFrame(np.c_[br - tc * se, br + tc * se], index=names, columns=["low", "high"]),
                  cov=pd.DataFrame(cov, index=names, columns=names), dist="t", df_resid=dfr, nobs=n, y=y,
                  fitted=X @ br, resid=er, wresid=er, wexog=X, notes=list(design.notes))
    r.ssr = float(er @ er)
    r.sigma = float(np.sqrt(s2r))
    r.rsquared = 1 - r.ssr / sst
    r.rsquared_adj = 1 - (1 - r.rsquared) * (n - 1) / dfr
    r.llf = float(llf)
    kk = k - J
    r.aic, r.bic = -2 * llf + 2 * kk, -2 * llf + np.log(n) * kk
    r.hqic = -2 * llf + 2 * kk * np.log(np.log(n))
    r.extra.update({"restrictions": labels or [], "F_restr": float(F), "p_restr": float(pF), "J": J,
                    "unrestricted_params": pd.Series(b, index=names)})
    r.notes.append(f"Restrictions imposed: {'; '.join(labels or [])}. "
                   f"F-test of the restrictions: F({J}, {n - k}) = {F:.3f}, p = {pF:.4f}"
                   + (" → the data reject the restrictions." if pF < 0.05 else " → the restrictions are not rejected."))
    return r


# ==========================================================
# Nonlinear least squares
# ==========================================================
NLS_FORMS = {
    "exponential": ("Exponential  Y = a·e^(bX)", "أسي  Y = a·e^(bX)", ["a", "b"]),
    "power": ("Power  Y = a·X^b", "قوة  Y = a·X^b", ["a", "b"]),
    "logistic": ("Logistic growth  Y = K / (1 + e^(−r(X − x₀)))", "النمو اللوجستي  Y = K / (1 + e^(−r(X − x₀)))", ["K", "r", "x0"]),
    "gompertz": ("Gompertz  Y = K·e^(−b·e^(−cX))", "جومبرتز  Y = K·e^(−b·e^(−cX))", ["K", "b", "c"]),
    "mitscherlich": ("Mitscherlich  Y = A(1 − e^(−c(X + b)))", "ميتشرليخ  Y = A(1 − e^(−c(X + b)))", ["A", "c", "b"]),
    "michaelis": ("Michaelis-Menten / hyperbolic  Y = aX / (b + X)", "ميكايليس-منتن (زائدي)  Y = aX / (b + X)", ["a", "b"]),
}


def _nls_fun(kind):
    return {
        "exponential": lambda x, a, b: a * np.exp(b * x),
        "power": lambda x, a, b: a * np.power(x, b),
        "logistic": lambda x, K, r, x0: K / (1 + np.exp(-r * (x - x0))),
        "gompertz": lambda x, K, b, c: K * np.exp(-b * np.exp(-c * x)),
        "mitscherlich": lambda x, A, c, b: A * (1 - np.exp(-c * (x + b))),
        "michaelis": lambda x, a, b: a * x / (b + x),
    }[kind]


def _nls_start(kind, x, y):
    ypos = y[y > 0]
    if kind == "exponential" and ypos.size == y.size:
        b, la = np.polyfit(x, np.log(y), 1)
        return [np.exp(la), b]
    if kind == "power" and (x > 0).all() and (y > 0).all():
        b, la = np.polyfit(np.log(x), np.log(y), 1)
        return [np.exp(la), b]
    K = y.max() * 1.1 if y.max() > 0 else 1.0
    span = (x.max() - x.min()) or 1.0
    if kind == "logistic":
        return [K, 4 / span, float(np.median(x))]
    if kind == "gompertz":
        return [K, 1.0, 2 / span]
    if kind == "mitscherlich":
        return [K, 2 / span, 0.0]
    if kind == "michaelis":
        return [K, float(np.median(x)) or 1.0]
    return [1.0, 0.0]


def fit_nls(df: pd.DataFrame, y: str, x: str, kind: str, p0=None, name: str = "NLS") -> RegResult:
    d = df[[y, x]].apply(pd.to_numeric, errors="coerce").dropna()
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    if kind == "power" and (xv <= 0).any():
        raise ValueError("The power form needs X > 0.")
    f = _nls_fun(kind)
    pnames = NLS_FORMS[kind][2]
    p0 = p0 or _nls_start(kind, xv, yv)
    popt, pcov = optimize.curve_fit(f, xv, yv, p0=p0, maxfev=20000)
    fitted = f(xv, *popt)
    resid = yv - fitted
    n, k = len(yv), len(popt)
    dfr = n - k
    se = np.sqrt(np.clip(np.diag(pcov), 0, None))
    tv = popt / se
    pv = 2 * stats.t.sf(np.abs(tv), dfr)
    tc = stats.t.ppf(0.975, dfr)
    ssr = float(resid @ resid)
    sst = float(np.sum((yv - yv.mean()) ** 2))
    llf = -n / 2 * (np.log(2 * np.pi) + np.log(ssr / n) + 1)
    # numerical Jacobian for diagnostics (Gauss-Newton regressors)
    J = np.empty((n, k))
    for j in range(k):
        h = 1e-6 * max(1.0, abs(popt[j]))
        pp = popt.copy()
        pp[j] += h
        J[:, j] = (f(xv, *pp) - fitted) / h
    r = RegResult(name=name, method="nls", cov_type="nonrobust", design=None, params=pd.Series(popt, index=pnames),
                  bse=pd.Series(se, index=pnames), tvalues=pd.Series(tv, index=pnames), pvalues=pd.Series(pv, index=pnames),
                  ci=pd.DataFrame(np.c_[popt - tc * se, popt + tc * se], index=pnames, columns=["low", "high"]),
                  cov=pd.DataFrame(pcov, index=pnames, columns=pnames), dist="t", df_resid=dfr, nobs=n, y=yv,
                  fitted=fitted, resid=resid, wresid=resid, wexog=J)
    r.ssr, r.sigma = ssr, float(np.sqrt(ssr / dfr))
    r.rsquared = 1 - ssr / sst
    r.rsquared_adj = 1 - (1 - r.rsquared) * (n - 1) / dfr
    r.llf = float(llf)
    r.aic, r.bic = -2 * llf + 2 * k, -2 * llf + np.log(n) * k
    r.hqic = -2 * llf + 2 * k * np.log(np.log(n))
    r.extra.update({"x": xv, "kind": kind, "x_name": x, "y_label": y, "fun": f})
    return r


# ==========================================================
# Prediction (with full transformation pipeline)
# ==========================================================
def design_rows(r: RegResult, values: pd.DataFrame) -> pd.DataFrame:
    """Design matrix for new data in the units of the original variables."""
    spec = r.design.spec
    v = values.copy()
    if spec.y not in v.columns:
        v[spec.y] = 1.0
    d = build_design(v, spec, info=r.design.info)
    X = d.X.reindex(columns=r.names, fill_value=0.0)
    return X


def smearing_factor(r: RegResult) -> float:
    """Duan (1983) smearing estimator for retransforming ln(Y) predictions."""
    return float(np.mean(np.exp(r.resid)))


def predict(r: RegResult, values: pd.DataFrame, level: float = 0.95) -> pd.DataFrame:
    """Point prediction, confidence and prediction intervals, returned in the units of Y."""
    X = design_rows(r, values).to_numpy(float)
    b = r.params.to_numpy()
    V = r.cov.to_numpy()
    yhat = X @ b
    se_mean = np.sqrt(np.einsum("ij,jk,ik->i", X, V, X))
    se_obs = np.sqrt(se_mean ** 2 + r.sigma ** 2)
    q = stats.t.ppf(0.5 + level / 2, r.df_resid) if r.dist == "t" else stats.norm.ppf(0.5 + level / 2)
    out = pd.DataFrame({"fit": yhat, "ci_low": yhat - q * se_mean, "ci_high": yhat + q * se_mean,
                        "pi_low": yhat - q * se_obs, "pi_high": yhat + q * se_obs, "se_fit": se_mean})
    if r.design.info["ytx"] == "log":
        sf = smearing_factor(r)
        out = pd.DataFrame({"fit": np.exp(out["fit"]) * sf, "ci_low": np.exp(out["ci_low"]) * sf,
                            "ci_high": np.exp(out["ci_high"]) * sf, "pi_low": np.exp(out["pi_low"]),
                            "pi_high": np.exp(out["pi_high"]), "se_fit": out["se_fit"]})
    return out


def baseline_values(df: pd.DataFrame, r: RegResult) -> pd.DataFrame:
    """Representative observation: means of numeric regressors (geometric mean in log forms), reference category."""
    spec = r.design.spec
    idx = r.design.index
    row = {}
    for v in spec.x + [a for pair in spec.interactions for a in pair if a not in spec.categorical]:
        s = pd.to_numeric(df.loc[idx, v], errors="coerce")
        row[v] = float(np.exp(np.log(s[s > 0]).mean())) if r.design.info["hows"].get(v) == "log" else float(s.mean())
    for c in spec.categorical:
        row[c] = r.design.info["levels"][c][0]
    return pd.DataFrame([row])


# ==========================================================
# Linear hypothesis tests on a fitted model
# ==========================================================
def wald_test(r: RegResult, R: np.ndarray, q: np.ndarray) -> dict:
    """Wald test of Rβ = q using the model's own covariance (robust if the model is robust)."""
    b = r.params.to_numpy()
    V = r.cov.to_numpy()
    diff = R @ b - q
    W = float(diff @ np.linalg.solve(R @ V @ R.T, diff))
    J = R.shape[0]
    F = W / J
    pF = float(stats.f.sf(F, J, r.df_resid)) if r.dist == "t" else np.nan
    return {"Wald χ²": W, "df": J, "p (χ²)": float(stats.chi2.sf(W, J)), "F": F, "p (F)": pF,
            "estimate": diff + q}


def nested_tests(df: pd.DataFrame, r: RegResult, drop: list) -> pd.DataFrame:
    """Exclusion restrictions: F, Wald, likelihood-ratio and Lagrange-multiplier tests (Engle 1984)."""
    X = r.design.X
    y = r.design.y.to_numpy(float)
    keep = [c for c in X.columns if c not in drop]
    Xu, Xr = X.to_numpy(float), X[keep].to_numpy(float)
    n, k = Xu.shape
    J = len(drop)
    eu = y - Xu @ np.linalg.lstsq(Xu, y, rcond=None)[0]
    er = y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]
    ssu, ssr_ = eu @ eu, er @ er
    F = ((ssr_ - ssu) / J) / (ssu / (n - k))
    LR = n * np.log(ssr_ / ssu)
    W = n * (ssr_ - ssu) / ssu
    g = er - er.mean() if "const" not in X.columns else er
    aux_fit = Xu @ np.linalg.lstsq(Xu, er, rcond=None)[0]
    LM = n * (1 - np.sum((er - aux_fit) ** 2) / np.sum(g ** 2))
    R_idx = [list(X.columns).index(c) for c in drop]
    R = np.zeros((J, k))
    R[np.arange(J), R_idx] = 1
    rw = wald_test(r, R, np.zeros(J))
    rows = [("F-test (classical)", F, f"({J}, {n - k})", stats.f.sf(F, J, n - k)),
            ("Wald χ² (classical)", W, str(J), stats.chi2.sf(W, J)),
            ("Likelihood ratio χ²", LR, str(J), stats.chi2.sf(LR, J)),
            ("Lagrange multiplier χ² (nR²)", LM, str(J), stats.chi2.sf(LM, J))]
    if r.cov_type != "nonrobust":
        rows.insert(1, (f"Wald χ² ({r.cov_type} covariance)", rw["Wald χ²"], str(J), rw["p (χ²)"]))
    return pd.DataFrame(rows, columns=["Test", "Statistic", "df", "p-value"])


def term_tests(r: RegResult) -> pd.DataFrame:
    """ANCOVA-style table: joint Wald/F test for each term (all dummies of a factor together), partial η²."""
    groups = r.design.info["groups"]
    rows = []
    X = r.design.X
    y = r.design.y.to_numpy(float)
    ssu = float(np.sum((y - X.to_numpy(float) @ np.linalg.lstsq(X.to_numpy(float), y, rcond=None)[0]) ** 2))
    for g, cols in groups.items():
        cols = [c for c in cols if c in r.names]
        if not cols:
            continue
        J = len(cols)
        R = np.zeros((J, r.k))
        for i, c in enumerate(cols):
            R[i, r.names.index(c)] = 1
        w = wald_test(r, R, np.zeros(J))
        keep = [c for c in X.columns if c not in cols]
        Xr = X[keep].to_numpy(float)
        ssr_ = float(np.sum((y - Xr @ np.linalg.lstsq(Xr, y, rcond=None)[0]) ** 2))
        ss_term = ssr_ - ssu
        rows.append({"Term": g, "df": J, "Partial SS": ss_term, "F": w["F"],
                     "p-value": w["p (F)"] if r.dist == "t" else w["p (χ²)"],
                     "Partial η²": ss_term / (ss_term + ssu)})
    rows.append({"Term": "Residual", "df": r.df_resid, "Partial SS": ssu, "F": np.nan, "p-value": np.nan,
                 "Partial η²": np.nan})
    return pd.DataFrame(rows)


def adjusted_means(df: pd.DataFrame, r: RegResult, factor: str) -> pd.DataFrame:
    """ANCOVA adjusted (least-squares) means of Y for each level, covariates at their means."""
    base = baseline_values(df, r)
    rows = []
    for lv in r.design.info["levels"][factor]:
        v = base.copy()
        v[factor] = lv
        p = predict(r, v)
        raw = pd.to_numeric(df.loc[r.design.index][df.loc[r.design.index, factor].astype(str) == str(lv)][r.design.spec.y],
                            errors="coerce")
        rows.append({"Level": lv, "Raw mean": raw.mean(), "Adjusted mean": p["fit"].iloc[0],
                     "CI low": p["ci_low"].iloc[0], "CI high": p["ci_high"].iloc[0], "n": int(raw.notna().sum())})
    return pd.DataFrame(rows)


# ==========================================================
# Model selection: stepwise, best subsets, hierarchical
# ==========================================================
def _ols_np(X: np.ndarray, y: np.ndarray):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    n, k = X.shape
    ssr = float(e @ e)
    llf = -n / 2 * (np.log(2 * np.pi) + np.log(ssr / n) + 1)
    return b, ssr, llf


def _crit(ssr, llf, n, k, sst, crit, s2_full=None):
    if crit == "aic":
        return -2 * llf + 2 * k
    if crit == "bic":
        return -2 * llf + np.log(n) * k
    if crit == "adjr2":
        return -(1 - (ssr / (n - k)) / (sst / (n - 1)))
    if crit == "cp":
        return ssr / s2_full - n + 2 * k
    raise ValueError(crit)


def stepwise(design: Design, direction: str = "both", criterion: str = "p", p_enter: float = 0.05,
             p_remove: float = 0.10, force: list | None = None, max_steps: int = 200) -> dict:
    """Forward / backward / bidirectional stepwise selection by p-values (SPSS style) or AIC / BIC (R step())."""
    X, y = design.X, design.y.to_numpy(float)
    force = [c for c in (force or []) if c in X.columns]
    base = ["const"] if "const" in X.columns else []
    base += [c for c in force if c not in base]
    cand = [c for c in X.columns if c not in base]
    n = len(y)
    sst = float(np.sum((y - y.mean()) ** 2))
    current = list(base) if direction != "backward" else base + cand
    log = []

    def stats_for(cols):
        Xm = X[cols].to_numpy(float)
        b, ssr, llf = _ols_np(Xm, y)
        k = len(cols)
        return Xm, b, ssr, llf, k

    def pvals(cols):
        res = sm.OLS(y, X[cols]).fit()
        return res.pvalues

    def score(cols):
        _, _, ssr, llf, k = stats_for(cols)
        return _crit(ssr, llf, n, k, sst, criterion if criterion != "p" else "aic")

    for _ in range(max_steps):
        changed = False
        if direction in ("forward", "both"):
            outside = [c for c in cand if c not in current]
            if outside:
                if criterion == "p":
                    best = min(((pvals(current + [c])[c], c) for c in outside), default=None)
                    if best and best[0] < p_enter:
                        current.append(best[1])
                        log.append(("enter", best[1], best[0]))
                        changed = True
                else:
                    now = score(current)
                    best = min(((score(current + [c]), c) for c in outside))
                    if best[0] < now - 1e-9:
                        current.append(best[1])
                        log.append(("enter", best[1], best[0]))
                        changed = True
        if direction in ("backward", "both"):
            removable = [c for c in current if c not in base]
            if removable:
                if criterion == "p":
                    pv = pvals(current)
                    worst = max(((pv[c], c) for c in removable))
                    if worst[0] > p_remove:
                        current.remove(worst[1])
                        log.append(("remove", worst[1], worst[0]))
                        changed = True
                else:
                    now = score(current)
                    best = min(((score([c for c in current if c != r_]), r_) for r_ in removable))
                    if best[0] < now - 1e-9:
                        current.remove(best[1])
                        log.append(("remove", best[1], best[0]))
                        changed = True
        if not changed:
            break
    steps = pd.DataFrame(log, columns=["Action", "Variable", "p-value" if criterion == "p" else criterion.upper()])
    return {"selected": current, "steps": steps}


def best_subsets(design: Design, max_vars: int | None = None, force: list | None = None,
                 limit: int = 14) -> pd.DataFrame:
    """All-subsets regression; best model of each size by R², with adj. R², Mallows Cp, AIC, BIC."""
    X, y = design.X, design.y.to_numpy(float)
    base = ["const"] if "const" in X.columns else []
    base += [c for c in (force or []) if c in X.columns and c not in base]
    cand = [c for c in X.columns if c not in base]
    if len(cand) > limit:
        raise ValueError(f"Best subsets is limited to {limit} candidate regressors ({len(cand)} given). "
                         "Use stepwise selection instead.")
    n = len(y)
    sst = float(np.sum((y - y.mean()) ** 2))
    _, ssr_full, _ = _ols_np(X.to_numpy(float), y)
    s2_full = ssr_full / (n - X.shape[1])
    rows = []
    max_vars = min(max_vars or len(cand), len(cand))
    for m in range(1, max_vars + 1):
        best = None
        for combo in combinations(cand, m):
            cols = base + list(combo)
            if len(cols) >= n:
                continue
            _, ssr, llf = _ols_np(X[cols].to_numpy(float), y)
            if best is None or ssr < best[0]:
                best = (ssr, llf, cols, combo)
        if best is None:
            continue
        ssr, llf, cols, combo = best
        k = len(cols)
        rows.append({"Size": m, "Variables": ", ".join(combo), "R²": 1 - ssr / sst,
                     "Adj. R²": 1 - (ssr / (n - k)) / (sst / (n - 1)), "Mallows Cp": ssr / s2_full - n + 2 * k,
                     "AIC": -2 * llf + 2 * k, "BIC": -2 * llf + np.log(n) * k, "_cols": cols})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Best by"] = ""
        for crit, fn in (("Adj. R²", "idxmax"), ("AIC", "idxmin"), ("BIC", "idxmin")):
            i = getattr(out[crit], fn)()
            out.loc[i, "Best by"] = (out.loc[i, "Best by"] + " " + crit).strip()
        cp_ok = out[out["Mallows Cp"] <= out["Size"] + len(base) + 1e-9]
        if not cp_ok.empty:
            i = cp_ok["Size"].idxmin()
            out.loc[i, "Best by"] = (out.loc[i, "Best by"] + " Cp").strip()
    return out


def hierarchical(design: Design, blocks: list[list[str]]) -> pd.DataFrame:
    """Hierarchical (blockwise) regression: R², ΔR², F-change and its p-value for each block."""
    X, y = design.X, design.y.to_numpy(float)
    n = len(y)
    sst = float(np.sum((y - y.mean()) ** 2))
    cols = ["const"] if "const" in X.columns else []
    prev_r2, prev_k = 0.0, len(cols)
    rows = []
    for i, blk in enumerate(blocks, 1):
        cols = cols + [c for c in blk if c in X.columns and c not in cols]
        _, ssr, llf = _ols_np(X[cols].to_numpy(float), y)
        k = len(cols)
        r2 = 1 - ssr / sst
        dk = k - prev_k
        dfr = n - k
        Fc = ((r2 - prev_r2) / dk) / ((1 - r2) / dfr) if dk > 0 else np.nan
        rows.append({"Block": i, "Added": ", ".join(blk), "k": k - (1 if "const" in cols else 0), "R²": r2,
                     "Adj. R²": 1 - (1 - r2) * (n - 1) / dfr, "ΔR²": r2 - prev_r2, "F change": Fc,
                     "df1": dk, "df2": dfr, "p (F change)": stats.f.sf(Fc, dk, dfr) if dk > 0 else np.nan,
                     "AIC": -2 * llf + 2 * k})
        prev_r2, prev_k = r2, k
    return pd.DataFrame(rows)


# ==========================================================
# Functional forms & Box-Cox
# ==========================================================
def compare_forms(df: pd.DataFrame, spec: ModelSpec, forms=None) -> pd.DataFrame:
    """Fit every functional form on a common sample (Y > 0, X > 0). The Jacobian-adjusted log-likelihood
    (Zarembka 1968) makes AIC comparable between models with Y and ln Y as dependent variable."""
    forms = forms or list(FORMS)
    d = df.copy()
    pos = pd.Series(True, index=d.index)
    for v in [spec.y] + [x for x in spec.x if x not in spec.nolog]:
        pos &= pd.to_numeric(d[v], errors="coerce") > 0
    d = d[pos]
    rows = []
    for f in forms:
        sp = ModelSpec(**{**spec.__dict__, "form": f})
        try:
            r = fit(d, sp)
        except Exception as e:  # noqa: BLE001
            rows.append({"Form": f, "Equation": FORMS[f][2], "error": str(e)})
            continue
        # r.y is ln(Y) in log models: the Jacobian of the transformation is −Σ ln Y
        adj = r.llf - (np.sum(r.y) if r.design.info["ytx"] == "log" else 0.0)
        aic_adj = -2 * adj + 2 * r.k
        rows.append({"Form": f, "Equation": FORMS[f][2], "N": r.nobs, "R² (own Y)": r.rsquared,
                     "Adj. R²": r.rsquared_adj, "LogL (Y-scale)": adj, "AIC (comparable)": aic_adj,
                     "BIC (comparable)": -2 * adj + np.log(r.nobs) * r.k})
    out = pd.DataFrame(rows)
    if "AIC (comparable)" in out:
        out["Best"] = np.where(out["AIC (comparable)"] == out["AIC (comparable)"].min(), "◀ best", "")
    return out


def box_cox(df: pd.DataFrame, spec: ModelSpec, grid=(-2.0, 2.0)) -> dict:
    """Box-Cox (1964) transformation of Y by profile maximum likelihood in the regression context.
    Returns λ̂, its 95% LR confidence interval, and LR tests of λ = 1 (linear) and λ = 0 (log)."""
    sp = ModelSpec(**{**spec.__dict__, "form": "linear" if spec.form in ("linear", "loglin") else "linlog"})
    design = build_design(df, sp)
    y = design.y.to_numpy(float)
    if (y <= 0).any():
        raise ValueError("Box-Cox requires Y > 0.")
    X = design.X.to_numpy(float)
    n = len(y)
    sly = float(np.sum(np.log(y)))

    def ll(lam):
        z = np.log(y) if abs(lam) < 1e-8 else (y ** lam - 1) / lam
        _, ssr, _ = _ols_np(X, z)
        return -n / 2 * np.log(ssr / n) + (lam - 1) * sly

    opt = optimize.minimize_scalar(lambda l: -ll(l), bounds=grid, method="bounded")
    lam = float(opt.x)
    lmax = ll(lam)
    crit = stats.chi2.ppf(0.95, 1) / 2

    def root(a, b):
        try:
            return optimize.brentq(lambda l: lmax - ll(l) - crit, a, b)
        except ValueError:
            return np.nan
    lo, hi = root(grid[0] * 2, lam), root(lam, grid[1] * 2)
    lr1 = 2 * (lmax - ll(1.0))
    lr0 = 2 * (lmax - ll(0.0))
    lams = np.linspace(grid[0], grid[1], 81)
    if abs(lam - 1) < 0.25 or stats.chi2.sf(lr1, 1) > 0.05 >= stats.chi2.sf(lr0, 1):
        rec = "linear"
    elif abs(lam) < 0.25 or stats.chi2.sf(lr0, 1) > 0.05:
        rec = "log"
    else:
        rec = "boxcox"
    return {"lambda": lam, "ci": (lo, hi), "LR_lambda1": lr1, "p_lambda1": float(stats.chi2.sf(lr1, 1)),
            "LR_lambda0": lr0, "p_lambda0": float(stats.chi2.sf(lr0, 1)), "profile": pd.DataFrame(
                {"lambda": lams, "loglik": [ll(l) for l in lams]}), "recommend": rec, "n": n}


# ==========================================================
# Moderation (interaction) & mediation
# ==========================================================
def simple_slopes(r: RegResult, focal: str, moderator: str, values=None) -> pd.DataFrame:
    """Conditional effect of `focal` at chosen moderator values (default mean − 1 SD, mean, mean + 1 SD)."""
    inter = f"{focal}×{moderator}" if f"{focal}×{moderator}" in r.names else f"{moderator}×{focal}"
    if inter not in r.names:
        raise ValueError("The model has no focal × moderator interaction term.")
    mcol = r.design.X[moderator]
    if values is None:
        m, s = mcol.mean(), mcol.std(ddof=1)
        values = {"Mean − 1 SD": m - s, "Mean": m, "Mean + 1 SD": m + s}
    V = r.cov
    q = stats.t.ppf(0.975, r.df_resid) if r.dist == "t" else stats.norm.ppf(0.975)
    rows = []
    for lab, mv in values.items():
        slope = r.params[focal] + r.params[inter] * mv
        se = np.sqrt(V.loc[focal, focal] + mv ** 2 * V.loc[inter, inter] + 2 * mv * V.loc[focal, inter])
        tval = slope / se
        p = 2 * (stats.t.sf(abs(tval), r.df_resid) if r.dist == "t" else stats.norm.sf(abs(tval)))
        rows.append({"Moderator value": lab, moderator: mv, "Slope": slope, "Std. Err.": se, "t": tval,
                     "p-value": p, "CI low": slope - q * se, "CI high": slope + q * se})
    return pd.DataFrame(rows)


def johnson_neyman(r: RegResult, focal: str, moderator: str, alpha: float = 0.05) -> dict:
    """Johnson-Neyman region: moderator values where the focal slope is significant at α."""
    inter = f"{focal}×{moderator}" if f"{focal}×{moderator}" in r.names else f"{moderator}×{focal}"
    b1, b3 = r.params[focal], r.params[inter]
    V = r.cov
    v11, v33, v13 = V.loc[focal, focal], V.loc[inter, inter], V.loc[focal, inter]
    tc = stats.t.ppf(1 - alpha / 2, r.df_resid)
    a = b3 ** 2 - tc ** 2 * v33
    b = 2 * (b1 * b3 - tc ** 2 * v13)
    c = b1 ** 2 - tc ** 2 * v11
    disc = b * b - 4 * a * c
    mcol = r.design.X[moderator]
    roots = [] if disc < 0 or abs(a) < 1e-15 else sorted([(-b - np.sqrt(disc)) / (2 * a), (-b + np.sqrt(disc)) / (2 * a)])
    grid = np.linspace(mcol.min(), mcol.max(), 100)
    slope = b1 + b3 * grid
    se = np.sqrt(v11 + grid ** 2 * v33 + 2 * grid * v13)
    return {"roots": roots, "a": a, "curve": pd.DataFrame({moderator: grid, "slope": slope, "low": slope - tc * se,
                                                            "high": slope + tc * se}),
            "range": (float(mcol.min()), float(mcol.max()))}


def mediation(df: pd.DataFrame, x: str, m: str, y: str, controls=None, reps: int = 2000, seed: int = 2026) -> dict:
    """Simple mediation X → M → Y: Baron-Kenny paths, Sobel test, percentile bootstrap CI of a·b
    (Preacher & Hayes 2008)."""
    controls = list(controls or [])
    d = df[[x, m, y] + controls].apply(pd.to_numeric, errors="coerce").dropna()
    n = len(d)

    def paths(dd):
        Xa = np.c_[np.ones(len(dd)), dd[[x] + controls].to_numpy(float)]
        a = np.linalg.lstsq(Xa, dd[m].to_numpy(float), rcond=None)[0][1]
        Xb = np.c_[np.ones(len(dd)), dd[[x, m] + controls].to_numpy(float)]
        bb = np.linalg.lstsq(Xb, dd[y].to_numpy(float), rcond=None)[0]
        c = np.linalg.lstsq(Xa, dd[y].to_numpy(float), rcond=None)[0][1]
        return a, bb[2], bb[1], c

    ra = sm.OLS(d[m], sm.add_constant(d[[x] + controls])).fit()
    rb = sm.OLS(d[y], sm.add_constant(d[[x, m] + controls])).fit()
    rc = sm.OLS(d[y], sm.add_constant(d[[x] + controls])).fit()
    a, sa = ra.params[x], ra.bse[x]
    b, sb = rb.params[m], rb.bse[m]
    cp, c = rb.params[x], rc.params[x]
    ind = a * b
    sobel_se = np.sqrt(b ** 2 * sa ** 2 + a ** 2 * sb ** 2)
    z = ind / sobel_se
    rng = np.random.default_rng(seed)
    arr = d.to_numpy()
    boot = np.empty(reps)
    for i in range(reps):
        s = pd.DataFrame(arr[rng.integers(0, n, n)], columns=d.columns)
        aa, bbb, _, _ = paths(s)
        boot[i] = aa * bbb
    lo, hi = np.percentile(boot, [2.5, 97.5])
    table = pd.DataFrame([
        {"Path": f"a: {x} → {m}", "Coef.": a, "Std. Err.": sa, "p-value": ra.pvalues[x]},
        {"Path": f"b: {m} → {y} (controlling {x})", "Coef.": b, "Std. Err.": sb, "p-value": rb.pvalues[m]},
        {"Path": f"c′: direct effect {x} → {y}", "Coef.": cp, "Std. Err.": rb.bse[x], "p-value": rb.pvalues[x]},
        {"Path": f"c: total effect {x} → {y}", "Coef.": c, "Std. Err.": rc.bse[x], "p-value": rc.pvalues[x]},
        {"Path": "Indirect effect a·b (Sobel)", "Coef.": ind, "Std. Err.": sobel_se, "p-value": 2 * stats.norm.sf(abs(z))},
    ])
    if (lo > 0) or (hi < 0):
        kind = "partial" if rb.pvalues[x] < 0.05 else "full"
    else:
        kind = "none"
    return {"table": table, "indirect": ind, "boot_ci": (lo, hi), "sobel_z": z, "sobel_p": 2 * stats.norm.sf(abs(z)),
            "prop_mediated": ind / c if c != 0 else np.nan, "type": kind, "n": n, "reps": reps, "boot": boot}
