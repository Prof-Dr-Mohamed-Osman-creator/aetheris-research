# ==========================================
# Aetheris Research - Count data & survival / duration models (Catalogue §1.5)
# ------------------------------------------
# Count: Poisson, negative binomial (NB2), zero-inflated Poisson / NB, hurdle Poisson / NB, PPML —
#   incidence-rate ratios, AME, overdispersion tests (Cameron-Trivedi, LR α = 0), Vuong tests,
#   observed vs predicted count frequencies.
# Survival: Kaplan-Meier (Greenwood / log-log CIs), Nelson-Aalen, log-rank test, Cox proportional
#   hazards (Efron / Breslow ties) with hazard ratios and scaled-Schoenfeld PH tests, parametric AFT
#   (exponential, Weibull, log-normal, log-logistic), discrete-time hazard (logit / cloglog).
# ==========================================

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import special, stats

from analytics import regression as rg
from analytics.ldv import _design, _groups, _lnPhi, _mle, _result

warnings.filterwarnings("ignore")

COUNT = {
    "poisson": ("Poisson", "بواسون"),
    "negbin": ("Negative binomial (NB2)", "ذو الحدين السالب (NB2)"),
    "zip": ("Zero-inflated Poisson (ZIP)", "بواسون المتضخم بالأصفار (ZIP)"),
    "zinb": ("Zero-inflated negative binomial (ZINB)", "ذو الحدين السالب المتضخم بالأصفار (ZINB)"),
    "hurdle_p": ("Hurdle Poisson (logit / truncated Poisson)", "عقبة بواسون (لوجيت / بواسون مبتور)"),
    "hurdle_nb": ("Hurdle negative binomial", "عقبة ذو الحدين السالب"),
    "ppml": ("Poisson pseudo-ML (PPML, robust)", "بواسون شبه الإمكان الأعظم (PPML، متين)"),
}
AFT = {
    "exponential": ("Exponential", "أسي"), "weibull": ("Weibull", "ويبل"),
    "lognormal": ("Log-normal", "لوغاريتمي طبيعي"), "loglogistic": ("Log-logistic", "لوجستي لوغاريتمي"),
}


# ==========================================================
# Count models
# ==========================================================
def fit_count(df: pd.DataFrame, spec: rg.ModelSpec, model: str = "poisson", inflate: list | None = None,
              exposure: str | None = None, cov_type: str = "nonrobust", cov_opts: dict | None = None,
              name: str = "") -> rg.RegResult:
    design = _design(df, spec)
    y = design.y.to_numpy(float)
    if (y < 0).any():
        raise ValueError("Counts must be non-negative.")
    if model != "ppml" and not np.allclose(y, np.round(y)):
        raise ValueError("Count models need integer outcomes (use PPML for non-negative continuous outcomes).")
    X = design.X
    n, k = X.shape
    off = np.log(pd.to_numeric(df.loc[design.index, exposure], errors="coerce").to_numpy(float)) if exposure else None
    kw = {}
    if cov_type in ("HC0", "robust") or model == "ppml":
        kw = {"cov_type": "HC0"}
    if cov_type == "cluster":
        kw = {"cov_type": "cluster", "cov_kwds": {"groups": _groups(df, design, cov_type, cov_opts)}}
    Xa = X.to_numpy(float)
    notes, extra = [], {"family": "count", "model": model}
    if model in ("poisson", "ppml"):
        res = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit(**kw)
        theta = np.asarray(res.params)
        cov = np.asarray(res.cov_params())
        llf = float(np.sum(stats.poisson.logpmf(np.round(y), res.fittedvalues))) if model == "poisson" else res.llf
        ll0 = float(sm.GLM(y, np.ones((n, 1)), family=sm.families.Poisson(), offset=off).fit().llf)
        names = list(X.columns)
        kx = k

        def pred(Xm, t):
            return np.exp(Xm @ t[:Xm.shape[1]])
        mu = res.fittedvalues
        # Cameron & Trivedi (1990) auxiliary regression for overdispersion (NB2 alternative)
        aux = ((y - mu) ** 2 - y) / mu
        ct = sm.OLS(aux, mu).fit()
        extra["overdispersion"] = {"alpha_hat": float(ct.params[0]), "t": float(ct.tvalues[0]),
                                   "p": float(stats.t.sf(ct.tvalues[0], n - 1))}
        extra["pearson_disp"] = float(np.sum((y - mu) ** 2 / mu) / (n - k))
        if model == "ppml":
            notes.append("PPML (Santos Silva & Tenreyro 2006): Poisson quasi-ML with robust standard errors — "
                         "consistent whenever E[y|x] = exp(xβ), whatever the distribution; zeros are kept.")
    elif model == "negbin":
        res = sm.NegativeBinomial(y, X, loglike_method="nb2", offset=off).fit(disp=0, maxiter=500, **kw)
        theta = np.asarray(res.params)
        cov = np.asarray(res.cov_params())
        llf, ll0 = res.llf, res.llnull
        names = list(X.columns) + ["alpha"]
        kx = k

        def pred(Xm, t):
            return np.exp(Xm @ t[:Xm.shape[1]])
        mu = np.exp(Xa @ theta[:k])
        lp = sm.GLM(y, X, family=sm.families.Poisson(), offset=off).fit().llf
        LR = max(0.0, 2 * (llf - lp))
        extra["lr_alpha"] = {"LR": float(LR), "p": float(0.5 * stats.chi2.sf(LR, 1))}
        notes.append(f"LR test of α = 0 (Poisson vs NB2): χ̄²₀₁ = {LR:.3f}, p = {0.5 * stats.chi2.sf(LR, 1):.4f} "
                     "(boundary-corrected, half χ²(1)).")
    elif model in ("zip", "zinb"):
        Zi = np.c_[np.ones(n), df.loc[design.index, inflate].apply(pd.to_numeric, errors="coerce").to_numpy(float)] \
            if inflate else np.ones((n, 1))
        cls = sm.ZeroInflatedPoisson if model == "zip" else sm.ZeroInflatedNegativeBinomialP
        mkw = {"exog_infl": Zi, "inflation": "logit", "offset": off}
        if model == "zinb":
            mkw["p"] = 2
        mod = cls(y, X, **mkw)
        res = None
        for meth in ("bfgs", "nm", "newton"):
            try:
                r0 = mod.fit(disp=0, maxiter=5000, method=meth, **kw)
                if res is None or r0.llf > res.llf:
                    res = r0
            except Exception:  # noqa: BLE001
                continue
        theta = np.asarray(res.params)
        cov = np.asarray(res.cov_params())
        llf = res.llf
        kz = Zi.shape[1]
        names = [f"inflate: {c}" for c in (["const"] + list(inflate or []))] + list(X.columns) + (["alpha"] if model == "zinb" else [])
        # statsmodels orders inflation parameters first
        ll0 = float(cls(y, np.ones((n, 1)), exog_infl=np.ones((n, 1)), **({"p": 2} if model == "zinb" else {})).fit(disp=0, maxiter=2000).llf)
        kx = k

        def pred(Xm, t, kz=kz, k=k):
            pi = special.expit(np.full(len(Xm), t[0])) if kz == 1 else np.full(len(Xm), np.nan)
            return (1 - pi) * np.exp(Xm @ t[kz:kz + k])
        if kz > 1:
            pred = None
            notes.append("Average effects on E[y] are reported for the count part only when the inflation equation has "
                         "covariates; see the inflation coefficients for the excess-zero process.")
        # Vuong test vs the non-inflated model
        base_mod = sm.Poisson(y, X, offset=off).fit(disp=0) if model == "zip" else \
            sm.NegativeBinomial(y, X, loglike_method="nb2", offset=off).fit(disp=0, maxiter=500)
        l1 = mod.loglikeobs(res.params)
        l2 = base_mod.model.loglikeobs(base_mod.params)
        m_ = l1 - l2
        v = np.sqrt(n) * m_.mean() / m_.std(ddof=1)
        extra["vuong"] = {"z": float(v), "p": float(stats.norm.sf(v))}
        notes.append(f"Vuong test, zero-inflated vs standard model: z = {v:.3f}, one-sided p = {stats.norm.sf(v):.4f}"
                     + (" → zero inflation is supported." if stats.norm.sf(v) < 0.05 else " → no evidence of zero inflation."))
        mu = (1 - special.expit(Zi @ theta[:kz])) * np.exp(Xa @ theta[kz:kz + k])
    elif model in ("hurdle_p", "hurdle_nb"):
        # Mullahy (1986) hurdle: binary hurdle for zero vs positive (logit, or the Poisson-type cloglog
        # link) + zero-truncated Poisson / NB2 for the positive counts. The parts are estimated separately
        # (the likelihood factorises), exactly as R pscl::hurdle.
        zd = (cov_opts or {}).get("zero_dist", "logit")
        pos = y > 0
        link = sm.families.links.Logit() if zd == "logit" else sm.families.links.CLogLog()
        zr = sm.GLM(pos.astype(float), X, family=sm.families.Binomial(link=link)).fit()
        Xp, yp = Xa[pos], y[pos]
        nb = model == "hurdle_nb"

        def llc(th):
            b = th[:k]
            lam = np.exp(Xp @ b)
            if not nb:
                return stats.poisson.logpmf(yp, lam) - np.log1p(-np.exp(-lam))
            a = np.exp(th[k])
            r_ = 1 / a
            lp = special.gammaln(yp + r_) - special.gammaln(r_) - special.gammaln(yp + 1) \
                + r_ * np.log(r_ / (r_ + lam)) + yp * np.log(lam / (r_ + lam))
            p0 = (r_ / (r_ + lam)) ** r_
            return lp - np.log1p(-p0)
        b0 = np.asarray(sm.GLM(yp, Xp, family=sm.families.Poisson()).fit().params)
        start = np.r_[b0, np.log(0.5)] if nb else b0
        cres = _mle(llc, start, list(X.columns) + (["ln alpha"] if nb else []), int(pos.sum()))
        cth = np.asarray(cres.params)
        ccov = np.asarray(cres.cov_params())
        if nb:
            T = np.eye(k + 1)
            T[k, k] = np.exp(cth[k])
            cth = np.r_[cth[:k], np.exp(cth[k])]
            ccov = T @ ccov @ T.T
        theta = np.r_[np.asarray(zr.params), cth]
        cov = np.zeros((len(theta), len(theta)))
        cov[:k, :k] = np.asarray(zr.cov_params())
        cov[k:, k:] = ccov
        llf = float(zr.llf + cres.llf)
        names = [f"zero: {c}" for c in X.columns] + list(X.columns) + (["alpha"] if nb else [])
        ll0 = np.nan
        kx = k
        Fz = special.expit if zd == "logit" else (lambda z: 1 - np.exp(-np.exp(np.clip(z, -50, 50))))

        def pred(Xm, t, k=k, nb=nb, Fz=Fz):
            p_pos = Fz(Xm @ t[:k])
            lam = np.exp(Xm @ t[k:2 * k])
            p0c = (1 + t[2 * k] * lam) ** (-1 / t[2 * k]) if nb else np.exp(-lam)
            return p_pos * lam / (1 - p0c)
        mu = pred(Xa, theta)
        extra["zero_dist"] = zd
        notes.append(f"Hurdle model (Mullahy 1986): {'logit' if zd == 'logit' else 'Poisson-type (cloglog)'} hurdle "
                     "for zero vs positive and a zero-truncated "
                     f"{'negative binomial' if nb else 'Poisson'} for positive counts; the two parts may have "
                     "different covariate effects.")
    else:
        raise ValueError(model)
    r = _result(name or COUNT[model][0], model, design, names, theta, cov, llf, n, y, np.asarray(mu), kx, pred=pred,
                outputs=None, llnull=ll0, notes=notes, cov_type=kw.get("cov_type", "nonrobust"), extra=extra)
    if model in ("poisson", "negbin", "ppml"):
        r.extra["irr"] = pd.DataFrame({"Variable": names[:k], "IRR": np.exp(theta[:k]),
                                       "CI low": np.exp(r.ci["low"].values[:k]), "CI high": np.exp(r.ci["high"].values[:k]),
                                       "p-value": r.pvalues.values[:k]})
    if model != "ppml":
        r.extra["freq"] = count_frequencies(r, y, model, theta, Xa, names)
    return r


def count_frequencies(r, y, model, theta, X, names, maxc: int = 10) -> pd.DataFrame:
    """Observed vs average predicted probability of each count (the basis of a rootogram)."""
    n, k = X.shape
    ks = np.arange(0, maxc + 1)
    if model == "poisson":
        mu = np.exp(X @ theta[:k])
        P = stats.poisson.pmf(ks[None, :], mu[:, None])
    elif model == "negbin":
        mu = np.exp(X @ theta[:k])
        a = theta[k]
        P = stats.nbinom.pmf(ks[None, :], 1 / a, (1 / a) / (1 / a + mu[:, None]))
    elif model in ("zip", "zinb"):
        kz = sum(1 for c in names if c.startswith("inflate:"))
        if kz > 1:
            return pd.DataFrame()
        pi = special.expit(theta[0])
        mu = np.exp(X @ theta[kz:kz + k])
        base = stats.poisson.pmf(ks[None, :], mu[:, None]) if model == "zip" else \
            stats.nbinom.pmf(ks[None, :], 1 / theta[-1], (1 / theta[-1]) / (1 / theta[-1] + mu[:, None]))
        P = (1 - pi) * base
        P[:, 0] += pi
    elif model in ("hurdle_p", "hurdle_nb"):
        p_pos = special.expit(X @ theta[:k]) if r.extra.get("zero_dist", "logit") == "logit" else 1 - np.exp(-np.exp(X @ theta[:k]))
        lam = np.exp(X @ theta[k:2 * k])
        if model == "hurdle_p":
            base, p0 = stats.poisson.pmf(ks[None, :], lam[:, None]), np.exp(-lam)
        else:
            a = theta[2 * k]
            base = stats.nbinom.pmf(ks[None, :], 1 / a, (1 / a) / (1 / a + lam[:, None]))
            p0 = (1 + a * lam) ** (-1 / a)
        P = p_pos[:, None] * base / (1 - p0[:, None])
        P[:, 0] = 1 - p_pos
    else:
        return pd.DataFrame()
    obs = np.array([np.mean(y == c) for c in ks])
    exp_ = P.mean(axis=0)
    return pd.DataFrame({"Count": [str(c) for c in ks[:-1]] + [f"{maxc}"], "Observed share": obs,
                         "Predicted share": exp_, "Difference": obs - exp_})


# ==========================================================
# Survival analysis
# ==========================================================
def kaplan_meier(time, event, conf: float = 0.95) -> pd.DataFrame:
    """Product-limit estimator with Greenwood variance and log(−log) confidence limits (as R / Stata)."""
    t = np.asarray(time, float)
    d = np.asarray(event, int)
    ut = np.unique(t[d == 1])
    rows = []
    S, gw = 1.0, 0.0
    z = stats.norm.ppf(0.5 + conf / 2)
    H = 0.0
    for u in ut:
        n_risk = int(np.sum(t >= u))
        n_ev = int(np.sum((t == u) & (d == 1)))
        n_cens = int(np.sum((t == u) & (d == 0)))
        S *= 1 - n_ev / n_risk
        if n_risk > n_ev:
            gw += n_ev / (n_risk * (n_risk - n_ev))
        H += n_ev / n_risk
        se = S * np.sqrt(gw)
        if 0 < S < 1:
            ll = np.log(-np.log(S))
            sl = np.sqrt(gw) / abs(np.log(S))
            lo, hi = np.exp(-np.exp(ll + z * sl)), np.exp(-np.exp(ll - z * sl))
        else:
            lo = hi = S
        rows.append({"time": u, "at risk": n_risk, "events": n_ev, "censored": n_cens, "survival": S, "std. err.": se,
                     "CI low": lo, "CI high": hi, "cum. hazard (Nelson-Aalen)": H})
    return pd.DataFrame(rows)


def km_median(km: pd.DataFrame) -> dict:
    def first_below(col):
        s = km[km[col] <= 0.5]
        return float(s["time"].iloc[0]) if not s.empty else np.nan
    return {"median": first_below("survival"), "ci_low": first_below("CI low"), "ci_high": first_below("CI high")}


def logrank(time, event, group) -> dict:
    """Mantel-Cox log-rank test for equality of survivor functions across k groups (χ², k − 1 df)."""
    t = np.asarray(time, float)
    d = np.asarray(event, int)
    g = pd.Series(group).astype(str).to_numpy()
    levels = sorted(np.unique(g))
    K = len(levels)
    O = np.zeros(K)
    E = np.zeros(K)
    V = np.zeros((K, K))
    for u in np.unique(t[d == 1]):
        at = t >= u
        n = at.sum()
        dd = ((t == u) & (d == 1)).sum()
        nk = np.array([(at & (g == lv)).sum() for lv in levels])
        dk = np.array([((t == u) & (d == 1) & (g == lv)).sum() for lv in levels])
        O += dk
        E += dd * nk / n
        if n > 1:
            V += dd * (n - dd) / (n - 1) * (np.diag(nk / n) - np.outer(nk, nk) / n ** 2)
    o_e = (O - E)[:-1]
    chi = float(o_e @ np.linalg.pinv(V[:-1, :-1]) @ o_e)
    return {"chi2": chi, "df": K - 1, "p": float(stats.chi2.sf(chi, K - 1)),
            "table": pd.DataFrame({"Group": levels, "Observed events": O, "Expected events": E})}


def _surv_design(df, spec: rg.ModelSpec, time: str, event: str):
    s = rg.ModelSpec(**{**spec.__dict__, "y": time})
    design = _design(df, s)
    ev = pd.to_numeric(df.loc[design.index, event], errors="coerce")
    keep = ev.notna() & (design.y > 0)
    if not keep.all():
        design = _design(df.loc[design.index[keep.to_numpy()]], s)
        ev = pd.to_numeric(df.loc[design.index, event], errors="coerce")
    return design, design.y.to_numpy(float), ev.to_numpy(int)


def fit_cox(df: pd.DataFrame, spec: rg.ModelSpec, time: str, event: str, ties: str = "efron",
            robust: bool = False, name: str = "") -> rg.RegResult:
    design, t, d = _surv_design(df, spec, time, event)
    X = design.X.drop(columns=[c for c in ("const",) if c in design.X.columns])
    n, k = X.shape
    mod = sm.PHReg(t, X, status=d, ties=ties)
    res = mod.fit(groups=np.arange(n)) if robust else mod.fit()
    theta = np.asarray(res.params)
    cov = np.asarray(res.cov_params())
    ll0 = mod.loglike(np.zeros(k))
    xb = X.to_numpy(float) @ theta

    def pred(Xm, th, k=k):
        Xm = Xm[:, 1:] if Xm.shape[1] == k + 1 else Xm
        return np.exp(Xm @ th[:k])
    r = _result(name or "Cox proportional hazards", "cox", design, list(X.columns), theta, cov, res.llf, n, t, xb, k,
                pred=pred, outputs=None, llnull=ll0, cov_type="robust" if robust else "nonrobust",
                extra={"family": "survival", "time": t, "event": d, "ties": ties, "sm": res, "k_beta": k,
                       "event_name": event})
    r.extra["stats"].update({"Events": int(d.sum()), "Concordance (Harrell C)": harrell_c(t, d, xb)})
    r.extra["hr"] = pd.DataFrame({"Variable": X.columns, "Hazard ratio": np.exp(theta), "CI low": np.exp(r.ci["low"].values),
                                  "CI high": np.exp(r.ci["high"].values), "p-value": r.pvalues.values})
    r.extra["ph_test"] = ph_test(res, t, d, list(X.columns))
    return r


def harrell_c(t, d, risk) -> float:
    """Harrell's concordance index (ties in risk count ½)."""
    t, d, risk = np.asarray(t), np.asarray(d), np.asarray(risk)
    order = np.argsort(t)
    t, d, risk = t[order], d[order], risk[order]
    num = den = 0.0
    for i in range(len(t)):
        if not d[i]:
            continue
        later = t > t[i]
        den += later.sum()
        num += np.sum(risk[i] > risk[later]) + 0.5 * np.sum(risk[i] == risk[later])
    return float(num / den) if den else np.nan


def ph_test(res, t, d, names, transform: str = "rank") -> pd.DataFrame:
    """Grambsch & Therneau (1994) test of proportional hazards: scaled Schoenfeld residuals regressed on a
    transform of time (rank, as in lifelines' default), per covariate and global."""
    sch = np.asarray(res.schoenfeld_residuals)
    ev = d == 1
    S = sch[ev]
    te = t[ev]
    g = stats.rankdata(te) if transform == "rank" else np.log(te) if transform == "log" else te
    g = g - g.mean()
    V = np.asarray(res.cov_params())
    m = ev.sum()
    scaled = S @ V * m        # scaled Schoenfeld residuals (Grambsch-Therneau)
    rows = []
    for j, nm in enumerate(names):
        # T = (Σ g·s*)² / (m · V_jj · Σ g²)   (one-df score-type statistic)
        Tj = float((g @ scaled[:, j]) ** 2 / (m * V[j, j] * (g @ g)))
        rows.append({"Variable": nm, "χ²": Tj, "df": 1, "p-value": float(stats.chi2.sf(Tj, 1))})
    U = g @ S
    Tg = float(U @ V @ U * m / (g @ g))
    rows.append({"Variable": "GLOBAL", "χ²": Tg, "df": len(names), "p-value": float(stats.chi2.sf(Tg, len(names)))})
    out = pd.DataFrame(rows)
    out["PH assumption"] = np.where(out["p-value"] >= 0.05, "holds", "violated")
    return out


def fit_aft(df: pd.DataFrame, spec: rg.ModelSpec, time: str, event: str, dist: str = "weibull",
            name: str = "") -> rg.RegResult:
    """Parametric accelerated-failure-time model ln T = xβ + σε (ε: extreme value → Weibull, σ ≡ 1 →
    exponential, normal → log-normal, logistic → log-logistic), right-censoring, maximum likelihood.
    Parameterisation identical to R's survreg (Scale = σ)."""
    design, t, d = _surv_design(df, spec, time, event)
    X = design.X.to_numpy(float)
    n, k = X.shape
    lt = np.log(t)

    def ll(th):
        b = th[:k]
        s = 1.0 if dist == "exponential" else np.exp(th[k])
        w = (lt - X @ b) / s
        if dist in ("weibull", "exponential"):
            lf, lS = w - np.exp(w), -np.exp(w)
        elif dist == "lognormal":
            lf, lS = stats.norm.logpdf(w), _lnPhi(-w)
        else:
            lf, lS = -w - 2 * np.log1p(np.exp(-w)), -np.log1p(np.exp(w))
        return d * (lf - np.log(s) - lt) + (1 - d) * lS
    b0 = np.linalg.lstsq(X, lt, rcond=None)[0]
    start = b0 if dist == "exponential" else np.r_[b0, np.log(np.std(lt - X @ b0) + 1e-3)]
    names = list(design.X.columns) + ([] if dist == "exponential" else ["ln σ (log scale)"])
    res = _mle(ll, start, names, n)
    th = np.asarray(res.params)
    cov = np.asarray(res.cov_params())

    def pred(Xm, t_, k=k):
        s = 1.0 if dist == "exponential" else np.exp(t_[k])
        mu = Xm @ t_[:k]
        if dist in ("weibull", "exponential"):
            return np.exp(mu) * special.gamma(1 + s)          # mean survival time
        if dist == "lognormal":
            return np.exp(mu + s ** 2 / 2)
        return np.exp(mu) * (np.pi * s / np.sin(np.pi * s)) if s < 1 else np.full(len(mu), np.inf)
    r = _result(name or f"AFT {AFT[dist][0]}", f"aft_{dist}", design, names, th, cov, res.llf, n, t, X @ th[:k], k,
                pred=pred, outputs=None, extra={"family": "survival", "dist": dist, "time": t, "event": d,
                                               "event_name": event})
    r.extra["stats"]["Events"] = int(d.sum())
    tr = pd.DataFrame({"Variable": design.X.columns, "Time ratio exp(β)": np.exp(th[:k]),
                       "CI low": np.exp(r.ci["low"].values[:k]), "CI high": np.exp(r.ci["high"].values[:k]),
                       "p-value": r.pvalues.values[:k]})
    r.extra["time_ratios"] = tr
    if dist in ("weibull", "exponential"):
        s = 1.0 if dist == "exponential" else float(np.exp(th[k]))
        r.extra["stats"]["Weibull shape p = 1/σ"] = 1 / s
        hr = np.exp(-th[:k] / s)
        r.extra["ph_form"] = pd.DataFrame({"Variable": design.X.columns, "Hazard ratio exp(−β/σ)": hr})
        r.notes.append(f"Weibull shape p = 1/σ = {1 / s:.3f} — " + (
            "hazard rising with duration (positive duration dependence)." if 1 / s > 1 else
            "hazard falling with duration (negative duration dependence)." if 1 / s < 1 else "constant hazard."))
    return r


def person_period(df: pd.DataFrame, time: str, event: str, keep: list, width: float = 1.0) -> pd.DataFrame:
    """Expand to person-period format for discrete-time hazard models (one row per unit and interval at risk)."""
    rows = []
    d = df[[time, event] + keep].dropna()
    for _, r in d.iterrows():
        T = int(np.ceil(float(r[time]) / width))
        for p in range(1, max(T, 1) + 1):
            row = {c: r[c] for c in keep}
            row["period"] = p
            row["event_now"] = int(r[event] == 1 and p == max(T, 1))
            rows.append(row)
    return pd.DataFrame(rows)


def fit_discrete_hazard(df: pd.DataFrame, spec: rg.ModelSpec, time: str, event: str, link: str = "cloglog",
                        width: float = 1.0, baseline: str = "dummies", name: str = "") -> rg.RegResult:
    """Discrete-time hazard (Allison 1982; Jenkins 1995): binary model on person-period data with a flexible
    baseline (period dummies or log(period)). cloglog = grouped-time proportional hazards (Prentice-Gloeckler)."""
    from analytics.ldv import fit_binary
    keep = list(dict.fromkeys(list(spec.x) + list(spec.categorical)))
    pp = person_period(df, time, event, keep, width)
    if baseline == "dummies":
        pp["period_f"] = pp["period"].astype(int).astype(str).str.zfill(3)
        s = rg.ModelSpec(y="event_now", x=list(spec.x), categorical=list(spec.categorical) + ["period_f"],
                         reference={"period_f": "001"}, form=spec.form, nolog=list(spec.nolog))
    else:
        pp["ln_period"] = np.log(pp["period"])
        s = rg.ModelSpec(y="event_now", x=list(spec.x) + ["ln_period"], categorical=list(spec.categorical),
                         form=spec.form, nolog=list(spec.nolog) + ["ln_period"])
    r = fit_binary(pp, s, link=link, name=name or f"Discrete-time hazard ({link})")
    r.extra.update({"family": "discrete_hazard", "person_period": pp, "n_units": int(len(df[[time, event]].dropna()))})
    r.notes.append(f"Person-period data: {len(pp)} unit-intervals from {r.extra['n_units']} units (interval width {width}).")
    return r
