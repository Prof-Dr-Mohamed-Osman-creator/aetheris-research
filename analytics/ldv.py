# ==========================================
# Aetheris Research - Limited dependent variable models (Catalogue §1.4)
# ------------------------------------------
# Binary: linear probability, logit, probit, complementary log-log — odds ratios, AME / MEM,
#   classification table, ROC / AUC, Hosmer-Lemeshow, Pregibon link test, pseudo-R² family.
# Multinomial logit (relative-risk ratios, AME per outcome, Hausman-McFadden IIA test).
# Ordered logit / probit (cut-points, category probabilities, AME, Brant parallel-lines test).
# Censored & selected: Tobit (McDonald-Moffitt decomposition), truncated regression, Heckman
#   (two-step with corrected SEs, and full ML), Cragg truncated-normal hurdle, Jones double hurdle,
#   fractional response (Papke-Wooldridge quasi-ML).
# Pure numpy / scipy / statsmodels — no Streamlit.
# ==========================================

from __future__ import annotations

import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import special, stats
from statsmodels.base.model import GenericLikelihoodModel

from analytics import regression as rg

warnings.filterwarnings("ignore", category=RuntimeWarning)

BINARY = {
    "lpm": ("Linear probability model (OLS, robust SE)", "نموذج الاحتمال الخطي (OLS بأخطاء متينة)"),
    "logit": ("Logit", "لوجيت"),
    "probit": ("Probit", "بروبيت"),
    "cloglog": ("Complementary log-log", "اللوغاريتم المزدوج التكميلي (cloglog)"),
}
CENSORED = {
    "tobit": ("Tobit (censored regression)", "توبيت (الانحدار المراقَب)"),
    "truncreg": ("Truncated regression", "الانحدار المبتور"),
    "heckman2": ("Heckman selection — two-step", "نموذج هيكمان للاختيار — خطوتان"),
    "heckman_ml": ("Heckman selection — maximum likelihood", "نموذج هيكمان للاختيار — الإمكان الأعظم"),
    "cragg": ("Cragg truncated-normal hurdle", "نموذج العقبة لكراج (طبيعي مبتور)"),
    "dhurdle": ("Double hurdle (Jones, independent)", "العقبة المزدوجة (جونز، مستقلة)"),
    "frac": ("Fractional response (Papke-Wooldridge)", "الاستجابة الكسرية (بابكي-وولدريدج)"),
}


# ==========================================================
# Helpers
# ==========================================================
def _cdf(link):
    return {"logit": special.expit, "probit": stats.norm.cdf,
            "cloglog": lambda z: 1 - np.exp(-np.exp(np.clip(z, -50, 50))), "lpm": lambda z: z}[link]


def _result(name, method, design, names, theta, cov, llf, n, y, fitted, kx, pred=None, outputs=None,
            llnull=np.nan, notes=None, extra=None, dist="z", cov_type="nonrobust"):
    theta = np.asarray(theta, float)
    cov = np.asarray(cov, float)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    z = theta / np.where(se > 0, se, np.nan)
    p = 2 * stats.norm.sf(np.abs(z))
    q = stats.norm.ppf(0.975)
    r = rg.RegResult(name=name, method=method, cov_type=cov_type, design=design, params=pd.Series(theta, index=names),
                     bse=pd.Series(se, index=names), tvalues=pd.Series(z, index=names), pvalues=pd.Series(p, index=names),
                     ci=pd.DataFrame(np.c_[theta - q * se, theta + q * se], index=names, columns=["low", "high"]),
                     cov=pd.DataFrame(cov, index=names, columns=names), dist=dist, df_resid=n - len(theta), nobs=n,
                     y=np.asarray(y, float) if y is not None else np.zeros(n), fitted=np.asarray(fitted, float),
                     resid=np.zeros(n), wresid=np.zeros(n), wexog=design.X.to_numpy(float) if design is not None else np.zeros((n, 1)),
                     notes=list(notes or []) + (list(design.notes) if design is not None else []))
    k = len(theta)
    r.llf = float(llf)
    r.aic, r.bic = -2 * llf + 2 * k, -2 * llf + np.log(n) * k
    r.hqic = -2 * llf + 2 * k * np.log(np.log(n))
    st = {"Log-likelihood": float(llf)}
    if np.isfinite(llnull):
        lr = 2 * (llf - llnull)
        dfm = kx - 1 if design is not None and "const" in design.X.columns else kx
        st.update({"Pseudo R²": 1 - llf / llnull, "LR χ²": lr, "Prob > χ²": float(stats.chi2.sf(lr, max(dfm, 1))),
                   "LL (null)": float(llnull)})
        r.rsquared = st["Pseudo R²"]
        r.fvalue, r.f_pvalue = lr, st["Prob > χ²"]
    r.extra.update({"stats": st, "kx": kx, "pred": pred, "outputs": outputs, **(extra or {})})
    return r


def _design(df, spec, y_raw=False, allow_logy=False):
    if spec.form not in ("linear", "linlog") and not allow_logy:
        spec = rg.ModelSpec(**{**spec.__dict__, "form": "linear"})
    return rg.build_design(df, spec, y_raw=y_raw)


# ==========================================================
# Binary response
# ==========================================================
def fit_binary(df: pd.DataFrame, spec: rg.ModelSpec, link: str = "logit", cov_type: str = "nonrobust",
               cov_opts: dict | None = None, name: str = "") -> rg.RegResult:
    cov_opts = dict(cov_opts or {})
    design = _design(df, spec)
    y = design.y.to_numpy(float)
    if not set(np.unique(y)) <= {0.0, 1.0}:
        raise ValueError("The dependent variable must be coded 0 / 1.")
    X = design.X
    n, k = X.shape
    groups = None
    if cov_type == "cluster":
        g = cov_opts.get("groups")
        groups = pd.factorize(df.loc[design.index, g].astype(str) if isinstance(g, str) else pd.Series(g))[0]
        cov_opts["groups_name"] = g if isinstance(g, str) else None
    if link == "lpm":
        r = rg.fit(df, spec if spec.form in ("linear", "linlog") else rg.ModelSpec(**{**spec.__dict__, "form": "linear"}),
                   cov_type="HC1" if cov_type == "nonrobust" else cov_type, cov_opts=cov_opts, name=name or "LPM")
        r.method = "lpm"
        out = np.mean((r.fitted < 0) | (r.fitted > 1))
        r.notes.append(f"{out:.1%} of fitted probabilities lie outside [0, 1] — a known limitation of the LPM; "
                       "robust (HC1) standard errors are used because the LPM is heteroskedastic by construction.")
        r.extra.update({"pred": lambda Xm, th: Xm @ th[: Xm.shape[1]], "kx": r.k, "outputs": None,
                        "family": "binary", "link": "lpm",
                        "stats": {"Mean of fitted": float(r.fitted.mean())}})
        r.extra["classification"] = classification(y, np.clip(r.fitted, 0, 1))
        return r
    kw = {}
    if cov_type in ("HC0", "robust"):
        kw = {"cov_type": "HC0"}
    elif cov_type == "cluster":
        kw = {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
    if link in ("logit", "probit"):
        mod = (sm.Logit if link == "logit" else sm.Probit)(y, X)
        res = mod.fit(disp=0, maxiter=200, **kw)
        llnull = res.llnull
    else:
        fam = sm.families.Binomial(link=sm.families.links.CLogLog())
        res = sm.GLM(y, X, family=fam).fit(**kw)
        llnull = sm.GLM(y, np.ones((n, 1)), family=fam).fit().llf
    F = _cdf(link)
    theta = np.asarray(res.params)
    p = F(X.to_numpy(float) @ theta)
    r = _result(name or BINARY[link][0], link, design, list(X.columns), theta, np.asarray(res.cov_params()), res.llf,
                n, y, p, k, pred=lambda Xm, th, F=F: F(Xm @ th[: Xm.shape[1]]), llnull=llnull,
                cov_type=("HC0" if kw.get("cov_type") == "HC0" else cov_type),
                extra={"family": "binary", "link": link, "sm": res})
    r.cov_opts = cov_opts
    r.extra["classification"] = classification(y, p)
    st = r.extra["stats"]
    ll0, llf = llnull, res.llf
    st["McFadden adj. R²"] = 1 - (llf - k) / ll0
    st["Cox-Snell R²"] = 1 - np.exp(2 * (ll0 - llf) / n)
    st["Nagelkerke R²"] = st["Cox-Snell R²"] / (1 - np.exp(2 * ll0 / n))
    st["Efron R²"] = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    if link == "logit":
        r.extra["odds_ratios"] = pd.DataFrame({"Variable": r.names, "Odds ratio": np.exp(theta),
                                               "CI low": np.exp(r.ci["low"].values), "CI high": np.exp(r.ci["high"].values),
                                               "p-value": r.pvalues.values})
    return r


def classification(y, p, cutoff: float = 0.5) -> dict:
    y = np.asarray(y, int)
    yhat = (np.asarray(p) >= cutoff).astype(int)
    tp, tn = int(((yhat == 1) & (y == 1)).sum()), int(((yhat == 0) & (y == 0)).sum())
    fp, fn = int(((yhat == 1) & (y == 0)).sum()), int(((yhat == 0) & (y == 1)).sum())
    n = len(y)
    acc = (tp + tn) / n
    maj = max(y.mean(), 1 - y.mean())
    table = pd.DataFrame([[tp, fn], [fp, tn]], index=["Actual 1", "Actual 0"], columns=["Predicted 1", "Predicted 0"])
    return {"table": table, "accuracy": acc, "sensitivity": tp / max(tp + fn, 1), "specificity": tn / max(tn + fp, 1),
            "ppv": tp / max(tp + fp, 1), "npv": tn / max(tn + fn, 1), "count_r2": acc,
            "adj_count_r2": (acc - maj) / (1 - maj) if maj < 1 else np.nan, "cutoff": cutoff}


def roc(y, p) -> dict:
    """ROC curve, AUC (Mann-Whitney) with Hanley-McNeil (1982) standard error, Youden-optimal cut-off."""
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    thr = np.r_[np.inf, np.unique(p)[::-1]]
    tpr = np.array([((p >= t) & (y == 1)).sum() for t in thr]) / max(y.sum(), 1)
    fpr = np.array([((p >= t) & (y == 0)).sum() for t in thr]) / max((1 - y).sum(), 1)
    n1, n0 = y.sum(), (1 - y).sum()
    ranks = stats.rankdata(p)
    auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    q1, q2 = auc / (2 - auc), 2 * auc ** 2 / (1 + auc)
    se = np.sqrt((auc * (1 - auc) + (n1 - 1) * (q1 - auc ** 2) + (n0 - 1) * (q2 - auc ** 2)) / (n1 * n0))
    j = np.argmax(tpr - fpr)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thr, "auc": float(auc), "se": float(se),
            "ci": (float(auc - 1.96 * se), float(min(1, auc + 1.96 * se))), "youden_cut": float(thr[j]),
            "youden_sens": float(tpr[j]), "youden_spec": float(1 - fpr[j])}


def hosmer_lemeshow(y, p, g: int = 10) -> dict:
    """Hosmer-Lemeshow (1980) goodness-of-fit test with g groups of predicted risk (χ², g − 2 df)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    q = pd.qcut(pd.Series(p).rank(method="first"), g, labels=False)
    tab = pd.DataFrame({"g": q, "y": y, "p": p}).groupby("g").agg(n=("y", "size"), obs=("y", "sum"), exp=("p", "sum"))
    hl = float(np.sum((tab.obs - tab.exp) ** 2 / (tab.exp * (1 - tab.exp / tab.n))))
    return {"stat": hl, "df": g - 2, "p": float(stats.chi2.sf(hl, g - 2)), "table": tab.reset_index()}


def link_test(r: rg.RegResult) -> dict:
    """Pregibon (1980) link test: refit on the linear predictor and its square; a significant square
    term signals a mis-specified link or omitted non-linearity."""
    X = r.design.X.to_numpy(float)
    xb = X @ r.params.to_numpy()[: X.shape[1]]
    Z = np.c_[np.ones(len(xb)), xb, xb ** 2]
    mod = sm.Logit if r.method == "logit" else sm.Probit if r.method == "probit" else None
    if mod is None:
        res = sm.GLM(r.y, Z, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
    else:
        res = mod(r.y, Z).fit(disp=0)
    return {"b_hatsq": float(res.params[2]), "se": float(res.bse[2]), "z": float(res.tvalues[2]),
            "p": float(res.pvalues[2])}


# ==========================================================
# Multinomial logit
# ==========================================================
def fit_mnlogit(df: pd.DataFrame, spec: rg.ModelSpec, base: str | None = None, cov_type: str = "nonrobust",
                name: str = "") -> rg.RegResult:
    design = _design(df, spec, y_raw=True)
    yraw = design.y.astype(str)
    cats = sorted(yraw.unique().tolist())
    if base in cats:
        cats.remove(base)
        cats = [base] + cats
    if len(cats) < 3:
        raise ValueError("Multinomial logit needs at least three outcome categories (use logit for two).")
    codes = yraw.map({c: i for i, c in enumerate(cats)}).to_numpy()
    X = design.X
    n, k = X.shape
    kw = {"cov_type": "HC0"} if cov_type in ("HC0", "robust") else {}
    res = sm.MNLogit(codes, X).fit(disp=0, maxiter=300, method="newton", **kw)
    J = len(cats)
    P = np.asarray(res.params)              # k × (J−1)
    theta = P.T.ravel()                      # outcome-major order
    names = [f"{cats[j + 1]}: {c}" for j in range(J - 1) for c in X.columns]
    Craw = np.asarray(res.cov_params())      # statsmodels orders outcome-major as well
    def pred(Xm, th, k=k, J=J):
        B = th.reshape(J - 1, k).T
        U = np.c_[np.zeros(len(Xm)), Xm @ B]
        U -= U.max(axis=1, keepdims=True)
        E = np.exp(U)
        return E / E.sum(axis=1, keepdims=True)
    probs = pred(X.to_numpy(float), theta)
    r = _result(name or "Multinomial logit", "mnlogit", design, names, theta, Craw, res.llf, n, codes,
                probs.max(axis=1), k * (J - 1) + 1, pred=pred, outputs=cats, llnull=res.llnull,
                cov_type="HC0" if kw else "nonrobust",
                extra={"family": "multinomial", "categories": cats, "sm": res, "probs": probs, "codes": codes})
    r.extra["rrr"] = pd.DataFrame({"Term": names, "RRR": np.exp(theta), "CI low": np.exp(r.ci["low"].values),
                                   "CI high": np.exp(r.ci["high"].values), "p-value": r.pvalues.values})
    pred_cls = probs.argmax(axis=1)
    r.extra["classification"] = pd.crosstab(pd.Series([cats[i] for i in codes], name="Actual"),
                                             pd.Series([cats[i] for i in pred_cls], name="Predicted"))
    r.extra["accuracy"] = float(np.mean(pred_cls == codes))
    return r


def iia_test(df: pd.DataFrame, r: rg.RegResult) -> pd.DataFrame:
    """Hausman-McFadden (1984) test of independence of irrelevant alternatives: drop each non-base outcome,
    re-estimate, and contrast the coefficients of the remaining outcomes."""
    cats = r.extra["categories"]
    codes = r.extra["codes"]
    X = r.design.X.to_numpy(float)
    k = X.shape[1]
    full = r.extra["sm"]
    rows = []
    for j in range(1, len(cats)):
        keep = codes != j
        kept = [c for c in range(len(cats)) if c != j]
        new_codes = pd.Series(codes[keep]).map({c: i for i, c in enumerate(kept)}).to_numpy()
        try:
            rr = sm.MNLogit(new_codes, X[keep]).fit(disp=0, maxiter=300, method="newton")
        except Exception:  # noqa: BLE001
            continue
        # common parameters: outcomes kept except base, excluding constants (Hausman-McFadden use slopes)
        slopes = [i for i, c in enumerate(r.design.X.columns) if c != "const"]
        idx_full, idx_r = [], []
        for pos_r, c in enumerate(kept[1:]):
            pos_f = c - 1
            idx_full += [pos_f * k + s for s in slopes]
            idx_r += [pos_r * k + s for s in slopes]
        bf = np.asarray(full.params).T.ravel()[idx_full]
        br = np.asarray(rr.params).T.ravel()[idx_r]
        Vf = np.asarray(full.cov_params())[np.ix_(idx_full, idx_full)]
        Vr = np.asarray(rr.cov_params())[np.ix_(idx_r, idx_r)]
        d = br - bf
        H = float(d @ np.linalg.pinv(Vr - Vf) @ d)
        dfh = len(d)
        rows.append({"Omitted outcome": cats[j], "χ²": H, "df": dfh,
                     "p-value": float(stats.chi2.sf(H, dfh)) if H >= 0 else np.nan,
                     "Verdict": "IIA holds" if H < 0 or stats.chi2.sf(H, dfh) >= 0.05 else "IIA rejected"})
    return pd.DataFrame(rows)


# ==========================================================
# Ordered logit / probit
# ==========================================================
def fit_ordered(df: pd.DataFrame, spec: rg.ModelSpec, distr: str = "logit", order: list | None = None,
                name: str = "") -> rg.RegResult:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    design = _design(df, spec, y_raw=True)
    yraw = design.y
    if order is None:
        num = pd.to_numeric(yraw, errors="coerce")
        cats = sorted(yraw.unique().tolist(), key=lambda v: float(v)) if num.notna().all() else sorted(yraw.astype(str).unique())
    else:
        cats = list(order)
    codes = yraw.map({c: i for i, c in enumerate(cats)}).to_numpy()
    if np.isnan(codes.astype(float)).any():
        raise ValueError("Some outcome values are not in the category order.")
    X = design.X.drop(columns=[c for c in ("const",) if c in design.X.columns])
    n, k = X.shape
    J = len(cats)
    res = OrderedModel(codes.astype(int), X, distr=distr).fit(method="bfgs", disp=0, maxiter=2000)
    raw = np.asarray(res.params)
    beta = raw[:k]
    th_raw = raw[k:]
    cuts = np.r_[th_raw[0], th_raw[0] + np.cumsum(np.exp(th_raw[1:]))]
    # delta method: ∂cut_j/∂θ_raw
    Jm = np.zeros((J - 1, J - 1))
    Jm[:, 0] = 1
    for j in range(1, J - 1):
        Jm[j:, j] = np.exp(th_raw[j])
    T = np.eye(k + J - 1)
    T[k:, k:] = Jm
    cov = T @ np.asarray(res.cov_params()) @ T.T
    theta = np.r_[beta, cuts]
    names = list(X.columns) + [f"cut{j + 1} ({cats[j]}|{cats[j + 1]})" for j in range(J - 1)]
    F = special.expit if distr == "logit" else stats.norm.cdf

    def pred(Xm, th, k=k, J=J, F=F):
        Xm = Xm[:, 1:] if Xm.shape[1] == k + 1 else Xm   # design rows include the constant column
        xb = Xm @ th[:k]
        c = th[k:]
        Fc = F(c[None, :] - xb[:, None])
        Fc = np.c_[np.zeros(len(xb)), Fc, np.ones(len(xb))]
        return np.diff(Fc, axis=1)
    Xfull = design.X.to_numpy(float)
    probs = pred(Xfull, theta)
    ll0 = float(np.sum(np.log(np.bincount(codes.astype(int), minlength=J)[codes.astype(int)] / n)))
    r = _result(name or f"Ordered {distr}", f"o{distr}", design, names, theta, cov, res.llf, n, codes,
                probs.max(axis=1), k, pred=pred, outputs=[str(c) for c in cats], llnull=ll0,
                extra={"family": "ordered", "categories": cats, "distr": distr, "probs": probs, "codes": codes,
                       "k_beta": k})
    r.extra["stats"]["LR χ²"] = 2 * (res.llf - ll0)
    r.extra["stats"]["Prob > χ²"] = float(stats.chi2.sf(2 * (res.llf - ll0), k))
    r.fvalue, r.f_pvalue = r.extra["stats"]["LR χ²"], r.extra["stats"]["Prob > χ²"]
    pc = probs.argmax(axis=1)
    r.extra["accuracy"] = float(np.mean(pc == codes))
    if distr == "logit":
        r.extra["odds_ratios"] = pd.DataFrame({"Variable": names[:k], "Odds ratio": np.exp(beta),
                                               "CI low": np.exp(r.ci["low"].values[:k]),
                                               "CI high": np.exp(r.ci["high"].values[:k]), "p-value": r.pvalues.values[:k]})
    return r


def brant_test(r: rg.RegResult) -> pd.DataFrame:
    """Brant (1990) test of the parallel-regression (proportional-odds) assumption of the ordered logit:
    J − 1 binary logits for y > j, joint covariance of the stacked estimates, Wald test that slopes are equal."""
    codes = r.extra["codes"].astype(int)
    k = r.extra["k_beta"]
    X = r.design.X.drop(columns=[c for c in ("const",) if c in r.design.X.columns]).to_numpy(float)
    names = r.names[:k]
    Xc = np.c_[np.ones(len(X)), X]
    J = len(r.extra["categories"])
    bets, pis, invs = [], [], []
    for j in range(J - 1):
        yj = (codes > j).astype(float)
        res = sm.Logit(yj, Xc).fit(disp=0, maxiter=300)
        b = np.asarray(res.params)
        p = special.expit(Xc @ b)
        bets.append(b)
        pis.append(p)
        invs.append(np.linalg.inv(Xc.T @ (Xc * (p * (1 - p))[:, None])))
    m = J - 1
    K1 = k + 1
    Vall = np.zeros((m * K1, m * K1))
    for i in range(m):
        for j in range(i, m):
            w = pis[j] - pis[i] * pis[j]          # for i ≤ j (P(y>j) ≤ P(y>i))
            Vij = invs[i] @ (Xc.T @ (Xc * w[:, None])) @ invs[j]
            Vall[i * K1:(i + 1) * K1, j * K1:(j + 1) * K1] = Vij
            Vall[j * K1:(j + 1) * K1, i * K1:(i + 1) * K1] = Vij.T
    # slopes only
    sl = [i * K1 + s for i in range(m) for s in range(1, K1)]
    B = np.concatenate([b[1:] for b in bets])
    V = Vall[np.ix_(sl, sl)]
    D = np.zeros(((m - 1) * k, m * k))
    for i in range(m - 1):
        D[i * k:(i + 1) * k, 0:k] = np.eye(k)
        D[i * k:(i + 1) * k, (i + 1) * k:(i + 2) * k] = -np.eye(k)
    W = float((D @ B) @ np.linalg.pinv(D @ V @ D.T) @ (D @ B))
    rows = [{"Variable": "All (omnibus)", "χ²": W, "df": (m - 1) * k, "p-value": float(stats.chi2.sf(W, (m - 1) * k))}]
    for v in range(k):
        Dv = np.zeros((m - 1, m * k))
        for i in range(m - 1):
            Dv[i, v] = 1
            Dv[i, (i + 1) * k + v] = -1
        Wv = float((Dv @ B) @ np.linalg.pinv(Dv @ V @ Dv.T) @ (Dv @ B))
        rows.append({"Variable": names[v], "χ²": Wv, "df": m - 1, "p-value": float(stats.chi2.sf(Wv, m - 1))})
    out = pd.DataFrame(rows)
    out["Parallel lines"] = np.where(out["p-value"] >= 0.05, "holds", "violated")
    return out


# ==========================================================
# Censored, truncated and selection models (maximum likelihood)
# ==========================================================
class _GLL(GenericLikelihoodModel):
    def __init__(self, endog, exog, ll, **kw):
        self._ll = ll
        super().__init__(endog, exog, **kw)

    def loglikeobs(self, params):
        return self._ll(params)


def _mle(ll, start, names, n, method="bfgs", cov_type="nonrobust", groups=None):
    endog = np.zeros(n)
    exog = np.ones((n, 1))
    mod = _GLL(endog, exog, ll)
    mod.exog_names[:] = []
    mod.data.xnames = names
    mod.nparams = len(start)
    mod.df_model = len(start) - 1
    mod.df_resid = n - len(start)
    kw = {}
    if cov_type in ("HC0", "robust"):
        kw = {"cov_type": "HC0"}
    elif cov_type == "cluster":
        kw = {"cov_type": "cluster", "cov_kwds": {"groups": groups}}
    res = mod.fit(start_params=np.asarray(start, float), method=method, maxiter=5000, disp=0, **kw)
    res2 = mod.fit(start_params=res.params, method="newton", maxiter=100, disp=0, **kw) if method != "newton" else res
    if np.isfinite(res2.llf) and res2.llf >= res.llf - 1e-8:
        res = res2
    return res


def _lnPhi(z):
    return special.log_ndtr(z)


def fit_tobit(df: pd.DataFrame, spec: rg.ModelSpec, left: float | None = 0.0, right: float | None = None,
              cov_type: str = "nonrobust", cov_opts: dict | None = None, name: str = "") -> rg.RegResult:
    """Tobit (1958) censored regression; McDonald & Moffitt (1980) decomposition of effects."""
    design = _design(df, spec)
    X = design.X.to_numpy(float)
    y = design.y.to_numpy(float)
    n, k = X.shape
    lo = (y <= left) if left is not None else np.zeros(n, bool)
    hi = (y >= right) if right is not None else np.zeros(n, bool)
    mid = ~(lo | hi)
    if lo.sum() + hi.sum() == 0:
        raise ValueError("No censored observations at the chosen limit(s) — use OLS.")

    def ll(th):
        b, s = th[:k], np.exp(th[k])
        xb = X @ b
        out = np.empty(n)
        out[mid] = stats.norm.logpdf((y[mid] - xb[mid]) / s) - np.log(s)
        if left is not None:
            out[lo] = _lnPhi((left - xb[lo]) / s)
        if right is not None:
            out[hi] = _lnPhi((xb[hi] - right) / s)
        return out
    b0 = np.linalg.lstsq(X[mid], y[mid], rcond=None)[0]
    s0 = np.log(np.std(y[mid] - X[mid] @ b0) + 1e-6)
    groups = _groups(df, design, cov_type, cov_opts)
    res = _mle(ll, np.r_[b0, s0], list(design.X.columns) + ["ln σ"], n, cov_type=cov_type, groups=groups)
    th = np.asarray(res.params)
    cov = np.asarray(res.cov_params())
    # report σ instead of ln σ
    T = np.eye(k + 1)
    T[k, k] = np.exp(th[k])
    theta = np.r_[th[:k], np.exp(th[k])]
    covs = T @ cov @ T.T
    a = left if left is not None else -np.inf
    names = list(design.X.columns) + ["sigma"]

    def pred(Xm, t, k=k, a=a):
        xb = Xm @ t[:k]
        s = t[k]
        z = (xb - a) / s
        Pz = stats.norm.cdf(z)
        lam = np.exp(stats.norm.logpdf(z) - _lnPhi(z))
        e_cond = xb + s * lam
        e_unc = Pz * e_cond + (1 - Pz) * (a if np.isfinite(a) else 0)
        return np.c_[xb, Pz, e_cond, e_unc]
    ll0 = _tobit_null(y, left, right)
    r = _result(name or "Tobit", "tobit", design, names, theta, covs, res.llf, n, y, pred(X, theta)[:, 3], k,
                pred=pred, outputs=["E[y*] (latent)", "P(uncensored)", "E[y | y > limit]", "E[y] (unconditional)"],
                llnull=ll0, cov_type=cov_type,
                extra={"family": "censored", "left": left, "right": right, "n_left": int(lo.sum()),
                       "n_right": int(hi.sum())})
    r.notes.append(f"{int(lo.sum())} left-censored, {int(hi.sum())} right-censored and {int(mid.sum())} uncensored observations.")
    return r


def _tobit_null(y, left, right):
    n = len(y)
    lo = (y <= left) if left is not None else np.zeros(n, bool)
    hi = (y >= right) if right is not None else np.zeros(n, bool)
    mid = ~(lo | hi)

    def nll(th):
        m, s = th[0], np.exp(th[1])
        v = np.sum(stats.norm.logpdf((y[mid] - m) / s) - np.log(s))
        if left is not None:
            v += np.sum(_lnPhi((left - m) / s)) if lo.any() else 0
        if right is not None:
            v += np.sum(_lnPhi((m - right) / s)) if hi.any() else 0
        return -v
    from scipy.optimize import minimize
    o = minimize(nll, [y.mean(), np.log(y.std() + 1e-6)], method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 5000})
    return -o.fun


def _groups(df, design, cov_type, cov_opts):
    if cov_type != "cluster":
        return None
    g = (cov_opts or {}).get("groups")
    return pd.factorize(df.loc[design.index, g].astype(str) if isinstance(g, str) else pd.Series(g))[0]


def fit_truncreg(df: pd.DataFrame, spec: rg.ModelSpec, lower: float | None = 0.0, upper: float | None = None,
                 name: str = "") -> rg.RegResult:
    """Truncated normal regression (the sample only contains lower < y < upper)."""
    design = _design(df, spec)
    keep = np.ones(len(design.y), bool)
    if lower is not None:
        keep &= design.y.to_numpy(float) > lower
    if upper is not None:
        keep &= design.y.to_numpy(float) < upper
    d = df.loc[design.index[keep]]
    design = _design(d, spec)
    X = design.X.to_numpy(float)
    y = design.y.to_numpy(float)
    n, k = X.shape

    def ll(th):
        b, s = th[:k], np.exp(th[k])
        xb = X @ b
        out = stats.norm.logpdf((y - xb) / s) - np.log(s)
        hi = stats.norm.cdf((upper - xb) / s) if upper is not None else 1.0
        lo = stats.norm.cdf((lower - xb) / s) if lower is not None else 0.0
        return out - np.log(np.clip(hi - lo, 1e-300, None))
    b0 = np.linalg.lstsq(X, y, rcond=None)[0]
    res = _mle(ll, np.r_[b0, np.log(np.std(y - X @ b0))], list(design.X.columns) + ["ln σ"], n)
    th = np.asarray(res.params)
    T = np.eye(k + 1)
    T[k, k] = np.exp(th[k])
    theta = np.r_[th[:k], np.exp(th[k])]
    r = _result(name or "Truncated regression", "truncreg", design, list(design.X.columns) + ["sigma"], theta,
                T @ np.asarray(res.cov_params()) @ T.T, res.llf, n, y, X @ theta[:k], k,
                pred=lambda Xm, t, k=k: Xm @ t[:k], outputs=None, cov_type="nonrobust",
                extra={"family": "censored", "lower": lower, "upper": upper})
    r.notes.append(f"Estimation uses the {n} observations inside the truncation limits.")
    return r


def fit_fractional(df: pd.DataFrame, spec: rg.ModelSpec, link: str = "logit", cov_type: str = "HC0",
                   cov_opts: dict | None = None, name: str = "") -> rg.RegResult:
    """Papke & Wooldridge (1996) fractional response: Bernoulli quasi-ML with a logit / probit mean and
    robust standard errors (the outcome may take any value in [0, 1], including the bounds)."""
    design = _design(df, spec)
    y = design.y.to_numpy(float)
    if (y < 0).any() or (y > 1).any():
        raise ValueError("The fractional outcome must lie in [0, 1] (proportions, shares, rates).")
    X = design.X
    n, k = X.shape
    lk = sm.families.links.Logit() if link == "logit" else sm.families.links.Probit()
    kw = {"cov_type": "HC0"} if cov_type in ("HC0", "robust", "nonrobust") else \
        {"cov_type": "cluster", "cov_kwds": {"groups": _groups(df, design, cov_type, cov_opts)}}
    res = sm.GLM(y, X, family=sm.families.Binomial(link=lk)).fit(**kw)
    F = _cdf(link)
    theta = np.asarray(res.params)
    mu = F(X.to_numpy(float) @ theta)
    ll = float(np.sum(y * np.log(np.clip(mu, 1e-300, 1)) + (1 - y) * np.log(np.clip(1 - mu, 1e-300, 1))))
    ybar = y.mean()
    ll0 = float(np.sum(y * np.log(ybar) + (1 - y) * np.log(1 - ybar)))
    r = _result(name or f"Fractional {link}", "frac", design, list(X.columns), theta, np.asarray(res.cov_params()),
                ll, n, y, mu, k, pred=lambda Xm, t, F=F: F(Xm @ t[: Xm.shape[1]]), llnull=ll0,
                cov_type=kw["cov_type"], extra={"family": "fractional", "link": link})
    r.extra["stats"]["Quasi-LL"] = r.extra["stats"].pop("Log-likelihood")
    r.notes.append("Quasi-maximum likelihood: the Bernoulli log-likelihood is used for estimation only; inference "
                   "relies on robust (sandwich) standard errors.")
    return r


# ---------------------------------------------------------------- Heckman
def fit_heckman(df: pd.DataFrame, outcome: rg.ModelSpec, selection: rg.ModelSpec, method: str = "twostep",
                name: str = "") -> rg.RegResult:
    """Heckman (1979) sample-selection model. `selection.y` is the 0/1 participation indicator; the outcome
    is used only where selection == 1. Two-step uses the Heckman / Greene corrected covariance; ML estimates
    (β, γ, atanh ρ, ln σ) jointly."""
    sel = _design(df, selection)
    s = sel.y.to_numpy(float)
    if not set(np.unique(s)) <= {0.0, 1.0}:
        raise ValueError("The selection variable must be 0 / 1.")
    Zdf = sel.X
    Z = Zdf.to_numpy(float)
    pro = sm.Probit(s, Z).fit(disp=0, maxiter=200)
    g = np.asarray(pro.params)
    zg = Z @ g
    lam_all = np.exp(stats.norm.logpdf(zg) - _lnPhi(zg))
    selected_idx = sel.index[s == 1]
    od = _design(df.loc[selected_idx], outcome, allow_logy=True)
    Xdf = od.X
    X = Xdf.to_numpy(float)
    y = od.y.to_numpy(float)
    pos = pd.Series(np.arange(len(sel.index)), index=sel.index).loc[od.index].to_numpy()
    lam = lam_all[pos]
    n1, k = X.shape
    N = len(s)
    excl = [c for c in Zdf.columns if c not in Xdf.columns]
    notes = []
    if not excl:
        notes.append("No exclusion restriction: the selection equation has no variable absent from the outcome "
                     "equation, so identification relies only on the non-linearity of the inverse Mills ratio.")
    if method == "twostep":
        Xs = np.c_[X, lam]
        bl = np.linalg.lstsq(Xs, y, rcond=None)[0]
        e = y - Xs @ bl
        bL = bl[-1]
        delta = lam * (lam + zg[pos])
        s2 = e @ e / n1 + delta.mean() * bL ** 2
        rho = bL / np.sqrt(s2)
        rho_c = np.clip(rho, -1, 1)
        XtXi = np.linalg.inv(Xs.T @ Xs)
        D = delta
        Q = rho_c ** 2 * (Xs.T * D) @ Z[pos] @ np.asarray(pro.cov_params()) @ Z[pos].T @ (Xs * D[:, None])
        V = s2 * XtXi @ (Xs.T @ (Xs * (1 - rho_c ** 2 * D)[:, None]) + Q) @ XtXi
        theta = np.r_[bl, g]
        names = [f"{c}" for c in Xdf.columns] + ["lambda (IMR)"] + [f"select: {c}" for c in Zdf.columns]
        cov = np.zeros((len(theta), len(theta)))
        cov[:k + 1, :k + 1] = V
        cov[k + 1:, k + 1:] = np.asarray(pro.cov_params())
        llf = np.nan
        sigma = float(np.sqrt(s2))
        extra = {"rho": float(rho), "sigma": sigma, "lambda": float(bL), "lambda_se": float(np.sqrt(V[-1, -1]))}
        notes.append(f"ρ̂ = {rho:.3f}, σ̂ = {sigma:.3f}; test of selection bias: t(λ) = {bL / np.sqrt(V[-1, -1]):.3f}, "
                     f"p = {2 * stats.norm.sf(abs(bL / np.sqrt(V[-1, -1]))):.4f}.")
    else:
        Xfull = np.zeros((N, k))
        Xfull[pos] = X
        yfull = np.zeros(N)
        yfull[pos] = y
        kz = Z.shape[1]

        def ll(th):
            b, gg, at, ls = th[:k], th[k:k + kz], th[k + kz], th[k + kz + 1]
            rho, sg = np.tanh(at), np.exp(ls)
            zz = Z @ gg
            out = np.empty(N)
            o = s == 0
            out[o] = _lnPhi(-zz[o])
            u = (yfull[~o] - Xfull[~o] @ b) / sg
            out[~o] = _lnPhi((zz[~o] + rho * u) / np.sqrt(1 - rho ** 2)) + stats.norm.logpdf(u) - ls
            return out
        two = fit_heckman(df, outcome, selection, "twostep")
        r0 = np.clip(two.extra["rho"], -0.9, 0.9)
        start = np.r_[two.params.values[:k], g, np.arctanh(r0), np.log(two.extra["sigma"])]
        res = _mle(ll, start, list(Xdf.columns) + [f"select: {c}" for c in Zdf.columns] + ["atanh ρ", "ln σ"], N)
        th = np.asarray(res.params)
        C = np.asarray(res.cov_params())
        at, ls = th[-2], th[-1]
        rho, sigma = float(np.tanh(at)), float(np.exp(ls))
        lam_ml = rho * sigma
        T = np.eye(len(th))
        T[-2, -2] = 1 - np.tanh(at) ** 2
        T[-1, -1] = np.exp(ls)
        theta = np.r_[th[:-2], rho, sigma]
        cov = T @ C @ T.T
        names = [f"{c}" for c in Xdf.columns] + [f"select: {c}" for c in Zdf.columns] + ["rho", "sigma"]
        llf = res.llf
        # LR test of independent equations (ρ = 0): probit LL + OLS LL
        e_ols = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
        ll_ols = -n1 / 2 * (np.log(2 * np.pi) + np.log(e_ols @ e_ols / n1) + 1)
        LR = 2 * (llf - (pro.llf + ll_ols))
        extra = {"rho": rho, "sigma": sigma, "lambda": lam_ml, "LR_rho": float(LR), "p_LR_rho": float(stats.chi2.sf(LR, 1)),
                 "Wald_rho_p": float(2 * stats.norm.sf(abs(at / np.sqrt(C[-2, -2]))))}
        notes.append(f"ρ̂ = {rho:.3f}, σ̂ = {sigma:.3f}, λ = ρσ = {lam_ml:.3f}. LR test of independent equations "
                     f"(ρ = 0): χ²(1) = {LR:.3f}, p = {stats.chi2.sf(LR, 1):.4f}.")
    kb = k

    def pred(Xm, t, kb=kb):
        return Xm @ t[:kb]
    r = _result(name or CENSORED["heckman2" if method == "twostep" else "heckman_ml"][0],
                "heckman2" if method == "twostep" else "heckman_ml", od, names, theta, cov, llf if np.isfinite(llf) else 0.0,
                n1, y, X @ theta[:k], k, pred=pred, outputs=None, notes=notes,
                extra={"family": "selection", "N_total": N, "N_selected": int(n1), "N_censored": int(N - n1),
                       "probit": pro, "selection_spec": selection, **extra})
    if not np.isfinite(llf):
        r.llf = r.aic = r.bic = r.hqic = np.nan
        r.extra["stats"] = {}
    r.extra["stats"].update({"ρ": extra["rho"], "σ": extra["sigma"], "λ": extra["lambda"],
                             "Selected / total": f"{n1} / {N}"})
    return r


# ---------------------------------------------------------------- hurdles
def fit_hurdle(df: pd.DataFrame, outcome: rg.ModelSpec, participation: rg.ModelSpec | None = None,
               kind: str = "cragg", name: str = "") -> rg.RegResult:
    """Cragg (1971) truncated-normal hurdle (`cragg`: zero ⇔ fails the participation hurdle; log-likelihood =
    probit + truncated regression) or Jones (1989) independent double hurdle (`dhurdle`: zeros may come
    from either hurdle). `participation` regressors default to the outcome regressors."""
    od = _design(df, outcome)
    participation = participation or rg.ModelSpec(y=outcome.y, x=outcome.x, categorical=outcome.categorical)
    pdz = _design(df, rg.ModelSpec(**{**participation.__dict__, "y": outcome.y}))
    idx = od.index.intersection(pdz.index)
    od, pdz = _design(df.loc[idx], outcome), _design(df.loc[idx], rg.ModelSpec(**{**participation.__dict__, "y": outcome.y}))
    X, Z = od.X.to_numpy(float), pdz.X.to_numpy(float)
    y = od.y.to_numpy(float)
    if (y < 0).any():
        raise ValueError("Hurdle models need a non-negative outcome with a mass point at zero.")
    pos = y > 0
    n, k = X.shape
    kz = Z.shape[1]

    def ll(th):
        gg, b, s = th[:kz], th[kz:kz + k], np.exp(th[-1])
        zg, xb = Z @ gg, X @ b
        out = np.empty(n)
        if kind == "cragg":
            out[~pos] = _lnPhi(-zg[~pos])
            out[pos] = _lnPhi(zg[pos]) + stats.norm.logpdf((y[pos] - xb[pos]) / s) - np.log(s) - _lnPhi(xb[pos] / s)
        else:
            out[~pos] = np.log1p(-np.clip(stats.norm.cdf(zg[~pos]) * stats.norm.cdf(xb[~pos] / s), 0, 1 - 1e-15))
            out[pos] = _lnPhi(zg[pos]) + stats.norm.logpdf((y[pos] - xb[pos]) / s) - np.log(s)
        return out
    pr = sm.Probit(pos.astype(float), Z).fit(disp=0)
    b0 = np.linalg.lstsq(X[pos], y[pos], rcond=None)[0]
    start = np.r_[pr.params, b0, np.log(np.std(y[pos] - X[pos] @ b0))]
    names = [f"hurdle: {c}" for c in pdz.X.columns] + list(od.X.columns) + ["ln σ"]
    res = _mle(ll, start, names, n)
    th = np.asarray(res.params)
    T = np.eye(len(th))
    T[-1, -1] = np.exp(th[-1])
    theta = np.r_[th[:-1], np.exp(th[-1])]
    names[-1] = "sigma"

    def pred(Xm, t, kz=kz, k=k):
        # the same original variables feed both equations: design rows are rebuilt from the outcome design,
        # so for prediction we use the outcome regressors for the level and the participation index
        xb = Xm @ t[kz:kz + k]
        s = t[-1]
        zg = Xm @ t[:kz] if Xm.shape[1] == kz else np.full(len(Xm), np.nan)
        p_part = stats.norm.cdf(zg)
        z = xb / s
        cond = xb + s * np.exp(stats.norm.logpdf(z) - _lnPhi(z))
        if kind == "cragg":
            return np.c_[p_part, cond, p_part * cond]
        return np.c_[p_part * stats.norm.cdf(z), cond, p_part * stats.norm.cdf(z) * cond]
    same = list(pdz.X.columns) == list(od.X.columns)
    r = _result(name or CENSORED[kind][0], kind, od, names, theta, T @ np.asarray(res.cov_params()) @ T.T, res.llf, n,
                y, X @ theta[kz:kz + k], k, pred=pred if same else None,
                outputs=["P(y > 0)", "E[y | y > 0]", "E[y]"], cov_type="nonrobust",
                extra={"family": "hurdle", "kind": kind, "n_zero": int((~pos).sum()), "same_regressors": same})
    if kind == "cragg":
        # Tobit nested in Cragg (Lin & Schmidt 1984 LR test) when regressors coincide
        if same:
            t = fit_tobit(df.loc[idx], outcome, left=0.0)
            LR = 2 * (res.llf - t.llf)
            dfl = kz
            r.extra["tobit_LR"] = (float(LR), dfl, float(stats.chi2.sf(LR, dfl)))
            r.notes.append(f"LR test Tobit vs Cragg (Lin & Schmidt 1984): χ²({dfl}) = {LR:.3f}, "
                           f"p = {stats.chi2.sf(LR, dfl):.4f}" + (" → Tobit's single-process assumption is rejected."
                                                                 if stats.chi2.sf(LR, dfl) < 0.05 else
                                                                 " → Tobit is not rejected."))
    if not same:
        r.notes.append("Participation and outcome regressors differ: average effects are not computed automatically.")
    return r
