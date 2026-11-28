# ==========================================
# Aetheris Research - Policy & impact evaluation (v1.11, Catalogue §1.20)
# Randomised trials (difference in means, Lin adjustment, balance, ITT / LATE), difference-in-differences (2×2, TWFE
# event study, Callaway & Sant'Anna ATT(g,t) with aggregations, Sun & Abraham), synthetic control (Abadie) and
# synthetic difference-in-differences (Arkhangelsky et al.), regression discontinuity (rdrobust / rddensity), selection
# on observables (propensity-score matching, IPW, AIPW) and IV / LATE. Conventions follow R estimatr, did, fixest,
# Synth, synthdid, MatchIt, WeightIt and ivreg, and Stata teffects.
# ==========================================

from __future__ import annotations

import contextlib
import io
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore")


def _star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def _row(name, b, se, df=None):
    t = b / se if se > 0 else np.nan
    p = 2 * (stats.t.sf(abs(t), df) if df else stats.norm.sf(abs(t)))
    q = stats.t.ppf(0.975, df) if df else 1.959963984540054
    return {"Term": name, "Estimate": float(b), "Std. Err.": float(se), "t / z": float(t), "p-value": float(p), "CI low": float(b - q * se),
            "CI high": float(b + q * se), "": _star(p)}


# ----------------------------------------------------------------------------------------------------------------------
# linear algebra helpers: OLS with HC0–HC3 or cluster-robust covariance (Stata / estimatr small-sample conventions)
# ----------------------------------------------------------------------------------------------------------------------
def ols(y, X, vcov: str = "HC2", cluster=None, w=None):
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    sw = np.ones(n) if w is None else np.sqrt(np.asarray(w, float))
    Xw, yw = X * sw[:, None], y * sw
    XtXi = np.linalg.pinv(Xw.T @ Xw)
    b = XtXi @ Xw.T @ yw
    e = yw - Xw @ b
    if cluster is not None:
        g = pd.factorize(np.asarray(cluster))[0]
        G = g.max() + 1
        S = np.zeros((G, k))
        np.add.at(S, g, Xw * e[:, None])
        V = XtXi @ (S.T @ S) @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k))
        dof = G - 1
    elif vcov == "iid":
        V = XtXi * (e @ e) / (n - k)
        dof = n - k
    else:
        h = np.einsum("ij,jk,ik->i", Xw, XtXi, Xw)
        om = {"HC0": e ** 2, "HC1": e ** 2 * n / (n - k), "HC2": e ** 2 / (1 - h), "HC3": e ** 2 / (1 - h) ** 2}[vcov]
        V = XtXi @ (Xw.T * om) @ Xw @ XtXi
        dof = n - k
    return b, V, e, dof


# ======================================================================================================================
# Randomised controlled trials
# ======================================================================================================================
def balance_table(df: pd.DataFrame, treat: str, covs: list, weights=None) -> pd.DataFrame:
    """Means by arm, difference, standardised mean difference (pooled SD as cobalt / MatchIt), variance ratio and a
    Welch t-test p-value (unweighted)."""
    rows = []
    D = df[treat].to_numpy() == 1
    w = np.ones(len(df)) if weights is None else np.asarray(weights, float)
    for c in covs:
        x = df[c].to_numpy(float)
        m1, m0 = np.average(x[D], weights=w[D]), np.average(x[~D], weights=w[~D])
        v1, v0 = x[D].var(ddof=1), x[~D].var(ddof=1)
        smd = (m1 - m0) / np.sqrt((v1 + v0) / 2) if v1 + v0 > 0 else 0.0
        p = stats.ttest_ind(x[D], x[~D], equal_var=False).pvalue if weights is None else np.nan
        rows.append({"Covariate": c, "Mean treated": m1, "Mean control": m0, "Difference": m1 - m0, "Std. mean diff.": smd,
                     "Variance ratio": v1 / v0 if v0 > 0 else np.nan, "p-value (Welch)": p, "Balanced (|SMD|<0.1)": abs(smd) < 0.1})
    return pd.DataFrame(rows)


def rct(df: pd.DataFrame, y: str, treat: str, covs: list | None = None, strata: str | None = None, cluster: str | None = None) -> dict:
    """Difference in means with Neyman (HC2) SEs, Lin (2013) fully interacted covariate adjustment (HC2, as
    estimatr::lm_lin), optional strata fixed effects and cluster-robust SEs; joint orthogonality F test."""
    covs = covs or []
    cols = [y, treat] + covs + ([strata] if strata else []) + ([cluster] if cluster else [])
    d = df[list(dict.fromkeys(cols))].dropna()
    yv, D = d[y].to_numpy(float), d[treat].to_numpy(float)
    n = len(d)
    cl = d[cluster].to_numpy() if cluster else None
    out = {"N": n, "N treated": int(D.sum()), "N control": int(n - D.sum())}
    rows = []
    X = np.column_stack([np.ones(n), D])
    b, V, _, dof = ols(yv, X, "HC2", cl)
    rows.append({"Estimator": "Difference in means", **_row(treat, b[1], np.sqrt(V[1, 1]), dof)})
    S = pd.get_dummies(d[strata], drop_first=True, dtype=float).to_numpy() if strata else np.empty((n, 0))
    if strata:
        b, V, _, dof = ols(yv, np.column_stack([X, S]), "HC2", cl)
        rows.append({"Estimator": "Strata fixed effects", **_row(treat, b[1], np.sqrt(V[1, 1]), dof)})
    if covs:
        Xc = d[covs].to_numpy(float)
        Xc = Xc - Xc.mean(0)
        b, V, _, dof = ols(yv, np.column_stack([X, Xc, S]), "HC2", cl)
        rows.append({"Estimator": "ANCOVA (additive)", **_row(treat, b[1], np.sqrt(V[1, 1]), dof)})
        b, V, _, dof = ols(yv, np.column_stack([X, Xc, Xc * D[:, None], S]), "HC2", cl)
        rows.append({"Estimator": "Lin (interacted)", **_row(treat, b[1], np.sqrt(V[1, 1]), dof)})
        bt = balance_table(d, treat, covs)
        bb, VV, _, _ = ols(D, np.column_stack([np.ones(n), d[covs].to_numpy(float)]), "HC2", cl)
        q = len(covs)
        F = float(bb[1:] @ np.linalg.solve(VV[1:, 1:], bb[1:]) / q)
        out["balance"] = bt
        out["joint F"] = {"F": F, "df1": q, "df2": n - q - 1, "p-value": float(stats.f.sf(F, q, n - q - 1))}
    out["table"] = pd.DataFrame(rows)
    out["means"] = d.groupby(treat)[y].agg(["mean", "std", "count"]).reset_index()
    return out


def rct_heterogeneity(df: pd.DataFrame, y: str, treat: str, by: str) -> pd.DataFrame:
    rows = []
    for g, s in df.groupby(by):
        if s[treat].nunique() < 2:
            continue
        r = rct(s, y, treat)
        rows.append({by: g, **{k: v for k, v in r["table"].iloc[0].items() if k != "Estimator"}, "N": r["N"]})
    out = pd.DataFrame(rows)
    d = df[[y, treat, by]].dropna()
    G = pd.get_dummies(d[by], drop_first=True, dtype=float).to_numpy()
    Dv = d[treat].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), Dv, G, G * Dv[:, None]])
    b, V, _, _ = ols(d[y], X, "HC2")
    j = slice(2 + G.shape[1], None)
    W = float(b[j] @ np.linalg.solve(V[j, j], b[j]))
    out.attrs["interaction test"] = {"Wald": W, "df": G.shape[1], "p-value": float(stats.chi2.sf(W, G.shape[1]))}
    return out


# ======================================================================================================================
# Instrumental variables / LATE
# ======================================================================================================================
def iv_2sls(df: pd.DataFrame, y: str, endog: str, instruments: list, exog: list | None = None, vcov: str = "HC1",
            cluster: str | None = None) -> dict:
    """Two-stage least squares (as ivreg / ivregress 2sls) with robust or cluster SEs, first-stage F (robust Wald / q),
    the reduced form and — for one binary instrument — the Wald (LATE) ratio. Complier share = first-stage coefficient."""
    exog = exog or []
    d = df[list(dict.fromkeys([y, endog] + instruments + exog + ([cluster] if cluster else [])))].dropna()
    n = len(d)
    W = np.column_stack([np.ones(n), d[exog].to_numpy(float)]) if exog else np.ones((n, 1))
    Z = np.column_stack([W, d[instruments].to_numpy(float)])
    Xe = d[endog].to_numpy(float)
    yv = d[y].to_numpy(float)
    X = np.column_stack([W, Xe])
    cl = d[cluster].to_numpy() if cluster else None
    # first stage
    b1, V1, _, dof1 = ols(Xe, Z, vcov, cl)
    q = len(instruments)
    j = slice(W.shape[1], None)
    F = float(b1[j] @ np.linalg.solve(V1[j, j], b1[j]) / q)
    # 2SLS
    PZ = Z @ np.linalg.pinv(Z.T @ Z) @ Z.T
    Xh = PZ @ X
    A = np.linalg.inv(Xh.T @ X)
    b = A @ Xh.T @ yv
    e = yv - X @ b
    k = X.shape[1]
    if cl is not None:
        g = pd.factorize(cl)[0]
        G = g.max() + 1
        S = np.zeros((G, k))
        np.add.at(S, g, Xh * e[:, None])
        V = A @ S.T @ S @ A.T * (G / (G - 1)) * ((n - 1) / (n - k))
        dof = G - 1
    elif vcov == "iid":
        V = A * (e @ e) / (n - k)
        dof = n - k
    else:
        adj = n / (n - k) if vcov == "HC1" else 1.0
        V = A @ (Xh.T * e ** 2) @ Xh @ A.T * adj
        dof = n - k
    names = ["const"] + exog + [endog]
    tab = pd.DataFrame([_row(nm, b[i], np.sqrt(V[i, i]), dof) for i, nm in enumerate(names)])
    rf, Vr, _, _ = ols(yv, Z, vcov, cl)
    bo, Vo, _, _ = ols(yv, X, vcov, cl)
    out = {"table": tab, "first stage": pd.DataFrame([_row(z, b1[W.shape[1] + i], np.sqrt(V1[W.shape[1] + i, W.shape[1] + i]), dof1) for i, z in enumerate(instruments)]),
           "first-stage F": F, "reduced form": pd.DataFrame([_row(z, rf[W.shape[1] + i], np.sqrt(Vr[W.shape[1] + i, W.shape[1] + i])) for i, z in enumerate(instruments)]),
           "OLS": _row(endog, bo[-1], np.sqrt(Vo[-1, -1])), "N": n, "late": float(b[-1]), "late se": float(np.sqrt(V[-1, -1])), "b": b, "V": V}
    if q == 1 and set(np.unique(d[instruments[0]])) <= {0, 1}:
        zz = d[instruments[0]].to_numpy()
        out["Wald"] = {"ITT (reduced form)": float(yv[zz == 1].mean() - yv[zz == 0].mean()),
                       "Complier share (first stage)": float(Xe[zz == 1].mean() - Xe[zz == 0].mean())}
        out["Wald"]["LATE = ITT / first stage"] = out["Wald"]["ITT (reduced form)"] / out["Wald"]["Complier share (first stage)"]
    if exog == [] and q >= 1:
        # Durbin-Wu-Hausman (control-function, robust)
        v = Xe - Z @ b1
        bc, Vc, _, _ = ols(yv, np.column_stack([X, v]), vcov, cl)
        tt = bc[-1] / np.sqrt(Vc[-1, -1])
        out["endogeneity test"] = {"t (control function)": float(tt), "p-value": float(2 * stats.norm.sf(abs(tt)))}
    return out


# ======================================================================================================================
# Difference-in-differences
# ======================================================================================================================
def did_2x2(df: pd.DataFrame, y: str, treat_group: str, post: str, cluster: str | None = None, covs: list | None = None) -> dict:
    covs = covs or []
    d = df[list(dict.fromkeys([y, treat_group, post] + covs + ([cluster] if cluster else [])))].dropna()
    G, P = d[treat_group].to_numpy(float), d[post].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), G, P, G * P] + ([d[covs].to_numpy(float)] if covs else []))
    b, V, _, dof = ols(d[y], X, "HC1", d[cluster].to_numpy() if cluster else None)
    names = ["const", "treated group", "post", "treated × post (DiD)"] + covs
    cell = d.groupby([treat_group, post])[y].mean().unstack()
    return {"table": pd.DataFrame([_row(nm, b[i], np.sqrt(V[i, i]), dof) for i, nm in enumerate(names)]), "cells": cell,
            "did": float(b[3]), "se": float(np.sqrt(V[3, 3])), "N": len(d)}


def _twoway_demean(A, unit_idx, time_idx, iters=200, tol=1e-12):
    """Alternating projections (exact in one pass for a balanced panel)."""
    A = np.array(A, float, copy=True)
    if A.ndim == 1:
        A = A[:, None]
    for _ in range(iters):
        old = A.copy()
        for idx in (unit_idx, time_idx):
            cnt = np.bincount(idx)
            m = np.zeros((cnt.size, A.shape[1]))
            np.add.at(m, idx, A)
            A -= (m / cnt[:, None])[idx]
        if np.max(np.abs(A - old)) < tol:
            break
    return A


def _fe_cluster(y, X, unit_idx, time_idx, cl, strict: bool = False, names=None):
    """Two-way FE OLS with cluster-robust SEs, fixest small-sample conventions (nested FE not counted). With `strict`,
    regressors absorbed by the fixed effects raise an informative error instead of being dropped."""
    Yd = _twoway_demean(y, unit_idx, time_idx)[:, 0]
    Xd = _twoway_demean(X, unit_idx, time_idx)
    keep = np.abs(Xd).sum(0) > 1e-10
    if strict:
        nrm = np.linalg.norm(Xd, axis=0)
        rank = np.linalg.matrix_rank(Xd / np.where(nrm > 0, nrm, 1))
        if not keep.all() or rank < Xd.shape[1]:
            bad = [n for n, k in zip(names or range(Xd.shape[1]), keep) if not k]
            raise ValueError("regressors are collinear or absorbed by the fixed effects" + (f": {', '.join(map(str, bad))}" if bad else ""))
    Xd = Xd[:, keep]
    n, k = Xd.shape
    XtXi = np.linalg.pinv(Xd.T @ Xd)
    b = XtXi @ Xd.T @ Yd
    e = Yd - Xd @ b
    g = pd.factorize(cl)[0]
    G = g.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, g, Xd * e[:, None])
    Kfe = len(np.unique(time_idx))       # time FE counted in full; unit FE nested in the unit cluster (fixest ssc)
    V = XtXi @ S.T @ S @ XtXi * (G / (G - 1)) * ((n - 1) / (n - k - Kfe))
    return b, V, keep, G - 1


def event_study_twfe(df: pd.DataFrame, unit: str, time: str, y: str, first_treated: str, leads: int = 5, lags: int = 5,
                     bin_ends: bool = True, ref: int = -1) -> pd.DataFrame:
    """Dynamic TWFE regression on relative-time dummies (never-treated coded 0 have none), reference period −1, end
    points binned. Cluster-robust SEs by unit. Biased under heterogeneous staggered effects — compare with CS / SA."""
    d = df[[unit, time, y, first_treated]].dropna().copy()
    g = d[first_treated].to_numpy()
    rel = np.where(g > 0, d[time].to_numpy() - g, np.nan)
    if bin_ends:
        rel = np.where(np.isnan(rel), rel, np.clip(rel, -leads, lags))
    ks = [k for k in range(-leads, lags + 1) if k != ref]
    X = np.column_stack([(rel == k).astype(float) for k in ks])
    ui, ti = pd.factorize(d[unit])[0], pd.factorize(d[time])[0]
    b, V, keep, dof = _fe_cluster(d[y].to_numpy(float), X, ui, ti, d[unit].to_numpy())
    ks = [k for k, kp in zip(ks, keep) if kp]
    rows = [{"Relative period": k, **_row(f"e={k}", b[i], np.sqrt(V[i, i]), dof)} for i, k in enumerate(ks)]
    rows.append({"Relative period": ref, "Term": f"e={ref}", "Estimate": 0.0, "Std. Err.": 0.0, "CI low": 0.0, "CI high": 0.0})
    return pd.DataFrame(rows).sort_values("Relative period").reset_index(drop=True)


def sun_abraham(df: pd.DataFrame, unit: str, time: str, y: str, first_treated: str, ref: int = -1) -> dict:
    """Interaction-weighted estimator (Sun & Abraham 2021) by cohort × relative-period saturation with two-way FE and
    never-treated units (0) as controls, aggregated per relative period by cohort shares (as fixest::sunab)."""
    d = df[[unit, time, y, first_treated]].dropna().copy()
    if (d[first_treated] == 0).sum() == 0:
        last = d[first_treated].max()
        d = d[d[time] < last]
        d.loc[d[first_treated] == last, first_treated] = 0
    g = d[first_treated].to_numpy()
    t = d[time].to_numpy()
    rel = t - g
    cells = sorted({(gg, r) for gg, r in zip(g[g > 0], rel[g > 0]) if r != ref})
    X = np.zeros((len(d), len(cells)))
    pos = {c: i for i, c in enumerate(cells)}
    for i, (gg, r) in enumerate(zip(g, rel)):
        if gg > 0 and r != ref:
            X[i, pos[(gg, r)]] = 1
    ui, ti = pd.factorize(d[unit])[0], pd.factorize(d[time])[0]
    b, V, keep, dof = _fe_cluster(d[y].to_numpy(float), X, ui, ti, d[unit].to_numpy())
    cells = [c for c, kp in zip(cells, keep) if kp]
    cnt = X[:, keep].sum(0)
    rows = []
    for r in sorted({c[1] for c in cells}):
        idx = [i for i, c in enumerate(cells) if c[1] == r]
        w = np.zeros(len(cells))
        w[idx] = cnt[idx] / cnt[idx].sum()
        est = w @ b
        rows.append({"Relative period": r, **_row(f"e={r}", est, np.sqrt(w @ V @ w), dof)})
    post = [i for i, c in enumerate(cells) if c[1] >= 0]
    w = np.zeros(len(cells))
    w[post] = cnt[post] / cnt[post].sum()
    att = _row("ATT (post, cohort-share weighted)", w @ b, np.sqrt(w @ V @ w), dof)
    cohort = pd.DataFrame([{"Cohort": c[0], "Relative period": c[1], "Estimate": b[i], "Std. Err.": np.sqrt(V[i, i])} for i, c in enumerate(cells)])
    return {"event": pd.DataFrame(rows), "att": att, "cohort": cohort}


def callaway_santanna(df: pd.DataFrame, unit: str, time: str, y: str, first_treated: str, control: str = "never",
                      base: str = "varying") -> dict:
    """ATT(g,t) of Callaway & Sant'Anna (2021) for balanced panels without covariates (unconditional parallel
    trends), never-treated or not-yet-treated controls, varying / universal base period; analytical influence-function
    SEs and the simple, dynamic (event-time), group and calendar aggregations including the estimated-weight term —
    as R did::att_gt + aggte (bstrap = FALSE)."""
    d = df[[unit, time, y, first_treated]].dropna()
    Y = d.pivot(index=unit, columns=time, values=y)
    if Y.isna().any().any():
        raise ValueError("Callaway-Sant'Anna here requires a balanced panel.")
    G = d.groupby(unit)[first_treated].first().reindex(Y.index).to_numpy()
    tl = np.array(sorted(Y.columns))
    Yv = Y.to_numpy(float)
    G = np.where(G > tl.max(), 0, G)
    n = len(G)
    glist = np.array(sorted(set(G[(G > tl.min())]) - {0}))
    att, inf, grp, per = [], [], [], []
    for gg in glist:
        gi = np.where(tl == gg)[0][0]
        for ti in range(1, len(tl)):
            tt = tl[ti]
            if base == "varying" and tt < gg:
                bi = ti - 1
            else:
                bi = gi - 1
            if bi < 0 or bi == ti:
                continue
            cur = max(tt, tl[bi])
            Dg = G == gg
            C = (G == 0) if control == "never" else ((G == 0) | ((G > cur) & (G != gg)))
            if C.sum() == 0:
                continue
            dY = Yv[:, ti] - Yv[:, bi]
            mg, mc = dY[Dg].mean(), dY[C].mean()
            psi = n * (Dg * (dY - mg) / Dg.sum() - C * (dY - mc) / C.sum())
            att.append(mg - mc)
            inf.append(psi)
            grp.append(gg)
            per.append(tt)
    att, grp, per = np.array(att), np.array(grp), np.array(per)
    IF = np.column_stack(inf)
    se = np.sqrt(np.mean(IF ** 2, 0) / n)
    tab = pd.DataFrame({"Group": grp, "Time": per, "ATT(g,t)": att, "Std. Err.": se, "CI low": att - 1.96 * se, "CI high": att + 1.96 * se,
                        "Post": per >= grp})
    pg_unit = {gg: np.mean(G == gg) for gg in glist}
    pg = np.array([pg_unit[gg] for gg in grp])

    def wif(keep):
        s = pg[keep].sum()
        ind = np.column_stack([(G == grp[k]).astype(float) - pg[k] for k in keep])
        if1 = ind / s
        if2 = ind.sum(1)[:, None] @ (pg[keep] / s ** 2)[None, :]
        return if1 - if2

    def agg(keep, w_given=None, with_wif=True):
        keep = np.asarray(keep)
        if w_given is None:
            w = pg[keep] / pg[keep].sum()
        else:
            w = w_given
        est = float(w @ att[keep])
        f = IF[:, keep] @ w
        if with_wif and w_given is None:
            f = f + wif(keep) @ att[keep]
        return est, f

    def se_of(f):
        return float(np.sqrt(np.mean(f ** 2) / n))
    k_post = np.where(grp <= per)[0]
    simple, f_simple = agg(k_post)
    # dynamic
    e = per - grp
    dyn = []
    fdyn = {}
    for ee in sorted(set(e)):
        k = np.where(e == ee)[0]
        est, f = agg(k)
        fdyn[ee] = f
        dyn.append({"Event time": int(ee), "ATT": est, "Std. Err.": se_of(f)})
    dyn = pd.DataFrame(dyn)
    pos = [ee for ee in fdyn if ee >= 0]
    f_dyn_all = np.mean([fdyn[ee] for ee in pos], 0)
    dyn_all = float(dyn[dyn["Event time"] >= 0].ATT.mean())
    # group
    gtab, fg, attg = [], [], []
    for gg in glist:
        k = np.where((grp == gg) & (per >= gg))[0]
        if len(k) == 0:
            continue
        w = np.full(len(k), 1 / len(k))
        est, f = agg(k, w)
        gtab.append({"Group": int(gg), "ATT": est, "Std. Err.": se_of(f)})
        fg.append(f)
        attg.append(est)
    gtab = pd.DataFrame(gtab)
    pgg = np.array([pg_unit[gg] for gg in gtab.Group])
    wg = pgg / pgg.sum()
    ind = np.column_stack([(G == gg).astype(float) - pg_unit[gg] for gg in gtab.Group])
    wif_g = ind / pgg.sum() - ind.sum(1)[:, None] @ (pgg / pgg.sum() ** 2)[None, :]
    f_group = np.column_stack(fg) @ wg + wif_g @ np.array(attg)
    group_all = float(wg @ np.array(attg))
    # calendar
    ctab, fc = [], []
    for tt in tl[tl >= glist.min()]:
        k = np.where((per == tt) & (grp <= tt))[0]
        if len(k) == 0:
            continue
        est, f = agg(k)
        ctab.append({"Time": int(tt), "ATT": est, "Std. Err.": se_of(f)})
        fc.append(f)
    ctab = pd.DataFrame(ctab)
    f_cal = np.mean(fc, 0)
    summary = pd.DataFrame([
        {"Aggregation": "Simple (weighted by group size)", "ATT": simple, "Std. Err.": se_of(f_simple)},
        {"Aggregation": "Dynamic (average over e ≥ 0)", "ATT": dyn_all, "Std. Err.": se_of(f_dyn_all)},
        {"Aggregation": "Group (selective timing)", "ATT": group_all, "Std. Err.": se_of(f_group)},
        {"Aggregation": "Calendar time", "ATT": float(ctab.ATT.mean()), "Std. Err.": se_of(f_cal)}])
    summary["CI low"] = summary.ATT - 1.96 * summary["Std. Err."]
    summary["CI high"] = summary.ATT + 1.96 * summary["Std. Err."]
    summary["p-value"] = 2 * stats.norm.sf(np.abs(summary.ATT / summary["Std. Err."]))
    pre = tab[~tab.Post]
    W = None
    if len(pre):
        k = np.where(per < grp)[0]
        Vp = IF[:, k].T @ IF[:, k] / n / n
        try:
            Wst = float(att[k] @ np.linalg.solve(Vp, att[k]))
            W = {"Wald (pre-trends)": Wst, "df": len(k), "p-value": float(stats.chi2.sf(Wst, len(k)))}
        except np.linalg.LinAlgError:
            W = None
    for t_ in (dyn, gtab, ctab):
        t_["CI low"] = t_.ATT - 1.96 * t_["Std. Err."]
        t_["CI high"] = t_.ATT + 1.96 * t_["Std. Err."]
    return {"att_gt": tab, "summary": summary, "dynamic": dyn, "group": gtab, "calendar": ctab, "pretrend": W, "n units": n,
            "groups": glist.tolist()}


def bacon_weights(df: pd.DataFrame, unit: str, time: str, y: str, first_treated: str) -> pd.DataFrame:
    """Goodman-Bacon (2021) decomposition of the static TWFE DiD into 2×2 comparisons (balanced panel)."""
    d = df[[unit, time, y, first_treated]].dropna()
    Y = d.pivot(index=unit, columns=time, values=y)
    G = d.groupby(unit)[first_treated].first().reindex(Y.index).to_numpy()
    tl = np.array(sorted(Y.columns))
    T = len(tl)
    Yv = Y.to_numpy(float)
    groups = sorted(set(G))
    share = {g: np.mean(G == g) for g in groups}
    Dbar = {g: (np.mean(tl >= g) if g > 0 else 0.0) for g in groups}
    rows = []

    def dd(gk, gl, window):
        a, b = Yv[G == gk][:, window], Yv[G == gl][:, window]
        tw = tl[window]
        postk = tw >= gk if gk > 0 else np.zeros(len(tw), bool)
        return (a[:, postk].mean() - a[:, ~postk].mean()) - (b[:, postk].mean() - b[:, ~postk].mean())
    treated = [g for g in groups if g > 0 and g > tl.min()]
    for k in treated:
        if 0 in groups:
            nk, nu = share[k], share[0]
            Dk = Dbar[k]
            s = (nk + nu) ** 2 * (nk / (nk + nu)) * (nu / (nk + nu)) * Dk * (1 - Dk)
            rows.append({"Comparison": "Treated vs never treated", "Treated": k, "Control": "never", "Estimate": dd(k, 0, slice(None)), "s": s})
        for l_ in treated:
            if l_ <= k:
                continue
            nk, nl = share[k], share[l_]
            Dk, Dl = Dbar[k], Dbar[l_]
            nkl = nk / (nk + nl)
            w1 = np.where(tl < l_)[0]
            s1 = ((nk + nl) * (1 - Dl)) ** 2 * nkl * (1 - nkl) * ((Dk - Dl) / (1 - Dl)) * ((1 - Dk) / (1 - Dl))
            rows.append({"Comparison": "Earlier vs later treated (later as control)", "Treated": k, "Control": str(l_), "Estimate": dd(k, l_, w1), "s": s1})
            w2 = np.where(tl >= k)[0]
            sub = Yv[:, w2]
            tw = tl[w2]
            postl = tw >= l_
            est2 = (sub[G == l_][:, postl].mean() - sub[G == l_][:, ~postl].mean()) - (sub[G == k][:, postl].mean() - sub[G == k][:, ~postl].mean())
            s2 = ((nk + nl) * Dk) ** 2 * nkl * (1 - nkl) * (Dl / Dk) * ((Dk - Dl) / Dk)
            rows.append({"Comparison": "Later vs earlier treated (earlier as control — 'forbidden')", "Treated": l_, "Control": str(k), "Estimate": est2, "s": s2})
    out = pd.DataFrame(rows)
    out["Weight"] = out.s / out.s.sum()
    out.attrs["weighted sum"] = float((out.Weight * out.Estimate).sum())
    del T
    return out.drop(columns="s")


# ======================================================================================================================
# Synthetic control and synthetic DiD
# ======================================================================================================================
def _simplex_qp(A, b, Vd=None):
    """min (b − A w)' V (b − A w) over the unit simplex: NNLS with a heavily weighted adding-up row, polished by an
    active-set refit (fast enough for nested V optimisation and placebo loops)."""
    from scipy.optimize import nnls
    Vd = np.ones(A.shape[0]) if Vd is None else np.asarray(Vd, float)
    sv = np.sqrt(Vd)
    As, bs = A * sv[:, None], b * sv
    big = 1e3 * max(1.0, np.abs(As).max())
    w, _ = nnls(np.vstack([As, np.full(A.shape[1], big)]), np.r_[bs, big], maxiter=50 * A.shape[1])
    s_ = w.sum()
    w = w / s_ if s_ > 0 else np.full(A.shape[1], 1 / A.shape[1])
    act = w > 1e-10
    if act.sum() > 1:   # exact equality-constrained LS on the active set (KKT system)
        Q = As[:, act].T @ As[:, act]
        m = act.sum()
        K = np.block([[2 * Q, np.ones((m, 1))], [np.ones((1, m)), np.zeros((1, 1))]])
        rhs = np.r_[2 * As[:, act].T @ bs, 1.0]
        try:
            sol = np.linalg.solve(K, rhs)[:m]
            if np.all(sol >= -1e-12):
                w = np.zeros_like(w)
                w[act] = np.clip(sol, 0, None)
                w /= w.sum()
        except np.linalg.LinAlgError:
            pass
    return w


def synth_control(df: pd.DataFrame, unit: str, time: str, y: str, treated_unit, treat_time, predictors: list | None = None,
                  outcome_lags: list | None = None, custom_v=None, placebo: bool = True) -> dict:
    """Abadie, Diamond & Hainmueller (2010) synthetic control. Predictors = pre-period means of `predictors` plus the
    outcome in `outcome_lags`; rows scaled by their cross-unit SD (as Synth::dataprep); V by nested optimisation of the
    pre-treatment outcome MSPE (or `custom_v`). In-space placebos give post/pre RMSPE ratios and a permutation p-value."""
    predictors = predictors or []
    d = df[[unit, time, y] + predictors].dropna()
    Y = d.pivot(index=time, columns=unit, values=y)
    tl = Y.index.to_numpy()
    pre = tl < treat_time
    if outcome_lags is None:
        pyrs = tl[pre]
        outcome_lags = sorted(set(np.quantile(pyrs, [0, 0.33, 0.67, 1.0], method="nearest").tolist()))
    units = list(Y.columns)

    def build(tu):
        rows = []
        for p in predictors:
            rows.append(d[d[time] < treat_time].groupby(unit)[p].mean().reindex(units))
        for L in outcome_lags:
            rows.append(Y.loc[L])
        P = pd.concat(rows, axis=1).T
        P.index = [f"mean {p}" for p in predictors] + [f"{y} ({L})" for L in outcome_lags]
        return P

    def fit_one(tu):
        P = build(tu)
        sd = P.std(axis=1, ddof=1).replace(0, 1)
        Ps = P.div(sd, axis=0)
        X1, X0 = Ps[tu].to_numpy(), Ps.drop(columns=tu).to_numpy()
        Z1, Z0 = Y.loc[pre, tu].to_numpy(), Y.loc[pre].drop(columns=tu).to_numpy()
        kx = X0.shape[0]
        if custom_v is not None:
            v = np.asarray(custom_v, float)
            v = v / v.sum()
            w = _simplex_qp(X0, X1, v)
        else:
            def loss(th):
                v = np.exp(th - th.max())
                v = v / v.sum()
                w = _simplex_qp(X0, X1, v)
                return np.mean((Z1 - Z0 @ w) ** 2)
            best = None
            starts = [np.zeros(kx)] + [np.where(np.arange(kx) == j, 4.0, 0.0) for j in range(kx)]
            if not placebo_mode:
                starts += [np.random.default_rng(s).normal(0, 1.5, kx) for s in range(4)]
            for s0 in starts:
                r = optimize.minimize(loss, s0, method="Nelder-Mead", options={"maxiter": 400 * kx if placebo_mode else 1000 * kx, "xatol": 1e-7, "fatol": 1e-13})
                if best is None or r.fun < best.fun:
                    best = r
            r = optimize.minimize(loss, best.x, method="Powell", options={"xtol": 1e-7, "ftol": 1e-13, "maxiter": 20000})
            if r.fun < best.fun:
                best = r
            v = np.exp(best.x - best.x.max())
            v = v / v.sum()
            w = _simplex_qp(X0, X1, v)
        synth = Y.drop(columns=tu).to_numpy() @ w
        gap = Y[tu].to_numpy() - synth
        return {"w": pd.Series(w, index=[u for u in units if u != tu]), "v": pd.Series(v, index=P.index), "synthetic": synth, "gap": gap, "P": P}
    placebo_mode = False
    main = fit_one(treated_unit)
    rm = lambda g: np.sqrt(np.mean(g ** 2))  # noqa: E731
    path = pd.DataFrame({time: tl, "Treated": Y[treated_unit].to_numpy(), "Synthetic": main["synthetic"], "Gap": main["gap"]})
    P = main["P"]
    bal = pd.DataFrame({"Predictor": P.index, "Treated": P[treated_unit].to_numpy(), "Synthetic": P.drop(columns=treated_unit).to_numpy() @ main["w"].to_numpy(),
                        "Donor average": P.drop(columns=treated_unit).mean(1).to_numpy(), "V weight": main["v"].to_numpy()})
    out = {"path": path, "weights": main["w"].sort_values(ascending=False).rename("Weight").reset_index().rename(columns={"index": unit}),
           "balance": bal, "pre RMSPE": rm(main["gap"][pre]), "post RMSPE": rm(main["gap"][~pre]), "ATT (post average gap)": float(main["gap"][~pre].mean())}
    if placebo:
        placebo_mode = True
        gaps, ratios = {treated_unit: main["gap"]}, {treated_unit: rm(main["gap"][~pre]) / rm(main["gap"][pre])}
        for u in units:
            if u == treated_unit:
                continue
            r = fit_one(u)
            gaps[u] = r["gap"]
            ratios[u] = rm(r["gap"][~pre]) / rm(r["gap"][pre])
        rr = pd.Series(ratios).sort_values(ascending=False)
        out["placebo gaps"] = pd.DataFrame(gaps, index=tl)
        out["ratios"] = rr.rename("post/pre RMSPE").reset_index().rename(columns={"index": unit})
        out["p-value (rank)"] = float((rr >= rr[treated_unit]).sum() / len(rr))
        pre_rm = {u: rm(g[pre]) for u, g in gaps.items()}
        out["pre RMSPE (all)"] = pre_rm
    return out


def _fw(Y, zeta, intercept=True, lam=None, min_decrease=1e-3, max_iter=1000):
    """synthdid::sc.weight.fw (Frank-Wolfe for the ridge-penalised simplex regression)."""
    N0, T0p = Y.shape
    T0 = T0p - 1
    if intercept:
        Y = Y - Y.mean(0)
    lam = np.full(T0, 1 / T0) if lam is None else lam.copy()
    A, b = Y[:, :T0], Y[:, T0]
    eta = N0 * zeta ** 2
    vals = []
    t = 0
    while t < max_iter and (t < 2 or vals[-2] - vals[-1] > min_decrease ** 2):
        Ax = A @ lam
        hg = (Ax - b) @ A + eta * lam
        i = int(np.argmin(hg))
        dx = -lam.copy()
        dx[i] = 1 - lam[i]
        if np.all(dx == 0):
            t += 1
            err = Y @ np.r_[lam, -1]
            vals.append(zeta ** 2 * np.sum(lam ** 2) + np.sum(err ** 2) / N0)
            continue
        derr = A[:, i] - Ax
        step = -np.sum(hg * dx) / (np.sum(derr ** 2) + eta * np.sum(dx ** 2))
        lam = lam + min(1.0, max(0.0, step)) * dx
        t += 1
        err = Y @ np.r_[lam, -1]
        vals.append(zeta ** 2 * np.sum(lam ** 2) + np.sum(err ** 2) / N0)
    return lam


def _sparsify(v):
    v = np.where(v <= v.max() / 4, 0, v)
    return v / v.sum()


def _sdid_core(Y, N0, T0, om0=None, lam0=None):
    N, T = Y.shape
    N1, T1 = N - N0, T - T0
    noise = np.std(np.diff(Y[:N0, :T0], axis=1), ddof=1)
    eta_o = (N1 * T1) ** 0.25
    z_o, z_l = eta_o * noise, 1e-6 * noise
    md = 1e-5 * noise
    Yl = np.column_stack([Y[:N0, :T0], Y[:N0, T0:].mean(1)])
    lam = _fw(Yl, z_l, True, lam0, md, 100)
    lam = _fw(Yl, z_l, True, _sparsify(lam), md, 10000)
    Yo = np.column_stack([Y[:N0, :T0].T, Y[N0:, :T0].mean(0)])
    om = _fw(Yo, z_o, True, om0, md, 100)
    om = _fw(Yo, z_o, True, _sparsify(om), md, 10000)
    tau = np.r_[-om, np.full(N1, 1 / N1)] @ Y @ np.r_[-lam, np.full(T1, 1 / T1)]
    return float(tau), om, lam


def synth_did(df: pd.DataFrame, unit: str, time: str, y: str, treat: str, reps: int = 200, seed: int = 1) -> dict:
    """Synthetic difference-in-differences (Arkhangelsky, Athey, Hirshberg, Imbens & Wager 2021) for block adoption,
    reproducing synthdid::synthdid_estimate (Frank-Wolfe weights with sparsification) and the placebo SE
    (vcov method = 'placebo'). `treat` is the unit-time treatment indicator."""
    d = df[[unit, time, y, treat]].dropna()
    Yw = d.pivot(index=unit, columns=time, values=y)
    Dw = d.pivot(index=unit, columns=time, values=treat)
    tr_units = Dw.index[Dw.max(1) > 0]
    co_units = Dw.index[Dw.max(1) == 0]
    T0 = int((Dw.loc[tr_units].max(0) == 0).sum())
    Yw = pd.concat([Yw.loc[co_units], Yw.loc[tr_units]])
    Y = Yw.to_numpy(float)
    N0 = len(co_units)
    tau, om, lam = _sdid_core(Y, N0, T0)
    N1 = len(tr_units)
    rng = np.random.default_rng(seed)
    est = []
    Yc = Y[:N0]
    if N0 <= N1:
        raise ValueError("The placebo SE needs more control than treated units.")
    for _ in range(reps):          # synthdid::placebo_se (warm-started at the estimated weights)
        idx = rng.permutation(N0)
        o0 = om[idx[:N0 - N1]]
        o0 = o0 / o0.sum() if o0.sum() > 0 else np.full(N0 - N1, 1 / (N0 - N1))
        est.append(_sdid_core(Yc[idx], N0 - N1, T0, o0, lam)[0])
    se = float(np.sqrt((reps - 1) / reps) * np.std(est, ddof=1))
    # DiD and SC comparators
    did_ = float(Y[N0:, T0:].mean() - Y[N0:, :T0].mean() - (Y[:N0, T0:].mean() - Y[:N0, :T0].mean()))
    tl = Yw.columns.to_numpy()
    syn = om @ Y[:N0]
    tr = Y[N0:].mean(0)
    path = pd.DataFrame({time: tl, "Treated": tr, "SDiD control (ω-weighted)": syn + (tr[:T0] @ lam - syn[:T0] @ lam), "Time weight λ": np.r_[lam, np.zeros(len(tl) - T0)]})
    return {"ATT": tau, "se": se, "CI": (tau - 1.96 * se, tau + 1.96 * se), "p-value": float(2 * stats.norm.sf(abs(tau / se))), "DiD": did_,
            "unit weights": pd.Series(om, index=co_units).sort_values(ascending=False), "time weights": pd.Series(lam, index=tl[:T0]),
            "path": path, "T0": T0, "N0": N0, "N1": N1}


# ======================================================================================================================
# Regression discontinuity (official rdrobust / rddensity packages)
# ======================================================================================================================
def _quiet(f, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def rdd(df: pd.DataFrame, y: str, running: str, cutoff: float, treat: str | None = None, p: int = 1, kernel: str = "triangular",
        bwselect: str = "mserd", covs: list | None = None, cluster: str | None = None) -> dict:
    """Local-polynomial RD (Calonico, Cattaneo & Titiunik 2014) via rdrobust: conventional, bias-corrected and robust
    estimates; fuzzy RD when `treat` is given (first stage reported). Includes the rddensity manipulation test
    (Cattaneo, Jansson & Ma 2020), bandwidth sensitivity, placebo cutoffs and covariate continuity."""
    from rdrobust import rdrobust
    cols = list(dict.fromkeys([y, running] + ([treat] if treat else []) + (covs or []) + ([cluster] if cluster else [])))
    d = df[cols].dropna()
    kw = dict(c=cutoff, p=p, kernel=kernel, bwselect=bwselect)
    if cluster:
        kw["cluster"] = d[cluster]
    if treat:
        kw["fuzzy"] = d[treat]
    r = _quiet(rdrobust, d[y], d[running], **kw)
    lab = ["Conventional", "Bias-corrected", "Robust"]
    tab = pd.DataFrame({"Method": lab, "Estimate": r.coef.to_numpy().ravel(), "Std. Err.": r.se.to_numpy().ravel(), "z": r.z.to_numpy().ravel(),
                        "p-value": r.pv.to_numpy().ravel(), "CI low": r.ci.iloc[:, 0].to_numpy(), "CI high": r.ci.iloc[:, 1].to_numpy()})
    out = {"table": tab, "h": float(r.bws.iloc[0, 0]), "b": float(r.bws.iloc[1, 0]), "N left": int(r.N[0]), "N right": int(r.N[1]),
           "N eff left": int(r.N_h[0]), "N eff right": int(r.N_h[1]), "kernel": kernel, "p": p}
    if treat:
        fs = _quiet(rdrobust, d[treat], d[running], c=cutoff, p=p, kernel=kernel, h=out["h"], b=out["b"])
        out["first stage"] = {"Jump in treatment": float(fs.coef.iloc[0, 0]), "Std. Err.": float(fs.se.iloc[0, 0]), "p-value (robust)": float(fs.pv.iloc[2, 0])}
    # bandwidth sensitivity
    sens = []
    for m in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        kk = dict(kw)
        kk.pop("bwselect", None)
        rr = _quiet(rdrobust, d[y], d[running], h=out["h"] * m, b=max(out["b"], out["h"] * m), **{k: v for k, v in kk.items()})
        sens.append({"Bandwidth × h*": m, "h": out["h"] * m, "Estimate": float(rr.coef.iloc[0, 0]), "Robust CI low": float(rr.ci.iloc[2, 0]),
                     "Robust CI high": float(rr.ci.iloc[2, 1]), "Robust p": float(rr.pv.iloc[2, 0]), "N eff": int(sum(rr.N_h))})
    out["sensitivity"] = pd.DataFrame(sens)
    # placebo cutoffs (on each side, using only that side's observations)
    x = d[running].to_numpy(float)
    plc = []
    for side, q in (("left", (0.25, 0.5, 0.75)), ("right", (0.25, 0.5, 0.75))):
        sub = d[x < cutoff] if side == "left" else d[x >= cutoff]
        for qq in q:
            c2 = float(np.quantile(sub[running], qq))
            try:
                rr = _quiet(rdrobust, sub[y], sub[running], c=c2, p=p, kernel=kernel)
                plc.append({"Placebo cutoff": c2, "Side": side, "Estimate": float(rr.coef.iloc[0, 0]), "Robust p": float(rr.pv.iloc[2, 0])})
            except Exception:  # noqa: BLE001
                pass
    out["placebo"] = pd.DataFrame(plc)
    if covs:
        cb = []
        for c in covs:
            rr = _quiet(rdrobust, d[c], d[running], c=cutoff, p=p, kernel=kernel)
            cb.append({"Covariate": c, "Jump": float(rr.coef.iloc[0, 0]), "Robust p": float(rr.pv.iloc[2, 0])})
        out["covariates"] = pd.DataFrame(cb)
    try:
        from rddensity import rddensity
        dn = _quiet(rddensity, X=d[running].to_numpy(float), c=cutoff)
        out["density test"] = {"T (jackknife)": float(dn.test["t_jk"]), "p-value": float(dn.test["p_jk"]),
                               "density left": float(dn.hat["left"]), "density right": float(dn.hat["right"])}
    except Exception:  # noqa: BLE001
        out["density test"] = None
    out["binned"] = rd_bins(d, y, running, cutoff)
    return out


def rd_bins(df, y, running, cutoff, nbins: int = 20, order: int = 4) -> dict:
    """Evenly-spaced bin means on each side plus a global polynomial fit (rdplot-style picture)."""
    x, yv = df[running].to_numpy(float), df[y].to_numpy(float)
    out = []
    fits = []
    for side in ("left", "right"):
        m = x < cutoff if side == "left" else x >= cutoff
        xs, ys = x[m], yv[m]
        edges = np.linspace(xs.min(), xs.max(), nbins + 1)
        idx = np.clip(np.digitize(xs, edges) - 1, 0, nbins - 1)
        for b in range(nbins):
            s = idx == b
            if s.any():
                out.append({"side": side, "x": xs[s].mean(), "y": ys[s].mean(), "n": int(s.sum())})
        cf = np.polyfit(xs - cutoff, ys, min(order, max(1, len(np.unique(xs)) - 1)))
        g = np.linspace(xs.min(), xs.max(), 100)
        fits.append(pd.DataFrame({"side": side, "x": g, "fit": np.polyval(cf, g - cutoff)}))
    return {"bins": pd.DataFrame(out), "fit": pd.concat(fits, ignore_index=True)}


# ======================================================================================================================
# Selection on observables: propensity scores, matching, IPW, AIPW
# ======================================================================================================================
def _logit(D, X):
    import statsmodels.api as sm
    try:
        m = sm.Logit(D, X).fit(disp=0, maxiter=200)
    except Exception as e:  # noqa: BLE001  (perfect separation / singular Hessian)
        raise ValueError("the propensity model cannot be estimated — a covariate (nearly) predicts treatment perfectly; remove it") from e
    ps = m.predict(X)
    if ps.min() < 1e-8 or ps.max() > 1 - 1e-8:
        raise ValueError("propensity scores of 0 or 1 (no overlap) — a covariate (nearly) predicts treatment perfectly; remove it or trim")
    return m.params, ps


def propensity(df, treat, covs):
    d = df[[treat] + covs].dropna()
    X = np.column_stack([np.ones(len(d)), d[covs].to_numpy(float)])
    _, ps = _logit(d[treat].to_numpy(float), X)
    return pd.Series(ps, index=d.index)


def psm(df: pd.DataFrame, y: str, treat: str, covs: list, k: int = 1, caliper: float | None = None, estimand: str = "ATT") -> dict:
    """Nearest-neighbour matching on the estimated logit propensity score, with replacement (as MatchIt method =
    'nearest', replace = TRUE, and teffects psmatch). Caliper in SDs of the logit of the score. Abadie-Imbens (2006)
    standard error with two same-arm neighbours for the conditional variances (ignores first-step estimation)."""
    d = df[[y, treat] + covs].dropna().reset_index(drop=True)
    X = np.column_stack([np.ones(len(d)), d[covs].to_numpy(float)])
    D = d[treat].to_numpy(int)
    yv = d[y].to_numpy(float)
    _, ps = _logit(D.astype(float), X)
    lp = np.log(ps / (1 - ps))
    cal = caliper * lp.std(ddof=1) if caliper else np.inf

    def match(src, dst):
        o = np.argsort(np.abs(lp[src][:, None] - lp[dst][None, :]), axis=1, kind="stable")[:, :k]
        dist = np.abs(lp[src][:, None] - lp[dst][o])
        return dst[o], dist <= cal
    T_, C_ = np.where(D == 1)[0], np.where(D == 0)[0]
    used = np.zeros(len(d))
    taus = []
    targets = [(T_, C_, 1)] if estimand == "ATT" else [(T_, C_, 1), (C_, T_, -1)]
    unmatched = 0
    for src, dst, sgn in targets:
        mt, ok = match(src, dst)
        for i, s in enumerate(src):
            js = mt[i][ok[i]]
            if len(js) == 0:
                unmatched += 1
                continue
            y0 = yv[js].mean()
            taus.append(sgn * (yv[s] - y0))
            used[js] += 1 / len(js)
    tau = float(np.mean(taus))
    # conditional variance by matching within arm (J = 2)
    sig2 = np.zeros(len(d))
    for grp in (T_, C_):
        o = np.argsort(np.abs(lp[grp][:, None] - lp[grp][None, :]), axis=1, kind="stable")[:, 1:3]
        sig2[grp] = 2 / 3 * (yv[grp] - yv[grp][o].mean(1)) ** 2
    if estimand == "ATT":
        mt, ok = match(T_, C_)
        keepT = T_[ok.any(1)]
        y0 = np.array([yv[mt[i][ok[i]]].mean() for i in range(len(T_)) if ok[i].any()])
        N1 = len(keepT)
        V = (np.sum((yv[keepT] - y0 - tau) ** 2) + np.sum((used[C_] * (used[C_] - 1 / k)) * sig2[C_])) / N1 ** 2
    else:
        V = np.var(taus, ddof=1) / len(taus) + np.sum(used * (used - 1 / k) * sig2) / len(taus) ** 2
    se = float(np.sqrt(max(V, 0)))
    w = np.where(D == 1, 1.0, used) if estimand == "ATT" else np.where(used > 0, used + 1, 1.0)
    bal_before = balance_table(d, treat, covs)
    bal_after = balance_table(d.assign(_w=w), treat, covs, weights=w)
    bal = bal_before[["Covariate", "Std. mean diff."]].rename(columns={"Std. mean diff.": "SMD before"})
    bal["SMD after matching"] = bal_after["Std. mean diff."].to_numpy()
    return {"estimate": _row(f"{estimand} (PSM, k={k})", tau, se), "balance": bal, "ps": ps, "D": D, "matched controls used": int((used[C_] > 0).sum()),
            "unmatched (caliper)": unmatched, "weights": w}


def ipw_aipw(df: pd.DataFrame, y: str, treat: str, covs: list, estimand: str = "ATE", trim: float | None = None) -> dict:
    """Inverse-probability weighting (normalised / Hájek weights) and augmented IPW (doubly robust, linear outcome
    models per arm), for the ATE or ATT. Standard errors from the stacked M-estimation sandwich (accounting for the
    estimated propensity and outcome models), as WeightIt::glm_weightit and teffects ipw / aipw."""
    d = df[[y, treat] + covs].dropna().reset_index(drop=True)
    X = np.column_stack([np.ones(len(d)), d[covs].to_numpy(float)])
    D = d[treat].to_numpy(float)
    yv = d[y].to_numpy(float)
    n, kx = X.shape
    gam, ps = _logit(D, X)
    keep = np.ones(n, bool)
    if trim:
        keep = (ps > trim) & (ps < 1 - trim)
        d, X, D, yv = d[keep].reset_index(drop=True), X[keep], D[keep], yv[keep]
        gam, ps = _logit(D, X)
        n = len(d)

    def ipw_est(theta):
        g = theta[:kx]
        e = 1 / (1 + np.exp(-X @ g))
        if estimand == "ATE":
            w1, w0 = D / e, (1 - D) / (1 - e)
        else:
            w1, w0 = D, (1 - D) * e / (1 - e)
        return w1, w0, e

    def psi_ipw(theta):
        m1, m0 = theta[kx], theta[kx + 1]
        w1, w0, e = ipw_est(theta)
        s = X * (D - e)[:, None]
        return np.column_stack([s, w1 * (yv - m1), w0 * (yv - m0)])
    w1, w0, _ = ipw_est(np.r_[gam, 0, 0])
    m1, m0 = np.sum(w1 * yv) / w1.sum(), np.sum(w0 * yv) / w0.sum()
    th = np.r_[gam, m1, m0]

    def sandwich(psi, th):
        P = psi(th)
        from statsmodels.tools.numdiff import approx_fprime
        A = approx_fprime(th, lambda t: psi(t).mean(0), centered=True)
        B = P.T @ P / n
        Ai = np.linalg.inv(A)
        return Ai @ B @ Ai.T / n
    V = sandwich(psi_ipw, th)
    c = np.zeros(len(th))
    c[kx], c[kx + 1] = 1, -1
    ipw_tau, ipw_se = m1 - m0, float(np.sqrt(c @ V @ c))
    # AIPW
    b1 = np.linalg.lstsq(X[D == 1], yv[D == 1], rcond=None)[0]
    b0 = np.linalg.lstsq(X[D == 0], yv[D == 0], rcond=None)[0]
    mu1, mu0 = X @ b1, X @ b0
    if estimand == "ATE":
        phi = mu1 - mu0 + D * (yv - mu1) / ps - (1 - D) * (yv - mu0) / (1 - ps)
        a_tau = float(phi.mean())
        a_se = float(phi.std(ddof=0) / np.sqrt(n))
    else:
        p1 = D.mean()
        phi = (D * (yv - mu0) - (1 - D) * ps / (1 - ps) * (yv - mu0)) / p1
        a_tau = float(phi.mean())
        a_se = float(np.sqrt(np.mean((phi - a_tau * D / p1) ** 2) / n))
    wts = np.where(D == 1, w1, w0)
    bal_w = balance_table(d, treat, covs, weights=wts)
    bal = balance_table(d, treat, covs)[["Covariate", "Std. mean diff."]].rename(columns={"Std. mean diff.": "SMD before"})
    bal["SMD after weighting"] = bal_w["Std. mean diff."].to_numpy()
    ess = lambda w: w.sum() ** 2 / np.sum(w ** 2)  # noqa: E731
    return {"table": pd.DataFrame([{"Estimator": f"IPW ({estimand})", **_row(treat, ipw_tau, ipw_se)},
                                   {"Estimator": f"AIPW, doubly robust ({estimand})", **_row(treat, a_tau, a_se)}]),
            "balance": bal, "ps": ps, "D": D, "weights": wts, "ESS treated": float(ess(wts[D == 1])), "ESS control": float(ess(wts[D == 0])),
            "trimmed": int((~keep).sum()), "ps range": (float(ps.min()), float(ps.max()))}


def overlap(ps, D, bins: int = 30) -> pd.DataFrame:
    e = np.linspace(0, 1, bins + 1)
    h1, _ = np.histogram(ps[D == 1], e, density=True)
    h0, _ = np.histogram(ps[D == 0], e, density=True)
    return pd.DataFrame({"ps": (e[:-1] + e[1:]) / 2, "Treated": h1, "Control": h0})
