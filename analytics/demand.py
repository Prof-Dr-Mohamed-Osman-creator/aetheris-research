# ==========================================
# Aetheris Research - Supply response, demand & consumption (v1.10, Catalogue §2.8)
# Nerlove supply response (partial adjustment, adaptive expectations, combined), cobweb dynamics, single-equation
# demand (static and dynamic), Engel curves, demand systems (LES, LA-AIDS, AIDS, QUAIDS, Rotterdam) with
# Marshallian / Hicksian / expenditure elasticities, consumption functions (Keynes, permanent income, Hall,
# Campbell-Mankiw) and welfare (consumer surplus, compensating and equivalent variation).
# ==========================================

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from analytics.production import sur

warnings.filterwarnings("ignore")


def _ols(y, X, names, hac: int | None = None):
    import statsmodels.api as sm
    m = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac}) if hac else sm.OLS(y, X).fit()
    tab = pd.DataFrame({"Parameter": names, "Coef.": m.params, "Std. Err.": m.bse, "t": m.tvalues, "p-value": m.pvalues})
    return m, tab


def _delta(fun, b, V):
    b = np.asarray(b, float)
    g = np.zeros(len(b))
    for i in range(len(b)):
        e = np.zeros(len(b)); e[i] = 1e-6 * max(1, abs(b[i]))
        g[i] = (fun(b + e) - fun(b - e)) / (2 * e[i])
    return float(fun(b)), float(np.sqrt(g @ V @ g))


# ==========================================================
# Supply response
# ==========================================================
def nerlove(df: pd.DataFrame, time: str, q: str, price: str, others: list | None = None, model: str = "pa",
            log: bool = True, trend: bool = True) -> dict:
    """Nerlovian supply (acreage) response with naive price expectations P*_t = P_{t−1}:
    'pa'  partial adjustment: Q_t = a + b P_{t−1} + c Q_{t−1} (+ others_{t−1}, trend): δ = 1 − c, SR = b, LR = b/(1 − c);
    'ae'  adaptive expectations: same reduced form (γ = 1 − c is the expectation coefficient);
    'paae' both: Q_t = a + b P_{t−1} + c1 Q_{t−1} + c2 Q_{t−2}, with δ + γ = 2 − c1 and (1 − δ)(1 − γ) = −c2.
    In logs the coefficients are elasticities. Mean lag and half-life of the adjustment follow."""
    others = list(others or [])
    d = df[[time, q, price] + others].dropna().sort_values(time).reset_index(drop=True)
    f = np.log if log else (lambda v: v)
    Q, Pp = f(d[q].to_numpy(float)), f(d[price].to_numpy(float))
    O = np.column_stack([f(d[o].to_numpy(float)) if log and (d[o] > 0).all() else d[o].to_numpy(float) for o in others]) if others else np.zeros((len(d), 0))
    L = 2 if model == "paae" else 1
    y = Q[L:]
    cols = [np.ones(len(y)), Pp[L - 1:-1], Q[L - 1:-1]] + ([Q[L - 2:-2]] if model == "paae" else []) + [O[L - 1:-1, j] for j in range(O.shape[1])]
    names = ["const", f"{price}(t−1)", f"{q}(t−1)"] + ([f"{q}(t−2)"] if model == "paae" else []) + [f"{o}(t−1)" for o in others]
    if trend:
        cols.append(np.arange(len(y), dtype=float))
        names.append("trend")
    X = np.column_stack(cols)
    m, tab = _ols(y, X, names)
    b, V = m.params, m.cov_params()
    out = {"table": tab, "model": m, "R²": float(m.rsquared), "N": len(y), "kind": model}
    sr = (b[1], np.sqrt(V[1, 1]))
    if model in ("pa", "ae"):
        lr = _delta(lambda p: p[1] / (1 - p[2]), b, V)
        c = b[2]
        out.update({"SR": sr, "LR": lr, "adjustment coefficient": (1 - c, np.sqrt(V[2, 2])), "mean lag": c / (1 - c) if c < 1 else np.inf,
                    "half-life": np.log(0.5) / np.log(c) if 0 < c < 1 else np.nan, "stable": bool(abs(c) < 1)})
        for j, o in enumerate(others):
            out[f"LR {o}"] = _delta(lambda p, j=j: p[3 + j] / (1 - p[2]), b, V)
    else:
        c1, c2 = b[2], b[3]
        s_ = 2 - c1                                   # δ + γ;  (1 − δ)(1 − γ) = −c2  ⇒  δγ = s_ − 1 − c2
        prod = s_ - 1 - c2
        disc = s_ ** 2 - 4 * prod
        roots = (np.nan, np.nan) if disc < 0 else ((s_ + np.sqrt(disc)) / 2, (s_ - np.sqrt(disc)) / 2)
        lr = _delta(lambda p: p[1] / (1 - p[2] - p[3]), b, V)
        out.update({"SR": sr, "LR": lr, "δ, γ (unordered)": roots, "stable": bool(np.all(np.abs(np.roots([1, -c1, -c2])) < 1))})
    out["fitted"] = pd.DataFrame({time: d[time].iloc[L:].to_numpy(), "actual": y, "fitted": m.fittedvalues})
    return out


def cobweb(a_d: float, b_d: float, a_s: float, b_s: float, p0: float, periods: int = 20) -> dict:
    """Linear cobweb: demand Q = a_d − b_d P, supply Q_t = a_s + b_s P_{t−1}. Equilibrium P* = (a_d − a_s)/(b_d + b_s);
    oscillations converge when b_s/b_d < 1, persist when = 1, explode when > 1."""
    pe = (a_d - a_s) / (b_d + b_s)
    qe = a_d - b_d * pe
    P, Q = [p0], []
    for _ in range(periods):
        q = a_s + b_s * P[-1]
        Q.append(q)
        P.append((a_d - q) / b_d)
    ratio = b_s / b_d
    return {"P": np.array(P), "Q": np.array(Q), "P*": pe, "Q*": qe, "ratio": ratio,
            "verdict": "convergent" if ratio < 1 else ("continuous" if np.isclose(ratio, 1) else "explosive")}


def cobweb_from_elasticities(p_mean, q_mean, e_supply, e_demand) -> tuple:
    """Linear curves through the sample means with the given elasticities (e_demand < 0)."""
    b_s = e_supply * q_mean / p_mean
    b_d = -e_demand * q_mean / p_mean
    return q_mean + b_d * p_mean, b_d, q_mean - b_s * p_mean, b_s


# ==========================================================
# Single-equation demand
# ==========================================================
def demand_equation(df: pd.DataFrame, q: str, own: str, cross: list | None = None, income: str | None = None,
                    form: str = "loglog", dynamic: bool = False, time: str | None = None, trend: bool = False) -> dict:
    """Demand for one good: log-log (constant elasticities) or linear (elasticities at the means); dynamic adds the
    lagged quantity (partial adjustment) and long-run elasticities e/(1 − λ) by the delta method; homogeneity test
    (own + cross + income elasticities = 0, log-log)."""
    cross = list(cross or [])
    cols = [q, own] + cross + ([income] if income else []) + ([time] if time else [])
    d = df[cols].dropna()
    if time:
        d = d.sort_values(time)
    vars_ = [own] + cross + ([income] if income else [])
    if form == "loglog":
        y = np.log(d[q].to_numpy(float))
        R = np.log(d[vars_].to_numpy(float))
    else:
        y = d[q].to_numpy(float)
        R = d[vars_].to_numpy(float)
    names = ["const"] + vars_
    cols_ = [np.ones(len(y)), R]
    if dynamic:
        y, R = y[1:], R[1:]
        lagq = (np.log(d[q].to_numpy(float)) if form == "loglog" else d[q].to_numpy(float))[:-1]
        cols_ = [np.ones(len(y)), R, lagq]
        names.append(f"{q}(t−1)")
    if trend:
        cols_.append(np.arange(len(y), dtype=float))
        names.append("trend")
    X = np.column_stack(cols_)
    m, tab = _ols(y, X, names)
    b, V = np.asarray(m.params), np.asarray(m.cov_params())
    k = len(vars_)
    rows = []
    for j, v in enumerate(vars_):
        if form == "loglog":
            sr = (b[1 + j], np.sqrt(V[1 + j, 1 + j]))
        else:
            sc = d[v].mean() / d[q].mean()
            sr = (b[1 + j] * sc, np.sqrt(V[1 + j, 1 + j]) * sc)
        row = {"Variable": v, "Type": "own price" if j == 0 else ("income" if income and v == income else "cross price"),
               "Short-run elasticity": sr[0], "SE (SR)": sr[1]}
        if dynamic:
            il = 1 + k
            if form == "loglog":
                lr = _delta(lambda p, j=j: p[1 + j] / (1 - p[il]), b, V)
            else:
                sc = d[v].mean() / d[q].mean()
                lr = _delta(lambda p, j=j: p[1 + j] * sc / (1 - p[il]), b, V)
            row.update({"Long-run elasticity": lr[0], "SE (LR)": lr[1]})
        rows.append(row)
    out = {"table": tab, "elasticities": pd.DataFrame(rows), "R²": float(m.rsquared), "N": len(y), "model": m}
    if form == "loglog" and k > 1:
        R_ = np.zeros(len(b)); R_[1:1 + k] = 1
        w = m.t_test(R_)
        out["homogeneity"] = {"sum": float(R_ @ b), "t": float(np.asarray(w.tvalue).ravel()[0]), "p": float(np.asarray(w.pvalue).ravel()[0])}
    if dynamic:
        out["adjustment coefficient"] = float(1 - b[1 + k])
    return out


# ==========================================================
# Engel curves
# ==========================================================
ENGEL_FORMS = {"wl": ("Working-Leser (w = a + b ln x)", "ووركنج-ليسر (w = a + b ln x)"), "qwl": ("Quadratic Working-Leser", "ووركنج-ليسر التربيعية"),
               "loglog": ("Double-log (ln e = a + b ln x)", "اللوغاريتمية المزدوجة"), "semilog": ("Semi-log (e = a + b ln x)", "نصف اللوغاريتمية"),
               "linear": ("Linear (e = a + b x)", "الخطية")}


def engel(df: pd.DataFrame, total: str, shares: list, form: str = "wl", demog: list | None = None) -> pd.DataFrame:
    """Expenditure (income) elasticities from Engel curves at the sample means, for each good; classification as
    inferior (< 0), necessity (0–1) or luxury (> 1). Shares can be budget shares (wl, qwl) — other forms use the
    implied expenditure e = w·x."""
    import statsmodels.api as sm
    demog = list(demog or [])
    d = df[[total] + shares + demog].dropna()
    x = d[total].to_numpy(float)
    lx = np.log(x)
    D = d[demog].to_numpy(float) if demog else np.zeros((len(d), 0))
    rows = []
    for s_ in shares:
        w = d[s_].to_numpy(float)
        e = w * x
        if form == "wl":
            Z = sm.add_constant(np.column_stack([lx, D]))
            m = sm.OLS(w, Z).fit()
            el = 1 + m.params[1] / w.mean()
            se = m.bse[1] / w.mean()
        elif form == "qwl":
            Z = sm.add_constant(np.column_stack([lx, lx ** 2, D]))
            m = sm.OLS(w, Z).fit()
            g = np.r_[0, 1, 2 * lx.mean(), np.zeros(D.shape[1])] / w.mean()
            el = 1 + g @ m.params
            se = float(np.sqrt(g @ m.cov_params() @ g))
        elif form == "loglog":
            Z = sm.add_constant(np.column_stack([lx, np.log(np.maximum(D, 1e-9)) if D.size else D]))
            m = sm.OLS(np.log(np.maximum(e, 1e-12)), Z).fit()
            el, se = m.params[1], m.bse[1]
        elif form == "semilog":
            Z = sm.add_constant(np.column_stack([lx, D]))
            m = sm.OLS(e, Z).fit()
            el, se = m.params[1] / e.mean(), m.bse[1] / e.mean()
        else:
            Z = sm.add_constant(np.column_stack([x, D]))
            m = sm.OLS(e, Z).fit()
            el, se = m.params[1] * x.mean() / e.mean(), m.bse[1] * x.mean() / e.mean()
        cls = "inferior" if el < 0 else ("necessity" if el < 1 else "luxury")
        rows.append({"Good": s_, "Mean share": w.mean(), "Expenditure elasticity": float(el), "Std. Err.": float(se), "Class": cls,
                     "R²": float(m.rsquared)})
    return pd.DataFrame(rows)


# ==========================================================
# Demand systems
# ==========================================================
SYSTEMS = {"laaids": ("LA-AIDS (Stone index)", "LA-AIDS (مؤشر ستون)"), "aids": ("AIDS (non-linear, translog index)", "AIDS (غير الخطي، مؤشر ترانسلوج)"),
           "quaids": ("QUAIDS", "QUAIDS"), "les": ("Linear expenditure system (LES)", "نظام الإنفاق الخطي (LES)"),
           "rotterdam": ("Rotterdam", "روتردام")}


def _restrictions(G, P_eq, layout, symmetric=True, homogeneous=True, extra=None):
    """Build R b = 0 for G−1 equations; layout(i) returns column indices dict for equation i with keys 'g' (list of G
    γ positions, the last for the numeraire price)."""
    rows = []
    K = (G - 1) * P_eq
    for i in range(G - 1):
        gi = layout(i)["g"]
        if homogeneous:
            r = np.zeros(K)
            r[[i * P_eq + c for c in gi]] = 1
            rows.append(r)
    if symmetric:
        for i in range(G - 1):
            for j in range(i + 1, G - 1):
                r = np.zeros(K)
                r[i * P_eq + layout(i)["g"][j]] = 1
                r[j * P_eq + layout(j)["g"][i]] = -1
                rows.append(r)
    return np.array(rows) if rows else None


def _prep_system(df, shares, prices, total, demog, normalize):
    demog = list(demog or [])
    d = df[shares + prices + [total] + demog].dropna()
    d = d[(d[prices + [total]] > 0).all(1)]
    W = d[shares].to_numpy(float)
    W = W / W.sum(1, keepdims=True)
    lp = np.log(d[prices].to_numpy(float))
    if normalize:
        lp = lp - lp.mean(0)
    lx = np.log(d[total].to_numpy(float))
    D = d[demog].to_numpy(float) if demog else np.zeros((len(d), 0))
    if demog:
        D = D - D.mean(0)
    return d, W, lp, lx, D


def aids(df: pd.DataFrame, shares: list, prices: list, total: str, model: str = "quaids", demog: list | None = None,
         alpha0: float | None = None, symmetric: bool = True, homogeneous: bool = True, normalize: bool = True,
         max_iter: int = 200, tol: float = 1e-9) -> dict:
    """Almost Ideal demand systems. 'laaids': linear approximation with the Stone index ln P = Σ w̄_i ln p_i; 'aids' and
    'quaids': exact translog price index ln a(p) and, for QUAIDS, b(p) = Π p_i^{β_i}, estimated by iterated linear
    least squares (Blundell & Robin 1999) — each step is a restricted iterated SUR given the price indices; converges to
    the ML estimator. The last share equation is dropped and recovered from adding-up; homogeneity and symmetry are
    imposed as cross-equation restrictions. Prices are normalised to geometric mean 1 when `normalize`. α0 defaults to
    slightly below the minimum of ln x (Banks, Blundell & Lewbel 1997). Demographics enter the intercepts."""
    d, W, lp, lx, D = _prep_system(df, shares, prices, total, demog, normalize)
    n, G = W.shape
    nd = D.shape[1]
    quad = model == "quaids"
    a0 = alpha0 if alpha0 is not None else (lx.min() - 0.1 if model != "laaids" else 0.0)
    # parameters per equation i: α_i, δ_i (demog), γ_i1..γ_iG, β_i, (λ_i)
    P_eq = 1 + nd + G + 1 + quad

    def layout(i):
        return {"a": 0, "d": list(range(1, 1 + nd)), "g": list(range(1 + nd, 1 + nd + G)), "b": 1 + nd + G, "l": 2 + nd + G if quad else None}
    R = _restrictions(G, P_eq, layout, symmetric, homogeneous)
    wbar = W.mean(0)
    lnP = lp @ wbar
    lnb = np.zeros(n)
    params = None
    it = 0
    for it in range(max_iter if model != "laaids" else 1):
        z = lx - (lnP if model != "laaids" else lp @ wbar)
        Xi = np.column_stack([np.ones(n), D, lp, z] + ([z ** 2 / np.exp(lnb)] if quad else []))
        fit = sur([W[:, i] for i in range(G - 1)], [Xi] * (G - 1), R)
        B = fit["b"].reshape(G - 1, P_eq)
        # recover the last equation by adding-up
        last = -B.sum(0)
        last[0] += 1.0
        Bf = np.vstack([B, last])
        al, gm, be = Bf[:, 0], Bf[:, 1 + nd:1 + nd + G], Bf[:, 1 + nd + G]
        lam = Bf[:, 2 + nd + G] if quad else np.zeros(G)
        dl = Bf[:, 1:1 + nd]
        new_lnP = a0 + lp @ al + 0.5 * np.einsum("ti,ij,tj->t", lp, gm, lp) if model != "laaids" else lnP
        new_lnb = lp @ be if quad else lnb
        if params is not None and np.max(np.abs(fit["b"] - params)) < tol:
            params = fit["b"]
            break
        params = fit["b"]
        lnP, lnb = new_lnP, new_lnb
    names = []
    for i in range(G - 1):
        names += [f"α[{shares[i]}]"] + [f"δ[{shares[i]},{v}]" for v in (demog or [])] + [f"γ[{shares[i]},{prices[j]}]" for j in range(G)] + \
                 [f"β[{shares[i]}]"] + ([f"λ[{shares[i]}]"] if quad else [])
    se = np.sqrt(np.maximum(np.diag(fit["V"]), 0))
    tab = pd.DataFrame({"Parameter": names, "Coef.": params, "Std. Err.": se, "z": params / np.where(se > 0, se, np.nan)})
    tab["p-value"] = 2 * stats.norm.sf(np.abs(tab["z"]))
    res = {"table": tab, "alpha": al, "gamma": gm, "beta": be, "lambda": lam, "delta": dl, "alpha0": a0, "model": model, "shares": shares,
           "prices": prices, "W": W, "lp": lp, "lx": lx, "D": D, "iterations": it + 1, "loglik": fit["loglik"], "N": n, "Sigma": fit["Sigma"],
           "V": fit["V"]}
    res["elasticities"] = aids_elasticities(res)
    return res


def _aids_point(res, lp=None, lx=None):
    G = len(res["shares"])
    lp = np.zeros(G) if lp is None else lp
    lx = float(np.mean(res["lx"])) if lx is None else lx
    al, gm, be, lam = res["alpha"], res["gamma"], res["beta"], res["lambda"]
    if res["model"] == "laaids":
        w = res["W"].mean(0)
        return w, None, None
    lna = res["alpha0"] + lp @ al + 0.5 * lp @ gm @ lp
    b = np.exp(lp @ be)
    z = lx - lna
    w = al + gm @ lp + be * z + lam / b * z ** 2
    return w, z, b


def aids_elasticities(res, lp=None, lx=None) -> dict:
    """Expenditure, Marshallian (uncompensated) and Hicksian (compensated) price elasticities at a point (default:
    normalised prices = 1 and mean ln x): QUAIDS formulas of Banks et al. (1997); LA-AIDS: Green & Alston (1990)."""
    G = len(res["shares"])
    al, gm, be, lam = res["alpha"], res["gamma"], res["beta"], res["lambda"]
    I = np.eye(G)
    if res["model"] == "laaids":
        w = res["W"].mean(0)
        e = 1 + be / w
        M = -I + (gm - np.outer(be, w)) / w[:, None]
    else:
        lp = np.zeros(G) if lp is None else lp
        w, z, b = _aids_point(res, lp, lx)
        mu = be + 2 * lam / b * z
        mu_ij = gm - np.outer(mu, al + gm @ lp) - np.outer(lam, be) / b * z ** 2
        e = mu / w + 1
        M = mu_ij / w[:, None] - I
    H = M + np.outer(e, w)
    names = res["shares"]
    return {"shares": w, "expenditure": pd.Series(e, index=names), "marshallian": pd.DataFrame(M, index=names, columns=names),
            "hicksian": pd.DataFrame(H, index=names, columns=names), "negativity": bool(np.all(np.linalg.eigvalsh((H * w[:, None] + (H * w[:, None]).T) / 2) <= 1e-8))}


def les(df: pd.DataFrame, shares: list, prices: list, total: str, max_iter: int = 100) -> dict:
    """Stone-Geary linear expenditure system p_i q_i = p_i γ_i + β_i (x − Σ_j p_j γ_j), Σβ = 1, estimated in share form by
    iterated feasible GLS non-linear least squares (G − 1 equations). Elasticities: expenditure β_i/w_i, Marshallian
    e_ii = (1 − β_i)γ_i/q_i − 1, e_ij = −β_i p_j γ_j/(p_i q_i); Frisch parameter −x/(x − Σ pγ)."""
    d, W, lp, lx, _ = _prep_system(df, shares, prices, total, None, False)
    P = np.exp(lp)
    x = np.exp(lx)
    n, G = W.shape
    Q = W * x[:, None] / P

    def resid(th, Sinv_chol=None):
        gam = th[:G]
        bet = np.r_[th[G:], 1 - th[G:].sum()]
        sub = P @ gam
        Wh = (P * gam) / x[:, None] + bet[None, :] * (1 - sub / x)[:, None]
        E = (W - Wh)[:, :-1]
        return (E @ Sinv_chol).ravel() if Sinv_chol is not None else E.ravel()
    th = np.r_[0.3 * Q.min(0), W.mean(0)[:-1]]
    C = None
    for _ in range(max_iter):
        r = optimize.least_squares(lambda t: resid(t, C), th, method="lm", max_nfev=20000)
        E = resid(r.x).reshape(n, G - 1)
        S = E.T @ E / n
        Cn = np.linalg.cholesky(np.linalg.inv(S))
        if np.max(np.abs(r.x - th)) < 1e-9:
            th = r.x
            break
        th, C = r.x, Cn
    J = r.jac
    V = np.linalg.pinv(J.T @ J)
    gam = th[:G]
    bet = np.r_[th[G:], 1 - th[G:].sum()]
    pm, xm = P.mean(0), x.mean()
    qm = Q.mean(0)
    wm = pm * qm / xm
    e = bet / wm
    M = -np.outer(bet, pm * gam) / (pm * qm)[:, None]
    M[np.diag_indices(G)] = (1 - bet) * gam / qm - 1
    H = M + np.outer(e, wm)
    names = [f"γ[{s}]" for s in shares] + [f"β[{s}]" for s in shares[:-1]]
    tab = pd.DataFrame({"Parameter": names + [f"β[{shares[-1]}]"], "Coef.": np.r_[th, bet[-1]],
                        "Std. Err.": np.r_[np.sqrt(np.maximum(np.diag(V), 0)), np.sqrt(np.ones(G - 1) @ V[G:, G:] @ np.ones(G - 1))]})
    frisch = -xm / (xm - pm @ gam)
    return {"table": tab, "gamma": gam, "beta": bet, "expenditure": pd.Series(e, index=shares), "marshallian": pd.DataFrame(M, index=shares, columns=shares),
            "hicksian": pd.DataFrame(H, index=shares, columns=shares), "frisch": float(frisch), "subsistence share": float(pm @ gam / xm), "N": n}


def rotterdam(df: pd.DataFrame, unit: str | None, time: str, shares: list, prices: list, total: str, symmetric: bool = True) -> dict:
    """Absolute-price Rotterdam model w̄_it Δln q_it = θ_i DQ_t + Σ_j π_ij Δln p_jt (+ e), DQ = Σ w̄ Δln q (Divisia volume),
    with homogeneity (Σ_j π_ij = 0) and symmetry imposed; expenditure elasticity θ_i/w_i, Hicksian π_ij/w_i, Marshallian
    π_ij/w_i − θ_i w_j/w_i. Differences are taken within `unit` (e.g. region) over `time`."""
    keys = ([unit] if unit else []) + [time]
    g = df.groupby(keys)[shares + prices + [total]].mean().reset_index().sort_values(keys)
    W = g[shares].to_numpy(float)
    W = W / W.sum(1, keepdims=True)
    lp = np.log(g[prices].to_numpy(float))
    lx = np.log(g[total].to_numpy(float))
    lq = np.log(W) + lx[:, None] - lp
    grp = g[unit].to_numpy() if unit else np.zeros(len(g))
    rows = [t for t in range(1, len(g)) if grp[t] == grp[t - 1]]
    rows = np.array(rows)
    if len(rows) < 3 * len(shares):
        raise ValueError(f"Only {len(rows)} first differences: the Rotterdam model needs more periods or groups.")
    wbar = 0.5 * (W[rows] + W[rows - 1])
    dlq = lq[rows] - lq[rows - 1]
    dlp = lp[rows] - lp[rows - 1]
    DQ = np.sum(wbar * dlq, 1)
    n, G = wbar.shape
    Xi = np.column_stack([DQ, dlp])
    P_eq = 1 + G

    def layout(i):
        return {"g": list(range(1, 1 + G))}
    R = _restrictions(G, P_eq, layout, symmetric, True)
    fit = sur([wbar[:, i] * dlq[:, i] for i in range(G - 1)], [Xi] * (G - 1), R)
    B = fit["b"].reshape(G - 1, P_eq)
    last = -B.sum(0)
    last[0] += 1
    Bf = np.vstack([B, last])
    th, pi = Bf[:, 0], Bf[:, 1:]
    w = W.mean(0)
    e = th / w
    H = pi / w[:, None]
    M = H - np.outer(th, w) / w[:, None]
    return {"theta": pd.Series(th, index=shares), "pi": pd.DataFrame(pi, index=shares, columns=shares), "expenditure": pd.Series(e, index=shares),
            "hicksian": pd.DataFrame(H, index=shares, columns=shares), "marshallian": pd.DataFrame(M, index=shares, columns=shares), "N": n}


# ==========================================================
# Consumption functions
# ==========================================================
def consumption(df: pd.DataFrame, time: str, c: str, y: str, log: bool = True) -> dict:
    """Keynesian (MPC, APC), permanent-income / Koyck (C_t = a + b Y_t + λ C_{t−1}: short-run MPC b, long-run b/(1 − λ)),
    Hall (1978) random-walk test (lagged ΔC and ΔY do not predict ΔC) and Campbell-Mankiw (1989) share λ of
    rule-of-thumb consumers (Δln C on Δln Y instrumented with lags 2–4 of Δln Y and Δln C)."""
    d = df[[time, c, y]].dropna().sort_values(time).reset_index(drop=True)
    C, Y = d[c].to_numpy(float), d[y].to_numpy(float)
    out = {}
    m, tab = _ols(C, np.column_stack([np.ones(len(C)), Y]), ["const", y], hac=4)
    out["keynes"] = {"table": tab, "MPC": float(m.params[1]), "APC": float(C.mean() / Y.mean()), "R²": float(m.rsquared),
                     "income elasticity": float(m.params[1] * Y.mean() / C.mean())}
    m2, tab2 = _ols(C[1:], np.column_stack([np.ones(len(C) - 1), Y[1:], C[:-1]]), ["const", y, f"{c}(t−1)"], hac=4)
    b, V = m2.params, m2.cov_params()
    out["pih"] = {"table": tab2, "MPC short run": float(b[1]), "MPC long run": _delta(lambda p: p[1] / (1 - p[2]), b, V), "λ": float(b[2])}
    dc = np.diff(np.log(C)) if log else np.diff(C)
    dy = np.diff(np.log(Y)) if log else np.diff(Y)
    X = np.column_stack([np.ones(len(dc) - 4)] + [dc[4 - j:-j] for j in range(1, 5)] + [dy[4 - j:-j] for j in range(1, 5)])
    m3, tab3 = _ols(dc[4:], X, ["const"] + [f"ΔC(t−{j})" for j in range(1, 5)] + [f"ΔY(t−{j})" for j in range(1, 5)], hac=4)
    R = np.zeros((8, 9)); R[np.arange(8), np.arange(1, 9)] = 1
    ft = m3.f_test(R)
    out["hall"] = {"table": tab3, "F": float(np.asarray(ft.fvalue).ravel()[0]), "p": float(ft.pvalue)}
    # Campbell-Mankiw by 2SLS
    yv, xv = dc[4:], dy[4:]
    Z = np.column_stack([np.ones(len(yv))] + [dy[4 - j:-j] for j in (2, 3, 4)] + [dc[4 - j:-j] for j in (2, 3, 4)])
    Xs = np.column_stack([np.ones(len(yv)), xv])
    Pz = Z @ np.linalg.pinv(Z)
    bh = np.linalg.solve(Xs.T @ Pz @ Xs, Xs.T @ Pz @ yv)
    e = yv - Xs @ bh
    A = np.linalg.inv(Xs.T @ Pz @ Xs)
    Xh = Pz @ Xs
    Vr = A @ (Xh.T * e ** 2) @ Xh @ A
    fs = np.linalg.lstsq(Z, xv, rcond=None)[0]
    r2fs = 1 - np.sum((xv - Z @ fs) ** 2) / np.sum((xv - xv.mean()) ** 2)
    out["cm"] = {"λ": float(bh[1]), "se": float(np.sqrt(Vr[1, 1])), "p": float(2 * stats.norm.sf(abs(bh[1] / np.sqrt(Vr[1, 1])))),
                 "first-stage R²": float(r2fs)}
    return out


# ==========================================================
# Welfare
# ==========================================================
def welfare_quaids(res: dict, dlnp: np.ndarray, x=None, lp0=None) -> pd.DataFrame:
    """Exact compensating and equivalent variation of a price change for each household from the (Q)AIDS expenditure
    function: ln e(p, u) = ln a(p) + b(p)·u / (1 − λ(p)·u) with λ(p) = Σ λ_i ln p_i (AIDS: λ = 0). CV = e(p1, u0) − x,
    EV = x − e(p0, u1); positive values are welfare losses. Also the first-order (Stone) and second-order
    (Friedman-Levinsohn) approximations, as % of expenditure."""
    if res["model"] == "laaids":
        raise ValueError("Exact welfare measures need AIDS or QUAIDS (LA-AIDS has no expenditure function).")
    al, gm, be, lam, a0 = res["alpha"], res["gamma"], res["beta"], res["lambda"], res["alpha0"]
    lp0 = res["lp"] if lp0 is None else lp0
    lx = res["lx"] if x is None else np.log(x)
    lp1 = lp0 + np.asarray(dlnp, float)[None, :]

    def lna(lp):
        return a0 + lp @ al + 0.5 * np.einsum("ti,ij,tj->t", lp, gm, lp)

    def bp(lp):
        return np.exp(lp @ be)

    def lamp(lp):
        return lp @ lam

    def u_of(lp, lx_):
        z = (lx_ - lna(lp)) / bp(lp)
        return 1 / (1 / z + lamp(lp)) if np.any(lam != 0) else z

    def lne(lp, u):
        return lna(lp) + (bp(lp) * u / (1 - lamp(lp) * u) if np.any(lam != 0) else bp(lp) * u)
    u0 = u_of(lp0, lx)
    u1 = u_of(lp1, lx)
    x0 = np.exp(lx)
    cv = np.exp(lne(lp1, u0)) - x0
    ev = x0 - np.exp(lne(lp0, u1))
    W = res["W"]
    first = W @ np.asarray(dlnp, float)
    E = res["elasticities"]["hicksian"].to_numpy()
    second = first + 0.5 * np.einsum("ti,ij,j,i->t", W, E, np.asarray(dlnp, float), np.asarray(dlnp, float))
    return pd.DataFrame({"expenditure": x0, "CV": cv, "EV": ev, "CV %": 100 * cv / x0, "EV %": 100 * ev / x0,
                         "first-order %": 100 * first, "second-order %": 100 * second})


def consumer_surplus_loglinear(q0: float, p0: float, p1: float, elasticity: float) -> float:
    """Change in Marshallian consumer surplus for a constant-elasticity demand q = A p^ε when the price moves p0 → p1
    (positive = loss)."""
    A = q0 / p0 ** elasticity
    if np.isclose(elasticity, -1):
        return float(A * np.log(p1 / p0))
    return float(A / (elasticity + 1) * (p1 ** (elasticity + 1) - p0 ** (elasticity + 1)))
