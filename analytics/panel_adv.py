# ==========================================
# Aetheris Research - Advanced panel models (Catalogue §1.15)
# Panel ARDL (PMG of Pesaran-Shin-Smith 1999, MG, DFE) with Hausman tests; panel cointegration (Kao 1999, Pedroni
# 1999/2004, Westerlund 2007) with Monte Carlo p-values for the actual N, T and number of regressors; interactive fixed
# effects (Bai 2009); panel threshold regression (Hansen 1999) with bootstrap; panel VAR (Love & Zicchino 2006,
# Abrigo & Love 2016) by GMM on forward-orthogonal deviations with impulse responses and Granger tests.
# ==========================================

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from analytics.panel import PanelData


def _units(pdata: PanelData, cols):
    d = pdata.df
    return [g[cols].to_numpy(float) for _, g in d.groupby("_id")]


def _lagm(a, L):
    out = np.full_like(a, np.nan, dtype=float)
    if L == 0:
        return a.astype(float).copy()
    out[L:] = a[:-L]
    return out


# ==========================================================
# Panel ARDL: PMG, MG, DFE
# ==========================================================
def _ardl_unit(Y, X, p, q):
    """Unit design for the ECM: Δy on [y_{-1}, x_{-1}], short-run [Δy lags 1..p-1, Δx lags 0..q-1], intercept."""
    y = Y
    dy = np.r_[np.nan, np.diff(y)]
    dX = np.vstack([np.full((1, X.shape[1]), np.nan), np.diff(X, axis=0)])
    lv = np.column_stack([_lagm(y, 1), _lagm(X, 1)])
    sr = [_lagm(dy, j) for j in range(1, p)] + [_lagm(dX[:, k], j) for k in range(X.shape[1]) for j in range(0, q)]
    W = np.column_stack(sr + [np.ones(len(y))]) if sr else np.ones((len(y), 1))
    ok = np.isfinite(dy) & np.isfinite(lv).all(1) & np.isfinite(W).all(1)
    return dy[ok], lv[ok, 0], lv[ok, 1:], W[ok]


def panel_ardl(pdata: PanelData, p: int = 1, q: int = 1, max_iter: int = 500, tol: float = 1e-10) -> dict:
    """Pooled mean group (ML, common long-run θ, heterogeneous φ_i and short run), mean group and dynamic fixed
    effects estimators of a panel ARDL(p, q) in error-correction form."""
    k = len(pdata.x)
    data = [_ardl_unit(u[:, 0], u[:, 1:], p, q) for u in _units(pdata, [pdata.y] + pdata.x)]
    N = len(data)
    # --- MG: unit-by-unit OLS of the unrestricted ECM
    mg_theta, mg_phi, mg_sr = [], [], []
    for dy, yl, xl, W in data:
        Z = np.column_stack([yl, xl, W])
        b = np.linalg.lstsq(Z, dy, rcond=None)[0]
        mg_phi.append(b[0]); mg_theta.append(-b[1:1 + k] / b[0]); mg_sr.append(b[1 + k:])
    mg_theta, mg_phi = np.array(mg_theta), np.array(mg_phi)
    # --- PMG by back-substitution (Pesaran, Shin & Smith 1999)
    theta = np.median(mg_theta, axis=0)
    for it in range(max_iter):
        num, den, s2 = np.zeros(k), np.zeros((k, k)), []
        phis = []
        for dy, yl, xl, W in data:
            ec = yl - xl @ theta
            Z = np.column_stack([ec, W])
            b = np.linalg.lstsq(Z, dy, rcond=None)[0]
            e = dy - Z @ b
            si = e @ e / len(e)
            phi = b[0]
            H = np.eye(len(dy)) - W @ np.linalg.pinv(W)
            num += phi * xl.T @ H @ (dy - phi * yl) / si
            den += phi ** 2 * xl.T @ H @ xl / si
            s2.append(si); phis.append(phi)
        new = -np.linalg.solve(den, num)
        if np.max(np.abs(new - theta)) < tol:
            theta = new
            break
        theta = new
    # final unit estimates and information matrix (θ, then for each unit φ_i and short-run γ_i)
    units, llf = [], 0.0
    blocks = []
    for dy, yl, xl, W in data:
        ec = yl - xl @ theta
        Z = np.column_stack([ec, W])
        b = np.linalg.lstsq(Z, dy, rcond=None)[0]
        e = dy - Z @ b
        si = e @ e / len(e)
        llf += -len(e) / 2 * (1 + np.log(2 * np.pi * si))
        units.append({"phi": b[0], "gamma": b[1:], "s2": si, "n": len(e)})
        J = np.column_stack([b[0] * xl, -ec[:, None] * -1, W])   # ∂(fitted)/∂(θ, φ, γ) up to sign
        J[:, :k] = -b[0] * xl
        blocks.append((J, si))
    m = blocks[0][0].shape[1] - k
    P = k + N * m
    I = np.zeros((P, P))
    for i, (J, si) in enumerate(blocks):
        idx = list(range(k)) + list(range(k + i * m, k + (i + 1) * m))
        I[np.ix_(idx, idx)] += J.T @ J / si
    V = np.linalg.inv(I)
    phi = np.array([u["phi"] for u in units])
    G = np.array([u["gamma"] for u in units])
    # mean of the heterogeneous short-run parameters (and φ) with MG-style standard errors
    sr_names = [f"L{j}.D.{pdata.y}" for j in range(1, p)] + [f"{'L' + str(j) + '.' if j else ''}D.{x}" for x in pdata.x for j in range(0, q)] + ["const"]
    pmg_lr = pd.DataFrame({"Variable": pdata.x, "Coef.": theta, "Std. Err.": np.sqrt(np.diag(V)[:k])})
    pmg_sr = [{"Variable": "ECT (φ)", "Coef.": phi.mean(), "Std. Err.": phi.std(ddof=1) / np.sqrt(N)}]
    for j, nm in enumerate(sr_names):
        pmg_sr.append({"Variable": nm, "Coef.": G[:, j].mean(), "Std. Err.": G[:, j].std(ddof=1) / np.sqrt(N)})
    pmg_sr = pd.DataFrame(pmg_sr)
    mg_lr = pd.DataFrame({"Variable": pdata.x, "Coef.": mg_theta.mean(0), "Std. Err.": mg_theta.std(0, ddof=1) / np.sqrt(N)})
    mg_sr = pd.DataFrame([{"Variable": "ECT (φ)", "Coef.": mg_phi.mean(), "Std. Err.": mg_phi.std(ddof=1) / np.sqrt(N)}] +
                         [{"Variable": nm, "Coef.": np.array(mg_sr)[:, j].mean(), "Std. Err.": np.array(mg_sr)[:, j].std(ddof=1) / np.sqrt(N)}
                          for j, nm in enumerate(sr_names)])
    # --- DFE: pooled ECM with unit intercepts (common slopes), delta-method long run
    Zs, ys, gs = [], [], []
    for i, (dy, yl, xl, W) in enumerate(data):
        Zs.append(np.column_stack([yl, xl, W[:, :-1]])); ys.append(dy); gs.append(np.full(len(dy), i))
    Z, yy, gg = np.vstack(Zs), np.concatenate(ys), np.concatenate(gs)
    Dm = pd.get_dummies(gg).to_numpy(float)
    Zf = np.column_stack([Z, Dm])
    XtXi = np.linalg.pinv(Zf.T @ Zf)
    b = XtXi @ Zf.T @ yy
    e = yy - Zf @ b
    Vd = e @ e / (len(e) - Zf.shape[1]) * XtXi
    ph = b[0]
    dfe_lr = []
    for j in range(k):
        th = -b[1 + j] / ph
        g = np.zeros(len(b)); g[1 + j] = -1 / ph; g[0] = b[1 + j] / ph ** 2
        dfe_lr.append({"Variable": pdata.x[j], "Coef.": th, "Std. Err.": float(np.sqrt(g @ Vd @ g))})
    dfe_lr = pd.DataFrame(dfe_lr)
    dfe_sr = pd.DataFrame([{"Variable": "ECT (φ)", "Coef.": ph, "Std. Err.": float(np.sqrt(Vd[0, 0]))}] +
                          [{"Variable": nm, "Coef.": b[1 + k + j], "Std. Err.": float(np.sqrt(Vd[1 + k + j, 1 + k + j]))}
                           for j, nm in enumerate(sr_names[:-1])])
    for t in (pmg_lr, pmg_sr, mg_lr, mg_sr, dfe_lr, dfe_sr):
        t["z"] = t["Coef."] / t["Std. Err."]
        t["p-value"] = 2 * stats.norm.sf(np.abs(t["z"]))
    # Hausman tests on the long-run coefficients (MG efficient alternative)
    def _haus(a, Va, bb, Vb):
        dlt = a - bb
        Vdl = Va - Vb
        H = abs(float(dlt @ np.linalg.pinv(Vdl) @ dlt))
        return {"stat": H, "df": len(dlt), "p": float(stats.chi2.sf(H, len(dlt)))}
    Vmg = np.cov(mg_theta.T, ddof=1).reshape(k, k) / N
    Vpmg = V[:k, :k]
    Vdfe = np.array([[np.nan]])
    h1 = _haus(mg_theta.mean(0), Vmg, theta, Vpmg)
    h2 = _haus(theta, Vpmg, dfe_lr["Coef."].to_numpy(), np.diag(dfe_lr["Std. Err."].to_numpy() ** 2))
    del Vdfe
    with np.errstate(invalid="ignore", divide="ignore"):
        half = np.where((phi < 0) & (phi > -1), np.log(0.5) / np.log(1 + np.clip(phi, -0.999999, -1e-6)), np.nan)
    return {"pmg": {"long_run": pmg_lr, "short_run": pmg_sr, "llf": llf, "iterations": it + 1, "phi_i": phi,
                    "half_life_i": half, "units": units},
            "mg": {"long_run": mg_lr, "short_run": mg_sr, "theta_i": mg_theta, "phi_i": mg_phi},
            "dfe": {"long_run": dfe_lr, "short_run": dfe_sr},
            "hausman_mg_pmg": h1, "hausman_pmg_dfe": h2, "N": N, "p": p, "q": q}


# ==========================================================
# Panel cointegration tests
# ==========================================================
def _nw(u, L):
    T = len(u)
    s = (u.T @ u) / T
    for l in range(1, L + 1):
        g = (u[l:].T @ u[:-l]) / T
        s = s + (1 - l / (L + 1)) * (g + g.T)
    return s


def _kao_stats(Ys, Xs):
    """Kao (1999) DF and ADF tests on LSDV residuals (homogeneous slopes, unit intercepts)."""
    N = len(Ys)
    T = len(Ys[0])
    yw = np.concatenate([y - y.mean() for y in Ys])
    Xw = np.vstack([X - X.mean(0) for X in Xs])
    b = np.linalg.lstsq(Xw, yw, rcond=None)[0]
    E = [y - y.mean() - (X - X.mean(0)) @ b for y, X in zip(Ys, Xs)]
    el = np.concatenate([e[:-1] for e in E])
    ec = np.concatenate([e[1:] for e in E])
    rho = (el @ ec) / (el @ el)
    v = ec - rho * el
    se = np.sqrt((v @ v) / (len(v) - 1) / (el @ el))
    t_rho = (rho - 1) / se
    # long-run and contemporaneous variances of (û, Δx)
    L = int(np.floor(4 * (T / 100) ** (2 / 9)))
    Sig, Om = 0, 0
    for y, X, e in zip(Ys, Xs, E):
        W = np.column_stack([np.diff(e), np.diff(X, axis=0)])       # innovations (Δê, Δx) under the null
        W = W - W.mean(0)
        Sig = Sig + W.T @ W / len(W)
        Om = Om + _nw(W, L)
    Sig, Om = Sig / N, Om / N
    s2v = Sig[0, 0] - Sig[0, 1:] @ np.linalg.solve(Sig[1:, 1:], Sig[1:, 0])
    s2ov = Om[0, 0] - Om[0, 1:] @ np.linalg.solve(Om[1:, 1:], Om[1:, 0])
    sv, sov = np.sqrt(s2v), np.sqrt(s2ov)
    DFr = (np.sqrt(N) * T * (rho - 1) + 3 * np.sqrt(N)) / np.sqrt(10.2)
    DFt = np.sqrt(1.25) * t_rho + np.sqrt(1.875 * N)
    DFrs = (np.sqrt(N) * T * (rho - 1) + 3 * np.sqrt(N) * s2v / s2ov) / np.sqrt(3 + 36 * s2v ** 2 / (5 * s2ov ** 2))
    DFts = (t_rho + np.sqrt(6 * N) * sv / (2 * sov)) / np.sqrt(s2ov / (2 * s2v) + 3 * s2v / (10 * s2ov))
    # ADF with one lag
    rows_y, rows_x = [], []
    for e in E:
        de = np.diff(e)
        rows_y.append(de[1:]); rows_x.append(np.column_stack([e[1:-1], de[:-1]]))
    Ya, Xa = np.concatenate(rows_y), np.vstack(rows_x)
    ba = np.linalg.lstsq(Xa, Ya, rcond=None)[0]
    ua = Ya - Xa @ ba
    Va = (ua @ ua) / (len(ua) - 2) * np.linalg.inv(Xa.T @ Xa)
    t_adf = ba[0] / np.sqrt(Va[0, 0])
    ADF = (t_adf + np.sqrt(6 * N) * sv / (2 * sov)) / np.sqrt(s2ov / (2 * s2v) + 3 * s2v / (10 * s2ov))
    return {"DF_rho": DFr, "DF_t": DFt, "DF*_rho": DFrs, "DF*_t": DFts, "ADF": ADF}


def kao(pdata: PanelData) -> pd.DataFrame:
    U = _units(pdata, [pdata.y] + pdata.x)
    if len({len(u) for u in U}) > 1:
        raise ValueError("Kao's test needs a balanced panel.")
    s = _kao_stats([u[:, 0] for u in U], [u[:, 1:] for u in U])
    return pd.DataFrame([{"Test": k, "Statistic": v, "p-value": float(stats.norm.cdf(v))} for k, v in s.items()])


def _pedroni_raw(Ys, Xs, trend=False):
    """Raw Pedroni (1999) statistics: panel v, ρ, PP(t), ADF(t); group ρ, PP(t), ADF(t)."""
    N, T = len(Ys), len(Ys[0])
    K = int(np.floor(4 * (T / 100) ** (2 / 9)))
    L11, s_i, s2_i, num_rho, den, num_pp, panel_adf_num, panel_adf_den = [], [], [], 0, 0, 0, 0, 0
    grho, gpp, gadf = 0, 0, 0
    sig_tilde_N, s_star = [], []
    store = []
    for y, X in zip(Ys, Xs):
        D = [np.ones(T)] + ([np.arange(1, T + 1, dtype=float)] if trend else [])
        Z = np.column_stack(D + [X])
        e = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        dy, dX = np.diff(y), np.diff(X, axis=0)
        eta = dy - dX @ np.linalg.lstsq(dX, dy, rcond=None)[0]
        L = (eta @ eta) / len(eta)
        for l in range(1, K + 1):
            L += 2 * (1 - l / (K + 1)) * (eta[l:] @ eta[:-l]) / len(eta)
        el, ec = e[:-1], e[1:]
        rho = (el @ ec) / (el @ el)
        mu = ec - rho * el
        s2 = (mu @ mu) / len(mu)
        lam = 0.0
        for l in range(1, K + 1):
            lam += (1 - l / (K + 1)) * (mu[l:] @ mu[:-l]) / len(mu)
        sig2 = s2 + 2 * lam
        # ADF residual regression with K lags
        de = np.diff(e)
        Xa = np.column_stack([e[K:-1]] + [de[K - j:len(de) - j] for j in range(1, K + 1)]) if K else e[:-1][:, None]
        Ya = de[K:]
        ba = np.linalg.lstsq(Xa, Ya, rcond=None)[0]
        ua = Ya - Xa @ ba
        s2s = (ua @ ua) / len(ua)
        store.append((el, ec, L, lam, s2, sig2, Xa[:, 0], Ya - Xa[:, 1:] @ ba[1:] if Xa.shape[1] > 1 else Ya, s2s))
    sig_NT = np.mean([s[5] / s[2] for s in store])
    s2_NT = np.mean([s[4] / s[2] for s in store])
    s2_star = np.mean([s[8] / s[2] for s in store])        # scale-invariant (ratio to L̂²₁₁ᵢ)
    A = sum(s[0] @ s[0] / s[2] for s in store)
    panel_v = T ** 2 * N ** 1.5 / A
    panel_rho = T * np.sqrt(N) * sum((s[0] @ (s[1] - s[0]) - T * s[3]) / s[2] for s in store) / A
    panel_pp = sum((s[0] @ (s[1] - s[0]) - T * s[3]) / s[2] for s in store) / np.sqrt(sig_NT * A)
    panel_adf = sum((s[6] @ s[7]) / s[2] for s in store) / np.sqrt(s2_star * sum((s[6] @ s[6]) / s[2] for s in store))
    group_rho = T / np.sqrt(N) * sum((s[0] @ (s[1] - s[0]) - T * s[3]) / (s[0] @ s[0]) for s in store)
    group_pp = 1 / np.sqrt(N) * sum((s[0] @ (s[1] - s[0]) - T * s[3]) / np.sqrt(s[5] * (s[0] @ s[0])) for s in store)
    group_adf = 1 / np.sqrt(N) * sum((s[6] @ s[7]) / np.sqrt(s[8] * (s[6] @ s[6])) for s in store)
    del L11, s_i, s2_i, num_rho, den, num_pp, panel_adf_num, panel_adf_den, grho, gpp, gadf, sig_tilde_N, s_star, s2_NT
    return {"Panel v": panel_v, "Panel ρ": panel_rho, "Panel PP": panel_pp, "Panel ADF": panel_adf,
            "Group ρ": group_rho, "Group PP": group_pp, "Group ADF": group_adf}


@lru_cache(maxsize=32)
def _pedroni_null(N, T, k, trend, reps=400, seed=1999):
    rng = np.random.default_rng(seed + N * 7 + T * 13 + k)
    out = []
    for _ in range(reps):
        W = np.cumsum(rng.standard_normal((N, T, k + 1)), axis=1)
        out.append(_pedroni_raw([w[:, 0] for w in W], [w[:, 1:] for w in W], trend))
    return pd.DataFrame(out)


def pedroni(pdata: PanelData, trend: bool = False, reps: int = 400) -> pd.DataFrame:
    """Pedroni (1999, 2004) residual-based panel cointegration tests (4 within-dimension, 3 between-dimension).
    Standardisation and p-values from a Monte Carlo of the null (no cointegration) for the actual N, T and k."""
    U = _units(pdata, [pdata.y] + pdata.x)
    if len({len(u) for u in U}) > 1:
        raise ValueError("Pedroni's tests need a balanced panel.")
    raw = _pedroni_raw([u[:, 0] for u in U], [u[:, 1:] for u in U], trend)
    sim = _pedroni_null(len(U), len(U[0]), len(pdata.x), trend, reps)
    rows = []
    for k_, v in raw.items():
        s = sim[k_]
        z = (v - s.mean()) / s.std(ddof=1)
        p = float((s >= v).mean()) if k_ == "Panel v" else float((s <= v).mean())
        rows.append({"Test": k_, "Statistic": v, "Standardised": z, "p-value": max(p, 1 / (len(s) + 1))})
    return pd.DataFrame(rows)


def _westerlund_raw(Ys, Xs, p=1, q=1, trend=False):
    N = len(Ys)
    Gt, Ga, a_list, se_list, num_pt, den_pt = 0, 0, [], [], 0, 0
    pooled_rows = []
    for y, X in zip(Ys, Xs):
        T = len(y)
        dy = np.r_[np.nan, np.diff(y)]
        dX = np.vstack([np.full((1, X.shape[1]), np.nan), np.diff(X, axis=0)])
        cols = [np.ones(T)] + ([np.arange(1, T + 1, dtype=float)] if trend else []) + [_lagm(y, 1)] + [_lagm(X[:, j], 1) for j in range(X.shape[1])]
        cols += [_lagm(dy, j) for j in range(1, p + 1)] + [_lagm(dX[:, k], j) for k in range(X.shape[1]) for j in range(0, q + 1)]
        Z = np.column_stack(cols)
        ok = np.isfinite(Z).all(1) & np.isfinite(dy)
        Z, yy = Z[ok], dy[ok]
        b = np.linalg.lstsq(Z, yy, rcond=None)[0]
        e = yy - Z @ b
        ia = 1 + (1 if trend else 0)
        s2 = e @ e / (len(e) - Z.shape[1])
        V = s2 * np.linalg.pinv(Z.T @ Z)
        a, se = b[ia], np.sqrt(V[ia, ia])
        # α_i(1) = 1 − Σ lags of Δy coefficients
        a1 = 1 - b[ia + 1 + X.shape[1]: ia + 1 + X.shape[1] + p].sum()
        Gt += a / se
        Ga += len(e) * a / a1
        pooled_rows.append((Z, yy, ia))
    Gt /= N
    Ga /= N
    # panel statistics: pooled α with unit-specific nuisance parameters (partial out all other regressors per unit)
    num = den = s2p = 0.0
    nobs = 0
    for Z, yy, ia in pooled_rows:
        O = np.delete(Z, ia, axis=1)
        M = lambda v: v - O @ np.linalg.lstsq(O, v, rcond=None)[0]  # noqa: E731
        yl, ym = M(Z[:, ia]), M(yy)
        num += yl @ ym; den += yl @ yl
        nobs += len(yy)
        pooled_rows_e = ym - (yl @ ym) / (yl @ yl) * yl
        s2p += pooled_rows_e @ pooled_rows_e
    ap = num / den
    s2p /= nobs
    se_p = np.sqrt(s2p / den)
    Pt = ap / se_p
    Tbar = nobs / N
    Pa = Tbar * ap
    return {"Gt": Gt, "Ga": Ga, "Pt": Pt, "Pa": Pa}


@lru_cache(maxsize=32)
def _westerlund_null(N, T, k, p, q, trend, reps=300, seed=2007):
    rng = np.random.default_rng(seed + N * 5 + T * 11 + k)
    out = []
    for _ in range(reps):
        W = np.cumsum(rng.standard_normal((N, T, k + 1)), axis=1)
        out.append(_westerlund_raw([w[:, 0] for w in W], [w[:, 1:] for w in W], p, q, trend))
    return pd.DataFrame(out)


def westerlund(pdata: PanelData, p: int = 1, q: int = 1, trend: bool = False, reps: int = 300) -> pd.DataFrame:
    """Westerlund (2007) error-correction-based tests: Gt, Ga (group mean; H1: some units cointegrated) and Pt, Pa
    (panel; H1: the panel is cointegrated). p-values from a Monte Carlo of the null for the actual N, T, k, lags."""
    U = _units(pdata, [pdata.y] + pdata.x)
    if len({len(u) for u in U}) > 1:
        raise ValueError("Westerlund's tests need a balanced panel.")
    raw = _westerlund_raw([u[:, 0] for u in U], [u[:, 1:] for u in U], p, q, trend)
    sim = _westerlund_null(len(U), len(U[0]), len(pdata.x), p, q, trend, reps)
    rows = []
    for k_, v in raw.items():
        s = sim[k_]
        rows.append({"Test": k_, "Statistic": v, "Standardised": (v - s.mean()) / s.std(ddof=1),
                     "p-value": max(float((s <= v).mean()), 1 / (len(s) + 1))})
    return pd.DataFrame(rows)


# ==========================================================
# Interactive fixed effects (Bai 2009)
# ==========================================================
def interactive_fe(pdata: PanelData, r: int = 1, max_iter: int = 1000, tol: float = 1e-9, additive: bool = True) -> dict:
    """Bai (2009) iterated principal components estimator of y_it = x_it'β + λ_i'F_t + ε_it (balanced panels),
    optionally with additive unit and time effects removed first. Homoskedastic standard errors (Bai 2009, Thm 3)."""
    d = pdata.df
    if not pdata.balanced:
        raise ValueError("Interactive fixed effects need a balanced panel.")
    N, T, k = pdata.N, pdata.T, len(pdata.x)
    Y = d.pivot(index="_t", columns="_id", values=pdata.y).to_numpy()
    Xs = [d.pivot(index="_t", columns="_id", values=v).to_numpy() for v in pdata.x]
    if additive:
        dm = lambda M: M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()  # noqa: E731
        Y = dm(Y); Xs = [dm(X) for X in Xs]
    Xmat = np.column_stack([X.T.ravel() for X in Xs])
    yv = Y.T.ravel()
    beta = np.linalg.lstsq(Xmat, yv, rcond=None)[0]
    for it in range(max_iter):
        Wm = Y - sum(b * X for b, X in zip(beta, Xs))
        w, Vv = np.linalg.eigh(Wm @ Wm.T / (N * T))
        F = Vv[:, ::-1][:, :r] * np.sqrt(T)
        MF = np.eye(T) - F @ F.T / T
        A = sum((MF @ np.column_stack([X[:, i] for X in Xs])).T @ np.column_stack([X[:, i] for X in Xs]) for i in range(N))
        bvec = sum(np.column_stack([X[:, i] for X in Xs]).T @ MF @ Y[:, i] for i in range(N))
        new = np.linalg.solve(A, bvec)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    Wm = Y - sum(b * X for b, X in zip(beta, Xs))
    Lam = Wm.T @ F / T
    E = Wm - F @ Lam.T
    dfr = N * T - (N + T) * r - k - ((N + T - 1) if additive else 0)
    s2 = (E ** 2).sum() / dfr
    LL = Lam.T @ Lam / N
    Zs = []
    Xi = [np.column_stack([X[:, i] for X in Xs]) for i in range(N)]
    MX = [MF @ X for X in Xi]
    for i in range(N):
        a_ik = Lam[i] @ np.linalg.solve(LL, Lam.T)
        Zs.append(MX[i] - sum(a_ik[j] * MX[j] for j in range(N)) / N)
    D = sum(Z.T @ Z for Z in Zs)
    V = s2 * np.linalg.inv(D)
    se = np.sqrt(np.diag(V))
    z = beta / se
    tab = pd.DataFrame({"Variable": pdata.x, "Coef.": beta, "Std. Err.": se, "z": z, "p-value": 2 * stats.norm.sf(np.abs(z))})
    # Bai-Ng (2002) IC_p2 for the number of factors (on the converged residual matrix)
    ic = []
    for rr in range(0, min(8, min(N, T) - 1) + 1):
        if rr == 0:
            V_ = (Wm ** 2).mean()
        else:
            w_, Vv_ = np.linalg.eigh(Wm @ Wm.T)
            Fr = Vv_[:, ::-1][:, :rr]
            V_ = ((Wm - Fr @ Fr.T @ Wm) ** 2).mean()
        ic.append({"Factors": rr, "IC_p2": np.log(V_) + rr * (N + T) / (N * T) * np.log(min(N, T))})
    return {"table": tab, "F": F, "Lambda": Lam, "iterations": it + 1, "sigma2": s2, "ic": pd.DataFrame(ic), "r": r}


# ==========================================================
# Panel threshold regression (Hansen 1999)
# ==========================================================
def _ssr_threshold(Yw_fun, y, X_reg, Xr, q, g, N, gammas):
    out = []
    for gm in gammas:
        Z = np.column_stack([X_reg, Xr * (q <= gm)[:, None], Xr * (q > gm)[:, None]])
        Zw = Yw_fun(Z)
        b = np.linalg.lstsq(Zw, y, rcond=None)[0]
        e = y - Zw @ b
        out.append(e @ e)
    return np.array(out)


def panel_threshold(pdata: PanelData, regime_vars: list, threshold: str, trim: float = 0.05, grid: int = 200,
                    boot: int = 300, seed: int = 1999) -> dict:
    """Hansen (1999) fixed-effect panel threshold model with one threshold: the slopes of `regime_vars` switch at
    q ≤ γ. Bootstrap p-value of the LR (F) test of no threshold and the LR-based 95% confidence interval for γ."""
    from analytics.panel import _within
    d = pdata.df
    g = d["_id"].to_numpy()
    N = pdata.N
    if threshold in d.columns:
        q = d[threshold].to_numpy(float)
    else:
        raise ValueError("The threshold variable must be one of the model variables.")
    yv = _within(d[pdata.y].to_numpy(float), g, N)
    others = [v for v in pdata.x if v not in regime_vars]
    Xo = d[others].to_numpy(float) if others else np.empty((len(d), 0))
    Xr = d[regime_vars].to_numpy(float)
    W = lambda Z: _within(Z, g, N)  # noqa: E731
    qs = np.unique(np.quantile(q, np.linspace(trim, 1 - trim, grid)))
    ssr = _ssr_threshold(W, yv, Xo, Xr, q, g, N, qs)
    j = int(np.argmin(ssr))
    gam = qs[j]
    Z0 = W(np.column_stack([Xo, Xr]))
    b0 = np.linalg.lstsq(Z0, yv, rcond=None)[0]
    e0 = yv - Z0 @ b0
    S0 = e0 @ e0
    n = len(yv)
    dfr = n - N - Z0.shape[1] - Xr.shape[1]
    s2 = ssr[j] / dfr
    F1 = (S0 - ssr[j]) / s2
    LR = (ssr - ssr[j]) / s2
    c95 = -2 * np.log(1 - np.sqrt(0.95))
    ci = qs[LR <= c95]
    # estimates at γ̂
    Z = np.column_stack([Xo, Xr * (q <= gam)[:, None], Xr * (q > gam)[:, None]])
    Zw = W(Z)
    XtXi = np.linalg.inv(Zw.T @ Zw)
    b = XtXi @ Zw.T @ yv
    e = yv - Zw @ b
    # cluster-robust (by unit) standard errors
    H = np.zeros((N, Zw.shape[1]))
    np.add.at(H, g, Zw * e[:, None])
    V = XtXi @ (H.T @ H) @ XtXi
    names = others + [f"{v} (q ≤ γ)" for v in regime_vars] + [f"{v} (q > γ)" for v in regime_vars]
    se = np.sqrt(np.diag(V))
    tab = pd.DataFrame({"Variable": names, "Coef.": b, "Std. Err. (cluster)": se, "t": b / se,
                        "p-value": 2 * stats.norm.sf(np.abs(b / se))})
    # bootstrap of F1 under H0 (Hansen 1999: resample unit residual vectors)
    rng = np.random.default_rng(seed)
    fitted0 = Z0 @ b0
    res_units = [e0[g == i] for i in range(N)]
    idx_units = [np.where(g == i)[0] for i in range(N)]
    Fb = []
    qs_b = qs[:: max(1, len(qs) // 60)]
    for _ in range(boot):
        draw = rng.integers(0, N, N)
        yb = fitted0.copy()
        ok = True
        for i in range(N):
            rb = res_units[draw[i]]
            if len(rb) != len(idx_units[i]):
                ok = False
                break
            yb[idx_units[i]] = fitted0[idx_units[i]] + rb
        if not ok:
            raise ValueError("The bootstrap needs a balanced panel.")
        b0b = np.linalg.lstsq(Z0, yb, rcond=None)[0]
        S0b = ((yb - Z0 @ b0b) ** 2).sum()
        ssb = _ssr_threshold(W, yb, Xo, Xr, q, g, N, qs_b)
        Fb.append((S0b - ssb.min()) / (ssb.min() / dfr))
    pv = float((np.array(Fb) >= F1).mean())
    return {"gamma": float(gam), "ci": (float(ci.min()), float(ci.max())), "F": float(F1), "p_boot": pv,
            "crit": {"10%": float(np.quantile(Fb, 0.9)), "5%": float(np.quantile(Fb, 0.95)), "1%": float(np.quantile(Fb, 0.99))},
            "table": tab, "LR": pd.DataFrame({"gamma": qs, "LR": LR}), "c95": c95, "share_low": float((q <= gam).mean())}


# ==========================================================
# Panel VAR (GMM on forward orthogonal deviations)
# ==========================================================
def _fod(M):
    """Helmert forward orthogonal deviations of a (T × k) block (rows with insufficient future data are NaN)."""
    T = len(M)
    out = np.full_like(M, np.nan, dtype=float)
    for t in range(T - 1):
        fut = M[t + 1:]
        c = np.sqrt((T - t - 1) / (T - t))
        out[t] = c * (M[t] - fut.mean(0))
    return out


def panel_var(pdata: PanelData, variables: list, p: int = 1, inst_lags: int | None = None) -> dict:
    """Panel VAR(p) of Abrigo & Love (2016): forward orthogonal deviations remove the unit effects; each equation is
    estimated by GMM (2SLS weighting) with the untransformed lags 1 … inst_lags of all variables as instruments
    ('GMM-style', zeros for missing). Cluster-robust covariance."""
    d = pdata.df
    K = len(variables)
    L = inst_lags or p
    Ys, Xs, Zs, G = [], [], [], []
    for i, g in d.groupby("_id"):
        M = g[variables].to_numpy(float)
        F = _fod(M)
        T = len(M)
        X = np.column_stack([_lagm(F, l) for l in range(1, p + 1)])
        Z = np.column_stack([_lagm(M, l) for l in range(1, L + 1)])
        ok = np.isfinite(F).all(1) & np.isfinite(X).all(1) & np.isfinite(Z).all(1)
        Ys.append(F[ok]); Xs.append(X[ok]); Zs.append(Z[ok]); G.append(np.full(ok.sum(), i))
    Y, X, Z, G = np.vstack(Ys), np.vstack(Xs), np.vstack(Zs), np.concatenate(G)
    ZZi = np.linalg.pinv(Z.T @ Z)
    Pz = Z @ ZZi @ Z.T if len(Z) < 4000 else None
    Xh = Z @ (ZZi @ (Z.T @ X))
    A = np.linalg.inv(Xh.T @ X)
    B = A @ Xh.T @ Y                         # (Kp × K)
    E = Y - X @ B
    # cluster-robust covariance of vec(B) (equation-stacked)
    units = np.unique(G)
    S = np.zeros((K * X.shape[1], K * X.shape[1]))
    for u in units:
        s = (Xh[G == u].T @ E[G == u])     # (Kp × K)
        v = s.T.ravel()
        S += np.outer(v, v)
    Ak = np.kron(np.eye(K), A)
    V = Ak @ S @ Ak.T
    del Pz
    names = [f"L{l}.{v}" for l in range(1, p + 1) for v in variables]
    rows = []
    for k_, eq in enumerate(variables):
        for j, nm in enumerate(names):
            b = B[j, k_]
            se = np.sqrt(V[k_ * len(names) + j, k_ * len(names) + j])
            rows.append({"Equation": eq, "Regressor": nm, "Coef.": b, "Std. Err.": se, "z": b / se,
                         "p-value": 2 * stats.norm.sf(abs(b / se))})
    Alist = [B[(l - 1) * K:l * K].T for l in range(1, p + 1)]
    comp = np.zeros((K * p, K * p))
    comp[:K] = np.hstack(Alist)
    if p > 1:
        comp[K:, :-K] = np.eye(K * (p - 1))
    mods = np.abs(np.linalg.eigvals(comp))
    Sigma = E.T @ E / len(E)
    # Granger (block Wald) per equation
    gr = []
    for k_, eq in enumerate(variables):
        for c, cv in enumerate(variables):
            if c == k_:
                continue
            idx = [k_ * len(names) + (l - 1) * K + c for l in range(1, p + 1)]
            bb = np.array([B[(l - 1) * K + c, k_] for l in range(1, p + 1)])
            W = float(bb @ np.linalg.solve(V[np.ix_(idx, idx)], bb))
            gr.append({"Equation": eq, "Excluded": cv, "Chi-sq": W, "df": p, "p-value": float(stats.chi2.sf(W, p))})
    return {"table": pd.DataFrame(rows), "A": Alist, "Sigma": Sigma, "V": V, "B": B, "roots": np.sort(mods)[::-1],
            "granger": pd.DataFrame(gr), "names": variables, "p": p, "nobs": len(Y), "n_units": len(units)}


def pvar_irf(res: dict, h: int = 10, draws: int = 200, seed: int = 11, level: float = 0.95):
    """Orthogonalised (Cholesky) impulse responses with Monte Carlo bands from N(B̂, V̂) (Love & Zicchino 2006)."""
    from analytics.var import ma_phi
    K, p = len(res["names"]), res["p"]
    P = np.linalg.cholesky(res["Sigma"])

    def irf_from(Alist):
        return ma_phi(Alist, h) @ P
    base = irf_from(res["A"])
    rng = np.random.default_rng(seed)
    vecB = res["B"].T.ravel()
    sims = []
    Lc = np.linalg.cholesky(res["V"] + 1e-12 * np.eye(len(vecB)))
    for _ in range(draws):
        bb = (vecB + Lc @ rng.standard_normal(len(vecB))).reshape(K, -1).T
        Al = [bb[(l - 1) * K:l * K].T for l in range(1, p + 1)]
        sims.append(irf_from(Al))
    sims = np.array(sims)
    a = (1 - level) / 2
    return base, np.quantile(sims, a, 0), np.quantile(sims, 1 - a, 0)
