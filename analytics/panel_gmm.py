# ==========================================
# Aetheris Research - Dynamic panel GMM (Catalogue §1.15)
# Arellano-Bond (1991) difference GMM and Blundell-Bond (1998) system GMM, one- and two-step, with collapsed or full
# instrument sets, Windmeijer (2005) corrected standard errors, Arellano-Bond AR(1)/AR(2) tests, Hansen J test and
# Wald tests. A faithful port of R plm::pgmm so that every number can be reproduced in R.
# ==========================================

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class GmmSpec:
    y: str
    ylags: list                       # lags of y in the equation, e.g. [1]
    x: dict                           # {var: [lags in the equation]}, e.g. {"k": [0], "l": [0, 1]}
    gmm: dict                         # {var: (min_lag, max_lag)} GMM-style instruments, e.g. {"y": (2, 99), "l": (2, 99)}
    iv: list = field(default_factory=list)   # variables (from x) used as standard instruments (in differences / levels)


@dataclass
class GmmResult:
    names: list
    coef: np.ndarray
    coef1: np.ndarray
    V: np.ndarray                    # robust (Windmeijer for two-step)
    V_nonrobust: np.ndarray
    n_units: int
    n_obs: int
    n_instr: int
    tests: dict
    spec: GmmSpec
    model: str
    transformation: str
    effect: str
    collapse: bool
    k_slopes: int
    extra: dict = field(default_factory=dict)

    def table(self, time_dummies: bool = False) -> pd.DataFrame:
        k = len(self.names) if time_dummies else self.k_slopes
        se = np.sqrt(np.diag(self.V))[:k]
        b = self.coef[:k]
        z = b / se
        return pd.DataFrame({"Variable": self.names[:k], "Coef.": b, "Std. Err.": se, "z": z,
                             "p-value": 2 * stats.norm.sf(np.abs(z)), "CI low": b - 1.96 * se, "CI high": b + 1.96 * se})


def _G(t):
    G = np.zeros((t, t))
    for i in range(t - 1):
        G[i, i], G[i, i + 1], G[i + 1, i] = 2, -1, -1
    G[t - 1, t - 1] = 2
    return G


def _FD(t):
    F = np.eye(t)[1:]
    for i in range(t - 1):
        F[i, i] = -1
    return F


def _FSM(t, fsm):
    if fsm == "G":
        return _G(t)
    if fsm == "I":
        return np.eye(t)
    if fsm == "full":
        return np.block([[_G(t - 1), _FD(t)], [_FD(t).T, np.eye(t)]])
    return np.block([[_G(t - 1), np.zeros((t - 1, t))], [np.zeros((t, t - 1)), np.eye(t)]])


def _lag(v, L):
    out = np.full(len(v), np.nan)
    if L == 0:
        return v.copy()
    out[L:] = v[:-L]
    return out


def _makegmm(x, g, TL1, collapse):
    T = len(x)
    rows = [x[max(1, y - g[1]) - 1:(y - g[0])] for y in range(TL1 + 1, T + 1)]
    if collapse:
        m = np.zeros((T - TL1, min(T - g[0], g[1] + 1 - g[0])))
        for i, r in enumerate(rows):
            r = r[::-1]
            m[i, :len(r)] = r
        return m
    lens = [len(r) for r in rows]
    tot = sum(lens)
    m = np.zeros((len(rows), tot))
    c = 0
    for i, r in enumerate(rows):
        m[i, c:c + len(r)] = r
        c += len(r)
    return m


def _makeW2(x, collapse):
    d = np.diff(x[:-1])
    return d[:, None] if collapse else np.diag(d)


def _bdiag(A, B):
    out = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
    out[:A.shape[0], :A.shape[1]] = A
    out[A.shape[0]:, A.shape[1]:] = B
    return out


def _inv(M):
    ev = np.linalg.eigvalsh((M + M.T) / 2)
    return np.linalg.pinv(M) if ev.min() < 1e-9 else np.linalg.inv(M)


def fit(df: pd.DataFrame, unit: str, time: str, spec: GmmSpec, model: str = "twosteps", transformation: str = "d",
        effect: str = "twoways", collapse: bool = False) -> GmmResult:
    """Difference ('d') or system ('ld') GMM, one- or two-step (R plm::pgmm conventions)."""
    units = sorted(df[unit].astype(str).unique())
    times = sorted(df[time].unique())
    T, N = len(times), len(units)
    full = pd.MultiIndex.from_product([units, times], names=[unit, time])
    d = df.assign(**{unit: df[unit].astype(str)}).set_index([unit, time]).reindex(full)
    main_max = max([max(spec.ylags)] + [max(v) for v in spec.x.values()] + [0])
    gmm_min = min(g[0] for g in spec.gmm.values())
    TL1 = max(main_max + 1, gmm_min)
    TL2 = max(main_max, gmm_min - 1)
    names = [f"L{L}.{spec.y}" if L else spec.y for L in spec.ylags] + \
            [f"L{L}.{v}" if L else v for v, lags in spec.x.items() for L in lags]
    iv_vars = [v for v in spec.x if v not in spec.gmm] if not spec.iv else spec.iv
    namesV = [str(t) for t in times]
    if effect == "twoways":
        namesV = namesV[TL1:] if transformation == "d" else ["(Intercept)"] + namesV[TL2 + 1:]
    elif transformation == "ld":
        namesV = ["(Intercept)"] + namesV[TL2 + 1:]
    yX_l, W_l = [], []
    if effect == "twoways":
        if transformation == "d":
            V1 = np.diff(np.eye(T - TL1 + 1), axis=0)[:, 1:]
        else:
            td = np.column_stack([np.ones(T), np.vstack([np.zeros((1, T - 1)), np.eye(T - 1)])])
            V2 = np.delete(td[TL2:], list(range(1, 1 + TL2)), axis=1)
            V1 = np.diff(V2, axis=0)
    for u in units:
        g = d.loc[u]
        yv = g[spec.y].to_numpy(float)
        cols = [yv] + [_lag(yv, L) for L in spec.ylags] + [_lag(g[v].to_numpy(float), L) for v, lags in spec.x.items() for L in lags]
        yX = np.column_stack(cols)
        yX1 = np.diff(yX, axis=0)[TL1 - 1:]
        W1 = np.column_stack([_makegmm(g[v].to_numpy(float), gl, TL1, collapse) for v, gl in spec.gmm.items()])
        Z1 = None
        if iv_vars:
            Z = np.column_stack([_lag(g[v].to_numpy(float), L) for v in iv_vars for L in spec.x.get(v, [0])])
            Z1 = np.diff(Z, axis=0)[TL1 - 1:]
        if transformation == "ld":
            parts = []
            for v in spec.gmm:
                w2 = _makeW2(g[v].to_numpy(float), collapse)
                parts.append(w2)
            W2 = np.column_stack(parts)
            if TL2 == 1:
                W2 = np.vstack([np.zeros((1, W2.shape[1])), W2])
            else:
                W2 = W2[W2.shape[0] - (T - TL2):]
            yX2 = yX[TL2:]
            Z2 = np.column_stack([_lag(g[v].to_numpy(float), L) for v in iv_vars for L in spec.x.get(v, [0])])[TL2:] if iv_vars else None
        if effect == "twoways":
            yX1 = np.column_stack([yX1, V1])
            if transformation == "d":
                W1 = np.column_stack([W1, V1])
            else:
                W2 = np.column_stack([W2, V2])
                yX2 = np.column_stack([yX2, V2])
        na = np.isnan(yX1).any(1)
        yX1[na] = 0
        W1 = np.nan_to_num(W1); W1[na] = 0
        if Z1 is not None:
            Z1 = np.nan_to_num(Z1); Z1[na] = 0
        if transformation == "ld":
            na2 = np.isnan(yX2).any(1)
            yX2[na2] = 0
            W2 = np.nan_to_num(W2); W2[na2] = 0
            if Z2 is not None:
                Z2 = np.nan_to_num(Z2); Z2[na2] = 0
            W1 = _bdiag(W1, W2)
            yX1 = np.vstack([yX1, yX2])
            if Z1 is not None:
                Z1 = _bdiag(Z1, Z2)
        if Z1 is not None:
            W1 = np.column_stack([W1, Z1])
        yX_l.append(yX1); W_l.append(W1)
    if effect == "twoways":
        names = names + namesV
    WX = sum(W.T @ yX[:, 1:] for W, yX in zip(W_l, yX_l))
    Wy = sum(W.T @ yX[:, 0] for W, yX in zip(W_l, yX_l))
    H = _FSM(T - TL1, "G") if transformation == "d" else _FSM(T - TL2, "full")
    A1 = _inv(sum(W.T @ H @ W for W in W_l)) * N
    B1 = np.linalg.inv(WX.T @ A1 @ WX)
    b1 = B1 @ (WX.T @ A1 @ Wy)
    res = [yX[:, 0] - yX[:, 1:] @ b1 for yX in yX_l]
    A2 = _inv(sum(W.T @ np.outer(e, e) @ W for W, e in zip(W_l, res)))
    if model == "twosteps":
        B2 = np.linalg.inv(WX.T @ A2 @ WX)
        b = B2 @ (WX.T @ A2 @ Wy)
        res2 = [yX[:, 0] - yX[:, 1:] @ b for yX in yX_l]
        Vnr = B2
    else:
        b, res2, Vnr, B2 = b1, res, B1, None
    # robust covariance (plm::vcovHC.pgmm, Windmeijer correction for two-step)
    SA2 = np.linalg.pinv(A2)
    Vr1 = B1 @ (WX.T @ A1 @ SA2 @ A1 @ WX) @ B1
    if model == "twosteps":
        K = yX_l[0].shape[1]
        We = sum(W.T @ e for W, e in zip(W_l, res2))
        D = []
        for k in range(1, K):
            wexkw = 0
            for W, yX, e1 in zip(W_l, yX_l, res):
                z = np.outer(yX[:, k], e1)
                exk = -z - z.T
                wexkw = wexkw + W.T @ exk @ W
            D.append(-B2 @ WX.T @ A2 @ wexkw @ A2 @ We)
        D = np.column_stack(D)
        Vr = B2 + D @ B2 + (D @ B2).T + D @ Vr1 @ D.T
    else:
        Vr = Vr1
    k_sl = len(spec.ylags) + sum(len(v) for v in spec.x.values())
    p = W_l[0].shape[1]
    z = sum(W.T @ e for W, e in zip(W_l, res2))
    J = float(z @ A2 @ z)
    tests = {"Hansen J": (J, p - len(b), float(stats.chi2.sf(J, p - len(b))))}
    Kt = len(namesV)
    A = A1 if model == "onestep" else A2
    B = Vnr
    for order in (1, 2):
        try:
            tests[f"AR({order})"] = _mtest(res2, yX_l, W_l, A, B, Vr, order, transformation, Kt)
        except Exception:  # noqa: BLE001
            pass
    bs = b[:k_sl]
    Vs = Vr[:k_sl, :k_sl]
    w = float(bs @ np.linalg.solve(Vs, bs))
    tests["Wald (slopes)"] = (w, k_sl, float(stats.chi2.sf(w, k_sl)))
    if effect == "twoways" and len(b) > k_sl:
        sel = list(range(k_sl, len(b)))
        if transformation == "ld":
            sel = sel[1:]
        if sel:
            bt = b[sel]
            wt = float(bt @ np.linalg.solve(Vr[np.ix_(sel, sel)], bt))
            tests["Wald (time dummies)"] = (wt, len(sel), float(stats.chi2.sf(wt, len(sel))))
    n_obs = int(sum((np.abs(yX[: (T - TL1)]).sum(1) > 0).sum() for yX in yX_l))
    return GmmResult(names, b, b1, Vr, Vnr, N, n_obs, p, tests, spec, model, transformation, effect, collapse, k_sl,
                     {"TL1": TL1, "TL2": TL2})


def _mtest(resid, yX_l, W_l, A, B, vv, order, transformation, Kt):
    if transformation == "d":
        r = resid
        rl = [np.r_[np.zeros(order), e[:len(e) - order]] for e in resid]
    else:
        r, rl = [], []
        for e in resid:
            keep = np.ones(len(e), bool)
            drop = np.arange(Kt - 1, 2 * Kt + 1)
            drop = drop[drop < len(e)]
            keep[drop] = False
            r.append(np.r_[e[keep], np.zeros(Kt)])
            rl.append(np.r_[np.zeros(order), e[:Kt - order - 1], np.zeros(Kt)])
    X = [yX[:, 1:] for yX in yX_l]
    EX = sum(el @ x for el, x in zip(rl, X))
    XZ = sum(W.T @ x for W, x in zip(W_l, X))
    EVE = sum((el @ e) ** 2 for e, el in zip(r, rl))
    ZVE = sum(W.T @ e * (e @ el) for W, e, el in zip(W_l, r, rl))
    num = sum(e @ el for e, el in zip(r, rl))
    den = EVE - 2 * EX @ B @ XZ.T @ A @ ZVE + EX @ vv @ EX
    st = float(num / np.sqrt(den))
    return (st, None, float(2 * stats.norm.sf(abs(st))))


def long_run(r: GmmResult) -> pd.DataFrame:
    """Long-run effects β/(1 − Σρ) of the regressors with delta-method standard errors."""
    ny = len(r.spec.ylags)
    rho = r.coef[:ny].sum()
    rows = []
    groups = {}
    j = ny
    for v, lags in r.spec.x.items():
        groups[v] = list(range(j, j + len(lags)))
        j += len(lags)
    for v, idx in groups.items():
        th = r.coef[idx].sum()
        lr = th / (1 - rho)
        g = np.zeros(len(r.coef))
        g[idx] = 1 / (1 - rho)
        g[:ny] = th / (1 - rho) ** 2
        se = float(np.sqrt(g @ r.V @ g))
        rows.append({"Variable": v, "Long-run effect": lr, "Std. Err.": se, "z": lr / se, "p-value": float(2 * stats.norm.sf(abs(lr / se)))})
    return pd.DataFrame(rows)
