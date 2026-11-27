# ==========================================
# Aetheris Research - Marginal effects engine (shared by limited-dependent, count and survival models)
# ------------------------------------------
# Average marginal effects (AME) and marginal effects at the means (MEM) for ANY model that supplies
# a prediction function pred(X, θ) → (n,) or (n, J). Works in the units of the ORIGINAL variables
# through the design pipeline (logs, polynomials, interactions, dummies); discrete changes for binary
# regressors and factor levels; delta-method standard errors with numerical gradients in θ.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from analytics import regression as rg


def _is_binary(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce").dropna().unique()
    return len(v) == 2 and set(np.round(v, 12)) <= {0.0, 1.0}


def _rows(r, values: pd.DataFrame) -> np.ndarray:
    return rg.design_rows(r, values).to_numpy(float)


def marginal_effects(df: pd.DataFrame, r, pred, theta=None, V=None, at: str = "average",
                     outputs: list | None = None, variables: list | None = None, level: float = 0.95) -> pd.DataFrame:
    """Return a long table: Variable, Outcome, Effect, Std. Err., z, p-value, CI, Type."""
    theta = np.asarray(r.params if theta is None else theta, float)
    V = np.asarray(r.cov if V is None else V, float)
    spec = r.design.spec
    idx = r.design.index
    sample = df.loc[idx].copy()
    if at == "mean":
        sample = rg.baseline_values(df, r)
        for v in spec.x:  # arithmetic means for the MEM (Stata `margins, atmeans`)
            sample[v] = pd.to_numeric(df.loc[idx, v], errors="coerce").mean()
        for c in spec.categorical:  # factors at their sample shares → use the mode (most frequent level)
            sample[c] = df.loc[idx, c].astype(str).mode().iloc[0]
    variables = variables or (list(spec.x) + list(spec.categorical))
    q = stats.norm.ppf(0.5 + level / 2)

    def effect_fn(make_pair):
        """make_pair() -> (X_hi, X_lo, scale); effect = mean((pred(hi) − pred(lo)) / scale)."""
        Xh, Xl, scale = make_pair()

        def f(th):
            d = (np.atleast_2d(pred(Xh, th).T).T - np.atleast_2d(pred(Xl, th).T).T) / scale
            return np.mean(d, axis=0)
        return f

    rows = []
    for v in variables:
        pairs = []
        if v in spec.categorical:
            levels = r.design.info["levels"][v]
            for lv in levels[1:]:
                def mk(lv=lv, v=v, ref=levels[0]):
                    hi, lo = sample.copy(), sample.copy()
                    hi[v], lo[v] = lv, ref
                    return _rows(r, hi), _rows(r, lo), 1.0
                pairs.append((f"{v} = {lv} (vs {levels[0]})", "discrete", mk))
        elif _is_binary(df.loc[idx, v]):
            def mk(v=v):
                hi, lo = sample.copy(), sample.copy()
                hi[v], lo[v] = 1.0, 0.0
                return _rows(r, hi), _rows(r, lo), 1.0
            pairs.append((f"{v} (0 → 1)", "discrete", mk))
        else:
            sd = float(pd.to_numeric(df.loc[idx, v], errors="coerce").std()) or 1.0
            h = 1e-4 * sd

            def mk(v=v, h=h):
                hi, lo = sample.copy(), sample.copy()
                base = pd.to_numeric(sample[v], errors="coerce")
                hi[v], lo[v] = base + h, base - h
                return _rows(r, hi), _rows(r, lo), 2 * h
            pairs.append((v, "continuous", mk))
        for label, kind, mk in pairs:
            f = effect_fn(mk)
            est = f(theta)
            G = np.empty((len(theta), est.size))
            for j in range(len(theta)):
                d = 1e-6 * max(1.0, abs(theta[j]))
                tp, tm = theta.copy(), theta.copy()
                tp[j] += d
                tm[j] -= d
                G[j] = (f(tp) - f(tm)) / (2 * d)
            se = np.sqrt(np.clip(np.einsum("jo,jk,ko->o", G, V, G), 0, None))
            for o in range(est.size):
                z = est[o] / se[o] if se[o] > 0 else np.nan
                rows.append({"Variable": label, "Outcome": (outputs[o] if outputs else ""), "Type": kind,
                             "Effect": est[o], "Std. Err.": se[o], "z": z,
                             "p-value": 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan,
                             "CI low": est[o] - q * se[o], "CI high": est[o] + q * se[o]})
    out = pd.DataFrame(rows)
    if not outputs:
        out = out.drop(columns="Outcome")
    return out


def scenario(df: pd.DataFrame, r, pred, changes: dict, mode: str = "pct", theta=None, V=None,
             outputs: list | None = None, level: float = 0.95) -> pd.DataFrame:
    """Average predicted outcome(s) over the sample before / after a policy change, with delta-method CI
    of the change (e.g. adoption probability if credit access is extended to everyone)."""
    theta = np.asarray(r.params if theta is None else theta, float)
    V = np.asarray(r.cov if V is None else V, float)
    spec = r.design.spec
    s0 = df.loc[r.design.index].copy()
    s1 = s0.copy()
    for v, ch in changes.items():
        if v in spec.categorical:
            s1[v] = ch
        elif mode == "set":
            s1[v] = ch
        elif mode == "pct":
            s1[v] = pd.to_numeric(s0[v], errors="coerce") * (1 + ch / 100)
        else:
            s1[v] = pd.to_numeric(s0[v], errors="coerce") + ch
    X0, X1 = _rows(r, s0), _rows(r, s1)

    def m(X, th):
        return np.mean(np.atleast_2d(pred(X, th).T).T, axis=0)
    p0, p1 = m(X0, theta), m(X1, theta)
    G = np.empty((len(theta), p0.size))
    for j in range(len(theta)):
        d = 1e-6 * max(1.0, abs(theta[j]))
        tp, tm = theta.copy(), theta.copy()
        tp[j] += d
        tm[j] -= d
        G[j] = ((m(X1, tp) - m(X0, tp)) - (m(X1, tm) - m(X0, tm))) / (2 * d)
    se = np.sqrt(np.clip(np.einsum("jo,jk,ko->o", G, V, G), 0, None))
    q = stats.norm.ppf(0.5 + level / 2)
    out = pd.DataFrame({"Outcome": outputs or ["E[y]"], "Baseline": p0, "Scenario": p1, "Change": p1 - p0,
                        "Std. Err.": se, "CI low": p1 - p0 - q * se, "CI high": p1 - p0 + q * se,
                        "p-value": 2 * stats.norm.sf(np.abs((p1 - p0) / np.where(se > 0, se, np.nan)))})
    return out
