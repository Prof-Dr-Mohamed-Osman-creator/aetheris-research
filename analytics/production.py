# ==========================================
# Aetheris Research - Production, cost & efficiency (v1.10, Catalogue §2.7)
# Production functions and their economics (elasticities, returns to scale, substitution, stages, isoquants),
# yield response and economic optimum, optimal input use, cost curves, scale and scope, translog cost / profit
# systems (restricted iterated SUR), stochastic frontiers (cross-section, Battese-Coelli 1995, Battese-Coelli 1992
# panel), meta-frontier, DEA and Malmquist, and TFP (Olley-Pakes, Levinsohn-Petrin, ACF, Törnqvist, Fisher).
# ==========================================

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy import optimize, stats

warnings.filterwarnings("ignore")

FORMS = {"cd": ("Cobb-Douglas", "كوب-دوجلاس"), "translog": ("Translog", "المتسامية اللوغاريتمية (ترانسلوج)"),
         "ces": ("CES", "مرونة الإحلال الثابتة (CES)"), "gl": ("Generalized Leontief", "ليونتيف المعممة"),
         "quadratic": ("Quadratic", "التربيعية"), "nquadratic": ("Normalized quadratic (inputs ÷ means)", "التربيعية المعيارية (المدخلات ÷ المتوسطات)"),
         "transcendental": ("Transcendental", "الترانسندنتالية"), "leontief": ("Leontief (fixed proportions)", "ليونتيف (نسب ثابتة)")}


@dataclass
class ProdResult:
    form: str
    inputs: list
    names: list
    b: np.ndarray
    V: np.ndarray
    df_resid: int
    predict: object              # f(X levels) -> y levels
    y: np.ndarray
    X: np.ndarray
    fitted: np.ndarray
    stats: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)

    @property
    def se(self):
        return np.sqrt(np.diag(self.V))

    def table(self) -> pd.DataFrame:
        se = self.se
        t = self.b / se
        return pd.DataFrame({"Parameter": self.names, "Coef.": self.b, "Std. Err.": se, "t": t,
                             "p-value": 2 * stats.t.sf(np.abs(t), self.df_resid)})


def _ols(y, Z, names, robust=False):
    import statsmodels.api as sm
    m = sm.OLS(y, Z).fit(cov_type="HC1" if robust else "nonrobust")
    return m.params, m.cov_params(), int(m.df_resid), m


def _sq_terms(L, inputs, half=True):
    cols, names = [], []
    for i, j in combinations_with_replacement(range(L.shape[1]), 2):
        f = (0.5 if (i == j and half) else 1.0)
        cols.append(f * L[:, i] * L[:, j])
        names.append(f"{inputs[i]}²" if i == j else f"{inputs[i]}×{inputs[j]}")
    return np.column_stack(cols), names


def production_function(df: pd.DataFrame, y: str, inputs: list, form: str = "cd", controls: list | None = None,
                        robust: bool = False) -> ProdResult:
    """Estimate a production function. Logs of inputs are centred at their sample means for the translog, so the
    first-order coefficients are output elasticities at the mean point."""
    controls = list(controls or [])
    d = df[[y] + inputs + controls].dropna()
    yv = d[y].to_numpy(float)
    X = d[inputs].to_numpy(float)
    C = d[controls].to_numpy(float) if controls else np.zeros((len(d), 0))
    n, k = X.shape
    if (X <= 0).any() or (yv <= 0).any():
        if form in ("cd", "translog", "ces", "transcendental", "gl"):
            raise ValueError("Output and inputs must be strictly positive for this functional form.")
    xm = X.mean(0)
    lxm = np.log(X).mean(0) if (X > 0).all() else None
    one = np.ones((n, 1))
    if form == "cd":
        Z = np.column_stack([one, np.log(X), C])
        names = ["const"] + [f"ln {v}" for v in inputs] + controls
        b, V, dfr, m = _ols(np.log(yv), Z, names, robust)

        def pred(Xn, b=b):
            return np.exp(b[0] + np.log(Xn) @ b[1:1 + k] + (C.mean(0) @ b[1 + k:] if controls else 0))
        res = ProdResult(form, inputs, names, np.asarray(b), np.asarray(V), dfr, pred, yv, X, np.exp(m.fittedvalues))
        # returns-to-scale test
        R = np.r_[0, np.ones(k), np.zeros(len(controls))]
        rts = R @ b
        se = np.sqrt(R @ V @ R)
        res.stats.update({"RTS": float(rts), "RTS se": float(se), "t (RTS = 1)": float((rts - 1) / se),
                          "p (RTS = 1)": float(2 * stats.t.sf(abs((rts - 1) / se), dfr)), "R²": float(m.rsquared),
                          "SSR": float(m.ssr), "N": n, "loglik": float(m.llf)})
        res.extra["model"] = m
        return res
    if form == "translog":
        L = np.log(X) - lxm
        Q, qn = _sq_terms(L, inputs)
        Z = np.column_stack([one, L, Q, C])
        names = ["const"] + [f"ln {v}" for v in inputs] + [f"½ ln {q}" if "²" in q else f"ln {q}" for q in qn] + controls
        b, V, dfr, m = _ols(np.log(yv), Z, names, robust)
        b = np.asarray(b)

        def pred(Xn, b=b):
            Ln = np.log(Xn) - lxm
            Qn, _ = _sq_terms(Ln, inputs)
            return np.exp(b[0] + Ln @ b[1:1 + k] + Qn @ b[1 + k:1 + k + Q.shape[1]] + (C.mean(0) @ b[1 + k + Q.shape[1]:] if controls else 0))
        res = ProdResult(form, inputs, names, b, np.asarray(V), dfr, pred, yv, X, np.exp(m.fittedvalues))
        # F test translog vs Cobb-Douglas
        mcd = production_function(df.loc[d.index], y, inputs, "cd", controls)
        F = ((mcd.stats["SSR"] - m.ssr) / Q.shape[1]) / (m.ssr / dfr)
        res.stats.update({"R²": float(m.rsquared), "SSR": float(m.ssr), "N": n, "loglik": float(m.llf),
                          "F (translog vs CD)": float(F), "p (translog vs CD)": float(stats.f.sf(F, Q.shape[1], dfr))})
        res.extra.update({"model": m, "lxm": lxm, "nq": Q.shape[1]})
        return res
    if form == "ces":
        lx = np.log(X)

        def f(p, Xn):
            lnA, nu, rho = p[0], p[1], p[2]
            dl = np.r_[p[3:], 1 - np.sum(p[3:])]
            if abs(rho) < 1e-6:
                return lnA + nu * (np.log(Xn) @ dl)
            z = (np.asarray(Xn, float) ** (-rho)) @ dl
            return lnA - nu / rho * np.log(z)
        cd = production_function(df.loc[d.index], y, inputs, "cd", None)
        nu0 = cd.stats["RTS"]
        dl0 = np.clip(cd.b[1:1 + k] / nu0, 0.05, 0.9)
        dl0 = dl0 / dl0.sum()
        best = None
        for rho0 in (-0.5, -0.2, 0.2, 0.5, 1.0):
            st = np.r_[cd.b[0], nu0, rho0, dl0[:-1]]
            try:
                r = optimize.least_squares(lambda p: f(p, X) - np.log(yv), st, method="lm", max_nfev=20000)
                if best is None or r.cost < best.cost:
                    best = r
            except Exception:  # noqa: BLE001
                continue
        p = best.x
        J = best.jac
        dfr = n - len(p)
        s2 = 2 * best.cost / dfr
        V = s2 * np.linalg.pinv(J.T @ J)
        names = ["ln A", "ν (returns to scale)", "ρ"] + [f"δ ({v})" for v in inputs[:-1]]
        res = ProdResult(form, inputs, names, p, V, dfr, lambda Xn, p=p: np.exp(f(p, Xn)), yv, X, np.exp(f(p, X)))
        sig = 1 / (1 + p[2])
        g = -1 / (1 + p[2]) ** 2
        res.stats.update({"σ (elasticity of substitution)": float(sig), "σ se": float(abs(g) * np.sqrt(V[2, 2])),
                          "RTS": float(p[1]), "R²": float(1 - 2 * best.cost / np.sum((np.log(yv) - np.log(yv).mean()) ** 2)),
                          "SSR": float(2 * best.cost), "N": n, "δ (last input)": float(1 - p[3:].sum()),
                          "Kmenta": kmenta(np.log(yv), lx) if k == 2 else None})
        return res
    if form == "leontief":
        ratio = X / yv[:, None]
        a_min = ratio.min(0)
        a_med = np.median(ratio, 0)
        pred = lambda Xn, a=a_med: np.min(np.asarray(Xn, float) / a, axis=1)  # noqa: E731
        res = ProdResult(form, inputs, [f"a ({v})" for v in inputs], a_med, np.diag((ratio.std(0) / np.sqrt(n)) ** 2), n - k, pred, yv, X, pred(X))
        limiting = np.argmin(X / a_med, axis=1)
        res.stats.update({"N": n})
        res.extra.update({"a_frontier": a_min, "a_median": a_med,
                          "limiting share": {inputs[j]: float(np.mean(limiting == j)) for j in range(k)}})
        return res
    if form in ("quadratic", "nquadratic"):
        Xs = X / xm if form == "nquadratic" else X
        Q, qn = _sq_terms(Xs, inputs)
        Z = np.column_stack([one, Xs, Q, C])
        names = ["const"] + list(inputs) + [f"½ {q}" if "²" in q else q for q in qn] + controls
        b, V, dfr, m = _ols(yv, Z, names, robust)
        b = np.asarray(b)

        def pred(Xn, b=b):
            Xq = np.asarray(Xn, float) / xm if form == "nquadratic" else np.asarray(Xn, float)
            Qn, _ = _sq_terms(Xq, inputs)
            return b[0] + Xq @ b[1:1 + k] + Qn @ b[1 + k:1 + k + Q.shape[1]] + (C.mean(0) @ b[1 + k + Q.shape[1]:] if controls else 0)
        res = ProdResult(form, inputs, names, b, np.asarray(V), dfr, pred, yv, X, np.asarray(m.fittedvalues))
        res.stats.update({"R²": float(m.rsquared), "SSR": float(m.ssr), "N": n})
        res.extra["model"] = m
        return res
    if form == "gl":
        cols, names = [], []
        for i, j in combinations_with_replacement(range(k), 2):
            cols.append(np.sqrt(X[:, i] * X[:, j]) * (1 if i == j else 2))
            names.append(f"β({inputs[i]})" if i == j else f"β({inputs[i]},{inputs[j]})")
        Z = np.column_stack(cols + ([C] if controls else []))
        names += controls
        b, V, dfr, m = _ols(yv, Z, names, robust)
        b = np.asarray(b)

        def pred(Xn, b=b):
            Xn = np.asarray(Xn, float)
            cc = [np.sqrt(Xn[:, i] * Xn[:, j]) * (1 if i == j else 2) for i, j in combinations_with_replacement(range(k), 2)]
            return np.column_stack(cc) @ b[:len(cc)] + (C.mean(0) @ b[len(cc):] if controls else 0)
        res = ProdResult(form, inputs, names, b, np.asarray(V), dfr, pred, yv, X, np.asarray(m.fittedvalues))
        res.stats.update({"R²": float(m.rsquared), "N": n})
        return res
    if form == "transcendental":
        Z = np.column_stack([one, np.log(X), X, C])
        names = ["const"] + [f"ln {v}" for v in inputs] + list(inputs) + controls
        b, V, dfr, m = _ols(np.log(yv), Z, names, robust)
        b = np.asarray(b)

        def pred(Xn, b=b):
            Xn = np.asarray(Xn, float)
            return np.exp(b[0] + np.log(Xn) @ b[1:1 + k] + Xn @ b[1 + k:1 + 2 * k] + (C.mean(0) @ b[1 + 2 * k:] if controls else 0))
        res = ProdResult(form, inputs, names, b, np.asarray(V), dfr, pred, yv, X, np.exp(m.fittedvalues))
        res.stats.update({"R²": float(m.rsquared), "N": n})
        return res
    raise ValueError(form)


def kmenta(lny, lx):
    """Kmenta (1967) linear approximation of the two-input CES: ln y = a + b1 ln x1 + b2 ln x2 + b3 (ln x1 − ln x2)²."""
    import statsmodels.api as sm
    Z = sm.add_constant(np.column_stack([lx[:, 0], lx[:, 1], (lx[:, 0] - lx[:, 1]) ** 2]))
    m = sm.OLS(lny, Z).fit()
    b = m.params
    nu = b[1] + b[2]
    delta = b[1] / nu
    rho = -2 * b[3] * nu / (b[1] * b[2]) if b[1] * b[2] != 0 else np.nan
    return {"ν": float(nu), "δ": float(delta), "ρ": float(rho), "σ": float(1 / (1 + rho)) if np.isfinite(rho) else np.nan}


# ----------------------------------------------------------------- derived economics
def _grad_hess(f, x, eps=1e-4):
    x = np.asarray(x, float)
    k = len(x)
    h = eps * np.maximum(np.abs(x), 1e-3)
    f0 = f(x[None, :])[0]
    g = np.zeros(k)
    H = np.zeros((k, k))
    for i in range(k):
        e = np.zeros(k); e[i] = h[i]
        fp, fm = f((x + e)[None, :])[0], f((x - e)[None, :])[0]
        g[i] = (fp - fm) / (2 * h[i])
        H[i, i] = (fp - 2 * f0 + fm) / h[i] ** 2
    for i in range(k):
        for j in range(i + 1, k):
            ei = np.zeros(k); ei[i] = h[i]
            ej = np.zeros(k); ej[j] = h[j]
            H[i, j] = H[j, i] = (f((x + ei + ej)[None, :])[0] - f((x + ei - ej)[None, :])[0] - f((x - ei + ej)[None, :])[0]
                                 + f((x - ei - ej)[None, :])[0]) / (4 * h[i] * h[j])
    return f0, g, H


def elasticities(r: ProdResult, at: str = "mean") -> dict:
    """Output elasticities, marginal and average products, RTS and Allen-Uzawa / Morishima substitution elasticities
    from the bordered Hessian of the fitted function (analytic for Cobb-Douglas and CES)."""
    X = r.X
    xp = X.mean(0) if at == "mean" else np.exp(np.log(X).mean(0))
    k = len(r.inputs)
    f0, g, H = _grad_hess(r.predict, xp)
    el = g * xp / f0
    B = np.zeros((k + 1, k + 1))
    B[0, 1:] = B[1:, 0] = g
    B[1:, 1:] = H
    detB = np.linalg.det(B)
    aes = np.full((k, k), np.nan)
    mes = np.full((k, k), np.nan)
    if abs(detB) > 1e-300:
        cof = np.linalg.inv(B).T * detB
        for i in range(k):
            for j in range(k):
                aes[i, j] = (xp @ g) / (xp[i] * xp[j]) * cof[i + 1, j + 1] / detB
        for i in range(k):
            for j in range(k):
                if i != j:
                    mes[i, j] = g[j] / xp[i] * cof[i + 1, j + 1] / detB - g[j] / xp[j] * cof[j + 1, j + 1] / detB
    if r.form == "cd":
        el = r.b[1:1 + k].copy()
        aes = np.ones((k, k))
        mes = np.where(np.eye(k) == 1, np.nan, 1.0)
    if r.form == "ces":
        s = r.stats["σ (elasticity of substitution)"]
        aes = np.full((k, k), s)
        mes = np.where(np.eye(k) == 1, np.nan, s)
    tab = pd.DataFrame({"Input": r.inputs, "Elasticity": el, "Marginal product": g, "Average product": f0 / xp,
                        "MP / AP": g / (f0 / xp)})
    return {"table": tab, "RTS": float(el.sum()), "AES": pd.DataFrame(aes, index=r.inputs, columns=r.inputs),
            "MES": pd.DataFrame(mes, index=r.inputs, columns=r.inputs), "point": xp, "y_at_point": f0}


def obs_elasticities(r: ProdResult) -> pd.DataFrame:
    """Output elasticities for every observation (translog: β_i + Σ_j β_ij ln x_j)."""
    k = len(r.inputs)
    if r.form == "translog":
        L = np.log(r.X) - r.extra["lxm"]
        B = np.zeros((k, k))
        idx = 1 + k
        for i, j in combinations_with_replacement(range(k), 2):
            B[i, j] = B[j, i] = r.b[idx]
            idx += 1
        E = r.b[1:1 + k] + L @ B
    elif r.form == "cd":
        E = np.tile(r.b[1:1 + k], (len(r.X), 1))
    else:
        E = np.zeros((len(r.X), k))
        for t in range(len(r.X)):
            f0, g, _ = _grad_hess(r.predict, r.X[t])
            E[t] = g * r.X[t] / f0
    out = pd.DataFrame(E, columns=r.inputs)
    out["RTS"] = E.sum(1)
    return out


def stages(r: ProdResult, j: int, grid: int = 200, upto: float = 2.5) -> dict:
    """Total, average and marginal product of input j with the others at their means; stage I ends where AP is
    maximal (MP = AP), stage II ends where MP = 0 (TP maximal)."""
    xm = r.X.mean(0)
    xs = np.linspace(r.X[:, j].max() * upto / grid, r.X[:, j].max() * upto, grid)
    Xg = np.tile(xm, (grid, 1))
    Xg[:, j] = xs
    tp = r.predict(Xg)
    ap = tp / xs
    mp = np.gradient(tp, xs)
    i_ap = int(np.argmax(ap))
    neg = np.where(mp <= 0)[0]
    x_mp0 = float(xs[neg[0]]) if len(neg) else np.nan
    x_ap = float(xs[i_ap]) if 0 < i_ap < grid - 1 else (np.nan if i_ap == 0 else float(xs[i_ap]))
    cur = r.X[:, j].mean()
    stage = "I" if np.isfinite(x_ap) and cur < x_ap else ("III" if np.isfinite(x_mp0) and cur > x_mp0 else "II")
    return {"x": xs, "TP": tp, "AP": ap, "MP": mp, "end stage I": x_ap, "end stage II": x_mp0, "stage at mean": stage, "mean": cur}


def isoquants(r: ProdResult, i: int, j: int, levels=None, grid: int = 80):
    xm = r.X.mean(0)
    a = np.linspace(r.X[:, i].min() * 0.6, r.X[:, i].max() * 1.2, grid)
    b = np.linspace(r.X[:, j].min() * 0.6, r.X[:, j].max() * 1.2, grid)
    A, Bm = np.meshgrid(a, b)
    Xg = np.tile(xm, (grid * grid, 1))
    Xg[:, i] = A.ravel()
    Xg[:, j] = Bm.ravel()
    Z = r.predict(Xg).reshape(grid, grid)
    if levels is None:
        q = np.quantile(r.y, [0.25, 0.5, 0.75])
        levels = q
    return a, b, Z, levels


def optimal_inputs(r: ProdResult, p_out: float, prices: dict) -> pd.DataFrame:
    """Value of the marginal product (VMP = p·MP at the sample geometric mean) against the input price: VMP/price > 1
    means the input is under-used (profit rises by using more), < 1 over-used. For Cobb-Douglas with decreasing
    returns, unconditional profit-maximising quantities are also reported."""
    e = elasticities(r, "geo")
    rows = []
    for idx, v in enumerate(r.inputs):
        w = prices.get(v)
        mp = e["table"]["Marginal product"].iloc[idx]
        vmp = p_out * mp
        rows.append({"Input": v, "Marginal product": mp, "VMP": vmp, "Price": w, "VMP / price": vmp / w if w else np.nan,
                     "Recommendation": ("increase" if vmp / w > 1.05 else ("reduce" if vmp / w < 0.95 else "near optimum")) if w else ""})
    out = pd.DataFrame(rows)
    if r.form == "cd" and r.stats.get("RTS", 1) < 1 and all(prices.get(v) for v in r.inputs):
        bb = r.b[1:1 + len(r.inputs)]
        s = bb.sum()
        A = np.exp(r.b[0])
        w = np.array([prices[v] for v in r.inputs])
        ystar = (A * np.prod((bb * p_out / w) ** bb)) ** (1 / (1 - s))
        out["Optimal quantity (CD)"] = bb * p_out * ystar / w
        out["Mean use"] = r.X.mean(0)
    return out


# ==========================================================
# Yield response and the economic optimum rate
# ==========================================================
YR_FORMS = {"quadratic": ("Quadratic", "التربيعية"), "quad_plateau": ("Quadratic-plateau", "التربيعية-الهضبة"),
            "linear_plateau": ("Linear-plateau", "الخطية-الهضبة"), "mitscherlich": ("Mitscherlich-Spillman", "ميتشرليخ-سبيلمان"),
            "sqrt": ("Square root", "الجذر التربيعي")}


def _yr_f(kind):
    def qp(x, a, b, c):
        x0 = -b / (2 * c) if c < 0 else np.inf
        xx = np.minimum(x, x0)
        return a + b * xx + c * xx ** 2
    return {"quadratic": lambda x, a, b, c: a + b * x + c * x ** 2, "quad_plateau": qp,
            "linear_plateau": lambda x, a, b, x0: a + b * np.minimum(x, x0),
            "mitscherlich": lambda x, A, B, R: A - B * R ** x, "sqrt": lambda x, a, b, c: a + b * np.sqrt(x) + c * x}[kind]


def yield_response(x, y, kind: str = "quad_plateau", p_out: float | None = None, p_in: float | None = None) -> dict:
    """Least-squares yield-response function and economic optimum rate (EOR: marginal product = input/output price
    ratio), agronomic maximum, and the EOR standard error by the delta method."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = _yr_f(kind)
    q = np.polyfit(x, y, 2)[::-1]
    starts = {"quadratic": [q], "quad_plateau": [q, [q[0], q[1] * 1.2, q[2] * 1.5]],
              "linear_plateau": [[y.min(), (y.max() - y.min()) / np.ptp(x) * 2, np.median(x)], [y.min(), (y.max() - y.min()) / np.ptp(x), x.max() * 0.7]],
              "mitscherlich": [[y.max(), y.max() - y.min(), 0.99], [y.max() * 1.05, y.max() - y.min(), 0.98]],
              "sqrt": [[y.min(), 0.1, 0.0]]}[kind]
    best = None
    for st in starts:
        try:
            r = optimize.least_squares(lambda p: f(x, *p) - y, st, method="trf", max_nfev=20000, x_scale="jac")
            if best is None or r.cost < best.cost:
                best = r
        except Exception:  # noqa: BLE001
            continue
    p = best.x
    n, k = len(y), len(p)
    J = best.jac
    s2 = 2 * best.cost / (n - k)
    V = s2 * np.linalg.pinv(J.T @ J)
    names = {"quadratic": ["a", "b", "c"], "quad_plateau": ["a", "b", "c"], "linear_plateau": ["a", "b", "x₀ (join point)"],
             "mitscherlich": ["A (asymptote)", "B", "R"], "sqrt": ["a", "b", "c"]}[kind]
    tab = pd.DataFrame({"Parameter": names, "Estimate": p, "Std. Err.": np.sqrt(np.diag(V))})
    tab["t"] = tab["Estimate"] / tab["Std. Err."]
    tab["p-value"] = 2 * stats.t.sf(np.abs(tab["t"]), n - k)
    ratio = (p_in / p_out) if p_in and p_out else None

    def opt(pp):
        a, b, c = pp
        if kind in ("quadratic", "quad_plateau"):
            amax = -b / (2 * c) if c < 0 else np.inf
            eor = (ratio - b) / (2 * c) if ratio is not None and c < 0 else np.nan
            return amax, (min(max(eor, 0), amax) if np.isfinite(eor) else np.nan)
        if kind == "linear_plateau":
            return c, (c if ratio is not None and b > ratio else 0.0) if ratio is not None else np.nan
        if kind == "mitscherlich":
            A, B, R = pp
            if not 0 < R < 1:
                return np.inf, np.nan
            eor = np.log(ratio / (-B * np.log(R))) / np.log(R) if ratio is not None else np.nan
            return np.inf, max(eor, 0) if np.isfinite(eor) else np.nan
        a, b, c = pp
        amax = (b / (-2 * c)) ** 2 if c < 0 else np.inf
        eor = (b / (2 * (ratio - c))) ** 2 if ratio is not None and ratio - c > 0 else np.nan
        return amax, eor
    amax, eor = opt(p)
    eor_se = np.nan
    if np.isfinite(eor):
        gr = np.zeros(k)
        for i in range(k):
            dp = np.zeros(k); dp[i] = 1e-6 * max(1, abs(p[i]))
            gr[i] = (opt(p + dp)[1] - opt(p - dp)[1]) / (2 * dp[i])
        eor_se = float(np.sqrt(gr @ V @ gr))
    out = {"table": tab, "params": p, "V": V, "kind": kind, "f": lambda xx, p=p: f(np.asarray(xx, float), *p),
           "RSS": float(2 * best.cost), "R²": float(1 - 2 * best.cost / np.sum((y - y.mean()) ** 2)),
           "AIC": float(n * np.log(2 * best.cost / n) + 2 * (k + 1)), "agronomic max": float(amax), "EOR": float(eor) if np.isfinite(eor) else np.nan,
           "EOR se": eor_se, "ratio": ratio}
    if np.isfinite(eor) and p_out:
        out["yield at EOR"] = float(f(np.array([eor]), *p)[0])
        if np.isfinite(amax):
            out["yield at max"] = float(f(np.array([amax]), *p)[0])
            out["profit gain vs max"] = float(p_out * (out["yield at EOR"] - out["yield at max"]) - p_in * (eor - amax))
    return out


# ==========================================================
# Cost curves, scale and scope
# ==========================================================
def cubic_cost(q, c) -> dict:
    """C = a + bQ + cQ² + dQ³ by OLS: fixed and variable cost, AVC, ATC, MC, shutdown point (min AVC),
    efficient scale (min ATC = break-even price) and scale economies 1 − MC/AC."""
    import statsmodels.api as sm
    q, c = np.asarray(q, float), np.asarray(c, float)
    Z = sm.add_constant(np.column_stack([q, q ** 2, q ** 3]))
    m = sm.OLS(c, Z).fit()
    a, b, cc, d = m.params
    q_avc = -cc / (2 * d) if d > 0 else np.nan
    atc = lambda x: (a + b * x + cc * x ** 2 + d * x ** 3) / x  # noqa: E731
    mc = lambda x: b + 2 * cc * x + 3 * d * x ** 2  # noqa: E731
    r = optimize.minimize_scalar(atc, bounds=(q.min() * 0.2, q.max() * 3), method="bounded")
    q_atc = float(r.x)
    tab = pd.DataFrame({"Parameter": ["a (fixed cost)", "b", "c", "d"], "Coef.": m.params, "Std. Err.": m.bse, "t": m.tvalues, "p-value": m.pvalues})
    return {"table": tab, "params": m.params, "R²": float(m.rsquared), "q shutdown": float(q_avc), "shutdown price": float(b + cc * q_avc + d * q_avc ** 2) if np.isfinite(q_avc) else np.nan,
            "q efficient": q_atc, "break-even price": float(atc(q_atc)), "ATC": atc, "MC": mc,
            "AVC": lambda x: b + cc * x + d * x ** 2, "scale economies at mean": float(1 - mc(q.mean()) / atc(q.mean())), "model": m}


def long_run_cost(df: pd.DataFrame, cost: str, q: str, prices: list | None = None) -> dict:
    """ln C = a + b ln Q + ½ c (ln Q)² (+ ln input prices, homogeneity imposed by normalising with the last price):
    cost elasticity ε = b + c ln Q; economies of scale where ε < 1; minimum efficient scale where ε = 1."""
    import statsmodels.api as sm
    prices = list(prices or [])
    d = df[[cost, q] + prices].dropna()
    lq = np.log(d[q].to_numpy(float))
    lqm = lq.mean()
    lc = np.log(d[cost].to_numpy(float))
    cols = [lq - lqm, 0.5 * (lq - lqm) ** 2]
    names = ["ln Q", "½ (ln Q)²"]
    if prices:
        lw = np.log(d[prices].to_numpy(float))
        lc = lc - lw[:, -1]
        for j in range(len(prices) - 1):
            cols.append(lw[:, j] - lw[:, -1])
            names.append(f"ln ({prices[j]} / {prices[-1]})")
    Z = sm.add_constant(np.column_stack(cols))
    m = sm.OLS(lc, Z).fit()
    b, c = m.params[1], m.params[2]
    eps = b + c * (lq - lqm)
    mes = float(np.exp(lqm + (1 - b) / c)) if c > 0 else np.nan
    tab = pd.DataFrame({"Parameter": ["const"] + names, "Coef.": m.params, "Std. Err.": m.bse, "t": m.tvalues, "p-value": m.pvalues})
    return {"table": tab, "elasticity at mean": float(b), "elasticity": eps, "share with scale economies": float(np.mean(eps < 1)),
            "minimum efficient scale": mes, "RTS at mean": float(1 / b), "model": m, "lq": lq}


def scope_economies(df: pd.DataFrame, cost: str, q1: str, q2: str) -> dict:
    """Quadratic two-output cost C = a + a1 q1 + a2 q2 + ½ b11 q1² + ½ b22 q2² + b12 q1 q2; economies of scope
    SC = [C(q1, 0) + C(0, q2) − C(q1, q2)] / C(q1, q2) at the means of multi-product firms (fixed cost shared),
    cost complementarity when b12 < 0, and product-specific scale economies."""
    import statsmodels.api as sm
    d = df[[cost, q1, q2]].dropna()
    a, b = d[q1].to_numpy(float), d[q2].to_numpy(float)
    Z = sm.add_constant(np.column_stack([a, b, 0.5 * a ** 2, 0.5 * b ** 2, a * b]))
    m = sm.OLS(d[cost].to_numpy(float), Z).fit()
    p = m.params
    C = lambda x1, x2: p[0] + p[1] * x1 + p[2] * x2 + 0.5 * p[3] * x1 ** 2 + 0.5 * p[4] * x2 ** 2 + p[5] * x1 * x2  # noqa: E731
    mm = (a > 0) & (b > 0)
    m1, m2 = a[mm].mean(), b[mm].mean()
    sc = (C(m1, 0) + C(0, m2) - C(m1, m2)) / C(m1, m2)
    tab = pd.DataFrame({"Parameter": ["a0 (fixed)", f"a1 ({q1})", f"a2 ({q2})", f"b11", f"b22", f"b12 ({q1}×{q2})"], "Coef.": p,
                        "Std. Err.": m.bse, "t": m.tvalues, "p-value": m.pvalues})
    return {"table": tab, "SC": float(sc), "complementarity": bool(p[5] < 0), "p (b12)": float(m.pvalues[5]), "at": (m1, m2), "R²": float(m.rsquared)}


# ==========================================================
# Restricted iterated SUR and translog cost / profit share systems
# ==========================================================
def sur(ys: list, Xs: list, R=None, r=None, iterate: bool = True, max_iter: int = 200, tol: float = 1e-10) -> dict:
    """Seemingly unrelated regressions with linear restrictions R b = r; iterated FGLS (= ML under normality),
    residual covariance without degrees-of-freedom correction (as systemfit methodResidCov = 'noDfCor')."""
    G = len(ys)
    n = len(ys[0])
    ks = [X.shape[1] for X in Xs]
    K = sum(ks)
    off = np.r_[0, np.cumsum(ks)]
    if R is not None:
        from scipy.linalg import null_space
        Rm_ = np.asarray(R, float)
        rv_ = np.zeros(Rm_.shape[0]) if r is None else np.asarray(r, float)
        Nr = null_space(Rm_)
        b0r = np.linalg.lstsq(Rm_, rv_, rcond=None)[0]
    XX = [[Xs[g].T @ Xs[h] for h in range(G)] for g in range(G)]
    Xy = [[Xs[g].T @ ys[h] for h in range(G)] for g in range(G)]

    def gls(Si):
        A = np.zeros((K, K))
        bvec = np.zeros(K)
        for g in range(G):
            for h in range(G):
                A[off[g]:off[g + 1], off[h]:off[h + 1]] = Si[g, h] * XX[g][h]
                bvec[off[g]:off[g + 1]] += Si[g, h] * Xy[g][h]
        if R is None:
            Ai = np.linalg.inv(A)
            return Ai @ bvec, Ai
        # restricted GLS by re-parameterising b = b0 + N θ with R b0 = r and N spanning the null space of R
        Ath = Nr.T @ A @ Nr
        th = np.linalg.solve(Ath, Nr.T @ (bvec - A @ b0r))
        return b0r + Nr @ th, Nr @ np.linalg.inv(Ath) @ Nr.T

    def resid(b):
        return np.column_stack([ys[g] - Xs[g] @ b[off[g]:off[g + 1]] for g in range(G)])
    S = np.eye(G)
    b, V = gls(S)
    for it in range(max_iter):
        E = resid(b)
        S = E.T @ E / n
        b_new, V = gls(np.linalg.inv(S))
        if not iterate or np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new
    E = resid(b)
    S = E.T @ E / n
    _, ld = np.linalg.slogdet(S)
    ll = -n * G / 2 * (1 + np.log(2 * np.pi)) - n / 2 * ld
    return {"b": b, "V": V, "Sigma": S, "resid": E, "loglik": float(ll), "iterations": it + 1, "ks": ks}


def translog_system(df: pd.DataFrame, kind: str, value: str, prices: list, shares: list, q: str | None = None,
                    fixed: list | None = None) -> dict:
    """Translog cost (kind='cost') or profit (kind='profit') function estimated jointly with its share equations by
    iterated SUR, imposing symmetry and linear homogeneity in prices (prices normalised by the last price; last share
    dropped). Variables are mean-centred in logs. Returns parameters, Allen and Morishima substitution elasticities,
    price elasticities of input demand (and output supply for profit), scale economies (cost) and curvature checks."""
    fixed = list(fixed or [])
    cols = [value] + prices + shares + ([q] if q else []) + fixed
    d = df[cols].dropna()
    d = d[(d[[value] + prices] > 0).all(1)]
    n = len(d)
    J = len(prices)
    lw = np.log(d[prices].to_numpy(float))
    lw = lw - lw.mean(0)
    S = d[shares].to_numpy(float)
    lv = np.log(d[value].to_numpy(float))
    lv = lv - lw[:, -1]
    lwn = lw[:, :-1] - lw[:, [-1]]
    Q = []
    qn = []
    if q:
        lq = np.log(d[q].to_numpy(float))
        Q.append(lq - lq.mean())
        qn.append(q)
    for f_ in fixed:
        lz = np.log(d[f_].to_numpy(float))
        Q.append(lz - lz.mean())
        qn.append(f_)
    Q = np.column_stack(Q) if Q else np.zeros((n, 0))
    m = Q.shape[1]
    # parameters: a0, a_1..a_{J-1}, g_ij (i<=j, i,j<J), aq_k, aqq_kl (k<=l), gq_ik (i<J)
    pairs = [(i, j) for i in range(J - 1) for j in range(i, J - 1)]
    qpairs = [(k, l) for k in range(m) for l in range(k, m)]
    names = ["α0"] + [f"α({prices[i]})" for i in range(J - 1)] + [f"γ({prices[i]},{prices[j]})" for i, j in pairs] + \
            [f"α({qn[k]})" for k in range(m)] + [f"α({qn[k]},{qn[l]})" for k, l in qpairs] + [f"γ({prices[i]},{qn[k]})" for i in range(J - 1) for k in range(m)]
    P = len(names)

    def cost_row(t):
        x = [1.0] + list(lwn[t])
        for i, j in pairs:
            x.append(0.5 * lwn[t, i] ** 2 if i == j else lwn[t, i] * lwn[t, j])
        x += list(Q[t])
        for k, l in qpairs:
            x.append(0.5 * Q[t, k] ** 2 if k == l else Q[t, k] * Q[t, l])
        for i in range(J - 1):
            for k in range(m):
                x.append(lwn[t, i] * Q[t, k])
        return x
    Xc = np.array([cost_row(t) for t in range(n)])
    # share equations for i < J : S_i = α_i + Σ_j γ_ij lwn_j + Σ_k γ_ik q_k  — expressed with the same parameter vector
    Xs = []
    for i in range(J - 1):
        X = np.zeros((n, P))
        X[:, 1 + i] = 1.0
        for p_, (a_, b_) in enumerate(pairs):
            col = 1 + (J - 1) + p_
            if a_ == i and b_ == i:
                X[:, col] = lwn[:, i]
            elif a_ == i:
                X[:, col] = lwn[:, b_]
            elif b_ == i:
                X[:, col] = lwn[:, a_]
        base = 1 + (J - 1) + len(pairs) + m + len(qpairs)
        for k in range(m):
            X[:, base + i * m + k] = Q[:, k]
        Xs.append(X)
    # stack: cost equation and J-1 share equations share the same parameters → restrictions b_eq = b_cost
    ys = [lv] + [S[:, i] for i in range(J - 1)]
    Xl = [Xc] + Xs
    G = len(ys)
    Rr = []
    for g in range(1, G):
        for p_ in range(P):
            row = np.zeros(G * P)
            row[p_] = 1
            row[g * P + p_] = -1
            Rr.append(row)
    # share-equation columns that are always zero make their own parameters unidentified: tie them via restrictions above
    out = sur(ys, Xl, np.array(Rr))
    b = out["b"][:P]
    Vb = out["V"][:P, :P]
    tab = pd.DataFrame({"Parameter": names, "Coef.": b, "Std. Err.": np.sqrt(np.maximum(np.diag(Vb), 0))})
    tab["z"] = tab["Coef."] / tab["Std. Err."]
    tab["p-value"] = 2 * stats.norm.sf(np.abs(tab["z"]))
    # full gamma matrix (J × J) by homogeneity
    Gm = np.zeros((J, J))
    for p_, (i, j) in enumerate(pairs):
        Gm[i, j] = Gm[j, i] = b[1 + (J - 1) + p_]
    Gm[:J - 1, J - 1] = -Gm[:J - 1, :J - 1].sum(1)
    Gm[J - 1, :J - 1] = Gm[:J - 1, J - 1]
    Gm[J - 1, J - 1] = -Gm[J - 1, :J - 1].sum()
    alpha = np.r_[b[1:J], 1 - b[1:J].sum()] if kind == "cost" else np.r_[b[1:J], 1 - b[1:J].sum()]
    Sm = alpha                                          # fitted shares at the (centred) mean point
    AES = (Gm + np.outer(Sm, Sm)) / np.outer(Sm, Sm)
    AES[np.diag_indices(J)] = (np.diag(Gm) + Sm ** 2 - Sm) / Sm ** 2
    eta = AES * Sm[None, :]                              # η_ij = σ_ij S_j  (cost: input demand; profit: netput)
    if kind == "profit":
        eta = Gm / Sm[:, None] + Sm[None, :] - np.eye(J)
    MES = np.full((J, J), np.nan)
    for i in range(J):
        for j in range(J):
            if i != j:
                MES[i, j] = eta[j, i] - eta[i, i]
    res = {"table": tab, "b": b, "V": Vb, "gamma": pd.DataFrame(Gm, index=prices, columns=prices), "shares at mean": Sm,
           "AES": pd.DataFrame(AES, index=prices, columns=prices), "elasticities": pd.DataFrame(eta, index=prices, columns=prices),
           "MES": pd.DataFrame(MES, index=prices, columns=prices), "loglik": out["loglik"], "N": n, "Sigma": out["Sigma"], "iterations": out["iterations"]}
    # curvature: concavity (cost) / convexity (profit) in prices at each observation
    fitted_sh = np.column_stack([X @ b for X in Xs])
    fitted_sh = np.column_stack([fitted_sh, 1 - fitted_sh.sum(1)])
    ok = 0
    for t in range(n):
        s_ = fitted_sh[t]
        H = Gm + np.outer(s_, s_) - np.diag(s_)
        ev = np.linalg.eigvalsh(H)
        ok += (ev.max() <= 1e-8) if kind == "cost" else (ev.min() >= -1e-8)
    res["monotonicity %"] = float(100 * np.mean((fitted_sh > 0).all(1))) if kind == "cost" else np.nan
    res["curvature %"] = float(100 * ok / n)
    if kind == "cost" and q:
        aq = b[1 + (J - 1) + len(pairs)]
        res["cost elasticity"] = float(aq)
        res["scale economies"] = float(1 - aq)
        res["RTS"] = float(1 / aq)
    return res


def cd_cost(df: pd.DataFrame, cost: str, q: str, prices: list) -> dict:
    """Cobb-Douglas cost function ln C = a + α_q ln Q + Σ α_i ln w_i with Σα_i = 1 (prices normalised by the last)."""
    import statsmodels.api as sm
    d = df[[cost, q] + prices].dropna()
    lw = np.log(d[prices].to_numpy(float))
    y = np.log(d[cost].to_numpy(float)) - lw[:, -1]
    Z = sm.add_constant(np.column_stack([np.log(d[q].to_numpy(float))] + [lw[:, j] - lw[:, -1] for j in range(len(prices) - 1)]))
    m = sm.OLS(y, Z).fit()
    a = np.r_[m.params[2:], 1 - m.params[2:].sum()]
    tab = pd.DataFrame({"Parameter": ["const", f"ln {q}"] + [f"ln {p}" for p in prices[:-1]], "Coef.": m.params, "Std. Err.": m.bse,
                        "t": m.tvalues, "p-value": m.pvalues})
    return {"table": tab, "input shares": dict(zip(prices, a)), "RTS": float(1 / m.params[1]), "cost elasticity": float(m.params[1]), "R²": float(m.rsquared)}


# ==========================================================
# Stochastic frontier analysis
# ==========================================================
SFA_DIST = {"hnormal": ("Half-normal", "نصف طبيعي"), "tnormal": ("Truncated normal", "طبيعي مبتور"), "exponential": ("Exponential", "أسي")}


def _sfa_ll(theta, y, X, Z, dist, s, k):
    b = theta[:k]
    lsv, lsu = theta[k], theta[k + 1]
    sv, su = np.exp(0.5 * lsv), np.exp(0.5 * lsu)
    e = y - X @ b
    if dist == "hnormal":
        sig = np.sqrt(sv ** 2 + su ** 2)
        lam = su / sv
        return np.log(2) - np.log(sig) + stats.norm.logpdf(e / sig) + stats.norm.logcdf(-s * lam * e / sig)
    if dist == "exponential":
        return -np.log(su) + sv ** 2 / (2 * su ** 2) + s * e / su + stats.norm.logcdf(-s * e / sv - sv / su)
    mu = Z @ theta[k + 2:]
    sig2 = sv ** 2 + su ** 2
    sig = np.sqrt(sig2)
    mus = (sv ** 2 * mu - s * su ** 2 * e) / sig2
    sigs = su * sv / sig
    return (-0.5 * np.log(2 * np.pi) - np.log(sig) - stats.norm.logcdf(mu / su) + stats.norm.logcdf(mus / sigs)
            - 0.5 * ((e + s * mu) / sig) ** 2)


def sfa(df: pd.DataFrame, y: str, x: list, dist: str = "hnormal", cost: bool = False, z: list | None = None,
        log: bool = True, ln_y: bool | None = None) -> dict:
    """Cross-sectional stochastic frontier y = Xβ + v − s·u (s = 1 production, −1 cost) by maximum likelihood:
    half-normal (Aigner-Lovell-Schmidt), truncated normal with constant mean (Stevenson) or with mean δ'z
    (Battese-Coelli 1995 inefficiency determinants), exponential (Meeusen-van den Broeck). Parameterised as Stata
    frontier (ln σv², ln σu², μ / δ). Technical efficiency E[exp(−u)|ε] (Battese-Coelli 1988) and JLMS E[u|ε]."""
    z = list(z or [])
    d = df[[y] + x + z].dropna()
    yy = d[y].to_numpy(float)
    Xr = d[x].to_numpy(float)
    if log:
        yy, Xr = np.log(yy), np.log(Xr)
    X = np.column_stack([np.ones(len(d)), Xr])
    n, k = X.shape
    s = -1.0 if cost else 1.0
    Z = np.column_stack([np.ones(n)] + ([d[z].to_numpy(float)] if z else [])) if dist == "tnormal" else np.zeros((n, 0))
    ols_b = np.linalg.lstsq(X, yy, rcond=None)[0]
    e = yy - X @ ols_b
    m3 = np.mean((e - e.mean()) ** 3)
    skew = float(m3 / np.std(e) ** 3)
    ll_ols = float(-n / 2 * (1 + np.log(2 * np.pi * np.mean(e ** 2))))
    # starting values (method of moments for half-normal)
    su = max(max(s * m3 / (np.sqrt(2 / np.pi) * (1 - 4 / np.pi)), 1e-9) ** (1 / 3), 0.05 * np.std(e))
    sv2 = max(np.var(e) - (1 - 2 / np.pi) * su ** 2, 0.05 * np.var(e))
    b0 = ols_b.copy()
    b0[0] += s * su * np.sqrt(2 / np.pi) if dist != "exponential" else s * su
    th0 = np.r_[b0, np.log(sv2), np.log(su ** 2), np.zeros(Z.shape[1])]
    f = lambda th: -np.sum(_sfa_ll(th, yy, X, Z, dist, s, k))  # noqa: E731
    best = None
    for st in (th0, np.r_[th0[:k], th0[k] - 0.5, th0[k + 1] + 0.5, th0[k + 2:]]):
        for meth in ("BFGS", "Nelder-Mead"):
            try:
                r = optimize.minimize(f, st, method=meth, options={"maxiter": 2000, "gtol": 1e-8} if meth == "BFGS" else {"maxiter": 6000, "xatol": 1e-9, "fatol": 1e-11})
                if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                    best = r
            except Exception:  # noqa: BLE001
                continue
    r = optimize.minimize(f, best.x, method="BFGS", options={"gtol": 1e-9, "maxiter": 2000})
    if r.fun <= best.fun:
        best = r
    th = best.x
    from statsmodels.tools.numdiff import approx_hess
    H = approx_hess(th, f)
    V = np.linalg.pinv(H)
    names = ["const"] + [(f"ln {v}" if log else v) for v in x] + ["ln σv²", "ln σu²"] + (
        (["μ"] if not z else ["δ0"] + [f"δ ({v})" for v in z]) if dist == "tnormal" else [])
    se = np.sqrt(np.maximum(np.diag(V), 0))
    tab = pd.DataFrame({"Parameter": names, "Coef.": th, "Std. Err.": se, "z": th / se, "p-value": 2 * stats.norm.sf(np.abs(th / se))})
    sv, su = np.exp(0.5 * th[k]), np.exp(0.5 * th[k + 1])
    e = yy - X @ th[:k]
    if dist == "exponential":
        mus = -s * e - sv ** 2 / su
        sigs = np.full(n, sv)
    else:
        mu = Z @ th[k + 2:] if dist == "tnormal" else np.zeros(n)
        sig2 = sv ** 2 + su ** 2
        mus = (sv ** 2 * mu - s * su ** 2 * e) / sig2
        sigs = np.full(n, su * sv / np.sqrt(sig2))
    Eu = mus + sigs * stats.norm.pdf(mus / sigs) / stats.norm.cdf(mus / sigs)
    sg = -1.0 if not cost else 1.0
    te = np.exp(stats.norm.logcdf(mus / sigs + sg * sigs) - stats.norm.logcdf(mus / sigs) + sg * mus + 0.5 * sigs ** 2)
    ll = -best.fun
    LR = 2 * (ll - ll_ols)
    gamma = su ** 2 / (sv ** 2 + su ** 2)
    out = {"table": tab, "theta": th, "V": V, "loglik": float(ll), "sigma_v": float(sv), "sigma_u": float(su), "lambda": float(su / sv),
           "gamma": float(gamma), "efficiency": te, "E[u|e]": Eu, "mean efficiency": float(te.mean()), "skewness OLS": skew,
           "wrong skew": bool(s * skew > 0), "LR (no inefficiency)": float(LR), "N": n, "dist": dist, "cost": cost,
           "beta": th[:k], "names": names, "index": d.index, "X": X}
    df_lr = 1 + (Z.shape[1] if dist == "tnormal" else 0)
    out["LR df"] = df_lr
    out["p LR"] = float(0.5 * stats.chi2.sf(LR, df_lr - 1) + 0.5 * stats.chi2.sf(LR, df_lr)) if df_lr > 1 else float(0.5 * stats.chi2.sf(LR, 1))
    if log and not cost:
        out["RTS"] = float(th[1:k].sum())
    return out


def sfa_panel(df: pd.DataFrame, unit: str, time: str, y: str, x: list, model: str = "tvd", cost: bool = False,
              mu: bool = True, log: bool = True) -> dict:
    """Panel stochastic frontier: Battese-Coelli (1992) time-varying decay u_it = exp(−η(t − T_i))·u_i (model 'tvd') or
    time-invariant (Pitt-Lee 1981 / Battese-Coelli 1988, model 'ti', η = 0), u_i ~ N⁺(μ, σu²) (μ = 0 if `mu` is False).
    Parameterised as Stata xtfrontier: ln σ², logit γ (γ = σu²/σ²), μ, η."""
    d = df[[unit, time, y] + x].dropna().sort_values([unit, time])
    yy = d[y].to_numpy(float)
    Xr = d[x].to_numpy(float)
    if log:
        yy, Xr = np.log(yy), np.log(Xr)
    X = np.column_stack([np.ones(len(d)), Xr])
    n, k = X.shape
    s = -1.0 if cost else 1.0
    tt = pd.factorize(d[time], sort=True)[0].astype(float)
    ids = d[unit].to_numpy()
    groups = [np.where(ids == u)[0] for u in pd.unique(ids)]
    Ti = np.array([tt[g].max() for g in groups])

    gid = pd.factorize(ids)[0]
    G = len(groups)
    Tn = np.bincount(gid, minlength=G).astype(float)
    Tobs = Ti[gid]

    def nll(th):
        b = th[:k]
        sig2 = np.exp(th[k])
        gam = 1 / (1 + np.exp(-th[k + 1]))
        su2, sv2 = gam * sig2, (1 - gam) * sig2
        m_ = th[k + 2] if mu else 0.0
        eta = th[k + 2 + mu] if model == "tvd" else 0.0
        e = yy - X @ b
        et = np.exp(-eta * (tt - Tobs))
        ee = np.bincount(gid, et * et, G)
        ete = np.bincount(gid, et * e, G)
        e2 = np.bincount(gid, e * e, G)
        den = sv2 + su2 * ee
        mus = (m_ * sv2 - s * su2 * ete) / den
        sigs2 = su2 * sv2 / den
        ll = (-Tn / 2 * np.log(2 * np.pi) - (Tn - 1) / 2 * np.log(sv2) - 0.5 * np.log(den) - 0.5 * e2 / sv2
              + 0.5 * mus ** 2 / sigs2 - 0.5 * m_ ** 2 / su2 + stats.norm.logcdf(mus / np.sqrt(sigs2)) - stats.norm.logcdf(m_ / np.sqrt(su2)))
        v = -float(np.sum(ll))
        return v if np.isfinite(v) else 1e12
    cs = sfa(d, y, x, "hnormal", cost, log=log)
    sig2 = cs["sigma_v"] ** 2 + cs["sigma_u"] ** 2
    g0 = min(max(cs["gamma"], 0.05), 0.95)
    th0 = np.r_[cs["beta"], np.log(sig2), np.log(g0 / (1 - g0)), [0.0] * mu, [0.0] * (model == "tvd")]
    best = None
    for st in (th0, np.r_[th0[:k], th0[k], 1.5, [0.2] * mu, [0.0] * (model == "tvd")]):
        for meth in ("BFGS", "Nelder-Mead"):
            try:
                r = optimize.minimize(nll, st, method=meth, options={"maxiter": 30000})
                if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                    best = r
            except Exception:  # noqa: BLE001
                continue
    r = optimize.minimize(nll, best.x, method="BFGS", options={"gtol": 1e-8})
    if r.fun <= best.fun:
        best = r
    th = best.x
    from statsmodels.tools.numdiff import approx_hess
    V = np.linalg.pinv(approx_hess(th, nll))
    names = ["const"] + [(f"ln {v}" if log else v) for v in x] + ["ln σ²", "logit γ"] + (["μ"] if mu else []) + (["η"] if model == "tvd" else [])
    se = np.sqrt(np.maximum(np.diag(V), 0))
    tab = pd.DataFrame({"Parameter": names, "Coef.": th, "Std. Err.": se, "z": th / se, "p-value": 2 * stats.norm.sf(np.abs(th / se))})
    sig2 = np.exp(th[k])
    gam = 1 / (1 + np.exp(-th[k + 1]))
    su2, sv2 = gam * sig2, (1 - gam) * sig2
    m_ = th[k + 2] if mu else 0.0
    eta = th[k + 2 + mu] if model == "tvd" else 0.0
    e = yy - X @ th[:k]
    te = np.zeros(n)
    sg = -1.0 if not cost else 1.0
    for g, T_ in zip(groups, Ti):
        et = np.exp(-eta * (tt[g] - T_))
        den = sv2 + su2 * (et @ et)
        mus = (m_ * sv2 - s * su2 * (et @ e[g])) / den
        sigs = np.sqrt(su2 * sv2 / den)
        te[g] = np.exp(stats.norm.logcdf(mus / sigs + sg * et * sigs) - stats.norm.logcdf(mus / sigs) + sg * et * mus + 0.5 * et ** 2 * sigs ** 2)
    eff = pd.DataFrame({unit: ids, time: d[time].to_numpy(), "TE": te})
    return {"table": tab, "theta": th, "loglik": float(-best.fun), "sigma2": float(sig2), "gamma": float(gam), "mu": float(m_), "eta": float(eta),
            "efficiency": eff, "mean efficiency by period": eff.groupby(time)["TE"].mean(), "N": n, "units": len(groups), "model": model}


def metafrontier(df: pd.DataFrame, group: str, y: str, x: list, dist: str = "hnormal") -> dict:
    """Battese, Rao & O'Donnell (2004) meta-frontier: group SFAs, then the meta-frontier β* minimising Σ|x'β* − x'β̂_g|
    subject to x'β* ≥ x'β̂_g (linear programme); technology-gap ratio TGR = exp(x'β̂_g − x'β*), meta-technical efficiency
    MTE = TE × TGR."""
    from scipy.optimize import linprog
    parts, rows = [], []
    for g_, dg in df.groupby(group):
        r = sfa(dg, y, x, dist)
        Xg = r["X"]
        parts.append((g_, r, Xg))
        rows.append({"Group": g_, "N": r["N"], "Mean TE": r["mean efficiency"], "λ": r["lambda"], "log L": r["loglik"],
                     **{nm: v for nm, v in zip(r["names"][:Xg.shape[1]], r["beta"])}})
    Xall = np.vstack([p[2] for p in parts])
    fit = np.concatenate([p[2] @ p[1]["beta"] for p in parts])
    k = Xall.shape[1]
    res = linprog(c=Xall.sum(0), A_ub=-Xall, b_ub=-fit, bounds=[(None, None)] * k, method="highs")
    bstar = res.x
    tgr = np.exp(fit - Xall @ bstar)
    te = np.concatenate([p[1]["efficiency"] for p in parts])
    grp = np.concatenate([[p[0]] * len(p[2]) for p in parts])
    obs = pd.DataFrame({"Group": grp, "TE (group frontier)": te, "TGR": tgr, "MTE": te * tgr})
    summ = obs.groupby("Group").mean().reset_index()
    return {"groups": pd.DataFrame(rows), "meta beta": pd.Series(bstar, index=parts[0][1]["names"][:k]), "obs": obs, "summary": summ}


# ==========================================================
# Data envelopment analysis and Malmquist
# ==========================================================
def dea(X, Y, rts: str = "crs", orientation: str = "input", Xref=None, Yref=None) -> np.ndarray:
    """Radial DEA efficiency (Farrell) by linear programming: input orientation θ ≤ 1 (min θ: Σλx ≤ θx₀, Σλy ≥ y₀);
    output orientation reported as 1/φ (max φ: Σλx ≤ x₀, Σλy ≥ φy₀). rts: 'crs' (CCR), 'vrs' (BCC), 'nirs'.
    A separate reference technology (Xref, Yref) gives cross-period distance functions for Malmquist."""
    from scipy.optimize import linprog
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    Xr = X if Xref is None else np.asarray(Xref, float)
    Yr = Y if Yref is None else np.asarray(Yref, float)
    n, m = X.shape
    s = Y.shape[1]
    N = Xr.shape[0]
    out = np.full(n, np.nan)
    for o in range(n):
        c = np.r_[1.0, np.zeros(N)] if orientation == "input" else np.r_[-1.0, np.zeros(N)]
        if orientation == "input":
            A = np.vstack([np.column_stack([-X[o][:, None], Xr.T]), np.column_stack([np.zeros((s, 1)), -Yr.T])])
            b = np.r_[np.zeros(m), -Y[o]]
        else:
            A = np.vstack([np.column_stack([np.zeros((m, 1)), Xr.T]), np.column_stack([Y[o][:, None], -Yr.T])])
            b = np.r_[X[o], np.zeros(s)]
        Aeq = beq = None
        if rts == "vrs":
            Aeq, beq = np.r_[0.0, np.ones(N)][None, :], [1.0]
        elif rts == "nirs":
            A = np.vstack([A, np.r_[0.0, np.ones(N)]])
            b = np.r_[b, 1.0]
        r = linprog(c, A_ub=A, b_ub=b, A_eq=Aeq, b_eq=beq, bounds=[(None, None)] + [(0, None)] * N, method="highs")
        if r.status == 0:
            out[o] = r.x[0] if orientation == "input" else 1 / r.x[0]
    return out


def dea_table(df: pd.DataFrame, inputs: list, outputs: list, orientation: str = "input", ids=None) -> pd.DataFrame:
    X, Y = df[inputs].to_numpy(float), df[outputs].to_numpy(float)
    crs = dea(X, Y, "crs", orientation)
    vrs = dea(X, Y, "vrs", orientation)
    nirs = dea(X, Y, "nirs", orientation)
    se = crs / vrs
    nature = np.where(np.isclose(se, 1, atol=1e-6), "CRS", np.where(np.isclose(nirs, crs, atol=1e-6), "IRS", "DRS"))
    out = pd.DataFrame({"DMU": ids if ids is not None else np.arange(1, len(df) + 1), "CRS (CCR)": crs, "VRS (BCC)": vrs, "Scale efficiency": se,
                        "Returns to scale": nature})
    return out


def malmquist(df: pd.DataFrame, unit: str, time: str, inputs: list, outputs: list) -> pd.DataFrame:
    """Output-oriented Malmquist TFP index (Färe, Grosskopf, Norris & Zhang 1994) under CRS between consecutive periods,
    decomposed into efficiency change (EC) and technical change (TC); M > 1 = productivity growth."""
    periods = sorted(df[time].unique())
    rows = []
    for t0, t1 in zip(periods[:-1], periods[1:]):
        a = df[df[time] == t0].set_index(unit)
        b = df[df[time] == t1].set_index(unit)
        common = a.index.intersection(b.index)
        a, b = a.loc[common], b.loc[common]
        Xa, Ya, Xb, Yb = a[inputs].to_numpy(float), a[outputs].to_numpy(float), b[inputs].to_numpy(float), b[outputs].to_numpy(float)
        D_aa = dea(Xa, Ya, "crs", "output")
        D_bb = dea(Xb, Yb, "crs", "output")
        D_ab = dea(Xb, Yb, "crs", "output", Xa, Ya)       # period-t technology, period-t+1 data
        D_ba = dea(Xa, Ya, "crs", "output", Xb, Yb)
        EC = D_bb / D_aa
        TC = np.sqrt((D_ab / D_bb) * (D_aa / D_ba))
        M = EC * TC
        for u_, m_, ec, tc in zip(common, M, EC, TC):
            rows.append({unit: u_, "from": t0, "to": t1, "Malmquist": m_, "Efficiency change": ec, "Technical change": tc})
    return pd.DataFrame(rows)


# ==========================================================
# Total factor productivity
# ==========================================================
def tfp_index(df: pd.DataFrame, time: str, outputs: list, out_prices: list, inputs: list, in_prices: list) -> pd.DataFrame:
    """Aggregate TFP growth across periods by chained Törnqvist and Fisher indices (outputs and inputs summed over
    units in each period; value shares as weights)."""
    g = df.groupby(time)
    Yq = g[outputs].sum()
    Xq = g[inputs].sum()
    Yp = pd.DataFrame({o: (df[o] * df[p]).groupby(df[time]).sum() / df[o].groupby(df[time]).sum() for o, p in zip(outputs, out_prices)})
    Xp = pd.DataFrame({i: (df[i] * df[p]).groupby(df[time]).sum() / df[i].groupby(df[time]).sum() for i, p in zip(inputs, in_prices)})
    rows = [{"Period": Yq.index[0], "Törnqvist output": 1.0, "Törnqvist input": 1.0, "Törnqvist TFP": 1.0, "Fisher output": 1.0, "Fisher input": 1.0, "Fisher TFP": 1.0}]
    cum = {k: 1.0 for k in ("to", "ti", "fo", "fi")}
    for a, b in zip(Yq.index[:-1], Yq.index[1:]):
        ra = (Yq.loc[a] * Yp.loc[a]) / (Yq.loc[a] * Yp.loc[a]).sum()
        rb = (Yq.loc[b] * Yp.loc[b]) / (Yq.loc[b] * Yp.loc[b]).sum()
        sa = (Xq.loc[a] * Xp.loc[a]) / (Xq.loc[a] * Xp.loc[a]).sum()
        sb = (Xq.loc[b] * Xp.loc[b]) / (Xq.loc[b] * Xp.loc[b]).sum()
        to = np.exp(np.sum(0.5 * (ra + rb) * np.log(Yq.loc[b] / Yq.loc[a])))
        ti = np.exp(np.sum(0.5 * (sa + sb) * np.log(Xq.loc[b] / Xq.loc[a])))

        def fisher(q0, q1, p0, p1):
            L = (q1 @ p0) / (q0 @ p0)
            P = (q1 @ p1) / (q0 @ p1)
            return np.sqrt(L * P)
        fo = fisher(Yq.loc[a].to_numpy(), Yq.loc[b].to_numpy(), Yp.loc[a].to_numpy(), Yp.loc[b].to_numpy())
        fi = fisher(Xq.loc[a].to_numpy(), Xq.loc[b].to_numpy(), Xp.loc[a].to_numpy(), Xp.loc[b].to_numpy())
        cum["to"] *= to
        cum["ti"] *= ti
        cum["fo"] *= fo
        cum["fi"] *= fi
        rows.append({"Period": b, "Törnqvist output": cum["to"], "Törnqvist input": cum["ti"], "Törnqvist TFP": cum["to"] / cum["ti"],
                     "Fisher output": cum["fo"], "Fisher input": cum["fi"], "Fisher TFP": cum["fo"] / cum["fi"]})
    return pd.DataFrame(rows)


def _poly(cols, deg=3):
    from sklearn.preprocessing import PolynomialFeatures
    return PolynomialFeatures(deg, include_bias=True).fit_transform(np.column_stack(cols))


def _lag(df, unit, col):
    return df.groupby(unit)[col].shift(1)


def prodest(df: pd.DataFrame, unit: str, time: str, y: str, free: list, state: str, proxy: str, method: str = "acf",
            deg: int = 3, boot: int = 0, seed: int = 1) -> dict:
    """Control-function production-function estimators (value added): Olley-Pakes (proxy = investment), Levinsohn-Petrin
    (proxy = intermediate inputs) and Ackerberg-Caves-Frazer (2015). First stage: y on a 3rd-order polynomial in the
    free input(s), state and proxy (OP / LP: free inputs enter linearly); second stage: ω = φ̂ − β'x, ω_t = g(ω_{t−1}) + ξ
    (3rd-order polynomial), moments E[ξ·(k_t, l_{t−1})] = 0 (ACF) or E[ξ·k_t] = 0 (OP / LP). Bootstrap over units for SEs."""
    d = df[[unit, time, y] + free + [state, proxy]].dropna().sort_values([unit, time]).reset_index(drop=True)
    if method == "op":
        d = d[d[proxy] > 0].reset_index(drop=True)

    def est(dd):
        yv = dd[y].to_numpy(float)
        L = dd[free].to_numpy(float)
        k = dd[state].to_numpy(float)
        px = dd[proxy].to_numpy(float)
        if method == "op":
            px = np.log(px)
        if method == "acf":
            P = _poly([L[:, j] for j in range(L.shape[1])] + [k, px], deg)
            phi = P @ np.linalg.lstsq(P, yv, rcond=None)[0]
            bl1 = None
        else:
            P = _poly([k, px], deg)
            Z = np.column_stack([L, P])
            co = np.linalg.lstsq(Z, yv, rcond=None)[0]
            bl1 = co[:L.shape[1]]
            phi = yv - (yv - Z @ co) - L @ bl1          # φ̂ = fitted minus free-input part
        dd = dd.assign(_phi=phi)
        dd["_phi_l"] = _lag(dd, unit, "_phi")
        lagcols = []
        for j, c in enumerate(free):
            dd[f"_l{j}"] = _lag(dd, unit, c)
            lagcols.append(f"_l{j}")
        dd["_k_l"] = _lag(dd, unit, state)
        ok = dd[["_phi_l", "_k_l"] + lagcols].notna().all(1).to_numpy()
        e = dd[ok]
        phi_t, phi_l = e["_phi"].to_numpy(), e["_phi_l"].to_numpy()
        kt, kl = e[state].to_numpy(float), e["_k_l"].to_numpy()
        Lt = e[free].to_numpy(float)
        Ll = e[lagcols].to_numpy(float)

        def xi(beta):
            if method == "acf":
                bl, bk = beta[:-1], beta[-1]
                w, wl = phi_t - Lt @ bl - bk * kt, phi_l - Ll @ bl - bk * kl
            else:
                bk = beta[0]
                w, wl = phi_t - bk * kt, phi_l - bk * kl
            G = _poly([wl], 3)
            return w - G @ np.linalg.lstsq(G, w, rcond=None)[0]
        if method == "acf":
            Zm = np.column_stack([kt, Ll])
            obj = lambda b_: np.sum((Zm.T @ xi(b_) / len(kt)) ** 2)  # noqa: E731
            ols = np.linalg.lstsq(np.column_stack([np.ones(len(dd)), dd[free].to_numpy(float), dd[state].to_numpy(float)]), dd[y].to_numpy(float), rcond=None)[0]
            best = None
            starts = [ols[1:]] + [np.r_[np.full(len(free), a) / len(free), b] for a in (0.3, 0.5, 0.7, 0.9) for b in (0.1, 0.3, 0.5)]
            for st in starts:
                r = optimize.minimize(obj, st, method="Nelder-Mead", options={"xatol": 1e-9, "fatol": 1e-16, "maxiter": 20000})
                if best is None or r.fun < best.fun:
                    best = r
            return np.asarray(best.x)
        obj = lambda b_: (kt @ xi(b_) / len(kt)) ** 2  # noqa: E731
        r = optimize.minimize_scalar(lambda bk: obj([bk]), bounds=(-0.5, 1.5), method="bounded", options={"xatol": 1e-10})
        return np.r_[bl1, r.x]
    b = est(d)
    names = list(free) + [state]
    se = np.full(len(b), np.nan)
    if boot:
        rng = np.random.default_rng(seed)
        units = d[unit].unique()
        draws = []
        for _ in range(boot):
            pick = rng.choice(units, len(units), replace=True)
            parts = [d[d[unit] == u_].assign(**{unit: f"{u_}_{i}"}) for i, u_ in enumerate(pick)]
            try:
                draws.append(est(pd.concat(parts, ignore_index=True)))
            except Exception:  # noqa: BLE001
                continue
        se = np.std(np.array(draws), axis=0, ddof=1)
    import statsmodels.api as sm
    ols = sm.OLS(d[y].to_numpy(float), sm.add_constant(d[names].to_numpy(float))).fit()
    tab = pd.DataFrame({"Input": names, "Coef.": b, "Std. Err. (bootstrap)": se, "OLS": ols.params[1:]})
    w = d[y].to_numpy(float) - d[names].to_numpy(float) @ b
    return {"table": tab, "beta": b, "RTS": float(b.sum()), "omega": pd.DataFrame({unit: d[unit], time: d[time], "ln TFP": w - w.mean()}),
            "method": method}
