# ==========================================
# Aetheris Research - Volatility models (Catalogue §1.9)
# Univariate: ARCH, GARCH(p,q), GARCH-M, GJR / threshold GARCH, EGARCH (Nelson) with normal, Student-t, GED and
# skewed-t (Fernandez-Steel) errors; ML with numerical-Hessian and Bollerslev-Wooldridge robust standard errors;
# diagnostics (Ljung-Box on z and z², ARCH-LM, Engle-Ng sign-bias), persistence, half-life, news impact,
# variance forecasts, Value-at-Risk / Expected Shortfall with Kupiec and Christoffersen backtests.
# Multivariate: CCC (Bollerslev 1990), DCC (Engle 2002), BEKK(1,1) (Engle-Kroner 1995) and the Engle-Sheppard
# test of constant correlation. Initialisation follows R fGarch (pre-sample ε² and σ² = mean of ε²).
# ==========================================

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

MODELS = {"arch": ("ARCH(q)", "ARCH(q)"), "garch": ("GARCH(p, q)", "GARCH(p, q)"),
          "gjr": ("GJR-GARCH / threshold GARCH", "GJR-GARCH / العتبي TGARCH"), "egarch": ("EGARCH (Nelson)", "EGARCH (نيلسون)")}
DISTS = {"norm": ("Normal", "طبيعي"), "std": ("Student-t", "ستيودنت t"), "ged": ("Generalised error (GED)", "الخطأ المعمم GED"),
         "sstd": ("Skewed Student-t", "ستيودنت t الملتوي")}


# ---------------- standardised densities (unit variance) ----------------
def _logpdf(z: np.ndarray, dist: str, shape=None, skew=None) -> np.ndarray:
    if dist == "norm":
        return -0.5 * (np.log(2 * np.pi) + z ** 2)
    if dist == "std":
        nu = shape
        s = np.sqrt(nu / (nu - 2))
        return stats.t.logpdf(z * s, nu) + np.log(s)
    if dist == "ged":
        nu = shape
        lam = np.sqrt(2 ** (-2 / nu) * special.gamma(1 / nu) / special.gamma(3 / nu))
        g = nu / (lam * 2 ** (1 + 1 / nu) * special.gamma(1 / nu))
        return np.log(g) - 0.5 * np.abs(z / lam) ** nu
    if dist == "sstd":
        nu, xi = shape, skew
        m1 = 2 * np.sqrt(nu - 2) / (nu - 1) / np.exp(special.betaln(0.5, nu / 2))
        mu = m1 * (xi - 1 / xi)
        sig = np.sqrt((1 - m1 ** 2) * (xi ** 2 + 1 / xi ** 2) + 2 * m1 ** 2 - 1)
        zz = z * sig + mu
        Xi = xi ** np.sign(zz)
        g = 2 / (xi + 1 / xi)
        s = np.sqrt(nu / (nu - 2))
        return np.log(g) + stats.t.logpdf(zz / Xi * s, nu) + np.log(s) + np.log(sig)
    raise ValueError(dist)


def _ppf(p, dist, shape=None, skew=None):
    if dist == "norm":
        return stats.norm.ppf(p)
    if dist == "std":
        return stats.t.ppf(p, shape) * np.sqrt((shape - 2) / shape)
    if dist == "ged":
        nu = shape
        lam = np.sqrt(2 ** (-2 / nu) * special.gamma(1 / nu) / special.gamma(3 / nu))
        q = stats.gamma.ppf(np.abs(2 * p - 1), 1 / nu)
        return np.sign(p - 0.5) * lam * (2 * q) ** (1 / nu)
    grid = np.linspace(-15, 15, 30001)
    cdf = np.cumsum(np.exp(_logpdf(grid, dist, shape, skew)))
    cdf /= cdf[-1]
    return np.interp(p, cdf, grid)


def _abs_moment(dist, shape=None, skew=None) -> float:
    if dist == "norm":
        return np.sqrt(2 / np.pi)
    if dist == "std":
        nu = shape
        return np.sqrt(nu - 2) * special.gamma((nu - 1) / 2) / (np.sqrt(np.pi) * special.gamma(nu / 2))
    grid = np.linspace(-30, 30, 60001)
    f = np.exp(_logpdf(grid, dist, shape, skew))
    return float(np.trapezoid(np.abs(grid) * f, grid) / np.trapezoid(f, grid))


# ---------------- univariate GARCH ----------------
@dataclass
class GarchResult:
    model: str
    dist: str
    p: int
    q: int
    mean: str
    names: list
    params: np.ndarray
    se: np.ndarray
    se_robust: np.ndarray
    llf: float
    resid: np.ndarray
    sigma2: np.ndarray
    y: np.ndarray
    index: object = None
    label: str = "r"
    stats: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    @property
    def z(self):
        return self.resid / np.sqrt(self.sigma2)

    def p_(self, name, default=0.0):
        return float(self.params[self.names.index(name)]) if name in self.names else default

    def table(self, robust: bool = True) -> pd.DataFrame:
        se = self.se_robust if robust else self.se
        t = self.params / se
        return pd.DataFrame({"Parameter": self.names, "Coef.": self.params, "Std. Err.": se, "z": t,
                             "p-value": 2 * stats.norm.sf(np.abs(t)), "CI low": self.params - 1.96 * se,
                             "CI high": self.params + 1.96 * se})

    @property
    def title(self):
        lab = {"arch": f"ARCH({self.q})", "garch": f"GARCH({self.p},{self.q})", "gjr": f"GJR-GARCH({self.p},{self.q})",
               "egarch": f"EGARCH({self.p},{self.q})"}[self.model]
        if self.mean.startswith("m"):
            lab += "-M"
        return lab + {"norm": "", "std": " t", "ged": " GED", "sstd": " skew-t"}[self.dist]


def _names(model, p, q, mean, dist):
    n = []
    if mean in ("const", "ar1", "m_sd", "m_var"):
        n.append("mu")
    if mean == "ar1":
        n.append("ar1")
    if mean in ("m_sd", "m_var"):
        n.append("lambda")
    n.append("omega")
    n += [f"alpha{i}" for i in range(1, q + 1)]
    if model in ("gjr", "egarch"):
        n += [f"gamma{i}" for i in range(1, q + 1)]
    n += [f"beta{j}" for j in range(1, p + 1)]
    if dist == "sstd":
        n.append("skew")
    if dist in ("std", "ged", "sstd"):
        n.append("shape")
    return n


def _unpack(theta, names):
    return dict(zip(names, theta))


def _filter(theta, y, model, p, q, mean, dist, names):
    """Return (eps, sigma2, loglik vector)."""
    P = _unpack(theta, names)
    T = len(y)
    mu = P.get("mu", 0.0)
    al = [P[f"alpha{i}"] for i in range(1, q + 1)]
    ga = [P.get(f"gamma{i}", 0.0) for i in range(1, q + 1)]
    be = [P[f"beta{j}"] for j in range(1, p + 1)]
    om = P["omega"]
    shape, skew = P.get("shape"), P.get("skew")
    if mean == "ar1":
        eps0 = y[1:] - mu - P["ar1"] * y[:-1]
        yy = y[1:]
    else:
        yy = y
        eps0 = y - mu if mean != "zero" else y.copy()
    Tn = len(yy)
    if mean in ("m_sd", "m_var"):
        eps0 = yy - mu          # provisional, for the pre-sample variance
    mv = float(np.mean(eps0 ** 2))
    h = np.empty(Tn)
    eps = np.empty(Tn)
    L = max(p, q)
    if model == "egarch":
        Ez = _abs_moment(dist, shape, skew)
        lh_pre = np.log(mv)
        lh = [lh_pre] * p
        zl = [0.0] * q
        azl = [Ez] * q
        for t in range(Tn):
            v = om + sum(al[i] * (azl[i] - Ez) + ga[i] * zl[i] for i in range(q)) + sum(be[j] * lh[j] for j in range(p))
            v = min(max(v, -50.0), 50.0)
            ht = np.exp(v)
            h[t] = ht
            e = (yy[t] - mu - (P["lambda"] * (np.sqrt(ht) if mean == "m_sd" else ht) if mean in ("m_sd", "m_var") else 0.0)) \
                if mean in ("m_sd", "m_var") else eps0[t]
            eps[t] = e
            zt = e / np.sqrt(ht)
            lh = [v] + lh[:-1] if p else lh
            zl = [zt] + zl[:-1]
            azl = [abs(zt)] + azl[:-1]
    else:
        pers = sum(al) + 0.5 * sum(ga) + sum(be)
        h_pre = om + pers * mv
        hl = [h_pre] * p
        el = [mv] * q
        nl = [0.5 * mv] * q        # pre-sample I(ε<0) ε² = ½ mean ε²
        for t in range(Tn):
            ht = om + sum(al[i] * el[i] + ga[i] * nl[i] for i in range(q)) + sum(be[j] * hl[j] for j in range(p))
            if t == 0:
                ht = h_pre
            ht = max(ht, 1e-12)
            h[t] = ht
            if mean in ("m_sd", "m_var"):
                e = yy[t] - mu - P["lambda"] * (np.sqrt(ht) if mean == "m_sd" else ht)
            else:
                e = eps0[t]
            eps[t] = e
            e2 = e * e
            hl = [ht] + hl[:-1] if p else hl
            el = [e2] + el[:-1]
            nl = [e2 if e < 0 else 0.0] + nl[:-1]
    del L
    z = eps / np.sqrt(h)
    ll = _logpdf(z, dist, shape, skew) - 0.5 * np.log(h)
    return eps, h, ll


def _start(y, model, p, q, mean, dist, names):
    v = np.var(y)
    st = {"mu": float(np.mean(y)), "ar1": 0.0, "lambda": 0.0, "omega": 0.1 * v * (1 - 0.9) / 0.1 if model != "egarch" else 0.1 * np.log(v)}
    if model == "egarch":
        st["omega"] = (1 - 0.9) * np.log(v)
    for i in range(1, q + 1):
        st[f"alpha{i}"] = (0.1 if model != "egarch" else 0.15) / q
        st[f"gamma{i}"] = (0.05 if model == "gjr" else -0.05) / q
    for j in range(1, p + 1):
        st[f"beta{j}"] = (0.8 if model != "egarch" else 0.95) / p
    if model == "arch":
        st["omega"] = 0.5 * v
    st["shape"] = {"std": 8.0, "ged": 1.5, "sstd": 8.0}.get(dist, 8.0)
    st["skew"] = 1.0
    st["omega"] = st["omega"] if model == "egarch" else max(st["omega"], 1e-4 * v)
    return np.array([st[n] for n in names])


def _bounds(names, model, y):
    v = np.var(y)
    b = []
    for n in names:
        if n in ("mu", "lambda"):
            b.append((None, None))
        elif n == "ar1":
            b.append((-0.999, 0.999))
        elif n == "omega":
            b.append((None, None) if model == "egarch" else (1e-8 * v, 10 * v))
        elif n.startswith("alpha"):
            b.append((None, None) if model == "egarch" else (0.0, 0.999))
        elif n.startswith("gamma"):
            b.append((None, None) if model == "egarch" else (-0.999, 0.999))
        elif n.startswith("beta"):
            b.append((-0.9999, 0.9999) if model == "egarch" else (0.0, 0.9999))
        elif n == "shape":
            b.append((2.05, 100.0) if "skew" in names or True else (0.3, 20))
        elif n == "skew":
            b.append((0.1, 10.0))
    return b


def fit_garch(y, model: str = "garch", p: int = 1, q: int = 1, dist: str = "norm", mean: str = "const",
              index=None, label: str = "r") -> GarchResult:
    """Maximum-likelihood estimation. model: 'arch' (p = 0), 'garch', 'gjr', 'egarch'; mean: 'zero', 'const',
    'ar1', 'm_sd' (GARCH-M in σ), 'm_var' (GARCH-M in σ²)."""
    from statsmodels.tools.numdiff import approx_fprime, approx_hess
    y = np.asarray(y, float)
    if model == "arch":
        p = 0
    names = _names(model, p, q, mean, dist)
    bnds = _bounds(names, model, y)
    if dist == "ged":
        bnds[names.index("shape")] = (0.3, 20.0)

    def nll(th):
        try:
            P = _unpack(th, names)
            if model != "egarch":
                pers = sum(P[f"alpha{i}"] for i in range(1, q + 1)) + 0.5 * sum(P.get(f"gamma{i}", 0) for i in range(1, q + 1)) \
                    + sum(P[f"beta{j}"] for j in range(1, p + 1))
                if pers >= 0.99999 or any(P[f"alpha{i}"] + P.get(f"gamma{i}", 0) < 0 for i in range(1, q + 1)):
                    return 1e10
            elif sum(P[f"beta{j}"] for j in range(1, p + 1)) >= 0.9999:
                return 1e10
            _, _, ll = _filter(th, y, model, p, q, mean, dist, names)
            v = -ll.sum()
            return v if np.isfinite(v) else 1e10
        except (FloatingPointError, ValueError, OverflowError):
            return 1e10

    x0 = _start(y, model, p, q, mean, dist, names)
    best = None
    for start in (x0, np.where([n.startswith("beta") for n in names], x0 * 1.05, x0)):
        try:
            r1 = optimize.minimize(nll, start, method="L-BFGS-B", bounds=bnds, options={"maxiter": 2000})
            r2 = optimize.minimize(nll, r1.x, method="Nelder-Mead", options={"maxiter": 6000, "xatol": 1e-9, "fatol": 1e-10})
            cand = r2 if r2.fun < r1.fun else r1
            r3 = optimize.minimize(nll, cand.x, method="L-BFGS-B", bounds=bnds, options={"maxiter": 2000, "ftol": 1e-14, "gtol": 1e-9})
            cand = r3 if r3.fun <= cand.fun else cand
            if best is None or cand.fun < best.fun:
                best = cand
        except Exception:  # noqa: BLE001
            continue
    th = best.x
    eps, h, ll = _filter(th, y, model, p, q, mean, dist, names)
    H = approx_hess(th, nll)
    try:
        Hi = np.linalg.inv(H)
        se = np.sqrt(np.abs(np.diag(Hi)))
        G = approx_fprime(th, lambda t: _filter(t, y, model, p, q, mean, dist, names)[2], centered=True)
        J = G.T @ G
        se_r = np.sqrt(np.abs(np.diag(Hi @ J @ Hi)))
    except np.linalg.LinAlgError:
        se = se_r = np.full(len(th), np.nan)
    T = len(ll)
    k = len(th)
    r = GarchResult(model, dist, p, q, mean, names, th, se, se_r, float(ll.sum()), eps, h,
                    y[1:] if mean == "ar1" else y, index[1:] if (index is not None and mean == "ar1") else index, label)
    r.stats = {"N": T, "Log-likelihood": r.llf, "AIC": -2 * r.llf + 2 * k, "BIC": -2 * r.llf + np.log(T) * k,
               "AIC/N": (-2 * r.llf + 2 * k) / T, "BIC/N": (-2 * r.llf + np.log(T) * k) / T}
    pers, hl, uv = persistence(r)
    r.stats.update({"Persistence": pers, "Half-life": hl, "Unconditional variance": uv})
    if pers >= 0.999:
        r.notes.append("Persistence is (almost) one: shocks to volatility are (nearly) permanent — an integrated GARCH.")
    if not best.success:
        r.notes.append("The optimiser reported non-convergence; check the estimates.")
    return r


def persistence(r: GarchResult) -> tuple:
    q, p = r.q, r.p
    al = sum(r.p_(f"alpha{i}") for i in range(1, q + 1))
    ga = sum(r.p_(f"gamma{i}") for i in range(1, q + 1))
    be = sum(r.p_(f"beta{j}") for j in range(1, p + 1))
    if r.model == "egarch":
        pers = be
        uv = np.exp(r.p_("omega") / (1 - be)) if be < 1 else np.inf
    else:
        kappa = 0.5
        if r.model == "gjr" and r.dist == "sstd":
            grid = np.linspace(-30, 0, 30001)
            f = np.exp(_logpdf(grid, r.dist, r.p_("shape"), r.p_("skew")))
            kappa = float(np.trapezoid(grid ** 2 * f, grid))
        pers = al + kappa * ga + be
        uv = r.p_("omega") / (1 - pers) if pers < 1 else np.inf
    hl = np.log(0.5) / np.log(pers) if 0 < pers < 1 else np.inf
    return float(pers), float(hl), float(uv)


def news_impact(r: GarchResult, grid=None) -> pd.DataFrame:
    """σ²_{t+1} as a function of the shock ε_t, other lags at the unconditional variance (Engle & Ng 1993)."""
    _, _, uv = persistence(r)
    if not np.isfinite(uv):
        uv = float(np.mean(r.sigma2))
    s = np.sqrt(uv)
    e = np.linspace(-5 * s, 5 * s, 201) if grid is None else grid
    if r.model == "egarch":
        Ez = _abs_moment(r.dist, r.p_("shape", None), r.p_("skew", None))
        z = e / s
        lh = r.p_("omega") + r.p_("alpha1") * (np.abs(z) - Ez) + r.p_("gamma1") * z + sum(r.p_(f"beta{j}") for j in range(1, r.p + 1)) * np.log(uv)
        h = np.exp(lh)
    else:
        h = r.p_("omega") + r.p_("alpha1") * e ** 2 + r.p_("gamma1") * (e < 0) * e ** 2 \
            + (sum(r.p_(f"alpha{i}") + 0.5 * r.p_(f"gamma{i}") for i in range(2, r.q + 1))) * uv \
            + sum(r.p_(f"beta{j}") for j in range(1, r.p + 1)) * uv
    return pd.DataFrame({"shock": e, "sigma2": h})


def garch_diagnostics(r: GarchResult, lags: int = 10) -> pd.DataFrame:
    from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
    z = r.z
    lb = acorr_ljungbox(z, lags=[lags], return_df=True)
    lb2 = acorr_ljungbox(z ** 2, lags=[lags], return_df=True)
    arch = het_arch(z, nlags=5)
    jb = stats.jarque_bera(z)
    sb = sign_bias(r)
    rows = [{"Test": f"Ljung-Box Q({lags}) on z", "H0": "no autocorrelation in the mean", "Statistic": float(lb.lb_stat.iloc[0]), "p-value": float(lb.lb_pvalue.iloc[0])},
            {"Test": f"Ljung-Box Q({lags}) on z²", "H0": "no remaining volatility clustering", "Statistic": float(lb2.lb_stat.iloc[0]), "p-value": float(lb2.lb_pvalue.iloc[0])},
            {"Test": "ARCH-LM (5) on z", "H0": "no remaining ARCH", "Statistic": float(arch[0]), "p-value": float(arch[1])},
            {"Test": "Jarque-Bera on z", "H0": "normal standardised residuals", "Statistic": float(jb.statistic), "p-value": float(jb.pvalue)}]
    rows += [{"Test": f"Engle-Ng {k}", "H0": "no asymmetry missed by the model", "Statistic": v[0], "p-value": v[1]} for k, v in sb.items()]
    return pd.DataFrame(rows)


def sign_bias(r: GarchResult) -> dict:
    """Engle & Ng (1993): z²_t on S⁻_{t−1}, S⁻ε_{t−1}, S⁺ε_{t−1}; t tests and the joint F test."""
    z2 = r.z[1:] ** 2
    e = r.resid[:-1]
    Sn = (e < 0).astype(float)
    X = np.column_stack([np.ones(len(e)), Sn, Sn * e, (1 - Sn) * e])
    b, *_ = np.linalg.lstsq(X, z2, rcond=None)
    u = z2 - X @ b
    df = len(z2) - 4
    s2 = u @ u / df
    V = s2 * np.linalg.inv(X.T @ X)
    t = b / np.sqrt(np.diag(V))
    R = np.eye(4)[1:]
    F = float(b[1:] @ np.linalg.solve(R @ V @ R.T, b[1:]) / 3)
    p2 = lambda tv: float(2 * stats.t.sf(abs(tv), df))  # noqa: E731
    return {"sign bias": (float(t[1]), p2(t[1])), "negative size bias": (float(t[2]), p2(t[2])),
            "positive size bias": (float(t[3]), p2(t[3])), "joint (F)": (F, float(stats.f.sf(F, 3, df)))}


def forecast_variance(r: GarchResult, h: int = 20, sims: int = 2000, seed: int = 1) -> pd.DataFrame:
    """h-step conditional-variance forecasts: analytic for (GJR-)GARCH, Monte Carlo for EGARCH."""
    pers, _, uv = persistence(r)
    if r.model != "egarch" and r.p <= 1 and r.q <= 1:
        e_T, h_T = r.resid[-1], r.sigma2[-1]
        f1 = r.p_("omega") + r.p_("alpha1") * e_T ** 2 + r.p_("gamma1") * (e_T < 0) * e_T ** 2 + r.p_("beta1") * h_T
        out = [f1]
        for _ in range(1, h):
            out.append(r.p_("omega") + pers * out[-1])
        f = np.array(out)
    else:
        rng = np.random.default_rng(seed)
        th = r.params
        paths = np.zeros((sims, h))
        for s_ in range(sims):
            yy = np.r_[r.y, np.zeros(h)]
            # simulate forward by appending draws from the fitted distribution
            for k in range(h):
                eps, hh, _ = _filter(th, yy[:len(r.y) + k + 1], r.model, r.p, r.q, r.mean, r.dist, r.names)
                zdraw = _ppf(rng.uniform(), r.dist, r.p_("shape", None), r.p_("skew", None))
                paths[s_, k] = hh[-1]
                yy[len(r.y) + k] = r.p_("mu") + np.sqrt(hh[-1]) * zdraw
            if s_ >= 199:
                paths = paths[:200]
                break
        f = paths.mean(0)
    idx = np.arange(1, h + 1)
    return pd.DataFrame({"step": idx, "variance": f, "volatility": np.sqrt(f), "unconditional": uv})


def var_es(r: GarchResult, level: float = 0.01) -> dict:
    """One-step-ahead parametric VaR / ES of the return (lower tail) from the fitted model, plus in-sample VaR
    series and Kupiec (1995) unconditional-coverage and Christoffersen (1998) independence tests."""
    sh, sk = r.p_("shape", None), r.p_("skew", None)
    qz = float(_ppf(level, r.dist, sh, sk))
    grid = np.linspace(-30, qz, 20001)
    f = np.exp(_logpdf(grid, r.dist, sh, sk))
    es_z = float(np.trapezoid(grid * f, grid) / np.trapezoid(f, grid))
    mu = r.y - r.resid
    var_series = mu + np.sqrt(r.sigma2) * qz
    hits = (r.y < var_series).astype(int)
    n, x = len(hits), hits.sum()
    pi = x / n
    lr_uc = -2 * ((n - x) * np.log(1 - level) + x * np.log(level) - ((n - x) * np.log(1 - pi) + (x * np.log(pi) if x else 0))) if 0 < pi < 1 else np.nan
    n00 = n01 = n10 = n11 = 0
    for a, b in zip(hits[:-1], hits[1:]):
        n00 += (a == 0) & (b == 0); n01 += (a == 0) & (b == 1); n10 += (a == 1) & (b == 0); n11 += (a == 1) & (b == 1)
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p_ = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    def _l(pp, k0, k1):
        return (k0 * np.log(1 - pp) if k0 and pp < 1 else 0) + (k1 * np.log(pp) if k1 and pp > 0 else 0)
    lr_ind = -2 * (_l(p_, n00 + n10, n01 + n11) - _l(p01, n00, n01) - _l(p11, n10, n11))
    fc = forecast_variance(r, 1)
    mu_next = r.p_("mu") + (r.p_("ar1") * r.y[-1] if r.mean == "ar1" else 0)
    sd_next = float(np.sqrt(fc["variance"].iloc[0]))
    return {"VaR": mu_next + sd_next * qz, "ES": mu_next + sd_next * es_z, "level": level, "series": var_series,
            "hits": hits, "violations": int(x), "expected": level * n, "kupiec": lr_uc,
            "p_kupiec": float(stats.chi2.sf(lr_uc, 1)) if np.isfinite(lr_uc) else np.nan,
            "christoffersen": float(lr_ind), "p_christoffersen": float(stats.chi2.sf(lr_ind, 1)),
            "p_cc": float(stats.chi2.sf(lr_uc + lr_ind, 2)) if np.isfinite(lr_uc) else np.nan}


def arch_lm(y, lags: int = 5, demean: bool = True) -> dict:
    """Engle (1982) ARCH-LM pre-test on the (demeaned) series."""
    from statsmodels.stats.diagnostic import het_arch
    x = np.asarray(y, float)
    x = x - x.mean() if demean else x
    st, p, f, fp = het_arch(x, nlags=lags)
    return {"LM": float(st), "p": float(p), "F": float(f), "p_F": float(fp), "lags": lags}


# ---------------- multivariate ----------------
@dataclass
class MgarchResult:
    model: str
    names: list
    uni: list
    params: dict
    se: dict
    llf: float
    H: np.ndarray            # T × K × K conditional covariances
    R: np.ndarray            # T × K × K conditional correlations
    index: object = None
    stats: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def corr_frame(self) -> pd.DataFrame:
        K = len(self.names)
        d = {f"{self.names[i]} – {self.names[j]}": self.R[:, i, j] for i in range(K) for j in range(i + 1, K)}
        return pd.DataFrame(d, index=self.index)


def _std_resid(Y, model, dist, p, q, names, index):
    uni = [fit_garch(Y[:, k], model, p, q, dist, "const", index, names[k]) for k in range(Y.shape[1])]
    Z = np.column_stack([u.z for u in uni])
    S = np.column_stack([np.sqrt(u.sigma2) for u in uni])
    return uni, Z, S


def fit_ccc(Y, names, model="garch", dist="norm", p=1, q=1, index=None) -> MgarchResult:
    Y = np.asarray(Y, float)
    uni, Z, S = _std_resid(Y, model, dist, p, q, names, index)
    Rb = np.corrcoef(Z.T)
    T, K = Z.shape
    Ri = np.linalg.inv(Rb)
    llc = -0.5 * (T * np.linalg.slogdet(Rb)[1] + np.einsum("ti,ij,tj->", Z, Ri, Z) - (Z ** 2).sum())
    H = np.einsum("ti,ij,tj->tij", S, Rb, S)
    R = np.broadcast_to(Rb, (T, K, K)).copy()
    ll = sum(u.llf for u in uni) + llc
    return MgarchResult("CCC", names, uni, {"R": Rb}, {}, float(ll), H, R, index,
                        {"Log-likelihood": ll, "N": T, "LL correlation part": llc})


def _dcc_filter(ab, Z, Qbar):
    a, b = ab
    T, K = Z.shape
    OP = np.einsum("ti,tj->tij", Z, Z)
    Q = np.empty((T, K, K))
    Q[0] = Qbar
    c = (1 - a - b) * Qbar
    for t in range(1, T):
        Q[t] = c + a * OP[t - 1] + b * Q[t - 1]
    d = 1 / np.sqrt(np.einsum("tii->ti", Q))
    R = Q * np.einsum("ti,tj->tij", d, d)
    _, ld = np.linalg.slogdet(R)
    quad = np.einsum("ti,ti->t", Z, np.linalg.solve(R, Z[:, :, None])[:, :, 0])
    ll = -0.5 * (ld + quad - (Z ** 2).sum(1)).sum()
    return float(ll), R


def fit_dcc(Y, names, model="garch", dist="norm", p=1, q=1, index=None) -> MgarchResult:
    """Engle (2002) two-step DCC(1,1): univariate GARCH margins, then the correlation likelihood in (a, b) with
    Q̄ = sample covariance of the standardised residuals (correlation targeting)."""
    from statsmodels.tools.numdiff import approx_hess
    Y = np.asarray(Y, float)
    uni, Z, S = _std_resid(Y, model, dist, p, q, names, index)
    T, K = Z.shape
    Qbar = Z.T @ Z / T

    def nll(ab):
        if ab[0] < 0 or ab[1] < 0 or ab[0] + ab[1] >= 0.9999:
            return 1e10
        return -_dcc_filter(ab, Z, Qbar)[0]

    best = None
    for st in ([0.03, 0.95], [0.05, 0.90], [0.01, 0.98]):
        r = optimize.minimize(nll, st, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-9, "maxiter": 4000})
        if best is None or r.fun < best.fun:
            best = r
    ab = best.x
    llc, R = _dcc_filter(ab, Z, Qbar)
    try:
        se = np.sqrt(np.abs(np.diag(np.linalg.inv(approx_hess(ab, nll)))))
    except np.linalg.LinAlgError:
        se = np.full(2, np.nan)
    H = np.einsum("ti,tij,tj->tij", S, R, S)
    ll = sum(u.llf for u in uni) + llc
    res = MgarchResult("DCC", names, uni, {"a": ab[0], "b": ab[1], "Qbar": Qbar}, {"a": se[0], "b": se[1]}, float(ll), H, R, index,
                       {"Log-likelihood": ll, "N": T, "LL correlation part": llc, "a + b": ab.sum()})
    res.notes.append("Two-step standard errors of a and b treat the univariate stage as known (Engle 2002); they are "
                     "slightly too small.")
    return res


def fit_bekk(Y, names, kind: str = "diagonal", index=None) -> MgarchResult:
    """BEKK(1,1): H_t = CC' + A'ε_{t−1}ε'_{t−1}A + B'H_{t−1}B with C lower-triangular; kind = 'scalar', 'diagonal' or
    'full'. Gaussian ML on demeaned returns, H₀ = sample covariance."""
    from statsmodels.tools.numdiff import approx_hess
    Y = np.asarray(Y, float)
    E = Y - Y.mean(0)
    T, K = E.shape
    S0 = E.T @ E / T
    il = np.tril_indices(K)
    nC = len(il[0])
    nA = {"scalar": 1, "diagonal": K, "full": K * K}[kind]

    def build(th):
        C = np.zeros((K, K))
        C[il] = th[:nC]
        a, b = th[nC:nC + nA], th[nC + nA:nC + 2 * nA]
        if kind == "scalar":
            A, B = a[0] * np.eye(K), b[0] * np.eye(K)
        elif kind == "diagonal":
            A, B = np.diag(a), np.diag(b)
        else:
            A, B = a.reshape(K, K), b.reshape(K, K)
        return C, A, B

    def filt(th):
        C, A, B = build(th)
        CC = C @ C.T
        AE = E @ A
        OP = np.einsum("ti,tj->tij", AE, AE)
        H = np.empty((T, K, K))
        H[0] = S0
        Bt = B.T
        for t in range(1, T):
            H[t] = CC + OP[t - 1] + Bt @ H[t - 1] @ B
        sg, ld = np.linalg.slogdet(H)
        if np.any(sg <= 0) or not np.all(np.isfinite(ld)):
            return None, -np.inf * np.ones(T)
        quad = np.einsum("ti,ti->t", E, np.linalg.solve(H, E[:, :, None])[:, :, 0])
        ll = -0.5 * (K * np.log(2 * np.pi) + ld + quad)
        return H, ll

    def nll(th):
        _, ll = filt(th)
        v = -ll.sum()
        return v if np.isfinite(v) else 1e12

    C0 = np.linalg.cholesky(S0 * 0.05)
    a0 = {"scalar": [0.25], "diagonal": [0.25] * K, "full": list((0.25 * np.eye(K)).ravel())}[kind]
    b0 = {"scalar": [0.95], "diagonal": [0.95] * K, "full": list((0.95 * np.eye(K)).ravel())}[kind]
    th0 = np.r_[C0[il], a0, b0]
    r1 = optimize.minimize(nll, th0, method="L-BFGS-B", options={"maxiter": 3000})
    r2 = optimize.minimize(nll, r1.x, method="Nelder-Mead", options={"maxiter": 4000 * len(th0), "xatol": 1e-8, "fatol": 1e-9})
    best = r2 if r2.fun < r1.fun else r1
    r3 = optimize.minimize(nll, best.x, method="BFGS", options={"gtol": 1e-6, "maxiter": 500})
    best = r3 if r3.fun < best.fun else best
    th = best.x
    # sign normalisation: diagonal of C positive, A₁₁ and B₁₁ positive
    C, A, B = build(th)
    C = C * np.sign(np.diag(C))[None, :]
    if A[0, 0] < 0:
        A = -A
    if B[0, 0] < 0:
        B = -B
    th = np.r_[C[il], (A[0, 0] if kind == "scalar" else np.diag(A) if kind == "diagonal" else A.ravel()),
               (B[0, 0] if kind == "scalar" else np.diag(B) if kind == "diagonal" else B.ravel())]
    H, ll = filt(th)
    try:
        se = np.sqrt(np.abs(np.diag(np.linalg.inv(approx_hess(th, nll)))))
    except np.linalg.LinAlgError:
        se = np.full(len(th), np.nan)
    d = np.sqrt(np.einsum("tii->ti", H))
    R = H / np.einsum("ti,tj->tij", d, d)
    pn = [f"C[{i + 1},{j + 1}]" for i, j in zip(*il)]
    if kind == "scalar":
        pn += ["a", "b"]
    elif kind == "diagonal":
        pn += [f"A[{i + 1},{i + 1}]" for i in range(K)] + [f"B[{i + 1},{i + 1}]" for i in range(K)]
    else:
        pn += [f"A[{i + 1},{j + 1}]" for i in range(K) for j in range(K)] + [f"B[{i + 1},{j + 1}]" for i in range(K) for j in range(K)]
    res = MgarchResult(f"BEKK ({kind})", names, [], dict(zip(pn, th)), dict(zip(pn, se)), float(ll.sum()), H, R, index,
                       {"Log-likelihood": float(ll.sum()), "N": T, "AIC": -2 * ll.sum() + 2 * len(th),
                        "BIC": -2 * ll.sum() + np.log(T) * len(th)})
    if kind == "full":
        res.notes.append("Off-diagonal A[i,j] (i ≠ j) measure shock spillovers from market i to market j; B[i,j] volatility "
                         "spillovers.")
    return res


def engle_sheppard(Z: np.ndarray, lags: int = 5) -> dict:
    """Engle & Sheppard (2001) test of constant conditional correlation on standardised residuals."""
    T, K = Z.shape
    Rb = np.corrcoef(Z.T)
    w, V = np.linalg.eigh(Rb)
    Rih = V @ np.diag(w ** -0.5) @ V.T
    X = Z @ Rih
    iu = np.triu_indices(K, 1)
    Ylist = []
    for t in range(T):
        Ylist.append(np.outer(X[t], X[t])[iu])
    Ym = np.array(Ylist)
    Y = Ym.ravel(order="F")
    regs = [np.ones(len(Y))]
    for L in range(1, lags + 1):
        lagged = np.vstack([np.zeros((L, Ym.shape[1])), Ym[:-L]])
        regs.append(lagged.ravel(order="F"))
    Xr = np.column_stack(regs)
    keep = np.tile(np.arange(T) >= lags, Ym.shape[1])
    Xr, Y = Xr[keep], Y[keep]
    d, *_ = np.linalg.lstsq(Xr, Y, rcond=None)
    e = Y - Xr @ d
    s2 = e @ e / (len(Y) - Xr.shape[1])
    st = float(d @ Xr.T @ Xr @ d / s2)
    df = lags + 1
    return {"stat": st, "df": df, "p": float(stats.chi2.sf(st, df)), "lags": lags}


def hedge_ratio(m: MgarchResult, i: int, j: int) -> np.ndarray:
    """Dynamic minimum-variance hedge ratio of asset i hedged with asset j: h_ij,t / h_jj,t (Kroner-Sultan 1993)."""
    return m.H[:, i, j] / m.H[:, j, j]


def returns(prices, kind: str = "log", scale: float = 100.0) -> np.ndarray:
    p = np.asarray(prices, float)
    if kind == "log":
        return scale * np.diff(np.log(p))
    return scale * (p[1:] / p[:-1] - 1)
