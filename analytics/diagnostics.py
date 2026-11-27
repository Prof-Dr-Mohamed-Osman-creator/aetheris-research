# ==========================================
# Aetheris Research - Regression Diagnostics (Catalogue §1.3)
# ------------------------------------------
# Every post-estimation test a referee asks for, computed on the model's
# whitened residuals (so WLS / FGLS / GLS models are tested correctly):
#   heteroskedasticity  Breusch-Pagan (Koenker & original), White, Glejser, Park, Goldfeld-Quandt
#   autocorrelation     Durbin-Watson with EXACT p-value (Imhof), Breusch-Godfrey, Ljung-Box, Durbin h
#   collinearity        VIF / tolerance, Belsley condition indices & variance-decomposition proportions
#   specification       Ramsey RESET, Chow break & predictive tests, CUSUM & CUSUM of squares
#   normality           Jarque-Bera, Shapiro-Wilk, D'Agostino K², Anderson-Darling
#   influence           leverage, studentized residuals, Cook's D, DFFITS, DFBETAS
# Pure functions (numpy / scipy / statsmodels) — unit-tested against statsmodels.
# ==========================================

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import integrate, linalg, stats

# ------------------------------------------
# Helpers
# ------------------------------------------
ALPHA = 0.05


def _aux_r2(y: np.ndarray, Z: np.ndarray) -> tuple[float, int]:
    """R² of an auxiliary OLS regression (Z must contain a constant). Returns (R², rank of Z)."""
    b, *_ = np.linalg.lstsq(Z, y, rcond=None)
    e = y - Z @ b
    sst = np.sum((y - y.mean()) ** 2)
    return (1 - e @ e / sst if sst > 0 else 0.0), int(np.linalg.matrix_rank(Z))


def _with_const(X: np.ndarray) -> np.ndarray:
    has_const = np.any(np.all(np.isclose(X, X[0]), axis=0) & ~np.all(np.isclose(X, 0), axis=0))
    return X if has_const else np.c_[np.ones(len(X)), X]


def _slopes(X: np.ndarray) -> np.ndarray:
    """Non-constant columns of X."""
    keep = ~np.all(np.isclose(X, X[0]), axis=0)
    return X[:, keep]


def _row(group, test, test_ar, stat, df, p, h0, h0_ar, ok=None, note=""):
    note, note_ar = (note if isinstance(note, tuple) else (note, note))
    if ok is None and p is not None and not pd.isna(p):
        ok = p >= ALPHA
    return {"group": group, "Test": test, "Test_ar": test_ar, "Statistic": stat, "df": df, "p-value": p,
            "H0": h0, "H0_ar": h0_ar, "ok": ok, "note": note, "note_ar": note_ar}


# ==========================================================
# Heteroskedasticity
# ==========================================================
def breusch_pagan(e, X, koenker: bool = True) -> dict:
    """Breusch-Pagan (1979) LM test; Koenker (1981) studentized version is robust to non-normality."""
    X = _with_const(X)
    n = len(e)
    e2 = e ** 2
    if koenker:
        r2, rank = _aux_r2(e2, X)
        lm = n * r2
    else:
        g = e2 / (e2.mean())
        b, *_ = np.linalg.lstsq(X, g, rcond=None)
        fit = X @ b
        lm = 0.5 * np.sum((fit - g.mean()) ** 2)
        rank = int(np.linalg.matrix_rank(X))
    df = rank - 1
    return {"stat": lm, "df": df, "p": float(stats.chi2.sf(lm, df))}


def white_test(e, X, cross: bool = True, max_terms: int = 300) -> dict:
    """White (1980) general test: ê² on regressors, their squares and (optionally) cross products."""
    S = _slopes(X)
    n, k = S.shape
    terms = [S, S ** 2]
    if cross and k > 1:
        pairs = [(i, j) for i in range(k) for j in range(i + 1, k)]
        if k + len(pairs) * 1 + k > min(n - 2, max_terms):
            cross = False
        else:
            terms.append(np.column_stack([S[:, i] * S[:, j] for i, j in pairs]))
    Z = np.column_stack([np.ones(n)] + terms)
    # drop duplicated / collinear columns (e.g. squares of dummies) — keep a full-rank subset
    Zs = Z / np.maximum(np.sqrt((Z ** 2).sum(axis=0)), 1e-300)
    q, rr, piv = linalg.qr(Zs, mode="economic", pivoting=True)
    rank = int(np.sum(np.abs(np.diag(rr)) > 1e-9 * np.abs(rr[0, 0])))
    Z = Z[:, np.sort(piv[:rank])]
    r2, rank = _aux_r2(e ** 2, Z)
    lm = n * r2
    df = rank - 1
    return {"stat": lm, "df": df, "p": float(stats.chi2.sf(lm, df)), "cross": cross}


def glejser(e, X) -> dict:
    """Glejser (1969): |ê| regressed on the regressors; joint F-test of the slopes."""
    Z = _with_const(X)
    a = np.abs(e)
    r2, rank = _aux_r2(a, Z)
    n = len(e)
    q = rank - 1
    F = (r2 / q) / ((1 - r2) / (n - rank))
    return {"stat": F, "df": (q, n - rank), "p": float(stats.f.sf(F, q, n - rank))}


def park(e, z) -> dict:
    """Park (1966): ln ê² = α + β ln|z| + v; t-test of β (z = fitted values or a chosen regressor)."""
    z = np.abs(np.asarray(z, float))
    ok = (z > 0) & (e != 0)
    ly, lz = np.log(e[ok] ** 2), np.log(z[ok])
    n = ok.sum()
    Z = np.c_[np.ones(n), lz]
    b, *_ = np.linalg.lstsq(Z, ly, rcond=None)
    res = ly - Z @ b
    s2 = res @ res / (n - 2)
    se = np.sqrt(s2 * np.linalg.inv(Z.T @ Z)[1, 1])
    t = b[1] / se
    return {"stat": t, "df": n - 2, "p": float(2 * stats.t.sf(abs(t), n - 2)), "beta": b[1]}


def goldfeld_quandt(y, X, z, drop: float = 0.2, alternative: str = "increasing") -> dict:
    """Goldfeld-Quandt (1965): order by z, omit the central `drop` share, compare residual variances."""
    order = np.argsort(np.asarray(z, float), kind="stable")
    y, X = np.asarray(y)[order], np.asarray(X)[order]
    n, k = X.shape
    c = int(round(n * drop))
    n1 = (n - c) // 2
    lo, hi = slice(0, n1), slice(n - n1, n)

    def rss(sl):
        b, *_ = np.linalg.lstsq(X[sl], y[sl], rcond=None)
        e = y[sl] - X[sl] @ b
        return e @ e
    df1 = df2 = n1 - k
    if df1 <= 0:
        return {"stat": np.nan, "df": None, "p": np.nan}
    F = (rss(hi) / df2) / (rss(lo) / df1)
    if alternative == "increasing":
        p = stats.f.sf(F, df2, df1)
    elif alternative == "decreasing":
        p = stats.f.cdf(F, df2, df1)
    else:
        p = 2 * min(stats.f.sf(F, df2, df1), stats.f.cdf(F, df2, df1))
    return {"stat": F, "df": (df2, df1), "p": float(p), "omitted": c}


# ==========================================================
# Autocorrelation
# ==========================================================
def durbin_watson(e) -> float:
    e = np.asarray(e, float)
    return float(np.sum(np.diff(e) ** 2) / np.sum(e ** 2))


def _imhof_cdf0(lam: np.ndarray) -> float:
    """P(Σ λᵢ zᵢ² < 0) for independent standard normals (Imhof 1961)."""
    lam = lam[np.abs(lam) > 1e-12]

    @np.errstate(over="ignore", invalid="ignore")
    def f(u):
        theta = 0.5 * np.sum(np.arctan(lam * u))
        rho = np.exp(0.25 * np.sum(np.log1p((lam * u) ** 2)))
        return np.sin(theta) / (u * rho)
    val, _ = integrate.quad(f, 0, np.inf, limit=500)
    return float(np.clip(0.5 - val / np.pi, 0.0, 1.0))


def durbin_watson_exact(e, X, max_n: int = 1500) -> dict:
    """Durbin-Watson d with its exact p-value for the given regressor matrix (Imhof algorithm, as in
    R's lmtest::dwtest). H1: positive first-order autocorrelation (ρ > 0); the two-sided p is also returned."""
    d = durbin_watson(e)
    n, k = X.shape
    out = {"stat": d, "rho_hat": 1 - d / 2, "p": np.nan, "p_two": np.nan, "method": "exact"}
    if n > max_n:
        # normal approximation (Durbin & Watson 1971 moments, crude) for very large samples
        z = (d - 2) / (2 / np.sqrt(n))
        out.update(p=float(stats.norm.cdf(z)), p_two=float(2 * stats.norm.sf(abs(z))), method="normal approx.")
        return out
    Q = linalg.null_space(X.T)                 # orthonormal basis of the residual space (n × (n−k))
    A = np.diag(np.r_[1, np.full(n - 2, 2), 1]) - np.eye(n, k=1) - np.eye(n, k=-1)
    nu = np.linalg.eigvalsh(Q.T @ A @ Q)
    p_low = _imhof_cdf0(nu - d)                # P(D < d)
    out.update(p=p_low, p_two=float(2 * min(p_low, 1 - p_low)))
    return out


def breusch_godfrey(e, X, nlags: int = 1) -> dict:
    """Breusch (1978) – Godfrey (1978) LM test for AR(p)/MA(p) errors; valid with lagged Y as regressor."""
    e = np.asarray(e, float)
    n = len(e)
    L = np.column_stack([np.r_[np.zeros(j), e[:-j]] for j in range(1, nlags + 1)])
    Z = np.c_[X, L]
    b, *_ = np.linalg.lstsq(Z, e, rcond=None)
    u = e - Z @ b
    sst = np.sum((e - e.mean()) ** 2)
    r2 = 1 - u @ u / sst
    lm = n * r2
    k = Z.shape[1]
    # F version: test the lag coefficients jointly
    b0, *_ = np.linalg.lstsq(X, e, rcond=None)
    u0 = e - X @ b0
    F = ((u0 @ u0 - u @ u) / nlags) / (u @ u / (n - k))
    return {"stat": lm, "df": nlags, "p": float(stats.chi2.sf(lm, nlags)), "F": F,
            "pF": float(stats.f.sf(F, nlags, n - k))}


def ljung_box(e, lags: int = 10) -> dict:
    e = np.asarray(e, float) - np.mean(e)
    n = len(e)
    acf = np.array([np.sum(e[j:] * e[:-j]) for j in range(1, lags + 1)]) / np.sum(e ** 2)
    Qs = n * (n + 2) * np.sum(acf ** 2 / (n - np.arange(1, lags + 1)))
    return {"stat": float(Qs), "df": lags, "p": float(stats.chi2.sf(Qs, lags)), "acf": acf}


def durbin_h(e, X, se_lag: float, d: float | None = None) -> dict:
    """Durbin (1970) h statistic for models with a lagged dependent variable."""
    n = len(e)
    d = durbin_watson(e) if d is None else d
    v = n * se_lag ** 2
    if v >= 1:
        return {"stat": np.nan, "df": None, "p": np.nan, "note": "n·Var(b_lag) ≥ 1: h undefined — use Breusch-Godfrey."}
    h = (1 - d / 2) * np.sqrt(n / (1 - v))
    return {"stat": float(h), "df": None, "p": float(2 * stats.norm.sf(abs(h)))}


# ==========================================================
# Multicollinearity
# ==========================================================
def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Centered VIFs (as in Stata `estat vif`): VIFⱼ = 1 / (1 − R²ⱼ) from xⱼ on the other regressors + constant."""
    cols = [c for c in X.columns if c != "const" and X[c].std() > 0]
    rows = []
    M = X[cols].to_numpy(float)
    n = len(M)
    for j, c in enumerate(cols):
        others = np.c_[np.ones(n), np.delete(M, j, axis=1)]
        r2, _ = _aux_r2(M[:, j], others) if len(cols) > 1 else (0.0, 1)
        vif = 1 / (1 - r2) if r2 < 1 else np.inf
        rows.append({"Variable": c, "VIF": vif, "Tolerance (1/VIF)": 1 / vif, "R²ⱼ": r2})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["Concern"] = np.where(out["VIF"] >= 10, "serious", np.where(out["VIF"] >= 5, "moderate", "none"))
    return out


def condition_indices(X: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Belsley, Kuh & Welsch (1980): columns scaled to unit length (constant included), condition indices
    and variance-decomposition proportions. CI > 30 with ≥ 2 proportions > .5 signals a damaging dependency."""
    M = X.to_numpy(float)
    S = M / np.sqrt((M ** 2).sum(axis=0))
    _, s, Vt = np.linalg.svd(S, full_matrices=False)
    ci = s.max() / s
    phi = (Vt.T ** 2) / s ** 2                  # k × dims
    pi = (phi / phi.sum(axis=1, keepdims=True)).T
    tab = pd.DataFrame(pi, columns=X.columns)
    tab.insert(0, "Condition index", ci)
    tab.insert(0, "Eigenvalue", s ** 2)
    tab.insert(0, "Dimension", np.arange(1, len(s) + 1))
    return tab, float(ci.max())


# ==========================================================
# Specification & stability
# ==========================================================
def reset_test(y, X, fitted, power: int = 3) -> dict:
    """Ramsey (1969) RESET: add ŷ², …, ŷᵖ and F-test them jointly."""
    y = np.asarray(y, float)
    f = np.asarray(fitted, float)
    fs = f / np.std(f) if np.std(f) > 0 else f
    extra = np.column_stack([fs ** p for p in range(2, power + 1)])
    Z = np.c_[X, extra]
    n, k = Z.shape
    q = extra.shape[1]

    def rss(M):
        b, *_ = np.linalg.lstsq(M, y, rcond=None)
        e = y - M @ b
        return e @ e
    r0, r1 = rss(X), rss(Z)
    F = ((r0 - r1) / q) / (r1 / (n - k))
    return {"stat": F, "df": (q, n - k), "p": float(stats.f.sf(F, q, n - k))}


def chow_test(y, X, break_at: int) -> dict:
    """Chow (1960) break test at observation position `break_at` (0-based; first obs of regime 2).
    Uses the predictive (forecast) version when one sub-sample has fewer than k observations."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape

    def rss(yy, XX):
        b, *_ = np.linalg.lstsq(XX, yy, rcond=None)
        e = yy - XX @ b
        return e @ e
    n1, n2 = break_at, n - break_at
    rp = rss(y, X)
    if n1 > k and n2 > k:
        r1, r2 = rss(y[:n1], X[:n1]), rss(y[n1:], X[n1:])
        F = ((rp - r1 - r2) / k) / ((r1 + r2) / (n - 2 * k))
        return {"stat": F, "df": (k, n - 2 * k), "p": float(stats.f.sf(F, k, n - 2 * k)), "kind": "break"}
    if n1 > k:
        r1 = rss(y[:n1], X[:n1])
        F = ((rp - r1) / n2) / (r1 / (n1 - k))
        return {"stat": F, "df": (n2, n1 - k), "p": float(stats.f.sf(F, n2, n1 - k)), "kind": "predictive"}
    return {"stat": np.nan, "df": None, "p": np.nan, "kind": "too few observations"}


def recursive_residuals(y, X) -> tuple[np.ndarray, int]:
    """Brown, Durbin & Evans (1975) recursive residuals wₜ, t = k+1 … n (Sherman-Morrison updating)."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    n, k = X.shape
    start = k
    while start < n and np.linalg.matrix_rank(X[:start]) < k:
        start += 1
    P = np.linalg.pinv(X[:start].T @ X[:start])
    b = P @ X[:start].T @ y[:start]
    w = []
    for t in range(start, n):
        x = X[t]
        f = 1 + x @ P @ x
        w.append((y[t] - x @ b) / np.sqrt(f))
        Px = P @ x
        P = P - np.outer(Px, Px) / f
        b = b + Px * (y[t] - x @ b) / f
    return np.asarray(w), start


@lru_cache(maxsize=64)
def _cusumsq_c0(m: int, alpha: float = 0.05, reps: int = 4000) -> float:
    """Critical value of max |Sᵣ − r/m| under H0 by simulation (seeded, reproducible)."""
    rng = np.random.default_rng(1975)
    z2 = rng.standard_normal((reps, m)) ** 2
    S = np.cumsum(z2, axis=1) / z2.sum(axis=1, keepdims=True)
    dev = np.abs(S - np.arange(1, m + 1) / m).max(axis=1)
    return float(np.quantile(dev, 1 - alpha))


def cusum(y, X, alpha: float = 0.05) -> dict:
    w, start = recursive_residuals(y, X)
    n, k = np.asarray(X).shape
    m = len(w)
    sig = np.std(w, ddof=1)
    W = np.cumsum(w) / sig
    a = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}[alpha]
    r = np.arange(1, m + 1)
    bound = a * (np.sqrt(m) + 2 * r / np.sqrt(m))
    S = np.cumsum(w ** 2) / np.sum(w ** 2)
    expected = r / m
    c0 = _cusumsq_c0(m, alpha)
    t = np.arange(start, n)
    out = pd.DataFrame({"t": t, "w": w, "CUSUM": W, "lower": -bound, "upper": bound,
                        "CUSUMSQ": S, "expected": expected, "sq_lower": expected - c0, "sq_upper": expected + c0})
    cusum_ok = bool(np.all(np.abs(W) <= bound))
    sq_ok = bool(np.all(np.abs(S - expected) <= c0))
    return {"path": out, "cusum_ok": cusum_ok, "cusumsq_ok": sq_ok, "c0": c0,
            "cusum_max": float(np.max(np.abs(W) / bound)), "sq_max": float(np.max(np.abs(S - expected)))}


# ==========================================================
# Normality
# ==========================================================
def normality(e) -> list[dict]:
    e = np.asarray(e, float)
    n = len(e)
    out = []
    jb = stats.jarque_bera(e)
    out.append({"key": "jb", "stat": float(jb.statistic), "df": 2, "p": float(jb.pvalue),
                "skew": float(stats.skew(e)), "kurt": float(stats.kurtosis(e, fisher=False))})
    if 3 <= n <= 5000:
        sw = stats.shapiro(e)
        out.append({"key": "sw", "stat": float(sw.statistic), "df": None, "p": float(sw.pvalue)})
    if n >= 20:
        k2 = stats.normaltest(e)
        out.append({"key": "k2", "stat": float(k2.statistic), "df": 2, "p": float(k2.pvalue)})
    from statsmodels.stats.diagnostic import normal_ad
    ad, pad = normal_ad(e)
    out.append({"key": "ad", "stat": float(ad), "df": None, "p": float(pad)})
    return out


# ==========================================================
# Influence
# ==========================================================
def influence(e, X, names=None, index=None) -> pd.DataFrame:
    """Leverage, internally/externally studentized residuals, Cook's D, DFFITS, DFBETAS (exact formulas)."""
    e, X = np.asarray(e, float), np.asarray(X, float)
    n, k = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)
    h = np.clip(h, 0, 1 - 1e-12)
    s2 = e @ e / (n - k)
    r_int = e / np.sqrt(s2 * (1 - h))
    s2_i = ((n - k) * s2 - e ** 2 / (1 - h)) / (n - k - 1)
    t_ext = e / np.sqrt(s2_i * (1 - h))
    cooks = r_int ** 2 * h / (k * (1 - h))
    dffits = t_ext * np.sqrt(h / (1 - h))
    dfbeta = (X @ XtXi) * (e / (1 - h))[:, None]
    dfbetas = dfbeta / np.sqrt(np.outer(s2_i, np.diag(XtXi)))
    out = pd.DataFrame({"obs": index if index is not None else np.arange(n), "Leverage": h,
                        "Std. residual": r_int, "Stud. residual": t_ext, "Cook's D": cooks, "DFFITS": dffits})
    names = names or [f"b{j}" for j in range(k)]
    for j, nm in enumerate(names):
        out[f"DFBETAS {nm}"] = dfbetas[:, j]
    lev_c, cook_c, dff_c, dfb_c = 2 * k / n, 4 / n, 2 * np.sqrt(k / n), 2 / np.sqrt(n)
    flags = []
    for i in range(n):
        f = []
        if h[i] > lev_c:
            f.append("leverage")
        if abs(t_ext[i]) > 3:
            f.append("outlier")
        if cooks[i] > cook_c:
            f.append("Cook")
        if abs(dffits[i]) > dff_c:
            f.append("DFFITS")
        if np.any(np.abs(dfbetas[i]) > dfb_c):
            f.append("DFBETAS")
        flags.append(", ".join(f))
    out["Flags"] = flags
    out.attrs["cutoffs"] = {"leverage": lev_c, "cook": cook_c, "dffits": dff_c, "dfbetas": dfb_c, "student": 3}
    return out


# ==========================================================
# Full panel
# ==========================================================
TEST_LABELS = {
    "bp": ("Breusch-Pagan / Koenker (studentized)", "بروش-باجان / كوينكر (المُعيَّر)"),
    "bp_orig": ("Breusch-Pagan (original, normal errors)", "بروش-باجان (الأصلي، بافتراض الطبيعية)"),
    "white": ("White (squares & cross products)", "وايت (المربعات والحدود التقاطعية)"),
    "white_nc": ("White (squares only)", "وايت (المربعات فقط)"),
    "glejser": ("Glejser", "جليجسر"),
    "park": ("Park", "بارك"),
    "gq": ("Goldfeld-Quandt", "جولدفيلد-كواندت"),
    "dw": ("Durbin-Watson (exact p, H₁: ρ > 0)", "دربن-واتسون (قيمة p دقيقة، الفرض البديل ρ > 0)"),
    "bg": ("Breusch-Godfrey LM", "بروش-جودفري LM"),
    "lb": ("Ljung-Box Q", "ليونج-بوكس Q"),
    "dh": ("Durbin h (lagged dependent variable)", "دربن h (مع متغير تابع مُبطأ)"),
    "reset": ("Ramsey RESET", "رامزي RESET"),
    "chow": ("Chow", "شاو"),
    "cusum": ("CUSUM", "CUSUM"),
    "cusumsq": ("CUSUM of squares", "CUSUM للمربعات"),
    "jb": ("Jarque-Bera", "جارك-بيرا"),
    "sw": ("Shapiro-Wilk", "شابيرو-ويلك"),
    "k2": ("D'Agostino-Pearson K²", "داجوستينو-بيرسون K²"),
    "ad": ("Anderson-Darling", "أندرسون-دارلنج"),
    "vif": ("Max VIF", "أكبر معامل تضخم تباين VIF"),
    "cond": ("Condition number (Belsley)", "رقم الحالة (بيلسلي)"),
    "infl": ("Influential observations (Cook's D > 4/n)", "المشاهدات المؤثرة (مسافة كوك > 4/n)"),
}
GROUPS = {
    "hetero": ("Heteroskedasticity", "عدم تجانس التباين", "H₀: constant error variance", "فرض العدم: تباين الخطأ ثابت"),
    "autocorr": ("Autocorrelation", "الارتباط الذاتي", "H₀: no serial correlation", "فرض العدم: لا ارتباط ذاتي"),
    "spec": ("Functional form & stability", "الشكل الدالي والاستقرار", "H₀: correct specification / stable parameters",
             "فرض العدم: توصيف صحيح ومعاملات مستقرة"),
    "normality": ("Normality of residuals", "طبيعية البواقي", "H₀: residuals are normal", "فرض العدم: البواقي طبيعية"),
    "multicol": ("Multicollinearity", "الازدواج الخطي", "rule: VIF < 10, CN < 30", "القاعدة: VIF < 10 ورقم الحالة < 30"),
    "influence": ("Influential observations", "المشاهدات المؤثرة", "rule: few points with Cook's D > 4/n",
                  "القاعدة: قلة النقاط ذات مسافة كوك > 4/n"),
}
REMEDIES = {
    "hetero": ("Use heteroskedasticity-robust SEs (HC3 for n < 250, HC1 otherwise), or WLS/FGLS when the variance "
               "pattern is known; a log transformation of Y often stabilises the variance.",
               "استخدم أخطاء معيارية متينة (HC3 للعينات أقل من 250 مشاهدة، وHC1 فيما عداها)، أو المربعات الصغرى "
               "الموزونة / المعممة الممكنة عند معرفة نمط التباين؛ وكثيراً ما يُثبِّت أخذ لوغاريتم المتغير التابع التباين."),
    "autocorr": ("Use Newey-West HAC SEs or GLS with AR(1) errors, and reconsider the dynamics (lags, trend). "
                 "Meaningful only when observations have a natural order (time).",
                 "استخدم أخطاء نيوي-ويست المتينة أو GLS بأخطاء AR(1)، وأعد النظر في الديناميكية (الإبطاءات والاتجاه). "
                 "ولا معنى لهذا الاختبار إلا إذا كان للمشاهدات ترتيب طبيعي (زمني)."),
    "spec": ("Revise the functional form (logs, quadratic terms, interactions) or add omitted variables; "
             "for a break, add regime dummies or split the sample.",
             "راجع الشكل الدالي (اللوغاريتمات، أو الحدود التربيعية، أو التفاعلات) أو أضف المتغيرات المحذوفة؛ "
             "وعند وجود انكسار أضف متغيرات صورية للفترات أو قسّم العينة."),
    "normality": ("With large n the t and F tests remain valid (central limit theorem); otherwise use bootstrap SEs, "
                  "inspect outliers, or estimate a robust regression.",
                  "مع العينات الكبيرة تبقى اختبارات t وF صالحة (نظرية النهاية المركزية)؛ وإلا فاستخدم أخطاء البوتستراب، "
                  "وافحص القيم الشاذة، أو قدّر انحداراً متيناً."),
    "multicol": ("Centre variables used in powers / interactions, combine or drop redundant regressors, or collect "
                 "more data. Coefficients stay unbiased but imprecise.",
                 "اطرح المتوسط من المتغيرات الداخلة في القوى والتفاعلات، أو ادمج المتغيرات المتكررة أو احذفها، أو زد "
                 "حجم البيانات. تبقى المعاملات غير متحيزة لكنها غير دقيقة."),
    "influence": ("Inspect the flagged observations for data errors; report results with and without them, "
                  "or use robust (Huber / bisquare) regression.",
                  "افحص المشاهدات المعلَّمة بحثاً عن أخطاء إدخال، واعرض النتائج بها وبدونها، أو استخدم الانحدار "
                  "المتين (هوبر أو ثنائي التربيع)."),
}


def run_all(r, lb_lags: int | None = None, bg_lags: int = 2, reset_power: int = 3, gq_by=None,
            chow_at: int | None = None, lagged_y: str | None = None, time_ordered: bool = True) -> dict:
    """Run the complete diagnostic battery on a fitted RegResult. Returns rows + detail objects."""
    e = np.asarray(r.wresid, float)
    X = np.asarray(r.wexog, float)
    n, k = X.shape
    names = r.names
    rows, detail = [], {}
    Xdf = pd.DataFrame(X, columns=names)
    linear = r.method != "nls"

    # --- heteroskedasticity
    g = "hetero"
    h0, h0a = GROUPS[g][2], GROUPS[g][3]
    if k > 1:
        bp = breusch_pagan(e, X, True)
        rows.append(_row(g, *TEST_LABELS["bp"], bp["stat"], bp["df"], bp["p"], h0, h0a))
        bpo = breusch_pagan(e, X, False)
        rows.append(_row(g, *TEST_LABELS["bp_orig"], bpo["stat"], bpo["df"], bpo["p"], h0, h0a))
        wt = white_test(e, X)
        rows.append(_row(g, *TEST_LABELS["white" if wt["cross"] else "white_nc"], wt["stat"], wt["df"], wt["p"], h0, h0a))
        gl = glejser(e, X)
        rows.append(_row(g, *TEST_LABELS["glejser"], gl["stat"], gl["df"], gl["p"], h0, h0a))
    zvar = np.asarray(r.fitted if gq_by is None else gq_by, float)
    if len(zvar) == n:
        pk = park(e, zvar)
        rows.append(_row(g, *TEST_LABELS["park"], pk["stat"], pk["df"], pk["p"], h0, h0a))
        ytarget = (X @ r.params.to_numpy() + e) if linear else np.asarray(r.y, float)
        gq = goldfeld_quandt(ytarget, X, zvar)
        rows.append(_row(g, *TEST_LABELS["gq"], gq["stat"], gq["df"], gq["p"], h0, h0a,
                         note=(f"ordered by {'fitted values' if gq_by is None else 'selected variable'}, central {gq.get('omitted', 0)} obs omitted",
                               f"مرتبة حسب {'القيم المقدَّرة' if gq_by is None else 'المتغير المختار'}، مع حذف {gq.get('omitted', 0)} مشاهدة وسطى")))

    # --- autocorrelation
    g = "autocorr"
    h0, h0a = GROUPS[g][2], GROUPS[g][3]
    note = "" if time_ordered else "data order may not be meaningful"
    dw = durbin_watson_exact(e, X) if linear else {"stat": durbin_watson(e), "p": np.nan, "rho_hat": np.nan}
    rows.append(_row(g, *TEST_LABELS["dw"], dw["stat"], None, dw["p"], h0, h0a,
                     note=(f"ρ̂ ≈ {dw['rho_hat']:.3f}", f"ρ̂ ≈ {dw['rho_hat']:.3f}")))
    detail["dw"] = dw
    for L in sorted({1, bg_lags}):
        bg = breusch_godfrey(e, X, L)
        rows.append(_row(g, TEST_LABELS["bg"][0] + f" (lags = {L})", TEST_LABELS["bg"][1] + f" (الإبطاءات = {L})",
                         bg["stat"], bg["df"], bg["p"], h0, h0a, note=f"F = {bg['F']:.3f}, p = {bg['pF']:.4f}"))
    L = lb_lags or int(max(1, min(10, n // 5)))
    lb = ljung_box(e, L)
    rows.append(_row(g, TEST_LABELS["lb"][0] + f" ({L})", TEST_LABELS["lb"][1] + f" ({L})", lb["stat"], lb["df"],
                     lb["p"], h0, h0a))
    detail["acf"] = lb["acf"]
    if lagged_y and lagged_y in names:
        dh = durbin_h(e, X, float(r.bse[lagged_y]), dw["stat"])
        rows.append(_row(g, *TEST_LABELS["dh"], dh["stat"], None, dh["p"], h0, h0a, note=(dh.get("note", ""), "n·Var(b_lag) ≥ 1: إحصاءة h غير معرَّفة — استخدم بروش-جودفري." if dh.get("note") else "")))

    # --- specification & stability
    g = "spec"
    h0, h0a = GROUPS[g][2], GROUPS[g][3]
    ytarget = (X @ r.params.to_numpy() + e)
    if linear:
        rs = reset_test(ytarget, X, X @ r.params.to_numpy(), reset_power)
        rows.append(_row(g, TEST_LABELS["reset"][0] + f" (ŷ² … ŷ^{reset_power})",
                         TEST_LABELS["reset"][1] + f" (ŷ² … ŷ^{reset_power})", rs["stat"], rs["df"], rs["p"], h0, h0a))
        if chow_at is not None and 0 < chow_at < n:
            ch = chow_test(ytarget, X, chow_at)
            rows.append(_row(g, TEST_LABELS["chow"][0] + f" ({ch['kind']}, break at obs {chow_at + 1})",
                             TEST_LABELS["chow"][1] + f" (الانكسار عند المشاهدة {chow_at + 1})",
                             ch["stat"], ch["df"], ch["p"], h0, h0a))
        if n - k >= 8:
            cs = cusum(ytarget, X)
            detail["cusum"] = cs
            rows.append(_row(g, *TEST_LABELS["cusum"], cs["cusum_max"], None, None, h0, h0a, ok=cs["cusum_ok"],
                             note=("max |W|/bound (≤ 1 = stable at 5%)", "أقصى |W|/الحد (≤ 1 = مستقر عند 5%)")))
            rows.append(_row(g, *TEST_LABELS["cusumsq"], cs["sq_max"], None, None, h0, h0a, ok=cs["cusumsq_ok"],
                             note=(f"max deviation; 5% critical value c₀ = {cs['c0']:.3f}",
                                   f"أقصى انحراف؛ القيمة الحرجة عند 5% c₀ = {cs['c0']:.3f}")))

    # --- normality
    g = "normality"
    h0, h0a = GROUPS[g][2], GROUPS[g][3]
    for t in normality(e):
        rows.append(_row(g, *TEST_LABELS[t["key"]], t["stat"], t["df"], t["p"], h0, h0a,
                         note=((f"skewness = {t['skew']:.3f}, kurtosis = {t['kurt']:.3f}",
                               f"الالتواء = {t['skew']:.3f}، التفلطح = {t['kurt']:.3f}") if t["key"] == "jb" else "")))
        if t["key"] == "jb":
            detail["skew"], detail["kurt"] = t["skew"], t["kurt"]

    # --- multicollinearity
    g = "multicol"
    if linear and k > 2:
        vt = vif_table(Xdf)
        detail["vif"] = vt
        if not vt.empty:
            mv = float(vt["VIF"].max())
            rows.append(_row(g, *TEST_LABELS["vif"], mv, None, None, GROUPS[g][2], GROUPS[g][3], ok=mv < 10,
                             note=f"{vt.loc[vt['VIF'].idxmax(), 'Variable']}"))
        ct, cn = condition_indices(Xdf)
        detail["cond"] = ct
        rows.append(_row(g, *TEST_LABELS["cond"], cn, None, None, GROUPS[g][2], GROUPS[g][3], ok=cn < 30))

    # --- influence
    g = "influence"
    if linear and n - k > 1:
        inf = influence(e, X, names, index=(r.design.index[-n:] if r.design is not None else None))
        detail["influence"] = inf
        share = float((inf["Cook's D"] > 4 / n).mean())
        rows.append(_row(g, *TEST_LABELS["infl"], int((inf["Cook's D"] > 4 / n).sum()), None, None,
                         GROUPS[g][2], GROUPS[g][3], ok=share <= 0.05,
                         note=(f"{share:.1%} of sample; max Cook's D = {inf["Cook's D"].max():.3f}",
                               f"{share:.1%} من العينة؛ أقصى مسافة كوك = {inf["Cook's D"].max():.3f}")))

    tab = pd.DataFrame(rows)
    if not time_ordered:  # autocorrelation / stability tests are informational for unordered cross-sections
        tab["ok"] = tab["ok"].astype(object)
        mask = tab["group"].eq("autocorr") | tab["Test"].str.startswith(("CUSUM", "Chow"))
        tab.loc[mask, "ok"] = None
        tab.loc[mask, "note"] = tab.loc[mask, "note"].astype(str) + " [informational: observations not time-ordered]"
        tab.loc[mask, "note_ar"] = tab.loc[mask, "note_ar"].astype(str) + " [للعلم فقط: المشاهدات غير مرتبة زمنياً]"
    problems = {}
    for g in GROUPS:
        sub = tab[tab["group"] == g]
        if sub.empty:
            continue
        failed = sub[sub["ok"] == False]  # noqa: E712
        n_tests = int(sub["ok"].notna().sum())
        if n_tests == 0:
            continue
        problems[g] = {"n_tests": n_tests, "n_failed": len(failed), "failed": failed["Test"].tolist(),
                       "failed_ar": failed["Test_ar"].tolist()}
    return {"table": tab, "detail": detail, "problems": problems, "n": n, "k": k}


def verdicts(diag: dict, lang: str = "en") -> list[dict]:
    """One line per assumption: holds / violated / mixed + remedy, for the dashboard and the AI layer."""
    out = []
    i = 1 if lang == "ar" else 0
    for g, info in diag["problems"].items():
        nf, nt = info["n_failed"], info["n_tests"]
        if nf == 0:
            status = "ok"
        elif g in ("hetero", "autocorr", "normality", "spec") and nf < max(2, nt / 2):
            status = "mixed"
        else:
            status = "fail"
        out.append({"group": g, "label": GROUPS[g][i], "status": status,
                    "failed": info["failed_ar"] if lang == "ar" else info["failed"],
                    "remedy": REMEDIES[g][i] if status != "ok" else ""})
    return out


def pre_checks(df: pd.DataFrame, spec, design) -> list[dict]:
    """Assumption checks BEFORE estimation: sample size, variation, missing data, skewness, outliers, VIF."""
    from analytics.regression import build_design  # noqa: F401  (design passed in)
    X, y = design.X, design.y
    n, k = X.shape
    out = []

    def add(ok, en, ar):
        out.append({"ok": ok, "en": en, "ar": ar})
    ratio = n / max(1, k - 1)
    add(ratio >= 10, f"Observations per regressor: {ratio:.1f} (rule of thumb ≥ 10–20).",
        f"عدد المشاهدات لكل متغير مستقل: {ratio:.1f} (القاعدة التقريبية ≥ 10–20).")
    total = len(df)
    lost = total - n
    add(lost / max(total, 1) < 0.2, f"Estimation sample: {n} of {total} rows ({lost} lost to missing / invalid values).",
        f"عينة التقدير: {n} من {total} صفاً (فُقد {lost} بسبب القيم المفقودة أو غير الصالحة).")
    sk = float(stats.skew(y)) if n > 2 else 0.0
    add(abs(sk) < 1, f"Skewness of the dependent variable ({design.y_label}): {sk:.2f}"
        + (" — consider a log form." if abs(sk) >= 1 and design.info["ytx"] == "level" else "."),
        f"التواء المتغير التابع ({design.y_label}): {sk:.2f}"
        + (" — فكّر في الصيغة اللوغاريتمية." if abs(sk) >= 1 and design.info["ytx"] == "level" else "."))
    z = np.abs(stats.zscore(y)) if n > 2 and np.std(y) > 0 else np.zeros(n)
    no = int((z > 3).sum())
    add(no <= max(1, 0.01 * n), f"Extreme values of Y (|z| > 3): {no}.", f"القيم المتطرفة في Y (|z| > 3): {no}.")
    if k > 2:
        vt = vif_table(X)
        if not vt.empty:
            mv = float(vt["VIF"].max())
            add(mv < 10, f"Highest VIF before estimation: {mv:.2f} ({vt.loc[vt['VIF'].idxmax(), 'Variable']}).",
                f"أعلى معامل تضخم تباين قبل التقدير: {mv:.2f} ({vt.loc[vt['VIF'].idxmax(), 'Variable']}).")
    for note in design.notes:
        add(False, note, note)
    return out
