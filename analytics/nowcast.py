# ==========================================
# Aetheris Research - Macro forecasting & nowcasting (v1.9, Catalogue §3.2)
# Bridge equations, MIDAS, dynamic factor models (mixed frequency), Bayesian VAR (Minnesota / NIW, GLP 2015
# marginal-likelihood hyperparameter) and local projections (Jordà 2005).
# ==========================================

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats
from scipy.special import multigammaln

warnings.filterwarnings("ignore")


# ==========================================================
# Mixed-frequency helpers
# ==========================================================
def monthly_frame(df: pd.DataFrame, date: str, cols: list) -> pd.DataFrame:
    d = df.copy()
    d[date] = pd.to_datetime(d[date])
    d = d.sort_values(date).set_index(date)
    out = d[cols].astype(float)
    out.index = out.index.to_period("M")
    return out


def target_quarterly(df: pd.DataFrame, date: str, target: str) -> pd.Series:
    m = monthly_frame(df, date, [target])[target]
    q = m.groupby(m.index.asfreq("Q")).apply(lambda s: s.dropna().iloc[-1] if s.notna().any() else np.nan)
    return q


def fill_ragged(x: pd.Series, until: pd.Period, max_lag: int = 6) -> tuple:
    """Extend a monthly indicator to `until` with AR(p) forecasts (p ≤ max_lag by AIC), as in bridge-equation practice.
    Returns the completed series and the list of filled months."""
    from statsmodels.tsa.ar_model import AutoReg, ar_select_order
    s = x.copy()
    last = s.last_valid_index()
    s = s.loc[:last]
    if last >= until:
        return x.loc[:until], []
    h = (until - last).n
    v = s.dropna().to_numpy()
    try:
        sel = ar_select_order(v, maxlag=min(max_lag, len(v) // 5), ic="aic", trend="c")
        lags = sel.ar_lags or [1]
        r = AutoReg(v, lags=lags, trend="c").fit()
        f = r.predict(start=len(v), end=len(v) + h - 1)
    except Exception:  # noqa: BLE001
        f = np.full(h, v[-12:].mean())
    idx = pd.period_range(last + 1, until, freq="M")
    filled = pd.concat([s, pd.Series(np.asarray(f), index=idx)])
    return filled, list(idx)


def _quarter_mean(m: pd.Series) -> pd.Series:
    g = m.groupby(m.index.asfreq("Q"))
    out = g.mean()
    out[g.count() < 3] = np.nan
    return out


# ==========================================================
# Bridge equations
# ==========================================================
def bridge(df: pd.DataFrame, date: str, target: str, indicators: list, y_lag: bool = False, eval_quarters: int = 12) -> dict:
    """y_q = α + Σ β_k x̄_k,q (+ ρ y_{q−1}) + e, with x̄ the quarterly mean of monthly indicators. Missing months in
    the quarters to nowcast are filled by AR forecasts of each indicator. Pseudo out-of-sample evaluation re-estimates
    the equation each quarter and compares with an AR(1) benchmark for the target (Diebold-Mariano)."""
    import statsmodels.api as sm
    from analytics.fcst import dm_test
    M = monthly_frame(df, date, indicators)
    yq = target_quarterly(df, date, target)
    last_obs = yq.last_valid_index()
    end_q = M.index.max().asfreq("Q")
    nq = [q for q in pd.period_range(last_obs + 1, end_q, freq="Q")]
    until = (end_q.asfreq("M", "end")) if nq else M.index.max()
    filled, info = {}, []
    for c in indicators:
        f, idx = fill_ragged(M[c], until)
        filled[c] = f
        if idx:
            info.append({"Indicator": c, "Months forecast (AR)": ", ".join(str(i) for i in idx)})
    Mf = pd.DataFrame(filled)
    Xq = pd.DataFrame({c: _quarter_mean(Mf[c]) for c in indicators})
    data = Xq.join(yq.rename("y"), how="left")
    if y_lag:
        data["y_lag"] = data["y"].shift(1)
    regs = indicators + (["y_lag"] if y_lag else [])
    est = data.dropna(subset=regs + ["y"])
    X = sm.add_constant(est[regs])
    m = sm.OLS(est["y"], X).fit()
    mh = sm.OLS(est["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    tab = pd.DataFrame({"Parameter": X.columns, "Coef.": m.params.values, "Std. Err.": m.bse.values, "Std. Err. (NW)": mh.bse.values,
                        "t": m.tvalues.values, "p-value": m.pvalues.values})
    rows = []
    prev = data["y"].copy()
    for q in nq:
        xr = data.loc[[q], indicators].copy()
        if y_lag:
            xr["y_lag"] = prev.get(q - 1, np.nan)
        xx = sm.add_constant(xr, has_constant="add")
        pr = m.get_prediction(xx).summary_frame(alpha=0.05)
        v = float(pr["mean"].iloc[0])
        prev[q] = v
        contrib = {c: float(m.params[c] * (xr[c].iloc[0] - est[c].mean())) for c in indicators}
        rows.append({"Quarter": str(q), "Nowcast": v, "lower 95%": float(pr["obs_ci_lower"].iloc[0]),
                     "upper 95%": float(pr["obs_ci_upper"].iloc[0]),
                     "Months observed (average)": float(np.mean([sum(1 for p in M[c].dropna().index if p.asfreq("Q") == q) for c in indicators])),
                     **{f"contrib: {c}": contrib[c] for c in indicators}})
    # pseudo out-of-sample
    ev = []
    qs = est.index[-eval_quarters:] if eval_quarters else []
    for q in qs:
        tr = est.loc[est.index < q]
        if len(tr) < len(regs) + 8:
            continue
        mm = sm.OLS(tr["y"], sm.add_constant(tr[regs])).fit()
        f = float(mm.predict(sm.add_constant(est.loc[[q], regs], has_constant="add")).iloc[0])
        yy = yq.loc[:q - 1].dropna()
        ar = sm.OLS(yy.iloc[1:].to_numpy(), sm.add_constant(yy.iloc[:-1].to_numpy())).fit()
        fb = float(ar.params[0] + ar.params[1] * yy.iloc[-1])
        ev.append({"Quarter": str(q), "Actual": float(est.loc[q, "y"]), "Bridge": f, "AR(1)": fb})
    ev = pd.DataFrame(ev)
    summ = {}
    if len(ev) >= 4:
        e1, e2 = ev["Actual"] - ev["Bridge"], ev["Actual"] - ev["AR(1)"]
        summ = {"RMSE bridge": float(np.sqrt(np.mean(e1 ** 2))), "RMSE AR(1)": float(np.sqrt(np.mean(e2 ** 2))),
                "relative RMSE": float(np.sqrt(np.mean(e1 ** 2)) / np.sqrt(np.mean(e2 ** 2))), **{f"DM {k}": v for k, v in dm_test(e1, e2).items() if k in ("DM", "p")}}
    return {"table": tab, "model": m, "nowcasts": pd.DataFrame(rows), "filled": pd.DataFrame(info), "quarterly": data,
            "evaluation": ev, "eval_summary": summ, "R²": float(m.rsquared), "N": int(m.nobs)}


# ==========================================================
# MIDAS (Ghysels, Santa-Clara & Valkanov 2004; Ghysels, Sinko & Valkanov 2007)
# ==========================================================
WEIGHTS = {"umidas": ("Unrestricted (U-MIDAS)", "غير مقيد (U-MIDAS)"), "nealmon": ("Normalised exponential Almon", "ألمون الأسي المعياري"),
           "beta": ("Beta (normalised)", "بيتا (معياري)"), "almon": ("Almon polynomial (degree 2)", "كثيرة حدود ألمون (درجة 2)")}


def midas_weights(theta, K: int, kind: str) -> np.ndarray:
    j = np.arange(1, K + 1, dtype=float)
    if kind == "nealmon":                       # midasr::nealmon: exp(θ1 j + θ2 j²) / Σ
        z = theta[0] * j + theta[1] * j ** 2
        z = np.exp(z - z.max())
        return z / z.sum()
    if kind == "beta":                          # midasr::nbeta with x = (j − 1) / (K − 1) moved into (ε, 1 − ε)
        eps = 1e-6
        x = (j - 1) / (K - 1)
        x = np.clip(x, eps, 1 - eps)
        a, b = np.exp(theta[0]), np.exp(theta[1])
        w = x ** (a - 1) * (1 - x) ** (b - 1)
        return w / w.sum()
    if kind == "almon":
        return theta[0] + theta[1] * j + theta[2] * j ** 2
    raise ValueError(kind)


def _midas_design(M: pd.Series, yq: pd.Series, K: int):
    """Row for quarter q: x at the last month of q and the K − 1 previous months (lag 0 … K − 1)."""
    rows = {}
    for q in yq.index:
        last = q.asfreq("M", "end")
        idx = pd.period_range(last - (K - 1), last, freq="M")[::-1]
        if idx[-1] < M.index.min() or last > M.index.max():
            continue
        rows[q] = M.reindex(idx).to_numpy()
    return pd.DataFrame(rows).T.rename(columns=lambda c: f"x(-{c})")


def midas(df: pd.DataFrame, date: str, target: str, indicator: str, K: int = 6, kind: str = "nealmon", y_lag: bool = True) -> dict:
    """MIDAS regression y_q = α + ρ y_{q−1} + β Σ_j w_j(θ) x_{3q−j+1} + e by non-linear least squares (U-MIDAS by OLS).
    Nowcasts fill missing months with AR forecasts. A Wald-type F test compares the restricted and U-MIDAS fits
    (the hAh test idea of Kvedaras & Zemlys 2012)."""
    import statsmodels.api as sm
    M = monthly_frame(df, date, [indicator])[indicator]
    yq = target_quarterly(df, date, target)
    end_q = M.index.max().asfreq("Q")
    full_idx = pd.period_range(yq.index.min(), end_q, freq="Q")
    yq = yq.reindex(full_idx)
    Mf, filled = fill_ragged(M, end_q.asfreq("M", "end"))
    Xall = _midas_design(Mf, yq, K)
    data = Xall.join(yq.rename("y"))
    if y_lag:
        data["y_lag"] = data["y"].shift(1)
    est = data.dropna()
    y = est["y"].to_numpy()
    X = est[[c for c in Xall.columns]].to_numpy()
    Z = np.column_stack([np.ones(len(y))] + ([est["y_lag"].to_numpy()] if y_lag else []))
    um = sm.OLS(y, np.column_stack([Z, X])).fit()
    rss_u, k_u = float(um.ssr), Z.shape[1] + K
    if kind == "umidas":
        coef = um.params
        names = ["const"] + (["y_lag"] if y_lag else []) + list(Xall.columns)
        tab = pd.DataFrame({"Parameter": names, "Coef.": coef, "Std. Err.": um.bse, "t": um.tvalues, "p-value": um.pvalues})
        w = um.params[Z.shape[1]:]
        pred = lambda zrow, xrow: float(np.r_[zrow, xrow] @ coef)  # noqa: E731
        rss, k = rss_u, k_u
        restr = None
        J = None
    else:
        nth = 3 if kind == "almon" else 2

        def resid(p):
            a = p[:Z.shape[1]]
            if kind == "almon":
                return y - Z @ a - X @ midas_weights(p[Z.shape[1]:], K, kind)
            return y - Z @ a - p[Z.shape[1]] * (X @ midas_weights(p[Z.shape[1] + 1:], K, kind))
        a0 = np.linalg.lstsq(Z, y, rcond=None)[0]
        best = None
        if kind == "almon":
            starts = [np.r_[a0, 0.1, 0.0, 0.0]]
        else:
            b0 = float(np.sum(um.params[Z.shape[1]:]))
            starts = [np.r_[um.params[:Z.shape[1]], b0, t1, t2] for t1, t2 in ((0.0, 0.0), (0.5, -0.1), (-0.5, 0.0), (1.0, -0.2), (0.1, 0.1))]
        for st in starts:
            try:
                r = optimize.least_squares(resid, st, method="lm", max_nfev=20000)
                if best is None or r.cost < best.cost:
                    best = r
            except Exception:  # noqa: BLE001
                continue
        p = best.x
        J = best.jac
        rss = float(np.sum(best.fun ** 2))
        k = len(p)
        s2 = rss / (len(y) - k)
        V = s2 * np.linalg.pinv(J.T @ J)
        names = ["const"] + (["y_lag"] if y_lag else []) + (["β (slope)"] if kind != "almon" else []) + [f"θ{i + 1}" for i in range(nth)]
        se = np.sqrt(np.diag(V))
        tab = pd.DataFrame({"Parameter": names, "Coef.": p, "Std. Err.": se, "t": p / se, "p-value": 2 * stats.t.sf(np.abs(p / se), len(y) - k)})
        if kind == "almon":
            w = midas_weights(p[Z.shape[1]:], K, kind)
        else:
            w = p[Z.shape[1]] * midas_weights(p[Z.shape[1] + 1:], K, kind)
        coef_z = p[:Z.shape[1]]
        pred = lambda zrow, xrow: float(zrow @ coef_z + xrow @ w)  # noqa: E731
        q_ = k_u - k
        F = ((rss - rss_u) / q_) / (rss_u / (len(y) - k_u)) if q_ > 0 else np.nan
        restr = {"F": float(F), "df": (q_, len(y) - k_u), "p": float(stats.f.sf(F, q_, len(y) - k_u)) if q_ > 0 else np.nan}
    # nowcasts for quarters without the target
    last_obs = yq.last_valid_index()
    rows = []
    prev = yq.copy()
    for q in pd.period_range(last_obs + 1, end_q, freq="Q"):
        if q not in Xall.index:
            continue
        zrow = np.r_[1.0, [prev.get(q - 1, np.nan)]] if y_lag else np.array([1.0])
        v = pred(zrow, Xall.loc[q].to_numpy())
        prev[q] = v
        rows.append({"Quarter": str(q), "Nowcast": v})
    sig = np.sqrt(rss / (len(y) - k))
    nw = pd.DataFrame(rows)
    if len(nw):
        nw["lower 95%"] = nw["Nowcast"] - 1.96 * sig
        nw["upper 95%"] = nw["Nowcast"] + 1.96 * sig
    lags = pd.DataFrame({"Monthly lag": np.arange(K), "Weight × β": np.asarray(w, float), "U-MIDAS": um.params[Z.shape[1]:]})
    del J
    return {"table": tab, "lag_weights": lags, "nowcasts": nw, "RMSE": float(np.sqrt(rss / len(y))), "sigma": float(sig),
            "R²": float(1 - rss / np.sum((y - y.mean()) ** 2)), "restriction_test": restr, "N": len(y), "K": K,
            "filled": [str(f) for f in filled], "AIC": float(len(y) * np.log(rss / len(y)) + 2 * k)}


# ==========================================================
# Dynamic factor model (mixed frequency; Bańbura & Modugno 2014 via statsmodels DynamicFactorMQ)
# ==========================================================
def dfm_nowcast(df: pd.DataFrame, date: str, target: str, indicators: list, factors: int = 1, factor_order: int = 2,
                maxiter: int = 300) -> dict:
    from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
    M = monthly_frame(df, date, indicators)
    yq = target_quarterly(df, date, target).dropna()
    Q = yq.to_frame(target)
    mod = DynamicFactorMQ(M, endog_quarterly=Q, factors=factors, factor_orders=factor_order, idiosyncratic_ar1=True, standardize=True)
    r = mod.fit(disp=False, maxiter=maxiter)
    end_q = M.index.max().asfreq("Q")
    pred = r.predict(start=str(M.index.min()), end=str(end_q.asfreq("M", "end")))
    path = pred[target]
    last = yq.index.max()
    rows = []
    for q in pd.period_range(last + 1, end_q, freq="Q"):
        mth = q.asfreq("M", "end")
        rows.append({"Quarter": str(q), "Nowcast": float(path.loc[mth])})
    fq = path[[p for p in path.index if p.month % 3 == 0]]
    fq.index = fq.index.asfreq("Q")
    fit = pd.DataFrame({"actual": yq, "fitted": fq.reindex(yq.index)})
    f = r.factors.smoothed
    load = pd.DataFrame({"Series": list(M.columns) + [target]})
    try:
        R2 = r.get_coefficients_of_determination(method="individual")
        load["R² on factor(s)"] = R2.iloc[:, 0].reindex(load["Series"]).to_numpy()
    except Exception:  # noqa: BLE001
        pass
    # sign normalisation: factor positively correlated with the first indicator
    sgn = np.sign(np.corrcoef(f.iloc[:, 0].to_numpy()[M.iloc[:, 0].notna().to_numpy()], M.iloc[:, 0].dropna().to_numpy())[0, 1]) or 1.0
    return {"nowcasts": pd.DataFrame(rows), "factor": f.iloc[:, 0] * sgn, "loadings": load, "fit": fit, "llf": float(r.llf),
            "RMSE in-sample": float(np.sqrt(np.nanmean((fit.actual - fit.fitted) ** 2))), "monthly_path": path, "results": r}


# ==========================================================
# Bayesian VAR — Minnesota / normal-inverse-Wishart (Giannone, Lenza & Primiceri 2015)
# ==========================================================
def _lagmat(Y: np.ndarray, p: int):
    T, n = Y.shape
    X = np.column_stack([np.ones(T - p)] + [Y[p - l:T - l] for l in range(1, p + 1)])
    return Y[p:], X


def bvar_posterior(Y: np.ndarray, p: int, lam: float, alpha: float = 2.0, psi=None, b_own: float | np.ndarray = 1.0,
                   const_var: float = 1e7) -> dict:
    """Conjugate NIW prior: Σ ~ IW(Ψ, d = n + 2), vec(B)|Σ ~ N(b, Σ ⊗ Ω) with Ω for lag l of variable j equal to
    λ² / (l^α ψ_j)·(d − n − 1) (GLP 2015) and a diffuse constant. ψ_j = residual variance of an AR(p) for variable j.
    Returns posterior moments and the closed-form log marginal likelihood."""
    Y = np.asarray(Y, float)
    T0, n = Y.shape
    if psi is None:
        psi = np.array([_ar_resvar(Y[:, j], p) for j in range(n)])
    d = n + 2
    y, X = _lagmat(Y, p)
    T, k = X.shape
    om = [const_var] + [lam ** 2 / (l ** alpha * psi[j]) * (d - n - 1) for l in range(1, p + 1) for j in range(n)]
    Om = np.diag(om)
    b = np.zeros((k, n))
    bo = np.full(n, b_own) if np.isscalar(b_own) else np.asarray(b_own, float)
    for j in range(n):
        b[1 + j, j] = bo[j]
    Oi = np.diag(1 / np.asarray(om))
    A = X.T @ X + Oi
    Bh = np.linalg.solve(A, X.T @ y + Oi @ b)
    E = y - X @ Bh
    Psi = np.diag(psi)
    S = Psi + E.T @ E + (Bh - b).T @ Oi @ (Bh - b)
    _, ldA = np.linalg.slogdet(A)
    _, ldS = np.linalg.slogdet(S)
    logml = (-n * T / 2 * np.log(np.pi) + multigammaln((T + d) / 2, n) - multigammaln(d / 2, n)
             - n / 2 * np.sum(np.log(om)) + d / 2 * np.sum(np.log(psi)) - n / 2 * ldA - (T + d) / 2 * ldS)
    return {"B": Bh, "S": S, "A": A, "dof": T + d, "logml": float(logml), "psi": psi, "lambda": lam, "X": X, "y": y, "p": p, "n": n}


def _ar_resvar(x, p):
    y, X = _lagmat(np.asarray(x, float).reshape(-1, 1), p)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    return float((e.T @ e).item() / (len(e) - X.shape[1]))


def bvar(Y: pd.DataFrame, p: int = 2, lam: float | None = None, alpha: float = 2.0, b_own: float = 1.0, h: int = 8,
         draws: int = 2000, seed: int = 1, levels=(68, 90)) -> dict:
    """Bayesian VAR with the Minnesota (NIW) prior; λ chosen by maximising the marginal likelihood when not given
    (GLP 2015 empirical Bayes). Forecasts by simulation from the exact posterior (Σ ~ IW, B|Σ ~ MN)."""
    from scipy.stats import invwishart
    names = list(Y.columns)
    Ya = Y.to_numpy(float)
    grid = None
    if lam is None:
        lams = np.exp(np.linspace(np.log(0.01), np.log(5), 60))
        lml = np.array([bvar_posterior(Ya, p, l_, alpha, b_own=b_own)["logml"] for l_ in lams])
        i = int(np.argmax(lml))
        r = optimize.minimize_scalar(lambda ll: -bvar_posterior(Ya, p, np.exp(ll), alpha, b_own=b_own)["logml"],
                                     bounds=(np.log(lams[max(i - 1, 0)]), np.log(lams[min(i + 1, len(lams) - 1)])), method="bounded")
        lam = float(np.exp(r.x))
        grid = pd.DataFrame({"λ": lams, "log ML": lml})
    post = bvar_posterior(Ya, p, lam, alpha, b_own=b_own)
    rng = np.random.default_rng(seed)
    n, k = post["n"], post["B"].shape[0]
    Ai = np.linalg.inv(post["A"])
    La = np.linalg.cholesky(Ai)
    paths = np.empty((draws, h, n))
    for i in range(draws):
        Sig = invwishart.rvs(df=post["dof"], scale=post["S"], random_state=rng)
        Ls = np.linalg.cholesky(Sig)
        B = post["B"] + La @ rng.standard_normal((k, n)) @ Ls.T
        hist = list(Ya[-p:])
        for t in range(h):
            x = np.r_[1.0, np.concatenate([hist[-l] for l in range(1, p + 1)])]
            v = x @ B + Ls @ rng.standard_normal(n)
            hist.append(v)
            paths[i, t] = v
    out = {}
    for j, nm in enumerate(names):
        P = paths[:, :, j]
        fr = pd.DataFrame({"step": np.arange(1, h + 1), "median": np.median(P, 0), "mean": P.mean(0)})
        for lv in levels:
            fr[f"lower {lv}%"] = np.quantile(P, 0.5 - lv / 200, 0)
            fr[f"upper {lv}%"] = np.quantile(P, 0.5 + lv / 200, 0)
        out[nm] = fr
    coef = pd.DataFrame(post["B"], columns=names, index=["const"] + [f"{v}.l{l}" for l in range(1, p + 1) for v in names])
    ols = np.linalg.lstsq(post["X"], post["y"], rcond=None)[0]
    return {"lambda": lam, "grid": grid, "logml": post["logml"], "coef": coef, "ols": pd.DataFrame(ols, columns=names, index=coef.index),
            "forecasts": out, "paths": paths, "posterior": post, "names": names}


# ==========================================================
# Local projections (Jordà 2005) — impulse responses and direct multi-step forecasts
# ==========================================================
def local_projection(Y: pd.DataFrame, shock: str, response: str, H: int = 12, p: int = 4, identification: str = "cholesky",
                     cumulative: bool = False, level: float = 0.90) -> pd.DataFrame:
    """y_{t+h} = α_h + B_h,1 y_{t−1} + … + B_h,p y_{t−p} + u (lpirfs::lp_lin): the response at h to the structural shock
    (Cholesky of the reduced-form VAR(p) residual covariance, ordering = column order, shock size one s.d.) is
    B_h,1 · d (h ≥ 1) and d at h = 0. Newey-West SEs with h + 1 lags."""
    import statsmodels.api as sm
    names = list(Y.columns)
    Ya = Y.to_numpy(float)
    T, n = Ya.shape
    yv, X = _lagmat(Ya, p)
    B = np.linalg.lstsq(X, yv, rcond=None)[0]
    U = yv - X @ B
    Sig = U.T @ U / (len(U) - X.shape[1])
    D = np.linalg.cholesky(Sig) if identification == "cholesky" else np.diag(np.sqrt(np.diag(Sig)))
    js, jr = names.index(shock), names.index(response)
    d = D[:, js]
    z = stats.norm.ppf(0.5 + level / 2)
    rows = [{"h": 0, "IRF": d[jr], "lower": d[jr], "upper": d[jr], "se": 0.0}]
    for h in range(1, H + 1):
        tgt, reg = [], []
        for t in range(p, T - h + 1):
            tgt.append(Ya[t:t + h, jr].sum() if cumulative else Ya[t + h - 1, jr])
            reg.append(np.r_[1.0, np.concatenate([Ya[t - l] for l in range(1, p + 1)])])
        m = sm.OLS(np.asarray(tgt), np.asarray(reg)).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        bh = m.params[1:1 + n]
        Vb = m.cov_params()[1:1 + n, 1:1 + n]
        irf = float(bh @ d) + (d[jr] if cumulative else 0.0)
        se = float(np.sqrt(d @ Vb @ d))
        rows.append({"h": h, "IRF": irf, "lower": irf - z * se, "upper": irf + z * se, "se": se})
    return pd.DataFrame(rows)


def lp_direct_forecast(Y: pd.DataFrame, target: str, H: int = 8, p: int = 2, level: float = 0.95) -> pd.DataFrame:
    """Direct multi-step forecasts: one regression of y_{t+h} on (y_t … y_{t−p+1}) for each horizon h (all variables in
    Y); intervals from the h-step regression residuals."""
    import statsmodels.api as sm
    Ya = Y.to_numpy(float)
    T, n = Ya.shape
    j = list(Y.columns).index(target)
    z = stats.norm.ppf(0.5 + level / 2)
    out = []
    for h in range(1, H + 1):
        tgt, reg = [], []
        for t in range(p - 1, T - h):
            tgt.append(Ya[t + h, j])
            reg.append(np.r_[1.0, np.concatenate([Ya[t - l] for l in range(p)])])
        m = sm.OLS(np.asarray(tgt), np.asarray(reg)).fit()
        x = np.r_[1.0, np.concatenate([Ya[T - 1 - l] for l in range(p)])]
        f = float(m.params @ x)
        s = float(np.sqrt(m.scale))
        out.append({"step": h, "forecast": f, f"lower {int(level * 100)}%": f - z * s, f"upper {int(level * 100)}%": f + z * s, "R²": m.rsquared})
    return pd.DataFrame(out)
