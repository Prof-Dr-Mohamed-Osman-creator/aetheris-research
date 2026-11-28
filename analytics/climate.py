# ==========================================
# Aetheris Research - Climate & energy economics (v1.11, Catalogue §1.26)
# Degree days and temperature bins from daily weather (Baskerville-Emin single sine), panel temperature-response models
# (GDD/KDD, bins, quadratic) with warming-scenario impacts, Ricardian land-value models, the Burke-Hsiang-Miguel
# growth-temperature damage function with projections and an illustrative social cost of carbon, STIRPAT (with ridge),
# emission projections and peak, the environmental Kuznets curve with turning-point inference, Tapio decoupling and
# LMDI (Kaya) decomposition.
# ==========================================

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import optimize, stats

from analytics.causal import _fe_cluster, _row, ols

warnings.filterwarnings("ignore")


# ======================================================================================================================
# Degree days
# ======================================================================================================================
def _sine_dd(tmin, tmax, base):
    """Degree days above `base` for one day under the single-sine model (Baskerville & Emin 1969)."""
    tmin, tmax = np.asarray(tmin, float), np.asarray(tmax, float)
    M = (tmax + tmin) / 2
    W = (tmax - tmin) / 2
    out = np.where(tmin >= base, M - base, 0.0)
    part = (tmin < base) & (tmax > base)
    with np.errstate(invalid="ignore", divide="ignore"):
        th = np.arcsin(np.clip((base - M) / np.where(W > 0, W, 1), -1, 1))
        val = ((M - base) * (np.pi / 2 - th) + W * np.cos(th)) / np.pi
    return np.where(part, val, out)


def degree_days(df: pd.DataFrame, tmin: str, tmax: str, group: list | None = None, base: float = 8.0, upper: float = 29.0,
                method: str = "sine", rain: str | None = None, bins: tuple = (10, 15, 20, 25, 30, 35)) -> pd.DataFrame:
    """Season aggregates per group: GDD between `base` and `upper` (Schlenker & Roberts 2009 convention: GDD(base) −
    GDD(upper)), KDD above `upper`, by the single-sine method or the daily average; plus days with Tmax in bins, frost
    days, hot days (Tmax ≥ 35), total rain, rainy days and the longest dry spell."""
    d = df.copy()
    lo, hi = d[tmin].to_numpy(float), d[tmax].to_numpy(float)
    if method == "sine":
        dd_b, dd_u = _sine_dd(lo, hi, base), _sine_dd(lo, hi, upper)
    else:
        m = (lo + hi) / 2
        dd_b, dd_u = np.clip(m - base, 0, None), np.clip(m - upper, 0, None)
    d["_gdd"] = dd_b - dd_u
    d["_kdd"] = dd_u
    d["_tmean"] = (lo + hi) / 2
    d["_frost"] = (lo < 0).astype(int)
    d["_hot"] = (hi >= 35).astype(int)
    edges = [-np.inf, *bins, np.inf]
    labs = [f"< {bins[0]}"] + [f"{a}–{b}" for a, b in zip(bins[:-1], bins[1:])] + [f"≥ {bins[-1]}"]
    bi = np.digitize(hi, edges[1:-1])
    for j, lab in enumerate(labs):
        d[f"days Tmax {lab}"] = (bi == j).astype(int)
    agg = {"_gdd": "sum", "_kdd": "sum", "_tmean": "mean", tmin: "mean", tmax: "mean", "_frost": "sum", "_hot": "sum"}
    agg.update({f"days Tmax {lab}": "sum" for lab in labs})
    if rain:
        d["_rainy"] = (d[rain] >= 1).astype(int)
        agg.update({rain: "sum", "_rainy": "sum"})
    g = group or []
    out = d.groupby(g).agg(agg).reset_index() if g else d.agg(agg).to_frame().T
    out["days"] = d.groupby(g).size().to_numpy() if g else len(d)
    if rain:
        def dry(s):
            run = best = 0
            for v in s:
                run = run + 1 if v < 1 else 0
                best = max(best, run)
            return best
        out["longest dry spell (days)"] = d.groupby(g)[rain].apply(dry).to_numpy() if g else dry(d[rain])
    ren = {"_gdd": f"GDD {base:g}–{upper:g}°C", "_kdd": f"KDD > {upper:g}°C", "_tmean": "mean Tavg", tmin: "mean Tmin", tmax: "mean Tmax",
           "_frost": "frost days", "_hot": "hot days (Tmax ≥ 35°C)", "_rainy": "rainy days (≥ 1 mm)"}
    if rain:
        ren[rain] = "total rain (mm)"
    return out.rename(columns=ren)


def warming_degree_days(df, tmin, tmax, group, base=8.0, upper=29.0, deltas=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """GDD / KDD per group under uniform warming of the daily record by Δ °C (shifting Tmin and Tmax)."""
    rows = []
    for dl in deltas:
        d = df.copy()
        d[tmin] = d[tmin] + dl
        d[tmax] = d[tmax] + dl
        a = degree_days(d, tmin, tmax, group, base, upper)
        a.insert(len(group), "warming (°C)", dl)
        rows.append(a[group + ["warming (°C)", f"GDD {base:g}–{upper:g}°C", f"KDD > {upper:g}°C"]])
    return pd.concat(rows, ignore_index=True)


# ======================================================================================================================
# Panel temperature-response models
# ======================================================================================================================
SPECS = {"gddkdd": ("Piecewise linear: GDD + KDD (Schlenker-Roberts)", "خطي مجزأ: درجات النمو + درجات الضرر الحراري"),
         "bins": ("Temperature bins (days in Tmax bins)", "فئات الحرارة (عدد الأيام في كل فئة)"),
         "quad": ("Quadratic in season mean temperature", "تربيعي في متوسط حرارة الموسم")}


def temp_response(df: pd.DataFrame, unit: str, time: str, y: str, spec: str = "gddkdd", gdd: str | None = None, kdd: str | None = None,
                  bins: list | None = None, ref_bin: str | None = None, tmean: str | None = None, precip: str | None = None,
                  trend: bool = False, controls: list | None = None) -> dict:
    """Panel fixed-effects yield (or other outcome) response to weather: unit and year fixed effects (or unit FE plus a
    common quadratic trend when `trend`), cluster-robust SEs by unit. Precipitation enters as a quadratic."""
    controls = controls or []
    X, names = [], []
    if spec == "gddkdd":
        X += [df[gdd], df[kdd]]
        names += [gdd, kdd]
    elif spec == "bins":
        bins = [b for b in bins if b != ref_bin]
        X += [df[b] for b in bins]
        names += bins
    else:
        X += [df[tmean], df[tmean] ** 2]
        names += [tmean, f"{tmean}²"]
    if precip:
        X += [df[precip], df[precip] ** 2]
        names += [precip, f"{precip}²"]
    X += [df[c] for c in controls]
    names += controls
    d = pd.concat([df[[unit, time, y]].reset_index(drop=True), pd.DataFrame(np.column_stack(X), columns=names)], axis=1).dropna()
    Xm = d[names].to_numpy(float)
    ui, ti = pd.factorize(d[unit])[0], pd.factorize(d[time])[0]
    if trend:
        tt = d[time].to_numpy(float) - d[time].min()
        Xm2 = np.column_stack([Xm, tt, tt ** 2])
        Z = Xm2 - pd.DataFrame(Xm2).groupby(ui).transform("mean").to_numpy()
        yd = d[y].to_numpy(float) - d.groupby(ui)[y].transform("mean").to_numpy()
        b, V, _, dof = ols(yd, Z, cluster=d[unit].to_numpy())
        b, V = b[:len(names)], V[:len(names), :len(names)]
    else:
        b, V, keep, dof = _fe_cluster(d[y].to_numpy(float), Xm, ui, ti, d[unit].to_numpy(), strict=True, names=names)
    tab = pd.DataFrame([_row(nm, b[i], np.sqrt(V[i, i]), dof) for i, nm in enumerate(names)])
    out = {"table": tab, "b": dict(zip(names, b)), "V": pd.DataFrame(V, index=names, columns=names), "N": len(d), "units": d[unit].nunique(),
           "spec": spec, "dof": dof}
    if spec == "quad":
        b1, b2 = b[0], b[1]
        out["optimum"] = -b1 / (2 * b2) if b2 < 0 else np.nan
    if spec == "gddkdd":
        # empirical sensitivity of the degree days to the season mean (within-unit slope)
        if tmean and tmean in df:
            dd = df[[unit, gdd, kdd, tmean]].dropna()
            tm = dd[tmean] - dd.groupby(unit)[tmean].transform("mean")
            out["dGDD/dT"] = float(np.sum(tm * (dd[gdd] - dd.groupby(unit)[gdd].transform("mean"))) / np.sum(tm ** 2))
            out["dKDD/dT"] = float(np.sum(tm * (dd[kdd] - dd.groupby(unit)[kdd].transform("mean"))) / np.sum(tm ** 2))
    return out


def warming_impact(res: dict, deltas=(1, 2, 3, 4), base_tmean: float | None = None, dgdd: float | None = None, dkdd: float | None = None,
                   bin_shift: dict | None = None) -> pd.DataFrame:
    """Predicted change in the (log) outcome for uniform warming Δ: GDD/KDD spec via the degree-day sensitivities
    (from daily data or the empirical within-unit slopes); quadratic spec at the sample mean temperature. Delta-method
    95 % intervals; % change = 100·(exp(Δ) − 1) for a log outcome."""
    names = list(res["b"])
    b = np.array([res["b"][n] for n in names])
    V = res["V"].to_numpy()
    rows = []
    for dl in deltas:
        g = np.zeros(len(b))
        if res["spec"] == "gddkdd":
            g[0] = (dgdd if dgdd is not None else res.get("dGDD/dT", 0)) * dl
            g[1] = (dkdd if dkdd is not None else res.get("dKDD/dT", 0)) * dl
        elif res["spec"] == "quad":
            T = base_tmean
            g[0] = dl
            g[1] = (T + dl) ** 2 - T ** 2
        else:
            for j, n in enumerate(names):
                if bin_shift and n in bin_shift:
                    g[j] = bin_shift[n] * dl
        est = g @ b
        se = float(np.sqrt(g @ V @ g))
        rows.append({"Warming (°C)": dl, "Δ outcome": est, "Std. Err.": se, "CI low": est - 1.96 * se, "CI high": est + 1.96 * se,
                     "% change": 100 * (np.exp(est) - 1), "% CI low": 100 * (np.exp(est - 1.96 * se) - 1), "% CI high": 100 * (np.exp(est + 1.96 * se) - 1)})
    return pd.DataFrame(rows)


def response_curve(res: dict, grid=None, ref=None) -> pd.DataFrame:
    """Marginal response for plotting: bins (coefficients per bin vs. the reference) or the quadratic curve relative to
    the optimum / reference temperature, with 95 % bands."""
    names = list(res["b"])
    b = np.array([res["b"][n] for n in names])
    V = res["V"].to_numpy()
    if res["spec"] == "bins":
        nb = [n for n in names if n.startswith("days")]
        return pd.DataFrame({"Bin": nb, "Effect of one day": [res["b"][n] for n in nb], "Std. Err.": [np.sqrt(res["V"].loc[n, n]) for n in nb]})
    if res["spec"] == "quad":
        grid = np.linspace(10, 35, 60) if grid is None else grid
        r0 = ref if ref is not None else (res.get("optimum") if np.isfinite(res.get("optimum", np.nan)) else np.mean(grid))
        G = np.zeros((len(grid), len(b)))
        G[:, 0] = grid - r0
        G[:, 1] = grid ** 2 - r0 ** 2
        est = G @ b
        se = np.sqrt(np.einsum("ij,jk,ik->i", G, V, G))
        return pd.DataFrame({"Temperature": grid, "Effect": est, "CI low": est - 1.96 * se, "CI high": est + 1.96 * se})
    return pd.DataFrame()


# ======================================================================================================================
# Ricardian model
# ======================================================================================================================
def ricardian(df: pd.DataFrame, y: str, climate: list, controls: list | None = None, log: bool = True, deltas: dict | None = None) -> dict:
    """Mendelsohn, Nordhaus & Shaw (1994): land value on climate normals and their squares plus controls (HC1 SEs).
    Marginal effects of each climate variable at the sample mean (delta-method SEs), and the non-marginal welfare
    effect of a climate scenario (average predicted change, in % for a log model and in levels otherwise)."""
    controls = controls or []
    d = df[[y] + climate + controls].dropna()
    yv = np.log(d[y].to_numpy(float)) if log else d[y].to_numpy(float)
    cols = []
    names = ["const"]
    for c in climate:
        cols += [d[c].to_numpy(float), d[c].to_numpy(float) ** 2]
        names += [c, f"{c}²"]
    cols += [d[c].to_numpy(float) for c in controls]
    names += controls
    X = np.column_stack([np.ones(len(d))] + cols)
    b, V, e, dof = ols(yv, X, "HC1")
    tab = pd.DataFrame([_row(nm, b[i], np.sqrt(V[i, i]), dof) for i, nm in enumerate(names)])
    me = []
    for j, c in enumerate(climate):
        i1, i2 = 1 + 2 * j, 2 + 2 * j
        xm = d[c].mean()
        g = np.zeros(len(b))
        g[i1], g[i2] = 1, 2 * xm
        est, se = g @ b, np.sqrt(g @ V @ g)
        turning = -b[i1] / (2 * b[i2]) if b[i2] != 0 else np.nan
        me.append({"Climate variable": c, "Mean": xm, "Marginal effect at mean": est, "Std. Err.": se, "p-value": 2 * stats.norm.sf(abs(est / se)),
                   "Shape": "hill (∩)" if b[i2] < 0 else "U-shaped", "Turning point": turning})
    out = {"table": tab, "marginal": pd.DataFrame(me), "R²": float(1 - e @ e / np.sum((yv - yv.mean()) ** 2)), "N": len(d), "b": b, "V": V,
           "log": log, "names": names}
    if deltas:
        X1 = X.copy()
        for j, c in enumerate(climate):
            if c in deltas:
                kind, val = deltas[c]
                new = d[c].to_numpy(float) + val if kind == "abs" else d[c].to_numpy(float) * (1 + val / 100)
                X1[:, 1 + 2 * j] = new
                X1[:, 2 + 2 * j] = new ** 2
        G = (X1 - X).mean(0)
        dlt = float(G @ b)
        se = float(np.sqrt(G @ V @ G))
        if log:
            per_farm = np.exp(X1 @ b) - np.exp(X @ b)
            out["scenario"] = {"Δ log land value (mean)": dlt, "Std. Err.": se, "% change": 100 * (np.exp(dlt) - 1),
                               "% CI low": 100 * (np.exp(dlt - 1.96 * se) - 1), "% CI high": 100 * (np.exp(dlt + 1.96 * se) - 1),
                               "Mean change per farm (units of y, naive retransform)": float(per_farm.mean()),
                               "Share of farms losing": float(np.mean(X1 @ b < X @ b))}
        else:
            out["scenario"] = {"Mean change per farm": dlt, "Std. Err.": se, "% of mean": 100 * dlt / d[y].mean(), "Share of farms losing": float(np.mean(X1 @ b < X @ b))}
    return out


# ======================================================================================================================
# Burke-Hsiang-Miguel damage function
# ======================================================================================================================
def bhm(df: pd.DataFrame, unit: str, time: str, growth: str, temp: str, precip: str | None = None, trends: str = "linear") -> dict:
    """Burke, Hsiang & Miguel (2015): growth_it = β1 T + β2 T² (+ P, P²) + μ_i + ν_t + unit-specific trends + ε,
    clustered by unit. Optimum temperature −β1/(2β2) with a delta-method CI and the response curve."""
    cols = [unit, time, growth, temp] + ([precip] if precip else [])
    d = df[cols].dropna().reset_index(drop=True)
    names = [temp, f"{temp}²"] + ([precip, f"{precip}²"] if precip else [])
    X = [d[temp], d[temp] ** 2] + ([d[precip], d[precip] ** 2] if precip else [])
    tt = (d[time] - d[time].min()).to_numpy(float)
    ui, ti = pd.factorize(d[unit])[0], pd.factorize(d[time])[0]
    U = pd.get_dummies(d[unit], dtype=float).to_numpy()
    if trends in ("linear", "quadratic"):
        X += [U[:, j] * tt for j in range(U.shape[1])]
        if trends == "quadratic":
            X += [U[:, j] * tt ** 2 for j in range(U.shape[1])]
    Xm = np.column_stack(X)
    b, V, keep, dof = _fe_cluster(d[growth].to_numpy(float), Xm, ui, ti, d[unit].to_numpy())   # one unit trend is collinear with year FE
    if not keep[:len(names)].all():
        raise ValueError("temperature terms are absorbed by the fixed effects / trends")
    k = len(names)
    b, V = b[:k], V[:k, :k]
    tab = pd.DataFrame([_row(nm, b[i], np.sqrt(V[i, i]), dof) for i, nm in enumerate(names)])
    b1, b2 = b[0], b[1]
    opt = -b1 / (2 * b2)
    g = np.array([-1 / (2 * b2), b1 / (2 * b2 ** 2)])
    se_opt = float(np.sqrt(g @ V[:2, :2] @ g))
    grid = np.linspace(d[temp].min(), d[temp].max(), 80)
    G = np.column_stack([grid - opt, grid ** 2 - opt ** 2])
    est = G @ b[:2]
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, V[:2, :2], G))
    curve = pd.DataFrame({"Temperature": grid, "Growth effect vs optimum": est, "CI low": est - 1.96 * se, "CI high": est + 1.96 * se})
    return {"table": tab, "b1": float(b1), "b2": float(b2), "V": V[:2, :2], "optimum": float(opt), "optimum se": se_opt,
            "optimum CI": (float(opt - 1.96 * se_opt), float(opt + 1.96 * se_opt)), "curve": curve, "N": len(d), "trends": trends}


def bhm_projection(res: dict, base: pd.DataFrame, unit: str, temp: str, gdp: str, growth0: float = 0.02, warming: float = 3.0,
                   years: int = 80, weight: str | None = None) -> dict:
    """Project GDP per capita with and without warming: each unit's temperature rises linearly by `warming` °C over the
    horizon; the climate-induced growth change is β1[(T+Δ_t) − T] + β2[(T+Δ_t)² − T²] (BHM 'pooled' response). Units
    start from their baseline GDP and grow at `growth0` absent warming. Returns unit-level losses and the aggregate
    (weighted by `weight`, e.g. population, else by GDP)."""
    b1, b2 = res["b1"], res["b2"]
    rows, paths = [], []
    for _, r in base.iterrows():
        T0, Y0 = r[temp], r[gdp]
        y_cc, y_nc = Y0, Y0
        for t in range(1, years + 1):
            dT = warming * t / years
            dg = b1 * dT + b2 * ((T0 + dT) ** 2 - T0 ** 2)
            y_nc *= 1 + growth0
            y_cc *= 1 + growth0 + dg
            paths.append({unit: r[unit], "year": t, "no warming": y_nc, "with warming": y_cc})
        rows.append({unit: r[unit], "Baseline T": T0, "GDP/cap end (no warming)": y_nc, "GDP/cap end (warming)": y_cc,
                     "% change": 100 * (y_cc / y_nc - 1), "w": r[weight] if weight else y_nc})
    u = pd.DataFrame(rows)
    agg = float(np.sum(u["w"] * u["% change"]) / u["w"].sum())
    return {"units": u.drop(columns="w"), "aggregate % change": agg, "paths": pd.DataFrame(paths), "share losing": float((u["% change"] < 0).mean())}


def scc_simple(gdp0: float = 105e12, g: float = 0.02, T0: float = 1.2, warming_rate: float = 0.025, damage: float = 0.00236,
               tcre: float = 0.45, discount: float = 0.03, horizon: int = 200, ramsey: tuple | None = None) -> dict:
    """Illustrative social cost of carbon: quadratic damages D(T) = a·T² of world GDP (a = 0.00236 as DICE-2016R), TCRE
    (°C per 1000 GtCO2), a baseline warming path T_t = T0 + rate·t, and constant or Ramsey (ρ, η) discounting. The
    marginal tonne raises temperature permanently by TCRE/1e12 °C. Returns USD per tCO2 and the discounted path."""
    t = np.arange(1, horizon + 1)
    Y = gdp0 * (1 + g) ** t
    T = T0 + warming_rate * t
    dT = tcre / 1e12
    md = 2 * damage * T * dT * Y
    r = discount if ramsey is None else ramsey[0] + ramsey[1] * g
    df_ = (1 + r) ** -t
    return {"SCC (USD/tCO2)": float(np.sum(md * df_)), "discount rate": r,
            "path": pd.DataFrame({"year": t, "GDP": Y, "Temperature": T, "Damages (% GDP)": 100 * damage * T ** 2, "Discounted marginal damage": md * df_})}


def damage_curve(a: float = 0.00236, temps=np.linspace(0, 6, 61), b1: float | None = None, b2: float | None = None, T_base: float | None = None):
    out = pd.DataFrame({"Warming (°C)": temps, "Quadratic damages (% GDP)": 100 * a * temps ** 2})
    if b1 is not None and T_base is not None:
        out["BHM annual growth effect (pp)"] = 100 * (b1 * temps + b2 * ((T_base + temps) ** 2 - T_base ** 2))
    return out


# ======================================================================================================================
# STIRPAT, EKC, projections
# ======================================================================================================================
def stirpat(df: pd.DataFrame, unit: str | None, time: str | None, impact: str, drivers: list, fe: str = "twoway", ridge: bool = False) -> dict:
    """STIRPAT ln I = a + Σ b_j ln X_j (Dietz & Rosa 1997; York, Rosa & Dietz 2003): pooled OLS (HC1), unit or two-way
    fixed effects (clustered by unit), and optionally ridge regression on standardised logs with the GCV-optimal penalty
    (to tame collinearity, reporting VIFs). Coefficients are elasticities."""
    cols = [c for c in [unit, time] if c] + [impact] + drivers
    d = df[cols].dropna()
    d = d[(d[[impact] + drivers] > 0).all(1)]
    Y = np.log(d[impact].to_numpy(float))
    X = np.log(d[drivers].to_numpy(float))
    names = [f"ln {v}" for v in drivers]
    R = np.corrcoef(X, rowvar=False)
    vif = np.diag(np.linalg.inv(R)) if len(drivers) > 1 else np.ones(1)
    if fe == "none" or unit is None:
        b, V, e, dof = ols(Y, np.column_stack([np.ones(len(d)), X]), "HC1")
        b, V = b[1:], V[1:, 1:]
    elif fe == "unit":
        ui = pd.factorize(d[unit])[0]
        Yd = Y - pd.Series(Y).groupby(ui).transform("mean").to_numpy()
        Xd = X - pd.DataFrame(X).groupby(ui).transform("mean").to_numpy()
        b, V, e, dof = ols(Yd, Xd, cluster=d[unit].to_numpy())
    else:
        b, V, keep, dof = _fe_cluster(Y, X, pd.factorize(d[unit])[0], pd.factorize(d[time])[0], d[unit].to_numpy(), strict=True, names=names)
    tab = pd.DataFrame([_row(nm, b[i], np.sqrt(V[i, i]), dof) for i, nm in enumerate(names)])
    tab["VIF"] = vif
    out = {"table": tab, "b": dict(zip(drivers, b)), "N": len(d), "fe": fe, "V": V}
    if ridge:
        Xs = (X - X.mean(0)) / X.std(0)
        ys = Y - Y.mean()
        U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
        n = len(ys)

        def gcv(lmb):
            dgn = s ** 2 / (s ** 2 + lmb)
            fit = U @ (dgn * (U.T @ ys))
            return n * np.sum((ys - fit) ** 2) / (n - dgn.sum()) ** 2
        lam = optimize.minimize_scalar(lambda L: gcv(np.exp(L)), bounds=(-10, np.log(10 * n)), method="bounded").x
        lam = float(np.exp(lam))
        br = Vt.T @ (s / (s ** 2 + lam) * (U.T @ ys))
        out["ridge"] = pd.DataFrame({"Variable": names, "Ridge elasticity": br / X.std(0), "λ (GCV)": lam})
    return out


def ekc(df: pd.DataFrame, unit: str, time: str, emissions: str, income: str, cubic: bool = False, per_capita: str | None = None,
        controls: list | None = None) -> dict:
    """Environmental Kuznets curve: ln E = β1 ln y + β2 (ln y)² (+ β3 (ln y)³) + controls + μ_i + ν_t, clustered by unit.
    Turning point y* = exp(−β1 / 2β2) (quadratic) with delta-method and Fieller 95 % intervals and the share of
    observations beyond it."""
    controls = controls or []
    d = df[[unit, time, emissions, income] + controls + ([per_capita] if per_capita else [])].dropna()
    E = d[emissions].to_numpy(float) / (d[per_capita].to_numpy(float) if per_capita else 1)
    d = d[E > 0]
    E = E[E > 0]
    ly = np.log(d[income].to_numpy(float))
    m = ly.mean()
    names = ["ln y", "(ln y)²"] + (["(ln y)³"] if cubic else []) + controls
    X = [ly, ly ** 2] + ([ly ** 3] if cubic else []) + [d[c].to_numpy(float) for c in controls]
    b, V, keep, dof = _fe_cluster(np.log(E), np.column_stack(X), pd.factorize(d[unit])[0], pd.factorize(d[time])[0], d[unit].to_numpy(), strict=True, names=names)
    tab = pd.DataFrame([_row(nm, b[i], np.sqrt(V[i, i]), dof) for i, nm in enumerate(names)])
    out = {"table": tab, "N": len(d), "b": b, "V": V, "names": names, "mean ln y": m}
    if not cubic:
        b1, b2 = b[0], b[1]
        tp = -b1 / (2 * b2)
        g = np.array([-1 / (2 * b2), b1 / (2 * b2 ** 2)])
        se = float(np.sqrt(g @ V[:2, :2] @ g))
        # Fieller: solve (b1 + 2 b2 x)² = z² var(b1 + 2 b2 x)
        z2 = 1.959963984540054 ** 2
        v11, v12, v22 = V[0, 0], V[0, 1], V[1, 1]
        A = 4 * (b2 ** 2 - z2 * v22)
        B = 4 * (b1 * b2 - z2 * v12)
        Cc = b1 ** 2 - z2 * v11
        disc = B ** 2 - 4 * A * Cc
        fiel = tuple(sorted(((-B - np.sqrt(disc)) / (2 * A), (-B + np.sqrt(disc)) / (2 * A)))) if disc > 0 and A > 0 else (np.nan, np.nan)
        out.update({"shape": "inverted U (EKC)" if b2 < 0 and b1 > 0 else "U-shaped" if b2 > 0 else "monotone",
                    "turning point ln y": float(tp), "turning point": float(np.exp(tp)), "turning point se (ln)": se,
                    "turning point CI (delta)": (float(np.exp(tp - 1.96 * se)), float(np.exp(tp + 1.96 * se))),
                    "turning point CI (Fieller)": (float(np.exp(fiel[0])), float(np.exp(fiel[1]))) if np.isfinite(fiel[0]) else (np.nan, np.nan),
                    "share beyond turning point": float(np.mean(ly > tp)),
                    "Lind-Mehlum slopes": {"slope at min": float(b1 + 2 * b2 * ly.min()), "slope at max": float(b1 + 2 * b2 * ly.max())}})
        grid = np.linspace(ly.min(), ly.max(), 80)
        G = np.column_stack([grid - m, grid ** 2 - m ** 2])
        est = G @ b[:2]
        s_ = np.sqrt(np.einsum("ij,jk,ik->i", G, V[:2, :2], G))
        out["curve"] = pd.DataFrame({"income": np.exp(grid), "ln E relative to mean income": est, "CI low": est - 1.96 * s_, "CI high": est + 1.96 * s_})
    return out


def emission_projection(elasticities: dict, base: dict, growth: dict, years: int = 30, start_year: int = 2024) -> pd.DataFrame:
    """Project an impact path from STIRPAT elasticities: ln I_t = ln I_0 + Σ b_j ln(X_jt / X_j0) with drivers growing at
    scenario rates (% per year); flags the peak year."""
    t = np.arange(0, years + 1)
    lnI = np.log(base["impact"]) + sum(elasticities[k] * t * np.log(1 + growth.get(k, 0) / 100) for k in elasticities)
    out = pd.DataFrame({"Year": start_year + t, "Impact": np.exp(lnI)})
    out["Peak"] = out.Impact == out.Impact.max()
    return out


# ======================================================================================================================
# Decoupling and decomposition
# ======================================================================================================================
TAPIO = [("Strong decoupling", "فك ارتباط قوي"), ("Weak decoupling", "فك ارتباط ضعيف"), ("Expansive coupling", "ارتباط توسعي"),
         ("Expansive negative decoupling", "فك ارتباط سلبي توسعي"), ("Strong negative decoupling", "فك ارتباط سلبي قوي"),
         ("Weak negative decoupling", "فك ارتباط سلبي ضعيف"), ("Recessive coupling", "ارتباط انكماشي"), ("Recessive decoupling", "فك ارتباط انكماشي")]


def tapio_state(dE, dG):
    """Tapio (2005) classification of the elasticity e = %ΔE / %ΔG with thresholds 0.8 and 1.2."""
    e = dE / dG if dG != 0 else np.nan
    if dG > 0:
        if dE < 0:
            return 0, e
        if e < 0.8:
            return 1, e
        if e <= 1.2:
            return 2, e
        return 3, e
    if dE > 0:
        return 4, e
    if e < 0.8:
        return 5, e
    if e <= 1.2:
        return 6, e
    return 7, e


def tapio(df: pd.DataFrame, unit: str | None, time: str, emissions: str, gdp: str, periods: list) -> pd.DataFrame:
    rows = []
    groups = df.groupby(unit) if unit else [("All", df.groupby(time)[[emissions, gdp]].sum().reset_index())]
    for u, s in groups:
        s = s.set_index(time)
        for a, b in periods:
            if a not in s.index or b not in s.index:
                continue
            dE = s.loc[b, emissions] / s.loc[a, emissions] - 1
            dG = s.loc[b, gdp] / s.loc[a, gdp] - 1
            k, e = tapio_state(dE, dG)
            rows.append({"Unit": u, "Period": f"{a}–{b}", "%Δ emissions": 100 * dE, "%Δ GDP": 100 * dG, "Elasticity": e, "State": TAPIO[k][0], "State_ar": TAPIO[k][1],
                         "code": k})
    return pd.DataFrame(rows)


def _L(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isclose(a, b), a, (a - b) / (np.log(a) - np.log(b)))


def lmdi_kaya(df: pd.DataFrame, unit: str | None, time: str, co2: str, pop: str, gdp: str, energy: str, t0, t1) -> dict:
    """Additive and multiplicative LMDI-I (Ang 2004, 2005) of the Kaya identity C = P · (G/P) · (E/G) · (C/E):
    population, affluence, energy intensity and carbon intensity effects; perfect decomposition (no residual), summed
    over units for the aggregate."""
    d = df[[c for c in [unit] if c] + [time, co2, pop, gdp, energy]].dropna()
    if unit is None:
        d = d.groupby(time)[[co2, pop, gdp, energy]].sum().reset_index()
        d["_u"] = "All"
        unit = "_u"
    a, b = d[d[time] == t0].set_index(unit), d[d[time] == t1].set_index(unit)
    idx = a.index.intersection(b.index)
    a, b = a.loc[idx], b.loc[idx]

    def fac(x):
        return {"Population": x[pop], "Affluence (GDP/capita)": x[gdp] / x[pop], "Energy intensity (E/GDP)": x[energy] / x[gdp], "Carbon intensity (CO2/E)": x[co2] / x[energy]}
    fa, fb = fac(a), fac(b)
    w = _L(b[co2], a[co2])
    eff = pd.DataFrame({k: w * np.log(fb[k] / fa[k]) for k in fa}, index=idx)
    eff["Total ΔCO2"] = b[co2] - a[co2]
    eff["Residual"] = eff["Total ΔCO2"] - eff[list(fa)].sum(1)
    tot = eff.sum()
    wm = _L(b[co2].sum(), a[co2].sum())
    mult = {k: float(np.exp(tot[k] / wm)) for k in fa}
    mult["Total (C_T / C_0)"] = float(b[co2].sum() / a[co2].sum())
    return {"by unit": eff.reset_index(), "aggregate": tot.to_frame("Effect").reset_index().rename(columns={"index": "Factor"}), "multiplicative": mult,
            "period": (t0, t1)}
