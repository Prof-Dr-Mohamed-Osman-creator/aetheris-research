# ==========================================
# Aetheris Research - Regression charts
# ------------------------------------------
# Residual diagnostics, influence, CUSUM / CUSUMSQ, coefficient (forest)
# plots, fitted curves with confidence & prediction bands, conditional
# effect curves, Box-Cox profile, Johnson-Neyman plot. Plotly only.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from analytics import regression as rg
from analytics.charts import PALETTES, style_academic

PAL = PALETTES["academic"]

AXIS_AR = {"Fitted": "القيم المقدَّرة", "Theoretical quantiles": "الكميات النظرية", "Leverage": "الرفع",
           "Predicted": "القيم المتنبأ بها", "Observed": "القيم المشاهدة", "Residual": "البواقي", "Density": "الكثافة",
           "Lag": "الإبطاء", "ACF of residuals": "دالة الارتباط الذاتي للبواقي", "Observation": "المشاهدة",
           "Coefficient (95% CI)": "المعامل (فترة الثقة 95%)", "Profile log-likelihood": "لوغاريتم دالة الإمكان الجانبية",
           "95% prediction band": "نطاق التنبؤ 95%", "95% confidence band": "نطاق الثقة 95%", "Fitted line": "الخط المقدَّر",
           "Predicted line": "القيم المتنبأ بها", "NLS fit": "منحنى NLS", "Conditional effect of": "الأثر الشرطي لـ"}


def _a(text: str, rtl: bool) -> str:
    return AXIS_AR.get(text, text) if rtl else text


def _lowess(x, y, frac=0.6):
    try:
        from statsmodels.nonparametric.smoothers_lowess import lowess
        z = lowess(y, x, frac=frac, return_sorted=True)
        return z[:, 0], z[:, 1]
    except Exception:  # noqa: BLE001
        o = np.argsort(x)
        return x[o], y[o]


def residual_panel(r: rg.RegResult, rtl=False, labels=None) -> go.Figure:
    """The four classic residual plots (as R's plot.lm): residuals vs fitted, normal Q-Q,
    scale-location, residuals vs leverage with Cook's distance contours."""
    L = labels or ["Residuals vs fitted", "Normal Q-Q", "Scale-location", "Residuals vs leverage"]
    e = np.asarray(r.wresid, float)
    X = np.asarray(r.wexog, float)
    n, k = X.shape
    fitted = X @ r.params.to_numpy()
    h = np.einsum("ij,jk,ik->i", X, np.linalg.pinv(X.T @ X), X)
    s = np.sqrt(e @ e / (n - k))
    rs = e / (s * np.sqrt(np.clip(1 - h, 1e-12, None)))
    fig = make_subplots(rows=2, cols=2, subplot_titles=L, horizontal_spacing=0.1, vertical_spacing=0.14)
    mk = dict(color=PAL[0], size=5, opacity=0.55)
    fig.add_trace(go.Scatter(x=fitted, y=e, mode="markers", marker=mk, name="e"), 1, 1)
    lx, ly = _lowess(fitted, e)
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=PAL[1], width=2)), 1, 1)
    fig.add_hline(y=0, line=dict(color="#777", dash="dot"), row=1, col=1)
    osm, osr = stats.probplot(rs, dist="norm", fit=False)
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", marker=mk), 1, 2)
    fig.add_trace(go.Scatter(x=[osm.min(), osm.max()], y=[osm.min(), osm.max()], mode="lines",
                             line=dict(color=PAL[1], dash="dash")), 1, 2)
    sl = np.sqrt(np.abs(rs))
    fig.add_trace(go.Scatter(x=fitted, y=sl, mode="markers", marker=mk), 2, 1)
    lx, ly = _lowess(fitted, sl)
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color=PAL[1], width=2)), 2, 1)
    cook = rs ** 2 * h / (k * np.clip(1 - h, 1e-12, None))
    fig.add_trace(go.Scatter(x=h, y=rs, mode="markers",
                             marker=dict(color=cook, colorscale=[[0, "#7f9fbf"], [0.15, "#c55a11"], [1, "#8b0000"]], cmin=0,
                                         cmax=max(float(cook.max()), 4 / n), size=6, opacity=0.8, showscale=False),
                             text=[f"obs {i}: Cook's D = {c:.3f}" for i, c in zip(range(n), cook)],
                             hoverinfo="text"), 2, 2)
    hh = np.linspace(max(h.min(), 1e-3), min(h.max() * 1.05, 0.99), 60)
    for D in (0.5, 1.0):
        bound = np.sqrt(D * k * (1 - hh) / hh)
        for sgn in (1, -1):
            fig.add_trace(go.Scatter(x=hh, y=sgn * bound, mode="lines", line=dict(color="#999", dash="dot", width=1),
                                     hoverinfo="skip"), 2, 2)
    lim = max(3.5, np.abs(rs).max() * 1.1)
    fig.update_yaxes(range=[-lim, lim], row=2, col=2)
    fig.update_xaxes(title_text=_a("Fitted", rtl), row=1, col=1)
    fig.update_xaxes(title_text=_a("Theoretical quantiles", rtl), row=1, col=2)
    fig.update_xaxes(title_text=_a("Fitted", rtl), row=2, col=1)
    fig.update_xaxes(title_text=_a("Leverage", rtl), row=2, col=2)
    style_academic(fig, height=720, legend="none", rtl=rtl)
    return fig


def actual_vs_fitted(r: rg.RegResult, x_axis=None, x_title=None, rtl=False, names=("Actual", "Fitted")) -> go.Figure:
    x_title = x_title or _a("Observation", rtl)
    x = np.arange(1, r.nobs + 1) if x_axis is None else np.asarray(x_axis)
    o = np.argsort(x, kind="stable")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[o], y=r.y[o], mode="markers+lines" if x_axis is None else "markers",
                             name=names[0], marker=dict(color=PAL[0], size=5), line=dict(width=1)))
    fig.add_trace(go.Scatter(x=x[o], y=r.fitted[o], mode="lines" if x_axis is None else "markers", name=names[1],
                             line=dict(color=PAL[1], width=2), marker=dict(color=PAL[1], size=5, symbol="x")))
    style_academic(fig, x_title=x_title, y_title=r.y_label, height=420, rtl=rtl)
    return fig


def observed_vs_predicted(r: rg.RegResult, rtl=False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r.fitted, y=r.y, mode="markers", marker=dict(color=PAL[0], size=6, opacity=0.6)))
    lo, hi = float(min(r.fitted.min(), r.y.min())), float(max(r.fitted.max(), r.y.max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=PAL[1], dash="dash")))
    style_academic(fig, x_title=_a("Predicted", rtl), y_title=_a("Observed", rtl), height=420, legend="none", rtl=rtl)
    return fig


def residual_hist(r: rg.RegResult, rtl=False) -> go.Figure:
    e = np.asarray(r.wresid, float)
    fig = go.Figure(go.Histogram(x=e, histnorm="probability density", marker_color=PAL[0], opacity=0.75,
                                 nbinsx=int(min(80, max(15, 2 * np.sqrt(len(e)))))))
    g = np.linspace(e.min(), e.max(), 200)
    fig.add_trace(go.Scatter(x=g, y=stats.norm.pdf(g, e.mean(), e.std(ddof=1)), mode="lines",
                             line=dict(color=PAL[1], width=2)))
    style_academic(fig, x_title=_a("Residual", rtl), y_title=_a("Density", rtl), height=380, legend="none", rtl=rtl)
    return fig


def acf_plot(acf: np.ndarray, n: int, rtl=False) -> go.Figure:
    lags = np.arange(1, len(acf) + 1)
    band = 1.96 / np.sqrt(n)
    fig = go.Figure(go.Bar(x=lags, y=acf, marker_color=PAL[0], width=0.3))
    fig.add_hline(y=band, line=dict(color=PAL[1], dash="dash"))
    fig.add_hline(y=-band, line=dict(color=PAL[1], dash="dash"))
    style_academic(fig, x_title=_a("Lag", rtl), y_title=_a("ACF of residuals", rtl), height=340, legend="none", rtl=rtl)
    return fig


def cusum_plots(cs: dict, rtl=False, titles=("CUSUM", "CUSUM of squares")) -> go.Figure:
    p = cs["path"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=titles, horizontal_spacing=0.1)
    fig.add_trace(go.Scatter(x=p["t"], y=p["CUSUM"], mode="lines", line=dict(color=PAL[0], width=2)), 1, 1)
    for c in ("lower", "upper"):
        fig.add_trace(go.Scatter(x=p["t"], y=p[c], mode="lines", line=dict(color=PAL[1], dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=p["t"], y=p["CUSUMSQ"], mode="lines", line=dict(color=PAL[0], width=2)), 1, 2)
    for c in ("sq_lower", "sq_upper"):
        fig.add_trace(go.Scatter(x=p["t"], y=p[c], mode="lines", line=dict(color=PAL[1], dash="dash")), 1, 2)
    fig.update_xaxes(title_text=_a("Observation", rtl))
    style_academic(fig, height=400, legend="none", rtl=rtl)
    return fig


def coef_plot(results: list, labels: list, drop_const=True, rtl=False) -> go.Figure:
    """Forest plot of coefficients with 95% CIs, one marker series per model."""
    fig = go.Figure()
    names = []
    for r in results:
        for n in r.names:
            if n not in names and not (drop_const and n == "const"):
                names.append(n)
    for j, (r, lab) in enumerate(zip(results, labels)):
        sel = [n for n in names if n in r.names]
        off = (j - (len(results) - 1) / 2) * 0.18
        y = [names.index(n) + off for n in sel]
        b = r.params[sel].to_numpy()
        lo, hi = r.ci.loc[sel, "low"].to_numpy(), r.ci.loc[sel, "high"].to_numpy()
        fig.add_trace(go.Scatter(x=b, y=y, mode="markers", name=lab, marker=dict(color=PAL[j % len(PAL)], size=9),
                                 error_x=dict(type="data", symmetric=False, array=hi - b, arrayminus=b - lo,
                                              thickness=1.6, width=4)))
    fig.add_vline(x=0, line=dict(color="#777", dash="dot"))
    fig.update_yaxes(tickvals=list(range(len(names))), ticktext=names, autorange="reversed")
    style_academic(fig, x_title=_a("Coefficient (95% CI)", rtl), height=max(320, 42 * len(names) + 120),
                   legend="bottom" if len(results) > 1 else "none", rtl=rtl)
    return fig


def effect_curve(df: pd.DataFrame, r: rg.RegResult, var: str, by: str | None = None, rtl=False,
                 show_points=True) -> go.Figure:
    """Predicted Y across the observed range of `var` (others at the representative point), with 95% CI
    band; one curve per level of `by` (categorical) — shows polynomials and interactions exactly."""
    base = rg.baseline_values(df, r)
    raw = pd.to_numeric(df.loc[r.design.index, var], errors="coerce")
    grid = np.linspace(raw.min(), raw.max(), 80)
    if r.design.info["hows"].get(var) == "log":
        grid = grid[grid > 0]
    fig = go.Figure()
    levels = r.design.info["levels"].get(by, [None]) if by else [None]
    for j, lv in enumerate(levels):
        vals = pd.concat([base] * len(grid), ignore_index=True)
        vals[var] = grid
        if lv is not None:
            vals[by] = lv
        p = rg.predict(r, vals)
        col = PAL[j % len(PAL)]
        fig.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[p["ci_high"], p["ci_low"][::-1]], fill="toself",
                                 fillcolor=col, opacity=0.15, line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=grid, y=p["fit"], mode="lines", line=dict(color=col, width=2.5),
                                 name=str(lv) if lv is not None else _a("Predicted", rtl)))
    if show_points:
        yraw = pd.to_numeric(df.loc[r.design.index, r.design.spec.y], errors="coerce")
        fig.add_trace(go.Scatter(x=raw, y=yraw, mode="markers", marker=dict(color="#888", size=4, opacity=0.3),
                                 name=_a("Observed", rtl), showlegend=False))
    style_academic(fig, x_title=var, y_title=r.design.spec.y, height=440,
                   legend="bottom" if by else "none", rtl=rtl)
    return fig


def slr_bands(df: pd.DataFrame, r: rg.RegResult, var: str, rtl=False) -> go.Figure:
    """Simple regression: fitted line with 95% confidence band (mean) and prediction band (new observation)."""
    base = rg.baseline_values(df, r)
    raw = pd.to_numeric(df.loc[r.design.index, var], errors="coerce")
    grid = np.linspace(raw.min(), raw.max(), 80)
    vals = pd.concat([base] * len(grid), ignore_index=True)
    vals[var] = grid
    p = rg.predict(r, vals)
    yraw = pd.to_numeric(df.loc[r.design.index, r.design.spec.y], errors="coerce")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=raw, y=yraw, mode="markers", marker=dict(color=PAL[0], size=6, opacity=0.55), name=_a("Observed", rtl)))
    for lo, hi, nm, op in (("pi_low", "pi_high", _a("95% prediction band", rtl), 0.10), ("ci_low", "ci_high", _a("95% confidence band", rtl), 0.25)):
        fig.add_trace(go.Scatter(x=np.r_[grid, grid[::-1]], y=np.r_[p[hi], p[lo][::-1]], fill="toself", fillcolor=PAL[1],
                                 opacity=op, line=dict(width=0), name=nm))
    fig.add_trace(go.Scatter(x=grid, y=p["fit"], mode="lines", line=dict(color=PAL[1], width=2.5), name=_a("Fitted", rtl)))
    style_academic(fig, x_title=var, y_title=r.design.spec.y, height=460, rtl=rtl)
    return fig


def boxcox_profile(bc: dict, rtl=False) -> go.Figure:
    pr = bc["profile"]
    fig = go.Figure(go.Scatter(x=pr["lambda"], y=pr["loglik"], mode="lines", line=dict(color=PAL[0], width=2.5)))
    cut = pr["loglik"].max() - stats.chi2.ppf(0.95, 1) / 2
    fig.add_hline(y=cut, line=dict(color=PAL[1], dash="dash"), annotation_text="95% CI")
    fig.add_vline(x=bc["lambda"], line=dict(color=PAL[2]), annotation_text=f"λ̂ = {bc['lambda']:.3f}")
    for v in (0, 1):
        fig.add_vline(x=v, line=dict(color="#999", dash="dot"))
    style_academic(fig, x_title=_a("λ", rtl), y_title=_a("Profile log-likelihood", rtl), height=400, legend="none", rtl=rtl)
    return fig


def jn_plot(jn: dict, moderator: str, focal: str, rtl=False) -> go.Figure:
    c = jn["curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.r_[c[moderator], c[moderator][::-1]], y=np.r_[c["high"], c["low"][::-1]],
                             fill="toself", fillcolor=PAL[0], opacity=0.18, line=dict(width=0), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=c[moderator], y=c["slope"], mode="lines", line=dict(color=PAL[0], width=2.5)))
    fig.add_hline(y=0, line=dict(color="#777", dash="dot"))
    for rt in jn["roots"]:
        if jn["range"][0] <= rt <= jn["range"][1]:
            fig.add_vline(x=rt, line=dict(color=PAL[1], dash="dash"), annotation_text=f"{rt:.3g}")
    style_academic(fig, x_title=moderator, y_title=f"{_a('Conditional effect of', rtl)} {focal}", height=420, legend="none", rtl=rtl)
    return fig


def simple_slopes_plot(r: rg.RegResult, focal: str, moderator: str, rtl=False) -> go.Figure:
    X = r.design.X
    xs = np.linspace(X[focal].min(), X[focal].max(), 50)
    m, s = X[moderator].mean(), X[moderator].std(ddof=1)
    inter = f"{focal}×{moderator}" if f"{focal}×{moderator}" in r.names else f"{moderator}×{focal}"
    others = r.design.X.mean()
    fig = go.Figure()
    for j, (lab, mv) in enumerate((("−1 SD", m - s), ("Mean", m), ("+1 SD", m + s))):
        row = others.copy()
        yhat = []
        for x in xs:
            row[focal], row[moderator], row[inter] = x, mv, x * mv
            yhat.append(float(row[r.names].to_numpy() @ r.params.to_numpy()))
        fig.add_trace(go.Scatter(x=xs, y=yhat, mode="lines", name=f"{moderator} {lab}",
                                 line=dict(color=PAL[j], width=2.5, dash=["dot", "solid", "dash"][j])))
    style_academic(fig, x_title=focal, y_title=r.y_label, height=420, rtl=rtl)
    return fig


def nls_curve(r: rg.RegResult, rtl=False) -> go.Figure:
    x, y = r.extra["x"], r.y
    g = np.linspace(x.min(), x.max(), 200)
    f = r.extra["fun"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", marker=dict(color=PAL[0], size=6, opacity=0.6), name=_a("Observed", rtl)))
    fig.add_trace(go.Scatter(x=g, y=f(g, *r.params.to_numpy()), mode="lines", line=dict(color=PAL[1], width=2.5),
                             name=_a("NLS fit", rtl)))
    style_academic(fig, x_title=r.extra["x_name"], y_title=r.extra["y_label"], height=440, rtl=rtl)
    return fig
