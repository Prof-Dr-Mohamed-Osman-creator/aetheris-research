# ==========================================
# Aetheris Research - Charts for v1.5 models
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import special, stats

from analytics import count_surv as cs
from analytics import margins as mg
from analytics import regression as rg
from analytics.charts import PALETTES, style_academic
from analytics.reg_charts import _a

PAL = PALETTES["academic"]
AR = {"False positive rate (1 − specificity)": "معدل الإيجابيات الكاذبة (1 − النوعية)",
      "True positive rate (sensitivity)": "معدل الإيجابيات الصحيحة (الحساسية)", "Predicted": "المتنبأ به",
      "Observed": "المشاهد", "Share of observations": "نسبة المشاهدات", "Count": "العدد", "Time": "الزمن",
      "Survival probability": "احتمال البقاء", "Average prediction": "متوسط التنبؤ", "Predicted probability": "الاحتمال المتنبأ به"}


def _t(s, rtl):
    return AR.get(s, s) if rtl else s


def roc_plot(roc: dict, rtl=False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", line=dict(color=PAL[0], width=2.5),
                             name=f"AUC = {roc['auc']:.3f}"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#999", dash="dot"), showlegend=False))
    fig.add_trace(go.Scatter(x=[1 - roc["youden_spec"]], y=[roc["youden_sens"]], mode="markers",
                             marker=dict(color=PAL[1], size=10), name=f"Youden cut-off {roc['youden_cut']:.2f}"))
    style_academic(fig, x_title=_t("False positive rate (1 − specificity)", rtl), y_title=_t("True positive rate (sensitivity)", rtl),
                   height=440, rtl=rtl)
    return fig


def prob_hist(y, p, rtl=False) -> go.Figure:
    fig = go.Figure()
    for j, v in enumerate((0, 1)):
        fig.add_trace(go.Histogram(x=p[np.asarray(y) == v], name=f"y = {v}", opacity=0.6, marker_color=PAL[j],
                                   nbinsx=30, histnorm="probability"))
    fig.update_layout(barmode="overlay")
    style_academic(fig, x_title=_t("Predicted probability", rtl), y_title=_t("Share of observations", rtl), height=380, rtl=rtl)
    return fig


def response_curve(df, r, var: str, pred, outputs=None, points: int = 25, rtl=False, scale: float = 1.0) -> go.Figure:
    """Average prediction over the sample as `var` is set to each grid value (others as observed), with
    delta-method 95% band (one line per outcome for multi-output models)."""
    raw = pd.to_numeric(df.loc[r.design.index, var], errors="coerce")
    grid = np.linspace(raw.quantile(0.01), raw.quantile(0.99), points)
    rows = []
    for gv in grid:
        sc = mg.scenario(df, r, pred, {var: gv}, mode="set", outputs=outputs)
        se = sc["Std. Err."].to_numpy()  # SE of the change; for the level use the scenario prediction only
        rows.append(sc["Scenario"].to_numpy())
    P = np.vstack(rows)
    fig = go.Figure()
    names = outputs or [_t("Average prediction", rtl)]
    for j in range(P.shape[1]):
        fig.add_trace(go.Scatter(x=grid, y=P[:, j] * scale, mode="lines", name=str(names[j]),
                                 line=dict(color=PAL[j % len(PAL)], width=2.5)))
    style_academic(fig, x_title=var, y_title=_t("Average prediction", rtl), height=420,
                   legend="bottom" if P.shape[1] > 1 else "none", rtl=rtl)
    return fig


def count_freq_plot(freq: pd.DataFrame, rtl=False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=freq["Count"], y=freq["Observed share"], name=_t("Observed", rtl), marker_color=PAL[0], opacity=0.75))
    fig.add_trace(go.Scatter(x=freq["Count"], y=freq["Predicted share"], name=_t("Predicted", rtl), mode="lines+markers",
                             line=dict(color=PAL[1], width=2.5)))
    style_academic(fig, x_title=_t("Count", rtl), y_title=_t("Share of observations", rtl), height=400, rtl=rtl)
    return fig


def km_plot(time, event, group=None, rtl=False, ci=True) -> go.Figure:
    fig = go.Figure()
    t, d = np.asarray(time, float), np.asarray(event, int)
    groups = [None] if group is None else sorted(pd.Series(group).astype(str).unique())
    g = None if group is None else pd.Series(group).astype(str).to_numpy()
    for j, lv in enumerate(groups):
        sel = np.ones(len(t), bool) if lv is None else g == lv
        km = cs.kaplan_meier(t[sel], d[sel])
        x = np.r_[0, km["time"].to_numpy()]
        s = np.r_[1, km["survival"].to_numpy()]
        col = PAL[j % len(PAL)]
        if ci:
            lo, hi = np.r_[1, km["CI low"].to_numpy()], np.r_[1, km["CI high"].to_numpy()]
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[hi, lo[::-1]], fill="toself", fillcolor=col, opacity=0.12,
                                     line=dict(width=0, shape="hv"), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=x, y=s, mode="lines", line=dict(color=col, width=2.5, shape="hv"),
                                 name="All" if lv is None else str(lv)))
    style_academic(fig, x_title=_t("Time", rtl), y_title=_t("Survival probability", rtl), height=440,
                   legend="bottom" if group is not None else "none", rtl=rtl)
    fig.update_yaxes(range=[0, 1.02])
    return fig


def aft_curves(df, r, var: str, rtl=False) -> go.Figure:
    """Model-based survivor functions S(t | x) at the 10th, 50th and 90th percentile of `var` (others at
    their representative values)."""
    dist = r.extra["dist"]
    k = r.design.X.shape[1]
    th = r.params.to_numpy()
    s = 1.0 if dist == "exponential" else np.exp(th[k])
    tmax = float(np.quantile(r.extra["time"], 0.95))
    tt = np.linspace(tmax / 200, tmax, 200)
    base = rg.baseline_values(df, r)
    raw = pd.to_numeric(df.loc[r.design.index, var], errors="coerce")
    fig = go.Figure()
    vals = [raw.quantile(q) for q in (0.1, 0.5, 0.9)] if raw.nunique() > 2 else sorted(raw.unique())
    for j, v in enumerate(vals):
        row = base.copy()
        row[var] = v
        mu = float(rg.design_rows(r, row).to_numpy(float)[0] @ th[:k])
        w = (np.log(tt) - mu) / s
        S = {"weibull": np.exp(-np.exp(w)), "exponential": np.exp(-np.exp(w)), "lognormal": stats.norm.sf(w),
             "loglogistic": special.expit(-w)}[dist]
        fig.add_trace(go.Scatter(x=tt, y=S, mode="lines", name=f"{var} = {v:.3g}", line=dict(color=PAL[j], width=2.5)))
    style_academic(fig, x_title=_t("Time", rtl), y_title=_t("Survival probability", rtl), height=420, rtl=rtl)
    return fig


def cox_curves(df, r, var: str, rtl=False) -> go.Figure:
    """Cox model survivor functions S(t|x) = S₀(t)^exp(xβ) (Breslow baseline) at percentiles of `var`."""
    res = r.extra["sm"]
    bch = res.baseline_cumulative_hazard[0]
    t0, H0 = np.asarray(bch[0]), np.asarray(bch[1])
    X = r.design.X.drop(columns=[c for c in ("const",) if c in r.design.X.columns])
    k = X.shape[1]
    th = r.params.to_numpy()[:k]
    Xc = X.to_numpy(float).mean(axis=0)            # statsmodels centres the covariates
    base = rg.baseline_values(df, r)
    raw = pd.to_numeric(df.loc[r.design.index, var], errors="coerce")
    vals = [raw.quantile(q) for q in (0.1, 0.5, 0.9)] if raw.nunique() > 2 else sorted(raw.unique())
    fig = go.Figure()
    for j, v in enumerate(vals):
        row = base.copy()
        row[var] = v
        x = rg.design_rows(r, row).to_numpy(float)[0]
        x = x[1:] if len(x) == k + 1 else x
        S = np.exp(-H0 * np.exp((x - Xc) @ th))
        fig.add_trace(go.Scatter(x=t0, y=S, mode="lines", line=dict(color=PAL[j], width=2.5, shape="hv"),
                                 name=f"{var} = {v:.3g}"))
    style_academic(fig, x_title=_t("Time", rtl), y_title=_t("Survival probability", rtl), height=420, rtl=rtl)
    return fig
