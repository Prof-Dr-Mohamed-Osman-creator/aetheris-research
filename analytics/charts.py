# ==========================================
# Aetheris Research - Academic chart engine
# ------------------------------------------
# Builds publication-quality Plotly figures. No Streamlit here: the same
# builders are reused by the Chart Builder, the report generator and the API.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from analytics.descriptive import lorenz_curve, gini

# ------------------------------------------
# Palettes (all checked for colour-vision deficiency except "vivid")
# ------------------------------------------
PALETTES = {
    "academic": ["#1f4e79", "#c55a11", "#548235", "#7f6000", "#7030a0", "#2e75b6", "#bf9000", "#843c0c"],
    "colorblind": px.colors.qualitative.Safe,
    "grayscale": ["#111111", "#555555", "#888888", "#aaaaaa", "#333333", "#777777", "#bbbbbb", "#222222"],
    "vivid": px.colors.qualitative.Bold,
}
PALETTE_LABELS = {
    "academic": ("Academic (journal)", "أكاديمي (مجلات علمية)"),
    "colorblind": ("Colour-blind safe", "آمن لعمى الألوان"),
    "grayscale": ("Grayscale (print)", "تدرج رمادي (للطباعة)"),
    "vivid": ("Vivid (presentations)", "ألوان زاهية (للعروض)"),
}
SEQUENTIAL = {"academic": "Blues", "colorblind": "Viridis", "grayscale": "Greys", "vivid": "Plasma"}

AGG_FUNCS = ["sum", "mean", "median", "count", "min", "max"]

# Chart catalogue: key -> (English, Arabic, required roles)
# roles: x, y, color, size, y2, value, names
CHART_TYPES: dict[str, tuple[str, str]] = {
    "bar": ("Bar chart (simple / grouped / stacked)", "أعمدة بيانية (بسيطة / مجمّعة / مكدّسة)"),
    "hbar": ("Horizontal bar chart", "أعمدة أفقية"),
    "pie": ("Pie / donut chart", "دائرة نسبية / حلقية"),
    "line": ("Line chart", "منحنى خطي"),
    "area": ("Area chart (stacked)", "رسم مساحي (مكدّس)"),
    "dual": ("Dual-axis line chart", "منحنى خطي بمحورين"),
    "scatter": ("Scatter plot + regression line", "شكل انتشار مع خط الانحدار"),
    "bubble": ("Bubble chart", "رسم فقاعي"),
    "histogram": ("Histogram", "مدرج تكراري"),
    "polygon": ("Frequency polygon", "مضلع تكراري"),
    "ecdf": ("Cumulative frequency curve (ECDF)", "المنحنى التكراري المتجمع"),
    "box": ("Box plot", "صندوق ذو الطرفين (Box plot)"),
    "violin": ("Violin plot", "رسم كماني (Violin)"),
    "pareto": ("Pareto chart", "مخطط باريتو"),
    "lorenz": ("Lorenz curve", "منحنى لورنز"),
    "heatmap": ("Heat map (two categories × value)", "خريطة حرارية (فئتان × قيمة)"),
    "map": ("World map (ISO3 country codes)", "خريطة العالم (رموز الدول ISO3)"),
}


def _aggregate(df: pd.DataFrame, keys: list[str], y: str, agg: str) -> pd.DataFrame:
    keys = [k for k in keys if k]
    if agg == "count":
        return df.groupby(keys, dropna=False).size().reset_index(name=y)
    return df.groupby(keys, dropna=False)[y].agg(agg).reset_index()


def build_chart(kind: str, df: pd.DataFrame, x: str | None = None, y: str | None = None,
                color: str | None = None, size: str | None = None, y2: str | None = None,
                agg: str = "sum", barmode: str = "group", palette: str = "academic",
                trendline: bool = True, donut: bool = False, bins: int = 20,
                show_values: bool = False, lang: str = "en") -> tuple[go.Figure, dict]:
    """Return (figure, info). `info` carries extra results, e.g. regression equation or Gini."""
    seq = PALETTES[palette]
    info: dict = {}
    d = df.copy()
    if color and color in d.columns:
        d[color] = d[color].astype(str)

    if kind in ("bar", "hbar"):
        data = _aggregate(d, [x, color], y, agg)
        if kind == "bar":
            fig = px.bar(data, x=x, y=y, color=color, barmode=barmode, color_discrete_sequence=seq,
                         text_auto=".3s" if show_values else False)
        else:
            data = data.sort_values(y)
            fig = px.bar(data, y=x, x=y, color=color, barmode=barmode, orientation="h",
                         color_discrete_sequence=seq, text_auto=".3s" if show_values else False)

    elif kind == "pie":
        data = _aggregate(d, [x], y, agg)
        fig = px.pie(data, names=x, values=y, hole=0.45 if donut else 0, color_discrete_sequence=seq)
        fig.update_traces(textinfo="percent+label", textposition="inside")

    elif kind in ("line", "area"):
        data = _aggregate(d, [x, color], y, agg).sort_values(x)
        f = px.line if kind == "line" else px.area
        fig = f(data, x=x, y=y, color=color, color_discrete_sequence=seq,
                **({"markers": True} if kind == "line" else {}))

    elif kind == "dual":
        data = d.groupby(x)[[y, y2]].agg(agg if agg != "count" else "mean").reset_index().sort_values(x)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=data[x], y=data[y], name=y, mode="lines+markers",
                                 line=dict(color=seq[0], width=2.5)), secondary_y=False)
        fig.add_trace(go.Scatter(x=data[x], y=data[y2], name=y2, mode="lines+markers",
                                 line=dict(color=seq[1], width=2.5, dash="dash")), secondary_y=True)
        fig.update_yaxes(title_text=y, secondary_y=False)
        fig.update_yaxes(title_text=y2, secondary_y=True)

    elif kind in ("scatter", "bubble"):
        cols = [c for c in [x, y, color, size] if c]
        data = d[cols].dropna()
        fig = px.scatter(data, x=x, y=y, color=color, size=size if kind == "bubble" else None,
                         color_discrete_sequence=seq, opacity=0.75, size_max=40)
        if trendline and kind == "scatter":
            xs, ys = pd.to_numeric(data[x], errors="coerce"), pd.to_numeric(data[y], errors="coerce")
            ok = xs.notna() & ys.notna()
            if ok.sum() >= 3 and xs[ok].nunique() > 1:
                lr = stats.linregress(xs[ok], ys[ok])
                grid = np.linspace(xs[ok].min(), xs[ok].max(), 100)
                fig.add_trace(go.Scatter(x=grid, y=lr.intercept + lr.slope * grid, mode="lines",
                                         name="OLS", line=dict(color="#c00000", width=2)))
                info = {"intercept": lr.intercept, "slope": lr.slope, "r2": lr.rvalue ** 2,
                        "p": lr.pvalue, "n": int(ok.sum())}
                sign = "+" if lr.slope >= 0 else "−"
                eq = f"ŷ = {lr.intercept:.4g} {sign} {abs(lr.slope):.4g}·x   R² = {lr.rvalue ** 2:.3f}"
                fig.add_annotation(xref="paper", yref="paper", x=0.01, y=0.99, text=eq, showarrow=False,
                                   align="left", bgcolor="rgba(255,255,255,0.85)", font=dict(size=13))

    elif kind == "histogram":
        fig = px.histogram(d, x=x, color=color, nbins=bins, barmode="overlay" if color else "relative",
                           color_discrete_sequence=seq, opacity=0.8 if color else 1)
        fig.update_traces(marker_line_color="white", marker_line_width=1)

    elif kind == "polygon":
        fig = go.Figure()
        groups = d.groupby(color) if color else [(None, d)]
        allx = pd.to_numeric(d[x], errors="coerce").dropna()
        edges = np.histogram_bin_edges(allx, bins=bins)
        mids = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]
        for i, (g, sub) in enumerate(groups):
            counts, _ = np.histogram(pd.to_numeric(sub[x], errors="coerce").dropna(), bins=edges)
            # close the polygon at zero on both sides (textbook construction)
            px_ = np.concatenate([[mids[0] - width], mids, [mids[-1] + width]])
            py_ = np.concatenate([[0], counts, [0]])
            fig.add_trace(go.Scatter(x=px_, y=py_, mode="lines+markers", name=str(g) if g is not None else x,
                                     line=dict(color=seq[i % len(seq)], width=2)))

    elif kind == "ecdf":
        fig = px.ecdf(d, x=x, color=color, color_discrete_sequence=seq)

    elif kind in ("box", "violin"):
        if x:
            d[x] = d[x].astype(str)
        f = px.box if kind == "box" else px.violin
        extra = {"points": "outliers"} if kind == "box" else {"box": True, "points": False}
        fig = f(d, x=x if x else None, y=y, color=color or (x if x else None),
                color_discrete_sequence=seq, **extra)

    elif kind == "pareto":
        data = _aggregate(d, [x], y, agg).sort_values(y, ascending=False)
        data["cum"] = data[y].cumsum() / data[y].sum() * 100
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=data[x].astype(str), y=data[y], name=y, marker_color=seq[0]), secondary_y=False)
        fig.add_trace(go.Scatter(x=data[x].astype(str), y=data["cum"], name="%", mode="lines+markers",
                                 line=dict(color=seq[1], width=2)), secondary_y=True)
        fig.add_hline(y=80, line_dash="dot", line_color="#888", secondary_y=True)
        fig.update_yaxes(range=[0, 105], ticksuffix="%", secondary_y=True)

    elif kind == "lorenz":
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Equality" if lang == "en" else "خط المساواة",
                                 line=dict(color="#888", dash="dash")))
        groups = d.groupby(color) if color else [(None, d)]
        ginis = {}
        for i, (g, sub) in enumerate(groups):
            lc = lorenz_curve(sub[y])
            gv = gini(sub[y])
            label = (str(g) if g is not None else y) + f" (G = {gv:.3f})"
            ginis[str(g) if g is not None else y] = gv
            fig.add_trace(go.Scatter(x=lc["population_share"], y=lc["value_share"], mode="lines", name=label,
                                     line=dict(color=seq[i % len(seq)], width=2.5),
                                     fill="tonexty" if not color else None))
        fig.update_xaxes(range=[0, 1], tickformat=".0%")
        fig.update_yaxes(range=[0, 1], tickformat=".0%", scaleanchor="x", scaleratio=1)
        info = {"gini": ginis}

    elif kind == "heatmap":
        data = _aggregate(d, [x, color], y, agg)
        pivot = data.pivot(index=color, columns=x, values=y)
        fig = px.imshow(pivot, text_auto=".3s", aspect="auto", color_continuous_scale=SEQUENTIAL[palette])

    elif kind == "map":
        data = _aggregate(d, [x], y, agg)
        fig = px.choropleth(data, locations=x, locationmode="ISO-3", color=y,
                            color_continuous_scale=SEQUENTIAL[palette])
        fig.update_geos(showframe=False, showcoastlines=True, projection_type="natural earth")

    else:
        raise ValueError(f"Unknown chart type: {kind}")

    return fig, info


def style_academic(fig: go.Figure, title: str = "", x_title: str | None = None, y_title: str | None = None,
                   source: str = "", font_size: int = 14, height: int = 520, legend: str = "bottom",
                   gridlines: bool = True, rtl: bool = False) -> go.Figure:
    """Journal-style layout: white background, thin axes, readable fonts, figure caption + source note."""
    legend_pos = {
        "bottom": dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        "right": dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        "top": dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        "none": None,
    }[legend]
    fig.update_layout(
        template="simple_white",
        title=dict(text=f"<b>{title}</b>" if title else "", x=0.5, xanchor="center", font=dict(size=font_size + 3)),
        font=dict(family="Times New Roman, Amiri, Arial, serif", size=font_size, color="#111"),
        height=height,
        margin=dict(l=60, r=40, t=70 if title else 30, b=110 if source or legend == "bottom" else 60),
        showlegend=legend != "none",
        legend=legend_pos or {},
        bargap=0.15,
    )
    axis = dict(showline=True, linewidth=1, linecolor="#111", ticks="outside", mirror=False,
                showgrid=gridlines, gridcolor="#e5e5e5", zeroline=False)
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    if x_title is not None:
        fig.update_xaxes(title_text=x_title)
    if y_title is not None:
        if _has_secondary(fig):  # dual-axis / Pareto: keep the right-hand axis title
            fig.update_yaxes(title_text=y_title, secondary_y=False)
        else:
            fig.update_yaxes(title_text=y_title)
    if source:
        fig.add_annotation(text=source, xref="paper", yref="paper", x=1 if rtl else 0, y=-0.30,
                           xanchor="right" if rtl else "left", showarrow=False,
                           font=dict(size=font_size - 2, color="#444"), align="right" if rtl else "left")
    return fig


def _has_secondary(fig: go.Figure) -> bool:
    return any(getattr(tr, "yaxis", None) == "y2" for tr in fig.data)


def plotly_export_config(fmt: str = "png", scale: int = 3, filename: str = "aetheris_figure") -> dict:
    """Modebar camera button exports high-resolution PNG (scale 3 ≈ 300 dpi) or vector SVG."""
    return {
        "displaylogo": False,
        "toImageButtonOptions": {"format": fmt, "filename": filename, "scale": scale},
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    }
