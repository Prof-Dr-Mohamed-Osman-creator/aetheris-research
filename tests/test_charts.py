"""Builds every chart type on the demo data (runs where plotly is installed, e.g. CI)."""
import pytest

pytest.importorskip("plotly")

from analytics import charts as ch  # noqa: E402
from data.samples import climate_crop_demo  # noqa: E402

ARGS = {
    "bar": dict(x="crop", y="production_t", color="governorate"),
    "hbar": dict(x="governorate", y="production_t"),
    "pie": dict(x="crop", y="production_t", donut=True),
    "line": dict(x="year", y="yield_t_per_feddan", color="crop"),
    "area": dict(x="year", y="production_t", color="crop"),
    "dual": dict(x="year", y="avg_temp_c", y2="yield_t_per_feddan"),
    "scatter": dict(x="avg_temp_c", y="yield_t_per_feddan"),
    "bubble": dict(x="avg_temp_c", y="yield_t_per_feddan", size="area_feddan", color="crop"),
    "histogram": dict(x="yield_t_per_feddan", color="crop"),
    "polygon": dict(x="yield_t_per_feddan", color="crop"),
    "ecdf": dict(x="yield_t_per_feddan"),
    "box": dict(x="crop", y="yield_t_per_feddan"),
    "violin": dict(y="yield_t_per_feddan"),
    "pareto": dict(x="governorate", y="production_t"),
    "lorenz": dict(y="production_t", color="crop"),
    "heatmap": dict(x="governorate", color="crop", y="yield_t_per_feddan", agg="mean"),
    "map": dict(x="iso3", y="production_t"),
}


@pytest.mark.parametrize("kind", list(ch.CHART_TYPES))
def test_every_chart_builds(kind):
    df = climate_crop_demo().assign(iso3="EGY")
    fig, info = ch.build_chart(kind, df, **ARGS[kind])
    ch.style_academic(fig, title="T", x_title="x", y_title="y", source="Source: test")
    assert fig.to_html(include_plotlyjs=False)
    if kind == "scatter":
        assert 0 <= info["r2"] <= 1
