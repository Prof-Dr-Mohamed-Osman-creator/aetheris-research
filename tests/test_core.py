"""Unit tests for the pure (non-UI) modules. Run: python -m pytest -q"""
import json

import numpy as np
import pandas as pd
import pytest

from analytics import descriptive as ds
from data.merge import merge_datasets
from data.engines import fred, worldbank as wb
from data.samples import climate_crop_demo


@pytest.fixture
def demo():
    return climate_crop_demo()


def test_summary_matches_pandas(demo):
    t = ds.summary_table(demo, ["avg_temp_c", "yield_t_per_feddan"])
    s = demo["avg_temp_c"]
    assert t.loc["avg_temp_c", "N"] == s.count()
    assert np.isclose(t.loc["avg_temp_c", "Mean"], s.mean())
    assert np.isclose(t.loc["avg_temp_c", "Std. Dev."], s.std())
    assert np.isclose(t.loc["avg_temp_c", "Skewness"], s.skew())
    assert np.isclose(t.loc["avg_temp_c", "Kurtosis (excess)"], s.kurt())
    lo, hi = t.loc["avg_temp_c", ["95% CI Lower", "95% CI Upper"]]
    assert lo < s.mean() < hi


def test_column_typing(demo):
    assert "year" in ds.time_columns(demo)
    assert {"governorate", "crop"} <= set(ds.categorical_columns(demo))


def test_normality_detects_skew():
    rng = np.random.default_rng(0)
    normal = pd.Series(rng.normal(size=400))
    skewed = pd.Series(rng.lognormal(size=400))
    assert ds.normality_tests(normal)["Normal at α"].all()
    assert not ds.normality_tests(skewed)["Normal at α"].any()


def test_correlation_matches_pandas(demo):
    cols = ["avg_temp_c", "rainfall_mm", "yield_t_per_feddan"]
    r, p, n = ds.correlation_matrix(demo, cols, "pearson")
    assert np.allclose(r.values, demo[cols].corr().values)
    rs, _, _ = ds.correlation_matrix(demo, cols, "spearman")
    assert np.allclose(rs.values, demo[cols].corr("spearman").values)
    star = ds.starred_matrix(r, p)
    assert star.iloc[0, 1] == "" and star.iloc[1, 1] == "1"


def test_growth_recovers_known_rate():
    years = np.arange(2000, 2021)
    df = pd.DataFrame({"year": years, "y": 100 * 1.05 ** (years - 2000)})
    g = ds.growth_analysis(df, "year", "y").iloc[0]
    assert np.isclose(g["CAGR (%)"], 5.0)
    assert np.isclose(g["Trend growth (%/yr, log-lin)"], 5.0)


def test_growth_by_entity(demo):
    wheat = demo[demo.crop == "Wheat"]
    g = ds.growth_analysis(wheat, "year", "yield_t_per_feddan", "governorate")
    assert len(g) == 8


def test_frequency_and_chi2(demo):
    ft = ds.frequency_table(demo["crop"])
    assert np.isclose(ft["Percent (%)"].sum(), 100)
    ct, res = ds.crosstab_chi2(demo, "governorate", "crop")
    assert res["p"] > 0.99  # balanced design -> independent


def test_interpretation_bilingual(demo):
    d = ds.summary_table(demo, ["rainfall_mm"]).loc["rainfall_mm"].to_dict()
    assert "mean" in ds.interpret_variable("rain", d, "en")
    assert "المتوسط" in ds.interpret_variable("rain", d, "ar")


def test_worldbank_parse_and_panel():
    raw = json.loads("""[{"page":1},[
      {"indicator":{"id":"NY.GDP.MKTP.KD.ZG","value":"GDP growth"},"country":{"id":"EG","value":"Egypt, Arab Rep."},
       "countryiso3code":"EGY","date":"2022","value":6.6},
      {"indicator":{"id":"NY.GDP.MKTP.KD.ZG","value":"GDP growth"},"country":{"id":"EG","value":"Egypt, Arab Rep."},
       "countryiso3code":"EGY","date":"2021","value":3.3},
      {"indicator":{"id":"FP.CPI.TOTL.ZG","value":"Inflation"},"country":{"id":"EG","value":"Egypt, Arab Rep."},
       "countryiso3code":"EGY","date":"2022","value":null}]]""")
    long_df = wb.parse_records(raw[1]).dropna(subset=["value"])
    panel = wb.to_panel(long_df, {"NY.GDP.MKTP.KD.ZG": "GDP growth"})
    assert list(panel["year"]) == [2021, 2022]
    assert panel["GDP growth"].tolist() == [3.3, 6.6]


@pytest.mark.parametrize("header", ["DATE", "observation_date"])
def test_fred_parse(header):
    text = f"{header},UNRATE\n2020-01-01,3.6\n2020-02-01,.\n2020-03-01,4.4\n"
    df = fred.parse_fred_csv(text, "UNRATE")
    assert df["UNRATE"].isna().sum() == 1 and len(df) == 3
    q = fred.resample(df, "QS")
    assert np.isclose(q["UNRATE"].iloc[0], 4.0)


def test_merge_harmonises_keys():
    a = pd.DataFrame({"iso3": ["EGY", "SAU"], "year": [2020, 2020], "x": [1, 2]})
    b = pd.DataFrame({"code": ["EGY "], "year": [2020], "gdp": [9.9]})
    m = merge_datasets(a, b, ["iso3", "year"], ["code", "year"], "left")
    assert m.loc[m.iso3 == "EGY", "gdp"].item() == 9.9
    assert "code" not in m.columns and len(m) == 2


def test_extended_measures():
    x = pd.Series([1.0, 2, 2, 4, 8, 16])
    d = ds.describe_series(x)
    from scipy import stats as st_
    assert d["Mode"] == 2
    assert np.isclose(d["Geometric mean"], st_.gmean(x))
    assert np.isclose(d["Harmonic mean"], st_.hmean(x))
    assert np.isclose(d["MAD (median)"], (x - x.median()).abs().median())
    assert np.isclose(d["Mean abs. deviation"], (x - x.mean()).abs().mean())
    assert np.isnan(ds.describe_series(pd.Series([1.0, 2, 3]))["Mode"])      # no repeated value
    assert np.isnan(ds.describe_series(pd.Series([-1.0, 2, 3]))["Geometric mean"])


def test_gini_and_lorenz():
    assert np.isclose(ds.gini([5, 5, 5, 5]), 0)
    assert np.isclose(ds.gini([0, 0, 0, 1]), 0.75)
    lc = ds.lorenz_curve([1, 2, 3, 4])
    assert lc.iloc[0].tolist() == [0, 0] and lc.iloc[-1].tolist() == [1, 1]
    sh = ds.income_shares(np.ones(20))
    assert np.isclose(sh["Share top 10% (%)"], 10) and np.isclose(sh["Palma ratio"], 0.25)


def test_summary_groups_and_brief(demo):
    brief = ds.summary_table(demo, ["avg_temp_c"], brief=True)
    assert list(brief.columns) == ds.BRIEF_MEASURES
    cen = ds.summary_table(demo, ["avg_temp_c"], groups=["central"])
    assert list(cen.columns) == ds.MEASURE_GROUPS["central"]
    full = ds.summary_table(demo, ["avg_temp_c"])
    assert all(m in full.columns for g in ds.MEASURE_GROUPS.values() for m in g)
    assert set(ds.MEASURE_LABELS_AR) >= {m for g in ds.MEASURE_GROUPS.values() for m in g}
    pt = ds.percentile_table(demo, ["avg_temp_c"], "deciles")
    assert list(pt.columns)[0] == "P10" and pt.shape[1] == 9
    gg = ds.gini_by_group(demo, "production_t", "crop")
    assert len(gg) == 4 and gg["Gini"].between(0, 1).all()
