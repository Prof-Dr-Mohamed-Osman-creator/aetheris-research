# ==========================================
# Aetheris Research - FRED Data Engine
# ------------------------------------------
# Federal Reserve Economic Data (St. Louis Fed).
# Uses the public fredgraph CSV endpoint, so no API key is needed.
# ==========================================

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv"
TIMEOUT = 30

# Popular series: id -> (English, Arabic, frequency)
SERIES: dict[str, tuple[str, str, str]] = {
    "GDPC1": ("US Real GDP (bn chained 2017 $)", "الناتج المحلي الحقيقي الأمريكي", "Q"),
    "A191RL1Q225SBEA": ("US Real GDP growth (% SAAR)", "نمو الناتج الحقيقي الأمريكي (%)", "Q"),
    "CPIAUCSL": ("US CPI, all items (index)", "الرقم القياسي لأسعار المستهلك الأمريكي", "M"),
    "PCEPILFE": ("US Core PCE price index", "مؤشر أسعار نفقات الاستهلاك الأساسي", "M"),
    "UNRATE": ("US Unemployment rate (%)", "معدل البطالة الأمريكي (%)", "M"),
    "PAYEMS": ("US Nonfarm payrolls (thousands)", "الوظائف غير الزراعية الأمريكية (ألف)", "M"),
    "INDPRO": ("US Industrial production index", "الرقم القياسي للإنتاج الصناعي الأمريكي", "M"),
    "FEDFUNDS": ("Effective Federal Funds Rate (%)", "سعر الفائدة الفيدرالي الفعلي (%)", "M"),
    "DGS10": ("10-Year Treasury yield (%)", "عائد سندات الخزانة 10 سنوات (%)", "D"),
    "T10Y2Y": ("10Y-2Y Treasury spread (%)", "فرق العائد 10 سنوات - سنتين (%)", "D"),
    "M2SL": ("US M2 money stock (bn $)", "عرض النقود M2 الأمريكي", "M"),
    "DCOILWTICO": ("Crude oil, WTI ($/barrel)", "سعر خام غرب تكساس (دولار/برميل)", "D"),
    "DCOILBRENTEU": ("Crude oil, Brent ($/barrel)", "سعر خام برنت (دولار/برميل)", "D"),
    "DTWEXBGS": ("Nominal broad US dollar index", "مؤشر الدولار الأمريكي الواسع", "D"),
    "DEXUSEU": ("US$ per Euro", "سعر صرف اليورو مقابل الدولار", "D"),
    "PWHEAMTUSDM": ("Global price of wheat ($/mt)", "السعر العالمي للقمح (دولار/طن)", "M"),
    "PMAIZMTUSDM": ("Global price of maize ($/mt)", "السعر العالمي للذرة (دولار/طن)", "M"),
    "PBARLUSDM": ("Global price of barley ($/mt)", "السعر العالمي للشعير (دولار/طن)", "M"),
    "PSOYBUSDM": ("Global price of soybeans ($/mt)", "السعر العالمي لفول الصويا (دولار/طن)", "M"),
    "PFOODINDEXM": ("Global price index of all food", "الرقم القياسي العالمي لأسعار الغذاء", "M"),
    "UMCSENT": ("U. Michigan consumer sentiment", "مؤشر ثقة المستهلك - ميشيغان", "M"),
    "USREC": ("US recession indicator (NBER)", "مؤشر الركود الأمريكي", "M"),
}

FREQ_RULES = {"Original": None, "Monthly": "MS", "Quarterly": "QS", "Annual": "YS"}


class FredError(RuntimeError):
    pass


def parse_fred_csv(text: str, series_id: str) -> pd.DataFrame:
    """Parse a fredgraph CSV (handles both 'DATE' and 'observation_date' headers)."""
    df = pd.read_csv(StringIO(text))
    if df.shape[1] < 2:
        raise FredError(f"Unexpected response for {series_id}")
    date_col = df.columns[0]
    value_col = series_id if series_id in df.columns else df.columns[1]
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        series_id: pd.to_numeric(df[value_col], errors="coerce"),  # "." -> NaN
    }).dropna(subset=["date"])
    return out


def fetch_series(series_id: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    params = {"id": series_id}
    if start:
        params["cosd"] = start
    if end:
        params["coed"] = end
    try:
        resp = requests.get(FRED_CSV, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise FredError(f"FRED request failed for {series_id}: {exc}") from exc
    if "<html" in resp.text[:200].lower():
        raise FredError(f"Series '{series_id}' was not found on FRED")
    return parse_fred_csv(resp.text, series_id)


def resample(df: pd.DataFrame, rule: str | None, how: str = "mean") -> pd.DataFrame:
    if not rule:
        return df
    out = df.set_index("date").resample(rule)
    out = out.last() if how == "last" else out.mean()
    return out.reset_index()


def fetch_many(series_ids: list[str], start: str | None, end: str | None,
               frequency: str = "Original", how: str = "mean") -> tuple[pd.DataFrame, list[str]]:
    """Fetch several series, align them on date. Returns (data, failed ids)."""
    merged, failed = None, []
    rule = FREQ_RULES.get(frequency)
    for sid in series_ids:
        try:
            s = resample(fetch_series(sid, start, end), rule, how)
        except FredError:
            failed.append(sid)
            continue
        merged = s if merged is None else merged.merge(s, on="date", how="outer")
    if merged is None:
        return pd.DataFrame(), failed
    return merged.sort_values("date").reset_index(drop=True), failed
