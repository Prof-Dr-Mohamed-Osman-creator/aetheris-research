# ==========================================
# Aetheris Research - World Bank Data Engine
# ------------------------------------------
# Uses the public World Bank Indicators API v2 (no key required).
# https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
# ==========================================

from __future__ import annotations

import pandas as pd
import requests

WB_BASE = "https://api.worldbank.org/v2"
TIMEOUT = 30

# Curated indicator catalog: code -> (English label, Arabic label, theme)
INDICATORS: dict[str, tuple[str, str, str]] = {
    # Growth & output
    "NY.GDP.MKTP.CD": ("GDP (current US$)", "الناتج المحلي الإجمالي (دولار جاري)", "growth"),
    "NY.GDP.MKTP.KD": ("GDP (constant 2015 US$)", "الناتج المحلي الإجمالي (دولار ثابت 2015)", "growth"),
    "NY.GDP.MKTP.KD.ZG": ("GDP growth (annual %)", "معدل نمو الناتج المحلي (%)", "growth"),
    "NY.GDP.PCAP.CD": ("GDP per capita (current US$)", "نصيب الفرد من الناتج (دولار جاري)", "growth"),
    "NY.GDP.PCAP.KD.ZG": ("GDP per capita growth (annual %)", "نمو نصيب الفرد من الناتج (%)", "growth"),
    "NE.GDI.TOTL.ZS": ("Gross capital formation (% of GDP)", "تكوين رأس المال الإجمالي (% من الناتج)", "growth"),
    "NY.GNS.ICTR.ZS": ("Gross savings (% of GDP)", "الادخار الإجمالي (% من الناتج)", "growth"),
    # Prices & money
    "FP.CPI.TOTL.ZG": ("Inflation, consumer prices (annual %)", "التضخم - أسعار المستهلك (%)", "prices"),
    "NY.GDP.DEFL.KD.ZG": ("Inflation, GDP deflator (annual %)", "التضخم - مخفض الناتج (%)", "prices"),
    "FM.LBL.BMNY.GD.ZS": ("Broad money (% of GDP)", "عرض النقود بالمعنى الواسع (% من الناتج)", "prices"),
    "FR.INR.LEND": ("Lending interest rate (%)", "سعر الفائدة على الإقراض (%)", "prices"),
    "FR.INR.RINR": ("Real interest rate (%)", "سعر الفائدة الحقيقي (%)", "prices"),
    "PA.NUS.FCRF": ("Official exchange rate (LCU per US$)", "سعر الصرف الرسمي (عملة محلية لكل دولار)", "prices"),
    # External sector
    "NE.EXP.GNFS.ZS": ("Exports of goods & services (% of GDP)", "الصادرات من السلع والخدمات (% من الناتج)", "external"),
    "NE.IMP.GNFS.ZS": ("Imports of goods & services (% of GDP)", "الواردات من السلع والخدمات (% من الناتج)", "external"),
    "NE.TRD.GNFS.ZS": ("Trade (% of GDP)", "الانفتاح التجاري (% من الناتج)", "external"),
    "BX.KLT.DINV.WD.GD.ZS": ("FDI, net inflows (% of GDP)", "صافي تدفقات الاستثمار الأجنبي المباشر (% من الناتج)", "external"),
    "BN.CAB.XOKA.GD.ZS": ("Current account balance (% of GDP)", "رصيد الحساب الجاري (% من الناتج)", "external"),
    "BX.TRF.PWKR.DT.GD.ZS": ("Personal remittances received (% of GDP)", "تحويلات العاملين (% من الناتج)", "external"),
    "DT.DOD.DECT.GN.ZS": ("External debt stocks (% of GNI)", "رصيد الدين الخارجي (% من الدخل القومي)", "external"),
    # Fiscal
    "GC.DOD.TOTL.GD.ZS": ("Central government debt (% of GDP)", "دين الحكومة المركزية (% من الناتج)", "fiscal"),
    "GC.REV.XGRT.GD.ZS": ("Revenue excl. grants (% of GDP)", "الإيرادات العامة بدون المنح (% من الناتج)", "fiscal"),
    "GC.XPN.TOTL.GD.ZS": ("Government expense (% of GDP)", "الإنفاق الحكومي (% من الناتج)", "fiscal"),
    # Labour & social
    "SL.UEM.TOTL.ZS": ("Unemployment (% of labor force, ILO)", "معدل البطالة (% من قوة العمل)", "labour"),
    "SL.TLF.CACT.ZS": ("Labor force participation rate (%)", "معدل المشاركة في قوة العمل (%)", "labour"),
    "SP.POP.TOTL": ("Population, total", "إجمالي السكان", "labour"),
    "SP.POP.GROW": ("Population growth (annual %)", "معدل النمو السكاني (%)", "labour"),
    "SP.URB.TOTL.IN.ZS": ("Urban population (% of total)", "سكان الحضر (% من الإجمالي)", "labour"),
    "SI.POV.GINI": ("Gini index", "مؤشر جيني", "labour"),
    "SI.POV.DDAY": ("Poverty headcount at $2.15/day (%)", "نسبة الفقر عند 2.15 دولار يومياً (%)", "labour"),
    "SP.DYN.LE00.IN": ("Life expectancy at birth (years)", "العمر المتوقع عند الولادة (سنوات)", "labour"),
    "SE.TER.ENRR": ("School enrollment, tertiary (% gross)", "الالتحاق بالتعليم العالي (%)", "labour"),
    # Agriculture, food & environment
    "NV.AGR.TOTL.ZS": ("Agriculture value added (% of GDP)", "القيمة المضافة للزراعة (% من الناتج)", "agri"),
    "SL.AGR.EMPL.ZS": ("Employment in agriculture (% of total)", "العمالة في الزراعة (% من الإجمالي)", "agri"),
    "AG.YLD.CREL.KG": ("Cereal yield (kg per hectare)", "إنتاجية الحبوب (كجم/هكتار)", "agri"),
    "AG.PRD.FOOD.XD": ("Food production index (2014-2016=100)", "الرقم القياسي لإنتاج الغذاء", "agri"),
    "AG.LND.ARBL.ZS": ("Arable land (% of land area)", "الأراضي الصالحة للزراعة (% من المساحة)", "agri"),
    "AG.LND.IRIG.AG.ZS": ("Agricultural irrigated land (% of agri land)", "الأراضي الزراعية المروية (%)", "agri"),
    "ER.H2O.FWTL.ZS": ("Freshwater withdrawal (% of internal resources)", "سحب المياه العذبة (% من الموارد الداخلية)", "agri"),
    "EN.GHG.CO2.PC.CE.AR5": ("CO2 emissions per capita (t CO2e)", "انبعاثات ثاني أكسيد الكربون للفرد (طن)", "agri"),
    "EG.FEC.RNEW.ZS": ("Renewable energy consumption (% of total)", "استهلاك الطاقة المتجددة (% من الإجمالي)", "agri"),
    # Institutions / digital
    "IT.NET.USER.ZS": ("Individuals using the Internet (% of pop.)", "مستخدمو الإنترنت (% من السكان)", "other"),
    "GB.XPD.RSDV.GD.ZS": ("R&D expenditure (% of GDP)", "الإنفاق على البحث والتطوير (% من الناتج)", "other"),
}

THEMES = {
    "growth": ("Growth & output", "النمو والإنتاج"),
    "prices": ("Prices, money & interest", "الأسعار والنقود والفائدة"),
    "external": ("External sector", "القطاع الخارجي"),
    "fiscal": ("Public finance", "المالية العامة"),
    "labour": ("Labour, population & welfare", "العمل والسكان والرفاه"),
    "agri": ("Agriculture, food & environment", "الزراعة والغذاء والبيئة"),
    "other": ("Technology & innovation", "التكنولوجيا والابتكار"),
}

# Curated economies: ISO3 -> (English, Arabic)
COUNTRIES: dict[str, tuple[str, str]] = {
    # Arab world
    "EGY": ("Egypt", "مصر"), "SAU": ("Saudi Arabia", "السعودية"), "ARE": ("United Arab Emirates", "الإمارات"),
    "QAT": ("Qatar", "قطر"), "KWT": ("Kuwait", "الكويت"), "BHR": ("Bahrain", "البحرين"), "OMN": ("Oman", "عُمان"),
    "JOR": ("Jordan", "الأردن"), "LBN": ("Lebanon", "لبنان"), "IRQ": ("Iraq", "العراق"), "SYR": ("Syria", "سوريا"),
    "YEM": ("Yemen", "اليمن"), "PSE": ("West Bank and Gaza", "فلسطين"), "SDN": ("Sudan", "السودان"),
    "LBY": ("Libya", "ليبيا"), "TUN": ("Tunisia", "تونس"), "DZA": ("Algeria", "الجزائر"), "MAR": ("Morocco", "المغرب"),
    "MRT": ("Mauritania", "موريتانيا"),
    # Major & emerging economies
    "USA": ("United States", "الولايات المتحدة"), "CHN": ("China", "الصين"), "JPN": ("Japan", "اليابان"),
    "DEU": ("Germany", "ألمانيا"), "GBR": ("United Kingdom", "المملكة المتحدة"), "FRA": ("France", "فرنسا"),
    "ITA": ("Italy", "إيطاليا"), "CAN": ("Canada", "كندا"), "IND": ("India", "الهند"), "BRA": ("Brazil", "البرازيل"),
    "RUS": ("Russian Federation", "روسيا"), "KOR": ("Korea, Rep.", "كوريا الجنوبية"), "MEX": ("Mexico", "المكسيك"),
    "IDN": ("Indonesia", "إندونيسيا"), "TUR": ("Türkiye", "تركيا"), "IRN": ("Iran", "إيران"),
    "PAK": ("Pakistan", "باكستان"), "BGD": ("Bangladesh", "بنغلاديش"), "MYS": ("Malaysia", "ماليزيا"),
    "ZAF": ("South Africa", "جنوب أفريقيا"), "NGA": ("Nigeria", "نيجيريا"), "ETH": ("Ethiopia", "إثيوبيا"),
    "KEN": ("Kenya", "كينيا"), "AUS": ("Australia", "أستراليا"), "ESP": ("Spain", "إسبانيا"),
    # Aggregates
    "WLD": ("World", "العالم"), "ARB": ("Arab World", "العالم العربي"),
    "MEA": ("Middle East & North Africa", "الشرق الأوسط وشمال أفريقيا"),
    "SSF": ("Sub-Saharan Africa", "أفريقيا جنوب الصحراء"), "EUU": ("European Union", "الاتحاد الأوروبي"),
    "HIC": ("High income", "الدول مرتفعة الدخل"), "LMC": ("Lower middle income", "الدول متوسطة الدخل الأدنى"),
    "LIC": ("Low income", "الدول منخفضة الدخل"),
}


class WorldBankError(RuntimeError):
    pass


def _get_json(url: str, params: dict) -> list:
    try:
        resp = requests.get(url, params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, ValueError) as exc:
        raise WorldBankError(f"World Bank API request failed: {exc}") from exc
    # Errors come back as [{"message": [...]}]
    if isinstance(payload, list) and payload and isinstance(payload[0], dict) and "message" in payload[0]:
        raise WorldBankError(str(payload[0]["message"]))
    if not isinstance(payload, list) or len(payload) < 2:
        return []
    return payload[1] or []


def parse_records(records: list) -> pd.DataFrame:
    """Convert raw World Bank API records to a tidy long DataFrame."""
    rows = []
    for r in records:
        rows.append({
            "country": (r.get("country") or {}).get("value"),
            "iso3": r.get("countryiso3code") or (r.get("country") or {}).get("id"),
            "year": r.get("date"),
            "indicator": (r.get("indicator") or {}).get("id"),
            "value": r.get("value"),
        })
    df = pd.DataFrame(rows, columns=["country", "iso3", "year", "indicator", "value"])
    if df.empty:
        return df
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df


def fetch_indicator(indicator: str, countries: list[str], start: int, end: int) -> pd.DataFrame:
    """Fetch one indicator for many countries (long format)."""
    if not countries:
        return pd.DataFrame(columns=["country", "iso3", "year", "indicator", "value"])
    url = f"{WB_BASE}/country/{';'.join(countries)}/indicator/{indicator}"
    params = {"format": "json", "date": f"{start}:{end}", "per_page": 20000}
    return parse_records(_get_json(url, params))


def to_panel(long_df: pd.DataFrame, labels: dict[str, str] | None = None) -> pd.DataFrame:
    """Pivot long data to a country-year panel with one column per indicator."""
    if long_df.empty:
        return long_df
    panel = (long_df.pivot_table(index=["country", "iso3", "year"], columns="indicator",
                                 values="value", aggfunc="first")
             .reset_index())
    panel.columns.name = None
    if labels:
        panel = panel.rename(columns=labels)
    panel = panel.sort_values(["country", "year"]).reset_index(drop=True)
    panel["year"] = panel["year"].astype(int)
    return panel


def fetch_panel(indicators: list[str], countries: list[str], start: int, end: int,
                labels: dict[str, str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Fetch several indicators and return (panel, list of indicators that returned no data)."""
    frames, empty = [], []
    for code in indicators:
        df = fetch_indicator(code, countries, start, end)
        if df.empty or df["value"].notna().sum() == 0:
            empty.append(code)
        frames.append(df)
    long_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    long_df = long_df.dropna(subset=["value"]) if not long_df.empty else long_df
    return to_panel(long_df, labels), empty


def search_indicators(query: str, max_results: int = 50) -> pd.DataFrame:
    """Search the full World Bank catalog (WDI source) by keyword."""
    url = f"{WB_BASE}/source/2/indicator"
    records = _get_json(url, {"format": "json", "per_page": 20000})
    q = query.lower().strip()
    hits = [{"code": r.get("id"), "name": r.get("name")}
            for r in records if q in (r.get("name") or "").lower() or q in (r.get("id") or "").lower()]
    return pd.DataFrame(hits[:max_results], columns=["code", "name"])
