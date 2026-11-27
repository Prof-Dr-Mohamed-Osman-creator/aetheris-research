# ==========================================
# Aetheris Research - Demo datasets
# ------------------------------------------
# SYNTHETIC data generated with a fixed seed so visitors can try every
# wing instantly. Clearly labelled: it must never be cited as real data.
# ==========================================

import numpy as np
import pandas as pd


def climate_crop_demo(seed: int = 42) -> pd.DataFrame:
    """Synthetic governorate-year-crop panel: climate drivers and yields."""
    rng = np.random.default_rng(seed)
    governorates = ["Matrouh", "New Valley", "Red Sea", "North Sinai",
                    "South Sinai", "Giza (desert)", "Minya (desert)", "Aswan (desert)"]
    crops = {"Wheat": (2.7, 0.012), "Barley": (1.6, 0.008), "Maize": (3.3, 0.010), "Faba bean": (1.4, 0.006)}
    rows = []
    for g_i, g in enumerate(governorates):
        base_t = 21 + g_i * 0.6
        for year in range(2000, 2024):
            t = base_t + 0.035 * (year - 2000) + rng.normal(0, 0.45)
            rain = max(0.0, rng.gamma(2.0, 20 if g_i < 2 else 6))
            for crop, (y0, trend) in crops.items():
                heat_penalty = -0.07 * max(0.0, t - 24)
                yld = y0 * (1 + trend) ** (year - 2000) * (1 + heat_penalty) * (1 + 0.0009 * rain)
                yld *= np.exp(rng.normal(0, 0.08))
                area = rng.lognormal(mean=8.5 - g_i * 0.15, sigma=0.3)
                rows.append({"governorate": g, "year": year, "crop": crop,
                             "avg_temp_c": round(t, 2), "rainfall_mm": round(rain, 1),
                             "area_feddan": round(area), "yield_t_per_feddan": round(yld, 3),
                             "production_t": round(area * yld)})
    return pd.DataFrame(rows)


def farm_survey_demo(seed: int = 7, n: int = 320) -> pd.DataFrame:
    """Synthetic cross-section of farms for regression training: Cobb-Douglas technology with
    heteroskedastic errors, an irrigation factor, education and credit effects."""
    rng = np.random.default_rng(seed)
    regions = np.array(["Delta", "Middle Egypt", "Upper Egypt", "New Lands"])
    irrigation = np.array(["Canal", "Groundwater", "Drip"])
    region = rng.choice(regions, n, p=[0.35, 0.25, 0.25, 0.15])
    irr = rng.choice(irrigation, n, p=[0.55, 0.30, 0.15])
    land = np.round(rng.lognormal(1.3, 0.6, n), 2)                          # feddan
    labour = np.round(land * rng.lognormal(3.4, 0.35, n))                    # man-days
    fert = np.round(land * rng.lognormal(4.6, 0.30, n))                      # kg N
    educ = rng.integers(0, 17, n)
    exper = rng.integers(2, 45, n)
    credit = rng.choice(["No", "Yes"], n, p=[0.6, 0.4])
    irr_eff = pd.Series(irr).map({"Canal": 0.0, "Groundwater": 0.06, "Drip": 0.18}).to_numpy()
    reg_eff = pd.Series(region).map({"Delta": 0.10, "Middle Egypt": 0.03, "Upper Egypt": -0.05, "New Lands": -0.08}).to_numpy()
    lnq = (1.2 + 0.52 * np.log(land) + 0.23 * np.log(labour) + 0.19 * np.log(fert) + irr_eff + reg_eff
           + 0.012 * educ + 0.006 * exper - 0.0001 * exper ** 2 + 0.05 * (credit == "Yes")
           + rng.normal(0, 0.12 + 0.04 * np.log1p(land), n))
    output = np.round(np.exp(lnq), 2)                                         # tonnes
    price = np.round(rng.normal(9.5, 0.8, n), 2)                              # EGP '000 per tonne
    return pd.DataFrame({"farm_id": np.arange(1, n + 1), "region": region, "irrigation": irr,
                         "land_feddan": land, "labour_days": labour, "fertilizer_kg": fert,
                         "education_years": educ, "experience_years": exper, "credit_access": credit,
                         "output_tonnes": output, "farm_price": price,
                         "revenue_k_egp": np.round(output * price, 1)})


def household_survey_demo(seed: int = 15, n: int = 1500) -> pd.DataFrame:
    """Synthetic farm-household survey for limited-dependent-variable, count and survival models:
    technology adoption (binary), technology choice (multinomial), satisfaction (ordered), fertilizer
    spending with zeros (Tobit / hurdle), off-farm wage observed only for participants (Heckman),
    share of output sold (fractional), extension visits (count with excess zeros) and years until
    adoption with right-censoring (survival)."""
    rng = np.random.default_rng(seed)
    region = rng.choice(["Delta", "Middle Egypt", "Upper Egypt", "New Lands"], n, p=[0.35, 0.25, 0.25, 0.15])
    female_head = rng.binomial(1, 0.18, n)
    educ = np.clip(np.round(rng.normal(7, 4, n)), 0, 18)
    age = np.clip(np.round(rng.normal(46, 11, n)), 20, 80)
    farm_size = np.round(rng.lognormal(1.0, 0.7, n), 2)
    dist_market = np.round(rng.gamma(3, 4, n), 1)
    credit = rng.binomial(1, 1 / (1 + np.exp(-(-1 + 0.08 * educ + 0.3 * np.log(farm_size)))), n)
    hh_size = rng.integers(2, 12, n)
    n_children = np.clip(hh_size - 2 - rng.integers(0, 3, n), 0, None)
    reg_eff = pd.Series(region).map({"Delta": 0.3, "Middle Egypt": 0.1, "Upper Egypt": -0.2, "New Lands": 0.0}).to_numpy()
    # extension visits: zero-inflated negative binomial
    structural_zero = rng.random(n) < 1 / (1 + np.exp(-(0.2 + 0.06 * dist_market - 0.1 * educ)))
    mu = np.exp(0.2 + 0.05 * educ + 0.25 * np.log(farm_size) - 0.03 * dist_market + 0.4 * credit + reg_eff)
    visits = np.where(structural_zero, 0, rng.negative_binomial(2, 2 / (2 + mu)))
    # adoption (probit latent index)
    xb = -1.2 + 0.09 * educ + 0.35 * np.log(farm_size) + 0.5 * credit + 0.12 * np.minimum(visits, 8) \
        - 0.03 * dist_market - 0.3 * female_head + reg_eff
    adopt = (xb + rng.normal(size=n) > 0).astype(int)
    # technology choice (multinomial logit utilities)
    u_seed = 0.2 + 0.05 * educ + 0.3 * credit - 0.02 * dist_market + rng.gumbel(size=n)
    u_drip = -1.5 + 0.08 * educ + 0.6 * np.log(farm_size) + 0.7 * credit + 0.3 * (region == "New Lands") + rng.gumbel(size=n)
    u_trad = rng.gumbel(size=n)
    choice = np.array(["Traditional", "Improved seed", "Drip irrigation"])[np.argmax(np.c_[u_trad, u_seed, u_drip], axis=1)]
    # satisfaction with extension service (ordered logit, 1–5)
    s_star = 0.25 * np.minimum(visits, 8) + 0.05 * educ - 0.02 * dist_market + 0.4 * adopt + rng.logistic(size=n)
    satisfaction = np.digitize(s_star, [-0.5, 0.6, 1.6, 2.8]) + 1
    # fertilizer spending: Tobit, censored at zero
    f_star = -300 + 180 * np.log(farm_size) + 25 * educ + 220 * credit - 6 * dist_market + 350 * adopt + rng.normal(0, 260, n)
    fert_spend = np.round(np.maximum(0, f_star), 1)
    # off-farm work (selection) and wage (outcome); n_children is the exclusion restriction
    u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n).T
    works = (0.3 + 0.04 * educ - 0.02 * (age - 45) - 0.25 * n_children - 0.2 * np.log(farm_size) + 0.2 * female_head + v > 0)
    ln_wage = 2.0 + 0.07 * educ + 0.02 * (age - 20) - 0.0003 * (age - 20) ** 2 - 0.15 * female_head + 0.35 * u
    offfarm_wage = np.where(works, np.round(np.exp(ln_wage), 2), np.nan)
    # share of output sold (fractional)
    share = 1 / (1 + np.exp(-(-0.5 + 0.4 * np.log(farm_size) - 0.04 * dist_market + 0.3 * credit + 0.05 * educ
                              + rng.normal(0, 0.6, n))))
    share_sold = np.round(np.clip(np.where(rng.random(n) < 0.05, 0, share), 0, 1), 3)
    # years until adoption (Weibull PH), censored at 10 years of observation
    lam = np.exp(-3.3 + 0.08 * educ + 0.45 * credit + 0.25 * np.log(farm_size) - 0.02 * dist_market)
    t = (-np.log(rng.random(n)) / lam) ** (1 / 1.3)
    years = np.round(np.minimum(t, 10), 2)
    adopted_event = (t <= 10).astype(int)
    return pd.DataFrame({
        "hh_id": np.arange(1, n + 1), "region": region, "female_head": female_head, "education_years": educ,
        "age": age, "farm_size_feddan": farm_size, "distance_market_km": dist_market, "credit_access": credit,
        "household_size": hh_size, "n_children": n_children, "extension_visits": visits, "adopt_technology": adopt,
        "technology_choice": choice, "satisfaction_1to5": satisfaction, "fertilizer_spend_egp": fert_spend,
        "works_offfarm": works.astype(int), "offfarm_wage_egp_day": offfarm_wage, "share_sold": share_sold,
        "years_to_adoption": years, "adopted_by_end": adopted_event})


def labour_iv_demo(seed: int = 21, n: int = 1200) -> pd.DataFrame:
    """Synthetic wage data where education is endogenous (unobserved ability) and distance to college and
    parents' schooling are instruments — for IV / 2SLS / LIML / GMM training."""
    rng = np.random.default_rng(seed)
    ability = rng.normal(size=n)
    dist_college = np.round(rng.gamma(2, 15, n), 1)
    father_educ = np.clip(np.round(rng.normal(8, 3.5, n)), 0, 20)
    mother_educ = np.clip(np.round(0.5 * father_educ + rng.normal(4, 3, n)), 0, 20)
    female = rng.binomial(1, 0.45, n)
    urban = rng.binomial(1, 0.55, n)
    exper = np.clip(np.round(rng.normal(15, 8, n)), 0, 40)
    educ = np.clip(np.round(8 + 0.25 * father_educ + 0.15 * mother_educ - 0.04 * dist_college + 1.2 * ability
                            + 0.8 * urban + rng.normal(0, 1.5, n)), 4, 22)
    ln_wage = 1.0 + 0.075 * educ + 0.035 * exper - 0.0006 * exper ** 2 - 0.18 * female + 0.10 * urban \
        + 0.12 * ability + rng.normal(0, 0.35 + 0.01 * exper, n)
    return pd.DataFrame({"person_id": np.arange(1, n + 1), "wage_egp_hour": np.round(np.exp(ln_wage), 2),
                         "education_years": educ, "experience_years": exper, "female": female, "urban": urban,
                         "distance_college_km": dist_college, "father_education": father_educ,
                         "mother_education": mother_educ, "region": rng.choice(["North", "South", "East", "West"], n)})


def market_sem_demo(seed: int = 3, n: int = 60) -> pd.DataFrame:
    """Synthetic annual market for a crop: demand and supply determined simultaneously.
    Demand: Q = 120 − 2.0 P + 0.8 Income + u;  Supply: Q = 20 + 1.5 P + 0.4 Rainfall − 1.2 InputCost + v."""
    rng = np.random.default_rng(seed)
    year = np.arange(2026 - n, 2026)
    income = 50 + 0.8 * np.arange(n) + rng.normal(0, 4, n)
    rain = rng.normal(100, 20, n)
    cost = 20 + 0.2 * np.arange(n) + rng.normal(0, 3, n)
    u, v = rng.normal(0, 6, n), rng.normal(0, 5, n)
    P = (120 + 0.8 * income + u - 20 - 0.4 * rain + 1.2 * cost - v) / (2.0 + 1.5)
    Q = 20 + 1.5 * P + 0.4 * rain - 1.2 * cost + v
    return pd.DataFrame({"year": year, "quantity": np.round(Q, 2), "price": np.round(P, 3),
                         "income": np.round(income, 2), "rainfall": np.round(rain, 1), "input_cost": np.round(cost, 2)})


def macro_ts_demo(seed: int = 11) -> pd.DataFrame:
    """Synthetic monthly series (Jan 2000 – Dec 2025) for time-series training: a seasonal I(1) price index
    with drift, an ARIMA(1,1,1) commodity price, a stationary AR(2) unemployment rate, a seasonal
    stationary rainfall series and a long-memory (d ≈ 0.3) river-flow index."""
    rng = np.random.default_rng(seed)
    n = 312
    date = pd.date_range("2000-01-01", periods=n, freq="MS")
    month = date.month.to_numpy()
    seas = 0.012 * np.sin(2 * np.pi * month / 12) + 0.006 * np.cos(2 * np.pi * month / 12)
    lcpi = 4.6 + np.cumsum(0.006 + rng.normal(0, 0.008, n)) + seas
    e = rng.normal(0, 4, n + 1)
    dp = np.zeros(n)
    for t in range(1, n):
        dp[t] = 0.5 * dp[t - 1] + e[t] + 0.3 * e[t - 1]
    wheat = 3000 + np.cumsum(dp)
    u = np.zeros(n)
    for t in range(2, n):
        u[t] = 1.1 * u[t - 1] - 0.3 * u[t - 2] + rng.normal(0, 0.25)
    unemp = 9 + u
    rain = np.maximum(0, 20 + 15 * np.cos(2 * np.pi * (month - 1) / 12) + rng.normal(0, 5, n))
    d = 0.3
    w = np.empty(600)
    w[0] = 1.0
    for k in range(1, 600):
        w[k] = w[k - 1] * (k - 1 + d) / k
    eps = rng.normal(size=n + 600)
    flow = np.array([w @ eps[t:t + 600][::-1] for t in range(n)])
    return pd.DataFrame({"date": date, "cpi_index": np.round(np.exp(lcpi), 2), "wheat_price_egp_t": np.round(wheat, 1),
                         "unemployment_rate": np.round(unemp, 2), "rainfall_mm": np.round(rain, 1),
                         "nile_flow_index": np.round(100 + 5 * flow, 2)})


def quarterly_demo(seed: int = 5) -> pd.DataFrame:
    """Synthetic quarterly agricultural output (1990Q1–2025Q4) with seasonal unit roots — for the HEGY test."""
    rng = np.random.default_rng(seed)
    n = 144
    x = np.zeros(n + 4)
    for t in range(4, n + 4):
        x[t] = x[t - 4] + 0.5 + rng.normal(0, 1.0)
    return pd.DataFrame({"quarter": pd.date_range("1990-01-01", periods=n, freq="QS"),
                         "ag_output_index": np.round(100 + x[4:], 2)})
