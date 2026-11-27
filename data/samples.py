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
