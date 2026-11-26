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
