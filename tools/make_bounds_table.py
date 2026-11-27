"""Simulate the asymptotic critical values of the PSS (2001) bounds F and t tests (T = 1000, 40 000 replications per
case and k = 0 … 10, the design of Pesaran, Shin & Smith 2001 and R ARDL) and write data/bounds_cv.csv.
Run once:  python tools/make_bounds_table.py   (≈ 1–2 hours on one core)."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analytics import ardl as ad  # noqa: E402

REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
rows = []
t0 = time.time()
for case in range(1, 6):
    for k in range(0, 11):
        sims = ad._bounds_stats(case, k, 1000, REPS, seed=2001 + 97 * case + k)
        q = ad._quantiles(sims)
        for j, a in enumerate(ad.ALPHAS):
            rows.append({"case": case, "k": k, "alpha": round(float(a), 3), "F0": q["F0"][j], "F1": q["F1"][j],
                         "t0": q["t0"][j], "t1": q["t1"][j]})
        print(case, k, f"{time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).round(5).to_csv(Path(__file__).resolve().parent.parent / "data" / "bounds_cv.csv", index=False)
