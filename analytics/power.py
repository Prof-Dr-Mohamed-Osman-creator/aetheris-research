# ==========================================
# Aetheris Research - Power analysis & sample size
# ------------------------------------------
# Exact power from non-central t / F distributions (as in G*Power),
# normal approximations for proportions and correlation, and survey
# sample-size formulas (Cochran, Yamane) used in field studies.
# ==========================================

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, stats


def power_t(d: float, n: float, alpha: float = 0.05, kind: str = "two", alternative: str = "two-sided",
            ratio: float = 1.0) -> float:
    """Power of a t-test. kind: 'two' (independent, n = size of group 1, n2 = ratio·n), 'one' or 'paired'."""
    if kind == "two":
        n1, n2 = n, n * ratio
        df = n1 + n2 - 2
        ncp = d * np.sqrt(n1 * n2 / (n1 + n2))
    else:
        df = n - 1
        ncp = d * np.sqrt(n)
    if alternative == "two-sided":
        tc = stats.t.ppf(1 - alpha / 2, df)
        return float(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))
    tc = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(tc, df, abs(ncp)))


def power_anova(f: float, k: int, n_per_group: float, alpha: float = 0.05) -> float:
    N = k * n_per_group
    df1, df2 = k - 1, N - k
    fc = stats.f.ppf(1 - alpha, df1, df2)
    return float(stats.ncf.sf(fc, df1, df2, f ** 2 * N))


def power_correlation(r: float, n: float, alpha: float = 0.05, alternative: str = "two-sided") -> float:
    z = np.arctanh(r) * np.sqrt(n - 3)
    if alternative == "two-sided":
        zc = stats.norm.ppf(1 - alpha / 2)
        return float(stats.norm.sf(zc - abs(z)) + stats.norm.cdf(-zc - abs(z)))
    return float(stats.norm.sf(stats.norm.ppf(1 - alpha) - abs(z)))


def power_two_proportions(p1: float, p2: float, n: float, alpha: float = 0.05,
                          alternative: str = "two-sided") -> float:
    h = abs(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))
    z = h * np.sqrt(n / 2)
    zc = stats.norm.ppf(1 - alpha / 2) if alternative == "two-sided" else stats.norm.ppf(1 - alpha)
    return float(stats.norm.sf(zc - z))


def power_chi2(w: float, N: float, df: int, alpha: float = 0.05) -> float:
    crit = stats.chi2.ppf(1 - alpha, df)
    return float(stats.ncx2.sf(crit, df, w ** 2 * N))


def solve_n(power_fn, target: float = 0.80, start: float = 3.0, max_n: float = 1e7) -> int:
    """Smallest integer n reaching the target power (bracket by doubling, then bisection)."""
    def pw(n):
        v = power_fn(n)
        return 1.0 if np.isnan(v) else v  # NaN only appears when the non-centrality is huge (power ≈ 1)

    lo, hi = start, start
    while pw(hi) < target:
        lo, hi = hi, hi * 2
        if hi > max_n:
            return -1
    if pw(lo) >= target:
        return math.ceil(lo)
    n = optimize.brentq(lambda v: pw(v) - target, lo, hi)
    n = math.ceil(n - 1e-9)
    while pw(n) < target:
        n += 1
    while n > start and pw(n - 1) >= target:
        n -= 1
    return n


def minimum_detectable_effect(power_fn_of_effect, target: float = 0.80, lo: float = 1e-4, hi: float = 5.0) -> float:
    return float(optimize.brentq(lambda e: power_fn_of_effect(e) - target, lo, hi))


# ---------------- survey sample size ----------------
def cochran_sample_size(p: float = 0.5, margin: float = 0.05, conf: float = 0.95,
                        population: int | None = None, design_effect: float = 1.0,
                        response_rate: float = 1.0) -> dict:
    """Cochran (1977) n₀ = z²p(1−p)/e², finite-population correction, design effect and non-response."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    n0 = z ** 2 * p * (1 - p) / margin ** 2
    n = n0 / (1 + (n0 - 1) / population) if population else n0
    n_deff = n * design_effect
    return {"z": z, "n0": n0, "n (after FPC)": n, "n (× design effect)": n_deff,
            "n to contact (non-response)": math.ceil(n_deff / response_rate),
            "Recommended n": math.ceil(n_deff)}


def yamane_sample_size(population: int, margin: float = 0.05) -> int:
    """Yamane (1967): n = N / (1 + N·e²) — common in social & agricultural surveys (assumes p = 0.5, 95%)."""
    return math.ceil(population / (1 + population * margin ** 2))


def mean_sample_size(sd: float, margin: float, conf: float = 0.95, population: int | None = None) -> int:
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    n0 = (z * sd / margin) ** 2
    n = n0 / (1 + (n0 - 1) / population) if population else n0
    return math.ceil(n)
