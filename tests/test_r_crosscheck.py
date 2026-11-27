"""Cross-validation against R (lm, sandwich, lmtest, MASS): the R code that Aetheris exports is executed and
must reproduce the platform's estimates, standard errors and diagnostic statistics. Skipped if R is absent."""
import shutil
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import diagnostics as dx
from analytics import reg_report as rp
from analytics import regression as rg
from data.samples import climate_crop_demo, farm_survey_demo

warnings.filterwarnings("ignore")
R_OK = shutil.which("Rscript") is not None and subprocess.run(
    ["Rscript", "-e", "for (p in c('lmtest','sandwich','car','tseries','strucchange','MASS')) library(p, character.only=TRUE)"],
    capture_output=True).returncode == 0 if shutil.which("Rscript") else False
pytestmark = pytest.mark.skipif(not R_OK, reason="R with lmtest/sandwich/car/tseries/strucchange/MASS not installed")

FULL = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days", "fertilizer_kg", "experience_years"],
                    categorical=["irrigation"], form="loglog", nolog=["experience_years"], poly={"experience_years": 2},
                    interactions=[("land_feddan", "irrigation")], center=True, reference={"irrigation": "Drip"})
SMALL = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days"], form="loglog")


def _run_r(r, df, tmp_path, tail):
    f = tmp_path / "d.csv"
    df.to_csv(f, index=False)
    code = rp.code_r(r, str(f)) + tail
    (tmp_path / "m.R").write_text(code)
    p = subprocess.run(["Rscript", str(tmp_path / "m.R")], capture_output=True, text=True, cwd=tmp_path)
    assert p.returncode == 0, p.stderr[-800:]
    return pd.read_csv(tmp_path / "out.csv")


COEF = '\nct <- coeftest(m, vcov. = V)\nwrite.csv(data.frame(b = ct[, 1], se = ct[, 2]), "out.csv", row.names = FALSE)\n'


@pytest.mark.parametrize("spec,kw,data", [
    (FULL, {}, "farm"), (FULL, {"cov_type": "HC0"}, "farm"), (FULL, {"cov_type": "HC1"}, "farm"),
    (FULL, {"cov_type": "HC2"}, "farm"), (FULL, {"cov_type": "HC3"}, "farm"),
    (FULL, {"cov_type": "cluster", "cov_opts": {"groups": "region"}}, "farm"),
    (rg.ModelSpec(y="yield_t_per_feddan", x=["avg_temp_c", "rainfall_mm", "year"], categorical=["crop"], form="loglin"),
     {"cov_type": "HAC", "cov_opts": {"maxlags": 5}}, "clim"),
    (rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days"], form="linlog"), {}, "farm"),
    (rg.ModelSpec(y="output_tonnes", x=["land_feddan"], form="reciprocal"), {}, "farm"),
    (rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days"], categorical=["irrigation"]), {"method": "fgls"}, "farm"),
    (SMALL, {"method": "huber"}, "farm"), (SMALL, {"method": "bisquare"}, "farm"),
])
def test_r_reproduces_estimates(spec, kw, data, tmp_path):
    df = farm_survey_demo() if data == "farm" else climate_crop_demo()
    r = rg.fit(df, spec, **kw)
    o = _run_r(r, df, tmp_path, COEF)
    tol = 1e-4 if kw.get("method") in ("huber", "bisquare") else 1e-8   # IRLS convergence tolerance differs
    assert np.allclose(o.b.values, r.params.values, rtol=tol, atol=tol)
    assert np.allclose(o.se.values, r.bse.values, rtol=tol, atol=tol)


def test_r_wls_and_rls(tmp_path):
    farm = farm_survey_demo()
    r = rg.fit(farm, SMALL, method="wls", weights="land_feddan")
    f = tmp_path / "d.csv"
    farm.to_csv(f, index=False)
    code = rp.code_r(r, str(f)).replace("weights = w)", "weights = land_feddan)") + COEF
    (tmp_path / "m.R").write_text(code)
    assert subprocess.run(["Rscript", "m.R"], cwd=tmp_path, capture_output=True).returncode == 0
    o = pd.read_csv(tmp_path / "out.csv")
    assert np.allclose(o.b.values, r.params.values) and np.allclose(o.se.values, r.bse.values)
    R, q, lab = rg.parse_restrictions("ln(land_feddan) + ln(labour_days) = 1", r.names)
    rr = rg.restricted_ls(farm, SMALL, R, q, lab)
    o = _run_r(rr, farm, tmp_path, "")  # noqa: F841 — runs linearHypothesis; now read F explicitly
    code = rp.code_r(rr, str(f)).replace("linearHypothesis(m,", "lh <- linearHypothesis(m,")
    code += '\nwrite.csv(data.frame(F = lh$F[2], p = lh$`Pr(>F)`[2]), "out.csv", row.names = FALSE)\n'
    (tmp_path / "m.R").write_text(code)
    assert subprocess.run(["Rscript", "m.R"], cwd=tmp_path, capture_output=True).returncode == 0
    o = pd.read_csv(tmp_path / "out.csv")
    assert np.isclose(o.F[0], rr.extra["F_restr"]) and np.isclose(o.p[0], rr.extra["p_restr"])


def test_r_diagnostics_match(tmp_path):
    farm = farm_survey_demo()
    r = rg.fit(farm, FULL)
    tail = ('d <- c(bp = unname(bptest(m)$statistic), bpo = unname(bptest(m, studentize = FALSE)$statistic), '
            'dw = unname(dwtest(m)$statistic), dwp = dwtest(m)$p.value, bg = unname(bgtest(m, order = 2)$statistic), '
            'reset = unname(resettest(m, power = 2:3, type = "fitted")$statistic), '
            'jb = unname(jarque.bera.test(resid(m))$statistic), lb = unname(Box.test(resid(m), lag = 10, type = "Ljung-Box")$statistic), '
            'cook = max(cooks.distance(m)), gq = unname(gqtest(m, fraction = 0.2, order.by = fitted(m))$statistic), '
            'vif = max(vif(m, type = "terms")))\n'
            'write.csv(data.frame(k = names(d), v = d), "out.csv", row.names = FALSE)\n')
    d = _run_r(r, farm, tmp_path, tail).set_index("k").v
    e, X = r.wresid, r.wexog
    mine = {"bp": dx.breusch_pagan(e, X, True)["stat"], "bpo": dx.breusch_pagan(e, X, False)["stat"],
            "dw": dx.durbin_watson(e), "bg": dx.breusch_godfrey(e, X, 2)["stat"],
            "reset": dx.reset_test(r.y, X, r.fitted)["stat"], "jb": dx.normality(e)[0]["stat"],
            "lb": dx.ljung_box(e, 10)["stat"], "cook": dx.influence(e, X)["Cook's D"].max(),
            "gq": dx.goldfeld_quandt(r.y, X, r.fitted)["stat"]}
    for k, v in mine.items():
        assert np.isclose(v, d[k], rtol=1e-8), k
    # exact DW p-value: R's dwtest (Pan algorithm) vs Aetheris (Imhof) — two numerical integrations
    assert abs(dx.durbin_watson_exact(e, X)["p"] - d["dwp"]) < 2e-3
