"""Cross-validation of the v1.11 engines against R reference packages (conda-forge R; set AETHERIS_RSCRIPT_V111):
- quantile regression: quantreg::rq / summary.rq (iid, nid, ker) and anova.rq (joint and separate slope equality);
- spatial: spdep (Moran, Geary, LISA, LM tests) and spatialreg (lagsarlm, errorsarlm, Durbin, sacsarlm, impacts); the
  fixed-effects spatial panel against lagsarlm / errorsarlm on the within-transformed data with I_T ⊗ W; GWR against
  PySAL mgwr;
- policy evaluation: did (att_gt / aggte, analytical SEs), fixest::sunab, ivreg + sandwich HC1, HC2 difference in
  means and Lin regression, MatchIt nearest-neighbour matching, WeightIt glm weights with M-estimation SEs, Synth (our
  weights reach a predictor loss no larger than Synth's ipop solution);
- climate: fixest two-way FE (GDD/KDD, bins, EKC), lm + HC1 (Ricardian), feols with explicit country trends (BHM).
"""
import os
import subprocess
import warnings

import numpy as np
import pandas as pd
import pytest

from analytics import causal as C
from analytics import climate as K
from analytics import quantile as Q
from analytics import spatial as S
from data import samples as smp

warnings.filterwarnings("ignore")
RS = os.environ.get("AETHERIS_RSCRIPT_V111", "/opt/mm/renv/bin/Rscript")


def _ok(pkgs):
    return os.path.exists(RS) and subprocess.run([RS, "-e", ";".join(f"suppressMessages(library({p}))" for p in pkgs)], capture_output=True).returncode == 0


def _run(code, tmp):
    (tmp / "s.R").write_text(code + '\nfor (n in names(o)) { cat("##", n, "\\n"); cat(format(as.vector(as.matrix(o[[n]])), digits = 15), "\\n") }\n')
    out = subprocess.run([RS, "s.R"], capture_output=True, text=True, cwd=tmp)
    assert out.returncode == 0, out.stderr[-2500:]
    B, cur = {}, None
    for line in out.stdout.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip()
            B[cur] = []
        elif cur and line.strip():
            B[cur] += [float(v) for v in line.split()]
    return {k: np.array(v) for k, v in B.items()}


def _close(a, b, tol):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    assert a.shape == b.shape, (a.shape, b.shape)
    assert np.max(np.abs(a - b) / np.maximum(1, np.abs(b))) < tol, (a[:8], b[:8])


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _ok(["quantreg"]), reason="R quantreg not available")
def test_quantile_matches_quantreg(tmp_path):
    d = smp.farm_survey_demo()
    d = d.assign(ly=np.log(d.output_tonnes), ll=np.log(d.labour_days), lf=np.log(d.fertilizer_kg))[["ly", "ll", "lf", "education_years"]].dropna()
    d.to_csv(tmp_path / "d.csv", index=False)
    B = _run('''suppressMessages(library(quantreg)); d <- read.csv("d.csv"); o <- list(); f <- ly ~ ll + lf + education_years
for (t in c(0.1, 0.5, 0.9)) for (s in c("iid", "nid", "ker")) { m <- rq(f, tau = t, data = d); o[[paste0(s, t)]] <- summary(m, se = s)$coefficients[, 1:2] }
ms <- rq(f, tau = c(0.25, 0.5, 0.75), data = d); a <- anova(ms, joint = TRUE); o$joint <- a$table$Tn
a2 <- anova(ms, joint = FALSE); o$sep <- a2$table$Tn''', tmp_path)
    x = ["ll", "lf", "education_years"]
    y, X = d["ly"].to_numpy(), np.column_stack([np.ones(len(d)), d[x].to_numpy()])
    for t in (0.1, 0.5, 0.9):
        for s, tol in (("iid", 1e-8), ("nid", 1e-5), ("ker", 1e-8)):
            r = Q.qr_fit(y, X, t, s)
            _close(np.r_[r["b"], np.sqrt(np.diag(r["V"]))], B[f"{s}{t}"], tol)
    se = Q.slope_equality(d, "ly", x, [0.25, 0.5, 0.75])
    _close(se.F.iloc[-1], B["joint"], 1e-5)
    _close(se.F.iloc[:-1], B["sep"], 1e-5)


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _ok(["spdep", "spatialreg"]), reason="R spdep / spatialreg not available")
def test_spatial_matches_spdep_spatialreg(tmp_path):
    D = smp.spatial_districts_demo()
    D["ln_dist"] = np.log(D.distance_market_km)
    x = ["irrigation_share", "ln_dist", "soil_index", "avg_temp_c"]
    W = S.weights(D[["longitude", "latitude"]].values, "knn", 6)
    D.to_csv(tmp_path / "d.csv", index=False)
    pd.DataFrame(W).to_csv(tmp_path / "w.csv", index=False)
    B = _run('''suppressMessages({library(spdep); library(spatialreg)})
d <- read.csv("d.csv"); W <- as.matrix(read.csv("w.csv")); lw <- mat2listw(W, style = "W"); o <- list()
f <- ln_land_value ~ irrigation_share + ln_dist + soil_index + avg_temp_c
m <- moran.test(d$ln_land_value, lw); o$moran <- c(m$estimate, m$statistic)
g <- geary.test(d$ln_land_value, lw); o$geary <- c(g$estimate, g$statistic)
t <- lm.RStests(lm(f, data = d), lw, test = "all"); o$lm <- sapply(t, function(z) z$statistic)
lg <- lagsarlm(f, data = d, listw = lw, method = "eigen"); o$sar <- c(coef(lg), logLik(lg), summary(lg)$Coef[, 2], lg$rho.se)
er <- errorsarlm(f, data = d, listw = lw, method = "eigen"); o$sem <- c(coef(er), logLik(er), summary(er)$Coef[, 2], er$lambda.se)
sd <- lagsarlm(f, data = d, listw = lw, method = "eigen", Durbin = TRUE); o$sdm <- c(coef(sd), logLik(sd))
im <- impacts(sd, listw = lw); o$imp <- c(im$direct, im$indirect, im$total)
sc <- sacsarlm(f, data = d, listw = lw, method = "eigen"); o$sac <- c(coef(sc), logLik(sc))
o$lisa <- localmoran(d$ln_land_value, lw)[, "Ii"]''', tmp_path)
    m = S.moran(D.ln_land_value, W, perms=0)
    _close([m["I"], m["E[I]"], m["Var (randomisation)"], m["z (randomisation)"]], B["moran"], 1e-10)
    g = S.geary(D.ln_land_value, W)
    _close([g["C"], g["E[C]"], g["Var"], g["z"]], B["geary"], 1e-10)
    _close(S.lm_tests(D.ln_land_value, np.column_stack([np.ones(150), D[x]]), W).Statistic, B["lm"], 1e-8)
    for mdl in ("sar", "sem"):
        r = S.spatial_model(D, "ln_land_value", x, W, mdl, 10)
        k = len(r["est"]) - 1
        _close(np.r_[r["est"][k], r["est"][:k], r["loglik"], r["table"]["Std. Err."]], B[mdl], 1e-5)
    r = S.spatial_model(D, "ln_land_value", x, W, "sdm", 10)
    k = len(r["est"]) - 1
    _close(np.r_[r["est"][k], r["est"][:k], r["loglik"]], B["sdm"], 1e-5)
    im = r["impacts"]
    _close(np.r_[im[im.Impact == "Direct"].Estimate, im[im.Impact.str.startswith("Indirect")].Estimate, im[im.Impact == "Total"].Estimate], B["imp"], 1e-6)
    r = S.spatial_model(D, "ln_land_value", x, W, "sac", 10)
    _close(np.r_[r["est"][-2:], r["est"][:-2], r["loglik"]], B["sac"], 1e-4)
    _close(S.lisa(D.ln_land_value, W, perms=9).Ii, B["lisa"], 1e-10)


@pytest.mark.skipif(not _ok(["spdep", "spatialreg"]), reason="R spatialreg not available")
def test_spatial_panel_matches_within_lagsarlm(tmp_path):
    P = smp.spatial_panel_demo()
    units = pd.unique(P.district)
    W = S.weights(P.drop_duplicates("district")[["longitude", "latitude"]].values, "knn", 6)
    d = P.sort_values(["year", "district"]).copy()
    d["district"] = pd.Categorical(d.district, units)
    d = d.sort_values(["year", "district"])
    for c in ("ln_yield", "ln_fertilizer", "rainfall_anomaly"):
        d[c + "_w"] = d[c] - d.groupby("district", observed=True)[c].transform("mean")
    d.to_csv(tmp_path / "d.csv", index=False)
    pd.DataFrame(W).to_csv(tmp_path / "w.csv", index=False)
    T = d.year.nunique()
    B = _run(f'''suppressMessages({{library(spdep); library(spatialreg)}}); d <- read.csv("d.csv"); W <- as.matrix(read.csv("w.csv")); o <- list()
lw <- mat2listw(kronecker(diag({T}), W), style = "W")
f <- ln_yield_w ~ ln_fertilizer_w + rainfall_anomaly_w - 1
m <- lagsarlm(f, data = d, listw = lw, method = "eigen"); o$sar <- c(coef(m), logLik(m))
e <- errorsarlm(f, data = d, listw = lw, method = "eigen"); o$sem <- c(coef(e), logLik(e))''', tmp_path)
    for mdl in ("sar", "sem"):
        r = S.spatial_panel(P, "district", "year", "ln_yield", ["ln_fertilizer", "rainfall_anomaly"], W, mdl)
        est = r["table"]["Coef."].to_numpy()
        _close(np.r_[est[-1], est[:-1], r["loglik"]], B[mdl], 1e-5)


def test_gwr_matches_mgwr():
    mgwr = pytest.importorskip("mgwr")
    from mgwr.gwr import GWR
    from mgwr.sel_bw import Sel_BW
    D = smp.spatial_districts_demo()
    D["ln_dist"] = np.log(D.distance_market_km)
    xy = D[["longitude", "latitude"]].values
    x = ["irrigation_share", "ln_dist", "soil_index", "avg_temp_c"]
    bw = Sel_BW(xy, D[["ln_land_value"]].values, D[x].values, fixed=False, kernel="bisquare", spherical=False).search(criterion="AICc")
    r = GWR(xy, D[["ln_land_value"]].values, D[x].values, bw, fixed=False, kernel="bisquare", spherical=False).fit()
    g = S.gwr(D, "ln_land_value", x, xy)
    assert g["bandwidth"] == bw
    assert abs(g["AICc"] - r.aicc) < 1e-4 and abs(g["R²"] - r.R2) < 1e-6
    assert np.abs(r.params - g["local"][["const"] + x].values).max() < 1e-5
    del mgwr


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _ok(["did", "fixest"]), reason="R did / fixest not available")
def test_did_matches_r_did_and_fixest(tmp_path):
    st = smp.staggered_did_demo()
    st.to_csv(tmp_path / "st.csv", index=False)
    B = _run('''suppressMessages({library(did); library(fixest)}); o <- list()
st <- read.csv("st.csv"); st$vid <- as.integer(factor(st$village))
for (cg in c("nevertreated","notyettreated")) {
 a <- att_gt("ln_water_productivity","year","vid","first_treated", data=st, control_group=cg, bstrap=FALSE, cband=FALSE)
 o[[paste0("gt_",cg)]] <- c(a$att, a$se)
 for (ty in c("simple","dynamic","group","calendar")) { g <- aggte(a, type=ty, bstrap=FALSE, cband=FALSE)
   o[[paste0(ty,"_",cg)]] <- c(g$overall.att, g$overall.se, if (ty!="simple") c(g$att.egt, g$se.egt)) } }
st$ft2 <- ifelse(st$first_treated==0, 10000, st$first_treated)
m <- feols(ln_water_productivity ~ sunab(ft2, year) | vid + year, data=st, cluster=~vid)
o$sa <- c(coef(m), se(m)); a <- summary(m, agg="ATT"); o$sa_att <- c(coef(a)["ATT"], se(a)["ATT"])
st$rel <- ifelse(st$first_treated==0, -1000, pmin(pmax(st$year-st$first_treated, -5), 5))
e <- feols(ln_water_productivity ~ i(rel, ref=c(-1, -1000)) | vid + year, data=st, cluster=~vid); o$es <- c(coef(e), se(e))''', tmp_path)
    for cg, ctl in (("nevertreated", "never"), ("notyettreated", "notyet")):
        r = C.callaway_santanna(st, "village", "year", "ln_water_productivity", "first_treated", ctl)
        _close(np.r_[r["att_gt"]["ATT(g,t)"], r["att_gt"]["Std. Err."]], B["gt_" + cg], 1e-10)
        S_ = r["summary"]
        _close([S_.ATT[0], S_["Std. Err."][0]], B["simple_" + cg], 1e-10)
        _close(np.r_[S_.ATT[1], S_["Std. Err."][1], r["dynamic"].ATT, r["dynamic"]["Std. Err."]], B["dynamic_" + cg], 1e-10)
        _close(np.r_[S_.ATT[2], S_["Std. Err."][2], r["group"].ATT, r["group"]["Std. Err."]], B["group_" + cg], 1e-10)
        _close(np.r_[S_.ATT[3], S_["Std. Err."][3], r["calendar"].ATT, r["calendar"]["Std. Err."]], B["calendar_" + cg], 1e-10)
    sa = C.sun_abraham(st, "village", "year", "ln_water_productivity", "first_treated")
    _close(np.r_[sa["event"].Estimate, sa["event"]["Std. Err."]], B["sa"], 1e-8)
    _close([sa["att"]["Estimate"], sa["att"]["Std. Err."]], B["sa_att"], 1e-8)
    es = C.event_study_twfe(st, "village", "year", "ln_water_productivity", "first_treated")
    es = es[es["Relative period"] != -1]
    _close(np.r_[es.Estimate, es["Std. Err."]], B["es"], 1e-8)


@pytest.mark.skipif(not _ok(["ivreg", "sandwich", "MatchIt", "WeightIt"]), reason="R ivreg / MatchIt / WeightIt not available")
def test_rct_iv_matching_weighting_match_r(tmp_path):
    rc, cr = smp.rct_extension_demo(), smp.credit_program_demo()
    rc.to_csv(tmp_path / "rc.csv", index=False)
    cr.to_csv(tmp_path / "cr.csv", index=False)
    B = _run('''suppressMessages({library(ivreg); library(sandwich); library(MatchIt); library(WeightIt)}); o <- list()
rc <- read.csv("rc.csv")
iv <- ivreg(ln_yield ~ participated | treatment_offered, data=rc); o$iv <- c(coef(iv), sqrt(diag(vcovHC(iv, type="HC1"))))
l1 <- lm(ln_yield ~ treatment_offered, data=rc); o$dim <- c(coef(l1)[2], sqrt(vcovHC(l1, "HC2")[2,2]))
for (v in c("land_feddan","education_years","baseline_ln_yield")) rc[[paste0(v,"_c")]] <- rc[[v]] - mean(rc[[v]])
l2 <- lm(ln_yield ~ treatment_offered*(land_feddan_c+education_years_c+baseline_ln_yield_c), data=rc); o$lin <- c(coef(l2)[2], sqrt(vcovHC(l2,"HC2")[2,2]))
cr <- read.csv("cr.csv"); f <- participant ~ age + education_years + land_feddan + female_head + distance_branch_km
md <- match.data(matchit(f, data=cr, method="nearest", distance="glm", replace=TRUE, estimand="ATT"))
o$psm <- coef(lm(ln_income ~ participant, data=md, weights=weights))[2]
for (es in c("ATE","ATT")) { w <- weightit(f, data=cr, method="glm", estimand=es); fit <- lm_weightit(ln_income ~ participant, data=cr, weightit=w)
  o[[paste0("ipw_",es)]] <- c(coef(fit)[2], sqrt(vcov(fit)[2,2])) }''', tmp_path)
    iv = C.iv_2sls(rc, "ln_yield", "participated", ["treatment_offered"])
    _close(np.r_[iv["table"].Estimate, iv["table"]["Std. Err."]], B["iv"], 1e-10)
    T = C.rct(rc, "ln_yield", "treatment_offered", ["land_feddan", "education_years", "baseline_ln_yield"])["table"]
    _close([T.Estimate[0], T["Std. Err."][0]], B["dim"], 1e-10)
    _close([T.Estimate[2], T["Std. Err."][2]], B["lin"], 1e-10)
    covs = ["age", "education_years", "land_feddan", "female_head", "distance_branch_km"]
    _close([C.psm(cr, "ln_income", "participant", covs)["estimate"]["Estimate"]], B["psm"], 1e-10)
    for es in ("ATE", "ATT"):
        q = C.ipw_aipw(cr, "ln_income", "participant", covs, es)["table"]
        _close([q.Estimate[0], q["Std. Err."][0]], B["ipw_" + es], 1e-5)


@pytest.mark.skipif(not _ok(["Synth"]), reason="R Synth not available")
def test_synth_weights_at_least_as_good_as_r_synth(tmp_path):
    sc = smp.scm_regions_demo()
    sc["rid"] = pd.factorize(sc.region)[0] + 1
    sc.to_csv(tmp_path / "sc.csv", index=False)
    v = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2]
    B = _run(f'''suppressMessages(library(Synth)); d <- read.csv("sc.csv"); o <- list()
tr <- unique(d$rid[d$region=="Sinai Coast"]); co <- setdiff(unique(d$rid), tr)
sink(tempfile()); dp <- dataprep(foo=d, predictors=c("investment_share","schooling_years"), predictors.op="mean", time.predictors.prior=1995:2011,
 special.predictors=list(list("ln_gdp_pc",1995,"mean"),list("ln_gdp_pc",2000,"mean"),list("ln_gdp_pc",2005,"mean"),list("ln_gdp_pc",2011,"mean")),
 dependent="ln_gdp_pc", unit.variable="rid", unit.names.variable="region", time.variable="year", treatment.identifier=tr, controls.identifier=co,
 time.optimize.ssr=1995:2011, time.plot=1995:2024)
s1 <- synth(dp, custom.v=c({",".join(map(str, v))})); s2 <- synth(dp); sink()
o$lossw <- s1$loss.w; o$lossv <- s2$loss.v''', tmp_path)
    r = C.synth_control(sc, "region", "year", "ln_gdp_pc", "Sinai Coast", 2012, ["investment_share", "schooling_years"], [1995, 2000, 2005, 2011],
                        custom_v=v, placebo=False)
    bal = r["balance"]
    # predictor loss under the same V (rows scaled by their cross-unit SD as Synth)
    Y = sc.pivot(index="year", columns="region", values="ln_gdp_pc")
    rows = [sc[sc.year < 2012].groupby("region")[p].mean() for p in ["investment_share", "schooling_years"]] + [Y.loc[L] for L in [1995, 2000, 2005, 2011]]
    P = pd.concat(rows, axis=1).T
    s_ = P.std(axis=1, ddof=1).to_numpy()
    gap = (bal.Treated - bal.Synthetic).to_numpy() / s_
    vv = np.array(v) / sum(v)
    assert gap @ (vv * gap) <= B["lossw"][0] + 1e-10
    rf = C.synth_control(sc, "region", "year", "ln_gdp_pc", "Sinai Coast", 2012, ["investment_share", "schooling_years"], [1995, 2000, 2005, 2011], placebo=False)
    assert rf["pre RMSPE"] ** 2 <= B["lossv"][0] * 1.0001


# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not _ok(["fixest", "sandwich"]), reason="R fixest not available")
def test_climate_models_match_fixest_and_lm(tmp_path):
    cp, rd, ce = smp.crop_climate_panel_demo(), smp.ricardian_demo(), smp.country_emissions_demo()
    cp.to_csv(tmp_path / "cp.csv", index=False)
    rd.to_csv(tmp_path / "rd.csv", index=False)
    ce.to_csv(tmp_path / "ce.csv", index=False)
    B = _run('''suppressMessages({library(fixest); library(sandwich)}); o <- list()
cp <- read.csv("cp.csv"); m <- feols(ln_yield ~ gdd + kdd + precip_mm + I(precip_mm^2) | district + year, cp, cluster = ~district); o$gk <- c(coef(m), se(m))
m <- feols(ln_yield ~ days_tmax_lt10 + days_tmax_10_15 + days_tmax_15_20 + days_tmax_25_30 + days_tmax_30_35 + days_tmax_gt35 + precip_mm + I(precip_mm^2) | district + year, cp, cluster = ~district); o$bins <- c(coef(m), se(m))
rd <- read.csv("rd.csv"); l <- lm(log(land_value_k_egp) ~ temp_winter + I(temp_winter^2) + temp_summer + I(temp_summer^2) + precip_winter + I(precip_winter^2) + precip_summer + I(precip_summer^2) + soil_quality + distance_city_km + irrigated, rd)
o$ric <- c(coef(l), sqrt(diag(vcovHC(l, "HC1"))))
ce <- read.csv("ce.csv"); ce$lpc <- log(ce$co2_mt / ce$population); ce$ly <- log(ce$gdp_pc_usd)
m <- feols(lpc ~ ly + I(ly^2) | country + year, ce, cluster = ~country); o$ekc <- c(coef(m), se(m))
ce$tt <- ce$year - min(ce$year); g <- ce[!is.na(ce$gdp_growth), ]
m <- feols(gdp_growth ~ temperature_c + I(temperature_c^2) + i(country, tt) | country + year, g, cluster = ~country); o$bhm <- c(coef(m)[1:2], se(m)[1:2])''', tmp_path)
    r = K.temp_response(cp, "district", "year", "ln_yield", "gddkdd", gdd="gdd", kdd="kdd", precip="precip_mm")
    _close(np.r_[r["table"].Estimate, r["table"]["Std. Err."]], B["gk"], 1e-8)
    bins = [c for c in cp if c.startswith("days")]
    r = K.temp_response(cp, "district", "year", "ln_yield", "bins", bins=bins, ref_bin="days_tmax_20_25", precip="precip_mm")
    _close(np.r_[r["table"].Estimate, r["table"]["Std. Err."]], B["bins"], 1e-8)
    ri = K.ricardian(rd, "land_value_k_egp", ["temp_winter", "temp_summer", "precip_winter", "precip_summer"], ["soil_quality", "distance_city_km", "irrigated"])
    _close(np.r_[ri["table"].Estimate, ri["table"]["Std. Err."]], B["ric"], 1e-6)   # squared normals: ill-conditioned
    e = K.ekc(ce, "country", "year", "co2_mt", "gdp_pc_usd", per_capita="population")
    _close(np.r_[e["table"].Estimate, e["table"]["Std. Err."]], B["ekc"], 1e-8)
    b = K.bhm(ce, "country", "year", "gdp_growth", "temperature_c")
    _close(np.r_[b["table"].Estimate, b["table"]["Std. Err."]], B["bhm"], 1e-6)


def test_spatial_panel_matches_pysal_spreg():
    """Coefficients and analytical standard errors against PySAL spreg Panel_FE_Lag / Panel_FE_Error (Elhorst)."""
    libpysal = pytest.importorskip("libpysal")
    spreg = pytest.importorskip("spreg")
    P = smp.spatial_panel_demo()
    units = pd.unique(P.district)
    W = S.weights(P.drop_duplicates("district")[["longitude", "latitude"]].values, "knn", 6)
    w = libpysal.weights.full2W(W)
    w.transform = "r"

    def wide(c):
        return P.pivot(index="district", columns="year", values=c).loc[units].to_numpy()
    y, X = wide("ln_yield"), np.hstack([wide("ln_fertilizer"), wide("rainfall_anomaly")])
    for cls, mdl in ((spreg.Panel_FE_Lag, "sar"), (spreg.Panel_FE_Error, "sem")):
        m = cls(y, X, w)
        r = S.spatial_panel(P, "district", "year", "ln_yield", ["ln_fertilizer", "rainfall_anomaly"], W, mdl)
        _close(r["table"]["Coef."], np.ravel(m.betas)[:3], 1e-6)
        _close(r["table"]["Std. Err."], np.sqrt(np.diag(m.vm))[:3], 1e-6)


def test_sdid_matches_synthdid_port():
    """Synthetic DiD against `synthdid` (Python port of the R package synthdid): ATT and unit / time weights."""
    sdp = pytest.importorskip("synthdid.synthdid")
    sc = smp.scm_regions_demo()
    sc["D"] = ((sc.region == "Sinai Coast") & (sc.year >= 2012)).astype(int)
    st = smp.staggered_did_demo()
    st = st[st.first_treated.isin([0, 2017])].copy()
    st["D"] = ((st.first_treated > 0) & (st.year >= 2017)).astype(int)
    for d, u, t, y in ((sc, "region", "year", "ln_gdp_pc"), (st, "village", "year", "ln_water_productivity")):
        m = sdp.Synthdid(d, u, t, "D", y).fit()
        o = C.synth_did(d, u, t, y, "D", reps=5)
        assert abs(m.att_info.att_time.iloc[0] - o["ATT"]) < 1e-10
        assert np.abs(np.sort(np.ravel(m.weights["omega"])) - np.sort(o["unit weights"].to_numpy())).max() < 1e-10
        assert np.abs(np.ravel(m.weights["lambda"]) - o["time weights"].to_numpy()).max() < 1e-10
