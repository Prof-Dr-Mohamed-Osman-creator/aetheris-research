"""v1.8 cross-validation against R plm: static models (pooled, FE one/two-way, RE Swamy-Arora, between, FD), robust
covariances (cluster, Driscoll-Kraay, PCSE), specification tests (F, BP/Honda, Hausman, robust Hausman, CD/LM/scaled LM,
Wooldridge AR and FD), unit roots (LLC, IPS, Maddala-Wu, Pm, Choi, logit, Hadri, CIPS), Dumitrescu-Hurlin, MG/CMG/CCE and
pgmm (difference and system GMM, one/two-step, collapse; balanced and unbalanced). Skipped when R/plm is missing."""
import shutil
import subprocess
import warnings

import numpy as np
import pytest

from analytics import panel as pn
from analytics import panel_gmm as pg
from analytics import panel_report as pr
from data.samples import firm_panel_demo, macro_panel_demo

warnings.filterwarnings("ignore")
R_OK = bool(shutil.which("Rscript")) and subprocess.run(["Rscript", "-e", "library(plm)"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(not R_OK, reason="R package plm is not installed")

F = firm_panel_demo()
U = F.drop(index=np.random.default_rng(1).choice(len(F), 300, replace=False)).reset_index(drop=True)
M = macro_panel_demo()
M = M.assign(lc=np.log(M.co2_t_pc), ly=np.log(M.gdp_pc_usd), le=np.log(M.energy_kgoe_pc))
X = ["ln_capital", "ln_labour"]


def _run(code, tmp):
    p = tmp / "s.R"
    p.write_text(code)
    out = subprocess.run(["Rscript", str(p)], capture_output=True, text=True, cwd=tmp)
    assert out.returncode == 0, out.stderr[-2000:]
    B, cur = {}, None
    for line in out.stdout.splitlines():
        if line.startswith("## "):
            cur = line[3:].strip(); B[cur] = []
        elif cur and line.strip():
            B[cur] += [float(v) for v in line.split()]
    return {k: np.array(v) for k, v in B.items()}


DUMP = 'for (n in names(o)) { cat("##", n, "\\n"); cat(format(as.vector(as.matrix(o[[n]])), digits = 15), "\\n") }'


@pytest.fixture(scope="module")
def static_ref(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v18s")
    F.to_csv(tmp / "f.csv", index=False); U.to_csv(tmp / "u.csv", index=False)
    return _run(r'''
suppressMessages(library(plm)); o <- list()
for (nm in c("f", "u")) {
 d <- pdata.frame(read.csv(paste0(nm, ".csv")), index = c("firm_id", "year")); f <- ln_sales ~ ln_capital + ln_labour
 po <- plm(f, d, model = "pooling"); fe <- plm(f, d, model = "within"); fe2 <- plm(f, d, model = "within", effect = "twoways")
 re <- plm(f, d, model = "random"); be <- plm(f, d, model = "between"); fd <- plm(f, d, model = "fd")
 for (m in c("po", "fe", "fe2", "re", "be", "fd")) { mm <- get(m); o[[paste(nm, m, "b")]] <- coef(mm); o[[paste(nm, m, "se")]] <- sqrt(diag(vcov(mm))) }
 o[[paste(nm, "re sig")]] <- ercomp(re)$sigma2
 for (m in c("po", "fe", "re")) o[[paste(nm, m, "cl")]] <- sqrt(diag(vcovHC(get(m), type = "sss", cluster = "group")))
 for (m in c("po", "fe")) { o[[paste(nm, m, "dk")]] <- sqrt(diag(vcovSCC(get(m)))); o[[paste(nm, m, "bk")]] <- sqrt(diag(vcovBK(get(m), cluster = "time"))) }
 o[[paste(nm, "F")]] <- pFtest(fe, po)$statistic
 o[[paste(nm, "bp")]] <- c(plmtest(po, type = "bp")$statistic, plmtest(po, type = "honda")$statistic)
 o[[paste(nm, "haus")]] <- phtest(fe, re)$statistic
 o[[paste(nm, "haus_aux")]] <- phtest(f, d, method = "aux", vcov = function(x) vcovHC(x, method = "arellano", type = "HC0"))$statistic
 o[[paste(nm, "cd")]] <- c(pcdtest(fe, test = "cd")$statistic, pcdtest(fe, test = "lm")$statistic, pcdtest(fe, test = "sclm")$statistic)
 o[[paste(nm, "war")]] <- pwartest(fe)$statistic; o[[paste(nm, "wfd")]] <- pwfdtest(f, d, h0 = "fe")$statistic
}
''' + DUMP, tmp)


@pytest.mark.parametrize("nm", ["f", "u"])
def test_static_models_and_tests_match_plm(static_ref, nm):
    B = static_ref
    P = pn.prepare(F if nm == "f" else U, "firm_id", "year", "ln_sales", X)
    assert P.balanced == (nm == "f")
    R = {}
    for m, key in (("pooled", "po"), ("fe", "fe"), ("fe2", "fe2"), ("re", "re"), ("be", "be"), ("fd", "fd")):
        R[key] = r = pn.fit(P, m)
        np.testing.assert_allclose(r.b, B[f"{nm} {key} b"], atol=1e-7, err_msg=key)
        np.testing.assert_allclose(r.se, B[f"{nm} {key} se"], atol=1e-7, err_msg=key)
    np.testing.assert_allclose([R["re"].stats["σ²(idiosyncratic)"], R["re"].stats["σ²(individual)"]], B[f"{nm} re sig"], rtol=1e-7)
    for key, m in (("fe", "fe"), ("po", "pooled"), ("re", "re")):
        np.testing.assert_allclose(pn.fit(P, m, "cluster").se, B[f"{nm} {key} cl"], atol=1e-7)
    for key, m in (("fe", "fe"), ("po", "pooled")):
        np.testing.assert_allclose(pn.fit(P, m, "dk").se, B[f"{nm} {key} dk"], atol=1e-7)
        np.testing.assert_allclose(pn.fit(P, m, "pcse").se, B[f"{nm} {key} bk"], atol=1e-7)
    np.testing.assert_allclose(R["fe"].stats["F (effects = 0)"], B[f"{nm} F"][0], rtol=1e-7)
    bp = pn.bp_lm(R["po"])
    np.testing.assert_allclose([bp["LM"], bp["Honda"]], B[f"{nm} bp"], rtol=1e-7)
    np.testing.assert_allclose(pn.hausman(R["fe"], R["re"])["stat"], B[f"{nm} haus"][0], rtol=1e-6)
    np.testing.assert_allclose(pn.hausman_robust(P)["stat"], B[f"{nm} haus_aux"][0], rtol=1e-6)
    np.testing.assert_allclose(pn.cross_section_dependence(R["fe"])["Statistic"].values[:3], B[f"{nm} cd"], rtol=1e-7)
    np.testing.assert_allclose(pn.wooldridge_ar(P)["stat"], B[f"{nm} war"][0], rtol=1e-7)
    np.testing.assert_allclose(pn.wooldridge_fd(P)["stat"], B[f"{nm} wfd"][0], rtol=1e-7)


def test_unit_roots_causality_mg_cce_match_plm(tmp_path):
    M.to_csv(tmp_path / "m.csv", index=False)
    B = _run(r'''
suppressMessages(library(plm)); d <- pdata.frame(read.csv("m.csv"), index = c("country", "year")); o <- list()
for (ex in c("intercept", "trend")) for (t in c("levinlin", "ips", "madwu", "Pm", "invnormal", "logit")) {
  o[[paste(t, ex, "fix")]] <- purtest(d$lc, test = t, exo = ex, lags = 2, p.approx = "MacKinnon1994")$statistic$statistic
  o[[paste(t, ex, "sic")]] <- purtest(d$lc, test = t, exo = ex, lags = "SIC", pmax = 4, p.approx = "MacKinnon1994")$statistic$statistic }
o[["levinlin none"]] <- purtest(d$lc, test = "levinlin", exo = "none", lags = 1)$statistic$statistic
o[["ips aic"]] <- purtest(d$lc, test = "ips", exo = "intercept", lags = "AIC", pmax = 4)$statistic$statistic
o[["ips hall"]] <- purtest(d$lc, test = "ips", exo = "intercept", lags = "Hall", pmax = 4)$statistic$statistic
o[["hadri intercept"]] <- purtest(d$lc, test = "hadri", exo = "intercept")$statistic$statistic
o[["hadri trend"]] <- purtest(d$lc, test = "hadri", exo = "trend")$statistic$statistic
for (ty in c("trend", "drift", "none")) o[[paste("cips", ty)]] <- suppressWarnings(cipstest(d$lc, lags = 1, type = ty)$statistic)
for (k in 1:2) o[[paste0("dh", k)]] <- unlist(lapply(c("Wbar", "Zbar", "Ztilde"), function(s) pgrangertest(lc ~ le, data = d, order = k, test = s)$statistic))
mg <- pmg(lc ~ ly + le, d, model = "mg"); o$mg <- c(coef(mg), sqrt(diag(vcov(mg))))
cmg <- pmg(lc ~ ly + le, d, model = "cmg"); o$cmg <- c(coef(cmg)[1:3], sqrt(diag(vcov(cmg)))[1:3])
pc <- pcce(lc ~ ly + le, d, model = "p"); o$ccep <- c(coef(pc), sqrt(diag(vcov(pc))))
''' + DUMP, tmp_path)
    P = pn.prepare(M, "country", "year", "lc", ["ly", "le"])
    S = pn.series_by_unit(P, "lc")
    got = {}
    for ex in ("intercept", "trend"):
        for t in ("levinlin", "ips", "madwu", "Pm", "invnormal", "logit"):
            got[f"{t} {ex} fix"] = pn.unit_root(S, t, ex, 2)["stat"]
            got[f"{t} {ex} sic"] = pn.unit_root(S, t, ex, "SIC", 4)["stat"]
        got[f"hadri {ex}"] = pn.unit_root(S, "hadri", ex)["stat"]
    got["levinlin none"] = pn.unit_root(S, "levinlin", "none", 1)["stat"]
    got["ips aic"] = pn.unit_root(S, "ips", "intercept", "AIC", 4)["stat"]
    got["ips hall"] = pn.unit_root(S, "ips", "intercept", "Hall", 4)["stat"]
    for ty in ("trend", "drift", "none"):
        got[f"cips {ty}"] = pn.cips(S, 1, ty)["stat"]
    for k in (1, 2):
        dh = pn.dumitrescu_hurlin(P, "le", "lc", k)
        got[f"dh{k}"] = [dh["Wbar"], dh["Zbar"], dh["Ztilde"]]
    for k, m in (("mg", pn.mean_group(P)), ("cmg", pn.mean_group(P, cce=True)), ("ccep", pn.cce_pooled(P))):
        got[k] = np.r_[m["b"], np.sqrt(np.diag(m["V"]))]
    for k, v in got.items():
        np.testing.assert_allclose(np.ravel(v), B[k], rtol=1e-6, atol=1e-8, err_msg=k)


CASES = [("d", "twosteps", "twoways", False), ("d", "onestep", "individual", False), ("ld", "twosteps", "twoways", True),
         ("ld", "onestep", "twoways", False), ("d", "twosteps", "twoways", True)]


def test_gmm_matches_pgmm(tmp_path):
    Ug = F.drop(index=np.random.default_rng(1).choice(len(F), 200, replace=False)).reset_index(drop=True)
    F.to_csv(tmp_path / "f.csv", index=False); Ug.to_csv(tmp_path / "u.csv", index=False)
    R = ["suppressMessages(library(plm)); o <- list()"]
    for dn in ("f", "u"):
        R.append(f'd <- pdata.frame(read.csv("{dn}.csv"), index = c("firm_id", "year"))')
        for i, (tr, mo, ef, co) in enumerate(CASES):
            R.append(f'm <- pgmm(ln_sales ~ lag(ln_sales, 1) + ln_capital + ln_labour | lag(ln_sales, 2:99) + lag(ln_labour, 2:99) + '
                     f'lag(ln_capital, 1:99), data = d, effect = "{ef}", model = "{mo}", transformation = "{tr}", collapse = {str(co).upper()}); '
                     f's <- summary(m, robust = TRUE); cc <- if (is.list(m$coefficients)) m$coefficients[[2]] else m$coefficients; '
                     f'o[["{dn}{i}"]] <- c(cc[1:3], s$coefficients[1:3, 2], s$sargan$statistic, s$m1$statistic, s$m2$statistic)')
    B = _run("\n".join(R) + "\n" + DUMP, tmp_path)
    spec = pg.GmmSpec("ln_sales", [1], {"ln_capital": [0], "ln_labour": [0]}, {"ln_sales": (2, 99), "ln_labour": (2, 99), "ln_capital": (1, 99)})
    for dn, D in (("f", F), ("u", Ug)):
        for i, (tr, mo, ef, co) in enumerate(CASES):
            r = pg.fit(D, "firm_id", "year", spec, mo, tr, ef, co)
            got = np.r_[r.coef[:3], np.sqrt(np.diag(r.V))[:3], r.tests["Hansen J"][0], r.tests["AR(1)"][0], r.tests["AR(2)"][0]]
            np.testing.assert_allclose(got, B[f"{dn}{i}"], rtol=1e-5, atol=1e-6, err_msg=f"{dn}{i}")


def test_exported_r_code_runs(tmp_path):
    F.to_csv(tmp_path / "data.csv", index=False)
    for kind, kw in (("static", {}), ("unitroot", {"var": "ln_sales"}),
                     ("gmm", {"transformation": "ld", "model": "twosteps", "effect": "twoways", "collapse": True,
                              "endo": ["ln_labour"], "pre": ["ln_capital"], "exo": []})):
        code = pr.code(kind, file_name="data.csv", unit="firm_id", time="year", y="ln_sales", x=X, **kw)["R"]
        (tmp_path / f"{kind}.R").write_text(code)
        out = subprocess.run(["Rscript", f"{kind}.R"], capture_output=True, text=True, cwd=tmp_path)
        assert out.returncode == 0, (kind, out.stderr[-1500:])
