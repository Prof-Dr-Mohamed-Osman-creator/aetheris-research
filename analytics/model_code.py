# ==========================================
# Aetheris Research - Reproducible code for v1.5 models (IV, limited dependent, count, survival)
# ------------------------------------------
# Re-uses the regression code generators for the data preparation (logs, centring, dummies,
# interactions) and adds the estimator-specific calls in Python, R and Stata.
# ==========================================

from __future__ import annotations

import copy
import re

import pandas as pd

from analytics import reg_report as rp
from analytics import regression as rg

S = rp._stata_name


def _proxy(r: rg.RegResult) -> rg.RegResult:
    p = copy.copy(r)
    cols = list(r.design.X.columns)
    p.params = pd.Series(0.0, index=cols)
    p.method, p.cov_type, p.cov_opts, p.extra = "ols", "nonrobust", {}, {}
    return p


def _py_prep(r, file_name):
    code = rp.code_python(_proxy(r), file_name)
    head = code.split("model = ")[0].rstrip().splitlines()
    return [ln for ln in head if not ln.startswith(("from statsmodels.stats import diagnostic",
                                                    "from statsmodels.stats.stattools",
                                                    "from statsmodels.stats.outliers_influence"))]


def _r_formula(r) -> tuple[str, str]:
    code = rp.code_r(_proxy(r), "x.csv")
    m = re.search(r"m <- lm\((.*), data = df\)", code)
    f = m.group(1)
    lhs, rhs = f.split(" ~ ", 1)
    return lhs, rhs


def _stata_prep(r, file_name):
    code = rp.code_stata(_proxy(r), file_name)
    lines = code.splitlines()
    reg = next(ln for ln in lines if ln.startswith("regress "))
    prep = lines[: lines.index(reg)]
    parts = reg.split()
    return prep, parts[1], " ".join(parts[2:])


def _r_head(file_name, pkgs):
    return [f"# install.packages(c({', '.join(repr(p) for p in pkgs)}))",
            "; ".join(f"library({p})" for p in pkgs), "", f'df <- read.csv("{file_name}", check.names = FALSE)']


# ==========================================================
def code_python(r: rg.RegResult, file_name: str = "your_data.csv") -> str:
    m = r.method
    if m in ("2sls", "liml", "fuller", "gmm", "ols") and "Z" in r.extra:
        return _py_iv(r, file_name)
    L = _py_prep(r, file_name)
    if m in ("logit", "probit"):
        L += [f"res = sm.{m.capitalize()}(y, X).fit()", "print(res.summary())",
              'print(res.get_margeff(at="overall", dummy=True).summary())  # average marginal effects']
        if m == "logit":
            L.append("print(np.exp(res.params))  # odds ratios")
    elif m == "cloglog":
        L += ["res = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()", "print(res.summary())"]
    elif m == "lpm":
        L += ['res = sm.OLS(y, X).fit(cov_type="HC1")', "print(res.summary())"]
    elif m == "mnlogit":
        cats = r.extra["categories"]
        L += [f"cats = {cats!r}  # first = base outcome", "codes = y.map({c: i for i, c in enumerate(cats)})",
              'res = sm.MNLogit(codes, X).fit(method="newton")', "print(res.summary())",
              "print(res.get_margeff().summary())"]
    elif m == "mnprobit":
        L += ["# No standard Python package estimates the frequentist multinomial probit; Aetheris computes it",
              "# exactly (Gauss-Hermite quadrature, independent errors) or by GHK simulated ML (correlated errors).",
              "# Stata: mprobit / cmmprobit (see the Stata tab)."]
    elif m in ("ologit", "oprobit"):
        L += ["from statsmodels.miscmodels.ordinal_model import OrderedModel",
              f"cats = {[str(c) for c in r.extra['categories']]!r}",
              "codes = y.astype(str).map({c: i for i, c in enumerate(cats)})",
              f'res = OrderedModel(codes, X.drop(columns="const"), distr="{r.extra["distr"]}").fit(method="bfgs")',
              "print(res.summary())  # thresholds: first cut-point, then log increments"]
    elif m == "frac":
        link = "Logit" if r.extra.get("link") == "logit" else "Probit"
        L += [f'res = sm.GLM(y, X, family=sm.families.Binomial(link=sm.families.links.{link}())).fit(cov_type="HC0")',
              "print(res.summary())"]
    elif m == "tobit":
        lim = r.extra.get("left", 0.0)
        L += ["from scipy import stats", "from statsmodels.base.model import GenericLikelihoodModel", "",
              "class Tobit(GenericLikelihoodModel):",
              f"    left = {lim!r}",
              "    def loglikeobs(self, p):",
              "        b, s = p[:-1], np.exp(p[-1]); xb = self.exog @ b; y = self.endog",
              "        cens = y <= self.left",
              "        return np.where(cens, stats.norm.logcdf((self.left - xb) / s),",
              "                        stats.norm.logpdf((y - xb) / s) - np.log(s))",
              "start = np.r_[np.linalg.lstsq(X, y, rcond=None)[0], np.log(y.std())]",
              "res = Tobit(y, X, extra_params_names=['ln_sigma']).fit(start_params=start, method='bfgs', maxiter=5000)",
              "print(res.summary())  # sigma = exp(ln_sigma)"]
    elif m in ("poisson", "ppml"):
        L += ["res = sm.GLM(y, X, family=sm.families.Poisson()).fit(" + ('cov_type="HC0"' if m == "ppml" or r.cov_type == "HC0" else "") + ")",
              "print(res.summary()); print(np.exp(res.params))  # incidence-rate ratios"]
    elif m == "negbin":
        L += ['res = sm.NegativeBinomial(y, X, loglike_method="nb2").fit(maxiter=500)', "print(res.summary())"]
    elif m in ("zip", "zinb"):
        inf = [c.replace("inflate: ", "") for c in r.names if c.startswith("inflate: ") and c != "inflate: const"]
        cls = "ZeroInflatedPoisson" if m == "zip" else "ZeroInflatedNegativeBinomialP"
        L += [f"Z = sm.add_constant(df.loc[X.index, {inf!r}])" if inf else "Z = np.ones((len(X), 1))",
              f"res = sm.{cls}(y, X, exog_infl=Z, inflation='logit'{', p=2' if m == 'zinb' else ''}).fit(method='bfgs', maxiter=5000)",
              "print(res.summary())"]
    elif m in ("hurdle_p", "hurdle_nb"):
        L += ["from statsmodels.discrete.truncated_model import HurdleCountModel",
              "# statsmodels' HurdleCountModel only offers a Poisson-type zero hurdle; Aetheris also offers the logit hurdle",
              f"res = HurdleCountModel(y, X, dist='{'poisson' if m == 'hurdle_p' else 'negbin'}', zerodist='poisson', p=2).fit(maxiter=2000)",
              "print(res.summary())"]
    elif m == "cox":
        t, e = r.design.spec.y, r.extra.get("event_name", "event")
        L += [f'res = sm.PHReg(y, X.drop(columns="const"), status=df.loc[X.index, "{e}"], ties="{r.extra["ties"]}").fit()',
              "print(res.summary()); print(np.exp(res.params))  # hazard ratios"]
    elif m.startswith("aft_"):
        dist = r.extra["dist"]
        cls = {"weibull": "WeibullAFTFitter", "lognormal": "LogNormalAFTFitter", "loglogistic": "LogLogisticAFTFitter",
               "exponential": "WeibullAFTFitter"}[dist]
        e = r.extra.get("event_name", "event")
        L += ["# pip install lifelines", f"from lifelines import {cls}",
              f'd = X.drop(columns="const").assign(T=y, E=df.loc[X.index, "{e}"])',
              f"fit = {cls}().fit(d, duration_col='T', event_col='E')" + ("  # exponential: fix rho = 1 (Weibull)" if dist == "exponential" else ""),
              "fit.print_summary()"]
    else:
        L += ["# See the Aetheris output: this estimator is implemented directly by maximum likelihood",
              "# (R and Stata equivalents are given in the other tabs)."]
    return "\n".join(L) + "\n"


def code_r(r: rg.RegResult, file_name: str = "your_data.csv") -> str:
    m = r.method
    if "Z" in r.extra:
        return _r_iv(r, file_name)
    lhs, rhs = _r_formula(r)
    if m in ("logit", "probit", "cloglog", "lpm", "frac"):
        L = _r_head(file_name, ["lmtest", "sandwich"])
        if m == "lpm":
            L += [f"m <- lm({lhs} ~ {rhs}, data = df)", 'coeftest(m, vcov. = vcovHC(m, type = "HC1"))']
        elif m == "frac":
            link = r.extra.get("link", "logit")
            L += [f"m <- glm({lhs} ~ {rhs}, family = quasibinomial(link = \"{link}\"), data = df)",
                  'coeftest(m, vcov. = sandwich(m))   # Papke-Wooldridge robust (HC0) standard errors']
        else:
            L += [f"m <- glm({lhs} ~ {rhs}, family = binomial(link = \"{m}\"), data = df)",
                  "coeftest(m)" if r.cov_type == "nonrobust" else "coeftest(m, vcov. = sandwich(m))",
                  "# average marginal effects: install.packages('marginaleffects'); marginaleffects::avg_slopes(m)"]
            if m == "logit":
                L.append("exp(cbind(OR = coef(m), confint.default(m)))   # odds ratios")
    elif m == "mnlogit":
        base = r.extra["categories"][0]
        L = _r_head(file_name, ["nnet"])
        L += [f'df${lhs.strip("`")} <- relevel(factor(df${lhs.strip("`")}), ref = "{base}")',
              f"m <- multinom({lhs} ~ {rhs}, data = df, maxit = 1000, reltol = 1e-12)", "summary(m)",
              "exp(coef(m))   # relative-risk ratios"]
    elif m == "mnprobit":
        L = _r_head(file_name, ["MNP"]) + ["# CRAN offers only the Bayesian multinomial probit (Imai & van Dyk):",
                                          f"m <- mnp({lhs} ~ {rhs}, data = df, base = \"{r.extra['categories'][0]}\", n.draws = 20000)",
                                          "summary(m)   # posterior estimates — comparable in sign / magnitude, not identical to ML"]
    elif m in ("ologit", "oprobit"):
        cats = [str(c) for c in r.extra["categories"]]
        L = _r_head(file_name, ["MASS"])
        L += [f"df${lhs.strip('`')} <- factor(df${lhs.strip('`')}, levels = c({', '.join(repr(c) for c in cats)}), ordered = TRUE)",
              f"m <- polr({lhs} ~ {rhs}, data = df, method = \"{'logistic' if m == 'ologit' else 'probit'}\", Hess = TRUE)",
              "summary(m)   # 'Intercepts' are the cut-points"]
    elif m == "tobit":
        L = _r_head(file_name, ["AER"])
        lim = r.extra.get("left", 0.0)
        L += [f"m <- tobit({lhs} ~ {rhs}, left = {lim}, data = df)", "summary(m)   # Log(scale) = ln σ"]
    elif m == "truncreg":
        L = _r_head(file_name, ["truncreg"])
        L += [f"m <- truncreg({lhs} ~ {rhs}, point = {r.extra.get('lower', 0)}, direction = 'left', data = df, iterlim = 1000)",
              "summary(m)   # check convergence: truncreg's optimiser can stall when y has a large scale (rescale y if needed)"]
    elif m in ("heckman2", "heckman_ml"):
        sel = r.extra["selection_spec"]
        L = _r_head(file_name, ["sampleSelection"])
        srhs = " + ".join(f"`{v}`" for v in sel.x + sel.categorical)
        L += [f"m <- selection(`{sel.y}` ~ {srhs}, {lhs} ~ {rhs}, data = df, method = \"{'2step' if m == 'heckman2' else 'ml'}\")",
              "summary(m)"]
    elif m in ("cragg", "dhurdle"):
        L = _r_head(file_name, ["mhurdle"])
        L += ["# Cragg / double-hurdle: mhurdle(y ~ participation | outcome | 0, dist = 'n', corr = FALSE)",
              "# mhurdle rescales y internally (scaled = TRUE): outcome coefficients and sd are in units of y / scale;",
              "# the ratios β / σ and the participation (h1) coefficients are directly comparable with Aetheris.",
              f"m <- mhurdle({lhs} ~ {rhs} | {rhs} | 0, data = df, dist = 'n', h2 = {'TRUE' if m == 'dhurdle' else 'FALSE'}, corr = FALSE)",
              "summary(m)"]
    elif m in ("poisson", "ppml", "negbin"):
        L = _r_head(file_name, ["MASS", "lmtest", "sandwich"])
        if m == "negbin":
            L += [f"m <- glm.nb({lhs} ~ {rhs}, data = df)", "summary(m)   # theta = 1 / alpha"]
        else:
            L += [f"m <- glm({lhs} ~ {rhs}, family = {'poisson' if m == 'poisson' else 'quasipoisson'}, data = df)",
                  "coeftest(m, vcov. = sandwich(m))" if m == "ppml" or r.cov_type == "HC0" else "summary(m)"]
        L.append("exp(coef(m))   # incidence-rate ratios")
    elif m in ("zip", "zinb"):
        inf = [c.replace("inflate: ", "") for c in r.names if c.startswith("inflate: ") and c != "inflate: const"]
        L = _r_head(file_name, ["pscl"])
        L += [f"m <- zeroinfl({lhs} ~ {rhs} | {' + '.join(f'`{v}`' for v in inf) or '1'}, dist = \"{'poisson' if m == 'zip' else 'negbin'}\", data = df)",
              "summary(m)", "vuong(m, glm(" + f"{lhs} ~ {rhs}, family = poisson, data = df))" if m == "zip" else "summary(m)"]
    elif m in ("hurdle_p", "hurdle_nb"):
        L = _r_head(file_name, ["pscl"])
        L += [f"m <- hurdle({lhs} ~ {rhs}, dist = \"{'poisson' if m == 'hurdle_p' else 'negbin'}\", zero.dist = \"{'binomial' if r.extra.get('zero_dist', 'logit') == 'logit' else 'poisson'}\", data = df)",
              "summary(m)"]
    elif m == "cox":
        e = r.extra.get("event_name", "event")
        L = _r_head(file_name, ["survival"])
        L += [f"m <- coxph(Surv({lhs}, `{e}`) ~ {rhs}, ties = \"{r.extra['ties']}\", data = df)", "summary(m)",
              "cox.zph(m, transform = 'rank')   # proportional-hazards test"]
    elif m.startswith("aft_"):
        e = r.extra.get("event_name", "event")
        L = _r_head(file_name, ["survival"])
        L += [f"m <- survreg(Surv({lhs}, `{e}`) ~ {rhs}, dist = \"{r.extra['dist']}\", data = df)", "summary(m)",
              "exp(coef(m))   # time ratios"]
    else:
        L = _r_head(file_name, ["stats"]) + [f"# formula: {lhs} ~ {rhs}"]
    return "\n".join(L) + "\n"


def code_stata(r: rg.RegResult, file_name: str = "your_data.csv") -> str:
    m = r.method
    if "Z" in r.extra:
        return _stata_iv(r, file_name)
    prep, yv, rhs = _stata_prep(r, file_name)
    L = list(prep)
    if m in ("logit", "probit", "cloglog"):
        L += [f"{m} {yv} {rhs}" + (", vce(robust)" if r.cov_type == "HC0" else ""), "margins, dydx(*)"]
        if m == "logit":
            L += ["logit, or", "estat classification", "lroc", "estat gof, group(10)", "linktest"]
    elif m == "lpm":
        L += [f"regress {yv} {rhs}, vce(robust)"]
    elif m in ("mnlogit", "mnprobit"):
        cats = r.extra["categories"]
        bpos = sorted(cats).index(cats[0]) + 1
        L += [f"encode {yv}, gen({yv[:28]}_n)   // outcomes numbered alphabetically; base = {cats[0]}"]
        if m == "mnlogit":
            L += [f"mlogit {yv[:28]}_n {rhs}, baseoutcome({bpos})", "mlogit, rrr", "margins, dydx(*)"]
        elif not r.extra.get("correlated"):
            L += [f"mprobit {yv[:28]}_n {rhs}, baseoutcome({bpos})", "margins, dydx(*)"]
        else:
            L += ["gen long id = _n", f"expand {len(cats)}", "bysort id: gen alt = _n",
                  f"gen chosen = (alt == {yv[:28]}_n)", "cmset id alt",
                  f"cmmprobit chosen, casevars({rhs}) basealternative({bpos}) intpoints(200)", "estat covariance"]
    elif m in ("ologit", "oprobit"):
        L += [f"{m} {yv} {rhs}", "margins, dydx(*)",
              "* parallel-lines test: ssc install spost13_ado; brant, detail" if m == "ologit" else ""]
    elif m == "frac":
        L += [f"fracreg {r.extra.get('link', 'logit')} {yv} {rhs}", "margins, dydx(*)"]
    elif m == "tobit":
        L += [f"tobit {yv} {rhs}, ll({r.extra.get('left', 0)})", "margins, dydx(*) predict(ystar(0,.))",
              "margins, dydx(*) predict(e(0,.))"]
    elif m == "truncreg":
        L += [f"truncreg {yv} {rhs}, ll({r.extra.get('lower', 0)})"]
    elif m in ("heckman2", "heckman_ml"):
        sel = r.extra["selection_spec"]
        L += [f"heckman {yv} {rhs}, select({S(sel.y)} = {' '.join(S(v) for v in sel.x)})" +
              (" twostep" if m == "heckman2" else "")]
    elif m in ("cragg", "dhurdle"):
        L += ["* ssc install craggit (Cragg) / ssc install dblhurdle (double hurdle)",
              f"craggit {yv} {rhs}, second({rhs})" if m == "cragg" else f"dblhurdle {yv} {rhs}, peq({rhs}) ll(0)"]
    elif m in ("poisson", "ppml"):
        L += [f"poisson {yv} {rhs}" + (", vce(robust)" if m == "ppml" or r.cov_type == "HC0" else ""), "poisson, irr",
              "estat gof", "margins, dydx(*)"]
    elif m == "negbin":
        L += [f"nbreg {yv} {rhs}", "nbreg, irr", "margins, dydx(*)"]
    elif m in ("zip", "zinb"):
        inf = [S(c.replace("inflate: ", "")) for c in r.names if c.startswith("inflate: ") and c != "inflate: const"]
        L += [f"{m} {yv} {rhs}, inflate({' '.join(inf) or '_cons'})"]
    elif m in ("hurdle_p", "hurdle_nb"):
        L += ["* hurdle count models: ssc install hplogit / hnblogit (logit zero hurdle)",
              f"{'hplogit' if m == 'hurdle_p' else 'hnblogit'} {yv} {rhs}"]
    elif m == "cox":
        e = S(r.extra.get("event_name", "event"))
        L += [f"stset {yv}, failure({e})", f"stcox {rhs}, {r.extra['ties']}", "estat phtest, detail",
              "sts graph", "estat concordance"]
    elif m.startswith("aft_"):
        e = S(r.extra.get("event_name", "event"))
        dist = {"weibull": "weibull", "exponential": "exponential", "lognormal": "lognormal",
                "loglogistic": "loglogistic"}[r.extra["dist"]]
        L += [f"stset {yv}, failure({e})", f"streg {rhs}, distribution({dist}) time", "streg, tr   // time ratios"]
    return "\n".join(x for x in L if x is not None) + "\n"


# ==========================================================
# IV (single equation)
# ==========================================================
def _iv_parts(r):
    spec = r.extra["spec"]
    return spec, r.extra["exog_cols"], r.extra["endog_cols"], r.extra["excl_cols"]


def _py_iv(r, file_name):
    spec, exog, endog, excl = _iv_parts(r)
    L = _py_prep(r, file_name)
    zf = []
    for c in excl:
        v = r.design.info["base"].get(c, c)
        v = next((vv for vv in spec.instruments if rg._tname(vv, rg.x_transform(spec.form) if vv not in spec.nolog else "level") == c), c)
        how = rg.x_transform(spec.form) if v not in spec.nolog else "level"
        zf.append(f'Zx["{c}"] = ' + ({"log": f'np.log(df["{v}"])', "recip": f'1 / df["{v}"]'}.get(how, f'df["{v}"]')))
    cls = {"2sls": "IV2SLS", "liml": "IVLIML", "fuller": "IVLIML", "gmm": "IVGMM", "ols": "IV2SLS"}[r.method]
    cov = {"nonrobust": "unadjusted", "HC0": "robust", "HC1": "robust", "HC2": "robust", "HC3": "robust",
           "cluster": "clustered", "HAC": "kernel"}.get(r.cov_type, "unadjusted")
    L += ["# pip install linearmodels", f"from linearmodels.iv import {cls}", "Zx = pd.DataFrame(index=X.index)"] + zf
    L += [f"mod = {cls}(y, X[{exog!r}], X[{endog!r}], Zx.loc[X.index]" + (", fuller=1" if r.method == "fuller" else "") + ")",
          f'res = mod.fit(cov_type="{cov}", debiased={r.extra.get("small", True)})', "print(res.summary)",
          "print(res.first_stage)", "print(res.sargan); print(res.wu_hausman())"]
    return "\n".join(L) + "\n"


def _r_iv(r, file_name):
    spec, exog, endog, excl = _iv_parts(r)
    lhs, rhs = _r_formula(r)
    terms = rhs.split(" + ")
    endog_t = [t for t in terms if any(e in t for e in [f"`{v}`" for v in spec.endog])]
    exog_t = [t for t in terms if t not in endog_t]
    z_t = []
    for v in spec.instruments:
        how = rg.x_transform(spec.form) if v not in spec.nolog else "level"
        z_t.append({"log": f"I(log(`{v}`))", "recip": f"I(1/`{v}`)"}.get(how, f"`{v}`"))
    L = _r_head(file_name, ["AER", "sandwich", "lmtest"])
    if r.method in ("2sls", "ols"):
        L += [f"m <- ivreg({lhs} ~ {' + '.join(exog_t + endog_t)} | {' + '.join(exog_t + z_t)}, data = df)",
              "summary(m, diagnostics = TRUE)   # weak instruments, Wu-Hausman, Sargan",
              {"nonrobust": "", "cluster": f"coeftest(m, vcov. = vcovCL(m, cluster = ~ `{r.cov_opts.get('groups_name', 'cluster')}`))"}.get(
                  r.cov_type, f'coeftest(m, vcov. = vcovHC(m, type = "{r.cov_type if r.cov_type.startswith("HC") else "HC1"}"))')]
    else:
        L += ["# LIML / Fuller / GMM: install.packages('ivreg') or use the gmm package",
              f"library(ivreg); m <- ivreg({lhs} ~ {' + '.join(exog_t + endog_t)} | {' + '.join(exog_t + z_t)}, data = df, method = '{'mm' if r.method == 'gmm' else '2sls'}')",
              "summary(m)"]
    return "\n".join(x for x in L if x) + "\n"


def _stata_iv(r, file_name):
    spec, exog, endog, excl = _iv_parts(r)
    prep, yv, rhs = _stata_prep(r, file_name)
    terms = rhs.split()
    endog_s = [t for t in terms if any(S(v) in t for v in spec.endog)]
    exog_s = [t for t in terms if t not in endog_s]
    L = list(prep)
    z_s = []
    for v in spec.instruments:
        how = rg.x_transform(spec.form) if v not in spec.nolog else "level"
        if how == "log":
            L.append(f"gen ln_{S(v)} = ln({S(v)})")
            z_s.append(f"ln_{S(v)}")
        else:
            z_s.append(S(v))
    est = {"2sls": "2sls", "liml": "liml", "fuller": "liml", "gmm": "gmm", "ols": "2sls"}[r.method]
    vce = {"nonrobust": "", "HC0": " vce(robust)", "HC1": " vce(robust)", "HC2": " vce(robust)", "HC3": " vce(robust)",
           "cluster": f" vce(cluster {S(r.cov_opts.get('groups_name', 'cluster_var'))})", "HAC": " vce(hac bartlett)"}.get(r.cov_type, "")
    L += [f"ivregress {est} {yv} {' '.join(exog_s)} ({' '.join(endog_s)} = {' '.join(z_s)}),{vce} small" +
          (" // Fuller: user-written ivreg2 ..., fuller(1)" if r.method == "fuller" else ""),
          "estat firststage", "estat overid", "estat endogenous"]
    return "\n".join(L) + "\n"


def system_code(sres, equations, exog_all, file_name: str = "your_data.csv") -> dict:
    """Python (linearmodels), R (systemfit) and Stata (reg3 / sureg) code for a system of equations."""
    meth = sres.method
    py = ["import pandas as pd", "from collections import OrderedDict",
          "from linearmodels.system import IV3SLS, SUR", f'df = pd.read_csv("{file_name}")', "df['const'] = 1.0",
          "eqs = OrderedDict()"]
    for e in equations:
        inst = [v for v in exog_all if v not in e.exog]
        if meth in ("sur", "itsur") or not e.endog:
            py.append(f"eqs['{e.name}'] = {{'dependent': df['{e.y}'], 'exog': df[{['const'] + e.endog + e.exog!r}]}}")
        else:
            py.append(f"eqs['{e.name}'] = {{'dependent': df['{e.y}'], 'exog': df[{['const'] + e.exog!r}], "
                      f"'endog': df[{e.endog!r}], 'instruments': df[{inst!r}]}}")
    py.append("res = " + ("SUR(eqs).fit(method='gls'" + (", iterate=True" if meth == "itsur" else "") + ")" if meth in ("sur", "itsur")
                          else "IV3SLS(eqs).fit(" + ("method='ols'" if meth in ("2sls", "ols", "ils") else "") + ")"))
    py.append("print(res.summary)")
    rr = ["# install.packages('systemfit')", "library(systemfit)", f'df <- read.csv("{file_name}", check.names = FALSE)']
    eqs_r = [f"{S(e.name)} = `{e.y}` ~ {' + '.join(f'`{v}`' for v in e.endog + e.exog) or '1'}" for e in equations]
    rr.append(f"eqs <- list({', '.join(eqs_r)})")
    rm = {"ols": "OLS", "2sls": "2SLS", "ils": "2SLS", "3sls": "3SLS", "sur": "SUR", "itsur": "SUR"}[meth]
    inst = f", inst = ~ {' + '.join(f'`{v}`' for v in exog_all)}" if rm in ("2SLS", "3SLS") else ""
    rr.append(f"m <- systemfit(eqs, method = \"{rm}\"{inst}, data = df, methodResidCov = \"noDfCor\""
              + (", maxit = 1000, tol = 1e-12" if meth == "itsur" else "") + ")")
    rr.append("summary(m)")
    st = [f'import delimited "{file_name}", clear varnames(1) case(preserve)']
    eqs_s = " ".join(f"({S(e.y)} {' '.join(S(v) for v in e.endog + e.exog)})" for e in equations)
    if meth in ("sur", "itsur"):
        st.append(f"sureg {eqs_s}" + (", isure" if meth == "itsur" else ""))
    else:
        st.append(f"reg3 {eqs_s}, exog({' '.join(S(v) for v in exog_all)}) endog({' '.join(sorted({S(v) for e in equations for v in e.endog}))})"
                  + (", 2sls" if meth in ("2sls", "ils") else ", ols" if meth == "ols" else ""))
    return {"Python": "\n".join(py) + "\n", "R": "\n".join(rr) + "\n", "Stata": "\n".join(st) + "\n"}
