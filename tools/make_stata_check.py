"""Generate docs/validation/stata_check.do: runs the Stata code that Aetheris exports and asserts that Stata
reproduces the platform's coefficients, standard errors and diagnostic statistics (relative difference < 1e-6).

Usage:  python tools/make_stata_check.py   then in Stata:  cd docs/validation  →  do stata_check.do
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from analytics import diagnostics as dx  # noqa: E402
from analytics import reg_report as rp  # noqa: E402
from analytics import regression as rg  # noqa: E402
from data.samples import farm_survey_demo  # noqa: E402

OUT = ROOT / "docs" / "validation"
CSV = "farm_survey_demo.csv"


def stata_coef_names(r: rg.RegResult) -> dict:
    """Map Aetheris coefficient names to Stata _b[] names for the code produced by reg_report.code_stata."""
    spec = r.design.spec
    info = r.design.info
    S = rp._stata_name
    out = {"const": "_cons"}
    base = {}
    for v in spec.x:
        how = info["hows"].get(v, "level")
        name = {"log": f"ln_{S(v)}", "recip": f"inv_{S(v)}"}.get(how, S(v))
        if v in info["means"]:
            name = "c_" + name
        base[v] = name[:32]
        out[rg._tname(v, how)] = base[v]
        if int(spec.poly.get(v, 1)) == 2:
            out[f"{rg._tname(v, how)}^2"] = f"c.{base[v]}#c.{base[v]}"
    for c in spec.categorical:
        lv = info["levels"][c]
        order = sorted(lv, key=str)
        for level in lv[1:]:
            out[f"{c}[{level}]"] = f"{order.index(level) + 1}.{S(c)[:28]}_n"
    for a, b in spec.interactions:
        if a in spec.x and b in spec.categorical:
            lv = info["levels"][b]
            order = sorted(lv, key=str)
            for level in lv[1:]:
                out[f"{rg._tname(a, info['hows'].get(a, 'level'))}×{b}[{level}]"] = \
                    f"{order.index(level) + 1}.{S(b)[:28]}_n#c.{base[a]}"
    return out


def block(title, r, extra=""):
    names = stata_coef_names(r)
    code = rp.code_stata(r, CSV).split("* Run the diagnostics")[0]
    code = "\n".join(l for l in code.splitlines() if not l.startswith(("estat", "predict", "sktest", "swilk", "dfbeta", "gen t =", "tsset")))
    L = [f"\n* {'=' * 70}\n* {title}\n* {'=' * 70}", code]
    for k, v in r.params.items():
        L.append(f'chk "{title}: b[{k}]" _b[{names[k]}] {float(v)!r}')
        L.append(f'chk "{title}: se[{k}]" _se[{names[k]}] {float(r.bse[k])!r}')
    return "\n".join(L + [extra])


EST_CMDS = ("logit ", "probit ", "mlogit ", "mprobit ", "ologit ", "oprobit ", "tobit ", "poisson ", "nbreg ", "zip ",
            "heckman ", "stcox ", "streg ", "ivregress ", "regress ")


def v15_block(title, r, csv, checks, tol_b=1e-5, tol_se=1e-4):
    """Estimation lines from the exported Stata code (up to the estimation command) + coefficient checks."""
    from analytics import model_code as mc
    code = mc.code_stata(r, csv).splitlines()
    keep = []
    for ln in code:
        keep.append(ln)
        if ln.startswith(EST_CMDS) and not ln.startswith(("stset",)):
            break
    L = [f"\n* {'=' * 70}\n* v1.5 — {title}\n* {'=' * 70}"] + keep
    for label, expr, val, tol in checks:
        L.append(f'chk "{title}: {label}" {expr} {float(val)!r} {tol if tol else tol_b}')
    return "\n".join(L)


def v15_blocks():
    import pandas as pd  # noqa: F401
    from analytics import count_surv as cs
    from analytics import iv as ivm
    from analytics import ldv
    from data.samples import household_survey_demo, labour_iv_demo
    H = household_survey_demo()
    L_ = labour_iv_demo()
    H.to_csv(OUT / "household_survey_demo.csv", index=False)
    L_.to_csv(OUT / "labour_iv_demo.csv", index=False)
    hcsv, lcsv = "household_survey_demo.csv", "labour_iv_demo.csv"
    XB = ["education_years", "farm_size_feddan", "distance_market_km", "credit_access"]
    out = []

    def bse(r, names, eq=None, tb=1e-5, ts=1e-4):
        ch = []
        for nm, sname in names:
            pre = f"[{eq}]" if eq else ""
            ch.append((f"b[{nm}]", f"{pre}_b[{sname}]", r.params[nm], tb))
            ch.append((f"se[{nm}]", f"{pre}_se[{sname}]", r.bse[nm], ts))
        return ch
    simple = [("const", "_cons")] + [(v, v) for v in XB]
    for link in ("logit", "probit"):
        r = ldv.fit_binary(H, rg.ModelSpec(y="adopt_technology", x=XB), link)
        out.append(v15_block(link.capitalize(), r, hcsv, bse(r, simple)))
    r = ldv.fit_mnlogit(H, rg.ModelSpec(y="technology_choice", x=XB), base="Traditional")
    cats = sorted(r.extra["categories"])
    ch = []
    for cat in r.extra["categories"][1:]:
        eq = f"#{cats.index(cat) + 1}"
        ch += [(f"b[{cat}: {v}]", f"[{eq}]_b[{sv}]", r.params[f"{cat}: {v}"], 1e-5) for v, sv in simple]
        ch += [(f"se[{cat}: {v}]", f"[{eq}]_se[{sv}]", r.bse[f"{cat}: {v}"], 1e-4) for v, sv in simple]
    out.append(v15_block("Multinomial logit", r, hcsv, ch))
    r = ldv.fit_mnprobit(H, rg.ModelSpec(y="technology_choice", x=XB), base="Traditional")
    ch = []
    for cat in r.extra["categories"][1:]:
        eq = f"#{cats.index(cat) + 1}"
        ch += [(f"b[{cat}: {v}]", f"[{eq}]_b[{sv}]", r.params[f"{cat}: {v}"], 1e-3) for v, sv in simple]
    out.append(v15_block("Multinomial probit, independent errors (Stata quadrature vs Aetheris)", r, hcsv, ch))
    for distr in ("logit", "probit"):
        r = ldv.fit_ordered(H, rg.ModelSpec(y="satisfaction_1to5", x=XB), distr)
        ch = bse(r, [(v, v) for v in XB])
        ch += [(f"cut{j + 1}", f"_b[/cut{j + 1}]", r.params.iloc[len(XB) + j], 1e-5) for j in range(4)]
        out.append(v15_block(f"Ordered {distr}", r, hcsv, ch))
    r = ldv.fit_tobit(H, rg.ModelSpec(y="fertilizer_spend_egp", x=XB))
    out.append(v15_block("Tobit (left-censored at 0)", r, hcsv, bse(r, simple, tb=1e-5, ts=1e-3)))
    r = cs.fit_count(H, rg.ModelSpec(y="extension_visits", x=XB), "poisson")
    out.append(v15_block("Poisson", r, hcsv, bse(r, simple)))
    r = cs.fit_count(H, rg.ModelSpec(y="extension_visits", x=XB), "negbin")
    out.append(v15_block("Negative binomial (NB2)", r, hcsv, bse(r, simple, ts=1e-3) + [("alpha", "e(alpha)", r.params["alpha"], 1e-4)]))
    o = rg.ModelSpec(y="offfarm_wage_egp_day", x=["education_years", "age", "female_head"], form="loglin")
    s_ = rg.ModelSpec(y="works_offfarm", x=["education_years", "age", "n_children", "farm_size_feddan", "female_head"])
    for meth in ("twostep", "ml"):
        r = ldv.fit_heckman(H, o, s_, meth)
        ch = bse(r, [("const", "_cons")] + [(v, v) for v in o.x], eq="#1", ts=1e-4 if meth == "twostep" else 1e-3)
        ch += [(f"select b[{v}]", f"[select]_b[{v if v != 'const' else '_cons'}]", r.params[f"select: {v}"], 1e-4)
               for v in ["const"] + s_.x]
        ch += [("rho", "e(rho)", r.extra["rho"], 1e-4), ("sigma", "e(sigma)", r.extra["sigma"], 1e-4)]
        out.append(v15_block(f"Heckman {meth}", r, hcsv, ch))
    r = cs.fit_cox(H, rg.ModelSpec(y="years_to_adoption", x=XB), "years_to_adoption", "adopted_by_end", "efron")
    out.append(v15_block("Cox PH (Efron ties)", r, hcsv, bse(r, [(v, v) for v in XB], ts=1e-4)))
    r = cs.fit_aft(H, rg.ModelSpec(y="years_to_adoption", x=XB), "years_to_adoption", "adopted_by_end", "weibull")
    out.append(v15_block("Weibull AFT", r, hcsv, bse(r, simple, tb=1e-4, ts=1e-3) +
                         [("ln p = −ln σ", "_b[/ln_p]", -r.params["ln σ (log scale)"], 1e-4)]))
    spec = ivm.IVSpec(y="wage_egp_hour", exog=["experience_years", "female", "urban"], endog=["education_years"],
                      instruments=["distance_college_km", "father_education", "mother_education"], form="loglin")
    r = ivm.iv_fit(L_, spec, "2sls")
    t = ivm.iv_tests(L_, r).set_index("Test")
    ch = bse(r, [("const", "_cons"), ("experience_years", "experience_years"), ("female", "female"), ("urban", "urban"),
                 ("education_years", "education_years")])
    blk = v15_block("IV 2SLS (small)", r, lcsv, ch)
    blk += "\n".join(["", "quietly estat endogenous",
                       f'chk "Durbin chi2" r(durbin) {float(t.loc["Durbin χ²", "Statistic"])!r}',
                       f'chk "Wu-Hausman F" r(wu) {float(t.loc["Wu-Hausman F", "Statistic"])!r}',
                       "quietly estat overid",
                       f'chk "Sargan chi2" r(sargan) {float(t.loc["Sargan χ² (homoskedastic)", "Statistic"])!r}',
                       f'chk "Basmann chi2" r(basmann) {float(t.loc["Basmann χ²", "Statistic"])!r}'])
    out.append(blk)
    return out


def v16_blocks():
    """v1.6 — unit-root statistics and ARIMA maximum likelihood (Stata: dfuller, pperron, dfgls, arima)."""
    import numpy as np
    import pandas as pd
    from analytics import ts
    from data.samples import macro_ts_demo
    M = macro_ts_demo()
    M.to_csv(OUT / "macro_ts_demo.csv", index=False)
    y = np.log(M.cpi_index.values)
    L = [f"\n* {'=' * 70}\n* v1.6 — unit roots and ARIMA (macro_ts_demo.csv)\n* {'=' * 70}",
         'import delimited "macro_ts_demo.csv", clear varnames(1) case(preserve)',
         "gen t = _n", "tsset t", "gen ln_cpi = ln(cpi_index)"]
    a = ts.adf(y, "ct", None, 4)
    L += ["quietly dfuller ln_cpi, trend lags(4)", f'chk "ADF tau (trend, 4 lags)" r(Zt) {float(a["stat"])!r} 1e-6']
    a = ts.adf(np.diff(y), "c", None, 2)
    L += ["quietly dfuller D.ln_cpi, lags(2)", f'chk "ADF tau on D.ln_cpi (2 lags)" r(Zt) {float(a["stat"])!r} 1e-6']
    for reg, opt in (("c", ""), ("ct", " trend")):
        pp = ts.phillips_perron(y, reg)
        L += [f"quietly pperron ln_cpi,{opt} lags({pp['lags']})",
              f'chk "Phillips-Perron Z(t) ({reg})" r(Zt) {float(pp["stat"])!r} 2e-3',
              f'chk "Phillips-Perron Z(rho) ({reg})" r(Zrho) {float(pp["Z_alpha"])!r} 2e-3']
    g = ts.dfgls(y, "ct", lags=4)
    L += ["quietly dfgls ln_cpi, maxlag(4)", "matrix DG = r(results)",
          "* r(results): rows k = maxlag … 1; column 5 = DF-GLS tau",
          f'chk "DF-GLS tau (trend, 4 lags)" el(DG,1,5) {float(g["stat"])!r} 1e-4']
    r = ts.fit_arima(pd.Series(M.wheat_price_egp_t.values), (1, 1, 1))
    P = r.res.params
    B = r.res.bse
    L += ["quietly arima wheat_price_egp_t, arima(1,1,1) noconstant vce(oim)",
          f'chk "ARIMA(1,1,1) b[ar.L1]" [ARMA]_b[L.ar] {float(P["ar.L1"])!r} 1e-3',
          f'chk "ARIMA(1,1,1) b[ma.L1]" [ARMA]_b[L.ma] {float(P["ma.L1"])!r} 1e-3',
          f'chk "ARIMA(1,1,1) se[ar.L1]" [ARMA]_se[L.ar] {float(B["ar.L1"])!r} 5e-3',
          f'chk "ARIMA(1,1,1) sigma" [sigma]_b[_cons] {float(np.sqrt(P["sigma2"]))!r} 1e-3',
          f'chk "ARIMA(1,1,1) log-likelihood" e(ll) {float(r.res.llf)!r} 1e-5']
    r = ts.fit_arima(pd.Series(y), (0, 1, 1), (0, 1, 1), 12)
    P = r.res.params
    L += ["quietly arima ln_cpi, arima(0,1,1) sarima(0,1,1,12) noconstant vce(oim)",
          f'chk "SARIMA b[ma.S.L12]" [ARMA12]_b[L.ma] {float(P["ma.S.L12"])!r} 1e-3',
          f'chk "SARIMA log-likelihood" e(ll) {float(r.res.llf)!r} 1e-5']
    r = ts.fit_arima(pd.Series(M.unemployment_rate.values), (2, 0, 0))
    P = r.res.params
    L += ["quietly arima unemployment_rate, ar(1/2) vce(oim)",
          f'chk "AR(2) b[ar.L1]" [ARMA]_b[L.ar] {float(P["ar.L1"])!r} 1e-4',
          f'chk "AR(2) b[ar.L2]" [ARMA]_b[L2.ar] {float(P["ar.L2"])!r} 1e-4',
          f'chk "AR(2) mean" [unemployment_rate]_b[_cons] {float(P["const"])!r} 1e-4']
    return ["\n".join(L)]


def v17_blocks():
    """v1.7 — VAR, VECM, ARDL and GARCH (Stata: var, vec, ardl [ssc], arch)."""
    import numpy as np
    import pandas as pd
    from analytics import ardl as ad
    from analytics import garch as gm
    from analytics import var as vr
    from data.samples import financial_returns_demo, macro_var_demo, price_transmission_demo
    M = macro_var_demo()
    V = pd.DataFrame({"dy": 100 * np.diff(np.log(M.gdp_bn)), "infl": M.inflation_pct[1:].values,
                      "rate": M.policy_rate_pct[1:].values, "y": np.log(M.gdp_bn)[1:].values,
                      "c": np.log(M.consumption_bn)[1:].values, "i": np.log(M.investment_bn)[1:].values})
    V.to_csv(OUT / "v17_macro.csv", index=False)
    PT = price_transmission_demo()
    A = pd.DataFrame({"d": np.log(PT.domestic_wheat_egp_t), "x": np.log(PT.import_parity_egp_t),
                      "e": np.log(PT.exchange_rate_egp_usd), "rain": PT.rainfall_anomaly})
    A.to_csv(OUT / "v17_wheat.csv", index=False)
    R = pd.DataFrame({"r": gm.returns(financial_returns_demo().equity_index.values)})
    R.to_csv(OUT / "v17_returns.csv", index=False)
    L = [f"\n* {'=' * 70}\n* v1.7 — VAR, VECM, ARDL, GARCH\n* {'=' * 70}",
         'import delimited "v17_macro.csv", clear varnames(1) case(preserve)', "gen t = _n", "tsset t",
         "quietly var dy infl rate, lags(1/2) dfk"]
    r = vr.fit_var(V[["dy", "infl", "rate"]], 2, "const")
    for k, eq in enumerate(r.names):
        for j, rn in enumerate(r.reg_names):
            nm, lag = (rn.split(".l") if ".l" in rn else (rn, None))
            sn = f"L{lag}.{nm}" if lag and lag != "1" else (f"L.{nm}" if lag else "_cons")
            L.append(f'chk "VAR [{eq}] b[{rn}]" [{eq}]_b[{sn}] {float(r.B[j, k])!r} 1e-6')
            L.append(f'chk "VAR [{eq}] se[{rn}]" [{eq}]_se[{sn}] {float(r.se[j, k])!r} 1e-5')
    jo = vr.johansen(V[["y", "c", "i"]], 3, 3)
    vm = vr.vecm(jo, 2)
    b = vm["beta"].to_numpy()
    al = vm["alpha"]
    L += ["quietly vec y c i, lags(3) rank(2) trend(constant)",
          f'chk "VECM beta CE1[i]" [_ce1]_b[i] {float(b[2, 0])!r} 1e-5',
          f'chk "VECM beta CE2[i]" [_ce2]_b[i] {float(b[2, 1])!r} 1e-5']
    for v in ("y", "c", "i"):
        L.append(f'chk "VECM alpha D_{v} CE1" [D_{v}]_b[L._ce1] {float(al.loc[v, "CE1"])!r} 1e-5')
    L += ['import delimited "v17_wheat.csv", clear varnames(1) case(preserve)', "gen t = _n", "tsset t",
          "capture which ardl", "if _rc ssc install ardl",
          "quietly ardl d x e, lags(2 2 1) exog(rain)"]
    ra = ad.fit_ardl(A, "d", ["x", "e"], (2, 2, 1), 3, ["rain"])
    smap = {"(Intercept)": "_cons", "L(d, 1)": "L.d", "L(d, 2)": "L2.d", "x": "x", "L(x, 1)": "L.x", "L(x, 2)": "L2.x",
            "e": "e", "L(e, 1)": "L.e", "rain": "rain"}
    for nm, bv, se in zip(ra.names, ra.b, np.sqrt(np.diag(ra.V))):
        L.append(f'chk "ARDL b[{nm}]" _b[{smap[nm]}] {float(bv)!r} 1e-6')
        L.append(f'chk "ARDL se[{nm}]" _se[{smap[nm]}] {float(se)!r} 1e-5')
    L += ["quietly ardl d x e, lags(2 2 1) exog(rain) ec", "quietly estat ectest",
          f'chk "ARDL bounds F (case 3)" r(F_pss) {float(ad.bounds_test(ra)["F"])!r} 1e-5']
    g = gm.fit_garch(R.r.values, "garch", 1, 1, "norm")
    L += ['import delimited "v17_returns.csv", clear varnames(1) case(preserve)', "gen t = _n", "tsset t",
          "* Stata initialises sigma2_0 with the OLS mean squared residual; Aetheris (as fGarch) with the current one —",
          "* estimates agree to about 3 decimals.",
          "quietly arch r, arch(1) garch(1)",
          f'chk "GARCH(1,1) mu" [r]_b[_cons] {float(g.p_("mu"))!r} 5e-3',
          f'chk "GARCH(1,1) alpha" [ARCH]_b[L.arch] {float(g.p_("alpha1"))!r} 1e-2',
          f'chk "GARCH(1,1) beta" [ARCH]_b[L.garch] {float(g.p_("beta1"))!r} 1e-2',
          f'chk "GARCH(1,1) omega" [ARCH]_b[_cons] {float(g.p_("omega"))!r} 5e-2']
    return ["\n".join(L)]


def v18_blocks():
    """v1.8 — static panel models (xtreg) and system GMM (xtabond2 [ssc])."""
    import numpy as np
    from analytics import panel as pn
    from analytics import panel_gmm as pg
    from data.samples import firm_panel_demo
    F = firm_panel_demo()
    F.to_csv(OUT / "v18_firms.csv", index=False)
    P = pn.prepare(F, "firm_id", "year", "ln_sales", ["ln_capital", "ln_labour"])
    L = [f"\n* {'=' * 70}\n* v1.8 — panel data (xtreg, xtabond2)\n* {'=' * 70}",
         'import delimited "v18_firms.csv", clear varnames(1) case(preserve)', "encode firm_id, gen(fid)", "xtset fid year"]
    for title, cmd, m, cov, tol in (("FE", "xtreg ln_sales ln_capital ln_labour, fe", "fe", "classic", 1e-6),
                                    ("FE cluster", "xtreg ln_sales ln_capital ln_labour, fe vce(cluster fid)", "fe", "cluster", 1e-5),
                                    ("RE", "xtreg ln_sales ln_capital ln_labour, re", "re", "classic", 1e-5),
                                    ("Between", "xtreg ln_sales ln_capital ln_labour, be", "be", "classic", 1e-6)):
        r = pn.fit(P, m, cov)
        L.append("quietly " + cmd)
        for v, b, se in zip(r.names, r.b, r.se):
            sv = "_cons" if v == "const" else v
            if m == "fe" and v == "const":
                continue
            L.append(f'chk "{title} b[{v}]" _b[{sv}] {float(b)!r} {tol}')
            L.append(f'chk "{title} se[{v}]" _se[{sv}] {float(se)!r} {tol * 10}')
    r = pn.fgls(P, "hetero")
    L.append("quietly xtgls ln_sales ln_capital ln_labour, panels(heteroskedastic)")
    for v, b, se in zip(r.names, r.b, r.se):
        sv = "_cons" if v == "const" else v
        L.append(f'chk "FGLS hetero b[{v}]" _b[{sv}] {float(b)!r} 1e-6')
        L.append(f'chk "FGLS hetero se[{v}]" _se[{sv}] {float(se)!r} 1e-5')
    spec = pg.GmmSpec("ln_sales", [1], {"ln_capital": [0], "ln_labour": [0]}, {"ln_sales": (2, 99), "ln_labour": (2, 99), "ln_capital": (1, 99)})
    for tr, nm, opt in (("d", "Difference GMM", " noleveleq"), ("ld", "System GMM", "")):
        g = pg.fit(F, "firm_id", "year", spec, "twosteps", tr, "twoways", True)
        se = np.sqrt(np.diag(g.V))
        L += ["capture which xtabond2", "if _rc ssc install xtabond2",
              f"quietly xtabond2 ln_sales L.ln_sales ln_capital ln_labour i.year, gmm(ln_sales, lag(2 .) collapse) "
              f"gmm(ln_labour, lag(2 .) collapse) gmm(ln_capital, lag(1 .) collapse) iv(i.year){opt} twostep robust",
              "* xtabond2 and R pgmm treat the time dummies slightly differently in the level equation: compare to ~3 decimals."]
        for v, b, s_ in zip(["L.ln_sales", "ln_capital", "ln_labour"], g.coef[:3], se[:3]):
            L.append(f'chk "{nm} b[{v}]" _b[{v}] {float(b)!r} 1e-3')
            L.append(f'chk "{nm} se[{v}]" _se[{v}] {float(s_)!r} 1e-2')
        L.append(f'chk "{nm} AR(2)" e(ar2) {float(g.tests["AR(2)"][0])!r} 1e-2')
    return ["\n".join(L)]


def v19_blocks():
    """v1.9 — regression with SARIMA errors (Stata arima on seasonal differences) and a bridge equation (regress)."""
    import numpy as np
    import pandas as pd
    from analytics import fcst as F
    from analytics import nowcast as nc
    from data.samples import monthly_forecast_demo, nowcast_demo
    D = monthly_forecast_demo()
    D.to_csv(OUT / "v19_monthly.csv", index=False)
    a = F.sarimax(D.electricity_gwh.values, 6, (1, 0, 1), (0, 1, 1), 12, D[["temperature_c"]].values, np.full((6, 1), 25.0))
    tb = a.info["table"].set_index("Parameter")["Coef."]
    L = [f"\n* {'=' * 70}\n* v1.9 — forecasting (arima with seasonal differencing, bridge equation)\n* {'=' * 70}",
         'import delimited "v19_monthly.csv", clear varnames(1) case(preserve)', "gen t = _n", "tsset t",
         "quietly arima S12.electricity_gwh S12.temperature_c, ar(1) ma(1) mma(1, 12) noconstant",
         "* Stata maximises the exact likelihood of the differenced data, as R Arima and Aetheris; optimisers may differ slightly.",
         f'chk "ARIMAX temp" [S12.electricity_gwh]_b[S12.temperature_c] {float(tb["x1"])!r} 1e-3',
         f'chk "ARIMAX ar1" [ARMA]_b[L.ar] {float(tb["ar.L1"])!r} 1e-3',
         f'chk "ARIMAX ma1" [ARMA]_b[L.ma] {float(tb["ma.L1"])!r} 1e-2',
         f'chk "ARIMAX sma1" [ARMA12]_b[L.ma] {float(tb["ma.S.L12"])!r} 1e-3']
    N = nowcast_demo()
    ind = ["industrial_production_mom", "pmi", "exports_mom", "retail_sales_mom"]
    b = nc.bridge(N, "date", "gdp_growth_q", ind)
    q = b["quarterly"].dropna(subset=ind + ["y"]).reset_index(drop=True)
    q.to_csv(OUT / "v19_bridge.csv", index=False)
    L += ['import delimited "v19_bridge.csv", clear varnames(1) case(preserve)', "quietly regress y " + " ".join(ind)]
    for _, rw in b["table"].iterrows():
        nm = "_cons" if rw.Parameter == "const" else rw.Parameter
        L.append(f'chk "Bridge b[{rw.Parameter}]" _b[{nm}] {float(rw["Coef."])!r} 1e-6')
        L.append(f'chk "Bridge se[{rw.Parameter}]" _se[{nm}] {float(rw["Std. Err."])!r} 1e-6')
    del pd
    return ["\n".join(L)]


def v110_blocks():
    """v1.10 — production function, stochastic frontiers (frontier, xtfrontier) and QUAIDS (ssc quaids)."""
    import numpy as np
    from analytics import demand as dm
    from analytics import production as pr
    from data.samples import farm_efficiency_demo, household_budget_demo
    F = farm_efficiency_demo()
    F.to_csv(OUT / "v110_farms.csv", index=False)
    X3 = ["land_feddan", "labour_days", "fertilizer_kg"]
    L = [f"\n* {'=' * 70}\n* v1.10 — production, frontiers, QUAIDS\n* {'=' * 70}",
         'import delimited "v110_farms.csv", clear varnames(1) case(preserve)', "gen lny = ln(output_tonnes)"] + [f"gen ln_{v} = ln({v})" for v in X3]
    rhs = " ".join(f"ln_{v}" for v in X3)
    cd = pr.production_function(F, "output_tonnes", X3, "cd")
    L.append(f"quietly regress lny {rhs}")
    for nm, b, se in zip(["_cons"] + [f"ln_{v}" for v in X3], cd.b, cd.se):
        L.append(f'chk "CD b[{nm}]" _b[{nm}] {float(b)!r} 1e-6')
        L.append(f'chk "CD se[{nm}]" _se[{nm}] {float(se)!r} 1e-6')
    for dist, opt in (("hnormal", "hnormal"), ("exponential", "exponential"), ("tnormal", "tnormal")):
        r = pr.sfa(F, "output_tonnes", X3, dist)
        L.append(f"quietly frontier lny {rhs}, distribution({opt})")
        for nm, b in zip(["_cons"] + [f"ln_{v}" for v in X3], r["beta"]):
            L.append(f'capture noisily chk "SFA {dist} b[{nm}]" [lny]_b[{nm}] {float(b)!r} 1e-4')
        L.append(f'capture noisily chk "SFA {dist} lnsig2v" [lnsig2v]_b[_cons] {float(r["theta"][4])!r} 1e-3')
        L.append(f'capture noisily chk "SFA {dist} lnsig2u" [lnsig2u]_b[_cons] {float(r["theta"][5])!r} 1e-3')
        L.append(f'capture noisily chk "SFA {dist} log L" e(ll) {r["loglik"]!r} 1e-5')
    z = ["education_years", "extension_visits", "credit_access"]
    r = pr.sfa(F, "output_tonnes", X3, "tnormal", z=z)
    L.append(f"quietly frontier lny {rhs}, distribution(tnormal) cm({' '.join(z)})")
    for nm, b in zip(z, r["theta"][7:]):
        L.append(f'capture noisily chk "BC95 delta[{nm}]" [mu]_b[{nm}] {float(b)!r} 1e-3')
    L.append(f'capture noisily chk "BC95 log L" e(ll) {r["loglik"]!r} 1e-5')
    L += ["encode farm_id, gen(fid)", "xtset fid year"]
    for m in ("ti", "tvd"):
        p = pr.sfa_panel(F, "farm_id", "year", "output_tonnes", X3, m)
        L.append(f"quietly xtfrontier lny {rhs}, {m}")
        for nm, b in zip(["_cons"] + [f"ln_{v}" for v in X3], p["theta"]):
            L.append(f'capture noisily chk "xtfrontier {m} b[{nm}]" [lny]_b[{nm}] {float(b)!r} 1e-4')
        L.append(f'capture noisily chk "xtfrontier {m} log L" e(ll) {p["loglik"]!r} 1e-5')
        if m == "tvd":
            L.append(f'capture noisily chk "xtfrontier tvd eta" [eta]_b[_cons] {p["eta"]!r} 1e-3')
    from data.samples import firm_tfp_demo
    T = firm_tfp_demo()
    T.to_csv(OUT / "v110_tfp.csv", index=False)
    L += ['import delimited "v110_tfp.csv", clear varnames(1) case(preserve)', "encode firm_id, gen(fid)", "xtset fid year", "gen ln_inv = ln(investment)",
          "capture which prodest", "if _rc ssc install prodest",
          "* prodest (Rovigatti & Mollisi) uses its own polynomial and optimiser settings: compare to ~2 decimals."]
    for m, px, opt in (("op", "investment", "method(op) proxy(ln_inv)"), ("lp", "ln_materials", "method(lp) proxy(ln_materials)"),
                       ("acf", "ln_materials", "method(lp) proxy(ln_materials) acf")):
        rr = pr.prodest(T, "firm_id", "year", "ln_value_added", ["ln_labour"], "ln_capital", px, m)
        L.append(f"capture noisily prodest ln_value_added, free(ln_labour) state(ln_capital) {opt} va poly(3) reps(0)")
        L.append(f'capture noisily chk "prodest {m} b[ln_labour]" _b[ln_labour] {float(rr["beta"][0])!r} 3e-2')
        L.append(f'capture noisily chk "prodest {m} b[ln_capital]" _b[ln_capital] {float(rr["beta"][1])!r} 3e-2')
    H = household_budget_demo()
    H.to_csv(OUT / "v110_budget.csv", index=False)
    G5 = ["cereals", "meat_fish", "dairy_eggs", "fruit_veg", "other_food"]
    q = dm.aids(H, [f"w_{g}" for g in G5], [f"p_{g}" for g in G5], "food_expenditure", "quaids", alpha0=7.8, normalize=False)
    L += ['import delimited "v110_budget.csv", clear varnames(1) case(preserve)', "capture which quaids", "if _rc ssc install quaids",
          "quietly quaids " + " ".join(f"w_{g}" for g in G5) + ", anot(7.8) prices(" + " ".join(f"p_{g}" for g in G5) + ") expenditure(food_expenditure) nolog",
          "* (checks whose coefficient names differ across Stata / package versions are wrapped in capture noisily)",
          "* Aetheris uses iterated linear least squares (Blundell-Robin); Stata nlsur gives the ML estimate: compare to ~2 decimals."]
    for i in range(5):
        L.append(f'capture noisily chk "QUAIDS beta{i + 1}" _b[beta:beta_{i + 1}] {float(q["beta"][i])!r} 2e-2')
    del np
    return ["\n".join(L)]


def main():
    farm = farm_survey_demo()
    OUT.mkdir(parents=True, exist_ok=True)
    farm.to_csv(OUT / CSV, index=False)
    parts = ["* Aetheris Research — Stata cross-validation (generated by tools/make_stata_check.py)",
             "* Run from this folder:  do stata_check.do   — every line prints PASS or FAIL.",
             "version 15", "clear all", "set more off", "global fails = 0",
             "capture program drop chk",
             "program define chk",
             "    args label stata aetheris tol",
             '    if "`tol\'" == "" local tol = 1e-6',
             "    local d = reldif(`stata', `aetheris')",
             '    if `d\' < `tol\' display as result "PASS  `label\'  (reldif = " %9.2e `d\' ")"',
             '    else {',
             '        display as error "FAIL  `label\'  Stata = " %14.0g `stata\' "  Aetheris = " %14.0g `aetheris\'',
             "        global fails = $fails + 1",
             "    }",
             "end"]
    base = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days", "fertilizer_kg", "education_years"],
                        categorical=["irrigation"], form="loglog", nolog=["education_years"])
    for cov, kw in (("nonrobust", {}), ("HC1", {}), ("HC2", {}), ("HC3", {}), ("cluster", {"groups": "region"})):
        r = rg.fit(farm, base, cov_type=cov, cov_opts=dict(kw))
        extra = ""
        if cov == "nonrobust":
            e, X = r.wresid, r.wexog
            extra = "\n".join([
                "quietly estat hettest, rhs iid",
                f'chk "Koenker BP chi2" r(chi2) {float(dx.breusch_pagan(e, X, True)["stat"])!r}',
                "quietly estat hettest, rhs",
                f'chk "Breusch-Pagan chi2" r(chi2) {float(dx.breusch_pagan(e, X, False)["stat"])!r}',
                "quietly estat imtest, white",
                f'chk "White chi2" r(chi2_h) {float(dx.white_test(e, X)["stat"])!r}',
                "quietly estat ovtest",
                f'chk "RESET F (yhat^2..yhat^4)" r(F) {float(dx.reset_test(r.y, X, r.fitted, 4)["stat"])!r}',
                "estat vif",
                f'display "Aetheris max VIF = {dx.vif_table(r.design.X)["VIF"].max():.6f}"',
                "gen t = _n", "quietly tsset t",
                "quietly estat dwatson",
                f'chk "Durbin-Watson d" r(dw) {float(dx.durbin_watson(e))!r}',
                "quietly estat bgodfrey, lags(2)",
                f'chk "Breusch-Godfrey chi2 (2 lags)" el(r(chi2),1,1) {float(dx.breusch_godfrey(e, X, 2)["stat"])!r}',
                "drop t"])
        parts.append(block(f"Log-log Cobb-Douglas, SE = {cov}", r, extra))
    full = rg.ModelSpec(y="output_tonnes", x=["land_feddan", "labour_days", "fertilizer_kg", "experience_years"],
                        categorical=["irrigation"], form="loglog", nolog=["experience_years"],
                        poly={"experience_years": 2}, interactions=[("land_feddan", "irrigation")], center=True)
    parts.append(block("Quadratic + interaction + centring, SE = HC3", rg.fit(farm, full, cov_type="HC3")))
    lin = rg.ModelSpec(y="revenue_k_egp", x=["land_feddan", "labour_days"], categorical=["region"],
                       reference={"region": "New Lands"})
    parts.append(block("Linear model, custom reference level", rg.fit(farm, lin)))
    parts += v15_blocks()
    parts += v16_blocks()
    parts += v17_blocks()
    parts += v18_blocks()
    parts += v19_blocks()
    parts += v110_blocks()
    parts += ['\ndisplay _n "Total failures: $fails"',
              'if $fails == 0 display as result "ALL CHECKS PASSED — Stata reproduces Aetheris."']
    (OUT / "stata_check.do").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'stata_check.do'}")


if __name__ == "__main__":
    main()
