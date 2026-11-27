# ==========================================
# Aetheris Research - Reporting for panel data (v1.8, §1.15)
# Model-choice logic, interpretation (EN/AR), methodology paragraphs, references and Python / R / Stata code.
# ==========================================

from __future__ import annotations

import numpy as np

REFS = {
    "baltagi": "Baltagi, B. H. (2021). Econometric analysis of panel data (6th ed.). Springer.",
    "wooldridge": "Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data (2nd ed.). MIT Press.",
    "croissant": "Croissant, Y., & Millo, G. (2008). Panel data econometrics in R: The plm package. Journal of Statistical Software, 27(2), 1–43.",
    "hausman": "Hausman, J. A. (1978). Specification tests in econometrics. Econometrica, 46(6), 1251–1271.",
    "bp": "Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. Review of Economic Studies, 47(1), 239–253.",
    "mundlak": "Mundlak, Y. (1978). On the pooling of time series and cross section data. Econometrica, 46(1), 69–85.",
    "sa": "Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. Econometrica, 40(2), 261–275.",
    "pesaran_cd": "Pesaran, M. H. (2004). General diagnostic tests for cross section dependence in panels (CESifo Working Paper No. 1229).",
    "bfk": "Baltagi, B. H., Feng, Q., & Kao, C. (2012). A Lagrange multiplier test for cross-sectional dependence in a fixed effects panel data model. Journal of Econometrics, 170(1), 164–177.",
    "drukker": "Drukker, D. M. (2003). Testing for serial correlation in linear panel-data models. Stata Journal, 3(2), 168–177.",
    "greene": "Greene, W. H. (2018). Econometric analysis (8th ed.). Pearson.",
    "py": "Pesaran, M. H., & Yamagata, T. (2008). Testing slope homogeneity in large panels. Journal of Econometrics, 142(1), 50–93.",
    "dk": "Driscoll, J. C., & Kraay, A. C. (1998). Consistent covariance matrix estimation with spatially dependent panel data. Review of Economics and Statistics, 80(4), 549–560.",
    "bk": "Beck, N., & Katz, J. N. (1995). What to do (and not to do) with time-series cross-section data. American Political Science Review, 89(3), 634–647.",
    "llc": "Levin, A., Lin, C.-F., & Chu, C.-S. J. (2002). Unit root tests in panel data: Asymptotic and finite-sample properties. Journal of Econometrics, 108(1), 1–24.",
    "ips": "Im, K. S., Pesaran, M. H., & Shin, Y. (2003). Testing for unit roots in heterogeneous panels. Journal of Econometrics, 115(1), 53–74.",
    "mw": "Maddala, G. S., & Wu, S. (1999). A comparative study of unit root tests with panel data and a new simple test. Oxford Bulletin of Economics and Statistics, 61(S1), 631–652.",
    "choi": "Choi, I. (2001). Unit root tests for panel data. Journal of International Money and Finance, 20(2), 249–272.",
    "hadri": "Hadri, K. (2000). Testing for stationarity in heterogeneous panel data. Econometrics Journal, 3(2), 148–161.",
    "cips": "Pesaran, M. H. (2007). A simple panel unit root test in the presence of cross-section dependence. Journal of Applied Econometrics, 22(2), 265–312.",
    "kao": "Kao, C. (1999). Spurious regression and residual-based tests for cointegration in panel data. Journal of Econometrics, 90(1), 1–44.",
    "pedroni99": "Pedroni, P. (1999). Critical values for cointegration tests in heterogeneous panels with multiple regressors. Oxford Bulletin of Economics and Statistics, 61(S1), 653–670.",
    "pedroni04": "Pedroni, P. (2004). Panel cointegration: Asymptotic and finite sample properties of pooled time series tests with an application to the PPP hypothesis. Econometric Theory, 20(3), 597–625.",
    "westerlund": "Westerlund, J. (2007). Testing for error correction in panel data. Oxford Bulletin of Economics and Statistics, 69(6), 709–748.",
    "ab": "Arellano, M., & Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. Review of Economic Studies, 58(2), 277–297.",
    "bb": "Blundell, R., & Bond, S. (1998). Initial conditions and moment restrictions in dynamic panel data models. Journal of Econometrics, 87(1), 115–143.",
    "windmeijer": "Windmeijer, F. (2005). A finite sample correction for the variance of linear efficient two-step GMM estimators. Journal of Econometrics, 126(1), 25–51.",
    "roodman": "Roodman, D. (2009). How to do xtabond2: An introduction to difference and system GMM in Stata. Stata Journal, 9(1), 86–136.",
    "pss99": "Pesaran, M. H., Shin, Y., & Smith, R. P. (1999). Pooled mean group estimation of dynamic heterogeneous panels. Journal of the American Statistical Association, 94(446), 621–634.",
    "ps95": "Pesaran, M. H., & Smith, R. (1995). Estimating long-run relationships from dynamic heterogeneous panels. Journal of Econometrics, 68(1), 79–113.",
    "cce": "Pesaran, M. H. (2006). Estimation and inference in large heterogeneous panels with a multifactor error structure. Econometrica, 74(4), 967–1012.",
    "bai": "Bai, J. (2009). Panel data models with interactive fixed effects. Econometrica, 77(4), 1229–1279.",
    "baing": "Bai, J., & Ng, S. (2002). Determining the number of factors in approximate factor models. Econometrica, 70(1), 191–221.",
    "hansen99": "Hansen, B. E. (1999). Threshold effects in non-dynamic panels: Estimation, testing, and inference. Journal of Econometrics, 93(2), 345–368.",
    "lz": "Love, I., & Zicchino, L. (2006). Financial development and dynamic investment behavior: Evidence from panel VAR. Quarterly Review of Economics and Finance, 46(2), 190–210.",
    "al": "Abrigo, M. R. M., & Love, I. (2016). Estimation of panel vector autoregression in Stata. Stata Journal, 16(3), 778–804.",
    "dh": "Dumitrescu, E.-I., & Hurlin, C. (2012). Testing for Granger non-causality in heterogeneous panels. Economic Modelling, 29(4), 1450–1460.",
}


def advice(tests: dict, lang: str = "en") -> str:
    """Pooled vs FE vs RE decision from the F, Breusch-Pagan and Hausman tests, plus the SE recommendation."""
    ar = lang == "ar"
    F, LM, H = tests.get("F"), tests.get("LM"), tests.get("H")
    out = []
    if F is not None:
        out.append(("اختبار F: " + ("الآثار الفردية معنوية ← FE أفضل من التجميعي." if F < 0.05 else "لا آثار فردية معنوية ← التجميعي مقبول."))
                   if ar else ("F test: " + ("unit effects are significant → FE beats pooled OLS." if F < 0.05 else "no significant unit effects → pooled OLS is acceptable.")))
    if LM is not None:
        out.append(("اختبار بروش-باجان: " + ("تباين الأثر العشوائي موجب ← RE أفضل من التجميعي." if LM < 0.05 else "لا أثر عشوائي ← التجميعي مقبول."))
                   if ar else ("Breusch-Pagan LM: " + ("random-effect variance > 0 → RE beats pooled OLS." if LM < 0.05 else "no random effect → pooled OLS is acceptable.")))
    if H is not None:
        out.append(("هاوسمان: " + ("الآثار مرتبطة بالمتغيرات المفسرة ← <b>الآثار الثابتة</b> (RE غير متسق)." if H < 0.05 else "لا ارتباط ← <b>الآثار العشوائية</b> متسقة وأكفأ."))
                   if ar else ("Hausman: " + ("effects are correlated with the regressors → <b>fixed effects</b> (RE is inconsistent)." if H < 0.05 else "no correlation → <b>random effects</b> are consistent and efficient.")))
    cd, ser, het = tests.get("CD"), tests.get("AR"), tests.get("HET")
    if cd is not None and cd < 0.05:
        out.append("⚠️ " + ("اعتماد مقطعي بين الوحدات ← استخدم أخطاء دريسكول-كراي المعيارية، وCCE واختبارات الجيل الثاني لجذور الوحدة (CIPS)." if ar else
                            "Cross-sectional dependence → use Driscoll-Kraay standard errors, CCE estimators and second-generation unit-root tests (CIPS)."))
    elif (ser is not None and ser < 0.05) or (het is not None and het < 0.05):
        out.append("⚠️ " + ("ارتباط ذاتي و/أو عدم تجانس تباين ← استخدم الأخطاء المعيارية العنقودية حسب الوحدة." if ar else
                            "Serial correlation and/or heteroskedasticity → use standard errors clustered by unit."))
    return "<br>".join(out)


def interpret_gmm(r, lang: str = "en") -> str:
    ar = lang == "ar"
    t = r.tests
    P = []
    a1, a2, J = t.get("AR(1)"), t.get("AR(2)"), t["Hansen J"]
    if ar:
        P.append(f"عدد الأدوات = {r.n_instr} مقابل {r.n_units} وحدة" + (" ⚠️ الأدوات أكثر من الوحدات: يضعف اختبار هانسن؛ استخدم «تجميع الأدوات» أو قلّل الإبطاءات." if r.n_instr > r.n_units else "."))
        if a2 is not None:
            P.append(f"AR(2) في الفروق: {a2[0]:.3f} (p = {a2[2]:.3f}) — " + ("لا ارتباط من الرتبة الثانية: الأدوات صالحة." if a2[2] >= 0.05 else "⚠️ ارتباط من الرتبة الثانية: الأدوات المبطأة مرتين غير صالحة؛ ابدأ من الإبطاء الثالث."))
        if a1 is not None:
            P.append(f"AR(1) = {a1[0]:.3f} (p = {a1[2]:.3f}) — " + ("متوقع أن يكون معنوياً." if a1[2] < 0.05 else "غير معنوي على غير المعتاد."))
        P.append(f"هانسن J = {J[0]:.2f} (د.ح {J[1]}، p = {J[2]:.3f}) — " + ("القيود الزائدة مقبولة." if 0.05 <= J[2] < 0.99 else ("⚠️ p قريبة من 1: علامة على كثرة الأدوات." if J[2] >= 0.99 else "⚠️ تُرفض القيود الزائدة.")))
    else:
        P.append(f"Instruments = {r.n_instr} for {r.n_units} units" + (" ⚠️ more instruments than units weakens the Hansen test; collapse the instruments or limit the lags." if r.n_instr > r.n_units else "."))
        if a2 is not None:
            P.append(f"AR(2) in differences: {a2[0]:.3f} (p = {a2[2]:.3f}) — " + ("no second-order serial correlation: the instruments are valid." if a2[2] >= 0.05 else "⚠️ second-order correlation: lag-2 instruments are invalid; start from lag 3."))
        if a1 is not None:
            P.append(f"AR(1) = {a1[0]:.3f} (p = {a1[2]:.3f}) — " + ("expected to be significant." if a1[2] < 0.05 else "unusually not significant."))
        P.append(f"Hansen J = {J[0]:.2f} ({J[1]} df, p = {J[2]:.3f}) — " + ("over-identifying restrictions accepted." if 0.05 <= J[2] < 0.99 else ("⚠️ p close to 1: a sign of instrument proliferation." if J[2] >= 0.99 else "⚠️ over-identifying restrictions rejected.")))
    return "<br>".join(P)


def methodology(kind: str, lang: str = "en", **kw) -> tuple:
    ar = lang == "ar"
    R = REFS
    if kind == "static":
        txt = ("قُدِّرت النماذج الساكنة للبيانات الطولية: التجميعي والآثار الثابتة (أحادية وثنائية الاتجاه) والآثار العشوائية "
               "(سوامي-أرورا) والبيني والفروق الأولى. واختير النموذج باختبار F للآثار الفردية واختبار بروش-باجان LM واختبار "
               "هاوسمان (وصيغته المتينة القائمة على الانحدار المساعد لموندلاك)، وفُحصت البواقي باختبارات الاعتماد المقطعي "
               "لبيساران (CD) وبروش-باجان وLM المعدل، واختبار وولدريدج للارتباط الذاتي، واختبار والد المعدل لعدم تجانس "
               "التباين بين الوحدات، واختبار بيساران-ياماجاتا لتجانس الميول، واستُخدمت أخطاء معيارية عنقودية أو "
               "دريسكول-كراي أو PCSE لبيك-كاتز بحسب نتائجها، وقُدِّر نموذج المربعات الصغرى المعممة الممكنة (FGLS) "
               "مع تباينات خاصة بكل وحدة.") if ar else (
               "Static panel models are estimated: pooled OLS, one- and two-way fixed effects, Swamy-Arora random effects, "
               "between and first differences. The specification is chosen with the F test for unit effects, the "
               "Breusch-Pagan LM test and the Hausman test (with its robust Mundlak auxiliary-regression version). Residuals "
               "are checked for cross-sectional dependence (Pesaran CD, Breusch-Pagan LM, scaled LM), serial correlation "
               "(Wooldridge), groupwise heteroskedasticity (modified Wald) and slope heterogeneity (Pesaran-Yamagata); "
               "cluster-robust, Driscoll-Kraay or Beck-Katz panel-corrected standard errors are used accordingly, and "
               "feasible GLS with unit-specific variances is reported.")
        refs = [R[k] for k in ("baltagi", "wooldridge", "hausman", "bp", "mundlak", "sa", "pesaran_cd", "drukker", "py", "dk", "bk", "croissant")]
    elif kind == "unitroot":
        txt = ("اختُبرت جذور الوحدة في البيانات الطولية باختبارات الجيل الأول: ليفين-لين-تشو (ميل مشترك)، وإم-بيساران-شين "
               "وفيشر لمادالا-وو واختبارات تشوي (ميول غير متجانسة)، واختبار هادري للسكون، وباختبار الجيل الثاني CIPS "
               "لبيساران الذي يراعي الاعتماد المقطعي. ثم اختُبر التكامل المشترك باختبارات كاو وبيدروني (سبع إحصاءات) "
               "وويسترلوند القائمة على تصحيح الخطأ.") if ar else (
               "Panel unit roots are tested with first-generation tests — Levin-Lin-Chu (common root), Im-Pesaran-Shin, the "
               "Maddala-Wu Fisher test and Choi's tests (heterogeneous roots), Hadri's stationarity test — and the "
               "second-generation CIPS test of Pesaran, robust to cross-sectional dependence. Cointegration is then tested "
               "with the Kao, Pedroni (seven statistics) and Westerlund error-correction tests.")
        refs = [R[k] for k in ("llc", "ips", "mw", "choi", "hadri", "cips", "kao", "pedroni99", "pedroni04", "westerlund")]
    elif kind == "gmm":
        txt = ("لأن المتغير التابع المبطأ يرتبط بالآثار الثابتة، قُدِّر النموذج الديناميكي بطريقة العزوم المعممة: "
               + ("GMM للفروق لأريلانو-بوند" if kw.get("transformation") == "d" else "GMM النظامي لبلونديل-بوند") +
               (" على خطوتين مع تصحيح ويندماير للأخطاء المعيارية" if kw.get("model") == "twosteps" else " على خطوة واحدة مع أخطاء متينة") +
               "، واستُخدمت إبطاءات المتغيرات كأدوات، وفُحصت صلاحيتها باختبار أريلانو-بوند للارتباط الذاتي من الرتبة "
               "الثانية واختبار هانسن للقيود الزائدة، مع مراقبة عدد الأدوات.") if ar else (
               "Because the lagged dependent variable is correlated with the fixed effects, the dynamic model is estimated by "
               + ("Arellano-Bond difference GMM" if kw.get("transformation") == "d" else "Blundell-Bond system GMM") +
               (", two-step with Windmeijer-corrected standard errors" if kw.get("model") == "twosteps" else ", one-step with robust standard errors") +
               ". Lags of the variables serve as instruments; their validity is checked with the Arellano-Bond test for "
               "second-order serial correlation and the Hansen test of over-identifying restrictions, keeping the instrument "
               "count below the number of units.")
        refs = [R[k] for k in ("ab", "bb", "windmeijer", "roodman")]
    elif kind == "pardl":
        txt = ("قُدِّر نموذج ARDL للبيانات الطولية بثلاث طرق: المتوسط الجماعي المجمّع (PMG) بالإمكان الأعظم الذي يفرض تجانس "
               "معاملات الأجل الطويل ويترك سرعة التعديل والديناميكية القصيرة مختلفة بين الوحدات، والمتوسط الجماعي (MG) "
               "دون قيود، والآثار الثابتة الديناميكية (DFE)، واختير بينها باختبار هاوسمان.") if ar else (
               "The panel ARDL is estimated with the pooled mean group (PMG) maximum-likelihood estimator, which constrains "
               "the long-run coefficients to be homogeneous while letting the adjustment speed and short-run dynamics vary "
               "across units, the unrestricted mean group (MG) and the dynamic fixed-effects (DFE) estimator; Hausman tests "
               "choose between them.")
        refs = [R[k] for k in ("pss99", "ps95")]
    elif kind == "advanced":
        txt = ("وللتعامل مع العوامل المشتركة غير المشاهدة قُدِّرت نماذج الآثار المرتبطة الشائعة (CCE) لبيساران والآثار الثابتة "
               "التفاعلية لباي، وقُدِّر نموذج العتبة لهانسن مع اختبار البوتستراب، ونموذج VAR للبيانات الطولية بطريقة GMM على "
               "الانحرافات الأمامية المتعامدة مع دوال الاستجابة وسببية جرانجر، واختُبرت السببية بطريقة دوميترسكو-هورلن.") if ar else (
               "To handle unobserved common factors, Pesaran's common correlated effects (CCE) and Bai's interactive fixed "
               "effects estimators are used; Hansen's panel threshold model is estimated with a bootstrap test; a panel VAR "
               "is estimated by GMM on forward orthogonal deviations with impulse responses and Granger tests; and causality "
               "is tested with the Dumitrescu-Hurlin procedure.")
        refs = [R[k] for k in ("cce", "bai", "baing", "hansen99", "lz", "al", "dh")]
    else:
        txt, refs = "", []
    return txt, sorted(refs)


def code(kind: str, file_name: str = "your_data.csv", **kw) -> dict:
    unit, time, y, x = kw.get("unit", "id"), kw.get("time", "year"), kw.get("y", "y"), kw.get("x", ["x"])
    xs = " + ".join(x)
    xsp = " ".join(x)
    py = ["import numpy as np, pandas as pd", f'df = pd.read_csv("{file_name}").set_index(["{unit}", "{time}"])']
    rr = ["library(plm)", f'df <- pdata.frame(read.csv("{file_name}"), index = c("{unit}", "{time}"))']
    st = [f'import delimited "{file_name}", clear varnames(1) case(preserve)', f"encode {unit}, gen(_id)", f"xtset _id {time}"]
    if kind == "static":
        py += ["# pip install linearmodels", "from linearmodels.panel import PooledOLS, PanelOLS, RandomEffects, BetweenOLS, FirstDifferenceOLS, compare",
               f"f = '{y} ~ 1 + {xs}'",
               "po = PooledOLS.from_formula(f, df).fit(cov_type='clustered', cluster_entity=True)",
               f"fe = PanelOLS.from_formula('{y} ~ {xs} + EntityEffects', df).fit(cov_type='clustered', cluster_entity=True)",
               f"fe2 = PanelOLS.from_formula('{y} ~ {xs} + EntityEffects + TimeEffects', df).fit(cov_type='kernel')   # Driscoll-Kraay",
               "re = RandomEffects.from_formula(f, df).fit()", "print(compare({'Pooled': po, 'FE': fe, 'FE2': fe2, 'RE': re}))"]
        rr += [f"f <- {y} ~ {xs}", "po <- plm(f, df, model = 'pooling'); fe <- plm(f, df, model = 'within')",
               "fe2 <- plm(f, df, model = 'within', effect = 'twoways'); re <- plm(f, df, model = 'random')",
               "be <- plm(f, df, model = 'between'); fd <- plm(f, df, model = 'fd')",
               "pFtest(fe, po); plmtest(po, type = 'bp'); phtest(fe, re)",
               "phtest(f, df, method = 'aux', vcov = function(x) vcovHC(x, method = 'arellano'))   # robust Hausman",
               "pcdtest(fe, test = 'cd'); pcdtest(fe, test = 'lm'); pwartest(fe); pwfdtest(f, df, h0 = 'fe')",
               "library(lmtest)", "coeftest(fe, vcov = vcovHC(fe, type = 'sss', cluster = 'group'))   # clustered (Stata)",
               "coeftest(fe, vcov = vcovSCC(fe))   # Driscoll-Kraay", "coeftest(fe, vcov = vcovBK(fe, cluster = 'time'))   # PCSE",
               "# FGLS with unit-specific variances (= Stata xtgls, panels(het)):",
               "e <- resid(lm(f, data = df)); w <- 1 / ave(e^2, index(df)[[1]])",
               "summary(lm(f, data = df, weights = w))$coefficients[, 1]   # coefficients; GLS SEs = sqrt(diag(solve(t(X) %*% (w * X))))"]
        st += [f"xtreg {y} {xsp}, fe", "estimates store fe", f"xtreg {y} {xsp}, re", "estimates store re", "hausman fe re",
               f"quietly xtreg {y} {xsp}, re", "xttest0   // Breusch-Pagan LM",
               f"xtreg {y} {xsp}, fe vce(cluster _id)", f"xtreg {y} {xsp} i.{time}, fe vce(cluster _id)   // two-way FE",
               "* ssc install xtscc ; ssc install xtcsd ; ssc install xtserial ; ssc install xttest3 ; ssc install xtoverid",
               f"xtscc {y} {xsp}, fe   // Driscoll-Kraay", f"quietly xtreg {y} {xsp}, fe", "xtcsd, pesaran abs", "xttest3",
               f"xtserial {y} {xsp}", f"quietly xtreg {y} {xsp}, re vce(cluster _id)", "xtoverid   // robust Hausman (Mundlak)",
               f"xtpcse {y} {xsp}   // Beck-Katz PCSE", f"xtgls {y} {xsp}, panels(heteroskedastic)   // FGLS"]
    elif kind == "unitroot":
        v = kw.get("var", y)
        py += ["# Panel unit-root tests are not in statsmodels/linearmodels: use R plm or Stata xtunitroot (or Aetheris)."]
        rr += ["pm <- min(4, max(1, floor(min(table(index(df)[[1]])) / 4)))   # max ADF lag, kept small for short T",
               f"purtest(df${v}, test = 'levinlin', exo = 'intercept', lags = 'SIC', pmax = pm)",
               f"purtest(df${v}, test = 'ips', exo = 'intercept', lags = 'SIC', pmax = pm)",
               f"purtest(df${v}, test = 'madwu', exo = 'intercept', lags = 'SIC', pmax = pm, p.approx = 'MacKinnon1994')",
               f"purtest(df${v}, test = 'Pm', exo = 'intercept', lags = 'SIC', pmax = pm, p.approx = 'MacKinnon1994')",
               f"purtest(df${v}, test = 'hadri', exo = 'intercept')", f"if (min(table(index(df)[[1]])) >= 10) cipstest(df${v}, lags = 1, type = 'drift')   # CIPS tables need T >= 10",
               f"pgrangertest({y} ~ {x[0]}, data = df, order = 1, test = if (min(table(index(df)[[1]])) > 8) 'Ztilde' else 'Zbar')   # Dumitrescu-Hurlin"]
        st += [f"xtunitroot llc {v}, lags(aic 4)", f"xtunitroot ips {v}, lags(aic 4)", f"xtunitroot fisher {v}, dfuller lags(2)",
               f"xtunitroot hadri {v}", "* ssc install multipurt ; ssc install xtcips", f"xtcips {v}, maxlags(1) bglags(1)",
               f"xtcointtest kao {y} {xsp}", f"xtcointtest pedroni {y} {xsp}", f"xtcointtest westerlund {y} {xsp}, allpanels",
               "* ssc install xtgcause", f"xtgcause {y} {x[0]}, lags(1)"]
    elif kind == "gmm":
        tr, mo, ef, co = kw.get("transformation", "d"), kw.get("model", "twosteps"), kw.get("effect", "twoways"), kw.get("collapse", False)
        endo, pre, exo = kw.get("endo", []), kw.get("pre", []), kw.get("exo", [])
        gm = [f"lag({y}, 2:99)"] + [f"lag({v}, 2:99)" for v in endo] + [f"lag({v}, 1:99)" for v in pre]
        rr += [f"m <- pgmm({y} ~ lag({y}, 1) + {xs} | {' + '.join(gm)}, data = df, effect = '{ef}', model = '{mo}', "
               f"transformation = '{tr}', collapse = {str(co).upper()})", "summary(m, robust = TRUE)   # AR(1), AR(2), Sargan/Hansen, Wald"]
        py += ["# Python: pip install pydynpd  (difference / system GMM)", "from pydynpd import regression",
               f"cmd = '{y} L1.{y} {xsp} | gmm({y}, 2:99) " + " ".join(f"gmm({v}, 2:99)" for v in endo) + " " +
               " ".join(f"gmm({v}, 1:99)" for v in pre) + " " + " ".join(f"iv({v})" for v in exo) +
               f" | {'nolevel ' if tr == 'd' else ''}{'collapse ' if co else ''}timedumm'",
               f"regression.abond(cmd, df.reset_index(), ['{unit}', '{time}'])"]
        st += ["* ssc install xtabond2",
               f"xtabond2 {y} L.{y} {xsp}{' i.' + time if ef == 'twoways' else ''}, gmm(L.{y}{' ' + ' '.join(endo) if endo else ''}, lag(1 .){' collapse' if co else ''})"
               + (f" gmm({' '.join(pre)}, lag(0 .){' collapse' if co else ''})" if pre else "")
               + (f" iv({' '.join(exo)})" if exo else "") + (f" iv(i.{time})" if ef == "twoways" else "")
               + f"{' noleveleq' if tr == 'd' else ''}{' twostep' if mo == 'twosteps' else ''} robust small"]
    elif kind == "pardl":
        p, q = kw.get("p", 1), kw.get("q", 1)
        rr += ["# PMG: no maintained CRAN package; MG via plm::pmg on the ECM form", f"dfx <- df; dfx$dy <- diff(df${y})",
               f"pmg(dy ~ lag({y}) + " + " + ".join(f"lag({v}) + diff({v})" for v in x) + ", dfx, model = 'mg')"]
        py += ["# Panel ARDL (PMG/MG/DFE): use Stata xtpmg or Aetheris; no maintained Python package"]
        st += ["* ssc install xtpmg", f"xtpmg d.{y} " + " ".join(f"d.{v}" for v in x) + f", lr(l.{y} {xsp}) ec(ECT) replace pmg",
               "estimates store pmg", f"xtpmg d.{y} " + " ".join(f"d.{v}" for v in x) + f", lr(l.{y} {xsp}) ec(ECT) replace mg",
               "estimates store mg", "hausman mg pmg, sigmamore",
               f"xtpmg d.{y} " + " ".join(f"d.{v}" for v in x) + f", lr(l.{y} {xsp}) ec(ECT) replace dfe"]
        del p, q
    else:   # advanced
        rr += [f"pcce({y} ~ {xs}, df, model = 'mg'); pcce({y} ~ {xs}, df, model = 'p')", f"pmg({y} ~ {xs}, df, model = 'cmg')",
               "# interactive FE: library(phtt); Eup(...)   ·   threshold: library(pdR); ptm(...)   ·   panel VAR: library(panelvar)"]
        py += ["# CCE / IFE / threshold / panel VAR: Stata (xtdcce2, regife, xthreg, pvar) or Aetheris"]
        st += ["* ssc install xtdcce2 ; ssc install regife ; ssc install xthreg ; ssc install pvar",
               f"xtdcce2 {y} {xsp}, cr({y} {xsp})   // CCE-MG", f"xtdcce2 {y} {xsp}, cr({y} {xsp}) pooled({xsp})   // CCE pooled",
               f"regife {y} {xsp}, factors(_id {time}, 1)", f"xthreg {y} {xsp}, rx({x[0]}) qx({x[-1]}) thnum(1) bs(300) trim(0.05)",
               f"pvar {y} {xsp}, lags(1) instlags(1/2) fod", "pvarstable", "pvargranger", "pvarirf, oirf mc(200)"]
    return {"Python": "\n".join(py) + "\n", "R": "\n".join(rr) + "\n", "Stata": "\n".join(st) + "\n"}


def fmt_p(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return ""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"
