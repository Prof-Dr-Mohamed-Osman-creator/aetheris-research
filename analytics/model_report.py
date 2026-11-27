# ==========================================
# Aetheris Research - Reporting for v1.5 models
# ------------------------------------------
# Diagnostic tables, bilingual interpretation, policy brief, methodology paragraphs and references
# for IV / systems, limited dependent variables, count and survival models.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import count_surv as cs
from analytics import iv as ivm
from analytics import ldv
from analytics import margins as mg
from analytics import reg_report as rp

REF = {
    **rp.REFERENCES,
    "theil53": "Theil, H. (1953). Repeated least squares applied to complete equation systems. Central Planning Bureau, The Hague.",
    "basmann": "Basmann, R. L. (1957). A generalized classical method of linear estimation of coefficients in a structural equation. Econometrica, 25(1), 77–83.",
    "zellner": "Zellner, A. (1962). An efficient method of estimating seemingly unrelated regressions and tests for aggregation bias. Journal of the American Statistical Association, 57(298), 348–368.",
    "zt": "Zellner, A., & Theil, H. (1962). Three-stage least squares: Simultaneous estimation of simultaneous equations. Econometrica, 30(1), 54–78.",
    "anderson": "Anderson, T. W., & Rubin, H. (1949). Estimation of the parameters of a single equation in a complete system of stochastic equations. Annals of Mathematical Statistics, 20(1), 46–63.",
    "fuller": "Fuller, W. A. (1977). Some properties of a modification of the limited information estimator. Econometrica, 45(4), 939–953.",
    "hansen82": "Hansen, L. P. (1982). Large sample properties of generalized method of moments estimators. Econometrica, 50(4), 1029–1054.",
    "sargan": "Sargan, J. D. (1958). The estimation of economic relationships using instrumental variables. Econometrica, 26(3), 393–415.",
    "stockyogo": "Stock, J. H., & Yogo, M. (2005). Testing for weak instruments in linear IV regression. In D. W. K. Andrews & J. H. Stock (Eds.), Identification and inference for econometric models (pp. 80–108). Cambridge University Press.",
    "cragg93": "Cragg, J. G., & Donald, S. G. (1993). Testing identifiability and specification in instrumental variable models. Econometric Theory, 9(2), 222–240.",
    "hausman78": "Hausman, J. A. (1978). Specification tests in econometrics. Econometrica, 46(6), 1251–1271.",
    "wu": "Wu, D.-M. (1973). Alternative tests of independence between stochastic regressors and disturbances. Econometrica, 41(4), 733–750.",
    "staiger": "Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak instruments. Econometrica, 65(3), 557–586.",
    "wooldridge": "Wooldridge, J. M. (2010). Econometric analysis of cross section and panel data (2nd ed.). MIT Press.",
    "mcfadden": "McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. In P. Zarembka (Ed.), Frontiers in econometrics (pp. 105–142). Academic Press.",
    "hosmer": "Hosmer, D. W., & Lemeshow, S. (1980). Goodness of fit tests for the multiple logistic regression model. Communications in Statistics – Theory and Methods, 9(10), 1043–1069.",
    "pregibon": "Pregibon, D. (1980). Goodness of link tests for generalized linear models. Applied Statistics, 29(1), 15–24.",
    "hanley": "Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. Radiology, 143(1), 29–36.",
    "hausmcf": "Hausman, J., & McFadden, D. (1984). Specification tests for the multinomial logit model. Econometrica, 52(5), 1219–1240.",
    "mckelvey": "McKelvey, R. D., & Zavoina, W. (1975). A statistical model for the analysis of ordinal level dependent variables. Journal of Mathematical Sociology, 4(1), 103–120.",
    "brant": "Brant, R. (1990). Assessing proportionality in the proportional odds model for ordinal logistic regression. Biometrics, 46(4), 1171–1178.",
    "tobin": "Tobin, J. (1958). Estimation of relationships for limited dependent variables. Econometrica, 26(1), 24–36.",
    "mcdonald": "McDonald, J. F., & Moffitt, R. A. (1980). The uses of Tobit analysis. Review of Economics and Statistics, 62(2), 318–321.",
    "heckman79": "Heckman, J. J. (1979). Sample selection bias as a specification error. Econometrica, 47(1), 153–161.",
    "cragg71": "Cragg, J. G. (1971). Some statistical models for limited dependent variables with application to the demand for durable goods. Econometrica, 39(5), 829–844.",
    "jones": "Jones, A. M. (1989). A double-hurdle model of cigarette consumption. Journal of Applied Econometrics, 4(1), 23–39.",
    "linschmidt": "Lin, T.-F., & Schmidt, P. (1984). A test of the Tobit specification against an alternative suggested by Cragg. Review of Economics and Statistics, 66(1), 174–177.",
    "papke": "Papke, L. E., & Wooldridge, J. M. (1996). Econometric methods for fractional response variables with an application to 401(k) plan participation rates. Journal of Applied Econometrics, 11(6), 619–632.",
    "cameron": "Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data (2nd ed.). Cambridge University Press.",
    "ct90": "Cameron, A. C., & Trivedi, P. K. (1990). Regression-based tests for overdispersion in the Poisson model. Journal of Econometrics, 46(3), 347–364.",
    "lambert": "Lambert, D. (1992). Zero-inflated Poisson regression, with an application to defects in manufacturing. Technometrics, 34(1), 1–14.",
    "mullahy": "Mullahy, J. (1986). Specification and testing of some modified count data models. Journal of Econometrics, 33(3), 341–365.",
    "vuong": "Vuong, Q. H. (1989). Likelihood ratio tests for model selection and non-nested hypotheses. Econometrica, 57(2), 307–333.",
    "santos": "Santos Silva, J. M. C., & Tenreyro, S. (2006). The log of gravity. Review of Economics and Statistics, 88(4), 641–658.",
    "km": "Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282), 457–481.",
    "mantel": "Mantel, N. (1966). Evaluation of survival data and two new rank order statistics arising in its consideration. Cancer Chemotherapy Reports, 50(3), 163–170.",
    "cox72": "Cox, D. R. (1972). Regression models and life-tables. Journal of the Royal Statistical Society B, 34(2), 187–220.",
    "efron77": "Efron, B. (1977). The efficiency of Cox's likelihood function for censored data. Journal of the American Statistical Association, 72(359), 557–565.",
    "grambsch": "Grambsch, P. M., & Therneau, T. M. (1994). Proportional hazards tests and diagnostics based on weighted residuals. Biometrika, 81(3), 515–526.",
    "harrell": "Harrell, F. E., Califf, R. M., Pryor, D. B., Lee, K. L., & Rosati, R. A. (1982). Evaluating the yield of medical tests. JAMA, 247(18), 2543–2546.",
    "kalbfleisch": "Kalbfleisch, J. D., & Prentice, R. L. (2002). The statistical analysis of failure time data (2nd ed.). Wiley.",
    "jenkins": "Jenkins, S. P. (1995). Easy estimation methods for discrete-time duration models. Oxford Bulletin of Economics and Statistics, 57(1), 129–138.",
    "allison": "Allison, P. D. (1982). Discrete-time methods for the analysis of event histories. Sociological Methodology, 13, 61–98.",
}

REF_KEYS = {
    "2sls": ["theil53", "basmann", "wooldridge", "staiger", "stockyogo", "cragg93", "sargan", "hausman78", "wu", "anderson"],
    "liml": ["anderson", "wooldridge", "stockyogo", "cragg93", "sargan", "hausman78"],
    "fuller": ["fuller", "anderson", "stockyogo", "cragg93"],
    "gmm": ["hansen82", "wooldridge", "stockyogo", "cragg93"],
    "ols": ["wooldridge", "hausman78", "wu"],
    "lpm": ["wooldridge", "white"], "logit": ["mcfadden", "wooldridge", "hosmer", "pregibon", "hanley"],
    "probit": ["wooldridge", "hosmer", "pregibon", "hanley"], "cloglog": ["wooldridge", "pregibon", "hanley"],
    "mnlogit": ["mcfadden", "hausmcf", "wooldridge"], "ologit": ["mckelvey", "brant", "wooldridge"],
    "oprobit": ["mckelvey", "wooldridge"], "tobit": ["tobin", "mcdonald", "wooldridge"],
    "truncreg": ["wooldridge", "greene"], "heckman2": ["heckman79", "greene"], "heckman_ml": ["heckman79", "greene"],
    "cragg": ["cragg71", "linschmidt", "wooldridge"], "dhurdle": ["jones", "cragg71"], "frac": ["papke", "wooldridge"],
    "poisson": ["cameron", "ct90"], "negbin": ["cameron", "ct90"], "zip": ["lambert", "vuong", "cameron"],
    "zinb": ["lambert", "vuong", "cameron"], "hurdle_p": ["mullahy", "cameron"], "hurdle_nb": ["mullahy", "cameron"],
    "ppml": ["santos", "cameron"], "cox": ["cox72", "efron77", "grambsch", "harrell", "km", "mantel"],
}


def references(method: str) -> list:
    keys = REF_KEYS.get(method, ["kalbfleisch", "km"] if method.startswith("aft") else ["wooldridge"])
    if method.startswith("aft"):
        keys = ["kalbfleisch", "km", "mantel"]
    return sorted({REF[k] for k in keys})


def family(r) -> str:
    if "Z" in r.extra:
        return "iv"
    return r.extra.get("family", "binary" if r.method == "lpm" else "other")


def _fmt(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ==========================================================
# Diagnostics per family
# ==========================================================
def diagnostics(df: pd.DataFrame, r, lang: str = "en") -> pd.DataFrame:
    ar = lang == "ar"
    fam = family(r)
    rows = []

    def add(test, test_ar, stat, dfv, p, verdict_en, verdict_ar):
        rows.append({"Test": test_ar if ar else test, "Statistic": stat, "df": dfv, "p-value": p,
                     "Reading": verdict_ar if ar else verdict_en})
    st = r.extra.get("stats", {})
    if "LR χ²" in st and np.isfinite(st.get("LR χ²", np.nan)):
        add("Likelihood-ratio test (all slopes = 0)", "اختبار نسبة الإمكان (كل الميول = 0)", st["LR χ²"], "", st["Prob > χ²"],
            "model significant" if st["Prob > χ²"] < 0.05 else "not significant",
            "النموذج معنوي" if st["Prob > χ²"] < 0.05 else "غير معنوي")
    if fam == "iv":
        t = ivm.iv_tests(df, r)
        for _, row in t.iterrows():
            p = row["p-value"]
            g = row["group"]
            if g == "weak":
                if row["Test"].startswith("Cragg"):
                    v_en = "strong instruments (F ≥ 10)" if row["Statistic"] >= 10 else "WEAK instruments (F < 10)"
                    v_ar = "أدوات قوية (F ≥ 10)" if row["Statistic"] >= 10 else "أدوات ضعيفة (F < 10)"
                else:
                    v_en = "endogenous effect ≠ 0 (weak-IV robust)" if p < 0.05 else "no effect detected"
                    v_ar = "أثر المتغير الداخلي ≠ 0 (متين لضعف الأدوات)" if p < 0.05 else "لا أثر معنوي"
            elif g == "overid":
                v_en = "—" if not np.isfinite(p) else ("instruments valid" if p >= 0.05 else "INVALID instruments (over-id rejected)")
                v_ar = "—" if not np.isfinite(p) else ("الأدوات صالحة" if p >= 0.05 else "أدوات غير صالحة (رُفض فرط التحديد)")
            else:
                v_en = "endogeneity confirmed → IV needed" if p < 0.05 else "exogeneity not rejected → OLS is consistent and more efficient"
                v_ar = "الداخلية مؤكدة ← يلزم IV" if p < 0.05 else "لا تُرفض الخارجية ← OLS متسق وأكفأ"
            rows.append({"Test": row["Test_ar"] if ar else row["Test"], "Statistic": row["Statistic"], "df": row["df"],
                         "p-value": p, "Reading": v_ar if ar else v_en + (f" — {row['note']}" if row["note"] and not ar else "")})
        h = ivm.hausman_ols_iv(df, r)
        add("Hausman (OLS vs IV coefficients)", "هاوسمان (مقارنة معاملات OLS وIV)", h["H"], h["df"], h["p"],
            "OLS and IV differ → endogeneity" if h["p"] < 0.05 else "no systematic difference",
            "يختلف OLS عن IV ← داخلية" if h["p"] < 0.05 else "لا فرق منهجي")
    elif fam in ("binary", "discrete_hazard"):
        y, p = r.y, np.clip(r.fitted, 1e-12, 1 - 1e-12)
        c = r.extra.get("classification") or ldv.classification(y, p)
        roc = ldv.roc(y, p)
        add("Area under ROC curve (AUC)", "المساحة تحت منحنى ROC (AUC)", roc["auc"], "", None,
            f"95% CI {roc['ci'][0]:.3f}–{roc['ci'][1]:.3f}; " + ("excellent" if roc["auc"] >= 0.8 else "acceptable" if roc["auc"] >= 0.7 else "weak") + " discrimination",
            f"فترة الثقة 95%: {roc['ci'][0]:.3f}–{roc['ci'][1]:.3f}؛ قدرة تمييز " + ("ممتازة" if roc["auc"] >= 0.8 else "مقبولة" if roc["auc"] >= 0.7 else "ضعيفة"))
        add("Correctly classified (cut-off 0.5)", "نسبة التصنيف الصحيح (حد القطع 0.5)", c["accuracy"], "", None,
            f"sensitivity {c['sensitivity']:.3f}, specificity {c['specificity']:.3f}, adj. count R² {c['adj_count_r2']:.3f}",
            f"الحساسية {c['sensitivity']:.3f}، النوعية {c['specificity']:.3f}، R² العددي المعدل {c['adj_count_r2']:.3f}")
        if r.method != "lpm":
            hl = ldv.hosmer_lemeshow(y, p)
            add("Hosmer-Lemeshow (10 groups)", "هوسمر-ليمشو (10 مجموعات)", hl["stat"], hl["df"], hl["p"],
                "good calibration" if hl["p"] >= 0.05 else "POOR calibration", "معايرة جيدة" if hl["p"] >= 0.05 else "معايرة ضعيفة")
            if fam == "binary":
                lt = ldv.link_test(r)
                add("Pregibon link test (ŷ²)", "اختبار الرابط لبريجيبون (ŷ²)", lt["z"], "", lt["p"],
                    "link adequate" if lt["p"] >= 0.05 else "MIS-specified link / omitted non-linearity",
                    "دالة الربط مناسبة" if lt["p"] >= 0.05 else "خطأ في دالة الربط أو لا خطية محذوفة")
    elif fam == "multinomial":
        add("Correctly classified", "نسبة التصنيف الصحيح", r.extra["accuracy"], "", None, "", "")
        for _, row in ldv.iia_test(df, r).iterrows():
            add(f"Hausman-McFadden IIA (omit '{row['Omitted outcome']}')", f"هاوسمان-ماكفادن IIA (حذف «{row['Omitted outcome']}»)",
                row["χ²"], row["df"], row["p-value"],
                "IIA holds" + (" (negative χ²: evidence for IIA)" if row["χ²"] < 0 else "") if row["Verdict"] == "IIA holds" else "IIA REJECTED → nested / mixed logit",
                "افتراض IIA مستوفى" if row["Verdict"] == "IIA holds" else "يُرفض IIA ← اللوجيت المتداخل أو المختلط")
    elif fam == "ordered":
        add("Correctly classified", "نسبة التصنيف الصحيح", r.extra["accuracy"], "", None, "", "")
        if r.extra["distr"] == "logit":
            for _, row in ldv.brant_test(r).iterrows():
                add(f"Brant parallel lines: {row['Variable']}", f"خطوط برانت المتوازية: {row['Variable']}", row["χ²"], row["df"],
                    row["p-value"], "parallel lines hold" if row["Parallel lines"] == "holds" else "VIOLATED → generalized / partial ordered logit",
                    "التوازي مستوفى" if row["Parallel lines"] == "holds" else "غير مستوفى ← اللوجيت المرتب المعمم")
    elif fam == "censored" and r.method == "tobit":
        add("Censored observations", "المشاهدات المراقَبة", r.extra["n_left"] + r.extra["n_right"], "", None,
            f"{r.extra['n_left']} at the lower and {r.extra['n_right']} at the upper limit",
            f"{r.extra['n_left']} عند الحد الأدنى و{r.extra['n_right']} عند الحد الأعلى")
    elif fam == "selection":
        if "LR_rho" in r.extra:
            add("LR test of independent equations (ρ = 0)", "اختبار نسبة الإمكان لاستقلال المعادلتين (ρ = 0)", r.extra["LR_rho"], 1,
                r.extra["p_LR_rho"], "selection bias present" if r.extra["p_LR_rho"] < 0.05 else "no selection bias",
                "تحيز الاختيار موجود" if r.extra["p_LR_rho"] < 0.05 else "لا تحيز اختيار")
        else:
            z = r.extra["lambda"] / r.extra["lambda_se"]
            p = float(2 * __import__("scipy").stats.norm.sf(abs(z)))
            add("t-test of the inverse Mills ratio (λ = 0)", "اختبار t لنسبة ميلز العكسية (λ = 0)", z, "", p,
                "selection bias present" if p < 0.05 else "no selection bias", "تحيز الاختيار موجود" if p < 0.05 else "لا تحيز اختيار")
    elif fam == "hurdle" and "tobit_LR" in r.extra:
        LR, d, p = r.extra["tobit_LR"]
        add("Lin-Schmidt LR: Tobit vs Cragg", "اختبار لين-شميدت: توبيت مقابل كراج", LR, d, p,
            "Tobit REJECTED → two separate decisions" if p < 0.05 else "Tobit adequate",
            "يُرفض توبيت ← قراران منفصلان" if p < 0.05 else "توبيت مناسب")
    elif fam == "count":
        if "overdispersion" in r.extra:
            o = r.extra["overdispersion"]
            add("Cameron-Trivedi overdispersion test (α̂)", "اختبار كاميرون-تريفيدي لفرط التشتت (α̂)", o["t"], "", o["p"],
                f"α̂ = {o['alpha_hat']:.3f} → " + ("OVERDISPERSION: use negative binomial / robust SEs" if o["p"] < 0.05 else "equidispersion holds"),
                f"α̂ = {o['alpha_hat']:.3f} ← " + ("فرط تشتت: استخدم ذا الحدين السالب أو أخطاء متينة" if o["p"] < 0.05 else "تساوي التشتت مستوفى"))
            add("Pearson dispersion (χ² / df)", "تشتت بيرسون (χ² / درجات الحرية)", r.extra["pearson_disp"], "", None,
                "≈ 1 under the Poisson assumption", "≈ 1 في ظل افتراض بواسون")
        if "lr_alpha" in r.extra:
            add("LR test α = 0 (Poisson vs NB2)", "اختبار نسبة الإمكان α = 0 (بواسون مقابل NB2)", r.extra["lr_alpha"]["LR"], "χ̄²₀₁",
                r.extra["lr_alpha"]["p"], "NB2 preferred" if r.extra["lr_alpha"]["p"] < 0.05 else "Poisson adequate",
                "ذو الحدين السالب أفضل" if r.extra["lr_alpha"]["p"] < 0.05 else "بواسون كافٍ")
        if "vuong" in r.extra:
            add("Vuong test (zero-inflated vs standard)", "اختبار فوونج (المتضخم بالأصفار مقابل العادي)", r.extra["vuong"]["z"], "",
                r.extra["vuong"]["p"], "zero inflation supported" if r.extra["vuong"]["p"] < 0.05 else "no zero inflation",
                "تضخم الأصفار مدعوم" if r.extra["vuong"]["p"] < 0.05 else "لا تضخم في الأصفار")
    elif fam == "survival":
        if "ph_test" in r.extra:
            for _, row in r.extra["ph_test"].iterrows():
                add(f"Proportional hazards (Schoenfeld): {row['Variable']}", f"تناسب المخاطر (شونفيلد): {row['Variable']}",
                    row["χ²"], row["df"], row["p-value"], "PH holds" if row["PH assumption"] == "holds" else "PH VIOLATED → time-varying effect / stratify",
                    "التناسب مستوفى" if row["PH assumption"] == "holds" else "غير مستوفى ← أثر متغير مع الزمن أو التقسيم الطبقي")
        if "Concordance (Harrell C)" in st:
            add("Harrell's concordance C", "مؤشر التوافق لهاريل C", st["Concordance (Harrell C)"], "", None,
                "0.5 = random, 1 = perfect ranking", "0.5 = عشوائي، 1 = ترتيب تام")
    return pd.DataFrame(rows)


# ==========================================================
# Interpretation
# ==========================================================
def interpret(df: pd.DataFrame, r, effects: pd.DataFrame | None, lang: str = "en", alpha: float = 0.05) -> str:
    ar = lang == "ar"
    fam = family(r)
    m = r.method
    st = r.extra.get("stats", {})
    P = []
    label = r.name
    P.append((f"قُدِّر نموذج <b>{label}</b> على {r.nobs} مشاهدة." if ar else
              f"<b>{label}</b> was estimated on {r.nobs} observations."))
    if "Pseudo R²" in st and np.isfinite(st.get("Pseudo R²", np.nan)):
        P.append((f"معامل التحديد الزائف لماكفادن = {st['Pseudo R²']:.3f}، واختبار نسبة الإمكان χ² = {st['LR χ²']:.2f} ({_fmt(st['Prob > χ²'])})"
                  + ("، فالمتغيرات المفسرة معنوية مجتمعة." if st["Prob > χ²"] < alpha else ".")) if ar else
                 (f"McFadden pseudo-R² = {st['Pseudo R²']:.3f}; likelihood-ratio χ² = {st['LR χ²']:.2f} ({_fmt(st['Prob > χ²'])})"
                  + (" — the regressors are jointly significant." if st["Prob > χ²"] < alpha else ".")))
    if fam == "iv":
        tests = ivm.iv_tests(df, r).set_index("Test")
        cd = tests.loc["Cragg-Donald Wald F (min. eigenvalue)", "Statistic"]
        wu = tests.loc["Wu-Hausman F", "p-value"]
        P.append(("الأدوات " + ("قوية" if cd >= 10 else "<b>ضعيفة</b>") + f" (إحصاءة كراج-دونالد F = {cd:.2f}). "
                  + ("واختبار وو-هاوسمان يؤكد داخلية المتغير، فتقدير IV مبرَّر." if wu < alpha else
                     "ولا يرفض اختبار وو-هاوسمان الخارجية، فمقدّر OLS متسق وأكفأ."))
                 if ar else ("Instruments are " + ("strong" if cd >= 10 else "<b>weak</b>") + f" (Cragg-Donald F = {cd:.2f}). "
                             + ("The Wu-Hausman test confirms endogeneity, so IV estimation is warranted." if wu < alpha else
                                "The Wu-Hausman test does not reject exogeneity: OLS is consistent and more efficient.")))
        if "Sargan χ² (homoskedastic)" in tests.index:
            ps = tests.loc["Hansen J (heteroskedasticity-robust)", "p-value"]
            P.append((f"اختبار هانسن J لفرط التحديد (p = {ps:.3f}) " + ("لا يرفض صلاحية الأدوات." if ps >= alpha else "<b>يرفض</b> صلاحية الأدوات.")) if ar else
                     f"Hansen's J over-identification test (p = {ps:.3f}) " + ("does not reject instrument validity." if ps >= alpha else "<b>rejects</b> instrument validity."))
        for c in r.extra["endog_cols"]:
            b, p = r.params[c], r.pvalues[c]
            P.append((f"• الأثر السببي المقدَّر لـ {c} = {b:.4g} ({_fmt(p)})." if ar else
                      f"• The IV estimate of the effect of {c} is {b:.4g} ({_fmt(p)})."))
    if effects is not None and not effects.empty:
        sig = effects[effects["p-value"] < alpha]
        head = {"binary": ("التغير في احتمال النجاح (نقاط مئوية)", "change in probability (percentage points)", 100),
                "discrete_hazard": ("التغير في احتمال وقوع الحدث في الفترة", "change in the per-period event probability", 100),
                "fractional": ("التغير في النسبة (نقاط مئوية)", "change in the share (percentage points)", 100),
                "count": ("التغير في العدد المتوقع", "change in the expected count", 1)}.get(fam)
        if head and not sig.empty:
            P.append("<b>" + (head[0] if ar else "Average marginal effects — " + head[1]) + ":</b>")
            for _, row in sig.head(8).iterrows():
                v = row["Effect"] * head[2]
                unit = (" نقطة مئوية" if head[2] == 100 else "") if ar else (" pp" if head[2] == 100 else "")
                P.append(f"• {row['Variable']}: {v:+.3f}{unit} ({_fmt(row['p-value'])})")
        elif fam in ("multinomial", "ordered") and not sig.empty:
            P.append("<b>" + ("أبرز الآثار الحدية على احتمالات الفئات:" if ar else "Largest marginal effects on category probabilities:") + "</b>")
            for _, row in sig.reindex(sig["Effect"].abs().sort_values(ascending=False).index).head(6).iterrows():
                P.append(f"• {row['Variable']} → P({row['Outcome']}): {100 * row['Effect']:+.2f} " + ("نقطة مئوية" if ar else "pp")
                         + f" ({_fmt(row['p-value'])})")
        elif fam in ("censored", "hurdle") and not sig.empty:
            uncond = sig[sig.get("Outcome", pd.Series(dtype=str)).astype(str).str.startswith("E[y]")] if "Outcome" in sig else sig
            P.append("<b>" + ("الآثار الحدية على القيمة المتوقعة غير الشرطية E[y]:" if ar else "Marginal effects on the unconditional mean E[y]:") + "</b>")
            for _, row in uncond.head(8).iterrows():
                P.append(f"• {row['Variable']}: {row['Effect']:+.4g} ({_fmt(row['p-value'])})")
    if fam == "survival":
        if "hr" in r.extra:
            for _, row in r.extra["hr"][r.extra["hr"]["p-value"] < alpha].iterrows():
                ch = 100 * (row["Hazard ratio"] - 1)
                P.append((f"• {row['Variable']}: نسبة الخطر {row['Hazard ratio']:.3f} — زيادة المتغير بوحدة {'ترفع' if ch > 0 else 'تخفض'} معدل وقوع الحدث بنسبة {abs(ch):.1f}% ({_fmt(row['p-value'])})."
                          if ar else f"• {row['Variable']}: hazard ratio {row['Hazard ratio']:.3f} — a one-unit increase {'raises' if ch > 0 else 'lowers'} the event rate by {abs(ch):.1f}% ({_fmt(row['p-value'])})."))
        if "time_ratios" in r.extra:
            for _, row in r.extra["time_ratios"][r.extra["time_ratios"]["p-value"] < alpha].iterrows():
                if row["Variable"] == "const":
                    continue
                ch = 100 * (row["Time ratio exp(β)"] - 1)
                P.append((f"• {row['Variable']}: نسبة الزمن {row['Time ratio exp(β)']:.3f} — {'يطيل' if ch > 0 else 'يقصّر'} الزمن حتى الحدث بنسبة {abs(ch):.1f}%."
                          if ar else f"• {row['Variable']}: time ratio {row['Time ratio exp(β)']:.3f} — {'lengthens' if ch > 0 else 'shortens'} the time to the event by {abs(ch):.1f}%."))
    if "odds_ratios" in r.extra and fam in ("binary", "ordered"):
        orr = r.extra["odds_ratios"]
        orr = orr[(orr["Variable"] != "const") & (orr["p-value"] < alpha)]
        if not orr.empty:
            P.append(("نسب الأرجحية: " if ar else "Odds ratios: ") + "; ".join(f"{row['Variable']} = {row['Odds ratio']:.3f}" for _, row in orr.iterrows()))
    for n in r.notes:
        P.append("ℹ️ " + n)
    if fam not in ("iv", "survival"):
        P.append(("ملاحظة: المعاملات في النماذج غير الخطية لا تُقرأ مباشرة كآثار؛ اعتمد على الآثار الحدية." if ar else
                  "Note: coefficients of non-linear models are not marginal effects; read the AME table for magnitudes."))
    return "<br>".join(P)


def methodology(r, lang: str = "en") -> tuple[str, list]:
    ar = lang == "ar"
    m = r.method
    xs = ", ".join(list(r.design.spec.x) + list(r.design.spec.categorical)) if r.design is not None else ""
    y = r.design.spec.y if r.design is not None else r.y_label
    T = {
        "iv": ("قُدِّرت المعادلة بطريقة المتغيرات الأداتية ({m}) لمعالجة داخلية {endog}، مستخدمةً الأدوات المستبعدة ({inst}). "
               "وقُيِّمت قوة الأدوات بإحصاءة كراج-دونالد وقيم ستوك-يوغو الحرجة، وصلاحيتها باختباري سارجان وهانسن J، "
               "والداخلية باختباري دربن ووو-هاوسمان، مع اختبار أندرسون-روبن المتين لضعف الأدوات.",
               "The equation is estimated by instrumental variables ({m}) to address the endogeneity of {endog}, using the "
               "excluded instruments {inst}. Instrument strength is assessed with the Cragg-Donald statistic against Stock-Yogo "
               "critical values, validity with the Sargan and Hansen J tests, and endogeneity with the Durbin and Wu-Hausman "
               "tests; the Anderson-Rubin test provides weak-instrument-robust inference."),
        "binary": ("لنمذجة المتغير الثنائي {y} قُدِّر نموذج {m} بطريقة الإمكان الأعظم على المتغيرات ({xs}). وتُعرض الآثار الحدية "
                   "المتوسطة (AME) بأخطاء معيارية بطريقة دلتا، وتُقيَّم جودة النموذج بالمساحة تحت منحنى ROC واختبار هوسمر-ليمشو "
                   "واختبار الرابط لبريجيبون.",
                   "The binary outcome {y} is modelled with a {m} model estimated by maximum likelihood on ({xs}). Average "
                   "marginal effects (AME) with delta-method standard errors are reported; fit is assessed with the area under "
                   "the ROC curve, the Hosmer-Lemeshow test and Pregibon's link test."),
        "multinomial": ("لأن {y} متغير نوعي غير مرتب متعدد الفئات قُدِّر نموذج اللوجيت متعدد الحدود، وتُعرض نسب المخاطر النسبية "
                        "والآثار الحدية على احتمال كل فئة، ويُختبر افتراض استقلال البدائل غير ذات الصلة (IIA) باختبار هاوسمان-ماكفادن.",
                        "Because {y} is an unordered categorical outcome, a multinomial logit is estimated; relative-risk ratios "
                        "and marginal effects on each category probability are reported, and the IIA assumption is checked with "
                        "the Hausman-McFadden test."),
        "ordered": ("لأن {y} متغير مرتب قُدِّر نموذج {m} المرتب، مع نقاط القطع والآثار الحدية على احتمال كل فئة، واختبار برانت "
                    "لافتراض خطوط الانحدار المتوازية.",
                    "Because {y} is ordinal, an ordered {m} model is estimated, reporting cut-points, marginal effects on each "
                    "category probability, and Brant's test of the parallel-regression assumption."),
        "censored": ("لأن {y} مراقَب أو مبتور قُدِّر نموذج {m} بطريقة الإمكان الأعظم، وتُجزَّأ الآثار وفق ماكدونالد وموفيت إلى أثر على "
                     "احتمال تجاوز الحد وأثر على القيمة المتوقعة بشرط تجاوزه.",
                     "Because {y} is censored / truncated, a {m} model is estimated by maximum likelihood, and effects are "
                     "decomposed (McDonald & Moffitt) into the effect on the probability of exceeding the limit and on the "
                     "conditional mean."),
        "selection": ("لتصحيح تحيز الاختيار في {y} (المشاهَد فقط للمشاركين) قُدِّر نموذج هيكمان ({m})، مع معادلة اختيار بروبيت "
                      "ومعادلة نتيجة مصحَّحة بنسبة ميلز العكسية.",
                      "To correct the sample-selection bias in {y} (observed only for participants) a Heckman model ({m}) is "
                      "estimated, with a probit selection equation and an outcome equation corrected by the inverse Mills ratio."),
        "hurdle": ("لأن قرار المشاركة قد يختلف عن قرار الكمية قُدِّر نموذج {m} الذي يفصل بين العقبة الأولى (المشاركة) ومستوى {y}.",
                   "Because the participation decision may differ from the quantity decision, a {m} model separates the "
                   "participation hurdle from the level of {y}."),
        "fractional": ("لأن {y} نسبة بين الصفر والواحد قُدِّر نموذج الاستجابة الكسرية لبابكي ووولدريدج بشبه الإمكان الأعظم وأخطاء متينة.",
                       "Because {y} is a proportion in [0, 1], the Papke-Wooldridge fractional response model is estimated by "
                       "quasi-maximum likelihood with robust standard errors."),
        "count": ("لأن {y} متغير عدّي قُدِّر نموذج {m} مع نسب معدلات الحدوث (IRR) والآثار الحدية، ويُختبر فرط التشتت بطريقة "
                  "كاميرون-تريفيدي، وتضخم الأصفار باختبار فوونج.",
                  "Because {y} is a count, a {m} model is estimated, reporting incidence-rate ratios and marginal effects; "
                  "overdispersion is tested (Cameron-Trivedi) and zero inflation with Vuong's test."),
        "survival": ("لتحليل الزمن حتى وقوع الحدث ({y}) مع مراعاة المشاهدات المراقَبة قُدِّر نموذج {m}، مع منحنيات كابلان-ماير "
                     "واختبار log-rank، واختبار تناسب المخاطر ببواقي شونفيلد.",
                     "To analyse the time to the event ({y}) with right-censoring, a {m} model is estimated alongside Kaplan-Meier "
                     "curves and log-rank tests; proportional hazards are tested with scaled Schoenfeld residuals."),
        "discrete_hazard": ("قُدِّر نموذج المخاطر في الزمن المتقطع ({m}) على بيانات الوحدة-الفترة مع خط أساس مرن للمخاطر.",
                            "A discrete-time hazard model ({m}) is estimated on person-period data with a flexible baseline hazard."),
    }
    fam = family(r)
    tmpl = T.get(fam, T["binary"])[0 if ar else 1]
    lab = r.name
    kw = {"m": lab, "y": y, "xs": xs}
    if fam == "iv":
        kw.update({"endog": ", ".join(r.extra["spec"].endog), "inst": ", ".join(r.extra["spec"].instruments)})
    return tmpl.format(**kw), references(m)


def policy_brief(df, r, effects: pd.DataFrame | None, lang: str = "en", alpha: float = 0.05) -> list:
    ar = lang == "ar"
    out = []
    if effects is None or effects.empty:
        return out
    e = effects[effects["p-value"] < alpha].copy()
    if "Outcome" in e:
        fam = family(r)
        if fam in ("censored", "hurdle"):
            e = e[e["Outcome"].astype(str).str.startswith("E[y]")]
        elif fam in ("multinomial", "ordered"):
            e = e.reindex(e["Effect"].abs().sort_values(ascending=False).index)
    e = e.reindex(e["Effect"].abs().sort_values(ascending=False).index).head(5)
    scale = 100 if family(r) in ("binary", "multinomial", "ordered", "fractional", "discrete_hazard") else 1
    for _, row in e.iterrows():
        what = f" → P({row['Outcome']})" if "Outcome" in row and str(row.get("Outcome", "")) and family(r) in ("multinomial", "ordered") else ""
        unit = (" نقطة مئوية" if ar else " percentage points") if scale == 100 else ""
        out.append(f"<b>{row['Variable']}</b>{what}: {row['Effect'] * scale:+.3f}{unit}")
    return out
