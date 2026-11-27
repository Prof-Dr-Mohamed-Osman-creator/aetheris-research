# ==========================================
# Aetheris Research - Reporting for multivariate time series and volatility (v1.7, §1.8–1.9)
# Interpretation (EN/AR), methodology paragraphs, references and reproducible Python / R / Stata code.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd

REFS = {
    "sims": "Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1–48.",
    "lutkepohl": "Lütkepohl, H. (2005). New introduction to multiple time series analysis. Springer.",
    "granger": "Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37(3), 424–438.",
    "ty": "Toda, H. Y., & Yamamoto, T. (1995). Statistical inference in vector autoregressions with possibly integrated processes. Journal of Econometrics, 66(1–2), 225–250.",
    "pesaran_shin98": "Pesaran, H. H., & Shin, Y. (1998). Generalized impulse response analysis in linear multivariate models. Economics Letters, 58(1), 17–29.",
    "amisano": "Amisano, G., & Giannini, C. (1997). Topics in structural VAR econometrics (2nd ed.). Springer.",
    "bq": "Blanchard, O. J., & Quah, D. (1989). The dynamic effects of aggregate demand and supply disturbances. American Economic Review, 79(4), 655–673.",
    "pfaff": "Pfaff, B. (2008). VAR, SVAR and SVEC models: Implementation within R package vars. Journal of Statistical Software, 27(4), 1–32.",
    "eg": "Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. Econometrica, 55(2), 251–276.",
    "johansen88": "Johansen, S. (1988). Statistical analysis of cointegration vectors. Journal of Economic Dynamics and Control, 12(2–3), 231–254.",
    "johansen95": "Johansen, S. (1995). Likelihood-based inference in cointegrated vector autoregressive models. Oxford University Press.",
    "jj90": "Johansen, S., & Juselius, K. (1990). Maximum likelihood estimation and inference on cointegration — with applications to the demand for money. Oxford Bulletin of Economics and Statistics, 52(2), 169–210.",
    "mhm": "MacKinnon, J. G., Haug, A. A., & Michelis, L. (1999). Numerical distribution functions of likelihood ratio tests for cointegration. Journal of Applied Econometrics, 14(5), 563–577.",
    "reinsel": "Reinsel, G. C., & Ahn, S. K. (1992). Vector autoregressive models with unit roots and reduced rank structure. Journal of Time Series Analysis, 13(4), 353–375.",
    "gh": "Gregory, A. W., & Hansen, B. E. (1996). Residual-based tests for cointegration in models with regime shifts. Journal of Econometrics, 70(1), 99–126.",
    "ph": "Phillips, P. C. B., & Hansen, B. E. (1990). Statistical inference in instrumental variables regression with I(1) processes. Review of Economic Studies, 57(1), 99–125.",
    "sw": "Stock, J. H., & Watson, M. W. (1993). A simple estimator of cointegrating vectors in higher order integrated systems. Econometrica, 61(4), 783–820.",
    "park": "Park, J. Y. (1992). Canonical cointegrating regressions. Econometrica, 60(1), 119–143.",
    "mackinnon10": "MacKinnon, J. G. (2010). Critical values for cointegration tests (Queen's Economics Department Working Paper No. 1227).",
    "pss": "Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches to the analysis of level relationships. Journal of Applied Econometrics, 16(3), 289–326.",
    "ps99": "Pesaran, M. H., & Shin, Y. (1999). An autoregressive distributed lag modelling approach to cointegration analysis. In S. Strøm (Ed.), Econometrics and economic theory in the 20th century (pp. 371–413). Cambridge University Press.",
    "narayan": "Narayan, P. K. (2005). The saving and investment nexus for China: Evidence from cointegration tests. Applied Economics, 37(17), 1979–1990.",
    "syg": "Shin, Y., Yu, B., & Greenwood-Nimmo, M. (2014). Modelling asymmetric cointegration and dynamic multipliers in a nonlinear ARDL framework. In R. Sickles & W. Horrace (Eds.), Festschrift in honor of Peter Schmidt (pp. 281–314). Springer.",
    "cks": "Cho, J. S., Kim, T.-H., & Shin, Y. (2015). Quantile cointegration in the autoregressive distributed-lag modeling framework. Journal of Econometrics, 188(1), 281–300.",
    "koenker": "Koenker, R. (2005). Quantile regression. Cambridge University Press.",
    "natsiopoulos": "Natsiopoulos, K., & Tzeremes, N. G. (2022). ARDL bounds test for cointegration: Replicating the Pesaran et al. (2001) results for the UK earnings equation using R. Journal of Applied Econometrics, 37(5), 1079–1090.",
    "engle82": "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with estimates of the variance of United Kingdom inflation. Econometrica, 50(4), 987–1007.",
    "bollerslev86": "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307–327.",
    "elr": "Engle, R. F., Lilien, D. M., & Robins, R. P. (1987). Estimating time varying risk premia in the term structure: The ARCH-M model. Econometrica, 55(2), 391–407.",
    "gjr": "Glosten, L. R., Jagannathan, R., & Runkle, D. E. (1993). On the relation between the expected value and the volatility of the nominal excess return on stocks. Journal of Finance, 48(5), 1779–1801.",
    "nelson": "Nelson, D. B. (1991). Conditional heteroskedasticity in asset returns: A new approach. Econometrica, 59(2), 347–370.",
    "bw": "Bollerslev, T., & Wooldridge, J. M. (1992). Quasi-maximum likelihood estimation and inference in dynamic models with time-varying covariances. Econometric Reviews, 11(2), 143–172.",
    "engle_ng": "Engle, R. F., & Ng, V. K. (1993). Measuring and testing the impact of news on volatility. Journal of Finance, 48(5), 1749–1778.",
    "fs": "Fernández, C., & Steel, M. F. J. (1998). On Bayesian modeling of fat tails and skewness. Journal of the American Statistical Association, 93(441), 359–371.",
    "kupiec": "Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. Journal of Derivatives, 3(2), 73–84.",
    "christoffersen": "Christoffersen, P. F. (1998). Evaluating interval forecasts. International Economic Review, 39(4), 841–862.",
    "ccc": "Bollerslev, T. (1990). Modelling the coherence in short-run nominal exchange rates: A multivariate generalized ARCH model. Review of Economics and Statistics, 72(3), 498–505.",
    "dcc": "Engle, R. F. (2002). Dynamic conditional correlation: A simple class of multivariate generalized autoregressive conditional heteroskedasticity models. Journal of Business & Economic Statistics, 20(3), 339–350.",
    "bekk": "Engle, R. F., & Kroner, K. F. (1995). Multivariate simultaneous generalized ARCH. Econometric Theory, 11(1), 122–150.",
    "es01": "Engle, R. F., & Sheppard, K. (2001). Theoretical and empirical properties of dynamic conditional correlation multivariate GARCH (NBER Working Paper No. 8554).",
    "nm94": "Newey, W. K., & McFadden, D. (1994). Large sample estimation and hypothesis testing. In R. F. Engle & D. L. McFadden (Eds.), Handbook of econometrics (Vol. 4, pp. 2111–2245). Elsevier.",
    "ks93": "Kroner, K. F., & Sultan, J. (1993). Time-varying distributions and dynamic hedging with foreign currency futures. Journal of Financial and Quantitative Analysis, 28(4), 535–551.",
}


def _p(p):
    if p is None or not np.isfinite(p):
        return ""
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


# ==========================================================
# Methodology
# ==========================================================
def methodology(kind: str, lang: str = "en", **kw) -> tuple:
    ar = lang == "ar"
    R = REFS
    if kind == "var":
        txt = ("قُدِّر نموذج الانحدار الذاتي المتجهي (VAR) بطريقة المربعات الصغرى معادلةً بمعادلة بعد التحقق من سكون المتغيرات، "
               "واختيرت فترة الإبطاء بمعايير AIC وHQ وSC وFPE على عينة مشتركة، وتُحقق من استقرار النموذج بجذور مصفوفة "
               "المرافقة، ومن خلو البواقي من الارتباط الذاتي باختبار بورتمانتو واختبار بروش-جودفري (LM) وتصحيح إدجرتون-شوكور، "
               "ومن طبيعيتها باختبار جارك-بيرا متعدد المتغيرات، ومن ثبات تباينها باختبار ARCH متعدد المتغيرات. واختُبرت "
               "السببية بمفهوم جرانجر باختبار والد، وحُسبت دوال الاستجابة الفورية المتعامدة (تحليل تشوليسكي) والمعممة "
               "(بيساران-شين) مع فترات ثقة بالبوتستراب، وتجزئة تباين خطأ التنبؤ.") if ar else (
               "A vector autoregression (VAR) is estimated equation by equation by OLS after verifying stationarity. The lag "
               "order is chosen with AIC, HQ, SC and FPE on a common sample; stability is checked with the companion-matrix "
               "roots, residual autocorrelation with the portmanteau and Breusch-Godfrey LM tests (with the Edgerton-Shukur "
               "correction), normality with the multivariate Jarque-Bera test and homoskedasticity with the multivariate ARCH "
               "test. Granger causality is tested with Wald tests; orthogonalised (Cholesky) and generalised (Pesaran-Shin) "
               "impulse responses are computed with bootstrap confidence bands, together with forecast-error variance "
               "decompositions.")
        refs = [R[k] for k in ("sims", "lutkepohl", "granger", "ty", "pesaran_shin98", "pfaff")]
    elif kind == "svar":
        txt = ("حُددت الصدمات الهيكلية بقيود قصيرة الأجل على مصفوفتي A وB (نموذج أميسانو-جياناني) وقُدِّرت بالإمكان الأعظم "
               "بخوارزمية Scoring، مع اختبار نسبة الإمكان للقيود الزائدة عن التعريف، أو بقيود طويلة الأجل لبلانشارد-كواه.") if ar else (
               "Structural shocks are identified with short-run restrictions on the A and B matrices (Amisano-Giannini) and "
               "estimated by maximum likelihood with the scoring algorithm, with a likelihood-ratio test of the over-identifying "
               "restrictions, or with Blanchard-Quah long-run restrictions.")
        refs = [R[k] for k in ("amisano", "bq", "lutkepohl", "pfaff")]
    elif kind == "coint":
        txt = ("اختُبر التكامل المشترك بطريقة إنجل-جرانجر (اختبار ADF على بواقي انحدار المستوى مع قيم ماكينون 2010)، "
               "وبطريقة جوهانسن (اختبارا الأثر والقيمة الذاتية العظمى) مع اختيار الحالة المناسبة للحدود المحددة، وقيم حرجة "
               "واحتمالية من التوزيعات التقاربية بمحاكاة مونت كارلو، وباختبار جريجوري-هانسن الذي يسمح بانكسار هيكلي واحد "
               "مجهول التاريخ. وقُدِّرت علاقة الأجل الطويل بطرق FMOLS وDOLS وCCR التي تصحح التحيز الناتج عن الارتباط "
               "الذاتي والداخلية.") if ar else (
               "Cointegration is tested with the Engle-Granger procedure (ADF on the levels-regression residuals, MacKinnon 2010 "
               "critical values), Johansen's trace and maximum-eigenvalue tests under the appropriate deterministic case "
               "(critical values and p-values simulated from the asymptotic distributions), and the Gregory-Hansen test allowing "
               "one unknown structural break. The long-run relation is estimated by FMOLS, DOLS and CCR, which correct the OLS "
               "bias caused by serial correlation and endogeneity.")
        refs = [R[k] for k in ("eg", "mackinnon10", "johansen88", "johansen95", "mhm", "gh", "ph", "sw", "park")]
    elif kind == "vecm":
        txt = (f"قُدِّر نموذج تصحيح الخطأ المتجهي (VECM) برتبة تكامل مشترك r = {kw.get('r', 1)} بطريقة جوهانسن، مع تطبيع "
               "متجهات التكامل بالصيغة المثلثية لفيليبس، وتقدير معاملات سرعة التعديل α والديناميكية قصيرة الأجل Γ، واختبار "
               "الضعف الخارجي (α = 0) وقيود الأجل الطويل على β باختبارات نسبة الإمكان.") if ar else (
               f"A vector error-correction model (VECM) with cointegration rank r = {kw.get('r', 1)} is estimated by Johansen's "
               "reduced-rank procedure; the cointegrating vectors are normalised in Phillips' triangular form, the adjustment "
               "coefficients α and the short-run dynamics Γ are estimated, and weak exogeneity (zero rows of α) and long-run "
               "restrictions on β are tested with likelihood-ratio tests.")
        refs = [R[k] for k in ("johansen95", "jj90", "lutkepohl", "pfaff")]
    elif kind == "ardl":
        txt = ("طُبِّق نهج الانحدار الذاتي ذي الإبطاءات الموزعة (ARDL) لبيساران وشين وسميث (2001) لأنه يصلح لمتغيرات من "
               "الدرجة I(0) أو I(1) أو خليط منهما ولعينات صغيرة نسبياً. اختيرت الإبطاءات بمعيار AIC على عينة مشتركة، "
               "واختُبرت علاقة الأجل الطويل باختبار الحدود F (وt حيث ينطبق) مع قيم حرجة تقاربية وأخرى للعينة الفعلية "
               "بالمحاكاة، ثم قُدِّرت معاملات الأجل الطويل بطريقة دلتا ونموذج تصحيح الخطأ المقيد، وفُحصت البواقي "
               "والاستقرار الهيكلي (CUSUM وCUSUMSQ).") if ar else (
               "The autoregressive distributed-lag (ARDL) bounds approach of Pesaran, Shin and Smith (2001) is used because it "
               "is valid for regressors that are I(0), I(1) or mutually cointegrated, and performs well in small samples. Lag "
               "orders are selected by AIC on a common sample; the existence of a level relationship is tested with the "
               "bounds F test (and the t test where applicable) using asymptotic and sample-specific simulated critical values; "
               "long-run coefficients (delta-method standard errors) and the restricted error-correction model are then "
               "estimated, with residual and stability (CUSUM, CUSUMSQ) diagnostics.")
        refs = [R[k] for k in ("pss", "ps99", "narayan", "natsiopoulos")]
    elif kind == "nardl":
        txt = ("لالتقاط عدم تماثل الأثر قُسِّم المتغير المفسر إلى مجموعين جزئيين للتغيرات الموجبة والسالبة (شين ويو وجرينوود-نيمو "
               "2014)، واختُبر التماثل في الأجلين الطويل والقصير باختبار والد، وحُسبت المضاعفات الديناميكية التراكمية مع "
               "فترات ثقة بالبوتستراب.") if ar else (
               "To capture asymmetric effects the regressor is decomposed into partial sums of positive and negative changes "
               "(Shin, Yu & Greenwood-Nimmo 2014); long- and short-run symmetry are tested with Wald tests, and cumulative "
               "dynamic multipliers are reported with bootstrap confidence bands.")
        refs = [R[k] for k in ("syg", "pss")]
    elif kind == "qardl":
        txt = ("قُدِّر نموذج ARDL الكمي (QARDL) لتشو وكيم وشين (2015) عند عدة كميات للتوزيع الشرطي بانحدار كمي دقيق، "
               "واشتُقت معاملات الأجل الطويل وسرعة التعديل لكل كمية بطريقة دلتا من مصفوفة التباين المتينة (nid).") if ar else (
               "The quantile ARDL (QARDL) of Cho, Kim and Shin (2015) is estimated at several quantiles of the conditional "
               "distribution by exact quantile regression; long-run coefficients and the speed of adjustment at each quantile "
               "are obtained with the delta method from the robust (nid) covariance matrix.")
        refs = [R[k] for k in ("cks", "koenker")]
    elif kind == "garch":
        txt = ("نُمذجت التقلبات الشرطية للعوائد بعائلة نماذج GARCH: GARCH لبولرسليف، وGJR-GARCH لالتقاط أثر الرافعة "
               "(استجابة أكبر للصدمات السالبة)، وEGARCH لنيلسون، مع توزيعات طبيعي وستيودنت t وخطأ معمم وt ملتوي، وقُدِّرت "
               "بالإمكان الأعظم مع أخطاء معيارية متينة لبولرسليف-وولدريدج. وفُحصت البواقي المعيارية باختبارات ليونج-بوكس "
               "وARCH-LM وانحياز الإشارة لإنجل-نج، وحُسبت الاستمرارية ونصف العمر ومنحنى أثر الأخبار والقيمة المعرضة للخطر مع "
               "اختبارات كوبيك وكريستوفرسن.") if ar else (
               "Conditional volatility is modelled with the GARCH family — Bollerslev's GARCH, the GJR-GARCH capturing the "
               "leverage effect (a larger response to negative shocks) and Nelson's EGARCH — with normal, Student-t, GED and "
               "skewed-t errors, estimated by maximum likelihood with Bollerslev-Wooldridge robust standard errors. Standardised "
               "residuals are checked with Ljung-Box, ARCH-LM and Engle-Ng sign-bias tests; persistence, half-life, the news "
               "impact curve and Value-at-Risk with Kupiec and Christoffersen backtests are reported.")
        refs = [R[k] for k in ("engle82", "bollerslev86", "elr", "gjr", "nelson", "bw", "engle_ng", "fs", "kupiec", "christoffersen")]
    else:   # mgarch
        txt = ("قُدِّرت التقلبات والارتباطات الشرطية المشتركة بنماذج GARCH متعددة المتغيرات: الارتباط الشرطي الثابت (CCC) "
               "والديناميكي (DCC) لإنجل على مرحلتين مع تصحيح الأخطاء المعيارية لأثر تقدير المرحلة الأولى (نيوي-ماكفادن 1994)، "
               "ونموذج BEKK الذي يلتقط انتقال الصدمات والتقلبات بين الأسواق، واختُبر "
               "ثبات الارتباط باختبار إنجل-شيبارد، وحُسبت نسبة التحوط الديناميكية.") if ar else (
               "Joint conditional volatilities and correlations are modelled with multivariate GARCH: Bollerslev's constant "
               "conditional correlation (CCC), Engle's two-step dynamic conditional correlation (DCC) and the BEKK model, which "
               "captures shock and volatility spillovers between markets. DCC standard errors are corrected for the first-stage "
               "estimation (two-step sandwich, Newey and McFadden 1994). Constant correlation is tested with the Engle-Sheppard "
               "test and dynamic hedge ratios are computed.")
        refs = [R[k] for k in ("ccc", "dcc", "bekk", "es01", "ks93", "nm94")]
    return txt, sorted(refs)


# ==========================================================
# Interpretation
# ==========================================================
def model_advice(orders: dict, lang: str = "en") -> str:
    """Which multivariate model fits the integration orders found (the usual applied decision tree)."""
    ar = lang == "ar"
    vals = [v for v in orders.values() if v is not None]
    if any(v is None for v in orders.values()) or any(v >= 2 for v in vals):
        return ("⚠️ " + ("بعض المتغيرات I(2) أو غير محسومة: لا تصلح ARDL؛ استخدم الفروق أو أعد النظر في التحويل." if ar else
                         "Some variables are I(2) or undetermined: ARDL is invalid; difference them or reconsider the transformation."))
    if all(v == 0 for v in vals):
        return ("كل المتغيرات مستقرة I(0) ← قدّر <b>VAR في المستويات</b> (أو ARDL)." if ar else
                "All variables are stationary I(0) → estimate a <b>VAR in levels</b> (or ARDL).")
    if all(v == 1 for v in vals):
        return ("كل المتغيرات I(1) ← اختبر التكامل المشترك (جوهانسن): إن وُجد فقدّر <b>VECM</b>، وإلا فقدّر <b>VAR في الفروق</b>. "
                "ويصلح ARDL أيضاً لعلاقة بمتغير تابع واحد." if ar else
                "All variables are I(1) → test for cointegration (Johansen): if present estimate a <b>VECM</b>, otherwise a "
                "<b>VAR in differences</b>. ARDL is also valid for a single-equation relation.")
    return ("خليط من I(0) وI(1) ← النهج المناسب هو <b>ARDL واختبار الحدود</b> (أو VAR بطريقة تودا-ياماموتو للسببية)." if ar else
            "A mix of I(0) and I(1) → the appropriate approach is <b>ARDL bounds testing</b> (or a Toda-Yamamoto VAR for causality).")


def interpret_johansen(jo: dict, lang: str = "en") -> str:
    ar = lang == "ar"
    rt, rm, n = jo["rank_trace"], jo["rank_max"], len(jo["names"])
    if ar:
        s = f"يشير اختبار الأثر إلى <b>{rt}</b> متجه تكامل مشترك، واختبار القيمة الذاتية العظمى إلى <b>{rm}</b> (عند 5%)."
        if rt == 0:
            s += " لا توجد علاقة توازنية طويلة الأجل: قدّر VAR في الفروق."
        elif rt == n:
            s += " الرتبة كاملة، أي أن المتغيرات مستقرة في المستوى: قدّر VAR في المستويات."
        else:
            s += f" توجد علاقة توازنية طويلة الأجل؛ قدّر VECM برتبة r = {rt}."
        if rt != rm:
            s += " وعند التعارض يُفضَّل عادةً اختبار الأثر لأنه أكثر متانة."
    else:
        s = f"The trace test indicates <b>{rt}</b> cointegrating vector(s), the maximum-eigenvalue test <b>{rm}</b> (5% level)."
        if rt == 0:
            s += " There is no long-run equilibrium: estimate a VAR in differences."
        elif rt == n:
            s += " Full rank means the variables are stationary in levels: estimate a VAR in levels."
        else:
            s += f" A long-run equilibrium exists; estimate a VECM with r = {rt}."
        if rt != rm:
            s += " When the two disagree, the trace test is usually preferred for its robustness."
    return s


def interpret_bounds(bt: dict, lang: str = "en") -> str:
    ar = lang == "ar"
    c5 = bt["critical"].iloc[1]
    v = bt["verdict_F"]
    if ar:
        s = (f"إحصاءة F لاختبار الحدود = {bt['F']:.3f} مقابل حدَّي 5%: I(0) = {c5['F I(0)']:.3f} وI(1) = {c5['F I(1)']:.3f}. ")
        s += {"cointegration": "الإحصاءة أكبر من الحد الأعلى ← <b>توجد علاقة توازنية طويلة الأجل</b> أياً كانت درجة تكامل المتغيرات.",
              "no cointegration": "الإحصاءة أقل من الحد الأدنى ← <b>لا توجد علاقة طويلة الأجل</b>؛ اكتفِ بنموذج الأجل القصير.",
              "inconclusive": "الإحصاءة بين الحدين ← <b>النتيجة غير حاسمة</b>؛ استعن بدرجات التكامل وبمعامل تصحيح الخطأ."}[v]
        if "t" in bt and "verdict_t" in bt:
            s += f" واختبار t ({bt['t']:.3f}) يؤدي إلى: {dict(cointegration='تكامل مشترك', inconclusive='غير حاسم').get(bt['verdict_t'], 'لا تكامل')}."
    else:
        s = (f"Bounds F = {bt['F']:.3f} against the 5% bounds I(0) = {c5['F I(0)']:.3f} and I(1) = {c5['F I(1)']:.3f}. ")
        s += {"cointegration": "It exceeds the upper bound → <b>a long-run level relationship exists</b> whatever the integration order of the regressors.",
              "no cointegration": "It is below the lower bound → <b>no long-run relationship</b>; model the short run only.",
              "inconclusive": "It lies between the bounds → <b>inconclusive</b>; use the integration orders and the error-correction term."}[v]
        if "verdict_t" in bt:
            s += f" The t test ({bt['t']:.3f}) points to: {bt['verdict_t']}."
    return s


def interpret_ecm(speed: float, half_life: float, lang: str = "en", per: str = "period") -> str:
    ar = lang == "ar"
    ok = -1 < speed < 0
    if ar:
        if not ok:
            return (f"معامل تصحيح الخطأ = {speed:.3f}؛ " + ("موجب، أي لا يوجد تقارب نحو التوازن — راجع النموذج." if speed >= 0
                                                         else "أقل من −1، ما يعني تذبذباً حول التوازن."))
        return (f"معامل تصحيح الخطأ = {speed:.3f} (سالب ومعنوي كما يُتوقع): يُصحَّح نحو {abs(speed) * 100:.1f}% من الانحراف عن التوازن "
                f"في كل فترة، ونصف العمر ≈ {half_life:.1f} فترة.")
    if not ok:
        return (f"Error-correction coefficient = {speed:.3f}; " + ("positive — no convergence to equilibrium; revisit the model." if speed >= 0
                                                              else "below −1 — oscillating adjustment."))
    return (f"Error-correction coefficient = {speed:.3f} (negative, as expected): about {abs(speed) * 100:.1f}% of any disequilibrium "
            f"is corrected each {per}; half-life ≈ {half_life:.1f} {per}s.")


def interpret_garch(r, diag: pd.DataFrame, lang: str = "en") -> str:
    ar = lang == "ar"
    st = r.stats
    pers, hl = st["Persistence"], st["Half-life"]
    P = []
    if ar:
        P.append(f"قُدِّر النموذج <b>{r.title}</b> على {st['N']} مشاهدة (AIC = {st['AIC']:.1f}).")
        P.append(f"الاستمرارية = {pers:.3f}: " + ("التقلب شديد الاستمرار؛ " if pers > 0.95 else "")
                 + (f"نصف عمر الصدمة ≈ {hl:.1f} فترة." if np.isfinite(hl) else "الصدمات دائمة الأثر."))
        if r.model == "gjr":
            g = r.p_("gamma1")
            P.append(f"معامل عدم التماثل γ = {g:.3f}" + ("؛ الصدمات السالبة (الأخبار السيئة) ترفع التقلب أكثر من الموجبة (أثر الرافعة)." if g > 0 else "."))
        if r.model == "egarch":
            g = r.p_("gamma1")
            P.append(f"γ = {g:.3f}" + ("؛ سالب، أي أن الأخبار السيئة تزيد التقلب أكثر (أثر الرافعة)." if g < 0 else "."))
        if r.mean.startswith("m"):
            P.append(f"علاوة المخاطرة λ = {r.p_('lambda'):.3f}: " + ("العائد يرتفع مع المخاطرة." if r.p_('lambda') > 0 else "لا علاوة موجبة."))
        if r.dist in ("std", "sstd"):
            P.append(f"درجات الحرية ν = {r.p_('shape'):.2f}: ذيول أثقل من الطبيعي.")
    else:
        P.append(f"<b>{r.title}</b> estimated on {st['N']} observations (AIC = {st['AIC']:.1f}).")
        P.append(f"Persistence = {pers:.3f}: " + ("volatility is highly persistent; " if pers > 0.95 else "")
                 + (f"a shock's half-life is ≈ {hl:.1f} periods." if np.isfinite(hl) else "shocks are permanent."))
        if r.model == "gjr":
            g = r.p_("gamma1")
            P.append(f"Asymmetry γ = {g:.3f}" + (": negative shocks (bad news) raise volatility more than positive ones (leverage effect)." if g > 0 else "."))
        if r.model == "egarch":
            g = r.p_("gamma1")
            P.append(f"γ = {g:.3f}" + (": negative — bad news raises volatility more (leverage effect)." if g < 0 else "."))
        if r.mean.startswith("m"):
            P.append(f"Risk premium λ = {r.p_('lambda'):.3f}: " + ("expected return rises with risk." if r.p_('lambda') > 0 else "no positive premium."))
        if r.dist in ("std", "sstd"):
            P.append(f"Degrees of freedom ν = {r.p_('shape'):.2f}: tails are fatter than normal.")
    d = diag.set_index("Test")["p-value"]
    k2 = [k for k in d.index if "z²" in k]
    if k2:
        ok = d[k2[0]] >= 0.05
        P.append(("لا يتبقى تجمع للتقلب في البواقي المعيارية — النموذج كافٍ." if ok else "⚠️ يتبقى تجمع للتقلب: جرّب رتباً أعلى أو نموذجاً غير متماثل.") if ar else
                 ("No volatility clustering remains in the standardised residuals — the model is adequate." if ok else
                  "⚠️ Volatility clustering remains: try higher orders or an asymmetric model."))
    for n_ in r.notes:
        P.append("⚠️ " + n_)
    return "<br>".join(P)


# ==========================================================
# Code export
# ==========================================================
def code(kind: str, file_name: str = "your_data.csv", **kw) -> dict:
    v = kw.get("vars", ["y1", "y2"])
    vs = " ".join(v)
    p = kw.get("p", 2)
    cs = ", ".join(f'"{x}"' for x in v)
    rv = ", ".join(v)
    py = ["import numpy as np, pandas as pd", f'df = pd.read_csv("{file_name}")']
    rr = [f'df <- read.csv("{file_name}")']
    st = [f'import delimited "{file_name}", clear varnames(1) case(preserve)', "gen t = _n", "tsset t"]
    if kind == "var":
        typ = kw.get("type", "const")
        py += ["from statsmodels.tsa.api import VAR", f"Y = df[[{cs}]]",
               f"print(VAR(Y).select_order(8).summary())",
               f"res = VAR(Y).fit({p}, trend='{dict(const='c', both='ct', trend='ct', none='n')[typ]}')", "print(res.summary())",
               "print(res.is_stable(), res.test_whiteness(12).summary(), res.test_normality().summary())",
               f"print(res.test_causality('{v[0]}', {[x for x in v[1:]]}, kind='wald').summary())",
               "res.irf(10).plot(orth=True); res.fevd(10).summary()", "print(res.forecast(Y.values[-res.k_ar:], 8))"]
        rr = ["library(vars)"] + rr + [f"Y <- df[, c({cs})]", f'VARselect(Y, lag.max = 8, type = "{typ}")',
                                       f'v <- VAR(Y, p = {p}, type = "{typ}")', "summary(v); roots(v)",
                                       "serial.test(v, lags.pt = 12); serial.test(v, lags.bg = 5, type = 'BG')",
                                       "normality.test(v); arch.test(v, lags.multi = 5)",
                                       f'causality(v, cause = "{v[0]}")',
                                       "plot(irf(v, n.ahead = 10, boot = TRUE, runs = 500)); fevd(v, n.ahead = 10)",
                                       "predict(v, n.ahead = 8)"]
        st += [f"varsoc {vs}, maxlag(8)", f"var {vs}, lags(1/{p})", "varstable", "varlmar, mlag(5)", "varnorm",
               "vargranger", "irf create m1, set(myirf) step(10) replace", "irf graph oirf", "irf table fevd",
               "fcast compute f_, step(8)"]
    elif kind == "svar":
        py += ["from statsmodels.tsa.vector_ar.svar_model import SVAR", f"Y = df[[{cs}]]",
               "A = np.array([['E', 0, 0], ['E', 'E', 0], ['E', 'E', 'E']])   # 'E' = free (adapt to your K)",
               f"res = SVAR(Y, svar_type='A', A=A).fit(maxlags={p})", "res.irf(10).plot()"]
        rr = ["library(vars)"] + rr + [f"v <- VAR(df[, c({cs})], p = {p}, type = 'const')",
                                       f"Amat <- diag({len(v)}); Amat[lower.tri(Amat)] <- NA",
                                       "s <- SVAR(v, Amat = Amat, estmethod = 'scoring'); s$A; s$Ase   # summary(s) can fail in vars when printing the LR test",
                                       "plot(irf(s, n.ahead = 10)); BQ(v)   # long-run (Blanchard-Quah)"]
        st += [f"matrix A = (1,0,0 \\ .,1,0 \\ .,.,1)   // adapt to K = {len(v)}", f"matrix B = (.,0,0 \\ 0,.,0 \\ 0,0,.)",
               f"svar {vs}, lags(1/{p}) aeq(A) beq(B)", "irf create s1, set(svirf) step(10) replace", "irf graph sirf"]
    elif kind == "coint":
        case = kw.get("case", 3)
        K = kw.get("K", 2)
        ec = {1: "none", 2: "const", 3: "none", 4: "trend", 5: "none"}[case]
        sttr = {1: "none", 2: "rconstant", 3: "constant", 4: "rtrend", 5: "trend"}[case]
        py += ["from statsmodels.tsa.vector_ar.vecm import coint_johansen, select_coint_rank",
               "from statsmodels.tsa.stattools import coint", f"Y = df[[{cs}]]",
               f"print(coint(Y['{v[0]}'], Y[{v[1:]}]))   # Engle-Granger",
               f"j = coint_johansen(Y, det_order={dict([(1, -1), (2, 0), (3, 0), (4, 1), (5, 1)])[case]}, k_ar_diff={K - 1})",
               "print(j.lr1, j.cvt)   # trace statistics and critical values",
               f"print(select_coint_rank(Y, det_order={dict([(1, -1), (2, 0), (3, 0), (4, 1), (5, 1)])[case]}, k_ar_diff={K - 1}).summary())"]
        rr = ["library(urca); library(cointReg)"] + rr + [
            f"j <- ca.jo(df[, c({cs})], type = 'trace', ecdet = '{ec}', K = {K}, spec = 'transitory'); summary(j)",
            f"summary(ca.jo(df[, c({cs})], type = 'eigen', ecdet = '{ec}', K = {K}))",
            f"eg <- lm({v[0]} ~ {' + '.join(v[1:])}, data = df); summary(ur.df(resid(eg), type = 'none', selectlags = 'AIC'))",
            f"cointRegFM(x = as.matrix(df[, c({', '.join(repr(x) for x in v[1:])})]), y = df${v[0]}, deter = matrix(1, nrow(df), 1))",
            f"cointRegD(x = as.matrix(df[, c({', '.join(repr(x) for x in v[1:])})]), y = df${v[0]}, deter = matrix(1, nrow(df), 1), n.lag = 2, n.lead = 2)"]
        st += [f"vecrank {vs}, lags({K}) trend({sttr}) max", f"regress {v[0]} {' '.join(v[1:])}", "predict u, resid",
               "dfuller u, noconstant lags(2)   // compare with Engle-Granger (MacKinnon) critical values",
               "* ssc install ghansen ; ssc install cointreg",
               f"ghansen {v[0]} {' '.join(v[1:])}, break(level) lagmethod(aic) maxlags(4)",
               f"cointreg {v[0]} {' '.join(v[1:])}, est(fmols)", f"cointreg {v[0]} {' '.join(v[1:])}, est(dols) nlead(2) nlag(2)",
               f"cointreg {v[0]} {' '.join(v[1:])}, est(ccr)"]
    elif kind == "vecm":
        case, K, r = kw.get("case", 3), kw.get("K", 2), kw.get("r", 1)
        ec = {1: "none", 2: "const", 3: "none", 4: "trend", 5: "none"}[case]
        sttr = {1: "none", 2: "rconstant", 3: "constant", 4: "rtrend", 5: "trend"}[case]
        smd = {1: "n", 2: "ci", 3: "co", 4: "colo", 5: "coli"}[case]
        py += ["from statsmodels.tsa.vector_ar.vecm import VECM", f"Y = df[[{cs}]]",
               f"m = VECM(Y, k_ar_diff={K - 1}, coint_rank={r}, deterministic='{smd}').fit()", "print(m.summary())",
               "m.irf(10).plot(orth=True)"]
        rr = ["library(urca); library(vars)"] + rr + [
            f"j <- ca.jo(df[, c({cs})], type = 'trace', ecdet = '{ec}', K = {K}, spec = 'transitory')",
            f"cajorls(j, r = {r})   # β normalised, α and Γ", f"alrtest(j, A = diag({len(v)})[, -1, drop = FALSE], r = {r})   # weak exogeneity of the first variable",
            f"plot(irf(vec2var(j, r = {r}), n.ahead = 10))"]
        st += [f"vec {vs}, lags({K}) rank({r}) trend({sttr})", "vecstable", "veclmar", "vecnorm",
               f"test [_ce1]_cons = 0   // example restriction", "irf create v1, set(vecirf) step(10) replace", "irf graph oirf"]
    elif kind in ("ardl", "nardl"):
        y, xs, order, case = kw.get("y", "y"), kw.get("x", ["x"]), kw.get("order", (1, 1)), kw.get("case", 3)
        fixed = kw.get("fixed", [])
        fx = (" | " + " + ".join(fixed)) if fixed else ""
        tr = " + trend(" + y + ")" if case in (4, 5) else ""
        if tr and not fixed:
            fx = " |"
        o = ", ".join(map(str, order))
        if kind == "nardl":
            a = kw.get("asym", xs[0])
            pre_py = [f"dx = df['{a}'].diff().fillna(0)", f"df['{a}_pos'] = dx.clip(lower=0).cumsum()", f"df['{a}_neg'] = dx.clip(upper=0).cumsum()"]
            pre_r = [f"dx <- c(0, diff(df${a}))", f"df${a}_pos <- cumsum(pmax(dx, 0)); df${a}_neg <- cumsum(pmin(dx, 0))"]
            pre_st = [f"gen d{a} = D.{a}", f"replace d{a} = 0 if missing(d{a})", f"gen {a}_pos = sum(max(d{a}, 0))", f"gen {a}_neg = sum(min(d{a}, 0))"]
        else:
            pre_py, pre_r, pre_st = [], [], []
        py += pre_py + ["from statsmodels.tsa.ardl import ARDL, UECM, ardl_select_order",
                        f"sel = ardl_select_order(df['{y}'], 4, df[{xs}], 4, trend='{'ct' if case in (4, 5) else 'c'}', ic='aic')",
                        f"res = ARDL(df['{y}'], {order[0]}, df[{xs}], {dict(zip(xs, order[1:]))}, trend='{'ct' if case in (4, 5) else 'c'}').fit()",
                        "print(res.summary())",
                        f"u = UECM.from_ardl(ARDL(df['{y}'], {order[0]}, df[{xs}], {dict(zip(xs, order[1:]))}, trend='{'ct' if case in (4, 5) else 'c'}')).fit()",
                        f"print(u.bounds_test(case={case}))   # PSS bounds test"]
        rr = ["library(ARDL)"] + rr + pre_r + [
            f"m <- ardl({y} ~ {' + '.join(xs)}{fx}{tr}, data = df, order = c({o}))", "summary(m)",
            f"auto_ardl({y} ~ {' + '.join(xs)}{fx}{tr}, data = df, max_order = 4)$top_orders",
            f"bounds_f_test(m, case = {case}); bounds_f_test(m, case = {case}, exact = TRUE)",
            *( [f"bounds_t_test(m, case = {case})"] if case in (1, 3, 5) else []),
            "multipliers(m)   # long-run", f"summary(recm(m, case = {case}))   # ECM"]
        if kind == "nardl":
            rr += [f"library(aod); wald.test(b = coef(uecm(m)), Sigma = vcov(uecm(m)), L = ...)   # symmetry tests"]
        st += pre_st + ["* ssc install ardl", f"ardl {y} {' '.join(xs)}{' ' + ' '.join(fixed) if fixed else ''}, lags({' '.join(map(str, order))}) ec{' trendvar(t)' if case in (4, 5) else ''}{' exog(' + ' '.join(fixed) + ')' if fixed else ''}",
                        "estat ectest   // PSS bounds test with Kripfganz-Schneider critical values", "estat bgodfrey; estat hettest; estat sbcusum"]
    elif kind == "qardl":
        y, xs, order = kw.get("y", "y"), kw.get("x", ["x"]), kw.get("order", (1, 1))
        py += ["import statsmodels.formula.api as smf"]
        terms = [f"{y}.shift({i})" for i in range(1, order[0] + 1)]
        py += [f"df['{y}_l{i}'] = df['{y}'].shift({i})" for i in range(1, order[0] + 1)]
        py += [f"df['{x}_l{L}'] = df['{x}'].shift({L})" for x, q in zip(xs, order[1:]) for L in range(1, q + 1)]
        rhs = " + ".join([f"{y}_l{i}" for i in range(1, order[0] + 1)] + [x if L == 0 else f"{x}_l{L}" for x, q in zip(xs, order[1:]) for L in range(q + 1)])
        py += [f"for tau in (0.25, 0.5, 0.75):", f"    print(tau, smf.quantreg('{y} ~ {rhs}', df.dropna()).fit(q=tau).params)"]
        lagdefs = [f"df${y}_l{i} <- lagv(df${y}, {i})" for i in range(1, order[0] + 1)]
        lagdefs += [f"df${x}_l{L} <- lagv(df${x}, {L})" for x, q in zip(xs, order[1:]) for L in range(1, q + 1)]
        rhs_r = " + ".join([f"{y}_l{i}" for i in range(1, order[0] + 1)] + [x if L == 0 else f"{x}_l{L}" for x, q in zip(xs, order[1:]) for L in range(q + 1)])
        rr = ["library(quantreg)"] + rr + ["lagv <- function(x, k) c(rep(NA, k), head(x, length(x) - k))"] + lagdefs + [
            "for (tau in c(0.25, 0.5, 0.75)) print(summary(rq(" + f"{y} ~ {rhs_r}" + ", data = na.omit(df), tau = tau), se = 'nid'))",
            "# long-run beta(tau) = sum(theta) / (1 - sum(phi)); speed rho(tau) = sum(phi) - 1"]
        st += [f"qreg {y} L(1/{order[0]}).{y} " + " ".join(f"L(0/{q}).{x}" for x, q in zip(xs, order[1:])) + ", quantile(0.5)",
               "* repeat for quantile(0.25) and quantile(0.75); long-run β(τ) = Σθ / (1 − Σφ) via nlcom"]
    elif kind == "garch":
        model, d, mean = kw.get("model", "garch"), kw.get("dist", "norm"), kw.get("mean", "const")
        y = kw.get("y", "r")
        ap = {"norm": "normal", "std": "t", "ged": "ged", "sstd": "skewt"}[d]
        vol = {"garch": "vol='GARCH', p=1, q=1", "arch": "vol='ARCH', p=1", "gjr": "vol='GARCH', p=1, o=1, q=1",
               "egarch": "vol='EGARCH', p=1, o=1, q=1"}[model]
        py += ["from arch import arch_model   # pip install arch", f"r = 100 * np.log(df['{y}']).diff().dropna()   # or your return column",
               f"res = arch_model(r, mean='{'Zero' if mean == 'zero' else 'AR' if mean == 'ar1' else 'Constant'}'{', lags=1' if mean == 'ar1' else ''}, {vol}, dist='{ap}').fit(disp='off')",
               "print(res.summary())", "print(res.forecast(horizon=10).variance.iloc[-1])"]
        fm = {"garch": "garch(1,1)", "arch": "garch(1,0)", "gjr": "aparch(1,1)", "egarch": "garch(1,1)"}[model]
        rr = ["library(fGarch)   # or rugarch for EGARCH / GARCH-M"] + rr + [
            f"r <- 100 * diff(log(df${y}))",
            f"m <- garchFit(~ {'arma(1,0) + ' if mean == 'ar1' else ''}{fm}, data = r, cond.dist = '{d}', include.mean = {str(mean != 'zero').upper()}{', include.delta = FALSE, delta = 2' if model == 'gjr' else ''})",
            "summary(m)   # Ljung-Box on z and z², LM-ARCH", "predict(m, n.ahead = 10)",
            *(["library(rugarch)", f"sp <- ugarchspec(variance.model = list(model = '{'eGARCH' if model == 'egarch' else 'sGARCH'}', garchOrder = c(1, 1)), mean.model = list(armaOrder = c(0, 0), archm = {'TRUE' if mean.startswith('m') else 'FALSE'}), distribution.model = '{d}')",
               "ugarchfit(sp, r)"] if model == "egarch" or mean.startswith("m") else [])]
        sd = {"norm": "", "std": " distribution(t)", "ged": " distribution(ged)", "sstd": " distribution(t)"}[d]
        spec = {"garch": "arch(1) garch(1)", "arch": "arch(1)", "gjr": "arch(1) tarch(1) garch(1)", "egarch": "earch(1) egarch(1)"}[model]
        st += [f"gen r = 100 * D.ln({y})", f"arch r{' L.r' if mean == 'ar1' else ''}, {spec}{sd}{' archm' if mean == 'm_var' else ''} vce(robust)",
               "predict h, variance", "predict z, residuals", "estat ic"]
    else:   # mgarch
        model = kw.get("model", "DCC")
        py += ["# DCC / BEKK: Python 'mgarch' packages are immature — use R rmgarch / MTS or Stata mgarch",
               "from arch import arch_model", f"R = 100 * np.log(df[[{cs}]]).diff().dropna()",
               "Z = pd.DataFrame({c: (lambda f: f.std_resid)(arch_model(R[c], vol='GARCH').fit(disp='off')) for c in R})",
               "print(Z.corr())   # CCC correlation"]
        rr = ["library(rmgarch); library(rugarch)"] + rr + [
            f"R <- 100 * apply(log(df[, c({cs})]), 2, diff)",
            "u <- ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(1, 1)), distribution.model = 'norm')",
            f"sp <- dccspec(multispec(replicate({len(v)}, u)), dccOrder = c(1, 1), distribution = 'mvnorm')",
            "fit <- dccfit(sp, R); fit; plot(fit, which = 4)",
            "library(MTS); BEKK11(scale(R, scale = FALSE))   # full BEKK(1,1)"]
        st += [f"foreach v in {vs} {{", "    gen r_`v' = 100 * D.ln(`v')", "}",
               f"mgarch {'dcc' if model == 'DCC' else 'ccc'} ({' '.join('r_' + x for x in v)} = ), arch(1) garch(1)",
               "predict H*, variance", "estat ic"]
    return {"Python": "\n".join(py) + "\n", "R": "\n".join(rr) + "\n", "Stata": "\n".join(st) + "\n"}
