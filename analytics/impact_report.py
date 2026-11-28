# ==========================================
# Aetheris Research - v1.11 reports: methodology paragraphs (AR/EN), references and Python / R / Stata code for
# quantile regression, spatial econometrics, policy evaluation and climate & energy economics (Catalogue §1.14, §1.16,
# §1.20, §1.26).
# ==========================================

from __future__ import annotations

import numpy as np

REFS = {
    "kb78": "Koenker, R. & Bassett, G. (1978). Regression quantiles. Econometrica, 46(1), 33–50.",
    "koenker05": "Koenker, R. (2005). Quantile Regression. Cambridge University Press.",
    "hs88": "Hall, P. & Sheather, S. J. (1988). On the distribution of a studentized quantile. Journal of the Royal Statistical Society B, 50(3), 381–391.",
    "kb82": "Koenker, R. & Bassett, G. (1982). Tests of linear hypotheses and l1 estimation. Econometrica, 50(6), 1577–1583.",
    "km99": "Koenker, R. & Machado, J. A. F. (1999). Goodness of fit and related inference processes for quantile regression. Journal of the American Statistical Association, 94(448), 1296–1310.",
    "mss19": "Machado, J. A. F. & Santos Silva, J. M. C. (2019). Quantiles via moments. Journal of Econometrics, 213(1), 145–173.",
    "hlow16": "Han, H., Linton, O., Oka, T. & Whang, Y.-J. (2016). The cross-quantilogram: measuring quantile dependence and testing directional predictability between time series. Journal of Econometrics, 193(1), 251–270.",
    "moran50": "Moran, P. A. P. (1950). Notes on continuous stochastic phenomena. Biometrika, 37(1/2), 17–23.",
    "cliff81": "Cliff, A. D. & Ord, J. K. (1981). Spatial Processes: Models and Applications. Pion.",
    "anselin95": "Anselin, L. (1995). Local indicators of spatial association — LISA. Geographical Analysis, 27(2), 93–115.",
    "anselin88": "Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer.",
    "abfy96": "Anselin, L., Bera, A. K., Florax, R. & Yoon, M. J. (1996). Simple diagnostic tests for spatial dependence. Regional Science and Urban Economics, 26(1), 77–104.",
    "lp09": "LeSage, J. & Pace, R. K. (2009). Introduction to Spatial Econometrics. CRC Press.",
    "elhorst14": "Elhorst, J. P. (2014). Spatial Econometrics: From Cross-Sectional Data to Spatial Panels. Springer.",
    "elhorst03": "Elhorst, J. P. (2003). Specification and estimation of spatial panel data models. International Regional Science Review, 26(3), 244–268.",
    "fbc02": "Fotheringham, A. S., Brunsdon, C. & Charlton, M. (2002). Geographically Weighted Regression. Wiley.",
    "bivand": "Bivand, R. S., Pebesma, E. & Gómez-Rubio, V. (2013). Applied Spatial Data Analysis with R (2nd ed.). Springer.",
    "neyman": "Imbens, G. W. & Rubin, D. B. (2015). Causal Inference for Statistics, Social, and Biomedical Sciences. Cambridge University Press.",
    "lin13": "Lin, W. (2013). Agnostic notes on regression adjustments to experimental data: reexamining Freedman's critique. Annals of Applied Statistics, 7(1), 295–318.",
    "duflo": "Duflo, E., Glennerster, R. & Kremer, M. (2007). Using randomization in development economics research: a toolkit. Handbook of Development Economics, 4, 3895–3962.",
    "ia94": "Imbens, G. W. & Angrist, J. D. (1994). Identification and estimation of local average treatment effects. Econometrica, 62(2), 467–475.",
    "sy05": "Stock, J. H. & Yogo, M. (2005). Testing for weak instruments in linear IV regression. In Identification and Inference for Econometric Models, 80–108. Cambridge University Press.",
    "ap09": "Angrist, J. D. & Pischke, J.-S. (2009). Mostly Harmless Econometrics. Princeton University Press.",
    "cs21": "Callaway, B. & Sant'Anna, P. H. C. (2021). Difference-in-differences with multiple time periods. Journal of Econometrics, 225(2), 200–230.",
    "sa21": "Sun, L. & Abraham, S. (2021). Estimating dynamic treatment effects in event studies with heterogeneous treatment effects. Journal of Econometrics, 225(2), 175–199.",
    "gb21": "Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. Journal of Econometrics, 225(2), 254–277.",
    "rambachan": "Roth, J., Sant'Anna, P. H. C., Bilinski, A. & Poe, J. (2023). What's trending in difference-in-differences? Journal of Econometrics, 235(2), 2218–2244.",
    "adh10": "Abadie, A., Diamond, A. & Hainmueller, J. (2010). Synthetic control methods for comparative case studies. Journal of the American Statistical Association, 105(490), 493–505.",
    "abadie21": "Abadie, A. (2021). Using synthetic controls: feasibility, data requirements, and methodological aspects. Journal of Economic Literature, 59(2), 391–425.",
    "aahiw21": "Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W. & Wager, S. (2021). Synthetic difference-in-differences. American Economic Review, 111(12), 4088–4118.",
    "cct14": "Calonico, S., Cattaneo, M. D. & Titiunik, R. (2014). Robust nonparametric confidence intervals for regression-discontinuity designs. Econometrica, 82(6), 2295–2326.",
    "cjm20": "Cattaneo, M. D., Jansson, M. & Ma, X. (2020). Simple local polynomial density estimators. Journal of the American Statistical Association, 115(531), 1449–1455.",
    "lee10": "Lee, D. S. & Lemieux, T. (2010). Regression discontinuity designs in economics. Journal of Economic Literature, 48(2), 281–355.",
    "rr83": "Rosenbaum, P. R. & Rubin, D. B. (1983). The central role of the propensity score in observational studies for causal effects. Biometrika, 70(1), 41–55.",
    "ai06": "Abadie, A. & Imbens, G. W. (2006). Large sample properties of matching estimators for average treatment effects. Econometrica, 74(1), 235–267.",
    "hir03": "Hirano, K., Imbens, G. W. & Ridder, G. (2003). Efficient estimation of average treatment effects using the estimated propensity score. Econometrica, 71(4), 1161–1189.",
    "rrz94": "Robins, J. M., Rotnitzky, A. & Zhao, L. P. (1994). Estimation of regression coefficients when some regressors are not always observed. Journal of the American Statistical Association, 89(427), 846–866.",
    "austin11": "Austin, P. C. (2011). An introduction to propensity score methods for reducing the effects of confounding in observational studies. Multivariate Behavioral Research, 46(3), 399–424.",
    "sr09": "Schlenker, W. & Roberts, M. J. (2009). Nonlinear temperature effects indicate severe damages to U.S. crop yields under climate change. PNAS, 106(37), 15594–15598.",
    "be69": "Baskerville, G. L. & Emin, P. (1969). Rapid estimation of heat accumulation from maximum and minimum temperatures. Ecology, 50(3), 514–517.",
    "dg07": "Deschênes, O. & Greenstone, M. (2007). The economic impacts of climate change: evidence from agricultural output and random fluctuations in weather. American Economic Review, 97(1), 354–385.",
    "dell14": "Dell, M., Jones, B. F. & Olken, B. A. (2014). What do we learn from the weather? The new climate–economy literature. Journal of Economic Literature, 52(3), 740–798.",
    "mns94": "Mendelsohn, R., Nordhaus, W. D. & Shaw, D. (1994). The impact of global warming on agriculture: a Ricardian analysis. American Economic Review, 84(4), 753–771.",
    "bhm15": "Burke, M., Hsiang, S. M. & Miguel, E. (2015). Global non-linear effect of temperature on economic production. Nature, 527(7577), 235–239.",
    "nordhaus17": "Nordhaus, W. D. (2017). Revisiting the social cost of carbon. PNAS, 114(7), 1518–1523.",
    "matthews09": "Matthews, H. D., Gillett, N. P., Stott, P. A. & Zickfeld, K. (2009). The proportionality of global warming to cumulative carbon emissions. Nature, 459(7248), 829–832.",
    "dr97": "Dietz, T. & Rosa, E. A. (1997). Effects of population and affluence on CO2 emissions. PNAS, 94(1), 175–179.",
    "yrd03": "York, R., Rosa, E. A. & Dietz, T. (2003). STIRPAT, IPAT and ImPACT: analytic tools for unpacking the driving forces of environmental impacts. Ecological Economics, 46(3), 351–365.",
    "hk70": "Hoerl, A. E. & Kennard, R. W. (1970). Ridge regression: biased estimation for nonorthogonal problems. Technometrics, 12(1), 55–67.",
    "gk95": "Grossman, G. M. & Krueger, A. B. (1995). Economic growth and the environment. Quarterly Journal of Economics, 110(2), 353–377.",
    "lm10": "Lind, J. T. & Mehlum, H. (2010). With or without U? The appropriate test for a U-shaped relationship. Oxford Bulletin of Economics and Statistics, 72(1), 109–118.",
    "stern04": "Stern, D. I. (2004). The rise and fall of the environmental Kuznets curve. World Development, 32(8), 1419–1439.",
    "tapio05": "Tapio, P. (2005). Towards a theory of decoupling: degrees of decoupling in the EU and the case of road traffic in Finland between 1970 and 2001. Transport Policy, 12(2), 137–151.",
    "ang05": "Ang, B. W. (2005). The LMDI approach to decomposition analysis: a practical guide. Energy Policy, 33(7), 867–871.",
    "kaya": "Kaya, Y. & Yokobori, K. (1997). Environment, Energy, and Economy: Strategies for Sustainability. United Nations University Press.",
}


def fmt_p(p):
    return "" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


TEXT = {
    "quantile": (
        "Conditional quantile functions were estimated by quantile regression (Koenker & Bassett 1978) over a grid of quantiles. Standard "
        "errors follow Koenker (2005): the Hendricks-Koenker sandwich with the Hall-Sheather bandwidth ('nid', robust to heteroskedasticity), "
        "the iid sparsity estimator or Powell's kernel estimator. Equality of slopes across quantiles was tested with the joint Wald test of "
        "Koenker & Bassett (1982), and goodness of fit is summarised by the pseudo-R¹ of Koenker & Machado (1999).",
        "قُدِّرت دوال الكميات الشرطية بالانحدار الكمّي (Koenker & Bassett 1978) على شبكة من الكميات، واتّبعت الأخطاء المعيارية Koenker (2005): "
        "مقدّر الساندويتش لهندركس-كوينكر بنطاق هول-شيذر ('nid'، المتين ضد عدم تجانس التباين)، أو مقدّر الكثافة تحت فرض التجانس، أو مقدّر "
        "النواة لباول. واختُبر تساوي الميول عبر الكميات باختبار والد المشترك (Koenker & Bassett 1982)، ولُخّصت جودة التوفيق بمعامل R¹ الكاذب "
        "لكوينكر وماتشادو (1999).",
        ("kb78", "koenker05", "hs88", "kb82", "km99")),
    "mmqr": (
        "Panel quantile effects were estimated by the method of moments quantile regression of Machado & Santos Silva (2019): a location-scale "
        "model with individual effects in both location and scale, the location by within-group OLS, the scale by a within regression of the "
        "absolute residuals, and β(τ) = β + q(τ)γ with q(τ) the τ-quantile of the standardised residual. Standard errors were obtained by a "
        "cluster (unit) bootstrap.",
        "قُدِّرت آثار الكميات في بيانات البانل بطريقة انحدار الكميات بالعزوم (Machado & Santos Silva 2019): نموذج موقع-مقياس بآثار فردية في "
        "الموقع والمقياس، يُقدَّر فيه الموقع بالمربعات الصغرى داخل المجموعات، والمقياس بانحدار داخلي للبواقي المطلقة، ثم β(τ) = β + q(τ)γ حيث "
        "q(τ) هي الكمية τ للبواقي المعيارية. وحُسبت الأخطاء المعيارية ببوتستراب عنقودي على مستوى الوحدة.",
        ("mss19", "koenker05")),
    "cq": (
        "Directional predictability in the tails was examined with the cross-quantilogram of Han, Linton, Oka & Whang (2016), the "
        "correlation of quantile-hit processes at lag k, with the portmanteau Q(p) statistic; confidence bands and critical values were "
        "obtained by the stationary bootstrap.",
        "فُحصت القدرة التنبؤية الاتجاهية في الأطراف بمخطط الكميات المتقاطع (Han et al. 2016)، وهو ارتباط عمليات تجاوز الكميات عند الفجوة k، مع "
        "إحصاءة Q(p) الإجمالية؛ وحُسبت حدود الثقة والقيم الحرجة بالبوتستراب الساكن.",
        ("hlow16",)),
    "spatial_auto": (
        "Spatial dependence was assessed with a row-standardised spatial weights matrix. Global autocorrelation was measured by Moran's I "
        "(Moran 1950) and Geary's C with inference under normality and randomisation (Cliff & Ord 1981) and by a permutation test; local "
        "clusters and outliers were identified with local Moran statistics (LISA; Anselin 1995) using conditional permutation pseudo p-values.",
        "قيس الاعتماد المكاني باستخدام مصفوفة أوزان مكانية معيارية الصفوف؛ فقيس الارتباط الذاتي الكلي بمعامل موران I (Moran 1950) ومعامل جيري C "
        "مع الاستدلال تحت فرضَي الطبيعية والعشوائية (Cliff & Ord 1981) وباختبار التباديل، وحُدّدت العناقيد والقيم الشاذة محلياً بإحصاءات موران "
        "المحلية (LISA؛ Anselin 1995) بقيم p تبادلية شرطية.",
        ("moran50", "cliff81", "anselin95", "bivand")),
    "spatial_reg": (
        "Lagrange multiplier tests and their robust versions (Anselin et al. 1996) on OLS residuals guided the specification. Spatial lag "
        "(SAR), spatial error (SEM), spatial Durbin (SDM, SDEM) and SAC models were estimated by maximum likelihood with the eigenvalue "
        "log-determinant (Anselin 1988). Because a spatial lag makes coefficients non-marginal, direct, indirect (spillover) and total "
        "impacts were computed following LeSage & Pace (2009), with standard errors from simulated parameter draws; models were compared by "
        "log-likelihood and AIC (Elhorst 2014).",
        "استرشد تحديد النموذج باختبارات مضاعف لاغرانج ونسخها المتينة (Anselin et al. 1996) على بواقي OLS، ثم قُدّرت نماذج الإبطاء المكاني (SAR) "
        "والخطأ المكاني (SEM) ودوربن المكاني (SDM وSDEM) ونموذج SAC بالإمكان الأعظم باستخدام اللوغاريتم المحدِّد عبر القيم الذاتية (Anselin 1988). "
        "ولأن الإبطاء المكاني يجعل المعاملات غير حدّية، حُسبت الآثار المباشرة وغير المباشرة (الانتشارية) والكلية وفق LeSage & Pace (2009) "
        "بأخطاء معيارية من سحب المحاكاة، وقورنت النماذج بدالة الإمكان ومعيار AIC (Elhorst 2014).",
        ("abfy96", "anselin88", "lp09", "elhorst14")),
    "gwr": (
        "Spatially varying relationships were estimated by geographically weighted regression (Fotheringham, Brunsdon & Charlton 2002) with "
        "an adaptive bisquare kernel; the bandwidth (number of nearest neighbours) was selected by minimising the corrected AIC. Local "
        "coefficients, standard errors and t-values were mapped.",
        "قُدِّرت العلاقات المتغيّرة مكانياً بالانحدار الموزون جغرافياً (Fotheringham et al. 2002) بنواة ثنائية التربيع تكيفية، واختير عرض "
        "النطاق (عدد أقرب الجيران) بتصغير معيار AIC المصحح، ورُسمت خرائط المعاملات المحلية وأخطائها وقيم t.",
        ("fbc02",)),
    "spatial_panel": (
        "A fixed-effects spatial panel model was estimated by concentrated maximum likelihood after the within transformation (Elhorst 2003, "
        "2014), with the log-likelihood including T·ln|I − ρW|; for the spatial lag specification direct, indirect and total impacts are "
        "reported.",
        "قُدّر نموذج بانل مكاني بآثار ثابتة بالإمكان الأعظم المركّز بعد التحويل الداخلي (Elhorst 2003، 2014) مع تضمين T·ln|I − ρW| في دالة "
        "الإمكان، وتُعرض للنموذج ذي الإبطاء المكاني الآثار المباشرة وغير المباشرة والكلية.",
        ("elhorst03", "elhorst14", "lp09")),
    "rct": (
        "Average treatment effects were estimated by the difference in means with Neyman (HC2) standard errors and, to gain precision, by "
        "the fully interacted regression adjustment of Lin (2013) with centred baseline covariates; covariate balance was checked with "
        "standardised mean differences and a joint orthogonality F test. With imperfect compliance the intention-to-treat effect was "
        "scaled by the first stage to recover the local average treatment effect (Imbens & Angrist 1994).",
        "قُدّر متوسط أثر المعالجة بالفرق بين المتوسطات مع أخطاء نيمان المعيارية (HC2)، ولزيادة الدقة بتعديل الانحدار المتفاعل كلياً (Lin 2013) "
        "بالمتغيرات الأساسية الممركزة، وفُحص توازن المتغيرات بالفروق المعيارية للمتوسطات واختبار F المشترك للتعامد. ومع عدم الامتثال الكامل "
        "قُسِم أثر النية في المعالجة على المرحلة الأولى للحصول على متوسط الأثر المحلي (Imbens & Angrist 1994).",
        ("neyman", "lin13", "duflo", "ia94")),
    "iv": (
        "The causal effect of the endogenous regressor was estimated by two-stage least squares with heteroskedasticity-robust standard errors; "
        "instrument relevance was assessed by the first-stage F statistic (Stock & Yogo 2005), the reduced form is reported, and under "
        "monotonicity the estimate is a local average treatment effect for compliers (Imbens & Angrist 1994).",
        "قُدّر الأثر السببي للمتغير الداخلي بالمربعات الصغرى على مرحلتين بأخطاء متينة، وقُيّمت قوة الأداة بإحصاءة F للمرحلة الأولى "
        "(Stock & Yogo 2005)، وتُعرض المعادلة المختزلة، وتحت فرض الرتابة يمثل التقدير متوسط الأثر المحلي للممتثلين (Imbens & Angrist 1994).",
        ("ia94", "sy05", "ap09")),
    "did": (
        "Because treatment was adopted at different dates, effects were estimated with heterogeneity-robust difference-in-differences: the "
        "group-time average treatment effects ATT(g,t) of Callaway & Sant'Anna (2021) under unconditional parallel trends with analytical "
        "influence-function standard errors, aggregated to event-time, group and calendar summaries, and the interaction-weighted estimator of "
        "Sun & Abraham (2021). The conventional two-way fixed-effects event study and the Goodman-Bacon (2021) decomposition are reported to "
        "show the bias from 'forbidden' comparisons; pre-treatment estimates test parallel trends (Roth et al. 2023).",
        "لأن المعالجة طُبّقت في تواريخ مختلفة، قُدّرت الآثار بطرق فرق الفروق المتينة لعدم تجانس الأثر: آثار المجموعة-الزمن ATT(g,t) لكالاواي "
        "وسانت آنا (2021) تحت فرض الاتجاهات المتوازية غير الشرطي بأخطاء معيارية من دالة التأثير، مجمّعةً حسب زمن الحدث والمجموعة والزمن "
        "التقويمي، ومقدّر سن وأبراهام (2021) الموزون بالتفاعلات. وتُعرض دراسة الحدث التقليدية بالآثار الثابتة الثنائية وتفكيك جودمان-بيكون "
        "(2021) لبيان التحيز الناتج عن المقارنات «الممنوعة»، وتختبر تقديرات ما قبل المعالجة فرض الاتجاهات المتوازية (Roth et al. 2023).",
        ("cs21", "sa21", "gb21", "rambachan")),
    "did22": (
        "The canonical 2×2 difference-in-differences was estimated by OLS with the treated-group × post interaction and cluster-robust "
        "standard errors (Angrist & Pischke 2009).",
        "قُدّر نموذج فرق الفروق الأساسي (2×2) بالمربعات الصغرى مع تفاعل المجموعة المعالجة × فترة ما بعد وأخطاء معيارية عنقودية "
        "(Angrist & Pischke 2009).",
        ("ap09",)),
    "synth": (
        "The counterfactual for the treated unit was built by the synthetic control method (Abadie, Diamond & Hainmueller 2010): non-negative "
        "donor weights summing to one minimise the distance between the treated unit's and the synthetic unit's pre-treatment predictors, "
        "with predictor weights V chosen to minimise the pre-treatment outcome MSPE. Inference uses in-space placebos and the ratio of post- to "
        "pre-treatment RMSPE (Abadie 2021). Synthetic difference-in-differences (Arkhangelsky et al. 2021), which combines unit and time "
        "weights with two-way fixed effects, is reported as a robustness check with a placebo standard error.",
        "بُني السيناريو المضاد للوحدة المعالجة بطريقة الضابط التركيبي (Abadie et al. 2010): أوزان غير سالبة مجموعها واحد للوحدات المانحة "
        "تُصغّر المسافة بين محددات الوحدة المعالجة والوحدة التركيبية قبل المعالجة، مع اختيار أوزان المحددات V لتصغير متوسط مربع خطأ التنبؤ "
        "بالنتيجة قبل المعالجة. ويعتمد الاستدلال على اختبارات الوهم المكانية ونسبة RMSPE بعد المعالجة إلى ما قبلها (Abadie 2021). ويُعرض فرق "
        "الفروق التركيبي (Arkhangelsky et al. 2021) الذي يجمع أوزان الوحدات والزمن مع الآثار الثابتة الثنائية كاختبار متانة بخطأ معياري وهمي.",
        ("adh10", "abadie21", "aahiw21")),
    "rdd": (
        "The discontinuity at the cutoff was estimated by local-polynomial regression with a triangular kernel and MSE-optimal bandwidth, "
        "reporting conventional, bias-corrected and robust confidence intervals (Calonico, Cattaneo & Titiunik 2014); for a fuzzy design the "
        "jump in the outcome is divided by the jump in treatment take-up. Validity checks include the manipulation test of Cattaneo, Jansson & "
        "Ma (2020), covariate continuity, placebo cutoffs and bandwidth sensitivity (Lee & Lemieux 2010).",
        "قُدّر الانقطاع عند نقطة القطع بانحدار متعدد الحدود المحلي بنواة مثلثية وعرض نطاق أمثل وفق متوسط مربع الخطأ، مع فترات ثقة تقليدية "
        "ومصححة التحيز ومتينة (Calonico et al. 2014)، وفي التصميم الغامض يُقسم القفز في النتيجة على القفز في نسبة تلقي المعالجة. وتشمل "
        "فحوص الصلاحية اختبار التلاعب (Cattaneo et al. 2020) واستمرارية المتغيرات المصاحبة ونقاط القطع الوهمية وحساسية عرض النطاق "
        "(Lee & Lemieux 2010).",
        ("cct14", "cjm20", "lee10")),
    "matching": (
        "Under selection on observables, the propensity score was estimated by logit (Rosenbaum & Rubin 1983). Effects were estimated by "
        "nearest-neighbour matching on the score with replacement and Abadie-Imbens (2006) standard errors, by inverse-probability weighting "
        "with normalised weights (Hirano, Imbens & Ridder 2003) and by augmented IPW, which is doubly robust (Robins, Rotnitzky & Zhao 1994); "
        "standard errors for the weighting estimators come from the stacked M-estimation sandwich. Balance was checked by standardised mean "
        "differences (|SMD| < 0.1) and overlap by the distribution of the score (Austin 2011).",
        "تحت فرض الانتقاء على المتغيرات المشاهدة قُدّر درجة الميل بنموذج لوجيت (Rosenbaum & Rubin 1983)، وقُدّرت الآثار بالمطابقة مع أقرب جار "
        "على الدرجة مع الإحلال وأخطاء أباديه-إمبنز (2006)، وبالترجيح بمقلوب الاحتمال بأوزان معيارية (Hirano et al. 2003)، وبالترجيح المعزز "
        "المزدوج المتانة (Robins et al. 1994)، مع أخطاء معيارية لمقدّرات الترجيح من ساندويتش تقدير-M المكدّس. وفُحص التوازن بالفروق المعيارية "
        "للمتوسطات (|SMD| < 0.1) والتداخل بتوزيع درجة الميل (Austin 2011).",
        ("rr83", "ai06", "hir03", "rrz94", "austin11")),
    "degree_days": (
        "Daily minimum and maximum temperatures were converted into growing degree days between 8 and 29 °C and killing (extreme) degree "
        "days above 29 °C with the single-sine method (Baskerville & Emin 1969), following Schlenker & Roberts (2009); exposure in temperature "
        "bins, frost and hot days, rainfall totals and dry spells were also computed.",
        "حُوّلت درجات الحرارة اليومية الصغرى والعظمى إلى درجات أيام النمو بين 8 و29 °م ودرجات الأيام الضارة فوق 29 °م بطريقة الجيب المفرد "
        "(Baskerville & Emin 1969) وفق Schlenker & Roberts (2009)، كما حُسب التعرض لفئات الحرارة وأيام الصقيع والأيام الحارة ومجموع الأمطار "
        "وفترات الجفاف.",
        ("be69", "sr09")),
    "temp_response": (
        "The weather-yield relationship was estimated with a panel fixed-effects model exploiting year-to-year weather variation within "
        "locations (Deschênes & Greenstone 2007; Dell, Jones & Olken 2014): log yield on growing and killing degree days (piecewise-linear "
        "temperature response; Schlenker & Roberts 2009), or on days in temperature bins or a quadratic in season temperature, with a quadratic "
        "in precipitation, location and year fixed effects and standard errors clustered by location. Warming scenarios translate a uniform "
        "rise in temperature into degree-day changes and yield impacts with delta-method intervals.",
        "قُدّرت العلاقة بين الطقس والإنتاجية بنموذج بانل بآثار ثابتة يستغل التغيرات السنوية في الطقس داخل المواقع (Deschênes & Greenstone 2007؛ "
        "Dell et al. 2014): لوغاريتم الإنتاجية على درجات أيام النمو والضرر (استجابة خطية مجزأة؛ Schlenker & Roberts 2009) أو على أيام فئات "
        "الحرارة أو على دالة تربيعية في حرارة الموسم، مع دالة تربيعية في الأمطار وآثار ثابتة للموقع والسنة وأخطاء معيارية عنقودية حسب الموقع. "
        "وتحوّل سيناريوهات الاحترار ارتفاعاً منتظماً في الحرارة إلى تغيرات في درجات الأيام وآثار على الإنتاجية بفترات ثقة بطريقة دلتا.",
        ("sr09", "dg07", "dell14")),
    "ricardian": (
        "Following the Ricardian approach (Mendelsohn, Nordhaus & Shaw 1994), land values were regressed on linear and quadratic climate "
        "normals and soil and market controls with robust standard errors; marginal climate effects are evaluated at the sample means and the "
        "welfare effect of a climate scenario is the average change in predicted land value, which nets out farmers' adaptation.",
        "وفق المنهج الريكاردي (Mendelsohn et al. 1994) انحُدرت قيم الأراضي على المتوسطات المناخية وتربيعاتها مع ضوابط التربة والسوق وأخطاء "
        "متينة، وقُيّمت الآثار الحدية للمناخ عند متوسطات العينة، والأثر الرفاهي لسيناريو مناخي هو متوسط التغير في القيمة المتنبأ بها للأرض، "
        "وهو يتضمن تكيّف المزارعين.",
        ("mns94",)),
    "bhm": (
        "The macroeconomic damage function follows Burke, Hsiang & Miguel (2015): annual growth of GDP per capita on a quadratic in "
        "temperature (and precipitation) with country and year fixed effects and country-specific trends, clustered by country. The "
        "temperature optimum −β1/2β2 is reported with a delta-method interval, and GDP paths are projected with and without a linear warming "
        "scenario. An illustrative social cost of carbon combines quadratic damages (Nordhaus 2017) with the transient climate response to "
        "cumulative emissions (Matthews et al. 2009).",
        "تتبع دالة الضرر الكلية Burke et al. (2015): نمو نصيب الفرد من الناتج على دالة تربيعية في الحرارة (والأمطار) بآثار ثابتة للدولة والسنة "
        "واتجاهات خاصة بكل دولة وأخطاء عنقودية حسب الدولة، وتُعرض الحرارة المثلى −β1/2β2 بفترة دلتا، وتُسقط مسارات الناتج بسيناريو احترار خطي "
        "وبدونه. وتجمع تكلفة الكربون الاجتماعية التوضيحية أضراراً تربيعية (Nordhaus 2017) مع استجابة المناخ للانبعاثات التراكمية (Matthews et al. 2009).",
        ("bhm15", "nordhaus17", "matthews09")),
    "stirpat": (
        "Drivers of emissions were analysed with the STIRPAT model (Dietz & Rosa 1997; York, Rosa & Dietz 2003), a log-linear stochastic form "
        "of I = PAT whose coefficients are elasticities, estimated with country and year fixed effects and clustered standard errors; ridge "
        "regression with a GCV-chosen penalty (Hoerl & Kennard 1970) addresses collinearity among the drivers (VIFs reported). Elasticities "
        "drive emission projections under growth scenarios.",
        "حُلّلت محددات الانبعاثات بنموذج STIRPAT (Dietz & Rosa 1997؛ York et al. 2003)، وهو صيغة عشوائية لوغاريتمية للمتطابقة I = PAT "
        "معاملاتها مرونات، قُدّر بآثار ثابتة للدولة والسنة وأخطاء عنقودية، ويعالج انحدار الحافة بمعامل جزاء مختار بالتحقق المتقاطع المعمم "
        "(Hoerl & Kennard 1970) الارتباط بين المحددات (مع عرض VIF)، وتُستخدم المرونات في إسقاط الانبعاثات وفق سيناريوهات النمو.",
        ("dr97", "yrd03", "hk70")),
    "ekc": (
        "The environmental Kuznets curve (Grossman & Krueger 1995) was estimated as log emissions per capita on log income and its square "
        "with country and year fixed effects and clustered standard errors. The turning point exp(−β1/2β2) is reported with delta-method and "
        "Fieller intervals together with the Lind & Mehlum (2010) end-point slopes; the curve's robustness is discussed in Stern (2004).",
        "قُدّر منحنى كوزنتس البيئي (Grossman & Krueger 1995) بانحدار لوغاريتم الانبعاثات للفرد على لوغاريتم الدخل ومربعه بآثار ثابتة للدولة "
        "والسنة وأخطاء عنقودية، وتُعرض نقطة التحول exp(−β1/2β2) بفترتي دلتا وفيلر مع ميلي الطرفين لليند وميلوم (2010)، وتُناقش متانة المنحنى "
        "وفق Stern (2004).",
        ("gk95", "lm10", "stern04")),
    "tapio": (
        "Decoupling between emissions and economic growth was classified with the Tapio (2005) elasticity e = %ΔCO2 / %ΔGDP into eight "
        "states using the 0.8 and 1.2 thresholds, and emission changes were decomposed with the additive and multiplicative LMDI-I method "
        "(Ang 2005) applied to the Kaya identity CO2 = P · (GDP/P) · (E/GDP) · (CO2/E), which leaves no residual.",
        "صُنّف فك الارتباط بين الانبعاثات والنمو الاقتصادي بمرونة تابيو (2005) e = %ΔCO2 / %ΔGDP إلى ثماني حالات باستخدام العتبتين 0.8 و1.2، "
        "وفُكّكت تغيرات الانبعاثات بطريقة LMDI-I الجمعية والضربية (Ang 2005) المطبقة على متطابقة كايا CO2 = P · (GDP/P) · (E/GDP) · (CO2/E) "
        "دون أي باقٍ.",
        ("tapio05", "ang05", "kaya")),
}


def methodology(kind: str, lang: str = "en") -> tuple:
    en, ar, keys = TEXT[kind]
    return (ar if lang == "ar" else en), [REFS[k] for k in keys]


def code(kind: str, file_name: str = "your_data.csv", **kw) -> dict:
    py = ["import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf", f"df = pd.read_csv('{file_name}')"]
    rr = [f"df <- read.csv('{file_name}')"]
    st = [f'import delimited "{file_name}", clear varnames(1) case(preserve)']
    y, x = kw.get("y", "y"), kw.get("x", ["x1", "x2"])
    fx = " + ".join(x)
    unit, time = kw.get("unit", "id"), kw.get("time", "year")
    if kind in ("quantile", "mmqr", "cq"):
        if kind == "quantile":
            py += [f"for q in (0.1, 0.25, 0.5, 0.75, 0.9):", f"    print(q, smf.quantreg('{y} ~ {fx}', df).fit(q=q).params.round(4).to_dict())"]
            rr += ["library(quantreg)", f"m <- rq({y} ~ {fx}, tau = c(0.1, 0.25, 0.5, 0.75, 0.9), data = df)", "summary(m, se = 'nid')",
                   "anova(m, joint = TRUE)   # equality of slopes", "plot(summary(m))"]
            st += [f"sqreg {y} {' '.join(x)}, quantiles(.1 .25 .5 .75 .9) reps(200)", "test [q25]" + x[0] + " = [q75]" + x[0]]
        elif kind == "mmqr":
            rr += ["# install.packages('quantreg'); MMQR: Stata xtqreg (Machado & Santos Silva) or Aetheris"]
            st += ["* ssc install xtqreg", f"xtset {unit} {time}", f"xtqreg {y} {' '.join(x)}, i({unit}) quantile(.1 .25 .5 .75 .9) ls"]
            py += ["# MMQR: see analytics.quantile.mmqr in Aetheris"]
        else:
            rr += ["# install.packages('quantilogram')  # cross-quantilogram (Han et al. 2016)"]
            py += ["# Cross-quantilogram: analytics.quantile.cross_quantilogram in Aetheris"]
            st += ["* user-written: crossquantilogram (Stata Journal)"]
    elif kind in ("spatial_auto", "spatial_reg", "gwr", "spatial_panel"):
        lon, lat = kw.get("lon", "longitude"), kw.get("lat", "latitude")
        k = kw.get("k", 6)
        rr += ["library(spdep); library(spatialreg)", f"nb <- knn2nb(knearneigh(cbind(df${lon}, df${lat}), k = {k})); lw <- nb2listw(nb, style = 'W')"]
        py += ["import libpysal, esda, spreg", f"w = libpysal.weights.KNN.from_array(df[['{lon}', '{lat}']].values, k={k}); w.transform = 'r'"]
        st += ["gen id_ = _n", "spset id_", f"spset, modify coord({lon} {lat})",
               f"spmatrix create idistance W, normalize(row)"]
        if kind == "spatial_auto":
            rr += [f"moran.test(df${y}, lw); moran.mc(df${y}, lw, nsim = 999); geary.test(df${y}, lw)", f"lisa <- localmoran(df${y}, lw)"]
            py += [f"mi = esda.Moran(df['{y}'], w); print(mi.I, mi.p_norm, mi.p_sim)", f"lisa = esda.Moran_Local(df['{y}'], w)"]
            st += [f"regress {y}", "estat moran, errorlag(W)"]
        elif kind == "spatial_reg":
            rr += [f"f <- {y} ~ {fx}", "lm.RStests(lm(f, data = df), lw, test = 'all')",
                   "sar <- lagsarlm(f, data = df, listw = lw); summary(sar); impacts(sar, listw = lw, R = 1000)",
                   "sem <- errorsarlm(f, data = df, listw = lw); sdm <- lagsarlm(f, data = df, listw = lw, Durbin = TRUE)", "AIC(sar, sem, sdm)"]
            py += [f"y = df[['{y}']].values; X = df[{x!r}].values",
                   "m = spreg.ML_Lag(y, X, w=w, name_y='" + y + "', name_x=" + repr(x) + "); print(m.summary)"]
            st += [f"spregress {y} {' '.join(x)}, ml dvarlag(W)", "estat impact", f"spregress {y} {' '.join(x)}, ml errorlag(W)",
                   f"spregress {y} {' '.join(x)}, ml dvarlag(W) ivarlag(W: {' '.join(x)})"]
        elif kind == "gwr":
            py += ["from mgwr.gwr import GWR; from mgwr.sel_bw import Sel_BW", f"xy = df[['{lon}', '{lat}']].values; yy = df[['{y}']].values; XX = df[{x!r}].values",
                   "bw = Sel_BW(xy, yy, XX).search(criterion='AICc'); print(GWR(xy, yy, XX, bw).fit().summary())"]
            rr += ["library(GWmodel)  # or spgwr", f"# bw <- bw.gwr({y} ~ {fx}, data = sp_df, adaptive = TRUE, kernel = 'bisquare')"]
            st += ["* ssc install gwr  (user-written)"]
        else:
            rr += ["# install.packages('splm')", "library(splm)",
                   f"m <- spml({y} ~ {fx}, data = df, index = c('{unit}', '{time}'), listw = lw, model = 'within', lag = TRUE, spatial.error = 'none'); summary(m)"]
            st += [f"xtset {unit} {time}", f"spxtregress {y} {' '.join(x)}, fe dvarlag(W)", "estat impact"]
            py += ["# spreg.Panel_FE_Lag(y, X, w) — PySAL spreg panel FE (y, X stacked by time)"]
    elif kind in ("rct", "iv"):
        tr_ = kw.get("treat", "treat")
        if kind == "rct":
            py += [f"m = smf.ols('{y} ~ {tr_}', df).fit(cov_type='HC2'); print(m.summary())"]
            rr += ["library(estimatr)", f"difference_in_means({y} ~ {tr_}, data = df)", f"lm_lin({y} ~ {tr_}, covariates = ~ {fx}, data = df)"]
            st += [f"regress {y} {tr_}, vce(hc2)", f"* Lin: center covariates and interact", f"teffects ra ({y} {' '.join(x)}) ({tr_})"]
        else:
            z = kw.get("z", ["z"])
            py += ["from linearmodels.iv import IV2SLS", f"m = IV2SLS.from_formula('{y} ~ 1 + [{tr_} ~ {' + '.join(z)}]', df).fit(cov_type='robust'); print(m.summary); print(m.first_stage)"]
            rr += ["library(ivreg); library(sandwich)", f"m <- ivreg({y} ~ {tr_} | {' + '.join(z)}, data = df); summary(m, vcov = sandwich, diagnostics = TRUE)"]
            st += [f"ivregress 2sls {y} ({tr_} = {' '.join(z)}), vce(robust)", "estat firststage", "estat endogenous"]
    elif kind in ("did", "did22"):
        g = kw.get("first_treated", "first_treated")
        rr += ["library(did); library(fixest)",
               f"a <- att_gt('{y}', '{time}', '{unit}', '{g}', data = df, control_group = 'nevertreated')  # {unit} must be numeric",
               "aggte(a, type = 'dynamic'); aggte(a, type = 'simple'); ggdid(aggte(a, type = 'dynamic'))",
               f"df$g2 <- ifelse(df${g} == 0, 10000, df${g}); sa <- feols({y} ~ sunab(g2, {time}) | {unit} + {time}, df, cluster = ~{unit}); iplot(sa); summary(sa, agg = 'ATT')",
               "# install.packages('bacondecomp'); bacondecomp::bacon(...)"]
        st += [f"* ssc install csdid drdid", f"csdid {y}, ivar({unit}) time({time}) gvar({g}) notyet", "estat event", "estat simple",
               f"* ssc install eventstudyinteract", f"* jwdid {y}, ivar({unit}) tvar({time}) gvar({g})"]
        py += ["# pip install differences  — or Aetheris analytics.causal.callaway_santanna / sun_abraham"]
    elif kind == "synth":
        tu_, tt = kw.get("treated", "Treated unit"), kw.get("treat_time", 2010)
        rr += ["library(Synth)  # dataprep() + synth(); placebo loop over donors", "# install.packages('synthdid'); synthdid::synthdid_estimate(Y, N0, T0)"]
        st += ["* ssc install synth", f"tsset {unit} {time}", f"synth {y} {' '.join(x)} {y}({tt - 1}), trunit(<id of {tu_}>) trperiod({tt}) fig",
               "* ssc install sdid", f"sdid {y} {unit} {time} treated, vce(placebo) reps(200)"]
        py += ["# pip install pysyncon  — or Aetheris analytics.causal.synth_control / synth_did"]
    elif kind == "rdd":
        run, c = kw.get("running", "x"), kw.get("cutoff", 0)
        py += ["from rdrobust import rdrobust, rdplot; from rddensity import rddensity", f"print(rdrobust(df['{y}'], df['{run}'], c={c}))",
               f"print(rddensity(X=df['{run}'].values, c={c}).test)"]
        rr += ["library(rdrobust); library(rddensity)", f"summary(rdrobust(df${y}, df${run}, c = {c}))", f"rdplot(df${y}, df${run}, c = {c})",
               f"summary(rddensity(df${run}, c = {c}))"]
        st += ["* net install rdrobust, from(https://raw.githubusercontent.com/rdpackages/rdrobust/master/stata) replace", f"rdrobust {y} {run}, c({c})",
               f"rdplot {y} {run}, c({c})", f"rddensity {run}, c({c})"]
    elif kind == "matching":
        tr_ = kw.get("treat", "treat")
        rr += ["library(MatchIt); library(WeightIt); library(cobalt)", f"f <- {tr_} ~ {fx}",
               "m <- matchit(f, data = df, method = 'nearest', replace = TRUE); bal.tab(m)",
               f"w <- weightit(f, data = df, method = 'glm', estimand = 'ATT'); summary(lm_weightit({y} ~ {tr_}, data = df, weightit = w))"]
        st += [f"teffects psmatch ({y}) ({tr_} {' '.join(x)}), atet", f"teffects ipw ({y}) ({tr_} {' '.join(x)})",
               f"teffects aipw ({y} {' '.join(x)}) ({tr_} {' '.join(x)})", "tebalance summarize", "teoverlap"]
        py += [f"ps = smf.logit('{tr_} ~ {fx}', df).fit().predict()", f"w = np.where(df['{tr_}'] == 1, 1/ps, 1/(1 - ps))",
               f"print(smf.wls('{y} ~ {tr_}', df, weights=w).fit(cov_type='HC1').params)"]
    elif kind in ("degree_days", "temp_response", "ricardian"):
        if kind == "degree_days":
            py += ["def sine_dd(tmin, tmax, base):", "    M, W = (tmax + tmin) / 2, (tmax - tmin) / 2",
                   "    th = np.arcsin(np.clip((base - M) / np.where(W > 0, W, 1), -1, 1))",
                   "    part = ((M - base) * (np.pi / 2 - th) + W * np.cos(th)) / np.pi",
                   "    return np.where(tmin >= base, M - base, np.where(tmax > base, part, 0))",
                   "df['gdd'] = sine_dd(df.tmin, df.tmax, 8) - sine_dd(df.tmin, df.tmax, 29); df['kdd'] = sine_dd(df.tmin, df.tmax, 29)",
                   "print(df.groupby(['station', 'season'])[['gdd', 'kdd']].sum())"]
            rr += ["# install.packages('pollen')  # gdd(); or code the single-sine formula as in the Python tab"]
            st += ["* compute degree days with the single-sine formula (gen / replace) and collapse (sum) gdd kdd, by(station season)"]
        elif kind == "temp_response":
            rr += ["library(fixest)", f"m <- feols({y} ~ {fx} + precip_mm + I(precip_mm^2) | {unit} + {time}, df, cluster = ~{unit}); etable(m)"]
            st += [f"reghdfe {y} {' '.join(x)} c.precip_mm##c.precip_mm, absorb({unit} {time}) vce(cluster {unit})"]
            py += ["from linearmodels.panel import PanelOLS", f"d = df.set_index(['{unit}', '{time}'])",
                   f"print(PanelOLS.from_formula('{y} ~ {fx} + EntityEffects + TimeEffects', d).fit(cov_type='clustered', cluster_entity=True))"]
        else:
            rr += [f"m <- lm(log({y}) ~ " + " + ".join(f"{v} + I({v}^2)" for v in x) + ", data = df); lmtest::coeftest(m, sandwich::vcovHC(m, 'HC1'))"]
            st += [f"gen ln{y} = ln({y})", f"regress ln{y} " + " ".join(f"c.{v}##c.{v}" for v in x) + ", vce(robust)", "margins, dydx(*)"]
            py += [f"m = smf.ols('np.log({y}) ~ " + " + ".join(f"{v} + I({v}**2)" for v in x) + "', df).fit(cov_type='HC1'); print(m.summary())"]
    elif kind in ("bhm", "stirpat", "ekc", "tapio"):
        if kind == "bhm":
            rr += ["library(fixest)", f"df$tt <- df${time} - min(df${time})",
                   f"m <- feols({y} ~ temperature_c + I(temperature_c^2) + i({unit}, tt) | {unit} + {time}, df, cluster = ~{unit})",
                   "b <- coef(m); -b[1]/(2*b[2])  # optimum temperature"]
            st += [f"reghdfe {y} c.temperature_c##c.temperature_c, absorb({unit} {time} i.{unit}#c.{time}) vce(cluster {unit})",
                   "nlcom -_b[temperature_c]/(2*_b[c.temperature_c#c.temperature_c])"]
        elif kind == "stirpat":
            rr += ["library(fixest)", f"m <- feols(log({y}) ~ " + " + ".join(f"log({v})" for v in x) + f" | {unit} + {time}, df, cluster = ~{unit}); etable(m)",
                   "# ridge: glmnet::cv.glmnet(X, y, alpha = 0)"]
            st += [f"gen ln{y} = ln({y})"] + [f"gen ln{v} = ln({v})" for v in x] + [f"reghdfe ln{y} {' '.join('ln' + v for v in x)}, absorb({unit} {time}) vce(cluster {unit})"]
        elif kind == "ekc":
            rr += ["library(fixest)", f"df$lpc <- log(df${y} / df$population); df$ly <- log(df${x[0]})",
                   f"m <- feols(lpc ~ ly + I(ly^2) | {unit} + {time}, df, cluster = ~{unit}); b <- coef(m); exp(-b[1]/(2*b[2]))  # turning point",
                   "# install.packages('utest')-style Lind-Mehlum test: see Aetheris"]
            st += [f"gen lpc = ln({y}/population)", f"gen ly = ln({x[0]})", f"reghdfe lpc c.ly##c.ly, absorb({unit} {time}) vce(cluster {unit})",
                   "nlcom exp(-_b[ly]/(2*_b[c.ly#c.ly]))", "* ssc install utest ; utest ly c.ly#c.ly"]
        else:
            py += ["# Tapio elasticity e = %ΔCO2 / %ΔGDP; LMDI weights L(a, b) = (a − b) / (ln a − ln b)",
                   "L = lambda a, b: (a - b) / (np.log(a) - np.log(b))"]
            rr += ["# LMDI: install.packages('LMDI') or code L(a,b) = (a-b)/(log(a)-log(b)) as in Aetheris"]
            st += ["* LMDI: code the logarithmic mean weights with gen / egen"]
    return {"Python": "\n".join(py) + "\n", "R": "\n".join(rr) + "\n", "Stata": "\n".join(st) + "\n"}
