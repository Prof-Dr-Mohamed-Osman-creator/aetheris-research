# ==========================================
# Aetheris Research - Applied economics reports (v1.10, §2.7–2.8): methodology, references, code
# ==========================================

from __future__ import annotations

import numpy as np

REFS = {
    "cd": "Cobb, C. W. & Douglas, P. H. (1928). A theory of production. American Economic Review, 18(1), 139–165.",
    "translog": "Christensen, L. R., Jorgenson, D. W. & Lau, L. J. (1973). Transcendental logarithmic production frontiers. Review of Economics and Statistics, 55(1), 28–45.",
    "ces": "Arrow, K. J., Chenery, H. B., Minhas, B. S. & Solow, R. M. (1961). Capital-labor substitution and economic efficiency. Review of Economics and Statistics, 43(3), 225–250.",
    "kmenta": "Kmenta, J. (1967). On estimation of the CES production function. International Economic Review, 8(2), 180–189.",
    "diewert": "Diewert, W. E. (1971). An application of the Shephard duality theorem: a generalized Leontief production function. Journal of Political Economy, 79(3), 481–507.",
    "halter": "Halter, A. N., Carter, H. O. & Hocking, J. G. (1957). A note on the transcendental production function. Journal of Farm Economics, 39(4), 966–974.",
    "bm": "Blackorby, C. & Russell, R. R. (1989). Will the real elasticity of substitution please stand up? American Economic Review, 79(4), 882–888.",
    "heady": "Heady, E. O. & Dillon, J. L. (1961). Agricultural Production Functions. Iowa State University Press.",
    "cerrato": "Cerrato, M. E. & Blackmer, A. M. (1990). Comparison of models for describing corn yield response to nitrogen fertilizer. Agronomy Journal, 82(1), 138–143.",
    "baumol": "Baumol, W. J., Panzar, J. C. & Willig, R. D. (1982). Contestable Markets and the Theory of Industry Structure. Harcourt.",
    "shephard": "Shephard, R. W. (1953). Cost and Production Functions. Princeton University Press.",
    "zellner": "Zellner, A. (1962). An efficient method of estimating seemingly unrelated regressions. Journal of the American Statistical Association, 57(298), 348–368.",
    "berndt": "Berndt, E. R. & Wood, D. O. (1975). Technology, prices, and the derived demand for energy. Review of Economics and Statistics, 57(3), 259–268.",
    "als": "Aigner, D., Lovell, C. A. K. & Schmidt, P. (1977). Formulation and estimation of stochastic frontier production function models. Journal of Econometrics, 6(1), 21–37.",
    "mvb": "Meeusen, W. & van den Broeck, J. (1977). Efficiency estimation from Cobb-Douglas production functions with composed error. International Economic Review, 18(2), 435–444.",
    "jlms": "Jondrow, J., Lovell, C. A. K., Materov, I. S. & Schmidt, P. (1982). On the estimation of technical inefficiency in the stochastic frontier production function model. Journal of Econometrics, 19(2–3), 233–238.",
    "bc88": "Battese, G. E. & Coelli, T. J. (1988). Prediction of firm-level technical efficiencies with a generalized frontier production function and panel data. Journal of Econometrics, 38(3), 387–399.",
    "bc92": "Battese, G. E. & Coelli, T. J. (1992). Frontier production functions, technical efficiency and panel data: with application to paddy farmers in India. Journal of Productivity Analysis, 3(1–2), 153–169.",
    "bc95": "Battese, G. E. & Coelli, T. J. (1995). A model for technical inefficiency effects in a stochastic frontier production function for panel data. Empirical Economics, 20(2), 325–332.",
    "kl": "Kumbhakar, S. C. & Lovell, C. A. K. (2000). Stochastic Frontier Analysis. Cambridge University Press.",
    "bro": "Battese, G. E., Rao, D. S. P. & O'Donnell, C. J. (2004). A metafrontier production function for estimation of technical efficiencies and technology gaps. Journal of Productivity Analysis, 21(1), 91–103.",
    "ccr": "Charnes, A., Cooper, W. W. & Rhodes, E. (1978). Measuring the efficiency of decision making units. European Journal of Operational Research, 2(6), 429–444.",
    "bcc": "Banker, R. D., Charnes, A. & Cooper, W. W. (1984). Some models for estimating technical and scale inefficiencies in DEA. Management Science, 30(9), 1078–1092.",
    "fgnz": "Färe, R., Grosskopf, S., Norris, M. & Zhang, Z. (1994). Productivity growth, technical progress, and efficiency change in industrialized countries. American Economic Review, 84(1), 66–83.",
    "op": "Olley, G. S. & Pakes, A. (1996). The dynamics of productivity in the telecommunications equipment industry. Econometrica, 64(6), 1263–1297.",
    "lp": "Levinsohn, J. & Petrin, A. (2003). Estimating production functions using inputs to control for unobservables. Review of Economic Studies, 70(2), 317–341.",
    "acf": "Ackerberg, D. A., Caves, K. & Frazer, G. (2015). Identification properties of recent production function estimators. Econometrica, 83(6), 2411–2451.",
    "diewert76": "Diewert, W. E. (1976). Exact and superlative index numbers. Journal of Econometrics, 4(2), 115–145.",
    "coelli": "Coelli, T. J., Rao, D. S. P., O'Donnell, C. J. & Battese, G. E. (2005). An Introduction to Efficiency and Productivity Analysis (2nd ed.). Springer.",
    "nerlove": "Nerlove, M. (1958). The Dynamics of Supply: Estimation of Farmers' Response to Price. Johns Hopkins Press.",
    "askari": "Askari, H. & Cummings, J. T. (1977). Estimating agricultural supply response with the Nerlove model: a survey. International Economic Review, 18(2), 257–292.",
    "ezekiel": "Ezekiel, M. (1938). The cobweb theorem. Quarterly Journal of Economics, 52(2), 255–280.",
    "working": "Working, H. (1943). Statistical laws of family expenditure. Journal of the American Statistical Association, 38(221), 43–56.",
    "leser": "Leser, C. E. V. (1963). Forms of Engel functions. Econometrica, 31(4), 694–703.",
    "stone": "Stone, R. (1954). Linear expenditure systems and demand analysis: an application to the pattern of British demand. Economic Journal, 64(255), 511–527.",
    "dm80": "Deaton, A. & Muellbauer, J. (1980). An almost ideal demand system. American Economic Review, 70(3), 312–326.",
    "bbl": "Banks, J., Blundell, R. & Lewbel, A. (1997). Quadratic Engel curves and consumer demand. Review of Economics and Statistics, 79(4), 527–539.",
    "br99": "Blundell, R. & Robin, J. M. (1999). Estimation in large and disaggregated demand systems: an estimator for conditionally linear systems. Journal of Applied Econometrics, 14(3), 209–232.",
    "ga": "Green, R. & Alston, J. M. (1990). Elasticities in AIDS models. American Journal of Agricultural Economics, 72(2), 442–445.",
    "theil": "Theil, H. (1965). The information approach to demand analysis. Econometrica, 33(1), 67–87.",
    "barten": "Barten, A. P. (1964). Consumer demand functions under conditions of almost additive preferences. Econometrica, 32(1–2), 1–38.",
    "friedman": "Friedman, M. (1957). A Theory of the Consumption Function. Princeton University Press.",
    "hall": "Hall, R. E. (1978). Stochastic implications of the life cycle–permanent income hypothesis. Journal of Political Economy, 86(6), 971–987.",
    "cm": "Campbell, J. Y. & Mankiw, N. G. (1989). Consumption, income, and interest rates. NBER Macroeconomics Annual, 4, 185–216.",
    "fl": "Friedman, J. & Levinsohn, J. (2002). The distributional impacts of Indonesia's financial crisis on household welfare. World Bank Economic Review, 16(3), 397–423.",
    "hicks": "Hicks, J. R. (1942). Consumers' surplus and index-numbers. Review of Economic Studies, 9(2), 126–137.",
}


def fmt_p(p):
    return "" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


TEXT = {
    "production": (
        "Production functions were estimated in several functional forms (Cobb-Douglas, translog with inputs centred at their means, CES "
        "by non-linear least squares, generalized Leontief, quadratic, transcendental). Output elasticities, returns to scale, marginal and "
        "average products, Allen-Uzawa and Morishima elasticities of substitution (from the bordered Hessian of the fitted function), the "
        "three stages of production and isoquants were derived; the translog was tested against the Cobb-Douglas with an F test.",
        "قُدِّرت دوال الإنتاج بعدة صور دالية (كوب-دوجلاس، والمتسامية اللوغاريتمية بمدخلات ممركزة حول متوسطاتها، وCES بالمربعات الصغرى "
        "غير الخطية، وليونتيف المعممة، والتربيعية، والترانسندنتالية)، واشتُقت منها المرونات الإنتاجية والعائد للسعة والناتج الحدي والمتوسط "
        "ومرونات الإحلال لألن-أوزاوا وموريشيما (من مصفوفة هس المحدودة للدالة المقدرة) ومراحل الإنتاج الثلاث ومنحنيات الناتج المتساوي، "
        "واختُبرت المتسامية اللوغاريتمية مقابل كوب-دوجلاس باختبار F.",
        ("cd", "translog", "ces", "kmenta", "diewert", "halter", "bm", "heady")),
    "yield": (
        "Yield-response functions (quadratic, quadratic-plateau, linear-plateau, Mitscherlich-Spillman, square root) were fitted by least "
        "squares and compared by AIC; the economic optimum rate equates the marginal physical product with the input/output price ratio, "
        "with a delta-method standard error, and is compared with the agronomic maximum.",
        "قُدِّرت دوال استجابة المحصول (التربيعية، والتربيعية-الهضبة، والخطية-الهضبة، وميتشرليخ-سبيلمان، والجذر التربيعي) بالمربعات الصغرى "
        "وقورنت بمعيار AIC، وحُسب المعدل الاقتصادي الأمثل الذي يساوي فيه الناتج الحدي نسبة سعر المدخل إلى سعر الناتج مع خطئه المعياري "
        "بطريقة دلتا، وقورن بالحد الأقصى الفني.",
        ("heady", "cerrato")),
    "cost": (
        "Cost structure was analysed with a cubic short-run cost function (average, marginal and variable costs, shutdown and break-even "
        "points), a long-run log-quadratic cost function (cost elasticity, minimum efficient scale), a quadratic multi-product cost function "
        "(economies of scope) and translog / Cobb-Douglas cost systems estimated jointly with the cost-share equations by iterated SUR under "
        "symmetry and linear homogeneity, giving Allen and Morishima substitution elasticities and input-demand price elasticities.",
        "حُللت بنية التكاليف بدالة تكاليف تكعيبية قصيرة الأجل (التكاليف المتوسطة والحدية والمتغيرة ونقطتا الإغلاق والتعادل)، ودالة تكاليف "
        "طويلة الأجل لوغاريتمية تربيعية (مرونة التكاليف والحجم الأدنى الكفء)، ودالة تكاليف تربيعية متعددة المنتجات (وفورات النطاق)، "
        "ونظامي تكاليف ترانسلوج وكوب-دوجلاس مع معادلات حصص التكاليف بطريقة SUR المكررة مع فرض التماثل والتجانس الخطي، لتُستخرج مرونات "
        "الإحلال لألن وموريشيما والمرونات السعرية للطلب على المدخلات.",
        ("shephard", "zellner", "berndt", "baumol", "translog")),
    "efficiency": (
        "Technical efficiency was measured with stochastic frontier models estimated by maximum likelihood (half-normal, truncated-normal "
        "and exponential inefficiency; Battese-Coelli 1995 inefficiency determinants; Battese-Coelli 1992 time-varying panel model), with "
        "firm-level efficiency E[exp(−u)|ε]; with a meta-frontier enveloping group frontiers (technology-gap ratios); and non-parametrically "
        "by data envelopment analysis (CCR, BCC, scale efficiency) and the Malmquist productivity index decomposed into efficiency and "
        "technical change.",
        "قيست الكفاءة الفنية بنماذج الحدود العشوائية المقدرة بالإمكان الأعظم (عدم كفاءة بتوزيع نصف طبيعي وطبيعي مبتور وأسي؛ ومحددات عدم "
        "الكفاءة لباتيس-كويلي 1995؛ ونموذج البانل المتغير زمنياً لباتيس-كويلي 1992) مع كفاءة كل وحدة E[exp(−u)|ε]، وبالحد الفوقي الذي "
        "يغلّف حدود المجموعات (نسب الفجوة التقنية)، ولامعلمياً بتحليل مغلف البيانات (CCR وBCC وكفاءة الحجم) ومؤشر مالمكويست للإنتاجية "
        "المجزأ إلى تغير الكفاءة والتغير التقني.",
        ("als", "mvb", "jlms", "bc88", "bc92", "bc95", "kl", "bro", "ccr", "bcc", "fgnz", "coelli")),
    "tfp": (
        "Production-function parameters and total factor productivity were estimated with control-function methods that correct the "
        "simultaneity between input choices and unobserved productivity: Olley-Pakes (investment proxy), Levinsohn-Petrin (intermediate-input "
        "proxy) and Ackerberg-Caves-Frazer (GMM second stage), with bootstrap standard errors; aggregate productivity growth was measured with "
        "superlative Törnqvist and Fisher indices.",
        "قُدِّرت معلمات دالة الإنتاج والإنتاجية الكلية للعوامل بطرق الدالة الضابطة التي تعالج التزامن بين اختيار المدخلات والإنتاجية غير "
        "المشاهدة: أولي-باكس (الاستثمار متغيراً بديلاً)، وليفينسون-بترين (المدخلات الوسيطة)، وأكربرغ-كيفز-فريزر (المرحلة الثانية بطريقة "
        "العزوم المعممة) مع أخطاء معيارية بالبوتستراب، وقيس نمو الإنتاجية الكلي بالأرقام القياسية الفائقة لتورنكفيست وفيشر.",
        ("op", "lp", "acf", "diewert76", "coelli")),
    "supply": (
        "Supply response was estimated with the Nerlovian partial-adjustment / adaptive-expectations model (short- and long-run elasticities, "
        "adjustment coefficient, mean lag) and its combined form; price-quantity dynamics were analysed with the cobweb model and its "
        "stability condition.",
        "قُدِّرت استجابة العرض بنموذج نيرلوف للتعديل الجزئي والتوقعات المتكيفة (المرونات قصيرة وطويلة الأجل، ومعامل التعديل، ومتوسط "
        "الإبطاء) وصيغته المركبة، وحُللت ديناميكية الأسعار والكميات بنموذج بيت العنكبوت وشرط استقراره.",
        ("nerlove", "askari", "ezekiel")),
    "demand": (
        "Single-equation demand functions (constant-elasticity and linear, static and partial-adjustment) gave own-price, cross-price and "
        "income elasticities with long-run values by the delta method; Engel curves (Working-Leser and alternatives) classified goods as "
        "necessities, luxuries or inferior.",
        "أعطت دوال الطلب المفردة (ثابتة المرونة والخطية، الساكنة وذات التعديل الجزئي) المرونات السعرية الذاتية والتقاطعية والدخلية مع "
        "قيمها طويلة الأجل بطريقة دلتا، وصنّفت منحنيات إنجل (ووركنج-ليسر وبدائلها) السلع إلى ضرورية أو كمالية أو دنيا.",
        ("working", "leser", "nerlove")),
    "system": (
        "Complete demand systems were estimated: the linear expenditure system; the almost ideal demand system in its linear approximate "
        "(Stone index), non-linear and quadratic (QUAIDS) forms by iterated linear least squares within restricted iterated SUR (adding-up, "
        "homogeneity and symmetry imposed); and the Rotterdam model. Expenditure, Marshallian and Hicksian elasticities were computed and the "
        "welfare effect of price changes measured by exact compensating and equivalent variations from the expenditure function.",
        "قُدِّرت أنظمة الطلب الكاملة: نظام الإنفاق الخطي، ونظام الطلب شبه المثالي بصيغته الخطية التقريبية (مؤشر ستون) وغير الخطية والتربيعية "
        "(QUAIDS) بالمربعات الصغرى الخطية المكررة ضمن SUR مكرر مقيد (مع فرض الجمع والتجانس والتماثل)، ونموذج روتردام، وحُسبت المرونات "
        "الإنفاقية ومرونات مارشال وهيكس، وقيس الأثر على الرفاه لتغير الأسعار بالتغير التعويضي والمكافئ الدقيقين من دالة الإنفاق.",
        ("stone", "dm80", "bbl", "br99", "ga", "theil", "barten", "hicks", "fl")),
    "consumption": (
        "Consumption behaviour was analysed with the Keynesian consumption function, the permanent-income (Koyck) form with short- and long-run "
        "marginal propensities, Hall's random-walk test and the Campbell-Mankiw share of rule-of-thumb consumers estimated by instrumental "
        "variables.",
        "حُلل سلوك الاستهلاك بدالة الاستهلاك الكينزية، وصيغة الدخل الدائم (كويك) مع الميل الحدي للاستهلاك في الأجلين القصير والطويل، واختبار "
        "السير العشوائي لهول، ونسبة المستهلكين وفق قاعدة الإبهام لكامبل ومانكيو المقدرة بالمتغيرات الأداتية.",
        ("friedman", "hall", "cm")),
}


def methodology(kind: str, lang: str = "en") -> tuple:
    en, ar, keys = TEXT[kind]
    return (ar if lang == "ar" else en), [REFS[k] for k in keys]


def code(kind: str, file_name: str = "your_data.csv", **kw) -> dict:
    py = ["import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf", f"df = pd.read_csv('{file_name}')"]
    rr = [f"df <- read.csv('{file_name}')"]
    st = [f'import delimited "{file_name}", clear varnames(1) case(preserve)']
    y = kw.get("y", "y")
    x = kw.get("x", ["x1", "x2"])
    lx = " + ".join(f"np.log({v})" for v in x)
    if kind == "production":
        py += [f"cd = smf.ols('np.log({y}) ~ {lx}', df).fit(); print(cd.summary())",
               "print('RTS =', cd.params.iloc[1:].sum()); print(cd.t_test(' + '.join(cd.params.index[1:]) + ' = 1'))"]
        tl = " + ".join(f"I(np.log({v}) - np.log({v}).mean())" for v in x)
        py += [f"# translog: add squares and cross-products of the centred logs", f"tl = smf.ols('np.log({y}) ~ ({tl})**2 + " +
               " + ".join(f"I(0.5*(np.log({v}) - np.log({v}).mean())**2)" for v in x) + "', df).fit()"]
        rr += [f"cd <- lm(log({y}) ~ {' + '.join(f'log({v})' for v in x)}, data = df); summary(cd)",
               "library(car); linearHypothesis(cd, paste(paste(names(coef(cd))[-1], collapse = ' + '), '= 1'))",
               f"# CES (two inputs) by NLS: install.packages('micEconCES'); micEconCES::cesEst('{y}', c({', '.join(repr(v) for v in x[:2])}), df)"]
        st += [f"gen ln{y} = ln({y})"] + [f"gen ln{v} = ln({v})" for v in x] + [f"regress ln{y} {' '.join('ln' + v for v in x)}",
               "test " + " + ".join("ln" + v for v in x) + " = 1"]
    elif kind == "yield":
        xv, yv = kw.get("x", "n"), y
        rr += [f"qp <- nls({yv} ~ ifelse({xv} < -b/(2*c), a + b*{xv} + c*{xv}^2, a - b^2/(4*c)), data = df, start = list(a = 3, b = 0.04, c = -1e-4))",
               "summary(qp); ratio <- 35/12000; eor <- (ratio - coef(qp)['b'])/(2*coef(qp)['c']); eor"]
        py += ["from scipy.optimize import curve_fit", "def qp(x, a, b, c): x0 = -b/(2*c); xx = np.minimum(x, x0); return a + b*xx + c*xx**2",
               f"p, V = curve_fit(qp, df['{xv}'], df['{yv}'], p0=[3, 0.04, -1e-4]); print(p)"]
        st += [f"nl ({yv} = cond({xv} < -{{b}}/(2*{{c}}), {{a}} + {{b}}*{xv} + {{c}}*{xv}^2, {{a}} - {{b}}^2/(4*{{c}}))), initial(a 3 b 0.04 c -0.0001)"]
    elif kind == "cost":
        pr = kw.get("prices", ["w1", "w2", "w3"])
        sh = kw.get("shares", ["s1", "s2", "s3"])
        q = kw.get("q", "q")
        c = kw.get("cost", "c")
        rr += ["library(systemfit)",
               f"df$lc <- log(df${c}/df${pr[-1]}); " + "; ".join(f"df$lw{i} <- log(df${pr[i]}/df${pr[-1]})" for i in range(len(pr) - 1)) + f"; df$lq <- log(df${q})",
               "# translog cost + share equations with symmetry: see Aetheris for the full restriction matrix",
               f"eqs <- list(share1 = {sh[0]} ~ " + " + ".join(f"lw{i}" for i in range(len(pr) - 1)) + " + lq, share2 = " + f"{sh[1]} ~ " + " + ".join(f"lw{i}" for i in range(len(pr) - 1)) + " + lq)",
               "fit <- systemfit(eqs, method = 'SUR', data = df, restrict.matrix = 'share1_lw1 - share2_lw0 = 0', control = systemfit.control(maxiter = 500)); summary(fit)"]
        py += [f"# Cobb-Douglas cost function (homogeneity imposed)", f"m = smf.ols('np.log({c}/{pr[-1]}) ~ np.log({q}) + " +
               " + ".join(f"np.log({p}/{pr[-1]})" for p in pr[:-1]) + "', df).fit(); print(m.summary()); print('RTS =', 1/m.params.iloc[1])"]
        st += [f"gen lc = ln({c}/{pr[-1]})", f"gen lq = ln({q})"] + [f"gen lw{i} = ln({pr[i]}/{pr[-1]})" for i in range(len(pr) - 1)] + \
              ["regress lc lq " + " ".join(f"lw{i}" for i in range(len(pr) - 1)), "* translog system: sureg with constraint() options"]
    elif kind == "efficiency":
        rr += ["# install.packages(c('frontier', 'Benchmarking'))", "library(frontier)",
               f"m <- sfa(log({y}) ~ {' + '.join(f'log({v})' for v in x)}, data = df); summary(m); efficiencies(m)",
               "library(Benchmarking)", f"e <- dea(as.matrix(df[, c({', '.join(repr(v) for v in x)})]), df${y}, RTS = 'crs'); summary(e)"]
        st += [f"gen ln{y} = ln({y})"] + [f"gen ln{v} = ln({v})" for v in x] + [
               f"frontier ln{y} {' '.join('ln' + v for v in x)}, distribution(hnormal)", "predict te, te",
               f"frontier ln{y} {' '.join('ln' + v for v in x)}, distribution(tnormal) cm(education_years extension_visits credit_access)",
               "* panel: xtset id year ; xtfrontier ... , tvd"]
        py += ["from scipy import optimize, stats",
               f"y = np.log(df['{y}'].values); X = np.column_stack([np.ones(len(df))] + [np.log(df[v].values) for v in {x!r}])",
               "def nll(t):  # half-normal production frontier (Aigner-Lovell-Schmidt)",
               "    b, sv, su = t[:-2], np.exp(0.5 * t[-2]), np.exp(0.5 * t[-1]); e = y - X @ b; s = np.hypot(sv, su)",
               "    return -np.sum(np.log(2 / s) + stats.norm.logpdf(e / s) + stats.norm.logcdf(-su / sv * e / s))",
               "b0 = np.linalg.lstsq(X, y, rcond=None)[0]; r = optimize.minimize(nll, np.r_[b0, -3, -2], method='BFGS'); print(r.x, -r.fun)",
               "# DEA: scipy.optimize.linprog on the envelopment form (see Aetheris for CCR / BCC / Malmquist)"]
    elif kind == "tfp":
        rr += ["# install.packages('prodest')", "library(prodest)",
               "acf <- prodestACF(Y = df$ln_value_added, fX = df$ln_labour, sX = df$ln_capital, pX = df$ln_materials, idvar = df$firm_id, timevar = df$year, R = 50); acf"]
        st += ["* ssc install prodest", "xtset firm_id year",
               "prodest ln_value_added, free(ln_labour) state(ln_capital) proxy(ln_materials) method(lp) acf va reps(50)"]
        py += ["# Aetheris implements OP / LP / ACF; Python alternative: 'pyprodest' (unofficial)"]
    elif kind == "supply":
        q, p = kw.get("q", "area"), kw.get("p", "price")
        py += [f"df['lq'] = np.log(df['{q}']); df['lp'] = np.log(df['{p}'])",
               "m = smf.ols('lq ~ lp.shift(1) + lq.shift(1)', df).fit(); b = m.params",
               "print('SR =', b.iloc[1], 'LR =', b.iloc[1] / (1 - b.iloc[2]), 'adjustment =', 1 - b.iloc[2])"]
        rr += [f"m <- lm(log({q}) ~ log(dplyr::lag({p})) + log(dplyr::lag({q})), data = df); b <- coef(m)",
               "c(SR = b[2], LR = b[2]/(1 - b[3])); car::deltaMethod(m, 'b1/(1 - b2)', parameterNames = c('a', 'b1', 'b2'))"]
        st += ["tsset year", f"regress ln{q} L.ln{p} L.ln{q}", f"nlcom _b[L.ln{p}]/(1 - _b[L.ln{q}])"]
    elif kind in ("demand", "system", "consumption"):
        sh = kw.get("shares", ["w1", "w2", "w3"])
        pr = kw.get("prices", ["p1", "p2", "p3"])
        tot = kw.get("total", "x")
        rr += ["# install.packages('micEconAids')", "library(micEconAids)",
               f"la <- aidsEst(c({', '.join(repr(p) for p in pr)}), c({', '.join(repr(s) for s in sh)}), '{tot}', data = df, method = 'LA')",
               "summary(la); aidsElas(coef(la), prices = colMeans(df[, c(" + ", ".join(repr(p) for p in pr) + ")]), shares = colMeans(df[, c(" + ", ".join(repr(s) for s in sh) + ")]))"]
        st += ["* ssc install quaids", f"quaids {' '.join(sh)}, anot(7.8) prices({' '.join(pr)}) expenditure({tot}) nolog", "estat expenditure e_*",
               "estat uncompensated u*, atmeans", "estat compensated c*, atmeans"]
        py += ["# Engel curve (Working-Leser) for each good", f"for s in {sh!r}:",
               f"    m = smf.ols(f'{{s}} ~ np.log({tot})', df).fit(); print(s, 'elasticity =', 1 + m.params.iloc[1] / df[s].mean())"]
    return {"Python": "\n".join(py) + "\n", "R": "\n".join(rr) + "\n", "Stata": "\n".join(st) + "\n"}
