# ==========================================
# Aetheris Research - Reporting for univariate time series (§1.7)
# Interpretation, methodology paragraphs, references and Python / R / Stata code.
# ==========================================

from __future__ import annotations

import numpy as np
import pandas as pd

from analytics import ts


def _p(p):
    return "p < 0.001" if p is not None and p < 0.001 else (f"p = {p:.3f}" if p is not None and np.isfinite(p) else "")


def interpret_unit_roots(io: dict, name: str, lang: str = "en") -> str:
    ar = lang == "ar"
    d = io["order"]
    tab = io["table"]
    parts = []
    lab_ar = {"level": "المستوى", "Δ (first difference)": "الفرق الأول Δ", "Δ^2": "الفرق الثاني Δ²"}
    for ser, sub in tab.groupby("Series", sort=False):
        ser = lab_ar.get(ser, ser) if ar else ser
        rej_ur = sub[sub["H0"] == "unit root"]["Reject H0 (5%)"]
        k = sub[sub["H0"] == "stationary"]
        txt = (f"<b>{ser}</b>: " + (f"رفضت {int(rej_ur.sum())} من {len(rej_ur)} اختبارات فرض جذر الوحدة" if ar else
                                    f"{int(rej_ur.sum())} of {len(rej_ur)} unit-root tests reject the unit root"))
        if len(k):
            kr = bool(k["Reject H0 (5%)"].iloc[0])
            txt += ("، و" + ("رفض KPSS فرض السكون" if kr else "لم يرفض KPSS فرض السكون")) if ar else \
                ("; KPSS " + ("rejects" if kr else "does not reject") + " stationarity")
        parts.append(txt + ".")
    if d is None:
        parts.append("لم تحسم الاختبارات درجة التكامل حتى الفرق الثاني؛ راجع وجود انكسار هيكلي (زيفوت-أندروز)." if ar else
                     "The tests do not settle the order of integration up to the second difference; check for structural breaks (Zivot-Andrews).")
    else:
        lab = {0: ("مستقرة في المستوى I(0)", "stationary in levels — I(0)"), 1: ("متكاملة من الدرجة الأولى I(1)", "integrated of order one — I(1)"),
               2: ("متكاملة من الدرجة الثانية I(2)", "integrated of order two — I(2)")}[d]
        parts.append((f"الخلاصة: السلسلة {name} <b>{lab[0]}</b>." if ar else f"Conclusion: {name} is <b>{lab[1]}</b>."))
        if d == 1:
            parts.append("ويعني ذلك أن الصدمات أثرها دائم، وأن الانحدار على المستوى قد يكون زائفاً ما لم يوجد تكامل مشترك "
                         "(انظر الإصدار القادم: ARDL وجوهانسن)." if ar else
                         "Shocks have permanent effects, and regressions in levels may be spurious unless the variables are "
                         "cointegrated (ARDL / Johansen arrive in the next release).")
    if io.get("verdict_last") == "conflicting":
        parts.append("⚠️ " + ("نتائج متعارضة بين الاختبارات (قد تشير إلى جذر قريب من الوحدة أو انكسار)." if ar else
                              "Conflicting test results (a near-unit root or a structural break is possible)."))
    return "<br>".join(parts)


def interpret_arima(r: ts.ArimaResult, diag: dict, lang: str = "en") -> str:
    ar = lang == "ar"
    P = []
    P.append((f"قُدِّر النموذج <b>{r.name}</b> بالإمكان الأعظم الدقيق على {r.stats['N']} مشاهدة (AIC = {r.stats['AIC']:.2f}، "
              f"BIC = {r.stats['BIC']:.2f})." if ar else
              f"<b>{r.name}</b> was estimated by exact maximum likelihood on {r.stats['N']} observations "
              f"(AIC = {r.stats['AIC']:.2f}, BIC = {r.stats['BIC']:.2f})."))
    sig = r.table[(r.table["p-value"] < 0.05) & (r.table["Parameter"] != "sigma2")]
    if not sig.empty:
        P.append(("المعلمات المعنوية: " if ar else "Significant parameters: ") +
                 ", ".join(f"{row['Parameter']} = {row['Coef.']:.4f}" for _, row in sig.iterrows()) + ".")
    ok = diag["lb_p"] >= 0.05
    P.append((f"اختبار ليونج-بوكس للبواقي (Q({diag['lb_lags']}) بدرجات حرية {diag['lb_df']}، {_p(diag['lb_p'])}) "
              + ("لا يكشف ارتباطاً ذاتياً متبقياً: البواقي ضوضاء بيضاء والنموذج كافٍ." if ok else
                 "يكشف <b>ارتباطاً ذاتياً متبقياً</b>: أضف حدوداً AR/MA أو موسمية.")) if ar else
             (f"Ljung-Box on the residuals (Q({diag['lb_lags']}), {diag['lb_df']} df, {_p(diag['lb_p'])}) "
              + ("finds no remaining autocorrelation: the residuals are white noise and the model is adequate." if ok else
                 "finds <b>remaining autocorrelation</b>: add AR/MA or seasonal terms.")))
    P.append((f"الطبيعية (جارك-بيرا {_p(diag['jb_p'])})، وأثر ARCH ({_p(diag['arch_p'])})"
              + ("؛ يوجد تقلب شرطي يستدعي نماذج GARCH (الإصدار القادم)." if diag["arch_p"] < 0.05 else ".")) if ar else
             (f"Normality (Jarque-Bera {_p(diag['jb_p'])}); ARCH effects ({_p(diag['arch_p'])})"
              + (" — conditional heteroskedasticity is present: consider GARCH (next release)." if diag["arch_p"] < 0.05 else ".")))
    try:
        arr = np.asarray(r.res.arroots)
        near = arr.size and np.min(np.abs(arr)) < 1.02
    except Exception:  # noqa: BLE001
        near = False
    if near:
        P.append("⚠️ " + ("أحد جذور AR قريب جداً من دائرة الوحدة: قد تحتاج السلسلة إلى فرق إضافي (راجع اختبارات جذر الوحدة)." if ar else
                          "An AR root lies very close to the unit circle: the series may need one more difference (see the unit-root tests)."))
    for n in r.notes:
        P.append("⚠️ " + n)
    return "<br>".join(P)


def methodology(kind: str, lang: str = "en", **kw) -> tuple[str, list]:
    ar = lang == "ar"
    R = ts.REFS
    if kind == "unitroot":
        txt = ("لتحديد درجة تكامل السلسلة طُبِّقت حزمة من الاختبارات: ديكي-فولر الموسع (ADF) واختبار فيليبس-بيرون (PP) واختبار "
               "DF-GLS لإليوت وروثنبرج وستوك، وفرض العدم فيها وجود جذر الوحدة، مع اختبار KPSS الذي فرضه العدم السكون، "
               "واختيرت فترات الإبطاء بمعيار أكايكي (ADF) والمعيار المعدل لنج وبيرون (DF-GLS) وقاعدة نيوي-ويست (PP وKPSS). "
               "وطُبِّقت الاختبارات على المستوى ثم على الفروق المتتالية حتى تحقق السكون. وأُكملت باختبارات نج-بيرون المعدلة "
               "واختبار زيفوت-أندروز الذي يسمح بانكسار هيكلي داخلي.") if ar else (
               "The order of integration is determined with a battery of tests: augmented Dickey-Fuller (ADF), Phillips-Perron "
               "(PP) and the Elliott-Rothenberg-Stock DF-GLS test (null: unit root), complemented by the KPSS test (null: "
               "stationarity). Lag lengths follow the AIC (ADF), the modified AIC of Ng and Perron (DF-GLS) and the Newey-West "
               "rule (PP, KPSS). Tests are applied to the level and to successive differences until stationarity is reached, "
               "and are supplemented by the Ng-Perron modified tests and the Zivot-Andrews test with an endogenous break.")
        refs = [R[k] for k in ("df", "said", "mackinnon10", "pp", "kpss", "ers", "ngperron", "za")]
    elif kind == "arima":
        name = kw.get("name", "ARIMA")
        txt = (f"اتُّبعت منهجية بوكس-جنكنز: التعرف على النموذج من دالتي الارتباط الذاتي والجزئي بعد تحقيق السكون، ثم المفاضلة "
               f"بين النماذج المرشحة بمعايير AIC وAICc وBIC، وتقدير النموذج المختار {name} بالإمكان الأعظم الدقيق (مرشح كالمان)، "
               "ثم التحقق من كفاءته باختبار ليونج-بوكس للبواقي وجارك-بيرا للطبيعية وARCH-LM لثبات التباين وجذور كثيرات "
               "الحدود (السكون وقابلية الانعكاس)، وأخيراً التنبؤ مع فترات الثقة وتقييم الدقة خارج العينة (RMSE وMAE وMAPE "
               "وإحصاءة ثايل U واختبار ديبولد-ماريانو).") if ar else (
               f"The Box-Jenkins methodology is followed: identification from the ACF and PACF after achieving stationarity, "
               f"selection among candidate models by AIC, AICc and BIC, estimation of the selected {name} by exact maximum "
               "likelihood (Kalman filter), diagnostic checking (Ljung-Box on residuals, Jarque-Bera, ARCH-LM, inverse "
               "characteristic roots for stationarity and invertibility) and forecasting with confidence intervals, evaluated "
               "out of sample (RMSE, MAE, MAPE, Theil's U and the Diebold-Mariano test).")
        refs = [R[k] for k in ("boxjenkins", "hamilton", "ljungbox", "engle", "dm", "hln", "hyndman")]
    elif kind == "filters":
        txt = ("فُصل الاتجاه العام عن الدورة بمرشح هودريك-بريسكوت (معامل التمهيد λ حسب تكرار البيانات) ومرشحي التمرير "
               "النطاقي لباكستر-كينج وكريستيانو-فيتزجيرالد، وفُكِّكت السلسلة إلى اتجاه وموسمية وبواقٍ بطريقة STL.") if ar else (
               "Trend and cycle are separated with the Hodrick-Prescott filter (smoothing λ by data frequency) and the "
               "Baxter-King and Christiano-Fitzgerald band-pass filters; the series is decomposed into trend, seasonal and "
               "remainder components with STL.")
        refs = [R[k] for k in ("hp", "ravn", "bk", "cf", "stl")]
    elif kind == "seasonal":
        txt = ("اختُبرت جذور الوحدة الموسمية باختبار HEGY للبيانات الربع سنوية عند التكرار الصفري ونصف السنوي والسنوي، "
               "مع قيم حرجة بمحاكاة مونت كارلو لحجم العينة الفعلي.") if ar else (
               "Seasonal unit roots are tested with the HEGY procedure for quarterly data at the zero, semi-annual and annual "
               "frequencies, with Monte Carlo critical values for the actual sample size.")
        refs = [R["hegy"]]
    else:
        txt = ("قُدِّرت معلمة الذاكرة الطويلة d بمقدّر جيويكي وبورتر-هوداك للانحدار على اللوغاريتم الدوري، وبمقدّر ويتل المحلي "
               "لروبنسون، وقُدِّر نموذج ARFIMA بإمكان ويتل في مجال التردد؛ وتدل 0 < d < 0.5 على ذاكرة طويلة مستقرة.") if ar else (
               "The long-memory parameter d is estimated with the Geweke-Porter-Hudak log-periodogram regression and Robinson's "
               "local Whittle estimator, and an ARFIMA model by the frequency-domain Whittle likelihood; 0 < d < 0.5 indicates "
               "stationary long memory.")
        refs = [R[k] for k in ("granger", "hosking", "gph", "robinson", "fox")]
    return txt, sorted(refs)


# ==========================================================
# Code export
# ==========================================================
def code(kind: str, value: str, time: str, file_name: str = "your_data.csv", s: int = 12, log: bool = False,
         order=(1, 1, 1), seasonal=(0, 0, 0), regression: str = "c") -> dict:
    yexpr_py = f'np.log(df["{value}"])' if log else f'df["{value}"]'
    yexpr_r = f'log(df[["{value}"]])' if log else f'df[["{value}"]]'
    sv = value.replace(" ", "_")
    yst = f"ln_{sv}" if log else sv
    py = ["import numpy as np, pandas as pd", f'df = pd.read_csv("{file_name}", parse_dates=["{time}"]).sort_values("{time}")',
          f"y = {yexpr_py}.to_numpy()"]
    rr = [f'df <- read.csv("{file_name}")', f"y <- ts({yexpr_r}, frequency = {s})"]
    st = [f'import delimited "{file_name}", clear varnames(1) case(preserve)',
          f"gen t = _n", "tsset t"] + ([f"gen {yst} = ln({sv})"] if log else [])
    if kind == "unitroot":
        rs = {"c": "drift", "ct": "trend", "n": "none"}[regression]
        py += ["from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews",
               f'print(adfuller(y, regression="{regression}", autolag="AIC"))',
               f'print(kpss(y, regression="{regression if regression != "n" else "c"}", nlags=int(4 * (len(y) / 100) ** 0.25)))',
               'print(zivot_andrews(y, regression="c", autolag="AIC"))',
               "# Phillips-Perron and DF-GLS need the 'arch' package (pip install arch):",
               "# from arch.unitroot import PhillipsPerron, DFGLS",
               f'# print(PhillipsPerron(y, trend="{regression}").summary())',
               f'# print(DFGLS(y, trend="{"ct" if regression == "ct" else "c"}", method="aic").summary())']
        rr = ["library(urca)"] + rr + [f'summary(ur.df(y, type = "{rs}", selectlags = "AIC", lags = 12))',
                                       f'summary(ur.pp(y, type = "Z-tau", model = "{"trend" if regression == "ct" else "constant"}", lags = "short"))',
                                       f'summary(ur.ers(y, type = "DF-GLS", model = "{"trend" if regression == "ct" else "constant"}", lag.max = 4))',
                                       f'summary(ur.kpss(y, type = "{"tau" if regression == "ct" else "mu"}", lags = "short"))',
                                       'summary(ur.za(y, model = "intercept", lag = 4))',
                                       "summary(ur.df(diff(y), type = \"drift\", selectlags = \"AIC\", lags = 12))   # first difference"]
        trend_opt = " trend" if regression == "ct" else (" noconstant" if regression == "n" else "")
        st += [f"dfuller {yst},{trend_opt} lags(4) regress", f"pperron {yst},{trend_opt}",
               f"dfgls {yst}, maxlag(12)" + ("" if regression == "ct" else " notrend"),
               "* ssc install kpss ; ssc install zandrews",
               f"kpss {yst}" + ("" if regression == "ct" else ", notrend"), f"zandrews {yst}, break(intercept)",
               f"dfuller D.{yst}, lags(4) regress"]
    elif kind == "arima":
        p, d, q = order
        P, D, Q = seasonal
        seas_py = f", seasonal_order=({P}, {D}, {Q}, {s})" if any(seasonal) and s > 1 else ""
        py += ["from statsmodels.tsa.arima.model import ARIMA", "from statsmodels.stats.diagnostic import acorr_ljungbox",
               "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf",
               f"res = ARIMA(y, order=({p}, {d}, {q}){seas_py}).fit(cov_type='approx')", "print(res.summary())",
               f"print(acorr_ljungbox(res.resid, lags=[{max(10, 2 * s)}], model_df={p + q + P + Q}))",
               "print(res.get_forecast(12).summary_frame())"]
        seas_r = f", seasonal = c({P}, {D}, {Q})" if any(seasonal) and s > 1 else ""
        rr = ["library(forecast)"] + rr + [f'm <- Arima(y, order = c({p}, {d}, {q}){seas_r}, method = "ML")', "summary(m)",
                                           "checkresiduals(m)", "autoplot(forecast(m, h = 12))",
                                           "auto.arima(y, stepwise = FALSE, approximation = FALSE)   # automatic search"]
        seas_s = f" sarima({P},{D},{Q},{s})" if any(seasonal) and s > 1 else ""
        nocons = " noconstant" if d + D > 0 else ""
        st += [f"ac {yst}" if d == 0 else f"ac D.{yst}", f"arima {yst}, arima({p},{d},{q}){seas_s}{nocons} vce(oim)", "estat ic", "estat aroots",
               "predict e, residuals", "wntestq e", "tsappend, add(12)", "predict f, y dynamic(.)"]
    elif kind == "filters":
        lam = ts.HP_LAMBDA.get(s, 1600)
        lo, hi, K = ts.BK_PARAMS.get(s, (6, 32, 12))
        py += ["from statsmodels.tsa.filters.hp_filter import hpfilter", "from statsmodels.tsa.filters.bk_filter import bkfilter",
               "from statsmodels.tsa.filters.cf_filter import cffilter", "from statsmodels.tsa.seasonal import STL",
               f"cycle, trend = hpfilter(y, lamb={lam})", f"bk = bkfilter(y, {lo}, {hi}, {K})", f"cf_cycle, cf_trend = cffilter(y, {lo}, {hi}, drift=True)",
               f"stl = STL(y, period={s}, robust=True).fit()" if s > 1 else ""]
        rr = ["library(mFilter)   # install.packages('mFilter')"] + rr + [f"hpfilter(y, freq = {lam}, type = 'lambda')",
                                                                        f"bkfilter(y, pl = {lo}, pu = {hi}, nfix = {K})", f"cffilter(y, pl = {lo}, pu = {hi}, drift = TRUE)",
                                                                        "plot(stl(y, s.window = 'periodic', robust = TRUE))" if s > 1 else ""]
        st += [f"tsfilter hp hp_cycle = {yst}, smooth({lam})", f"tsfilter bk bk_cycle = {yst}, minperiod({lo}) maxperiod({hi}) smaorder({K})",
               f"tsfilter cf cf_cycle = {yst}, minperiod({lo}) maxperiod({hi}) drift"]
    elif kind == "seasonal":
        py += ["# HEGY: no maintained Python implementation — use R uroot::hegy.test or Stata hegy4"]
        rr = ["library(uroot)"] + [f'df <- read.csv("{file_name}")', f"y <- ts({yexpr_r}, frequency = 4)",
                                   "hegy.test(y, deterministic = c(1, 0, 1), lag.method = 'fixed', maxlag = 0)"]
        st += ["* ssc install hegy4", "gen q = quarter(dofq(t))", f"hegy4 {yst}, det(seas) lags(0)"]
    else:
        py += ["# Long memory: GPH / local Whittle (Aetheris) — R fracdiff, Stata arfima"]
        rr = ["library(fracdiff)"] + rr + ["fdGPH(y, bandw.exp = 0.5)", "summary(fracdiff(y - mean(y), nar = 0, nma = 0))"]
        st += [f"arfima {yst}", "estat acplot"]
    return {"Python": "\n".join(x for x in py if x) + "\n", "R": "\n".join(x for x in rr if x) + "\n",
            "Stata": "\n".join(x for x in st if x) + "\n"}
