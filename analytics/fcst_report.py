# ==========================================
# Aetheris Research - Forecasting reports: interpretation, methodology, references, code (v1.9, §3)
# ==========================================

from __future__ import annotations

import numpy as np

REFS = {
    "fpp": "Hyndman, R. J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice (3rd ed.). OTexts.",
    "hk": "Hyndman, R. J. & Khandakar, Y. (2008). Automatic time series forecasting: the forecast package for R. Journal of Statistical Software, 27(3), 1–22.",
    "hkos": "Hyndman, R. J., Koehler, A. B., Ord, J. K. & Snyder, R. D. (2008). Forecasting with Exponential Smoothing: The State Space Approach. Springer.",
    "theta": "Assimakopoulos, V. & Nikolopoulos, K. (2000). The theta model: a decomposition approach to forecasting. International Journal of Forecasting, 16(4), 521–530.",
    "hb": "Hyndman, R. J. & Billah, B. (2003). Unmasking the Theta method. International Journal of Forecasting, 19(2), 287–290.",
    "harvey": "Harvey, A. C. (1989). Forecasting, Structural Time Series Models and the Kalman Filter. Cambridge University Press.",
    "dk": "Durbin, J. & Koopman, S. J. (2012). Time Series Analysis by State Space Methods (2nd ed.). Oxford University Press.",
    "bsts": "Scott, S. L. & Varian, H. R. (2014). Predicting the present with Bayesian structural time series. International Journal of Mathematical Modelling and Numerical Optimisation, 5(1–2), 4–23.",
    "bj": "Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). Time Series Analysis: Forecasting and Control (5th ed.). Wiley.",
    "bridge": "Baffigi, A., Golinelli, R. & Parigi, G. (2004). Bridge models to forecast the euro area GDP. International Journal of Forecasting, 20(3), 447–460.",
    "midas": "Ghysels, E., Sinko, A. & Valkanov, R. (2007). MIDAS regressions: further results and new directions. Econometric Reviews, 26(1), 53–90.",
    "umidas": "Foroni, C., Marcellino, M. & Schumacher, C. (2015). Unrestricted mixed data sampling (MIDAS). Journal of the Royal Statistical Society A, 178(1), 57–82.",
    "kz": "Kvedaras, V. & Zemlys, V. (2012). Testing the functional constraints on parameters in regressions with variables of different frequency. Economics Letters, 116(2), 250–254.",
    "bm": "Bańbura, M. & Modugno, M. (2014). Maximum likelihood estimation of factor models on datasets with arbitrary pattern of missing data. Journal of Applied Econometrics, 29(1), 133–160.",
    "grs": "Giannone, D., Reichlin, L. & Small, D. (2008). Nowcasting: the real-time informational content of macroeconomic data. Journal of Monetary Economics, 55(4), 665–676.",
    "litterman": "Litterman, R. B. (1986). Forecasting with Bayesian vector autoregressions — five years of experience. Journal of Business & Economic Statistics, 4(1), 25–38.",
    "glp": "Giannone, D., Lenza, M. & Primiceri, G. E. (2015). Prior selection for vector autoregressions. Review of Economics and Statistics, 97(2), 436–451.",
    "bgr": "Bańbura, M., Giannone, D. & Reichlin, L. (2010). Large Bayesian vector auto regressions. Journal of Applied Econometrics, 25(1), 71–92.",
    "jorda": "Jordà, Ò. (2005). Estimation and inference of impulse responses by local projections. American Economic Review, 95(1), 161–182.",
    "pmw": "Plagborg-Møller, M. & Wolf, C. K. (2021). Local projections and VARs estimate the same impulse responses. Econometrica, 89(2), 955–980.",
    "bg": "Bates, J. M. & Granger, C. W. J. (1969). The combination of forecasts. Operational Research Quarterly, 20(4), 451–468.",
    "gr": "Granger, C. W. J. & Ramanathan, R. (1984). Improved methods of combining forecasting. Journal of Forecasting, 3(2), 197–204.",
    "timmermann": "Timmermann, A. (2006). Forecast combinations. In Handbook of Economic Forecasting, Vol. 1, 135–196. Elsevier.",
    "corsi": "Corsi, F. (2009). A simple approximate long-memory model of realized volatility. Journal of Financial Econometrics, 7(2), 174–196.",
    "patton": "Patton, A. J. (2011). Volatility forecast comparison using imperfect volatility proxies. Journal of Econometrics, 160(1), 246–256.",
    "kupiec": "Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. Journal of Derivatives, 3(2), 73–84.",
    "christoffersen": "Christoffersen, P. F. (1998). Evaluating interval forecasts. International Economic Review, 39(4), 841–862.",
    "bollerslev": "Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics, 31(3), 307–327.",
    "breiman": "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32.",
    "friedman": "Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. Annals of Statistics, 29(5), 1189–1232.",
    "lgbm": "Ke, G. et al. (2017). LightGBM: a highly efficient gradient boosting decision tree. Advances in Neural Information Processing Systems, 30.",
    "svr": "Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A. & Vapnik, V. (1997). Support vector regression machines. Advances in Neural Information Processing Systems, 9.",
    "conformal": "Stankevičiūtė, K., Alaa, A. M. & van der Schaar, M. (2021). Conformal time-series forecasting. Advances in Neural Information Processing Systems, 34.",
    "prophet": "Taylor, S. J. & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37–45.",
    "dm": "Diebold, F. X. & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.",
    "hln": "Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. International Journal of Forecasting, 13(2), 281–291.",
    "hk06": "Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures of forecast accuracy. International Journal of Forecasting, 22(4), 679–688.",
    "tashman": "Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: an analysis and review. International Journal of Forecasting, 16(4), 437–450.",
    "m4": "Makridakis, S., Spiliotis, E. & Assimakopoulos, V. (2020). The M4 Competition: 100,000 time series and 61 forecasting methods. International Journal of Forecasting, 36(1), 54–74.",
    "growth": "Meade, N. & Islam, T. (2006). Modelling and forecasting the diffusion of innovation — a 25-year review. International Journal of Forecasting, 22(3), 519–545.",
    "mz": "Mincer, J. A. & Zarnowitz, V. (1969). The evaluation of economic forecasts. In Economic Forecasts and Expectations, 3–46. NBER.",
}


def fmt_p(p):
    return "" if p is None or not np.isfinite(p) else ("< 0.001" if p < 0.001 else f"{p:.3f}")


def methodology(kind: str, lang: str = "en", **kw) -> tuple:
    ar = lang == "ar"
    R = REFS
    if kind == "race":
        txt = ("قورنت دقة مجموعة من نماذج التنبؤ (المرجعية الساذجة والموسمية والسير العشوائي بانجراف، والتمهيد الأسي ETS "
               "المختار آلياً بمعيار AICc، وARIMA المختار آلياً، ونموذج ثيتا، والنموذج الهيكلي في الفضاء الحالتي، ونماذج "
               "التعلم الآلي، ونموذج على نمط Prophet، وتجميع أفضل ثلاثة نماذج) بالتقييم التاريخي بالأصل المتدحرج: أعيد تقدير كل "
               f"نموذج عند {kw.get('origins', 8)} أصول تنبؤ متتالية وقورنت تنبؤاته للأفق {kw.get('h', 12)} بالقيم الفعلية. ورُتّبت النماذج "
               "حسب متوسط الخطأ المطلق المقيَّس (MASE)، واختُبر الفرق بين كل نموذج والأفضل باختبار ديبولد-ماريانو مع تصحيح "
               "هارفي-ليبورن-نيوبولد.") if ar else (
               "The predictive accuracy of a set of forecasting methods — naive, seasonal naive and drift benchmarks, ETS selected "
               "by AICc, automatic ARIMA, the theta method, a structural state-space model, machine-learning models, a Prophet-style "
               "model and a combination of the best three — was compared by rolling-origin evaluation: each model was re-estimated "
               f"at {kw.get('origins', 8)} successive forecast origins and its forecasts up to horizon {kw.get('h', 12)} were compared with "
               "the outcomes. Models are ranked by the mean absolute scaled error (MASE) and each is compared with the best by the "
               "Diebold-Mariano test with the Harvey-Leybourne-Newbold correction.")
        refs = [R[k] for k in ("fpp", "tashman", "hk06", "dm", "hln", "m4", "timmermann")]
    elif kind == "classical":
        txt = ("استُخدمت النماذج الإحصائية الكلاسيكية للتنبؤ: النماذج المرجعية، وإطار التمهيد الأسي في الفضاء الحالتي (ETS) "
               "المقدَّر بالإمكان الأعظم مع اختيار النموذج بمعيار AICc، ونموذج ثيتا بصيغة هيندمان وبلّاه، ونماذج ARIMA/SARIMA "
               "ومعها المتغيرات الخارجية (SARIMAX) المقدرة بالإمكان الأعظم الدقيق، والنماذج الهيكلية بمرشح كالمان، والسلاسل "
               "الزمنية الهيكلية البايزية بمعاينة جيبس، ومنحنيات النمو. وفترات التنبؤ مبنية على تباين خطأ التنبؤ النظري "
               "أو على المحاكاة.") if ar else (
               "Classical statistical forecasting models were used: benchmark methods, the state-space exponential-smoothing framework "
               "(ETS) estimated by maximum likelihood with model selection by AICc, the theta method in the Hyndman-Billah form, "
               "ARIMA / SARIMA and regression with ARIMA errors (SARIMAX) by exact maximum likelihood, structural time-series models "
               "by the Kalman filter, Bayesian structural time series by Gibbs sampling and growth curves. Prediction intervals are "
               "based on the theoretical forecast-error variance or on simulation.")
        refs = [R[k] for k in ("fpp", "hkos", "hk", "hb", "bj", "harvey", "dk", "bsts", "growth")]
    elif kind == "nowcast":
        txt = ("لتقدير القيمة الحالية للمتغير ربع السنوي قبل نشره (التنبؤ الآني) استُخدمت ثلاث مقاربات للبيانات مختلطة "
               "التكرار: معادلات الجسر التي تربط المتغير ربع السنوي بمتوسطات المؤشرات الشهرية بعد استكمال الأشهر الناقصة "
               "بنماذج AR، وانحدار MIDAS بأوزان إبطاء ألمون الأسية المعيارية أو بيتا أو غير المقيدة، ونموذج العوامل الديناميكية "
               "مختلط التكرار المقدَّر بالإمكان الأعظم مع خوارزمية EM ومرشح كالمان. وقُيّمت معادلة الجسر خارج العينة مقارنة "
               "بنموذج AR(1).") if ar else (
               "To estimate the current value of the quarterly variable before its release (nowcasting) three mixed-frequency "
               "approaches were used: bridge equations linking the quarterly variable to quarterly averages of monthly indicators, "
               "with missing months filled by AR forecasts; MIDAS regressions with normalised exponential Almon, beta or unrestricted "
               "lag weights; and a mixed-frequency dynamic factor model estimated by maximum likelihood with the EM algorithm and "
               "the Kalman filter. The bridge equation was evaluated out of sample against an AR(1) benchmark.")
        refs = [R[k] for k in ("bridge", "midas", "umidas", "kz", "grs", "bm")]
    elif kind == "macro":
        txt = ("قُدِّر نموذج VAR بايزي بالتوزيع القبلي لمينيسوتا في صيغته المترافقة (طبيعي-معكوس ويشارت)، واختير معامل "
               "الإحكام λ بتعظيم الإمكان الهامشي (بايز التجريبي، جيانوني ولينزا وبريميتشيري 2015)، وحُسبت التنبؤات بالمحاكاة "
               "من التوزيع البعدي الدقيق. وقُدّرت دوال الاستجابة بالإسقاطات المحلية لجوردا بتعريف تسلسلي (تحليل تشولسكي) "
               "وبأخطاء معيارية لنيوي-ويست، كما حُسبت تنبؤات مباشرة متعددة الآفاق.") if ar else (
               "A Bayesian VAR with the conjugate (normal-inverse-Wishart) Minnesota prior was estimated; the tightness λ was set by "
               "maximising the marginal likelihood (empirical Bayes, Giannone, Lenza & Primiceri 2015) and forecasts were simulated "
               "from the exact posterior. Impulse responses were estimated by Jordà local projections with recursive (Cholesky) "
               "identification and Newey-West standard errors, and direct multi-step forecasts were computed.")
        refs = [R[k] for k in ("litterman", "bgr", "glp", "jorda", "pmw")]
    elif kind == "volatility":
        txt = ("قُورنت تنبؤات التباين ليوم واحد من نماذج GARCH(1,1) وGJR-GARCH وEWMA ونموذج HAR-RV لكورسي (بالمستوى "
               "واللوغاريتم) مع التباين المحقق خارج العينة بدالتي الخسارة MSE وQLIKE المتينتين لعدم دقة مقياس التقلب، "
               "وانحدار مينسر-زارنوفيتز، واختبار ديبولد-ماريانو. وقُيِّمت تنبؤات القيمة المعرضة للخطر باختبارات كوبيك "
               "للتغطية غير المشروطة وكريستوفرسن للاستقلال والتغطية المشروطة، ورُتّبت بخسارة المئين.") if ar else (
               "One-day-ahead variance forecasts from GARCH(1,1), GJR-GARCH, EWMA and Corsi's HAR-RV (levels and logs) were compared "
               "with realized variance out of sample using the MSE and QLIKE losses (robust to noise in the volatility proxy), "
               "Mincer-Zarnowitz regressions and Diebold-Mariano tests. Value-at-Risk forecasts were evaluated with Kupiec's "
               "unconditional-coverage and Christoffersen's independence and conditional-coverage tests and ranked by the quantile loss.")
        refs = [R[k] for k in ("bollerslev", "corsi", "patton", "mz", "kupiec", "christoffersen")]
    elif kind == "ml":
        txt = ("طُبّقت نماذج التعلم الآلي (الغابات العشوائية، وتعزيز التدرج، وLightGBM، وانحدار المتجهات الداعمة) على "
               "خصائص مشتقة من السلسلة (الإبطاءات، والإبطاء الموسمي، والمتوسطات المتحركة، والاتجاه، وحدود فورييه الموسمية) "
               "بإستراتيجية تنبؤ تكرارية، وعلى الفروق الأولى عند عدم سكون السلسلة لأن الأشجار لا تستقرئ الاتجاه. وبُنيت فترات "
               "التنبؤ بالطريقة التوافقية (Conformal) من أخطاء التقييم المتدحرج. وطُبّق نموذج على نمط Prophet: اتجاه خطي متقطع "
               "مع نقاط تغيّر بتوزيع قبلي لابلاس وموسمية فورييه.") if ar else (
               "Machine-learning models (random forest, gradient boosting, LightGBM, support vector regression) were trained on "
               "features derived from the series (lags, the seasonal lag, rolling means, trend and Fourier seasonal terms) with a "
               "recursive strategy, on first differences when the series is non-stationary because trees cannot extrapolate a "
               "trend. Prediction intervals are split-conformal, from rolling-origin errors. A Prophet-style model — piecewise-linear "
               "trend with Laplace-prior changepoints and Fourier seasonality — was also fitted.")
        refs = [R[k] for k in ("breiman", "friedman", "lgbm", "svr", "conformal", "prophet")]
    elif kind == "eval":
        txt = ("قيست الدقة بمقاييس MAE وRMSE وMAPE وsMAPE وMASE ومعامل ثايل U، وقُيّمت النماذج بالأصل المتدحرج مع إعادة "
               "التقدير عند كل أصل، واختُبر تساوي الدقة باختبار ديبولد-ماريانو مع تصحيح العينات الصغيرة. ودُمجت التنبؤات "
               "بالمتوسط البسيط والوسيط والمتوسط المشذّب وأوزان مقلوب MSE وبيتس-جرينجر وجرينجر-راماناثان المقيدة وأوزان أكايكي.") if ar else (
               "Accuracy was measured by MAE, RMSE, MAPE, sMAPE, MASE and Theil's U; models were evaluated on a rolling origin with "
               "re-estimation at every origin and equal accuracy was tested with the small-sample-corrected Diebold-Mariano test. "
               "Forecasts were combined by the simple average, median, trimmed mean, inverse-MSE, Bates-Granger, constrained "
               "Granger-Ramanathan and Akaike weights.")
        refs = [R[k] for k in ("hk06", "tashman", "dm", "hln", "bg", "gr", "timmermann")]
    else:  # scenario
        txt = ("بُنيت تنبؤات السيناريوهات بنموذج انحدار بأخطاء ARIMA يربط المتغير المستهدف بمحركاته، ثم تُنبّأ بالمتغير "
               "المستهدف شرطياً على مسارات بديلة للمحركات (سيناريو أساسي بإسقاط الاتجاه، وسيناريوهات متفائلة ومتشائمة يحددها "
               "المستخدم). وفترات التنبؤ مشروطة بمسار المحركات ولا تتضمن عدم اليقين حوله.") if ar else (
               "Scenario forecasts use a regression with ARIMA errors linking the target to its drivers; the target is then forecast "
               "conditionally on alternative paths for the drivers (a baseline trend projection and user-defined optimistic and "
               "pessimistic scenarios). Prediction intervals are conditional on the driver paths and exclude uncertainty about them.")
        refs = [R[k] for k in ("fpp", "bj")]
    return txt, refs


def interpret_race(r: dict, lang: str = "en") -> str:
    ar = lang == "ar"
    tab = r["table"]
    b = tab.iloc[0]
    naive_rows = tab[tab.key.isin(["naive", "snaive"])]
    bench = naive_rows.MASE.min() if len(naive_rows) else np.nan
    gain = (1 - b.MASE / bench) * 100 if np.isfinite(bench) and bench > 0 else np.nan
    dm = r.get("dm")
    nsig = int((dm["p"] < 0.05).sum()) if dm is not None and len(dm) else 0
    if ar:
        s = (f"الأفضل في الاختبار التاريخي: <b>{b.Model}</b> (MASE = {b.MASE:.3f}، RMSE = {num(b.RMSE)}). ")
        if np.isfinite(gain):
            s += f"يتفوق على أفضل نموذج ساذج بنحو {gain:.0f}% في MASE. "
        s += f"ويتفوق معنوياً (ديبولد-ماريانو، 5%) على {nsig} من {len(dm) if dm is not None else 0} نموذجاً. "
        s += "قيمة MASE أقل من 1 تعني أداءً أفضل من التنبؤ الساذج داخل العينة."
    else:
        s = f"Best in the backtest: <b>{b.Model}</b> (MASE = {b.MASE:.3f}, RMSE = {num(b.RMSE)}). "
        if np.isfinite(gain):
            s += f"It improves on the best naive benchmark by about {gain:.0f}% in MASE. "
        s += f"It is significantly more accurate (Diebold-Mariano, 5%) than {nsig} of {len(dm) if dm is not None else 0} models. "
        s += "A MASE below 1 means better than the in-sample naive forecast."
    return s


def num(v) -> str:
    v = float(v)
    return f"{v:,.0f}" if abs(v) >= 1000 else (f"{v:,.2f}" if abs(v) >= 10 else f"{v:.4g}")


def interpret_fc(f, y_last: float, lang: str = "en", unit: str = "", name: str | None = None) -> str:
    ar = lang == "ar"
    h = len(f.mean)
    nm = name or f.name
    ch = (f.mean[-1] / y_last - 1) * 100 if y_last else np.nan
    lo, hi = f.pi[95] if 95 in f.pi else (f.pi[max(f.pi)] if f.pi else (None, None))
    if ar:
        s = f"يتوقع نموذج <b>{nm}</b> أن تبلغ القيمة بعد {h} فترة نحو {num(f.mean[-1])}{unit}"
        if np.isfinite(ch):
            s += f" ({'+' if ch >= 0 else ''}{ch:.1f}% عن آخر قيمة مشاهدة)"
        if lo is not None:
            s += f"، وفترة التنبؤ 95% بين {num(lo[-1])} و{num(hi[-1])}"
        return s + "."
    s = f"<b>{nm}</b> forecasts {num(f.mean[-1])}{unit} after {h} periods"
    if np.isfinite(ch):
        s += f" ({'+' if ch >= 0 else ''}{ch:.1f}% vs the last observation)"
    if lo is not None:
        s += f", with a 95% prediction interval of {num(lo[-1])} to {num(hi[-1])}"
    return s + "."


def code(kind: str, file_name: str = "your_data.csv", time: str = "date", y: str = "y", h: int = 12, s: int = 12, **kw) -> dict:
    py = ["import numpy as np, pandas as pd", f"df = pd.read_csv('{file_name}', parse_dates=['{time}'])"]
    rr = [f"df <- read.csv('{file_name}')"]
    st = [f'import delimited "{file_name}", clear varnames(1) case(preserve)']
    if kind in ("race", "classical", "ml", "eval"):
        py += [f"y = df.set_index('{time}')['{y}'].asfreq('MS' if {s} == 12 else 'QS')",
               "from statsmodels.tsa.exponential_smoothing.ets import ETSModel",
               "from statsmodels.tsa.statespace.sarimax import SARIMAX",
               "from statsmodels.tsa.forecasting.theta import ThetaModel",
               f"ets = ETSModel(y, error='mul', trend='add', damped_trend=True, seasonal='mul', seasonal_periods={s}).fit(disp=False)",
               f"print(ets.summary()); print(ets.get_prediction(len(y), len(y) + {h} - 1).summary_frame())",
               f"sar = SARIMAX(y, order=(1, 0, 1), seasonal_order=(0, 1, 1, {s}), trend='c').fit(disp=False)",
               f"print(sar.get_forecast({h}).summary_frame())",
               f"print(ThetaModel(y, period={s}).fit().forecast({h}))"]
        rr += ["library(forecast)", f"y <- ts(df${y}, frequency = {s})",
               f"fc_ets <- forecast(ets(y), h = {h}); fc_arima <- forecast(auto.arima(y), h = {h}); fc_theta <- thetaf(y, h = {h})",
               f"fc_snaive <- snaive(y, h = {h}); fc_drift <- rwf(y, h = {h}, drift = TRUE)",
               "autoplot(fc_ets); summary(fc_ets$model)",
               f"# rolling-origin evaluation (tsCV) and Diebold-Mariano",
               f"e1 <- tsCV(y, function(x, h) forecast(ets(x), h = h), h = 1, initial = length(y) - 36)",
               f"e2 <- tsCV(y, snaive, h = 1, initial = length(y) - 36)",
               "dm.test(na.omit(e1), na.omit(e2), h = 1)",
               "accuracy(fc_ets)"]
        st += [f"gen t = tm(2000m1) + _n - 1" if s == 12 else "gen t = _n", "tsset t",
               f"tssmooth hwinters hw = {y}, forecast({h})   // Holt-Winters",
               f"arima {y}, arima(1,0,1) sarima(0,1,1,{s})",
               f"tsappend, add({h})", "predict f_arima, y dynamic(tm(2025m12))" if s == 12 else "predict f_arima, y dynamic(.)",
               "* Stata has no built-in ETS / theta; see the R or Python code"]
        if kind == "ml":
            py += ["# --- machine learning with lag features (recursive) ---", "from sklearn.ensemble import HistGradientBoostingRegressor",
                   "X = pd.concat({f'lag{k}': y.shift(k) for k in range(1, 13)}, axis=1).dropna()",
                   "m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05).fit(X, y.loc[X.index])",
                   "# Prophet (optional): from prophet import Prophet"]
            rr += ["# install.packages('ranger'); random forest on lags", "library(ranger)",
                   "L <- embed(as.numeric(y), 13); rf <- ranger(x = L[, -1], y = L[, 1], num.trees = 500)",
                   "# install.packages('prophet'); library(prophet)"]
    elif kind == "nowcast":
        ind = kw.get("indicators", ["x1", "x2"])
        tgt = kw.get("target", y)
        py += ["from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ",
               f"m = df.set_index(pd.PeriodIndex(df['{time}'], freq='M'))",
               f"q = m['{tgt}'].dropna(); q.index = q.index.asfreq('Q')",
               f"mod = DynamicFactorMQ(m[{ind!r}], endog_quarterly=q.to_frame(), factors=1, factor_orders=2, idiosyncratic_ar1=True)",
               "res = mod.fit(disp=False); print(res.predict(end=str(m.index.max().asfreq('Q').asfreq('M', 'end'))))"]
        rr += ["# install.packages('midasr')", "library(midasr)",
               f"y <- ts(na.omit(df${tgt}), frequency = 4); x <- ts(df${ind[0]}, frequency = 12)",
               "m_nealmon <- midas_r(y ~ mls(y, 1, 1) + mls(x, 0:5, 3, nealmon), start = list(x = c(0.5, 0, 0)))",
               "summary(m_nealmon); hAh_test(m_nealmon)",
               "m_u <- midas_u(y ~ mls(y, 1, 1) + mls(x, 0:5, 3))"]
        st += ["* Stata 18: dfactor / 'nowcast' community commands; MIDAS: ssc install midasreg",
               f"* bridge: collapse (mean) {' '.join(ind)} (lastnm) {tgt}, by(quarter)",
               f"regress {tgt} {' '.join(ind)}"]
    elif kind == "macro":
        vs = kw.get("variables", ["y1", "y2"])
        py += [f"Y = df[{vs!r}].dropna()", "# BVAR: Aetheris (closed-form NIW Minnesota); local projections:",
               "import statsmodels.api as sm",
               "for h in range(9):", f"    yy = Y['{vs[0]}'].shift(-h); X = sm.add_constant(pd.concat([Y.shift(l) for l in (1, 2)], axis=1))",
               "    print(h, sm.OLS(yy, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': h + 1}).params.iloc[1:4].values)"]
        rr += ["# install.packages(c('BVAR', 'lpirfs'))", "library(BVAR); library(lpirfs)",
               f"Y <- df[, c({', '.join(repr(v) for v in vs)})]",
               "b <- bvar(Y, lags = 2, priors = bv_priors(hyper = 'lambda', mn = bv_minnesota(lambda = bv_lambda(mode = 0.2))))",
               "summary(b); predict(b, horizon = 8)",
               "lp <- lp_lin(endog_data = Y, lags_endog_lin = 2, trend = 0, shock_type = 0, confint = 1.64, hor = 12)", "plot(lp)"]
        st += [f"var {' '.join(vs)}, lags(1/2)", "* Bayesian VAR (Stata 17+):",
               f"bayes, rseed(1): var {' '.join(vs)}, lags(1/2)", "* local projections (Stata 18): lpirf " + " ".join(vs)]
    elif kind == "volatility":
        r = kw.get("ret", "return")
        rv = kw.get("rv", "rv")
        py += ["import statsmodels.api as sm", f"rv = df['{rv}']",
               "X = pd.DataFrame({'d': rv, 'w': rv.rolling(5).mean(), 'm': rv.rolling(22).mean()}).shift(1)",
               "har = sm.OLS(rv, sm.add_constant(X), missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': 5}); print(har.summary())",
               "# from arch import arch_model; arch_model(df['" + r + "'], vol='GARCH', p=1, o=1, q=1, dist='t').fit()"]
        rr += ["# install.packages('rugarch')", "library(rugarch)",
               "sp <- ugarchspec(variance.model = list(model = 'gjrGARCH'), distribution.model = 'std')",
               f"roll <- ugarchroll(sp, df${r}, n.start = 1000, refit.every = 20, refit.window = 'moving', calculate.VaR = TRUE, VaR.alpha = 0.01)",
               "report(roll, type = 'VaR', VaR.alpha = 0.01)",
               f"rv <- df${rv}; d <- data.frame(y = rv[-1], d = head(rv, -1))  # HAR: add weekly and monthly means"]
        st += [f"tsset t", f"arch {r}, arch(1) garch(1) distribution(t)", "predict h, variance",
               f"gen rv_w = (L.{rv} + L2.{rv} + L3.{rv} + L4.{rv} + L5.{rv}) / 5",
               f"regress {rv} L.{rv} rv_w, vce(robust)"]
    else:  # scenario
        xs = kw.get("drivers", ["x1"])
        py += ["from statsmodels.tsa.statespace.sarimax import SARIMAX", f"m = SARIMAX(df['{y}'], exog=df[{xs!r}], order=(1, 1, 1)).fit(disp=False)",
               f"future_x = pd.DataFrame({{c: [df[c].iloc[-1] * 1.02 ** k for k in range(1, {h} + 1)] for c in {xs!r}}})",
               f"print(m.get_forecast({h}, exog=future_x).summary_frame())"]
        rr += ["library(forecast)", f"fit <- Arima(df${y}, order = c(1, 1, 1), xreg = as.matrix(df[, c({', '.join(repr(v) for v in xs)})]))",
               f"newx <- matrix(rep(tail(as.matrix(df[, c({', '.join(repr(v) for v in xs)})]), 1), each = {h}), ncol = {len(xs)}) * 1.02",
               "forecast(fit, xreg = newx)"]
        st += ["gen t = _n", "tsset t", f"arima {y} {' '.join(xs)}, arima(1,1,1)", f"tsappend, add({h})",
               "* fill the driver paths for the new periods, then:", "predict f, y dynamic(.)"]
    return {"Python": "\n".join(py) + "\n", "R": "\n".join(rr) + "\n", "Stata": "\n".join(st) + "\n"}
