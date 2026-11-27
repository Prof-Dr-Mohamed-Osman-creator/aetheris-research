# Aetheris Research

**Global Research Platform for Economists & Decision-Makers**

Aetheris Research is a commercial research platform that enables economists, researchers, and decision-makers to:

- Access reliable international data
- Upload their own datasets (CSV, Excel, etc.)
- Perform rigorous statistical and econometric analysis
- Generate professional reports
- Access training content and community features

## Vision

A complete research workflow inside one integrated environment:  
**Data → Analysis → Insights → Reports → Recommendations**

## Current Status

- Version: `1.10.0-alpha`
- Live: https://aetheris-research.streamlit.app
- Languages: English + Arabic (full RTL support)
- ✅ Data Hub: file upload, World Bank WDI, FRED, Data Basket (multi-dataset memory + merge)
- ✅ Descriptive Statistics wing: 40 measures in 6 groups (central tendency, dispersion, shape, position, concentration), Gini & Lorenz, automatic academic interpretation, CSV/Excel export
- ✅ Academic Chart Builder: 17 chart types, journal styling, figure numbering & source notes, 300-dpi PNG / SVG / HTML export
- ✅ Inferential Statistics wing: Test Advisor, 30+ parametric & non-parametric tests, post-hoc (Tukey, Games-Howell, Dunn) with compact letters, two-way & repeated-measures ANOVA, effect sizes, APA reporting, bootstrap CIs, power & sample size (G*Power-validated), survey sample size
- ✅ **Econometrics Lab (v1.4 — catalogue §1.1 + §1.3):** guided & expert regression builder; 5 functional forms + Box-Cox; dummies/ANCOVA, polynomials, interactions (simple slopes, Johnson-Neyman), mediation (Sobel + bootstrap); stepwise, best subsets, hierarchical; OLS, WLS, FGLS, GLS-AR(1), robust M (Huber/Tukey), restricted LS, nonlinear LS; HC0–HC3, Newey-West, cluster & bootstrap SEs; 25+ diagnostics (BP/Koenker, White, Glejser, Park, Goldfeld-Quandt, exact Durbin-Watson, Breusch-Godfrey, Ljung-Box, Durbin h, RESET, Chow, CUSUM/CUSUMSQ, JB/SW/K²/AD, VIF & Belsley, Cook/DFFITS/DFBETAS); Wald/LR/LM; elasticities & marginal effects (delta method); journal-style multi-model tables; bilingual interpretation; policy brief + what-if; methodology paragraph with APA references; Excel / Word / LaTeX export and reproducible Python / R / Stata code. Validated against statsmodels (`tests/test_regression.py`) and against R — the exported R code is executed and must reproduce every estimate, SE and diagnostic (`tests/test_r_crosscheck.py`, skipped if R is absent). A self-checking Stata do-file is in `docs/validation/stata_check.do` (regenerate with `python tools/make_stata_check.py`).
- ✅ **Econometrics Lab v1.5 (catalogue §1.2, §1.4, §1.5):** IV — 2SLS, LIML, Fuller, efficient GMM with first-stage diagnostics, Cragg-Donald / Stock-Yogo, Anderson-Rubin, Sargan / Basmann / Hansen J, Durbin / Wu-Hausman; systems — identification, ILS, 2SLS, 3SLS, SUR / iterated SUR; qualitative response — LPM, logit, probit, cloglog, multinomial logit (IIA), ordered logit / probit (Brant); censored & selection — Tobit, truncated, Heckman (two-step / ML), Cragg and double hurdle, fractional response; count — Poisson, NB2, ZIP / ZINB, hurdle, PPML; survival — Kaplan-Meier, log-rank, Cox (Schoenfeld test, Harrell C), parametric AFT, discrete-time hazard. Average marginal effects and policy scenarios for every model; validated against R (`tests/test_r_crosscheck_v15.py`), linearmodels and lifelines (`tests/test_v15.py`).
- ✅ **Econometrics Lab v1.6 (catalogue §1.7):** univariate time series — series preparation (panel filters, logs, period detection), STL / classical decomposition, HP / Baxter-King / Christiano-Fitzgerald filters; unit roots — ADF, Phillips-Perron, KPSS, DF-GLS, Ng-Perron, Zivot-Andrews, seasonal HEGY and an automatic I(0)/I(1)/I(2) strategy; Box-Jenkins — ACF/PACF, automatic order search (AIC/AICc/BIC), exact-ML ARMA / ARIMA / SARIMA, residual diagnostics and inverse roots, forecasts with fan charts (log back-transform), hold-out evaluation with Diebold-Mariano; long memory — GPH, local Whittle, ARFIMA (Whittle). Validated against R urca / uroot / fracdiff / arima / forecast (`tests/test_r_crosscheck_ts.py`) and `tests/test_ts.py`; Stata checks added to `docs/validation/stata_check.do`.
- ✅ **Econometrics Lab v1.7 (catalogue §1.8, §1.9):** VAR (lag selection, stability, portmanteau / LM / Edgerton-Shukur / multivariate JB and ARCH, Granger, instantaneous and Toda-Yamamoto causality, orthogonalised and generalised IRF with bootstrap bands, FEVD, forecasts); structural VAR (A / B / AB by scoring ML, Blanchard-Quah); cointegration (Engle-Granger + ECM, Johansen with the five deterministic cases and simulated p-values, Gregory-Hansen, FMOLS / DOLS / CCR); VECM (β, α, Γ, β and weak-exogeneity LR tests, levels IRF); ARDL bounds testing (five cases, simulated asymptotic and finite-sample critical values, long-run multipliers, restricted ECM, CUSUM), NARDL with asymmetry tests and dynamic multipliers, QARDL; volatility — ARCH, GARCH, GARCH-M, GJR / TGARCH, EGARCH with normal / t / GED / skew-t errors, robust SEs, news impact, VaR / ES backtests, and CCC / DCC / BEKK with the Engle-Sheppard test and hedge ratios. Validated against R vars, urca, cointReg, ARDL, quantreg, fGarch (`tests/test_r_crosscheck_v17.py`) and the Python arch package; Stata checks added to `docs/validation/stata_check.do`.
- ✅ **Econometrics Lab v1.8 (catalogue §1.15 — panel data):** pooled, one- and two-way FE, Swamy-Arora RE, between, FD and FGLS; F, Breusch-Pagan / Honda, Hausman and robust (Mundlak) Hausman, Pesaran CD / LM / scaled LM, Wooldridge serial correlation, modified Wald, Pesaran-Yamagata slope homogeneity; cluster, Driscoll-Kraay and PCSE standard errors; panel unit roots (LLC, IPS, Maddala-Wu, Choi, Hadri, CIPS) and cointegration (Kao, Pedroni, Westerlund); difference and system GMM (Arellano-Bond / Blundell-Bond, Windmeijer, Hansen, AR(1)/AR(2), collapse, long-run effects); panel ARDL (PMG, MG, DFE with Hausman tests); MG / CCE-MG / CCE pooled, interactive fixed effects (Bai), Hansen panel threshold with bootstrap, panel VAR with IRFs; Dumitrescu-Hurlin causality. Validated against R plm / pgmm (`tests/test_r_crosscheck_v18.py`); Stata checks (xtreg, xtgls, xtabond2) added to `docs/validation/stata_check.do`.
- ✅ **Forecasting Studio v1.9 (catalogue §3.1–3.4, §3.6):** a new wing with an automatic model horse-race (rolling-origin backtests, MASE / RMSE ranking, Diebold-Mariano tests, combination of the top three); benchmarks, ETS (R forecast::ets conventions), theta, ARIMA / SARIMAX with automatic order selection, structural state-space models, BSTS (Gibbs sampler) and growth curves; nowcasting with bridge equations, MIDAS (U-MIDAS, exponential Almon, beta, Almon) and a mixed-frequency dynamic factor model; Bayesian VAR (Minnesota / NIW prior with marginal-likelihood λ) and local projections; HAR-RV, GARCH / EWMA / historical-simulation VaR backtests (Kupiec, Christoffersen) and volatility-forecast comparison (QLIKE, Mincer-Zarnowitz); random forest, gradient boosting, LightGBM, SVR with conformal intervals and a Prophet-style model; forecast combinations (inverse-MSE, Bates-Granger, Granger-Ramanathan); fan charts and scenario forecasts for decision-makers. Validated against R forecast and stats::nls (`tests/test_r_crosscheck_v19.py`), the R BVAR package and independent base-R implementations of MIDAS, bridge equations and local projections (`tests/test_r_crosscheck_extra.py`); XGBoost, CatBoost and Meta's Prophet are integrated (machine-learning tab and horse-race); Stata checks added to `docs/validation/stata_check.do`.
- ✅ **Applied Economics Lab v1.10 (catalogue §2.7–2.8):** a new wing with production functions (Cobb-Douglas, translog, CES, generalized Leontief, quadratic, transcendental, Leontief) and their economics (elasticities, RTS, Allen / Morishima substitution, stages of production, isoquants); yield response with the economic optimum rate; optimal input use (VMP = price); cubic and long-run cost, scale and scope economies; translog cost and profit systems by restricted iterated SUR; stochastic frontiers (half-normal, truncated, exponential, Battese-Coelli 1995 determinants, Battese-Coelli 1992 panel), meta-frontier, DEA and Malmquist; Olley-Pakes / Levinsohn-Petrin / ACF and Törnqvist / Fisher productivity; Nerlove supply response and cobweb; demand, Engel curves, LES, LA-AIDS, AIDS, QUAIDS and Rotterdam with exact welfare measures; consumption functions. Validated against R nls, systemfit and lpSolve (`tests/test_r_crosscheck_v110.py`) and independently coded R likelihoods for ALS, exponential, Battese-Coelli 1995 and 1992 frontiers (`tests/test_r_crosscheck_extra.py`); Stata checks (frontier, xtfrontier, quaids, prodest) added to `docs/validation/stata_check.do`.
- Strategic plan: see [ROADMAP.md](ROADMAP.md) · Full model catalogue: [docs/MODEL_CATALOG.md](docs/MODEL_CATALOG.md)

## Project Structure

```
aetheris-research/
├── app.py                     # Main entry point & navigation
├── config/                    # Settings and language dictionary
├── core/
│   ├── memory.py              # Smart Memory (multi-dataset library + active dataset)
│   └── utils.py               # i18n helpers, RTL styling, shared widgets
├── data/
│   ├── engines/
│   │   ├── worldbank.py       # World Bank Indicators API (no key)
│   │   └── fred.py            # FRED CSV endpoint (no key)
│   ├── upload.py              # My Data (CSV, Excel, TSV, JSON)
│   ├── global_data.py         # Global Data UI
│   ├── basket.py              # Data Basket UI
│   ├── merge.py               # Dataset merging engine
│   └── samples.py             # Synthetic demo dataset
├── analytics/
│   ├── descriptive.py         # Descriptive statistics engine (pure, tested)
│   ├── descriptive_ui.py      # Descriptive Statistics wing
│   ├── inferential.py         # Hypothesis tests, post-hoc, effect sizes, CIs (pure, tested)
│   ├── inferential_ui.py      # Inferential Statistics wing + Test Advisor
│   ├── power.py               # Power analysis & sample size
│   ├── regression.py          # Regression engine §1.1 (pure, tested)
│   ├── diagnostics.py         # Full diagnostic battery §1.3 (pure, tested)
│   ├── reg_report.py          # Journal tables, elasticities, interpretation, what-if, methodology, code export
│   ├── reg_charts.py          # Residual / influence / CUSUM / effect charts
│   ├── econometrics_ui.py     # Econometrics Lab wing
│   ├── iv.py                  # IV / GMM / systems §1.2 (pure, tested)
│   ├── ldv.py                 # Limited dependent variables §1.4 (pure, tested)
│   ├── count_surv.py          # Count & survival models §1.5 (pure, tested)
│   ├── margins.py             # Average marginal effects & policy scenarios for any model
│   ├── model_report.py        # v1.5 diagnostics, interpretation, methodology, references
│   ├── model_code.py          # v1.5 reproducible Python / R / Stata code
│   ├── model_charts.py        # ROC, probability curves, rootograms, survival curves
│   ├── econ_models_ui.py      # v1.5 Econometrics Lab tabs
│   ├── ts.py                  # Univariate time series & unit roots §1.7 (pure, tested)
│   ├── ts_report.py           # v1.6 interpretation, methodology, Python / R / Stata code
│   ├── ts_ui.py               # v1.6 Time-series tab
│   ├── var.py                 # VAR / SVAR / cointegration / VECM §1.8 (pure, tested)
│   ├── ardl.py                # ARDL bounds, NARDL, QARDL §1.8 (pure, tested)
│   ├── garch.py               # GARCH family and multivariate GARCH §1.9 (pure, tested)
│   ├── mts_report.py          # v1.7 interpretation, methodology, Python / R / Stata code
│   ├── mts_ui.py              # v1.7 tabs: VAR & cointegration, ARDL / NARDL, volatility
│   ├── panel.py               # v1.8 static panel models, tests, robust SEs, unit roots, CIPS, DH, MG / CCE
│   ├── panel_gmm.py           # v1.8 difference / system GMM (pgmm / xtabond2 conventions)
│   ├── panel_adv.py           # v1.8 panel ARDL, cointegration, IFE, threshold, panel VAR
│   ├── panel_report.py        # v1.8 advice, methodology, Python / R / Stata code
│   ├── panel_ui.py            # v1.8 panel-data tab
│   ├── fcst.py                # v1.9 forecasting engines, evaluation, combination, horse-race, VaR
│   ├── nowcast.py             # v1.9 bridge, MIDAS, dynamic factors, BVAR, local projections
│   ├── fcst_report.py         # v1.9 interpretation, methodology, Python / R / Stata code
│   ├── forecast_ui.py         # v1.9 Forecasting Studio wing
│   ├── production.py          # v1.10 production, cost, SUR, frontiers, DEA, Malmquist, TFP
│   ├── demand.py              # v1.10 supply response, demand, Engel, demand systems, welfare, consumption
│   ├── applied_report.py      # v1.10 methodology, references, Python / R / Stata code
│   ├── applied_ui.py          # v1.10 Applied Economics Lab wing
│   ├── charts.py              # Academic chart engine (17 chart types)
│   └── chart_builder_ui.py    # Chart Builder
├── tests/                     # Unit tests (pytest)
├── reports/  academy/  community/
├── ROADMAP.md
└── requirements.txt
```

## How to Run (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Run the tests:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Roadmap (summary)

1. ✅ Core structure + bilingual support + Smart Memory
2. ✅ Data Hub (Upload + World Bank + FRED + Data Basket)
3. ✅ Descriptive Statistics
4. ✅ Inferential Statistics + Test Advisor
5. ⏳ Econometrics Lab (OLS diagnostics, ARDL, panel, VAR, SFA…)
6. ⏳ Operations Research (LP, DEA, transportation…)
7. ⏳ Report Studio + grounded AI assistant
8. ⏳ Academy (courses, training, certificates) & Community

---

Built with focus on cleanliness, expandability, and professional research experience.

### Optional validation environment
The BVAR cross-check uses the R package `BVAR` from conda-forge; point `AETHERIS_RSCRIPT_BVAR` to that Rscript (default `/opt/mm/renv/bin/Rscript`). Tests are skipped when a reference package is missing.
