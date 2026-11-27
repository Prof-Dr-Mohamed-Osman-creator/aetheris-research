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

- Version: `1.5.0-alpha`
- Live: https://aetheris-research.streamlit.app
- Languages: English + Arabic (full RTL support)
- ✅ Data Hub: file upload, World Bank WDI, FRED, Data Basket (multi-dataset memory + merge)
- ✅ Descriptive Statistics wing: 40 measures in 6 groups (central tendency, dispersion, shape, position, concentration), Gini & Lorenz, automatic academic interpretation, CSV/Excel export
- ✅ Academic Chart Builder: 17 chart types, journal styling, figure numbering & source notes, 300-dpi PNG / SVG / HTML export
- ✅ Inferential Statistics wing: Test Advisor, 30+ parametric & non-parametric tests, post-hoc (Tukey, Games-Howell, Dunn) with compact letters, two-way & repeated-measures ANOVA, effect sizes, APA reporting, bootstrap CIs, power & sample size (G*Power-validated), survey sample size
- ✅ **Econometrics Lab (v1.4 — catalogue §1.1 + §1.3):** guided & expert regression builder; 5 functional forms + Box-Cox; dummies/ANCOVA, polynomials, interactions (simple slopes, Johnson-Neyman), mediation (Sobel + bootstrap); stepwise, best subsets, hierarchical; OLS, WLS, FGLS, GLS-AR(1), robust M (Huber/Tukey), restricted LS, nonlinear LS; HC0–HC3, Newey-West, cluster & bootstrap SEs; 25+ diagnostics (BP/Koenker, White, Glejser, Park, Goldfeld-Quandt, exact Durbin-Watson, Breusch-Godfrey, Ljung-Box, Durbin h, RESET, Chow, CUSUM/CUSUMSQ, JB/SW/K²/AD, VIF & Belsley, Cook/DFFITS/DFBETAS); Wald/LR/LM; elasticities & marginal effects (delta method); journal-style multi-model tables; bilingual interpretation; policy brief + what-if; methodology paragraph with APA references; Excel / Word / LaTeX export and reproducible Python / R / Stata code. Validated against statsmodels (`tests/test_regression.py`) and against R — the exported R code is executed and must reproduce every estimate, SE and diagnostic (`tests/test_r_crosscheck.py`, skipped if R is absent). A self-checking Stata do-file is in `docs/validation/stata_check.do` (regenerate with `python tools/make_stata_check.py`).
- ✅ **Econometrics Lab v1.5 (catalogue §1.2, §1.4, §1.5):** IV — 2SLS, LIML, Fuller, efficient GMM with first-stage diagnostics, Cragg-Donald / Stock-Yogo, Anderson-Rubin, Sargan / Basmann / Hansen J, Durbin / Wu-Hausman; systems — identification, ILS, 2SLS, 3SLS, SUR / iterated SUR; qualitative response — LPM, logit, probit, cloglog, multinomial logit (IIA), ordered logit / probit (Brant); censored & selection — Tobit, truncated, Heckman (two-step / ML), Cragg and double hurdle, fractional response; count — Poisson, NB2, ZIP / ZINB, hurdle, PPML; survival — Kaplan-Meier, log-rank, Cox (Schoenfeld test, Harrell C), parametric AFT, discrete-time hazard. Average marginal effects and policy scenarios for every model; validated against R (`tests/test_r_crosscheck_v15.py`), linearmodels and lifelines (`tests/test_v15.py`).
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
