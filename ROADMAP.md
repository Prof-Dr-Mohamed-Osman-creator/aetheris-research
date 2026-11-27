# Aetheris Research — Strategic Roadmap

> **الملخص التنفيذي:** الهدف أن تصبح Aetheris المنصة البحثية الأولى للاقتصاديين باللغتين العربية والإنجليزية،
> وأن تجمع في مكان واحد ما يتوزع اليوم بين EViews وStata وSPSS وGAMS وبرامج كتابة التقارير:
> **البيانات ← التحليل ← التفسير الأكاديمي ← التقرير ← التدريب**، مع مساعد ذكاء اصطناعي يعتمد على النتائج
> المحسوبة فعلاً ولا يختلق الأرقام.

> 📚 **The full list of every model, test and chart to be delivered — and the release order — is in [docs/MODEL_CATALOG.md](docs/MODEL_CATALOG.md).** That catalogue is the binding development reference.

---

## 1. Positioning — why researchers will choose Aetheris

| Gap in today's tools | Aetheris answer |
|---|---|
| EViews / Stata / SPSS are expensive desktop software with English-only output | Free-to-start web platform, **Arabic-first academic output** (tables, interpretation, methodology text) |
| Data, analysis and writing live in different programs | One workflow: **Data Hub → Analysis wings → Report Studio** sharing one Smart Memory |
| Software gives numbers, not meaning | **Automatic academic interpretation** under every result (already live in Descriptive Statistics) |
| Generic AI chatbots invent numbers and references | **Grounded AI**: the assistant only reads results computed by the platform |
| Results are hard to reproduce | **Code export**: every analysis can be exported as Python / R / Stata code |
| Little regional data in one place | Curated Arab-world & Egypt data + agriculture, climate and food-security sources |
| Agricultural & development economics tools are scattered | Built-in growth/instability analysis, supply response, efficiency (DEA/SFA), farm planning LP |

---

## 2. Release plan

### ✅ Phase 1 — Core (v1.0.0-alpha)
Modular structure, bilingual UI, Smart Memory, file upload.

### ✅ Phase 2 — Data Hub + Descriptive Statistics (this release)
- **Global data:** World Bank WDI (43 curated indicators in 7 themes + search of the full catalogue, 50+ economies and regional aggregates) and FRED (22 key US / commodity series, any FRED ID, frequency alignment).
- **Data Basket:** keep many datasets in memory, switch the active one, merge (e.g. survey data + World Bank indicators).
- **Descriptive Statistics wing:** full summary table (CI, CV, skewness, kurtosis, outliers), 4 normality tests with histogram + Q-Q plot, Pearson/Spearman/Kendall matrices with significance stars, frequency tables & χ² / Cramér's V, group comparison, trend growth rates (log-linear) and instability index — every result with automatic Arabic/English interpretation and CSV export.
- Synthetic demo dataset, RTL layout for Arabic, unit tests.

### ✅ Phase 2.5 — Complete descriptive toolkit + Chart Builder
- 40 measures in 6 groups (incl. geometric/harmonic/trimmed means, MAD, quartile dispersion, percentiles, Gini, Palma ratio), brief journal view, Excel export.
- Concentration tab: Lorenz curves and Gini by group.
- Chart Builder: 17 chart types, academic styling, grayscale print mode, figure numbering, source notes, 300-dpi PNG / SVG / interactive HTML export.

### ✅ Phase 3 — Inferential Statistics (v1.3)
- One- & two-sample t-tests, paired t, one-/two-way ANOVA with Tukey/Bonferroni post-hoc.
- Non-parametric: Mann-Whitney, Wilcoxon, Kruskal-Wallis, Friedman, Fisher exact.
- Effect sizes (Cohen's d, η², Cramér's V) and power / sample-size calculator.
- **Test Advisor:** checks assumptions (normality, homogeneity of variance via Levene) and recommends the correct test.
- Delivered additionally: Welch & two-way (Type II/III) & repeated-measures ANOVA (Greenhouse-Geisser), Games-Howell, Dunn, compact letter display, McNemar, Cochran's Q, partial correlation, χ² with adjusted residuals / Fisher / G², BCa bootstrap CIs, survey sample size (Cochran, Yamane), APA 7 result strings.

### ✅ Phase 4a — Econometrics Lab: regression & diagnostics (v1.4)
Catalogue §1.1 (classical & cross-sectional regression) and §1.3 (full diagnostics) with journal tables, delivered
with the full 10-element model standard. Next: v1.5 — simultaneous equations & IV, limited dependent variables, count & survival.

### ✅ Phase 4b — IV & systems, limited dependent variables, count & survival (v1.5)
Catalogue §1.2, §1.4 and §1.5 with the full model standard and cross-validation against R. Next: v1.6 — univariate time series & unit roots.

### ✅ Phase 4c — Univariate time series & unit roots (v1.6)
Catalogue §1.7: decomposition and filters, the full unit-root battery (incl. Ng-Perron, Zivot-Andrews, HEGY), Box-Jenkins
ARIMA / SARIMA with forecasting and out-of-sample evaluation, and long memory (ARFIMA). Next: v1.7 — VAR / VECM,
cointegration, ARDL / NARDL and volatility models.

### ✅ Phase 4d — Multivariate time series, cointegration and volatility (v1.7)
Catalogue §1.8–1.9: VAR / SVAR, Granger and Toda-Yamamoto causality, Johansen / Engle-Granger / Gregory-Hansen, FMOLS /
DOLS / CCR, VECM, ARDL bounds testing, NARDL, QARDL, the GARCH family and CCC / DCC / BEKK.

### ✅ Phase 4e — Panel data (v1.8)
Catalogue §1.15: pooled / FE / RE / between / FD / FGLS with the full specification-test battery and cluster / Driscoll-Kraay / PCSE errors, panel unit roots and cointegration, Arellano-Bond and Blundell-Bond GMM, panel ARDL (PMG / MG / DFE), CCE, interactive fixed effects, panel threshold, panel VAR and Dumitrescu-Hurlin causality. Next: v1.9 — forecasting studio.

### Phase 4 — Econometrics Lab (flagship, continuing)
- **Cross-section:** OLS with full diagnostics (Breusch-Pagan/White, VIF, RESET, Jarque-Bera), robust & clustered SEs, Logit/Probit with marginal effects, quantile regression, IV/2SLS.
- **Time series:** ADF/PP/KPSS/Zivot-Andrews unit roots, **ARDL bounds testing** (Pesaran) with ECM, Johansen cointegration, VAR/VECM, Granger causality, impulse responses, ARIMA/SARIMA forecasting, structural breaks.
- **Panel data:** pooled/FE/RE, Hausman, Driscoll-Kraay SEs, panel unit roots, dynamic panel (GMM).
- **Causal inference:** difference-in-differences, event study, propensity-score matching.
- **Applied economics:** Cobb-Douglas & translog production functions, stochastic frontier analysis, Nerlove supply response, AIDS demand systems, elasticities.
- Publication-ready multi-model comparison tables (APA / journal style).

### Phase 5 — Operations Research
Linear, integer and goal programming with sensitivity analysis and shadow prices; transportation & assignment; **DEA** (CCR, BCC, Malmquist index); farm planning & MOTAD risk programming; CPM/PERT; inventory & queueing models; decision analysis under uncertainty.

### Phase 6 — Report Studio + AI Assistant
- Assemble any results into a report; export **Word (.docx) and PDF** with APA tables and figures.
- Auto-generated *methodology* and *results* sections in academic Arabic or English.
- **AI assistant (provider-agnostic):** Claude, OpenAI or open-source models (via OpenRouter / Ollama), keys stored in Streamlit secrets. Uses only the platform's computed results, cites the data source, and flags what it cannot verify.
- **Chat with your data:** ask questions in Arabic ("ما أثر الحرارة على إنتاجية القمح؟") → the assistant proposes the analysis, runs it, explains it.

### Phase 7 — Academy (courses & training)
- Structured tracks: Statistics for economists, Econometrics (beginner → advanced), Time series, Panel data, Operations research, Research methodology & academic writing.
- Lessons linked to live datasets — the learner runs the analysis inside the platform.
- Quizzes, practical assignments, **certificates**; training packages for universities, ministries and research centres.
- **Learn mode** in every wing: shows the formula, assumptions and how to interpret each output.

### Phase 8 — Community & marketplace
Researcher profiles, Q&A, shared (anonymised) projects, peer feedback, expert consultation booking, dataset sharing.

---

## 3. Platform foundations (run in parallel)

| Area | Plan |
|---|---|
| Accounts & projects | Login (Streamlit auth / Supabase), saved projects, version history |
| Monetisation | Free tier + Pro / Institutional subscriptions (Paymob for Egypt & MENA, Stripe globally) |
| More data | IMF, FAOSTAT, UN Comtrade, OECD, ILOSTAT, CAPMAS (Egypt), Our World in Data, NASA POWER & ERA5 climate data |
| Reproducibility | Export every analysis as Python / R / Stata code + full audit log |
| Quality | Unit tests for every statistical engine, validated against Stata/R outputs; CI on GitHub Actions |
| Languages | Arabic, English → French (Maghreb) |
| Performance | Caching, lazy loading, move heavy jobs to a backend API when user base grows |

---

## 4. Architecture principle

Every wing is split into a **pure engine** (`analytics/*.py`, `data/engines/*.py` — no Streamlit, unit-tested)
and a **UI layer** (`*_ui.py`). The same engines will be reused by the report generator, the AI assistant
and a future public API.
