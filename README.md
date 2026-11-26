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

- Version: `1.3.0-alpha`
- Live: https://aetheris-research.streamlit.app
- Languages: English + Arabic (full RTL support)
- ✅ Data Hub: file upload, World Bank WDI, FRED, Data Basket (multi-dataset memory + merge)
- ✅ Descriptive Statistics wing: 40 measures in 6 groups (central tendency, dispersion, shape, position, concentration), Gini & Lorenz, automatic academic interpretation, CSV/Excel export
- ✅ Academic Chart Builder: 17 chart types, journal styling, figure numbering & source notes, 300-dpi PNG / SVG / HTML export
- ✅ Inferential Statistics wing: Test Advisor, 30+ parametric & non-parametric tests, post-hoc (Tukey, Games-Howell, Dunn) with compact letters, two-way & repeated-measures ANOVA, effect sizes, APA reporting, bootstrap CIs, power & sample size (G*Power-validated), survey sample size
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
