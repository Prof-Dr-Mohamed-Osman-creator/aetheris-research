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

- Version: `1.0.0-alpha`
- Architecture: Modular (easy to extend with new wings)
- Languages: English (primary) + Arabic (full support)
- Smart Memory system: Implemented
- Core structure: Ready

## Project Structure

```
aetheris-research/
├── app.py                 # Main entry point
├── config/
│   ├── settings.py
│   └── languages.py
├── core/
│   ├── memory.py          # Smart Memory system
│   └── utils.py
├── data/
│   ├── engines/           # International data sources
│   └── upload.py          # File upload handling
├── analytics/             # Statistical & econometric modules
├── reports/               # Report generation
├── academy/
├── community/
└── requirements.txt
```

## How to Run (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Roadmap (MVP)

1. ✅ Core structure + bilingual support + Smart Memory
2. ⏳ Data Hub (Upload + World Bank + FRED)
3. ⏳ Descriptive Statistics
4. ⏳ Inferential Statistics (basic)
5. ⏳ Econometrics Lab (starting with regression)
6. ⏳ Simple Report Generator

---

Built with focus on cleanliness, expandability, and professional research experience.
