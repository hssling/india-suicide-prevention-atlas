# India Suicide Prevention Atlas

[![Licence: CC BY 4.0](https://img.shields.io/badge/Licence-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Jekyll site](https://img.shields.io/badge/Live-Atlas-blue)](https://hssling.github.io/india-suicide-prevention-atlas/)

Open-access decision-analytic synthesis of five Indian suicide-surveillance systems (NCRB, GBD, WHO, SRS, MCCD), anchored to the population-based Sample Registration System, with state-level National Suicide Prevention Strategy (NSPS) 2030 trajectory forecasts, Population-Attributable Fractions for four modifiable risk factors, optimal investment allocation, cost-of-inaction monetisation, and an NSPS Achievability Index per state.

**Live atlas:** https://hssling.github.io/india-suicide-prevention-atlas/

## What this repository contains

```
.
├── README.md                  this file
├── REPRODUCIBILITY.md         step-by-step pipeline reproduction guide
├── CITATION.cff               machine-readable citation
├── LICENSE                    CC BY 4.0
├── requirements.txt           Python dependencies
├── run_pipeline.py            end-to-end pipeline orchestrator
├── _config.yml                Jekyll site config (for GitHub Pages atlas)
├── Gemfile                    Jekyll dependencies
├── index.md                   atlas landing page
├── map.md                     atlas interactive Leaflet map
├── interventions/             atlas intervention pages
├── assets/                    atlas JS / CSS / static files
├── src/                       Python analysis package
│   ├── config.py              paths and constants
│   ├── ingest/                NCRB / SRS / MCCD PDF extraction
│   ├── clean/                 population projections, sex × age panel
│   ├── analysis/              five-source triangulation, SES gradient, sex/age decomposition
│   ├── bayes/                 joint Bayesian measurement-error model (PyMC)
│   ├── policy/                PAF, NSPS forecast, optimal allocation, cost-of-inaction, Achievability Index
│   ├── viz/                   figure generation
│   └── tables/                manuscript table builders
├── data/
│   ├── processed/             analysis-ready CSVs and JSONs (~40 files)
│   ├── intervention_library.csv   TOPSIS intervention library
│   ├── topsis_ranking.csv         TOPSIS ranking under 3 weight schemes
│   ├── portfolio_projections.json optimal portfolio impact
│   ├── state_rates.json           NCRB 2023 state rates (for atlas map)
│   └── india_states.geojson       India state boundaries (for atlas map)
└── figures/                   5 manuscript figures (PNG)
```

## Quick reproduction

```bash
git clone https://github.com/hssling/india-suicide-prevention-atlas
cd india-suicide-prevention-atlas
pip install -r requirements.txt
python run_pipeline.py --stage policy
```

This regenerates every result in the published manuscript from the committed `data/processed/` CSVs. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for stage-by-stage detail.

## Headline findings (from the manuscript)

- The SRS-anchored 2023 Indian suicide baseline is **252,000 deaths** (NCRB captures 68%).
- Four modifiable risk factors account for **81% of the burden additive** (untreated depression 35.1%, pesticide method access 23.1%, IPV in women 15-49 50.9% within subset, heavy alcohol 11.1%).
- An **INR 6,000 crore six-year investment envelope** (0.15% of MoHFW annual budget) saves **296,363 lives** cumulatively over 2025-2030.
- **Pesticide regulation alone delivers 86% of those lives saved** at INR 51,304 per life.
- **Benefit-cost ratio = 103:1**; cost per DALY averted = INR 12,401.
- **Ten EMERGENCY-tier states** (Uttar Pradesh, Bihar, Jharkhand, Uttarakhand, Tripura, Manipur, Sikkim, Assam, Mizoram, Arunachal Pradesh) need priority NSPS Health-Grant allocation.

## Five surveillance systems triangulated

| Source | Type | Coverage | Capture vs SRS (2023) |
|---|---|---|---|
| NCRB ADSI | Police FIR | National + 36 states/UTs | 0.68 |
| IHME GBD | Modelled | National + 31 state-equivalents | 0.79 |
| WHO GHO MH_12 | Modelled | National sex-stratified | (rate-derived) |
| **SRS Cause of Death** | **Population-based (verbal autopsy)** | **National** | **1.00 (anchor)** |
| MCCD | Medical certification only | National + state | 0.03 |

## Atlas (Jekyll site)

The `_config.yml`, `index.md`, `map.md`, `interventions/`, `assets/` files build the public-facing atlas at https://hssling.github.io/india-suicide-prevention-atlas/. The atlas surfaces:

- State NCRB 2023 suicide rates on an interactive Leaflet choropleth
- Per-intervention TOPSIS ranking with three weighting schemes
- Per-state recommended-intervention details
- Downloadable open data

## Author

**Dr Siddalingaiah H S**
Professor, Department of Community Medicine
Shridevi Institute of Medical Sciences and Research Hospital, Tumkur, Karnataka, India
Email: hssling@yahoo.com · ORCID: [0000-0002-4771-8285](https://orcid.org/0000-0002-4771-8285)

## How to cite

See [CITATION.cff](CITATION.cff) for the machine-readable citation. Suggested human-readable citation:

> Siddalingaiah HS. India Suicide Prevention Atlas (v1.0). 2026. https://github.com/hssling/india-suicide-prevention-atlas

## License

Code and content are released under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgement of AI assistance

The analytic code and manuscript drafts in this repository were prepared with assistance from a large language model (Anthropic Claude, Opus model series). All scientific decisions, data interpretation, analytical choices, and final conclusions are those of the sole author. Every numerical finding and every reference was independently verified against primary sources before publication. The complete code is open-source and re-runnable for independent verification.
