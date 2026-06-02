# Reproducibility Guide

This document gives the step-by-step recipe to reproduce every numerical result, table and figure in the manuscript **"An evidence-based investment roadmap for India's National Suicide Prevention Strategy 2030 target"** from the source data.

## TL;DR

```bash
git clone https://github.com/hssling/india-suicide-prevention-atlas
cd india-suicide-prevention-atlas
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python run_pipeline.py --stage policy
```

The `policy` stage runs end-to-end from the published `data/processed/` CSVs and reproduces the 6 policy modules (PAF, forecast, allocation, cost-of-inaction, Achievability Index, figures). Total runtime is ~2 minutes on a modern laptop without re-running the upstream PDF-extraction stages.

## Pipeline architecture

| Stage | Inputs | Outputs |
|---|---|---|
| `ingest` | Raw NCRB ADSI / SRS-CoD / MCCD PDFs in `data/raw/` (not redistributed) | `data/processed/ncrb_*`, `srs_*`, `mccd_*` CSVs |
| `clean` | Census 2011 + IHME GHDx export | `data/processed/state_populations_*.csv`, `gbd_sex_age_*.parquet` |
| `analysis` | All five surveillance datasets | `data/processed/five_source_triangulation.csv`, `gbd_apc_*.csv`, `state_ses_*.csv` |
| `bayes` | NCRB state counts + GBD state estimates | `data/processed/state_sex_bayes_v1.parquet` |
| `policy` | All processed data | `data/processed/paf_*.csv`, `nsps_*.csv`, `optimal_nsps_*.csv`, `cost_of_inaction_*.csv` + figures |

## Raw data sources (not in repository)

The `ingest` stage requires the following PDFs to be downloaded by the user:

| Source | Files | Download URL |
|---|---|---|
| NCRB ADSI 2018-2024 | 7 annual reports (Chapter 2 suicides) | https://ncrb.gov.in/accidental-deaths-suicides-india |
| SRS Cause of Death 2014-2024 | 8 rolling three-year reports | https://censusindia.gov.in/nada/index.php/catalog/SRS |
| MCCD Annual Reports 2016-2023 | 8 annual reports | https://censusindia.gov.in/nada/index.php/catalog/MCCD |
| IHME GBD 2023 release | Self-harm cause, India state-level, 2018-2023 | https://vizhub.healthdata.org/gbd-results/ (free account required) |
| WHO GHO MH_12 | India 2010-2021 sex-stratified | https://ghoapi.azureedge.net/api/MH_12 (OData API) |

Place these in `data/raw/` following the structure in `src/config.py`.

## Re-running with `data/processed/` CSVs only

Most users will want to reproduce the policy outputs without re-extracting from raw PDFs. The repository ships with all intermediate `data/processed/` CSVs and JSONs. Running:

```bash
python run_pipeline.py --stage policy
```

regenerates:

- `data/processed/paf_modifiable_risk_factors.csv`
- `data/processed/nsps_trajectory_state_scenarios.csv`
- `data/processed/optimal_nsps_allocation.csv` (+ `_greedy`, `_by_state`, `_by_intervention`)
- `data/processed/cost_of_inaction_summary.csv`
- `data/processed/nsps_achievability_index.csv`
- `figures/Figure_1_nsps_trajectory.png` through `Figure_5_achievability_index.png`

All published numbers in the IJMR manuscript can be matched cell-for-cell against these regenerated files.

## Stage-by-stage commands

| To regenerate | Run |
|---|---|
| PAF table (Table 2) | `python -m src.policy.pop_attributable_fractions` |
| Trajectory forecast (Table S2) | `python -m src.policy.forecast_nsps_trajectory` |
| Optimal allocation (Table 3, Table S3) | `python -m src.policy.optimal_allocation_greedy` |
| LP cross-validation (Table S7) | `python -m src.policy.optimal_investment_allocation` |
| Cost-of-inaction (Table S6) | `python -m src.policy.cost_of_inaction` |
| Achievability Index (Table 4) | `python -m src.policy.nsps_achievability_index` |
| All 5 figures | `python -m src.policy.regenerate_v31_figures` |
| 5-source triangulation (Table S1) | `python -m src.analysis.five_source_triangulation` |

## Verification checklist

After running the policy stage, verify against the manuscript:

| Manuscript value | Reproduce from | Expected |
|---|---|---|
| SRS-anchored national 2023 baseline | `data/processed/five_source_triangulation.csv` (year=2023, srs_count) | 252,000 |
| Status-quo 2030 projection | `data/processed/nsps_trajectory_state_scenarios.csv` (sum of s0) | 363,522 |
| Integrated NSPS 2030 projection | (sum of s4_integrated_2030_count) | 277,771 |
| NSPS 10% target margin | 327,169 - 277,771 | 49,398 lives |
| Optimal allocation total lives saved | `data/processed/optimal_nsps_allocation.csv` (sum lives_cum) | 296,363 |
| Pesticide regulation lives saved | (intervention=Pesticide_ban, sum lives_cum) | 254,241 |
| Pesticide cost per life saved | (intervention=Pesticide_ban, weighted) | INR 51,304 |
| Status-quo cost-of-inaction | `data/processed/cost_of_inaction_summary.csv` | INR 61.8 lakh crore |
| Benefit-cost ratio | (same) | 103:1 |
| EMERGENCY-tier states | `data/processed/nsps_achievability_index.csv` (tier=EMERGENCY) | 10 states (UP, Bihar, Jharkhand, Uttarakhand, Tripura, Manipur, Sikkim, Assam, Mizoram, Arunachal Pradesh) |

## Software environment

- Python 3.11 (CPython)
- See `requirements.txt` for library versions
- Tested on Windows 11, macOS 14 (Apple Silicon), Ubuntu 22.04

## Computational cost

| Stage | Approximate runtime (laptop) |
|---|---|
| `ingest` | 5-15 minutes (PDF extraction) |
| `clean` | < 30 seconds |
| `analysis` | 1-2 minutes |
| `bayes` | 15-25 minutes (4 chains × 2000 draws) |
| `policy` | 1-2 minutes |
| `all` | 25-40 minutes end-to-end |

## License

All code in this repository is released under CC BY 4.0. See `LICENSE`.
