---
layout: default
title: India Suicide Prevention Intervention Atlas
---

# India Suicide Prevention Intervention Atlas

**Version 1.0** | Licence: CC BY 4.0 | Last updated: 2026-05-30

## What this atlas provides

- **State-level suicide burden** (NCRB 2018-2024 official totals + WHO/GBD modelled rates)
- **14 evidence-graded interventions** scored on 7 criteria (effect, evidence, cost, feasibility, equity, scale, India-applicability)
- **TOPSIS multi-criteria ranking** with three weighting schemes (equal, equity-heavy, cost-heavy)
- **NSPS-2030 burden-reduction projections** per intervention and for optimal portfolios
- **Downloadable open data** for replication and re-weighting

## Top 5 interventions (TOPSIS equal-weights)

1. **Decriminalisation impact monitoring (MHCA 2017 Sec 115)** - closeness 0.685
2. **Responsible media reporting guidelines** - closeness 0.558
3. **Pesticide regulation (means restriction)** - closeness 0.500
4. **Tele-MANAS national helpline expansion** - closeness 0.479
5. **Alcohol availability restriction (state-level)** - closeness 0.421

## Optimal portfolios

- **Top-3 portfolio:** Decriminalisation + Media + Pesticide - meets NSPS 10% target at INR 3,477 cr over 2025-2030
- **Top-5 portfolio:** Above + Tele-MANAS + Alcohol restriction - meets NSPS at INR 4,427 cr

## Interactive state map

See [/map/](map/) for the interactive Leaflet choropleth of state suicide rates with hover-to-recommend.

## Intervention pages

- [/interventions/](interventions/) - one page per intervention with full evidence, cost, and feasibility detail

## Methodology

1. **Burden data:** NCRB ADSI 2018-2024 + IHME GBD 2020-2023 + WHO GHO MH_12 2010-2021
2. **Joint Bayesian measurement-error model** (PyMC; Gamma prior on under-registration coefficient)
3. **Intervention library:** Curated from WHO LIVE LIFE (2021), mhGAP-IG v2.0, NSPS-2022, and 14 published Indian RCTs / programme evaluations
4. **TOPSIS:** Hwang & Yoon 1981; weights default equal across 7 criteria
5. **Impact projection:** Per-intervention effect x 2024 NCRB baseline (170,746) over 6-year window; subadditive cap at NSPS target

## Data downloads

- [intervention_library.csv](data/intervention_library.csv) - raw 14-intervention scoring matrix
- [topsis_ranking.csv](data/topsis_ranking.csv) - ranking under 4 weight schemes
- [intervention_impact_projection.csv](data/intervention_impact_projection.csv) - per-intervention lives/cost/NSPS
- [portfolio_projections.json](data/portfolio_projections.json) - top-3 + top-5 portfolios

## Citation

> Siddalingaiah HS. (2026). *India Suicide Prevention Intervention Atlas v1.0*.
> Companion to: State-level reconciliation of suicide mortality in India to inform
> the National Suicide Prevention Strategy 2030 target (manuscript submitted to
> Indian Journal of Medical Research).
> Available at: https://hssling.github.io/india-suicide-prevention-atlas/.
> Source: https://github.com/hssling/india-suicide-prevention-atlas.
> Licence: CC BY 4.0. DOI: pending Zenodo mint on first GitHub release.
> ORCID: 0000-0002-4771-8285.

## Reproducibility

All analysis code at: `companion_lancet_rh_seasia/src/topsis/`
- `rank_interventions.py` - TOPSIS engine
- `impact_projection.py` - burden-reduction projection
