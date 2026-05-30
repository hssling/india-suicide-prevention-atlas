# India Suicide Prevention Intervention Atlas

Open-access companion atlas for the suicide reconciliation manuscript (IJMR submission)
and the intervention-mapping companion paper (Lancet Regional Health SE-Asia).

## Stack

- **Jekyll** (static site generator; works on GitHub Pages out of the box)
- **Leaflet.js** for the interactive state choropleth
- **CC BY 4.0** licence

## Local preview

```bash
cd atlas
bundle install
bundle exec jekyll serve
# open http://127.0.0.1:4000
```

## Deploy to GitHub Pages

1. Push the `atlas/` directory contents to the gh-pages branch of the project repo
2. Enable Pages in repo Settings -> Pages -> Source: gh-pages branch
3. Optional: set custom domain in CNAME

## Data inputs

The `data/` folder is symlinked or copied from `companion_lancet_rh_seasia/data/`:

- `intervention_library.csv` (14 interventions x 7 criteria)
- `topsis_ranking.csv` (TOPSIS closeness under 4 weight schemes)
- `intervention_impact_projection.csv` (lives saved, cost, NSPS contribution)
- `portfolio_projections.json` (top-3 + top-5 optimal portfolios)
- `state_rates.json` (NCRB 2023 state suicide rates - generated from main project)
- `india_states.geojson` (India state boundaries from data_raw/)

## Generate Zenodo DOI

After first GitHub release:
1. Link the repo to Zenodo (https://zenodo.org/account/settings/github/)
2. Cut a v1.0 release
3. Zenodo auto-mints DOI; update citation block in `index.md`
