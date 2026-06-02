"""Project-wide constants and paths for meta_suicide_reconciliation_india."""
from __future__ import annotations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
OUT_FIGS = OUTPUTS / "figures"
OUT_TABLES = OUTPUTS / "tables"
MANUSCRIPT = ROOT / "manuscript"
MANUSCRIPT_JOIAC = MANUSCRIPT / "joiac"
SUBMISSION = ROOT / "submission"
DOCS = ROOT / "docs"

# Workspace artefacts to reuse
WORKSPACE_ROOT = ROOT.parent
EXISTING_NCRB_DIR = WORKSPACE_ROOT / "data_raw" / "ncrb"
EXISTING_NCRB_CSV = WORKSPACE_ROOT / "data_interim" / "ncrb_accidental_deaths_2023.csv"
EXISTING_GBD_HARMONIZED = WORKSPACE_ROOT / "data_interim" / "gbd_harmonized.csv"
EXISTING_GEOJSON_STATES = WORKSPACE_ROOT / "data_raw" / "india_states.geojson"
EXISTING_NFHS5_DISTRICTS = WORKSPACE_ROOT / "data_raw" / "NFHS_5_India_Districts_Factsheet_Data(Sheet1).csv"

# Analytic constants
START_YEAR = 2018
END_YEAR = 2023
ENDPOINT_YEAR = 2023

# Counterfactual benchmarks
SCENARIO_A_LABEL = "Indian best-performing state (25th percentile)"
SCENARIO_B_LABEL = "National-average suicide rate"
SCENARIO_C_LABEL = "WHO global suicide rate"
WHO_GLOBAL_SUICIDE_RATE_2021 = 9.0   # per 100,000 — verify from WHO GHO at execution

# Monte Carlo
N_MC_DRAWS = 1000
SEED = 20260528
NCRB_RECALL_SD = 0.10  # wider than RTI (0.05) per spec rationale

# Bayesian SAE
BAYES_CHAINS = 4
BAYES_WARMUP = 2000
BAYES_DRAWS = 2000
BAYES_TARGET_RHAT = 1.01
BAYES_TARGET_ESS = 1000

# Small-state threshold (population)
SMALL_STATE_POP_THRESHOLD = 2_000_000

# Single author (constant across all manuscript files)
AUTHOR_NAME = "Dr Siddalingaiah H S"
AUTHOR_DEGREE = "MD"
AUTHOR_TITLE = "Professor"
AUTHOR_DEPT = "Department of Community Medicine"
AUTHOR_INSTITUTION = "Shridevi Institute of Medical Sciences and Research Hospital"
AUTHOR_CITY = "Tumkur, Karnataka, India"
AUTHOR_EMAIL = "hssling@yahoo.com"
AUTHOR_PHONE = "+91 8941087719"
AUTHOR_ORCID = "0000-0002-4771-8285"
