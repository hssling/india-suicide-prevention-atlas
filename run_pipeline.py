"""Top-level orchestrator that re-runs the entire analytic pipeline end-to-end.

Usage (from the repository root):
    python run_pipeline.py            # all stages
    python run_pipeline.py --stage paf  # single stage

Stages are ordered to respect data dependencies:
    ingest    -> raw NCRB / SRS / MCCD PDF extraction
    clean     -> population projections, sex x age panel
    analysis  -> 5-source triangulation, sex/age decomposition, SRS calibration
    bayes     -> joint Bayesian measurement-error model
    policy    -> PAF, NSPS trajectory, optimal allocation, cost-of-inaction,
                 NSPS Achievability Index, figure regeneration

Note: ingest stages require raw PDFs in data/raw/ (not redistributed in this
repository; download from NCRB, SRS-CoD and MCCD publishers as listed in
README.md). All downstream stages can be re-run from the published
data/processed/ CSVs alone.
"""
from __future__ import annotations
import argparse
import importlib
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

STAGES = {
    "ingest": [
        ("src.ingest.fetch_ncrb_historical",
         "Extract NCRB ADSI 2018-2024 national totals + state rates from PDFs"),
        ("src.ingest.fetch_srs_mccd",
         "Extract SRS Cause of Death + MCCD national suicide totals from PDFs"),
    ],
    "clean": [
        ("src.clean.state_populations",
         "Project Census 2011 state populations to 2018-2024 with decadal growth"),
        ("src.clean.build_sex_age_panel",
         "Build sex x age x year panel from IHME GBD export"),
    ],
    "analysis": [
        ("src.analysis.five_source_triangulation",
         "Triangulate NCRB + GBD + WHO + SRS + MCCD national totals 2014-2024"),
        ("src.analysis.sex_age_apc",
         "Annual percent change by sex and age (GBD 2018-2023)"),
        ("src.analysis.srs_calibrated_state_rates",
         "Apply SRS national calibration to state-level Bayesian estimates"),
        ("src.analysis.ses_gradient",
         "NFHS-5 socioeconomic gradient versus state suicide rate"),
    ],
    "bayes": [
        ("src.bayes.joint_measurement_model_v2",
         "Joint NCRB-Poisson + GBD-Normal model with Gamma(2,2) prior on c_s"),
    ],
    "policy": [
        ("src.policy.pop_attributable_fractions",
         "PAF for four modifiable Indian suicide risk factors"),
        ("src.policy.forecast_nsps_trajectory",
         "State-level NSPS 2030 forecast under 5 policy scenarios"),
        ("src.policy.optimal_allocation_greedy",
         "Greedy cost-effectiveness allocation of INR 6,000 crore envelope"),
        ("src.policy.cost_of_inaction",
         "Status-quo cost-of-inaction + benefit-cost ratio"),
        ("src.policy.nsps_achievability_index",
         "State NSPS Achievability Index (EMERGENCY/WATCH/ON-TRACK tiers)"),
        ("src.policy.regenerate_v31_figures",
         "Regenerate the 5 IJMR manuscript figures from latest data"),
    ],
}


def _run_module(name: str, description: str) -> None:
    """Import a module and run its main() function, with timing and error capture."""
    print(f"\n>>> {name}  ({description})")
    t0 = time.time()
    try:
        mod = importlib.import_module(name)
        if hasattr(mod, "main"):
            mod.main()
        else:
            print(f"    [skip: module has no main()]")
    except Exception as e:
        print(f"    !! FAILED: {type(e).__name__}: {e}")
        return
    print(f"    done in {time.time() - t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=list(STAGES.keys()) + ["all"], default="all",
                    help="Pipeline stage to run (default: all)")
    args = ap.parse_args()

    stages_to_run = list(STAGES.keys()) if args.stage == "all" else [args.stage]
    for stage in stages_to_run:
        print(f"\n========== STAGE: {stage} ==========")
        for module_name, description in STAGES[stage]:
            _run_module(module_name, description)


if __name__ == "__main__":
    main()
