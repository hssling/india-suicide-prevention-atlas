"""Greedy optimal NSPS investment allocation.

Sorts state x intervention pairs by lives-saved-per-rupee (cost-effectiveness
ratio), then funds in descending order until the INR 6,000 crore budget envelope
is exhausted.

This is equivalent to the LP optimal solution for unconstrained continuous
allocation (which is what we have here without integrality constraints).
"""
from __future__ import annotations
import pandas as pd
from src import config

COST_PER_CAPITA = {"DMHP_universal": 6.0, "Tele_MANAS_full": 3.0, "Pesticide_ban": 1.5}
EFFECT = {"DMHP_universal": 0.12, "Tele_MANAS_full": 0.10, "Pesticide_ban": 0.30}
TOTAL_BUDGET_CRORE = 6_000
TOTAL_BUDGET_INR = TOTAL_BUDGET_CRORE * 1e7


def main() -> None:
    scen = pd.read_csv(config.DATA_PROCESSED / "nsps_trajectory_state_scenarios.csv")
    interventions = ["DMHP_universal", "Tele_MANAS_full", "Pesticide_ban"]

    candidates = []
    for _, r in scen.iterrows():
        pop_persons = r["pop_M_2023"] * 1_000_000
        baseline_count = r["s0_status_quo_2030_count"]
        feas_map = {
            "DMHP_universal":  max(0, 1 - r["dmhp_coverage"]),
            "Tele_MANAS_full": max(0, 1 - r["telemanas_coverage"]),
            "Pesticide_ban":   r["pesticide_feasibility"],
        }
        for intv in interventions:
            cost_inr = pop_persons * COST_PER_CAPITA[intv] * 6
            lives_2030 = baseline_count * EFFECT[intv] * feas_map[intv]
            lives_cum = lives_2030 * 3.5  # ~ cumulative 2025-2030
            if lives_cum <= 0:
                continue
            ce_ratio = cost_inr / lives_cum  # rupees per life saved
            candidates.append({"state": r["state"], "intervention": intv,
                               "cost_inr": cost_inr, "lives_cum": lives_cum,
                               "ce_ratio_rs_per_life": ce_ratio})

    # Sort by cost-effectiveness (lowest cost per life saved first)
    df = pd.DataFrame(candidates).sort_values("ce_ratio_rs_per_life")

    # Greedy funding until budget exhausted
    cum_cost = 0
    funded = []
    for _, r in df.iterrows():
        if cum_cost + r["cost_inr"] <= TOTAL_BUDGET_INR:
            funded.append(r)
            cum_cost += r["cost_inr"]
    funded_df = pd.DataFrame(funded)
    funded_df["cost_crore"] = (funded_df.cost_inr / 1e7).round(2)
    funded_df["lives_cum"] = funded_df.lives_cum.astype(int)

    total_lives = funded_df.lives_cum.sum()
    total_cost_crore = funded_df.cost_crore.sum()

    print(f"=== Greedy optimal NSPS allocation (budget envelope INR {TOTAL_BUDGET_CRORE:,} crore) ===")
    print(f"  Total cost utilised:     INR {total_cost_crore:,.1f} crore")
    print(f"  Total lives saved:        {total_lives:,} (cumulative 2025-2030)")
    print(f"  Funded state-interventions: {len(funded_df)} of {len(df)} candidates")

    # By intervention
    by_intv = funded_df.groupby("intervention").agg(
        n_states=("state", "count"),
        cost_crore=("cost_crore", "sum"),
        lives_saved=("lives_cum", "sum"),
    ).reset_index()
    by_intv["cost_per_life_inr"] = (by_intv.cost_crore * 1e7 / by_intv.lives_saved).round(0).astype(int)
    print(f"\n=== Aggregate by intervention ===")
    print(by_intv.to_string(index=False))

    # By state
    by_state = funded_df.groupby("state").agg(
        n_interventions_funded=("intervention", "count"),
        cost_crore=("cost_crore", "sum"),
        lives_saved=("lives_cum", "sum"),
    ).reset_index().sort_values("lives_saved", ascending=False)
    by_state.to_csv(config.DATA_PROCESSED / "optimal_nsps_by_state.csv", index=False)
    funded_df.to_csv(config.DATA_PROCESSED / "optimal_nsps_allocation.csv", index=False)
    by_intv.to_csv(config.DATA_PROCESSED / "optimal_nsps_by_intervention.csv", index=False)

    print(f"\n=== Top 10 states by lives saved ===")
    print(by_state.head(10).to_string(index=False))
    print(f"\n=== Wrote 3 CSVs to data/processed/ ===")


if __name__ == "__main__":
    main()
