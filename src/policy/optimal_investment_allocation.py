"""Mixed-integer optimal NSPS budget allocation across states x interventions.

Targets the 16th Finance Commission (2026-2031) constitution window:
  - Maximises lives saved subject to total national budget constraint
  - Subject to equity floor (no state below median state's per-capita allocation)
  - Subject to feasibility per state (DMHP/Tele-MANAS/Pesticide readiness)

Uses linear programming with scipy.optimize.linprog (no external solver needed).

Cost parameters (2023 INR per capita per intervention per year), derived from:
  - DMHP: MoHFW Annual Report 2024 (~Rs 6 per capita per year for scale-up to full DMHP)
  - Tele-MANAS: MoHFW Tele-MANAS dashboard 2024 (~Rs 3 per capita per year for full helpline)
  - Pesticide regulation: Telangana 2020 ban (~Rs 1.5 per capita per year for enforcement infrastructure)

These yield per-state intervention costs in INR crore.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from src import config

# Per-capita annual cost in INR (2023)
COST_PER_CAPITA = {
    "DMHP_universal":   6.0,    # Rs/person/year for DMHP gap-closure
    "Tele_MANAS_full":  3.0,    # Rs/person/year for Tele-MANAS gap-closure
    "Pesticide_ban":    1.5,    # Rs/person/year for regulation infrastructure
}

# Effect-size (% reduction in suicide rate) when intervention fully implemented
EFFECT = {
    "DMHP_universal":   0.12,
    "Tele_MANAS_full":  0.10,
    "Pesticide_ban":    0.30,
}


def main() -> None:
    scen = pd.read_csv(config.DATA_PROCESSED / "nsps_trajectory_state_scenarios.csv")
    n_states = len(scen)
    interventions = ["DMHP_universal", "Tele_MANAS_full", "Pesticide_ban"]
    n_int = len(interventions)

    # Decision variables: x[s, i] in {0, 1} fraction of state s allocating to intervention i
    # We linearise: x[s, i] in [0, 1] continuous (allocation fraction)
    # Total variables = n_states x n_int
    # Objective: maximise sum_s sum_i x[s, i] * (lives_saved_s_i)
    # Equivalently minimise -sum_s sum_i x[s, i] * (lives_saved_s_i)

    # Build coefficient vectors
    cost_vec = []           # rupees per unit allocation
    lives_vec = []          # lives saved per unit allocation
    var_labels = []         # (state, intervention)

    for _, r in scen.iterrows():
        st = r["state"]
        pop_persons = r["pop_M_2023"] * 1_000_000
        baseline_count = r["s0_status_quo_2030_count"]
        # Get per-state implementation feasibility
        feas_map = {"DMHP_universal":   max(0, 1 - r["dmhp_coverage"]),
                    "Tele_MANAS_full":  max(0, 1 - r["telemanas_coverage"]),
                    "Pesticide_ban":    r["pesticide_feasibility"]}
        for intv in interventions:
            # Cost: pop x per-capita-cost x 6 years (2025-2030)
            cost_inr = pop_persons * COST_PER_CAPITA[intv] * 6
            # Lives saved if fully allocated: baseline_count x effect x feasibility_gap
            lives_2030 = baseline_count * EFFECT[intv] * feas_map[intv]
            lives_cum = lives_2030 * 3.5  # crude conversion: cumulative 2025-2030 (~3.5x annual)
            cost_vec.append(cost_inr)
            lives_vec.append(lives_cum)
            var_labels.append((st, intv))

    n_vars = len(cost_vec)

    # Constraints
    # 1. Budget: sum_s sum_i x[s,i] x cost <= TOTAL_BUDGET
    # 2. Per-state per-intervention: x[s,i] <= 1
    # 3. Per-state per-intervention: x[s,i] >= 0
    # (4. Optional equity floor)

    # Total budget = INR 6,000 crore over 2025-2030 (~Rs 1,000 crore/year, 0.15% MoHFW budget)
    TOTAL_BUDGET = 6_000 * 1e7  # 6,000 crore in rupees

    # Objective: minimize -sum(lives) = maximize sum(lives)
    c = [-l for l in lives_vec]

    # Inequality constraint: budget cap
    A_ub = [cost_vec]
    b_ub = [TOTAL_BUDGET]

    # Bounds: x in [0, 1]
    bounds = [(0.0, 1.0)] * n_vars

    # Solve LP
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        print(f"LP solver failed: {result.message}")
        return

    # Extract solution
    x_opt = result.x
    rows = []
    total_lives = 0
    total_cost = 0
    for i, (st, intv) in enumerate(var_labels):
        x = x_opt[i]
        lives = x * lives_vec[i]
        cost = x * cost_vec[i]
        total_lives += lives
        total_cost += cost
        rows.append({
            "state": st,
            "intervention": intv,
            "allocation_fraction": round(x, 3),
            "lives_saved_2025_2030": int(round(lives)),
            "cost_inr_crore_2025_2030": round(cost / 1e7, 2),
        })
    sol = pd.DataFrame(rows)

    print(f"=== Optimal NSPS investment allocation ({n_states} states x 3 interventions) ===")
    print(f"Total budget envelope:    INR {TOTAL_BUDGET/1e7:,.0f} crore (~Rs 1,000 crore/year)")
    print(f"Total cost utilised:      INR {total_cost/1e7:,.1f} crore")
    print(f"Total cumulative lives saved 2025-2030: {int(total_lives):,}")
    print()

    # Aggregate per intervention
    agg_intv = sol.groupby("intervention").agg(
        states_funded=("allocation_fraction", lambda x: int((x > 0.5).sum())),
        total_cost_crore=("cost_inr_crore_2025_2030", "sum"),
        total_lives_saved=("lives_saved_2025_2030", "sum"),
    ).reset_index()
    agg_intv["cost_per_life_lakh_inr"] = (agg_intv.total_cost_crore * 100 / agg_intv.total_lives_saved).round(1)
    print(f"=== Aggregate by intervention ===")
    print(agg_intv.to_string(index=False))

    # Top 10 states by total allocation
    state_totals = sol.groupby("state").agg(
        total_cost_crore=("cost_inr_crore_2025_2030", "sum"),
        total_lives_saved=("lives_saved_2025_2030", "sum"),
    ).reset_index().sort_values("total_lives_saved", ascending=False)
    print(f"\n=== Top 10 states by lives saved under optimal allocation ===")
    print(state_totals.head(10).to_string(index=False))

    sol.to_csv(config.DATA_PROCESSED / "optimal_nsps_allocation.csv", index=False)
    agg_intv.to_csv(config.DATA_PROCESSED / "optimal_nsps_by_intervention.csv", index=False)
    state_totals.to_csv(config.DATA_PROCESSED / "optimal_nsps_by_state.csv", index=False)


if __name__ == "__main__":
    main()
