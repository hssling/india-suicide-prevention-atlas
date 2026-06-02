"""NSPS Achievability Index per state.

Targets the MoHFW NSPS dashboard finalisation:
  composite scoring of each state on its probability of meeting NSPS 2030 target,
  based on:
    1. Current burden level (SRS-anchored rate)
    2. Trajectory direction (NCRB APC 2018-2023)
    3. DMHP infrastructure readiness
    4. Tele-MANAS rollout completeness
    5. Pesticide-regulation feasibility (agri-belt + political readiness)
    6. State health budget per capita (capacity)
    7. NCRB under-registration (data-quality penalty)

Each domain z-normalised; composite = weighted mean; states ranked into
3 priority tiers (EMERGENCY / WATCH / ON-TRACK) for NSPS programme.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import config

# State health budget per capita (Rs 2023, MoHFW NHM allocations + state ratios)
# Source: 15th Finance Commission tabulations; State Health Mission allocations
STATE_HEALTH_BUDGET_PER_CAPITA = {
    "Andhra Pradesh":   2400, "Arunachal Pradesh": 4800, "Assam": 2100, "Bihar": 1300,
    "Chhattisgarh":     1900, "Goa": 5200, "Gujarat": 2300, "Haryana": 2400,
    "Himachal Pradesh": 4500, "Jammu & Kashmir": 4200, "Jharkhand": 1700, "Karnataka": 2200,
    "Kerala": 3200, "Madhya Pradesh": 1800, "Maharashtra": 2100, "Manipur": 3600,
    "Meghalaya": 4400, "Mizoram": 4000, "Nagaland": 3800, "Odisha": 2000, "Punjab": 2300,
    "Rajasthan": 2100, "Sikkim": 5500, "Tamil Nadu": 2600, "Telangana": 2500,
    "Tripura": 3500, "Uttar Pradesh": 1500, "Uttarakhand": 2700, "West Bengal": 1800,
}


def main() -> None:
    # Pull state-level forecast and SRS-anchored baseline
    scen = pd.read_csv(config.DATA_PROCESSED / "nsps_trajectory_state_scenarios.csv")
    scen["budget_per_capita"] = scen.state.map(STATE_HEALTH_BUDGET_PER_CAPITA)

    # Pull NCRB c_anchored (under-registration; lower c = lower NCRB capture)
    cal = pd.read_parquet(config.DATA_PROCESSED / "state_rates_srs_calibrated.parquet")
    scen = scen.merge(cal[["state", "c_anchored_to_srs"]], on="state", how="left")

    # Domains for composite (higher = better for NSPS achievability)
    # 1. burden_level: higher current rate = LOWER achievability (negative weight)
    # 2. trajectory: lower APC = HIGHER achievability (negative APC weighted positively)
    # 3. DMHP coverage: higher = better
    # 4. Tele-MANAS: higher = better
    # 5. pesticide feasibility: higher = better
    # 6. budget per capita: higher = better
    # 7. NCRB under-registration: higher c (closer to 1) = better data quality

    def zscore(x):
        x = pd.to_numeric(x, errors="coerce")
        return (x - x.mean()) / x.std(ddof=0)

    scen["z_burden"]       = -zscore(scen.baseline_rate_2023)  # higher rate = worse
    scen["z_trajectory"]   = -zscore(scen.apc_pct)             # higher APC = worse
    scen["z_dmhp"]         = zscore(scen.dmhp_coverage)
    scen["z_telemanas"]    = zscore(scen.telemanas_coverage)
    scen["z_pesticide"]    = zscore(scen.pesticide_feasibility)
    scen["z_budget"]       = zscore(scen.budget_per_capita)
    # NCRB data quality: penalise extreme (very low or very high c)
    # Distance from 1.0 = under/over-registration; lower distance = better
    scen["c_dist_from_1"] = (scen.c_anchored_to_srs - 1.0).abs()
    scen["z_data_quality"] = -zscore(scen.c_dist_from_1)       # higher distance = worse

    # Weights (sum to 1)
    weights = {"z_burden": 0.20, "z_trajectory": 0.15, "z_dmhp": 0.15,
               "z_telemanas": 0.10, "z_pesticide": 0.10, "z_budget": 0.15,
               "z_data_quality": 0.15}

    scen["achievability_index"] = sum(weights[k] * scen[k] for k in weights)

    # Tier states
    q33, q67 = scen.achievability_index.quantile([0.33, 0.67])
    def tier(x):
        if x < q33: return "EMERGENCY"
        if x < q67: return "WATCH"
        return "ON-TRACK"
    scen["tier"] = scen.achievability_index.apply(tier)

    out_cols = ["state", "baseline_rate_2023", "apc_pct", "dmhp_coverage", "telemanas_coverage",
                "pesticide_feasibility", "budget_per_capita", "c_anchored_to_srs",
                "achievability_index", "tier"]
    out = scen[out_cols].sort_values("achievability_index", ascending=False).round(3)
    out.to_csv(config.DATA_PROCESSED / "nsps_achievability_index.csv", index=False)

    print(f"=== NSPS Achievability Index per state ===")
    print(out.head(15).to_string(index=False))
    print(f"\n=== EMERGENCY-tier states (lowest 33% of Achievability Index) ===")
    emergency = out[out.tier == "EMERGENCY"]
    for _, r in emergency.iterrows():
        print(f"  {r.state:25s} index={r.achievability_index:+.2f}  baseline_rate={r.baseline_rate_2023:.1f}  APC={r.apc_pct:+.1f}%  DMHP={r.dmhp_coverage:.0%}")

    print(f"\n=== ON-TRACK states (highest 33%) ===")
    on_track = out[out.tier == "ON-TRACK"]
    for _, r in on_track.iterrows():
        print(f"  {r.state:25s} index={r.achievability_index:+.2f}  baseline_rate={r.baseline_rate_2023:.1f}  APC={r.apc_pct:+.1f}%  DMHP={r.dmhp_coverage:.0%}")


if __name__ == "__main__":
    main()
