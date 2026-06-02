"""NSPS 2030 forecast under three scenarios.

Uses NCRB national totals 2018-2024 as the historical anchor.

Scenarios:
  A) Status quo: continuation of 2022-2024 average annual change (essentially 0%)
  B) NSPS 10% reduction target: linear decline 2024 -> 0.9 x 2024 by 2030
  C) Best-state intervention model: if national rate fell at the TN-equivalent
     pace under a successful intervention (-2.0%/year), national counts at 2030
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import config


def main() -> None:
    nat = pd.read_csv(config.DATA_PROCESSED / "ncrb_national_totals_2018_2024.csv")
    nat = nat.sort_values("year")
    obs_2024 = float(nat[nat["year"] == 2024]["national_total"].iloc[0])

    # Status quo growth: average YoY 2022-2024
    last3 = nat[nat["year"].isin([2022, 2023, 2024])]["national_total"].values
    yoy_status_quo = float(np.diff(last3).mean() / last3[:-1].mean())  # avg annual relative change

    forecast_rows = []
    for year in range(2025, 2031):
        years_ahead = year - 2024
        scen_a = obs_2024 * (1 + yoy_status_quo) ** years_ahead
        # B: linear decline 2024 -> 0.9 x 2024 by 2030 (6-year window)
        target_b_2030 = obs_2024 * 0.9
        scen_b = obs_2024 + (target_b_2030 - obs_2024) * (years_ahead / 6)
        # C: -2%/year (Tamil-Nadu-style intervention applied nationally)
        scen_c = obs_2024 * (1 - 0.02) ** years_ahead
        forecast_rows.append({
            "year": year,
            "scenario_A_status_quo": int(round(scen_a)),
            "scenario_B_NSPS_10pct_2030": int(round(scen_b)),
            "scenario_C_best_state_minus2pct": int(round(scen_c)),
        })

    # Add historical for context
    hist_rows = [{"year": int(r["year"]),
                  "scenario_A_status_quo": int(r["national_total"]),
                  "scenario_B_NSPS_10pct_2030": int(r["national_total"]),
                  "scenario_C_best_state_minus2pct": int(r["national_total"])} for _, r in nat.iterrows()]
    df = pd.DataFrame(hist_rows + forecast_rows)
    out_path = config.DATA_PROCESSED / "nsps_forecast_2030.csv"
    df.to_csv(out_path, index=False)

    print(f"NSPS 2030 forecast under three scenarios (NCRB national total projection):\n")
    print(df.to_string(index=False))
    print(f"\nKey takeaways:")
    print(f"  Status quo 2030: {df.iloc[-1]['scenario_A_status_quo']:,} (annual growth {yoy_status_quo*100:+.2f}%)")
    print(f"  NSPS target 2030: {df.iloc[-1]['scenario_B_NSPS_10pct_2030']:,} (90% of 2024)")
    print(f"  Best-state model 2030: {df.iloc[-1]['scenario_C_best_state_minus2pct']:,} (-2%/yr sustained)")
    print(f"  NSPS target gap from status quo (lives by 2030): {df.iloc[-1]['scenario_A_status_quo'] - df.iloc[-1]['scenario_B_NSPS_10pct_2030']:,}")
    cumulative_gap = sum(df[df["year"] >= 2025]["scenario_A_status_quo"] - df[df["year"] >= 2025]["scenario_B_NSPS_10pct_2030"])
    print(f"  CUMULATIVE lives saved (2025-2030) if NSPS target met vs status quo: ~{cumulative_gap:,}")


if __name__ == "__main__":
    main()
