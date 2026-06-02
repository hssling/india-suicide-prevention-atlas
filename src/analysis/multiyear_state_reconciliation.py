"""Multi-year state-level NCRB-vs-GBD reconciliation 2020-2023.

For each state with NCRB rates (Figure 2.4) AND GBD modelled rates for
2020-2023, computes the NCRB/GBD ratio per year and the average ratio
+ year-over-year change.

This addresses 'reconciliation across the full overlap window' rather
than only 2023.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import config
from src.clean.state_populations import project_population


def main() -> None:
    # 1. NCRB rates 2018-2023 (from Figure 2.4 extraction)
    ncrb = pd.read_csv(config.DATA_PROCESSED / "ncrb_state_rates_2018_2024.csv")
    # 2. GBD state deaths 2020-2023
    gbd = pd.read_parquet(config.DATA_PROCESSED / "gbd_trend_2020_2023.parquet")
    # 3. Populations
    pops = pd.read_csv(config.DATA_PROCESSED / "state_populations_2018_2024.csv")

    # Convert GBD count to rate per 100k using year-matched populations
    gbd = gbd.merge(pops, on=["state", "year"], how="left")
    gbd["gbd_rate_per_100k"] = gbd["gbd_deaths"] / (gbd["population_m"] * 10.0)

    # Merge NCRB rate and GBD rate per state-year
    merged = ncrb.merge(gbd[["state", "year", "gbd_rate_per_100k", "gbd_deaths", "population_m"]],
                        on=["state", "year"], how="inner")
    merged["ncrb_rate_per_100k_observed"] = merged["ncrb_rate_per_100k"]
    merged["ncrb_gbd_rate_ratio"] = merged["ncrb_rate_per_100k_observed"] / merged["gbd_rate_per_100k"]

    out_long = merged[["state", "year", "ncrb_rate_per_100k_observed", "gbd_rate_per_100k", "ncrb_gbd_rate_ratio", "population_m"]]
    out_long.to_csv(config.DATA_PROCESSED / "state_year_reconciliation_2020_2023.csv", index=False)

    # Per-state summary across years
    summary_rows = []
    for st in out_long["state"].unique():
        sub = out_long[out_long["state"] == st].sort_values("year")
        if len(sub) < 2:
            continue
        summary_rows.append({
            "state": st,
            "n_years_overlap": int(len(sub)),
            "ncrb_rate_2020": float(sub[sub["year"] == 2020]["ncrb_rate_per_100k_observed"].iloc[0]) if 2020 in sub["year"].values else float("nan"),
            "ncrb_rate_2023": float(sub[sub["year"] == 2023]["ncrb_rate_per_100k_observed"].iloc[0]) if 2023 in sub["year"].values else float("nan"),
            "gbd_rate_2020": float(sub[sub["year"] == 2020]["gbd_rate_per_100k"].iloc[0]) if 2020 in sub["year"].values else float("nan"),
            "gbd_rate_2023": float(sub[sub["year"] == 2023]["gbd_rate_per_100k"].iloc[0]) if 2023 in sub["year"].values else float("nan"),
            "ratio_2020": float(sub[sub["year"] == 2020]["ncrb_gbd_rate_ratio"].iloc[0]) if 2020 in sub["year"].values else float("nan"),
            "ratio_2023": float(sub[sub["year"] == 2023]["ncrb_gbd_rate_ratio"].iloc[0]) if 2023 in sub["year"].values else float("nan"),
            "mean_ratio_2020_2023": float(sub["ncrb_gbd_rate_ratio"].mean()),
            "ratio_change_2020_to_2023": float(
                sub[sub["year"] == 2023]["ncrb_gbd_rate_ratio"].iloc[0] - sub[sub["year"] == 2020]["ncrb_gbd_rate_ratio"].iloc[0]
            ) if (2020 in sub["year"].values and 2023 in sub["year"].values) else float("nan"),
        })
    summary = pd.DataFrame(summary_rows).sort_values("mean_ratio_2020_2023", ascending=False)
    summary.to_csv(config.DATA_PROCESSED / "state_reconciliation_summary_2020_2023.csv", index=False)

    print(f"State-year overlap (NCRB rate AND GBD rate both present): {len(out_long)} state-year pairs")
    print(f"Unique states: {out_long['state'].nunique()}")

    # Print the per-state summary highlighting interesting states
    print("\nStates ranked by mean NCRB/GBD rate ratio (2020-2023):")
    cols = ["state", "n_years_overlap", "mean_ratio_2020_2023", "ratio_2020", "ratio_2023", "ratio_change_2020_to_2023"]
    print(summary[cols].head(10).to_string(index=False))
    print("...")
    print(summary[cols].tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
