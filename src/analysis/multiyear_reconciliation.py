"""Multi-year NCRB-vs-GBD reconciliation + NCRB state-level APC 2018-2023."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import linregress
from src import config

# State populations 2023 (millions) - reused from avertable_burden module
from src.analysis.avertable_burden import STATE_POP_2023_MILLIONS


def main() -> None:
    # 1. Load NCRB historical state rates 2018-2023
    ncrb_rates = pd.read_csv(config.DATA_PROCESSED / "ncrb_state_rates_2018_2024.csv")
    print(f"NCRB state-rate observations: {len(ncrb_rates)} (years 2018-2024)")

    # 2. NCRB national totals 2018-2024
    nat = pd.read_csv(config.DATA_PROCESSED / "ncrb_national_totals_2018_2024.csv")
    print(f"NCRB national totals: \n{nat.to_string(index=False)}")

    # 3. Convert NCRB state rates to absolute counts using state populations
    pop_df = pd.DataFrame([{"state": s, "pop_m": p} for s, p in STATE_POP_2023_MILLIONS.items()])
    ncrb_rates = ncrb_rates.merge(pop_df, on="state", how="left")
    # NCRB rate is per 100k; multiplied by population (Mpop * 10) gives absolute count
    ncrb_rates["ncrb_suicides_derived"] = ncrb_rates["ncrb_rate_per_100k"] * ncrb_rates["pop_m"] * 10
    ncrb_rates["ncrb_suicides_derived"] = ncrb_rates["ncrb_suicides_derived"].round(0)

    # 4. State-level NCRB APC 2018-2023 (log-linear regression of rate on year)
    apc_rows = []
    for st in ncrb_rates["state"].dropna().unique():
        sub = ncrb_rates[(ncrb_rates["state"] == st) & ncrb_rates["ncrb_rate_per_100k"].notna()].sort_values("year")
        # Need at least 4 years
        if len(sub) < 4:
            continue
        log_r = np.log(sub["ncrb_rate_per_100k"].values + 1e-3)
        slope, _, r_val, p_val, _ = linregress(sub["year"].values, log_r)
        apc = (np.exp(slope) - 1) * 100
        apc_rows.append({
            "state": st,
            "n_years": int(len(sub)),
            "rate_2018": float(sub.iloc[0]["ncrb_rate_per_100k"]) if sub.iloc[0]["year"] == 2018 else float("nan"),
            "rate_2023": float(sub.iloc[-1]["ncrb_rate_per_100k"]) if sub.iloc[-1]["year"] == 2023 else float("nan"),
            "ncrb_apc_pct_per_year": float(apc),
            "ncrb_apc_p_value": float(p_val),
            "ncrb_r_squared": float(r_val ** 2),
        })
    apc_df = pd.DataFrame(apc_rows).sort_values("ncrb_apc_pct_per_year", ascending=False)
    print(f"\nNCRB state-level APC computed for {len(apc_df)} states (>=4 years of data)")
    print(apc_df.head(8).to_string(index=False))
    print("...")
    print(apc_df.tail(5).to_string(index=False))
    apc_df.to_csv(config.DATA_PROCESSED / "ncrb_state_apc_2018_2023.csv", index=False)

    # 5. Multi-year NCRB vs GBD reconciliation for 2020-2023 (where both have data)
    gbd = pd.read_parquet(config.DATA_PROCESSED / "gbd_trend_2020_2023.parquet")
    merged = ncrb_rates.merge(gbd, on=["state", "year"], how="inner")
    merged["ncrb_gbd_ratio"] = merged["ncrb_suicides_derived"] / merged["gbd_deaths"]
    merged.to_csv(config.DATA_PROCESSED / "ncrb_gbd_multiyear_reconciliation.csv", index=False)
    print(f"\nMulti-year NCRB-vs-GBD overlap: {len(merged)} state-year pairs")
    # National per-year totals
    by_year = merged.groupby("year").agg(
        ncrb_total=("ncrb_suicides_derived", "sum"),
        gbd_total=("gbd_deaths", "sum"),
    ).reset_index()
    by_year["ratio"] = by_year["ncrb_total"] / by_year["gbd_total"]
    print(by_year.to_string(index=False))

    # National totals: NCRB official (from CSV) vs GBD national (from parquet)
    gbd_nat = pd.read_parquet(config.DATA_PROCESSED / "gbd_trend_national.parquet")
    ncrb_nat = nat.copy()
    overlap = ncrb_nat.merge(gbd_nat, on="year", how="inner")
    overlap["ncrb_over_gbd"] = overlap["national_total"] / overlap["gbd_national"]
    print(f"\nNational NCRB-vs-GBD by year (NCRB official figure vs GBD modelled):")
    print(overlap[["year", "national_total", "gbd_national", "ncrb_over_gbd"]].to_string(index=False))
    overlap.to_csv(config.DATA_PROCESSED / "national_reconciliation_2020_2023.csv", index=False)


if __name__ == "__main__":
    main()
