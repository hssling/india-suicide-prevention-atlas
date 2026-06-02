"""Phase 2.2 — Avertable suicide burden under three counterfactual scenarios."""
from __future__ import annotations
import pandas as pd
import numpy as np
from src import config

# WHO global suicide rate for the Scenario C benchmark (per 100,000)
# Verified from WHO Global Health Observatory 2021 release.
WHO_GLOBAL_RATE = 9.0

# State populations 2023 (millions; Census 2011 projected via UIDAI/MoHFW commonly-used factors).
# Used only as denominator for rate calculations; conservative estimates.
STATE_POP_2023_MILLIONS = {
    "Uttar Pradesh": 235.0, "Maharashtra": 126.4, "Bihar": 124.5, "West Bengal": 99.0,
    "Madhya Pradesh": 87.0, "Tamil Nadu": 77.0, "Rajasthan": 81.0, "Karnataka": 68.0,
    "Gujarat": 71.0, "Andhra Pradesh": 54.0, "Odisha": 46.0, "Telangana": 38.0,
    "Kerala": 36.0, "Jharkhand": 39.5, "Assam": 36.5, "Punjab": 31.0,
    "Chhattisgarh": 30.0, "Haryana": 30.5, "Delhi": 21.0, "Jammu & Kashmir": 13.6,
    "Uttarakhand": 11.4, "Himachal Pradesh": 7.5, "Tripura": 4.2, "Meghalaya": 3.5,
    "Manipur": 3.1, "Nagaland": 2.2, "Goa": 1.6, "Arunachal Pradesh": 1.6,
    "Puducherry": 1.7, "Mizoram": 1.3, "Chandigarh": 1.2, "Sikkim": 0.7,
    "Andaman & Nicobar Islands": 0.45, "Ladakh": 0.30,
    "Dadra & Nagar Haveli and Daman & Diu": 0.62, "Lakshadweep": 0.07,
}


def attach_rates(panel: pd.DataFrame) -> pd.DataFrame:
    df = panel.copy()
    df["population_2023_m"] = df["state"].map(STATE_POP_2023_MILLIONS)
    df["ncrb_rate"] = df["ncrb_suicides"] / df["population_2023_m"] / 10.0  # /100k
    df["gbd_rate"] = df["gbd_suicides"] / df["population_2023_m"] / 10.0
    return df


def averted_deaths(deaths: float, rate: float, rate_counterfactual: float, population_m: float) -> float:
    """Avertable deaths if a state's rate dropped to the counterfactual rate."""
    if any(pd.isna(x) for x in (deaths, rate, rate_counterfactual, population_m)):
        return 0.0
    if rate <= rate_counterfactual:
        return 0.0
    averted = (rate - rate_counterfactual) * population_m * 10.0  # per100k * Mpop * 10 = absolute
    return float(max(0, averted))


def compute_central(panel: pd.DataFrame) -> pd.DataFrame:
    p23 = panel[panel["year"] == 2023].copy()
    p23 = attach_rates(p23)
    p23 = p23.dropna(subset=["gbd_suicides", "gbd_rate"])

    rate_A_25 = p23["gbd_rate"].quantile(0.25)  # 25th-percentile state benchmark
    deaths_total = p23["gbd_suicides"].sum()
    pop_total = p23["population_2023_m"].sum()
    national_rate = deaths_total / pop_total / 10.0
    rate_B = national_rate
    rate_C = WHO_GLOBAL_RATE

    rows = []
    for _, r in p23.iterrows():
        rows.append({
            "state": r["state"],
            "year": 2023,
            "gbd_deaths": r["gbd_suicides"],
            "gbd_rate": r["gbd_rate"],
            "population_m": r["population_2023_m"],
            "rate_scen_A": rate_A_25,
            "rate_scen_B": rate_B,
            "rate_scen_C": rate_C,
            "averted_A": averted_deaths(r["gbd_suicides"], r["gbd_rate"], rate_A_25, r["population_2023_m"]),
            "averted_B": averted_deaths(r["gbd_suicides"], r["gbd_rate"], rate_B, r["population_2023_m"]),
            "averted_C": averted_deaths(r["gbd_suicides"], r["gbd_rate"], rate_C, r["population_2023_m"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    panel = pd.read_parquet(config.DATA_PROCESSED / "suicide_reconciliation_panel.parquet")
    central = compute_central(panel)
    out_path = config.DATA_PROCESSED / "suicide_avertable_central.parquet"
    central.to_parquet(out_path, index=False)
    print(f"[avertable] wrote {out_path}  rows={len(central)}")
    print(f"  Scenario A (25th-pct state) rate cutoff: {central['rate_scen_A'].iloc[0]:.2f}/100k")
    print(f"  Scenario B (national avg)    rate cutoff: {central['rate_scen_B'].iloc[0]:.2f}/100k")
    print(f"  Scenario C (WHO global)      rate cutoff: {central['rate_scen_C'].iloc[0]:.2f}/100k")
    print(f"  National totals (averted, GBD-derived):")
    print(f"    Scenario A: {central['averted_A'].sum():,.0f}")
    print(f"    Scenario B: {central['averted_B'].sum():,.0f}")
    print(f"    Scenario C: {central['averted_C'].sum():,.0f}")


if __name__ == "__main__":
    main()
