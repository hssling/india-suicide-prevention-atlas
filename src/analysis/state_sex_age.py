"""State x sex x age suicide burden analysis (GBD 2023, all 31 states + India).

Outputs:
  - Per-state F:M ratio + age-band pattern (2023)
  - Per-state sex-stratified APC over the available window
  - Female young-adult (15-29) absolute burden ranking - the priority subgroup
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import linregress
from src import config


def main() -> None:
    full = pd.read_parquet(config.DATA_PROCESSED / "gbd_sex_age_full.parquet")
    print(f"Full panel: {len(full)} rows; locations: {full.location.nunique()}; years: {sorted(full.year.unique())}")

    # === State x sex (all-ages) 2023 ===
    deaths23 = full[(full["year"] == 2023) & (full["measure"] == "Deaths")
                    & (full["metric"] == "Number") & (full["age"] == "All ages")
                    & (full["location"] != "India")]
    pivot = deaths23.pivot_table(index="location", columns="sex", values="value")
    pivot["F_to_M"] = pivot["Female"] / pivot["Male"]
    pivot = pivot.sort_values("Female", ascending=False)
    pivot.to_csv(config.DATA_PROCESSED / "state_sex_2023.csv")
    print(f"\nTop 10 states by Female suicide deaths (2023):")
    print(pivot.head(10).round({"Both":0,"Female":0,"Male":0,"F_to_M":2}).to_string())

    # === State female young-adult burden (15-29, summed across 3 age bands) ===
    yng_ages = ["15-19 years", "20-24 years", "25-29 years"]
    yng = full[(full["year"] == 2023) & (full["measure"] == "Deaths") & (full["metric"] == "Number")
               & (full["age"].isin(yng_ages)) & (full["sex"] == "Female")
               & (full["location"] != "India")]
    yng_state = yng.groupby("location").value.sum().reset_index().rename(columns={"value": "female_15_29_2023"})
    yng_state = yng_state.sort_values("female_15_29_2023", ascending=False)
    yng_state.to_csv(config.DATA_PROCESSED / "state_female_young_adult_2023.csv", index=False)
    print(f"\nTop 10 states by Female 15-29 suicide deaths (2023):")
    print(yng_state.head(10).round({"female_15_29_2023":0}).to_string(index=False))

    # === State Female all-ages APC (where >=4 years available) ===
    print(f"\n=== Female all-ages APC per state ===")
    apc_rows = []
    fem_all = full[(full["measure"] == "Deaths") & (full["metric"] == "Number")
                   & (full["age"] == "All ages") & (full["sex"] == "Female")
                   & (full["location"] != "India")]
    for st in fem_all["location"].unique():
        sub = fem_all[fem_all["location"] == st].sort_values("year")
        if len(sub) < 4: continue
        log_v = np.log(sub["value"].values)
        slope, _, r, p, _ = linregress(sub["year"].values, log_v)
        apc = (np.exp(slope) - 1) * 100
        apc_rows.append({"state": st, "n_years": int(len(sub)),
                         "female_2018": float(sub[sub.year == sub.year.min()]["value"].iloc[0]),
                         "female_2023": float(sub[sub.year == 2023]["value"].iloc[0]) if 2023 in sub.year.values else float("nan"),
                         "apc_pct_per_year": float(apc), "p_value": float(p)})
    apc_df = pd.DataFrame(apc_rows).sort_values("apc_pct_per_year", ascending=False)
    apc_df.to_csv(config.DATA_PROCESSED / "state_female_apc.csv", index=False)
    print(f"Top 5 states by Female suicide APC (increasing):")
    print(apc_df.head(5).round({"female_2018":0,"female_2023":0,"apc_pct_per_year":2,"p_value":4}).to_string(index=False))
    print(f"\nBottom 5 states by Female APC (decreasing):")
    print(apc_df.tail(5).round({"female_2018":0,"female_2023":0,"apc_pct_per_year":2,"p_value":4}).to_string(index=False))
    print(f"\nN states with significant (p<0.05) female trend: {(apc_df['p_value']<0.05).sum()}/{len(apc_df)}")


if __name__ == "__main__":
    main()
