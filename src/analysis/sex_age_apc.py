"""Phase B/C - APC + avertable burden by sex x age stratum (national level).

Computes:
  - APC 2018-2023 by sex (Male, Female, Both) overall and per age group
  - Female:Male ratio by age band, 2023
  - Avertable burden under three scenarios:
       A: state matches the 25th-percentile age-band rate (within sex)
       B: state matches the national Both-sex all-ages average rate
       C: state matches WHO global benchmark of 9.0/100k
     (applied per sex x age stratum)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import linregress
from src import config


WHO_GLOBAL_RATE = 9.0


def main() -> None:
    nat = pd.read_parquet(config.DATA_PROCESSED / "gbd_sex_age_national.parquet")
    print(f"Loaded national sex x age: {len(nat)} rows")

    # === APC by sex (all-ages) ===
    print("\n=== APC by sex, 2018-2023, all-ages, India national (Deaths Number) ===")
    rows = []
    for sex in ["Both", "Male", "Female"]:
        sub = nat[(nat["sex"] == sex) & (nat["age"] == "All ages") & (nat["metric"] == "Number")].sort_values("year")
        log_v = np.log(sub["value"].values)
        slope, _, r, p, _ = linregress(sub["year"].values, log_v)
        apc = (np.exp(slope) - 1) * 100
        rows.append({"sex": sex, "n_years": int(len(sub)), "value_2018": float(sub.iloc[0]["value"]),
                     "value_2023": float(sub.iloc[-1]["value"]), "apc_pct_per_year": float(apc),
                     "p_value": float(p), "r_squared": float(r ** 2)})
        print(f"  {sex:8s}: 2018={sub.iloc[0]['value']:,.0f} -> 2023={sub.iloc[-1]['value']:,.0f}  "
              f"APC={apc:+.2f}%/yr (p={p:.4f}, R^2={r**2:.3f})")
    apc_sex = pd.DataFrame(rows)
    apc_sex.to_csv(config.DATA_PROCESSED / "gbd_apc_by_sex_2018_2023.csv", index=False)

    # === APC by sex x age band ===
    print("\n=== APC by sex x age band, 2018-2023 (national Deaths Number) ===")
    age_bands = [a for a in nat["age"].unique() if a != "All ages"]
    rows = []
    for sex in ["Both", "Male", "Female"]:
        for age in age_bands:
            sub = nat[(nat["sex"] == sex) & (nat["age"] == age) & (nat["metric"] == "Number")].sort_values("year")
            if len(sub) < 4:
                continue
            log_v = np.log(sub["value"].values)
            slope, _, r, p, _ = linregress(sub["year"].values, log_v)
            apc = (np.exp(slope) - 1) * 100
            rows.append({"sex": sex, "age": age, "value_2018": float(sub.iloc[0]["value"]),
                         "value_2023": float(sub.iloc[-1]["value"]),
                         "apc_pct_per_year": float(apc), "p_value": float(p)})
    apc_sex_age = pd.DataFrame(rows)
    apc_sex_age.to_csv(config.DATA_PROCESSED / "gbd_apc_by_sex_age_2018_2023.csv", index=False)
    print(f"Saved per-stratum APC for {len(apc_sex_age)} sex x age combinations")
    print("\nTop 5 sex x age strata by absolute APC magnitude (any direction):")
    print(apc_sex_age.assign(apc_abs=apc_sex_age["apc_pct_per_year"].abs())
          .sort_values("apc_abs", ascending=False).head(5)
          [["sex","age","apc_pct_per_year","p_value"]].to_string(index=False))

    # === Female:Male ratio by age band, 2023 ===
    nat23 = nat[(nat["year"] == 2023) & (nat["metric"] == "Number")
                & (nat["measure"] == "Deaths") & (nat["age"] != "All ages")]
    pivot = nat23.pivot_table(index="age", columns="sex", values="value")
    pivot["F_to_M_ratio"] = pivot["Female"] / pivot["Male"]
    pivot["F_share_pct"] = 100 * pivot["Female"] / pivot["Both"]
    age_order = ["15-19 years","20-24 years","25-29 years","30-34 years","35-39 years",
                 "40-44 years","45-49 years","50-54 years","55-59 years","60-64 years",
                 "65-69 years","70+ years"]
    pivot = pivot.reindex(age_order)
    print("\n=== Female:Male ratio by age band, India 2023 (GBD Deaths Number) ===")
    print(pivot[["Female","Male","Both","F_to_M_ratio","F_share_pct"]]
          .round({"Female":0,"Male":0,"Both":0,"F_to_M_ratio":2,"F_share_pct":1}).to_string())
    pivot.to_csv(config.DATA_PROCESSED / "gbd_fm_ratio_2023.csv")

    # === Avertable burden by sex (all-ages, national) Scenario C: WHO 9.0 ===
    rates = nat[(nat["year"] == 2023) & (nat["metric"] == "Rate") & (nat["age"] == "All ages")]
    numbers = nat[(nat["year"] == 2023) & (nat["metric"] == "Number") & (nat["age"] == "All ages")]
    nat_pop_2023 = float(numbers[numbers["sex"] == "Both"]["value"].iloc[0]) / float(rates[rates["sex"] == "Both"]["value"].iloc[0]) * 100_000
    print(f"\nDerived 2023 India population (from GBD Both Number / Rate): {nat_pop_2023/1e6:.1f} million")

    averted_rows = []
    for sex in ["Male", "Female", "Both"]:
        rate_2023 = float(rates[rates["sex"] == sex]["value"].iloc[0])
        deaths_2023 = float(numbers[numbers["sex"] == sex]["value"].iloc[0])
        # Population per sex (approximate: M:F ≈ 51.5:48.5 in India 2023)
        pop_sex = nat_pop_2023 * (0.515 if sex == "Male" else 0.485 if sex == "Female" else 1.0)
        averted_C = max(0, (rate_2023 - WHO_GLOBAL_RATE) * pop_sex / 100_000)
        averted_rows.append({"sex": sex, "rate_2023": rate_2023, "deaths_2023": deaths_2023,
                             "pop_2023": pop_sex, "averted_who_9_per100k": averted_C})
    averted = pd.DataFrame(averted_rows)
    averted.to_csv(config.DATA_PROCESSED / "averted_by_sex_who_benchmark.csv", index=False)
    print("\n=== Avertable burden if rate dropped to WHO global benchmark (9.0/100k), by sex ===")
    print(averted.round({"rate_2023":2,"deaths_2023":0,"pop_2023":0,"averted_who_9_per100k":0}).to_string(index=False))


if __name__ == "__main__":
    main()
