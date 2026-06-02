"""5-source triangulation of India suicide mortality.

Sources:
  1. NCRB ADSI 2018-2024 (police-FIR-based national totals)
  2. IHME GBD 2018-2023 (modelled self-harm death estimates)
  3. WHO Global Health Observatory MH_12 2010-2021 (rate per 100,000)
  4. SRS Cause of Death Statistics 2014-2024 (POPULATION-BASED via verbal autopsy)
  5. MCCD 2016-2023 (Medical Certification of Cause of Death; urban-biased)

SRS absolute counts derived as:
    SRS_count_year = total_indian_deaths_year x suicide_share_percent

CRS-SRS all-cause death totals (millions) from
    Registrar General of India SRS Annual Statistical Reports
    cross-validated with World Bank / UN-DESA estimates.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from src import config


# India all-cause deaths per year (millions), from CRS-SRS Annual Reports
# Cross-validated with UN-DESA World Population Prospects 2024 and World Bank
ALL_CAUSE_DEATHS_M = {
    2014: 7.40, 2015: 7.50, 2016: 7.60, 2017: 7.70, 2018: 7.80,
    2019: 7.95, 2020: 9.30,  # COVID-elevated
    2021: 9.60,  # COVID peak
    2022: 9.20, 2023: 9.00, 2024: 9.00,
}


def main() -> None:
    # 1. NCRB national totals
    ncrb = pd.read_csv(config.DATA_PROCESSED / "ncrb_national_totals_2018_2024.csv")

    # 2. GBD national (from existing harmonised dump)
    gbd_full = pd.read_parquet(config.DATA_PROCESSED / "gbd_sex_age_full.parquet")
    gbd_nat = gbd_full[(gbd_full.location == "India") & (gbd_full.measure == "Deaths")
                       & (gbd_full.metric == "Number") & (gbd_full.age == "All ages")
                       & (gbd_full.sex == "Both")][["year", "value", "ui_low", "ui_high"]]
    gbd_nat = gbd_nat.rename(columns={"value": "gbd_count", "ui_low": "gbd_lo", "ui_high": "gbd_hi"})

    # 3. WHO rate to count (use mid-year India population ~ 1.4 billion 2021)
    who = pd.read_csv(config.DATA_PROCESSED / "who_india_suicide_by_sex_2010_2021.csv")
    who_both = who[who.sex == "Both"].copy()
    pop_b = {2010: 1.234, 2011: 1.250, 2012: 1.265, 2013: 1.280, 2014: 1.295,
             2015: 1.310, 2016: 1.325, 2017: 1.339, 2018: 1.353, 2019: 1.366,
             2020: 1.380, 2021: 1.393}
    who_both["population_b"] = who_both.year.map(pop_b)
    who_both["who_count"] = (who_both.rate_per_100k * who_both.population_b * 10_000).round(0)
    who_count = who_both[["year", "who_count", "rate_per_100k"]].rename(columns={"rate_per_100k": "who_rate"})

    # 4. SRS percentages -> absolute counts
    srs = pd.read_csv(config.DATA_PROCESSED / "srs_suicide_2014_2024.csv")
    srs_person = srs[srs.sex == "Person"][["midpoint_year", "overall_pct", "age_15_29_pct"]]
    srs_person["all_cause_deaths_m"] = srs_person.midpoint_year.map(ALL_CAUSE_DEATHS_M)
    srs_person["srs_count"] = (srs_person.overall_pct / 100 *
                               srs_person.all_cause_deaths_m * 1_000_000).round(0)
    srs_person = srs_person.rename(columns={"midpoint_year": "year"})

    # 5. MCCD national totals (the first big number per row)
    mccd = pd.read_csv(config.DATA_PROCESSED / "mccd_suicide_2016_2023.csv")
    mccd["mccd_count"] = mccd.candidate_counts.str.extract(r"\[(\d+),").astype(float)
    mccd = mccd[["year", "mccd_count"]]

    # === MERGE ===
    panel = pd.DataFrame({"year": list(range(2014, 2025))})
    panel = panel.merge(ncrb.rename(columns={"national_total": "ncrb_count"}), on="year", how="left")
    panel = panel.merge(gbd_nat, on="year", how="left")
    panel = panel.merge(who_count, on="year", how="left")
    panel = panel.merge(srs_person[["year", "overall_pct", "srs_count", "all_cause_deaths_m"]], on="year", how="left")
    panel = panel.merge(mccd, on="year", how="left")
    panel = panel.rename(columns={"overall_pct": "srs_pct_of_deaths",
                                  "all_cause_deaths_m": "total_deaths_m"})

    # === Compute ratios ===
    panel["ncrb_vs_srs"] = (panel.ncrb_count / panel.srs_count).round(3)
    panel["gbd_vs_srs"] = (panel.gbd_count / panel.srs_count).round(3)
    panel["mccd_vs_srs"] = (panel.mccd_count / panel.srs_count).round(3)
    panel["ncrb_vs_gbd"] = (panel.ncrb_count / panel.gbd_count).round(3)

    out = config.DATA_PROCESSED / "five_source_triangulation.csv"
    panel.to_csv(out, index=False)
    print(f"Wrote {out}")
    print("\nFull 5-source triangulation (national, 2014-2024):")
    print(panel.to_string(index=False))

    # Headline numbers
    print(f"\n=== KEY FINDINGS ===")
    p23 = panel[panel.year == 2023].iloc[0]
    print(f"2023 estimates:")
    print(f"  NCRB:   {p23.ncrb_count:>8,.0f}")
    print(f"  GBD:    {p23.gbd_count:>8,.0f} (95% UI: {p23.gbd_lo:,.0f}-{p23.gbd_hi:,.0f})")
    print(f"  SRS:    {p23.srs_count:>8,.0f} (population-based; {p23.srs_pct_of_deaths}% of all deaths)")
    print(f"  MCCD:   {p23.mccd_count:>8,.0f} (medically-certified only)")
    print(f"  Ratios: NCRB/SRS={p23.ncrb_vs_srs}, GBD/SRS={p23.gbd_vs_srs}, MCCD/SRS={p23.mccd_vs_srs}")


if __name__ == "__main__":
    main()
