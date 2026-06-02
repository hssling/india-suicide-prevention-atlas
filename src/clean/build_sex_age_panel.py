"""Build sex x age x year analytic panel from GHDx export.

Filters the raw GHDx CSV to:
  - Cause: Self-harm (only present cause)
  - Locations: India national (state-level requires re-download with all subnationals)
  - Sexes: Male, Female, Both
  - Ages: All-ages + 12 five-year bands
  - Years: 2018-2023

Writes:
  data/processed/gbd_sex_age_national.parquet  (long-format)
  data/processed/gbd_sex_age_national_wide.csv (wide for inspection)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from src import config

# Prefer the full state-coverage export (8011e3b1) when present; fall back to partial (d6f85fdc)
_FULL = config.DATA_RAW / "IHME-GBD_2023_DATA-8011e3b1-1" / "IHME-GBD_2023_DATA-8011e3b1-1.csv"
_PARTIAL = config.DATA_RAW / "IHME-GBD_2023_DATA-d6f85fdc-1" / "IHME-GBD_2023_DATA-d6f85fdc-1.csv"
GHDX_CSV = _FULL if _FULL.exists() else _PARTIAL


def main() -> None:
    df = pd.read_csv(GHDX_CSV)
    print(f"Raw GHDx rows: {len(df)}; locations: {df.location_name.unique()}")

    # Standardise column names
    df = df.rename(columns={
        "location_name": "location",
        "measure_name": "measure",
        "metric_name": "metric",
        "sex_name": "sex",
        "age_name": "age",
        "cause_name": "cause",
        "val": "value",
        "upper": "ui_high",
        "lower": "ui_low",
    })

    # Save full long-format panel
    keep = ["location", "year", "sex", "age", "measure", "metric", "value", "ui_low", "ui_high", "cause"]
    full = df[keep].copy()
    out_long = config.DATA_PROCESSED / "gbd_sex_age_full.parquet"
    full.to_parquet(out_long, index=False)
    print(f"[clean] wrote {out_long}  rows={len(full)}")

    # National sex x age subset (Deaths + Rate; one row per sex-age-year combo)
    nat = full[(full["location"] == "India")
               & (full["measure"] == "Deaths")
               & (full["metric"].isin(["Number", "Rate"]))].copy()
    out_nat = config.DATA_PROCESSED / "gbd_sex_age_national.parquet"
    nat.to_parquet(out_nat, index=False)
    print(f"[clean] wrote {out_nat}  rows={len(nat)}")

    # Wide pivot: sex x age x year for Deaths Number (so we can inspect counts)
    pivot = (nat[nat["metric"] == "Number"]
             .pivot_table(index=["age", "sex"], columns="year", values="value")
             .reset_index())
    out_wide = config.DATA_PROCESSED / "gbd_sex_age_national_wide.csv"
    pivot.to_csv(out_wide, index=False)
    print(f"[clean] wrote {out_wide}")
    print(f"\nIndia 2023 sex x age suicide deaths (counts):")
    p23 = pivot[[c for c in pivot.columns if c in ("age", "sex", 2023)]]
    print(p23.to_string(index=False))


if __name__ == "__main__":
    main()
