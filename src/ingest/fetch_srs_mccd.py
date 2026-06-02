"""Extract suicide data from SRS Cause-of-Death reports + MCCD reports.

SRS (Sample Registration System) reports give:
  - % share of all-cause deaths attributable to suicide (national)
  - Sex-stratified (Tables 1.3A Person / 1.3B Male / 1.3C Female)
  - Age-band breakdown (0-4, 5-14, 15-29, 30-44, 45-54, 55-69, 70+)
  - SRS is population-based with rural representation (the key strength
    over NCRB police-FIR and over modelled GBD/WHO estimates).

MCCD (Medical Certification of Cause of Death) reports give:
  - State-wise medically-certified deaths
  - Cause-grouped under ICD-10 Chapter XX (External causes) with
    X60-X84 = Intentional self-harm rows
  - Urban-biased (only medically certified deaths captured)
"""
from __future__ import annotations
import re
import pandas as pd
import pdfplumber
import logging
from pathlib import Path
from src import config

logging.getLogger("pdfminer").setLevel(logging.ERROR)

SRS_DIR = config.DATA_RAW / "SRS Cause of death"

# Map SRS report -> midpoint year (rolling 3-year window)
SRS_REPORTS = {
    "SRS_COD-STATISTICS_2014-2016.pdf": (2014, 2016, 2015),
    "SRS_COD-STATISTICS_2016-2018.pdf": (2016, 2018, 2017),
    "SRS_COD-STATISTICS_2017-2019.pdf": (2017, 2019, 2018),
    "SRS_COD-STATISTICS_2018-2020.pdf": (2018, 2020, 2019),
    "SRS_COD-STATISTICS_2019-2021.pdf": (2019, 2021, 2020),
    "SRS_COD-STATISTICS_2020-2022.pdf": (2020, 2022, 2021),
    "SRS_COD-STATISTICS_2021-2023.pdf": (2021, 2023, 2022),
    "SRS_COD-STATISTICS_2022-2024.pdf": (2022, 2024, 2023),
}

# Column headers in Table 1.3: Person/Male/Female | 0-4 | 5-14 | 15-29 | 30-44 | 45-54 | 55-69 | 70+
AGE_COLS = ["overall_pct", "age_0_4_pct", "age_5_14_pct", "age_15_29_pct",
            "age_30_44_pct", "age_45_54_pct", "age_55_69_pct", "age_70p_pct"]


def parse_srs_suicide_row(text: str) -> dict | None:
    """Parse 'Intentional injuries: Suicide' row -> dict of 8 percentages."""
    for line in text.splitlines():
        if "Intentional injuries: Suicide" in line:
            nums = re.findall(r"\b\d+\.\d+\b", line)
            if len(nums) >= 8:
                return dict(zip(AGE_COLS, [float(n) for n in nums[:8]]))
    return None


def extract_srs() -> pd.DataFrame:
    rows = []
    for fname, (yr_start, yr_end, yr_mid) in SRS_REPORTS.items():
        p = SRS_DIR / fname
        if not p.exists():
            print(f"  missing {fname}"); continue
        with pdfplumber.open(p) as pdf:
            tables = {"Person": None, "Male": None, "Female": None}
            for i in range(min(30, len(pdf.pages))):
                try:
                    txt = pdf.pages[i].extract_text() or ""
                except Exception:
                    continue
                if "Table 1.3A" in txt and "Person" in txt:
                    tables["Person"] = parse_srs_suicide_row(txt)
                if "Table 1.3B" in txt and "Male" in txt:
                    tables["Male"] = parse_srs_suicide_row(txt)
                if "Table 1.3C" in txt and ("Female" in txt or "Femal" in txt):
                    tables["Female"] = parse_srs_suicide_row(txt)
            for sex, d in tables.items():
                if d is None: continue
                rec = {"window_start": yr_start, "window_end": yr_end,
                       "midpoint_year": yr_mid, "sex": sex, **d}
                rows.append(rec)
                print(f"  {fname[:40]:40s} {sex:6s}: overall={d['overall_pct']}%, 15-29={d['age_15_29_pct']}%")
    return pd.DataFrame(rows)


def extract_mccd() -> pd.DataFrame:
    """MCCD: search for 'Intentional self-harm' or 'X60-X84' row at national + state level."""
    rows = []
    for fname in sorted(SRS_DIR.glob("Annual_Report_on_MCCD_*.pdf")):
        year = int(re.search(r"(\d{4})", fname.name).group(1))
        with pdfplumber.open(fname) as pdf:
            for i in range(len(pdf.pages)):
                try:
                    txt = pdf.pages[i].extract_text() or ""
                except Exception:
                    continue
                low = txt.lower()
                if "self-harm" in low or "self harm" in low or "x60" in low:
                    # Look for lines with "Intentional self-harm" + a count
                    for line in txt.splitlines():
                        if "intentional self-harm" in line.lower() or "intentional self harm" in line.lower():
                            nums = re.findall(r"\d[\d,]*\.?\d*", line.replace(",", ""))
                            big = [int(n) for n in nums if n.isdigit() and 1000 < int(n) < 500000]
                            if big:
                                rows.append({"year": year, "page": i+1, "raw_line": line.strip()[:200],
                                             "candidate_counts": big})
                                print(f"  MCCD {year} p{i+1}: {line.strip()[:120]}")
                                break  # one hit per page
                    if rows and rows[-1].get("year") == year:
                        break  # one hit per file
    return pd.DataFrame(rows)


def main() -> None:
    print("\n=== SRS extraction ===")
    srs = extract_srs()
    out_srs = config.DATA_PROCESSED / "srs_suicide_2014_2024.csv"
    srs.to_csv(out_srs, index=False)
    print(f"\n  wrote {out_srs} ({len(srs)} rows)")

    print("\n=== MCCD extraction ===")
    mccd = extract_mccd()
    out_mccd = config.DATA_PROCESSED / "mccd_suicide_2016_2023.csv"
    mccd.to_csv(out_mccd, index=False)
    print(f"  wrote {out_mccd} ({len(mccd)} rows)")


if __name__ == "__main__":
    main()
