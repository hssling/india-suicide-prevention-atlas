"""Extract state-level suicide rates from NCRB ADSI Chapter 2 PDFs, 2018-2024.

NCRB Figure 2.4 (or equivalent) in each year's Chapter 2 lists STATE  RATE_PER_100K
as a simple two-column figure. We regex-parse that for each year.

We also regex the narrative for the national total (Indian comma style: 1,71,418).

Output:
  data/processed/ncrb_state_rates_2018_2024.csv  - state-year rates
  data/processed/ncrb_national_totals_2018_2024.csv  - year totals (where parseable)
"""
from __future__ import annotations
import re
import pandas as pd
import pdfplumber
import logging
from pathlib import Path
from src import config

logging.getLogger("pdfminer").setLevel(logging.ERROR)

NCRB_DIR = config.EXISTING_NCRB_DIR

# Map of year -> Chapter 2 PDF filename
CHAPTER2_PDFS = {
    2018: "1695990798chapter-2-suicides-2018.pdf",
    2019: "chapter-2-suicides-2019.pdf",
    2020: "169599053803ADSI-2020Chapter-2Suicides-ExcludingFarmerCauses.pdf",
    2021: "169599016203ADSI-2021Chapter-2Suicides-ExcludingFarmerCauses.pdf",
    2022: "170161093707Chapter-2Suicides.pdf",
    2023: "08ADSI-2023Chapter-2Suicides-ExcludingFarmerCauses.pdf",
    2024: "8ADSI-2024Chapter-2Suicides-ExcludingFarmerCauses.pdf",
}

# Canonical state names matching the workspace
CANONICAL = {
    "ANDHRA PRADESH": "Andhra Pradesh",
    "ARUNACHAL PRADESH": "Arunachal Pradesh",
    "ASSAM": "Assam",
    "BIHAR": "Bihar",
    "CHHATTISGARH": "Chhattisgarh",
    "GOA": "Goa",
    "GUJARAT": "Gujarat",
    "HARYANA": "Haryana",
    "HIMACHAL PRADESH": "Himachal Pradesh",
    "JAMMU & KASHMIR": "Jammu & Kashmir",
    "JAMMU AND KASHMIR": "Jammu & Kashmir",
    "JAMMU&KASHMIR": "Jammu & Kashmir",
    "J & K": "Jammu & Kashmir",
    "JHARKHAND": "Jharkhand",
    "KARNATAKA": "Karnataka",
    "KERALA": "Kerala",
    "MADHYA PRADESH": "Madhya Pradesh",
    "MAHARASHTRA": "Maharashtra",
    "MANIPUR": "Manipur",
    "MEGHALAYA": "Meghalaya",
    "MIZORAM": "Mizoram",
    "NAGALAND": "Nagaland",
    "ODISHA": "Odisha",
    "PUNJAB": "Punjab",
    "RAJASTHAN": "Rajasthan",
    "SIKKIM": "Sikkim",
    "TAMIL NADU": "Tamil Nadu",
    "TELANGANA": "Telangana",
    "TRIPURA": "Tripura",
    "UTTAR PRADESH": "Uttar Pradesh",
    "UTTARAKHAND": "Uttarakhand",
    "WEST BENGAL": "West Bengal",
    "A & N ISLANDS": "Andaman & Nicobar Islands",
    "A&N ISLANDS": "Andaman & Nicobar Islands",
    "ANDAMAN & NICOBAR ISLANDS": "Andaman & Nicobar Islands",
    "ANDAMAN AND NICOBAR ISLANDS": "Andaman & Nicobar Islands",
    "CHANDIGARH": "Chandigarh",
    "D & N HAVELI AND DAMAN DIU": "Dadra & Nagar Haveli and Daman & Diu",
    "D & N HAVELI AND DAMAN & DIU": "Dadra & Nagar Haveli and Daman & Diu",
    "DADRA & NAGAR HAVELI": "Dadra & Nagar Haveli and Daman & Diu",
    "DAMAN & DIU": "Dadra & Nagar Haveli and Daman & Diu",
    "DELHI": "Delhi",
    "DELHI UT": "Delhi",
    "DELHI (UT)": "Delhi",
    "LADAKH": "Ladakh",
    "LAKSHADWEEP": "Lakshadweep",
    "PUDUCHERRY": "Puducherry",
}


def _normalise(s: str) -> str | None:
    """Map any NCRB-formatted state token to its canonical name (or None if unknown)."""
    s = (s or "").strip().upper()
    # Strip "UT" suffix etc.
    s = re.sub(r"\s+UT$", "", s)
    s = re.sub(r"\s+\(UT\)$", "", s)
    if s in CANONICAL:
        return CANONICAL[s]
    # Try with hyphens/spaces normalised
    s2 = re.sub(r"\s+", " ", s).strip()
    return CANONICAL.get(s2)


def extract_rates_from_text(text: str) -> dict[str, float]:
    """Regex out 'STATE NAME  RATE' lines from a page's text.

    NCRB Figure 2.4-style format: STATE in uppercase followed by a rate like 49.6.
    """
    rates: dict[str, float] = {}
    for line in text.splitlines():
        # Pattern: 1+ uppercase words possibly with & or AND, then a float
        m = re.match(r"^([A-Z][A-Z\s&\.\-]+?)\s+(\d+\.\d+)\s*$", line.strip())
        if m:
            state_token, rate_str = m.group(1).strip(), m.group(2)
            canon = _normalise(state_token)
            if canon:
                try:
                    rate = float(rate_str)
                    # Sanity-check: suicide rates 0-80/100k plausible
                    if 0 < rate < 80:
                        rates[canon] = rate
                except ValueError:
                    pass
    return rates


def extract_national_total_from_text(text: str) -> int | None:
    """Find national suicide total in the narrative."""
    # Indian comma style: 1,71,418 or 1,53,052; or international 171,418
    patterns = [
        r"(\d{1,2}[,\s]\d{2}[,\s]\d{3})\s+(?:total\s+)?suicides?",
        r"total\s+(?:of\s+)?(\d{1,2}[,\s]\d{2}[,\s]\d{3})\s+suicid",
        r"(\d{6})\s+suicides?",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            num_str = re.sub(r"[,\s]", "", m.group(1))
            try:
                n = int(num_str)
                if 100000 < n < 250000:  # Indian annual suicide range
                    return n
            except ValueError:
                continue
    return None


def process_one(pdf_path: Path, year: int) -> tuple[pd.DataFrame, int | None]:
    rates_all = {}
    national_total = None
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(len(pdf.pages)):
            try:
                txt = pdf.pages[i].extract_text() or ""
            except Exception:
                continue
            # Try to grab any state rates
            page_rates = extract_rates_from_text(txt)
            for st, r in page_rates.items():
                # Keep first non-None reading
                if st not in rates_all:
                    rates_all[st] = r
            # Try national total
            if national_total is None:
                nt = extract_national_total_from_text(txt)
                if nt:
                    national_total = nt
    df = pd.DataFrame([{"year": year, "state": st, "ncrb_rate_per_100k": r}
                       for st, r in rates_all.items()])
    return df, national_total


def main() -> None:
    rate_frames = []
    nat_rows = []
    for year, fname in CHAPTER2_PDFS.items():
        pdf_path = NCRB_DIR / fname
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found")
            continue
        print(f"[ncrb] processing {year}: {fname}")
        df, nat = process_one(pdf_path, year)
        print(f"  rates extracted for {len(df)} states; national total: {nat}")
        rate_frames.append(df)
        nat_rows.append({"year": year, "national_total": nat})
    rates_all = pd.concat(rate_frames, ignore_index=True) if rate_frames else pd.DataFrame()
    nat_all = pd.DataFrame(nat_rows)
    rates_path = config.DATA_PROCESSED / "ncrb_state_rates_2018_2024.csv"
    nat_path = config.DATA_PROCESSED / "ncrb_national_totals_2018_2024.csv"
    rates_all.to_csv(rates_path, index=False)
    nat_all.to_csv(nat_path, index=False)
    print(f"\n[ncrb] wrote {rates_path}  rows={len(rates_all)}")
    print(f"[ncrb] wrote {nat_path}  rows={len(nat_all)}")
    print(f"\nState-year coverage matrix:")
    if not rates_all.empty:
        pivot = rates_all.pivot_table(index="state", columns="year", values="ncrb_rate_per_100k", aggfunc="first")
        print(pivot.to_string())
    print(f"\nNational totals by year:")
    print(nat_all.to_string(index=False))


if __name__ == "__main__":
    main()
