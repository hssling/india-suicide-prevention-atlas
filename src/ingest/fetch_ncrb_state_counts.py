"""Extract absolute state suicide counts from NCRB ADSI Chapter 2 Figure 2.2 maps.

Each year's Chapter 2 PDF contains a state-map figure (Figure 2.2 in 2024,
similar elsewhere) where state names appear with their absolute suicide counts
laid out by geography. This is a more reliable extraction target than the
Table 2.2 multi-column tables that pdfplumber struggles with.

Output: data/processed/ncrb_state_counts_2018_2024.csv
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
CHAPTER2_PDFS = {
    2018: "1695990798chapter-2-suicides-2018.pdf",
    2019: "chapter-2-suicides-2019.pdf",
    2020: "169599053803ADSI-2020Chapter-2Suicides-ExcludingFarmerCauses.pdf",
    2021: "169599016203ADSI-2021Chapter-2Suicides-ExcludingFarmerCauses.pdf",
    2022: "170161093707Chapter-2Suicides.pdf",
    2023: "08ADSI-2023Chapter-2Suicides-ExcludingFarmerCauses.pdf",
    2024: "8ADSI-2024Chapter-2Suicides-ExcludingFarmerCauses.pdf",
}

# Canonical state-name regex (multiword + casing variants)
STATE_PATTERNS = {
    "Andhra Pradesh": r"Andhra\s+Pradesh",
    "Arunachal Pradesh": r"Arunachal\s+Pradesh",
    "Assam": r"\bAssam\b",
    "Bihar": r"\bBihar\b",
    "Chhattisgarh": r"Chhatt?isgarh",
    "Goa": r"\bGoa\b",
    "Gujarat": r"\bGujarat\b",
    "Haryana": r"\bHaryana\b",
    "Himachal Pradesh": r"Himachal\s+Pradesh",
    "Jammu & Kashmir": r"Jammu\s*[&@]?\s*Kashmir",
    "Jharkhand": r"\bJharkhand\b",
    "Karnataka": r"\bKarnataka\b",
    "Kerala": r"\bKerala\b",
    "Madhya Pradesh": r"Madhya\s+Pradesh",
    "Maharashtra": r"\bMaharashtra\b",
    "Manipur": r"\bManipur\b",
    "Meghalaya": r"\bMeghalaya\b",
    "Mizoram": r"\bMizoram\b",
    "Nagaland": r"\bNagaland\b",
    "Odisha": r"\bOdisha\b",
    "Punjab": r"\bPunjab\b",
    "Rajasthan": r"\bRajasthan\b",
    "Sikkim": r"\bSikkim\b",
    "Tamil Nadu": r"Tamil\s+Nadu",
    "Telangana": r"\bTelangana\b",
    "Tripura": r"\bTripura\b",
    "Uttar Pradesh": r"Uttar\s+Pradesh",
    "Uttarakhand": r"\bUttarakhand\b",
    "West Bengal": r"West\s+Bengal",
    "Andaman & Nicobar Islands": r"Andaman[\s\w]*Nicobar|A\s*[&@]\s*N\s*Island",
    "Chandigarh": r"\bChandigarh\b",
    "Dadra & Nagar Haveli and Daman & Diu": r"D\s*[&@]\s*N\s*Haveli|Dadra[\s\w]*Daman",
    "Delhi": r"\bDelhi\b",
    "Lakshadweep": r"\bLakshadweep\b",
    "Puducherry": r"\bPuducherry\b",
    "Ladakh": r"\bLadakh\b",
}


def extract_state_counts_from_map_page(text: str) -> dict[str, int]:
    """Parse a state-map page where state names appear next to their counts.

    Strategy: find each state name; capture the nearest number after it.
    """
    counts: dict[str, int] = {}
    # Each state name pattern; find first occurrence; capture next number within 60 chars
    for canon, pat in STATE_PATTERNS.items():
        # Match state name + up to 60 chars of intervening text + integer (1-7 digits, with optional commas/spaces between digits)
        m = re.search(pat + r"[\s\S]{0,60}?([\d][\d,\s]{0,8})\b", text, re.IGNORECASE)
        if m and m.group(1):
            num_str = re.sub(r"[,\s]", "", m.group(1))
            try:
                n = int(num_str)
                # Sanity: 0..50000 plausible for state suicides annually
                if 0 <= n <= 60000:
                    counts[canon] = n
            except ValueError:
                continue
    return counts


def extract_state_counts(pdf_path: Path, year: int) -> dict[str, int]:
    """Search Chapter 2 PDF pages 1-8 for the state-map figure (Figure 2.2 = counts).

    Prefer the page where extracted numbers are LARGEST (counts page beats
    rates page; rates are 0-50, counts are 0-25,000).
    """
    best: dict[str, int] = {}
    best_score = -1.0
    with pdfplumber.open(pdf_path) as pdf:
        for i in range(min(10, len(pdf.pages))):
            try:
                txt = pdf.pages[i].extract_text() or ""
            except Exception:
                continue
            states_present = sum(1 for pat in STATE_PATTERNS.values()
                                 if re.search(pat, txt, re.IGNORECASE))
            if states_present < 15:
                continue
            cand = extract_state_counts_from_map_page(txt)
            if not cand:
                continue
            # Score: number of states found, weighted by median value
            # (count page has median > 1000; rate page has median < 30)
            import statistics
            med = statistics.median(cand.values())
            score = len(cand) * (med + 1)
            if score > best_score:
                best_score = score
                best = cand
    return best


def main() -> None:
    rows = []
    for year, fname in CHAPTER2_PDFS.items():
        pdf_path = NCRB_DIR / fname
        if not pdf_path.exists():
            print(f"WARNING: {pdf_path} not found")
            continue
        print(f"[ncrb_counts] {year}: {fname}")
        counts = extract_state_counts(pdf_path, year)
        print(f"  extracted absolute counts for {len(counts)} states; total={sum(counts.values()):,}")
        for st, n in counts.items():
            rows.append({"year": year, "state": st, "ncrb_suicides": n})
    if not rows:
        print("NO DATA")
        return
    df = pd.DataFrame(rows)
    out_path = config.DATA_PROCESSED / "ncrb_state_counts_2018_2024.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[ncrb_counts] wrote {out_path}  rows={len(df)}")
    pivot = df.pivot_table(index="state", columns="year", values="ncrb_suicides", aggfunc="first")
    print(f"\nCoverage matrix:")
    print(pivot.to_string())
    print(f"\nNational totals (sum of extracted):")
    print(df.groupby("year").ncrb_suicides.agg(["sum", "count"]).to_string())


if __name__ == "__main__":
    main()
