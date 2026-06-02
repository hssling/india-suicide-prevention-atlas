"""State-specific projected populations 2018-2024.

Methodology: Census 2011 actual + decadal growth rate (Census 2001->2011)
projected forward annually. Decadal growth from Office of the Registrar
General of India (RGI). For states formed after 2011 (Telangana 2014,
Ladakh 2019), uses parent-state growth and the post-formation split.

This is the standard mid-year-projected-population convention NCRB uses
for its Figure 2.4 rate calculations.
"""
from __future__ import annotations
import pandas as pd
from src import config

# Census 2011 actual + decadal % growth Census 2001->2011 (RGI Census of India)
# Columns: state, pop_2011_M (millions), decadal_growth_2001_2011 (%)
CENSUS_AND_GROWTH = {
    "Andhra Pradesh": (49.58, 11.0),     # post-Telangana bifurcation; pre-bifurcation was 84.58M
    "Arunachal Pradesh": (1.38, 25.9),
    "Assam": (31.21, 16.9),
    "Bihar": (104.10, 25.1),
    "Chhattisgarh": (25.55, 22.6),
    "Goa": (1.46, 8.2),
    "Gujarat": (60.44, 19.3),
    "Haryana": (25.35, 19.9),
    "Himachal Pradesh": (6.86, 12.8),
    "Jammu & Kashmir": (12.27, 23.6),     # pre-2019 J&K; post-2019 split into J&K UT + Ladakh
    "Jharkhand": (32.99, 22.4),
    "Karnataka": (61.10, 15.7),
    "Kerala": (33.41, 4.9),
    "Madhya Pradesh": (72.63, 20.3),
    "Maharashtra": (112.37, 15.99),
    "Manipur": (2.86, 24.5),
    "Meghalaya": (2.97, 27.95),
    "Mizoram": (1.10, 23.5),
    "Nagaland": (1.98, -0.6),
    "Odisha": (41.97, 14.0),
    "Punjab": (27.74, 13.9),
    "Rajasthan": (68.55, 21.4),
    "Sikkim": (0.61, 12.9),
    "Tamil Nadu": (72.15, 15.6),
    "Telangana": (35.20, 11.0),           # bifurcated from AP 2014; uses pre-bifurcation growth
    "Tripura": (3.67, 14.8),
    "Uttar Pradesh": (199.81, 20.2),
    "Uttarakhand": (10.09, 19.2),
    "West Bengal": (91.28, 13.8),
    "Andaman & Nicobar Islands": (0.38, 6.9),
    "Chandigarh": (1.05, 17.2),
    "Dadra & Nagar Haveli and Daman & Diu": (0.59, 53.5),  # post-2020 merged UT
    "Delhi": (16.79, 21.2),
    "Lakshadweep": (0.06, 6.3),
    "Puducherry": (1.25, 28.1),
    "Ladakh": (0.27, 23.6),               # carved out of J&K 2019; uses J&K decadal growth
}


def project_population(state: str, year: int) -> float | None:
    """Return projected population in millions for the given state and year (2011-2030)."""
    if state not in CENSUS_AND_GROWTH:
        return None
    pop_2011, decadal_growth = CENSUS_AND_GROWTH[state]
    annual_growth = (1 + decadal_growth / 100) ** (1 / 10) - 1
    years_elapsed = year - 2011
    return pop_2011 * (1 + annual_growth) ** years_elapsed


def build_state_year_population_panel() -> pd.DataFrame:
    """Build a state x year (2018-2024) population panel."""
    rows = []
    for state in CENSUS_AND_GROWTH:
        for year in range(2018, 2025):
            rows.append({
                "state": state,
                "year": year,
                "population_m": round(project_population(state, year), 3),
            })
    df = pd.DataFrame(rows)
    out_path = config.DATA_PROCESSED / "state_populations_2018_2024.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def main() -> None:
    df = build_state_year_population_panel()
    print(f"[populations] wrote state-year panel: {len(df)} rows for {df.state.nunique()} states x {df.year.nunique()} years")
    # National totals
    nat = df.groupby("year").population_m.sum().reset_index()
    nat["population_total_M"] = nat["population_m"].round(1)
    print(f"\nNational population totals (millions, projected from Census 2011):")
    print(nat[["year", "population_total_M"]].to_string(index=False))
    # Spot-checks
    print(f"\nSpot-checks (millions in 2023):")
    for state in ["Uttar Pradesh", "Maharashtra", "Bihar", "Tamil Nadu", "Kerala", "Telangana"]:
        p = project_population(state, 2023)
        print(f"  {state:20s}: {p:.1f} M")


if __name__ == "__main__":
    main()
