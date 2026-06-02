"""District-level NFHS-5 SES composite + state-level suicide rate gradient.

Strategy:
  1. Build per-district SES composite from 8 NFHS-5 indicators (PCA first PC):
       electricity, water, sanitation, clean fuel, iodized salt, insurance,
       women literacy, women >= 10 years schooling
  2. Aggregate to state-level: mean SES + % districts in lowest national quintile (Q1)
  3. Link to state NCRB suicide rate (2023) + state Bayesian theta
  4. Compute SES gradient: state suicide rate ~ state SES + rank correlation
  5. Approximate SII / RII / Concentration Index at state level using SES rank
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, linregress
from src import config

NFHS_CSV = config.ROOT.parent / "data_raw/NFHS_5_India_Districts_Factsheet_Data(Sheet1).csv"

SES_INDICATORS = [
    "Population living in households with electricity (%)",
    "Population living in households with an improved drinking-water source1 (%)",
    "Population living in households that use an improved sanitation facility2 (%)",
    "Households using clean fuel for cooking3 (%)",
    "Households using iodized salt (%)",
    "Households with any usual member covered under a health insurance/financing scheme (%)",
    "Women (age 15-49) who are literate4 (%)",
    "Women (age 15-49)  with 10 or more years of schooling (%)",
]


def main() -> None:
    df = pd.read_csv(NFHS_CSV, low_memory=False)
    print(f"NFHS-5 district factsheet: {len(df)} districts, {len(df.columns)} columns")

    # Clean: convert to numeric, coerce missing
    nfhs = df[["District Names", "State/UT"] + SES_INDICATORS].copy()
    for c in SES_INDICATORS:
        nfhs[c] = pd.to_numeric(nfhs[c], errors="coerce")
    nfhs = nfhs.dropna(subset=SES_INDICATORS, how="all")
    print(f"After dropna: {len(nfhs)} districts with SES data")

    # Impute remaining missing with column median (NFHS districts vary)
    nfhs[SES_INDICATORS] = nfhs[SES_INDICATORS].fillna(nfhs[SES_INDICATORS].median())

    # PCA composite (higher PC1 score = higher SES)
    X = StandardScaler().fit_transform(nfhs[SES_INDICATORS].values)
    pca = PCA(n_components=1)
    pc1 = pca.fit_transform(X).ravel()
    # Make sure direction is high=affluent: positively correlated with women schooling
    if np.corrcoef(pc1, nfhs[SES_INDICATORS[-1]])[0, 1] < 0:
        pc1 = -pc1
    nfhs["ses_index"] = pc1
    nfhs["ses_quintile_natl"] = pd.qcut(pc1, 5, labels=["Q1_poorest","Q2","Q3","Q4","Q5_richest"])
    print(f"PCA: variance explained by PC1 = {pca.explained_variance_ratio_[0]*100:.1f}%")
    print(f"PC1 loadings: {dict(zip(SES_INDICATORS, pca.components_[0].round(3)))}")

    # State aggregation
    by_state = nfhs.groupby("State/UT").agg(
        n_districts=("District Names", "count"),
        mean_ses=("ses_index", "mean"),
        pct_in_q1_poorest=("ses_quintile_natl", lambda s: (s == "Q1_poorest").mean() * 100),
        pct_in_q5_richest=("ses_quintile_natl", lambda s: (s == "Q5_richest").mean() * 100),
    ).reset_index().rename(columns={"State/UT": "state"})

    # Harmonise state names to match workspace canonical
    NORM = {
        "Andaman & Nicobar Islands": "Andaman & Nicobar Islands",
        "Dadra & Nagar Haveli": "Dadra & Nagar Haveli and Daman & Diu",
        "Daman & Diu": "Dadra & Nagar Haveli and Daman & Diu",
        "Jammu & Kashmir": "Jammu & Kashmir",
        "NCT of Delhi": "Delhi",
    }
    by_state["state"] = by_state["state"].replace(NORM)
    by_state = by_state.groupby("state").agg(
        n_districts=("n_districts", "sum"),
        mean_ses=("mean_ses", "mean"),
        pct_in_q1_poorest=("pct_in_q1_poorest", "mean"),
        pct_in_q5_richest=("pct_in_q5_richest", "mean"),
    ).reset_index()

    # NCRB state rates 2023
    rates = pd.read_csv(config.DATA_PROCESSED / "ncrb_state_rates_2018_2024.csv")
    r23 = rates[rates["year"] == 2023][["state", "ncrb_rate_per_100k"]]
    merged = by_state.merge(r23, on="state", how="inner")
    print(f"\nState-level SES x suicide-rate panel: {len(merged)} states")

    # Spearman rank correlation: SES vs suicide rate
    rho, p_rho = spearmanr(merged["mean_ses"], merged["ncrb_rate_per_100k"])
    print(f"\nSpearman rho (state mean SES vs NCRB 2023 suicide rate): {rho:+.3f} (p={p_rho:.4f})")

    # Linear regression: suicide rate ~ SES
    slope, intercept, r_lin, p_lin, _ = linregress(merged["mean_ses"], merged["ncrb_rate_per_100k"])
    print(f"Linear: suicide_rate = {intercept:.2f} + {slope:.2f} * SES   (R^2={r_lin**2:.3f}, p={p_lin:.4f})")

    # SII (Slope Index of Inequality) at state level: regress rate on SES rank fraction
    merged = merged.sort_values("mean_ses").reset_index(drop=True)
    n = len(merged)
    merged["ses_rank_frac"] = (merged.index + 0.5) / n   # 0-1 with midpoint adjustment
    sii_slope, sii_intercept, _, sii_p, _ = linregress(merged["ses_rank_frac"], merged["ncrb_rate_per_100k"])
    sii = sii_slope  # rate change from Q1 (rank=0) to Q5 (rank=1)
    rii = (sii_intercept + sii_slope) / sii_intercept if sii_intercept > 0 else float("nan")
    print(f"\nState-level SII (NFHS SES rank): {sii:+.2f} per 100k (p={sii_p:.4f})")
    print(f"State-level RII (Q5/Q1 ratio): {rii:.2f}")

    merged.to_csv(config.DATA_PROCESSED / "state_ses_suicide_2023.csv", index=False)
    print(f"\n[ses] wrote state_ses_suicide_2023.csv")
    print(f"\nFull state panel sorted by SES (poorest -> richest):")
    print(merged[["state","mean_ses","pct_in_q1_poorest","ncrb_rate_per_100k"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
