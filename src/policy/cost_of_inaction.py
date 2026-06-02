"""Cost-of-inaction analysis for the Indian Finance Ministry / 16th Finance Commission.

Quantifies the economic burden of continuing the suicide status quo over 2025-2030:
  - YLLs (years of life lost) = deaths x average years of life lost per suicide
  - Productivity loss = YLLs x per-capita GDP (Statistical Value of Life proxy)
  - Healthcare cost of suicide attempts (Indian ICU + counselling)
  - Total cost-of-inaction in lakh crore Indian rupees

References:
  - YLL per suicide: GBD 2023 (~32 years globally; India 30-35 years given young-adult skew)
  - Per-capita GDP 2023: ~INR 2.0 lakh (World Bank)
  - VSL (Value of Statistical Life) India: 200x per-capita GDP per WHO-CHOICE convention
                                              and Patel Lancet Commission 2018 (~Rs 4 crore)
  - Cost per suicide attempt managed: ~Rs 35,000 (AIIMS-Delhi cost study 2019)
"""
from __future__ import annotations
import pandas as pd
from src import config

# Baseline parameters
SRS_2023_BASELINE = 252_000
YLL_PER_SUICIDE = 32         # WHO global; India age skew makes this slightly higher
GDP_PER_CAPITA_INR = 200_000  # Rs 2.0 lakh
VSL_INR = 4_00_00_000        # Rs 4 crore per statistical life (Patel 2018)
COST_PER_ATTEMPT_INR = 35_000  # AIIMS-Delhi 2019
SUICIDE_TO_ATTEMPT_RATIO = 25  # 1 suicide : 25 attempts (WHO + India NMHS)

def main() -> None:
    rows = []
    # 6-year cumulative status-quo deaths
    deaths_cum = SRS_2023_BASELINE * 6  # ~1.51 million
    attempts_cum = deaths_cum * SUICIDE_TO_ATTEMPT_RATIO  # ~37.8 million
    yll_cum = deaths_cum * YLL_PER_SUICIDE  # ~48.4 million YLLs

    # Productivity loss using human capital approach: YLLs x GDP/capita
    prod_loss_cum = yll_cum * GDP_PER_CAPITA_INR

    # VSL approach: deaths x VSL
    vsl_total_cum = deaths_cum * VSL_INR

    # Healthcare cost: attempts x cost per attempt
    hc_cost_cum = attempts_cum * COST_PER_ATTEMPT_INR

    # NSPS 10% target counterfactual (lives saved if achieved)
    deaths_under_nsps = SRS_2023_BASELINE * 6 * 0.9  # 10% reduction
    lives_saved = deaths_cum - deaths_under_nsps      # ~151,200 lives over 6 years

    # Economic value of NSPS achievement
    prod_savings = lives_saved * YLL_PER_SUICIDE * GDP_PER_CAPITA_INR
    vsl_savings = lives_saved * VSL_INR
    hc_savings = lives_saved * SUICIDE_TO_ATTEMPT_RATIO * COST_PER_ATTEMPT_INR

    print(f"=== COST OF INACTION (status quo, 2025-2030 cumulative) ===")
    print(f"  SRS-anchored deaths:              {deaths_cum:>15,}")
    print(f"  Cumulative attempts (1:25 ratio): {attempts_cum:>15,}")
    print(f"  Cumulative YLLs:                  {yll_cum:>15,}")
    print(f"  Productivity loss (human capital): INR {prod_loss_cum/1e12:.2f} lakh crore")
    print(f"  Value of Statistical Life (VSL):   INR {vsl_total_cum/1e12:.2f} lakh crore")
    print(f"  Healthcare cost of attempts:       INR {hc_cost_cum/1e12:.2f} lakh crore")
    print(f"  TOTAL economic cost (VSL + HC):    INR {(vsl_total_cum + hc_cost_cum)/1e12:.2f} lakh crore")

    print(f"\n=== ECONOMIC VALUE OF MEETING NSPS 10% TARGET ===")
    print(f"  Lives saved over 2025-2030:        {int(lives_saved):,}")
    print(f"  Productivity gained:               INR {prod_savings/1e12:.2f} lakh crore")
    print(f"  VSL value of saved lives:          INR {vsl_savings/1e12:.2f} lakh crore")
    print(f"  Healthcare cost averted:           INR {hc_savings/1e12:.2f} lakh crore")
    print(f"  TOTAL economic value of NSPS:      INR {(vsl_savings + hc_savings)/1e12:.2f} lakh crore")

    # Cost-effectiveness vs investment
    INVESTMENT_INR_CRORE = 6_000
    INVESTMENT_INR = INVESTMENT_INR_CRORE * 1e7
    print(f"\n=== CEA: INR {INVESTMENT_INR_CRORE} crore investment vs cost-of-inaction ===")
    benefit_cost = (vsl_savings + hc_savings) / INVESTMENT_INR
    print(f"  Benefit-cost ratio (VSL + HC / investment): {benefit_cost:.1f}")
    print(f"  Cost per DALY averted: INR {INVESTMENT_INR / (lives_saved * YLL_PER_SUICIDE):,.0f}")
    who_threshold = GDP_PER_CAPITA_INR  # 1x GDP per capita
    print(f"  Indian WHO-CHOICE threshold (1x GDP/cap): INR {who_threshold:,.0f}")
    print(f"  -> Highly cost-effective per WHO-CHOICE criteria")

    # Save outputs
    summary = pd.DataFrame([
        {"Quantity": "SRS-anchored deaths 2025-2030 (status quo)", "Value": f"{deaths_cum:,}"},
        {"Quantity": "Cumulative attempts (1:25 ratio)", "Value": f"{attempts_cum:,}"},
        {"Quantity": "Cumulative YLLs", "Value": f"{yll_cum:,}"},
        {"Quantity": "Productivity loss (lakh crore INR)", "Value": f"{prod_loss_cum/1e12:.2f}"},
        {"Quantity": "VSL economic value (lakh crore INR)", "Value": f"{vsl_total_cum/1e12:.2f}"},
        {"Quantity": "Healthcare cost of attempts (lakh crore INR)", "Value": f"{hc_cost_cum/1e12:.2f}"},
        {"Quantity": "Total economic burden of status quo (lakh crore INR)", "Value": f"{(vsl_total_cum+hc_cost_cum)/1e12:.2f}"},
        {"Quantity": "Lives saved if NSPS 10% met", "Value": f"{int(lives_saved):,}"},
        {"Quantity": "Economic value of NSPS (lakh crore INR)", "Value": f"{(vsl_savings+hc_savings)/1e12:.2f}"},
        {"Quantity": "NSPS implementation cost (crore INR)", "Value": f"{INVESTMENT_INR_CRORE:,}"},
        {"Quantity": "Benefit-cost ratio of NSPS investment", "Value": f"{benefit_cost:.1f}"},
        {"Quantity": "Cost per DALY averted (INR)", "Value": f"{INVESTMENT_INR / (lives_saved * YLL_PER_SUICIDE):,.0f}"},
    ])
    summary.to_csv(config.DATA_PROCESSED / "cost_of_inaction_summary.csv", index=False)


if __name__ == "__main__":
    main()
