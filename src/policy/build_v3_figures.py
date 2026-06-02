"""Build 5 policy-focused figures for v3 manuscript."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import numpy as np
from pathlib import Path
from src import config

OUT = Path("submission_final/IJMR_v3/figures")
OUT.mkdir(parents=True, exist_ok=True)


def fig1_nsps_trajectory():
    scen = pd.read_csv(config.DATA_PROCESSED / "nsps_trajectory_state_scenarios.csv")
    totals = {
        "S0 Status quo":      scen.s0_status_quo_2030_count.sum(),
        "S1 DMHP universal":  scen.s1_dmhp_2030_count.sum(),
        "S2 Tele-MANAS full": scen.s2_telemanas_2030_count.sum(),
        "S3 Pesticide ban":   scen.s3_pesticide_2030_count.sum(),
        "S4 Integrated NSPS": scen.s4_integrated_2030_count.sum(),
    }
    nsps_target = int(totals["S0 Status quo"] * 0.9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    labels = list(totals.keys()); vals = list(totals.values())
    colors = ["#444", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
    bars = ax1.bar(labels, vals, color=colors, alpha=0.85)
    ax1.axhline(nsps_target, color="black", linestyle="--", linewidth=1.5,
                label="NSPS 10% target ({:,})".format(nsps_target))
    ax1.set_ylabel("Projected 2030 national suicide count")
    ax1.set_title("(a) NSPS 2030 forecast under 5 policy scenarios")
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 5000,
                 f"{int(val):,}", ha="center", fontsize=9, fontweight="bold")
    ax1.legend(loc="upper right")
    ax1.tick_params(axis="x", rotation=20)
    ax1.grid(True, axis="y", alpha=0.3)
    top10 = scen.nlargest(10, "s4_vs_s0_lives_saved_2030")[["state", "s4_vs_s0_lives_saved_2030"]]
    ax2.barh(range(len(top10)), top10.s4_vs_s0_lives_saved_2030, color="#d62728", alpha=0.85)
    ax2.set_yticks(range(len(top10)))
    ax2.set_yticklabels(top10.state)
    ax2.invert_yaxis()
    ax2.set_xlabel("Lives saved in 2030 (S4 vs S0)")
    ax2.set_title("(b) Top 10 states by lives saved under Integrated NSPS")
    ax2.grid(True, axis="x", alpha=0.3)
    fig.suptitle("Figure 1. NSPS 2030 trajectory forecast under five policy scenarios",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_1_nsps_trajectory.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved Fig 1")


def fig2_paf():
    paf = pd.read_csv(config.DATA_PROCESSED / "paf_modifiable_risk_factors.csv")
    paf["paf_pct"] = paf["PAF"].str.extract(r"(\d+\.\d+)%")[0].astype(float)
    paf["paf_lo"] = paf["PAF"].str.extract(r"\(([\d\.]+)-").astype(float)
    paf["paf_hi"] = paf["PAF"].str.extract(r"-([\d\.]+)%\)").astype(float)
    paf = paf.sort_values("paf_pct", ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(paf))
    ax.barh(y, paf.paf_pct, color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"], alpha=0.85)
    ax.errorbar(paf.paf_pct, y, xerr=[paf.paf_pct - paf.paf_lo, paf.paf_hi - paf.paf_pct],
                fmt="none", color="black", capsize=4)
    ax.set_yticks(y)
    short = [r["Risk factor"].split("(")[0].strip()[:50] for _, r in paf.iterrows()]
    ax.set_yticklabels(short, fontsize=10)
    ax.set_xlabel("Population-Attributable Fraction (%)")
    ax.set_title("Figure 2. Population-Attributable Fractions for modifiable Indian suicide risk factors")
    ax.grid(True, axis="x", alpha=0.3)
    for i, r in paf.reset_index(drop=True).iterrows():
        ax.text(r.paf_pct + 1, i, r["Attributable deaths 2023"].split(" (")[0],
                va="center", fontsize=9, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_2_paf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved Fig 2")


def fig3_optimal_allocation():
    alloc = pd.read_csv(config.DATA_PROCESSED / "optimal_nsps_allocation.csv")
    intv_agg = alloc.groupby("intervention").agg(
        total_cost_crore=("cost_crore", "sum"),
        total_lives=("lives_cum", "sum"),
    ).reset_index()
    intv_agg["cost_per_life_lakh"] = (intv_agg.total_cost_crore * 100 / intv_agg.total_lives).round(1)
    intv_agg = intv_agg.sort_values("cost_per_life_lakh")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    bars = ax1.bar(intv_agg.intervention, intv_agg.cost_per_life_lakh,
                   color=["#2ca02c", "#1f77b4", "#d62728"], alpha=0.85)
    ax1.set_ylabel("Cost per life saved (INR lakh)")
    ax1.set_title("(a) Cost-effectiveness ladder (lower = more efficient)")
    for bar, val in zip(bars, intv_agg.cost_per_life_lakh):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.5,
                 "INR {} lakh".format(val), ha="center", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="x", rotation=15)
    ax1.grid(True, axis="y", alpha=0.3)
    ax2.bar(intv_agg.intervention, intv_agg.total_lives,
            color=["#2ca02c", "#1f77b4", "#d62728"], alpha=0.85)
    ax2.set_ylabel("Cumulative lives saved 2025-2030")
    ax2.set_title("(b) Lives saved by intervention under optimal allocation")
    for i, (intv, lives) in enumerate(zip(intv_agg.intervention, intv_agg.total_lives)):
        ax2.text(i, lives + 5000, f"{int(lives):,}", ha="center", fontsize=10, fontweight="bold")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Figure 3. Optimal NSPS investment allocation (INR 6,000 crore envelope, 2025-2030)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_3_optimal_allocation.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved Fig 3")


def fig4_achievability():
    ach = pd.read_csv(config.DATA_PROCESSED / "nsps_achievability_index.csv")
    ach = ach.sort_values("achievability_index")
    tier_colors = {"EMERGENCY": "#d62728", "WATCH": "#ff7f0e", "ON-TRACK": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(9, 11))
    y = np.arange(len(ach))
    colors = [tier_colors[t] for t in ach.tier]
    ax.barh(y, ach.achievability_index, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(ach.state, fontsize=9)
    ax.set_xlabel("NSPS Achievability Index (z-scored, weighted composite)")
    ax.set_title("Figure 4. State NSPS 2030 Achievability Index\n"
                 "Red EMERGENCY ; Orange WATCH ; Green ON-TRACK",
                 fontsize=12, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)
    legend_elems = [Patch(facecolor=c, alpha=0.85, label=t) for t, c in tier_colors.items()]
    ax.legend(handles=legend_elems, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_4_achievability_index.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved Fig 4")


def fig5_cost_of_inaction():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    labels = ["Productivity\nloss", "Healthcare\ncost", "VSL value\nof lives lost"]
    vals = [9.68, 1.32, 60.48]
    colors = ["#ff7f0e", "#d62728", "#1f77b4"]
    ax1.bar(labels, vals, color=colors, alpha=0.85)
    ax1.set_ylabel("Cost-of-inaction (INR lakh crore, 2025-2030)")
    ax1.set_title("(a) Cost-of-inaction: status quo SRS-anchored baseline")
    for i, v in enumerate(vals):
        ax1.text(i, v + 1, f"INR {v} lakh cr", ha="center", fontsize=10, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)
    investment_crore = 6_000
    benefit_lc = 6.18
    ax2.bar(["NSPS\ninvestment", "NSPS economic\nbenefit"],
            [investment_crore / 1e3, benefit_lc * 100],
            color=["#444", "#2ca02c"], alpha=0.85)
    ax2.set_ylabel("INR thousand crore")
    ax2.set_title("(b) Benefit-Cost ratio: NSPS investment vs averted cost")
    ax2.text(0, investment_crore / 1e3 + 10, f"INR {investment_crore/1e3:.1f}k crore",
             ha="center", fontsize=11, fontweight="bold")
    ax2.text(1, benefit_lc * 100 + 15, f"INR {benefit_lc*100:.0f}k crore\n(B:C = 103:1)",
             ha="center", fontsize=11, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Figure 5. Economic case for the NSPS 2030 implementation roadmap",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "Figure_5_cost_of_inaction.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved Fig 5")


if __name__ == "__main__":
    fig1_nsps_trajectory()
    fig2_paf()
    fig3_optimal_allocation()
    fig4_achievability()
    fig5_cost_of_inaction()
