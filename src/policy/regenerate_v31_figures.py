"""Regenerate v3.1 figures with fixes for layout/overlap/title errors.

Fixes:
  - Fig 2: move death-count labels above each bar (off the error-bar)
  - Fig 3: human-readable intervention names; increase top margin so labels don't clip
  - Fig 4: rename suptitle to 'Figure 4'; fix Panel (b) label overlap with subtitle
  - Fig 5: rename suptitle to 'Figure 5'
"""
from __future__ import annotations
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


def fig2_paf():
    """Fix: move death-count labels above bars to avoid error-bar overlap."""
    paf = pd.read_csv(config.DATA_PROCESSED / "paf_modifiable_risk_factors.csv")
    paf["paf_pct"] = paf["PAF"].str.extract(r"(\d+\.\d+)%")[0].astype(float)
    paf["paf_lo"] = paf["PAF"].str.extract(r"\(([\d\.]+)-").astype(float)
    paf["paf_hi"] = paf["PAF"].str.extract(r"-([\d\.]+)%\)").astype(float)
    paf = paf.sort_values("paf_pct", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    y = np.arange(len(paf))
    bar_height = 0.55
    ax.barh(y, paf.paf_pct, height=bar_height,
            color=["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"], alpha=0.85,
            label="_nolegend_")
    ax.errorbar(paf.paf_pct, y, xerr=[paf.paf_pct - paf.paf_lo, paf.paf_hi - paf.paf_pct],
                fmt="none", color="black", capsize=5, lw=1.4)
    ax.set_yticks(y)
    short_map = {
        "Pesticide method access (organophosphate, paraquat)": "Pesticide method access",
        "Heavy alcohol use disorder": "Heavy alcohol\nuse disorder",
        "Untreated depression / mental disorder": "Untreated\ndepression",
        "Intimate-partner violence in women aged 15-49": "Intimate-partner violence\n(women 15-49)",
    }
    ax.set_yticklabels([short_map.get(r["Risk factor"], r["Risk factor"][:35])
                        for _, r in paf.iterrows()], fontsize=10)
    ax.set_xlabel("Population-Attributable Fraction (%)", fontsize=11)
    ax.set_title("Figure 2. Population-Attributable Fractions (with 95% UI)\n"
                 "for modifiable Indian suicide risk factors",
                 fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, 85)
    # Move death-count labels ABOVE each bar to avoid error-bar overlap
    for i, r in paf.reset_index(drop=True).iterrows():
        deaths_str = r["Attributable deaths 2023"].split(" (")[0]
        # Place text just above bar (slightly off centre to avoid all collisions)
        ax.text(r.paf_hi + 2, i, f"{deaths_str} deaths",
                va="center", ha="left", fontsize=10, fontweight="bold",
                color="black",
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.2"))
    plt.subplots_adjust(left=0.30, right=0.96, top=0.86, bottom=0.12)
    fig.savefig(OUT / "Figure_2_paf.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("regenerated Fig 2")


def fig3_optimal_allocation():
    """Fix: human-readable intervention names; bigger top margin."""
    alloc = pd.read_csv(config.DATA_PROCESSED / "optimal_nsps_allocation.csv")
    intv_agg = alloc.groupby("intervention").agg(
        total_cost_crore=("cost_crore", "sum"),
        total_lives=("lives_cum", "sum"),
    ).reset_index()
    intv_agg["cost_per_life_lakh"] = (intv_agg.total_cost_crore * 100 / intv_agg.total_lives).round(1)
    intv_agg = intv_agg.sort_values("cost_per_life_lakh")
    # Human-readable names
    NAME = {
        "Pesticide_ban":   "Pesticide regulation",
        "Tele_MANAS_full": "Tele-MANAS expansion",
        "DMHP_universal":  "DMHP universal",
    }
    intv_agg["pretty"] = intv_agg.intervention.map(NAME)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5))
    bars = ax1.bar(intv_agg.pretty, intv_agg.cost_per_life_lakh,
                   color=["#2ca02c", "#1f77b4", "#d62728"], alpha=0.85)
    ax1.set_ylabel("Cost per life saved (INR lakh)", fontsize=11)
    ax1.set_title("(a) Cost-effectiveness ladder (lower = more efficient)", fontsize=11)
    ax1.set_ylim(0, intv_agg.cost_per_life_lakh.max() * 1.20)
    for bar, val in zip(bars, intv_agg.cost_per_life_lakh):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.4,
                 f"INR {val} lakh", ha="center", fontsize=10, fontweight="bold")
    ax1.tick_params(axis="x", rotation=12)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar(intv_agg.pretty, intv_agg.total_lives,
            color=["#2ca02c", "#1f77b4", "#d62728"], alpha=0.85)
    ax2.set_ylabel("Cumulative lives saved 2025-2030", fontsize=11)
    ax2.set_title("(b) Lives saved by intervention under optimal allocation", fontsize=11)
    ax2.set_ylim(0, intv_agg.total_lives.max() * 1.15)
    for i, (pretty, lives) in enumerate(zip(intv_agg.pretty, intv_agg.total_lives)):
        ax2.text(i, lives + 5500, f"{int(lives):,}", ha="center",
                 fontsize=10, fontweight="bold")
    ax2.tick_params(axis="x", rotation=12)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 3. Optimal NSPS investment allocation\n"
                 "(INR 6,000 crore envelope, 2025-2030)",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.84, bottom=0.13)
    fig.savefig(OUT / "Figure_3_optimal_allocation.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("regenerated Fig 3")


def fig4_economic_case():
    """Fix: correct suptitle 'Figure 4'; resolve Panel (b) label overlap."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # Panel (a) — cost-of-inaction breakdown
    labels = ["Productivity\nloss", "Healthcare\ncost", "VSL value\nof lives lost"]
    vals = [9.68, 1.32, 60.48]
    colors = ["#ff7f0e", "#d62728", "#1f77b4"]
    ax1.bar(labels, vals, color=colors, alpha=0.85)
    ax1.set_ylabel("Cost-of-inaction (INR lakh crore, 2025-2030)", fontsize=10.5)
    ax1.set_title("(a) Cost-of-inaction: status quo SRS-anchored baseline", fontsize=11)
    ax1.set_ylim(0, 75)
    for i, v in enumerate(vals):
        ax1.text(i, v + 2, f"INR {v} lakh cr", ha="center", fontsize=10, fontweight="bold")
    ax1.grid(True, axis="y", alpha=0.3)

    # Panel (b) — benefit-cost: use kilo-crore values; cap y-axis appropriately
    investment_kc = 6.0  # INR 6,000 crore = INR 6.0 thousand crore
    benefit_kc = 618.0   # INR 6.18 lakh crore = INR 618 thousand crore
    ax2.bar(["NSPS\ninvestment", "NSPS economic\nbenefit"], [investment_kc, benefit_kc],
            color=["#444", "#2ca02c"], alpha=0.85)
    ax2.set_ylabel("INR thousand crore (Rs lakh)", fontsize=10.5)
    ax2.set_title("(b) Benefit-Cost ratio: NSPS investment vs averted cost", fontsize=11)
    ax2.set_ylim(0, 750)
    ax2.text(0, investment_kc + 25, f"INR {investment_kc:.1f}k crore",
             ha="center", fontsize=10, fontweight="bold")
    ax2.text(1, benefit_kc + 25, f"INR {benefit_kc:.0f}k crore\n(B:C ratio = 103:1)",
             ha="center", fontsize=10, fontweight="bold")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Figure 4. Economic case for NSPS implementation, India 2025-2030",
                 fontsize=12, fontweight="bold", y=0.99)
    plt.subplots_adjust(top=0.86, bottom=0.13, wspace=0.30)
    fig.savefig(OUT / "Figure_4_economic_case.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("regenerated Fig 4")


def fig5_achievability():
    """Fix: correct suptitle 'Figure 5'."""
    ach = pd.read_csv(config.DATA_PROCESSED / "nsps_achievability_index.csv")
    ach = ach.sort_values("achievability_index")
    tier_colors = {"EMERGENCY": "#d62728", "WATCH": "#ff7f0e", "ON-TRACK": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(9, 11))
    y = np.arange(len(ach))
    colors = [tier_colors[t] for t in ach.tier]
    ax.barh(y, ach.achievability_index, color=colors, alpha=0.85)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y); ax.set_yticklabels(ach.state, fontsize=9)
    ax.set_xlabel("NSPS Achievability Index (z-scored, weighted composite)", fontsize=10.5)
    ax.set_title("Figure 5. State NSPS 2030 Achievability Index, India\n"
                 "Red = EMERGENCY tier;  Orange = WATCH;  Green = ON-TRACK",
                 fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, axis="x", alpha=0.3)
    legend_elems = [Patch(facecolor=c, alpha=0.85, label=t) for t, c in tier_colors.items()]
    ax.legend(handles=legend_elems, loc="lower right")
    plt.subplots_adjust(left=0.25, right=0.96, top=0.93, bottom=0.07)
    fig.savefig(OUT / "Figure_5_achievability_index.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print("regenerated Fig 5")


if __name__ == "__main__":
    fig2_paf()
    fig3_optimal_allocation()
    fig4_economic_case()
    fig5_achievability()
