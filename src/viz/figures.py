"""Phase 5 — Build Figures 1-6 for the suicide-reconciliation manuscript."""
from __future__ import annotations
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src import config


def _save(fig, basename: str) -> None:
    config.OUT_FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "tif"):
        fig.savefig(config.OUT_FIGS / f"{basename}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1_reconciliation_choropleth() -> None:
    """Fig 1 — NCRB/GBD ratio per state (choropleth-style bar chart since shapefile is light)."""
    import geopandas as gpd
    panel = pd.read_parquet(config.DATA_PROCESSED / "suicide_reconciliation_panel.parquet")
    p = panel[panel["year"] == 2023].dropna(subset=["ratio_ncrb_gbd"]).copy()
    p = p.sort_values("ratio_ncrb_gbd")

    # Try a choropleth if the GeoJSON loads
    try:
        gdf = gpd.read_file(config.EXISTING_GEOJSON_STATES)
        # Heuristic: detect state-name column
        name_col = next((c for c in gdf.columns if c.lower() in ("st_nm", "name", "state", "state_name")), None)
        if name_col:
            merged = gdf.merge(p[["state", "ratio_ncrb_gbd"]], left_on=name_col, right_on="state", how="left")
            for variant, cmap in (("colour", "RdBu"), ("bw", "Greys")):
                fig, ax = plt.subplots(figsize=(7, 8))
                merged.plot(column="ratio_ncrb_gbd", cmap=cmap, vmin=0, vmax=2,
                            legend=True, ax=ax, missing_kwds={"color": "#cccccc"})
                ax.set_title("Figure 1. NCRB-vs-GBD suicide-mortality ratio by state, India 2023")
                ax.axis("off")
                _save(fig, f"fig1_ncrb_gbd_ratio_{variant}")
            return
    except Exception as e:
        print(f"[fig1] geopandas choropleth failed ({e}); falling back to bar chart")

    # Fallback bar chart
    for variant, palette in (("colour", "RdBu_r"), ("bw", "Greys")):
        fig, ax = plt.subplots(figsize=(8, 10))
        bars = ax.barh(p["state"], p["ratio_ncrb_gbd"], color="lightgrey", edgecolor="black")
        ax.axvline(1.0, color="red" if variant == "colour" else "black", linestyle="--", label="Parity (NCRB=GBD)")
        ax.set_xlabel("NCRB-attributed suicides / GBD-estimated suicides, 2023")
        ax.set_title("Figure 1. NCRB-vs-GBD ratio by state")
        ax.legend()
        _save(fig, f"fig1_ncrb_gbd_ratio_{variant}")


def fig2_avertable_burden() -> None:
    """Fig 2 — Avertable burden Scenario C per 100k, by state."""
    central = pd.read_parquet(config.DATA_PROCESSED / "suicide_avertable_central.parquet")
    central["averted_per100k"] = central["averted_C"] / central["population_m"] / 10.0
    p = central.sort_values("averted_per100k")
    for variant, color in (("colour", "#1f77b4"), ("bw", "0.4")):
        fig, ax = plt.subplots(figsize=(8, 9))
        ax.barh(p["state"], p["averted_per100k"], color=color, edgecolor="black")
        ax.set_xlabel("Avertable suicides per 100,000 if state met WHO global rate (9.0/100k)")
        ax.set_title("Figure 2. Avertable suicide burden under Scenario C (WHO global benchmark), 2023")
        _save(fig, f"fig2_avertable_scenario_C_{variant}")


def fig3_forest_scenarios() -> None:
    """Fig 3 — Avertable deaths by state, Scenarios A/B/C."""
    central = pd.read_parquet(config.DATA_PROCESSED / "suicide_avertable_central.parquet")
    p = central.sort_values("averted_C")
    fig, ax = plt.subplots(figsize=(9, 10))
    y = np.arange(len(p))
    ax.barh(y - 0.25, p["averted_A"], height=0.25, label="Scen A (25-pct state, 10/100k)", color="#cccccc", edgecolor="black")
    ax.barh(y,         p["averted_B"], height=0.25, label="Scen B (national avg, 14/100k)", color="#888888", edgecolor="black")
    ax.barh(y + 0.25, p["averted_C"], height=0.25, label="Scen C (WHO global, 9/100k)",  color="#222222", edgecolor="black")
    ax.set_yticks(y); ax.set_yticklabels(p["state"])
    ax.set_xlabel("Avertable suicide deaths (annual; based on GBD 2023 rates)")
    ax.set_title("Figure 3. Avertable suicide deaths by state under three counterfactual scenarios, 2023")
    ax.legend(loc="lower right")
    _save(fig, "fig3_forest_three_scenarios")


def fig4_equity_summary() -> None:
    """Fig 4 — Equity indices summary bar chart with concentration curve."""
    central = pd.read_parquet(config.DATA_PROCESSED / "suicide_avertable_central.parquet")
    central = central.dropna(subset=["gbd_rate", "population_m"]).sort_values("gbd_rate")
    p_share = central["population_m"] / central["population_m"].sum()
    deaths_share = (central["gbd_rate"] * central["population_m"]) / (central["gbd_rate"] * central["population_m"]).sum()
    cum_p = np.cumsum(p_share)
    cum_d = np.cumsum(deaths_share)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Line of equality")
    ax.plot(np.concatenate([[0], cum_p]), np.concatenate([[0], cum_d]),
            "-o", color="#222222", markersize=3, label="Concentration curve (suicide deaths)")
    ax.fill_between(np.concatenate([[0], cum_p]), np.concatenate([[0], cum_p]),
                    np.concatenate([[0], cum_d]), alpha=0.2, color="grey")
    ax.set_xlabel("Cumulative population share (states ranked low to high suicide rate)")
    ax.set_ylabel("Cumulative suicide-death share")
    ax.set_title("Figure 4. Concentration curve of state-level Indian suicide mortality, GBD 2023")
    ax.legend()
    _save(fig, "fig4_concentration_curve")


def fig5_bayesian_state_estimates() -> None:
    """Fig 5 — Joint Bayesian posterior (true rate theta) with NCRB observed and GBD observed comparators.

    Falls back to single-source Bayesian model if joint model not yet available.
    """
    joint_path = config.DATA_PROCESSED / "suicide_joint_bayes_v2_summary.parquet"
    if not joint_path.exists():
        joint_path = config.DATA_PROCESSED / "suicide_joint_bayes_summary.parquet"
    single_path = config.DATA_PROCESSED / "suicide_bayes_summary.parquet"
    if joint_path.exists():
        b = pd.read_parquet(joint_path).sort_values("theta_true_rate_mean_per100k")
        fig, ax = plt.subplots(figsize=(8, 10))
        y = np.arange(len(b))
        ax.errorbar(b["theta_true_rate_mean_per100k"], y,
                    xerr=[b["theta_true_rate_mean_per100k"] - b["theta_true_rate_lo95"],
                          b["theta_true_rate_hi95"] - b["theta_true_rate_mean_per100k"]],
                    fmt="o", color="black", capsize=2,
                    label=r"$\theta$ joint Bayes posterior (mean $\pm$ 95% CrI)")
        ax.scatter(b["ncrb_rate_observed_per100k"], y, marker="s", color="grey",
                   label="NCRB observed", alpha=0.6)
        ax.scatter(b["gbd_rate_observed_per100k"], y, marker="^", color="white",
                   edgecolor="black", label="GBD modelled", alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(b["state"])
        ax.set_xlabel("Suicide rate per 100,000")
        ax.set_title("Figure 5. Joint Bayesian posterior true state suicide rates ($\\theta$) "
                     "with NCRB observed and GBD observed comparators, India 2023")
        ax.legend(loc="lower right")
        _save(fig, "fig5_joint_bayes_state_rates")
    elif single_path.exists():
        b = pd.read_parquet(single_path).sort_values("bayes_rate_mean_per100k")
        fig, ax = plt.subplots(figsize=(8, 10))
        y = np.arange(len(b))
        ax.errorbar(b["bayes_rate_mean_per100k"], y,
                    xerr=[b["bayes_rate_mean_per100k"] - b["bayes_rate_lo95"],
                          b["bayes_rate_hi95"] - b["bayes_rate_mean_per100k"]],
                    fmt="o", color="black", capsize=2,
                    label="Bayes posterior (mean ± 95% HDI)")
        ax.scatter(b["ncrb_observed_rate_per100k"], y, marker="s", color="grey",
                   label="NCRB observed", alpha=0.6)
        ax.scatter(b["gbd_rate_per100k"], y, marker="^", color="white",
                   edgecolor="black", label="GBD modelled", alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels(b["state"])
        ax.set_xlabel("Suicide rate per 100,000")
        ax.set_title("Figure 5. Bayesian posterior state suicide rates (single-source model)")
        ax.legend(loc="lower right")
        _save(fig, "fig5_bayesian_state_rates")
    else:
        print("[fig5] no Bayesian results available; skipping")


def fig6_trend_2018_2024() -> None:
    """Fig 6 — National NCRB 2018-2024 + GBD 2020-2023 overlay + top-state trends."""
    ncrb_nat = pd.read_csv(config.DATA_PROCESSED / "ncrb_national_totals_2018_2024.csv")
    gbd_nat = pd.read_parquet(config.DATA_PROCESSED / "gbd_trend_national.parquet")
    gbd_states = pd.read_parquet(config.DATA_PROCESSED / "gbd_trend_2020_2023.parquet")
    top5 = gbd_states[gbd_states["year"] == 2023].nlargest(5, "gbd_deaths")["state"].tolist()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Panel (a): NCRB 2018-2024 + GBD 2020-2023 overlay
    ax1.plot(ncrb_nat["year"], ncrb_nat["national_total"], "o-", color="#1f77b4", linewidth=2.5,
             markersize=8, label="NCRB official total")
    ax1.plot(gbd_nat["year"], gbd_nat["gbd_national"], "s-", color="#d62728", linewidth=2,
             markersize=7, label="GBD modelled total")
    ax1.fill_between(gbd_nat["year"], gbd_nat["lo95"], gbd_nat["hi95"], alpha=0.15, color="#d62728",
                     label="GBD 95% UI")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("National suicide deaths")
    ax1.set_title("(a) NCRB official vs GBD modelled national totals, India 2018-2024")
    ax1.legend(loc="lower right")
    ax1.set_xticks(list(range(2018, 2025)))
    ax1.grid(True, alpha=0.3)
    # Annotate COVID peak
    ax1.axvspan(2020, 2021.2, alpha=0.1, color="grey")
    ax1.annotate("COVID-19\nperiod", xy=(2020.6, ncrb_nat["national_total"].max()*0.55),
                 ha="center", fontsize=9, alpha=0.7)

    # Panel (b): top 5 states GBD trend
    for st in top5:
        sub = gbd_states[gbd_states["state"] == st].sort_values("year")
        ax2.plot(sub["year"], sub["gbd_deaths"], "o-", label=st, linewidth=1.5)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("State suicide deaths (GBD-modelled)")
    ax2.set_title("(b) Top-5 burden states GBD trend, 2020-2023")
    ax2.legend(loc="best", fontsize=9)
    ax2.set_xticks([2020, 2021, 2022, 2023])
    ax2.grid(True, alpha=0.3)
    fig.suptitle("Figure 6. National (NCRB 2018-2024 + GBD 2020-2023) and state-level suicide-mortality trend, India",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "fig6_trend_2018_2024")


def fig_s11_who_sex_trend() -> None:
    """Supplement Fig S11 - WHO GHO India sex-stratified suicide rate trend 2010-2021."""
    who = pd.read_csv(config.DATA_PROCESSED / "who_india_sex_trend.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(who["year"], who["Both"], "o-", color="black", linewidth=2.5, label="Both sexes", markersize=8)
    ax.plot(who["year"], who["Male"], "s-", color="#1f77b4", linewidth=2, label="Male", markersize=7)
    ax.plot(who["year"], who["Female"], "^-", color="#d62728", linewidth=2, label="Female", markersize=7)
    ax.set_xlabel("Year")
    ax.set_ylabel("Suicide mortality rate (per 100,000)")
    ax.set_title("Supplement Figure S11. WHO GHO India suicide rate by sex, 2010-2021")
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(range(2010, 2022)))
    plt.setp(ax.get_xticklabels(), rotation=45)
    # Annotate APC
    ax.text(0.02, 0.97, "Both APC: -2.34%/yr (p<0.0001)\nMale APC: -2.16%/yr (p=0.0001)\nFemale APC: -2.63%/yr (p<0.0001)",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="grey", alpha=0.9))
    _save(fig, "figS11_who_india_sex_trend")


def fig_s10_nsps_forecast() -> None:
    """Supplement Fig S10 - NSPS 2030 forecast under three scenarios."""
    fc = pd.read_csv(config.DATA_PROCESSED / "nsps_forecast_2030.csv")
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(fc["year"], fc["scenario_A_status_quo"], "o-", color="#444444", linewidth=2,
            label="Status quo (-0.05%/year continuation)")
    ax.plot(fc["year"], fc["scenario_B_NSPS_10pct_2030"], "s-", color="#1f77b4", linewidth=2,
            label="NSPS target: 10% reduction by 2030")
    ax.plot(fc["year"], fc["scenario_C_best_state_minus2pct"], "^-", color="#2ca02c", linewidth=2,
            label="Best-state model (-2.0%/year)")
    ax.axvline(2024, color="grey", linestyle=":", alpha=0.7, label="2024 (last observed)")
    ax.fill_between([2024, 2030], 130000, 200000, alpha=0.05, color="grey")
    ax.set_xlabel("Year")
    ax.set_ylabel("National annual suicide deaths (NCRB)")
    ax.set_title("Supplement Figure S10. NSPS 2030 forecast under three policy scenarios, India")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(list(range(2018, 2031)))
    plt.setp(ax.get_xticklabels(), rotation=45)
    _save(fig, "figS10_nsps_forecast_2030")


def fig_s9_under_registration_drivers() -> None:
    """Fig 6 — Scatter of state rate vs NCRB/GBD ratio (identifies systematic under-registration patterns)."""
    panel = pd.read_parquet(config.DATA_PROCESSED / "suicide_reconciliation_panel.parquet")
    central = pd.read_parquet(config.DATA_PROCESSED / "suicide_avertable_central.parquet")
    merged = panel[panel["year"] == 2023].merge(central[["state", "gbd_rate", "population_m"]], on="state", how="inner")
    merged = merged.dropna(subset=["ratio_ncrb_gbd", "gbd_rate"])

    fig, ax = plt.subplots(figsize=(8, 7))
    sizes = (merged["population_m"] / merged["population_m"].max() * 800).clip(40, None)
    ax.scatter(merged["gbd_rate"], merged["ratio_ncrb_gbd"], s=sizes, alpha=0.7, edgecolor="black", facecolor="grey")
    for _, r in merged.iterrows():
        ax.annotate(r["state"][:10], (r["gbd_rate"], r["ratio_ncrb_gbd"]), fontsize=7, alpha=0.8)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="Parity (NCRB=GBD)")
    ax.set_xlabel("GBD-modelled suicide rate per 100,000")
    ax.set_ylabel("NCRB/GBD ratio (1.0 = parity)")
    ax.set_title("Supplement Figure S9. State suicide rate vs NCRB-GBD discrepancy ratio (bubble size = population), India 2023")
    ax.legend()
    _save(fig, "figS9_under_registration_drivers")


def main() -> None:
    fig1_reconciliation_choropleth()
    print("[viz] Fig 1 done")
    fig2_avertable_burden()
    print("[viz] Fig 2 done")
    fig3_forest_scenarios()
    print("[viz] Fig 3 done")
    fig4_equity_summary()
    print("[viz] Fig 4 done")
    fig5_bayesian_state_estimates()
    print("[viz] Fig 5 done (may be deferred)")
    fig6_trend_2018_2024()
    print("[viz] Fig 6 (NCRB 2018-2024 + GBD overlay) done")
    fig_s9_under_registration_drivers()
    print("[viz] Supplement Fig S9 done")
    fig_s10_nsps_forecast()
    print("[viz] Supplement Fig S10 (NSPS forecast) done")
    fig_s11_who_sex_trend()
    print("[viz] Supplement Fig S11 (WHO sex trend) done")


if __name__ == "__main__":
    main()
