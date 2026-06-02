"""Phase 5 — Build IJMR Tables 1-4 for the suicide-reconciliation manuscript."""
from __future__ import annotations
import json
import pandas as pd
from src import config


def build_table1() -> pd.DataFrame:
    """Table 1 — State suicide burden 2023: NCRB count, GBD estimate, NCRB/GBD ratio."""
    panel = pd.read_parquet(config.DATA_PROCESSED / "suicide_reconciliation_panel.parquet")
    p = panel[panel["year"] == 2023].copy()
    p = p.dropna(subset=["ncrb_suicides", "gbd_suicides"])
    p = p.sort_values("ncrb_suicides", ascending=False)
    t = pd.DataFrame({
        "State/UT": p["state"],
        "NCRB suicides 2023": p["ncrb_suicides"].astype(int),
        "GBD suicides 2023": p["gbd_suicides"].round(0).astype(int),
        "GBD 95% UI": p.apply(lambda r: f"{r['gbd_lo']:.0f} - {r['gbd_hi']:.0f}", axis=1),
        "NCRB/GBD ratio": p["ratio_ncrb_gbd"].round(3),
    })
    total = pd.DataFrame([{
        "State/UT": "INDIA (all states with both)",
        "NCRB suicides 2023": int(p["ncrb_suicides"].sum()),
        "GBD suicides 2023": int(p["gbd_suicides"].sum().round()),
        "GBD 95% UI": "",
        "NCRB/GBD ratio": round(p["ncrb_suicides"].sum() / p["gbd_suicides"].sum(), 3),
    }])
    return pd.concat([t, total], ignore_index=True)


def build_table2() -> pd.DataFrame:
    """Table 2 — Avertable suicides under three scenarios, per state."""
    central = pd.read_parquet(config.DATA_PROCESSED / "suicide_avertable_central.parquet")
    central = central.sort_values("averted_C", ascending=False)
    t = pd.DataFrame({
        "State/UT": central["state"],
        "Population (millions)": central["population_m"].round(1),
        "GBD rate / 100k": central["gbd_rate"].round(2),
        "Averted under Scen A (25-pct state, 10.04/100k)": central["averted_A"].round(0).astype(int),
        "Averted under Scen B (national avg, 14.43/100k)": central["averted_B"].round(0).astype(int),
        "Averted under Scen C (WHO global, 9.00/100k)": central["averted_C"].round(0).astype(int),
    })
    total = pd.DataFrame([{
        "State/UT": "INDIA (national total)",
        "Population (millions)": round(central["population_m"].sum(), 1),
        "GBD rate / 100k": round((central["gbd_rate"] * central["population_m"]).sum() / central["population_m"].sum(), 2),
        "Averted under Scen A (25-pct state, 10.04/100k)": int(central["averted_A"].sum()),
        "Averted under Scen B (national avg, 14.43/100k)": int(central["averted_B"].sum()),
        "Averted under Scen C (WHO global, 9.00/100k)": int(central["averted_C"].sum()),
    }])
    return pd.concat([t, total], ignore_index=True)


def build_table3() -> pd.DataFrame:
    """Table 3 — Equity indices summary."""
    eq = pd.read_parquet(config.DATA_PROCESSED / "suicide_equity_indices.parquet").iloc[0].to_dict()
    rows = [
        {"Measure": "Number of states/UTs in analysis", "Value": eq["n_states"]},
        {"Measure": "National suicide rate (GBD-derived) per 100,000", "Value": round(eq["national_rate_per_100k"], 2)},
        {"Measure": "Minimum state rate / 100k", "Value": f"{eq['min_rate_state']} ({eq['min_rate_value']:.2f})"},
        {"Measure": "Maximum state rate / 100k", "Value": f"{eq['max_rate_state']} ({eq['max_rate_value']:.2f})"},
        {"Measure": "State-level Slope Index of Inequality (per 100k)", "Value": round(eq["SII_state_level_per100k"], 2)},
        {"Measure": "State-level Relative Index of Inequality", "Value": round(eq["RII_state_level"], 3)},
        {"Measure": "Erreygers Concentration Index (state level)", "Value": round(eq["Concentration_index_state_level"], 3)},
        {"Measure": "Theil index (state level)", "Value": round(eq["Theil_index_state_level"], 4)},
        {"Measure": "Sex Slope Index of Inequality (per 100k)", "Value": round(eq["SII_sex"], 2)},
        {"Measure": "Sex Relative Index of Inequality", "Value": round(eq["RII_sex"], 3)},
    ]
    return pd.DataFrame(rows)


def build_table4() -> pd.DataFrame:
    """Table 4 — Bayesian state-rate posterior summary (loaded from MCMC outputs if available)."""
    bayes_path = config.DATA_PROCESSED / "suicide_bayes_summary.parquet"
    diag_path = config.DATA_PROCESSED / "suicide_bayes_diagnostics.json"
    if not bayes_path.exists():
        return pd.DataFrame([{"Note": "Bayesian model not yet completed; table 4 deferred."}])
    b = pd.read_parquet(bayes_path)
    b = b.sort_values("bayes_rate_mean_per100k", ascending=False)
    t = pd.DataFrame({
        "State/UT": b["state"],
        "NCRB rate / 100k": b["ncrb_observed_rate_per100k"].round(2),
        "GBD rate / 100k": b["gbd_rate_per100k"].round(2),
        "Bayes posterior mean / 100k": b["bayes_rate_mean_per100k"].round(2),
        "Bayes 95% HDI": b.apply(lambda r: f"{r['bayes_rate_lo95']:.2f} - {r['bayes_rate_hi95']:.2f}", axis=1),
    })
    if diag_path.exists():
        diag = json.loads(diag_path.read_text())
        footer = pd.DataFrame([{
            "State/UT": "MCMC diagnostics",
            "NCRB rate / 100k": "",
            "GBD rate / 100k": f"R-hat max = {diag['max_r_hat']:.4f}",
            "Bayes posterior mean / 100k": f"ESS min = {diag['min_ess_bulk']:.0f}",
            "Bayes 95% HDI": "OK" if diag["convergence_ok"] else "CHECK",
        }])
        t = pd.concat([t, footer], ignore_index=True)
    return t


def build_table_s10_trend() -> pd.DataFrame:
    """Supplement S10 — GBD self-harm state trend 2020-2023 with APC."""
    apc = pd.read_parquet(config.DATA_PROCESSED / "gbd_trend_apc.parquet")
    trend = pd.read_parquet(config.DATA_PROCESSED / "gbd_trend_2020_2023.parquet")
    wide = trend.pivot_table(index="state", columns="year", values="gbd_deaths", aggfunc="first").reset_index()
    wide.columns = [f"GBD {c}" if isinstance(c, int) else "State/UT" for c in wide.columns]
    merged = wide.merge(apc, left_on="State/UT", right_on="state", how="inner")
    merged["APC (% per year)"] = merged["apc_pct_per_year"].round(2)
    merged["APC p-value"] = merged["p_value"].round(3)
    cols = ["State/UT", "GBD 2020", "GBD 2021", "GBD 2022", "GBD 2023", "APC (% per year)", "APC p-value"]
    out = merged[cols].sort_values("GBD 2023", ascending=False)
    for c in ["GBD 2020", "GBD 2021", "GBD 2022", "GBD 2023"]:
        out[c] = out[c].round(0).astype(int)
    return out


def build_table_s11_national_trend() -> pd.DataFrame:
    """Supplement S11 — National GBD trend with 95% UI 2020-2023."""
    nat = pd.read_parquet(config.DATA_PROCESSED / "gbd_trend_national.parquet")
    nat["GBD national (95% UI)"] = nat.apply(lambda r: f"{r['gbd_national']:,.0f} ({r['lo95']:,.0f} - {r['hi95']:,.0f})", axis=1)
    nat["YoY change %"] = (nat["gbd_national"].pct_change() * 100).round(1)
    return nat[["year", "GBD national (95% UI)", "YoY change %"]].rename(columns={"year": "Year"})


def build_table_s12_forecast() -> pd.DataFrame:
    """Supplement S12 — NSPS 2030 forecast under three scenarios."""
    fc = pd.read_csv(config.DATA_PROCESSED / "nsps_forecast_2030.csv")
    fc["Year"] = fc["year"]
    fc["Scenario A: status quo"] = fc["scenario_A_status_quo"].apply(lambda x: f"{int(x):,}")
    fc["Scenario B: NSPS 10% by 2030"] = fc["scenario_B_NSPS_10pct_2030"].apply(lambda x: f"{int(x):,}")
    fc["Scenario C: best-state model (-2%/yr)"] = fc["scenario_C_best_state_minus2pct"].apply(lambda x: f"{int(x):,}")
    out = fc[["Year", "Scenario A: status quo", "Scenario B: NSPS 10% by 2030", "Scenario C: best-state model (-2%/yr)"]]
    return out


def build_table_s17_ses_gradient() -> pd.DataFrame:
    """Supplement S17 - state SES gradient (NFHS-5 composite vs NCRB 2023 rate)."""
    s = pd.read_csv(config.DATA_PROCESSED / "state_ses_suicide_2023.csv")
    out = pd.DataFrame({
        "State/UT": s["state"],
        "N districts (NFHS-5)": s["n_districts"].astype(int),
        "Mean SES (PCA PC1)": s["mean_ses"].round(2),
        "% districts in poorest national Q1": s["pct_in_q1_poorest"].round(1),
        "NCRB 2023 suicide rate (per 100k)": s["ncrb_rate_per_100k"].round(1),
        "SES rank fraction": s["ses_rank_frac"].round(3),
    }).sort_values("Mean SES (PCA PC1)")
    return out


def build_table_s18_state_sex_bayes() -> pd.DataFrame:
    """Supplement S18 - state x sex joint Bayesian posterior rates with 95% CrI."""
    b = pd.read_parquet(config.DATA_PROCESSED / "state_sex_bayes_v1.parquet")
    out = pd.DataFrame({
        "State/UT": b["state"],
        "Male theta (95% CrI)": b.apply(lambda r: f"{r['theta_male_mean']:.1f} ({r['theta_male_lo95']:.1f}-{r['theta_male_hi95']:.1f})", axis=1),
        "Female theta (95% CrI)": b.apply(lambda r: f"{r['theta_female_mean']:.1f} ({r['theta_female_lo95']:.1f}-{r['theta_female_hi95']:.1f})", axis=1),
        "Both theta (95% CrI)": b.apply(lambda r: f"{r['theta_both_mean']:.1f} ({r['theta_both_lo95']:.1f}-{r['theta_both_hi95']:.1f})", axis=1),
    }).sort_values("Both theta (95% CrI)", ascending=False)
    return out


def build_table_s16_gbd_sex_age_apc() -> pd.DataFrame:
    """Supplement S16 - GBD sex x age band APC 2018-2023 (national)."""
    apc = pd.read_csv(config.DATA_PROCESSED / "gbd_apc_by_sex_age_2018_2023.csv")
    apc["Significant (p<0.05)"] = (apc["p_value"] < 0.05).map({True: "*", False: ""})
    out = pd.DataFrame({
        "Sex": apc["sex"],
        "Age band": apc["age"],
        "2018 deaths": apc["value_2018"].round(0).astype(int),
        "2023 deaths": apc["value_2023"].round(0).astype(int),
        "APC (% / year)": apc["apc_pct_per_year"].round(2),
        "p-value": apc["p_value"].round(4),
        "Sig.": apc["Significant (p<0.05)"],
    }).sort_values(["Sex", "Age band"])
    return out


def build_table_s14_who_sex_trend() -> pd.DataFrame:
    """Supplement S14 — WHO GHO India sex-stratified suicide rate 2010-2021."""
    pivot = pd.read_csv(config.DATA_PROCESSED / "who_india_sex_trend.csv")
    out = pd.DataFrame({
        "Year": pivot["year"],
        "Both (per 100k)": pivot["Both"].round(2),
        "Male (per 100k)": pivot["Male"].round(2),
        "Female (per 100k)": pivot["Female"].round(2),
        "Male:Female ratio": pivot["MF_ratio"].round(2),
        "Male - Female difference (per 100k)": pivot["MF_diff_per100k"].round(2),
    })
    return out


def build_table_s15_state_year_recon() -> pd.DataFrame:
    """Supplement S15 — State-level multi-year NCRB-vs-GBD reconciliation 2020-2023."""
    s = pd.read_csv(config.DATA_PROCESSED / "state_reconciliation_summary_2020_2023.csv")
    out = pd.DataFrame({
        "State/UT": s["state"],
        "N years overlap": s["n_years_overlap"],
        "NCRB rate 2020": s["ncrb_rate_2020"].round(2),
        "NCRB rate 2023": s["ncrb_rate_2023"].round(2),
        "GBD rate 2020": s["gbd_rate_2020"].round(2),
        "GBD rate 2023": s["gbd_rate_2023"].round(2),
        "Mean NCRB/GBD ratio (2020-23)": s["mean_ratio_2020_2023"].round(2),
        "Ratio change (2020 -> 2023)": s["ratio_change_2020_to_2023"].round(2),
    }).sort_values("Mean NCRB/GBD ratio (2020-23)", ascending=False)
    return out


def build_table_s13_joint_bayes() -> pd.DataFrame:
    """Supplement S13 — Joint Bayesian model: theta_s and c_s posteriors per state."""
    joint_path = config.DATA_PROCESSED / "suicide_joint_bayes_v2_summary.parquet"
    if not joint_path.exists():
        joint_path = config.DATA_PROCESSED / "suicide_joint_bayes_summary.parquet"
    if not joint_path.exists():
        return pd.DataFrame([{"Note": "Joint Bayesian model not yet completed."}])
    j = pd.read_parquet(joint_path).sort_values("theta_true_rate_mean_per100k", ascending=False)
    out = pd.DataFrame({
        "State/UT": j["state"],
        "NCRB observed rate (per 100k)": j["ncrb_rate_observed_per100k"].round(2),
        "GBD modelled rate (per 100k)": j["gbd_rate_observed_per100k"].round(2),
        "Joint Bayes true rate theta (95% CrI)":
            j.apply(lambda r: f"{r['theta_true_rate_mean_per100k']:.2f} ({r['theta_true_rate_lo95']:.2f} - {r['theta_true_rate_hi95']:.2f})", axis=1),
        "NCRB under-registration c (95% CrI)":
            j.apply(lambda r: f"{r['c_underreg_mean']:.3f} ({r['c_underreg_lo95']:.3f} - {r['c_underreg_hi95']:.3f})", axis=1),
    })
    return out


def main() -> None:
    config.OUT_TABLES.mkdir(parents=True, exist_ok=True)
    t1 = build_table1(); t1.to_csv(config.OUT_TABLES / "table1_burden_2023.csv", index=False); t1.to_excel(config.OUT_TABLES / "table1_burden_2023.xlsx", index=False)
    t2 = build_table2(); t2.to_csv(config.OUT_TABLES / "table2_avertable_burden.csv", index=False); t2.to_excel(config.OUT_TABLES / "table2_avertable_burden.xlsx", index=False)
    t3 = build_table3(); t3.to_csv(config.OUT_TABLES / "table3_equity_indices.csv", index=False); t3.to_excel(config.OUT_TABLES / "table3_equity_indices.xlsx", index=False)
    t4 = build_table4(); t4.to_csv(config.OUT_TABLES / "table4_bayesian_rates.csv", index=False); t4.to_excel(config.OUT_TABLES / "table4_bayesian_rates.xlsx", index=False)
    s10 = build_table_s10_trend(); s10.to_csv(config.OUT_TABLES / "tableS10_trend_apc.csv", index=False); s10.to_excel(config.OUT_TABLES / "tableS10_trend_apc.xlsx", index=False)
    s11 = build_table_s11_national_trend(); s11.to_csv(config.OUT_TABLES / "tableS11_national_trend.csv", index=False); s11.to_excel(config.OUT_TABLES / "tableS11_national_trend.xlsx", index=False)
    s12 = build_table_s12_forecast(); s12.to_csv(config.OUT_TABLES / "tableS12_nsps_forecast_2030.csv", index=False); s12.to_excel(config.OUT_TABLES / "tableS12_nsps_forecast_2030.xlsx", index=False)
    s13 = build_table_s13_joint_bayes(); s13.to_csv(config.OUT_TABLES / "tableS13_joint_bayes.csv", index=False); s13.to_excel(config.OUT_TABLES / "tableS13_joint_bayes.xlsx", index=False)
    s14 = build_table_s14_who_sex_trend(); s14.to_csv(config.OUT_TABLES / "tableS14_who_sex_trend.csv", index=False); s14.to_excel(config.OUT_TABLES / "tableS14_who_sex_trend.xlsx", index=False)
    s15 = build_table_s15_state_year_recon(); s15.to_csv(config.OUT_TABLES / "tableS15_state_year_reconciliation.csv", index=False); s15.to_excel(config.OUT_TABLES / "tableS15_state_year_reconciliation.xlsx", index=False)
    s16 = build_table_s16_gbd_sex_age_apc(); s16.to_csv(config.OUT_TABLES / "tableS16_gbd_sex_age_apc.csv", index=False); s16.to_excel(config.OUT_TABLES / "tableS16_gbd_sex_age_apc.xlsx", index=False)
    s17 = build_table_s17_ses_gradient(); s17.to_csv(config.OUT_TABLES / "tableS17_ses_gradient.csv", index=False); s17.to_excel(config.OUT_TABLES / "tableS17_ses_gradient.xlsx", index=False)
    s18 = build_table_s18_state_sex_bayes(); s18.to_csv(config.OUT_TABLES / "tableS18_state_sex_bayes.csv", index=False); s18.to_excel(config.OUT_TABLES / "tableS18_state_sex_bayes.xlsx", index=False)
    print(f"[tables] Wrote 4 main tables + 9 supplement tables to {config.OUT_TABLES}")


if __name__ == "__main__":
    main()
