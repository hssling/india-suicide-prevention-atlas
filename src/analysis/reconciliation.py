"""Phase 2.1 — NCRB-vs-GBD state-level suicide reconciliation."""
from __future__ import annotations
import pandas as pd
from src import config


def build_reconciliation_panel() -> pd.DataFrame:
    """Build the per-state 2023 NCRB-vs-GBD reconciliation panel."""
    ncrb = pd.read_csv(config.EXISTING_NCRB_CSV)
    ncrb = ncrb[ncrb["cause_ncrb"] == "suicide_total"].copy()
    ncrb = ncrb[["state_name_harmonized", "year", "deaths_n"]].rename(
        columns={"deaths_n": "ncrb_suicides"}
    )

    gbd = pd.read_csv(config.EXISTING_GBD_HARMONIZED)
    gbd = gbd[
        (gbd["cause_gbd"] == "Self-harm")
        & (gbd["measure"] == "Deaths")
        & (gbd["metric_type"] == "Number")
        & (gbd["sex"] == "Both")
        & (gbd["age_group"] == "All ages")
    ].copy()
    gbd = gbd[["state_name_harmonized", "year", "value", "lower_ui", "upper_ui"]].rename(
        columns={"value": "gbd_suicides", "lower_ui": "gbd_lo", "upper_ui": "gbd_hi"}
    )

    panel = ncrb.merge(gbd, on=["state_name_harmonized", "year"], how="outer")
    panel = panel.rename(columns={"state_name_harmonized": "state"})
    panel["ratio_ncrb_gbd"] = panel["ncrb_suicides"] / panel["gbd_suicides"]
    panel["log_ratio"] = panel["ratio_ncrb_gbd"].apply(
        lambda x: None if pd.isna(x) or x <= 0 else __import__("math").log(x)
    )
    return panel.sort_values(["state", "year"]).reset_index(drop=True)


def summarise_reconciliation(panel: pd.DataFrame) -> dict:
    both = panel.dropna(subset=["ncrb_suicides", "gbd_suicides"])
    p23 = both[both["year"] == 2023]
    return {
        "n_states_with_both_2023": int(len(p23)),
        "national_ncrb_2023": float(p23["ncrb_suicides"].sum()),
        "national_gbd_2023": float(p23["gbd_suicides"].sum()),
        "national_ratio_2023": float(p23["ncrb_suicides"].sum() / p23["gbd_suicides"].sum()),
        "state_ratio_median_2023": float(p23["ratio_ncrb_gbd"].median()),
        "state_ratio_iqr_2023": (
            float(p23["ratio_ncrb_gbd"].quantile(0.25)),
            float(p23["ratio_ncrb_gbd"].quantile(0.75)),
        ),
        "state_with_lowest_ratio": p23.loc[p23["ratio_ncrb_gbd"].idxmin(), "state"]
        if len(p23) > 0
        else None,
        "lowest_ratio_value": float(p23["ratio_ncrb_gbd"].min()),
        "state_with_highest_ratio": p23.loc[p23["ratio_ncrb_gbd"].idxmax(), "state"]
        if len(p23) > 0
        else None,
        "highest_ratio_value": float(p23["ratio_ncrb_gbd"].max()),
    }


def main() -> None:
    panel = build_reconciliation_panel()
    out_path = config.DATA_PROCESSED / "suicide_reconciliation_panel.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(out_path, index=False)
    summary = summarise_reconciliation(panel)
    print(f"[reconciliation] wrote {out_path}  rows={len(panel)}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
