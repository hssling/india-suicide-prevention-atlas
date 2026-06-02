"""Apply SRS-based national calibration to state-level joint Bayesian estimates.

Logic:
  - SRS provides the population-based national 'true' total (2023: ~252,000)
  - State-level joint Bayesian theta_s sums to a different national total
  - Apply a uniform scaling factor: state_calibrated_s = theta_s x (SRS_national / sum_theta_s_pop)

This produces SRS-anchored state estimates that:
  (a) preserve the relative state ranking from the joint Bayesian model
  (b) anchor the absolute total to the population-based SRS gold standard
  (c) yield a defensible NCRB/SRS-rate under-registration coefficient per state
"""
from __future__ import annotations
import pandas as pd
from src import config
from src.clean.state_populations import project_population


def main() -> None:
    # SRS 2023 anchor
    five = pd.read_csv(config.DATA_PROCESSED / "five_source_triangulation.csv")
    p23 = five[five.year == 2023].iloc[0]
    srs_2023 = float(p23.srs_count)            # 252,000
    gbd_2023 = float(p23.gbd_count)            # 200,060
    ncrb_2023 = float(p23.ncrb_count)          # 171,418
    calib = srs_2023 / gbd_2023                 # ~ 1.26
    print(f"2023 national anchors:")
    print(f"  NCRB={ncrb_2023:,.0f}  GBD={gbd_2023:,.0f}  SRS={srs_2023:,.0f}")
    print(f"  SRS/GBD calibration factor = {calib:.3f}")

    # Joint Bayes v2 state posteriors
    b = pd.read_parquet(config.DATA_PROCESSED / "suicide_joint_bayes_v2_summary.parquet")
    b["theta_calibrated_per100k"] = (b.theta_true_rate_mean_per100k * calib).round(2)
    b["theta_calibrated_lo95"] = (b.theta_true_rate_lo95 * calib).round(2)
    b["theta_calibrated_hi95"] = (b.theta_true_rate_hi95 * calib).round(2)

    # NCRB-vs-SRS-anchored under-registration coefficient
    # c_s_anchored = NCRB observed rate / SRS-anchored true rate
    b["c_anchored_to_srs"] = (b.ncrb_rate_observed_per100k / b.theta_calibrated_per100k).round(3)

    # State populations to get absolute counts
    b["population_m"] = b.state.apply(lambda s: project_population(s, 2023))
    b["theta_calibrated_count"] = (b.theta_calibrated_per100k * b.population_m * 10).round(0)
    b["ncrb_count_observed"] = (b.ncrb_rate_observed_per100k * b.population_m * 10).round(0)

    out = config.DATA_PROCESSED / "state_rates_srs_calibrated.parquet"
    b.to_parquet(out, index=False)
    print(f"\nwrote {out}")
    print(f"\n=== SRS-calibrated state rates (Top 10 by theta_calibrated) ===")
    cols = ["state", "ncrb_rate_observed_per100k", "gbd_rate_observed_per100k",
            "theta_true_rate_mean_per100k", "theta_calibrated_per100k", "c_anchored_to_srs"]
    print(b[cols].sort_values("theta_calibrated_per100k", ascending=False).head(10).to_string(index=False))

    # National under-registration check
    print(f"\n=== National under-registration vs SRS ===")
    sum_calibrated = (b.theta_calibrated_count).sum()
    sum_ncrb = b.ncrb_count_observed.sum()
    print(f"  Sum of state-calibrated theta counts: {sum_calibrated:,.0f}")
    print(f"  Sum of state NCRB-observed counts:    {sum_ncrb:,.0f}")
    print(f"  Implied national NCRB/SRS ratio:      {sum_ncrb/sum_calibrated:.3f}")
    print(f"  (Direct: NCRB national / SRS national = {ncrb_2023/srs_2023:.3f})")

    # Most under-counting + most over-counting states relative to SRS
    print(f"\n=== Top 5 most under-counting states (lowest c_anchored_to_srs) ===")
    print(b.sort_values("c_anchored_to_srs").head(5)[["state", "ncrb_rate_observed_per100k",
                                                        "theta_calibrated_per100k", "c_anchored_to_srs"]].to_string(index=False))
    print(f"\n=== Top 5 most over-counting states (highest c_anchored_to_srs) ===")
    print(b.sort_values("c_anchored_to_srs", ascending=False).head(5)[["state", "ncrb_rate_observed_per100k",
                                                                          "theta_calibrated_per100k", "c_anchored_to_srs"]].to_string(index=False))


if __name__ == "__main__":
    main()
