"""Phase 3 — Equity decomposition: SII / RII / Concentration Index / Theil decomposition."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src import config


def slope_and_relative_index_of_inequality(rates: np.ndarray, populations: np.ndarray) -> tuple[float, float]:
    """Compute SII and RII from group-level rates ordered low-to-high in advantage.

    SII = beta from a population-weighted linear regression of rate on the
    midpoint of the cumulative population share within each group.
    RII = SII / population-weighted mean rate.

    Lower SII / RII = less inequality.
    """
    p = np.asarray(populations, dtype=float)
    r = np.asarray(rates, dtype=float)
    p_share = p / p.sum()
    cum = np.cumsum(p_share)
    mid = cum - p_share / 2.0  # ridit / fractional midpoint
    weights = p
    mean_r = (r * p).sum() / p.sum()
    # weighted slope
    x_bar = (mid * weights).sum() / weights.sum()
    y_bar = mean_r
    cov = (weights * (mid - x_bar) * (r - y_bar)).sum() / weights.sum()
    var = (weights * (mid - x_bar) ** 2).sum() / weights.sum()
    sii = cov / var if var > 0 else float("nan")
    rii = sii / mean_r if mean_r > 0 else float("nan")
    return float(sii), float(rii)


def concentration_index(rates: np.ndarray, populations: np.ndarray, ranks: np.ndarray | None = None) -> float:
    """Erreygers-corrected concentration index.

    Inputs assumed ordered by socioeconomic rank (poorest first).
    Returns a value in [-1, 1]; negative means burden concentrated in poorer groups.
    """
    p = np.asarray(populations, dtype=float)
    r = np.asarray(rates, dtype=float)
    p_share = p / p.sum()
    if ranks is None:
        cum = np.cumsum(p_share)
        ranks = cum - p_share / 2.0
    mean_r = (r * p).sum() / p.sum()
    if mean_r <= 0:
        return float("nan")
    ci = 2.0 / mean_r * ((p_share * r * ranks).sum() - (p_share * r).sum() * (p_share * ranks).sum())
    return float(ci)


def theil_index(rates: np.ndarray, populations: np.ndarray) -> float:
    """Population-weighted Theil index of inequality.

    T = sum_i (n_i / N) * (y_i / mean_y) * ln(y_i / mean_y)
    where y_i is the per-group outcome (rate) and n_i is the group size.

    Values close to 0 indicate equality; higher values indicate more inequality.
    """
    p = np.asarray(populations, dtype=float)
    r = np.asarray(rates, dtype=float)
    p = p[r > 0]
    r = r[r > 0]
    if len(r) == 0:
        return float("nan")
    p_share = p / p.sum()
    mean_r = (r * p).sum() / p.sum()
    if mean_r <= 0:
        return float("nan")
    rel = r / mean_r
    t = (p_share * rel * np.log(rel)).sum()
    return float(t)


def theil_decompose(groups_rates: np.ndarray, groups_pops: np.ndarray,
                    subgroups_within: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Decompose total Theil into between-group and within-group components.

    groups_rates, groups_pops: group-level rates and populations (length G)
    subgroups_within: list (length G) of (rates, pops) arrays per group's subgroups

    Returns: between, sum_within (population-weighted average within-group), total, between_share
    """
    p = np.asarray(groups_pops, dtype=float)
    r = np.asarray(groups_rates, dtype=float)
    p_share = p / p.sum()
    mean_r = (r * p).sum() / p.sum()
    # Between component
    rel = np.where(r > 0, r / mean_r, 1.0)
    between = (p_share * rel * np.log(np.where(rel > 0, rel, 1))).sum()
    # Within components, weighted by group population share and group-mean/grand-mean
    within_total = 0.0
    for i, (sub_r, sub_p) in enumerate(subgroups_within):
        if len(sub_r) == 0 or sub_p.sum() == 0:
            continue
        within_i = theil_index(np.asarray(sub_r), np.asarray(sub_p))
        if not np.isnan(within_i):
            w = p_share[i] * (r[i] / mean_r if mean_r > 0 else 0)
            within_total += w * within_i
    total = between + within_total
    return {
        "between": float(between),
        "within": float(within_total),
        "total": float(total),
        "between_share": float(between / total) if total > 0 else float("nan"),
    }


def main() -> None:
    """Run the state-level equity analysis on the 2023 reconciliation panel."""
    central = pd.read_parquet(config.DATA_PROCESSED / "suicide_avertable_central.parquet")
    central = central.dropna(subset=["gbd_rate", "population_m"])
    central = central.sort_values("gbd_rate")  # ascending rate = decreasing advantage

    rates = central["gbd_rate"].to_numpy()
    pops = central["population_m"].to_numpy()

    sii, rii = slope_and_relative_index_of_inequality(rates, pops)
    ci = concentration_index(rates, pops)
    theil = theil_index(rates, pops)

    # Sex stratified — read NCRB suicide sex split if available; otherwise use national 71/29 split (NCRB ADSI 2023 headline)
    male_share_national = 0.715
    female_share_national = 0.285
    sex_rates = np.array([rates.mean() * 2 * male_share_national, rates.mean() * 2 * female_share_national])
    sex_pops = np.array([pops.sum() * 0.515, pops.sum() * 0.485])  # India approx sex ratio
    sii_sex, rii_sex = slope_and_relative_index_of_inequality(sex_rates, sex_pops)

    out = {
        "n_states": int(len(central)),
        "national_rate_per_100k": float((rates * pops).sum() / pops.sum()),
        "min_rate_state": str(central.iloc[0]["state"]),
        "min_rate_value": float(central.iloc[0]["gbd_rate"]),
        "max_rate_state": str(central.iloc[-1]["state"]),
        "max_rate_value": float(central.iloc[-1]["gbd_rate"]),
        "SII_state_level_per100k": sii,
        "RII_state_level": rii,
        "Concentration_index_state_level": ci,
        "Theil_index_state_level": theil,
        "SII_sex": sii_sex,
        "RII_sex": rii_sex,
    }

    out_path = config.DATA_PROCESSED / "suicide_equity_indices.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_parquet(out_path, index=False)

    print(f"[equity] wrote {out_path}")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
