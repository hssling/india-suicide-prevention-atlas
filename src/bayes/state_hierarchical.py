"""Phase 4 — Bayesian hierarchical model of state suicide rates.

Estimates the true state suicide rate as a partial-pooling Bayesian model:

  ncrb_count_s ~ Poisson(pop_s * theta_s)
  log(theta_s) = alpha + u_s
  u_s          ~ Normal(0, sigma)
  alpha        ~ Normal(log(national_gbd_rate), 0.5)
  sigma        ~ HalfNormal(0.5)

The posterior of theta_s provides shrinkage-stabilised state suicide rates with
95% credible intervals. The implied posterior of the NCRB-vs-GBD ratio is
also reported.

Outputs:
  data/processed/suicide_bayes_summary.parquet — posterior mean / 95% CrI per state
  data/processed/suicide_bayes_diagnostics.json — R-hat, ESS, summary diagnostics
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from src import config
from src.analysis.avertable_burden import STATE_POP_2023_MILLIONS


def main() -> None:
    panel = pd.read_parquet(config.DATA_PROCESSED / "suicide_reconciliation_panel.parquet")
    p23 = panel[panel["year"] == 2023].dropna(subset=["ncrb_suicides", "gbd_suicides"]).copy()
    p23["population"] = p23["state"].map(STATE_POP_2023_MILLIONS) * 1_000_000.0
    p23 = p23.dropna(subset=["population"]).reset_index(drop=True)

    counts = p23["ncrb_suicides"].astype(int).to_numpy()
    pop = p23["population"].astype(float).to_numpy()

    # GBD national rate as informative prior centre
    national_rate = p23["gbd_suicides"].sum() / pop.sum()

    print(f"Fitting Bayesian hierarchical model on n={len(p23)} states ...")
    with pm.Model() as model:
        alpha = pm.Normal("alpha", mu=np.log(national_rate), sigma=0.5)
        sigma = pm.HalfNormal("sigma", sigma=0.5)
        u = pm.Normal("u", mu=0.0, sigma=sigma, shape=len(p23))
        log_theta = alpha + u
        theta = pm.Deterministic("theta", pm.math.exp(log_theta))
        rate_per_100k = pm.Deterministic("rate_per_100k", theta * 100_000)
        _ = pm.Poisson("y", mu=pop * theta, observed=counts)

        trace = pm.sample(
            draws=4000,
            tune=2000,
            chains=4,
            target_accept=0.98,
            progressbar=False,
            random_seed=config.SEED,
            cores=1,  # Windows compatibility
        )

    # Diagnostics
    summary = az.summary(trace, var_names=["alpha", "sigma", "rate_per_100k"])
    max_rhat = float(summary["r_hat"].max())
    min_ess = float(summary["ess_bulk"].min())
    print(f"[bayes] max R-hat: {max_rhat:.4f} (target < {config.BAYES_TARGET_RHAT})")
    print(f"[bayes] min ESS:   {min_ess:.0f} (target > {config.BAYES_TARGET_ESS})")

    # Per-state posterior summary — compute 95% HDI directly (arviz column names vary by version)
    rate_samples = trace.posterior["rate_per_100k"].stack(sample=("chain", "draw")).values  # shape (n_states, n_samples)
    hdi_95 = np.column_stack([np.quantile(rate_samples, 0.025, axis=1), np.quantile(rate_samples, 0.975, axis=1)])  # shape (n_states, 2)
    rows = []
    for i, st in enumerate(p23["state"]):
        rows.append({
            "state": st,
            "ncrb_observed_rate_per100k": float(counts[i] / pop[i] * 100_000),
            "gbd_rate_per100k": float(p23.iloc[i]["gbd_suicides"] / pop[i] * 100_000),
            "bayes_rate_mean_per100k": float(rate_samples[i].mean()),
            "bayes_rate_lo95": float(hdi_95[i, 0]),
            "bayes_rate_hi95": float(hdi_95[i, 1]),
        })
    out = pd.DataFrame(rows)
    out_path = config.DATA_PROCESSED / "suicide_bayes_summary.parquet"
    out.to_parquet(out_path, index=False)
    print(f"[bayes] wrote {out_path}  rows={len(out)}")

    alpha_samples = trace.posterior["alpha"].values.flatten()
    sigma_samples = trace.posterior["sigma"].values.flatten()
    alpha_hdi = np.quantile(alpha_samples, [0.025, 0.975])
    sigma_hdi = np.quantile(sigma_samples, [0.025, 0.975])
    diag = {
        "n_states": int(len(p23)),
        "national_rate_used_per100k": float(national_rate * 100_000),
        "alpha_mean": float(alpha_samples.mean()),
        "alpha_hdi95": [float(alpha_hdi[0]), float(alpha_hdi[1])],
        "sigma_mean": float(sigma_samples.mean()),
        "sigma_hdi95": [float(sigma_hdi[0]), float(sigma_hdi[1])],
        "max_r_hat": max_rhat,
        "min_ess_bulk": min_ess,
        "convergence_ok": bool(max_rhat <= config.BAYES_TARGET_RHAT and min_ess > 500),
        "chains": int(trace.posterior.sizes["chain"]),
        "warmup": 2000,  # tune param actually used
        "draws": int(trace.posterior.sizes["draw"]),
        "seed": config.SEED,
        "target_accept": 0.98,
    }
    diag_path = config.DATA_PROCESSED / "suicide_bayes_diagnostics.json"
    diag_path.write_text(json.dumps(diag, indent=2))
    print(f"[bayes] wrote {diag_path}")

    print("\nHeadline state-level posterior rates (per 100,000):")
    print(out.sort_values("bayes_rate_mean_per100k", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
