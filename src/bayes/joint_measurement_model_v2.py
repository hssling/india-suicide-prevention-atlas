"""Joint Bayesian measurement-error model v2 — Gamma prior on c_s allowing values > 1.

Replaces the Beta(2,1) prior of v1 with a Gamma(2, 2) prior centred on
c_s = 1 with positive support, so states where NCRB > GBD modelled
(Sikkim, Delhi, Chhattisgarh) are not capped at 1.0.

Model:
    theta_s   ~ LogNormal(mu_theta, sigma_theta^2)
    c_s       ~ Gamma(2, 2)                                # mean 1, sd 0.71, mode 0.5; allows c>1
    Y_NCRB,s  ~ Poisson(pop_s * theta_s * c_s / 100,000)
    Y_GBD,s   ~ Normal(pop_s * theta_s / 100,000, sigma_GBD,s^2)
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pymc as pm
from src import config
from src.clean.state_populations import project_population


def main() -> None:
    panel = pd.read_parquet(config.DATA_PROCESSED / "suicide_reconciliation_panel.parquet")
    p23 = panel[panel["year"] == 2023].dropna(subset=["ncrb_suicides", "gbd_suicides", "gbd_lo", "gbd_hi"]).copy()
    p23["population"] = p23["state"].apply(lambda s: project_population(s, 2023))
    p23["population"] = p23["population"] * 1_000_000.0
    p23 = p23.dropna(subset=["population"]).reset_index(drop=True)

    counts_ncrb = p23["ncrb_suicides"].astype(int).to_numpy()
    counts_gbd = p23["gbd_suicides"].to_numpy()
    pop = p23["population"].to_numpy()
    sigma_gbd = (p23["gbd_hi"].to_numpy() - p23["gbd_lo"].to_numpy()) / (2 * 1.96)
    sigma_gbd = np.maximum(sigma_gbd, 1.0)

    n_states = len(p23)
    national_rate_per_100k = (p23["gbd_suicides"].sum() / p23["population"].sum()) * 100_000

    print(f"Joint Bayes v2 (Gamma prior on c) on n={n_states} states")
    print(f"GBD national rate (per 100k) used as prior centre: {national_rate_per_100k:.2f}")

    with pm.Model() as model:
        mu_theta = pm.Normal("mu_theta", mu=np.log(national_rate_per_100k), sigma=0.5)
        sigma_theta = pm.HalfNormal("sigma_theta", sigma=0.5)
        theta = pm.LogNormal("theta", mu=mu_theta, sigma=sigma_theta, shape=n_states)
        # Gamma(2, 2) -> mean 1, sd 0.71, mode 0.5; supports c in (0, inf)
        c = pm.Gamma("c", alpha=2.0, beta=2.0, shape=n_states)
        ncrb_expected = pm.Deterministic("ncrb_expected", pop * theta * c / 100_000.0)
        gbd_expected = pm.Deterministic("gbd_expected", pop * theta / 100_000.0)
        _ = pm.Poisson("Y_NCRB", mu=ncrb_expected, observed=counts_ncrb)
        _ = pm.Normal("Y_GBD", mu=gbd_expected, sigma=sigma_gbd, observed=counts_gbd)

        trace = pm.sample(
            draws=2000, tune=2000, chains=4, target_accept=0.95,
            progressbar=False, random_seed=config.SEED + 2, cores=1,
        )

    theta_samples = trace.posterior["theta"].stack(sample=("chain", "draw")).values
    c_samples = trace.posterior["c"].stack(sample=("chain", "draw")).values

    rows = []
    for i, st in enumerate(p23["state"]):
        rows.append({
            "state": st,
            "ncrb_observed": int(counts_ncrb[i]),
            "gbd_observed": float(counts_gbd[i]),
            "population_2023": float(pop[i]),
            "ncrb_rate_observed_per100k": float(counts_ncrb[i] / pop[i] * 100_000),
            "gbd_rate_observed_per100k": float(counts_gbd[i] / pop[i] * 100_000),
            "theta_true_rate_mean_per100k": float(theta_samples[i].mean()),
            "theta_true_rate_lo95": float(np.quantile(theta_samples[i], 0.025)),
            "theta_true_rate_hi95": float(np.quantile(theta_samples[i], 0.975)),
            "c_underreg_mean": float(c_samples[i].mean()),
            "c_underreg_lo95": float(np.quantile(c_samples[i], 0.025)),
            "c_underreg_hi95": float(np.quantile(c_samples[i], 0.975)),
        })
    out = pd.DataFrame(rows).sort_values("theta_true_rate_mean_per100k", ascending=False)
    out_path = config.DATA_PROCESSED / "suicide_joint_bayes_v2_summary.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n[joint_bayes_v2] wrote {out_path}")

    mu_s = trace.posterior["mu_theta"].values.flatten()
    sg_s = trace.posterior["sigma_theta"].values.flatten()
    chains = int(trace.posterior.sizes["chain"])
    draws = int(trace.posterior.sizes["draw"])

    def _rhat(samples):
        if chains < 2: return 1.0
        s = samples.reshape(chains, draws)
        m = s.mean(axis=1); v = s.var(axis=1, ddof=1)
        W = v.mean(); B = draws * m.var(ddof=1)
        if W <= 0: return 1.0
        return float(np.sqrt(((1 - 1/draws) * W + (1/draws) * B) / W))

    diag = {
        "n_states": n_states,
        "national_rate_used_per100k": float(national_rate_per_100k),
        "mu_theta_mean": float(mu_s.mean()),
        "mu_theta_95ci": [float(np.quantile(mu_s, 0.025)), float(np.quantile(mu_s, 0.975))],
        "sigma_theta_mean": float(sg_s.mean()),
        "sigma_theta_95ci": [float(np.quantile(sg_s, 0.025)), float(np.quantile(sg_s, 0.975))],
        "max_rhat_theta": max(_rhat(theta_samples[i]) for i in range(n_states)),
        "max_rhat_c": max(_rhat(c_samples[i]) for i in range(n_states)),
        "chains": chains, "warmup": 2000, "draws": draws,
        "seed": config.SEED + 2,
        "c_prior": "Gamma(2, 2): mean 1, supports c in (0, inf)",
        "model_version": "v2-gamma-prior-on-c",
    }
    (config.DATA_PROCESSED / "suicide_joint_bayes_v2_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(f"\nv2 diagnostics:")
    print(json.dumps(diag, indent=2))

    print(f"\nStates with c > 1.0 (NCRB > true) under v2 Gamma prior:")
    over = out[out["c_underreg_mean"] > 1.0].sort_values("c_underreg_mean", ascending=False)
    print(over[["state", "ncrb_rate_observed_per100k", "gbd_rate_observed_per100k",
                "theta_true_rate_mean_per100k", "c_underreg_mean", "c_underreg_lo95", "c_underreg_hi95"]].to_string(index=False))
    print(f"\nMean c_s (v2): {out['c_underreg_mean'].mean():.3f}; range [{out['c_underreg_mean'].min():.3f}, {out['c_underreg_mean'].max():.3f}]")


if __name__ == "__main__":
    main()
