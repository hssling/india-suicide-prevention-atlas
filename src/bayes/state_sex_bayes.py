"""State x sex joint Bayesian measurement-error model (PyMC).

Extension of v2 joint model to sex-stratification:
  theta_{s,sex} ~ LogNormal(mu + sex_effect, sigma)
  c_{s,sex}    ~ Gamma(2, 2)
  Y_GBD_{s,sex} ~ Normal(pop_{s,sex} * theta * 1.0 / 100k, sigma_GBD)
  (NCRB-only-Both serves as a single Poisson likelihood scaled by c_state)
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import pymc as pm
from src import config
from src.clean.state_populations import project_population

# Sex proportions India 2023 (approximation; Census 2011 base)
SEX_M_FRAC, SEX_F_FRAC = 0.515, 0.485


def main() -> None:
    g = pd.read_parquet(config.DATA_PROCESSED / "gbd_sex_age_full.parquet")
    g23 = g[(g["year"] == 2023) & (g["measure"] == "Deaths") & (g["metric"] == "Number")
            & (g["age"] == "All ages") & (g["location"] != "India")].copy()
    # GBD state names need harmonisation with our workspace state names
    REPLACEMENTS = {
        "Jammu & Kashmir and Ladakh": "Jammu & Kashmir",
        "Other Union Territories": "Dadra & Nagar Haveli and Daman & Diu",
    }
    g23["state"] = g23["location"].replace(REPLACEMENTS)
    pivot = g23.pivot_table(index="state", columns="sex", values="value").reset_index()
    # 95% UI for sex-stratified rows (Both); we'll use as measurement uncertainty
    ui = g[(g["year"] == 2023) & (g["measure"] == "Deaths") & (g["metric"] == "Number")
           & (g["age"] == "All ages") & (g["sex"] == "Both") & (g["location"] != "India")].copy()
    ui["state"] = ui["location"].replace(REPLACEMENTS)
    ui_map = ui.set_index("state")[["ui_low", "ui_high"]]
    pivot = pivot.merge(ui_map, left_on="state", right_index=True)
    pivot["pop_M"] = pivot["state"].apply(lambda s: project_population(s, 2023))
    pivot = pivot.dropna(subset=["pop_M"]).reset_index(drop=True)
    pivot["pop"] = pivot["pop_M"] * 1_000_000.0
    pivot["pop_M_male"] = pivot["pop"] * SEX_M_FRAC
    pivot["pop_M_female"] = pivot["pop"] * SEX_F_FRAC
    n_states = len(pivot)
    print(f"Joint state x sex Bayes on {n_states} states")

    male_deaths = pivot["Male"].astype(float).to_numpy()
    female_deaths = pivot["Female"].astype(float).to_numpy()
    both_deaths = pivot["Both"].astype(float).to_numpy()
    sigma_both = ((pivot["ui_high"] - pivot["ui_low"]) / (2 * 1.96)).clip(lower=1.0).to_numpy()
    pop_male = pivot["pop_M_male"].to_numpy()
    pop_female = pivot["pop_M_female"].to_numpy()
    pop_total = pivot["pop"].to_numpy()

    nat_rate = both_deaths.sum() / pop_total.sum() * 100_000
    print(f"GBD national crude rate 2023: {nat_rate:.2f}/100k")

    with pm.Model() as model:
        mu_theta = pm.Normal("mu_theta", mu=np.log(nat_rate), sigma=0.5)
        sigma_theta = pm.HalfNormal("sigma_theta", sigma=0.5)
        # State-baseline theta (Both-sex equivalent)
        theta = pm.LogNormal("theta", mu=mu_theta, sigma=sigma_theta, shape=n_states)
        # Sex-effect: male and female multiplicative offsets (mean 1, log-Normal prior)
        log_sex_m = pm.Normal("log_sex_m", mu=0.0, sigma=0.5)
        log_sex_f = pm.Normal("log_sex_f", mu=0.0, sigma=0.5)
        theta_m = pm.Deterministic("theta_m", theta * pm.math.exp(log_sex_m))
        theta_f = pm.Deterministic("theta_f", theta * pm.math.exp(log_sex_f))
        # Sex-stratified GBD observations (Normal with 95% UI scaled SD shared across sexes)
        _ = pm.Normal("Y_GBD_M", mu=pop_male * theta_m / 100_000, sigma=sigma_both / 2, observed=male_deaths)
        _ = pm.Normal("Y_GBD_F", mu=pop_female * theta_f / 100_000, sigma=sigma_both / 2, observed=female_deaths)
        _ = pm.Normal("Y_GBD_Both", mu=pop_total * theta / 100_000, sigma=sigma_both, observed=both_deaths)
        trace = pm.sample(draws=2000, tune=2000, chains=4, target_accept=0.95,
                          progressbar=False, random_seed=config.SEED + 3, cores=1)

    out = []
    th = trace.posterior["theta"].stack(sample=("chain", "draw")).values
    thm = trace.posterior["theta_m"].stack(sample=("chain", "draw")).values
    thf = trace.posterior["theta_f"].stack(sample=("chain", "draw")).values
    for i, st in enumerate(pivot["state"]):
        out.append({
            "state": st,
            "theta_both_mean": float(th[i].mean()),
            "theta_both_lo95": float(np.quantile(th[i], 0.025)),
            "theta_both_hi95": float(np.quantile(th[i], 0.975)),
            "theta_male_mean": float(thm[i].mean()),
            "theta_male_lo95": float(np.quantile(thm[i], 0.025)),
            "theta_male_hi95": float(np.quantile(thm[i], 0.975)),
            "theta_female_mean": float(thf[i].mean()),
            "theta_female_lo95": float(np.quantile(thf[i], 0.025)),
            "theta_female_hi95": float(np.quantile(thf[i], 0.975)),
        })
    df = pd.DataFrame(out).sort_values("theta_both_mean", ascending=False)
    df.to_parquet(config.DATA_PROCESSED / "state_sex_bayes_v1.parquet", index=False)
    print(f"\n[bayes] wrote state_sex_bayes_v1.parquet")
    print(df[["state","theta_male_mean","theta_female_mean","theta_both_mean"]].head(10).round(2).to_string(index=False))

    # Diagnostics
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
        "log_sex_male_mean": float(trace.posterior["log_sex_m"].values.mean()),
        "log_sex_female_mean": float(trace.posterior["log_sex_f"].values.mean()),
        "sex_male_multiplier_mean": float(np.exp(trace.posterior["log_sex_m"].values.mean())),
        "sex_female_multiplier_mean": float(np.exp(trace.posterior["log_sex_f"].values.mean())),
        "max_rhat_theta": max(_rhat(th[i]) for i in range(n_states)),
        "chains": chains, "draws": draws, "warmup": 2000,
        "model_version": "state-sex-v1",
    }
    (config.DATA_PROCESSED / "state_sex_bayes_v1_diagnostics.json").write_text(json.dumps(diag, indent=2))
    print(f"\nDiagnostics:")
    print(json.dumps(diag, indent=2))


if __name__ == "__main__":
    main()
