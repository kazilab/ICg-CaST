"""Cohort generators for ICg-Bench DGP variants.

For v0.1 we provide two generators:

- ``generate_linear_lowhet``: a thin wrapper around the package simulator
  reproducing the discrete-archetype, low-host-heterogeneity baseline.
- ``generate_nonlinear_mixhost``: continuous KCC mixtures sampled from a
  Dirichlet over the archetype space, non-linear KCC-to-state coupling, and
  widened host susceptibility distributions.

Both generators emit cohort rows with the same schema as the core simulator so
downstream evaluation code (MB-CNet, ICg-Bench tasks) is variant-agnostic.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit

from ..config import SimConfig
from ..constants import ARCHETYPE_KCC, ARCHETYPE_SIGNATURE, EPI_MODULES, KCC_NAMES, TRANSCRIPT_MODULES
from ..omics import generate_omics
from ..signatures import make_signature_profiles
from ..simulator import (
    noisy_kcc_vector,
    sample_archetype,
    simulate_cohort,
    simulate_state_trajectory,
    summarize_trajectory,
)


def generate_linear_lowhet(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Reproduce the starter-kit cohort (discrete archetypes, low heterogeneity)."""
    cfg = SimConfig(n=n, months=months, seed=seed)
    return simulate_cohort(cfg)


def _sample_continuous_kcc(
    archetype_kcc: dict[str, Sequence[float]],
    rng: np.random.Generator,
    alpha: float = 0.6,
) -> tuple[str, np.ndarray]:
    """Sample a continuous KCC vector as a Dirichlet mixture of archetype profiles.

    The dominant archetype (highest mixing weight) is returned for downstream
    interpretability of expected signatures.
    """
    names = list(archetype_kcc.keys())
    weights = rng.dirichlet(np.full(len(names), alpha))
    base = np.array([archetype_kcc[a] for a in names])
    kcc = (weights[:, None] * base).sum(axis=0)
    noise = rng.normal(0.0, 0.07, size=kcc.size)
    kcc = np.clip(kcc + noise, 0.0, 1.0)
    dominant = names[int(np.argmax(weights))]
    return dominant, kcc


def _simulate_state_trajectory_nonlinear(
    kcc: np.ndarray,
    dose: float,
    susceptibility: dict[str, float],
    months: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """qAOP recursion with non-linear KCC couplings.

    Differs from the starter kit by (i) squared KCC terms in the DNA-adduct and
    ROS inputs, (ii) a saturating tanh non-linearity on proliferation, (iii) a
    multiplicative KCC4*KCC5 interaction on epigenetic age, and (iv) a stronger
    clone-fraction selection pressure. The structural-equation form of
    `latent_risk` is preserved so the same `starter_kit_latent_risk` function
    can be reused for intervention-augmented training.
    """
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = kcc
    repair_capacity = float(np.clip(susceptibility["repair_capacity"] * (1.0 - 0.35 * k3), 0.15, 1.2))
    antioxidant = float(susceptibility["antioxidant_capacity"])
    immune = float(np.clip(susceptibility["immune_surveillance"] * (1.0 - 0.45 * k7), 0.05, 1.2))
    detox = float(susceptibility["detox_balance"])
    baseline_prolif = float(susceptibility["baseline_proliferation"])

    DNA = 0.0
    ROS = 0.0
    inflammation = 0.0
    epigenetic_age = 0.0
    clone_fraction = 1e-5 * float(rng.lognormal(mean=0.0, sigma=0.45))
    driver_count_proxy = 0.0

    rows: list[dict[str, float | int]] = []
    for t in range(months):
        cyclical = 0.85 + 0.15 * math.sin(2.0 * math.pi * t / 12.0)
        pulse = float(rng.lognormal(mean=-0.02, sigma=0.18))
        internal_dose = dose * cyclical * pulse * detox

        adduct_input = internal_dose * (0.20 + 1.35 * (k1 ** 1.4)) * (0.25 + 1.10 * (k2 ** 1.3))
        DNA = 0.68 * DNA + adduct_input - 0.36 * repair_capacity * DNA
        DNA = float(np.clip(DNA, 0.0, 20.0))

        ros_input = internal_dose * (0.10 + 1.45 * (k5 ** 1.3) + 0.35 * (k1 ** 1.2))
        ROS = 0.74 * ROS + ros_input - 0.42 * antioxidant * ROS
        ROS = float(np.clip(ROS, 0.0, 20.0))

        cytotoxicity = float(expit(-2.2 + 0.35 * DNA + 0.25 * ROS))
        inflammation = (
            0.80 * inflammation
            + 0.10 * ROS
            + 0.35 * cytotoxicity
            + 0.95 * internal_dose * (k6 ** 1.2)
            - 0.24 * immune * inflammation
        )
        inflammation = float(np.clip(inflammation, 0.0, 20.0))

        epigenetic_age += (
            0.011
            + internal_dose * (0.055 * k4 + 0.018 * k5 + 0.018 * k6)
            + 0.040 * (k4 * k5)
            + 0.003 * ROS
        )
        epigenetic_age = float(np.clip(epigenetic_age, 0.0, 50.0))

        proliferation = float(
            np.tanh(
                -0.6
                + baseline_prolif
                + 0.27 * inflammation
                + 0.20 * cytotoxicity
                + 1.15 * k8 * internal_dose
                + 0.90 * k10
                + 0.16 * epigenetic_age
            )
            * 0.5
            + 0.5
        )

        mutation_rate = 0.0012 * (
            1.0
            + 4.8 * DNA * k2
            + 1.2 * ROS
            + 2.0 * (1.0 - repair_capacity)
            + 0.9 * proliferation
        )
        mutation_rate = float(np.clip(mutation_rate, 0.0, 1.0))

        driver_count_proxy += mutation_rate * (0.35 + 2.0 * proliferation) * (1.0 + 0.35 * k9)

        selection = (
            -0.020
            + 0.36 * proliferation
            + 0.080 * inflammation
            + 0.050 * epigenetic_age
            + 0.22 * k9
            - 0.26 * immune
            + 0.10 * driver_count_proxy
        )
        noise = float(rng.normal(0.0, 0.006))
        clone_fraction = clone_fraction + clone_fraction * (selection * (1.0 - clone_fraction) + noise)
        clone_fraction = float(np.clip(clone_fraction, 1e-8, 0.99))

        latent_risk = float(
            expit(
                -7.5
                + 1.20 * math.log1p(DNA)
                + 0.70 * math.log1p(ROS)
                + 0.95 * math.log1p(inflammation)
                + 1.50 * proliferation
                + 0.85 * math.log1p(epigenetic_age)
                + 1.80 * math.log1p(driver_count_proxy)
                + 4.50 * clone_fraction
                - 0.90 * immune
            )
        )
        rows.append(
            {
                "month": t + 1,
                "DNA_adducts": DNA,
                "ROS": ROS,
                "inflammation": inflammation,
                "epigenetic_age": epigenetic_age,
                "proliferation": proliferation,
                "mutation_rate": mutation_rate,
                "clone_fraction": clone_fraction,
                "driver_count_proxy": driver_count_proxy,
                "immune_clearance": immune,
                "latent_risk": latent_risk,
            }
        )
    return pd.DataFrame(rows)


def generate_nonlinear_mixhost(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Generate the nonlinear_mixhost ICg-Bench cohort.

    - Continuous KCC vectors via Dirichlet mixtures of the 8 archetypes.
    - Non-linear KCC -> state coupling (powers, interactions, tanh saturation).
    - Wider host susceptibility distributions (high heterogeneity).

    The output schema matches `icg_cast.simulate_cohort` so downstream code is
    unchanged.
    """
    rng = np.random.default_rng(seed)
    sig_labels, sig_profiles = make_signature_profiles()

    rows: list[dict[str, float | int]] = []
    trajectories: dict[str, pd.DataFrame] = {}

    for i in range(n):
        archetype, kcc = _sample_continuous_kcc(ARCHETYPE_KCC, rng, alpha=0.5)
        dose = float(np.clip(rng.lognormal(mean=-0.10, sigma=1.10), 0.02, 8.0))
        susceptibility = {
            "repair_capacity":         float(np.clip(rng.normal(0.82, 0.28), 0.25, 1.40)),
            "antioxidant_capacity":    float(np.clip(rng.normal(0.85, 0.27), 0.25, 1.40)),
            "immune_surveillance":     float(np.clip(rng.normal(0.82, 0.32), 0.15, 1.50)),
            "detox_balance":           float(np.clip(rng.lognormal(mean=0.0, sigma=0.35), 0.45, 1.95)),
            "baseline_proliferation":  float(np.clip(rng.normal(0.0, 0.85), -1.8, 1.8)),
        }

        traj = _simulate_state_trajectory_nonlinear(kcc, dose, susceptibility, months, rng)
        states = summarize_trajectory(traj)
        omics = generate_omics(
            archetype, kcc, states, susceptibility, rng, sig_labels, sig_profiles,
        )

        final_risk = states["state_final_latent_risk"]
        cumulative_hazard = 0.020 * months * final_risk
        event_probability = 1.0 - math.exp(-cumulative_hazard)
        event = int(rng.uniform() < event_probability)

        row: dict[str, float | int] = {
            "sample_id": f"ICG_NLM_{i:05d}",
            "chemical_archetype": archetype,
            "dose": dose,
            "future_cancer_transition_event": event,
            "future_event_probability": event_probability,
        }
        for j, name in enumerate(KCC_NAMES, start=1):
            row[f"kcc{j}_{name}"] = float(kcc[j - 1])
        for key, val in susceptibility.items():
            row[f"host_{key}"] = val
        row.update(states)
        row.update(omics)
        rows.append(row)

        if len(trajectories) < 8 and archetype not in trajectories:
            trajectories[archetype] = traj.assign(sample_id=row["sample_id"], chemical_archetype=archetype)

    df = pd.DataFrame(rows)
    cutoff = float(df["state_final_latent_risk"].quantile(0.72))
    df["high_risk_transition_state"] = (df["state_final_latent_risk"] >= cutoff).astype(int)
    return df, trajectories


def generate_partial_observability(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
    masking_rate: float = 0.30,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """`nonlinear_mixhost` cohort with per-subject random masking of omics modules.

    Each `tx_*` / `epi_*` column is independently set to NaN with probability
    ``masking_rate`` per subject (Bernoulli sampling). KCC vectors, host
    susceptibility, mutational-signature features, and the qAOP state summaries
    are preserved unchanged. The latent_risk structural equation is unchanged.
    """
    df, traj = generate_nonlinear_mixhost(n=n, months=months, seed=seed)
    rng = np.random.default_rng(seed + 1000)
    omics_cols = [c for c in df.columns if c.startswith(("tx_", "epi_"))]
    mask = rng.uniform(size=(len(df), len(omics_cols))) < masking_rate
    arr = df[omics_cols].to_numpy(dtype=float)
    arr[mask] = np.nan
    df[omics_cols] = arr
    return df, traj


def _nonlinear_obs_generate_omics(
    archetype: str,
    kcc: np.ndarray,
    states: dict[str, float],
    susceptibility: dict[str, float],
    rng: np.random.Generator,
    signature_labels: Iterable[str],
    signature_profiles: dict[str, np.ndarray],
) -> dict[str, float]:
    """Like the starter kit's `generate_omics`, but with non-linear state transformations.

    The observed omics are computed from saturating / log-transformed / interacting
    state quantities rather than from the raw state AUCs. The latent state itself
    is unchanged; only the observation operator differs. This is the harder inverse
    problem for stage 1 of MB-CNet.
    """
    out: dict[str, float] = {}
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = kcc

    DNA_raw = states["state_auc_DNA_adducts"]
    ROS_raw = states["state_auc_ROS"]
    inflam_raw = states["state_auc_inflammation"]
    epi_age_raw = states["state_final_epigenetic_age"]
    prolif_raw = states["state_auc_proliferation"]
    mut_rate_raw = states["state_auc_mutation_rate"]
    clone_raw = states["state_final_clone_fraction"]
    drivers_raw = states["state_final_driver_count_proxy"]

    # Non-linear transformations of the latent state.
    DNA = float(np.tanh(DNA_raw / 4.0))
    ROS = float(np.log1p(np.maximum(ROS_raw, 0.0)))
    inflam = float(np.tanh(inflam_raw / 3.0))
    epi_age = float(np.log1p(np.maximum(epi_age_raw, 0.0)))
    prolif = float(prolif_raw ** 1.5)
    clone = float(np.sqrt(np.maximum(clone_raw, 0.0)))
    drivers = float(np.log1p(np.maximum(drivers_raw, 0.0)))

    # State-state interactions used selectively in the transcript modules.
    inflam_x_ROS = float(inflam * ROS)
    clone_x_inflam = float(clone * inflam)

    tx_values = {
        "DNA_adduct_response":         0.8 * DNA + 1.2 * k1 + 1.0 * k2,
        "p53_checkpoint":              0.6 * DNA + 0.7 * ROS + 0.8 * k2,
        "base_excision_repair":        0.6 * ROS - 0.8 * k3 + 0.3 * susceptibility["repair_capacity"],
        "nucleotide_excision_repair":  0.7 * DNA - 0.7 * k3 + 0.2 * susceptibility["repair_capacity"],
        "xenobiotic_metabolism_CYP":   1.3 * k1 + 0.8 * k8 + 0.5 * DNA,
        "oxidative_stress_response":   1.1 * ROS + 1.0 * k5,
        "mitochondrial_dysfunction":   0.9 * ROS + 0.4 * inflam_x_ROS,
        "inflammatory_cytokines":      1.2 * inflam + 0.8 * k6,
        "NFkB_activation":             0.9 * inflam + 0.5 * ROS + 0.6 * k6,
        "cell_cycle_E2F":              1.8 * prolif + 0.7 * k10 + 0.2 * clone,
        "replicative_DNA_synthesis":   1.5 * prolif + 0.5 * k8 + 0.7 * k10,
        "apoptosis_escape":            0.7 * k9 + 0.8 * clone_x_inflam + 0.3 * inflam,
        "ECM_fibrosis":                0.9 * inflam + 0.8 * epi_age + 0.4 * k4,
        "angiogenesis_nutrient_supply": 0.6 * prolif + 0.7 * inflam + 0.5 * k10,
        "immune_evasion":              0.9 * k7 + 0.7 * clone - 0.5 * susceptibility["immune_surveillance"],
        "nuclear_receptor_program":    1.6 * k8 + 0.5 * prolif,
        "stemness_PRC2":               0.9 * epi_age + 0.6 * k4 + 0.3 * drivers,
        "senescence_SASP":             0.4 * ROS + 0.9 * inflam + 0.4 * epi_age,
    }
    for name in TRANSCRIPT_MODULES:
        val = tx_values[name] + rng.normal(0.0, 0.30)
        out[f"tx_{name}"] = float(val)

    epi_values = {
        "epigenetic_age_acceleration":      epi_age + 0.5 * k4 + 0.2 * ROS,
        "PRC2_mitotic_clock":               epi_age * (0.75 + 0.8 * prolif) + 0.3 * drivers,
        "global_hypomethylation":           0.5 * epi_age + 0.7 * k4 + 0.25 * ROS,
        "tumor_suppressor_hypermethylation": 0.6 * epi_age + 0.6 * k4 + 0.35 * clone,
        "enhancer_reprogramming":           0.6 * k4 + 0.4 * k8 + 0.25 * inflam,
        "histone_activation_loss":          0.8 * k4 + 0.4 * epi_age,
        "histone_repression_gain":          0.9 * k4 + 0.3 * epi_age + 0.25 * drivers,
        "chromatin_accessibility_shift":    0.4 * k4 + 0.3 * ROS + 0.3 * k8,
    }
    for name in EPI_MODULES:
        out[f"epi_{name}"] = float(epi_values[name] + rng.normal(0.0, 0.22))

    # Mutational signature mixture identical to the starter kit (we are not
    # perturbing the signature observation operator in this variant).
    primary_sig = ARCHETYPE_SIGNATURE[archetype]
    sig_mix = 0.45 * signature_profiles["aging"].copy()
    primary_weight = float(np.clip(0.1 + 0.40 * k2 + 0.12 * DNA + 0.06 * ROS, 0.05, 0.85))
    sig_mix = (1.0 - primary_weight) * sig_mix + primary_weight * signature_profiles[primary_sig]
    if k5 > 0.5:
        sig_mix = 0.80 * sig_mix + 0.20 * signature_profiles["oxidative_like"]
    sig_mix = sig_mix / sig_mix.sum()
    total_mutations = int(rng.poisson(80 + 350 * mut_rate_raw + 40 * drivers_raw + 35 * DNA_raw + 15 * ROS_raw))
    total_mutations = max(total_mutations, 10)
    counts = rng.multinomial(total_mutations, sig_mix)
    signature_labels = list(signature_labels)
    for label, count in zip(signature_labels, counts, strict=False):
        safe = label.replace(">", "to").replace("[", "_").replace("]", "_")
        out[f"sig96_{safe}"] = int(count)
    for sig_name, profile in signature_profiles.items():
        out[f"sig_activity_{sig_name}"] = float(np.dot(counts / max(1, counts.sum()), profile) / np.dot(profile, profile))
    out["mut_total_count"] = int(total_mutations)
    return out


def generate_nonlinear_obs(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Linear KCC->state coupling (starter kit) but non-linear observation operator.

    Stresses stage-1 bottleneck recovery: the same latent state must be inferred
    from saturating, log-transformed, and multiplicatively-interacting omics
    readouts. The latent_risk structural equation is preserved.
    """
    cfg = SimConfig(n=n, months=months, seed=seed)
    rng = np.random.default_rng(seed)
    sig_labels, sig_profiles = make_signature_profiles()

    rows: list[dict[str, float | int]] = []
    trajectories: dict[str, pd.DataFrame] = {}

    for i in range(n):
        archetype = sample_archetype(rng)
        kcc = noisy_kcc_vector(archetype, rng)
        dose = float(np.clip(rng.lognormal(mean=-0.10, sigma=0.75), 0.02, 6.0))
        susceptibility = {
            "repair_capacity":         float(np.clip(rng.normal(0.82, 0.18), 0.35, 1.25)),
            "antioxidant_capacity":    float(np.clip(rng.normal(0.85, 0.17), 0.35, 1.25)),
            "immune_surveillance":     float(np.clip(rng.normal(0.82, 0.20), 0.25, 1.30)),
            "detox_balance":           float(np.clip(rng.lognormal(mean=0.0, sigma=0.20), 0.55, 1.65)),
            "baseline_proliferation":  float(np.clip(rng.normal(0.0, 0.55), -1.2, 1.2)),
        }
        traj = simulate_state_trajectory(kcc, dose, susceptibility, cfg.months, rng)
        states = summarize_trajectory(traj)
        omics = _nonlinear_obs_generate_omics(
            archetype, kcc, states, susceptibility, rng, sig_labels, sig_profiles,
        )

        final_risk = states["state_final_latent_risk"]
        cumulative_hazard = cfg.event_hazard_scale * cfg.months * final_risk
        event_probability = 1.0 - math.exp(-cumulative_hazard)
        event = int(rng.uniform() < event_probability)
        row: dict[str, float | int] = {
            "sample_id": f"ICG_NLO_{i:05d}",
            "chemical_archetype": archetype,
            "dose": dose,
            "future_cancer_transition_event": event,
            "future_event_probability": event_probability,
        }
        for j, name in enumerate(KCC_NAMES, start=1):
            row[f"kcc{j}_{name}"] = float(kcc[j - 1])
        for key, val in susceptibility.items():
            row[f"host_{key}"] = val
        row.update(states)
        row.update(omics)
        rows.append(row)

        if len(trajectories) < 8 and archetype not in trajectories:
            trajectories[archetype] = traj.assign(sample_id=row["sample_id"], chemical_archetype=archetype)

    df = pd.DataFrame(rows)
    cutoff = float(df["state_final_latent_risk"].quantile(0.72))
    df["high_risk_transition_state"] = (df["state_final_latent_risk"] >= cutoff).astype(int)
    return df, trajectories


def _misspecified_latent_risk_epi_age_flipped(S: pd.DataFrame, intercept: float = -6.0) -> np.ndarray:
    """Latent-risk equation with the sign on `state_final_epigenetic_age` flipped.

    The single deviation from `starter_kit_latent_risk` is that
    `state_final_epigenetic_age` enters with coefficient **-0.85** rather than
    **+0.85**, meaning that, in this synthetic DGP, accelerated epigenetic
    age is *protective* against future cancer transition. The intercept is
    adjusted from -7.5 to -6.0 to keep the marginal event rate in the same
    ballpark as `linear_lowhet`. This variant is used to test whether
    sign-constrained MB-CNet is robust to a mis-specified prior on the
    structural-equation signs.
    """
    def col(name: str) -> np.ndarray:
        if name in S.columns:
            return np.clip(S[name].to_numpy(dtype=float), 0.0, None)
        return np.zeros(len(S), dtype=float)
    return 1.0 / (1.0 + np.exp(-(
        intercept
        + 1.20 * np.log1p(col("state_final_DNA_adducts"))
        + 0.70 * np.log1p(col("state_final_ROS"))
        + 0.95 * np.log1p(col("state_final_inflammation"))
        + 1.50 * col("state_final_proliferation")
        - 0.85 * np.log1p(col("state_final_epigenetic_age"))   # FLIPPED sign
        + 1.80 * np.log1p(col("state_final_driver_count_proxy"))
        + 4.50 * col("state_final_clone_fraction")
        - 0.90 * col("state_final_immune_clearance")
    )))


def generate_misspecified_signs(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """`linear_lowhet` cohort relabelled under a mis-specified-sign latent-risk DGP.

    The KCC sampling, qAOP trajectory, and omics observation operator are all
    identical to `linear_lowhet`. Only the *outcome* labels are resampled
    from a latent-risk equation whose sign on `state_final_epigenetic_age` is
    flipped relative to the structural prior in `bottleneck.STRUCTURAL_SIGNS`.

    Use case: a falsification test for sign-constrained MB-CNet. If the prior
    is wrong, the sign-constrained model is biased; the unconstrained model
    learns the truth.
    """
    df, traj = generate_linear_lowhet(n=n, months=months, seed=seed)
    cfg = SimConfig(n=n, months=months, seed=seed)
    rng = np.random.default_rng(seed + 5000)

    state_cols = [c for c in df.columns if c.startswith("state_final_")]
    S = df[state_cols]
    new_latent = _misspecified_latent_risk_epi_age_flipped(S)
    df = df.copy()
    df["state_final_latent_risk"] = new_latent
    cumulative_hazard = cfg.event_hazard_scale * cfg.months * new_latent
    event_probability = 1.0 - np.exp(-cumulative_hazard)
    df["future_event_probability"] = event_probability
    df["future_cancer_transition_event"] = (rng.uniform(size=len(df)) < event_probability).astype(int)
    cutoff = float(np.quantile(new_latent, 0.72))
    df["high_risk_transition_state"] = (new_latent >= cutoff).astype(int)
    return df, traj


def _misspecified_latent_risk_v2_epi_and_immune_flipped(
    S: pd.DataFrame,
    intercept: float = -5.2,
) -> np.ndarray:
    """Two simultaneous sign flips vs the starter-kit structural equation.

    Deviations from ``starter_kit_latent_risk``:

    - ``state_final_epigenetic_age`` enters with **-0.85** on ``log1p(age)``
      (protective in the DGP; prior expects harmful).
    - ``state_final_immune_clearance`` enters with **+0.90** (harmful in the
      DGP; prior expects protective).

    Intercept adjusted to keep cohort prevalence near ``linear_lowhet``.
    """
    def col(name: str) -> np.ndarray:
        if name in S.columns:
            return np.clip(S[name].to_numpy(dtype=float), 0.0, None)
        return np.zeros(len(S), dtype=float)
    return 1.0 / (1.0 + np.exp(-(
        intercept
        + 1.20 * np.log1p(col("state_final_DNA_adducts"))
        + 0.70 * np.log1p(col("state_final_ROS"))
        + 0.95 * np.log1p(col("state_final_inflammation"))
        + 1.50 * col("state_final_proliferation")
        - 0.85 * np.log1p(col("state_final_epigenetic_age"))
        + 1.80 * np.log1p(col("state_final_driver_count_proxy"))
        + 4.50 * col("state_final_clone_fraction")
        + 0.90 * col("state_final_immune_clearance")
    )))


def generate_misspecified_signs_v2(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """`linear_lowhet` base with *two* coefficients in latent_risk flipped vs the prior.

    Compared to :func:`generate_misspecified_signs`, this cohort mis-specifies
    both ``state_final_epigenetic_age`` and ``state_final_immune_clearance``.
    Expected DGP directions for ``do_epigenetic_memory_reset`` and
    ``do_immune_surveillance_restore`` are both reversed relative to the prior.
    """
    df, traj = generate_linear_lowhet(n=n, months=months, seed=seed)
    cfg = SimConfig(n=n, months=months, seed=seed)
    rng = np.random.default_rng(seed + 6000)

    state_cols = [c for c in df.columns if c.startswith("state_final_")]
    S = df[state_cols]
    new_latent = _misspecified_latent_risk_v2_epi_and_immune_flipped(S)
    df = df.copy()
    df["state_final_latent_risk"] = new_latent
    cumulative_hazard = cfg.event_hazard_scale * cfg.months * new_latent
    event_probability = 1.0 - np.exp(-cumulative_hazard)
    df["future_event_probability"] = event_probability
    df["future_cancer_transition_event"] = (rng.uniform(size=len(df)) < event_probability).astype(int)
    cutoff = float(np.quantile(new_latent, 0.72))
    df["high_risk_transition_state"] = (new_latent >= cutoff).astype(int)
    return df, traj


_GENERATORS = {
    "linear_lowhet":         generate_linear_lowhet,
    "nonlinear_mixhost":     generate_nonlinear_mixhost,
    "partial_observability": generate_partial_observability,
    "nonlinear_obs":         generate_nonlinear_obs,
    "misspecified_signs":    generate_misspecified_signs,
    "misspecified_signs_v2": generate_misspecified_signs_v2,
}


def list_generator_names() -> list[str]:
    return list(_GENERATORS.keys())


def generate(variant_name: str, n: int = 1200, months: int = 72, seed: int = 7) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if variant_name not in _GENERATORS:
        raise KeyError(f"unknown generator variant: {variant_name!r}")
    return _GENERATORS[variant_name](n=n, months=months, seed=seed)
