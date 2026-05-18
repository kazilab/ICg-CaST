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

import functools
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from ..coefficients import (
    clear_registry_derived_caches,
    register_registry_derived_cache,
    use_registry,
)
from ..coefficients import registry as _registry
from ..config import SimConfig
from ..constants import ARCHETYPE_KCC, ARCHETYPE_SIGNATURE, EPI_MODULES, KCC_NAMES, TRANSCRIPT_MODULES
from ..omics import generate_omics
from ..signatures import make_signature_profiles
from ..simulator import (
    _bounded_retention,
    _event_probability_from_cumulative_risk,
    _per_month_hazard,
    _sample_susceptibility,
    noisy_kcc_vector,
    sample_archetype,
    simulate_cohort,
    simulate_state_trajectory,
    summarize_trajectory,
)


# ----------------------------------------------------------------------
# nonlinear_mixhost coefficient bundle (mirrors simulator._DynamicsCoefficients
# pattern: every numeric literal lives in the registry under
# benchmark.nonlinear_mixhost.dynamics.*).
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _NonlinDynamicsCoefficients:
    # repair / immune transforms
    repair_kcc3_inhibition: float
    repair_min: float
    repair_max: float
    immune_kcc7_inhibition: float
    immune_min: float
    immune_max: float
    antiox_kcc5_depletion: float
    antiox_min: float
    antiox_max: float
    # dose modulation
    dose_cyclical_baseline: float
    dose_cyclical_amplitude: float
    dose_cyclical_period_months: float
    dose_pulse_lognormal_mean: float
    dose_pulse_lognormal_sigma: float
    # DNA adducts
    dna_decay: float
    dna_repair_strength: float
    dna_max: float
    dna_input_kcc1_intercept: float
    dna_input_kcc1_slope: float
    dna_input_kcc1_power: float
    dna_input_kcc2_intercept: float
    dna_input_kcc2_slope: float
    dna_input_kcc2_power: float
    # ROS
    ros_decay: float
    ros_antioxidant_strength: float
    ros_max: float
    ros_input_intercept: float
    ros_input_kcc5_slope: float
    ros_input_kcc5_power: float
    ros_input_kcc1_slope: float
    ros_input_kcc1_power: float
    # cytotoxicity
    cyto_intercept: float
    cyto_dna_coupling: float
    cyto_ros_coupling: float
    # inflammation
    inf_decay: float
    inf_ros_coupling: float
    inf_cyto_coupling: float
    inf_dose_kcc6_coupling: float
    inf_kcc6_power: float
    inf_immune_clearance: float
    inf_max: float
    # epigenetic age
    epi_background_rate: float
    epi_dose_kcc4_coupling: float
    epi_dose_kcc5_coupling: float
    epi_dose_kcc6_coupling: float
    epi_kcc4_kcc5_interaction: float
    epi_ros_coupling: float
    epi_max: float
    # proliferation (tanh-saturated)
    prolif_intercept: float
    prolif_inflammation_coupling: float
    prolif_cyto_coupling: float
    prolif_dose_kcc8_coupling: float
    prolif_kcc10_coupling: float
    prolif_epi_coupling: float
    # mutation rate
    mut_scale: float
    mut_dna_kcc2: float
    mut_ros: float
    mut_repair_deficit: float
    mut_prolif: float
    mut_max: float
    # driver count
    driver_intercept: float
    driver_prolif: float
    driver_kcc9: float
    # clone fraction
    clone_init_scale: float
    clone_init_sigma: float
    clone_min: float
    clone_max: float
    clone_selection_intercept: float
    clone_selection_prolif: float
    clone_selection_inf: float
    clone_selection_epi: float
    clone_selection_kcc9: float
    clone_selection_immune: float
    clone_selection_drivers: float
    clone_selection_noise_sigma: float
    # latent risk
    risk_intercept: float
    risk_dna: float
    risk_ros: float
    risk_inf: float
    risk_prolif: float
    risk_epi: float
    risk_driver: float
    risk_clone: float
    risk_immune: float


@dataclass(frozen=True)
class _NonlinCohortCoefficients:
    continuous_kcc_dirichlet_alpha: float
    dose_lognormal_mean: float
    dose_lognormal_sigma: float
    dose_min: float
    dose_max: float
    event_hazard_scale: float


_NONLIN_DYN_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("repair_kcc3_inhibition", "benchmark.nonlinear_mixhost.dynamics.repair.kcc3_inhibition"),
    ("repair_min", "benchmark.nonlinear_mixhost.dynamics.repair.min"),
    ("repair_max", "benchmark.nonlinear_mixhost.dynamics.repair.max"),
    ("immune_kcc7_inhibition", "benchmark.nonlinear_mixhost.dynamics.immune.kcc7_inhibition"),
    ("immune_min", "benchmark.nonlinear_mixhost.dynamics.immune.min"),
    ("immune_max", "benchmark.nonlinear_mixhost.dynamics.immune.max"),
    ("antiox_kcc5_depletion", "benchmark.nonlinear_mixhost.dynamics.antioxidant.kcc5_depletion"),
    ("antiox_min", "benchmark.nonlinear_mixhost.dynamics.antioxidant.min"),
    ("antiox_max", "benchmark.nonlinear_mixhost.dynamics.antioxidant.max"),
    ("dose_cyclical_baseline", "benchmark.nonlinear_mixhost.dynamics.dose.cyclical_baseline"),
    ("dose_cyclical_amplitude", "benchmark.nonlinear_mixhost.dynamics.dose.cyclical_amplitude"),
    ("dose_cyclical_period_months", "benchmark.nonlinear_mixhost.dynamics.dose.cyclical_period_months"),
    ("dose_pulse_lognormal_mean", "benchmark.nonlinear_mixhost.dynamics.dose.pulse_lognormal_mean"),
    ("dose_pulse_lognormal_sigma", "benchmark.nonlinear_mixhost.dynamics.dose.pulse_lognormal_sigma"),
    ("dna_decay", "benchmark.nonlinear_mixhost.dynamics.dna_adducts.decay"),
    ("dna_repair_strength", "benchmark.nonlinear_mixhost.dynamics.dna_adducts.repair_strength"),
    ("dna_max", "benchmark.nonlinear_mixhost.dynamics.dna_adducts.max"),
    ("dna_input_kcc1_intercept", "benchmark.nonlinear_mixhost.dynamics.dna_adducts.input_kcc1_intercept"),
    ("dna_input_kcc1_slope", "benchmark.nonlinear_mixhost.dynamics.dna_adducts.input_kcc1_slope"),
    ("dna_input_kcc1_power", "benchmark.nonlinear_mixhost.dynamics.dna_adducts.input_kcc1_power"),
    ("dna_input_kcc2_intercept", "benchmark.nonlinear_mixhost.dynamics.dna_adducts.input_kcc2_intercept"),
    ("dna_input_kcc2_slope", "benchmark.nonlinear_mixhost.dynamics.dna_adducts.input_kcc2_slope"),
    ("dna_input_kcc2_power", "benchmark.nonlinear_mixhost.dynamics.dna_adducts.input_kcc2_power"),
    ("ros_decay", "benchmark.nonlinear_mixhost.dynamics.ros.decay"),
    ("ros_antioxidant_strength", "benchmark.nonlinear_mixhost.dynamics.ros.antioxidant_strength"),
    ("ros_max", "benchmark.nonlinear_mixhost.dynamics.ros.max"),
    ("ros_input_intercept", "benchmark.nonlinear_mixhost.dynamics.ros.input_intercept"),
    ("ros_input_kcc5_slope", "benchmark.nonlinear_mixhost.dynamics.ros.input_kcc5_slope"),
    ("ros_input_kcc5_power", "benchmark.nonlinear_mixhost.dynamics.ros.input_kcc5_power"),
    ("ros_input_kcc1_slope", "benchmark.nonlinear_mixhost.dynamics.ros.input_kcc1_slope"),
    ("ros_input_kcc1_power", "benchmark.nonlinear_mixhost.dynamics.ros.input_kcc1_power"),
    ("cyto_intercept", "benchmark.nonlinear_mixhost.dynamics.cytotoxicity.intercept"),
    ("cyto_dna_coupling", "benchmark.nonlinear_mixhost.dynamics.cytotoxicity.dna_coupling"),
    ("cyto_ros_coupling", "benchmark.nonlinear_mixhost.dynamics.cytotoxicity.ros_coupling"),
    ("inf_decay", "benchmark.nonlinear_mixhost.dynamics.inflammation.decay"),
    ("inf_ros_coupling", "benchmark.nonlinear_mixhost.dynamics.inflammation.ros_coupling"),
    ("inf_cyto_coupling", "benchmark.nonlinear_mixhost.dynamics.inflammation.cytotoxicity_coupling"),
    ("inf_dose_kcc6_coupling", "benchmark.nonlinear_mixhost.dynamics.inflammation.dose_kcc6_coupling"),
    ("inf_kcc6_power", "benchmark.nonlinear_mixhost.dynamics.inflammation.kcc6_power"),
    ("inf_immune_clearance", "benchmark.nonlinear_mixhost.dynamics.inflammation.immune_clearance"),
    ("inf_max", "benchmark.nonlinear_mixhost.dynamics.inflammation.max"),
    ("epi_background_rate", "benchmark.nonlinear_mixhost.dynamics.epigenetic_age.background_rate"),
    ("epi_dose_kcc4_coupling", "benchmark.nonlinear_mixhost.dynamics.epigenetic_age.dose_kcc4_coupling"),
    ("epi_dose_kcc5_coupling", "benchmark.nonlinear_mixhost.dynamics.epigenetic_age.dose_kcc5_coupling"),
    ("epi_dose_kcc6_coupling", "benchmark.nonlinear_mixhost.dynamics.epigenetic_age.dose_kcc6_coupling"),
    ("epi_kcc4_kcc5_interaction", "benchmark.nonlinear_mixhost.dynamics.epigenetic_age.kcc4_kcc5_interaction"),
    ("epi_ros_coupling", "benchmark.nonlinear_mixhost.dynamics.epigenetic_age.ros_coupling"),
    ("epi_max", "benchmark.nonlinear_mixhost.dynamics.epigenetic_age.max"),
    ("prolif_intercept", "benchmark.nonlinear_mixhost.dynamics.proliferation.intercept"),
    ("prolif_inflammation_coupling", "benchmark.nonlinear_mixhost.dynamics.proliferation.inflammation_coupling"),
    ("prolif_cyto_coupling", "benchmark.nonlinear_mixhost.dynamics.proliferation.cytotoxicity_coupling"),
    ("prolif_dose_kcc8_coupling", "benchmark.nonlinear_mixhost.dynamics.proliferation.dose_kcc8_coupling"),
    ("prolif_kcc10_coupling", "benchmark.nonlinear_mixhost.dynamics.proliferation.kcc10_coupling"),
    ("prolif_epi_coupling", "benchmark.nonlinear_mixhost.dynamics.proliferation.epigenetic_coupling"),
    ("mut_scale", "benchmark.nonlinear_mixhost.dynamics.mutation_rate.scale"),
    ("mut_dna_kcc2", "benchmark.nonlinear_mixhost.dynamics.mutation_rate.dna_kcc2_coupling"),
    ("mut_ros", "benchmark.nonlinear_mixhost.dynamics.mutation_rate.ros_coupling"),
    ("mut_repair_deficit", "benchmark.nonlinear_mixhost.dynamics.mutation_rate.repair_deficit_coupling"),
    ("mut_prolif", "benchmark.nonlinear_mixhost.dynamics.mutation_rate.proliferation_coupling"),
    ("mut_max", "benchmark.nonlinear_mixhost.dynamics.mutation_rate.max"),
    ("driver_intercept", "benchmark.nonlinear_mixhost.dynamics.driver_count.intercept"),
    ("driver_prolif", "benchmark.nonlinear_mixhost.dynamics.driver_count.proliferation_coupling"),
    ("driver_kcc9", "benchmark.nonlinear_mixhost.dynamics.driver_count.kcc9_coupling"),
    ("clone_init_scale", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.init_scale"),
    ("clone_init_sigma", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.init_lognormal_sigma"),
    ("clone_min", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.min"),
    ("clone_max", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.max"),
    ("clone_selection_intercept", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.selection_intercept"),
    ("clone_selection_prolif", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.selection_proliferation"),
    ("clone_selection_inf", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.selection_inflammation"),
    ("clone_selection_epi", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.selection_epigenetic"),
    ("clone_selection_kcc9", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.selection_kcc9"),
    ("clone_selection_immune", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.selection_immune"),
    ("clone_selection_drivers", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.selection_drivers"),
    ("clone_selection_noise_sigma", "benchmark.nonlinear_mixhost.dynamics.clone_fraction.selection_noise_sigma"),
    ("risk_intercept", "benchmark.nonlinear_mixhost.dynamics.latent_risk.intercept"),
    ("risk_dna", "benchmark.nonlinear_mixhost.dynamics.latent_risk.dna_coupling"),
    ("risk_ros", "benchmark.nonlinear_mixhost.dynamics.latent_risk.ros_coupling"),
    ("risk_inf", "benchmark.nonlinear_mixhost.dynamics.latent_risk.inflammation_coupling"),
    ("risk_prolif", "benchmark.nonlinear_mixhost.dynamics.latent_risk.proliferation_coupling"),
    ("risk_epi", "benchmark.nonlinear_mixhost.dynamics.latent_risk.epigenetic_coupling"),
    ("risk_driver", "benchmark.nonlinear_mixhost.dynamics.latent_risk.driver_coupling"),
    ("risk_clone", "benchmark.nonlinear_mixhost.dynamics.latent_risk.clone_coupling"),
    ("risk_immune", "benchmark.nonlinear_mixhost.dynamics.latent_risk.immune_coupling"),
)


_NONLIN_COHORT_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("continuous_kcc_dirichlet_alpha", "benchmark.nonlinear_mixhost.continuous_kcc.dirichlet_alpha"),
    ("dose_lognormal_mean", "benchmark.nonlinear_mixhost.dose.lognormal_mean"),
    ("dose_lognormal_sigma", "benchmark.nonlinear_mixhost.dose.lognormal_sigma"),
    ("dose_min", "benchmark.nonlinear_mixhost.dose.min"),
    ("dose_max", "benchmark.nonlinear_mixhost.dose.max"),
    ("event_hazard_scale", "benchmark.nonlinear_mixhost.event_hazard_scale"),
)


_SUSCEPT_OVERLAY_MAP: tuple[tuple[str, str], ...] = (
    ("susceptibility.repair_capacity.mean", "benchmark.nonlinear_mixhost.susceptibility.repair_capacity.mean"),
    ("susceptibility.repair_capacity.sd", "benchmark.nonlinear_mixhost.susceptibility.repair_capacity.sd"),
    ("susceptibility.repair_capacity.min", "benchmark.nonlinear_mixhost.susceptibility.repair_capacity.min"),
    ("susceptibility.repair_capacity.max", "benchmark.nonlinear_mixhost.susceptibility.repair_capacity.max"),
    ("susceptibility.antioxidant_capacity.mean", "benchmark.nonlinear_mixhost.susceptibility.antioxidant_capacity.mean"),
    ("susceptibility.antioxidant_capacity.sd", "benchmark.nonlinear_mixhost.susceptibility.antioxidant_capacity.sd"),
    ("susceptibility.antioxidant_capacity.min", "benchmark.nonlinear_mixhost.susceptibility.antioxidant_capacity.min"),
    ("susceptibility.antioxidant_capacity.max", "benchmark.nonlinear_mixhost.susceptibility.antioxidant_capacity.max"),
    ("susceptibility.immune_surveillance.mean", "benchmark.nonlinear_mixhost.susceptibility.immune_surveillance.mean"),
    ("susceptibility.immune_surveillance.sd", "benchmark.nonlinear_mixhost.susceptibility.immune_surveillance.sd"),
    ("susceptibility.immune_surveillance.min", "benchmark.nonlinear_mixhost.susceptibility.immune_surveillance.min"),
    ("susceptibility.immune_surveillance.max", "benchmark.nonlinear_mixhost.susceptibility.immune_surveillance.max"),
    ("susceptibility.detox_balance.lognormal_mean", "benchmark.nonlinear_mixhost.susceptibility.detox_balance.lognormal_mean"),
    ("susceptibility.detox_balance.lognormal_sigma", "benchmark.nonlinear_mixhost.susceptibility.detox_balance.lognormal_sigma"),
    ("susceptibility.detox_balance.min", "benchmark.nonlinear_mixhost.susceptibility.detox_balance.min"),
    ("susceptibility.detox_balance.max", "benchmark.nonlinear_mixhost.susceptibility.detox_balance.max"),
    ("susceptibility.baseline_proliferation.mean", "benchmark.nonlinear_mixhost.susceptibility.baseline_proliferation.mean"),
    ("susceptibility.baseline_proliferation.sd", "benchmark.nonlinear_mixhost.susceptibility.baseline_proliferation.sd"),
    ("susceptibility.baseline_proliferation.min", "benchmark.nonlinear_mixhost.susceptibility.baseline_proliferation.min"),
    ("susceptibility.baseline_proliferation.max", "benchmark.nonlinear_mixhost.susceptibility.baseline_proliferation.max"),
)


@functools.cache
def _nonlin_dyn() -> _NonlinDynamicsCoefficients:
    g = _registry().get
    return _NonlinDynamicsCoefficients(**{attr: g(key) for attr, key in _NONLIN_DYN_FIELD_MAP})


@functools.cache
def _nonlin_cohort() -> _NonlinCohortCoefficients:
    g = _registry().get
    return _NonlinCohortCoefficients(**{attr: g(key) for attr, key in _NONLIN_COHORT_FIELD_MAP})


register_registry_derived_cache(_nonlin_dyn.cache_clear)
register_registry_derived_cache(_nonlin_cohort.cache_clear)


def _high_heterogeneity_susceptibility_overlay():
    """Return the active registry with susceptibility.* cards replaced by the
    high-heterogeneity values from ``benchmark.nonlinear_mixhost.susceptibility.*``.

    Allows ``_sample_susceptibility`` (which reads ``susceptibility.*`` via
    ``_sus()``) to be reused unchanged under the wider-prior regime.
    """
    base = _registry()
    replacements = []
    for target_name, source_name in _SUSCEPT_OVERLAY_MAP:
        target_card = base.card(target_name)
        replacements.append(replace(target_card, default_value=base.get(source_name)))
    return base.replace_cards(replacements)


def _validate_generator_args(n: int, months: int, seed: int) -> None:
    if not isinstance(n, int) or isinstance(n, bool) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(months, int) or isinstance(months, bool) or months <= 0:
        raise ValueError("months must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")


def generate_linear_lowhet(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Reproduce the starter-kit cohort (discrete archetypes, low heterogeneity)."""
    _validate_generator_args(n, months, seed)
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
    noise_sigma = _registry().get("benchmark.continuous_kcc.noise_sigma")
    noise = rng.normal(0.0, noise_sigma, size=kcc.size)
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

    Differs from the starter kit by (i) KCC-power terms in the DNA-adduct and
    ROS inputs, (ii) a saturating tanh non-linearity on proliferation, (iii) a
    multiplicative KCC4*KCC5 interaction on epigenetic age, and (iv) a stronger
    clone-fraction selection pressure. The structural-equation form of
    ``latent_risk`` is preserved so the same ``starter_kit_latent_risk``
    function can be reused for intervention-augmented training. Every
    coefficient is sourced from the registry under
    ``benchmark.nonlinear_mixhost.dynamics.*``.
    """
    D = _nonlin_dyn()
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = kcc
    repair_capacity = float(
        np.clip(
            susceptibility["repair_capacity"] * (1.0 - D.repair_kcc3_inhibition * k3),
            D.repair_min,
            D.repair_max,
        )
    )
    antioxidant = float(
        np.clip(
            susceptibility["antioxidant_capacity"] * (1.0 - D.antiox_kcc5_depletion * k5),
            D.antiox_min,
            D.antiox_max,
        )
    )
    immune = float(
        np.clip(
            susceptibility["immune_surveillance"] * (1.0 - D.immune_kcc7_inhibition * k7),
            D.immune_min,
            D.immune_max,
        )
    )
    detox = float(susceptibility["detox_balance"])
    baseline_prolif = float(susceptibility["baseline_proliferation"])

    DNA = 0.0
    ROS = 0.0
    inflammation = 0.0
    epigenetic_age = 0.0
    clone_fraction = D.clone_init_scale * float(
        rng.lognormal(mean=0.0, sigma=D.clone_init_sigma)
    )
    driver_count_proxy = 0.0

    rows: list[dict[str, float | int]] = []
    for t in range(months):
        cyclical = D.dose_cyclical_baseline + D.dose_cyclical_amplitude * math.sin(
            2.0 * math.pi * t / D.dose_cyclical_period_months
        )
        pulse = float(
            rng.lognormal(mean=D.dose_pulse_lognormal_mean, sigma=D.dose_pulse_lognormal_sigma)
        )
        internal_dose = dose * cyclical * pulse * detox

        adduct_input = (
            internal_dose
            * (D.dna_input_kcc1_intercept + D.dna_input_kcc1_slope * (k1 ** D.dna_input_kcc1_power))
            * (D.dna_input_kcc2_intercept + D.dna_input_kcc2_slope * (k2 ** D.dna_input_kcc2_power))
        )
        DNA = D.dna_decay * DNA + adduct_input - D.dna_repair_strength * repair_capacity * DNA
        DNA = float(np.clip(DNA, 0.0, D.dna_max))

        ros_input = internal_dose * (
            D.ros_input_intercept
            + D.ros_input_kcc5_slope * (k5 ** D.ros_input_kcc5_power)
            + D.ros_input_kcc1_slope * (k1 ** D.ros_input_kcc1_power)
        )
        ROS = D.ros_decay * ROS + ros_input - D.ros_antioxidant_strength * antioxidant * ROS
        ROS = float(np.clip(ROS, 0.0, D.ros_max))

        cytotoxicity = float(
            expit(D.cyto_intercept + D.cyto_dna_coupling * DNA + D.cyto_ros_coupling * ROS)
        )
        # Inflammation retention is routed through expit-bounded logit-space
        # subtraction (matching DNA/ROS) so the prior-sample regime cannot
        # produce a negative effective retention.
        inflammation_input = (
            D.inf_ros_coupling * ROS
            + D.inf_cyto_coupling * cytotoxicity
            + D.inf_dose_kcc6_coupling * internal_dose * (k6 ** D.inf_kcc6_power)
        )
        inflammation_retention = float(
            _bounded_retention(D.inf_decay, D.inf_immune_clearance * immune)
        )
        inflammation = inflammation_retention * inflammation + inflammation_input
        inflammation = float(np.clip(inflammation, 0.0, D.inf_max))

        epigenetic_age += (
            D.epi_background_rate
            + internal_dose
            * (
                D.epi_dose_kcc4_coupling * k4
                + D.epi_dose_kcc5_coupling * k5
                + D.epi_dose_kcc6_coupling * k6
            )
            + D.epi_kcc4_kcc5_interaction * (k4 * k5)
            + D.epi_ros_coupling * ROS
        )
        epigenetic_age = float(np.clip(epigenetic_age, 0.0, D.epi_max))

        # tanh saturation in pre-activation space, then rescaled to [0, 1].
        proliferation = float(
            np.tanh(
                D.prolif_intercept
                + baseline_prolif
                + D.prolif_inflammation_coupling * inflammation
                + D.prolif_cyto_coupling * cytotoxicity
                + D.prolif_dose_kcc8_coupling * k8 * internal_dose
                + D.prolif_kcc10_coupling * k10
                + D.prolif_epi_coupling * epigenetic_age
            )
            * 0.5
            + 0.5
        )

        # See simulator._qaop_step for the rationale: clamp the deficit at 0
        # so high-repair hosts don't silently *reduce* the baseline mutation
        # rate via a negative "deficit."
        repair_deficit = max(0.0, 1.0 - repair_capacity)
        mutation_rate = D.mut_scale * (
            1.0
            + D.mut_dna_kcc2 * DNA * k2
            + D.mut_ros * ROS
            + D.mut_repair_deficit * repair_deficit
            + D.mut_prolif * proliferation
        )
        mutation_rate = float(np.clip(mutation_rate, 0.0, D.mut_max))

        driver_count_proxy += (
            mutation_rate
            * (D.driver_intercept + D.driver_prolif * proliferation)
            * (1.0 + D.driver_kcc9 * k9)
        )

        selection = (
            D.clone_selection_intercept
            + D.clone_selection_prolif * proliferation
            + D.clone_selection_inf * inflammation
            + D.clone_selection_epi * epigenetic_age
            + D.clone_selection_kcc9 * k9
            - D.clone_selection_immune * immune
            + D.clone_selection_drivers * driver_count_proxy
        )
        noise = float(rng.normal(0.0, D.clone_selection_noise_sigma))
        clone_fraction = clone_fraction + clone_fraction * (selection * (1.0 - clone_fraction) + noise)
        clone_fraction = float(np.clip(clone_fraction, D.clone_min, D.clone_max))

        latent_risk = float(
            expit(
                D.risk_intercept
                + D.risk_dna * math.log1p(DNA)
                + D.risk_ros * math.log1p(ROS)
                + D.risk_inf * math.log1p(inflammation)
                + D.risk_prolif * proliferation
                + D.risk_epi * math.log1p(epigenetic_age)
                + D.risk_driver * math.log1p(driver_count_proxy)
                + D.risk_clone * clone_fraction
                + D.risk_immune * immune
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
    - Wider host susceptibility distributions (``host_heterogeneity="high"``),
      applied as a registry overlay so the same ``_sample_susceptibility``
      helper as the main simulator can be reused.

    The output schema matches ``icg_cast.simulate_cohort``.
    """
    _validate_generator_args(n, months, seed)
    overlay = _high_heterogeneity_susceptibility_overlay()
    with use_registry(overlay):
        clear_registry_derived_caches()
        try:
            return _generate_nonlinear_mixhost_impl(n=n, months=months, seed=seed)
        finally:
            clear_registry_derived_caches()


def _generate_nonlinear_mixhost_impl(
    *, n: int, months: int, seed: int
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(seed)
    sig_labels, sig_profiles = make_signature_profiles()
    C = _nonlin_cohort()
    r = _registry()
    high_risk_quantile = r.get("cohort.high_risk_quantile")

    rows: list[dict[str, float | int]] = []
    trajectories: dict[str, pd.DataFrame] = {}

    for i in range(n):
        archetype, kcc = _sample_continuous_kcc(
            ARCHETYPE_KCC, rng, alpha=C.continuous_kcc_dirichlet_alpha
        )
        dose = float(
            np.clip(
                rng.lognormal(mean=C.dose_lognormal_mean, sigma=C.dose_lognormal_sigma),
                C.dose_min,
                C.dose_max,
            )
        )
        susceptibility = _sample_susceptibility(rng)

        traj = _simulate_state_trajectory_nonlinear(kcc, dose, susceptibility, months, rng)
        states = summarize_trajectory(traj)
        omics = generate_omics(
            archetype, kcc, states, susceptibility, rng, sig_labels, sig_profiles,
        )

        event_probability = float(
            _event_probability_from_cumulative_risk(
                C.event_hazard_scale,
                states["state_cumulative_latent_risk"],
            )
        )
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
    cutoff = float(df["state_final_latent_risk"].quantile(high_risk_quantile))
    df["high_risk_transition_state"] = (df["state_final_latent_risk"] >= cutoff).astype(int)
    return df, trajectories


def generate_partial_observability(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
    masking_rate: float = 0.30,
    mask_seed: int | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """``nonlinear_mixhost`` cohort with per-subject random masking of omics modules.

    Each ``tx_*`` / ``epi_*`` column is independently set to NaN with
    probability ``masking_rate`` per subject (Bernoulli sampling). KCC vectors,
    host susceptibility, mutational-signature features, and the qAOP state
    summaries are preserved unchanged. The latent_risk structural equation is
    unchanged.

    ``mask_seed`` controls the masking RNG independently of ``seed`` (which
    drives cohort sampling). When ``mask_seed`` is ``None``, an independent
    sub-stream is spawned from ``seed`` via ``np.random.SeedSequence`` so the
    mask realisation is fully reproducible from ``seed`` without colliding
    with the cohort's RNG draws.
    """
    _validate_generator_args(n, months, seed)
    if mask_seed is not None and (not isinstance(mask_seed, int) or isinstance(mask_seed, bool)):
        raise ValueError("mask_seed must be an integer or None")
    if not np.isfinite(float(masking_rate)) or not (0.0 <= masking_rate <= 1.0):
        raise ValueError("masking_rate must be finite and in [0, 1]")
    df, traj = generate_nonlinear_mixhost(n=n, months=months, seed=seed)
    if mask_seed is None:
        mask_ss = np.random.SeedSequence(seed).spawn(2)[1]
        rng = np.random.default_rng(mask_ss)
    else:
        rng = np.random.default_rng(mask_seed)
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
    _validate_generator_args(n, months, seed)
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

        event_probability = float(
            _event_probability_from_cumulative_risk(
                cfg.event_hazard_scale,
                states["state_cumulative_latent_risk"],
            )
        )
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


def _signed_risk_from_components(
    intercept: float,
    *,
    dna: np.ndarray,
    ros: np.ndarray,
    inflammation: np.ndarray,
    proliferation: np.ndarray,
    epigenetic_age: np.ndarray,
    driver_count_proxy: np.ndarray,
    clone_fraction: np.ndarray,
    immune_clearance: np.ndarray,
    epi_sign: float,
    immune_sign: float,
) -> np.ndarray:
    """Starter-kit ``latent_risk`` with configurable signs on the epi-age and
    immune-clearance terms. Structural coefficient magnitudes (and signs) are
    sourced from the registry so a change to ``dynamics.latent_risk.*``
    propagates to the misspecified cohorts automatically.

    Sign-flip convention
    --------------------
    ``epi_sign`` and ``immune_sign`` are multiplicative factors applied on top
    of the registry coefficient. Under the registry's default convention
    (``epi_coupling=+0.85``, ``immune_coupling=-0.90``), ``+1`` reproduces the
    starter-kit DGP and ``-1`` flips that term's contribution to risk.
    """
    r = _registry()
    clip = lambda a: np.clip(np.asarray(a, dtype=float), 0.0, None)
    return expit(
        intercept
        + r.get("dynamics.latent_risk.dna_coupling") * np.log1p(clip(dna))
        + r.get("dynamics.latent_risk.ros_coupling") * np.log1p(clip(ros))
        + r.get("dynamics.latent_risk.inflammation_coupling") * np.log1p(clip(inflammation))
        + r.get("dynamics.latent_risk.proliferation_coupling") * clip(proliferation)
        + epi_sign
        * r.get("dynamics.latent_risk.epigenetic_coupling")
        * np.log1p(clip(epigenetic_age))
        + r.get("dynamics.latent_risk.driver_coupling")
        * np.log1p(clip(driver_count_proxy))
        + r.get("dynamics.latent_risk.clone_coupling") * clip(clone_fraction)
        + immune_sign * r.get("dynamics.latent_risk.immune_coupling") * clip(immune_clearance)
    )


_FINAL_STATE_COLUMN_MAP: tuple[tuple[str, str], ...] = (
    ("dna", "state_final_DNA_adducts"),
    ("ros", "state_final_ROS"),
    ("inflammation", "state_final_inflammation"),
    ("proliferation", "state_final_proliferation"),
    ("epigenetic_age", "state_final_epigenetic_age"),
    ("driver_count_proxy", "state_final_driver_count_proxy"),
    ("clone_fraction", "state_final_clone_fraction"),
    ("immune_clearance", "state_final_immune_clearance"),
)

_TRAJ_COLUMN_MAP: tuple[tuple[str, str], ...] = (
    ("dna", "DNA_adducts"),
    ("ros", "ROS"),
    ("inflammation", "inflammation"),
    ("proliferation", "proliferation"),
    ("epigenetic_age", "epigenetic_age"),
    ("driver_count_proxy", "driver_count_proxy"),
    ("clone_fraction", "clone_fraction"),
    ("immune_clearance", "immune_clearance"),
)


def _components_from_frame(
    frame: pd.DataFrame, column_map: tuple[tuple[str, str], ...]
) -> dict[str, np.ndarray]:
    n_rows = len(frame)
    out: dict[str, np.ndarray] = {}
    for component, col in column_map:
        if col in frame.columns:
            out[component] = frame[col].to_numpy(dtype=float)
        else:
            out[component] = np.zeros(n_rows, dtype=float)
    return out


def _solve_intercept_for_prevalence_parity(
    *,
    cumulative_flipped_risk_at: Callable[[float], np.ndarray],
    target_event_prob_mean: float,
    hazard_scale: float,
    bracket: tuple[float, float] = (-30.0, 10.0),
) -> float:
    """Root-find the intercept ``b`` such that the mean event probability under
    the flipped latent-risk equation matches ``target_event_prob_mean``. Both
    sides use the simulator's cumulative-hazard parameterisation
    (``1 - exp(-hazard_scale * cumulative_latent_risk)``).
    """
    def residual(b: float) -> float:
        cumulative = cumulative_flipped_risk_at(b)
        event_probability = _event_probability_from_cumulative_risk(hazard_scale, cumulative)
        return float(np.mean(event_probability) - target_event_prob_mean)
    lo, hi = bracket
    return float(brentq(residual, lo, hi, xtol=1e-8, rtol=1e-10))


def _generate_misspecified_signs_impl(
    *,
    n: int,
    months: int,
    seed: int,
    rng_offset: int,
    epi_sign: float,
    immune_sign: float,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    cfg = SimConfig(n=n, months=months, seed=seed, retain_trajectories=True)
    df, per_subject_traj = simulate_cohort(cfg)
    rng = np.random.default_rng(seed + rng_offset)
    r = _registry()
    high_risk_quantile = r.get("cohort.high_risk_quantile")
    sample_ids = df["sample_id"].tolist()

    # Stack per-month state components for every subject. Each component array
    # has shape (n_subjects, months) so the flipped latent_risk can be evaluated
    # vectorised across the whole cohort at once during root-finding.
    component_arrays: dict[str, np.ndarray] = {comp: [] for comp, _ in _TRAJ_COLUMN_MAP}
    for sid in sample_ids:
        traj = per_subject_traj[sid]
        comps = _components_from_frame(traj, _TRAJ_COLUMN_MAP)
        for comp, _ in _TRAJ_COLUMN_MAP:
            component_arrays[comp].append(comps[comp])
    stacked = {comp: np.vstack(arrs) for comp, arrs in component_arrays.items()}

    def cumulative_flipped_risk_at(intercept: float) -> np.ndarray:
        # Each row of ``per_month`` is a per-month event probability (sigmoid
        # output); the cumulative hazard is Σ -log(1 - p_t), matching the
        # simulator's redefined ``state_cumulative_latent_risk``.
        per_month = _signed_risk_from_components(
            intercept,
            epi_sign=epi_sign,
            immune_sign=immune_sign,
            **stacked,
        )
        return _per_month_hazard(per_month).sum(axis=1)

    target_mean = float(df["future_event_probability"].mean())
    intercept = _solve_intercept_for_prevalence_parity(
        cumulative_flipped_risk_at=cumulative_flipped_risk_at,
        target_event_prob_mean=target_mean,
        hazard_scale=cfg.event_hazard_scale,
    )

    cumulative_flipped = cumulative_flipped_risk_at(intercept)
    final_components = _components_from_frame(df, _FINAL_STATE_COLUMN_MAP)
    new_final_latent = _signed_risk_from_components(
        intercept,
        epi_sign=epi_sign,
        immune_sign=immune_sign,
        **final_components,
    )
    event_probability = _event_probability_from_cumulative_risk(
        cfg.event_hazard_scale, cumulative_flipped
    )

    df = df.copy()
    df["state_final_latent_risk"] = new_final_latent
    df["state_cumulative_latent_risk"] = cumulative_flipped
    df["future_event_probability"] = event_probability
    df["future_cancer_transition_event"] = (
        rng.uniform(size=len(df)) < event_probability
    ).astype(int)
    cutoff = float(np.quantile(new_final_latent, high_risk_quantile))
    df["high_risk_transition_state"] = (new_final_latent >= cutoff).astype(int)

    # Preserve the public trajectory schema: one representative trajectory per
    # archetype, matching ``generate_linear_lowhet``'s default.
    trimmed: dict[str, pd.DataFrame] = {}
    for sid, archetype in zip(sample_ids, df["chemical_archetype"].tolist(), strict=True):
        if archetype not in trimmed:
            trimmed[archetype] = per_subject_traj[sid]
    return df, trimmed


def generate_misspecified_signs(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """``linear_lowhet`` cohort relabelled under a mis-specified-sign latent-risk DGP.

    The KCC sampling, qAOP trajectory, and omics observation operator are
    identical to ``linear_lowhet``. Only the outcome labels are resampled from
    a latent-risk equation whose sign on ``state_final_epigenetic_age`` is
    flipped relative to the structural prior in ``bottleneck.STRUCTURAL_SIGNS``.

    The intercept is solved at call-time so the mean event probability matches
    the linear baseline; structural-equation magnitudes come from the registry
    so a change to ``dynamics.latent_risk.*`` propagates without drift.
    """
    _validate_generator_args(n, months, seed)
    # Under the registry's new convention (immune_coupling=-0.90), immune_sign=+1
    # reproduces the starter-kit DGP; only the epigenetic-age sign is flipped here.
    return _generate_misspecified_signs_impl(
        n=n, months=months, seed=seed,
        rng_offset=5000,
        epi_sign=-1.0, immune_sign=+1.0,
    )


def generate_misspecified_signs_v2(
    n: int = 1200,
    months: int = 72,
    seed: int = 7,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """``linear_lowhet`` base with *two* coefficients in latent_risk flipped vs the prior.

    Compared to :func:`generate_misspecified_signs`, this cohort mis-specifies
    both ``state_final_epigenetic_age`` and ``state_final_immune_clearance``.
    Expected DGP directions for ``do_epigenetic_memory_reset`` and
    ``do_immune_surveillance_restore`` are both reversed relative to the prior.
    Intercept solved at call-time for prevalence parity with ``linear_lowhet``.
    """
    _validate_generator_args(n, months, seed)
    # Both signs flipped relative to the new registry convention
    # (immune_coupling=-0.90); immune_sign=-1 makes immune harmful in this DGP.
    return _generate_misspecified_signs_impl(
        n=n, months=months, seed=seed,
        rng_offset=6000,
        epi_sign=-1.0, immune_sign=-1.0,
    )


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
    _validate_generator_args(n, months, seed)
    if variant_name not in _GENERATORS:
        raise KeyError(f"unknown generator variant: {variant_name!r}")
    return _GENERATORS[variant_name](n=n, months=months, seed=seed)
