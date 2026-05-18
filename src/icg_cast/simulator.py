"""Synthetic exposure-to-qAOP-to-cancer-transition simulator.

Every numeric coefficient used here is sourced from the coefficient
registry — see :mod:`icg_cast.coefficients` and
``materials/coefficient_cards.yaml``. There are no inline numeric
literals in the qAOP dynamics, susceptibility distributions, or
cohort-sampling code below; the only literals that remain are
mathematical constants (zero, one, two-times-pi) and the state-name
prefixes used to label DataFrame columns.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import expit

from .coefficients import (
    clear_registry_derived_caches,
    register_registry_derived_cache,
    sampled_registry,
    use_registry,
)
from .coefficients import registry as _registry
from .config import SimConfig
from .constants import ARCHETYPE_KCC, ARCHETYPE_ORDER, KCC_NAMES, STATE_NAMES
from .omics import generate_omics
from .signatures import make_signature_profiles

if TYPE_CHECKING:
    from .calibration.bundle import CalibrationBundle


# ----------------------------------------------------------------------
# Coefficient namespaces (populated once from the registry, then cached)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _DynamicsCoefficients:
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
    dna_input_kcc2_intercept: float
    dna_input_kcc2_slope: float
    # ROS
    ros_decay: float
    ros_antioxidant_strength: float
    ros_max: float
    ros_input_intercept: float
    ros_input_kcc5_slope: float
    ros_input_kcc1_slope: float
    # cytotoxicity
    cyto_intercept: float
    cyto_dna_coupling: float
    cyto_ros_coupling: float
    # inflammation
    inf_decay: float
    inf_ros_coupling: float
    inf_cyto_coupling: float
    inf_dose_kcc6_coupling: float
    inf_immune_clearance: float
    inf_max: float
    # epigenetic age
    epi_background_rate: float
    epi_dose_kcc4_coupling: float
    epi_dose_kcc5_coupling: float
    epi_dose_kcc6_coupling: float
    epi_ros_coupling: float
    epi_max: float
    # proliferation
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
class _SusceptibilityCoefficients:
    repair_mean: float
    repair_sd: float
    repair_min: float
    repair_max: float
    antiox_mean: float
    antiox_sd: float
    antiox_min: float
    antiox_max: float
    immune_mean: float
    immune_sd: float
    immune_min: float
    immune_max: float
    detox_lognormal_mean: float
    detox_lognormal_sigma: float
    detox_min: float
    detox_max: float
    baseline_prolif_mean: float
    baseline_prolif_sd: float
    baseline_prolif_min: float
    baseline_prolif_max: float


@dataclass(frozen=True)
class _ArchetypeCoefficients:
    sample_prior: tuple[float, ...]
    kcc_noise_sigma: float


# Single source of truth that maps each ``_DynamicsCoefficients`` field name to
# its dotted registry key. Adding a new dynamics coefficient is a two-step edit:
# declare the dataclass field above, then add the tuple here.
_DYN_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("repair_kcc3_inhibition", "dynamics.repair.kcc3_inhibition"),
    ("repair_min", "dynamics.repair.min"),
    ("repair_max", "dynamics.repair.max"),
    ("immune_kcc7_inhibition", "dynamics.immune.kcc7_inhibition"),
    ("immune_min", "dynamics.immune.min"),
    ("immune_max", "dynamics.immune.max"),
    ("antiox_kcc5_depletion", "dynamics.antioxidant.kcc5_depletion"),
    ("antiox_min", "dynamics.antioxidant.min"),
    ("antiox_max", "dynamics.antioxidant.max"),
    ("dose_cyclical_baseline", "dynamics.dose.cyclical_baseline"),
    ("dose_cyclical_amplitude", "dynamics.dose.cyclical_amplitude"),
    ("dose_cyclical_period_months", "dynamics.dose.cyclical_period_months"),
    ("dose_pulse_lognormal_mean", "dynamics.dose.pulse_lognormal_mean"),
    ("dose_pulse_lognormal_sigma", "dynamics.dose.pulse_lognormal_sigma"),
    ("dna_decay", "dynamics.dna_adducts.decay"),
    ("dna_repair_strength", "dynamics.dna_adducts.repair_strength"),
    ("dna_max", "dynamics.dna_adducts.max"),
    ("dna_input_kcc1_intercept", "dynamics.dna_adducts.input_kcc1_intercept"),
    ("dna_input_kcc1_slope", "dynamics.dna_adducts.input_kcc1_slope"),
    ("dna_input_kcc2_intercept", "dynamics.dna_adducts.input_kcc2_intercept"),
    ("dna_input_kcc2_slope", "dynamics.dna_adducts.input_kcc2_slope"),
    ("ros_decay", "dynamics.ros.decay"),
    ("ros_antioxidant_strength", "dynamics.ros.antioxidant_strength"),
    ("ros_max", "dynamics.ros.max"),
    ("ros_input_intercept", "dynamics.ros.input_intercept"),
    ("ros_input_kcc5_slope", "dynamics.ros.input_kcc5_slope"),
    ("ros_input_kcc1_slope", "dynamics.ros.input_kcc1_slope"),
    ("cyto_intercept", "dynamics.cytotoxicity.intercept"),
    ("cyto_dna_coupling", "dynamics.cytotoxicity.dna_coupling"),
    ("cyto_ros_coupling", "dynamics.cytotoxicity.ros_coupling"),
    ("inf_decay", "dynamics.inflammation.decay"),
    ("inf_ros_coupling", "dynamics.inflammation.ros_coupling"),
    ("inf_cyto_coupling", "dynamics.inflammation.cytotoxicity_coupling"),
    ("inf_dose_kcc6_coupling", "dynamics.inflammation.dose_kcc6_coupling"),
    ("inf_immune_clearance", "dynamics.inflammation.immune_clearance"),
    ("inf_max", "dynamics.inflammation.max"),
    ("epi_background_rate", "dynamics.epigenetic_age.background_rate"),
    ("epi_dose_kcc4_coupling", "dynamics.epigenetic_age.dose_kcc4_coupling"),
    ("epi_dose_kcc5_coupling", "dynamics.epigenetic_age.dose_kcc5_coupling"),
    ("epi_dose_kcc6_coupling", "dynamics.epigenetic_age.dose_kcc6_coupling"),
    ("epi_ros_coupling", "dynamics.epigenetic_age.ros_coupling"),
    ("epi_max", "dynamics.epigenetic_age.max"),
    ("prolif_intercept", "dynamics.proliferation.intercept"),
    ("prolif_inflammation_coupling", "dynamics.proliferation.inflammation_coupling"),
    ("prolif_cyto_coupling", "dynamics.proliferation.cytotoxicity_coupling"),
    ("prolif_dose_kcc8_coupling", "dynamics.proliferation.dose_kcc8_coupling"),
    ("prolif_kcc10_coupling", "dynamics.proliferation.kcc10_coupling"),
    ("prolif_epi_coupling", "dynamics.proliferation.epigenetic_coupling"),
    ("mut_scale", "dynamics.mutation_rate.scale"),
    ("mut_dna_kcc2", "dynamics.mutation_rate.dna_kcc2_coupling"),
    ("mut_ros", "dynamics.mutation_rate.ros_coupling"),
    ("mut_repair_deficit", "dynamics.mutation_rate.repair_deficit_coupling"),
    ("mut_prolif", "dynamics.mutation_rate.proliferation_coupling"),
    ("driver_intercept", "dynamics.driver_count.intercept"),
    ("driver_prolif", "dynamics.driver_count.proliferation_coupling"),
    ("driver_kcc9", "dynamics.driver_count.kcc9_coupling"),
    ("clone_init_scale", "dynamics.clone_fraction.init_scale"),
    ("clone_init_sigma", "dynamics.clone_fraction.init_lognormal_sigma"),
    ("clone_min", "dynamics.clone_fraction.min"),
    ("clone_max", "dynamics.clone_fraction.max"),
    ("clone_selection_intercept", "dynamics.clone_fraction.selection_intercept"),
    ("clone_selection_prolif", "dynamics.clone_fraction.selection_proliferation"),
    ("clone_selection_inf", "dynamics.clone_fraction.selection_inflammation"),
    ("clone_selection_epi", "dynamics.clone_fraction.selection_epigenetic"),
    ("clone_selection_kcc9", "dynamics.clone_fraction.selection_kcc9"),
    ("clone_selection_immune", "dynamics.clone_fraction.selection_immune"),
    ("clone_selection_drivers", "dynamics.clone_fraction.selection_drivers"),
    ("clone_selection_noise_sigma", "dynamics.clone_fraction.selection_noise_sigma"),
    ("risk_intercept", "dynamics.latent_risk.intercept"),
    ("risk_dna", "dynamics.latent_risk.dna_coupling"),
    ("risk_ros", "dynamics.latent_risk.ros_coupling"),
    ("risk_inf", "dynamics.latent_risk.inflammation_coupling"),
    ("risk_prolif", "dynamics.latent_risk.proliferation_coupling"),
    ("risk_epi", "dynamics.latent_risk.epigenetic_coupling"),
    ("risk_driver", "dynamics.latent_risk.driver_coupling"),
    ("risk_clone", "dynamics.latent_risk.clone_coupling"),
    ("risk_immune", "dynamics.latent_risk.immune_coupling"),
)


_SUS_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("repair_mean", "susceptibility.repair_capacity.mean"),
    ("repair_sd", "susceptibility.repair_capacity.sd"),
    ("repair_min", "susceptibility.repair_capacity.min"),
    ("repair_max", "susceptibility.repair_capacity.max"),
    ("antiox_mean", "susceptibility.antioxidant_capacity.mean"),
    ("antiox_sd", "susceptibility.antioxidant_capacity.sd"),
    ("antiox_min", "susceptibility.antioxidant_capacity.min"),
    ("antiox_max", "susceptibility.antioxidant_capacity.max"),
    ("immune_mean", "susceptibility.immune_surveillance.mean"),
    ("immune_sd", "susceptibility.immune_surveillance.sd"),
    ("immune_min", "susceptibility.immune_surveillance.min"),
    ("immune_max", "susceptibility.immune_surveillance.max"),
    ("detox_lognormal_mean", "susceptibility.detox_balance.lognormal_mean"),
    ("detox_lognormal_sigma", "susceptibility.detox_balance.lognormal_sigma"),
    ("detox_min", "susceptibility.detox_balance.min"),
    ("detox_max", "susceptibility.detox_balance.max"),
    ("baseline_prolif_mean", "susceptibility.baseline_proliferation.mean"),
    ("baseline_prolif_sd", "susceptibility.baseline_proliferation.sd"),
    ("baseline_prolif_min", "susceptibility.baseline_proliferation.min"),
    ("baseline_prolif_max", "susceptibility.baseline_proliferation.max"),
)


@functools.cache
def _dyn() -> _DynamicsCoefficients:
    """Materialise the qAOP-dynamics coefficient bundle from the active registry."""
    g = _registry().get
    return _DynamicsCoefficients(**{attr: g(key) for attr, key in _DYN_FIELD_MAP})


@functools.cache
def _sus() -> _SusceptibilityCoefficients:
    """Materialise the susceptibility-distribution bundle from the active registry."""
    g = _registry().get
    return _SusceptibilityCoefficients(**{attr: g(key) for attr, key in _SUS_FIELD_MAP})


@functools.cache
def _arch() -> _ArchetypeCoefficients:
    r = _registry()
    return _ArchetypeCoefficients(
        sample_prior=r.get_vector("archetypes.sample_prior"),
        kcc_noise_sigma=r.get("archetypes.kcc_noise_sigma"),
    )


register_registry_derived_cache(_dyn.cache_clear)
register_registry_derived_cache(_sus.cache_clear)
register_registry_derived_cache(_arch.cache_clear)


def _clear_coefficient_caches() -> None:
    """Clear every cache derived from the active coefficient registry."""
    clear_registry_derived_caches()


def _active_archetype_table() -> dict[str, tuple[float, ...]]:
    r = _registry()
    names: list[str] = []
    for card_name in r.names():
        if not card_name.startswith("archetypes.") or not card_name.endswith(".kcc"):
            continue
        name = card_name.removeprefix("archetypes.").removesuffix(".kcc")
        names.append(name)
    ordered = [name for name in ARCHETYPE_ORDER if name in names]
    ordered.extend(name for name in names if name not in ordered)
    return {name: r.get_vector(f"archetypes.{name}.kcc") for name in ordered}


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------


def sample_archetype(
    rng: np.random.Generator,
    prior: dict[str, float] | None = None,
    archetype_table: Mapping[str, tuple[float, ...]] | None = None,
) -> str:
    """Sample an exposure archetype from the default or user-supplied prior."""
    table = archetype_table if archetype_table is not None else ARCHETYPE_KCC
    archetypes = list(table.keys())
    a = _arch()
    if prior is None:
        if tuple(archetypes) == ARCHETYPE_ORDER:
            probs = np.array(a.sample_prior, dtype=float)
            if probs.size != len(archetypes):
                raise ValueError(
                    f"archetypes.sample_prior has {probs.size} entries but "
                    f"the active archetype table has {len(archetypes)}"
                )
        else:
            probs = np.ones(len(archetypes), dtype=float) / len(archetypes)
    else:
        missing = [name for name in prior if name not in table]
        if missing:
            raise ValueError(f"unknown archetype(s) in prior: {missing}")
        probs = np.array([float(prior.get(name, 0.0)) for name in archetypes], dtype=float)
    if probs.sum() <= 0:
        raise ValueError("archetype prior must contain at least one positive weight")
    probs = probs / probs.sum()
    return str(rng.choice(archetypes, p=probs))


def noisy_kcc_vector(
    archetype: str,
    rng: np.random.Generator,
    archetype_table: Mapping[str, tuple[float, ...]] | None = None,
) -> np.ndarray:
    """Return a noisy 10-dimensional KCC vector clipped to ``[0, 1]``."""
    table = archetype_table if archetype_table is not None else ARCHETYPE_KCC
    if archetype not in table:
        raise KeyError(f"archetype not in KCC table: {archetype!r}")
    base = np.array(table[archetype], dtype=float)
    noise = rng.normal(0.0, _arch().kcc_noise_sigma, size=base.size)
    return np.clip(base + noise, 0.0, 1.0)


def _validate_kcc_vector(kcc: np.ndarray) -> np.ndarray:
    arr = np.asarray(kcc, dtype=float)
    if arr.ndim != 1 or arr.size != len(KCC_NAMES):
        raise ValueError(f"kcc must be a one-dimensional vector of length {len(KCC_NAMES)}")
    if not np.isfinite(arr).all():
        raise ValueError("kcc values must be finite")
    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("kcc values must be in [0, 1]")
    return arr


def _validate_susceptibility(susceptibility: Mapping[str, float]) -> dict[str, float]:
    required = {
        "repair_capacity": (_sus().repair_min, _sus().repair_max),
        "antioxidant_capacity": (_sus().antiox_min, _sus().antiox_max),
        "immune_surveillance": (_sus().immune_min, _sus().immune_max),
        "detox_balance": (_sus().detox_min, _sus().detox_max),
        "baseline_proliferation": (_sus().baseline_prolif_min, _sus().baseline_prolif_max),
    }
    missing = [name for name in required if name not in susceptibility]
    if missing:
        raise ValueError("susceptibility is missing required keys: " + ", ".join(missing))

    out: dict[str, float] = {}
    for name, (low, high) in required.items():
        value = float(susceptibility[name])
        if not np.isfinite(value):
            raise ValueError(f"susceptibility[{name!r}] must be finite")
        if value < low or value > high:
            raise ValueError(
                f"susceptibility[{name!r}]={value} is outside registry bounds [{low}, {high}]"
            )
        out[name] = value
    return out


def _bounded_retention(base_persistence: float, clearance) -> np.ndarray:
    """Return a positive monthly retention coefficient in ``(0, 1)``.

    ``clearance`` is interpreted as a non-negative reduction in retention; a
    negative value would silently invert the qAOP dynamics (clearance *boosts*
    retention) so it is floored at zero defensively. The registry's lognormal
    priors keep the upstream products positive in practice; this guard exists
    to catch future card edits or unusual ``prior_params``.
    """
    eps = np.finfo(float).eps
    base = np.clip(float(base_persistence), eps, 1.0 - eps)
    clearance = np.maximum(np.asarray(clearance, dtype=float), 0.0)
    return expit(np.log(base) - np.log1p(-base) - clearance)


_LATENT_RISK_CLIP_EPS = 1e-12


def _per_month_hazard(latent_risk):
    """Map per-month event probability ``p`` to hazard contribution ``-log(1-p)``.

    ``latent_risk`` is the sigmoid output from ``_qaop_step``, interpreted as
    a per-month event probability in ``[0, 1]``. The corresponding monthly
    hazard contribution is ``-log(1 - p)``, so that the horizon event
    probability is ``1 - exp(-Σ_t -log(1 - p_t)) = 1 - Π_t (1 - p_t)``.
    """
    p = np.clip(latent_risk, 0.0, 1.0 - _LATENT_RISK_CLIP_EPS)
    return -np.log1p(-p)


def _qaop_step(
    *,
    D: _DynamicsCoefficients,
    t: int,
    dna_adducts,
    ros,
    inflammation,
    epigenetic_age,
    clone_fraction,
    driver_count_proxy,
    repair_capacity,
    antioxidant,
    immune,
    detox,
    baseline_prolif,
    k1,
    k2,
    k3,
    k4,
    k5,
    k6,
    k7,
    k8,
    k9,
    k10,
    dose,
    pulse,
    clone_noise,
):
    """One month of the qAOP recurrence. Works element-wise on scalars or arrays.

    All RNG draws are passed in explicitly so callers control the random stream
    layout; the physics formulas live here and only here.
    """
    cyclical = D.dose_cyclical_baseline + D.dose_cyclical_amplitude * np.sin(
        2.0 * np.pi * t / D.dose_cyclical_period_months
    )
    internal_dose = dose * cyclical * pulse * detox

    adduct_input = (
        internal_dose
        * (D.dna_input_kcc1_intercept + D.dna_input_kcc1_slope * k1)
        * (D.dna_input_kcc2_intercept + D.dna_input_kcc2_slope * k2)
    )
    dna_retention = _bounded_retention(D.dna_decay, D.dna_repair_strength * repair_capacity)
    dna_adducts = dna_retention * dna_adducts + adduct_input
    dna_adducts = np.clip(dna_adducts, 0.0, D.dna_max)

    ros_input = internal_dose * (
        D.ros_input_intercept + D.ros_input_kcc5_slope * k5 + D.ros_input_kcc1_slope * k1
    )
    ros_retention = _bounded_retention(D.ros_decay, D.ros_antioxidant_strength * antioxidant)
    ros = ros_retention * ros + ros_input
    ros = np.clip(ros, 0.0, D.ros_max)

    cytotoxicity = expit(
        D.cyto_intercept + D.cyto_dna_coupling * dna_adducts + D.cyto_ros_coupling * ros
    )
    inflammation_input = (
        D.inf_ros_coupling * ros
        + D.inf_cyto_coupling * cytotoxicity
        + D.inf_dose_kcc6_coupling * internal_dose * k6
    )
    inflammation_retention = _bounded_retention(
        D.inf_decay, D.inf_immune_clearance * immune
    )
    inflammation = inflammation_retention * inflammation + inflammation_input
    inflammation = np.clip(inflammation, 0.0, D.inf_max)

    epigenetic_age = epigenetic_age + (
        D.epi_background_rate
        + internal_dose
        * (
            D.epi_dose_kcc4_coupling * k4
            + D.epi_dose_kcc5_coupling * k5
            + D.epi_dose_kcc6_coupling * k6
        )
        + D.epi_ros_coupling * ros
    )
    epigenetic_age = np.clip(epigenetic_age, 0.0, D.epi_max)

    proliferation = expit(
        D.prolif_intercept
        + baseline_prolif
        + D.prolif_inflammation_coupling * inflammation
        + D.prolif_cyto_coupling * cytotoxicity
        + D.prolif_dose_kcc8_coupling * k8 * internal_dose
        + D.prolif_kcc10_coupling * k10
        + D.prolif_epi_coupling * epigenetic_age
    )
    proliferation = np.clip(proliferation, 0.0, 1.0)

    # ``repair_capacity`` is clipped to ``[D.repair_min, D.repair_max]`` with
    # ``D.repair_max > 1``; ``max(0, 1 - repair_capacity)`` keeps the
    # repair-deficit term acting as a deficit (zero at full repair) instead of
    # turning into a bonus mutation suppressor for high-repair hosts.
    repair_deficit = np.maximum(0.0, 1.0 - repair_capacity)
    mutation_rate = D.mut_scale * (
        1.0
        + D.mut_dna_kcc2 * dna_adducts * k2
        + D.mut_ros * ros
        + D.mut_repair_deficit * repair_deficit
        + D.mut_prolif * proliferation
    )
    mutation_rate = np.clip(mutation_rate, 0.0, None)
    driver_count_proxy = driver_count_proxy + (
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
    log_growth = selection * (1.0 - clone_fraction) + clone_noise
    clone_fraction = clone_fraction * np.exp(log_growth)
    clone_fraction = np.clip(clone_fraction, D.clone_min, D.clone_max)

    latent_risk = expit(
        D.risk_intercept
        + D.risk_dna * np.log1p(dna_adducts)
        + D.risk_ros * np.log1p(ros)
        + D.risk_inf * np.log1p(inflammation)
        + D.risk_prolif * proliferation
        + D.risk_epi * np.log1p(epigenetic_age)
        + D.risk_driver * np.log1p(driver_count_proxy)
        + D.risk_clone * clone_fraction
        + D.risk_immune * immune
    )

    return {
        "DNA_adducts": dna_adducts,
        "ROS": ros,
        "inflammation": inflammation,
        "epigenetic_age": epigenetic_age,
        "proliferation": proliferation,
        "mutation_rate": mutation_rate,
        "clone_fraction": clone_fraction,
        "driver_count_proxy": driver_count_proxy,
        "immune_clearance": immune,
        "latent_risk": latent_risk,
    }


def simulate_state_trajectory(
    kcc: np.ndarray,
    dose: float,
    susceptibility: dict[str, float],
    months: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run the qAOP-like latent-state recurrence for one synthetic subject.

    All coefficients are sourced from the coefficient registry; see
    ``materials/coefficient_cards.yaml`` for provenance.
    """
    if not isinstance(months, int) or isinstance(months, bool) or months <= 0:
        raise ValueError("months must be a positive integer")
    if not np.isfinite(float(dose)) or dose < 0:
        raise ValueError("dose must be finite and non-negative")
    kcc = _validate_kcc_vector(kcc)
    susceptibility = _validate_susceptibility(susceptibility)
    D = _dyn()
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
    detox = susceptibility["detox_balance"]
    baseline_prolif = susceptibility["baseline_proliferation"]

    state = {
        "DNA_adducts": 0.0,
        "ROS": 0.0,
        "inflammation": 0.0,
        "epigenetic_age": 0.0,
        "clone_fraction": D.clone_init_scale * float(
            rng.lognormal(mean=0.0, sigma=D.clone_init_sigma)
        ),
        "driver_count_proxy": 0.0,
    }

    rows: list[dict[str, float | int]] = []
    for t in range(months):
        pulse = float(
            rng.lognormal(mean=D.dose_pulse_lognormal_mean, sigma=D.dose_pulse_lognormal_sigma)
        )
        clone_noise = float(rng.normal(0.0, D.clone_selection_noise_sigma))
        step = _qaop_step(
            D=D,
            t=t,
            dna_adducts=state["DNA_adducts"],
            ros=state["ROS"],
            inflammation=state["inflammation"],
            epigenetic_age=state["epigenetic_age"],
            clone_fraction=state["clone_fraction"],
            driver_count_proxy=state["driver_count_proxy"],
            repair_capacity=repair_capacity,
            antioxidant=antioxidant,
            immune=immune,
            detox=detox,
            baseline_prolif=baseline_prolif,
            k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
            k6=k6, k7=k7, k8=k8, k9=k9, k10=k10,
            dose=float(dose),
            pulse=pulse,
            clone_noise=clone_noise,
        )
        # Carry forward the persistent state for the next iteration.
        state["DNA_adducts"] = float(step["DNA_adducts"])
        state["ROS"] = float(step["ROS"])
        state["inflammation"] = float(step["inflammation"])
        state["epigenetic_age"] = float(step["epigenetic_age"])
        state["clone_fraction"] = float(step["clone_fraction"])
        state["driver_count_proxy"] = float(step["driver_count_proxy"])

        rows.append(
            {
                "month": t + 1,
                "DNA_adducts": float(step["DNA_adducts"]),
                "ROS": float(step["ROS"]),
                "inflammation": float(step["inflammation"]),
                "epigenetic_age": float(step["epigenetic_age"]),
                "proliferation": float(step["proliferation"]),
                "mutation_rate": float(step["mutation_rate"]),
                "clone_fraction": float(step["clone_fraction"]),
                "driver_count_proxy": float(step["driver_count_proxy"]),
                "immune_clearance": float(step["immune_clearance"]),
                "latent_risk": float(step["latent_risk"]),
            }
        )
    trajectory = pd.DataFrame(rows)
    trajectory.attrs["cumulative_latent_risk"] = float(
        _per_month_hazard(trajectory["latent_risk"].to_numpy(dtype=float)).sum()
    )
    return trajectory


def summarize_trajectory(traj: pd.DataFrame) -> dict[str, float]:
    """Summarize a per-month trajectory into final and AUC state features.

    ``state_cumulative_latent_risk`` is the cumulative hazard
    ``Σ_t -log(1 - p_t)`` derived from each month's sigmoid output ``p_t`` —
    not the raw sum of probabilities. This makes
    ``_event_probability_from_cumulative_risk`` a correct survival expression
    (``1 - exp(-hazard_scale * Σ -log(1 - p_t))``), which collapses to the
    canonical ``1 - Π (1 - p_t)`` when ``hazard_scale = 1``.
    """
    out: dict[str, float] = {}
    try:
        trapezoid = np.trapezoid
    except AttributeError:  # NumPy < 2.0
        trapezoid = np.trapz
    for col in STATE_NAMES:
        values = traj[col].to_numpy(dtype=float)
        out[f"state_final_{col}"] = float(traj[col].iloc[-1])
        if len(values) == 1:
            out[f"state_auc_{col}"] = float(values[0])
        else:
            # Trapezoidal integral has length ``months - 1``; dividing by that
            # produces the time-averaged state. Dividing by ``len(values)``
            # would under-report by a factor of (months - 1) / months — a
            # ~1.4% bias at months=72 but ~8% at months=12.
            out[f"state_auc_{col}"] = float(trapezoid(values, dx=1.0) / (len(values) - 1))
    out["state_cumulative_latent_risk"] = float(
        _per_month_hazard(traj["latent_risk"].to_numpy(dtype=float)).sum()
    )
    return out


def _event_probability_from_cumulative_risk(
    hazard_scale: float,
    cumulative_latent_risk,
):
    """Survival-model event probability from a cumulative hazard.

    ``cumulative_latent_risk`` must already be a cumulative hazard, i.e.
    ``Σ_t -log(1 - p_t)`` when ``p_t`` is the per-month event probability
    (see :func:`_per_month_hazard`). The returned probability is then
    ``1 - exp(-hazard_scale * cumulative)``; with ``hazard_scale = 1`` this
    is the canonical ``1 - Π_t (1 - p_t)``.
    """
    cumulative = np.clip(cumulative_latent_risk, 0.0, None)
    return 1.0 - np.exp(-hazard_scale * cumulative)


def _sample_susceptibility(rng: np.random.Generator) -> dict[str, float]:
    s = _sus()
    return {
        "repair_capacity": float(np.clip(rng.normal(s.repair_mean, s.repair_sd), s.repair_min, s.repair_max)),
        "antioxidant_capacity": float(
            np.clip(rng.normal(s.antiox_mean, s.antiox_sd), s.antiox_min, s.antiox_max)
        ),
        "immune_surveillance": float(
            np.clip(rng.normal(s.immune_mean, s.immune_sd), s.immune_min, s.immune_max)
        ),
        "detox_balance": float(
            np.clip(
                rng.lognormal(mean=s.detox_lognormal_mean, sigma=s.detox_lognormal_sigma),
                s.detox_min,
                s.detox_max,
            )
        ),
        "baseline_proliferation": float(
            np.clip(
                rng.normal(s.baseline_prolif_mean, s.baseline_prolif_sd),
                s.baseline_prolif_min,
                s.baseline_prolif_max,
            )
        ),
    }


def simulate_cohort(
    cfg: SimConfig,
    calibration: CalibrationBundle | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Generate a synthetic ICg-CaST cohort and representative trajectories.

    If ``calibration`` is provided, calibrated mutational-signature profiles
    and per-chemical KCC archetypes (e.g. from ToxCast) override the default
    constants. In ``coefficient_mode="prior_sample"``, numeric coefficient
    cards are sampled once per cohort from their evidence-level priors.
    """
    cfg.validate()
    if cfg.coefficient_mode == "point":
        _clear_coefficient_caches()
        try:
            return _simulate_cohort_impl(cfg, calibration=calibration, coefficient_seed=-1)
        finally:
            _clear_coefficient_caches()

    coefficient_seed = cfg.resolved_coefficient_seed()
    sampled = sampled_registry(seed=coefficient_seed)
    with use_registry(sampled):
        _clear_coefficient_caches()
        try:
            return _simulate_cohort_impl(
                cfg,
                calibration=calibration,
                coefficient_seed=coefficient_seed,
            )
        finally:
            _clear_coefficient_caches()


def _simulate_cohort_impl(
    cfg: SimConfig,
    *,
    calibration: CalibrationBundle | None,
    coefficient_seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if cfg.simulator_backend == "vectorized":
        return _simulate_cohort_impl_vectorized(
            cfg,
            calibration=calibration,
            coefficient_seed=coefficient_seed,
        )
    return _simulate_cohort_impl_python(
        cfg,
        calibration=calibration,
        coefficient_seed=coefficient_seed,
    )


def _simulate_cohort_impl_python(
    cfg: SimConfig,
    *,
    calibration: CalibrationBundle | None,
    coefficient_seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(cfg.seed)
    sig_labels, sig_profiles = make_signature_profiles(calibration=calibration)
    archetype_table: Mapping[str, tuple[float, ...]] = (
        calibration.archetype_kcc_arrays()
        if calibration is not None
        else _active_archetype_table()
    )
    rows: list[dict[str, float | int | str]] = []
    trajectories: dict[str, pd.DataFrame] = {}
    r = _registry()
    high_risk_quantile = r.get("cohort.high_risk_quantile")
    n_archetypes_to_retain = len(ARCHETYPE_ORDER)

    for i in range(cfg.n):
        archetype = sample_archetype(
            rng,
            dict(cfg.archetype_prior) if cfg.archetype_prior else None,
            archetype_table=archetype_table,
        )
        kcc = noisy_kcc_vector(archetype, rng, archetype_table=archetype_table)
        dose = float(
            np.clip(
                rng.lognormal(mean=cfg.dose_lognormal_mean, sigma=cfg.dose_lognormal_sigma),
                cfg.dose_min,
                cfg.dose_max,
            )
        )
        susceptibility = _sample_susceptibility(rng)
        traj = simulate_state_trajectory(kcc, dose, susceptibility, cfg.months, rng)
        states = summarize_trajectory(traj)
        omics = generate_omics(
            archetype,
            kcc,
            states,
            susceptibility,
            rng,
            sig_labels,
            sig_profiles,
            calibration=calibration,
        )

        event_probability = float(
            _event_probability_from_cumulative_risk(
                cfg.event_hazard_scale,
                states["state_cumulative_latent_risk"],
            )
        )
        row: dict[str, float | int | str] = {
            "sample_id": f"ICG_{i:05d}",
            "chemical_archetype": archetype,
            "dose": dose,
            "coefficient_seed": coefficient_seed,
            "future_cancer_transition_event": int(rng.uniform() < event_probability),
            "future_event_probability": event_probability,
        }
        for j, name in enumerate(KCC_NAMES, start=1):
            row[f"kcc{j}_{name}"] = float(kcc[j - 1])
        for key, val in susceptibility.items():
            row[f"host_{key}"] = val
        row.update(states)
        row.update(omics)
        rows.append(row)

        if cfg.retain_trajectories:
            trajectories[str(row["sample_id"])] = traj.assign(
                sample_id=row["sample_id"],
                chemical_archetype=archetype,
            )
        elif len(trajectories) < n_archetypes_to_retain and archetype not in trajectories:
            trajectories[archetype] = traj.assign(sample_id=row["sample_id"], chemical_archetype=archetype)

    df = pd.DataFrame(rows)
    cutoff = float(df["state_final_latent_risk"].quantile(high_risk_quantile))
    df["high_risk_transition_state"] = (df["state_final_latent_risk"] >= cutoff).astype(int)
    return df, trajectories


def _simulate_state_trajectories_vectorized(
    kcc_matrix: np.ndarray,
    doses: np.ndarray,
    susceptibility_rows: list[dict[str, float]],
    months: int,
    rng: np.random.Generator,
    retained_indices: set[int],
) -> tuple[list[dict[str, float]], dict[int, pd.DataFrame]]:
    """Run the monthly recurrence for all subjects in vectorised NumPy arrays.

    Validates each KCC row and susceptibility dict before running, matching
    the per-subject validation that ``simulate_state_trajectory`` performs in
    the Python backend; the two backends are otherwise expected to be drop-in
    interchangeable.
    """
    if not isinstance(months, int) or isinstance(months, bool) or months <= 0:
        raise ValueError("months must be a positive integer")
    kcc_matrix = np.asarray(kcc_matrix, dtype=float)
    doses = np.asarray(doses, dtype=float)
    if kcc_matrix.ndim != 2 or kcc_matrix.shape[1] != len(KCC_NAMES):
        raise ValueError(
            f"kcc_matrix must have shape (n, {len(KCC_NAMES)}); got {kcc_matrix.shape}"
        )
    if doses.shape != (kcc_matrix.shape[0],):
        raise ValueError(
            f"doses must be a 1-d array of length {kcc_matrix.shape[0]}; got {doses.shape}"
        )
    if not np.isfinite(doses).all() or np.any(doses < 0):
        raise ValueError("doses must be finite and non-negative")
    if len(susceptibility_rows) != kcc_matrix.shape[0]:
        raise ValueError(
            f"susceptibility_rows must have length {kcc_matrix.shape[0]}; "
            f"got {len(susceptibility_rows)}"
        )
    for i in range(kcc_matrix.shape[0]):
        _validate_kcc_vector(kcc_matrix[i])
        susceptibility_rows[i] = _validate_susceptibility(susceptibility_rows[i])

    D = _dyn()
    n = int(kcc_matrix.shape[0])
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = kcc_matrix.T

    repair = np.array([s["repair_capacity"] for s in susceptibility_rows], dtype=float)
    antioxidant_raw = np.array([s["antioxidant_capacity"] for s in susceptibility_rows], dtype=float)
    immune_raw = np.array([s["immune_surveillance"] for s in susceptibility_rows], dtype=float)
    detox = np.array([s["detox_balance"] for s in susceptibility_rows], dtype=float)
    baseline_prolif = np.array([s["baseline_proliferation"] for s in susceptibility_rows], dtype=float)

    repair_capacity = np.clip(
        repair * (1.0 - D.repair_kcc3_inhibition * k3),
        D.repair_min,
        D.repair_max,
    )
    antioxidant = np.clip(
        antioxidant_raw * (1.0 - D.antiox_kcc5_depletion * k5),
        D.antiox_min,
        D.antiox_max,
    )
    immune = np.clip(
        immune_raw * (1.0 - D.immune_kcc7_inhibition * k7),
        D.immune_min,
        D.immune_max,
    )

    dna_adducts = np.zeros(n, dtype=float)
    ros = np.zeros(n, dtype=float)
    inflammation = np.zeros(n, dtype=float)
    epigenetic_age = np.zeros(n, dtype=float)
    clone_fraction = D.clone_init_scale * rng.lognormal(
        mean=0.0,
        sigma=D.clone_init_sigma,
        size=n,
    )
    driver_count_proxy = np.zeros(n, dtype=float)
    cumulative_latent_risk = np.zeros(n, dtype=float)

    auc = {name: np.zeros(n, dtype=float) for name in STATE_NAMES}
    retained_rows: dict[int, list[dict[str, float | int]]] = {
        idx: [] for idx in sorted(retained_indices)
    }

    last_step: dict[str, np.ndarray] | None = None
    for t in range(months):
        pulse = rng.lognormal(
            mean=D.dose_pulse_lognormal_mean,
            sigma=D.dose_pulse_lognormal_sigma,
            size=n,
        )
        clone_noise = rng.normal(0.0, D.clone_selection_noise_sigma, size=n)
        step = _qaop_step(
            D=D,
            t=t,
            dna_adducts=dna_adducts,
            ros=ros,
            inflammation=inflammation,
            epigenetic_age=epigenetic_age,
            clone_fraction=clone_fraction,
            driver_count_proxy=driver_count_proxy,
            repair_capacity=repair_capacity,
            antioxidant=antioxidant,
            immune=immune,
            detox=detox,
            baseline_prolif=baseline_prolif,
            k1=k1, k2=k2, k3=k3, k4=k4, k5=k5,
            k6=k6, k7=k7, k8=k8, k9=k9, k10=k10,
            dose=doses,
            pulse=pulse,
            clone_noise=clone_noise,
        )
        # Carry persistent state forward; ``immune_clearance`` is constant.
        dna_adducts = step["DNA_adducts"]
        ros = step["ROS"]
        inflammation = step["inflammation"]
        epigenetic_age = step["epigenetic_age"]
        clone_fraction = step["clone_fraction"]
        driver_count_proxy = step["driver_count_proxy"]
        last_step = step
        cumulative_latent_risk += _per_month_hazard(step["latent_risk"])

        if months > 1:
            weight = 0.5 if t in (0, months - 1) else 1.0
            for name in STATE_NAMES:
                auc[name] += weight * step[name]

        for idx, rows in retained_rows.items():
            rows.append({"month": t + 1, **{name: float(step[name][idx]) for name in STATE_NAMES}})

    assert last_step is not None  # months is validated > 0 by the caller path
    state_rows: list[dict[str, float]] = []
    for i in range(n):
        row: dict[str, float] = {}
        for name in STATE_NAMES:
            row[f"state_final_{name}"] = float(last_step[name][i])
            if months == 1:
                row[f"state_auc_{name}"] = float(last_step[name][i])
            else:
                # Trapezoidal integral length is months - 1; match the Python
                # backend's time-averaged interpretation.
                row[f"state_auc_{name}"] = float(auc[name][i] / (months - 1))
        row["state_cumulative_latent_risk"] = float(cumulative_latent_risk[i])
        state_rows.append(row)

    trajectories = {
        idx: pd.DataFrame(rows)
        for idx, rows in retained_rows.items()
    }
    for trajectory in trajectories.values():
        trajectory.attrs["cumulative_latent_risk"] = float(
            _per_month_hazard(trajectory["latent_risk"].to_numpy(dtype=float)).sum()
        )
    return state_rows, trajectories


def _simulate_cohort_impl_vectorized(
    cfg: SimConfig,
    *,
    calibration: CalibrationBundle | None,
    coefficient_seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rng = np.random.default_rng(cfg.seed)
    sig_labels, sig_profiles = make_signature_profiles(calibration=calibration)
    archetype_table: Mapping[str, tuple[float, ...]] = (
        calibration.archetype_kcc_arrays()
        if calibration is not None
        else _active_archetype_table()
    )
    r = _registry()
    high_risk_quantile = r.get("cohort.high_risk_quantile")
    n_archetypes_to_retain = len(ARCHETYPE_ORDER)

    archetypes: list[str] = []
    kcc_rows: list[np.ndarray] = []
    doses: list[float] = []
    susceptibility_rows: list[dict[str, float]] = []
    retained_indices: set[int] = set()
    retained_archetypes: set[str] = set()

    for i in range(cfg.n):
        archetype = sample_archetype(
            rng,
            dict(cfg.archetype_prior) if cfg.archetype_prior else None,
            archetype_table=archetype_table,
        )
        kcc = noisy_kcc_vector(archetype, rng, archetype_table=archetype_table)
        dose = float(
            np.clip(
                rng.lognormal(mean=cfg.dose_lognormal_mean, sigma=cfg.dose_lognormal_sigma),
                cfg.dose_min,
                cfg.dose_max,
            )
        )
        susceptibility = _sample_susceptibility(rng)

        archetypes.append(archetype)
        kcc_rows.append(kcc)
        doses.append(dose)
        susceptibility_rows.append(susceptibility)
        if cfg.retain_trajectories:
            retained_indices.add(i)
        elif len(retained_archetypes) < n_archetypes_to_retain and archetype not in retained_archetypes:
            retained_indices.add(i)
            retained_archetypes.add(archetype)

    states_rows, retained_trajectories = _simulate_state_trajectories_vectorized(
        np.vstack(kcc_rows),
        np.asarray(doses, dtype=float),
        susceptibility_rows,
        cfg.months,
        rng,
        retained_indices,
    )

    rows: list[dict[str, float | int | str]] = []
    trajectories: dict[str, pd.DataFrame] = {}
    for i, (archetype, kcc, dose, susceptibility, states) in enumerate(
        zip(archetypes, kcc_rows, doses, susceptibility_rows, states_rows, strict=True)
    ):
        omics = generate_omics(
            archetype,
            kcc,
            states,
            susceptibility,
            rng,
            sig_labels,
            sig_profiles,
            calibration=calibration,
        )
        event_probability = float(
            _event_probability_from_cumulative_risk(
                cfg.event_hazard_scale,
                states["state_cumulative_latent_risk"],
            )
        )
        row: dict[str, float | int | str] = {
            "sample_id": f"ICG_{i:05d}",
            "chemical_archetype": archetype,
            "dose": float(dose),
            "coefficient_seed": coefficient_seed,
            "future_cancer_transition_event": int(rng.uniform() < event_probability),
            "future_event_probability": event_probability,
        }
        for j, name in enumerate(KCC_NAMES, start=1):
            row[f"kcc{j}_{name}"] = float(kcc[j - 1])
        for key, val in susceptibility.items():
            row[f"host_{key}"] = val
        row.update(states)
        row.update(omics)
        rows.append(row)

        if i in retained_trajectories:
            trajectory_key = str(row["sample_id"]) if cfg.retain_trajectories else archetype
            trajectories[trajectory_key] = retained_trajectories[i].assign(
                sample_id=row["sample_id"],
                chemical_archetype=archetype,
            )

    df = pd.DataFrame(rows)
    cutoff = float(df["state_final_latent_risk"].quantile(high_risk_quantile))
    df["high_risk_transition_state"] = (df["state_final_latent_risk"] >= cutoff).astype(int)
    return df, trajectories
