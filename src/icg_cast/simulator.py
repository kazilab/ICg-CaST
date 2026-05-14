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
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import expit

from .coefficients import registry as _registry
from .coefficients import sampled_registry, use_registry
from .config import SimConfig
from .constants import ARCHETYPE_KCC, ARCHETYPE_ORDER, KCC_NAMES, STATE_NAMES
from .omics import _omics_coefficients, generate_omics
from .signatures import _sig_coeffs, make_signature_profiles

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


@functools.cache
def _dyn() -> _DynamicsCoefficients:
    r = _registry()
    g = r.get
    return _DynamicsCoefficients(
        repair_kcc3_inhibition=g("dynamics.repair.kcc3_inhibition"),
        repair_min=g("dynamics.repair.min"),
        repair_max=g("dynamics.repair.max"),
        immune_kcc7_inhibition=g("dynamics.immune.kcc7_inhibition"),
        immune_min=g("dynamics.immune.min"),
        immune_max=g("dynamics.immune.max"),
        dose_cyclical_baseline=g("dynamics.dose.cyclical_baseline"),
        dose_cyclical_amplitude=g("dynamics.dose.cyclical_amplitude"),
        dose_cyclical_period_months=g("dynamics.dose.cyclical_period_months"),
        dose_pulse_lognormal_mean=g("dynamics.dose.pulse_lognormal_mean"),
        dose_pulse_lognormal_sigma=g("dynamics.dose.pulse_lognormal_sigma"),
        dna_decay=g("dynamics.dna_adducts.decay"),
        dna_repair_strength=g("dynamics.dna_adducts.repair_strength"),
        dna_max=g("dynamics.dna_adducts.max"),
        dna_input_kcc1_intercept=g("dynamics.dna_adducts.input_kcc1_intercept"),
        dna_input_kcc1_slope=g("dynamics.dna_adducts.input_kcc1_slope"),
        dna_input_kcc2_intercept=g("dynamics.dna_adducts.input_kcc2_intercept"),
        dna_input_kcc2_slope=g("dynamics.dna_adducts.input_kcc2_slope"),
        ros_decay=g("dynamics.ros.decay"),
        ros_antioxidant_strength=g("dynamics.ros.antioxidant_strength"),
        ros_max=g("dynamics.ros.max"),
        ros_input_intercept=g("dynamics.ros.input_intercept"),
        ros_input_kcc5_slope=g("dynamics.ros.input_kcc5_slope"),
        ros_input_kcc1_slope=g("dynamics.ros.input_kcc1_slope"),
        cyto_intercept=g("dynamics.cytotoxicity.intercept"),
        cyto_dna_coupling=g("dynamics.cytotoxicity.dna_coupling"),
        cyto_ros_coupling=g("dynamics.cytotoxicity.ros_coupling"),
        inf_decay=g("dynamics.inflammation.decay"),
        inf_ros_coupling=g("dynamics.inflammation.ros_coupling"),
        inf_cyto_coupling=g("dynamics.inflammation.cytotoxicity_coupling"),
        inf_dose_kcc6_coupling=g("dynamics.inflammation.dose_kcc6_coupling"),
        inf_immune_clearance=g("dynamics.inflammation.immune_clearance"),
        inf_max=g("dynamics.inflammation.max"),
        epi_background_rate=g("dynamics.epigenetic_age.background_rate"),
        epi_dose_kcc4_coupling=g("dynamics.epigenetic_age.dose_kcc4_coupling"),
        epi_dose_kcc5_coupling=g("dynamics.epigenetic_age.dose_kcc5_coupling"),
        epi_dose_kcc6_coupling=g("dynamics.epigenetic_age.dose_kcc6_coupling"),
        epi_ros_coupling=g("dynamics.epigenetic_age.ros_coupling"),
        epi_max=g("dynamics.epigenetic_age.max"),
        prolif_intercept=g("dynamics.proliferation.intercept"),
        prolif_inflammation_coupling=g("dynamics.proliferation.inflammation_coupling"),
        prolif_cyto_coupling=g("dynamics.proliferation.cytotoxicity_coupling"),
        prolif_dose_kcc8_coupling=g("dynamics.proliferation.dose_kcc8_coupling"),
        prolif_kcc10_coupling=g("dynamics.proliferation.kcc10_coupling"),
        prolif_epi_coupling=g("dynamics.proliferation.epigenetic_coupling"),
        mut_scale=g("dynamics.mutation_rate.scale"),
        mut_dna_kcc2=g("dynamics.mutation_rate.dna_kcc2_coupling"),
        mut_ros=g("dynamics.mutation_rate.ros_coupling"),
        mut_repair_deficit=g("dynamics.mutation_rate.repair_deficit_coupling"),
        mut_prolif=g("dynamics.mutation_rate.proliferation_coupling"),
        driver_intercept=g("dynamics.driver_count.intercept"),
        driver_prolif=g("dynamics.driver_count.proliferation_coupling"),
        driver_kcc9=g("dynamics.driver_count.kcc9_coupling"),
        clone_init_scale=g("dynamics.clone_fraction.init_scale"),
        clone_init_sigma=g("dynamics.clone_fraction.init_lognormal_sigma"),
        clone_min=g("dynamics.clone_fraction.min"),
        clone_max=g("dynamics.clone_fraction.max"),
        clone_selection_intercept=g("dynamics.clone_fraction.selection_intercept"),
        clone_selection_prolif=g("dynamics.clone_fraction.selection_proliferation"),
        clone_selection_inf=g("dynamics.clone_fraction.selection_inflammation"),
        clone_selection_epi=g("dynamics.clone_fraction.selection_epigenetic"),
        clone_selection_kcc9=g("dynamics.clone_fraction.selection_kcc9"),
        clone_selection_immune=g("dynamics.clone_fraction.selection_immune"),
        clone_selection_drivers=g("dynamics.clone_fraction.selection_drivers"),
        clone_selection_noise_sigma=g("dynamics.clone_fraction.selection_noise_sigma"),
        risk_intercept=g("dynamics.latent_risk.intercept"),
        risk_dna=g("dynamics.latent_risk.dna_coupling"),
        risk_ros=g("dynamics.latent_risk.ros_coupling"),
        risk_inf=g("dynamics.latent_risk.inflammation_coupling"),
        risk_prolif=g("dynamics.latent_risk.proliferation_coupling"),
        risk_epi=g("dynamics.latent_risk.epigenetic_coupling"),
        risk_driver=g("dynamics.latent_risk.driver_coupling"),
        risk_clone=g("dynamics.latent_risk.clone_coupling"),
        risk_immune=g("dynamics.latent_risk.immune_coupling"),
    )


@functools.cache
def _sus() -> _SusceptibilityCoefficients:
    r = _registry()
    g = r.get
    return _SusceptibilityCoefficients(
        repair_mean=g("susceptibility.repair_capacity.mean"),
        repair_sd=g("susceptibility.repair_capacity.sd"),
        repair_min=g("susceptibility.repair_capacity.min"),
        repair_max=g("susceptibility.repair_capacity.max"),
        antiox_mean=g("susceptibility.antioxidant_capacity.mean"),
        antiox_sd=g("susceptibility.antioxidant_capacity.sd"),
        antiox_min=g("susceptibility.antioxidant_capacity.min"),
        antiox_max=g("susceptibility.antioxidant_capacity.max"),
        immune_mean=g("susceptibility.immune_surveillance.mean"),
        immune_sd=g("susceptibility.immune_surveillance.sd"),
        immune_min=g("susceptibility.immune_surveillance.min"),
        immune_max=g("susceptibility.immune_surveillance.max"),
        detox_lognormal_mean=g("susceptibility.detox_balance.lognormal_mean"),
        detox_lognormal_sigma=g("susceptibility.detox_balance.lognormal_sigma"),
        detox_min=g("susceptibility.detox_balance.min"),
        detox_max=g("susceptibility.detox_balance.max"),
        baseline_prolif_mean=g("susceptibility.baseline_proliferation.mean"),
        baseline_prolif_sd=g("susceptibility.baseline_proliferation.sd"),
        baseline_prolif_min=g("susceptibility.baseline_proliferation.min"),
        baseline_prolif_max=g("susceptibility.baseline_proliferation.max"),
    )


@functools.cache
def _arch() -> _ArchetypeCoefficients:
    r = _registry()
    return _ArchetypeCoefficients(
        sample_prior=r.get_vector("archetypes.sample_prior"),
        kcc_noise_sigma=r.get("archetypes.kcc_noise_sigma"),
    )


def _clear_coefficient_caches() -> None:
    """Clear coefficient-derived caches after the active registry changes."""
    _dyn.cache_clear()
    _sus.cache_clear()
    _arch.cache_clear()
    _omics_coefficients.cache_clear()
    _sig_coeffs.cache_clear()


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
    D = _dyn()
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = kcc

    repair_capacity = float(
        np.clip(
            susceptibility["repair_capacity"] * (1.0 - D.repair_kcc3_inhibition * k3),
            D.repair_min,
            D.repair_max,
        )
    )
    antioxidant = susceptibility["antioxidant_capacity"]
    immune = float(
        np.clip(
            susceptibility["immune_surveillance"] * (1.0 - D.immune_kcc7_inhibition * k7),
            D.immune_min,
            D.immune_max,
        )
    )
    detox = susceptibility["detox_balance"]
    baseline_prolif = susceptibility["baseline_proliferation"]

    dna_adducts = 0.0
    ros = 0.0
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
        pulse = rng.lognormal(mean=D.dose_pulse_lognormal_mean, sigma=D.dose_pulse_lognormal_sigma)
        internal_dose = dose * cyclical * pulse * detox

        adduct_input = (
            internal_dose
            * (D.dna_input_kcc1_intercept + D.dna_input_kcc1_slope * k1)
            * (D.dna_input_kcc2_intercept + D.dna_input_kcc2_slope * k2)
        )
        dna_adducts = (
            D.dna_decay * dna_adducts
            + adduct_input
            - D.dna_repair_strength * repair_capacity * dna_adducts
        )
        dna_adducts = float(np.clip(dna_adducts, 0.0, D.dna_max))

        ros_input = internal_dose * (
            D.ros_input_intercept + D.ros_input_kcc5_slope * k5 + D.ros_input_kcc1_slope * k1
        )
        ros = D.ros_decay * ros + ros_input - D.ros_antioxidant_strength * antioxidant * ros
        ros = float(np.clip(ros, 0.0, D.ros_max))

        cytotoxicity = expit(
            D.cyto_intercept + D.cyto_dna_coupling * dna_adducts + D.cyto_ros_coupling * ros
        )
        inflammation = (
            D.inf_decay * inflammation
            + D.inf_ros_coupling * ros
            + D.inf_cyto_coupling * cytotoxicity
            + D.inf_dose_kcc6_coupling * internal_dose * k6
            - D.inf_immune_clearance * immune * inflammation
        )
        inflammation = float(np.clip(inflammation, 0.0, D.inf_max))

        epigenetic_age += (
            D.epi_background_rate
            + internal_dose
            * (
                D.epi_dose_kcc4_coupling * k4
                + D.epi_dose_kcc5_coupling * k5
                + D.epi_dose_kcc6_coupling * k6
            )
            + D.epi_ros_coupling * ros
        )
        epigenetic_age = float(np.clip(epigenetic_age, 0.0, D.epi_max))

        proliferation = expit(
            D.prolif_intercept
            + baseline_prolif
            + D.prolif_inflammation_coupling * inflammation
            + D.prolif_cyto_coupling * cytotoxicity
            + D.prolif_dose_kcc8_coupling * k8 * internal_dose
            + D.prolif_kcc10_coupling * k10
            + D.prolif_epi_coupling * epigenetic_age
        )
        proliferation = float(np.clip(proliferation, 0.0, 1.0))

        mutation_rate = D.mut_scale * (
            1.0
            + D.mut_dna_kcc2 * dna_adducts * k2
            + D.mut_ros * ros
            + D.mut_repair_deficit * (1.0 - repair_capacity)
            + D.mut_prolif * proliferation
        )
        mutation_rate = float(np.clip(mutation_rate, 0.0, 1.0))
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
        noise = rng.normal(0.0, D.clone_selection_noise_sigma)
        clone_fraction = clone_fraction + clone_fraction * (
            selection * (1.0 - clone_fraction) + noise
        )
        clone_fraction = float(np.clip(clone_fraction, D.clone_min, D.clone_max))

        latent_risk = expit(
            D.risk_intercept
            + D.risk_dna * math.log1p(dna_adducts)
            + D.risk_ros * math.log1p(ros)
            + D.risk_inf * math.log1p(inflammation)
            + D.risk_prolif * proliferation
            + D.risk_epi * math.log1p(epigenetic_age)
            + D.risk_driver * math.log1p(driver_count_proxy)
            + D.risk_clone * clone_fraction
            - D.risk_immune * immune
        )
        rows.append(
            {
                "month": t + 1,
                "DNA_adducts": dna_adducts,
                "ROS": ros,
                "inflammation": inflammation,
                "epigenetic_age": epigenetic_age,
                "proliferation": proliferation,
                "mutation_rate": mutation_rate,
                "clone_fraction": clone_fraction,
                "driver_count_proxy": driver_count_proxy,
                "immune_clearance": immune,
                "latent_risk": float(latent_risk),
            }
        )
    return pd.DataFrame(rows)


def summarize_trajectory(traj: pd.DataFrame) -> dict[str, float]:
    """Summarize a per-month trajectory into final and AUC state features."""
    out: dict[str, float] = {}
    for col in STATE_NAMES:
        out[f"state_final_{col}"] = float(traj[col].iloc[-1])
        out[f"state_auc_{col}"] = float(np.trapezoid(traj[col].to_numpy(), dx=1.0) / max(1, len(traj)))
    return out


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

        final_risk = states["state_final_latent_risk"]
        event_probability = 1.0 - math.exp(-cfg.event_hazard_scale * cfg.months * final_risk)
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

        if len(trajectories) < n_archetypes_to_retain and archetype not in trajectories:
            trajectories[archetype] = traj.assign(sample_id=row["sample_id"], chemical_archetype=archetype)

    df = pd.DataFrame(rows)
    cutoff = float(df["state_final_latent_risk"].quantile(high_risk_quantile))
    df["high_risk_transition_state"] = (df["state_final_latent_risk"] >= cutoff).astype(int)
    return df, trajectories
