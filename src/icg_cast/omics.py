"""Synthetic multi-omics observation model.

Every module coefficient, the per-modality measurement noise, the
signature-mixing weights, and the Poisson rate for the total mutation
count are sourced from the coefficient registry — see
:mod:`icg_cast.coefficients` and the ``omics.*`` block in
``materials/coefficient_cards.yaml``.

Each transcriptomic and epigenomic module is expressed as a linear
combination of named input features. The ``PRC2_mitotic_clock`` recipe
is rewritten in linear form using ``epi_age * proliferation`` as a
synthetic input so every recipe is a flat list of ``(input, coef)``
pairs.
"""

from __future__ import annotations

import functools
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.special import expit

from .coefficients import register_registry_derived_cache
from .coefficients import registry as _registry
from .constants import ARCHETYPE_SIGNATURE, EPI_MODULES, TRANSCRIPT_MODULES

if TYPE_CHECKING:
    from .calibration.bundle import CalibrationBundle

# ----------------------------------------------------------------------
# Recipe table — declarative list of inputs per module.
# Each entry is (module_name, list_of_input_names). The actual numeric
# coefficient for (module, input) is looked up in the registry under
#   omics.transcript.<module>.<input>_coeff
# (or omics.epi.<module>.<input>_coeff for the epi block).
# The "input" name maps onto one of the entries in the ``inputs`` dict
# constructed inside generate_omics().
# ----------------------------------------------------------------------

_TX_INPUTS: dict[str, list[str]] = {
    "DNA_adduct_response":         ["dna", "kcc1", "kcc2"],
    "p53_checkpoint":              ["dna", "ros", "kcc2"],
    "base_excision_repair":        ["ros", "kcc3", "susceptibility_repair"],
    "nucleotide_excision_repair":  ["dna", "kcc3", "susceptibility_repair"],
    "xenobiotic_metabolism_CYP":   ["kcc1", "kcc8", "dna"],
    "oxidative_stress_response":   ["ros", "kcc5"],
    "mitochondrial_dysfunction":   ["ros", "inflammation"],
    "inflammatory_cytokines":      ["inflammation", "kcc6"],
    "NFkB_activation":             ["inflammation", "ros", "kcc6"],
    "cell_cycle_E2F":              ["proliferation", "kcc10", "clone"],
    "replicative_DNA_synthesis":   ["proliferation", "kcc8", "kcc10"],
    "apoptosis_escape":            ["kcc9", "clone", "inflammation"],
    "ECM_fibrosis":                ["inflammation", "epi_age", "kcc4"],
    "angiogenesis_nutrient_supply":["proliferation", "inflammation", "kcc10"],
    "immune_evasion":              ["kcc7", "clone", "susceptibility_immune"],
    "nuclear_receptor_program":    ["kcc8", "proliferation"],
    "stemness_PRC2":               ["epi_age", "kcc4", "drivers"],
    "senescence_SASP":             ["ros", "inflammation", "epi_age"],
}

_EPI_INPUTS: dict[str, list[str]] = {
    "epigenetic_age_acceleration":      ["epi_age", "kcc4", "ros"],
    "PRC2_mitotic_clock":               ["epi_age", "epi_age_times_proliferation", "drivers"],
    "global_hypomethylation":           ["epi_age", "kcc4", "ros"],
    "tumor_suppressor_hypermethylation":["epi_age", "kcc4", "clone"],
    "enhancer_reprogramming":           ["kcc4", "kcc8", "inflammation"],
    "histone_activation_loss":          ["kcc4", "epi_age"],
    "histone_repression_gain":          ["kcc4", "epi_age", "drivers"],
    "chromatin_accessibility_shift":    ["kcc4", "ros", "kcc8"],
}


@dataclass(frozen=True)
class _OmicsCoefficients:
    tx_recipes: dict[str, list[tuple[str, float]]]
    epi_recipes: dict[str, list[tuple[str, float]]]
    tx_noise_sigma: float
    epi_noise_sigma: float
    aging_base_weight: float
    primary_weight_intercept: float
    primary_weight_kcc2: float
    primary_weight_dna: float
    primary_weight_ros: float
    primary_weight_center: float
    primary_weight_scale: float
    primary_weight_min: float
    primary_weight_max: float
    oxidative_kcc5_threshold: float
    oxidative_base_weight: float
    oxidative_blend_weight: float
    mut_total_intercept: float
    mut_total_mutation_rate: float
    mut_total_drivers: float
    mut_total_dna: float
    mut_total_ros: float
    mut_total_min: int


def _build_recipe(prefix: str, modules: dict[str, list[str]]) -> dict[str, list[tuple[str, float]]]:
    r = _registry()
    out: dict[str, list[tuple[str, float]]] = {}
    for module_name, inputs in modules.items():
        entries: list[tuple[str, float]] = []
        for input_name in inputs:
            entries.append(
                (input_name, r.get(f"{prefix}.{module_name}.{input_name}_coeff"))
            )
        out[module_name] = entries
    return out


@functools.cache
def _omics_coefficients() -> _OmicsCoefficients:
    """Materialise the omics-block coefficient bundle from the active registry."""
    r = _registry()
    g = r.get
    return _OmicsCoefficients(
        tx_recipes=_build_recipe("omics.transcript", _TX_INPUTS),
        epi_recipes=_build_recipe("omics.epi", _EPI_INPUTS),
        tx_noise_sigma=g("omics.transcript.noise_sigma"),
        epi_noise_sigma=g("omics.epi.noise_sigma"),
        aging_base_weight=g("omics.signature_mix.aging_base_weight"),
        primary_weight_intercept=g("omics.signature_mix.primary_weight_intercept"),
        primary_weight_kcc2=g("omics.signature_mix.primary_weight_kcc2_coeff"),
        primary_weight_dna=g("omics.signature_mix.primary_weight_dna_coeff"),
        primary_weight_ros=g("omics.signature_mix.primary_weight_ros_coeff"),
        primary_weight_center=g("omics.signature_mix.primary_weight_center"),
        primary_weight_scale=g("omics.signature_mix.primary_weight_scale"),
        primary_weight_min=g("omics.signature_mix.primary_weight_min"),
        primary_weight_max=g("omics.signature_mix.primary_weight_max"),
        oxidative_kcc5_threshold=g("omics.signature_mix.oxidative_kcc5_threshold"),
        oxidative_base_weight=g("omics.signature_mix.oxidative_base_weight"),
        oxidative_blend_weight=g("omics.signature_mix.oxidative_blend_weight"),
        mut_total_intercept=g("omics.mut_total.intercept"),
        mut_total_mutation_rate=g("omics.mut_total.mutation_rate_coeff"),
        mut_total_drivers=g("omics.mut_total.drivers_coeff"),
        mut_total_dna=g("omics.mut_total.dna_coeff"),
        mut_total_ros=g("omics.mut_total.ros_coeff"),
        mut_total_min=int(g("omics.mut_total.min_count")),
    )


register_registry_derived_cache(_omics_coefficients.cache_clear)


def _cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    numerator = float(np.dot(first, second))
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if not np.isfinite(denominator) or denominator <= 0.0:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _bounded_primary_weight(
    score: float,
    *,
    lower: float,
    upper: float,
    center: float,
    scale: float,
) -> float:
    if upper <= lower:
        raise ValueError("primary signature weight max must exceed min")
    if scale <= 0.0:
        raise ValueError("primary signature weight scale must be positive")
    return float(lower + (upper - lower) * expit((score - center) / scale))


def _soft_bounded_score(score: float) -> float:
    """Bound signed external scores without tanh's early high-score collapse."""
    return score / (1.0 + abs(score))


def generate_omics(
    archetype: str,
    kcc: np.ndarray,
    states: dict[str, float],
    susceptibility: dict[str, float],
    rng: np.random.Generator,
    signature_labels: list[str],
    signature_profiles: dict[str, np.ndarray],
    calibration: CalibrationBundle | None = None,
) -> dict[str, float]:
    """Generate synthetic omics from hidden qAOP states."""
    C = _omics_coefficients()
    module_multipliers = _lincs_module_multipliers(calibration, archetype)
    out: dict[str, float] = {}
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10 = kcc

    dna = states["state_auc_DNA_adducts"]
    ros = states["state_auc_ROS"]
    inflammation = states["state_auc_inflammation"]
    epi_age = states["state_final_epigenetic_age"]
    proliferation = states["state_auc_proliferation"]
    mutation_rate = states["state_auc_mutation_rate"]
    clone = states["state_final_clone_fraction"]
    drivers = states["state_final_driver_count_proxy"]

    inputs: dict[str, float] = {
        "dna": dna,
        "ros": ros,
        "inflammation": inflammation,
        "epi_age": epi_age,
        "proliferation": proliferation,
        "clone": clone,
        "drivers": drivers,
        "epi_age_times_proliferation": epi_age * proliferation,
        "kcc1": k1, "kcc2": k2, "kcc3": k3, "kcc4": k4, "kcc5": k5,
        "kcc6": k6, "kcc7": k7, "kcc8": k8, "kcc9": k9, "kcc10": k10,
        "susceptibility_repair": susceptibility["repair_capacity"],
        "susceptibility_immune": susceptibility["immune_surveillance"],
    }

    for module in TRANSCRIPT_MODULES:
        value = 0.0
        for input_name, coef in C.tx_recipes[module]:
            value += coef * inputs[input_name]
        value *= module_multipliers.get(module, 1.0)
        out[f"tx_{module}"] = float(value + rng.normal(0.0, C.tx_noise_sigma))

    for module in EPI_MODULES:
        value = 0.0
        for input_name, coef in C.epi_recipes[module]:
            value += coef * inputs[input_name]
        out[f"epi_{module}"] = float(value + rng.normal(0.0, C.epi_noise_sigma))

    primary_sig = _active_archetype_signature(archetype)
    if primary_sig not in signature_profiles:
        fallback = next(iter(signature_profiles))
        warnings.warn(
            f"archetype {archetype!r} expects signature {primary_sig!r} but it is "
            f"absent from the supplied profiles; falling back to {fallback!r}. "
            "Check that the calibration bundle covers every active archetype.",
            RuntimeWarning,
            stacklevel=2,
        )
        primary_sig = fallback
    sig_mix = C.aging_base_weight * signature_profiles.get(
        "aging", signature_profiles[primary_sig]
    ).copy()
    primary_score = float(
        C.primary_weight_intercept
        + C.primary_weight_kcc2 * k2
        + C.primary_weight_dna * dna
        + C.primary_weight_ros * ros
    )
    primary_weight = _bounded_primary_weight(
        primary_score,
        lower=C.primary_weight_min,
        upper=C.primary_weight_max,
        center=C.primary_weight_center,
        scale=C.primary_weight_scale,
    )
    sig_mix = (1.0 - primary_weight) * sig_mix + primary_weight * signature_profiles[primary_sig]
    if k5 > C.oxidative_kcc5_threshold and "oxidative_like" in signature_profiles:
        sig_mix = (
            C.oxidative_base_weight * sig_mix
            + C.oxidative_blend_weight * signature_profiles["oxidative_like"]
        )
    sig_mix = sig_mix / sig_mix.sum()

    lam = (
        C.mut_total_intercept
        + C.mut_total_mutation_rate * mutation_rate
        + C.mut_total_drivers * drivers
        + C.mut_total_dna * dna
        + C.mut_total_ros * ros
    )
    total_mutations = int(rng.poisson(lam))
    total_mutations = max(total_mutations, C.mut_total_min)
    counts = rng.multinomial(total_mutations, sig_mix)

    for label, count in zip(signature_labels, counts, strict=True):
        safe = label.replace(">", "to").replace("[", "_").replace("]", "_")
        out[f"sig96_{safe}"] = int(count)
    empirical = counts / max(1, counts.sum())
    for sig_name, profile in signature_profiles.items():
        out[f"sig_activity_{sig_name}"] = _cosine_similarity(empirical, profile)
    out["mut_total_count"] = int(total_mutations)
    return out


def _active_archetype_signature(archetype: str) -> str:
    """Return an archetype signature from the active registry when available."""
    card_name = f"archetypes.{archetype}.signature"
    r = _registry()
    if card_name in r:
        return r.get_str(card_name)
    return ARCHETYPE_SIGNATURE.get(archetype, "aging")


# Magnitude of the simulator-time LINCS module-score multiplier; mirrors the
# corresponding policy in ``calibration.coupling`` so registry overlays and
# simulator-time calibration agree by construction.
_LINCS_MULTIPLIER_MAGNITUDE: float = 0.15


def _lincs_module_multipliers(
    calibration: CalibrationBundle | None,
    archetype: str,
) -> dict[str, float]:
    """Map LINCS per-(perturbagen, module) priors to module weight multipliers."""
    if calibration is None or not calibration.transcript_module_priors:
        return {}
    rows = pd.DataFrame(calibration.transcript_module_priors)
    required = {"perturbagen", "module", "mean_score"}
    if not required.issubset(rows.columns):
        return {}
    exact = rows[rows["perturbagen"].astype(str) == str(archetype)]
    if exact.empty:
        exact = rows[rows["perturbagen"].astype(str).str.lower() == str(archetype).lower()]
    if exact.empty:
        return {}
    multipliers: dict[str, float] = {}
    for module, sub in exact.groupby("module"):
        score = float(pd.to_numeric(sub["mean_score"], errors="coerce").mean())
        if np.isfinite(score):
            multipliers[str(module)] = (
                1.0 + _LINCS_MULTIPLIER_MAGNITUDE * _soft_bounded_score(score)
            )
    return multipliers
