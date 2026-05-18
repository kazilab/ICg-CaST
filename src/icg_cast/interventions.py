"""Shared do-interventions and their expected risk directions.

These specs are consumed by the CLI bench runner, the Streamlit app, and the
intervention-augmented MB-CNet stage-2 fit. Keeping them in one module avoids
the cross-import of CLI privates that the previous layout required. Baseline
feature-space counterfactuals use explicit column allowlists from this module
so future feature names cannot be perturbed by incidental substring matches.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

INTERVENTIONS: dict[str, dict[str, float]] = {
    "do_DNA_repair_rescue":             {"state_final_DNA_adducts":        0.55},
    "do_ROS_inflammation_blockade":     {"state_final_ROS":                0.50,
                                         "state_final_inflammation":       0.50},
    "do_epigenetic_memory_reset":       {"state_final_epigenetic_age":     0.45},
    "do_proliferation_suppression":     {"state_final_proliferation":      0.50},
    "do_immune_surveillance_restore":   {"state_final_immune_clearance":   1.50},
    "do_repair_inhibition":             {"state_final_DNA_adducts":        1.80},
    "do_artificial_proliferation":      {"state_final_proliferation":      1.80},
}


@dataclass(frozen=True)
class FeatureIntervention:
    """Exact feature-column perturbation used by baseline counterfactual tests."""

    column_scales: Mapping[str, float]
    expected_direction: int
    severity_weight: float | None = None

    def resolved_severity_weight(self) -> float:
        """Return an intervention-severity weight independent of model response."""
        if self.severity_weight is not None:
            weight = float(self.severity_weight)
            if not np.isfinite(weight) or weight < 0.0:
                raise ValueError("severity_weight must be finite and non-negative")
            return weight
        return intervention_severity_weight(self.column_scales)


def intervention_severity_weight(column_scales: Mapping[str, float]) -> float:
    """Compute severity as mean absolute log fold-change across perturbed columns."""
    magnitudes: list[float] = []
    for scale in column_scales.values():
        value = float(scale)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("intervention scale factors must be positive and finite")
        if not np.isclose(value, 1.0):
            magnitudes.append(abs(float(np.log(value))))
    return float(np.mean(magnitudes)) if magnitudes else 0.0


COUNTERFACTUAL_FEATURE_INTERVENTIONS: dict[str, FeatureIntervention] = {
    "do_DNA_repair_rescue": FeatureIntervention(
        column_scales={
            "state_final_DNA_adducts": 0.55,
            "state_auc_DNA_adducts": 0.55,
            "tx_DNA_adduct_response": 0.55,
            "sig_activity_SBS4_like": 0.55,
            "sig_activity_SBS24_like": 0.55,
            "sig_activity_SBS22_like": 0.55,
            "mut_total_count": 0.55,
        },
        expected_direction=-1,
    ),
    "do_ROS_inflammation_blockade": FeatureIntervention(
        column_scales={
            "state_final_ROS": 0.50,
            "state_auc_ROS": 0.50,
            "state_final_inflammation": 0.50,
            "state_auc_inflammation": 0.50,
            "tx_oxidative_stress_response": 0.50,
            "tx_inflammatory_cytokines": 0.50,
            "tx_NFkB_activation": 0.50,
            "tx_senescence_SASP": 0.50,
            "sig_activity_oxidative_like": 0.50,
        },
        expected_direction=-1,
    ),
    "do_epigenetic_memory_reset": FeatureIntervention(
        column_scales={
            "state_final_epigenetic_age": 0.45,
            "state_auc_epigenetic_age": 0.45,
            "tx_stemness_PRC2": 0.45,
            "epi_epigenetic_age_acceleration": 0.45,
            "epi_PRC2_mitotic_clock": 0.45,
            "epi_global_hypomethylation": 0.45,
            "epi_tumor_suppressor_hypermethylation": 0.45,
            "epi_enhancer_reprogramming": 0.45,
            "epi_histone_activation_loss": 0.45,
            "epi_histone_repression_gain": 0.45,
            "epi_chromatin_accessibility_shift": 0.45,
        },
        expected_direction=-1,
    ),
    "do_proliferation_suppression": FeatureIntervention(
        column_scales={
            "state_final_proliferation": 0.50,
            "state_auc_proliferation": 0.50,
            "state_final_clone_fraction": 0.50,
            "state_auc_clone_fraction": 0.50,
            "tx_cell_cycle_E2F": 0.50,
            "tx_replicative_DNA_synthesis": 0.50,
            "tx_angiogenesis_nutrient_supply": 0.50,
        },
        expected_direction=-1,
    ),
    "do_lower_immune_evasion": FeatureIntervention(
        column_scales={"tx_immune_evasion": 0.35},
        expected_direction=-1,
    ),
    "do_raise_immune_clearance": FeatureIntervention(
        column_scales={
            "state_final_immune_clearance": 1.25,
            "state_auc_immune_clearance": 1.25,
            "host_immune_surveillance": 1.25,
        },
        expected_direction=-1,
    ),
    "do_immune_surveillance_restore": FeatureIntervention(
        column_scales={
            "tx_immune_evasion": 0.35,
            "state_final_immune_clearance": 1.25,
            "state_auc_immune_clearance": 1.25,
            "host_immune_surveillance": 1.25,
        },
        expected_direction=-1,
    ),
}


def _structural_signs() -> dict[str, int]:
    from .bottleneck import structural_signs_from_registry

    return structural_signs_from_registry()


def _direction_for_intervention(
    spec: Mapping[str, float],
    structural_signs: Mapping[str, int],
) -> int:
    directions: set[int] = set()
    for unit, scale in spec.items():
        sign = int(structural_signs.get(unit, 0))
        change = int(np.sign(float(scale) - 1.0))
        direction = sign * change
        if direction != 0:
            directions.add(direction)
    return directions.pop() if len(directions) == 1 else 0


def expected_directions_prior() -> dict[str, int]:
    """Derive prior intervention directions from active registry sign metadata."""
    signs = _structural_signs()
    return {
        name: _direction_for_intervention(spec, signs)
        for name, spec in INTERVENTIONS.items()
    }


class _ExpectedDirectionsPrior(Mapping[str, int]):
    def _data(self) -> dict[str, int]:
        return expected_directions_prior()

    def __getitem__(self, key: str) -> int:
        return self._data()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data())

    def __len__(self) -> int:
        return len(self._data())

    def __repr__(self) -> str:
        return repr(self._data())

    def copy(self) -> dict[str, int]:
        return self._data()


EXPECTED_DIRECTIONS_PRIOR: Mapping[str, int] = _ExpectedDirectionsPrior()


EXPECTED_DIRECTIONS_DGP_OVERRIDES: dict[str, dict[str, int]] = {
    "misspecified_signs": {
        "do_epigenetic_memory_reset": +1,
    },
    "misspecified_signs_v2": {
        "do_epigenetic_memory_reset": +1,
        "do_immune_surveillance_restore": +1,
    },
}


def dgp_directions(cohort: str) -> dict[str, int]:
    """Return the prior intervention directions with cohort-specific overrides applied."""
    base = dict(EXPECTED_DIRECTIONS_PRIOR)
    base.update(EXPECTED_DIRECTIONS_DGP_OVERRIDES.get(cohort, {}))
    return base


def risk_function_directions(
    states: pd.DataFrame,
    interventions: Mapping[str, Mapping[str, float]],
    risk_fn,
    *,
    tolerance: float = 1e-6,
) -> dict[str, int]:
    """Infer intervention directions from a risk equation evaluated on true states."""
    base = np.asarray(risk_fn(states), dtype=float)
    directions: dict[str, int] = {}
    for name, spec in interventions.items():
        after = states.copy()
        for unit, scale in spec.items():
            if unit in after.columns:
                after[unit] = after[unit].astype(float) * float(scale)
        delta = float(np.mean(np.asarray(risk_fn(after), dtype=float) - base))
        directions[name] = 0 if abs(delta) < tolerance else int(np.sign(delta))
    return directions
