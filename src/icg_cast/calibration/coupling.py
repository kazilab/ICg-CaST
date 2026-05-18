"""Calibration-driven coefficient updates.

Turns the passive calibration bundle into an optional coefficient overlay.
The default simulator remains synthetic and unchanged, but a caller can save
a calibrated registry and run with it through ``ICG_CAST_COEFFICIENTS_PATH``
or an explicit ``use_registry(...)`` context.
"""

from __future__ import annotations

import math
import re
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from icg_cast.coefficients import CoefficientCard, CoefficientRegistry
from icg_cast.constants import KCC_NAMES

from .bundle import CalibrationBundle

# ---------------------------------------------------------------------------
# Policy constants for the calibration overlay. These are heuristics about how
# strongly each evidence source should re-weight existing coefficient cards.
# They live here (not in the coefficient registry) because they describe how
# the overlay *maps* evidence -> registry edits, not how the qAOP dynamics
# behave. Tune by editing the constants and updating the unit tests.
# ---------------------------------------------------------------------------

# String -> numeric confidence used by ``_confidence`` for evidence rows whose
# weight field is a categorical label.
_CONFIDENCE_FROM_LABEL: dict[str, float] = {
    "high": 0.90, "strong": 0.90, "e1": 0.90, "e2": 0.90,
    "moderate": 0.65, "medium": 0.65, "e3": 0.65,
    "low": 0.35, "weak": 0.35, "e4": 0.35,
}
_DEFAULT_CONFIDENCE: float = 0.60

# LINCS module-prior magnitude: a per-(perturbagen, module) score is softly
# bounded and multiplied by this factor to produce a coefficient scaling.
_LINCS_MODULE_SCALE_MAGNITUDE: float = 0.15

# AOP-Wiki KER strength is converted to a coefficient scale via
# ``intercept + slope * confidence``. With confidence in [0, 1] the resulting
# scale ranges from 0.75 (no-confidence rescaling) to 1.25 (high-confidence).
_AOPWIKI_SCALE_INTERCEPT: float = 0.75
_AOPWIKI_SCALE_SLOPE: float = 0.50

# ToxCast: per-KCC mean activation in [0, 1] is converted to a coefficient
# scaling via ``1 + slope * activation``.
_TOXCAST_KCC_SCALE_SLOPE: float = 0.25

_KCC_RE = re.compile(r"\bkcc\s*([1-9]|10)\b", re.IGNORECASE)


def _soft_bounded_lincs_score(score: float) -> float:
    """Bound signed LINCS scores while preserving high-score separation."""
    return score / (1.0 + abs(score))

_KCC_COUPLING_CARDS: dict[int, tuple[str, ...]] = {
    1: (
        "dynamics.dna_adducts.input_kcc1_slope",
        "dynamics.ros.input_kcc1_slope",
        "omics.transcript.DNA_adduct_response.kcc1_coeff",
        "omics.transcript.xenobiotic_metabolism_CYP.kcc1_coeff",
    ),
    2: (
        "dynamics.dna_adducts.input_kcc2_slope",
        "dynamics.mutation_rate.dna_kcc2_coupling",
        "omics.transcript.DNA_adduct_response.kcc2_coeff",
        "omics.transcript.p53_checkpoint.kcc2_coeff",
    ),
    3: (
        "dynamics.repair.kcc3_inhibition",
        "omics.transcript.base_excision_repair.kcc3_coeff",
        "omics.transcript.nucleotide_excision_repair.kcc3_coeff",
    ),
    4: (
        "dynamics.epigenetic_age.dose_kcc4_coupling",
        "omics.transcript.ECM_fibrosis.kcc4_coeff",
        "omics.transcript.stemness_PRC2.kcc4_coeff",
        "omics.epi.epigenetic_age_acceleration.kcc4_coeff",
        "omics.epi.global_hypomethylation.kcc4_coeff",
        "omics.epi.tumor_suppressor_hypermethylation.kcc4_coeff",
        "omics.epi.enhancer_reprogramming.kcc4_coeff",
        "omics.epi.histone_activation_loss.kcc4_coeff",
        "omics.epi.histone_repression_gain.kcc4_coeff",
        "omics.epi.chromatin_accessibility_shift.kcc4_coeff",
    ),
    5: (
        "dynamics.ros.input_kcc5_slope",
        "dynamics.epigenetic_age.dose_kcc5_coupling",
        "omics.transcript.oxidative_stress_response.kcc5_coeff",
        "omics.signature_mix.oxidative_kcc5_threshold",
    ),
    6: (
        "dynamics.inflammation.dose_kcc6_coupling",
        "dynamics.epigenetic_age.dose_kcc6_coupling",
        "omics.transcript.inflammatory_cytokines.kcc6_coeff",
        "omics.transcript.NFkB_activation.kcc6_coeff",
    ),
    7: (
        "dynamics.immune.kcc7_inhibition",
        "omics.transcript.immune_evasion.kcc7_coeff",
    ),
    8: (
        "dynamics.proliferation.dose_kcc8_coupling",
        "omics.transcript.xenobiotic_metabolism_CYP.kcc8_coeff",
        "omics.transcript.replicative_DNA_synthesis.kcc8_coeff",
        "omics.transcript.nuclear_receptor_program.kcc8_coeff",
        "omics.epi.enhancer_reprogramming.kcc8_coeff",
        "omics.epi.chromatin_accessibility_shift.kcc8_coeff",
    ),
    9: (
        "dynamics.driver_count.kcc9_coupling",
        "dynamics.clone_fraction.selection_kcc9",
        "omics.transcript.apoptosis_escape.kcc9_coeff",
    ),
    10: (
        "dynamics.proliferation.kcc10_coupling",
        "omics.transcript.cell_cycle_E2F.kcc10_coeff",
        "omics.transcript.replicative_DNA_synthesis.kcc10_coeff",
        "omics.transcript.angiogenesis_nutrient_supply.kcc10_coeff",
    ),
}

_AOP_STATE_COUPLINGS: dict[tuple[int, str], tuple[str, ...]] = {
    (1, "dna_adducts"): ("dynamics.dna_adducts.input_kcc1_slope",),
    (1, "ros"): ("dynamics.ros.input_kcc1_slope",),
    (2, "dna_adducts"): ("dynamics.dna_adducts.input_kcc2_slope",),
    (2, "mutation_rate"): ("dynamics.mutation_rate.dna_kcc2_coupling",),
    (3, "repair"): ("dynamics.repair.kcc3_inhibition",),
    (4, "epigenetic_age"): ("dynamics.epigenetic_age.dose_kcc4_coupling",),
    (5, "ros"): ("dynamics.ros.input_kcc5_slope",),
    (5, "epigenetic_age"): ("dynamics.epigenetic_age.dose_kcc5_coupling",),
    (6, "inflammation"): ("dynamics.inflammation.dose_kcc6_coupling",),
    (6, "epigenetic_age"): ("dynamics.epigenetic_age.dose_kcc6_coupling",),
    (8, "proliferation"): ("dynamics.proliferation.dose_kcc8_coupling",),
    (9, "driver_count"): ("dynamics.driver_count.kcc9_coupling",),
    (9, "clone_fraction"): ("dynamics.clone_fraction.selection_kcc9",),
    (10, "proliferation"): ("dynamics.proliferation.kcc10_coupling",),
}

_STATE_ALIASES: dict[str, str] = {
    "dna": "dna_adducts",
    "dnaadducts": "dna_adducts",
    "dna_adducts": "dna_adducts",
    "adduct": "dna_adducts",
    "adducts": "dna_adducts",
    "ros": "ros",
    "oxidativestress": "ros",
    "oxidative_stress": "ros",
    "inflammation": "inflammation",
    "chronicinflammation": "inflammation",
    "chronic_inflammation": "inflammation",
    "epigenetic": "epigenetic_age",
    "epigeneticage": "epigenetic_age",
    "epigenetic_age": "epigenetic_age",
    "proliferation": "proliferation",
    "cellproliferation": "proliferation",
    "mutation": "mutation_rate",
    "mutationrate": "mutation_rate",
    "mutation_rate": "mutation_rate",
    "repair": "repair",
    "dnarepair": "repair",
    "dna_repair": "repair",
    "clone": "clone_fraction",
    "clonefraction": "clone_fraction",
    "clone_fraction": "clone_fraction",
    "driver": "driver_count",
    "drivercount": "driver_count",
    "driver_count": "driver_count",
    "driver_count_proxy": "driver_count",
}


def _safe_name(value: object) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_").lower()
    return text or "unnamed"


def _kcc_index(value: object) -> int | None:
    text = str(value)
    match = _KCC_RE.search(text)
    if match:
        return int(match.group(1))
    lowered = text.strip().lower()
    for idx, name in enumerate(KCC_NAMES, start=1):
        if lowered == name.lower():
            return idx
    return None


def _state_token(value: object) -> str | None:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip()).strip("_").lower()
    if not text:
        return None
    if text in _STATE_ALIASES:
        return _STATE_ALIASES[text]
    compact = text.replace("_", "")
    return _STATE_ALIASES.get(compact)


def _confidence(row: dict[str, Any]) -> float:
    for key in (
        "confidence",
        "weight",
        "weight_of_evidence",
        "evidence_weight",
        "evidence_score",
        "ker_confidence",
    ):
        if key not in row or pd.isna(row[key]):
            continue
        value = row[key]
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            text = str(value).strip().lower()
            if text in _CONFIDENCE_FROM_LABEL:
                return _CONFIDENCE_FROM_LABEL[text]
    return _DEFAULT_CONFIDENCE


def _with_numeric_update(
    registry: CoefficientRegistry,
    name: str,
    *,
    scale: float,
    evidence_level: str,
    source: str,
    note: str,
    review_date: str,
) -> tuple[CoefficientRegistry, dict[str, object]] | tuple[CoefficientRegistry, None]:
    if name not in registry:
        return registry, None
    card = registry.card(name)
    if card.is_string:
        return registry, None
    old = card.default_value
    if isinstance(old, tuple):
        return registry, None
    new_value = float(old) * float(scale)
    updated = replace(
        card,
        default_value=new_value,
        evidence_level=evidence_level,
        source=source,
        notes=f"{card.notes} | {note}".strip(" |"),
        last_reviewed=review_date,
    )
    return registry.replace_cards([updated]), {
        "coefficient": name,
        "old_value": float(old),
        "new_value": new_value,
        "scale": float(scale),
        "evidence_level": evidence_level,
        "source": source,
        "reason": note,
    }


def _module_coefficients(registry: CoefficientRegistry, module: str) -> list[str]:
    prefix = f"omics.transcript.{module}."
    return [
        name
        for name in registry.names()
        if name.startswith(prefix) and name.endswith("_coeff")
    ]


def _apply_lincs_module_priors(
    registry: CoefficientRegistry,
    bundle: CalibrationBundle,
    review_date: str,
) -> tuple[CoefficientRegistry, list[dict[str, object]]]:
    if not bundle.transcript_module_priors:
        return registry, []
    rows = pd.DataFrame(bundle.transcript_module_priors)
    required = {"perturbagen", "module", "mean_score", "n_genes"}
    if not required.issubset(rows.columns):
        return registry, []

    updates: list[dict[str, object]] = []
    cards_to_add: list[CoefficientCard] = []
    for row in rows.to_dict(orient="records"):
        perturbagen = _safe_name(row["perturbagen"])
        module = str(row["module"])
        mean_score = float(row["mean_score"])
        n_genes = int(row["n_genes"])
        card_name = f"calibration.lincs.{perturbagen}.{module}.mean_score"
        cards_to_add.append(
            CoefficientCard(
                name=card_name,
                default_value=mean_score,
                units="LINCS module score",
                evidence_level="E3",
                source="LINCS L1000 local calibration",
                notes=f"per-perturbagen module prior from {n_genes} mapped genes",
                last_reviewed=review_date,
                prior_distribution="normal",
                prior_params={"sd": max(0.05, 1.0 / math.sqrt(max(1, n_genes)))},
            )
        )
        updates.append(
            {
                "coefficient": card_name,
                "old_value": None,
                "new_value": mean_score,
                "scale": None,
                "evidence_level": "E3",
                "source": "LINCS L1000 local calibration",
                "reason": f"per-perturbagen module prior for {row['perturbagen']} / {module}",
            }
        )

    registry = registry.replace_cards(cards_to_add)

    for module, sub in rows.groupby("module"):
        module_score = float(pd.to_numeric(sub["mean_score"], errors="coerce").mean())
        if not np.isfinite(module_score):
            continue
        scale = 1.0 + _LINCS_MODULE_SCALE_MAGNITUDE * _soft_bounded_lincs_score(module_score)
        for name in _module_coefficients(registry, str(module)):
            registry, update = _with_numeric_update(
                registry,
                name,
                scale=scale,
                evidence_level="E3",
                source="LINCS L1000 local calibration",
                note=f"module-level LINCS prior mean_score={module_score:.3f}",
                review_date=review_date,
            )
            if update is not None:
                updates.append(update)
    return registry, updates


def _apply_aopwiki_ker_scaling(
    registry: CoefficientRegistry,
    bundle: CalibrationBundle,
    review_date: str,
) -> tuple[CoefficientRegistry, list[dict[str, object]]]:
    if not bundle.graph_edges:
        return registry, []
    updates: list[dict[str, object]] = []
    for row in bundle.graph_edges:
        source = row.get("source", row.get("from", ""))
        target = row.get("target", row.get("to", ""))
        kcc = _kcc_index(source) or _kcc_index(target)
        state = _state_token(target) or _state_token(source)
        if kcc is None or state is None:
            continue
        strength = _confidence(row)
        scale = _AOPWIKI_SCALE_INTERCEPT + _AOPWIKI_SCALE_SLOPE * strength
        for name in _AOP_STATE_COUPLINGS.get((kcc, state), ()):
            registry, update = _with_numeric_update(
                registry,
                name,
                scale=scale,
                evidence_level="E3",
                source="AOP-Wiki KER local calibration",
                note=f"AOP-Wiki KER {source}->{target} confidence={strength:.2f}",
                review_date=review_date,
            )
            if update is not None:
                updates.append(update)
    return registry, updates


def _apply_toxcast_calibration(
    registry: CoefficientRegistry,
    bundle: CalibrationBundle,
    review_date: str,
) -> tuple[CoefficientRegistry, list[dict[str, object]]]:
    if not bundle.archetype_kcc:
        return registry, []
    updates: list[dict[str, object]] = []
    cards_to_add: list[CoefficientCard] = []

    matrix = []
    for chemical, values in bundle.archetype_kcc.items():
        safe = _safe_name(chemical)
        vec = tuple(float(np.clip(v, 0.0, 1.0)) for v in values)
        matrix.append(vec)
        card_name = f"archetypes.{safe}.kcc"
        old = registry.card(card_name).default_value if card_name in registry else None
        cards_to_add.append(
            CoefficientCard(
                name=card_name,
                default_value=vec,
                units="10-vector of KCC activations in [0, 1]",
                evidence_level="E2",
                source="ToxCast local hit-call calibration",
                notes=f"per-chemical KCC vector calibrated from local ToxCast rows for {chemical}",
                last_reviewed=review_date,
                prior_distribution="logit_normal",
                prior_params={},
            )
        )
        cards_to_add.append(
            CoefficientCard(
                name=f"archetypes.{safe}.signature",
                default_value="aging",
                units="signature label",
                evidence_level="E2",
                source="ToxCast local hit-call calibration",
                notes=f"default signature for ToxCast-calibrated chemical {chemical}",
                last_reviewed=review_date,
                prior_distribution="fixed",
            )
        )
        updates.append(
            {
                "coefficient": card_name,
                "old_value": list(old) if isinstance(old, tuple) else old,
                "new_value": list(vec),
                "scale": None,
                "evidence_level": "E2",
                "source": "ToxCast local hit-call calibration",
                "reason": "per-chemical KCC values",
            }
        )

    registry = registry.replace_cards(cards_to_add)

    mean_kcc = np.asarray(matrix, dtype=float).mean(axis=0)
    for idx, activation in enumerate(mean_kcc, start=1):
        if activation <= 0.0:
            continue
        scale = 1.0 + _TOXCAST_KCC_SCALE_SLOPE * float(activation)
        for name in _KCC_COUPLING_CARDS.get(idx, ()):
            registry, update = _with_numeric_update(
                registry,
                name,
                scale=scale,
                evidence_level="E2",
                source="ToxCast local hit-call calibration",
                note=f"ToxCast KCC{idx} mean activation={activation:.3f}",
                review_date=review_date,
            )
            if update is not None:
                updates.append(update)
    return registry, updates


def count_evidence_upgrade(old_registry: CoefficientRegistry, new_registry: CoefficientRegistry) -> int:
    """Count coefficients that moved from E4/E5 into E1/E2/E3."""
    upgraded = 0
    for card in new_registry:
        old_level = old_registry.card(card.name).evidence_level if card.name in old_registry else "E5"
        if card.evidence_level in {"E1", "E2", "E3"} and old_level not in {"E1", "E2", "E3"}:
            upgraded += 1
    return upgraded


def calibrated_registry_from_bundle(
    base: CoefficientRegistry,
    bundle: CalibrationBundle,
    *,
    review_date: str | None = None,
) -> tuple[CoefficientRegistry, dict[str, object]]:
    """Return ``base`` plus coefficient updates implied by a calibration bundle."""
    stamp = review_date or date.today().isoformat()
    registry = base
    updates: list[dict[str, object]] = []

    for applier in (
        _apply_lincs_module_priors,
        _apply_aopwiki_ker_scaling,
        _apply_toxcast_calibration,
    ):
        registry, chunk = applier(registry, bundle, stamp)
        updates.extend(chunk)

    before = sum(1 for c in base if c.evidence_level in {"E1", "E2", "E3"})
    after = sum(1 for c in registry if c.evidence_level in {"E1", "E2", "E3"})
    return registry, {
        "review_date": stamp,
        "n_updates": len(updates),
        "e1_e3_before": before,
        "e1_e3_after": after,
        "evidence_upgrade_count": count_evidence_upgrade(base, registry),
        "updates": updates,
    }
