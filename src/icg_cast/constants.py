"""Shared scientific constants for the synthetic ICg-CaST simulator.

Numeric coefficients (ARCHETYPE_KCC values, the archetype-sampling prior,
and the per-archetype expected signatures) are read from the coefficient
registry and exposed here for backward compatibility. KCC / state / module
*names* stay as ordered tuples in this module — they are vocabulary, not
coefficients.
"""

from __future__ import annotations

from .coefficients import registry as _registry

KCC_NAMES: tuple[str, ...] = (
    "electrophilic_or_metabolically_activated",
    "genotoxic",
    "DNA_repair_or_genomic_instability",
    "epigenetic_alteration",
    "oxidative_stress",
    "chronic_inflammation",
    "immunosuppression",
    "receptor_mediated",
    "immortalization",
    "proliferation_death_nutrient_supply",
)

# Archetype iteration order is the order of declaration in the registry YAML
# (and historically: inert_control, pah_tobacco_like, aflatoxin_like,
# aristolochic_like, cyp2e1_ros_like, metal_epigenetic_like,
# endocrine_receptor_like, immune_suppression_like).
ARCHETYPE_ORDER: tuple[str, ...] = (
    "inert_control",
    "pah_tobacco_like",
    "aflatoxin_like",
    "aristolochic_like",
    "cyp2e1_ros_like",
    "metal_epigenetic_like",
    "endocrine_receptor_like",
    "immune_suppression_like",
)


def _build_archetype_kcc() -> dict[str, tuple[float, ...]]:
    r = _registry()
    return {name: r.get_vector(f"archetypes.{name}.kcc") for name in ARCHETYPE_ORDER}


def _build_archetype_signature() -> dict[str, str]:
    r = _registry()
    return {name: r.get_str(f"archetypes.{name}.signature") for name in ARCHETYPE_ORDER}


ARCHETYPE_KCC: dict[str, tuple[float, ...]] = _build_archetype_kcc()
ARCHETYPE_SIGNATURE: dict[str, str] = _build_archetype_signature()

TRANSCRIPT_MODULES: tuple[str, ...] = (
    "DNA_adduct_response",
    "p53_checkpoint",
    "base_excision_repair",
    "nucleotide_excision_repair",
    "xenobiotic_metabolism_CYP",
    "oxidative_stress_response",
    "mitochondrial_dysfunction",
    "inflammatory_cytokines",
    "NFkB_activation",
    "cell_cycle_E2F",
    "replicative_DNA_synthesis",
    "apoptosis_escape",
    "ECM_fibrosis",
    "angiogenesis_nutrient_supply",
    "immune_evasion",
    "nuclear_receptor_program",
    "stemness_PRC2",
    "senescence_SASP",
)

EPI_MODULES: tuple[str, ...] = (
    "epigenetic_age_acceleration",
    "PRC2_mitotic_clock",
    "global_hypomethylation",
    "tumor_suppressor_hypermethylation",
    "enhancer_reprogramming",
    "histone_activation_loss",
    "histone_repression_gain",
    "chromatin_accessibility_shift",
)

STATE_NAMES: tuple[str, ...] = (
    "DNA_adducts",
    "ROS",
    "inflammation",
    "epigenetic_age",
    "proliferation",
    "mutation_rate",
    "clone_fraction",
    "driver_count_proxy",
    "immune_clearance",
    "latent_risk",
)
