"""Coefficient registry: the single source of truth for numeric coefficients.

PLAN.md reference: section 25 (Coefficient credibility roadmap),
Milestones 8 and 9.

Every numeric coefficient that drives the qAOP dynamics, the
``latent_risk`` equation, the chemical archetype tables, and the host
susceptibility distributions is declared in
``materials/coefficient_cards.yaml`` and loaded through this module.

The omics observation model (`omics.py`) and the toy mutational signature
recipe (`signatures.py`) are also covered by the registry. Milestone 9 adds
seedable prior sampling for coefficient uncertainty.

Public API::

    from icg_cast.coefficients import registry

    decay = registry.get("dynamics.dna_adducts.decay")
    kcc   = registry.get_vector("archetypes.pah_tobacco_like.kcc")
    sig   = registry.get_str("archetypes.pah_tobacco_like.signature")

    # Filter for audit purposes:
    unsourced = registry.filter(evidence_level="E5")

    # Draw a seedable uncertainty realization:
    from icg_cast.coefficients import sampled_registry
    sampled = sampled_registry(seed=42)
"""

from __future__ import annotations

from .priors import (
    get_prior_params,
    prior_sigma_for_evidence,
    sample_card_prior,
    sample_coefficient,
    sampled_registry,
)
from .registry import (
    EVIDENCE_LEVELS,
    PRIOR_DISTRIBUTIONS,
    CoefficientCard,
    CoefficientRegistry,
    default_registry_path,
    load_registry,
    registry,
    save_registry,
    use_registry,
)

__all__ = [
    "EVIDENCE_LEVELS",
    "PRIOR_DISTRIBUTIONS",
    "CoefficientCard",
    "CoefficientRegistry",
    "default_registry_path",
    "load_registry",
    "get_prior_params",
    "prior_sigma_for_evidence",
    "registry",
    "sample_card_prior",
    "sample_coefficient",
    "sampled_registry",
    "save_registry",
    "use_registry",
]
