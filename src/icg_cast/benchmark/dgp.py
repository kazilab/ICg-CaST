"""ICg-Bench v0.1 data-generating-process variants.

Each variant is a named configuration of the simulator that varies one or
more axes of difficulty:

- coupling: linear vs. non-linear KCC -> qAOP state transitions
- archetype mode: discrete archetypes vs. continuous KCC mixtures
- host heterogeneity: low vs. high variance in susceptibility
- observability: full multi-omics vs. randomly masked modalities

This module exposes the schema and the variant registry so that downstream
tests, scoring code, and the leaderboard schema can be written and exercised
against fixtures.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

CouplingMode = Literal["linear", "nonlinear"]
ArchetypeMode = Literal["discrete", "continuous_mixture"]
HostHeterogeneity = Literal["low", "high"]
Observability = Literal["full", "partial"]


@dataclass(frozen=True)
class DGPVariant:
    """A versioned DGP configuration used by ICg-Bench."""

    name: str
    description: str
    coupling: CouplingMode
    archetype_mode: ArchetypeMode
    host_heterogeneity: HostHeterogeneity
    observability: Observability
    n: int = 400
    months: int = 36
    seed: int = 7
    masking_rate: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:12]


_REGISTRY: dict[str, DGPVariant] = {
    "linear_lowhet": DGPVariant(
        name="linear_lowhet",
        description=(
            "Linear KCC->state coupling, low host heterogeneity, discrete archetypes, "
            "full multi-omics observability. The easy baseline."
        ),
        coupling="linear",
        archetype_mode="discrete",
        host_heterogeneity="low",
        observability="full",
    ),
    "nonlinear_mixhost": DGPVariant(
        name="nonlinear_mixhost",
        description=(
            "Non-linear KCC->state coupling, continuous KCC mixtures, high host "
            "heterogeneity, full multi-omics observability. Stresses latent recovery."
        ),
        coupling="nonlinear",
        archetype_mode="continuous_mixture",
        host_heterogeneity="high",
        observability="full",
    ),
    "partial_observability": DGPVariant(
        name="partial_observability",
        description=(
            "Non-linear coupling with random per-subject masking of transcriptomic "
            "and epigenomic modules. Stresses model robustness to missing modalities."
        ),
        coupling="nonlinear",
        archetype_mode="continuous_mixture",
        host_heterogeneity="high",
        observability="partial",
        masking_rate=0.30,
    ),
    "nonlinear_obs": DGPVariant(
        name="nonlinear_obs",
        description=(
            "Linear KCC->state coupling with a non-linear, multiplicatively "
            "interacting observation operator. Stresses stage-1 latent recovery."
        ),
        coupling="linear",
        archetype_mode="discrete",
        host_heterogeneity="low",
        observability="full",
    ),
    "misspecified_signs": DGPVariant(
        name="misspecified_signs",
        description=(
            "linear_lowhet base, but the latent_risk DGP flips the sign on "
            "`state_final_epigenetic_age` relative to the structural prior. "
            "Falsification test for sign-constrained MB-CNet."
        ),
        coupling="linear",
        archetype_mode="discrete",
        host_heterogeneity="low",
        observability="full",
    ),
    "misspecified_signs_v2": DGPVariant(
        name="misspecified_signs_v2",
        description=(
            "linear_lowhet base with *two* latent_risk flips vs the prior: "
            "`state_final_epigenetic_age` and `state_final_immune_clearance`. "
            "Stress test for whether v0.1 unconstrained recovery survives "
            "multiple simultaneous prior errors."
        ),
        coupling="linear",
        archetype_mode="discrete",
        host_heterogeneity="low",
        observability="full",
    ),
}


def list_variants() -> list[str]:
    """Return registered variant names in stable order."""
    return list(_REGISTRY.keys())


def load_variant(name: str) -> DGPVariant:
    """Return the variant by name, raising KeyError for unknown names."""
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown DGP variant: {name!r}. registered: {sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]


def register_variant(variant: DGPVariant) -> None:
    """Register a new variant. Useful for adding research variants in user code."""
    if variant.name in _REGISTRY:
        raise ValueError(f"variant already registered: {variant.name!r}")
    _REGISTRY[variant.name] = variant
