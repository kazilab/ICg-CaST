"""Typed loader for the ICg-CaST coefficient registry.

The registry is a flat list of "cards", each describing one coefficient
(or one vector / one categorical label) with provenance metadata. Cards
are loaded from a YAML file; see ``materials/coefficient_cards.yaml``.

Card schema::

    name:            dotted namespace, e.g. ``dynamics.dna_adducts.decay``
    default_value:   scalar (float/int), vector (list of numbers), or string
    effect_direction:
                    optional net direction of effect on downstream risk
                    (-1 protective, +1 harmful, 0 neutral/unknown)
    units:           free-text units description
    evidence_level:  one of E1..E5 (see below)
    source:          DOI, dataset name, or ``"starter kit"``
    notes:           free-text explanation
    last_reviewed:   ISO date string
    prior_distribution:
                    one of auto/fixed/normal/lognormal/signed_lognormal/
                    logit_normal/dirichlet
    prior_params:   optional sampler parameters; evidence level supplies
                    defaults when omitted

Evidence levels::

    E1   published quantitative literature value
    E2   published qualitative direction or magnitude
    E3   AOP-Wiki / AOP-DB / KER weight-of-evidence
    E4   expert estimate, plausible biological order of magnitude
    E5   no source ("hand-tuned to produce interesting cohorts")

Default evidence is E5; explicit annotations override.
"""

from __future__ import annotations

import contextvars
import functools
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

EVIDENCE_LEVELS: tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5")

_SchemaVersion = "0.2"

_ScalarValue = float | int | str | tuple
PRIOR_DISTRIBUTIONS: tuple[str, ...] = (
    "auto",
    "fixed",
    "normal",
    "lognormal",
    "signed_lognormal",
    "logit_normal",
    "dirichlet",
)


@dataclass(frozen=True)
class CoefficientCard:
    """One numeric coefficient (or vector / categorical label) plus provenance."""

    name: str
    default_value: _ScalarValue
    effect_direction: int | None = None
    units: str = ""
    evidence_level: str = "E5"
    source: str = ""
    notes: str = ""
    last_reviewed: str = ""
    prior_distribution: str = "auto"
    prior_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_level not in EVIDENCE_LEVELS:
            raise ValueError(
                f"card {self.name!r} has invalid evidence_level "
                f"{self.evidence_level!r}; expected one of {EVIDENCE_LEVELS}"
            )
        if self.prior_distribution not in PRIOR_DISTRIBUTIONS:
            raise ValueError(
                f"card {self.name!r} has invalid prior_distribution "
                f"{self.prior_distribution!r}; expected one of {PRIOR_DISTRIBUTIONS}"
            )
        if not isinstance(self.prior_params, Mapping):
            raise ValueError(f"card {self.name!r} prior_params must be a mapping")
        object.__setattr__(self, "prior_params", dict(self.prior_params))
        if self.effect_direction is not None:
            direction = int(self.effect_direction)
            if direction not in (-1, 0, 1):
                raise ValueError(
                    f"card {self.name!r} has invalid effect_direction "
                    f"{self.effect_direction!r}; expected -1, 0, 1, or null"
                )
            object.__setattr__(self, "effect_direction", direction)
        # normalise list/tuple vectors to immutable tuples of floats so the
        # cached registry returns the same object identity each call.
        v = self.default_value
        if isinstance(v, list):
            object.__setattr__(self, "default_value", tuple(float(x) for x in v))
        elif isinstance(v, tuple):
            object.__setattr__(self, "default_value", tuple(float(x) for x in v))

    @property
    def is_vector(self) -> bool:
        return isinstance(self.default_value, tuple)

    @property
    def is_string(self) -> bool:
        return isinstance(self.default_value, str)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "name": self.name,
            "default_value": (
                list(self.default_value) if self.is_vector else self.default_value
            ),
            "units": self.units,
            "evidence_level": self.evidence_level,
            "source": self.source,
            "notes": self.notes,
            "last_reviewed": self.last_reviewed,
            "prior_distribution": self.prior_distribution,
            "prior_params": dict(self.prior_params),
        }
        if self.effect_direction is not None:
            out["effect_direction"] = self.effect_direction
        return out


class CoefficientRegistry:
    """Indexed collection of :class:`CoefficientCard` records."""

    def __init__(self, cards: Sequence[CoefficientCard], schema_version: str = _SchemaVersion) -> None:
        seen: dict[str, CoefficientCard] = {}
        for card in cards:
            if card.name in seen:
                raise ValueError(f"duplicate coefficient name: {card.name!r}")
            seen[card.name] = card
        self._cards: dict[str, CoefficientCard] = seen
        self.schema_version = schema_version

    def __contains__(self, name: str) -> bool:
        return name in self._cards

    def __len__(self) -> int:
        return len(self._cards)

    def __iter__(self):
        return iter(self._cards.values())

    def names(self) -> list[str]:
        return list(self._cards.keys())

    def to_dict(self) -> dict[str, Any]:
        """Serialise the registry to the on-disk YAML schema."""
        return {
            "schema_version": self.schema_version,
            "cards": [card.to_dict() for card in self],
        }

    def card(self, name: str) -> CoefficientCard:
        try:
            return self._cards[name]
        except KeyError as exc:
            raise KeyError(f"unknown coefficient: {name!r}") from exc

    def get(self, name: str) -> float:
        card = self.card(name)
        if card.is_vector:
            raise TypeError(f"{name!r} is a vector; use get_vector()")
        if card.is_string:
            raise TypeError(f"{name!r} is a string; use get_str()")
        return float(card.default_value)

    def get_vector(self, name: str) -> tuple[float, ...]:
        card = self.card(name)
        if not card.is_vector:
            raise TypeError(f"{name!r} is not a vector")
        return card.default_value  # type: ignore[return-value]

    def get_str(self, name: str) -> str:
        card = self.card(name)
        if not card.is_string:
            raise TypeError(f"{name!r} is not a string")
        return str(card.default_value)

    def filter(
        self,
        *,
        evidence_level: str | None = None,
        prefix: str | None = None,
    ) -> list[CoefficientCard]:
        results = list(self._cards.values())
        if evidence_level is not None:
            if evidence_level not in EVIDENCE_LEVELS:
                raise ValueError(
                    f"evidence_level must be one of {EVIDENCE_LEVELS}; got {evidence_level!r}"
                )
            results = [c for c in results if c.evidence_level == evidence_level]
        if prefix is not None:
            results = [c for c in results if c.name.startswith(prefix)]
        return results

    def replace_card(self, name: str, **changes: Any) -> CoefficientRegistry:
        """Return a new registry with one card replaced."""
        card = self.card(name)
        updated = replace(card, **changes)
        return self.replace_cards([updated])

    def replace_cards(self, cards: Sequence[CoefficientCard]) -> CoefficientRegistry:
        """Return a new registry with the supplied cards added or replaced."""
        updated = dict(self._cards)
        for card in cards:
            updated[card.name] = card
        return CoefficientRegistry(list(updated.values()), schema_version=self.schema_version)


def default_registry_path() -> Path:
    """Locate the canonical coefficient YAML.

    Search order:
        1. ``$ICG_CAST_COEFFICIENTS_PATH`` environment variable.
        2. ``materials/coefficient_cards.yaml`` walking up from this file
           toward the repo root.
        3. ``src/icg_cast/coefficients/cards.yaml`` (installed bundle).
    """
    env = os.environ.get("ICG_CAST_COEFFICIENTS_PATH")
    if env:
        path = Path(env)
        if not path.exists():
            raise FileNotFoundError(
                f"ICG_CAST_COEFFICIENTS_PATH={env!r} but no such file"
            )
        return path

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "materials" / "coefficient_cards.yaml"
        if candidate.exists():
            return candidate

    bundled = here.parent / "cards.yaml"
    if bundled.exists():
        return bundled

    raise FileNotFoundError(
        "coefficient_cards.yaml not found. Set ICG_CAST_COEFFICIENTS_PATH or place "
        "the file under materials/coefficient_cards.yaml relative to the repo root."
    )


def load_registry(path: str | Path | None = None) -> CoefficientRegistry:
    """Load and validate a registry from ``path`` (default: registry path)."""
    p = Path(path) if path is not None else default_registry_path()
    with p.open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    if not isinstance(doc, dict):
        raise ValueError(f"expected a YAML mapping at top level; got {type(doc).__name__}")

    schema_version = str(doc.get("schema_version", _SchemaVersion))
    defaults: dict[str, Any] = dict(doc.get("defaults") or {})
    raw_cards = doc.get("cards") or []
    if not isinstance(raw_cards, list):
        raise ValueError("`cards` must be a list of card mappings")

    cards: list[CoefficientCard] = []
    for entry in raw_cards:
        if not isinstance(entry, dict):
            raise ValueError(f"each card must be a mapping; got {type(entry).__name__}")
        merged = {**defaults, **entry}
        if "name" not in merged or "default_value" not in merged:
            raise ValueError(f"card missing required field(s): {merged!r}")
        cards.append(
            CoefficientCard(
                name=str(merged["name"]),
                default_value=merged["default_value"],
                effect_direction=(
                    None
                    if merged.get("effect_direction") is None
                    else int(merged["effect_direction"])
                ),
                units=str(merged.get("units", "")),
                evidence_level=str(merged.get("evidence_level", "E5")),
                source=str(merged.get("source", "")),
                notes=str(merged.get("notes", "")),
                last_reviewed=str(merged.get("last_reviewed", "")),
                prior_distribution=str(merged.get("prior_distribution", "auto")),
                prior_params=dict(merged.get("prior_params") or {}),
            )
        )
    return CoefficientRegistry(cards, schema_version=schema_version)


def save_registry(registry_obj: CoefficientRegistry, path: str | Path) -> Path:
    """Write a coefficient registry YAML file and return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(registry_obj.to_dict(), sort_keys=False),
        encoding="utf-8",
    )
    return p


_active_registry: contextvars.ContextVar[CoefficientRegistry | None] = contextvars.ContextVar(
    "icg_cast_active_coefficient_registry",
    default=None,
)


# Caches that derive values from the active registry register themselves here at
# import time. ``clear_registry_derived_caches()`` clears all of them at once so
# the simulator does not have to enumerate them by name.
_REGISTRY_DERIVED_CACHES: list[Callable[[], None]] = []


def register_registry_derived_cache(clear: Callable[[], None]) -> None:
    """Register a ``cache_clear``-style callable to be invoked on registry swaps."""
    _REGISTRY_DERIVED_CACHES.append(clear)


def clear_registry_derived_caches() -> None:
    """Clear every cache that was derived from a coefficient registry."""
    for clear in _REGISTRY_DERIVED_CACHES:
        clear()


@contextmanager
def use_registry(active: CoefficientRegistry) -> Iterator[CoefficientRegistry]:
    """Temporarily make ``registry()`` return ``active`` in this context."""
    token = _active_registry.set(active)
    try:
        yield active
    finally:
        _active_registry.reset(token)


@functools.lru_cache(maxsize=1)
def _cached_default_registry() -> CoefficientRegistry:
    return load_registry()


def registry() -> CoefficientRegistry:
    """Return the active registry, or the cached default registry."""
    active = _active_registry.get()
    if active is not None:
        return active
    return _cached_default_registry()


def reset_cache() -> None:
    """Clear the cached default registry (for tests / env-var overrides)."""
    _cached_default_registry.cache_clear()
