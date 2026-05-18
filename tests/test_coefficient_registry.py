"""Tests for the coefficient registry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from icg_cast.coefficients import (
    EVIDENCE_LEVELS,
    PRIOR_DISTRIBUTIONS,
    CoefficientCard,
    load_registry,
    prior_sigma_for_evidence,
    registry,
    sampled_registry,
    save_registry,
)
from icg_cast.constants import ARCHETYPE_KCC, ARCHETYPE_ORDER, ARCHETYPE_SIGNATURE


def test_registry_loads_default_yaml() -> None:
    r = registry()
    assert len(r) > 100
    assert r.schema_version == "0.2"


def test_registry_cards_include_prior_metadata() -> None:
    r = registry()
    for card in r:
        assert card.prior_distribution in PRIOR_DISTRIBUTIONS
        assert isinstance(card.prior_params, dict)
        assert card.effect_direction in (None, -1, 0, 1)


def test_registry_exposes_load_bearing_effect_directions() -> None:
    r = registry()
    assert r.card("dynamics.latent_risk.dna_coupling").effect_direction == 1
    assert r.card("dynamics.latent_risk.immune_coupling").effect_direction == -1
    assert r.card("dynamics.mutation_rate.scale").effect_direction == 1


def test_registry_returns_correct_types() -> None:
    r = registry()
    # scalar
    assert r.get("dynamics.dna_adducts.decay") == pytest.approx(0.68)
    # vector
    kcc = r.get_vector("archetypes.pah_tobacco_like.kcc")
    assert isinstance(kcc, tuple) and len(kcc) == 10
    # string
    assert r.get_str("archetypes.pah_tobacco_like.signature") == "SBS4_like"


def test_registry_get_rejects_wrong_kind() -> None:
    r = registry()
    with pytest.raises(TypeError):
        r.get("archetypes.pah_tobacco_like.kcc")  # is a vector
    with pytest.raises(TypeError):
        r.get_vector("dynamics.dna_adducts.decay")  # is a scalar
    with pytest.raises(TypeError):
        r.get_str("dynamics.dna_adducts.decay")  # is a scalar
    with pytest.raises(KeyError):
        r.get("does.not.exist")


def test_registry_filter_by_evidence_and_prefix() -> None:
    r = registry()
    e5 = r.filter(evidence_level="E5")
    e4 = r.filter(evidence_level="E4")
    assert len(e5) > 0
    assert len(e4) > 0
    assert len(e5) + len(e4) <= len(r)
    dyn = r.filter(prefix="dynamics.")
    assert all(c.name.startswith("dynamics.") for c in dyn)
    # invalid evidence level
    with pytest.raises(ValueError):
        r.filter(evidence_level="X9")


def test_evidence_level_validation_in_cards() -> None:
    with pytest.raises(ValueError, match="invalid evidence_level"):
        CoefficientCard(name="x", default_value=1.0, evidence_level="X1")
    for level in EVIDENCE_LEVELS:
        CoefficientCard(name=f"x.{level}", default_value=1.0, evidence_level=level)


def test_prior_spread_tracks_evidence_level() -> None:
    assert prior_sigma_for_evidence("E1") < prior_sigma_for_evidence("E5")
    with pytest.raises(ValueError):
        prior_sigma_for_evidence("X9")


def test_sampled_registry_is_seedable_and_bounded() -> None:
    base = registry()
    first = sampled_registry(base, seed=123)
    second = sampled_registry(base, seed=123)
    third = sampled_registry(base, seed=124)

    assert first.get("dynamics.dna_adducts.decay") == pytest.approx(
        second.get("dynamics.dna_adducts.decay")
    )
    assert first.get_vector("archetypes.pah_tobacco_like.kcc") == pytest.approx(
        second.get_vector("archetypes.pah_tobacco_like.kcc")
    )
    assert first.get("dynamics.dna_adducts.decay") != pytest.approx(
        third.get("dynamics.dna_adducts.decay")
    )
    assert first.get_str("archetypes.pah_tobacco_like.signature") == base.get_str(
        "archetypes.pah_tobacco_like.signature"
    )
    assert first.get("signatures.background.seed") == base.get("signatures.background.seed")

    prior = np.asarray(first.get_vector("archetypes.sample_prior"), dtype=float)
    assert prior.sum() == pytest.approx(1.0)
    assert np.all(prior > 0.0)

    kcc = np.asarray(first.get_vector("archetypes.pah_tobacco_like.kcc"), dtype=float)
    assert np.all((0.0 <= kcc) & (kcc <= 1.0))


def test_constants_archetype_tables_are_built_from_registry() -> None:
    """ARCHETYPE_KCC / ARCHETYPE_SIGNATURE must mirror the registry exactly."""
    r = registry()
    for name in ARCHETYPE_ORDER:
        assert ARCHETYPE_KCC[name] == r.get_vector(f"archetypes.{name}.kcc")
        assert ARCHETYPE_SIGNATURE[name] == r.get_str(f"archetypes.{name}.signature")


def test_load_registry_accepts_override_yaml(tmp_path: Path) -> None:
    yaml_text = """
schema_version: "0.2"
defaults:
  evidence_level: "E5"
  source: "test fixture"
cards:
  - name: test.alpha
    default_value: 0.42
  - name: test.vec
    default_value: [1.0, 2.0, 3.0]
  - name: test.label
    default_value: "hello"
"""
    p = tmp_path / "custom.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    r = load_registry(p)
    assert r.get("test.alpha") == pytest.approx(0.42)
    assert r.get_vector("test.vec") == (1.0, 2.0, 3.0)
    assert r.get_str("test.label") == "hello"


def test_load_registry_rejects_duplicate_names(tmp_path: Path) -> None:
    yaml_text = """
schema_version: "0.2"
cards:
  - name: dup
    default_value: 1.0
  - name: dup
    default_value: 2.0
"""
    p = tmp_path / "dup.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate coefficient"):
        load_registry(p)


def test_save_registry_roundtrip(tmp_path: Path) -> None:
    r = registry()
    path = save_registry(r.replace_card("dynamics.dna_adducts.decay", default_value=0.77), tmp_path / "cards.yaml")
    loaded = load_registry(path)
    assert loaded.get("dynamics.dna_adducts.decay") == pytest.approx(0.77)
    assert loaded.schema_version == r.schema_version


def test_load_registry_rejects_missing_required_fields(tmp_path: Path) -> None:
    yaml_text = """
schema_version: "0.2"
cards:
  - name: missing_value
"""
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="missing required field"):
        load_registry(p)


def test_registry_is_singleton() -> None:
    a = registry()
    b = registry()
    assert a is b


def test_no_inline_numeric_literals_in_covered_sites() -> None:
    """Spot-check that every covered file reads its coefficients from the registry."""
    import inspect

    from icg_cast import omics, signatures, simulator

    for module, sentinels in (
        (simulator, ("0.68 *", "0.36 *", "-7.5", "4.50 * clone_fraction")),
        (omics, ("0.8 * dna + 1.2 * k1", "0.45 * signature_profiles", "0.40 * k2")),
        (signatures, ("default_rng(123)", "shape=1.0, scale=0.2", "+= 2.0")),
    ):
        src = inspect.getsource(module)
        assert "from .coefficients import registry" in src, (
            f"{module.__name__} no longer imports the registry"
        )
        for sentinel in sentinels:
            assert sentinel not in src, (
                f"{module.__name__} still contains inline literal: {sentinel!r}"
            )


def test_registry_acceptance_check_sources_present() -> None:
    """Audit signal: every card has a non-empty source field."""
    r = registry()
    unsourced = [c for c in r if not c.source.strip()]
    assert unsourced == [], (
        f"{len(unsourced)} coefficient cards have empty source fields; "
        "this is allowed during draft work but should fail PRs that try to "
        "ship without provenance."
    )
