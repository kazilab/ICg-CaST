"""Synthetic mutational signature profiles.

All recipe parameters (background RNG seed, gamma shape/scale, per-context
boost magnitudes, the lower clip applied before normalisation) come from
the coefficient registry — see :mod:`icg_cast.coefficients` and the
``signatures.*`` block in ``materials/coefficient_cards.yaml``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .coefficients import registry as _registry

if TYPE_CHECKING:
    from .calibration.bundle import CalibrationBundle


@dataclass(frozen=True)
class _SignatureCoefficients:
    seed: int
    gamma_shape: float
    gamma_scale: float
    min_clip: float
    aging_ct: float
    aging_cpg: float
    sbs4_ca: float
    sbs4_at_end: float
    sbs24_ca: float
    sbs24_pattern: float
    sbs22_ta: float
    sbs22_ct_lead: float
    ox_ca_or_cg: float
    ox_tg: float


@functools.cache
def _sig_coeffs() -> _SignatureCoefficients:
    r = _registry()
    g = r.get
    return _SignatureCoefficients(
        seed=int(g("signatures.background.seed")),
        gamma_shape=g("signatures.background.gamma_shape"),
        gamma_scale=g("signatures.background.gamma_scale"),
        min_clip=g("signatures.background.min_clip"),
        aging_ct=g("signatures.aging.C_to_T_boost"),
        aging_cpg=g("signatures.aging.CpG_boost"),
        sbs4_ca=g("signatures.SBS4_like.C_to_A_boost"),
        sbs4_at_end=g("signatures.SBS4_like.AT_ending_extra_boost"),
        sbs24_ca=g("signatures.SBS24_like.C_to_A_boost"),
        sbs24_pattern=g("signatures.SBS24_like.context_pattern_boost"),
        sbs22_ta=g("signatures.SBS22_like.T_to_A_boost"),
        sbs22_ct_lead=g("signatures.SBS22_like.CT_leading_extra_boost"),
        ox_ca_or_cg=g("signatures.oxidative_like.C_to_AorG_boost"),
        ox_tg=g("signatures.oxidative_like.T_to_G_boost"),
    )


def mutation_context_labels() -> list[str]:
    """Return the 96 SBS-style trinucleotide contexts used by the toy simulator."""
    bases = ["A", "C", "G", "T"]
    substitutions = ["C>A", "C>G", "C>T", "T>A", "T>C", "T>G"]
    return [
        f"{left}[{substitution}]{right}"
        for substitution in substitutions
        for left in bases
        for right in bases
    ]


def make_signature_profiles(
    calibration: CalibrationBundle | None = None,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Create simplified 96-channel profiles for synthetic experiments only.

    If a :class:`CalibrationBundle` with calibrated signature profiles is
    provided, calibrated profiles are merged in and override any toy profile
    with the same name. Toy profiles required by the default
    ``ARCHETYPE_SIGNATURE`` mapping but absent from the bundle are preserved so
    the simulator can still resolve every archetype.
    """
    S = _sig_coeffs()
    labels = mutation_context_labels()
    profiles: dict[str, np.ndarray] = {}

    def normalized(weights: np.ndarray) -> np.ndarray:
        weights = np.clip(weights, S.min_clip, None)
        return weights / weights.sum()

    rng = np.random.default_rng(S.seed)
    background = rng.gamma(shape=S.gamma_shape, scale=S.gamma_scale, size=len(labels))

    w = background.copy()
    for i, label in enumerate(labels):
        if "[C>T]" in label:
            w[i] += S.aging_ct
        if label.startswith(("A[C>T]G", "C[C>T]G", "G[C>T]G", "T[C>T]G")):
            w[i] += S.aging_cpg
    profiles["aging"] = normalized(w)

    w = background.copy()
    for i, label in enumerate(labels):
        if "[C>A]" in label:
            w[i] += S.sbs4_ca
            if label.endswith(("A", "T")):
                w[i] += S.sbs4_at_end
    profiles["SBS4_like"] = normalized(w)

    w = background.copy()
    for i, label in enumerate(labels):
        if "[C>A]" in label:
            w[i] += S.sbs24_ca
            if label[0] in {"G", "T"} and label[-1] in {"G", "C"}:
                w[i] += S.sbs24_pattern
    profiles["SBS24_like"] = normalized(w)

    w = background.copy()
    for i, label in enumerate(labels):
        if "[T>A]" in label:
            w[i] += S.sbs22_ta
            if label[0] in {"C", "T"}:
                w[i] += S.sbs22_ct_lead
    profiles["SBS22_like"] = normalized(w)

    w = background.copy()
    for i, label in enumerate(labels):
        if "[C>A]" in label or "[C>G]" in label:
            w[i] += S.ox_ca_or_cg
        if "[T>G]" in label:
            w[i] += S.ox_tg
    profiles["oxidative_like"] = normalized(w)

    if calibration is not None:
        calibrated = calibration.signature_profile_arrays()
        if calibrated is not None:
            labels, override = calibrated
            for name, arr in override.items():
                profiles[name] = arr
    return labels, profiles
