"""Validation utilities for ICg-CaST.

This subpackage groups predictive and mechanistic validation helpers that
operate on cohorts and fitted models. Predictive metrics live in
``calibration``; mechanistic checks live in ``biological_coherence``;
cross-species relevance estimates live in ``cross_species``.

The historical entry points in :mod:`icg_cast.models` continue to work; the
modules in this subpackage are thin, focused wrappers (with the exception of
``cross_species``, which is new).
"""

from __future__ import annotations

from .biological_coherence import (
    biological_coherence_score,
    biological_coherence_summary,
    effect_weighted_biological_coherence,
    pathway_attribution_consistency,
    severity_effect_weighted_biological_coherence,
    severity_weighted_biological_coherence,
)
from .calibration import (
    calibration_curve,
    calibration_metrics,
    expected_calibration_error,
)
from .cross_species import (
    HRTIResult,
    human_relevance_transfer_index,
)

__all__ = [
    "HRTIResult",
    "biological_coherence_score",
    "biological_coherence_summary",
    "calibration_curve",
    "calibration_metrics",
    "effect_weighted_biological_coherence",
    "expected_calibration_error",
    "human_relevance_transfer_index",
    "pathway_attribution_consistency",
    "severity_effect_weighted_biological_coherence",
    "severity_weighted_biological_coherence",
]
