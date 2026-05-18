from __future__ import annotations

import numpy as np
import pytest

from icg_cast.calibration.bundle import CalibrationBundle
from icg_cast.omics import (
    _bounded_primary_weight,
    _cosine_similarity,
    _lincs_module_multipliers,
)


def test_signature_activity_uses_bounded_cosine_similarity() -> None:
    empirical = np.array([1.0, 0.0])
    profile = np.array([0.9, 0.1])

    activity = _cosine_similarity(empirical, profile)

    assert activity == pytest.approx(0.9 / np.sqrt(0.82))
    assert 0.0 <= activity <= 1.0


def test_primary_signature_weight_is_soft_bounded_not_clipped() -> None:
    first = _bounded_primary_weight(
        0.9,
        lower=0.05,
        upper=0.85,
        center=0.45,
        scale=0.20,
    )
    second = _bounded_primary_weight(
        1.2,
        lower=0.05,
        upper=0.85,
        center=0.45,
        scale=0.20,
    )

    assert 0.05 < first < second < 0.85


def test_lincs_multiplier_preserves_high_score_separation() -> None:
    calibration = CalibrationBundle(
        transcript_module_priors=[
            {"perturbagen": "chem", "module": "module_low", "mean_score": 5.0},
            {"perturbagen": "chem", "module": "module_high", "mean_score": 20.0},
        ]
    )

    multipliers = _lincs_module_multipliers(calibration, "chem")

    assert multipliers["module_high"] > multipliers["module_low"]
    assert multipliers["module_high"] - multipliers["module_low"] > 0.01
