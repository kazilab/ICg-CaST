from __future__ import annotations

import numpy as np
import pandas as pd

from icg_cast.coefficients import registry, use_registry
from icg_cast.counterfactuals import counterfactual_tests
from icg_cast.interventions import EXPECTED_DIRECTIONS_PRIOR


class _ColumnRiskModel:
    def __init__(self, column: str) -> None:
        self.column = column

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = X[self.column].to_numpy(dtype=float)
        return np.column_stack([1.0 - proba, proba])


def test_counterfactuals_use_explicit_allowlists_not_substrings() -> None:
    df = pd.DataFrame({"tx_crossROSstudy": [0.4, 0.6]})
    result = counterfactual_tests(
        _ColumnRiskModel("tx_crossROSstudy"),
        df,
        ["tx_crossROSstudy"],
    )

    ros = result.set_index("intervention").loc["do_ROS_inflammation_blockade"]
    assert ros["mean_absolute_risk_change"] == 0.0
    assert ros["intervention_severity_weight"] > 0.0


def test_counterfactuals_split_immune_evasion_and_clearance() -> None:
    df = pd.DataFrame(
        {
            "tx_immune_evasion": [0.5, 0.5],
            "state_final_immune_clearance": [0.5, 0.5],
        }
    )

    result = counterfactual_tests(
        _ColumnRiskModel("tx_immune_evasion"),
        df,
        ["tx_immune_evasion", "state_final_immune_clearance"],
    )

    interventions = set(result["intervention"])
    assert "do_lower_immune_evasion" in interventions
    assert "do_raise_immune_clearance" in interventions
    assert "do_immune_surveillance_restore" in interventions

    by_name = result.set_index("intervention")
    assert by_name.loc["do_lower_immune_evasion", "mean_absolute_risk_change"] < 0.0
    assert by_name.loc["do_raise_immune_clearance", "mean_absolute_risk_change"] == 0.0
    assert by_name.loc["do_immune_surveillance_restore", "mean_absolute_risk_change"] < 0.0
    assert (by_name["intervention_severity_weight"] > 0.0).all()


def test_expected_directions_prior_follow_active_registry_signs() -> None:
    assert EXPECTED_DIRECTIONS_PRIOR["do_immune_surveillance_restore"] == -1

    modified = registry().replace_card(
        "dynamics.latent_risk.immune_coupling",
        effect_direction=1,
    )
    with use_registry(modified):
        assert EXPECTED_DIRECTIONS_PRIOR["do_immune_surveillance_restore"] == 1
