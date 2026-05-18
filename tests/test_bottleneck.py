from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor

from icg_cast.biology.biological_risk_equation import biological_risk_equation
from icg_cast.bottleneck import (
    STRUCTURAL_SIGNS,
    MechanismBottleneckClassifier,
    augment_with_interventions,
    structural_signs_from_registry,
)
from icg_cast.coefficients import registry, use_registry
from icg_cast.oracle.reference_risk_oracle import reference_risk_oracle


class _ConstantMultiOutputRegressor(BaseEstimator):
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def fit(self, X, y):
        self.n_outputs_ = np.asarray(y).shape[1]
        return self

    def predict(self, X):
        return np.full((len(X), self.n_outputs_), self.value, dtype=float)


class _NoOpInterventionModel:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        proba = np.full(len(X), 0.5, dtype=float)
        return np.column_stack([1.0 - proba, proba])

    def clear_interventions(self) -> None:
        pass

    def intervene(self, unit: str, scale: float | None = None, shift: float | None = None) -> None:
        pass


def test_structural_signs_are_derived_from_coefficient_registry() -> None:
    signs = structural_signs_from_registry()
    r = registry()

    assert STRUCTURAL_SIGNS == signs
    assert signs["state_final_DNA_adducts"] == r.card(
        "dynamics.latent_risk.dna_coupling"
    ).effect_direction
    assert signs["state_final_immune_clearance"] == r.card(
        "dynamics.latent_risk.immune_coupling"
    ).effect_direction
    assert signs["state_auc_mutation_rate"] == r.card(
        "dynamics.mutation_rate.scale"
    ).effect_direction


def test_resolve_signs_uses_active_registry_metadata() -> None:
    modified = registry().replace_card(
        "dynamics.latent_risk.dna_coupling",
        effect_direction=-1,
    )
    with use_registry(modified):
        model = MechanismBottleneckClassifier(
            bottleneck_units=("state_final_DNA_adducts",)
        )
        model.bottleneck_units_ = model.bottleneck_units
        assert structural_signs_from_registry()["state_final_DNA_adducts"] == -1
        assert model._resolve_signs() == [-1]


def test_resolve_signs_raises_for_missing_signs_by_default() -> None:
    model = MechanismBottleneckClassifier(
        bottleneck_units=("state_final_unregistered_state",),
        stage2_kind="sign_constrained",
    )
    model.bottleneck_units_ = model.bottleneck_units

    with pytest.raises(KeyError, match="state_final_unregistered_state"):
        model._resolve_signs()


def test_resolve_signs_can_warn_and_fall_back_when_not_strict() -> None:
    model = MechanismBottleneckClassifier(
        bottleneck_units=("state_final_unregistered_state",),
        stage2_kind="sign_constrained",
        strict_signs=False,
    )
    model.bottleneck_units_ = model.bottleneck_units

    with pytest.warns(RuntimeWarning, match="treating as unconstrained"):
        assert model._resolve_signs() == [0]


def test_intervention_conformity_requires_responsive_delta() -> None:
    model = _NoOpInterventionModel()
    score, table = MechanismBottleneckClassifier.score_intervention_conformity(
        model,
        pd.DataFrame({"x": [1.0, 2.0]}),
        interventions={"noop": {"state_final_DNA_adducts": 2.0}},
        expected_directions={"noop": 1},
    )

    assert score == 0.0
    assert not bool(table.loc[0, "passed_directionality"])


def test_augment_with_interventions_weights_have_total_parity() -> None:
    S = pd.DataFrame({"state_final_DNA_adducts": [1.0, 2.0, 3.0]})

    _, _, weights = augment_with_interventions(
        S,
        interventions={
            "half": {"state_final_DNA_adducts": 0.5},
            "double": {"state_final_DNA_adducts": 2.0},
        },
        latent_risk_fn=lambda rows: np.full(len(rows), 0.5, dtype=float),
        hazard_scale=1.0,
        months=1,
        rng=np.random.default_rng(0),
        samples_per_intervention=2,
    )

    assert float(weights.sum()) == pytest.approx(float(len(S)))


def test_augmented_fit_uses_stage1_predictions_for_augmented_regime() -> None:
    X = pd.DataFrame({"tx_a": np.linspace(0.0, 1.0, 20)})
    S = pd.DataFrame({"state_final_DNA_adducts": np.full(len(X), 2.0)})
    y = np.tile([0, 1], len(X) // 2)
    observed_augmented_means: list[float] = []

    def latent_risk(rows: pd.DataFrame) -> np.ndarray:
        observed_augmented_means.append(float(rows["state_final_DNA_adducts"].mean()))
        return np.full(len(rows), 0.5, dtype=float)

    model = MechanismBottleneckClassifier(
        bottleneck_units=("state_final_DNA_adducts",),
        feature_columns=("tx_a",),
        stage1_estimator=_ConstantMultiOutputRegressor(value=10.0),
        stage2_estimator=LogisticRegression(max_iter=1000),
        stage2_kind="sign_constrained_augmented",
        augment_interventions={"double": {"state_final_DNA_adducts": 2.0}},
        augment_latent_risk_fn=latent_risk,
        augment_hazard_scale=1.0,
        augment_months=1,
        random_state=0,
    )

    model.fit(X, y, S=S)

    assert observed_augmented_means == pytest.approx([20.0])


def test_structural_signs_match_default_risk_equation_directions() -> None:
    states = pd.DataFrame(
        {
            "state_final_DNA_adducts": [2.0],
            "state_final_ROS": [1.5],
            "state_final_inflammation": [1.2],
            "state_final_epigenetic_age": [3.0],
            "state_final_proliferation": [0.4],
            "state_final_driver_count_proxy": [2.0],
            "state_final_clone_fraction": [0.05],
            "state_final_immune_clearance": [0.8],
        }
    )
    direct_units = tuple(states.columns)

    for risk_fn in (reference_risk_oracle, biological_risk_equation):
        base = float(np.asarray(risk_fn(states))[0])
        for unit in direct_units:
            after = states.copy()
            after[unit] = after[unit] * 1.1
            delta = float(np.asarray(risk_fn(after))[0]) - base
            assert int(np.sign(delta)) == STRUCTURAL_SIGNS[unit], unit


def test_stage1_estimator_override_is_cloned_and_used() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "tx_a": rng.normal(size=48),
            "tx_b": rng.normal(size=48),
        }
    )
    S = pd.DataFrame(
        {
            "state_final_DNA_adducts": X["tx_a"] + 0.1 * rng.normal(size=48),
            "state_final_immune_clearance": X["tx_b"] + 0.1 * rng.normal(size=48),
        }
    )
    y = (S["state_final_DNA_adducts"] > S["state_final_DNA_adducts"].median()).astype(int)
    stage1 = DecisionTreeRegressor(random_state=0)

    model = MechanismBottleneckClassifier(
        bottleneck_units=tuple(S.columns),
        feature_columns=tuple(X.columns),
        stage1_estimator=stage1,
        stage2_estimator=LogisticRegression(max_iter=1000),
        random_state=0,
    )
    model.fit(X, y, S=S)

    assert isinstance(model.stage1_, DecisionTreeRegressor)
    assert model.stage1_ is not stage1
    assert model.predict_bottleneck(X).shape == S.shape


def test_default_stage1_tracks_missingness_indicators() -> None:
    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "tx_a": rng.normal(size=60),
            "epi_b": rng.normal(size=60),
            "dose": rng.uniform(size=60),
        }
    )
    X.loc[::3, "tx_a"] = np.nan
    S = pd.DataFrame(
        {
            "state_final_DNA_adducts": rng.uniform(size=60),
            "state_final_immune_clearance": rng.uniform(0.2, 1.0, size=60),
        }
    )
    y = (S["state_final_DNA_adducts"] > S["state_final_DNA_adducts"].median()).astype(int)
    model = MechanismBottleneckClassifier(
        bottleneck_units=tuple(S.columns),
        feature_columns=tuple(X.columns),
        stage2_estimator=LogisticRegression(max_iter=1000),
        random_state=0,
    )

    model.fit(X, y, S=S)
    report = model.missingness_report()

    assert model.stage1_.named_steps["simpleimputer"].add_indicator is True
    assert report.loc[report["feature"] == "tx_a", "missing_fraction"].iloc[0] > 0.0
    assert set(report["modality"]) >= {"transcriptomic", "epigenomic", "dose"}


def test_fit_rejects_invalid_public_inputs() -> None:
    X = pd.DataFrame({"tx_a": [0.1, 0.2, 0.3]})
    S = pd.DataFrame({"state_final_DNA_adducts": [0.1, 0.2, 0.3]})
    model = MechanismBottleneckClassifier(bottleneck_units=tuple(S.columns))

    with pytest.raises(ValueError, match="y length"):
        model.fit(X, [0, 1], S=S)
    with pytest.raises(ValueError, match="binary labels"):
        model.fit(X, [0, 1, 2], S=S)
    with pytest.raises(ValueError, match="must not contain \\+/-inf"):
        model.fit(pd.DataFrame({"tx_a": [0.1, np.inf, 0.3]}), [0, 1, 0], S=S)


def test_invalid_effect_direction_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid effect_direction"):
        registry().replace_card("dynamics.latent_risk.dna_coupling", effect_direction=2)
