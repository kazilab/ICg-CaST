from __future__ import annotations


def test_public_package_name_and_version() -> None:
    import icg_cast

    assert icg_cast.__project_name__ == "ICg-CaST"
    assert icg_cast.__version__


def test_public_bottleneck_exports() -> None:
    from icg_cast import DEFAULT_BOTTLENECK_UNITS, MechanismBottleneckClassifier

    assert len(DEFAULT_BOTTLENECK_UNITS) >= 5
    assert MechanismBottleneckClassifier.__name__ == "MechanismBottleneckClassifier"


def test_public_model_exports() -> None:
    from icg_cast import evaluate_bundle, feature_sets, train_baselines

    assert callable(feature_sets)
    assert callable(train_baselines)
    assert callable(evaluate_bundle)
