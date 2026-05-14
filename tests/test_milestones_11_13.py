from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from icg_cast import SimConfig, simulate_cohort
from icg_cast.audit.coefficient_sensitivity import run_coefficient_sensitivity_audit
from icg_cast.audit.validate_reviews import validate_coefficient_reviews
from icg_cast.biology.biological_risk_equation import biological_risk_equation
from icg_cast.bottleneck import starter_kit_latent_risk
from icg_cast.coefficients import registry
from icg_cast.oracle.reference_risk_oracle import get_oracle_version, reference_risk_oracle


def test_reference_oracle_and_biology_are_separate_and_callable() -> None:
    cohort, _ = simulate_cohort(SimConfig(n=12, months=6, seed=9))
    states = cohort[[c for c in cohort.columns if c.startswith("state_final_")]]

    oracle = reference_risk_oracle(states)
    biology = biological_risk_equation(states)

    assert get_oracle_version() == "v1.0"
    assert isinstance(oracle, np.ndarray)
    assert isinstance(biology, np.ndarray)
    assert np.allclose(oracle, starter_kit_latent_risk(states))
    # Default biology uses the same point coefficients; calibrated/prior modes
    # can diverge without changing the frozen oracle.
    prior_biology = biological_risk_equation(states, use_priors=True, seed=123)
    assert not np.allclose(biology, prior_biology)


def test_coefficient_sensitivity_audit_changes_coefficients_and_writes_outputs(tmp_path: Path) -> None:
    cfg = SimConfig(n=30, months=12, seed=4, archetype_prior={"pah_tobacco_like": 1.0})
    df = run_coefficient_sensitivity_audit(
        simulate_cohort,
        cfg,
        coefficients_to_test=["dynamics.latent_risk.clone_coupling"],
        output_dir=tmp_path,
        n_samples=30,
    )

    assert not df.empty
    assert set(df["scale"]) == {0.5, 1.0, 2.0}
    assert df["relative_change"].abs().max() > 0.0
    assert (tmp_path / "coefficient_sensitivity.csv").exists()
    assert (tmp_path / "coefficient_sensitivity_heatmap.png").exists()
    assert (tmp_path / "coefficient_flags.csv").exists()


def test_default_coefficient_review_file_has_required_columns() -> None:
    review_path = Path("materials/coefficient_review.csv")
    review = pd.read_csv(review_path)
    assert {"coefficient", "reviewed_by", "review_date", "decision", "dissent_notes"} <= set(
        review.columns
    )
    assert review["reviewed_by"].astype(str).str.strip().ne("").all()
    assert "dynamics.latent_risk.clone_coupling" in set(review["coefficient"])


def test_review_validator_checks_load_bearing_rows(tmp_path: Path) -> None:
    flags = tmp_path / "flags.csv"
    pd.DataFrame(
        [
            {
                "coefficient": "dynamics.latent_risk.clone_coupling",
                "flag": "LOAD_BEARING",
            }
        ]
    ).to_csv(flags, index=False)
    assert validate_coefficient_reviews(
        review_file="materials/coefficient_review.csv",
        flags_file=str(flags),
    )
    assert "dynamics.latent_risk.clone_coupling" in registry()

