from __future__ import annotations

import pytest

from icg_cast.biology.biological_risk_equation import biological_risk_equation
from icg_cast.oracle.reference_risk_oracle import reference_risk_oracle


def test_reference_risk_oracle_rejects_missing_state_alias_by_default() -> None:
    with pytest.raises(KeyError, match="dna_adducts"):
        reference_risk_oracle(dna_adduct=1.0)


def test_biological_risk_equation_allows_partial_state_inputs() -> None:
    risk = biological_risk_equation(dna_adducts=1.0)

    assert isinstance(risk, float)
    assert 0.0 <= risk <= 1.0
