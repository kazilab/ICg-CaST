from __future__ import annotations

import numpy as np
import pandas as pd

from icg_cast import restricted_mean_survival, time_to_event


def test_time_to_event_crossing_and_censoring() -> None:
    crossed = pd.DataFrame({"latent_risk": [0.1, 0.3, 0.55, 0.7]})
    censored = pd.DataFrame({"latent_risk": [0.1, 0.3, 0.49, 0.4]})

    assert time_to_event(crossed, threshold=0.5) == (3, 1)
    assert time_to_event(censored, threshold=0.5) == (4, 0)


def test_restricted_mean_survival_is_finite() -> None:
    rmst = restricted_mean_survival(
        times=np.array([2, 4, 6]),
        events=np.array([1, 0, 1]),
        horizon=6,
    )

    assert np.isfinite(rmst)
    assert 0 <= rmst <= 6
