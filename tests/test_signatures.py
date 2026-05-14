from __future__ import annotations

import numpy as np

from icg_cast import make_signature_profiles, mutation_context_labels


def test_signature_profiles_are_valid_toy_96_channel_profiles() -> None:
    labels = mutation_context_labels()
    profile_labels, profiles = make_signature_profiles()

    assert len(labels) == 96
    assert profile_labels == labels
    assert profiles
    for profile in profiles.values():
        assert profile.shape == (96,)
        assert np.all(profile >= 0)
        assert np.isclose(profile.sum(), 1.0)
