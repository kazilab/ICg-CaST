"""COSMIC-driven mutational signature calibration."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..data_sources.common import DataSourceBundle


def calibrate_signatures_from_cosmic(
    bundle: DataSourceBundle,
    name_map: Mapping[str, str] | None = None,
    signature_columns: list[str] | None = None,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Build normalised 96-channel signature profiles from a COSMIC adapter bundle.

    Args:
        bundle: result of ``load_cosmic_sbs_matrix(...)``. Already validated to
            be a 96-context matrix with non-negative signature columns.
        name_map: optional ``{cosmic_column -> internal_name}`` rename, e.g.
            ``{"SBS4": "SBS4_like"}`` to drop calibrated profiles into the
            simulator's toy archetype mapping without touching constants.
        signature_columns: restrict to a subset of signature columns. Defaults
            to all columns reported by the COSMIC adapter metadata.

    Returns:
        ``(labels, profiles)`` where ``labels`` is the 96 trinucleotide contexts
        from the COSMIC file (in file order) and ``profiles`` maps each
        (renamed) signature name to a length-96 probability vector.
    """
    if bundle.metadata.get("adapter") != "cosmic":
        raise ValueError("calibrate_signatures_from_cosmic expects a COSMIC adapter bundle")
    data = bundle.data
    context_col = str(bundle.metadata["context_column"])
    labels = [str(x) for x in data[context_col].tolist()]
    if len(labels) != 96:
        raise ValueError("COSMIC bundle must contain 96 contexts")

    file_cols = signature_columns if signature_columns is not None else list(
        bundle.metadata["signature_columns"]
    )
    profiles: dict[str, np.ndarray] = {}
    for col in file_cols:
        if col not in data.columns:
            raise KeyError(f"signature column not in COSMIC bundle: {col!r}")
        values = data[col].astype(float).to_numpy()
        if (values < 0).any():
            raise ValueError(f"signature column {col!r} has negative values")
        total = float(values.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError(f"signature column {col!r} has non-positive total mass")
        out_name = name_map[col] if (name_map and col in name_map) else col
        profiles[out_name] = values / total
    return labels, profiles
