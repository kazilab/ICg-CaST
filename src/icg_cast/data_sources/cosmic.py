"""Local COSMIC mutational-signature matrix adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import DataSourceBundle, make_bundle, read_local_table


def load_cosmic_sbs_matrix(path: str | Path, context_column: str = "context") -> DataSourceBundle:
    """Load a local 96-channel SBS signature matrix.

    The loader expects one row per trinucleotide context and one or more
    non-negative numeric signature columns. It records provenance but does not
    download COSMIC files or resolve licensing.
    """
    data = read_local_table(path)
    if context_column not in data.columns:
        context_column = str(data.columns[0])
    if len(data) != 96:
        raise ValueError(f"COSMIC SBS matrices must contain 96 contexts; got {len(data)}")
    signature_cols = [c for c in data.columns if c != context_column]
    if not signature_cols:
        raise ValueError("COSMIC SBS matrix must include at least one signature column")
    numeric = data[signature_cols].astype(float)
    if (numeric < 0).any().any():
        raise ValueError("COSMIC SBS signature values must be non-negative")
    col_sums = numeric.sum(axis=0).to_numpy(dtype=float)
    if not np.all(np.isfinite(col_sums)) or np.any(col_sums <= 0):
        raise ValueError("each COSMIC SBS signature column must have positive finite mass")
    metadata = {
        "adapter": "cosmic",
        "context_column": context_column,
        "n_contexts": int(len(data)),
        "n_signatures": int(len(signature_cols)),
        "signature_columns": signature_cols,
    }
    return make_bundle(path, "COSMIC Mutational Signatures", data, metadata=metadata)
