"""Local LINCS L1000 signature adapter."""

from __future__ import annotations

from pathlib import Path

from .common import DataSourceBundle, make_bundle, read_local_table


def load_lincs_signatures(path: str | Path, metadata_path: str | Path | None = None) -> DataSourceBundle:
    """Load a local LINCS signature matrix and optional metadata table."""
    data = read_local_table(path)
    metadata = {
        "adapter": "lincs",
        "row_count": int(len(data)),
        "metadata_rows": 0,
    }
    if metadata_path is not None:
        meta = read_local_table(metadata_path)
        metadata["metadata_rows"] = int(len(meta))
        metadata["metadata_columns"] = list(meta.columns)
    return make_bundle(path, "LINCS L1000 / Connectivity Map", data, metadata=metadata)
