"""Local EPA ToxCast/CompTox export adapter."""

from __future__ import annotations

from pathlib import Path

from .common import DataSourceBundle, make_bundle, read_local_table


def load_toxcast_summary(path: str | Path, mapping_path: str | Path | None = None) -> DataSourceBundle:
    """Load a user-supplied ToxCast summary table and optional KCC mapping."""
    data = read_local_table(path)
    metadata = {
        "adapter": "toxcast",
        "row_count": int(len(data)),
        "mapping_rows": 0,
    }
    if mapping_path is not None:
        mapping = read_local_table(mapping_path)
        metadata["mapping_rows"] = int(len(mapping))
        metadata["mapping_columns"] = list(mapping.columns)
    return make_bundle(path, "EPA ToxCast/CompTox", data, metadata=metadata)
