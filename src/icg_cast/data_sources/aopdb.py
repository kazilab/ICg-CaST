"""Local EPA AOP-DB export adapter."""

from __future__ import annotations

from pathlib import Path

from .common import DataSourceBundle, make_bundle, read_local_table


def load_aopdb_export(path: str | Path, table: str | None = None) -> DataSourceBundle:
    """Load a user-supplied AOP-DB CSV/TSV/JSON/SQLite-derived export."""
    data = read_local_table(path, table=table)
    metadata = {
        "adapter": "aopdb",
        "row_count": int(len(data)),
        "table": table or "auto",
    }
    return make_bundle(path, "EPA AOP-DB", data, metadata=metadata)
