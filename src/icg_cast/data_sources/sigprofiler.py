"""Local SigProfiler output adapter."""

from __future__ import annotations

from pathlib import Path

from .common import DataSourceBundle, make_bundle, read_local_table


def load_sigprofiler_activities(path: str | Path) -> DataSourceBundle:
    """Load a local SigProfiler activity table."""
    data = read_local_table(path)
    metadata = {
        "adapter": "sigprofiler",
        "row_count": int(len(data)),
        "columns": list(data.columns),
    }
    return make_bundle(path, "SigProfilerExtractor", data, metadata=metadata)
