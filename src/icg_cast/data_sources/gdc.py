"""Local NCI Genomic Data Commons manifest adapter."""

from __future__ import annotations

from pathlib import Path

from .common import DataSourceBundle, make_bundle, read_local_table


def load_gdc_manifest(path: str | Path) -> DataSourceBundle:
    """Load a local GDC/TCGA/CPTAC manifest or open-access metadata table."""
    data = read_local_table(path)
    metadata = {
        "adapter": "gdc",
        "row_count": int(len(data)),
        "controlled_access_warning": "do not commit patient-level controlled-access data",
    }
    return make_bundle(path, "NCI Genomic Data Commons", data, metadata=metadata)
