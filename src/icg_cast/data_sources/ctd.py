"""Local Comparative Toxicogenomics Database adapter."""

from __future__ import annotations

from pathlib import Path

from .common import DataSourceBundle, make_bundle, read_local_table


def load_ctd_chemical_gene_disease(path: str | Path) -> DataSourceBundle:
    """Load a local CTD chemical-gene-disease or related export table."""
    data = read_local_table(path)
    metadata = {
        "adapter": "ctd",
        "row_count": int(len(data)),
        "columns": list(data.columns),
    }
    return make_bundle(path, "Comparative Toxicogenomics Database", data, metadata=metadata)
