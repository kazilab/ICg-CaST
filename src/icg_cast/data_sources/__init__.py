"""Optional local-file data-source adapters for ICg-CaST.

The adapters in this package are intentionally offline by default. They accept
user-supplied local files, return lightweight bundles, and record provenance.
They do not download public datasets and do not handle controlled-access data.
"""

from .aopdb import load_aopdb_export
from .aopwiki import load_aopwiki_export, map_aop_to_theory_graph
from .common import (
    DataSourceBundle,
    Provenance,
    calibration_provenance_payload,
    read_local_table,
    validate_calibration_provenance,
    validate_provenance_record,
)
from .cosmic import load_cosmic_sbs_matrix
from .ctd import load_ctd_chemical_gene_disease
from .gdc import load_gdc_manifest
from .lincs import load_lincs_signatures
from .sigprofiler import load_sigprofiler_activities
from .toxcast import load_toxcast_summary

__all__ = [
    "DataSourceBundle",
    "Provenance",
    "calibration_provenance_payload",
    "load_aopdb_export",
    "load_aopwiki_export",
    "load_cosmic_sbs_matrix",
    "load_ctd_chemical_gene_disease",
    "load_gdc_manifest",
    "load_lincs_signatures",
    "load_sigprofiler_activities",
    "load_toxcast_summary",
    "map_aop_to_theory_graph",
    "read_local_table",
    "validate_calibration_provenance",
    "validate_provenance_record",
]
