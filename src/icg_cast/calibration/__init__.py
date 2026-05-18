"""Public-data calibration prototype for ICg-CaST.

The calibration layer consumes the local-file adapters in
``icg_cast.data_sources`` and produces opt-in overrides for the synthetic
simulator and theory graph. It is intentionally non-invasive: the default
``simulate_cohort`` / ``build_theory_graph`` behaviour is unchanged unless a
caller passes a ``CalibrationBundle``.

Real data workflows are documented but not required for tests. Tests in this
package use tiny synthetic fixtures only; no real COSMIC, LINCS, ToxCast, or
AOP-Wiki files are downloaded or committed.
"""

from __future__ import annotations

from .bundle import CalibrationBundle, load_calibration_bundle
from .coupling import calibrated_registry_from_bundle
from .graph_enrichment import enrich_theory_graph
from .kcc_priors import calibrate_kcc_priors_from_toxcast
from .modules import calibrate_transcript_modules_from_lincs
from .pipeline import build_calibration_bundle
from .signatures import calibrate_signatures_from_cosmic

__all__ = [
    "CalibrationBundle",
    "build_calibration_bundle",
    "calibrate_kcc_priors_from_toxcast",
    "calibrated_registry_from_bundle",
    "calibrate_signatures_from_cosmic",
    "calibrate_transcript_modules_from_lincs",
    "enrich_theory_graph",
    "load_calibration_bundle",
]
