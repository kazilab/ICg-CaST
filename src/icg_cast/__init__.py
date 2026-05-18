"""ICg-CaST public package interface.

ICg-CaST: Integrated Carcinogenomics Causal State Theory. 
The Python import is ``icg_cast`` and the console command is ``icg-cast``. 
All brand-level strings (display name, version, distribution name, default output filenames) live in
``icg_cast._branding`` so this package has a single source of truth.
"""

from . import _branding
from ._branding import (
    PROJECT_LONG_NAME as __project_long_name__,
)
from ._branding import (
    PROJECT_NAME as __project_name__,
)
from ._branding import (
    VERSION as __version__,
)
from .biology import biological_risk_equation
from .bottleneck import (
    DEFAULT_BOTTLENECK_UNITS,
    MechanismBottleneckClassifier,
    SignConstrainedLogisticRegression,
)
from .calibration import (
    CalibrationBundle,
    build_calibration_bundle,
    load_calibration_bundle,
)
from .config import ChemicalArchetype, SimConfig
from .constants import ARCHETYPE_KCC, KCC_NAMES, STATE_NAMES
from .graph import build_theory_graph, export_graph
from .models import evaluate_bundle, feature_sets, train_baselines, validate_no_target_leakage
from .oracle import reference_risk_oracle
from .signatures import make_signature_profiles, mutation_context_labels
from .simulator import simulate_cohort, simulate_state_trajectory, summarize_trajectory
from .survival import restricted_mean_survival, time_to_event

__all__ = [
    "_branding",
    "__project_name__",
    "__project_long_name__",
    "__version__",
    "DEFAULT_BOTTLENECK_UNITS",
    "ARCHETYPE_KCC",
    "CalibrationBundle",
    "ChemicalArchetype",
    "KCC_NAMES",
    "MechanismBottleneckClassifier",
    "STATE_NAMES",
    "SignConstrainedLogisticRegression",
    "SimConfig",
    "build_calibration_bundle",
    "build_theory_graph",
    "biological_risk_equation",
    "export_graph",
    "evaluate_bundle",
    "feature_sets",
    "load_calibration_bundle",
    "make_signature_profiles",
    "mutation_context_labels",
    "reference_risk_oracle",
    "restricted_mean_survival",
    "simulate_cohort",
    "simulate_state_trajectory",
    "summarize_trajectory",
    "time_to_event",
    "train_baselines",
    "validate_no_target_leakage",
]
