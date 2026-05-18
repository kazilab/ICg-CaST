"""ICg-Bench: a public causal benchmark with a known data-generating process.

Four scored tasks:

1. ``task_risk_prediction``        AUROC / AUPRC / Brier / calibration on held-out subjects.
2. ``task_latent_recovery``        Per-state R^2 between a model's predicted bottleneck and the true qAOP state.
3. ``task_intervention_conformity`` Fraction of ``do_*`` interventions whose mean predicted-risk change matches the expected sign.
4. ``task_cross_host_generalization`` Source-to-target AUROC and transfer gap when host-susceptibility distributions shift.

The benchmark is intentionally a synthetic-DGP-with-known-ground-truth setup
because that is the only setting in biology where causal estimands can be
scored exactly. Real-data calibration is handled by `data_sources/` adapters
and is out of scope for ICg-Bench scoring.
"""

from .dgp import DGPVariant, list_variants, load_variant
from .generators import (
    generate,
    generate_linear_lowhet,
    generate_misspecified_signs,
    generate_misspecified_signs_v2,
    generate_nonlinear_mixhost,
    generate_nonlinear_obs,
    generate_partial_observability,
    list_generator_names,
)
from .leaderboard import (
    LeaderboardEntry,
    LeaderboardSchemaError,
    append_entry,
    load_leaderboard,
    migrate_leaderboard_entries,
    validate_leaderboard_entry,
    write_leaderboard,
)
from .scoring import BenchmarkResult, run_benchmark, score_summary
from .tasks import (
    task_cross_host_generalization,
    task_intervention_conformity,
    task_latent_recovery,
    task_risk_prediction,
)

__all__ = [
    "DGPVariant",
    "list_variants",
    "load_variant",
    "generate",
    "generate_linear_lowhet",
    "generate_misspecified_signs",
    "generate_misspecified_signs_v2",
    "generate_nonlinear_mixhost",
    "generate_nonlinear_obs",
    "generate_partial_observability",
    "list_generator_names",
    "LeaderboardEntry",
    "LeaderboardSchemaError",
    "append_entry",
    "load_leaderboard",
    "migrate_leaderboard_entries",
    "validate_leaderboard_entry",
    "write_leaderboard",
    "BenchmarkResult",
    "run_benchmark",
    "score_summary",
    "task_cross_host_generalization",
    "task_intervention_conformity",
    "task_latent_recovery",
    "task_risk_prediction",
]
