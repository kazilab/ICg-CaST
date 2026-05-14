# Validation

ICg-CaST separates predictive performance from mechanistic coherence.
Synthetic AUROC is useful software evidence, but it is not biological
validation.

The helpers in [src/icg_cast/validation/](../src/icg_cast/validation/) group
the three families of checks below:

```python
from icg_cast.validation import (
    biological_coherence_score,
    calibration_curve,
    expected_calibration_error,
    human_relevance_transfer_index,
    pathway_attribution_consistency,
)
```

## Predictive Metrics

Baseline training and evaluation report:

- ROC AUC.
- Average precision.
- Brier score.
- Event rate.
- Mean predicted risk.
- Calibration bins and expected calibration error.

`validation.calibration` adds two leaner entry points:

- `expected_calibration_error(y, proba, n_bins=10)` returns the ECE scalar.
- `calibration_curve(y, proba, n_bins=10)` returns
  `(mean_predicted, observed_fraction, counts)` arrays for reliability plots.

## Mechanistic Checks

The package includes counterfactual directionality tests for mechanism-linked
feature perturbations. A model can score well predictively while failing a
directionality test. Such failures are reported as biological-coherence
diagnostics, not as software errors.

The biological-coherence score is:

```text
correct_direction_count / tested_intervention_count
```

`validation.biological_coherence` provides:

- `biological_coherence_score(counterfactual_table)` returns the scalar
  directly.
- `pathway_attribution_consistency(importance, pathway_map)` aggregates
  per-feature permutation importance into per-pathway shares, so feature
  weight can be inspected at the modality / pathway level.

For *by-construction* (rather than post-hoc) coherence, see
[docs/bottleneck.md](bottleneck.md) and the
`task_intervention_conformity` task in
[docs/benchmark.md](benchmark.md).

## Cross-Species Human Relevance

`validation.cross_species.human_relevance_transfer_index` implements the HRTI
estimate from PLAN.md §9.4:

```text
HRTI = conserved_human_KE_activation
      / (conserved_human_KE_activation + rodent_specific_KE_activation)
```

It takes an explicit table with `key_event`, `conservation`,
`human_activation`, and `rodent_activation` columns and returns an
`HRTIResult` with the score, contributing counts, and per-key-event reason
strings. It does **not** wrap a classifier and does not look up KE conservation
databases automatically — the caller supplies the conservation labels. This is
intentional: HRTI is a transparent ratio, not a regulatory conclusion.

## Simulator Sanity Checks

Internal consistency checks should focus on synthetic-world expectations:

- Inert controls should usually have lower risk than active archetypes.
- Genotoxic archetypes should elevate DNA-damage and mutational features.
- ROS archetypes should elevate oxidative and inflammatory features.
- Receptor-mediated archetypes should elevate proliferative modules.
- Immunosuppressive archetypes should reduce clearance-related features.

These are checks on the simulator assumptions, not real-world claims.

## External Validation

Real-data validation is future work and requires local files, provenance,
appropriate permissions, and domain review. Human genomic data can be
identifying. Controlled-access datasets require the proper approvals before any
analysis.
