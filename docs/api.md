# API

The public API is intentionally small. Import from `icg_cast` for the stable
surface and from submodules when working on experimental internals.

## Simulation

```python
from icg_cast import SimConfig, simulate_cohort

cfg = SimConfig(n=120, months=72, seed=7)
cohort, trajectories = simulate_cohort(cfg)

uncertainty_cfg = SimConfig(
    n=120,
    months=72,
    seed=7,
    coefficient_mode="prior_sample",
    coefficient_seed=42,
)
uncertain_cohort, _ = simulate_cohort(uncertainty_cfg)
```

Key functions:

- `SimConfig`: simulation configuration dataclass.
- `simulate_cohort(cfg)`: returns a cohort table and retained trajectories.
- `simulate_state_trajectory(...)`: lower-level state recurrence.
- `summarize_trajectory(traj)`: final and AUC state summaries.

`SimConfig.coefficient_mode="point"` uses registry point values.
`"prior_sample"` draws one seedable coefficient realization for the whole
cohort and records it in the `coefficient_seed` column.

## Signatures

```python
from icg_cast import make_signature_profiles, mutation_context_labels

labels = mutation_context_labels()
labels, profiles = make_signature_profiles()
```

Synthetic profiles are simplified approximations. Use optional local COSMIC
loader stubs only with user-supplied files and explicit provenance.

## Graph

```python
from icg_cast import build_theory_graph, export_graph

graph = build_theory_graph()
export_graph(graph, "outputs/demo")
```

## Modeling

```python
from icg_cast import train_baselines, evaluate_bundle

metrics, importance, counterfactual, bundle = train_baselines(cohort, seed=7)
eval_metrics, calibration, cf, coherence = evaluate_bundle(cohort, bundle)
```

The feature-set builder rejects target-derived columns such as future endpoint
probabilities and latent-risk summaries.

## Optional Data Sources

Optional local-file adapters live under `icg_cast.data_sources`. They return a
`DataSourceBundle` containing `data`, `provenance`, and adapter metadata.

```python
from icg_cast.data_sources import load_cosmic_sbs_matrix

bundle = load_cosmic_sbs_matrix("local_cosmic_sbs.csv")
```

Adapters do not download data and do not authorize use of restricted datasets.
