# Mechanism-Bottleneck Causal Networks (MB-CNet)

MB-CNet is ICg-CaST's primary methodological contribution. It is a two-stage
model whose risk predictions are forced to flow through a hidden layer pinned
to the qAOP latent state vector. The bottleneck pin is what converts the
package's coherence story from a **post-hoc evaluation** (predict, then ask
whether the explanation matches the biology) into a **by-construction
constraint** (any risk prediction must factor through bottleneck units, so
do-operations on those units are well-defined interventions on the model
itself, not on the input features).

Source: [src/icg_cast/bottleneck.py](../src/icg_cast/bottleneck.py).

## Architecture

```text
stage 1: g_phi : omics_features  ->  hat{qAOP_state}      (multi-output regressor)
stage 2: h_theta: hat{qAOP_state} ->  risk probability    (calibrated classifier)
```

The default bottleneck units are the nine `state_final_*` qAOP states
(`DEFAULT_BOTTLENECK_UNITS` in `bottleneck.py`):

```text
state_final_DNA_adducts
state_final_ROS
state_final_inflammation
state_final_epigenetic_age
state_final_proliferation
state_final_mutation_rate
state_final_clone_fraction
state_final_driver_count_proxy
state_final_immune_clearance
```

Stage 1 is a `MultiOutputRegressor` (default: `RandomForestRegressor`). Pass a
custom sklearn-compatible `stage1_estimator` to swap in a stronger latent-state
recoverer while preserving the bottleneck contract:

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import HistGradientBoostingRegressor

model = MechanismBottleneckClassifier(
    stage1_estimator=MultiOutputRegressor(HistGradientBoostingRegressor()),
)
```

For the default stage-1 pipeline, NaN-valued omics features are imputed with
`SimpleImputer(strategy="mean", add_indicator=True)`. The added indicators make
modality dropout observable to the stage-1 regressor instead of silently
collapsing masked values onto the feature mean. After fitting, call
`model.missingness_report()` to audit per-feature missing fractions by modality.

Stage 2 has three available kinds:

- **`calibrated_logistic`** (v0.1 default) — unconstrained logistic regression
  with isotonic calibration.
- **`sign_constrained`** — `SignConstrainedLogisticRegression`; coefficients
  per bottleneck unit are bounded to the half-line implied by coefficient-card
  `effect_direction` metadata. `STRUCTURAL_SIGNS` remains as a derived
  compatibility export, not the source of truth.
- **`sign_constrained_augmented`** — the same sign constraints, but stage 2
  is additionally trained on intervention-implied synthetic samples drawn
  through `augment_with_interventions` using the simulator's
  `starter_kit_latent_risk` structural equation. This closes the loophole
  where the constraint can drive a coefficient to zero without ever
  responding to the intervention.

The module uses **scikit-learn only** — no `torch`, `jax`, or
`pytorch-geometric`. The differentiable, end-to-end, neural-ODE / UDE version
is explicitly deferred (PLAN.md §7.4).

## Do-operations on bottleneck units

```python
from icg_cast.bottleneck import MechanismBottleneckClassifier

model = MechanismBottleneckClassifier(stage2_kind="sign_constrained_augmented")
model.fit(X_train.join(S_train), y_train)

base = model.predict_proba(X_test)[:, 1]
model.intervene(unit="state_final_DNA_adducts", scale=0.5)   # do(DNA_adducts := 0.5 x)
after = model.predict_proba(X_test)[:, 1]
model.clear_interventions()
```

Because the intervention is applied to the *bottleneck row* rather than to
the input features, the resulting prediction is the model's analogue of a
structural do-operation on the qAOP state. Conformity to the simulator's
DGP-implied direction is what ICg-Bench's
[task_intervention_conformity](../src/icg_cast/benchmark/tasks.py) measures.

## Survival outcomes

[src/icg_cast/survival.py](../src/icg_cast/survival.py) supplies the
time-to-threshold variant of the outcome:

- `time_to_event(trajectory, column="latent_risk", threshold=0.5)` — returns
  `(time_index, event_observed)`, right-censored at the trajectory horizon.
- `add_survival_columns(cohort, trajectories, threshold, horizon)` — appends
  `time_to_high_risk_threshold` and `event_observed` to a cohort.
- `restricted_mean_survival(times, events, horizon)` — RMST via the
  step-function Kaplan-Meier integral. No `lifelines` dependency.
- `counterfactual_rmst_difference(model, cohort, intervention, horizon)` —
  bootstrap-CI estimate of `RMST(after) - RMST(before)` under a callable
  intervention on the cohort.

The binary `future_cancer_transition_event` column from
`simulate_cohort` is preserved unchanged; the survival columns are purely
additive.

## Acceptance criteria

- `bottleneck_recovery.csv` reports per-state recovery R² — see
  [outputs/bottleneck_v0_5/per_state_recovery_r2.csv](../outputs/bottleneck_v0_5/per_state_recovery_r2.csv).
- Mean recovery R² ≥ 0.60 across the 10 qAOP states on the default cohort.
- Intervention-conformity score ≥ 0.85 across the seven `do_*` interventions
  defined in `cli.py` (`_INTERVENTIONS`).
- AUROC within 0.03 of the best unconstrained multi-omics baseline.
- `bottleneck.py` does not import torch, jax, or any heavy ML dependency.
- Survival outcomes reproduce the binary event under threshold equivalence;
  RMST is finite for every archetype on the default cohort.

When a result falls below an acceptance threshold it is reported as
`failed_directionality_test` (or the corresponding flag), not as a software
failure — see PLAN.md §7.3.

## CLI usage

```bash
# train one cohort/variant/seed
icg-cast bench run --cohort linear_lowhet --variant sign_constrained_augmented --seed 7

# audit which structural signs actually bind (relax one at a time)
icg-cast bench audit --cohort linear_lowhet --variant sign_constrained --seed 7

# full sweep + figures (delegated to scripts/bottleneck_proof_of_concept.py)
icg-cast bench sweep
icg-cast bench plots
```

See also [docs/benchmark.md](benchmark.md) for the ICg-Bench scoring tasks
that MB-CNet is the reference participant for.
