# CLI

Install in editable mode before using the command-line interface:

```bash
python -m pip install -e ".[dev]"
```

## Simulate

```bash
icg-cast simulate --n 1200 --months 72 --seed 7 --outdir outputs/demo
```

Writes a synthetic cohort, metadata, and optional trajectory plots.

To sample one coefficient-prior realization for the whole cohort:

```bash
icg-cast simulate \
  --n 1200 \
  --months 72 \
  --seed 7 \
  --coefficient-mode prior_sample \
  --coefficient-seed 42 \
  --outdir outputs/demo_uncertainty
```

## Make Demo

```bash
icg-cast make-demo --n 120 --months 72 --seed 7 --outdir outputs/demo
```

Runs the reproducible demo workflow:

- `simulate`
- `train`
- `evaluate`
- `graph`
- plot generation

It writes `demo_manifest.json` alongside the generated cohort, metrics, model
bundle, graph exports, and PNG plots.

`make-demo` accepts the same `--coefficient-mode` and `--coefficient-seed`
flags as `simulate`.

## Train

```bash
icg-cast train \
  --cohort outputs/demo/synthetic_icg_cohort.csv \
  --outdir outputs/demo \
  --seed 7
```

Writes:

- `model_metrics.csv`
- `permutation_importance.csv`
- `counterfactual_tests.csv`
- `biological_coherence.csv`
- `model_bundle.joblib`
- `model_card.md`
- `modality_auc.png` unless plots are disabled

The saved bundle contains the best `multiomics_plus_qAOP` baseline model and
the held-out split indices used during training.

## Evaluate

```bash
icg-cast evaluate \
  --cohort outputs/demo/synthetic_icg_cohort.csv \
  --model outputs/demo/model_bundle.joblib \
  --outdir outputs/demo
```

Writes evaluation metrics, calibration diagnostics, counterfactual tests,
biological-coherence summary, and a model card. By default the command evaluates
on the held-out split stored in the bundle.

## Graph

```bash
icg-cast graph --outdir outputs/demo
```

Writes:

- `icg_theory_graph.graphml`
- `icg_theory_graph_edges.json`

## Bench

```bash
icg-cast bench list
icg-cast bench info linear_lowhet
icg-cast bench run --cohort linear_lowhet --variant sign_constrained_augmented --seed 7
```

Benchmark commands run synthetic data-generating-process variants with known
mechanistic structure for recovery and intervention-conformity experiments.
