# ICg-Bench

PLAN.md reference: Milestone 5.5.

ICg-Bench is the package's public causal benchmark: a versioned set of
synthetic data-generating processes (DGPs) with full ground truth plus four
scored tasks. The DGPs are *synthetic on purpose* — that is the only setting
in biology where causal estimands can be evaluated exactly. Real-data
calibration is handled separately through the [data sources](data_sources.md)
and [calibration](calibration.md) layers and is out of scope for benchmark
scoring.

Source: [src/icg_cast/benchmark/](../src/icg_cast/benchmark/).

## DGP variants

Registered in [src/icg_cast/benchmark/dgp.py](../src/icg_cast/benchmark/dgp.py)
and listed via `icg-cast bench list`. Each variant has a stable SHA-256 hash
derived from its dataclass fields, recorded on every leaderboard row for
reproducibility.

| Variant | Description |
| --- | --- |
| `linear_lowhet` | Linear KCC→state coupling, low host heterogeneity, discrete archetypes, full multi-omics observability. The easy baseline. |
| `nonlinear_mixhost` | Non-linear coupling, continuous KCC mixtures, high host heterogeneity. Stresses latent recovery. |
| `partial_observability` | Non-linear coupling with random per-subject masking (30%) of transcriptomic and epigenomic modules. Stresses robustness to missing modalities. |
| `nonlinear_obs` | Linear KCC→state coupling with a non-linear, multiplicatively-interacting observation operator. Stresses stage-1 latent recovery while keeping stage-2 simple. |
| `misspecified_signs` | `linear_lowhet` base with one flipped sign in the `latent_risk` DGP relative to the structural prior. Falsification cohort for sign-constrained and intervention-augmented MB-CNet variants. |
| `misspecified_signs_v2` | Two simultaneous sign flips. Stress test for whether unconstrained recovery survives multiple prior errors. |

## Four scored tasks

Implemented in [src/icg_cast/benchmark/tasks.py](../src/icg_cast/benchmark/tasks.py).
Tasks are deliberately model-agnostic: `task_risk_prediction` only requires
`predict_proba`, while `task_latent_recovery` and
`task_intervention_conformity` additionally require `predict_bottleneck` /
`intervene` (i.e. an MB-CNet-shaped model).

### `task_risk_prediction`

Held-out discrimination plus calibration on a single variant. Returns
`auroc`, `auprc`, `brier`, `mean_proba`, `event_rate`.

### `task_latent_recovery`

Per-state R² between the model's predicted bottleneck and the *true* qAOP
state (which is known because the DGP wrote it). Returns one `r2__<state>`
entry per state plus `r2_mean` and `n_states`. This task is what makes the
benchmark *causal* rather than purely predictive: a model can hit a high
AUROC without recovering the latent state, and that asymmetry is exactly the
contribution ICg-Bench tries to make visible.

### `task_intervention_conformity`

For each `do_*` intervention in the registry, the model is forced through
the intervention via `intervene(unit, scale)`, and the mean change in
predicted risk is compared against the expected sign. The CLI's bench-run
reports three sub-metrics so prior-fragility is exposed:

- **`prior_conformity`** — fraction matching the structural-prior direction.
- **`dgp_conformity`** — fraction matching the true DGP direction (this
  differs from the prior in the `misspecified_signs*` variants).
- **`responsive_dgp_conformity`** — fraction matching the DGP direction
  *and* moving by at least `|Δ risk| ≥ responsive_threshold` (default
  `0.005`). This closes the loophole where a sign constraint can drive a
  coefficient to zero, registering a technically-correct sign without any
  intervention response.

### `task_cross_host_generalization`

Source vs. target AUROC under a host-susceptibility distribution shift.
Returns `auroc_source`, `auroc_target`, `transfer_gap = auroc_source -
auroc_target`. Re-fitting is forbidden by the task contract.

## Scoring and the leaderboard

Aggregation is implemented in
[src/icg_cast/benchmark/scoring.py](../src/icg_cast/benchmark/scoring.py)
and [src/icg_cast/benchmark/leaderboard.py](../src/icg_cast/benchmark/leaderboard.py).

Each `BenchmarkResult` is a single `(variant, model, package_version)` row.
`score_summary` emits five headline numbers plus a `composite` that is an
arithmetic mean of `(auroc, r2_mean, conformity_score)` whenever those are
finite. The composite is **explicitly not** the canonical metric and is
provided only for ranking convenience.

Leaderboard files are append-only CSV plus a full-history JSON
(`leaderboard.csv` / `leaderboard.json`). Every entry carries:

- `schema_version` (currently `"0.1"`),
- `submitted_at` (UTC ISO timestamp),
- `variant_name` + `variant_hash` (first 12 chars of the SHA-256 of the
  variant's dataclass fields),
- `model_name`, `package_version`,
- per-task summary scores,
- free-form `notes`.

This means a leaderboard entry can be re-run from its CSV row alone.

## CLI

```bash
icg-cast bench list                                                # registered variants and hashes
icg-cast bench info misspecified_signs                             # inspect one variant, prints flipped signs
icg-cast bench run --cohort linear_lowhet --variant v0_1 --seed 7  # one (cohort, variant, seed) experiment
icg-cast bench audit --cohort linear_lowhet --variant sign_constrained --seed 7
icg-cast bench sweep                                               # full 5×3×3 sweep -> outputs/bottleneck_v0_5/
icg-cast bench plots                                               # manuscript figures -> outputs/figures/
```

The current canonical sweep lives at
[outputs/bottleneck_v0_5/](../outputs/bottleneck_v0_5/) and the figures it
drives at [outputs/figures/](../outputs/figures/). They are reproducible via
the two CLI commands above.

## Example script

[examples/run_icg_bench.py](../examples/run_icg_bench.py) runs a tiny
end-to-end benchmark on three variants and writes a small leaderboard.

```bash
python examples/run_icg_bench.py
```

It uses `n = 200` subjects and `months = 24` so it finishes in well under a
minute on a laptop and never touches real data. For full reproducibility of
the manuscript-grade numbers run `icg-cast bench sweep` instead.

## Submission

To add a result to the leaderboard, run the benchmark on one or more variants
and write the resulting `LeaderboardEntry` objects via
`benchmark.leaderboard.append_entry(entry, outdir)`. Entries are intended to
be reviewed via PR so the model name, package version, variant hash, and
notes can all be inspected before merge.

The benchmark is **synthetic by construction**; entries should not be
interpreted as carcinogenicity classifications or as predictions about real
human cohorts. See
[docs/ethics_and_limitations.md](ethics_and_limitations.md).
