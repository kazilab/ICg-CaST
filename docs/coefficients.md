# Coefficient registry

PLAN.md reference: section 25 (Coefficient credibility roadmap),
Milestones 8 and 9.

Every numeric coefficient that drives the qAOP dynamics, the `latent_risk`
equation, the chemical archetype tables, and the host susceptibility
distributions is declared in
[materials/coefficient_cards.yaml](../materials/coefficient_cards.yaml)
and loaded through [src/icg_cast/coefficients/](../src/icg_cast/coefficients/).

Inline numeric literals in those code sites are forbidden going forward
(PLAN.md §17.5). A pytest spot-check in
[tests/test_coefficient_registry.py](../tests/test_coefficient_registry.py)
catches the most obvious slips.

## Card schema

```yaml
- name: dynamics.dna_adducts.decay
  default_value: 0.68
  units: "month^-1"
  evidence_level: E4
  prior_distribution: auto
  prior_params: {}
  source: "starter kit (PLAN.md sections 6 and 7)"
  notes: "first-order monthly persistence of the DNA-adduct burden between exposures"
  last_reviewed: "2026-05-13"
```

Fields:

| Field            | Required | Meaning |
|------------------|:--------:|---------|
| `name`           | yes | Dotted namespace; must be unique. |
| `default_value`  | yes | Scalar (float/int), vector (list of numbers), or string label. |
| `units`          | no  | Free-text units description. |
| `evidence_level` | no  | One of `E1`..`E5`. Default `E5` (no source). |
| `prior_distribution` | no | One of `auto`, `fixed`, `normal`, `lognormal`, `signed_lognormal`, `logit_normal`, or `dirichlet`. Default `auto`. |
| `prior_params` | no | Optional sampler overrides such as `sigma`, `low`, `high`, or `concentration`. |
| `source`         | no  | DOI, dataset name, or `"starter kit"`. |
| `notes`          | no  | Free-text explanation. |
| `last_reviewed`  | no  | ISO date string. |

Top-level `defaults` apply to every card unless the card overrides them.

## Evidence levels

| Level | Meaning |
|-------|---------|
| `E1` | Published quantitative literature value. |
| `E2` | Published qualitative direction or magnitude. |
| `E3` | AOP-Wiki / AOP-DB / KER weight-of-evidence. |
| `E4` | Expert estimate, plausible biological order of magnitude. |
| `E5` | No source ("hand-tuned to produce interesting cohorts"). |

A coefficient is flagged "load-bearing" when Milestone 12's sensitivity
audit shows >20% effect on any downstream metric; load-bearing
coefficients must reach at least `E2`.

## Coefficient uncertainty

The point value in `default_value` is the center of the card's prior. In
`auto` mode, strings, seeds, hard minima, hard maxima, and minimum counts
stay fixed; bounded scalar/vector values use logit-normal priors; positive
scalars use log-normal priors; signed scalars use signed log-normal priors;
and probability vectors use Dirichlet priors. Evidence level controls the
default spread: `E1` is tightest and `E5` is widest.

Use point mode for exact reproducibility against previous demos:

```python
from icg_cast import SimConfig, simulate_cohort

cohort, _ = simulate_cohort(SimConfig(coefficient_mode="point"))
```

Use prior-sample mode to draw one seedable coefficient realization for the
whole cohort:

```python
cohort, _ = simulate_cohort(
    SimConfig(coefficient_mode="prior_sample", coefficient_seed=42)
)
```

The resulting cohort includes `coefficient_seed`; point mode writes `-1`.

## Python API

```python
from icg_cast.coefficients import registry

r = registry()
decay = r.get("dynamics.dna_adducts.decay")          # 0.68
kcc   = r.get_vector("archetypes.pah_tobacco_like.kcc")
sig   = r.get_str("archetypes.pah_tobacco_like.signature")

# Audit: find coefficients with no source
unsourced = r.filter(evidence_level="E5")

# Loading a custom YAML (useful for tests or alternate priors)
from icg_cast.coefficients import load_registry
r2 = load_registry("path/to/custom.yaml")

# Seedable prior draw
from icg_cast.coefficients import sampled_registry
r3 = sampled_registry(seed=42)
```

The default registry is cached at module level; the same object is
returned on every `registry()` call. Set `ICG_CAST_COEFFICIENTS_PATH` to
override the file location for one process.

## CLI

```bash
# How many cards at each evidence level?
icg-cast coeffs audit

# List every unsourced (E5) coefficient
icg-cast coeffs list --evidence E5

# Inspect a namespace
icg-cast coeffs list --prefix dynamics.latent_risk

# Machine-readable
icg-cast coeffs list --prefix archetypes --json

# Draw one coefficient-prior realization for a simulated cohort
icg-cast simulate --coefficient-mode prior_sample --coefficient-seed 42

# Run the full demo under coefficient uncertainty
icg-cast make-demo --coefficient-mode prior_sample --coefficient-seed 42
```

`coeffs audit` is the smallest useful command to run on any branch that
touches the registry. It surfaces the registry's evidence-level
distribution and is intended to be wired into review prompts.

## Coverage status (Milestones 8-9 complete)

| Site | Status |
|------|--------|
| `simulator.py` (qAOP dynamics, susceptibility, cohort sampling) | covered |
| `constants.py` (`ARCHETYPE_KCC`, `ARCHETYPE_SIGNATURE`) | covered |
| `omics.py` (transcriptomic / epigenomic module weights, signature mixing, total-mutation Poisson rate) | covered |
| `signatures.py` (toy SBS recipe parameters: background gamma, per-context boosts, lower clip) | covered |
| coefficient priors (`prior_distribution`, `prior_params`, seedable draws) | covered |

Current registry breakdown (run `icg-cast coeffs audit` to refresh):

| Namespace | Cards | Notes |
|---|---:|---|
| `dynamics.*` | 74 | qAOP recurrence + `latent_risk` |
| `susceptibility.*` | 20 | host distribution parameters |
| `archetypes.*` | 18 | 8 KCC vectors + 8 signature labels + prior + noise |
| `cohort.*` | 1 | high-risk quantile |
| `omics.transcript.*` | 51 | 18 modules × ~3 inputs + measurement noise |
| `omics.epi.*` | 24 | 8 modules × ~3 inputs + measurement noise |
| `omics.signature_mix.*` | 10 | aging baseline + primary blend + oxidative blend |
| `omics.mut_total.*` | 6 | Poisson rate intercept + 4 slope terms + min clip |
| `signatures.*` | 14 | background gamma + per-signature boost recipe |
| **total** | **218** | |

Of these, **9 are `E4`** (archetype KCC vectors and the mutation-rate
scale, where there is at least a literature order of magnitude) and the
remaining 209 are `E5`. The `E5` count is the canonical pre-Phase-3
baseline. Calibration updates from Milestone 10 are expected to upgrade
the COSMIC-tied signature, ToxCast-tied KCC, AOP-Wiki-tied coupling, and
LINCS-tied transcriptomic module coefficients out of `E5`.

## How to edit a coefficient

1. Edit
   [materials/coefficient_cards.yaml](../materials/coefficient_cards.yaml).
   Update `default_value`, `evidence_level`, `prior_distribution`,
   `prior_params`, `source`, `notes`, and `last_reviewed` together.
2. Run `pytest tests/test_coefficient_registry.py -q` to verify the
   schema still parses.
3. Run the full suite (`pytest -q`) to confirm determinism-sensitive
   tests still pass — a coefficient change is expected to change cohort
   numerics, so deterministic tests may need their reference values
   updated in the same PR.
4. Run `icg-cast coeffs audit` and paste the table into the PR
   description.

## Roadmap

Milestone 8 covers traceability, and Milestone 9 adds seedable
coefficient uncertainty. The next phases are:
- **Milestone 10** — calibration adapters (COSMIC, LINCS, ToxCast,
  AOP-Wiki) gain ownership of specific coefficient groups and can
  upgrade evidence levels.
- **Milestone 11** — split `starter_kit_latent_risk` into
  `reference_risk_oracle()` and `biological_risk_equation()` to remove
  the circularity between MB-CNet's labelling oracle and its conformity
  metric.
- **Milestone 12** — per-coefficient sensitivity audit, with `<1%` and
  `>20%` auto-flags.
- **Milestone 13** — domain-expert review process via
  `materials/coefficient_review.csv`.
