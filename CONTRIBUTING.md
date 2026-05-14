# Contributing

ICg-CaST is a synthetic research package. Contributions should keep the core
package reproducible, auditable, and clear about its limits.

## Development Setup

```bash
python -m pip install -e ".[dev]"
```

## Checks

Run before opening a pull request:

```bash
ruff check .
pytest -q
python -m build
```

## Coefficient Changes

Coefficient changes are scientific changes, not formatting changes. A pull
request that changes `materials/coefficient_cards.yaml`, calibration logic, or
any simulator equation must include:

- An updated coefficient card with `default_value`, `evidence_level`, `source`,
  `notes`, `last_reviewed`, `prior_distribution`, and `prior_params`.
- A sensitivity diff from `icg-cast coeffs sensitivity`. Include the changed
  rows from `outputs/audit/coefficient_sensitivity.csv` and any new
  `LOAD_BEARING` or `CANDIDATE_FOR_REMOVAL` rows from
  `outputs/audit/coefficient_flags.csv`.
- An expert-review entry in `materials/coefficient_review.csv` for every
  load-bearing coefficient. `reviewed_by` must name the reviewer or review
  group, `decision` must be one of `approved`, `rejected`,
  `draft_pending_expert_signoff`, or `needs_revision`, and dissent or
  uncertainty must be recorded in `dissent_notes`.
- If calibration changed a coefficient, attach the generated
  `calibration_provenance.json` and the `calibrated_coefficients.yaml` diff.

Run the review gate before requesting merge:

```bash
python -m icg_cast.audit.validate_reviews
```

Load-bearing coefficients without expert sign-off remain in draft status even
when tests pass.

## Data Rules

- Do not commit controlled-access, patient-level, or identifying data.
- Do not add network downloads to tests or default examples.
- Real-data adapters must accept local files and record provenance.
- Add tiny synthetic or mock fixtures for tests.

## Scientific Claims

Do not describe synthetic benchmark performance as clinical, regulatory, or
environmental safety validation. New docs and examples should repeat this
boundary when they could be misread.
