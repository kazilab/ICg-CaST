# Simulation

The synthetic cohort generator lives in `src/icg_cast/simulator.py` and is
configured through `SimConfig`.

## Data-Generating Process

For each synthetic cohort, ICg-CaST optionally draws one coefficient-prior
realization when `SimConfig.coefficient_mode="prior_sample"`. Then for each
synthetic sample it:

1. Samples a chemical archetype from the default or user-provided archetype
   prior.
2. Draws a noisy ten-dimensional KCC vector around the archetype profile.
3. Samples a bounded log-normal dose.
4. Samples host susceptibility factors.
5. Simulates monthly qAOP-like state dynamics for `months` time steps.
6. Summarizes each state by final value and time-normalized area under the
   curve.
7. Generates transcriptomic, epigenomic, and mutational-signature features from
   the states, KCCs, and archetype.
8. Samples the synthetic future event label from the final latent risk.

The cohort table includes generated features, endpoint columns, the
`coefficient_seed` metadata column, and a `high_risk_transition_state` label
derived from the latent-risk distribution. Feature-set builders exclude
endpoint-derived columns before training.

## Reproducibility

Every cohort is controlled by `SimConfig.seed`. Coefficient-prior draws are
controlled by `SimConfig.coefficient_seed`, which defaults to `seed` when
omitted. Reusing the same configuration and package version should produce the
same cohort table.

Example:

```python
from icg_cast import SimConfig, simulate_cohort

cohort, trajectories = simulate_cohort(SimConfig(n=120, months=72, seed=7))

uncertain, _ = simulate_cohort(
    SimConfig(
        n=120,
        months=72,
        seed=7,
        coefficient_mode="prior_sample",
        coefficient_seed=42,
    )
)
```

## Main Outputs

`icg-cast simulate` writes:

- `synthetic_icg_cohort.csv`
- `simulation_metadata.json`
- `example_state_trajectories.png` unless plots are disabled

Important cohort field groups are documented in
`materials/data_dictionary.csv`.

## Assumptions

The recurrence equations, weights, archetype KCC vectors, and intervention
expectations are simulation assumptions. They are designed to be inspectable and
configurable, not to encode validated biological truth.
