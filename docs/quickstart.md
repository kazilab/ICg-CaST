# Quickstart for Domain Scientists

ICg-CaST has four moving parts. You can use the package without reading the
full theory notes first if you keep this path in mind:

1. **KCC exposure coordinates.** Each chemical profile is represented by ten
   Key Characteristics of Carcinogens coordinates, scaled from 0 to 1.
2. **qAOP state trajectories.** The simulator maps KCC activity, dose, and host
   susceptibility into monthly latent states: DNA adducts, ROS, inflammation,
   epigenetic age, proliferation, mutation rate, clone fraction, driver-count
   proxy, immune clearance, and latent risk.
3. **MB-CNet bottleneck.** The mechanism-bottleneck model first reconstructs
   the qAOP state vector from omics-like features, then predicts transition
   risk from that state vector.
4. **Do-interventions.** Intervention checks perturb a named qAOP state, such
   as `do_ROS_inflammation_blockade`, and verify whether predicted risk moves
   in the expected direction.

## Browser Path

The Streamlit app wraps the common workflows without requiring command-line
flags for every step:

```bash
python -m pip install -e ".[app]"
streamlit run streamlit_app.py
```

The app writes runs under `outputs/streamlit/<run-name>` and can simulate
cohorts, train/evaluate models, export the theory graph, and run benchmark
experiments.

## Command-Line Path

```bash
icg-cast make-demo --n 120 --months 72 --seed 7 --outdir outputs/demo
icg-cast train --cohort outputs/demo/synthetic_icg_cohort.csv --outdir outputs/demo
icg-cast evaluate \
  --cohort outputs/demo/synthetic_icg_cohort.csv \
  --model outputs/demo/model_bundle.joblib \
  --outdir outputs/demo
```

## Python Path

```python
from icg_cast import SimConfig, simulate_cohort, train_baselines

cfg = SimConfig(n=120, months=72, seed=7)
cohort, trajectories = simulate_cohort(cfg)
metrics, importance, counterfactual, bundle = train_baselines(cohort, seed=7)
```

Use [simulation](simulation.md) for simulator details,
[bottleneck](bottleneck.md) for MB-CNet, and [benchmark](benchmark.md) for
ICg-Bench variants and leaderboard outputs.
