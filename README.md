# ICg-CaST
<!-- PyPI version badge -->
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://icg-cast.streamlit.app)
[![Documentation Status](https://readthedocs.org/projects/icg-cast/badge/?version=latest)](https://icg-cast.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://img.shields.io/pypi/v/icg-cast.svg)](https://pypi.org/project/icg-cast/)
[![Documentation Status](https://readthedocs.org/projects/icg-cast/badge/?version=latest)](https://icg-cast.readthedocs.io/en/latest/?badge=latest)

<!-- 
[![bioRxiv](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.03.22.713456-b31b1b.svg)](https://doi.org/10.64898/2026.03.22.713456)
-->
<!-- PyPI version badge -->
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/ICg--CaST-181717?logo=github&logoColor=white)](https://github.com/kazilab/ICg-CaST)
[![@KaziLab.se](https://img.shields.io/website?url=https://www.kazilab.se/)](https://www.kazilab.se/)
<!-- PyPI version badge -->

**ICg-CaST** is a research package for synthetic integrated carcinogenomics experiments:
causal-state simulation, mechanism-bottleneck models, and ICg-Bench benchmark
variants with known data-generating processes.

The distribution name is `icg-cast`, the Python import is `icg_cast`, and the
command-line entry point is `icg-cast`.

## Current Status

This repository is in an early cleanup and migration stage. The new package
currently focuses on:

- Mechanism-Bottleneck Causal Networks in `src/icg_cast/bottleneck.py`.
- Time-to-event/RMST helpers in `src/icg_cast/survival.py`.
- Coefficient registry and seedable coefficient-prior uncertainty in
  `src/icg_cast/coefficients/`.
- ICg-Bench DGP variants, task scoring, and leaderboard helpers in
  `src/icg_cast/benchmark/`.

The starter kit has been migrated into the active package under
`src/icg_cast`. Active docs, tests, and demos use that package layout.

## Install

```bash
python -m pip install -e ".[dev]"
```

## Quickstart

Mental model for domain scientists:

1. Choose or simulate a chemical exposure profile as a 10-dimensional KCC vector.
2. The simulator turns KCC activity into time-varying qAOP state trajectories
   such as DNA adducts, ROS, inflammation, proliferation, and clone fraction.
3. MB-CNet learns a bottleneck that predicts those qAOP states from omics-like
   observations before predicting future transition risk.
4. `do_*` interventions perturb bottleneck states and check whether predicted
   risk moves in the biologically expected direction.

Prefer a browser workflow before using the CLI? Install the optional app extra
and run the local Streamlit app:

```bash
python -m pip install -e ".[app]"
streamlit run streamlit_app.py
```

Generate a synthetic cohort:

```bash
icg-cast simulate --n 120 --months 12 --seed 7 --outdir outputs/demo
```

Generate a cohort under one coefficient-prior draw:

```bash
icg-cast simulate \
  --n 120 \
  --months 12 \
  --seed 7 \
  --coefficient-mode prior_sample \
  --coefficient-seed 42 \
  --outdir outputs/demo_uncertainty
```

Run the full reproducible package demo in one command:

```bash
icg-cast make-demo --n 120 --months 72 --seed 7 --outdir outputs/demo
```

Train baseline models:

```bash
icg-cast train \
  --cohort outputs/demo/synthetic_icg_cohort.csv \
  --outdir outputs/demo \
  --seed 7
```

Evaluate the saved model bundle:

```bash
icg-cast evaluate \
  --cohort outputs/demo/synthetic_icg_cohort.csv \
  --model outputs/demo/model_bundle.joblib \
  --outdir outputs/demo
```

Export the theory graph:

```bash
icg-cast graph --outdir outputs/demo
```

Use the Python API:

```python
from icg_cast import SimConfig, simulate_cohort, train_baselines

cohort, trajectories = simulate_cohort(SimConfig(n=120, months=72, seed=7))
metrics, importance, counterfactual, bundle = train_baselines(cohort, seed=7)
```

List registered benchmark variants:

```bash
icg-cast bench list
```

Inspect a variant:

```bash
icg-cast bench info misspecified_signs
```

Run one small benchmark experiment:

```bash
icg-cast bench run \
  --cohort linear_lowhet \
  --variant sign_constrained_augmented \
  --seed 7 \
  --n 400 \
  --months 36
```

For `requirements.txt` based Streamlit deployments, install with
`python -m pip install -r requirements.txt`.

The Streamlit app writes results under `outputs/streamlit/<run-name>` and
wraps simulation, demo, training/evaluation, graph export, and benchmark
workflows.

## Documentation and Materials

Project documentation starts at `docs/index.md`; the shortest conceptual
route is `docs/quickstart.md`. Field and intervention
dictionaries live under `materials/`:

Build the documentation locally:

```bash
python -m pip install -e ".[docs]"
make docs
```

- `materials/data_dictionary.csv`
- `materials/intervention_dictionary.csv`
- `materials/provenance_template.json`
- `materials/calibration_provenance.schema.json`

Optional real-data adapters live under `icg_cast.data_sources`. They accept
local files only and record provenance; they do not download public datasets.

## Citation and License

Citation metadata is in `CITATION.cff`. The package metadata and `LICENSE` file
declare Apache-2.0 for the source code unless a maintainer changes that before
release.

## Scope and Limitations

This is a synthetic theory-development and benchmarking scaffold. It is not a
clinical diagnostic, individual-risk model, chemical safety classifier, medical
device, or substitute for experimental toxicology, epidemiology, or regulatory
review.

Synthetic benchmark performance does not imply real-world biological validity.
Real-data adapters, if added, must remain optional and provenance-tracked.
