# Streamlit Browser App

The Streamlit app wraps the ICg-CaST Python workflows in a local browser UI.
It can run synthetic simulations, the full demo workflow, baseline model
training and evaluation, ICg-Bench experiments, and output inspection.

## Install

```bash
python -m pip install -e ".[app]"
```

For Streamlit Community Cloud or other `requirements.txt` based deployments,
use the repository-level requirements file:

```bash
python -m pip install -r requirements.txt
```

For development with tests and linting:

```bash
python -m pip install -e ".[dev,app]"
```

## Run

```bash
streamlit run streamlit_app.py
```

or:

```bash
make app
```

The app writes run outputs under `outputs/streamlit/<run-name>`. Those outputs
use the same file formats as the CLI: CSV tables, PNG plots, JSON metadata,
GraphML graph exports, and `model_bundle.joblib` model bundles.
