PYTHON ?= python

.PHONY: install lint test build docs demo app check clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check .

test:
	pytest -q

build:
	$(PYTHON) -m build

docs:
	sphinx-build -b html docs docs/_build/html

demo:
	icg-cast make-demo --outdir outputs/demo

app:
	$(PYTHON) -m streamlit run streamlit_app.py

check: lint test build

clean:
	rm -rf build dist src/icg_cast.egg-info .pytest_cache .ruff_cache
	rm -rf docs/_build
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
