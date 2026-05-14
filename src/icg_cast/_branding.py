"""Central branding constants for ICg-CaST.

Single source of truth for the project's identity. Edit values here to
rebrand the package globally. ``pyproject.toml`` reads ``VERSION`` via
setuptools' dynamic-attribute mechanism, ``__init__`` re-exports the
constants as ``__project_name__`` / ``__version__``, and the CLI, I/O
helpers, docs, and tests reference these names instead of hardcoding
their own strings.
"""

from __future__ import annotations

# ---- core identity -------------------------------------------------------

PROJECT_NAME: str = "ICg-CaST"
PROJECT_LONG_NAME: str = "Integrated Carcinogenomics Causal State Theory"
PROJECT_TAGLINE: str = (
    "Causal-state simulation and mechanism-bottleneck benchmarking "
    "for integrated carcinogenomics."
)
PROJECT_ABSTRACT: str = (
    "Synthetic causal-state simulation and mechanism-bottleneck "
    "benchmarking for integrated carcinogenomics theory development."
)

# Distribution / import / console-script names. These three SHOULD agree
# with pyproject.toml; keep them in lock-step if any rename ever happens.
DIST_NAME: str = "icg-cast"           # PyPI / pip install name
IMPORT_NAME: str = "icg_cast"         # `import icg_cast`
CLI_NAME: str = "icg-cast"            # console_scripts entry point

VERSION: str = "0.1.0.dev0"

# ---- secondary identities ------------------------------------------------

BENCH_NAME: str = "ICg-Bench"

# ---- standardised output filenames --------------------------------------
# Used by io.write_cohort, graph.export_graph, CLI helpers, and tests so
# that filenames track the brand from one place.

COHORT_FILENAME: str = "synthetic_icg_cohort.csv"
GRAPH_GRAPHML_FILENAME: str = "icg_theory_graph.graphml"
GRAPH_EDGES_JSON_FILENAME: str = "icg_theory_graph_edges.json"

__all__ = [
    "PROJECT_NAME",
    "PROJECT_LONG_NAME",
    "PROJECT_TAGLINE",
    "PROJECT_ABSTRACT",
    "DIST_NAME",
    "IMPORT_NAME",
    "CLI_NAME",
    "VERSION",
    "BENCH_NAME",
    "COHORT_FILENAME",
    "GRAPH_GRAPHML_FILENAME",
    "GRAPH_EDGES_JSON_FILENAME",
]
