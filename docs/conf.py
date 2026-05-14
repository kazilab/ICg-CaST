"""Sphinx configuration for the ICg-CaST documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from icg_cast._branding import PROJECT_LONG_NAME, PROJECT_NAME, PROJECT_TAGLINE, VERSION  # noqa: E402

project = PROJECT_NAME
author = "ICg-CaST contributors"
copyright = "2026, ICg-CaST contributors"
release = VERSION
version = VERSION

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
]

templates_path = ["_templates"]

html_theme = "sphinx_rtd_theme"
html_title = f"{PROJECT_NAME} documentation"
html_short_title = PROJECT_NAME
html_static_path = ["_static"] if (Path(__file__).parent / "_static").exists() else []

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "linkify",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3
myst_substitutions = {
    "project_name": PROJECT_NAME,
    "project_long_name": PROJECT_LONG_NAME,
    "project_tagline": PROJECT_TAGLINE,
    "version": VERSION,
}

autosummary_generate = True
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

suppress_warnings = [
    "myst.xref_missing",
]
