"""Sphinx configuration for ReMoDe docs."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "ReMoDe"
author = "ReMoDe contributors"

try:
    release = pkg_version("remode")
except PackageNotFoundError:
    release = "0+unknown"
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_member_order = "bysource"
autodoc_mock_imports = ["numpy", "pandas", "scipy", "matplotlib"]
suppress_warnings = ["myst.header"]

myst_enable_extensions = ["colon_fence", "deflist"]

nb_execution_mode = "off"
nb_execution_show_tb = False

html_theme = "pydata_sphinx_theme"
html_title = "ReMoDe documentation"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/sodascience/remode",
    "show_toc_level": 2,
    "navigation_with_keys": True,
}
