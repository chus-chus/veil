import os
import sys
from datetime import datetime

# -- Path setup --------------------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT_DIR)

# -- Project information -----------------------------------------------------
project = "Veil"
author = "Chus Antonanzas"
current_year = datetime.now().year
copyright = f"{current_year}, {author}"

try:
    from veil import __version__ as version  # type: ignore
except Exception:
    version = "0.1.0"
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "linkify",
    "substitution",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Use Sphinx native support for type hints
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Avoid label collisions when multiple pages share the same heading
autosectionlabel_prefix_document = True
autosummary_generate = True
autodoc_mock_imports = [
    "fastapi",
    "pydantic",
    "uvicorn",
    "spacy",
    "torch",
    "transformers",
    "gliner",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
html_logo = None
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
}


