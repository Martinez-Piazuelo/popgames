# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as _version

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src"))
)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PopGames"
copyright = "2025, Juan Martinez-Piazuelo"
author = "Juan Martinez-Piazuelo"
raw = _version("popgames")
version = raw.split(".dev", 1)[0]  # e.g. 1.0.0

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_autodoc_typehints",
]

autosummary_generate = True
autodoc_typehints = "description"
autoclass_content = "both"  # shows class docstring AND __init__ docstring

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
}

templates_path = ["_templates"]
exclude_patterns = []
autodoc_mock_imports = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_favicon = "_static/favicon.png"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme = "furo"
html_theme_options = {
    "source_repository": "https://github.com/Martinez-Piazuelo/popgames",
    "source_branch": "main",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Martinez-Piazuelo/popgames",
            "html": """
                <svg stroke="currentColor" fill="currentColor" stroke-width="0"
                     viewBox="0 0 16 16">
                    <path fill-rule="evenodd"
                    d="M8 0C3.58 0 0 3.58 0 8a8 8 0 005.47 7.59c.4.07.55-.17.55-.38
                    0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13
                    -.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87
                    2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95
                    0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21
                    2.2.82a7.65 7.65 0 012-.27c.68 0 1.36.09 2 .27
                    1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08
                    2.12.51.56.82 1.27.82 2.15
                    0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54
                    1.48 0 1.07-.01 1.93-.01 2.2
                    0 .21.15.46.55.38A8.01 8.01 0
                    0016 8c0-4.42-3.58-8-8-8z">
                    </path>
                </svg>
            """,
            "class": "",
        },
    ],
}
html_title = f"{project} v{version}"
