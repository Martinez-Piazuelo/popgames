# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from importlib.metadata import version as _version

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'popgames'
copyright = '2025, Juan Martinez-Piazuelo'
author = 'Juan Martinez-Piazuelo'
raw = _version("popgames")
release = raw.split("+", 1)[0]          # e.g. 1.0.0.dev1
version = release.split(".dev", 1)[0]   # e.g. 1.0.0

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
]

autodoc_typehints = 'description'
autoclass_content = 'both'  # shows class docstring AND __init__ docstring

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'inherited-members': True,
}

templates_path = ['_templates']
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
