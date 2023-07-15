# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "kani"
copyright = "2023, Andrew Zhu and Liam Dugan"
author = "Andrew Zhu and Liam Dugan"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxext.opengraph",  # https://sphinxext-opengraph.readthedocs.io/en/latest/
    "sphinx_inline_tabs",  # https://sphinx-inline-tabs.readthedocs.io/en/latest/usage.html
    "sphinx_copybutton",  # https://sphinx-copybutton.readthedocs.io/en/latest/
    "sphinxemoji.sphinxemoji",  # https://sphinxemojicodes.readthedocs.io/en/stable/
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

# spinxext.opengraph
ogp_social_cards = {
    "enable": False,  # the bundled Roboto font does not support kanji
}

# sphinx_copybutton
copybutton_exclude = ".linenos, .gp, .go"
