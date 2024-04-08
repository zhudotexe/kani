# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import pathlib
import sys

import matplotlib.font_manager

# ensure kani is available in path
sys.path.append("..")

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "kani"
copyright = "2023-present, Andrew Zhu, Liam Dugan, and Alyssa Hwang"
author = "Andrew Zhu, Liam Dugan, and Alyssa Hwang"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx.ext.viewcode",
    "sphinxext.opengraph",  # https://sphinxext-opengraph.readthedocs.io/en/latest/
    "sphinx_inline_tabs",  # https://sphinx-inline-tabs.readthedocs.io/en/latest/usage.html
    "sphinx_copybutton",  # https://sphinx-copybutton.readthedocs.io/en/latest/
    "sphinxemoji.sphinxemoji",  # https://sphinxemojicodes.readthedocs.io/en/stable/
    "sphinx_sitemap",  # https://sphinx-sitemap.readthedocs.io/en/latest/getting-started.html
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

maximum_signature_line_length = 120

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_extra_path = ["_extra"]
html_logo = "_static/kani-logo@512.png"
html_favicon = "_extra/favicon.ico"
html_baseurl = "https://kani.readthedocs.io/en/latest/"

nitpicky = True
nitpick_ignore_regex = [
    (r"py:class", r"aiohttp\..*"),  # aiohttp intersphinx is borked
    (r"py:class", r"torch\..*"),  # idk if torch has intersphinx
    (r"py:class", r"openai\..*"),  # openai does not use sphinx for docs
    (r"py:class", r"anthropic\..*"),  # anthropic does not use sphinx for docs
    (r"py:class", r"asyncio\.\w+\..*"),  # asyncio submodule intersphinx is borked
    (r"py:class", r"kani\..*\.[\w_0-9]*T"),  # ignore generics and other typevars
]

# sphinx.ext.autodoc
autoclass_content = "both"
autodoc_member_order = "bysource"

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# sphinxext.opengraph
# load correct fonts for card generation
ogp_fonts = {
    "Noto Sans Japanese": "NotoSansJP-Regular.ttf",
    "Roboto": "Roboto-Regular.ttf",
    "Noto Sans Emoji": "NotoColorEmoji-Regular.ttf",
}
for fontname, fontpath in ogp_fonts.items():
    path_font = pathlib.Path(__file__).parent / "_static/fonts" / fontpath
    font = matplotlib.font_manager.FontEntry(fname=str(path_font), name=fontname)
    matplotlib.font_manager.fontManager.ttflist.append(font)

ogp_social_cards = {
    "font": [*ogp_fonts.keys(), "sans-serif"],
}

# sphinx_copybutton
copybutton_exclude = ".linenos, .gp, .go"
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True
copybutton_copy_empty_lines = False

# sphinxemoji.sphinxemoji
sphinxemoji_style = "twemoji"

# sphinx_sitemap
sitemap_url_scheme = "{link}"
