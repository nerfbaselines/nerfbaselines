# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import importlib
import inspect

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ext"))
project = 'NerfBaselines'
copyright = '2024, Jonas Kulhanek'
author = 'Jonas Kulhanek'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'sphinxext.opengraph',
    'sphinx_copybutton',
    'sphinx_click',
    'myst_parser',
    'sphinx_nerfbaselines',
    'sphinxcontrib.apidoc',
]

# Hack to fix commands for sphinx_click
import nerfbaselines.__main__, click
nerfbaselines_cli = nerfbaselines.__main__.main
ctx = click.Context(nerfbaselines_cli)
command_names = nerfbaselines_cli.list_commands(ctx)
nerfbaselines_cli.commands = {x: nerfbaselines_cli.get_command(ctx, x) for x in command_names}
try:
    nerfbaselines_cli._lazy_commands = {}
except AttributeError:
    pass

# Get current commit
commit = os.popen("git rev-parse --short HEAD").read().strip()
code_url = f"https://github.com/jkulhanek/nerfbaselines/blob/{commit}/nerfbaselines"

def linkcode_resolve(domain, info):
    # Non-linkable objects from the starter kit in the tutorial.
    if domain == "js" or info["module"] == "connect4":
        return

    assert domain == "py", "expected only Python objects"

    mod = importlib.import_module(info["module"])
    if "." in info["fullname"]:
        objname, attrname = info["fullname"].split(".")
        obj = getattr(mod, objname)
        try:
            # object is a method of a class
            obj = getattr(obj, attrname)
        except AttributeError:
            # object is an attribute of a class
            return None
    else:
        obj = getattr(mod, info["fullname"])

    try:
        file = inspect.getsourcefile(obj)
        lines = inspect.getsourcelines(obj)
    except TypeError:
        # e.g. object is a typing.Union
        return None
    import nerfbaselines
    file = os.path.relpath(file, os.path.dirname(os.path.abspath(nerfbaselines.__file__)))
    start, end = lines[1], lines[1] + len(lines[0]) - 1

    return f"{code_url}/{file}#L{start}-L{end}"

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
root_doc = "index"

# Add support for versions
html_context = {
  'current_version' : "1.0",
  'versions' : "latest,1.0,2.0",
  'base_path': '',
}
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",
        "sidebar/scroll-end.html",
        "versions.html",
    ]
}

import nerfbaselines
nerfbaselines_source = os.path.dirname(os.path.abspath(nerfbaselines.__file__))
apidoc_module_dir = nerfbaselines_source
apidoc_separate_modules = True
apidoc_template_dir = "_templates"
apidoc_excluded_paths = [
    "methods/*.py",
    "backends/*.py",
    "datasets/*_spec.py",
    "viewer/*.py",
    "cli/*.py",
    "cli.py",
    "web/*.py",
]
apidoc_extra_args = ["--no-toc", "--remove-old"]

#
# Generate files if not present
#
local_path = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(local_path, "cli.rst")):
    with open(os.path.join(local_path, "cli.rst"), "w", encoding="utf8") as f:
        f.write(""".. click:: nerfbaselines.__main__:main
     :prog: nerfbaselines
     :nested: full
""")
# Detect if nerfbaselines has methods and datasets for ancient versions of NB
has_methods = True
has_datasets = True
try:
    import nerfbaselines.registry
    if not hasattr(nerfbaselines.registry.get("gaussian-splatting"), "metadata"):
        has_methods = False
except Exception:
    pass
try:
    import nerfbaselines.datasets.mipnerf360_spec
except ImportError:
    has_datasets = False
if not os.path.exists(os.path.join(local_path, "index.md")):
    with open(os.path.join(local_path, "index.md"), "w", encoding="utf8") as f:
        eol = "\n"
        f.write(f"""# NerfBaselines Documentation
```{{toctree}}
:maxdepth: 2
:caption: Reference
{('Methods <methods>' + eol) if has_methods else ''}{('Datasets <datasets>' + eol) if has_datasets else ''}CLI <cli>
API <api/nerfbaselines>
```
""")
if has_methods and not os.path.exists(os.path.join(local_path, "methods.md")):
    with open(os.path.join(local_path, "methods.md"), "w", encoding="utf8") as f:
        f.write("""# Methods
```{nerfbaselines}
:names-regex: ^methods/[^:]*$
```
""")
if has_datasets and not os.path.exists(os.path.join(local_path, "datasets.md")):
    with open(os.path.join(local_path, "datasets.md"), "w", encoding="utf8") as f:
        f.write("""# Datasets
```{nerfbaselines}
:names-regex: ^datasets/.*$
```
""")


# Add _static/styles.css and _static/scripts.js to the HTML context
html_static_path = ['_static']
html_css_files = ['styles.css']
html_theme = 'furo'#'sphinx_rtd_theme'  # furo
html_logo = "_static/logo.png"
html_theme_options = {
    "top_of_page_buttons": [],
    "footer_icons": [{
        "name": "GitHub",
        "url": "https://github.com/jkulhanek/nerfbaselines",
        "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
        """,
        "class": "",
    }, {
        "name": "Website",
        "url": "https://jkulhanek.com/nerfbaselines",
        "html": """
            <svg viewBox="2 2 21 21" fill="none"  stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path stroke="none" d="M0 0h24v24H0z" fill="none"/><path d="M3 12a9 9 0 1 0 18 0a9 9 0 0 0 -18 0" /><path d="M3.6 9h16.8" /><path d="M3.6 15h16.8" /><path d="M11.5 3a17 17 0 0 0 0 18" /><path d="M12.5 3a17 17 0 0 1 0 18" />
            </svg>
        """,
        "class": "",
    }],
}

# Override furo sidebar
def _html_page_context(app, pagename, templatename, context, doctree):
    del app, pagename, templatename, doctree    
    from furo.navigation import get_navigation_tree
    toctree = context["toctree"]
    toctree_html = toctree(
        collapse=False,
        titles_only=True,
        maxdepth=2,
        includehidden=True,
    )
    context["furo_navigation_tree"] = get_navigation_tree(toctree_html)


def setup(app):
    app.connect("html-page-context", _html_page_context, 1000)
