# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import re
import fnmatch
import argparse
from pathlib import Path
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
    #'myst_parser',
    'sphinx_nerfbaselines',
    'myst_nb',
    'sphinx_design',
    # 'sphinxcontrib.apidoc',
]

nb_execution_mode = "off"
nb_merge_streams = True

# Hack to fix commands for sphinx_click
import nerfbaselines.__main__, click
try:
    from nerfbaselines._constants import WEBPAGE_URL, CODE_REPOSITORY
except ImportError:
    # Fill for older versions
    WEBPAGE_URL = "https://nerfbaselines.github.io"
    CODE_REPOSITORY = "github.com/nerfbaselines/nerfbaselines"

# Hack for older versions of NerfBaselines requiring viser
import importlib
from unittest.mock import MagicMock
for module in ["viser", "viser.transforms", "viser.theme", "splines", "splines.quaternion", "mediapy"]:
    try:
        importlib.import_module(module)
    except ImportError:
        # Add a fake module to avoid import errors
        sys.modules[module] = MagicMock()
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
code_url = f"https://{CODE_REPOSITORY}/blob/{commit}/nerfbaselines"

def linkcode_resolve(domain, info):
    try:
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
    except Exception:
        import traceback
        traceback.print_exc()
        return None

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
root_doc = "index"

# Add support for versions
html_context = {
  'current_version' : "1.0",
  'versions' : "1.1,1.0,2.0",
  'version_names' : "latest,1.0,2.0",
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
    ]
}

# import nerfbaselines
# nerfbaselines_source = os.path.dirname(os.path.abspath(nerfbaselines.__file__))
# apidoc_module_dir = nerfbaselines_source
# apidoc_separate_modules = True
apidoc_template_dir = "_templates"
# apidoc_module_first = True
apidoc_excluded_paths = [
    "methods/*.py",
    "datasets/*_spec.py",
    "cli/*.py",
    "cli.py",
]
apidoc_extra_args = ["--remove-old"]

# Generate API documentation
out_path = os.path.join(os.path.dirname(__file__), "api")
rootpath = os.path.dirname(os.path.abspath(nerfbaselines.__file__))
excludes = tuple(
    re.compile(fnmatch.translate(os.path.join(rootpath, exclude)))
    for exclude in dict.fromkeys(apidoc_excluded_paths)
)

# Generate modules
opts = argparse.Namespace()
opts.followlinks = False
opts.includeprivate = False
opts.modulefirst = True
opts.separatemodules = False
opts.noheadings = False
opts.maxdepth = 1
opts.destdir = out_path
opts.suffix = "rst"
opts.dryrun = False
opts.force = True
opts.quiet = False
opts.implicit_namespaces = False
opts.header = "API Reference"
from sphinx.ext.apidoc import (
    walk,
    ReSTRenderer, write_file, is_packagedir, is_initpy, is_skipped_package, has_child_module, module_join, is_skipped_module,
    create_module_file
)
imported_members = False
try:
    # For NerfBaselines >0.1.3, we can use the imported members
    from nerfbaselines import Method as _
    imported_members = True
except ImportError:
    pass

def recurse_tree(
    rootpath: str,
    excludes,
    opts,
    user_template_dir: str | None = None,
) -> tuple[list[Path], list[str]]:
    root_package = rootpath.split(os.path.sep)[-1]
    written_files = []
    cutoff_depth = 2
    for root, subs, files in walk(rootpath, excludes, opts):
        del subs
        # Document modules
        depth = root[len(rootpath) :].count(os.path.sep)
        subpackage = root[len(rootpath) :].lstrip(os.path.sep).replace(os.path.sep, '.')
        for file in files:
            filename = os.path.join(root, file)
            if is_skipped_module(filename, opts, excludes) or is_initpy(file):
                continue
            if depth+1 >= cutoff_depth:
                continue
            basename = os.path.splitext(file)[0]
            written_files.append(
                create_module_file(
                    module_join(root_package, subpackage), basename, opts, user_template_dir
                )
            )

        # Document packages
        if not is_packagedir(root) or is_skipped_package(root, opts, excludes) or not has_child_module(root, excludes, opts):
            continue

        if depth >= cutoff_depth:
            continue
            
        # If at cutoff_depth, we include submodules in the package file
        submodules = []
        if depth == cutoff_depth - 1:
            # build a list of sub modules
            submodules = [
                sub.split('.')[0]
                for sub in files
                if not is_skipped_module(Path(root, sub), opts, excludes) and not is_initpy(sub)
            ]
            submodules = sorted(set(submodules))
            submodules = [module_join(root_package, subpackage, modname) for modname in submodules]
        pkgname = module_join(root_package, subpackage)
        context = {
            'pkgname': pkgname,
            'subpackages': [],
            'submodules': submodules,
            'modulefirst': opts.modulefirst,
            'separatemodules': False,
            'automodule_options': (['imported-members'] if imported_members else []) + ['members', 'undoc-members', 'show-inheritance'],
            'show_headings': not opts.noheadings,
            'maxdepth': opts.maxdepth,
        }
        text = ReSTRenderer([user_template_dir]).render('package.rst.jinja', context)
        written_files.append(write_file(pkgname, text, opts))
    return written_files
os.makedirs(out_path, exist_ok=True)
written_files = recurse_tree(rootpath, excludes, opts, apidoc_template_dir)
all_modules = [os.path.relpath(x, out_path)[:-len(opts.suffix)-1] for x in written_files]
all_modules.sort()

# Generate TOC file
context = {
    'header': opts.header,
    'maxdepth': opts.maxdepth,
    'docnames': all_modules,
}
text = ReSTRenderer([apidoc_template_dir]).render('toc.rst.jinja', context)
written_files.append(write_file("modules", text, opts))

# Remove old files
for existing in Path(out_path).glob(f'**/*.{opts.suffix}'):
    if existing not in written_files:
        existing.unlink()

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
if not os.path.exists(os.path.join(rootpath, "datasets", "mipnerf360_spec.py")):
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
        "url": f"https://{CODE_REPOSITORY}",
        "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
        """,
        "class": "",
    }, {
        "name": "Website",
        "url": WEBPAGE_URL,
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


def _get_autodoc_documenter_instance():
    # Unfortunatelly, autodoc documenter instance is not exported in the callbacks
    # so we have to get it from the stack
    for frame in inspect.stack():
        if frame.filename.endswith("autodoc/__init__.py"):
            return frame.frame.f_locals["self"]


def _autodoc_skip_member(app, what, name, obj, would_skip, options):
    # If obj is just reimported in the file and it is not publishing reimport
    # (from underscored file), we will skip it
    documenter = _get_autodoc_documenter_instance()
    # Test if the object is reimported with a simpler path
    module = documenter.object.__name__.split(".")
    for i in range(1, len(module)):
        try:
            reimported = getattr(sys.modules[".".join(module[:i])], name)
            if reimported is obj:
                # If object is just visible from higher up, we skip it here
                return True
        except AttributeError:
            pass

    if hasattr(obj, '__module__'):
        if not getattr(obj, '__module__', '').startswith("nerfbaselines"):
            return True


def setup(app):
    app.connect("html-page-context", _html_page_context, 1000)
    app.connect("autodoc-skip-member", _autodoc_skip_member)
