from unittest import mock
import subprocess
import argparse
import sys
import re
import os
import ast
from functools import lru_cache, partial
import os, subprocess, sys, json



@lru_cache(maxsize=None)
def _get_file_imports(file_path, module_path):
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    offset_level = 1 if file_path.endswith("__init__.py") else 0
    imports = {}

    def visit_node(node):
        if isinstance(node, ast.Import):
            for n in node.names:
                name = n.asname if n.asname is not None else n.name
                imports[name] = n.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            level = node.level-offset_level
            _path = module_path.split(".")[:-level] if level > 0 else module_path.split(".")
            module = ".".join(_path + ([module] if module is not None else [])) if node.level else module
            for n in node.names:
                name = n.asname if n.asname is not None else n.name
                imports[name] = ".".join((module, n.name))
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return
        for child in ast.iter_child_nodes(node):
            visit_node(child)
    visit_node(tree)
    return imports


def _resolve_import(root_path, import_path):
    path = os.path.join(root_path, import_path.replace(".", os.path.sep))
    pardir, name = os.path.split(path)
    members = os.listdir(pardir) if os.path.exists(pardir) else []
    if (name+".py") in members:
        return import_path, path+".py"
    elif name in members and os.path.isdir(path) and (os.path.exists(os.path.join(path, "__init__.py"))):
        return import_path, os.path.join(path, "__init__.py")
    elif os.path.exists(os.path.join(pardir, "__init__.py")):
        return ".".join(import_path.split(".")[:-1]), os.path.join(pardir, "__init__.py")
    elif os.path.exists(pardir+".py"):
        return ".".join(import_path.split(".")[:-1]), pardir+".py"
    else:
        raise FileNotFoundError(f"Module {import_path} not found")


def _spider_import(root_path, import_path):
    module_path, file_path = _resolve_import(root_path, import_path)
    if module_path == import_path:
        # Full module import
        return file_path
    imports = _get_file_imports(file_path, module_path)
    name = import_path.split(".")[-1]
    if name not in imports:
        # Not reimported in the file
        return file_path
    return _spider_import(root_path, imports[name])


def get_dependency_tree(root_path, 
                        module_path=None, 
                        file_path=None, 
                        module_filter="nerfbaselines.*", 
                        include_transient_dependencies: bool = False,
                        _ignore=None):
    if _ignore is None:
        _ignore = set()
    if file_path is None:
        assert module_path is not None
        file_path = os.path.join(root_path, module_path.replace(".", os.path.sep) + ".py")
        if not os.path.exists(file_path):
            file_path = os.path.join(root_path, module_path.replace(".", os.path.sep), "__init__.py")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Module {module_path} not found")
    elif module_path is None:
        assert file_path is not None
        module_path = os.path.relpath(file_path, root_path).replace(os.path.sep, ".")
        if module_path.endswith(".__init__.py"):
            module_path = module_path[:-len(".__init__.py")]
        elif module_path.endswith(".py"):
            module_path = module_path[:-3]

    file_imports = _get_file_imports(file_path, module_path)
    imports = {}
    _to_add_imports = []
    for import_path in file_imports.values():
        if not re.match(module_filter, import_path):
            continue
        if not include_transient_dependencies:
            _file_path = _spider_import(root_path, import_path)
        else:
            _, _file_path = _resolve_import(root_path, import_path)
        assert _file_path is not None, f"Import {import_path} not found"
        if _file_path not in _ignore:
            _to_add_imports.append(_file_path)

    _ignore = _ignore.union(_to_add_imports + [file_path])
    for _file_path in _to_add_imports:
        imports.update(get_dependency_tree(
            root_path, 
            file_path=_file_path, 
            module_filter=module_filter, 
            _ignore=_ignore))
    return {
        file_path: imports
    }


def format_dependency_tree(tree, root_path, level=0):
    out = ""
    for key, value in tree.items():
        out += ("   "*level) + os.path.relpath(key, root_path) + "\n"
        if value:
            out += format_dependency_tree(value, root_path, level+1)
    return out


def find_registered_method_specs(path):
    import nerfbaselines._registry
    registry = {}
    for package in os.listdir(path):
        if package.endswith("_spec.py") and not package.startswith("_"):
            package = package[:-3]
            fpath = os.path.join(path, package + ".py")

            registered_calls = []
            with nerfbaselines._registry.collect_register_calls(registered_calls), \
                    mock.patch("nerfbaselines._registry._make_entrypoint_absolute", lambda x: x):
                # Execute the file
                with open(fpath, "r") as file:
                    contents = file.read()
                try:
                    globals = {'__file__': fpath}
                    exec(contents, globals, {})
                except Exception as e:
                    raise RuntimeError(f"Error while processing {package}.py") from e
            registry.update({x["id"]: fpath for x in registered_calls})
    return registry


def get_method_dependency_tree(root_path, method_name, backend=None):
    dep_tree = {}

    # Get method spec and spec path
    registered_files = find_registered_method_specs(root_path + "/nerfbaselines/methods")
    if method_name not in registered_files:
        raise ValueError(f"Method {method_name} not found in registered methods")
    dep_tree.update(get_dependency_tree(
        root_path, 
        file_path=registered_files[method_name]))

    # Now we get path to the actual impl
    from nerfbaselines import get_method_spec
    method_spec = get_method_spec(method_name)
    _method_module_name = method_spec["method_class"].split(":")[0]
    method_module_name = _method_module_name.lstrip(".")
    method_module_level = len(_method_module_name) - len(method_module_name)
    if method_module_level:
        method_module_name = "nerfbaselines.methods".split(".")[:-method_module_level] + [method_module_name]
    if "method_class" in method_spec and method_module_name.startswith("nerfbaselines.methods"):
        dep_tree.update(get_dependency_tree(
            root_path=root_path,
            module_path=method_module_name))

    if backend is not None and backend != "python":
        # Add backend impl dependencies
        backend_tree = get_dependency_tree(
            root_path=root_path,
            module_path=f"nerfbaselines.backends._{backend}")

        # Filter backend_tree, remove _common.py -> _... links
        def _walk(tree, name=None):
            for k, v in list(tree.items()):
                if name == os.path.join(root_path, "nerfbaselines/backends/_common.py"):
                    if os.path.join(root_path, "nerfbaselines/backends/_"):
                        tree.pop(k)
                _walk(v, name=k)
        _walk(backend_tree)
        dep_tree.update(backend_tree)

    return dep_tree


def dependency_tree_to_list(tree):
    out = set()
    for k, v in tree.items():
        out.add(k)
        out.update(dependency_tree_to_list(v))
    return sorted(out)


def git_has_change(base_commit, *paths):
    # Test if there is a change in the paths from the last commit or uncommited changes
    import subprocess
    try:
        root_path = None
        if len(paths) > 1:
            root_path = os.path.commonpath(paths)
        elif len(paths) == 1:
            root_path = os.path.dirname(paths[0])
        # Check for changes in the last commit
        subprocess.check_output("git diff --exit-code".split() + [base_commit, "--"] + list(paths), cwd=root_path)
        # Check for untracked files
        subprocess.run("git ls-files --exclude-standard --error-unmatch --".split() + list(paths), 
                       cwd=root_path, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                       check=True)
    except subprocess.CalledProcessError:
        return True
    return False


def method_has_changes(method, backend=None, base_commit=None):
    if base_commit is None:
        base_commit = get_base_commit()
    root_path = os.path.abspath(os.path.join(__file__, "../.."))
    dep_tree = get_method_dependency_tree(root_path, method, backend)
    dep_tree_formatted = format_dependency_tree(dep_tree, root_path + "/nerfbaselines").splitlines(keepends=True)
    dep_tree_formatted = ["   " + line for line in dep_tree_formatted]
    print(f"Method: \033[96m{method}\033[0m", file=sys.stderr)
    print("Dependency tree: \n" + "".join(dep_tree_formatted), file=sys.stderr)

    dependencies = dependency_tree_to_list(dep_tree)
    if os.path.join(root_path, "nerfbaselines/_version.py") in dependencies:
        dependencies.remove(os.path.join(root_path, "nerfbaselines/_version.py"))
    changes = list(map(partial(git_has_change, base_commit), dependencies))
    dependencies_formatted = [
        f"   \033[31m{os.path.relpath(x, root_path)}\033[0m\n" if change else f"   {os.path.relpath(x, root_path)}\n"
        for x, change in zip(dependencies, changes)]
    print(f"Dependencies ({len(dependencies)}, {sum(changes)} changes):\n" + "".join(dependencies_formatted), file=sys.stderr)
    return any(changes)


def _cmd(*args: str) -> str:
    """Run a git command and return stripped stdout (empty on error)."""
    try:
        return subprocess.check_output(args).decode().strip()
    except subprocess.CalledProcessError as e:
        print(e)
        return ""


def get_default_branch():
    """
    Works without any local refs:
      1.  In Actions → use the event payload (always present and 100 % reliable);
      2.  Anywhere else  → ask the remote (`origin/HEAD`), which always tracks the
          repository’s default branch  [oai_citation:0‡Stack Overflow](https://stackoverflow.com/questions/8839958/how-does-origin-head-get-set).
    """
    # 1) CI: the payload file always knows the default branch
    if path := os.getenv("GITHUB_EVENT_PATH"):
        with open(path) as f:
            if d := json.load(f)["repository"].get("default_branch"):
                return d

    # 2) Fallback: read origin/HEAD (works off-line too)
    ref = _cmd("git", "symbolic-ref", "--short", "refs/remotes/origin/HEAD")
    if ref:
        return ref.split("/", 1)[1]           # origin/main → main

    # 3) Last-ditch: look for a local main or master
    for cand in ("main", "master"):
        if _cmd("git", "show-ref", "--verify", f"refs/heads/{cand}"):
            return cand
    raise RuntimeError("Cannot determine repository default branch")


def _current_branch():
    """
    Best-effort branch name for the *checked-out commit*.

    • PR runs  → `GITHUB_HEAD_REF`  
    • Pushes  → `GITHUB_REF_NAME`  
    • Local   → `git symbolic-ref HEAD`  
    • Detached→ None
    """
    return (
        os.getenv("GITHUB_HEAD_REF")
        or os.getenv("GITHUB_REF_NAME")
        or _cmd("git", "symbolic-ref", "--short", "-q", "HEAD") or None
    )

def get_base_commit():
    """
    • On the default branch → compare to the previous commit (HEAD~1)  
    • On any other ref      → compare to the merge-base with the default branch
      (uses the remote ref so it works in detached-HEAD CI jobs).
    """
    default_branch = get_default_branch()
    branch = _current_branch()

    if branch == default_branch:
        base = _cmd("git", "rev-parse", "HEAD~1")
    else:
        # In CI the local ref for the default branch usually doesn't exist,
        # so ensure we have the remote one:
        subprocess.check_call(
            ["git", "fetch", "--quiet", "origin", default_branch, "--depth", "1"]
        )
        base = _cmd("git", "merge-base", "HEAD", f"origin/{default_branch}")

    print(
        f"[debug] HEAD branch: {branch or '(detached)'}  "
        f"default: {default_branch}  base: {base}",
        file=sys.stderr,
    )
    if not base:
        raise RuntimeError("Could not determine base commit")
    return base


def main():
    parser = argparse.ArgumentParser(description="Check if a method has changed")
    parser.add_argument("method", help="Method name")
    parser.add_argument("--backend", help="Backend name", default=None)
    parser.add_argument("--base-commit", help="Base commit", default=None)
    args = parser.parse_args()

    changes = method_has_changes(args.method, args.backend, base_commit=args.base_commit)
    sys.exit(changes)


if __name__ == "__main__":
    main()
