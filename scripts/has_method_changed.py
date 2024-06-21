import argparse
import sys
import re
import os
import ast


def get_dependency_tree(root_path, module_path=None, file_path=None, module_filter="nerfbaselines.*", _ignore=None):
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
        if module_path.endswith("__init__.py"):
            module_path = module_path[:-9]
        elif module_path.endswith(".py"):
            module_path = module_path[:-3]

    if file_path in _ignore:
        return {}
    offset_level = 1 if file_path.endswith("__init__.py") else 0
    imports = {}

    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    def visit_node(node):
        if isinstance(node, ast.Import):
            for n in node.names:
                if re.match(module_filter, n.name):
                    imports.update(get_dependency_tree(root_path, n.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module
            module = ".".join(module_path.split(".")[:-node.level+offset_level] + ([module] if module is not None else [])) if node.level else module
            if module is not None and re.match(module_filter, module):
                imports.update(get_dependency_tree(root_path, module, module_filter=module_filter, _ignore=_ignore.union((file_path,))))
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == "TYPE_CHECKING":
            return
        for child in ast.iter_child_nodes(node):
            visit_node(child)
    visit_node(tree)
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
    registry = {}
    for package in os.listdir(path):
        if package.endswith("_spec.py") and not package.startswith("_"):
            package = package[:-3]
            fpath = os.path.join(path, package + ".py")
            with open(fpath, "r") as file:
                tree = ast.parse(file.read(), filename=package + ".py")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "register":
                    for arg in node.keywords:
                        if arg.arg == "name" and isinstance(arg.value, ast.Constant):
                            registry[arg.value.value] = fpath
                            break
                    else:
                        raise RuntimeError(f"Argument 'name' not found in register call while processing {package}.py")
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
    from nerfbaselines.registry import get_method_spec
    method_spec = get_method_spec(method_name)
    _method_module_name = method_spec["method"].split(":")[0]
    method_module_name = _method_module_name.lstrip(".")
    method_module_level = len(_method_module_name) - len(method_module_name)
    if method_module_level:
        method_module_name = "nerfbaselines.methods".split(".")[:-method_module_level] + [method_module_name]
    if "method" in method_spec:
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


def git_has_change(*paths):
    # Test if there is a change in the paths from the last commit or uncommited changes
    import subprocess
    try:
        root_path = None
        if len(paths) > 1:
            root_path = os.path.commonpath(paths)
        elif len(paths) == 1:
            root_path = os.path.dirname(paths[0])
        subprocess.check_output(["git", "diff", "--exit-code", "HEAD~1", "--", *paths], cwd=root_path)
    except subprocess.CalledProcessError:
        return True
    return False


def method_has_changes(method, backend=None):
    root_path = os.path.abspath(os.path.join(__file__, "../.."))
    dep_tree = get_method_dependency_tree(root_path, method, backend)
    dep_tree_formatted = format_dependency_tree(dep_tree, root_path + "/nerfbaselines").splitlines(keepends=True)
    dep_tree_formatted = ["   " + line for line in dep_tree_formatted]
    print(f"Method: \033[96m{method}\033[0m", file=sys.stderr)
    print("Dependency tree: \n" + "".join(dep_tree_formatted), file=sys.stderr)

    dependencies = dependency_tree_to_list(dep_tree)
    changes = list(map(git_has_change, dependencies))
    dependencies_formatted = [
        f"   \033[31m{os.path.relpath(x, root_path)}\033[0m\n" if change else f"   {os.path.relpath(x, root_path)}\n"
        for x, change in zip(dependencies, changes)]
    print(f"Dependencies ({len(dependencies)}, {sum(changes)} changes):\n" + "".join(dependencies_formatted), file=sys.stderr)
    return any(changes)


def main():
    parser = argparse.ArgumentParser(description="Check if a method has changed")
    parser.add_argument("method", help="Method name")
    parser.add_argument("--backend", help="Backend name", default=None)
    args = parser.parse_args()

    changes = method_has_changes(args.method, args.backend)
    sys.exit(changes)


if __name__ == "__main__":
    main()
