from contextlib import contextmanager
from threading import local
from functools import partial
import logging
import itertools
from typing import Callable
import types
import ast
import sys
import importlib
import importlib.abc
import importlib.util


_local = local()
_local.data = None


def _get_overrides_and_patches():
    if _local.data is None:
        return {}, {}
    return _local.data


def _parse_patch(patch):
    pl = patch.splitlines()
    i = 0
    fullpatch = {}
    while i < len(pl):
        while i < len(pl) and not pl[i].startswith("--- "):
            i += 1
        if i >= len(pl):
            break

        old_hdr = pl[i][4:].strip()
        i += 1
        if i >= len(pl) or not pl[i].startswith("+++ "):
            raise ValueError('Malformed patch: missing "+++" line')
        new_hdr = pl[i][4:].strip()
        i += 1

         # choose the filename that exists after applying the patch
        if new_hdr != "/dev/null":
            file = new_hdr[2:] if new_hdr.startswith("b/") else new_hdr
        else:
            file = old_hdr[2:] if old_hdr.startswith("a/") else old_hdr

        updates_mod = 0
        while i < len(pl) and pl[i].startswith("@@ "):
            old, new = pl[i][3:].split(" ", 3)[:2]
            i += 1
            assert old.startswith("-")
            assert new.startswith("+")
            old = tuple(map(int, old[1:].split(",")))
            if len(old) == 1:
                old = (old[0], 1)
            new = tuple(map(int, new[1:].split(",")))
            if len(new) == 1:
                new = (new[0], 1)
            oldlines = []
            newlines = []
            while i < len(pl):
                if pl[i].startswith("-"):
                    oldlines.append(pl[i][1:])
                elif pl[i].startswith("+"):
                    newlines.append(pl[i][1:])
                elif pl[i].startswith(" "):
                    oldlines.append(pl[i][1:])
                    newlines.append(pl[i][1:])
                elif len(pl[i]) == 0:
                    oldlines.append("")
                    newlines.append("")
                elif pl[i].startswith("\\"):
                    pass
                else:
                    break
                i += 1
                if i >= len(pl) or (pl[i] and pl[i][0] not in (" ", "-", "+", "\\")):
                    break
            updates = fullpatch.setdefault(file, [])
            assert len(oldlines) == old[1]
            assert len(newlines) == new[1]
            updates.append((old[0] + updates_mod, oldlines, newlines))
            updates_mod += new[1] - old[1]
    return fullpatch


def _apply_patch(content, updates):
    has_trailing_nl = content.endswith("\n")
    lines = content.splitlines()
    if has_trailing_nl:
        lines.append("")

    for lineno, oldlines, newlines in updates:
        actuallines = lines[lineno - 1:lineno + len(oldlines) - 1]
        assert "\n".join(actuallines) == "\n".join(oldlines), \
            f"Expected {oldlines} at line {lineno}, got {actuallines}"
        lines = lines[:lineno - 1] + newlines + lines[lineno + len(oldlines) - 1:]

    out = "\n".join(lines)
    if has_trailing_nl and not out.endswith("\n"):
        out += "\n"
    return out


class _MetaFinder(importlib.abc.MetaPathFinder):
    class _Loader(importlib.abc.Loader):
        _being_imported = {}

        def create_module(self, spec):
            overrides, patches = _get_overrides_and_patches()
            if spec.name not in overrides and spec.name not in patches:
                return None
            if spec.name in self._being_imported:
                return self._being_imported[spec.name]
            logging.debug("Patching %s", spec.name)
            assert spec.origin is not None, "origin is required"
            with open(spec.origin) as f:
                code = f.read()
            for callback in patches.get(spec.name, []):
                code = callback(code)
            ast_module = ast.parse(code)
            for callback in overrides.get(spec.name, []):
                callback(ast_module)
            module = types.ModuleType(spec.name)
            module.__spec__ = spec
            module.__loader__ = self
            module.__name__ = spec.name
            self._being_imported[spec.name] = module
            ast_module = ast.fix_missing_locations(ast_module)
            try:
                exec(compile(ast_module, spec.origin, "exec"), module.__dict__)
            finally:
                self._being_imported.pop(spec.name)
            return module

        def exec_module(self, module):
            del module
            return

    def find_spec(self, fullname, path, target=None):
        del path, target
        overrides, patches = _get_overrides_and_patches()
        if fullname in overrides or fullname in patches:
            # Temporarily remove the custom finder from sys.meta_path to avoid recursion
            original_meta_path = sys.meta_path[:]
            try:
                sys.meta_path = [finder for finder in sys.meta_path if not isinstance(finder, _MetaFinder)]
                spec = importlib.util.find_spec(fullname)
            finally:
                sys.meta_path = original_meta_path
            if spec:
                # Set the loader to be the custom loader
                spec.loader = _MetaFinder._Loader()
            return spec
        return None


def apply_patch(patch):
    patchdata = _parse_patch(patch)
    for path, updates in patchdata.items():
        assert path.endswith(".py"), "Only .py files are supported"
        path = path[:-3]
        path = path.replace("/", ".")
        if path.endswith(".__init__"):
            path = path[:-9]
        def _apply_patch_locally(updates, path, code):
            try:
                return _apply_patch(code, updates)
            except Exception as e:
                raise RuntimeError(f"Failed to apply patch to '{path}'") from e
        _local.patch_code(path, partial(_apply_patch_locally, updates, path))



class Context:
    def __init__(self):
        self._module_overrides = {}
        self._module_patches = {}
        self._backup = None

    def __enter__(self):
        if _local.data is None:
            # Register the custom finder
            if sys.meta_path and not sys.meta_path[0].__class__.__name__ == "_MetaFinder":
                sys.meta_path.insert(0, _MetaFinder())
        self._backup = _local.data
        _local.data = (self._module_overrides, self._module_patches)
        for module in itertools.chain(self._module_overrides, self._module_patches):
            if module in sys.modules:
                sys.modules.pop(module)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        _local.data = self._backup
        self._backup = None
        if _local.data is None:
            # Remove the custom finder
            sys.meta_path = [finder for finder in sys.meta_path if not isinstance(finder, _MetaFinder)]
        for module in itertools.chain(self._module_overrides, self._module_patches):
            if module in sys.modules:
                del sys.modules[module]
        importlib.invalidate_caches()

    def patch_ast_import(self, module: str):
        def wrap(callback: Callable[[ast.Module], None]):
            def _callback(*args, **kwargs):
                try:
                    return callback(*args, **kwargs)
                except Exception as e:
                    raise ImportError(f"Error in patching {module}: {e}")
            if module not in self._module_overrides:
                self._module_overrides[module] = []
            self._module_overrides[module].append(_callback)
            return callback
        return wrap

    def patch_code(self, module: str, callback):
        if module not in self._module_patches:
            self._module_patches[module] = []
        self._module_patches[module].append(callback)

    def apply_patch(self, patch):
        patchdata = _parse_patch(patch)
        for path, updates in patchdata.items():
            assert path.endswith(".py"), "Only .py files are supported"
            path = path[:-3]
            path = path.replace("/", ".")
            if path.endswith(".__init__"):
                path = path[:-9]
            def _apply_patch_locally(updates, path, code):
                try:
                    return _apply_patch(code, updates)
                except Exception as e:
                    raise RuntimeError(f"Failed to apply patch to '{path}'") from e
            self.patch_code(path, partial(_apply_patch_locally, updates, path))
