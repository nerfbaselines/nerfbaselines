from typing import Callable
import types
import ast
import sys
import importlib
import importlib.abc
import importlib.util


class _MetaFinder(importlib.abc.MetaPathFinder):
    _module_overrides = {}

    class _Loader(importlib.abc.Loader):
        _being_imported = {}

        def create_module(self, spec):
            if spec.name not in _MetaFinder._module_overrides:
                return None
            if spec.name in self._being_imported:
                return self._being_imported[spec.name]
            assert spec.origin is not None, "origin is required"
            with open(spec.origin) as f:
                ast_module = ast.parse(f.read())
            for callback in _MetaFinder._module_overrides[spec.name]:
                callback(ast_module)
            module = types.ModuleType(spec.name)
            module.__spec__ = spec
            module.__loader__ = self
            module.__name__ = spec.name
            self._being_imported[spec.name] = module
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
        if fullname in _MetaFinder._module_overrides:
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

    @staticmethod
    def patch_ast_import(module: str):
        def wrap(callback: Callable[[ast.Module], None]):
            if module not in _MetaFinder._module_overrides:
                _MetaFinder._module_overrides[module] = []
            _MetaFinder._module_overrides[module].append(callback)
        if module in sys.modules:
            sys.modules.pop(module)
        return wrap


sys.meta_path.insert(0, _MetaFinder())
patch_ast_import = _MetaFinder.patch_ast_import
