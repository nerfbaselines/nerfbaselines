from collections import deque, defaultdict
from functools import wraps
import inspect
import threading
import types
from functools import partial
import importlib
from pathlib import Path
import subprocess
from typing import Optional, Type
from typing import  Union, Set, Callable, TYPE_CHECKING, List, cast, Tuple
from typing import Sequence
from ..types import Method
if TYPE_CHECKING:
    from ..registry import MethodSpec


_mounted_paths = {}
_forwarded_ports = {}
_active_backend = {}


def _get_ast_class_interface(cls: Type):
    import ast

    body = []
    imports = defaultdict(set)

    def _get_ast_type(t):
        # Handle generic types
        if t is None:
            return ast.Name(id="None", ctx=ast.Load())
        if hasattr(t, "__origin__"):
            if t.__name__ == "Optional" and t.__module__ == "typing":
                imports["typing"].add("Optional")
                return ast.Subscript(
                    value=ast.Name(id="Optional", ctx=ast.Load()),
                    slice=_get_ast_type(t.__args__[0]),
                    ctx=ast.Load(),
                )
            if t.__module__ == "typing":
                return ast.Subscript(
                    value=ast.Name(id=t.__name__, ctx=ast.Load()),
                    slice=ast.Tuple(elts=[_get_ast_type(x) for x in t.__args__]) 
                        if len(t.__args__) > 1 
                        else _get_ast_type(t.__args__[0]),
                    ctx=ast.Load(),
                )
        if t.__module__ == "builtins":
            return ast.Name(id=t.__name__, ctx=ast.Load())
        imports[t.__module__].add(t.__name__)
        return ast.Name(id=t.__name__, ctx=ast.Load())

    for name, method in cls.__dict__.items():
        # Add method to the class
        if name.startswith("_"):
            continue
        signature = inspect.signature(method)
        return_annotation = signature.return_annotation
        if return_annotation == inspect.Signature.empty:
            return_annotation = None
        else:
            return_annotation = _get_ast_type(return_annotation)
        def _handle_arg(arg):
            if arg.annotation == inspect.Parameter.empty:
                return ast.arg(arg.name, None)
            return ast.arg(arg.name, _get_ast_type(arg.annotation))
        for p in signature.parameters.values():
            if p.kind in {inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY}:
                raise ValueError(f"Unsupported parameter kind {p.kind}")
        method_ast = ast.FunctionDef(
            name=name,
            lineno=0,
            args=ast.arguments(
                vararg=None,
                kwarg=None,
                args=[
                    _handle_arg(arg)
                    for arg in signature.parameters.values()
                    if arg.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                ],
                posonlyargs=[],
                kwonlyargs=[
                    _handle_arg(arg)
                    for arg in signature.parameters.values()
                    if arg.kind == inspect.Parameter.KEYWORD_ONLY
                ],
                kw_defaults=[
                    ast.Constant(arg.default) if arg.default != inspect.Parameter.empty else None
                    for arg in signature.parameters.values()
                    if arg.kind == inspect.Parameter.KEYWORD_ONLY
                ],
                defaults=[
                    ast.Constant(arg.default) if arg.default != inspect.Parameter.empty else None
                    for arg in signature.parameters.values()
                    if arg.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY}
                ],
            ),
            returns=return_annotation,
            body=[
                ast.Expr(value=ast.Ellipsis()),
            ],
            decorator_list=[],
        )
        body.append(method_ast)

    return ast.Module(
        body=[
            *[
                ast.ImportFrom(module, [ast.alias(name, None) for name in names], 0)
                for module, names in imports.items()
            ],
            ast.ClassDef(
                name=cls.__name__,
                bases=[],
                keywords=[],
                body=body,
                decorator_list=[],
            )
        ],
        type_ignores=[],
    )


def _to_snake_case(name: str) -> str:
    out = []
    for i, c in enumerate(name):
        if c.isupper() and i != 0:
            out.append("_")
        out.append(c.lower())
    return "".join(out)


def build_backend_server(interfaces, class_name="BackendServer"):
    """
    Build a backend server class that can be used to run methods on the backend.
    It will automatically generate the methods implementing supplied interfaces.

    Args:
        interfaces: List of classes that should be implemented
        class_name: Name of the class
    """
    def exec_body(ns):
        def __init__(self):
            self._instances = {}
            self._iterators = {}

        ns["__init__"] = __init__

        def _instance_call(self, instance, method, *args, **kwargs):
            if instance not in self._instances:
                raise ValueError(f"Instance {instance} not found")
            obj = self._instances[instance]
            try:
                fn = getattr(obj, method)
            except AttributeError:
                raise AttributeError(f"Method {method} not found")
            return fn(*args, **kwargs)

        ns["_instance_call"] = _instance_call

        # Add init method preparing the list of instances
        prefix = _to_snake_case(Method.__name__) + "_"
        def _resolve_cls(self, cls):
            del self
            fn, fnname = cls.split(":", 1)
            fn = importlib.import_module(fn)
            for part in fnname.split("."):
                fn = getattr(fn, part)
            return fn
        ns["_" + prefix + "_resolve_cls"] = _resolve_cls
        for name, method in Method.__dict__.items():
            if name.startswith("_") and name != "__init__":
                continue
            has_instance = False
            if name == "__init__":
                name = "init"
                signature = inspect.signature(method)
                has_instance = False
            elif isinstance(method, staticmethod):
                raise ValueError("Static methods are not supported")
            elif isinstance(method, classmethod):
                # Add method to the class
                signature = inspect.signature(method.__wrapped__)
            else:
                # Add method to the class
                signature = inspect.signature(method)
                has_instance = True

            # Replace POSITIONAL_OR_KEYWORD to KEYWORD_ONLY
            args = dict(
                self=inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            )
            if has_instance:
                args["instance"] = inspect.Parameter("instance", inspect.Parameter.KEYWORD_ONLY, annotation=int)
            else:
                args["cls"] = inspect.Parameter("cls", inspect.Parameter.KEYWORD_ONLY, annotation=str)
            for i, (argname, arg) in enumerate(signature.parameters.items()):
                if (i == 0 and argname in {"self", "cls"}):
                    continue
                args[argname] = arg.replace(kind=inspect.Parameter.KEYWORD_ONLY)
            signature = signature.replace(parameters=args.values())

            is_iterator = (
                name != "__init__" and
                signature.return_annotation is not None and
                signature.return_annotation.__module__ == "typing" and 
                signature.return_annotation.__name__ == "Iterable"
            )
            if is_iterator:
                assert has_instance, "Iterators are only supported for instance methods"

            if is_iterator:
                iterator_annotation = signature.return_annotation.__args__[0]
                signature = signature.replace(return_annotation=Tuple[int, iterator_annotation])
                # 1) Change return annotation
                # 2) Rename method to ..._start{name} - we will add the next 
                #    and stop methods later
                def _start(self, instance, *args, **kwargs):
                    iterable = self._instance_call(instance, method.__name__, *args, **kwargs)
                    iterator = iter(iterable)
                    out = next(iterator)
                    self._iterators[id(iterator)] = iterator
                    return id(iterator), out

                _start.__signature__ = signature
                _start.__name__ = prefix + name + "_start"
                ns[_start.__name__] = _start

                def _next(self, iterator):
                    iterator = self._iterators.pop(iterator)
                    out = next(iterator)
                    self._iterators[id(iterator)] = iterator
                    return out

                _next.__signature__ = inspect.Signature(
                    parameters=[
                        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
                        inspect.Parameter("iterator", inspect.Parameter.KEYWORD_ONLY, annotation=int),
                    ],
                    return_annotation=iterator_annotation,
                )
                _next.__name__ = prefix + name + "_next"
                ns[_next.__name__] = _next

                def _stop(self, iterator):
                    iterator = self._iterators.pop(iterator)
                    del iterator

                _stop.__signature__ = _next.__signature__.replace(return_annotation=None)
                _stop.__name__ = prefix + name + "_stop"
                ns[_stop.__name__] = _stop
            else:
                if has_instance:
                    def _call(self, instance, *args, **kwargs):
                        return self._instance_call(instance, method.__name__, *args, **kwargs)
                else:
                    def _call(self, *args, **kwargs):
                        return self._call(method.__name__, *args, **kwargs)
                _call = wraps(method)(_call)
                _call.__signature__ = signature
                _call.__name__ = prefix + name
                ns[_call.__name__] = _call


    cls = types.new_class(class_name, (), {}, exec_body=exec_body)

    # Build the backend server module
    return cls


def _implement_client_interface(interface, class_name="BackendClient"):
    pass


if __name__ == "__main__":
    import ast
    from nerfbaselines.types import Method
    cls = build_backend_server([Method])
    ast_module = _get_ast_class_interface(cls)
    print(ast.unparse(ast_module))
