import types
import functools
import importlib
import logging
from typing import Any, Type, cast
from contextlib import contextmanager
from typing import Optional
from nerfbaselines import Method, MethodSpec, BackendName


def _wrap_method_class(method_class: Type[Method], spec: MethodSpec):
    def wrap_get_info(get_info, spec):
        @functools.wraps(get_info)
        def __get_info(*args, **kwargs):
            info = get_info(*args, **kwargs)
            info["method_id"] = spec["id"]
            return info

        __get_info.__name__ = get_info.__name__  # type: ignore
        return __get_info

    # Add autocast to render output to make remote backends faster
    def wrap_render(render, spec):
        from nerfbaselines.utils import convert_image_dtype
        del spec
        output_types = None

        @functools.wraps(render)
        def __render(self, *args, options=None, **kwargs):
            nonlocal output_types
            if output_types is None:
                output_types = {
                    (v if isinstance(v, str) else v["name"]): (v if isinstance(v, str) else v.get("type", v["name"]))
                    for v in self.get_info().get("supported_outputs", {})}
            for out in render(self, *args, **kwargs):
                if not isinstance(out, dict):
                    yield out
                    continue
                for k, v in out.items():
                    output_type = output_types.get(k, k)
                    if options is not None and options.get("output_type_dtypes") is not None:
                        dtype = options["output_type_dtypes"].get(output_type, None)
                        if dtype is not None:
                            v = convert_image_dtype(v, dtype)
                            out[k] = v
                yield out
        try:
            __render.__name__ = render.__name__  # type: ignore
        except AttributeError:
            pass
        return __render

    # Update get_info and get_method_info with method_id
    ns = {}
    ns["get_info"] = wrap_get_info(method_class.get_info, spec)
    ns["get_method_info"] = staticmethod(wrap_get_info(method_class.get_method_info, spec))
    ns["render"] = wrap_render(method_class.render, spec)
    newcls = types.new_class(method_class.__name__, bases=(method_class,), exec_body=lambda _ns: _ns.update(ns))
    newcls.__module__ = method_class.__module__
    return newcls


def _build_method_class_internal(spec: "MethodSpec") -> Type[Method]:
    name = spec["method_class"]
    package, name = name.split(":")
    cls: Any = importlib.import_module(package)
    for p in name.split("."):
        cls = getattr(cls, p)
    newcls = _wrap_method_class(cls, spec)
    return newcls


@contextmanager
def build_method_class(spec: MethodSpec, backend: Optional[BackendName] = None):
    """
    Build a method class from a method spec. It automatically selects the backend based on the method spec if none is provided.

    Args:
        spec: Method spec
        backend: Backend name
    """
    from . import backends
    backend_impl = backends.get_backend(spec, backend)
    method = spec["id"]
    logging.info(f"Using method: {method}, backend: {backend_impl.name}")
    with backend_impl:
        backend_impl.install()
        build_method = _build_method_class_internal
        yield cast(Type[Method], backend_impl.static_call(f"{build_method.__module__}:{build_method.__name__}", spec))
