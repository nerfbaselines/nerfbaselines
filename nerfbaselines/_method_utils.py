import json
import os
import types
import functools
import importlib
import logging
from typing import Any, Type, cast, Tuple, Dict
from contextlib import contextmanager, ExitStack
from typing import Optional, Generator
from nerfbaselines import Method, MethodSpec, BackendName, get_method_spec


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
            out = render(self, *args, options=options, **kwargs)
            nonlocal output_types
            if output_types is None:
                output_types = {
                    (v if isinstance(v, str) else v["name"]): (v if isinstance(v, str) else v.get("type", v["name"]))
                    for v in self.get_info().get("supported_outputs", {})}
            if isinstance(out, dict):
                for k, v in out.items():
                    output_type = output_types.get(k, k)
                    if options is not None and options.get("output_type_dtypes") is not None:
                        dtype = options["output_type_dtypes"].get(output_type, None)
                        if dtype is not None:
                            v = convert_image_dtype(v, dtype)
                            out[k] = v
            return out
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


@contextmanager
def load_checkpoint(checkpoint: str, *, backend: Optional[BackendName] = None) -> Generator[Tuple[Method, Dict], None, None]:
    """
    This is a utility function to open the checkpoint directory,
    mount it, start the backend, build the model class and load the checkpoint.
    The checkpoint can be a local path, a remote path or a path inside a zip file.
    The function returns a context manager that yields the model instance and nb-info.

    Args:
        checkpoint: Path to the checkpoint. Can be a local path or a remote path. Can also be a path inside a zip file.
        backend: Backend name

    Returns:
        Context manager that yields a tuple of model instance and the nb-info dictionary.

    """
    from nerfbaselines.io import open_any_directory, deserialize_nb_info
    from nerfbaselines import backends

    logging.info(f"Loading checkpoint {checkpoint}")
    with ExitStack() as stack:
        # Load the checkpoint
        checkpoint_path = stack.enter_context(open_any_directory(checkpoint, mode="r"))
        stack.enter_context(backends.mount(checkpoint_path, checkpoint_path))
        assert os.path.exists(checkpoint_path), f"checkpoint path {checkpoint} does not exist"
        assert os.path.exists(os.path.join(checkpoint_path, "nb-info.json")), \
            f"checkpoint path {checkpoint} does not contain nb-info.json"
        # Read method nb-info
        with open(os.path.join(checkpoint_path, "nb-info.json"), "r", encoding="utf8") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        # Load the method
        method_name = nb_info["method"]
        method_spec = get_method_spec(method_name)
        method_cls = stack.enter_context(build_method_class(method_spec, backend=backend))
        model = method_cls(checkpoint=str(checkpoint_path))
        yield model, nb_info
