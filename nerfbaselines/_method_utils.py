import numpy as np
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
    from nerfbaselines.backends._common import backend_allocate_ndarray

    def wrap_get_info(get_info, spec):
        @functools.wraps(get_info)
        def __get_info(*args, **kwargs):
            info = get_info(*args, **kwargs)
            info["method_id"] = spec["id"]
            return info

        __get_info.__name__ = get_info.__name__  # type: ignore
        return __get_info

    def format_output_base(method, output, options):
        from nerfbaselines.utils import convert_image_dtype
        if not isinstance(output, dict):
            return output
        supported_outputs = method.get_info().get("supported_outputs", [])
        output_types = {
            v["name"]: v["type"]
            for v in supported_outputs if not isinstance(v, str) and "type" in v}
        selected_outputs = None
        out = {}
        for k, v in output.items():
            # Remove unused outputs to reduce data transfer
            if selected_outputs is not None and k not in selected_outputs:
                continue

            # Try casting to the requested type
            output_type_name = output_types.get(k, k)
            if options is not None and options.get("output_type_dtypes") is not None:
                dtype = options["output_type_dtypes"].get(output_type_name, None)
                if dtype is not None:
                    v = convert_image_dtype(v, dtype)

            # Finally, if the output is a tensor, convert it to a numpy array
            # Use shared memory to avoid unnecessary data transfer
            if getattr(v, "__module__", None) == "torch":
                if not (options or {}).get("keep_torch", False):
                    # First, we allocate the output in shared memory
                    import torch
                    dtype_name = str(v.dtype).split(".")[-1]
                    if hasattr(np, dtype_name):
                        new_v = backend_allocate_ndarray(v.shape, dtype=dtype_name)
                        new_v_torch = torch.from_numpy(new_v)
                        new_v_torch.copy_(v)
                    else:
                        new_v = v.cpu().numpy()
                    v = new_v
            out[k] = v
        return out

    # Override format output to avoid unnecessary data transfer between CPU and GPU
    # Instead, we will perform casting on GPU and load directly to shared memory
    # This is to avoid unnecessary data transfer between CPU and GPU
    def wrap_format_output(format_output, spec):
        del spec
        @functools.wraps(format_output)
        def __format_output(self, out, options):
            if not isinstance(out, dict): return out
            return format_output_base(self, out, options)
        try:
            __format_output.__name__ = format_output.__name__  # type: ignore
        except AttributeError:
            pass
        return __format_output

    # Add autocast to render output to make remote backends faster
    def wrap_render(render, spec):
        del spec

        @functools.wraps(render)
        def __render(self, *args, options=None, **kwargs):
            out = render(self, *args, options=options, **kwargs)
            if not isinstance(out, dict): return out
            return format_output_base(self, out, options)
        try:
            __render.__name__ = render.__name__  # type: ignore
        except AttributeError:
            pass
        return __render

    # Update get_info and get_method_info with method_id
    ns = {}
    ns["get_info"] = wrap_get_info(method_class.get_info, spec)
    ns["get_method_info"] = staticmethod(wrap_get_info(method_class.get_method_info, spec))
    old_format_output = getattr(method_class, "_format_output", None)
    if old_format_output is not None:
        ns["_format_output"] = wrap_format_output(old_format_output, spec)
    else:
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
