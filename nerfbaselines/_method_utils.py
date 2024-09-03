import pprint
import types
import functools
import importlib
import json
import logging
import os
from typing import Any, Type, cast, Tuple
from contextlib import contextmanager
from typing import Optional
from nerfbaselines import Method, MethodSpec, BackendName, Dataset


def _dict_equal(a, b):
    if a is b:
        return True
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        if len(a) != len(b):
            return False
        for k, v in a.items():
            if k not in b:
                return False
            if not _dict_equal(v, b[k]):
                return False
        return True
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        for i, v in enumerate(a):
            if not _dict_equal(v, b[i]):
                return False
        return True
    return a == b


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


@contextmanager
def build_method(spec: Optional[MethodSpec] = None,
                 checkpoint: Optional[str] = None,
                 method: Optional[str] = None,
                 backend: Optional[BackendName] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None,
                 presets: Optional[Tuple[str, ...]] = None):
    """
    Build a method instance from a method spec, checkpoint, or a training datasets. This is an utility function that
    automatically infers the method spec from the checkpoint, or the method ID from the method spec.
    It also applies presets based on the dataset metadata and the user-provided presets.
    """

    # Validate spec conflicts
    if spec is not None and method is not None:
        if method != spec.get("id"):
            logging.error(f"Method spec's id={spec['id']} is in conflict with the method argument {method}.")

    if checkpoint is not None:
        import nerfbaselines
        from nerfbaselines.io import deserialize_nb_info
        from nerfbaselines.io import open_any_directory

        # We open the checkpoint first
        ckpt_config_overrides = None
        with open_any_directory(checkpoint, "r") as _checkpoint_path:
            if not os.path.exists(_checkpoint_path):
                raise FileNotFoundError(f"checkpoint path {checkpoint} does not exist")
            nb_info = {}
            if os.path.exists(os.path.join(_checkpoint_path, "nb-info.json")):
                with open(os.path.join(_checkpoint_path, "nb-info.json"), "r", encoding="utf8") as f:
                    nb_info = json.load(f)
                nb_info = deserialize_nb_info(nb_info)

                # Validate explicit argument's conflicts
                if spec is not None and spec.get("id") is not None and spec.get("id") != nb_info.get("method"):
                    logging.error(f"Method spec's id={spec['id']} is in conflict with the checkpoint's method {nb_info['method']}.")
                if method is not None and method != nb_info.get("method"):
                    logging.error(f"Argument method={method} is in conflict with the checkpoint's method {nb_info['method']}.")

                if method is None and nb_info.get("method") is not None:
                    method = nb_info.get("method")
                ckpt_config_overrides = nb_info.get("config_overrides")

                if presets is None and nb_info.get("applied_presets") is not None:
                    presets = nb_info.get("presets")
                if presets is not None and nb_info.get("applied_presets") is not None:
                    if tuple(sorted(presets)) != tuple(sorted(nb_info.get("applied_presets"))):
                        logging.error(f"Presets {presets} are in conflict with the checkpoint's applied presets {nb_info['applied_presets']}.")
                if config_overrides is None and nb_info.get("config_overrides") is not None:
                    config_overrides = nb_info.get("config_overrides")
                if config_overrides is not None and nb_info.get("config_overrides") is not None:
                    if not _dict_equal(config_overrides, nb_info.get("config_overrides")):
                        logging.error(f"Config overrides {pprint.pformat(config_overrides)} are in conflict with the checkpoint's config overrides {pprint.pformat(nb_info.get('config_overrides'))}.")

            # At this points, we expect config_overrides and presets to be provided by checkpoint or explicitly
            if spec is None and method is not None:
                spec = nerfbaselines.get_method_spec(method)
            if spec is None:
                raise RuntimeError("Neither method ID nor method spec is provided and cannot be inferred from the checkpoint.")
            if method is None:
                method = spec.get("id")
            if method is None:
                raise RuntimeError("Method ID is not specified in the method spec.")

            if presets is None:
                # In case presets are still None here, it means they couldn't be loaded from the checkpoint,
                # perhaps in case of an old checkpoint, and weren't provided explicitly. In that case
                # we reconstruct them using the train dataset (if provided) otherwise we fail here.
                if train_dataset is None:
                    raise RuntimeError("Presets are not provided and cannot be inferred from the checkpoint. Please either manually specify the presets or provide the train dataset to infer them.")
                _presets = get_presets_to_apply(spec, train_dataset["metadata"], presets)
                dataset_overrides = get_config_overrides_from_presets(spec, _presets)
                if train_dataset["metadata"].get("name") is None:
                    logging.warning("Dataset name not specified, dataset-specific config overrides may not be applied")
                if dataset_overrides is not None:
                    dataset_overrides = dataset_overrides.copy()
                    dataset_overrides.update(config_overrides or {})
                    
                    # Validate against current config overrides
                    if ckpt_config_overrides is not None and not _dict_equal(config_overrides, dataset_overrides):
                        logging.error(f"Checkpoint's config overrides {pprint.pformat(ckpt_config_overrides)} are in conflict with the dataset-specific config overrides {pprint.pformat(dataset_overrides)}.")
                del dataset_overrides

            nb_info["method"] = method
            nb_info["config_overrides"] = config_overrides
            nb_info["applied_presets"] = tuple(sorted(_presets))

            # If config overrides match nb-info.json, we do not need to pass them again
            if ckpt_config_overrides is not None and _dict_equal(config_overrides, ckpt_config_overrides):
                config_overrides = None
            else:
                # Log the current set of config overrides
                logging.info(f"Active presets: {', '.join(_presets)}")
                logging.info(f"Previous config overrides: {pprint.pformat(ckpt_config_overrides)}")
                logging.info(f"Overriding config overrides to: {pprint.pformat(config_overrides)}")

            with _build_method_class(spec, backend=backend) as method_cls:
                yield method_cls(
                    checkpoint=str(checkpoint), 
                    train_dataset=train_dataset, 
                    config_overrides=config_overrides), nb_info
            return

    if spec is None and method is not None:
        spec = nerfbaselines.get_method_spec(method)
    if spec is None:
        raise RuntimeError("Either method ID, method spec, or valid checkpoint path must be provided.")
    if method is None:
        method = spec["id"]
    if method is None:
        raise RuntimeError("Method ID is not specified in the method spec.")
    if train_dataset is None:
        raise RuntimeError("Either a checkpoint or train_dataset must be provided")

    # Apply config overrides for the train dataset
    _presets = get_presets_to_apply(spec, train_dataset["metadata"], presets)
    dataset_overrides = get_config_overrides_from_presets(spec, _presets)
    if train_dataset["metadata"].get("name") is None:
        logging.warning("Dataset name not specified, dataset-specific config overrides may not be applied")
    if dataset_overrides is not None:
        dataset_overrides = dataset_overrides.copy()
        dataset_overrides.update(config_overrides or {})
        config_overrides = dataset_overrides
    del dataset_overrides

    # Log the current set of config overrides
    logging.info(f"Active presets: {', '.join(_presets)}")
    logging.info(f"Using config overrides: {pprint.pformat(config_overrides)}")
    nb_info_base = {
        "method": method,
        "config_overrides": config_overrides,
        "applied_presets": tuple(sorted(_presets)),
    }

    with _build_method_class(spec, backend=backend) as method_cls:
        yield method_cls(train_dataset=train_dataset, config_overrides=config_overrides), nb_info_base
