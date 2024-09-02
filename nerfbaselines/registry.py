import functools
import contextlib
import importlib.util
import types
import sys
import logging
import inspect
import os
import importlib
import contextlib
from typing import Optional, Type, Any, Tuple, Dict, List, cast, Union, Sequence, TYPE_CHECKING, Callable, TypeVar, Set
from .types import Method, TypedDict, Required, FrozenSet, NotRequired, LoadDatasetFunction, DownloadDatasetFunction, EvaluationProtocol
from .types import DatasetSpecMetadata, Logger, Literal, DatasetFeature, CameraModel, RenderOutputType, MethodSpec, DatasetSpec, EvaluationProtocolSpec, BackendName

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

T = TypeVar("T")
METHOD_SPECS_PATH = os.path.join(os.path.expanduser("~/.config/nerfbaselines/methods"))
SpecType = Literal["method", "dataset", "evaluation_protocol"]


def _assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


methods_registry: Dict[str, 'MethodSpec'] = {}
datasets_registry: Dict[str, 'DatasetSpec'] = {}
evaluation_protocols_registry: Dict[str, 'EvaluationProtocolSpec'] = {}
loggers_registry: Dict[str, Callable[..., Logger]] = {}
loggers_registry["wandb"] = lambda *args, **kwargs: _import_type("nerfbaselines.logging:WandbLogger")(*args, **kwargs)
loggers_registry["tensorboard"] = lambda path, **kwargs: _import_type("nerfbaselines.logging:TensorboardLogger")(os.path.join(path, "tensorboard"), **kwargs)
evaluation_protocols_registry["default"] = {
    "id": "default", 
    "evaluation_protocol": "nerfbaselines.evaluation:DefaultEvaluationProtocol"}
evaluation_protocols_registry["nerf"] = {
    "id": "nerf", 
    "evaluation_protocol": "nerfbaselines.evaluation:NerfEvaluationProtocol"}
_collected_register_calls = None


@contextlib.contextmanager
def collect_register_calls(output: List[Union["MethodSpec", "DatasetSpec", "EvaluationProtocolSpec"]]):
    """
    Context manager to disable and collect all calls to nerfbaselines.registry.register

    Args:
        output: List to which the calls will be appended
    """
    global _collected_register_calls
    old = _collected_register_calls
    try:
        _collected_register_calls = output
        yield
    finally:
        _collected_register_calls = old


def _load_locally_installed_methods():
    if not os.path.exists(METHOD_SPECS_PATH):
        return []
    # Register locally installed methods
    output = []
    with collect_register_calls(output):
        for file in os.listdir(METHOD_SPECS_PATH):
            if not file.endswith(".py"):
                continue
            # Import the module from the file
            path = os.path.join(METHOD_SPECS_PATH, file)
            spec = importlib.util.spec_from_file_location(file, path)
            if spec is None or spec.loader is None:
                continue
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)
    return output


def _discover_specs() -> List[Tuple["MethodSpec", str]]:
    """
    Discovers all methods, datasets, evaluation protocols registered using the `nerfbaselines.specs` entrypoint.
    And also methods, datasets, evaluation protocols in the NERFBASELINES_METHODS, NERFBASELINES_DATASETS, 
    NERFBASELINES_REGISTER environment variables.
    """
    global _collected_register_calls
    _types = []
    registered_methods = set()

    types_to_register = (
        os.environ.get("NERFBASELINES_METHODS", "") + "," +
        os.environ.get("NERFBASELINES_DATASETS", "") + "," +
        os.environ.get("NERFBASELINES_REGISTER", "")
    )
    try:
        for definition in types_to_register.split(","):
            if not definition:
                continue
            name, path = None, definition
            if "=" in definition:
                name, path = definition.split("=")
            logging.info(f"Loading object {name} from environment variable")
            register_calls = []
            with collect_register_calls(register_calls):
                modname, qualname_separator, qualname = path.partition(":")
                spec = importlib.import_module(modname)
                if qualname_separator:
                    for attr in qualname.split("."):
                        spec = getattr(spec, attr)

            # If spec is a module, we register all register_calls
            if isinstance(spec, types.ModuleType):
                assert name is None, "If the registered type is module, no name should be provided"
                for _spec in register_calls:
                    _types.append((_spec, "environment_variable"))
                    del _spec
            else:
                # If it's a dict, we register it directly
                # Register based on object type
                assert "id" in spec, "Spec does not have an id"
                get_spec_type(cast(Any, spec))
                _types.append((spec, "environment_variable"))
    except Exception as e:
        logging.exception(e)
        logging.error("Could not load methods from environment variables NERFBASELINES_METHODS, NERFBASELINES_DATASETS, NERFBASELINES_REGISTER")
    registered_methods = set((spec["id"], get_spec_type(spec)) for spec, _ in _types)

    discovered_entry_points = entry_points(group="nerfbaselines.specs")
    for name in discovered_entry_points.names:
        temptypes = []
        with collect_register_calls(temptypes):
            spec = discovered_entry_points[name].load()
            for spec in temptypes:
                name = spec["id"]
                if "method" in spec:
                    # For method spec, we set the default backend to python, we also drop other backends
                    spec = spec.copy()
                    spec["backends_order"] = ["python"] + [b for b in spec.get("backends_order", []) if b != "python"]
                if (name, get_spec_type(spec)) in registered_methods:
                    logging.warning(f"Not registering spec {name} as was supplied from an environment variable")
                    continue
                _types.append((spec, "spec"))
    registered_methods = set((spec["id"], get_spec_type(spec)) for spec, _ in _types)

    # Register locally installed methods
    # We skip the ones registered using entrypoints (can be inside PYTHON backend 
    for spec in _load_locally_installed_methods():
        name = spec["id"]
        if (name, get_spec_type(spec)) in registered_methods:
            logging.debug(f"Skipping locally installed spec {name} as it is already registered")
            continue
        _types.append((spec, "local"))
    return _types


# Auto register
_auto_register_completed = False
_registration_fastpath = None


def _auto_register(force=False):
    global _auto_register_completed
    global _registration_fastpath
    if _auto_register_completed and not force:
        return
    from . import methods, datasets

    assert __package__ is not None, "Package must be set"
    # Method registration
    _registration_fastpath = __package__ + ".methods"
    for package in os.listdir(os.path.dirname(methods.__file__)):
        if package.endswith("_spec.py") and not package.startswith("_"):
            package = package[:-3]
            importlib.import_module(f".methods.{package}", __package__)

    # Dataset registration
    _registration_fastpath = __package__ + ".datasets"
    for package in os.listdir(os.path.dirname(datasets.__file__)):
        if package.endswith("_spec.py") and not package.startswith("_"):
            package = package[:-3]
            importlib.import_module(f".datasets.{package}", __package__)
    # Reset the fastpath since we will be loading modules dynamically now
    _registration_fastpath = None

    # Register all external methods
    for spec, _ in _discover_specs():
        register(spec)

    # If we restrict methods to some subset, remove all other registered methods from the registry
    allowed_methods = set(v for v in os.environ.get("NERFBASELINES_ALLOWED_METHODS", "").split(",") if v)
    if allowed_methods:
        for k in list(methods_registry.keys()):
            if k not in allowed_methods:
                del methods_registry[k]

    _auto_register_completed = True


def _make_entrypoint_absolute(entrypoint: str) -> str:
    module, name = entrypoint.split(":")
    if module.startswith("."):
        if _registration_fastpath is not None and module != ".":
            module_base = _registration_fastpath
            module = module_base + module
        else:
            module_base = current_module = _make_entrypoint_absolute.__module__
            index = 1
            while module_base == current_module:
                frame = inspect.stack()[index]
                module_base = _assert_not_none(inspect.getmodule(frame[0])).__name__
                index += 1
            if module == ".":
                module = module_base
            else:
                module = ".".join(module_base.split(".")[:-1]) + module
    return ":".join((module, name))


def get_spec_type(spec: Union["MethodSpec", "DatasetSpec", "EvaluationProtocolSpec"]) -> SpecType:
    if "method" in spec:
        return "method"
    elif "load_dataset_function" in spec and "priority" in spec:
        return "dataset"
    elif "evaluation_protocol" in spec:
        return "evaluation_protocol"
    else:
        raise ValueError(f"Could not determine type of object {spec}")


def register(spec: Union["MethodSpec", "DatasetSpec", "EvaluationProtocolSpec"]) -> None:
    spec = spec.copy()
    spec_type = get_spec_type(spec)
    if "id" not in spec:
        raise ValueError(f"Spec does not have an id: {spec}")
    name = spec.get("id")
    if spec_type == "method":
        spec = cast("MethodSpec", spec)
        if _collected_register_calls is None:
            assert name not in methods_registry, f"Method {name} already registered"
        spec["method"] = _make_entrypoint_absolute(spec["method"])
        if _collected_register_calls is not None:
            _collected_register_calls.append(spec)
        else:
            methods_registry[name] = spec
    elif spec_type == "dataset":
        spec = cast("DatasetSpec", spec)
        if _collected_register_calls is None:
            assert name not in datasets_registry, f"Dataset {name} already registered"
        spec["load_dataset_function"] = _make_entrypoint_absolute(spec["load_dataset_function"])
        download_fn = spec.get("download_dataset_function")
        if download_fn is not None:
            spec["download_dataset_function"] = _make_entrypoint_absolute(download_fn)
        eval_protocol = spec.get("evaluation_protocol")
        if isinstance(eval_protocol, dict):
            eval_protocol_name = eval_protocol["id"]
            register(eval_protocol)
            spec["evaluation_protocol"] = eval_protocol_name
        if _collected_register_calls is not None:
            _collected_register_calls.append(spec)
        else:
            datasets_registry[name] = spec
    elif spec_type == "evaluation_protocol":
        spec = cast("EvaluationProtocolSpec", spec)
        if _collected_register_calls is None:
            assert name not in evaluation_protocols_registry, f"Evaluation protocol {name} already registered"
        spec["evaluation_protocol"] = _make_entrypoint_absolute(spec["evaluation_protocol"])
        if _collected_register_calls is not None:
            _collected_register_calls.append(spec)
        else:
            evaluation_protocols_registry[name] = spec
    else:
        raise ValueError(f"Could not determine type of object {spec}")


def register_logger(id: str, logger: Callable[..., Logger]) -> None:
    loggers_registry[id] = logger


def get_method_spec(id: str) -> MethodSpec:
    """
    Get a method by name
    """
    _auto_register()
    if id not in methods_registry:
        raise RuntimeError(f"Method {id} not registered.\nRegistered methods: {','.join(methods_registry.keys())}")
    return methods_registry[id]


def get_config_overrides_from_presets(spec: MethodSpec, presets: Union[Set[str], Sequence[str]]) -> Dict[str, Any]:
    """
    Apply presets to a method spec and return the config overrides.

    Args:
        spec: Method spec
        presets: List of presets to apply

    Returns:
        A dictionary of config overrides
    """
    _config_overrides = {}
    _presets = set(presets)
    for preset_name, preset in spec.get("presets", {}).items():
        if preset_name not in _presets:
            continue
        _config_overrides.update({
            k: v for k, v in preset.items()
            if not k.startswith("@")
        })
    return _config_overrides


def get_presets_to_apply(spec: MethodSpec, dataset_metadata: Dict[str, Any], presets: Union[Set[str], Sequence[str], None] = None) -> Set[str]:
    """
    Given a method spec, dataset metadata, and the optional list of presets from the user,
    this function returns the list of presets that should be applied.

    Args:
        spec: Method spec
        dataset_metadata: Dataset metadata
        presets: List of presets to apply or a special "@auto" preset that will automatically apply presets based on the dataset metadata

    Returns:
        List of presets to apply
    """
    # Validate presets for MethodSpec
    auto_presets = presets is None
    _presets = set(presets or ())
    _condition_data = dataset_metadata.copy()
    _condition_data["dataset"] = _condition_data.pop("name", "")

    for preset in presets or []:
        if preset == "@auto":
            if auto_presets:
                raise ValueError("Cannot specify @auto preset multiple times")
            auto_presets = True
            _presets.remove("@auto")
            continue
        if preset not in spec.get("presets", {}):
            raise ValueError(f"Preset {preset} not found in method spec {spec['id']}. Available presets: {','.join(spec.get('presets', {}).keys())}")
    if auto_presets:
        for preset_name, preset in spec.get("presets", {}).items():
            apply = preset.get("@apply", [])
            if not apply:
                continue
            for condition in apply:
                if all(_condition_data.get(k, "") == v for k, v in condition.items()):
                    _presets.add(preset_name)
    return _presets


def get_supported_methods(backend_name: Optional[BackendName] = None) -> FrozenSet[str]:
    from . import backends
    _auto_register()
    if backend_name is None:
        return frozenset(methods_registry.keys())
    else:
        return frozenset(name for name, spec in methods_registry.items() if backend_name in backends._get_implemented_backends(spec))


def _import_type(name: str) -> Any:
    package, name = name.split(":")
    obj: Any = importlib.import_module(package)
    for p in name.split("."):
        obj = getattr(obj, p)
    return obj


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


def _build_method(spec: "MethodSpec") -> Type[Method]:
    cls = cast(Type[Method], _import_type(spec["method"]))
    newcls = _wrap_method_class(cls, spec)
    return newcls


@contextlib.contextmanager
def build_method(spec: MethodSpec, backend: Optional[BackendName] = None):
    from . import backends
    backend_impl = backends.get_backend(spec, backend)
    method = spec["id"]
    logging.info(f"Using method: {method}, backend: {backend_impl.name}")
    with backend_impl:
        backend_impl.install()
        yield cast(Type[Method], backend_impl.static_call(f"{_build_method.__module__}:{_build_method.__name__}", spec))


def get_supported_datasets() -> FrozenSet[str]:
    _auto_register()
    return frozenset(datasets_registry.keys())


def get_dataset_loaders() -> Sequence[Tuple[str, LoadDatasetFunction]]:
    _auto_register()
    datasets = list(datasets_registry.items())
    datasets.sort(key=lambda x: -x[1]["priority"])
    return [(name, _import_type(spec["load_dataset_function"])) for name, spec in datasets]


def get_dataset_spec(id: str) -> DatasetSpec:
    _auto_register()
    if id not in datasets_registry:
        raise RuntimeError(f"Could not find dataset {id} in registry. Supported datasets: {','.join(datasets_registry.keys())}")
    return datasets_registry[id]


def get_dataset_downloaders() -> Dict[str, DownloadDatasetFunction]:
    _auto_register()
    datasets = [(k,v) for k,v in datasets_registry.items() if v.get("download_dataset_function") is not None]
    datasets.sort(key=lambda x: -x[1]["priority"])
    return {name: _import_type(_assert_not_none(spec.get("download_dataset_function"))) for name, spec in datasets}


def build_evaluation_protocol(id: str) -> 'EvaluationProtocol':
    _auto_register()
    spec = evaluation_protocols_registry.get(id)
    if spec is None:
        raise RuntimeError(f"Could not find evaluation protocol {id} in registry. Supported protocols: {','.join(evaluation_protocols_registry.keys())}")
    return cast('EvaluationProtocol', _import_type(spec["evaluation_protocol"])())

