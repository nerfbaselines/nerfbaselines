import contextlib
import importlib.util
import types
import sys
import logging
import inspect
import os
import importlib
import contextlib
from typing import Optional, Any, Tuple, Dict, List, cast, Union, TypeVar, FrozenSet
from . import MethodSpec, DatasetSpec, EvaluationProtocolSpec, BackendName, LoggerSpec
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

T = TypeVar("T")
METHOD_SPECS_PATH = os.path.join(os.path.expanduser("~/.config/nerfbaselines/methods"))
SpecType = Literal["method", "dataset", "evaluation_protocol", "logger"]


def _assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


methods_registry: Dict[str, 'MethodSpec'] = {}
datasets_registry: Dict[str, 'DatasetSpec'] = {}
evaluation_protocols_registry: Dict[str, 'EvaluationProtocolSpec'] = {}
loggers_registry: Dict[str, 'LoggerSpec'] = {}
loggers_registry["wandb"] = {
    "id": "wandb", "logger_class": "nerfbaselines.logging:WandbLogger" }
loggers_registry["tensorboard"] = { 
    "id": "tensorboard", "logger_class": "nerfbaselines.logging:TensorboardLogger" }
evaluation_protocols_registry["default"] = {
    "id": "default", 
    "evaluation_protocol_class": "nerfbaselines.evaluation:DefaultEvaluationProtocol"}
evaluation_protocols_registry["nerf"] = {
    "id": "nerf", 
    "evaluation_protocol_class": "nerfbaselines.evaluation:NerfEvaluationProtocol"}
_collected_register_calls = None


@contextlib.contextmanager
def collect_register_calls(output: List[Union["MethodSpec", "DatasetSpec", "EvaluationProtocolSpec", "LoggerSpec"]]):
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
                if "method_class" in spec:
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
    nb_path = os.path.dirname(os.path.abspath(__file__))

    assert __package__ is not None, "Package must be set"
    # Method registration
    _registration_fastpath = __package__ + ".methods"
    for package in os.listdir(os.path.join(nb_path, "methods")):
        if package.endswith("_spec.py") and not package.startswith("_"):
            package = package[:-3]
            importlib.import_module(f".methods.{package}", __package__)

    # Dataset registration
    _registration_fastpath = __package__ + ".datasets"
    for package in os.listdir(os.path.join(nb_path, "datasets")):
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


def get_spec_type(spec: Union["MethodSpec", "DatasetSpec", "EvaluationProtocolSpec", "LoggerSpec"]) -> SpecType:
    if "method_class" in spec:
        return "method"
    elif "load_dataset_function" in spec and "priority" in spec:
        return "dataset"
    elif "evaluation_protocol_class" in spec:
        return "evaluation_protocol"
    elif "logger_class" in spec:
        return "logger"
    else:
        raise ValueError(f"Could not determine type of object {spec}")


def register(spec: Union["MethodSpec", "DatasetSpec", "EvaluationProtocolSpec", "LoggerSpec"]) -> None:
    """
    Register a method, dataset, logger, or evaluation protocol spec.

    Args:
        spec: Spec to register (MethodSpec, DatasetSpec, EvaluationProtocolSpec, LoggerSpec)
    """
    spec = spec.copy()
    spec_type = get_spec_type(spec)
    if "id" not in spec:
        raise ValueError(f"Spec does not have an id: {spec}")
    name = spec.get("id")
    if spec_type == "method":
        spec = cast("MethodSpec", spec)
        if _collected_register_calls is None:
            assert name not in methods_registry, f"Method {name} already registered"
        spec["method_class"] = _make_entrypoint_absolute(spec["method_class"])
        if _collected_register_calls is not None:
            _collected_register_calls.append(spec)
        else:
            methods_registry[name] = spec
    elif spec_type == "logger":
        spec = cast("LoggerSpec", spec)
        if _collected_register_calls is None:
            assert name not in loggers_registry, f"Logger {name} already registered"
        spec["logger_class"] = _make_entrypoint_absolute(spec["logger_class"])
        if _collected_register_calls is not None:
            _collected_register_calls.append(spec)
        else:
            loggers_registry[name] = spec
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
        spec["evaluation_protocol_class"] = _make_entrypoint_absolute(spec["evaluation_protocol_class"])
        if _collected_register_calls is not None:
            _collected_register_calls.append(spec)
        else:
            evaluation_protocols_registry[name] = spec
    else:
        raise ValueError(f"Could not determine type of object {spec}")


def get_supported_methods(backend_name: Optional[BackendName] = None) -> FrozenSet[str]:
    """
    Get all supported methods. Optionally, filter the methods that support a specific backend.

    Args:
        backend_name: Backend name

    Returns:
        Set of method IDs
    """
    from .backends import get_implemented_backends
    _auto_register()
    if backend_name is None:
        return frozenset(methods_registry.keys())
    else:
        return frozenset(name for name, spec in methods_registry.items() if backend_name in get_implemented_backends(spec))


def get_method_spec(id: str) -> MethodSpec:
    """
    Get a method by method ID.

    Args:
        id: Method ID

    Returns:
        Method spec
    """
    _auto_register()
    if id not in methods_registry:
        raise RuntimeError(f"Method {id} not registered.\nRegistered methods: {','.join(methods_registry.keys())}")
    return methods_registry[id]


def get_supported_datasets(automatic_download: bool = False) -> List[str]:
    """
    Get all supported datasets. Optionally, filter the datasets that support automatic download.
    The list of supported datasets is sorted by priority.

    Args:
        automatic_download: If True, only return datasets that support automatic download

    Returns:
        List of dataset IDs (sorted by priority)
    """
    _auto_register()
    datasets = list(datasets_registry.keys())
    if automatic_download:
        datasets = [k for k in datasets if datasets_registry[k].get("download_dataset_function") is not None]
    datasets.sort(key=lambda x: datasets_registry[x]["priority"])
    return datasets


def get_dataset_spec(id: str) -> DatasetSpec:
    """
    Get a dataset specification by registered dataset ID.

    Args:
        id: Dataset ID

    Returns:
        Dataset specification
    """
    _auto_register()
    if id not in datasets_registry:
        raise RuntimeError(f"Could not find dataset {id} in registry. Supported datasets: {','.join(datasets_registry.keys())}")
    return datasets_registry[id]


def get_supported_loggers() -> FrozenSet[str]:
    """
    Get all supported loggers.

    Returns:
        Set of logger IDs
    """
    _auto_register()
    return frozenset(loggers_registry.keys())


def get_logger_spec(id: str) -> LoggerSpec:
    """
    Get a logger specification by registered logger ID.

    Args:
        id: Logger ID

    Returns:
        Logger specification
    """
    _auto_register()
    if id not in loggers_registry:
        raise RuntimeError(f"Could not find logger {id} in registry. Supported loggers: {','.join(loggers_registry.keys())}")
    return loggers_registry[id]


def get_supported_evaluation_protocols() -> FrozenSet[str]:
    """
    Get all supported evaluation protocols.

    Returns:
        Set of evaluation protocol IDs
    """
    _auto_register()
    return frozenset(evaluation_protocols_registry.keys())


def get_evaluation_protocol_spec(id: str) -> EvaluationProtocolSpec:
    """
    Get an evaluation protocol specification by registered evaluation protocol ID.

    Args:
        id: Evaluation protocol ID

    Returns:
        Evaluation protocol specification
    """
    _auto_register()
    if id not in evaluation_protocols_registry:
        raise RuntimeError(f"Could not find evaluation protocol {id} in registry. Supported evaluation protocols: {','.join(evaluation_protocols_registry.keys())}")
    return evaluation_protocols_registry[id]
