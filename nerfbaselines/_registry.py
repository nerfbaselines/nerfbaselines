import contextlib
import importlib.util
import sys
import logging
import inspect
import os
import importlib
import contextlib
from typing import Optional, Dict, List, cast, Union, TypeVar, FrozenSet
from . import MethodSpec, DatasetSpec, EvaluationProtocolSpec, BackendName, LoggerSpec, DatasetLoaderSpec
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points

T = TypeVar("T")
METHOD_SPECS_PATH = os.path.join(os.path.expanduser("~/.config/nerfbaselines/specs"))
SpecType = Literal["method", "dataset", "dataset_loader", "evaluation_protocol", "logger"]
AnySpec = Union[MethodSpec, DatasetSpec, DatasetLoaderSpec, EvaluationProtocolSpec, LoggerSpec]


def _assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


methods_registry: Dict[str, MethodSpec] = {}
datasets_registry: Dict[str, DatasetSpec] = {}
dataset_loaders_registry: Dict[str, DatasetLoaderSpec] = {}
evaluation_protocols_registry: Dict[str, EvaluationProtocolSpec] = {}
loggers_registry: Dict[str, LoggerSpec] = {}
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
def collect_register_calls(output: List[AnySpec]):
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


def _load_locally_installed_specs():
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
                logging.error(f"Could not load spec file {file} (loaded from {METHOD_SPECS_PATH})")
                continue
            try:
                cfg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cfg)
            except Exception as e:
                logging.exception(e)
                logging.error(f"Could not load spec file {file} (loaded from {METHOD_SPECS_PATH})")
    return output


def _load_specs_from_environment() -> List[MethodSpec]:
    separator = ";" if os.name == "nt" else ":"
    # Register methods from environment variables
    output = []
    with collect_register_calls(output):
        for spec_path in os.environ.get("NERFBASELINES_REGISTER", "").split(separator):  
            spec_path = spec_path.strip()
            if not spec_path:
                # Skip empty definitions
                continue

            if not spec_path.endswith(".py"):
                logging.error(f"Spec file {spec_path} (loaded from NERFBASELINES_REGISTER env variable) is not a python file")
                continue

            if not os.path.exists(spec_path):
                logging.error(f"Spec file {spec_path} (loaded from NERFBASELINES_REGISTER env variable) does not exist")
                continue

            # Import the module from the file
            spec = importlib.util.spec_from_file_location(os.path.split(spec_path)[-1], spec_path)
            if spec is None or spec.loader is None:
                logging.error(f"Could not load spec file {spec_path} (loaded from NERFBASELINES_REGISTER env variable)")
                continue
            try:
                cfg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cfg)
            except Exception as e:
                logging.exception(e)
                logging.error(f"Could not load spec file {spec_path} (loaded from NERFBASELINES_REGISTER env variable)")
    return output


def _load_specs_from_entrypoints() -> List[MethodSpec]:
    output = []
    discovered_entry_points = entry_points(group="nerfbaselines.specs")
    with collect_register_calls(output):
        for _name in discovered_entry_points.names:
            try:
                discovered_entry_points[_name].load()
            except Exception as e:
                logging.exception(e)
                logging.error(f"Could not load spec from entrypoint {_name}")

    # For installed methods, we set the default backend to python, we also drop other backends
    # This is done because the installed packages are assumed to be already in the target environment
    # as in the case of user installing NerfBaselines to their environment.
    for i, spec in enumerate(output):
        if "method_class" in spec:
            # For method spec, we set the default backend to python, we also drop other backends
            spec = spec.copy()
            spec["backends_order"] = ["python"] + [b for b in spec.get("backends_order", []) if b != "python"]
            output[i] = spec
    return output


def _discover_specs():
    """
    Discovers all methods, datasets, evaluation protocols, and loggers from the following sources:
      1) `nerfbaselines.specs` entrypoint
      2) NERFBASELINES_REGISTER environment variable
      3) locally installed methods in ~/.config/nerfbaselines/specs
    """
    output = []
    _registered = {}
    for spec in _load_specs_from_environment():
        spec_type = get_spec_type(spec)
        if (spec["id"], spec_type) in _registered:
            logging.warning(f"Not registering spec {spec['id']} as was already registered in environment variable")
            continue
        _registered[(spec["id"], spec_type)] = "env"
        output.append(spec)

    for spec in _load_specs_from_entrypoints():
        spec_type = get_spec_type(spec)
        source = _registered.get((spec["id"], spec_type))
        if source == "env":
            logging.warning(f"Not registering spec {spec['id']} as was supplied from an environment variable")
            continue
        if source == "entrypoint":
            logging.warning(f"Not registering spec {spec['id']} as was already registered in another entrypoint")
            continue
        _registered[(spec["id"], spec_type)] = "entrypoint"
        output.append(spec)

    # Register locally installed methods
    # We skip the ones registered using entrypoints (can be inside PYTHON backend 
    for spec in _load_locally_installed_specs():
        spec_type = get_spec_type(spec)
        source = _registered.get((spec["id"], spec_type))
        if source is not None:
            logging.debug(f"Not registering spec {spec['id']} as was already registered (source {source})")
            continue
        _registered[(spec["id"], spec_type)] = "local"
        output.append(spec)
    return output


# Auto register
_auto_register_completed = False
_registration_fastpath = None


def _is_registered(spec):
    spec_type = get_spec_type(spec)
    if spec_type == "method":
        return spec["id"] in methods_registry
    elif spec_type == "dataset":
        return spec["id"] in datasets_registry
    elif spec_type == "dataset_loader":
        return spec["id"] in dataset_loaders_registry
    elif spec_type == "evaluation_protocol":
        return spec["id"] in evaluation_protocols_registry
    elif spec_type == "logger":
        return spec["id"] in loggers_registry
    else:
        raise ValueError(f"Could not determine type of object {spec}")


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
            module_spec = importlib.util.find_spec(f"nerfbaselines.methods.{package}", __package__)
            assert module_spec is not None and module_spec.loader is not None, f"Could not find spec for {package}"
            cfg = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(cfg)

    # Dataset registration
    _registration_fastpath = __package__ + ".datasets"
    for package in os.listdir(os.path.join(nb_path, "datasets")):
        if package.endswith("_spec.py") and not package.startswith("_"):
            package = package[:-3]
            module_spec = importlib.util.find_spec(f"nerfbaselines.datasets.{package}", __package__)
            assert module_spec is not None and module_spec.loader is not None, f"Could not find spec for {package}"
            cfg = importlib.util.module_from_spec(module_spec)
            module_spec.loader.exec_module(cfg)

    # Reset the fastpath since we will be loading modules dynamically now
    _registration_fastpath = None

    # Register all external methods
    for spec in _discover_specs():
        if _is_registered(spec):
            logging.warning(f"Skipping registration of {spec['id']} as it would overwrite an existing NerfBaselines object")
            continue
        register(spec)

    # If we restrict methods to some subset, remove all other registered methods from the registry
    _auto_register_completed = True


def _filter_visible_methods(method_ids):
    nb_allowed_methods = os.environ.get("NERFBASELINES_ALLOWED_METHODS", "")
    if not nb_allowed_methods:
        yield from method_ids
    allowed_methods = set(nb_allowed_methods.split(","))
    for m in method_ids:
        if m in allowed_methods:
            yield m


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


def get_spec_type(spec: AnySpec) -> SpecType:
    if "method_class" in spec:
        return "method"
    elif "load_dataset_function" in spec:
        return "dataset_loader"
    elif "download_dataset_function" in spec:
        return "dataset"
    elif "evaluation_protocol_class" in spec:
        return "evaluation_protocol"
    elif "logger_class" in spec:
        return "logger"
    else:
        raise ValueError(f"Could not determine type of object {spec}")


def register(spec: AnySpec) -> None:
    """
    Register a method, dataset, logger, or evaluation protocol spec.

    Args:
        spec: Spec to register (MethodSpec, DatasetSpec, DatasetLoaderSpec, EvaluationProtocolSpec, LoggerSpec)
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
    elif spec_type == "dataset_loader":
        spec = cast("DatasetLoaderSpec", spec)
        if _collected_register_calls is None:
            assert name not in dataset_loaders_registry, f"Dataset loader {name} already registered"
        spec["load_dataset_function"] = _make_entrypoint_absolute(spec["load_dataset_function"])
        if _collected_register_calls is not None:
            _collected_register_calls.append(spec)
        else:
            dataset_loaders_registry[name] = spec
    elif spec_type == "dataset":
        spec = cast("DatasetSpec", spec)
        if _collected_register_calls is None:
            assert name not in datasets_registry, f"Dataset {name} already registered"
        spec["download_dataset_function"] = _make_entrypoint_absolute(spec["download_dataset_function"])
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
    method_ids = _filter_visible_methods(methods_registry.keys())
    if backend_name is None:
        return frozenset(
            method_ids)
    else:
        return frozenset(
            name for name in method_ids if backend_name in get_implemented_backends(get_method_spec(name)))


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


def get_supported_datasets() -> FrozenSet[str]:
    """
    Get all supported datasets.

    Returns:
        Set of dataset IDs
    """
    _auto_register()
    return frozenset(datasets_registry.keys())


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


def get_supported_dataset_loaders() -> FrozenSet[str]:
    """
    Get all supported dataset loaders. The loaders are sorted by priority.

    Returns:
        List of dataset loader IDs (sorted by priority)
    """
    _auto_register()
    return frozenset(dataset_loaders_registry.keys())


def get_dataset_loader_spec(id: str) -> DatasetLoaderSpec:
    """
    Get a dataset loader specification by registered dataset loader ID.

    Args:
        id: Dataset loader ID

    Returns:
        Dataset loader specification
    """
    _auto_register()
    if id not in dataset_loaders_registry:
        raise RuntimeError(f"Could not find dataset loader {id} in registry. Supported dataset loaders: {','.join(dataset_loaders_registry.keys())}")
    return dataset_loaders_registry[id]


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
