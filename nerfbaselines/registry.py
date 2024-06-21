import types
import sys
import logging
import inspect
import os
import importlib
import contextlib
from typing import Optional, Type, Any, Tuple, Dict, List, cast, Union, Sequence, TYPE_CHECKING, Callable, TypeVar

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points
from .types import Method, TypedDict, Required, FrozenSet, NotRequired, LoadDatasetFunction, DownloadDatasetFunction, EvaluationProtocol
from .types import DatasetSpecMetadata, Logger

if TYPE_CHECKING:
    from .backends import BackendName, CondaBackendSpec, DockerBackendSpec, ApptainerBackendSpec
else:
    BackendName = str
    CondaBackendSpec = DockerBackendSpec = ApptainerBackendSpec = dict
T = TypeVar("T")


def assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


class MethodSpec(TypedDict, total=False):
    method: Required[str]
    conda: NotRequired[CondaBackendSpec]
    docker: NotRequired[DockerBackendSpec]
    apptainer: NotRequired[ApptainerBackendSpec]
    kwargs: Dict[str, Any]
    metadata: Dict[str, Any]
    backends_order: List[BackendName]
    dataset_overrides: Dict[str, Any]


class EvaluationProtocolSpec(TypedDict, total=False):
    evaluation_protocol: Required[str]


class EvaluationProtocolWithNameSpec(EvaluationProtocolSpec):
    name: Required[str]


class DatasetSpec(TypedDict, total=False):
    load_dataset_function: Required[str]
    priority: Required[int]
    download_dataset_function: str
    evaluation_protocol: Union[str, EvaluationProtocolWithNameSpec]
    metadata: DatasetSpecMetadata


methods_registry: Dict[str, 'MethodSpec'] = {}
datasets_registry: Dict[str, 'DatasetSpec'] = {}
evaluation_protocols_registry: Dict[str, 'EvaluationProtocolSpec'] = {}
loggers_registry: Dict[str, Callable[..., Logger]] = {}
loggers_registry["wandb"] = lambda *args, **kwargs: _import_type("nerfbaselines.logging:WandbLogger")(*args, **kwargs)
loggers_registry["tensorboard"] = lambda path, **kwargs: _import_type("nerfbaselines.logging:TensorboardLogger")(os.path.join(path, "tensorboard"), **kwargs)
evaluation_protocols_registry["default"] = {"evaluation_protocol": "nerfbaselines.evaluation:DefaultEvaluationProtocol"}
evaluation_protocols_registry["nerf"] = {"evaluation_protocol": "nerfbaselines.evaluation:NerfEvaluationProtocol"}


def _discover_specs() -> List[Tuple[str, "MethodSpec"]]:
    """
    Discovers all methods, datasets, evaluation protocols registered using the `nerfbaselines.specs` entrypoint.
    And also methods, datasets, evaluation protocols in the NERFBASELINES_METHODS, NERFBASELINES_DATASETS, 
    NERFBASELINES_REGISTER environment variables.
    """
    types = []
    discovered_entry_points = entry_points(group="nerfbaselines.specs")
    for name in discovered_entry_points.names:
        spec = discovered_entry_points[name].load()
        if not isinstance(spec, dict):
            logging.warning(f"Could not entry point {spec} as it is not an instance of dict")
            continue
        if "method" not in spec and ("load_dataset_function" not in spec or "priority" not in spec) and "evaluation_protocol" not in spec:
            logging.warning(f"Could not process entry point {spec} as it is not an instance of MethodSpec or DatasetSpec")
            continue
        types.append((name, spec))

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
            modname, qualname_separator, qualname = path.partition(":")
            spec = importlib.import_module(modname)
            if qualname_separator:
                for attr in qualname.split("."):
                    spec = getattr(spec, attr)
            _name: Optional[str] = spec.pop("name", None)
            if name is None and _name is not None:
                name = _name
            if name is None:
                raise ValueError(f"Could not find name for object {spec}")
            assert isinstance(spec, dict), "Invalid instance type"

            # Register based on object type
            if "method" in spec:
                types.append((name, spec))
            elif "load_dataset_function" in spec and "priority" in spec:
                types.append((name, spec))
            elif "evaluation_protocol" in spec:
                types.append((name, spec))
            else:
                raise ValueError(f"Could not determine type of object {spec}")
    except Exception as e:
        logging.exception(e)
        logging.error("Could not load methods from environment variables NERFBASELINES_METHODS, NERFBASELINES_DATASETS, NERFBASELINES_REGISTER")

    return types


def partialmethod(cls, method_name, **kwargs):
    def build(ns):
        cls_dict = cls.__dict__
        ns["__module__"] = cls_dict["__module__"]
        ns["__doc__"] = cls_dict["__doc__"]
        for k, v in kwargs.items():
            if k == "config_overrides":
                continue
            ns[f"_{k}"] = v
        ns["_method_name"] = method_name

        if kwargs.get("config_overrides", None):
            old_init = cls.__init__
            old_config_overrides = kwargs["config_overrides"]
            def wrapped_init(self, *args, **kwargs):
                config_overrides = old_config_overrides.copy()
                if "config_overrides" in kwargs:
                    config_overrides.update(kwargs["config_overrides"])
                return old_init(self, *args, **kwargs)

            wrapped_init.__original_func__ = old_init  # type: ignore
            wrapped_init.__args__ = ()  # type: ignore
            wrapped_init.__kwargs__ = kwargs  # type: ignore
            wrapped_init.__doc__ = old_init.__doc__
            ns["__init__"] = wrapped_init
        return ns

    return types.new_class(cls.__name__, bases=(cls,), exec_body=build)


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
    for name, spec in _discover_specs():
        register(spec, name=name)

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
                module_base = assert_not_none(inspect.getmodule(frame[0])).__name__
                index += 1
            if module == ".":
                module = module_base
            else:
                module = ".".join(module_base.split(".")[:-1]) + module
    return ":".join((module, name))


def register(spec: Union["MethodSpec", "DatasetSpec", "EvaluationProtocolSpec"], *, 
             name: str, 
             metadata=None, 
             kwargs=None,
             dataset_overrides=None) -> None:
    if metadata is None:
        metadata = {}
    spec = spec.copy()
    if "method" in spec:
        assert name not in methods_registry, f"Method {name} already registered"
        spec["method"] = _make_entrypoint_absolute(spec["method"])
        spec.update(
            kwargs={**(spec.get("kwargs") or {}), **(kwargs or {})}, 
            metadata={**(spec.get("metadata") or {}), **(metadata or {})})
        if dataset_overrides is not None:
            spec["dataset_overrides"] = dataset_overrides
        methods_registry[name] = spec
    elif "load_dataset_function" in spec and "priority" in spec:
        assert name not in datasets_registry, f"Dataset {name} already registered"
        assert dataset_overrides is None, "Parameter dataset_overrides is only valid for methods"
        spec["load_dataset_function"] = _make_entrypoint_absolute(spec["load_dataset_function"])
        download_fn = spec.get("download_dataset_function")
        if download_fn is not None:
            spec["download_dataset_function"] = _make_entrypoint_absolute(download_fn)
        eval_protocol = spec.get("evaluation_protocol")
        if isinstance(eval_protocol, dict):
            eval_protocol_name = eval_protocol["name"]
            eval_protocol = cast(EvaluationProtocolSpec, {
                **{k: v for k, v in eval_protocol.items() if k not in ("evaluation_protocol", "name")},
                "evaluation_protocol": _make_entrypoint_absolute(eval_protocol["evaluation_protocol"]),
            })
            register(eval_protocol, name=eval_protocol_name)
            spec["evaluation_protocol"] = eval_protocol_name
        datasets_registry[name] = spec
    elif "evaluation_protocol" in spec:
        assert name not in evaluation_protocols_registry, f"Evaluation protocol {name} already registered"
        assert dataset_overrides is None, "Parameter dataset_overrides is only valid for methods"
        spec["evaluation_protocol"] = _make_entrypoint_absolute(spec["evaluation_protocol"])
        evaluation_protocols_registry[name] = spec
    else:
        raise ValueError(f"Could not determine type of object {spec}")


def register_logger(name: str, logger: Callable[..., Logger]) -> None:
    loggers_registry[name] = logger


def get_method_spec(name: str) -> MethodSpec:
    """
    Get a method by name
    """
    _auto_register()
    if name not in methods_registry:
        raise RuntimeError(f"Method {name} not registered.\nRegistered methods: {','.join(methods_registry.keys())}")
    return methods_registry[name]


def get_dataset_overrides(method_name: str, dataset_metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    spec = get_method_spec(method_name)
    dataset_name = dataset_metadata.get("name")
    scene = dataset_metadata.get("scene")
    dataset_overrides = spec.get("dataset_overrides") or {}
    if f"{dataset_name}/{scene}" in dataset_overrides:
        return dataset_overrides[f"{dataset_name}/{scene}"]
    if dataset_name is not None and dataset_name in dataset_overrides:
        return dataset_overrides[dataset_name]
    return None


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


def _build_method(method_name, spec: "MethodSpec") -> Type[Method]:
    cls = cast(Type[Method], _import_type(spec["method"]))

    # Apply kwargs to the class
    ns = {}
    kwargs = spec.get("kwargs", {}).copy()
    _config_overrides = kwargs.pop("config_overrides", None)
    ns["_method_name"] = method_name
    for k, v in kwargs.items():
        ns[f"_{k}"] = v
    
    # Apply config overrides
    if _config_overrides:
        old_init = cls.__init__
        def __init__(self, *, config_overrides=None, **kwargs):
            co = (config_overrides or {}).copy()
            co.update(_config_overrides)
            old_init(self, **kwargs, config_overrides=co)

        ns["__init__"] = __init__
    newcls = types.new_class(cls.__name__, bases=(cls,), exec_body=lambda _ns: _ns.update(ns))
    newcls.__module__ = cls.__module__
    return newcls


@contextlib.contextmanager
def build_method(method: str, backend: Optional[BackendName] = None):
    from . import backends
    method_spec = methods_registry.get(method)
    if method_spec is None:
        raise RuntimeError(f"Could not find method {method} in registry. Supported methods: {','.join(methods_registry.keys())}")
    backend_impl = backends.get_backend(method_spec, backend)
    logging.info(f"Using method: {method}, backend: {backend_impl.name}")
    with backend_impl:
        backend_impl.install()
        yield cast(Type[Method], backend_impl.wrap(_build_method)(method, method_spec))


def get_supported_datasets() -> FrozenSet[str]:
    _auto_register()
    return frozenset(datasets_registry.keys())


def get_dataset_loaders() -> Sequence[Tuple[str, LoadDatasetFunction]]:
    _auto_register()
    datasets = list(datasets_registry.items())
    datasets.sort(key=lambda x: -x[1]["priority"])
    return [(name, _import_type(spec["load_dataset_function"])) for name, spec in datasets]


def get_dataset_spec(name: str) -> DatasetSpec:
    _auto_register()
    if name not in datasets_registry:
        raise RuntimeError(f"Could not find dataset {name} in registry. Supported datasets: {','.join(datasets_registry.keys())}")
    return datasets_registry[name]


def get_dataset_downloaders() -> Sequence[Tuple[str, DownloadDatasetFunction]]:
    _auto_register()
    datasets = [(k,v) for k,v in datasets_registry.items() if v.get("download_dataset_function") is not None]
    datasets.sort(key=lambda x: -x[1]["priority"])
    return [(name, _import_type(assert_not_none(spec.get("download_dataset_function")))) for name, spec in datasets]


def build_evaluation_protocol(name: str) -> 'EvaluationProtocol':
    _auto_register()
    spec = evaluation_protocols_registry.get(name)
    if spec is None:
        raise RuntimeError(f"Could not find evaluation protocol {name} in registry. Supported protocols: {','.join(evaluation_protocols_registry.keys())}")
    return cast('EvaluationProtocol', _import_type(spec["evaluation_protocol"])())

