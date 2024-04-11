import contextlib
import types
import sys
import logging
import inspect
import os
import importlib
from typing import Optional, Type, Any, Tuple, Dict, List, cast

if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points
from .types import Method, TypedDict, Required, FrozenSet, NotRequired
from .backends import BackendName
from .backends import CondaBackendSpec, DockerBackendSpec, ApptainerBackendSpec
from . import backends
from .utils import assert_not_none


registry: Dict[str, 'MethodSpec'] = {}


def _discover_methods() -> List[Tuple[str, "MethodSpec"]]:
    """
    Discovers all methods registered using the `nerfbaselines.method_configs` entrypoint.
    And also methods in the NERFBASELINES_METHODS environment variable.
    """
    methods = []
    discovered_entry_points = entry_points(group="nerfbaselines.method_configs")
    for name in discovered_entry_points.names:
        spec = discovered_entry_points[name].load()
        if not isinstance(spec, dict):
            logging.warning(f"Could not entry point {spec} as it is not an instance of {MethodSpec.__name__}")
            continue
        methods.append((name, spec))

    if "NERFBASELINES_METHODS" in os.environ:
        try:
            strings = os.environ["NERFBASELINES_METHODS"].split(",")
            for definition in strings:
                if not definition:
                    continue
                name, path = None, definition
                if "=" in definition:
                    name, path = definition.split("=")
                logging.info(f"Loading method {name} from environment variable")
                modname, qualname_separator, qualname = path.partition(":")
                spec = importlib.import_module(modname)
                if qualname_separator:
                    for attr in qualname.split("."):
                        spec = getattr(spec, attr)
                _name: Optional[str] = spec.pop("name")
                if name is None and _name is not None:
                    name = _name
                if name is None:
                    raise ValueError(f"Could not find name for method {spec}")

                # check for valid instance type
                if not isinstance(spec, dict):
                    raise TypeError(f"Method is not an instance of {MethodSpec.__name__}")

                # save to methods
                methods.append((name, spec))
        except Exception as e:
            logging.exception(e)
            logging.error("Could not load methods from environment variable NERFBASELINES_METHODS")

    return methods


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
    from . import methods

    # TODO: do this more robustly
    assert __package__ is not None, "Package must be set"
    _registration_fastpath = __package__ + ".methods"
    for package in os.listdir(os.path.dirname(methods.__file__)):
        if package.endswith(".py") and not package.startswith("_") and package != "__init__.py":
            package = package[:-3]
            importlib.import_module(f".methods.{package}", __package__)
    # Reset the fastpath since we will be loading modules dynamically now
    _registration_fastpath = None

    # If we restrict methods to some subset, remove all other registered methods from the registry
    allowed_methods = set(v for v in os.environ.get("NERFBASELINES_ALLOWED_METHODS", "").split(",") if v)
    if allowed_methods:
        for k in list(registry.keys()):
            if k not in allowed_methods:
                del registry[k]


    for name, spec in _discover_methods():
        register(spec, name=name)
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


class MethodSpec(TypedDict, total=False):
    method: Required[str]
    conda: NotRequired[CondaBackendSpec]
    docker: NotRequired[DockerBackendSpec]
    apptainer: NotRequired[ApptainerBackendSpec]
    kwargs: Dict[str, Any]
    metadata: Dict[str, Any]
    backends_order: List[BackendName]


def register(spec: "MethodSpec", *, name: str, metadata=None, kwargs=None) -> None:
    assert spec.get("conda") is not None or spec.get("docker") is not None, "MethodSpec requires at least conda or docker backend"
    if metadata is None:
        metadata = {}
    spec = spec.copy()
    spec["method"] = _make_entrypoint_absolute(spec["method"])
    spec.update(
        kwargs={**(spec.get("kwargs") or {}), **(kwargs or {})}, 
        metadata={**(spec.get("metadata") or {}), **(metadata or {})})
    assert name not in registry, f"Method {name} already registered"
    registry[name] = spec


def get(name: str) -> MethodSpec:
    _auto_register()
    if name not in registry:
        raise RuntimeError(f"Method {name} not registered.\nRegistered methods: {','.join(registry.keys())}")
    return registry[name]


def supported_methods(backend_name: Optional[BackendName] = None) -> FrozenSet[str]:
    _auto_register()
    if backend_name is None:
        return frozenset(registry.keys())
    else:
        return frozenset(name for name, spec in registry.items() if backend_name in backends._get_implemented_backends(spec))


def _build_method(method_name, spec: "MethodSpec") -> Type[Method]:
    package, name = spec["method"].split(":")
    cls = cast(Type[Method], importlib.import_module(package))
    for p in name.split("."):
        cls = cast(Type[Method], getattr(cls, p))

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
    method_spec = registry.get(method)
    if method_spec is None:
        raise RuntimeError(f"Could not find method {method} in registry. Supported methods: {','.join(registry.keys())}")
    backend_impl = backends.get_backend(method_spec, backend)
    logging.info(f"Using method: {method}, backend: {backend_impl.name}")
    with backend_impl as _backend_imple_active:
        backend_impl.install()
        yield cast(Type[Method], backend_impl.wrap(_build_method)(method, method_spec))
