import sys
import logging
import inspect
import types
import typing
import os
import importlib
import dataclasses
import subprocess
from typing import Optional, Type, Any, Tuple, Dict, Set, Union
from dataclasses import field, dataclass

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
try:
    from typing import Required
except ImportError:
    from typing_extensions import Required  # type: ignore
try:
    from typing import TypedDict
except ImportError:
    from typing_extensions import TypedDict  # type: ignore
try:
    from typing import FrozenSet
except ImportError:
    from typing_extensions import FrozenSet  # type: ignore
try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args  # type: ignore
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points
from .types import Method
from .backends import DockerMethod, CondaMethod, ApptainerMethod, CondaMethodSpecDict, DockerMethodSpecDict, ApptainerMethodSpecDict
from .utils import partialclass


DEFAULT_DOCKER_IMAGE = "kulhanek/nerfbaselines:v1"
Backend = Literal["conda", "docker", "apptainer", "python"]
ALL_BACKENDS = list(get_args(Backend))
registry = {}


def discover_methods() -> Dict[str, Union["MethodSpec", "MethodSpecDict"]]:
    """
    Discovers all methods registered using the `nerfbaselines.method_configs` entrypoint.
    And also methods in the NERFBASELINES_METHODS environment variable.
    """
    methods = {}
    discovered_entry_points = entry_points(group="nerfbaselines.method_configs")
    for name in discovered_entry_points.names:
        spec = discovered_entry_points[name].load()
        if not isinstance(spec, (MethodSpec, dict)):
            logging.warning(f"Could not entry point {spec} as it is not an instance of {MethodSpec.__name__} or {MethodSpecDict.__name__}")
            continue
        methods[name] = spec

    if "NERFBASELINES_METHODS" in os.environ:
        try:
            strings = os.environ["NERFBASELINES_METHODS"].split(",")
            for definition in strings:
                if not definition:
                    continue
                name, path = definition.split("=")
                logging.info(f"Loading method {name} from environment variable")
                modname, qualname_separator, qualname = path.partition(":")
                method_config = importlib.import_module(modname)
                if qualname_separator:
                    for attr in qualname.split("."):
                        method_config = getattr(method_config, attr)

                # method_config specified as function or class -> instance
                if callable(method_config):
                    method_config = method_config()

                # check for valid instance type
                if not isinstance(spec, (MethodSpec, dict)):
                    raise TypeError(f"Method is not an instance of {MethodSpec.__name__} or {MethodSpecDict.__name__}")

                # save to methods
                methods[name] = method_config
        except Exception as e:
            logging.exception(e)
            logging.error("Could not load methods from environment variable NERFBASELINES_METHODS")

    return methods


# Auto register
_auto_register_completed = False


def auto_register(force=False):
    global _auto_register_completed
    if _auto_register_completed and not force:
        return
    from . import methods

    # TODO: do this more robustly
    for package in os.listdir(os.path.dirname(methods.__file__)):
        if package.endswith(".py") and not package.startswith("_") and package != "__init__.py":
            package = package[:-3]
            importlib.import_module(f".methods.{package}", __package__)
    for name, spec in discover_methods().items():
        register(spec, name)
    _auto_register_completed = True


def register(spec: Union["MethodSpec", "MethodSpecDict"], name: str, *args, metadata=None, **kwargs) -> "MethodSpec":
    if not isinstance(spec, MethodSpec):
        assert isinstance(spec, dict), f"Method spec must be either {MethodSpec.__name__} or {MethodSpecDict.__name__}"
        spec = convert_spec_dict_to_spec(spec)
    assert name not in registry, f"Method {name} already registered"
    if metadata is None:
        metadata = {}
    metadata = {**spec.metadata, **metadata}
    spec = dataclasses.replace(spec, args=spec.args + args, kwargs={**spec.kwargs, **kwargs}, metadata=metadata)
    registry[name] = spec
    return spec


def _make_entrypoint_absolute(entrypoint: str) -> str:
    module, name = entrypoint.split(":")
    if module.startswith("."):
        module_base = current_module = _make_entrypoint_absolute.__module__
        index = 1
        while module_base == current_module:
            frame = inspect.stack()[index]
            module_base = inspect.getmodule(frame[0]).__name__
            index += 1
        if module == ".":
            module = module_base
        else:
            module = ".".join(module_base.split(".")[:-1]) + module
    return ":".join((module, name))


class _LazyMethodMeta(type):
    def __getitem__(cls, __name: str) -> Type[Method]:
        __name = _make_entrypoint_absolute(__name)
        qualname = __name.split(":", maxsplit=1)[-1]
        name = qualname.split(".")[-1]
        module = __name.split(":", maxsplit=1)[0]

        def build(ns):
            def new(ncls, *args, **kwargs):
                old_init = ncls.__init__

                # For partialclass
                if hasattr(old_init, "__original_func__"):
                    args = old_init.__args__ + args
                    kwargs = {**old_init.__kwargs__, **kwargs}

                method = importlib.import_module(module)
                for attr in qualname.split("."):
                    method = getattr(method, attr)

                assert inspect.isclass(method)
                return method(*args, **kwargs)

            ns["__new__"] = new

        ncls = types.new_class(name, exec_body=build, bases=(Method,))
        ncls.__module__ = module
        ncls.__name__ = name
        return typing.cast(Type[Method], ncls)


class LazyMethod(object, metaclass=_LazyMethodMeta):
    def __class_getitem__(cls, __name: Tuple[str, str]) -> Type[Method]:
        return _LazyMethodMeta.__getitem__(cls, __name)


class MethodSpecDict(TypedDict, total=False):
    method: Required[str]
    conda: CondaMethodSpecDict
    docker: DockerMethodSpecDict
    apptainer: ApptainerMethodSpecDict
    args: Tuple[Any, ...]
    kwargs: Dict[str, Any]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class MethodSpec:
    method: Type[Method]
    conda: Optional[Type[CondaMethod]] = None
    docker: Optional[Type[DockerMethod]] = None
    apptainer: Optional[Type[ApptainerMethod]] = None
    args: Tuple[Any, ...] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert self.conda is not None or self.docker is not None, f"{MethodSpec.__name__} requires at least conda or docker backend"
        assert not hasattr(self.method, "method_name"), "method cannot have method_name property"
        assert not hasattr(self.conda, "method_name"), "conda cannot have method_name property"
        assert not hasattr(self.docker, "method_name"), "docker cannot have method_name property"
        assert not hasattr(self.apptainer, "method_name"), "apptainer cannot have method_name property"

    def get_default_backend(self) -> Backend:
        if self.conda is not None:
            return "conda"
        should_install = []
        if self.docker is not None:
            try:
                retcode = subprocess.run(["docker", "-v"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if retcode == 0:
                    return "docker"
            except FileNotFoundError:
                pass
            should_install.append("docker")
        if self.apptainer is not None:
            try:
                retcode = subprocess.run(["apptainer", "-v"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if retcode == 0:
                    return "apptainer"
            except FileNotFoundError:
                pass
            should_install.append("apptainer")
        raise RuntimeError("No backend available, please install " + " or ".join(should_install))

    @property
    def implemented_backends(self) -> FrozenSet[Backend]:
        backends: Set[Backend] = set(("python",))
        if self.conda is not None:
            backends.add("conda")
            backends.add("docker")
            backends.add("apptainer")
        if self.docker is not None:
            backends.add("docker")
            backends.add("apptainer")
        if self.apptainer is not None:
            backends.add("apptainer")
        return frozenset(backends)

    def build(self, *args, backend: Optional[Backend] = None, **kwargs) -> Tuple[Type[Method], Backend]:
        if backend is None:
            backend = self.get_default_backend()
        if backend not in self.implemented_backends:
            raise RuntimeError(f"Backend {backend} is not implemented for selected method.\nImplemented backends: {','.join(self.implemented_backends)}")
        method: Optional[Type[Method]]
        if backend == "python":
            method = self.method
        elif backend == "conda":
            method = self.conda
        elif backend == "docker":
            if self.docker is not None:
                method = self.docker
            elif self.conda is not None:
                method = DockerMethod.wrap(self.conda, image=DEFAULT_DOCKER_IMAGE)
            else:
                raise NotImplementedError()
        elif backend == "apptainer":
            if self.apptainer is not None:
                method = self.apptainer
            elif self.docker is not None:
                method = self.docker.to_apptainer()
            elif self.conda is not None:
                method = ApptainerMethod.wrap(self.conda, image="docker://" + DEFAULT_DOCKER_IMAGE)
            else:
                raise NotImplementedError()
        else:
            raise ValueError(f"Unknown backend {backend}")
        return partialclass(method, *self.args, *args, **self.kwargs, **kwargs), backend

    def register(self, name, *args, **kwargs) -> None:
        register(self, name, *args, **kwargs)


def convert_spec_dict_to_spec(spec: MethodSpecDict) -> MethodSpec:
    kwargs = spec.copy()

    kwargs["method"] = method = LazyMethod[spec["method"]]
    if "conda" in kwargs:
        kwargs["conda"] = CondaMethod.wrap(
            method,
            **kwargs.pop("conda"),
        )
    if "docker" in kwargs:
        kwargs["docker"] = DockerMethod.wrap(
            method,
            **kwargs.pop("docker"),
        )
    if "apptainer" in kwargs:
        kwargs["apptainer"] = ApptainerMethod.wrap(
            method,
            **kwargs.pop("apptainer"),
        )
    return MethodSpec(**kwargs)


def get(name: str) -> MethodSpec:
    auto_register()
    if name not in registry:
        raise RuntimeError(f"Method {name} not registered.\nRegistered methods: {','.join(registry.keys())}")
    return registry[name]


def supported_methods() -> FrozenSet[str]:
    auto_register()
    return frozenset(registry.keys())
