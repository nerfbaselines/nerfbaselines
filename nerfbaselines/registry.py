import inspect
import types
import typing
import os
import importlib
import dataclasses
import subprocess
from typing import Optional, Type, Any, Tuple, Dict, Set

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal  # type: ignore
try:
    from typing import FrozenSet
except ImportError:
    from typing_extensions import FrozenSet  # type: ignore
try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args  # type: ignore
from dataclasses import dataclass, field
from .types import Method
from .backends import DockerMethod, CondaMethod, ApptainerMethod
from .utils import partialclass


DEFAULT_DOCKER_IMAGE = "kulhanek/nerfbaselines:v1"
Backend = Literal["conda", "docker", "apptainer", "python"]
ALL_BACKENDS = list(get_args(Backend))
registry = {}


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
    _auto_register_completed = True


def register(spec: "MethodSpec", name: str, *args, metadata=None, **kwargs) -> "MethodSpec":
    assert name not in registry, f"Method {name} already registered"
    if metadata is None:
        metadata = {}
    metadata = {**spec.metadata, **metadata}
    spec = dataclasses.replace(spec, args=spec.args + args, kwargs={**spec.kwargs, **kwargs}, metadata=metadata)
    registry[name] = spec
    return spec


class _LazyMethodMeta(type):
    def __getitem__(cls, __name: Tuple[str, str]) -> Type[Method]:
        from . import methods

        module, name = __name
        module_base = methods.__package__

        def build(ns):
            def new(ncls, *args, **kwargs):
                old_init = ncls.__init__

                # For partialclass
                if hasattr(old_init, "__original_func__"):
                    args = old_init.__args__ + args
                    kwargs = {**old_init.__kwargs__, **kwargs}

                mod = importlib.import_module(module, methods.__package__)
                ncls = getattr(mod, name)
                assert inspect.isclass(ncls)
                return ncls(*args, **kwargs)

            ns["__new__"] = new

        ncls = types.new_class(name, exec_body=build, bases=(Method,))
        ncls.__module__ = module_base + module if module.startswith(".") else module
        ncls.__name__ = name
        return typing.cast(Type[Method], ncls)


class LazyMethod(object, metaclass=_LazyMethodMeta):
    def __class_getitem__(cls, __name: Tuple[str, str]) -> Type[Method]:
        return _LazyMethodMeta.__getitem__(cls, __name)


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
        assert self.conda is not None or self.docker is not None, "MethodSpec requires at least conda or docker backend"
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


def get(name: str) -> MethodSpec:
    auto_register()
    if name not in registry:
        raise RuntimeError(f"Method {name} not registered.\nRegistered methods: {','.join(registry.keys())}")
    return registry[name]


def supported_methods() -> FrozenSet[str]:
    auto_register()
    return frozenset(registry.keys())
