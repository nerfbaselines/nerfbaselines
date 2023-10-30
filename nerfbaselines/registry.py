import os
import importlib
import dataclasses
import subprocess
import typing
from typing import Optional, Type, Literal, Any, Tuple, Dict
from dataclasses import dataclass, field
from .types import Method
from .backends import DockerMethod, CondaMethod, ApptainerMethod
from .utils import partialclass


DEFAULT_DOCKER_IMAGE = "kulhanek/nerfbaselines:latest"
Backend = Literal["conda", "docker", "apptainer", "python"]
ALL_BACKENDS = list(typing.get_args(Backend))
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
        if package.endswith(".py") and package != "__init__.py":
            package = package[:-3]
            importlib.import_module(f".methods.{package}", __package__)
    _auto_register_completed = True


def register(spec: 'MethodSpec', name: str, *args, **kwargs) -> 'MethodSpec':
    assert name not in registry, f"Method {name} already registered"
    spec = dataclasses.replace(spec, args=spec.args + args, kwargs={**spec.kwargs, **kwargs})
    registry[name] = spec
    return spec


@dataclass(frozen=True)
class MethodSpec:
    method: Method
    conda: Optional[Type[CondaMethod]] = None
    docker: Optional[Type[DockerMethod]] = None
    apptainer: Optional[Type[ApptainerMethod]] = None
    args: Tuple[Any] = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)

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
            retcode = subprocess.call(["docker", "-v"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if retcode == 0:
                return "docker"
            should_install.append("docker")
        if self.apptainer is not None:
            retcode = subprocess.call(["apptainer", "-v"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if retcode == 0:
                return "apptainer"
            should_install.append("apptainer")
        raise RuntimeError("No backend available, please install " + " or ".join(should_install))

    @property
    def implemented_backends(self) -> set[Backend]:
        backends = set("python")
        if self.conda is not None:
            backends.add("conda")
            backends.add("docker")
            backends.add("apptainer")
        if self.docker is not None:
            backends.add("docker")
        if self.apptainer is not None:
            backends.add("apptainer")
        return backends

    def build(self, *args, backend: Optional[Backend] = None, **kwargs) -> Tuple[Method, Backend]:
        if backend is None:
            backend = self.get_default_backend()
        if backend not in self.implemented_backends:
            raise RuntimeError(f"Backend {backend} is not implemented for selected method.\nImplemented backends: {','.join(self.implemented_backends)}")
        if backend == "python":
            method = self.method
        elif backend == "conda":
            method = self.conda
        elif backend == "docker":
            if self.docker is not None:
                method = self.docker
            elif self.conda is not None:
                method = DockerMethod.wrap(
                    self.method,
                    image=DEFAULT_DOCKER_IMAGE)
            else:
                raise NotImplementedError()
        elif backend == "apptainer":
            if self.apptainer is not None:
                method = self.apptainer
            elif self.conda is not None:
                method = ApptainerMethod.wrap(
                    self.method,
                    image="docker://" + DEFAULT_DOCKER_IMAGE)
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


def supported_methods() -> set[str]:
    auto_register()
    return set(registry.keys())