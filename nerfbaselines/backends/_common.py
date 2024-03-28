from collections import deque
from functools import wraps
import inspect
import threading
import types
from functools import partial
import importlib
from pathlib import Path
import subprocess
from typing import Optional
from typing import  Union, Set, Callable, TYPE_CHECKING, List, cast
from typing import Sequence
from ..utils import CancellationToken, cancellable
from ..types import Method, Literal, get_args
if TYPE_CHECKING:
    from ..registry import MethodSpec


BackendName = Literal["conda", "docker", "apptainer", "python"]
ALL_BACKENDS = list(get_args(BackendName))

_mounted_paths = {}
_forwarded_ports = {}
_active_backend = {}


def mount(ps: Union[str, Path], pd: Union[str, Path]):
    tid = threading.get_ident()
    if _active_backend.get(tid):
        raise RuntimeError("Cannot mount while backend is active")
    if tid not in _mounted_paths:
        _mounted_paths[tid] = {}
    dest = str(Path(pd).absolute())
    _mounted_paths[tid][dest] = str(Path(ps).absolute())
    class _Mount:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            if tid in _mounted_paths and dest in _mounted_paths[tid]:
                _mounted_paths[tid].pop(dest)
            if tid in _mounted_paths and not _mounted_paths[tid]:
                del _mounted_paths[tid]
    return _Mount()


def get_mounts():
    tid = threading.get_ident()
    out = []
    for dest, src in _mounted_paths.get(tid, {}).items():
        out.append((src, dest))
    return out


def forward_port(ps: int, pd: int):
    tid = threading.get_ident()
    if _active_backend.get(tid):
        raise RuntimeError("Cannot forward ports while backend is active")
    if tid not in _forwarded_ports:
        _forwarded_ports[tid] = {}
    _forwarded_ports[tid][pd] = ps
    class _Forward:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            if tid in _forwarded_ports and pd in _forwarded_ports[tid]:
                _forwarded_ports[tid].pop(pd)
            if tid in _forwarded_ports and not _forwarded_ports[tid]:
                del _forwarded_ports[tid]
    return _Forward()


def get_forwarded_ports():
    tid = threading.get_ident()
    out = []
    for dest, src in _forwarded_ports.get(tid, {}).items():
        out.append((src, dest))
    return out


def _get_implemented_backends(method_spec: 'MethodSpec') -> Sequence[BackendName]:
    from ._apptainer import get_apptainer_spec
    from ._docker import get_docker_spec

    backends: Set[BackendName] = set(("python",))
    if method_spec.get("conda") is not None:
        backends.add("conda")

    if get_docker_spec(method_spec) is not None:
        backends.add("docker")

    if get_apptainer_spec(method_spec) is not None:
        backends.add("apptainer")

    backends_order: List[BackendName] = ["conda", "docker", "apptainer", "python"]
    bo = method_spec.get("backends_order")
    if bo is not None:
        backends_order = list(bo) + [x for x in backends_order if x not in bo]
    return [x for x in backends_order if x in backends]


def _get_default_backend(implemented_backends: Sequence[BackendName]) -> BackendName:
    should_install = []
    for backend in implemented_backends:
        if backend not in implemented_backends:
            continue
        if backend == "python":
            return "python"
        try:
            if backend == "conda":
                test_args = ["conda", "--version"]
            elif backend == "docker":
                test_args = ["docker", "-v"]
            elif backend == "apptainer":
                test_args = ["apptainer", "-v"]
            else:
                raise ValueError(f"Unknown backend {backend}")
            ret = subprocess.run(test_args, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if ret.returncode == 0:
                return backend
        except FileNotFoundError:
            pass
            should_install.append(backend)
    raise RuntimeError("No backend available, please install " + " or ".join(should_install))


def get_backend(method_spec: "MethodSpec", backend: Optional[str]) -> 'Backend':
    implemented_backends = _get_implemented_backends(method_spec)
    if backend is None:
        backend = _get_default_backend(implemented_backends)
    elif backend not in implemented_backends:
        raise RuntimeError(f"Backend {backend} is not implemented for selected method.\nImplemented backends: {','.join(implemented_backends)}")

    if backend == "python":
        return SimpleBackend()
    elif backend == "conda":
        from ._conda import CondaBackend
        spec = method_spec.get("conda")
        assert spec is not None, "conda_spec is not defined"
        return CondaBackend(spec)
    elif backend == "docker":
        from ._docker import DockerBackend, get_docker_spec
        spec = get_docker_spec(method_spec)
        assert spec is not None, "docker_spec is not defined"
        return DockerBackend(spec)
    elif backend == "apptainer":
        from ._apptainer import ApptainerBackend, get_apptainer_spec
        spec = get_apptainer_spec(method_spec)
        assert spec is not None, "apptainer_spec is not defined"
        return ApptainerBackend(spec)
    else:
        raise ValueError(f"Unknown backend {backend}")


class _BackendMeta(type):
    @property
    def current(cls) -> Optional['Backend']:
        tid = threading.get_ident()
        if tid in _active_backend and _active_backend[tid]:
            return _active_backend[tid][-1]
        return None


class Backend(metaclass=_BackendMeta):
    name = "unknown"

    def __enter__(self):
        tid = threading.get_ident()
        if tid not in _active_backend:
            _active_backend[tid] = deque()
        _active_backend[tid].append(self)
        return self

    def __exit__(self, *args):
        tid = threading.get_ident()
        if tid in _active_backend and _active_backend[tid]:
            _active_backend[tid].pop()
        if not _active_backend[tid]:
            del _active_backend[tid]

    def install(self):
        pass

    def shell(self):
        raise NotImplementedError("shell not implemented")

    def wrap(self, function: Union[str, Callable], spec=None):
        return wrap_with_backend(self, function, spec=spec)

    def static_call(self, function: str, *args, **kwargs):
        raise NotImplementedError("static_call not implemented")

    def static_getattr(self, attr: str):
        raise NotImplementedError("static_getattr not implemented")

    def instance_call(self, instance: int, method: str, *args, **kwargs):
        raise NotImplementedError("instance_call not implemented")

    def instance_getattr(self, instance: int, attr: str):
        raise NotImplementedError("instance_getattr not implemented")

    def instance_del(self, instance: int):
        raise NotImplementedError("instance_del not implemented")


class SimpleBackend(Backend):
    name = "python"

    def __init__(self):
        self._instances = {}

    def static_getattr(self, attr: str):
        obj, attrname = attr.split(":", 1)
        obj = importlib.import_module(obj)
        for part in attrname.split("."):
            obj = getattr(obj, part)
        return obj

    def static_call(self, function: str, *args, **kwargs):
        fn, fnname = function.split(":", 1)
        fn = importlib.import_module(fn)
        for part in fnname.split("."):
            fn = getattr(fn, part)
        fn = cast(Callable, getattr(fn, "__run_on_host_original__", fn))
        if CancellationToken.current is not None:
            fn = cancellable(fn, cancellation_token=CancellationToken.current)
        return fn(*args, **kwargs)

    def instance_call(self, instance: int, method: str, *args, **kwargs):
        instance = self._instances[instance]
        fn = getattr(instance, method)
        if CancellationToken.current is not None:
            fn = cancellable(fn, cancellation_token=CancellationToken.current)
        return fn(*args, **kwargs)

    def instance_getattr(self, instance: int, attr: str):
        instance = self._instances[instance]
        return getattr(instance, attr)

    def instance_del(self, instance: int):
        del self._instances[instance]


def wrap_with_backend(backend: 'Backend', cls, spec=None):
    if inspect.isfunction(spec or cls):
        if spec is None:
            path = f'{cls.__module__}:{cls.__name__}'
        else:
            path = cls

        @wraps(cls)
        def inner(*args, **kwargs):
            return backend.static_call(path, *args, **kwargs)
        return inner

    if inspect.isclass(spec or cls):
        if spec is None:
            path = f'{cls.__module__}:{cls.__name__}'
            bases = tuple(x for x in inspect.getmro(cls) if (
                x.__module__ in ("builtins", Method.__module__)
            ))
        else:
            path = cls
            bases = (spec,)
            cls = spec

        members = dict(inspect.getmembers(cls))

        ns = {}
        def partialproperty(k):
            return property(lambda *_: backend.static_getattr(f'{path}.{k}'))

        for k, member in members.items():
            if isinstance(member, classmethod):
                ns[k] = classmethod(partial(backend.static_getattr, f'{path}.{k}'))  # type: ignore
            elif callable(member):
                ns[k] = staticmethod(partial(backend.static_call, f'{path}.{k}'))  # type: ignore
            elif isinstance(member, property):
                ns[k] = partialproperty(k)
        ns.pop("__init__", None)
        ns.pop("__getattribute__", None)
        ns.pop("__getattr__", None)
        ns.pop("__setattr__", None)
        repr_str = f"<VirtualClass {cls}>"
        ns["__new__"] = lambda _cls, *args, **kwargs: backend.static_call(path, *args, **kwargs)
        ns["__repr__"] = lambda x: repr_str
        ns["__str__"] = lambda x: repr_str
        return types.new_class("VirtualInstanceWrapper", bases, {}, exec_body=lambda _ns: _ns.update(ns))
    
    raise ValueError(f"Unsupported type {cls}")
