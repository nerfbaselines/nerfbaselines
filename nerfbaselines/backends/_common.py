import numpy as np
import dataclasses
import contextlib
import os
import logging
import functools
import sys
from collections import deque
import threading
import importlib
from pathlib import Path
import shutil
from typing import Optional
from typing import  Union, Set, Callable, List, cast, Dict, Any, Tuple
from typing import Sequence
from nerfbaselines import BackendName, MethodSpec
from nerfbaselines._constants import WEBPAGE_URL
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


_mounted_paths = {}
_active_backend = {}
_backend_options = {}
_allocators = {}


def backend_allocate(size: int) -> Optional[Tuple[int, memoryview]]:
    """
    Allocates a memory block in the shared memory block
    valid for the next call to the backend. The function
    is only valid on the worker side handling request.
    Everywhere else it will return None.

    Args:
        size: The size of the memory block to allocate.

    Returns:
        Optional[Tuple[int, memoryview]]: If the memory block
            was allocated, the function returns a tuple of
            id and memory view. Otherwise, it returns None.
    """
    tid = threading.get_ident()
    allocator = _allocators.get(tid)
    if allocator is None:
        return None
    return allocator.allocate(size)


def backend_allocate_ndarray(shape, dtype):
    tid = threading.get_ident()
    allocator = _allocators.get(tid)
    if allocator is None:
        return np.empty(shape, dtype)
    return allocator.allocate_ndarray(shape, dtype)


@contextlib.contextmanager
def set_allocator(allocator):
    tid = threading.get_ident()
    old_allocator = _allocators.get(tid)
    try:
        _allocators[tid] = allocator
        yield
    finally:
        if old_allocator is None:
            del _allocators[tid]
        else:
            _allocators[tid] = old_allocator


@dataclasses.dataclass(frozen=True)
class BackendOptions:
    """
    Backend options for the current thread.
    """
    zero_copy: bool = False
    """If True, zero-copy is enabled for the current thread."""


def current_backend_options() -> BackendOptions:
    '''
    Returns the current backend options for the current thread.
    '''
    tid = threading.get_ident()
    return _backend_options.get(tid, BackendOptions())


@contextlib.contextmanager
def zero_copy(zero_copy: bool = True):
    '''
    A context manager that enables zero-copy for the current thread.
    Zero-copy is used for all subsequent calls for all backends.
    A zero-copy mode instructs the backend to reuse the shared memory
    used for data transfer. This is useful when the data is large and
    speed is important. However, it is important to only use results
    of backend calls inside the context manager. The data will be
    overwritten by subsequent calls.

    Args:
        zero_copy: If True, zero-copy is enabled for the current thread.
            If False, zero-copy is disabled.
    '''
    tid = threading.get_ident()
    if tid not in _backend_options:
        _backend_options[tid] = BackendOptions()
    change = dict(zero_copy=zero_copy)
    backup = {k: getattr(_backend_options[tid], k) for k in change}
    _backend_options[tid] = dataclasses.replace(_backend_options[tid], **change)
    try:
        yield
    finally:
        _backend_options[tid] = dataclasses.replace(_backend_options[tid], **backup)


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
            del args
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


@functools.lru_cache(maxsize=None)
def _is_backend_available(backend: BackendName) -> bool:
    """
    Tests if the backend is available on the current platform. 
    On Linux, the supported backends can be any of "conda", "docker", "apptainer", "python", 
    depending on which ones are installed. 
    On Windows or MacOS, "conda" is not supported.

    Returns:
        bool: True if the backend is available
    """
    if backend == "python":
        return True
    if sys.platform == "darwin" and backend == "conda":
        # Conda cannot be supported because it needs CUDA
        # Apptainer
        return False
    if os.name == "nt" and backend == "conda":
        # Conda is not supported directly on Windows.
        # It is only supported through WSL2.
        # The bridge is not maintained in NerfBaselines
        return False
    return shutil.which(backend) is not None


def get_implemented_backends(method_spec: 'MethodSpec') -> Sequence[BackendName]:
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
    for backend in implemented_backends:
        if _is_backend_available(backend):
            return backend
    backends_to_install = [x for x in implemented_backends if _is_backend_available(x) and x != "python"]
    raise RuntimeError("No backend available, please install " + " or ".join(backends_to_install))


def get_backend(method_spec: "MethodSpec", backend: Optional[str]) -> 'Backend':
    implemented_backends = get_implemented_backends(method_spec)
    if backend is None:
        backend = _get_default_backend([x for x in implemented_backends if x != "python"])
    elif not _is_backend_available(backend):
        raise RuntimeError(f"Backend {backend} is not available on this platform. "
            f"Please follow the installation instructions on {WEBPAGE_URL}/docs/.")
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        del exc_type, exc_val, exc_tb
        tid = threading.get_ident()
        if tid in _active_backend and _active_backend[tid]:
            _active_backend[tid].pop()
        if not _active_backend[tid]:
            del _active_backend[tid]

    def install(self):
        pass

    def shell(self, args: Optional[Tuple[str, ...]] = None):
        del args
        raise NotImplementedError("shell not implemented")

    def static_call(self, function: str, *args, **kwargs):
        del function, args, kwargs
        raise NotImplementedError("static_call not implemented")

    def instance_call(self, instance: int, method: str, *args, **kwargs):
        del instance, method, args, kwargs
        raise NotImplementedError("instance_call not implemented")

    def instance_del(self, instance: int):
        del instance
        raise NotImplementedError("instance_del not implemented")


class SimpleBackend(Backend):
    name = "python"

    def __init__(self):
        self._instances = {}

    def static_call(self, function: str, *args, **kwargs):
        logging.debug(f"Calling function {function}")
        fn, fnname = function.split(":", 1)
        fn = importlib.import_module(fn)
        for part in fnname.split("."):
            fn = getattr(fn, part)
        fn = cast(Callable, getattr(fn, "__run_on_host_original__", fn))
        return fn(*args, **kwargs)

    def instance_call(self, instance: int, method: str, *args, **kwargs):
        logging.debug(f"Calling method {method} on instance {instance}")
        instance_obj = self._instances[instance]
        fn = getattr(instance_obj, method)
        return fn(*args, **kwargs)

    def instance_del(self, instance: int):
        obj = self._instances.pop(instance, None)
        del obj


def run_on_host():
    def wrap(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if Backend.current is not None:
                return Backend.current.static_call(f"{fn.__module__}:{fn.__name__}", *args, **kwargs)
            return fn(*args, **kwargs)
        wrapped.__run_on_host_original__ = fn  # type: ignore
        return wrapped
    return wrap


def setup_logging(verbose: Union[bool, Literal['disabled']]):
    import logging

    class Formatter(logging.Formatter):
        def format(self, record: logging.LogRecord):
            levelname = record.levelname[0]
            message = record.getMessage()
            if levelname == "D":
                return f"\033[0;36mdebug:\033[0m {message}"
            elif levelname == "I":
                return f"\033[1;36minfo:\033[0m {message}"
            elif levelname == "W":
                return f"\033[0;1;33mwarning: {message}\033[0m"
            elif levelname == "E":
                return f"\033[0;1;31merror: {message}\033[0m"
            else:
                return message

    kwargs: Dict[str, Any] = {}
    if sys.version_info >= (3, 8):
        kwargs["force"] = True
    if verbose == "disabled":
        logging.basicConfig(level=logging.FATAL, **kwargs)
        logging.getLogger('PIL').setLevel(logging.FATAL)
        try:
            import tqdm as _tqdm
            old_init = _tqdm.tqdm.__init__
            _tqdm.tqdm.__init__ = lambda *args, disable=None, **kwargs: old_init(*args, disable=True, **kwargs)
        except ImportError:
            pass
    elif verbose:
        logging.basicConfig(level=logging.DEBUG, **kwargs)
        logging.getLogger('PIL').setLevel(logging.WARNING)
    else:
        import warnings
        logging.basicConfig(level=logging.INFO, **kwargs)
        warnings.formatwarning = lambda message, *args, **kwargs: message
    for handler in logging.root.handlers:
        handler.setFormatter(Formatter())
    logging.captureWarnings(True)
