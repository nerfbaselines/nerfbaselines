import re
import shlex
import dataclasses
import threading
from collections import deque
import contextlib
import time
import click
import sys
import math
import os
import subprocess
import inspect
import struct
from pathlib import Path
from functools import wraps
from typing import Any, Optional, Dict, TYPE_CHECKING, Union, List, TypeVar, Iterable, overload, Callable
from typing import BinaryIO, Tuple, cast, Set
import logging
import types
import numpy as np
from PIL import Image
try:
    from typing import get_args, get_origin
except ImportError:
    from typing_extensions import get_args, get_origin
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
try:
    from typing import NotRequired  # noqa: F401
    from typing import Required  # noqa: F401
    from typing import TypedDict
except ImportError:
    from typing_extensions import NotRequired  # noqa: F401
    from typing_extensions import Required  # noqa: F401
    from typing_extensions import TypedDict
try:
    from shlex import join as shlex_join
except ImportError:
    def shlex_join(split_command):
        return ' '.join(shlex.quote(arg) for arg in split_command)

if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp

if TYPE_CHECKING:
    cached_property = property
else:
    try:
        from functools import cached_property
    except ImportError:

        def cached_property(func):  # type: ignore
            key = "__cached_prop_" + func.__name__

            @wraps(func)
            def fn_cached(self):
                if key not in self.__dict__:
                    self.__dict__[key] = result = func(self)
                else:
                    result = self.__dict__[key]
                return result

            return property(fn_cached)


T = TypeVar("T")
TCallable = TypeVar("TCallable", bound=Callable)
TTensor = TypeVar("TTensor", np.ndarray, "torch.Tensor", "jnp.ndarray")


def _get_xnp(tensor: TTensor):
    if isinstance(tensor, np.ndarray):
        return np
    if tensor.__module__.startswith("jax"):
        return cast('jnp', sys.modules["jax.numpy"])
    if tensor.__module__ == "torch":
        return cast('torch', sys.modules["torch"])
    raise ValueError(f"Unknown tensor type {type(tensor)}")


def _xnp_astype(tensor: TTensor, dtype, xnp: Any) -> TTensor:
    if xnp.__name__ == "torch":
        return tensor.to(dtype)  # type: ignore
    return tensor.astype(dtype)  # type: ignore


def assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value

class NoGPUError(RuntimeError):
    def __init__(self, message="No GPUs available"):
        super().__init__(message)


def remap_error(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except RuntimeError as e:
            # torch driver error
            if "Found no NVIDIA driver on your system." in str(e):
                raise NoGPUError from e
            raise
        except EnvironmentError as e:
            # tcnn import error
            if "unknown compute capability. ensure pytorch with cuda support is installed." in str(e).lower():
                raise NoGPUError from e
            raise
        except ImportError as e:
            # pyngp import error
            if "libcuda.so.1: cannot open shared object file" in str(e):
                raise NoGPUError from e
            raise

    return wrapped


def build_measure_iter_time():
    total_time = 0

    def measure_iter_time(iterable: Iterable) -> Iterable:
        nonlocal total_time
        
        total_time = 0
        start = time.perf_counter()
        for x in iterable:
            total_time += time.perf_counter() - start
            yield x
            start = time.perf_counter()

    def get_total_time():
        return total_time

    return measure_iter_time, get_total_time


class CancelledException(Exception):
    pass


class _CancellationTokenMeta(type):
    _current_stack = {}

    @property
    def current(cls) -> Optional["CancellationToken"]:
        thread_id = threading.get_ident()
        stack = _CancellationTokenMeta._current_stack.get(thread_id, None)
        if not stack:
            return None
        return stack[-1]

    def _push_token(cls, token):
        thread_id = threading.get_ident()
        stack = _CancellationTokenMeta._current_stack.get(thread_id, None)
        if stack is None:
            _CancellationTokenMeta._current_stack[thread_id] = deque([token])
        else:
            stack.append(token)
    
    def _pop_token(cls, token):
        thread_id = threading.get_ident()
        _CancellationTokenMeta._current_stack[thread_id].pop()


class CancellationToken(metaclass=_CancellationTokenMeta):
    def __init__(self, parent_token: Optional['CancellationToken'] = None):
        self.parent = parent_token
        self._callbacks = []
        self._cancelled = False
        self.del_hooks = []
        if parent_token is not None:
            parent_token.register_callback(self.cancel)

    def __del__(self):
        for hook in self.del_hooks:
            hook(self)
        self.del_hooks.clear()

    def cancel(self):
        self._cancelled = True
        for cb in self._callbacks:
            cb()

    @property
    def cancelled(self):
        if self.parent is not None and self.parent.cancelled:
            return True
        return self._cancelled

    def register_callback(self, callback):
        if callback not in self._callbacks:
            self._callbacks.append(callback)
    
    def unregister_callback(self, callback):
        self._callbacks.remove(callback)

    def _trace(self, frame, event, arg):
        if event == "line":
            if self.cancelled:
                raise CancelledException
        return self._trace

    def _invoke_generator(self, fn, *args, **kwargs):
        try:
            sys.settrace(self._trace)
            for r in fn(*args, **kwargs):
                yield r
        finally:
            sys.settrace(None)

    def invoke(self, fn, *args, **kwargs):
        if inspect.isgeneratorfunction(fn):
            return self._invoke_generator(fn, *args, **kwargs)

        try:
            sys.settrace(self._trace)
            return fn(*args, **kwargs)
        finally:
            sys.settrace(None)

    def raise_for_cancelled(self):
        if self.cancelled:
            raise CancelledException

    def __enter__(self):
        type(self)._push_token(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        type(self)._pop_token(self)


@overload
def cancellable(fn: TCallable, *, mark_only: bool =..., cancellation_token: Optional[CancellationToken] = ...) -> TCallable:
    ...

@overload
def cancellable(*, mark_only: bool =..., cancellation_token: Optional[CancellationToken] = ...) -> Callable[[TCallable], TCallable]:
    ...

# TODO: fix signature of wrapped function
def cancellable(fn=None, *, mark_only: bool = False, cancellation_token: Optional[CancellationToken] = None):
    def wrap(fn):
        if mark_only:
            fn.__cancellable__ = True
            return fn

        if getattr(fn, "__cancellable__", False) and cancellation_token is None:
            return fn

        if inspect.isgeneratorfunction(fn):
            @wraps(fn)
            def wrapped_generator(*args, **kwargs):
                with (cancellation_token or contextlib.nullcontext()):
                    token = CancellationToken.current
                    if token is None or getattr(fn, "__cancellable__", False):
                        yield from fn(*args, **kwargs)
                    else:
                        yield from token.invoke(fn, *args, **kwargs)
            wrapped_generator.__cancellable__ = True  # type: ignore
            return wrapped_generator
        else:
            @wraps(fn)
            def wrapped_function(*args, **kwargs):
                with (cancellation_token or contextlib.nullcontext()):
                    token = CancellationToken.current
                    if token is None or getattr(fn, "__cancellable__", False):
                        return fn(*args, **kwargs)
                    else:
                        return token.invoke(fn, *args, **kwargs)
            wrapped_function.__cancellable__ = True  # type: ignore
            return wrapped_function

    return wrap if fn is None else wrap(fn)


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


def setup_logging(verbose: bool):
    kwargs: Dict[str, Any] = {}
    if sys.version_info >= (3, 8):
        kwargs["force"] = True
    if verbose >= 1:
        logging.basicConfig(level=logging.DEBUG, **kwargs)
        logging.getLogger('PIL').setLevel(logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO, **kwargs)
    for handler in logging.root.handlers:
        handler.setFormatter(Formatter())
    logging.captureWarnings(True)


def handle_cli_error(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            write_to_logger = getattr(e, "write_to_logger", None)
            if write_to_logger is not None:
                write_to_logger()
                sys.exit(1)
            else:
                raise e

    return wrapped


def partialmethod(func, *args, **kwargs):
    def wrapped(self, *args2, **kwargs2):
        return func(self, *args, *args2, **kwargs, **kwargs2)

    wrapped.__original_func__ = func  # type: ignore
    wrapped.__args__ = args  # type: ignore
    wrapped.__kwargs__ = kwargs  # type: ignore
    return wrapped


def partialclass(cls, *args, **kwargs):
    def build(ns):
        cls_dict = cls.__dict__
        ns["__module__"] = cls_dict["__module__"]
        ns["__doc__"] = cls_dict["__doc__"]
        if args or kwargs:
            ns["__init__"] = partialmethod(cls.__init__, *args, **kwargs)
        return ns

    return types.new_class(cls.__name__, bases=(cls,), exec_body=build)


class Indices:
    def __init__(self, steps):
        self._steps = steps
        self.total: Optional[int] = None

    def __contains__(self, x):
        if isinstance(self._steps, list):
            steps = self._steps
            if any(x < 0 for x in self._steps):
                assert self.total is not None, "total must be specified for negative steps"
                steps = set(x if x >= 0 else self.total + x for x in self._steps)
            return x in steps
        elif isinstance(self._steps, slice):
            start: int = self._steps.start or 0
            if start < 0:
                assert self.total is not None, "total must be specified for negative start"
                start = self.total - start
            stop: Optional[int] = self._steps.stop or self.total
            if stop is not None and stop < 0:
                assert self.total is not None, "total must be specified for negative stop"
                stop = self.total - stop
            step: int = self._steps.step or 1
            return x >= start and (stop is None or x < stop) and (x - start) % step == 0

    @classmethod
    def every_iters(cls, iters: int, zero: bool = False):
        start = iters if zero else 0
        return cls(slice(start, None, iters))

    def __repr__(self):
        if isinstance(self._steps, list):
            return ",".join(map(str, self._steps))
        elif isinstance(self._steps, slice):
            out = f"{self._steps.start or ''}:{self._steps.stop or ''}"
            if self._steps.step is not None:
                out += f":{self._steps.step}"
            return out
        else:
            return repr(self._steps)

    def __str__(self):
        return repr(self)


def batched(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i : i + batch_size]


def padded_stack(tensors: Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]) -> np.ndarray:
    if not isinstance(tensors, (tuple, list)):
        return tensors
    max_shape = tuple(max(s) for s in zip(*[x.shape for x in tensors]))
    out_tensors = []
    for x in tensors:
        pads = [(0, m - s) for s, m in zip(x.shape, max_shape)]
        out_tensors.append(np.pad(x, pads))
    return np.stack(out_tensors, 0)


def is_broadcastable(shape1, shape2):
    for a, b in zip(shape1[::-1], shape2[::-1]):
        if a == 1 or b == 1 or a == b:
            pass
        else:
            return False
    return True


def convert_image_dtype(image: np.ndarray, dtype) -> np.ndarray:
    if image.dtype == dtype:
        return image
    if image.dtype != np.uint8 and dtype != np.uint8:
        return image.astype(dtype)
    if image.dtype == np.uint8 and dtype != np.uint8:
        return image.astype(dtype) / 255.0
    if image.dtype != np.uint8 and dtype == np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"cannot convert image from {image.dtype} to {dtype}")


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def image_to_srgb(tensor, dtype, color_space: Optional[str] = None, allow_alpha: bool = False, background_color: Optional[np.ndarray] = None):
    # Remove alpha channel in uint8
    if color_space is None:
        color_space = "srgb"
    if tensor.shape[-1] == 4 and not allow_alpha:
        # NOTE: here we blend with black background
        if tensor.dtype == np.uint8:
            tensor = convert_image_dtype(tensor, np.float32)
        alpha = tensor[..., -1:]
        tensor = tensor[..., :3] * tensor[..., -1:]
        # Default is black background [0, 0, 0]
        if background_color is not None:
            tensor += (1 - alpha) * convert_image_dtype(background_color, np.float32)

    if color_space == "linear":
        tensor = convert_image_dtype(tensor, np.float32)
        tensor = linear_to_srgb(tensor)

    # Round to 8-bit for fair comparisons
    tensor = convert_image_dtype(tensor, np.uint8)
    tensor = convert_image_dtype(tensor, dtype)
    return tensor


def save_image(file: Union[BinaryIO, str, Path], tensor: np.ndarray):
    if isinstance(file, (str, Path)):
        with open(file, "wb") as f:
            return save_image(f, tensor)
    path = Path(file.name)
    if str(path).endswith(".bin"):
        if tensor.shape[2] < 4:
            tensor = np.dstack((tensor, np.ones([tensor.shape[0], tensor.shape[1], 4 - tensor.shape[2]])))
        file.write(struct.pack("ii", tensor.shape[0], tensor.shape[1]))
        file.write(tensor.astype(np.float16).tobytes())
    else:
        from PIL import Image

        tensor = convert_image_dtype(tensor, np.uint8)
        image = Image.fromarray(tensor)
        image.save(file, format="png")


def read_image(file: Union[BinaryIO, str, Path]) -> np.ndarray:
    if isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            return read_image(f)
    path = Path(file.name)
    if str(path).endswith(".bin"):
        h, w = struct.unpack("ii", file.read(8))
        itemsize = 2
        img = np.frombuffer(file.read(h * w * 4 * itemsize), dtype=np.float16, count=h * w * 4, offset=8).reshape([h, w, 4])
        assert img.itemsize == itemsize
        return img.astype(np.float32)
    else:
        from PIL import Image

        return np.array(Image.open(file))


def save_depth(file: Union[BinaryIO, str, Path], tensor: np.ndarray):
    if isinstance(file, (str, Path)):
        with open(file, "wb") as f:
            return save_depth(f, tensor)
    path = Path(file.name)
    assert str(path).endswith(".bin")
    file.write(struct.pack("ii", tensor.shape[0], tensor.shape[1]))
    file.write(tensor.astype(np.float16).tobytes())


def mark_host(fn):
    fn.__host__ = True
    return fn


def _zipnerf_power_transformation(x, lam: float):
    m = abs(lam - 1) / lam
    return (((x / abs(lam - 1)) + 1) ** lam - 1) * m


def apply_colormap(array: TTensor, *, pallete: str = "viridis", invert: bool = False) -> TTensor:
    xnp = _get_xnp(array)
    # TODO: remove matplotlib dependency
    import matplotlib
    import matplotlib.colors

    # Map to a color scale
    array_long = cast(TTensor, _xnp_astype(array * 255, xnp.int32, xnp=xnp).clip(0, 255))
    colormap = matplotlib.colormaps[pallete]
    colormap_colors = None
    if isinstance(colormap, matplotlib.colors.ListedColormap):
        colormap_colors = colormap.colors
    else:
        colormap_colors = [list(colormap(i / 255))[:3] for i in range(256)]
    if xnp.__name__ == "torch":
        import torch
        pallete_array = cast(TTensor, torch.tensor(colormap_colors, dtype=torch.float32, device=cast(torch.Tensor, array).device))
    else:
        pallete_array = cast(TTensor, xnp.array(colormap_colors, dtype=xnp.float32))  # type: ignore
    if invert:
        array_long = 255 - array_long
    out = cast(TTensor, pallete_array[array_long])
    return _xnp_astype(out * 255, xnp.uint8, xnp=xnp)


def visualize_depth(depth: np.ndarray, expected_scale: Optional[float] = None, near_far: Optional[np.ndarray] = None, pallete: str = "viridis") -> np.ndarray:
    # We will squash the depth to range [0, 1] using Barron's power transformation
    xnp = _get_xnp(depth)
    eps = xnp.finfo(xnp.float32).eps  # type: ignore
    if near_far is not None:
        depth_squashed = (depth - near_far[0]) / (near_far[1] - near_far[0])
    elif expected_scale is not None:
        depth = depth / max(0.3 * expected_scale, eps)

        # We use the power series -> for lam=-1.5 the limit is 5/3
        depth_squashed = _zipnerf_power_transformation(depth, -1.5) / (5 / 3)
    else:
        depth_squashed = depth
    depth_squashed = depth_squashed.clip(0, 1)

    # Map to a color scale
    return apply_colormap(depth_squashed, pallete=pallete)


def run_on_host():
    def wrap(fn):
        @wraps(fn)
        def wrapped(*args, **kwargs):
            from .backends import Backend
            if Backend.current is not None:
                return Backend.current.wrap(fn)(*args, **kwargs)
            return fn(*args, **kwargs)
        wrapped.__run_on_host_original__ = fn  # type: ignore
        return wrapped
    return wrap


class ResourcesUtilizationInfo(TypedDict, total=False):
    memory: int
    gpu_memory: int
    gpu_name: str


@run_on_host()
def get_resources_utilization_info(pid: Optional[int] = None) -> ResourcesUtilizationInfo:
    if pid is None:
        pid = os.getpid()

    info: ResourcesUtilizationInfo = {}

    # Get all cpu memory and running processes
    all_processes = set((pid,))
    try:
        mem = 0
        out = subprocess.check_output("ps -ax -o pid= -o ppid= -o rss=".split(), text=True).splitlines()
        mem_used: Dict[int, int] = {}
        children = {}
        for line in out:
            cpid, ppid, used_memory = map(int, line.split())
            mem_used[cpid] = used_memory
            children.setdefault(ppid, set()).add(cpid)
        all_processes = set()
        stack = [pid]
        while stack:
            cpid = stack.pop()
            all_processes.add(cpid)
            mem += mem_used[cpid]
            stack.extend(children.get(cpid, []))
        info["memory"] = (mem + 1024 - 1) // 1024
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass

    try:
        gpu_memory = 0
        gpus = {}
        uuids = set()
        out = subprocess.check_output("nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid,gpu_name --format=csv,noheader,nounits".split(), text=True).splitlines()
        for line in out:
            cpid, used_memory, uuid, gpu_name = tuple(x.strip() for x in line.split(",", 3))
            cpid = int(cpid)
            used_memory = int(used_memory)
            if cpid in all_processes:
                gpu_memory += used_memory
                if uuid not in uuids:
                    uuids.add(uuid)
                    gpus[gpu_name] = gpus.get(gpu_name, 0) + 1
        info["gpu_name"] = ",".join(f"{k}:{v}" if v > 1 else k for k, v in gpus.items())
        info["gpu_memory"] = gpu_memory
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass

    return info


def cast_value(tp, value):
    origin = get_origin(tp)
    if origin is Literal:
        for val in get_args(tp):
            try:
                value_casted = cast_value(type(val), value)
                if val == value_casted:
                    return value_casted
            except ValueError:
                pass
            except TypeError:
                pass
        raise TypeError(f"Value {value} is not in {get_args(tp)}")
            
    if origin is Union:
        for t in get_args(tp):
            try:
                return cast_value(t, value)
            except ValueError:
                pass
            except TypeError:
                pass
        raise TypeError(f"Value {value} is not in {tp}")
    if tp is type(None):
        if str(value).lower() == "none":
            return None
        else:
            raise TypeError(f"Value {value} is not None")
    if tp is bool:
        if str(value).lower() in {"true", "1", "yes"}:
            return True
        elif str(value).lower() in {"false", "0", "no"}:
            return False
        else:
            raise TypeError(f"Value {value} is not a bool")
    if tp in {int, float, bool, str}:
        return tp(value)
    if isinstance(value, tp):
        return value
    raise TypeError(f"Cannot cast value {value} to type {tp}")


def make_image_grid(*images: np.ndarray, ncol=None, padding=2, max_width=1920, background: Union[None, Tuple[float, float, float], np.ndarray] = None):
    if ncol is None:
        ncol = len(images)
    dtype = images[0].dtype
    if background is None:
        background = np.full((3,), 255 if dtype == np.uint8 else 1, dtype=dtype)
    elif isinstance(background, tuple):
        background = np.array(background, dtype=dtype)
    elif isinstance(background, np.ndarray):
        background = convert_image_dtype(background, dtype=dtype)
    else:
        raise ValueError(f"Invalid background type {type(background)}")
    nrow = int(math.ceil(len(images) / ncol))
    scale_factor = 1
    height, width = tuple(map(int, np.max([x.shape[:2] for x in images], axis=0).tolist()))
    if max_width is not None:
        scale_factor = min(1, (max_width - padding * (ncol - 1)) / (ncol * width))
        height = int(height * scale_factor)
        width = int(width * scale_factor)

    def interpolate(image) -> np.ndarray:
        img = Image.fromarray(image)
        img_width, img_height = img.size
        aspect = img_width / img_height
        img_width = int(min(width, aspect * height))
        img_height = int(img_width / aspect)
        img = img.resize((img_width, img_height))
        return np.array(img)

    images = tuple(map(interpolate, images))
    grid: np.ndarray = np.ndarray(
        (height * nrow + padding * (nrow - 1), width * ncol + padding * (ncol - 1), images[0].shape[-1]),
        dtype=dtype,
    )
    grid[..., :] = background
    for i, image in enumerate(images):
        x = i % ncol
        y = i // ncol
        h, w = image.shape[:2]
        offx = x * (width + padding) + (width - w) // 2
        offy = y * (height + padding) + (height - h) // 2
        grid[offy : offy + h, 
             offx : offx + w] = image
    return grid


class IndicesClickType(click.ParamType):
    name = "indices"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, Indices):
            return value
        if ":" in value:
            parts = [int(x) if x else None for x in value.split(":")]
            assert len(parts) <= 3, "too many parts in slice"
            return Indices(slice(*parts))
        return Indices([int(x) for x in value.split(",")])


class SetParamOptionType(click.ParamType):
    name = "key-value"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, tuple):
            return value
        if "=" not in value:
            self.fail(f"expected key=value pair, got {value}", param, ctx)
        k, v = value.split("=", 1)
        return k, v


def get_package_dependencies(extra=None, ignore: Optional[Set[str]] = None, ignore_viewer: bool = False):
    assert __package__ is not None, "Package must be set"
    if sys.version_info < (3, 10):
        from importlib_metadata import distribution
        import importlib_metadata
    else:
        from importlib import metadata as importlib_metadata
        from importlib.metadata import distribution

    requires = set()
    requires_with_conditions = None
    try:
        requires_with_conditions = distribution(__package__).requires
    except importlib_metadata.PackageNotFoundError:
        # Package not installed
        pass
    for r in (requires_with_conditions or ()):
        if ";" in r:
            r, condition = r.split(";")
            r = r.strip().replace(" ", "")
            condition = condition.strip().replace(" ", "")
            if condition.startswith("extra=="):
                extracond = condition.split("==")[1][1:-1]
                if extra is not None and extracond in extra:
                    requires.add(r)
                continue
            elif condition.startswith("python_version"):
                requires.add(r)
                continue
            else:
                raise ValueError(f"Unknown condition {condition}")
        r = r.strip().replace(" ", "")
        requires.add(r)
    if ignore_viewer:
        # NOTE: Viewer is included in the package by default
        # See https://github.com/pypa/setuptools/pull/1503
        ignore = set(ignore or ())
        ignore.add("viser")

    if ignore is not None:
        ignore = set(x.lower() for x in ignore)
        for r in list(requires):
            rsimple = re.sub(r"[^a-zA-Z0-9_-].*", "", r).lower()
            if rsimple in ignore:
                requires.remove(r)
    return sorted(requires)


def flatten_hparams(hparams: Any, *, separator: str = "/", _prefix: str = "") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)}
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator).items())
        else:
            flat[k] = v
    return flat


MetricAccumulationMode = Literal["average", "last", "sum"]


class MetricsAccumulator:
    def __init__(
        self,
        options: Optional[Dict[str, MetricAccumulationMode]] = None,
    ):
        self.options = options or {}
        self._state = None

    def update(self, metrics: Dict[str, Union[int, float]]) -> None:
        if self._state is None:
            self._state = {}
        state = self._state
        n_iters_since_update = state["n_iters_since_update"] = state.get("n_iters_since_update", {})
        for k, v in metrics.items():
            accumulation_mode = self.options.get(k, "average")
            n_iters_since_update[k] = n = n_iters_since_update.get(k, 0) + 1
            if k not in state:
                state[k] = 0
            if accumulation_mode == "last":
                state[k] = v
            elif accumulation_mode == "average":
                state[k] = state[k] * ((n - 1) / n) + v / n
            elif accumulation_mode == "sum":
                state[k] += v
            else:
                raise ValueError(f"Unknown accumulation mode {accumulation_mode}")

    def pop(self) -> Dict[str, Union[int, float]]:
        if self._state is None:
            return {}
        state = self._state
        self._state = None
        state.pop("n_iters_since_update", None)
        return state


@contextlib.contextmanager
def run_inside_eval_container(backend_name: Optional[str] = None):
    """
    Ensures PyTorch is available to compute extra metrics (lpips)
    """
    from .backends import get_backend
    try:
        import torch as _
        yield None
        return
    except ImportError:
        pass

    logging.warning("PyTorch is not available in the current environment, we will create a new environment to compute extra metrics (lpips)")
    if backend_name is None:
        backend_name = os.environ.get("NERFBASELINES_BACKEND", None)
    backend = get_backend({
        "method": "base",
        "conda": {
            "environment_name": "_metrics", 
            "install_script": ""
        }}, backend=backend_name)
    with backend:
        backend.install()
        yield None

