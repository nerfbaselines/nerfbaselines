import sys
import os
import subprocess
import inspect
import struct
from pathlib import Path
import types
from functools import wraps
from typing import Any, Optional, Dict, TYPE_CHECKING, Union, List
from typing import BinaryIO
import numpy as np
import logging

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
        except ImportError as e:
            # pyngp import error
            if "libcuda.so.1: cannot open shared object file" in str(e):
                raise NoGPUError from e
            raise

    return wrapped


class CancelledException(Exception):
    pass


class CancellationToken:
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return self._cancelled

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


def cancellable(fn=None, mark_only=False):
    def wrap(fn):
        if getattr(fn, "__cancellable__", False):
            return fn
        if mark_only:
            fn.__cancellable__ = True
            return fn

        if inspect.isgeneratorfunction(fn):

            @wraps(fn)
            def wrapped(*args, cancellation_token: Optional[CancellationToken] = None, **kwargs):
                if cancellation_token is not None:
                    yield from cancellation_token.invoke(fn, *args, **kwargs)
                else:
                    yield from fn(*args, **kwargs)

        else:

            @wraps(fn)
            def wrapped(*args, cancellation_token: Optional[CancellationToken] = None, **kwargs):
                if cancellation_token is not None:
                    return cancellation_token.invoke(fn, *args, **kwargs)
                else:
                    return fn(*args, **kwargs)

        wrapped.__cancellable__ = True
        return wrapped

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
            if hasattr(e, "write_to_logger"):
                e.write_to_logger()
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


def padded_stack(tensors: List[np.ndarray]) -> np.ndarray:
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


def convert_image_dtype(image, dtype):
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


def image_to_srgb(tensor, dtype, color_space="srgb", allow_alpha: bool = False, background_color: Optional[np.ndarray] = None):
    # Remove alpha channel in uint8
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


def visualize_depth(depth: np.ndarray, expected_scale: Optional[float] = None, near_far: Optional[np.ndarray] = None, pallete: str = "viridis", xnp=np) -> np.ndarray:
    # TODO: remove matplotlib dependency
    import matplotlib

    # We will squash the depth to range [0, 1] using Barron's power transformation
    eps = xnp.finfo(xnp.float32).eps
    if near_far is not None:
        depth_squashed = (depth - near_far[0]) / (near_far[1] - near_far[0])
    elif expected_scale is not None:
        depth = depth / max(2 * expected_scale, eps)

        # We use the power series -> for lam=-1.5 the limit is 5/3
        depth_squashed = _zipnerf_power_transformation(depth, -1.5) / (5 / 3)
    else:
        depth_squashed = depth
    depth_squashed = depth_squashed.clip(0, 1)

    # Map to a color scale
    depth_long = (depth_squashed * 255).astype(xnp.int32).clip(0, 255)
    out = xnp.array(matplotlib.colormaps[pallete].colors, dtype=xnp.float32)[255 - depth_long]
    return (out * 255).astype(xnp.uint8)


def get_resources_utilization_info(pid=None):
    if pid is None:
        pid = os.getpid()

    info = {}

    # Get all cpu memory and running processes
    all_processes = set((pid,))
    try:
        mem = 0
        out = subprocess.check_output("ps -ax -o pid= -o ppid= -o rss=".split(), text=True).splitlines()
        mem_used = {}
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
        out = subprocess.check_output("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits".split(), text=True).splitlines()
        for line in out:
            cpid, used_memory = map(int, line.split(","))
            if cpid in all_processes:
                gpu_memory += used_memory
        info["gpu_memory"] = gpu_memory
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass

    return info
