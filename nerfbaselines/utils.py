import types
from functools import wraps
import sys
from typing import Any, Optional, Dict, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    cached_property = property
else:
    try:
        from functools import cached_property
    except ImportError:

        def cached_property(func):  # type: ignore
            cache = None

            @wraps(func)
            def fn_cached(self):
                nonlocal cache
                if cache is None:
                    cache = (func(self),)
                return cache[0]

            return property(fn_cached)


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
            ns["__init__"] = partialmethod(cls_dict["__init__"], *args, **kwargs)
        return ns

    return types.new_class(cls.__name__, bases=(cls,), exec_body=build)


class Indices:
    def __init__(self, steps):
        self._steps = steps
        self.total: Optional[int] = None

    def __contains__(self, x):
        if isinstance(self._steps, list):
            return x in self._steps
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


def batched(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i : i + batch_size]


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


def mark_host(fn):
    fn.__host__ = True
    return fn
