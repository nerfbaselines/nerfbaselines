import types
from functools import partial
import dataclasses
from typing import Callable, List, Any, Optional, Tuple
import numpy as np
from jaxtyping import Float32, Int32
from .distortion import Distortions
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


def setup_logging(verbose: bool):
    if verbose >= 1:
        logging.basicConfig(level=logging.DEBUG, force=True)
    else:
        logging.basicConfig(level=logging.INFO, force=True)
    for handler in logging.root.handlers:
        handler.setFormatter(Formatter())


def partialmethod(func, *args, **kwargs):
    def wrapped(self, *args2, **kwargs2):
        return func(self, *args, *args2, **kwargs, **kwargs2)
    return wrapped


def partialclass(cls, *args, **kwargs):
    def build(ns):
        cls_dict = cls.__dict__
        ns["__module__"] = cls_dict["__module__"]
        ns["__doc__"] = cls_dict["__doc__"]
        if args or kwargs:
            ns["__init__"] = partialmethod(cls_dict["__init__"], *args, **kwargs)
        return ns
    return types.new_class(
        cls.__name__,
        bases=(cls,),
        exec_body=build)


class Indices:
    def __init__(self, steps):
        self._steps = steps
        self.total = None

    def __contains__(self, x):
        if isinstance(self._steps, list):
            return x in self._steps
        elif isinstance(self._steps, slice):
            start = self._steps.start or 0
            if start < 0:
                start = self.total - start
            stop = self._steps.stop or self.total
            if stop < 0:
                stop = self.total - stop
            step = self._steps.step or 1
            return x >= start and x < stop and (x - start) % step == 0

    @classmethod
    def every_iters(cls, iters: int):
        return cls(slice(iters, None, iters))


def batched(array, batch_size):
    for i in range(0, len(array), batch_size):
        yield array[i:i+batch_size]


def get_rays(camera_poses: Float32[np.ndarray, "batch 3 4"],
             camera_intrinsics: Float32[np.ndarray, "batch 4"],
             xy: Float32[np.ndarray, "batch 2"],
             distortions: Optional[Distortions]) -> Tuple[Float32[np.ndarray, "batch 3"], Float32[np.ndarray, "batch 3"]]:
    out_shape = xy.shape[:-1]
    fx = camera_intrinsics[..., 0]
    fy = camera_intrinsics[..., 1]
    cx = camera_intrinsics[..., 2]
    cy = camera_intrinsics[..., 3]

    # Already broadcasted to correct shape
    directions = np.stack([(xy[..., 0] - cx) / fx, -(xy[..., 1] - cy) / fy, -np.ones_like(xy[..., 0])], -1)
    if distortions is not None:
        distortions = distortions.distort(directions)

    rotation = camera_poses[..., :3, :3]  # (..., 3, 3)
    directions = (directions[..., None, :] * rotation).sum(-1)
    eps = np.finfo(directions.dtype).eps
    norm = np.clip(np.linalg.norm(directions, axis=-1, keepdims=True), a_min=eps, a_max=None)
    directions = directions / norm
    origins = camera_poses[..., :3, 3].expand(out_shape + (3,))
    return origins, directions
