import sys
from typing import TYPE_CHECKING, Union, Any, overload
import numpy as np


if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp

if TYPE_CHECKING:
    XTensor = Union[np.ndarray, 'torch.Tensor', 'jnp.ndarray']
else:
    XTensor = np.ndarray


if TYPE_CHECKING:
    @overload
    def astensor(x: Any, xnp=torch) -> torch.Tensor:
        ...

    @overload
    def astensor(x: Any, xnp=jnp) -> jnp.ndarray:
        ...

    @overload
    def astensor(x: Any, xnp=np) -> np.ndarray:
        ...


def astensor(x: Any, xnp) -> XTensor:
    source_module = getattr(x, '__class__', None)
    if source_module is not None:
        source_module = getattr(source_module, '__module__', None)
    if xnp is np or xnp.__name__ == "jax.numpy":
        if source_module is not None and source_module.startswith("torch"):
            x = x.detach().cpu().numpy()
        return xnp.asarray(x)
    elif xnp.__name__ == "torch":
        if TYPE_CHECKING:
            xnp = torch
        if isinstance(x, xnp.Tensor):
            return x
        else:
            return xnp.from_numpy(astensor(x, np))
    else:
        raise ValueError(f"Unknown numpy-like library {xnp}")


if TYPE_CHECKING:
    def getbackend(x: XTensor):
        return np
else:
    def getbackend(x: XTensor):
        if isinstance(x, np.ndarray):
            return np
        source_module = getattr(x, '__class__', None)
        if source_module is not None:
            source_module = getattr(source_module, '__module__', None)
        if source_module is not None:
            if source_module.startswith("jax"):
                import jax.numpy as jnp
                return jnp
            return sys.modules[source_module]
        else:
            raise RuntimeError("Unknown backend for tensor: ", x)


def copy(x, xnp=None):
    if xnp is None:
        xnp = getbackend(x)
    if xnp.__name__ == "torch":
        if TYPE_CHECKING:
            import torch
            xnp = torch
        return xnp.clone(x)
    return xnp.copy(x)


def astype(x, dtype, xnp=None):
    if xnp is None:
        xnp = getbackend(x)
    if xnp.__name__ == "torch":
        return x.to(dtype=dtype)
    return x.astype(dtype=dtype)


def assert_same_xnp(*args, xnp=None):
    if xnp is None:
        xnp = getbackend(args[0])
    backends = [getbackend(arg).__name__ for arg in args[1:]]
    for backend in backends:
        if backend != xnp.__name__:
            raise ValueError(f"All tensors must be of the same type ({xnp}). Got: {backends}")
