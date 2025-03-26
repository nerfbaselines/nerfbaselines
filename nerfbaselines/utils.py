import json
import numpy as np
from operator import mul
from functools import reduce
import threading
import sys
from typing import Any, Optional, Dict, TYPE_CHECKING, Union, List, TypeVar, Callable, Tuple, cast
import numpy as np
if TYPE_CHECKING:
    import torch
    import jax.numpy as jnp
# For Python>=3.9, use importlib.resources
if sys.version_info >= (3, 9):
    import importlib.resources as importlib_resources
else:
    import importlib_resources


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


def _xnp_astype(tensor: TTensor, dtype) -> TTensor:
    xnp = _get_xnp(tensor)
    if xnp.__name__ == "torch":
        return tensor.to(dtype=dtype)  # type: ignore
    return tensor.astype(dtype)  # type: ignore


class CancelledException(Exception):
    """
    Exception raised when an operation is cancelled using the ``CancellationToken``.
    """
    pass


class _CancellationTokenMeta(type):
    _local = threading.local()

    @property
    def current(cls) -> Optional["CancellationToken"]:
        return _CancellationTokenMeta._local.__dict__.get("current", None)

    def _set_token(cls, token):
        _CancellationTokenMeta._local.current = token
    

class CancellationToken(metaclass=_CancellationTokenMeta):
    """``CancellationToken`` is a context manager that can be used to cancel a long running operation. ``CancellationToken.current`` is a thread-local variable that can be used to access the current token.

    Example:
        ::

            # Create the token
            token = CancellationToken()
            
            # Now, you would  pass the token to another 
            # thread to allow it to cancel the operation

            # Make the token the current token for the thread
            with token:
                # Do something
                token.cancel_if_requested()
                # Do something else
                token.cancel_if_requested()

            # From the different thread, run.
            # It will stop the main thread at the nearest `cancel_if_requested`
            token.cancel()

    """
    def __init__(self):
        self._callbacks = []
        self._cancelled = False
        self._old_token = None

    def cancel(self):
        """
        Cancel the operation. This will raise a ``CancelledException`` in the current context.
        """
        self._cancelled = True
        for cb in self._callbacks:
            cb()

    def cancel_if_requested(self: Optional['CancellationToken'] = None):
        """
        Check if the operation has been cancelled and raise a ``CancelledException`` if it has.
        Can also be used as a static method: ``CancellationToken.cancel_if_requested()``
        """
        if self is None:
            # Static method
            if CancellationToken.current is not None:
                CancellationToken.current.cancel_if_requested()
        elif self._cancelled:
            raise CancelledException

    def __enter__(self):
        self._old_token = CancellationToken.current
        CancellationToken._set_token(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del exc_type, exc_value, traceback
        CancellationToken._set_token(self._old_token)


class Indices:
    """
    A class that represents a set of indices or slices. This is useful for specifying subsets of data
    training iterations or evaluation steps.
    """
    def __init__(self, steps, total=None):
        self._steps = steps
        self.total: Optional[int] = total

    def with_total(self, total):
        """
        Set the total number of elements in the sequence.

        Args:
            total: The total number of elements.

        Returns:
            The updated ``Indices`` object.
        """
        return Indices(self._steps, total=total)

    def __iter__(self):
        i = 0
        while self.total is None or i < self.total:
            if i in self:
                yield i
            i += 1

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
        """
        Create an ``Indices`` object that represents every ``iters`` iterations.
        For zero=False, this is equivalent to ``Indices(range(iters, total, iters))``.

        Args:
            iters: The number of iterations.
            zero: Whether to include 0 in the indices.

        Returns:
            The created ``Indices``
        """
        start = 0 if zero else iters
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


def padded_stack(tensors: Union[np.ndarray, Tuple[np.ndarray, ...], List[np.ndarray]]) -> np.ndarray:
    """
    Stack a list of tensors, padding them to the maximum shape.

    Args:
        tensors: A list of tensors to stack.

    Returns:
        The stacked tensor.
    """
    if not isinstance(tensors, (tuple, list)):
        return tensors
    max_shape = tuple(max(s) for s in zip(*[x.shape for x in tensors]))
    out_tensors = []
    for x in tensors:
        pads = [(0, m - s) for s, m in zip(x.shape, max_shape)]
        out_tensors.append(np.pad(x, pads))
    return np.stack(out_tensors, 0)


def convert_image_dtype(image: TTensor, dtype) -> TTensor:
    """
    Convert an image to a given dtype.

    Args:
        image: The input image.
        dtype: The output dtype.
    
    Returns:
        The converted image.
    """
    xnp = _get_xnp(image)
    is_torch = xnp.__name__ == "torch"
    if isinstance(dtype, str):
        dtype = getattr(xnp, dtype)
    uint8 = xnp.uint8
    if image.dtype == dtype:
        return image
    if image.dtype != uint8 and dtype != uint8:
        return _xnp_astype(image, dtype)
    if image.dtype == uint8 and dtype != uint8:
        if is_torch:
            # Faster torch cast
            if TYPE_CHECKING:
                assert isinstance(image, torch.Tensor)
            return image.to(dtype=dtype).div_(255.0)
        else:
            return _xnp_astype(image, dtype) / 255.0
    if image.dtype != uint8 and dtype == uint8:
        if is_torch:
            # Faster torch cast
            image_ = cast(Any, image)
            del image
            image_ = image_.mul(255.)
            if image_.dtype == xnp.float16:
                # NOTE: Clamp kernel is not available for half
                image_ = image_.to(dtype=xnp.float32)
            return cast(TTensor, image_.clamp_(0, 255).to(dtype=dtype))
        else:
            return _xnp_astype((image * 255.0).clip(0, 255), dtype)
    raise ValueError(f"cannot convert image from {image.dtype} to {dtype}")


def _srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)

del _srgb_to_linear


def _linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def image_to_srgb(tensor, dtype, color_space: Optional[str] = None, allow_alpha: bool = False, background_color: Optional[np.ndarray] = None):
    """
    Convert an image to sRGB color space (if it is not in sRGB color space already). 
    If the image has an alpha channel, it will be blended with a specified background color,
    or black if no background color is specified. The image will be converted to the specified dtype.
    In case the linear->sRGB conversion happens, the following formula is used:
    
            sRGB = 1.055 * linear^1/2.4 - 0.055, if linear > 0.0031308
            sRGB = 12.92 * linear, if linear <= 0.0031308


    Args:
        tensor: The input image tensor.
        dtype: The output dtype.
        color_space: The input color space. If None, it is assumed to be sRGB.
        allow_alpha: Whether to allow an alpha channel. If False, the alpha channel will be removed by blending with a black background.
        background_color: The background color to blend with if the image has an alpha channel. If None, it will be black.
        
    Returns:
        The converted image tensor.
    """
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
        tensor = _linear_to_srgb(tensor)

    # Round to 8-bit for fair comparisons
    tensor = convert_image_dtype(tensor, np.uint8)
    tensor = convert_image_dtype(tensor, dtype)
    return tensor


def _zipnerf_power_transformation(x, lam: float):
    m = abs(lam - 1) / lam
    return (((x / abs(lam - 1)) + 1) ** lam - 1) * m


def get_supported_palette_names() -> List[str]:
    """
    Get the names of the supported palettes.

    Returns:
        The names of the supported palettes.
    """
    import nerfbaselines
    with importlib_resources.open_text(nerfbaselines, 'palettes.json') as f:
        palettes_dict = json.load(f)
    return list(palettes_dict.keys())


def get_palette(palette: str) -> np.ndarray:
    """
    Get a color palette.

    Args:
        palette: The name of the palette.

    Returns:
        The color palette as a numpy array of shape [N, 3] and dtype uint8.
        N is usually 256, but it can be different for some palettes.
    """
    import nerfbaselines
    with importlib_resources.open_text(nerfbaselines, 'palettes.json') as f:
        palettes_dict = json.load(f)
    return np.array(palettes_dict[palette], dtype=np.uint8).reshape(-1, 3)



def apply_colormap(array: TTensor, *, pallete: str = "viridis", invert: bool = False) -> TTensor:
    """
    Apply a colormap to an array.

    Args:
        array: The input array.
        pallete: The matplotlib colormap to use.
        invert: Whether to invert the colormap.

    Returns:
        The array with the colormap applied.
    """
    xnp = _get_xnp(array)

    # Map to a color scale
    array_long = cast(TTensor, _xnp_astype(array * 255, xnp.int32).clip(0, 255))
    palette_np = get_palette(pallete)
    if xnp.__name__ == "torch":
        import torch
        pallete_array = cast(TTensor, torch.tensor(palette_np, dtype=torch.uint8, device=cast(torch.Tensor, array).device))
    else:
        pallete_array = cast(TTensor, xnp.array(palette_np, dtype=xnp.uint8))  # type: ignore
    if invert:
        array_long = 255 - array_long
    return cast(TTensor, pallete_array[array_long])  # type: ignore


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


#
# Pose utilities
#
def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def get_transform_and_scale(transform):
    """
    Decomposes a transform-scale matrix into a scale and a transform matrix.
    The scale is applied after the transform.

    Args:
        transform: A 4x4 or 3x4 matrix.

    Returns:
        A tuple of the transform matrix and the scale.
    """
    assert len(transform.shape) == 2, "Transform should be a 4x4 or a 3x4 matrix."
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    assert np.allclose(scale, scale[0], rtol=1e-3, atol=0)
    scale = float(np.mean(scale).item())
    transform = transform.copy()
    transform[:3, :] /= scale
    return transform, scale


def apply_transform(transform, poses):
    """
    Applies a transform-scale matrix to a set of poses.
    The scale is applied after the transform.

    Args:
        transform: A 4x4 or 3x4 matrix.
        poses: A set of poses.

    Returns:
        The transformed poses.
    """
    transform, scale = get_transform_and_scale(transform)
    poses = unpad_poses(transform @ pad_poses(poses))
    poses[..., :3, 3] *= scale
    return poses


def invert_transform(transform, has_scale=False):
    """
    Inverts a transform or a transform-scale matrix. By default,
    it assumes the transform has no scale.

    Args:
        transform: A 4x4 or 3x4 matrix.
        has_scale: Whether the transform has scale.

    Returns:
        The inverted transform.
    """
    scale = None
    if has_scale:
        transform, scale = get_transform_and_scale(transform)
    else:
        transform = transform.copy()
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    transform[..., :3, :] = np.concatenate([R.T, -np.matmul(R.T, t[..., None])], axis=-1)
    if scale is not None:
        transform[..., :3, :3] *= 1/scale
    return transform


def quaternion_multiply(q1, q2):
    """
    Multiply two sets of quaternions.

    Args:
        q1: A quaternion.
        q2: A quaternion.

    Returns:
        The multiplied quaternions.
    """
    a = q1[..., 0]*q2[...,0] - q1[...,1]*q2[...,1] - q1[...,2]*q2[...,2] - q1[...,3]*q2[...,3]
    b = q1[..., 0]*q2[...,1] + q1[...,1]*q2[...,0] + q1[...,2]*q2[...,3] - q1[...,3]*q2[...,2]
    c = q1[..., 0]*q2[...,2] - q1[...,1]*q2[...,3] + q1[...,2]*q2[...,0] + q1[...,3]*q2[...,1]
    d = q1[..., 0]*q2[...,3] + q1[...,1]*q2[...,2] - q1[...,2]*q2[...,1] + q1[...,3]*q2[...,0]
    return np.stack([a, b, c, d], axis=-1)


def quaternion_conjugate(q):
    """
    Return quaternion-conjugate of quaternion q̄

    Args:
        q: A quaternion.
    
    Returns:
        The quaternion conjugate.
    """
    return np.stack([+q[...,0], -q[...,1], -q[..., 2], -q[...,3]], -1)

def quaternion_to_rotation_matrix(r):
    """
    Convert input quaternion to a rotation matrix.

    Args:
        r: A quaternion.

    Returns:
        The rotation matrix.
    """
    norm = np.sqrt((r**2).sum(-1))

    q = r / norm[..., None]

    R = np.zeros(r.shape[:-1] + (3, 3), dtype=r.dtype)

    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R[..., 0, 0] = 1 - 2 * (y*y + z*z)
    R[..., 0, 1] = 2 * (x*y - r*z)
    R[..., 0, 2] = 2 * (x*z + r*y)
    R[..., 1, 0] = 2 * (x*y + r*z)
    R[..., 1, 1] = 1 - 2 * (x*x + z*z)
    R[..., 1, 2] = 2 * (y*z - r*x)
    R[..., 2, 0] = 2 * (x*z - r*y)
    R[..., 2, 1] = 2 * (y*z + r*x)
    R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def rotation_matrix_to_quaternion(R):
    """Convert input 3x3 rotation matrix to unit quaternion.

    Assuming an orthogonal 3x3 matrix ℛ rotates a vector v such that

        v' = ℛ * v,

    we can also express this rotation in terms of a unit quaternion R such that

        v' = R * v * R⁻¹,

    where v and v' are now considered pure-vector quaternions.  This function
    returns that quaternion.  If `rot` is not orthogonal, the "closest" orthogonal
    matrix is used; see Notes below.

    Parameters
    ----------
    R : (...Nx3x3) float array
        Each 3x3 matrix represents a rotation by multiplying (from the left)
        a column vector to produce a rotated column vector.  Note that this
        input may actually have ndims>3; it is just assumed that the last
        two dimensions have size 3, representing the matrix.

    Returns
    -------
    q : array of quaternions
        Unit quaternions resulting in rotations corresponding to input
        rotations.  Output shape is rot.shape[:-2].

    Raises
    ------
    LinAlgError
        If any of the eigenvalue solutions does not converge

    Notes
    -----
    This function uses Bar-Itzhack's algorithm to allow for
    non-orthogonal matrices.  [J. Guidance, Vol. 23, No. 6, p. 1085
    <http://dx.doi.org/10.2514/2.4654>]  This will almost certainly be quite a bit
    slower than simpler versions, though it will be more robust to numerical errors
    in the rotation matrix.  Also note that the Bar-Itzhack paper uses some pretty
    weird conventions.  The last component of the quaternion appears to represent
    the scalar, and the quaternion itself is conjugated relative to the convention
    used throughout the quaternionic module.
    """

    from scipy import linalg
    rot = np.array(R, copy=False)
    shape = rot.shape[:-2]


    K3 = np.empty(shape+(4, 4), dtype=rot.dtype)
    K3[..., 0, 0] = (rot[..., 0, 0] - rot[..., 1, 1] - rot[..., 2, 2])/3
    K3[..., 0, 1] = (rot[..., 1, 0] + rot[..., 0, 1])/3
    K3[..., 0, 2] = (rot[..., 2, 0] + rot[..., 0, 2])/3
    K3[..., 0, 3] = (rot[..., 1, 2] - rot[..., 2, 1])/3
    K3[..., 1, 0] = K3[..., 0, 1]
    K3[..., 1, 1] = (rot[..., 1, 1] - rot[..., 0, 0] - rot[..., 2, 2])/3
    K3[..., 1, 2] = (rot[..., 2, 1] + rot[..., 1, 2])/3
    K3[..., 1, 3] = (rot[..., 2, 0] - rot[..., 0, 2])/3
    K3[..., 2, 0] = K3[..., 0, 2]
    K3[..., 2, 1] = K3[..., 1, 2]
    K3[..., 2, 2] = (rot[..., 2, 2] - rot[..., 0, 0] - rot[..., 1, 1])/3
    K3[..., 2, 3] = (rot[..., 0, 1] - rot[..., 1, 0])/3
    K3[..., 3, 0] = K3[..., 0, 3]
    K3[..., 3, 1] = K3[..., 1, 3]
    K3[..., 3, 2] = K3[..., 2, 3]
    K3[..., 3, 3] = (rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2])/3

    if not shape:
        q = np.empty((4,), dtype=rot.dtype)
        eigvals, eigvecs = linalg.eigh(K3.T, subset_by_index=(3, 3))
        del eigvals
        q[0] = eigvecs[-1].item()
        q[1:] = -eigvecs[:-1].flatten()
        return q
    else:
        q = np.empty(shape+(4,), dtype=rot.dtype)
        for flat_index in range(reduce(mul, shape)):
            multi_index = np.unravel_index(flat_index, shape)
            eigvals, eigvecs = linalg.eigh(K3[multi_index], subset_by_index=(3, 3))
            del eigvals
            q[multi_index+(0,)] = eigvecs[-1]
            q[multi_index+(slice(1,None),)] = -eigvecs[:-1].flatten()
        return q
