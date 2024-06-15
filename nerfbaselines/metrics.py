from functools import wraps
import numpy
from typing import Optional, Callable, Union, Sequence, Tuple, cast
import numpy as np
import warnings


warnings.filterwarnings("ignore", category=UserWarning, message=".*?The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future*?", module="torchvision")
warnings.filterwarnings("ignore", category=UserWarning, message=".*?Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13*?", module="torchvision")
warnings.filterwarnings("ignore", category=DeprecationWarning, message="distutils.*? is deprecated and will be removed in a future version.*?", module="torchmetrics")
warnings.filterwarnings("ignore", category=FutureWarning, message="Importing `spectral_angle_mapper`*?", module="torchmetrics")


def _wrap_metric_arbitrary_shape(fn):
    @wraps(fn)
    def wrapped(a, b, **kwargs):
        bs = a.shape[:-3]
        a = np.reshape(a, (-1, *a.shape[-3:]))
        b = np.reshape(b, (-1, *b.shape[-3:]))
        out = fn(a, b, **kwargs)
        return np.reshape(out, bs)

    return wrapped


@_wrap_metric_arbitrary_shape
def dmpix_ssim(
    a: np.ndarray,
    b: np.ndarray,
    *,
    max_val: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
    return_map: bool = False,
    filter_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> np.ndarray:
    """Computes the structural similarity index (SSIM) between image pairs.

    This function is based on the standard SSIM implementation from:
    Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli,
    "Image quality assessment: from error visibility to structural similarity",
    in IEEE Transactions on Image Processing, vol. 13, no. 4, pp. 600-612, 2004.

    This function was modeled after tf.image.ssim, and should produce comparable
    output.

    Note: the true SSIM is only defined on grayscale. This function does not
    perform any colorspace transform. If the input is in a color space, then it
    will compute the average SSIM.

    NOTE: This function exactly matches dm_pix.ssim

    Args:
        a: First image (or set of images).
        b: Second image (or set of images).
        max_val: The maximum magnitude that `a` or `b` can have.
        kernel_size: Window size (>= 1). Image dims must be at least this small.
        sigma: The bandwidth of the Gaussian used for filtering (> 0.).
        k1: One of the SSIM dampening parameters (> 0.).
        k2: One of the SSIM dampening parameters (> 0.).
        return_map: If True, will cause the per-pixel SSIM "map" to be returned.
        precision: The numerical precision to use when performing convolution.

    Returns:
        Each image's mean SSIM, or a tensor of individual values if `return_map`.
    """
    # DO NOT REMOVE - Logging usage.

    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"

    if filter_fn is None:
        # Construct a 1D Gaussian blur filter.
        hw = kernel_size // 2
        shift = (2 * hw - kernel_size + 1) / 2
        f_i = ((np.arange(kernel_size) - hw + shift) / sigma) ** 2
        filt = np.exp(-0.5 * f_i)
        filt /= np.sum(filt)

        # Construct a 1D convolution.
        def filter_fn_1(z):
            return np.convolve(z, filt, mode="valid")

        # jax.vmap(filter_fn_1)
        filter_fn_vmap = lambda x: np.stack([filter_fn_1(y) for y in x], 0)  # noqa: E731

        # Apply the vectorized filter along the y axis.
        def filter_fn_y(z):
            z_flat = np.moveaxis(z, -3, -1).reshape((-1, z.shape[-3]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-2],
                z.shape[-1],
                -1,
            )
            z_filtered = np.moveaxis(filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -3)
            return z_filtered

        # Apply the vectorized filter along the x axis.
        def filter_fn_x(z):
            z_flat = np.moveaxis(z, -2, -1).reshape((-1, z.shape[-2]))
            z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
                z.shape[-3],
                z.shape[-1],
                -1,
            )
            z_filtered = np.moveaxis(filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -2)
            return z_filtered

        # Apply the blur in both x and y.
        filter_fn = lambda z: filter_fn_y(filter_fn_x(z))  # noqa: E731

    mu0 = filter_fn(a)
    mu1 = filter_fn(b)
    mu00 = mu0 * mu0
    mu11 = mu1 * mu1
    mu01 = mu0 * mu1
    sigma00 = filter_fn(a**2) - mu00
    sigma11 = filter_fn(b**2) - mu11
    sigma01 = filter_fn(a * b) - mu01

    # Clip the variances and covariances to valid values.
    # Variance must be non-negative:
    epsilon = np.finfo(np.float32).eps ** 2
    sigma00 = np.maximum(epsilon, sigma00)
    sigma11 = np.maximum(epsilon, sigma11)
    sigma01 = np.sign(sigma01) * np.minimum(np.sqrt(sigma00 * sigma11), np.abs(sigma01))

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
    denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
    ssim_map = numer / denom
    ssim_value = np.mean(ssim_map, tuple(range(-3, 0)))
    return ssim_map if return_map else ssim_value


def _gaussian(kernel_size: int, sigma: float, dtype: np.dtype) -> np.ndarray:
    """Compute 1D gaussian kernel.

    Args:
        kernel_size: size of the gaussian kernel
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor

    Example:
        >>> _gaussian(3, 1, torch.float, 'cpu')
        tensor([0.2741, 0.4519, 0.2741])

    """
    dist = np.arange((1 - kernel_size) / 2, (1 + kernel_size) / 2, 1, dtype=dtype)
    gauss = np.exp(-np.power(dist / sigma, 2) / 2)
    return gauss / gauss.sum()  # (kernel_size)


def _gaussian_kernel_2d(
    kernel_size: Sequence[int],
    sigma: Sequence[float],
    dtype: np.dtype,
) -> np.ndarray:
    """Compute 2D gaussian kernel.

    Args:
        channel: number of channels in the image
        kernel_size: size of the gaussian kernel as a tuple (h, w)
        sigma: Standard deviation of the gaussian kernel
        dtype: data type of the output tensor

    Example:
        >>> _gaussian_kernel_2d(1, (5,5), (1,1), torch.float, "cpu")
        tensor([[0.0030, 0.0133, 0.0219, 0.0133, 0.0030],
                [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                [0.0219, 0.0983, 0.1621, 0.0983, 0.0219],
                [0.0133, 0.0596, 0.0983, 0.0596, 0.0133],
                [0.0030, 0.0133, 0.0219, 0.0133, 0.0030]])

    """
    gaussian_kernel_x = _gaussian(kernel_size[0], sigma[0], dtype)[None]
    gaussian_kernel_y = _gaussian(kernel_size[1], sigma[1], dtype)[None]
    return np.matmul(gaussian_kernel_x.T, gaussian_kernel_y)  # (kernel_size, 1) * (1, kernel_size)


@_wrap_metric_arbitrary_shape
def torchmetrics_ssim(
    a: np.ndarray,
    b: np.ndarray,
    *,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    data_range: Optional[Union[float, Tuple[float, float]]] = None,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute Structural Similarity Index Measure.

    NOTE: this metric exactly matches torchmetrics.ssim

    Args:
        preds: estimated image
        target: ground truth image
        gaussian_kernel: If true (default), a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.

    """
    assert a.ndim == b.ndim and a.ndim == 4, f"Expected preds and target to have dimension less than 5, got {a.ndim} and {b.ndim}"
    a = np.transpose(a, (0, 3, 1, 2))
    b = np.transpose(b, (0, 3, 1, 2))

    def conv2d(a, f):
        shape = a.shape
        a = np.reshape(a, (-1, a.shape[-2], a.shape[-1]))

        def conv2d_single(a, f):
            s = f.shape + tuple(np.subtract(a.shape, f.shape) + 1)
            strd = numpy.lib.stride_tricks.as_strided
            subM = strd(a, shape=s, strides=a.strides * 2)
            return np.einsum("ij,ijkl->kl", f, subM)

        out = np.stack([conv2d_single(a[i], f) for i in range(len(a))])
        return np.reshape(out, shape[:-2] + out.shape[-2:])

    if not isinstance(kernel_size, Sequence):
        kernel_size = 2 * [kernel_size]
    if not isinstance(sigma, Sequence):
        sigma = 2 * [sigma]

    if len(kernel_size) != len(b.shape) - 2:
        raise ValueError(f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality," f" which is: {len(b.shape)}")
    if len(kernel_size) not in (2, 3):
        raise ValueError(f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}")
    if len(sigma) != len(b.shape) - 2:
        raise ValueError(f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality," f" which is: {len(b.shape)}")
    if len(sigma) not in (2, 3):
        raise ValueError(f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}")

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    if data_range is None:
        data_range = max(a.max() - a.min(), b.max() - b.min())
    elif isinstance(data_range, tuple):
        a = np.clip(a, data_range[0], data_range[1])
        b = np.clip(b, data_range[0], data_range[1])
        data_range = data_range[1] - data_range[0]
    assert isinstance(data_range, float), f"Expected data_range to be float, got {type(data_range)}"

    c1 = pow(k1 * data_range, 2)
    c2 = pow(k2 * data_range, 2)

    dtype = a.dtype
    gauss_kernel_size = [int(3.5 * s + 0.5) * 2 + 1 for s in sigma]

    pad_h = (gauss_kernel_size[0] - 1) // 2
    pad_w = (gauss_kernel_size[1] - 1) // 2

    a = np.pad(a, ((0, 0), (0, 0), (pad_w, pad_w), (pad_h, pad_h)), mode="reflect")
    b = np.pad(b, ((0, 0), (0, 0), (pad_w, pad_w), (pad_h, pad_h)), mode="reflect")
    if gaussian_kernel:
        kernel = _gaussian_kernel_2d(gauss_kernel_size, sigma, dtype)
    else:
        kernel = np.ones(kernel_size, dtype=dtype) / np.prod(np.array(kernel_size, dtype=dtype))

    input_list = np.concatenate((a, b, a * a, b * b, a * b))  # (5 * B, C, H, W)

    outputs: np.ndarray = conv2d(input_list, kernel)

    output_list = np.split(outputs, 5)

    mu_pred_sq = np.power(output_list[0], 2)
    mu_target_sq = np.power(output_list[1], 2)
    mu_pred_target = output_list[0] * output_list[1]

    sigma_pred_sq: np.ndarray = output_list[2] - mu_pred_sq
    sigma_target_sq: np.ndarray = output_list[3] - mu_target_sq
    sigma_pred_target: np.ndarray = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target.astype(dtype) + c2
    lower = (sigma_pred_sq + sigma_target_sq).astype(dtype) + c2

    ssim_idx_full_image = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)

    ssim_idx: np.ndarray = ssim_idx_full_image[..., pad_h:-pad_h, pad_w:-pad_w]
    return np.reshape(ssim_idx, (ssim_idx.shape[0], -1)).mean(-1)


def _mean(metric):
    return np.mean(metric, (-3, -2, -1))


def _normalize_input(a):
    return np.clip(a, 0, 1).astype(np.float32)


def ssim(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Structural Similarity Index Measure (the higher the better).
    Args:
        a: Tensor of prediction images [B, H, W, C].
        b: Tensor of target images [B, H, W, C].
    Returns:
        Tensor of mean SSIM values for each image [B].
    """
    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"
    a = _normalize_input(a)
    b = _normalize_input(b)
    return dmpix_ssim(a, b)


def mse(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Mean Squared Error (the lower the better).
    Args:
        a: Tensor of prediction images [B, H, W, C].
        b: Tensor of target images [B, H, W, C].
    Returns:
        Tensor of mean squared error values for each image [B].
    """
    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"
    a = _normalize_input(a)
    b = _normalize_input(b)
    return _mean((a - b) ** 2)


def mae(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Mean Absolute Error (the lower the better).
    Args:
        a: Tensor of prediction images [B, H, W, C].
        b: Tensor of target images [B, H, W, C].
    Returns:
        Tensor of mean absolute error values for each image [B].
    """
    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"
    a = _normalize_input(a)
    b = _normalize_input(b)
    return _mean(np.abs(a - b))


def psnr(a: Union[np.ndarray, np.float32, np.float64], b: Optional[np.ndarray] = None) -> Union[np.ndarray, np.float32, np.float64]:
    """
    Compute Peak Signal to Noise Ratio (the higher the better).
    It can reuse computed MSE values if b is None.
    Args:
        a: Tensor of prediction images [B, H, W, C] or a tensor of MSE values [B] (b must be None in that case).
        b: Tensor of target images [B, H, W, C] or None (if a are MSE values).
    Returns:
        Tensor of PSNR values for each image [B].
    """
    mse_value = a if b is None else mse(cast(np.ndarray, a), b)
    return -10 * np.log10(mse_value)


_LPIPS_CACHE = {}
_LPIPS_GPU_AVAILABLE = None


def _lpips(a, b, net, version="0.1"):
    global _LPIPS_GPU_AVAILABLE
    assert a.shape == b.shape, f"Images must have the same shape, got {a.shape} and {b.shape}"
    assert a.dtype.kind == "f" and b.dtype.kind == "f", f"Expected floating point inputs, got {a.dtype} and {b.dtype}"

    import torch

    lp_net = _LPIPS_CACHE.get(net)
    if lp_net is None:
        from ._metrics_lpips import LPIPS

        lp_net = LPIPS(net=net, version=version)
        _LPIPS_CACHE[net] = lp_net

    device = torch.device("cpu")
    if _LPIPS_GPU_AVAILABLE is None:
        _LPIPS_GPU_AVAILABLE = torch.cuda.is_available()
        if _LPIPS_GPU_AVAILABLE:
            try:
                lp_net.cuda()
                torch.zeros((1,), device="cuda").cpu()
            except Exception:
                _LPIPS_GPU_AVAILABLE = False

    if _LPIPS_GPU_AVAILABLE:
        device = torch.device("cuda")

    batch_shape = a.shape[:-3]
    img_shape = a.shape[-3:]
    a = _normalize_input(a)
    b = _normalize_input(b)
    with torch.no_grad():
        a = torch.from_numpy(a).float().view(-1, *img_shape).permute(0, 3, 1, 2).mul_(2).sub_(1).to(device)
        b = torch.from_numpy(b).float().view(-1, *img_shape).permute(0, 3, 1, 2).mul_(2).sub_(1).to(device)
        out = cast(torch.Tensor, lp_net.to(device).forward(a, b))
        out = out.detach().cpu().numpy().reshape(batch_shape)
        return out


def lpips_alex(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Learned Perceptual Image Patch Similarity (the lower the better).
    Args:
        a: Tensor of prediction images [B..., H, W, C].
        b: Tensor of target images [B..., H, W, C].
    Returns:
        Tensor of LPIPS values for each image [B...].
    """
    return _lpips(a, b, net="alex")


def lpips_vgg(a: np.ndarray, b: np.ndarray) -> Union[np.ndarray, np.float32]:
    """
    Compute Learned Perceptual Image Patch Similarity (the lower the better).
    Args:
        a: Tensor of prediction images [B..., H, W, C].
        b: Tensor of target images [B..., H, W, C].
    Returns:
        Tensor of LPIPS values for each image [B...].
    """
    return _lpips(a, b, net="vgg")


lpips = lpips_alex
