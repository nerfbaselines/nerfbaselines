import sys
from typing import cast
import numpy as np
import pytest
from nerfbaselines.metrics import torchmetrics_ssim, dmpix_ssim
from nerfbaselines import metrics


@pytest.mark.extras
@pytest.mark.filterwarnings("ignore::UserWarning:torchvision")
@pytest.mark.parametrize("kernel_size", [None, 3])
@pytest.mark.parametrize("sigma", [None, 0.5])
def test_torchmetrics_ssim(kernel_size, sigma):
    import torch
    import torchmetrics
    import dm_pix

    np.random.seed(42)
    a = np.random.rand(3, 47, 41, 3) * 0.9 + 0.05
    b = np.random.rand(3, 47, 41, 3) * 0.9 + 0.05

    kwargs = {}
    if kernel_size is not None:
        kwargs["kernel_size"] = kernel_size
    if sigma is not None:
        kwargs["sigma"] = sigma

    a_torch = torch.from_numpy(a).permute(0, 3, 1, 2)
    b_torch = torch.from_numpy(b).permute(0, 3, 1, 2)
    reference_ssim = cast(
        torch.Tensor, torchmetrics.functional.structural_similarity_index_measure(a_torch, b_torch, reduction="none", data_range=(0.0, 1.0), **kwargs)
    ).numpy()
    ssim = torchmetrics_ssim(a, b, data_range=(0.0, 1.0), **kwargs)

    assert isinstance(ssim, np.ndarray)
    assert ssim.shape == (3,)
    np.testing.assert_allclose(ssim, reference_ssim, atol=1e-5, rtol=0)

    if kernel_size is None and sigma is None:
        # SSIM matches for default parameters
        reference2 = dm_pix.ssim(a, b, **kwargs)
        np.testing.assert_allclose(ssim, reference2, atol=1e-5, rtol=0)


@pytest.mark.extras
@pytest.mark.filterwarnings("ignore::UserWarning:torchvision")
@pytest.mark.parametrize("kernel_size", [None, 3])
@pytest.mark.parametrize("sigma", [None, 0.5])
def test_dmpix_ssim(kernel_size, sigma):
    import dm_pix

    np.random.seed(42)
    a = np.random.rand(3, 47, 41, 3) * 0.9 + 0.05
    b = np.random.rand(3, 47, 41, 3) * 0.9 + 0.05

    kwargs = {}
    kwargs_dmpix = {}
    if kernel_size is not None:
        kwargs["kernel_size"] = kernel_size
        kwargs_dmpix["filter_size"] = kernel_size
    if sigma is not None:
        kwargs["sigma"] = sigma
        kwargs_dmpix["filter_sigma"] = sigma
    reference_ssim = dm_pix.ssim(a, b, **kwargs_dmpix)
    ssim = dmpix_ssim(a, b, **kwargs)

    assert isinstance(ssim, np.ndarray)
    assert ssim.shape == (3,)
    np.testing.assert_allclose(ssim, reference_ssim, atol=1e-5, rtol=0)


@pytest.mark.filterwarnings("ignore::UserWarning:torchvision")
@pytest.mark.parametrize("metric", ["torchmetrics_ssim", "dmpix_ssim", "ssim", "mse", "mae", "psnr"])
def test_metric(metric):
    np.random.seed(42)
    batch_shapes = [
        (3,),
        (2, 2),
        (
            2,
            1,
            1,
        ),
    ]
    for bs in batch_shapes:
        a = np.random.rand(*bs, 47, 31, 3)
        b = np.random.rand(*bs, 47, 31, 3)

        val = getattr(metrics, metric)(a, b)
        assert isinstance(val, np.ndarray)
        assert val.shape == bs

        # Different shape raises error
        with pytest.raises(Exception):
            getattr(metrics, metric)(a, b[:-1])


def test_psnr():
    np.random.seed(42)
    batch_shapes = [
        (3,),
        (2, 2),
        (
            2,
            1,
            1,
        ),
    ]
    for bs in batch_shapes:
        a = np.random.rand(*bs, 47, 31, 3)
        b = np.random.rand(*bs, 47, 31, 3)

        val = metrics.psnr(a, b)
        val2 = metrics.psnr(metrics.mse(a, b))
        assert isinstance(val, np.ndarray)
        assert val.shape == bs
        assert isinstance(val2, np.ndarray)
        assert val2.shape == bs

        np.testing.assert_allclose(val, val2, atol=1e-5, rtol=0)


@pytest.mark.extras
@pytest.mark.filterwarnings("ignore:The parameter 'pretrained' is deprecated since 0.13")
@pytest.mark.filterwarnings("ignore:Importing `spectral_angle_mapper` from `torchmetrics.functional` was deprecated")
@pytest.mark.parametrize("metric", ["lpips_vgg", "lpips_alex"])
def test_lpips(metric):
    metrics._LPIPS_CACHE.clear()
    metrics._LPIPS_GPU_AVAILABLE = None
    sys.modules.pop("nerfbaselines._metrics_lpips", None)
    np.random.seed(42)
    batch_shapes = [
        (3,),
        (2, 2),
        (
            2,
            1,
            1,
        ),
    ]
    for bs in batch_shapes:
        a = np.random.rand(*bs, 33, 38, 3)
        b = np.random.rand(*bs, 33, 38, 3)

        val = getattr(metrics, metric)(a, b)
        assert isinstance(val, np.ndarray)
        assert val.shape == bs

        # Different shape raises error
        with pytest.raises(Exception):
            getattr(metrics, metric)(a, b[:-1])
