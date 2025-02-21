import numpy as np
from unittest import mock
import pytest
from time import sleep, perf_counter
from nerfbaselines.utils import Indices
from nerfbaselines.utils import CancellationToken, CancelledException


def test_indices_last():
    indices = Indices([-1])
    indices.total = 12
    for i in range(12):
        if i == indices.total - 1:
            assert i in indices
        else:
            assert i not in indices


class TimeLimitCancellationToken(CancellationToken):
    def __init__(self, timeout=0.005):
        super().__init__()
        self.timeout = timeout
        self.start = perf_counter()

    @property
    def _cancelled(self):
        return perf_counter() - self.start > self.timeout

    @_cancelled.setter
    def _cancelled(self, value):
        del value
        pass


def test_cancel():
    was_called = False

    def fn():
        nonlocal was_called
        was_called = True
        for _ in range(100):
            CancellationToken.cancel_if_requested()
            sleep(0.001)
        raise Exception("Should not be reached")

    token = TimeLimitCancellationToken()
    with pytest.raises(CancelledException):
        with token:
            fn()
    assert was_called, "Function was not called"


def test_cancel_generator():
    was_called = False

    def fn():
        nonlocal was_called
        was_called = True
        for i in range(100):
            yield i
            CancellationToken.cancel_if_requested()
            sleep(0.001)
        raise Exception("Should not be reached")

    token = TimeLimitCancellationToken()
    out = []
    with pytest.raises(CancelledException):
        with token:
            for i in fn():
                out.append(i)
    assert len(out) > 1, "Function was not called"
    assert was_called, "Function was not called"


def test_get_resources_utilization_info():
    from nerfbaselines.training import get_resources_utilization_info

    info = get_resources_utilization_info()
    assert isinstance(info, dict)


def test_tuple_click_type():
    import click
    from nerfbaselines.cli._common import TupleClickType

    with mock.patch("sys.argv", ["test"]), \
        pytest.raises(SystemExit) as excinfo:
        @click.command()
        @click.option("--val", type=TupleClickType(), default=None)
        def cmd(val):
            assert val is None
        cmd()
    assert excinfo.value.code == 0

    with mock.patch("sys.argv", ["test"]), \
        pytest.raises(SystemExit) as excinfo:
        @click.command()
        @click.option("--val", type=TupleClickType(), default=())
        def cmd(val):
            assert val == ()
        cmd()
    assert excinfo.value.code == 0

    with mock.patch("sys.argv", ["test", "--val", "1,2"]), \
        pytest.raises(SystemExit) as excinfo:
        @click.command()
        @click.option("--val", type=TupleClickType(), default=())
        def cmd(val):
            assert val == ("1","2")
        cmd()
    assert excinfo.value.code == 0


def test_convert_image_dtype_numpy():
    from nerfbaselines.utils import convert_image_dtype

    # Test keep same dtype
    for dtype in ['uint8', 'float32', 'float64', 'float16']:
        arr = np.full((10, 10), 128, dtype=getattr(np, dtype))
        out = convert_image_dtype(arr, getattr(np, dtype))
        assert out is arr
        out = convert_image_dtype(arr, dtype)
        assert out is arr

    # Test uint8 -> float32
    arr = np.full((10, 10), 128, dtype=np.uint8)
    for dtype in ['float32', 'float64', 'float16']:
        out = convert_image_dtype(arr, getattr(np, dtype))
        assert out.dtype == getattr(np, dtype)
        assert out.shape == arr.shape
        assert abs(out[0, 0] - 128/255) < 1e-5
        out = convert_image_dtype(arr, dtype)
        assert out.dtype == getattr(np, dtype)
        assert out.shape == arr.shape
        assert abs(out[0, 0] - 128/255) < 1e-5

    # Test float -> uint8
    for dtype in ['float32', 'float64', 'float16']:
        arr = np.array([-1, 0, 0.2, 1, 2], dtype=getattr(np, dtype))
        out = convert_image_dtype(arr, np.uint8)
        assert out.dtype == np.uint8
        assert out.shape == arr.shape
        assert tuple(out.tolist()) == (0, 0, 51, 255, 255)
        out = convert_image_dtype(arr, 'uint8')
        assert out.dtype == np.uint8
        assert out.shape == arr.shape
        assert tuple(out.tolist()) == (0, 0, 51, 255, 255)


def _test_convert_image_dtype_torch(torch):
    from nerfbaselines.utils import convert_image_dtype

    # Test keep same dtype
    xnp = torch
    for dtype in ['uint8', 'float32', 'float64', 'float16']:
        arr = xnp.full((10, 10), 128, dtype=getattr(xnp, dtype))
        out = convert_image_dtype(arr, getattr(xnp, dtype))
        assert out is arr
        out = convert_image_dtype(arr, dtype)
        assert out is arr

    # Test uint8 -> float32
    arr = xnp.full((10, 10), 128, dtype=xnp.uint8)
    for dtype in ['float32', 'float64', 'float16']:
        out = convert_image_dtype(arr, getattr(xnp, dtype))
        assert out.dtype == getattr(xnp, dtype)
        assert out.shape == arr.shape
        assert abs(out[0, 0] - 128/255) < 1e-5
        out = convert_image_dtype(arr, dtype)
        assert out.dtype == getattr(xnp, dtype)
        assert out.shape == arr.shape
        assert abs(out[0, 0] - 128/255) < 1e-5

    # Test float -> uint8
    for dtype in ['float32', 'float64', 'float16']:
        arr = xnp.tensor([-1, 0, 0.2, 1, 2], dtype=getattr(xnp, dtype))
        out = convert_image_dtype(arr, xnp.uint8)
        assert out.dtype == xnp.uint8
        assert out.shape == arr.shape
        assert tuple(out.tolist()) == (0, 0, 51, 255, 255)
        out = convert_image_dtype(arr, 'uint8')
        assert out.dtype == xnp.uint8
        assert out.shape == arr.shape
        assert tuple(out.tolist()) == (0, 0, 51, 255, 255)


@pytest.mark.extras
def test_convert_image_dtype_torch_cpu(torch_cpu):
    del torch_cpu
    import torch
    _test_convert_image_dtype_torch(torch)


def test_convert_image_dtype_torch_mock(mock_torch):
    del mock_torch
    import torch
    _test_convert_image_dtype_torch(torch)
