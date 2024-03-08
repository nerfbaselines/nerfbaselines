import pytest
from time import sleep, perf_counter
from nerfbaselines.utils import Indices
from nerfbaselines.utils import cancellable, CancellationToken, CancelledException

try:
    from typing import Literal, Optional
except ImportError:
    from typing_extensions import Literal, Optional


def test_indices_last():
    indices = Indices([-1])
    indices.total = 12
    for i in range(12):
        if i == indices.total - 1:
            assert i in indices
        else:
            assert i not in indices


class TimeLimitCancellationToken(CancellationToken):
    def __init__(self, timeout=0.003):
        super().__init__()
        self.timeout = timeout
        self.start = perf_counter()

    @property
    def cancelled(self):
        return super().cancelled or perf_counter() - self.start > self.timeout


def test_cancellable():
    was_called = False

    @cancellable
    def fn():
        nonlocal was_called
        was_called = True
        for _ in range(100):
            sleep(0.001)
        raise Exception("Should not be reached")

    token = TimeLimitCancellationToken()
    with pytest.raises(CancelledException):
        with token:
            fn()
    assert was_called, "Function was not called"


def test_cancellable_generator():
    was_called = False

    @cancellable
    def fn():
        nonlocal was_called
        was_called = True
        for i in range(100):
            sleep(0.001)
            yield i
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
    from nerfbaselines.utils import get_resources_utilization_info

    info = get_resources_utilization_info()
    assert isinstance(info, dict)


def test_cast_value():
    from nerfbaselines.utils import cast_value

    assert cast_value(int, 1) == 1
    assert cast_value(int, "1") == 1
    assert cast_value(Optional[int], 1) == 1
    assert cast_value(str, "1") == "1"

    assert cast_value(Optional[int], "none") is None
    assert cast_value(Literal[3, "ok"], "ok") == "ok"
    assert cast_value(Literal[3, "ok"], "3") == 3
    with pytest.raises(TypeError):
        assert cast_value(Literal[3, "ok"], "5")
