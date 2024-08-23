from unittest import mock
import pytest
import click.core
from time import sleep, perf_counter
from nerfbaselines.utils import Indices
from nerfbaselines.utils import CancellationToken, CancelledException

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
    def _cancelled(self):
        return perf_counter() - self.start > self.timeout

    @_cancelled.setter
    def _cancelled(self, value):
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
            CancellationToken.cancel_if_requested()
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


def test_tuple_click_type():
    import click
    from nerfbaselines.utils import TupleClickType

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
