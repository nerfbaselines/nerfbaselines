from unittest import mock
import contextlib
import pytest
from typing import Iterable
import numpy as np
from time import sleep
from nerfbaselines import Method, MethodInfo, RenderOutput, ModelInfo
from nerfbaselines.utils import CancellationToken, CancelledException
from nerfbaselines.types import new_cameras


def test_render(use_remote_method):
    class TestMethodRenderCancellable(Method):
        def __init__(self):
            pass

        @classmethod
        def get_method_info(cls) -> MethodInfo:
            out: MethodInfo = {"name": "_test", "required_features": frozenset(), "supported_camera_models": frozenset()}
            return out

        def optimize_embeddings(self, *args, **kwargs):
            raise NotImplementedError()

        def get_info(self) -> ModelInfo:
            return {**self.get_method_info(), "num_iterations": 13}

        def render(self, cameras, embeddings=None) -> Iterable[RenderOutput]:
            yield {"color": np.full(tuple(), 23)}
            yield {"color": np.full(tuple(), 26)}

        def train_iteration(self, step: int):
            pass

        def save(self, path):
            pass

        def get_train_embedding(self, *args, **kwargs):
            raise NotImplementedError()

    with use_remote_method(TestMethodRenderCancellable) as remote_method_cls:
        remote_method = remote_method_cls()
        all_cameras = new_cameras(
            poses=np.eye(4, dtype=np.float32)[None, :3, :4],
            intrinsics=np.zeros((1, 4), dtype=np.float32),
            camera_types=np.zeros((1,), dtype=np.int32),
            distortion_parameters=np.zeros((1, 6), dtype=np.float32),
            image_sizes=np.array((64, 48), dtype=np.int32),
            nears_fars=None,
        )
        vals = [int(x["color"]) for x in remote_method.render(all_cameras)]
        assert vals == [23, 26]


@pytest.fixture
def use_remote_method():
    @contextlib.contextmanager
    def wrap(method=None, intercept=None):
        from nerfbaselines.backends import SimpleBackend

        if method is None:

            class TestMethod(Method):
                def __init__(self):
                    pass

                @classmethod
                def get_method_info(cls) -> MethodInfo:
                    out: MethodInfo = {"name": "_test"}
                    return out

                def get_info(self) -> ModelInfo:
                    return {**self.get_method_info(), "num_iterations": 13}

                def optimize_embeddings(self, *args, **kwargs):
                    raise NotImplementedError()

                def render(self, cameras, embeddings=None) -> Iterable[RenderOutput]:
                    for i in range(100):
                        sleep(0.001)
                        yield {"color": np.full(tuple(), i)}

                def setup_train(self, train_dataset, **kwargs):
                    pass

                def train_iteration(self, step: int):
                    pass

                def save(self, path):
                    pass

            method = TestMethod

        from importlib import import_module
        module = import_module(method.__module__)
        # Make default
        setattr(module, method.__name__, getattr(module, method.__name__, None))
        with SimpleBackend() as backend, \
                mock.patch.object(module, method.__name__, method):
            if intercept:
                oci = backend.instance_call
                backend.instance_call = lambda *args, **kwargs: intercept(oci, *args, **kwargs)
                ocs = backend.static_call
                backend.static_call = lambda *args, **kwargs: intercept(ocs, *args, **kwargs)
            remote_method = backend.wrap(method)
            yield remote_method

    return wrap


def test_get_resource_utilization(use_remote_method):
    from nerfbaselines.utils import get_resources_utilization_info

    called = False
    def intercept(callback, fn, *args, **kwargs):
        nonlocal called
        called = True
        assert fn == f'{get_resources_utilization_info.__module__}:{get_resources_utilization_info.__name__}'
        return callback(fn, *args, **kwargs)

    with use_remote_method(intercept=intercept) as _:
        info = get_resources_utilization_info()
        assert isinstance(info, dict)

    assert called, "intercept was not called"


def test_compute_metrics(use_remote_method):
    from nerfbaselines.evaluation import compute_metrics

    called = False
    def intercept(callback, function, *args, **kwargs):
        nonlocal called
        called = True
        assert function == f'{compute_metrics.__module__}:{compute_metrics.__name__}'
        return callback(function, *args, **kwargs)

    with use_remote_method(intercept=intercept) as _, mock.patch('nerfbaselines.metrics.lpips', lambda x, y: np.array([0.0, 0.0, 0.0])):
        im1 = np.zeros((21, 27, 3), dtype=np.float32)
        info = compute_metrics(im1, im1)
        assert isinstance(info, dict)

    assert called, "intercept was not called"


def test_render_cancellable(use_remote_method):
    from nerfbaselines.utils import cancellable

    class TestMethodRenderCancellable(Method):
        def __init__(self):
            pass

        def optimize_embeddings(self, *args, **kwargs):
            raise NotImplementedError()

        @classmethod
        def get_method_info(cls) -> MethodInfo:
            out: MethodInfo = {
                "name": "_test",
                "required_features": frozenset(("color",)),
                "supported_camera_models": frozenset(("opencv",))
            }
            return out

        def get_info(self) -> ModelInfo:
            return {**self.get_method_info(), "num_iterations": 13}

        @cancellable
        def render(self, cameras, embeddings=None) -> Iterable[RenderOutput]:
            for i in range(400):
                sleep(0.001)
                yield {"color": np.full(tuple(), i)}

        def train_iteration(self, step: int):
            pass

        def save(self, path):
            pass

        def get_train_embedding(self, *args, **kwargs):
            raise NotImplementedError()


    with use_remote_method(TestMethodRenderCancellable) as remote_method_cls:
        remote_method = remote_method_cls()
        all_cameras = new_cameras(
            poses=np.eye(4, dtype=np.float32)[None, :3, :4],
            intrinsics=np.zeros((1, 4), dtype=np.float32),
            camera_types=np.zeros((1,), dtype=np.int32),
            distortion_parameters=np.zeros((1, 6), dtype=np.float32),
            image_sizes=np.array((64, 48), dtype=np.int32),
            nears_fars=None,
        )
        cancelation_token = CancellationToken()
        vals = []
        with pytest.raises(CancelledException), cancelation_token:
            for vw in remote_method.render(all_cameras):
                v = int(vw["color"])
                vals.append(v)
                if v > 3:
                    cancelation_token.cancel()
        assert len(vals) < 100

        vals = []
        with pytest.raises(CancelledException), cancelation_token:
            for vw in cancellable(remote_method.render)(all_cameras):
                v = int(vw["color"])
                vals.append(v)
                if v > 3:
                    cancelation_token.cancel()
        assert len(vals) < 100
