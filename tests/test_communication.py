import contextlib
import pytest
from typing import Iterable
import numpy as np
from time import sleep
import threading
from nerfbaselines import Method, MethodInfo, Cameras, RenderOutput
from nerfbaselines.utils import CancellationToken, CancelledException


def test_render(use_remote_method):
    class TestMethodRenderCancellable(Method):
        def get_info(self):
            return MethodInfo()

        def render(self, cameras, progress_callback=None) -> Iterable[RenderOutput]:
            yield {"color": np.full(tuple(), 23)}
            yield {"color": np.full(tuple(), 26)}

        def setup_train(self, train_dataset, **kwargs):
            pass

        def train_iteration(self, step: int):
            pass

        def save(self, path):
            pass

    with use_remote_method(TestMethodRenderCancellable()) as remote_method:
        cameras = Cameras(
            poses=np.eye(4, dtype=np.float32)[None, :3, :4],
            normalized_intrinsics=np.zeros((1, 4), dtype=np.float32),
            camera_types=np.zeros((1,), dtype=np.int32),
            distortion_parameters=np.zeros((1, 6), dtype=np.float32),
            image_sizes=np.array((64, 48), dtype=np.int32),
            nears_fars=None,
        )
        vals = [int(x["color"]) for x in remote_method.render(cameras)]
        assert vals == [23, 26]


@pytest.fixture
def use_remote_method():
    @contextlib.contextmanager
    def wrap(method=None):
        from nerfbaselines.communication import start_backend, RemoteMethod, ConnectionParams

        if method is None:

            class TestMethod(Method):
                def get_info(self):
                    return MethodInfo()

                def render(self, cameras, progress_callback=None) -> Iterable[RenderOutput]:
                    for i in range(100):
                        sleep(0.001)
                        yield {"color": np.full(tuple(), i)}

                def setup_train(self, train_dataset, **kwargs):
                    pass

                def train_iteration(self, step: int):
                    pass

                def save(self, path):
                    pass

            method = TestMethod()

        connection_params = ConnectionParams()
        thread = threading.Thread(target=start_backend, args=(method, connection_params), daemon=True)
        thread.start()
        sleep(0.02)
        try:
            remote_method = RemoteMethod(connection_params=connection_params)
            try:
                yield remote_method
            finally:
                remote_method.close()
        finally:
            thread.join(0.02)

    return wrap


def test_get_resource_utilization(use_remote_method):
    from nerfbaselines.train import method_get_resources_utilization_info

    with use_remote_method() as remote_method:
        info = method_get_resources_utilization_info(remote_method)
        assert isinstance(info, dict)


def test_render_cancellable(use_remote_method):
    from nerfbaselines.utils import cancellable

    class TestMethodRenderCancellable(Method):
        def get_info(self):
            return MethodInfo()

        def render(self, cameras, progress_callback=None) -> Iterable[RenderOutput]:
            for i in range(400):
                sleep(0.001)
                yield {"color": np.full(tuple(), i)}

        def setup_train(self, train_dataset, **kwargs):
            pass

        def train_iteration(self, step: int):
            pass

        def save(self, path):
            pass

    with use_remote_method(TestMethodRenderCancellable()) as remote_method:
        assert getattr(remote_method.render, "__cancellable__", False)
        cameras = Cameras(
            poses=np.eye(4, dtype=np.float32)[None, :3, :4],
            normalized_intrinsics=np.zeros((1, 4), dtype=np.float32),
            camera_types=np.zeros((1,), dtype=np.int32),
            distortion_parameters=np.zeros((1, 6), dtype=np.float32),
            image_sizes=np.array((64, 48), dtype=np.int32),
            nears_fars=None,
        )
        cancelation_token = CancellationToken()
        vals = []
        with pytest.raises(CancelledException):
            for vw in remote_method.render(cameras, cancellation_token=cancelation_token):
                v = int(vw["color"])
                vals.append(v)
                if v > 3:
                    cancelation_token.cancel()
        assert len(vals) < 100

        vals = []
        with pytest.raises(CancelledException):
            for vw in cancellable(remote_method.render)(cameras, cancellation_token=cancelation_token):
                v = int(vw["color"])
                vals.append(v)
                if v > 3:
                    cancelation_token.cancel()
        assert len(vals) < 100
