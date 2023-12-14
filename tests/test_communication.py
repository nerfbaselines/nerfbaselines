import pytest
import numpy as np
from time import sleep
import threading
from nerfbaselines import Method, MethodInfo, Cameras
from nerfbaselines.utils import CancellationToken, CancelledException


def test_render():
    from nerfbaselines.communication import start_backend, RemoteMethod, ConnectionParams

    class TestMethodRenderCancellable(Method):
        def get_info():
            return MethodInfo()

        def render(self, cameras, progress_callback=None):
            yield "test"
            yield "test2"

        def setup_train(self, train_dataset, *, num_iterations: int):
            pass

        def train_iteration(self, step: int):
            pass

        def save(self, path):
            pass

    connection_params = ConnectionParams()
    method = TestMethodRenderCancellable()
    thread = threading.Thread(target=start_backend, args=(method, connection_params), daemon=True)
    thread.start()
    sleep(0.02)
    try:
        remote_method = RemoteMethod(connection_params=connection_params)
        cameras = Cameras(
            poses=np.eye(4, dtype=np.float32)[None, :3, :4],
            normalized_intrinsics=np.zeros((1, 4), dtype=np.float32),
            camera_types=np.zeros((1,), dtype=np.int32),
            distortion_parameters=np.zeros((1, 6), dtype=np.float32),
            image_sizes=np.array((64, 48), dtype=np.int32),
            nears_fars=None,
        )
        vals = list(remote_method.render(cameras))
        remote_method.close()
        assert vals == ["test", "test2"]
    finally:
        thread.join(0.02)


def test_render_cancellable():
    from nerfbaselines.communication import start_backend, RemoteMethod, ConnectionParams, cancellable

    class TestMethodRenderCancellable(Method):
        def get_info():
            return MethodInfo()

        def render(self, cameras, progress_callback=None):
            for i in range(100):
                sleep(0.001)
                yield i

        def setup_train(self, train_dataset, *, num_iterations: int):
            pass

        def train_iteration(self, step: int):
            pass

        def save(self, path):
            pass

    connection_params = ConnectionParams()
    method = TestMethodRenderCancellable()
    thread = threading.Thread(target=start_backend, args=(method, connection_params), daemon=True)
    thread.start()
    sleep(0.02)
    try:
        remote_method = RemoteMethod(connection_params=connection_params)
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
            for v in remote_method.render(cameras, cancellation_token=cancelation_token):
                vals.append(v)
                if v > 3:
                    cancelation_token.cancel()
        assert len(vals) < 60

        vals = []
        with pytest.raises(CancelledException):
            for v in cancellable(remote_method.render)(cameras, cancellation_token=cancelation_token):
                vals.append(v)
                if v > 3:
                    cancelation_token.cancel()
        assert len(vals) < 60
        remote_method.close()
    finally:
        thread.join(0.02)
