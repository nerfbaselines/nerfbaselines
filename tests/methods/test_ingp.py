import pytest
import contextlib
import numpy as np
import sys
from unittest import mock


METHOD_NAME = "instant-ngp"


@contextlib.contextmanager
def mock_instant_ngp():
    testbed = mock.MagicMock()
    pyngp = mock.Mock()
    pyngp.Testbed = lambda: testbed
    pyngp.__file__ = "pyngp/__init__.py"
    testbed.training_step = 0
    w, h = 200, 100

    def inc_training_step():
        testbed.training_step += 1
        return True

    testbed.nerf = mock.Mock()
    np.random.seed(42 + testbed.training_step + 11)
    testbed.render.return_value = np.random.rand(h, w, 4).astype(np.float32)
    testbed.nerf.training = mock.Mock()
    testbed.nerf.training.dataset = mock.MagicMock()
    testbed.nerf.training.dataset.n_images = 10
    testbed.nerf.training.dataset.metadata.__getitem__.return_value = metadata_mock = mock.Mock()
    metadata_mock.resolution = (w, h)
    image_sizes = None
    test_view = 0

    def _set_camera_to_training_view(x):
        nonlocal test_view
        test_view = x
        w, h = image_sizes[test_view]
        np.random.seed(42 + x + 13)
        testbed.render.return_value = np.random.rand(h, w, 4).astype(np.float32)
        metadata_mock.resolution = (w, h)

    testbed.set_camera_to_training_view = mock.Mock(side_effect=_set_camera_to_training_view)
    testbed.frame = mock.Mock(side_effect=inc_training_step)
    testbed.loss = 0.1
    with mock.patch.dict(sys.modules, {"pyngp": pyngp}):
        from nerfbaselines.methods._impl import instant_ngp

        old_setup_train = instant_ngp.InstantNGP.setup_train

        def new_setup_train(self, train_dataset, *args, **kwargs):
            nonlocal image_sizes
            image_sizes = train_dataset.cameras.image_sizes
            testbed.nerf.training.dataset.n_images = len(train_dataset.cameras)
            return old_setup_train(self, train_dataset, *args, **kwargs)

        old_render = instant_ngp.InstantNGP.render

        def new_render(self, cameras, *args, **kwargs):
            nonlocal image_sizes
            nonlocal test_view
            test_view = 0
            old_image_sizes = image_sizes
            old_n_images = testbed.nerf.training.dataset.n_images
            image_sizes = cameras.image_sizes
            testbed.nerf.training.dataset.n_images = len(cameras)
            try:
                out = old_render(self, cameras, *args, **kwargs)
            finally:
                image_sizes = old_image_sizes
                testbed.nerf.training.dataset.n_images = old_n_images
            return out

        with mock.patch.object(instant_ngp.InstantNGP, "render", new_render), mock.patch.object(instant_ngp.InstantNGP, "setup_train", new_setup_train):
            yield None


@pytest.mark.method(METHOD_NAME)
@mock_instant_ngp()
def test_train_instant_ngp_mocked(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.apptainer
def test_train_instant_ngp_apptainer(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.docker
def test_train_instant_ngp_docker(run_test_train):
    run_test_train()
