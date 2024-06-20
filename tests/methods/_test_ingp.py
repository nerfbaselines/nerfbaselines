from enum import Enum
from pathlib import Path
import tempfile
import pytest
import contextlib
import numpy as np
import sys
from unittest import mock


METHOD_NAME = "instant-ngp"


class ColorSpace(Enum):
    SRGB = 0
    Linear = 1


@contextlib.contextmanager
def mock_instant_ngp():
    testbed = mock.MagicMock()
    pyngp = mock.Mock()
    pyngp.Testbed = lambda: testbed
    pyngp.__file__ = "pyngp/__init__.py"
    pyngp.ColorSpace = ColorSpace
    
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
    testbed.nerf.training.dataset["metadata"].__getitem__.return_value = metadata_mock = mock.Mock()
    metadata_mock.resolution = (w, h)
    image_sizes = None
    test_view = 0

    def _set_camera_to_training_view(x):
        nonlocal test_view
        assert image_sizes is not None
        test_view = x
        w, h = image_sizes[test_view]
        np.random.seed(42 + x + 13)
        testbed.render.return_value = np.random.rand(h, w, 4).astype(np.float32)
        metadata_mock.resolution = (w, h)

    testbed.set_camera_to_training_view = mock.Mock(side_effect=_set_camera_to_training_view)
    testbed.frame = mock.Mock(side_effect=inc_training_step)
    testbed.loss = 0.1
    gzip = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"pyngp": pyngp, "gzip": gzip, "msgpack": mock.MagicMock()}), \
            tempfile.TemporaryDirectory() as tempdir:
        import nerfbaselines.methods.instant_ngp as instant_ngp  # noqa
        
        pyngp.__file__ = str(Path(tempdir).joinpath("pyngp/__init__.py"))
        Path(tempdir).joinpath("configs", "nerf").mkdir(exist_ok=True, parents=True)
        Path(tempdir).joinpath("configs", "nerf", "base.json").write_text("{}")

        old_setup = instant_ngp.InstantNGP._setup

        def new_setup(self, train_dataset, *args, **kwargs):
            nonlocal image_sizes
            image_sizes = train_dataset["cameras"].image_sizes
            print(image_sizes)
            testbed.nerf.training.dataset.n_images = len(train_dataset["cameras"])
            return old_setup(self, train_dataset, *args, **kwargs)

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

        intt = instant_ngp.InstantNGP.__init__
        def init(self, *args, **kwargs):
            intt(self, *args, **kwargs)
            self.num_iterations = 13
        print(init)

        with mock.patch.object(instant_ngp.InstantNGP, "render", new_render), \
              mock.patch.object(instant_ngp.InstantNGP, "_setup", new_setup), \
              mock.patch.object(instant_ngp.InstantNGP, "__init__", init):
            yield None


@pytest.mark.method(METHOD_NAME)
@mock_instant_ngp()
def test_train_instant_ngp_mocked(run_test_train, mock_torch):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.apptainer
def test_train_instant_ngp_apptainer(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.docker
def test_train_instant_ngp_docker(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.conda
def test_train_instant_ngp_conda(run_test_train):
    run_test_train()
