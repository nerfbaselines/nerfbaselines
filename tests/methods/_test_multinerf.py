from pathlib import Path
import functools
from typing import cast
import os
import gc
import contextlib
import pytest
import numpy as np
import sys
from unittest import mock
import nerfbaselines.registry


def _enable_gc(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        gc.enable()
        return out

    return wrapper


class _MNDataset:
    def __init__(self, split, none, config):
        self.size: int = len(self.dataset)  # type: ignore
        self._render_spherical = False
        self._n_examples = self.size
        self.split = split
        self._load_renderings(config)  # type: ignore
        self.cameras = np.random.rand(self.size, 3, 4).astype(np.float32)  # type: ignore

    def generate_ray_batch(self, i):
        mm = mock.MagicMock()
        mm.rays = mock.MagicMock()
        mm.rays._i = i
        return mm

    def __iter__(self):
        i = 0
        while True:
            mm = mock.MagicMock()
            mm.rays = mock.MagicMock()
            mm.rays._i = i
            yield mm
            i += 1


@contextlib.contextmanager
def mock_multinerf():
    with mock.patch.dict(
        sys.modules,
        {
            "gin": mock.MagicMock(),
            "jax": mock.MagicMock(),
            "jax.numpy": np,
            "flax": mock.MagicMock(),
            "flax.training": mock.MagicMock(),
            "internal": mock.Mock(),
            "internal.datasets": mock.Mock(),
            "train": mock.MagicMock(),
        },
    ):
        image_sizes = None
        cast(mock.MagicMock, sys.modules["gin"]).operative_config_str.return_value = ""
        cast(mock.MagicMock, sys.modules["train"]).__file__ = "train.py"
        cast(mock.MagicMock, sys.modules["jax"]).host_id.return_value = 0
        cast(mock.MagicMock, sys.modules["jax"]).process_index.return_value = 0
        cast(mock.MagicMock, sys.modules["jax"]).device_count.return_value = 4
        cast(mock.MagicMock, sys.modules["jax"]).numpy = np
        cast(mock.MagicMock, sys.modules["jax"]).random = random = mock.MagicMock()
        cast(mock.MagicMock, sys.modules["flax"]).jax_utils = flax_jax_utils = mock.MagicMock()
        flax_jax_utils.prefetch_to_device = lambda x, y: x
        random.split.return_value = 0, 0
        internal_datasets = cast(mock.MagicMock, sys.modules["internal.datasets"])
        internal_datasets.Dataset = _MNDataset
        internal = cast(mock.MagicMock, sys.modules["internal"])
        internal.configs = mock.MagicMock()
        internal.configs.Config.return_value = config = mock.MagicMock()
        internal.camera_utils = camera_utils = mock.MagicMock()
        internal.models = models = mock.MagicMock()
        config.lr_init = 0.01
        config.batch_size = 16384
        config.near = 2.0
        config.far = 6.0
        config.gc_every = 2
        config.enable_robustnerf_loss = False
        config.max_steps = 30
        camera_utils.unpad_poses = lambda x: x[..., :3, :4]
        camera_utils.pad_poses = lambda x: np.concatenate([x, np.tile(np.array([0, 0, 0, 1], dtype=np.float32), (x.shape[0], 1, 1))], -2)
        camera_utils.transform_poses_pca = lambda x: (x, np.eye(4, dtype=np.float32))
        camera_utils.intrinsic_matrix = lambda *args: np.eye(3, dtype=np.float32)
        internal.train_utils = train_utils = mock.MagicMock()
        model = mock.MagicMock()
        model.num_glo_embeddings = 10000
        model.num_glo_feature = 0
        stats = {}
        train_pstep = mock.Mock(return_value=(mock.MagicMock(), stats, mock.MagicMock()))
        train_utils.setup_model.return_value = (
            model,
            mock.MagicMock(),
            mock.MagicMock(),
            train_pstep,
            mock.MagicMock(),
        )
        models.construct_model.return_value = (model, mock.MagicMock())
        train_utils.create_optimizer.return_value = (mock.MagicMock(), mock.MagicMock())

        def render_image(_, val, *args, **kwargs):
            assert image_sizes is not None
            np.random.seed(42 + val._i)
            w, h = image_sizes[val._i]
            return {
                "acc": np.random.rand(h, w).astype(np.float32),
                "rgb": np.random.rand(h, w, 3).astype(np.float32),
                "distance_mean": np.random.rand(h, w).astype(np.float32),
            }

        models.render_image = render_image

        def fix_spec(v):
            if v["method"].split(":")[-1].lower() == "multinerf":
                v = v.copy()
                v["kwargs"] = (v.get("kwargs") or {}).copy()
                v["kwargs"]["config_overrides"] = {
                    **(v["kwargs"].get("config_overrides") or {}),
                    "Config.batch_size": 128,
                }
            return v

        new_registry = {k: fix_spec(v) for k, v in nerfbaselines.registry.methods_registry.items()}

        with mock.patch.object(nerfbaselines.registry, "methods_registry", new_registry):
            from nerfbaselines.methods.multinerf import MultiNeRF

            old_setup_train = MultiNeRF._setup_train
            old_save = MultiNeRF.save

            def new_setup_train(self, train_dataset, *args, **kwargs):
                nonlocal image_sizes
                image_sizes = train_dataset["cameras"].image_sizes
                return old_setup_train(self, train_dataset, *args, **kwargs)

            def new_save(self, path):
                old_save(self, path)
                os.makedirs(Path(path) / f"checkpoint_{self.step}")

            with mock.patch.object(MultiNeRF, "_setup_train", new_setup_train), mock.patch.object(MultiNeRF, "save", new_save):
                yield None


@pytest.mark.parametrize(
    "method_name",
    [pytest.param(k, marks=[pytest.mark.method(k)]) for k in ["mipnerf360", "mipnerf360:single-gpu"]],
)
@mock_multinerf()
@_enable_gc
def test_train_multinerf_mocked(run_test_train, method_name, mock_torch):
    run_test_train()


@pytest.mark.apptainer
@pytest.mark.method("mipnerf360")
@_enable_gc
def test_train_multinerf_apptainer(run_test_train):
    run_test_train(config_overrides=[
        ("Config.batch_size", "128")
    ])


@pytest.mark.docker
@pytest.mark.method("mipnerf360")
@_enable_gc
def test_train_multinerf_docker(run_test_train):
    run_test_train(config_overrides=[
        ("Config.batch_size", "128")
    ])


@pytest.mark.conda
@pytest.mark.method("mipnerf360")
@_enable_gc
def test_train_multinerf_conda(run_test_train):
    run_test_train(config_overrides=[
        ("Config.batch_size", "128")
    ])
