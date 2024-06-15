import tarfile
import json
import sys
import pickle
from functools import partial
import inspect
from unittest import mock
import contextlib
import os
import pytest
from pathlib import Path
import numpy as np
from PIL import Image
# from nerfbaselines.utils import setup_logging
# 
# setup_logging(True)


class _nullcontext(contextlib.nullcontext):
    def __call__(self, fn):
        return fn


def make_dataset(path: Path, num_images=10):
    from nerfbaselines.datasets import _colmap_utils as colmap_utils

    (path / "images").mkdir(parents=True)
    (path / "sparse" / "0").mkdir(parents=True)
    cameras = {
        2: colmap_utils.Camera(2, "OPENCV", 180, 190, np.array([30.0, 24.0, 80.0, 70.0, 0.3, 0.4, 0.1, 0.4], dtype=np.float32)),
        3: colmap_utils.Camera(3, "OPENCV_FISHEYE", 180, 190, np.array([30, 24, 80, 70, 0.3, 0.4, 0.1, 0.4], dtype=np.float32)),
    }
    images = {
        i + 1: colmap_utils.Image(i + 1, np.random.randn(4), np.random.rand(3) * 4, list(cameras.keys())[i % len(cameras)], f"{i+1}.jpg", np.random.rand(7, 2), np.random.randint(0, 11, (7,)))
        for i in range(num_images)
    }
    colmap_utils.write_cameras_binary(cameras, str(path / "sparse" / "0" / "cameras.bin"))
    colmap_utils.write_points3D_binary(
        {i + 1: colmap_utils.Point3D(i + 1, np.random.rand(3), np.random.randint(0, 255, (3,)), 0.01, np.random.randint(0, num_images, (2,)), np.random.randint(0, 7, (2,))) for i in range(11)},
        str(path / "sparse" / "0" / "points3D.bin"),
    )
    colmap_utils.write_images_binary(images, str(path / "sparse" / "0" / "images.bin"))
    for i in range(num_images):
        camera = cameras[images[i + 1].camera_id]
        Image.fromarray((np.random.rand(1, 1, 3) * 255 + np.random.rand(camera.height, camera.width, 3) * 15).astype(np.uint8)).convert("RGB").save(path / "images" / f"{i+1}.jpg")


def make_blender_dataset(path: Path, num_images=10):
    path = Path(path) / "lego"
    path.mkdir(parents=True)
    w, h = 64, 64

    def create_split(split, num_images=3):
        (path / split).mkdir(parents=True)
        for i in range(num_images):
            Image.fromarray((np.random.rand(1, 1, 4) * 255 + np.random.rand(h, w, 4) * 15).astype(np.uint8)).convert("RGBA").save(path / split / f"{i}.png")
        meta = {
            "camera_angle_x": 0.5,
            "frames": [{"file_path": f"{split}/{i}", "transform_matrix": np.random.rand(4, 4).tolist()} for i in range(num_images)],
        }
        json.dump(meta, (path / f"transforms_{split}.json").open("w"))

    create_split("train")
    create_split("test")
    create_split("val")
    return path


@pytest.fixture
def colmap_dataset_path(tmp_path):
    make_dataset(tmp_path)
    yield tmp_path
    return tmp_path


@pytest.fixture
def blender_dataset_path(tmp_path):
    tmp_path = make_blender_dataset(tmp_path)
    yield tmp_path
    return tmp_path


@pytest.fixture(autouse=True)
def patch_prefix(tmp_path):
    _ns_prefix_backup = os.environ.get("NS_PREFIX", None)
    try:
        os.environ["NS_PREFIX"] = prefix = str(tmp_path)
        yield prefix
    finally:
        if _ns_prefix_backup is not None:
            os.environ["NS_PREFIX"] = _ns_prefix_backup
        else:
            os.environ.pop("NS_PREFIX", None)


def run_test_train(tmp_path, dataset_path, method_name, backend="python", config_overrides=None):
    from nerfbaselines.training import Trainer
    from nerfbaselines import metrics
    metrics._LPIPS_CACHE.clear()
    metrics._LPIPS_GPU_AVAILABLE = None
    sys.modules.pop("nerfbaselines._metrics_lpips", None)
    from nerfbaselines.training import train_command
    from nerfbaselines.io import get_checkpoint_sha
    from nerfbaselines.cli import render_command
    from nerfbaselines.utils import Indices, remap_error
    from nerfbaselines.utils import NoGPUError
    from nerfbaselines.io import deserialize_nb_info

    # train_command.callback(method, checkpoint, data, output, no_wandb, verbose, backend, eval_single_iters, eval_all_iters)
    (tmp_path / "output").mkdir()
    num_steps = [13]
    try:
        train_cmd = remap_error(train_command.callback)
        old_init = Trainer.__init__
        
        def __init__(self, *args, **kwargs):
            old_init(self, *args, **kwargs)
            self.num_iterations = num_steps[0]
            for v in vars(self).values():
                if isinstance(v, Indices):
                    v.total = self.num_iterations + 1
        with mock.patch.object(Trainer, '__init__', __init__):
            train_cmd(method_name, None, str(dataset_path), str(tmp_path / "output"), False, backend, Indices.every_iters(9), Indices.every_iters(5), Indices([-1]), logger="none", config_overrides=config_overrides)
    except NoGPUError:
        pytest.skip("no GPU available")

    # Test if model was saved at the end
    assert (tmp_path / "output" / "checkpoint-13").exists()
    assert "nb-info.json" in os.listdir(tmp_path / "output" / "checkpoint-13")
    info = deserialize_nb_info(json.load((tmp_path / "output" / "checkpoint-13" / "nb-info.json").open("r")))
    assert "resources_utilization" in info

    # By default, the model should render all images at the end
    print(os.listdir(tmp_path / "output"))
    assert (tmp_path / "output" / "predictions-13.tar.gz").exists()
    with tarfile.open(tmp_path / "output" / "predictions-13.tar.gz", "r:gz") as tar:
        tar.extract(tar.getmember("info.json"), tmp_path / "tmpinfo")
        with open(tmp_path / "tmpinfo" / "info.json", "r") as f:
            info = json.load(f)
        info = deserialize_nb_info(info)
        (tmp_path / "tmp-renders").mkdir(parents=True)
        tar.extractall(tmp_path / "tmp-renders")
    assert info["checkpoint_sha256"] is not None, "checkpoint sha not saved in info.json"
    assert get_checkpoint_sha(str(tmp_path / "output" / "checkpoint-13")) == info["checkpoint_sha256"], "checkpoint sha mismatch"

    # Test restore checkpoint and render
    render_cmd = remap_error(render_command.callback)
    # render_command(checkpoint, data, output, split, verbose, backend):
    render_cmd(str(tmp_path / "output" / "checkpoint-13"), str(dataset_path), str(tmp_path / "output-render"), "test", verbose=False, backend_name=backend)

    print(os.listdir(tmp_path / "output-render"))
    assert (tmp_path / "output-render").exists()
    # Verify the renders match the previous renders
    for fname in (tmp_path / "output-render" / "color").glob("**/*"):
        if not fname.is_file():
            continue
        fname = fname.relative_to(tmp_path / "output-render")
        render = np.array(Image.open(tmp_path / "output-render" / fname))
        render_old = np.array(Image.open(tmp_path / "tmp-renders" / fname))
        np.testing.assert_allclose(render, render_old, err_msg=f"render mismatch for {fname}")

    # Test can restore checkpoint and continue training
    (tmp_path / "output" / "predictions-13.tar.gz").unlink()
    num_steps[0] = 14
    with mock.patch.object(Trainer, '__init__', __init__):
        train_cmd(
            method_name,
            tmp_path / "output" / "checkpoint-13",
            str(dataset_path),
            str(tmp_path / "output"),
            False,
            backend,
            Indices.every_iters(9), 
            Indices.every_iters(5),
            Indices([-1]),
            logger="none",
        )
    assert (tmp_path / "output" / "checkpoint-14").exists()
    assert (tmp_path / "output" / "predictions-14.tar.gz").exists()
    assert not (tmp_path / "output" / "predictions-13.tar.gz").exists()


@pytest.fixture(name="run_test_train", params=["blender", "colmap"])
def run_test_train_fixture(tmp_path_factory, request: pytest.FixtureRequest):
    dataset_name = request.param
    dataset_path = request.getfixturevalue(f"{dataset_name}_dataset_path")
    test_name = request.node.name
    assert test_name.startswith("test_")
    assert "_train" in test_name
    backend = "python"
    if request.node.get_closest_marker("apptainer") is not None:
        backend = "apptainer"
    elif request.node.get_closest_marker("docker") is not None:
        backend = "docker"
    elif request.node.get_closest_marker("conda") is not None:
        backend = "conda"

    default_method_name = None
    method_marker = request.node.get_closest_marker("method")
    if method_marker is not None:
        default_method_name = method_marker.args[0]

    def run(method_name=default_method_name, backend=backend, **kwargs):
        with tmp_path_factory.mktemp("output") as tmp_path:
            run_test_train(tmp_path, dataset_path, method_name, backend=backend, **kwargs)

    run.dataset_name = dataset_name  # type: ignore
    return run


@pytest.fixture
def mock_torch():
    torch = mock.MagicMock()

    class Tensor(np.ndarray):
        def __new__(cls, value):
            if isinstance(value, np.ndarray):
                return value.view(Tensor)
            return np.array(value).view(Tensor)

        def cuda(self):
            return self

        def backward(self):
            pass

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            if isinstance(self, Tensor):
                return super().view(np.ndarray)
            return self

        def permute(self, *args):
            return np.transpose(self, args)

        def dim(self):
            return len(self.shape)

        def float(self):
            return self.astype(np.float32)

        def expand(self, args):
            return torch.broadcast_to(self, args)

        def contiguous(self):
            return self

        def long(self):
            return self.astype(np.int64)

        def to(self, *args, dtype=None, **kwargs):
            if dtype is not None:
                return self.astype(dtype)
            else:
                return self

        def view(self, *shape):  # type: ignore
            return self.reshape(shape)

        def mul_(self, value):
            self *= value
            return self

        def sub_(self, value):
            self -= value
            return self

        def unsqueeze(self, dim):
            return np.expand_dims(self, dim)

        def bmm(self, other):
            return np.matmul(self, other)

        def sum(self, dim=None, dtype=None, keepdim=False):  # type: ignore
            self = np.ndarray.view(self, np.ndarray)
            if isinstance(dim, list):
                dim = tuple(dim)
            return Tensor(np.sum(self, axis=dim, dtype=dtype, keepdims=keepdim))

        def mean(self, dim=None, dtype=None, keepdim=False):  # type: ignore
            self = np.ndarray.view(self, np.ndarray)
            if isinstance(dim, list):
                dim = tuple(dim)
            return Tensor(np.mean(self, axis=dim, dtype=dtype, keepdims=keepdim))
        
    def from_numpy(x):
        assert not isinstance(x, Tensor)
        return x.view(Tensor)

    for k, v in vars(np).items():
        if callable(v) and not inspect.isclass(v):

            def v2(ok, *args, device=None, **kwargs):
                out = getattr(np, ok)(*args, **kwargs)
                if isinstance(out, np.ndarray) and not isinstance(out, Tensor):
                    out = from_numpy(out)
                return out

            v = partial(v2, k)
        if not k.startswith("__"):
            setattr(torch, k, v)
    torch.no_grad = lambda: _nullcontext()
    torch.tensor = torch.array
    torch.clamp = torch.clip
    torch.from_numpy = from_numpy
    torch.bool = bool
    torch.long = np.int64
    torch.cat = torch.concatenate
    torch.sum = Tensor.sum
    torch.mean = Tensor.mean
    # torch.sum = lambda x, dim=None, dtype=None, keepdim=False: np.sum(x, axis=dim, dtype=dtype, keepdims=keepdim).view(Tensor)
    # torch.mean = lambda x, dim=None, dtype=None, keepdim=False: np.mean(x, axis=dim, dtype=dtype, keepdims=keepdim).view(Tensor)
    torch.Tensor = Tensor

    def save(value, file):
        if not hasattr(file, "write"):
            with open(file, "wb") as f:
                save(value, f)
        else:

            def to_numpy_rec(x):
                if isinstance(x, dict):
                    return {k: to_numpy_rec(v) for k, v in x.items()}
                if hasattr(x, "numpy"):
                    return x.numpy()
                return x

            pickle.dump(to_numpy_rec(value), file)

    def load(file, map_location=None):
        if not hasattr(file, "write"):
            with open(file, "rb") as f:
                return load(f)

        def from_numpy_rec(x):
            if isinstance(x, dict):
                return {k: from_numpy_rec(v) for k, v in x.items()}
            if isinstance(x, np.ndarray):
                return x.view(Tensor)
            return x

        return from_numpy_rec(pickle.load(file))

    torch.save = save
    torch.load = load
    torchvision = mock.MagicMock()
    alexnet = mock.MagicMock()
    alexnet.features = torch.zeros((512, 1, 1, 3))
    torchvision.models.alexnet = mock.MagicMock(return_value=alexnet)
    torch.nn = mock.MagicMock()

    class Module:
        def __init__(self, *args, **kwargs):
            self._modules = []

        @property
        def modules(self):
            return self._modules or [Module()]
        
        @modules.setter
        def modules(self, value):
            self._modules = value

        def forward(self, *args, **kwargs):
            return torch.Tensor(np.random.rand(1, 1, 3, 3))

        def register_buffer(self, name, value):
            setattr(self, name, value)

        def train(self, train=True):
            return self

        def eval(self):
            return self.train(False)

        def to(self, *args, **kwargs):
            return self

        def cpu(self):
            return self

        def cuda(self):
            return self
        
        def load_state_dict(self, state_dict, strict=True):
            pass
        
        def parameters(self):
            return []

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    torch.nn.Module = Module
    torch.nn.ModuleList = list
    torch.nn.Conv2d = Module

    class Sequential(Module):
        def __init__(self, *args):
            self.modules = list(args)

        def add_module(self, name, module):
            self.modules.append(module)

        def forward(self, x):  # type: ignore
            for m in self.modules:
                x = m(x)
            return x

    torch.nn.Sequential = Sequential
    
    with mock.patch.dict(sys.modules, {
        "torch": torch, 
        "torch.nn": torch.nn,
        "torchvision": torchvision}):
        yield torch


@pytest.fixture
def mock_extras(mock_torch):
    yield {
        "scipy": mock.MagicMock(),
        "torch": mock_torch,
    }

@pytest.fixture(autouse=True)
def no_extras(request):
    if request.node.get_closest_marker("extras") is not None:
        yield None
        return

    class FailedTorch:
        def __getattribute__(self, __name: str):
            raise ImportError("torch not available")

    with mock.patch.dict(sys.modules, {
        "torch": FailedTorch(),
    }):
        yield None
        return
