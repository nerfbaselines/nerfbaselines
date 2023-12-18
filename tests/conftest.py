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


def run_test_train(tmp_path, dataset_path, method_name, backend="python"):
    from nerfbaselines.train import train_command
    from nerfbaselines.render import get_checkpoint_sha, render_command
    from nerfbaselines.utils import Indices, remap_error
    from nerfbaselines.utils import NoGPUError

    # train_command.callback(method, checkpoint, data, output, no_wandb, verbose, backend, eval_single_iters, eval_all_iters)
    (tmp_path / "output").mkdir()
    try:
        train_cmd = remap_error(train_command.callback)
        train_cmd(method_name, None, str(dataset_path), str(tmp_path / "output"), False, backend, Indices.every_iters(5), Indices([-1]), num_iterations=13, disable_extra_metrics=True, vis="none")
    except NoGPUError:
        pytest.skip("no GPU available")

    # Test if model was saved at the end
    assert (tmp_path / "output" / "checkpoint-13").exists()
    assert "nb-info.json" in os.listdir(tmp_path / "output" / "checkpoint-13")
    info = json.load((tmp_path / "output" / "checkpoint-13" / "nb-info.json").open("r"))
    assert "resources_utilization" in info

    # By default, the model should render all images at the end
    print(os.listdir(tmp_path / "output"))
    assert (tmp_path / "output" / "predictions-13.tar.gz").exists()
    with tarfile.open(tmp_path / "output" / "predictions-13.tar.gz", "r:gz") as tar:
        tar.extract(tar.getmember("info.json"), tmp_path / "tmpinfo")
        with open(tmp_path / "tmpinfo" / "info.json", "r") as f:
            info = json.load(f)
        (tmp_path / "tmp-renders").mkdir(parents=True)
        tar.extractall(tmp_path / "tmp-renders")
    assert info["checkpoint_sha256"] is not None, "checkpoint sha not saved in info.json"
    assert get_checkpoint_sha(tmp_path / "output" / "checkpoint-13") == info["checkpoint_sha256"], "checkpoint sha mismatch"

    # Test restore checkpoint and render
    render_cmd = remap_error(render_command.callback)
    # render_command(checkpoint, data, output, split, verbose, backend):
    render_cmd(tmp_path / "output" / "checkpoint-13", str(dataset_path), str(tmp_path / "output-render"), "test", verbose=False, backend=backend)

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
    train_cmd(
        method_name,
        tmp_path / "output" / "checkpoint-13",
        str(dataset_path),
        str(tmp_path / "output"),
        False,
        backend,
        Indices.every_iters(5),
        Indices([-1]),
        num_iterations=14,
        disable_extra_metrics=True,
        vis="none",
    )
    assert (tmp_path / "output" / "checkpoint-14").exists()
    assert (tmp_path / "output" / "predictions-14.tar.gz").exists()
    assert not (tmp_path / "output" / "predictions-13.tar.gz").exists()


@pytest.fixture(name="run_test_train", params=["blender", "colmap"])
def run_test_train_fixture(tmp_path_factory, request: pytest.FixtureRequest):
    dataset_type = request.param
    dataset_path = request.getfixturevalue(f"{dataset_type}_dataset_path")
    test_name = request.node.name
    assert test_name.startswith("test_")
    assert "_train" in test_name
    backend = "python"
    if request.node.get_closest_marker("apptainer") is not None:
        backend = "apptainer"
    elif request.node.get_closest_marker("docker") is not None:
        backend = "docker"

    default_method_name = None
    if request.node.get_closest_marker("method") is not None:
        default_method_name = request.node.get_closest_marker("method").args[0]

    def run(method_name=default_method_name, backend=backend):
        with tmp_path_factory.mktemp("output") as tmp_path:
            run_test_train(tmp_path, dataset_path, method_name, backend=backend)

    run.dataset_type = dataset_type
    return run


@pytest.fixture
def mock_torch():
    torch = mock.MagicMock()

    class Tensor(np.ndarray):
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

        def view(self, *shape):
            return self.reshape(shape)

        def mul_(self, value):
            self *= value
            return self

        def sub_(self, value):
            self -= value
            return self

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
    with mock.patch.dict(sys.modules, {"torch": torch}):
        yield torch


@pytest.fixture
def mock_extras(mock_torch):
    lpips = mock.MagicMock()

    with mock.patch.dict(sys.modules, {"lpips": lpips}):
        yield {
            "torch": mock_torch,
            "lpips": lpips,
        }


@pytest.fixture(autouse=True)
def no_extras(request):
    if request.node.get_closest_marker("extras") is not None:
        yield None
        return

    from nerfbaselines import evaluate

    def raise_import_error(*args, **kwargs):
        raise ImportError()

    with mock.patch.object(evaluate, "test_extra_metrics", raise_import_error):
        yield None
        return
