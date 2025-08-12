from contextlib import nullcontext
import shutil
import subprocess
import importlib
import gc
import copy
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


@pytest.fixture
def load_source_code():
    paths_to_pop = []
    def load(git_repo, commit_sha):
        reponame = git_repo.split("/")[-1].replace(".git", "")
        cached_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f".cache-{reponame}")
        if not os.path.exists(cached_path):
            if os.path.exists(cached_path+".tmp"):
                shutil.rmtree(cached_path+".tmp", ignore_errors=True)
            subprocess.run(["git", "clone", git_repo, cached_path+".tmp", "-q"], check=True)
            subprocess.run(["git", "checkout", commit_sha], cwd=cached_path+".tmp", check=True)
            os.rename(cached_path+".tmp", cached_path)
        if cached_path not in sys.path:
            sys.path.append(cached_path)
            paths_to_pop.append(cached_path)
    try:
        yield load
    finally:
        for path in paths_to_pop:
            sys.path.remove(path)


@pytest.fixture
def isolated_modules():
    mods = dict(sys.modules.items())
    try:
        yield None
    finally:
        for key in list(sys.modules.keys()):
            if key not in mods:
                del sys.modules[key]
        for key, value in mods.items():
            sys.modules[key] = value
        importlib.invalidate_caches()
        gc.collect()


@contextlib.contextmanager
def patch_modules(update):
    _empty = object()
    old_values = {k: sys.modules.get(k, _empty) for k in update}
    try:
        for k, v in update.items():
            sys.modules[k] = v  # type: ignore
        yield None
    finally:
        for k, v in old_values.items():
            if v is _empty:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v  # type: ignore

@pytest.fixture(name="patch_modules")
def patch_modules_fixture():
    return patch_modules


@pytest.fixture
def mock_module():
    old_values = dict()
    try:
        def mock_module(module):
            old_values[module] = sys.modules.get(module)
            sys.modules[module] = mock.MagicMock()
            return sys.modules[module]
        yield mock_module
    finally:
        for module, old in old_values.items():
            if old is None:
                del sys.modules[module]
            else:
                sys.modules[module] = old


def make_dataset(path: Path, num_images=10):
    from nerfbaselines.datasets import _colmap_utils as colmap_utils

    (path / "images").mkdir(parents=True)
    (path / "sparse" / "0").mkdir(parents=True)
    cameras = {
        2: colmap_utils.Camera(2, "OPENCV", 180, 190, np.array([30.0, 24.0, 80.0, 70.0, 0.3, 0.4, 0.1, 0.4], dtype=np.float32)),
        3: colmap_utils.Camera(3, "OPENCV_FISHEYE", 180, 190, np.array([30, 24, 80, 70, 0.3, 0.4, 0.1, 0.4], dtype=np.float32)),
    }
    images = {
        i + 1: colmap_utils.Image(i + 1, np.random.randn(4), np.random.rand(3) * 4, list(cameras.keys())[i % len(cameras)], str(i+1)+".jpg", np.random.rand(7, 2), np.random.randint(-1, 11, (7,)))
        for i in range(num_images)
    }
    colmap_utils.write_cameras_binary(cameras, str(path / "sparse" / "0" / "cameras.bin"))
    colmap_utils.write_points3D_binary(
        {i: colmap_utils.Point3D(i, np.random.rand(3), np.random.randint(0, 255, (3,)), 0.01, np.random.randint(1, num_images, (2,)), np.random.randint(0, 7, (2,))) for i in range(12)},
        str(path / "sparse" / "0" / "points3D.bin"),
    )
    colmap_utils.write_images_binary(images, str(path / "sparse" / "0" / "images.bin"))
    for i in range(num_images):
        camera = cameras[images[i + 1].camera_id]
        Image.fromarray((np.random.rand(1, 1, 3) * 255 + np.random.rand(camera.height, camera.width, 3) * 15).astype(np.uint8)).convert("RGB").save(path / "images" / (str(i+1)+".jpg"))


def make_blender_dataset(path: Path, num_images=10):
    del num_images
    path = Path(path) / "lego"
    path.mkdir(parents=True)
    w, h = 64, 64

    def create_split(split, num_images=3):
        (path / split).mkdir(parents=True)
        for i in range(num_images):
            Image.fromarray((np.random.rand(1, 1, 4) * 255 + np.random.rand(h, w, 4) * 15).astype(np.uint8)).convert("RGBA").save(path / split / (str(i)+".png"))
        meta = {
            "w": w,
            "h": h,
            "camera_angle_x": 0.5,
            "frames": [{"file_path": split+"/"+str(i), "transform_matrix": np.random.rand(4, 4).tolist()} for i in range(num_images)],
        }
        json.dump(meta, (path / ("transforms_"+split+".json")).open("w"))

    create_split("train")
    create_split("test")
    create_split("val")
    with path.joinpath("nb-info.json").open("w") as f:
        json.dump({
            "id": "blender",
            "scene": "lego",
            "type": "object-centric",
            "evaluation_protocol": "nerf",
            "loader": "nerfbaselines.datasets.blender:load_blender_dataset",
        }, f)
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


def is_gpu_error(e: Exception) -> bool:
    if isinstance(e, RuntimeError) and "Found no NVIDIA driver on your system." in str(e):
        return True
    if isinstance(e, EnvironmentError) and "unknown compute capability. ensure pytorch with cuda support is installed." in str(e).lower():
        return True
    if isinstance(e, ImportError) and "libcuda.so.1: cannot open shared object file" in str(e):
        return True
    if isinstance(e, RuntimeError) and "No suitable GPU found for rendering" in str(e):
        return True
    return False


@pytest.fixture(name="is_gpu_error")
def is_gpu_error_fixture():
    return is_gpu_error


def run_test_train(tmp_path, dataset_path, method_name, backend="python", config_overrides=None):
    from nerfbaselines.training import Trainer
    from nerfbaselines import metrics
    metrics._LPIPS_CACHE.clear()
    metrics._LPIPS_GPU_AVAILABLE = None
    sys.modules.pop("nerfbaselines._metrics_lpips", None)
    from nerfbaselines.cli._train import train_command
    from nerfbaselines.io import get_checkpoint_sha
    from nerfbaselines.cli._render import render_command
    from nerfbaselines.utils import Indices
    from nerfbaselines.io import deserialize_nb_info

    # train_command.callback(method, checkpoint, data, output, no_wandb, backend, eval_single_iters, eval_all_iters)
    (tmp_path / "output").mkdir()
    num_steps = [13]
    workdir = os.getcwd()
    try:
        train_cmd = train_command.callback
        assert train_cmd is not None
        old_init = Trainer.__init__
        
        def __init__(self, *args, **kwargs):
            old_init(self, *args, **kwargs)
            self.num_iterations = num_steps[0]
            for v in vars(self).values():
                if isinstance(v, Indices):
                    v.total = self.num_iterations + 1
        with mock.patch.object(Trainer, '__init__', __init__):
            train_cmd(method_name, None, str(dataset_path), str(tmp_path / "output"), backend, Indices.every_iters(9), Indices.every_iters(5), Indices([-1]), logger="none", config_overrides=config_overrides)
    except Exception as e:
        if is_gpu_error(e):
            pytest.skip("no GPU available")
        raise
    finally:
        # Restore working directory
        os.chdir(workdir)

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
    render_cmd = render_command.callback
    assert render_cmd is not None
    # render_command(checkpoint, data, output, split, verbose, backend):
    render_cmd(str(tmp_path / "output" / "checkpoint-13"), str(dataset_path), str(tmp_path / "output-render"), "test", backend_name=backend)

    print(os.listdir(tmp_path / "output-render"))
    assert (tmp_path / "output-render").exists()
    # Verify the renders match the previous renders
    for fname in (tmp_path / "output-render" / "color").glob("**/*"):
        if not fname.is_file():
            continue
        fname = fname.relative_to(tmp_path / "output-render")
        render = np.array(Image.open(tmp_path / "output-render" / fname))
        render_old = np.array(Image.open(tmp_path / "tmp-renders" / fname))
        np.testing.assert_allclose(render, render_old, err_msg="render mismatch for "+str(fname))

    # Test can restore checkpoint and continue training if supported
    from nerfbaselines import get_method_spec
    from nerfbaselines.results import get_method_info_from_spec
    method_info = get_method_info_from_spec(get_method_spec(method_name))
    if not method_info.get("can_resume_training", True):
        return
    (tmp_path / "output" / "predictions-13.tar.gz").unlink()
    num_steps[0] = 14
    cwd = os.getcwd()
    try:
        with mock.patch.object(Trainer, '__init__', __init__):
            train_cmd(
                method_name,
                tmp_path / "output" / "checkpoint-13",
                str(dataset_path),
                str(tmp_path / "output"),
                backend,
                Indices.every_iters(9), 
                Indices.every_iters(5),
                Indices([-1]),
                logger="none",
            )
    finally:
        os.chdir(cwd)
    assert (tmp_path / "output" / "checkpoint-14").exists()
    assert (tmp_path / "output" / "predictions-14.tar.gz").exists()
    assert not (tmp_path / "output" / "predictions-13.tar.gz").exists()


@pytest.fixture(name="run_test_train", params=["blender", "colmap"])
def run_test_train_fixture(tmp_path_factory, request: pytest.FixtureRequest):
    dataset_name = request.param
    dataset_path = request.getfixturevalue(dataset_name+"_dataset_path")
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
        tmp_path = tmp_path_factory.mktemp("output")
        run_test_train(tmp_path, dataset_path, method_name, backend=backend, **kwargs)

    run.dataset_name = dataset_name  # type: ignore
    return run


class Tensor(np.ndarray):
    def __new__(cls, value):
        if isinstance(value, np.ndarray):
            return np.ndarray.view(value, Tensor)
        return np.array(value).view(Tensor)

    def min(self, other=None, dim=None, axis=None):
        self = np.ndarray.view(self, np.ndarray)
        if isinstance(other, int):
            dim = other
            other = None
        if other is not None:
            other = np.ndarray.view(other, np.ndarray)
            return Tensor(np.minimum(self, other))
        if dim is not None or axis is not None:
            ind = np.argmin(self, axis=dim or axis).view(Tensor)
            values = np.min(self, axis=dim or axis).view(Tensor)
            return values, ind
        return Tensor(np.min(self))

    def max(self, other=None, dim=None, axis=None):
        self = np.ndarray.view(self, np.ndarray)
        if isinstance(other, int):
            dim = other
            other = None
        if other is not None:
            other = np.ndarray.view(other, np.ndarray)
            return Tensor(np.maximum(self, other))
        if dim is not None or axis is not None:
            maxv = Tensor(np.max(self, axis=dim or axis))
            maxind = Tensor(np.argmax(self, axis=dim or axis))
            return maxv, maxind
        return Tensor(np.max(self))


    def size(self, dim=None):
        if dim is None:
            return self.shape
        return self.shape[dim]

    def mm(self, other):
        return np.matmul(self, other)

    def t(self):
        return np.transpose(self)

    @property
    def grad(self):
        return self*0

    @property
    def device(self):
        return "cpu"

    def clone(self):
        return self.copy()

    def __eq__(self, other):
        if other is None:
            return False
        if isinstance(self, Tensor):
            self = np.ndarray.view(self, np.ndarray)
        if isinstance(other, Tensor):
            other = np.ndarray.view(other, np.ndarray)
        return Tensor(self == other)

    def __ne__(self, other):
        if other is None:
            return False
        if isinstance(self, Tensor):
            self = np.ndarray.view(self, np.ndarray)
        if isinstance(other, Tensor):
            other = np.ndarray.view(other, np.ndarray)
        return Tensor(self != other)

    def is_floating_point(self):
        return self.dtype in [np.float32, np.float64, np.float16]

    def float(self):
        return self.astype(np.float32)

    def half(self):
        return self.astype(np.float16)

    def int(self):
        return self.astype(np.int32)

    def cuda(self, *args):
        del args
        return self

    def copy_(self, other):
        self[:] = other
        return self

    def type_as(self, other):
        return self.astype(other.dtype)

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

    def is_cuda(self):
        return False

    def get_device(self):
        return self.device

    def expand(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        self = np.ndarray.view(self, np.ndarray)
        return Tensor(np.broadcast_to(self, args))

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
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return self.reshape(shape)

    def add_(self, value):
        self += value
        return self

    def mul_(self, value):
        self *= value
        return self

    def sub_(self, value):
        self -= value
        return self

    def abs(self):
        return np.abs(self)

    def squeeze(self, dim=None):
        return np.squeeze(np.ndarray.view(self, np.ndarray), dim).view(Tensor)

    def unsqueeze(self, dim):
        return np.expand_dims(self, dim)

    def bmm(self, other):
        return np.matmul(self, other)

    def sum(self, dim=None, dtype=None, keepdim=False):  # type: ignore
        self = np.ndarray.view(self, np.ndarray)
        if isinstance(dim, list):
            dim = tuple(dim)
        return Tensor(np.sum(self, axis=dim, dtype=dtype, keepdims=keepdim))

    def pow(self, value):
        return np.power(self, value)

    def prod(self, dim=None, dtype=None, keepdim=False):  # type: ignore
        self = np.ndarray.view(self, np.ndarray)
        if isinstance(dim, list):
            dim = tuple(dim)
        return Tensor(np.prod(self, axis=dim, dtype=dtype, keepdims=keepdim))

    def mean(self, dim=None, dtype=None, keepdim=False):  # type: ignore
        self = np.ndarray.view(self, np.ndarray)
        if isinstance(dim, list):
            dim = tuple(dim)
        return Tensor(np.mean(self, axis=dim, dtype=dtype, keepdims=keepdim))

    def clamp(self, min=None, max=None):
        return np.clip(self, min, max)

    def clamp_min(self, min=None):
        return np.clip(self, min, None)

    def flatten(self, start_dim=0, end_dim=-1):
        self = np.ndarray.view(self, np.ndarray)
        shape = self.shape
        if end_dim < 0:
            end_dim = len(shape) + end_dim
        shape = shape[:start_dim] + (-1,) + shape[end_dim + 1:]
        return Tensor(self.reshape(shape))

    def clone(self):
        return self.copy()

    def inverse(self):
        return np.linalg.inv(self)

    def norm(self, dim=None, keepdim=False):
        return np.linalg.norm(self, axis=dim, keepdims=keepdim).view(Tensor)

    def transpose(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        self = np.ndarray.view(self, np.ndarray)
        self = np.swapaxes(self, *args)
        return Tensor(self)

    def repeat(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        self = np.ndarray.view(self, np.ndarray)
        return Tensor(np.tile(self, args))

    def requires_grad_(self, *args, **kwargs):
        del args, kwargs
        return self

    def nonzero(self):
        self = np.ndarray.view(self, np.ndarray)
        return Tensor(np.nonzero(self)[0])

    def split(self, split_size, dim=0):
        self = np.ndarray.view(self, np.ndarray)
        if isinstance(split_size, list):
            split_size = tuple(split_size)
        if isinstance(dim, list):
            dim = tuple(dim)
        splits = np.cumsum(split_size)[:-1]
        return [x.view(Tensor) for x in np.split(self, splits, axis=dim)]

    def __repr__(self):
        self = np.ndarray.view(self, np.ndarray)
        return f"Tensor({self.__repr__()})"

    def __str__(self):
        self = np.ndarray.view(self, np.ndarray)
        return f"Tensor({self.__str__()})"

    def __getitem__(self, i):
        out = np.ndarray.view(self, np.ndarray).__getitem__(i).view(Tensor)
        if not isinstance(out, Tensor):
            out = Tensor(out)
        return out

    def expand(self, *shape):
        shape = [s if s >= 0 else s2 for s, s2 in zip(shape, self.shape)]
        return np.broadcast_to(self, shape).view(Tensor)

    def gather(self, dim, index):
        return np.ndarray.view(np.take_along_axis(self, index, axis=dim), Tensor)

Tensor.__module__ = "torch"


class Optimizer:
    def __init__(self, params, lr=None, eps=None, betas=None):
        self.param_groups = [
            p if isinstance(p, dict) else {} for p in params
        ]
        self.lr = lr
        self.eps = eps

    def zero_grad(self, set_to_none=False): pass
    def step(self, closure=None): pass
    def state_dict(self):
        return vars(self)
    def load_state_dict(self, state_dict):
        vars(self).update(state_dict)



@pytest.fixture
def mock_torch(patch_modules):
    torch = mock.MagicMock()
        
    def from_numpy(x):
        assert not isinstance(x, Tensor)
        return x.view(Tensor)

    for k, v in vars(np).items():
        if callable(v) and not inspect.isclass(v):

            def v2(ok, *args, device=None, requires_grad=None, **kwargs):
                out = getattr(np, ok)(*args, **kwargs)
                if (isinstance(out, np.float32) or isinstance(out, np.float64)):
                    out = np.array(out)
                if isinstance(out, np.ndarray) and not isinstance(out, Tensor):
                    out = from_numpy(out)
                return out

            v = partial(v2, k)
        if not k.startswith("__"):
            setattr(torch, k, v)
    def stack(x, dim=0):
        return np.stack(x, axis=dim).view(Tensor)

    def sort(x, dim=0):
        indices = np.ndarray.view(np.argsort(x, axis=dim), Tensor)
        return x[indices], indices

    torch.sort = sort
    torch.no_grad = lambda: _nullcontext()
    torch.tensor = torch.array
    torch.stack = stack
    torch.clamp = Tensor.clamp
    torch.from_numpy = from_numpy
    torch.bool = bool
    torch.long = np.int64
    torch.int = np.int32
    def concatenate(tensors, dim):
        return np.concatenate(tensors, axis=dim).view(Tensor)
    torch.bmm = Tensor.bmm
    torch.cat = concatenate
    torch.concat = concatenate
    torch.sum = Tensor.sum
    torch.mean = Tensor.mean
    torch.max = Tensor.max
    torch.min = Tensor.min
    def sigmoid(x):
        x = np.ndarray.view(x, np.ndarray)
        return Tensor(1 / (1 + np.exp(-x)))
    torch.sigmoid = sigmoid
    torch.norm = Tensor.norm
    torch.clamp_min = Tensor.clamp_min
    # torch.sum = lambda x, dim=None, dtype=None, keepdim=False: np.sum(x, axis=dim, dtype=dtype, keepdims=keepdim).view(Tensor)
    # torch.mean = lambda x, dim=None, dtype=None, keepdim=False: np.mean(x, axis=dim, dtype=dtype, keepdims=keepdim).view(Tensor)
    def zeros(*shape, dtype=None, device=None):
        del device
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        print(shape, dtype)
        return np.zeros(shape, dtype=dtype).view(Tensor)
    def ones(*shape, dtype=None, device=None):
        del device
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return np.ones(shape, dtype=dtype).view(Tensor)
    def randn_like(x):
        return np.random.randn(*x.shape).view(Tensor)
    def inverse(x):
        return np.ndarray.view(np.linalg.inv(x), Tensor)
    def cross(x, y, dim=-1):
        return np.cross(x, y, axis=dim).view(Tensor)
    torch.cross = cross
    torch.inverse = inverse
    torch.zeros = zeros
    torch.ones = ones
    torch.Tensor = Tensor
    torch.randn_like = randn_like

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

    def load(file, map_location=None, weights_only=None):
        del map_location, weights_only
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

    def unique(x, dim=None, return_inverse=False):
        axis = None
        if dim is not None:
            axis = [x for x in range(len(x.shape)) if x != dim]
        if return_inverse:
            _, indices = np.unique(x, axis=axis, return_index=True)
            return x[indices], indices
        else:
            return np.unique(x, axis=axis)

    def empty(*shape, dtype=None, device=None):
        del device
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        return np.empty(shape, dtype=dtype).view(Tensor)

    torch.save = save
    torch.load = load
    torch.unique = unique
    torch.empty = empty
    torch.float = 'float32'
    torchvision = mock.MagicMock()
    alexnet = mock.MagicMock()
    alexnet.features = torch.zeros((512, 1, 1, 3))
    torchvision.models.alexnet = mock.MagicMock(return_value=alexnet)
    torch.optim.Adam = Optimizer

    class Module:
        def __init__(self, *args, **kwargs):
            self._modules = []
            self.training = True

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
            self.training = train
            return self

        def eval(self):
            return self.train(False)

        def to(self, *args, **kwargs):
            return self

        def cpu(self):
            return self

        def cuda(self, device=None):
            del device
            return self

        def half(self):
            return self

        def state_dict(self):
            return {}
        
        def load_state_dict(self, state_dict, strict=True):
            pass
        
        def parameters(self):
            return []

        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    def Parameter(value):
        return value

    torch.nn.Module = Module
    torch.nn.Parameter = Parameter
    torch.nn.ModuleList = list
    torch.nn.Conv2d = Module

    class Sequential(Module):
        def __init__(self, *args):
            super().__init__()
            self.modules = list(args)

        def add_module(self, name, module):
            del name
            self.modules.append(module)

        def forward(self, x):  # type: ignore
            for m in self.modules:
                x = m(x)
            return x

    torch.nn.Sequential = Sequential
    def to_tensor(x):
        if isinstance(x, Image.Image):
            mode = x.mode
            x = np.array(x)
            if mode == "L": return Tensor(x.astype(np.float32))
            return Tensor(x.astype(np.float32) / 255.0)
        raise NotImplementedError(f"to_tensor not implemented for {x}")
    def to_pil_image(x):
        if len(x.shape) == 3:
            x = x.permute(1, 2, 0)
            return Image.fromarray((x * 255).astype(np.uint8))
        else:
            return Image.fromarray(x)
    torchvision.transforms.ToTensor = lambda: to_tensor
    torchvision.transforms.ToPILImage = lambda: to_pil_image
    def Compose(transforms):
        def compose2(x):
            for t in transforms:
                x = t(x)
            return x
        return compose2
    torchvision.transforms.Compose = Compose

    def conv2d(x, weight, bias=None, groups=None, stride=1, padding=0):
        del bias, groups
        assert x.shape[-3] == weight.shape[0]
        outdims = weight.shape[1]
        outshape = list(x.shape)
        outshape[-3] = outdims
        h, w = x.shape[-2:]
        h = h - weight.shape[-2] + 1 + 2 * padding
        w = w - weight.shape[-1] + 1 + 2 * padding
        h //= stride
        w //= stride
        outshape[-2:] = h, w
        return Tensor(np.full(outshape, float(x.mean()), dtype=x.dtype))

    def interpolate(x, size, mode="bilinear", align_corners=False):
        del mode, align_corners
        return x[:size[0], :size[1]]

    class Linear(Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features

        def forward(self, x):
            return torch.zeros((x.shape[0], self.out_features), dtype=x.dtype)

    torch.nn.functional.normalize = lambda x, dim=-1, p=2: x / x.norm(dim=dim, keepdim=True)
    torch.nn.functional.interpolate = interpolate
    torch.nn.functional.conv2d = conv2d
    torch.nn.functional.relu = lambda x: np.maximum(x, 0)
    torch.nn.Tanh = lambda: lambda x: torch.tanh(x)
    torch.nn.Sigmoid = lambda: lambda x: torch.sigmoid(x)
    torch.nn.Linear = Linear
    torch.nn.ReLU = lambda inplace=False: lambda x: torch.nn.functional.relu(x)


    torch.autograd.Variable = Tensor

    class DataLoader:
        def __init__(self, dataset, collate_fn, **kwargs):
            del kwargs
            self.dataset = dataset
            self.collate_fn = collate_fn
            self._index = None

        def __iter__(self):
            self._index = 0
            return self

        def __next__(self):
            if self._index >= len(self.dataset):
                raise StopIteration
            self._index += 1
            return self.collate_fn([self.dataset[self._index - 1]])

    class Dataset:
        def __init__(self, *args, **kwargs):
            del args, kwargs

    torch.utils.data.Dataset = Dataset
    torch.utils.data.DataLoader = DataLoader

    cuda = torch.cuda
    cuda.is_current_stream_capturing = lambda: False
    class nvtx_range(nullcontext):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def __call__(self, x): return x
    cuda.nvtx.range = nvtx_range

    with patch_modules({
        "torch": torch, 
        "torch._C": torch._C,
        "torch.autograd": torch.autograd,
        "torch.utils": torch.utils,
        "torch.utils.data": torch.utils.data,
        "torch.utils.tensorboard": torch.utils.tensorboard,
        "torch.utils.tensorboard.writer": torch.utils.tensorboard.writer,
        "torch.utils.cpp_extension": torch.utils.cpp_extension,
        "torch.cuda": torch.cuda,
        "torch.nn": torch.nn,
        "torch.optim": torch.optim,
        "torch.optim.optimizer": torch.optim.optimizer,
        "torch.mps": torch.mps,
        "torch.autograd": torch.autograd,
        "torch.nn.functional": torch.nn.functional,
        "torchvision": torchvision,
        "torchvision.transforms": torchvision.transforms,
        "torchvision.utils": torchvision.utils,
    }):
        from nerfbaselines.metrics import clear_cache as metrics_clear_cache
        metrics_clear_cache()
        yield torch
        metrics_clear_cache()


@pytest.fixture
def torch_cpu():
    import torch.utils
    import torchvision
    del torchvision
    backup = {}
    tensor_backup = {}
    def patch(name, value):
        if name not in backup:
            backup[name] = getattr(torch, name)
        setattr(torch, name, value)
    def patchtensor(name, value):
        if name not in backup:
            tensor_backup[name] = getattr(torch.Tensor, name)
        setattr(torch.Tensor, name, value)
    def patchdevice(call):
        def call2(*args, **kwargs):
            if "device" in kwargs:
                del kwargs["device"]
            return call(*args, **kwargs)
        return call2
    try:
        patch("Tensor", copy.copy(torch.Tensor))
        cuda = mock.MagicMock()
        cuda.is_current_stream_capturing = lambda: False
        class nvtx_range(nullcontext):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def __call__(self, x): return x
        cuda.nvtx.range = nvtx_range
        patch("cuda", cuda)
        def _cuda(self, device=None): return self
        patchtensor("cuda", _cuda)
        oldto = torch.Tensor.to
        def to(self, *args, **kwargs):
            if args and (args[0] == "cuda" or isinstance(args[0], torch.device)):
                args = ("cpu",) + args[1:]
            if kwargs.get("device", None) in ["cuda", "mps"]:
                kwargs["device"] = "cpu"
            return oldto(self, *args, **kwargs)  # type: ignore
        patchtensor("to", to)
        patchtensor("pin_memory", lambda self, *args, **kwargs: self)
        for name in ['zeros', 'ones', 'full', 'rand', 'tensor', 'zeros_like', 'ones_like', 'rand_like', 'eye', 'randint']:
            patch(name, patchdevice(getattr(torch, name)))
        yield None
    finally:
        for name, value in backup.items():
            setattr(torch, name, value)
        for name, value in tensor_backup.items():
            setattr(torch.Tensor, name, value)


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
            raise ImportError("torch not available", name="torch." + __name)

    old_torch = sys.modules.get("torch", None)
    try:
        sys.modules["torch"] = FailedTorch()  # type: ignore
        yield None
    finally:
        if old_torch is not None:
            sys.modules["torch"] = old_torch
        else:
            sys.modules.pop("torch", None)
