import contextlib
import sys
import shutil
import json
from typing import cast
from pathlib import Path
import os
import numpy as np
from nerfbaselines import Method, MethodInfo, Cameras, RenderOutput, ModelInfo
from nerfbaselines.utils import Indices
from nerfbaselines.datasets import _colmap_utils as colmap_utils
from nerfbaselines import metrics
from unittest import mock
import tarfile
import pytest
from PIL import Image


def make_dataset(path: Path, num_images=10):
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


class _TestMethod(Method):
    _save_paths = []
    _last_step = None
    _setup_train_dataset = []
    _render_call_step = []

    def __init__(self, *args, train_dataset=None, **kwargs):
        del args, kwargs
        if train_dataset is not None:
            self._setup_train_dataset.append(train_dataset)

    @staticmethod
    def _reset():
        _TestMethod._save_paths = []
        _TestMethod._last_step = None
        _TestMethod._setup_train_dataset = []
        _TestMethod._render_call_step = []

    @classmethod
    def get_method_info(cls) -> MethodInfo:
        return {
            "method_id": "_test",
            "required_features": frozenset(("color",)),
            "supported_camera_models": frozenset(("pinhole",)),
        }


    def get_info(self) -> ModelInfo:
        return {
            "method_id": "_test",
            "loaded_step": None,
            "supported_camera_models": frozenset(
                (
                    "pinhole",
                    "opencv",
                    "opencv_fisheye",
                    "full_opencv",
                )
            ),
            "num_iterations": 13,
        }

    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        del options
        _TestMethod._render_call_step.append(_TestMethod._last_step)
        cam = camera.item()
        assert cam.image_sizes is not None
        return {
            "color": np.zeros([cam.image_sizes[1], cam.image_sizes[0], 3], dtype=np.float32),
        }

    def train_iteration(self, step: int) -> dict:
        _TestMethod._last_step = step
        return {"loss": 0.1}

    def save(self, path: str):
        self._save_paths.append(path)



def _patch_wandb_for_py37():
    try:
        from typing import Literal
    except ImportError:
        import typing
        from typing_extensions import Literal
        typing.Literal = Literal


@pytest.fixture
def wandb_init_run():
    @contextlib.contextmanager
    def fn():
        _patch_wandb_for_py37()
        import wandb.sdk.wandb_run

        mock_run = mock.Mock(wandb.sdk.wandb_run.Run)
        with mock.patch.object(wandb, "init", mock.Mock(wandb.init, return_value=mock_run)), mock.patch.object(wandb, "run", mock_run):
            yield
    return fn


@pytest.mark.parametrize("vis", ["none", "tensorboard"])
def test_train_command(mock_extras, tmp_path, vis):
    del mock_extras
    # if sys.version_info[:2] == (3, 7) and vis == "tensorboard":
    #     # TODO: Investigate why this test fails in Python 3.7
    #     pytest.skip("for some reason this test fails in Python 3.7 when run together with the other tests, but passes when run alone.")
    metrics._LPIPS_CACHE.clear()
    metrics._LPIPS_GPU_AVAILABLE = None
    sys.modules.pop("nerfbaselines._metrics_lpips", None)
    from nerfbaselines.cli._train import train_command
    from nerfbaselines import MethodSpec
    from nerfbaselines._registry import methods_registry as registry

    _ns_prefix_backup = os.environ.get("NS_PREFIX", None)
    cwd = os.getcwd()
    try:
        os.environ["NS_PREFIX"] = str(tmp_path / "prefix")
        spec: MethodSpec = {
            "id": "_test",
            "method_class": _TestMethod.__module__ + ":_TestMethod",
            "conda": {
                "environment_name": "_test",
                "python_version": "3.10",
                "install_script": "",
            }
        }
        registry["_test"] = spec

        # train_command.callback(method, checkpoint, data, output, backend, eval_single_iters, eval_all_iters, logger="none")
        make_dataset(tmp_path / "data", num_images=10)
        (tmp_path / "output").mkdir()
        assert train_command.callback is not None
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), "python", Indices.every_iters(9), Indices.every_iters(5), Indices([-1]), logger=vis)

        # Test if model was saved at the end
        assert len(_TestMethod._save_paths) > 0
        assert "13" in _TestMethod._save_paths[-1]
        assert _TestMethod._last_step == 12
        assert _TestMethod._setup_train_dataset is not None
        assert (tmp_path / "output" / "checkpoint-13").exists()
        assert _TestMethod._render_call_step == [4, 4, 9, 9, 12, 12]

        if "tensorboard" in vis:
            assert (tmp_path / "output" / "tensorboard").exists()
        else:
            assert not (tmp_path / "output" / "tensorboard").exists()

        # By default, the model should render all images at the end
        print(os.listdir(tmp_path / "output"))
        assert (tmp_path / "output" / "checkpoint-13").exists()
        assert (tmp_path / "output" / "predictions-13.tar.gz").exists()
        assert (tmp_path / "output" / "results-13.json").exists()
    finally:
        os.chdir(cwd)
        _TestMethod._reset()
        if _ns_prefix_backup is not None:
            os.environ["NS_PREFIX"] = _ns_prefix_backup
        else:
            os.environ.pop("NS_PREFIX", None)
        registry.pop("_test", None)


@pytest.mark.extras
@pytest.mark.parametrize("vis", ["none", "wandb", "wandb,tensorboard"])
def test_train_command_extras(tmp_path, vis, wandb_init_run):
    with wandb_init_run():
        metrics._LPIPS_CACHE.clear()
        metrics._LPIPS_GPU_AVAILABLE = None
        sys.modules.pop("nerfbaselines._metrics_lpips", None)
        from nerfbaselines.cli._train import train_command
        from nerfbaselines import MethodSpec
        from nerfbaselines._registry import methods_registry as registry
        _patch_wandb_for_py37()
        import wandb

        assert train_command.callback is not None

        cwd = os.getcwd()
        try:
            spec: MethodSpec = {
                "id": "_test",
                "method_class": _TestMethod.__module__ + ":_TestMethod",
                "conda": {
                    "environment_name": "_test",
                    "python_version": "3.10",
                    "install_script": "",
                }
            }
            registry["_test"] = spec

            # train_command.callback(method, checkpoint, data, output, no_wandb, backend, eval_single_iters, eval_all_iters)
            make_dataset(tmp_path / "data", num_images=10)
            (tmp_path / "output").mkdir()
            train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), "python", Indices.every_iters(9), Indices.every_iters(5), Indices([-1]), logger=vis)

            # By default, the model should render all images at the end
            print(os.listdir(tmp_path / "output"))
            assert (tmp_path / "output" / "checkpoint-13").exists()
            assert (tmp_path / "output" / "predictions-13.tar.gz").exists()
            assert (tmp_path / "output" / "results-13.json").exists()
            # LPIPS should be computed by default
            with open(tmp_path / "output" / "results-13.json", "r") as f:
                results = json.load(f)
                assert "lpips" in results["metrics"], "lpips should be in results"
            if "tensorboard" in vis:
                assert (tmp_path / "output" / "output.zip").exists(), "output artifact should be generated by default"

            _TestMethod._reset()
            (tmp_path / "output1").mkdir()
            train_command.callback(
                "_test", None, str(tmp_path / "data"), str(tmp_path / "output1"), "python", Indices.every_iters(9), Indices.every_iters(5), Indices([-1]), generate_output_artifact=False, logger="tensorboard"
            )
            print(os.listdir(tmp_path / "output1"))
            assert (tmp_path / "output1" / "checkpoint-13").exists()
            assert (tmp_path / "output1" / "predictions-13.tar.gz").exists()
            assert (tmp_path / "output1" / "results-13.json").exists()
            # LPIPS should be computed by default
            with open(tmp_path / "output1" / "results-13.json", "r") as f:
                results = json.load(f)
                assert "lpips" in results["metrics"], "lpips should be in results"
            assert not (tmp_path / "output1" / "output.zip").exists(), "output artifact should not be generated without extra metrics"

            # Test if output artifact is generated
            _TestMethod._reset()
            (tmp_path / "output2").mkdir()
            train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output2"), "python", Indices.every_iters(9), Indices.every_iters(5), Indices([-1]), generate_output_artifact=True, logger="tensorboard")
            assert (tmp_path / "output2" / "checkpoint-13").exists()
            assert (tmp_path / "output2" / "predictions-13.tar.gz").exists()
            assert (tmp_path / "output2" / "results-13.json").exists()
            if "tensorboard" in vis:
                assert (tmp_path / "output2" / "output.zip").exists(), "output artifact should be generated with extra metrics"

            wandb_init_mock: mock.Mock = cast(mock.Mock, wandb.init)
            wandb_mock: mock.Mock = cast(mock.Mock, wandb.run)
            if "wandb" not in vis:
                wandb_init_mock.assert_not_called()
                wandb_mock.log.assert_not_called()
            else:
                wandb_init_mock.assert_called_once()
                wandb_mock.log.assert_called()

                # Last log is the final evaluation
                assert wandb_mock.log.call_args[1]["step"] == 13
                all_keys = set(sum((list(k[0][0].keys()) for k in wandb_mock.log.call_args_list), []))
                print(all_keys)
                assert "eval-all-test/color" in all_keys

                eval_single_calls = []
                eval_all_calls = []
                train_calls = []
                must_have = {"train/loss", "eval-few-test/psnr", "eval-all-test/psnr"}
                print(wandb_mock.log.call_args_list)
                for args, kwargs in wandb_mock.log.call_args_list:
                    if "eval-few-test/color" in args[0]:
                        eval_single_calls.append(kwargs["step"])
                    if "eval-all-test/color" in args[0]:
                        eval_all_calls.append(kwargs["step"])
                    if "train/loss" in args[0]:
                        train_calls.append(kwargs["step"])
                    must_have.difference_update(args[0].keys())
                assert eval_single_calls == [5, 10]
                assert eval_all_calls == [13]
                assert train_calls == [13]
                assert len(must_have) == 0

            if "tensorboard" in vis:
                assert (tmp_path / "output2" / "tensorboard").exists()


        finally:
            os.chdir(cwd)
            _TestMethod._reset()
            registry.pop("_test", None)


def test_train_command_undistort(tmp_path, mock_extras):
    del mock_extras
    metrics._LPIPS_CACHE.clear()
    metrics._LPIPS_GPU_AVAILABLE = None
    sys.modules.pop("nerfbaselines._metrics_lpips", None)
    from nerfbaselines.cli._train import train_command
    from nerfbaselines import MethodSpec
    from nerfbaselines._registry import methods_registry as registry

    assert train_command.callback is not None
    render_was_called = False
    setup_data_was_called = False

    class _Method(_TestMethod):
        def get_info(self) -> ModelInfo:
            info: ModelInfo = {**super().get_info()}
            info["supported_camera_models"] = frozenset(("pinhole",))
            return info

        def __init__(self, *args, train_dataset, **kwargs):
            nonlocal setup_data_was_called
            setup_data_was_called = True
            assert all(train_dataset["cameras"].camera_models == 0)
            super().__init__(*args, train_dataset=train_dataset, **kwargs)

        def render(self, camera, *args, **kwargs):
            nonlocal render_was_called
            render_was_called = True
            assert camera.camera_models == 0, "Camera should be undistorted"
            return super().render(camera, *args, **kwargs)

    test_train_command_undistort._TestMethod = _Method  # type: ignore

    _ns_prefix_backup = os.environ.get("NS_PREFIX", None)
    cwd = os.getcwd()
    try:
        os.environ["NS_PREFIX"] = str(tmp_path / "prefix")
        spec: MethodSpec = {
            "id": "_test",
            "method_class": _TestMethod.__module__ + ":test_train_command_undistort._TestMethod",
            "conda": {
                "environment_name": "_test",
                "python_version": "3.10",
                "install_script": "",
            }
        }
        registry["_test"] = spec

        # train_command.callback(method, checkpoint, data, output, no_wandb, backend, eval_single_iters, eval_all_iters)
        make_dataset(tmp_path / "data", num_images=10)
        (tmp_path / "output").mkdir()
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), "python", Indices.every_iters(9), Indices([1]), Indices([]), logger="none")
        assert render_was_called
    finally:
        os.chdir(cwd)
        _TestMethod._reset()
        if _ns_prefix_backup is not None:
            os.environ["NS_PREFIX"] = _ns_prefix_backup
        else:
            os.environ.pop("NS_PREFIX", None)
        registry.pop("_test", None)


@pytest.mark.parametrize("output_type", ["folder", "tar"])
def test_render_command(tmp_path, output_type):
    metrics._LPIPS_CACHE.clear()
    metrics._LPIPS_GPU_AVAILABLE = None
    sys.modules.pop("nerfbaselines._metrics_lpips", None)
    from nerfbaselines.cli._train import train_command
    from nerfbaselines.cli._render import render_command
    from nerfbaselines import MethodSpec
    from nerfbaselines._registry import methods_registry as registry

    _ns_prefix_backup = os.environ.get("NS_PREFIX", None)
    cwd = os.getcwd()
    try:
        os.environ["NS_PREFIX"] = str(tmp_path / "prefix")
        spec: MethodSpec = {
            "method_class": _TestMethod.__module__ + ":_TestMethod",
            "conda": {
                "environment_name": "_test",
                "python_version": "3.10",
                "install_script": "",
            },
            "id": "_test",
        }
        registry["_test"] = spec

        make_dataset(tmp_path / "data", num_images=10)
        (tmp_path / "output").mkdir()
        # Generate checkpoint
        assert train_command.callback is not None
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"),
                               "python", Indices([]), Indices([]), Indices([]), logger="none")

        # render_command(checkpoint, data, output, split, backend)
        if output_type == "folder":
            output = tmp_path / "output2"
        else:
            output = tmp_path / "output2.tar.gz"
        assert render_command.callback is not None
        render_command.callback(str(tmp_path / "output" / "checkpoint-13"), str(tmp_path / "data"), str(output), "train", "python")

        assert output.exists()
        if output_type == "folder":
            assert output.is_dir()
            assert (output / "color").exists()
            assert (output / "gt-color").exists()
            assert (output / "info.json").exists()
        else:
            # Check tar file
            assert output.is_file()
            with tarfile.open(output, "r:gz") as tar:
                print(tar.getmembers())
                assert tar.getmember("color/2.png").isreg()
                assert tar.getmember("gt-color/2.png").isreg()
                assert tar.getmember("info.json").isreg()
                assert tar.getmember("color/2.png").size > 0
                assert tar.getmember("gt-color/2.png").size > 0

    finally:
        os.chdir(cwd)
        _TestMethod._reset()
        if _ns_prefix_backup is not None:
            os.environ["NS_PREFIX"] = _ns_prefix_backup
        else:
            os.environ.pop("NS_PREFIX", None)
        registry.pop("_test", None)
