import shutil
import json
from typing import Iterable
from pathlib import Path
import os
import numpy as np
from nerfbaselines import Method, MethodInfo, Cameras, RenderOutput, CurrentProgress, Indices
from nerfbaselines.backends.conda import CondaMethod
from nerfbaselines.datasets import _colmap_utils as colmap_utils
from nerfbaselines.cameras import CameraModel
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
    _install_called_count = 0
    _save_paths = []
    _last_step = None
    _setup_train_dataset = []
    _render_call_step = []

    @staticmethod
    def _reset():
        _TestMethod._install_called_count = 0
        _TestMethod._save_paths = []
        _TestMethod._last_step = None
        _TestMethod._setup_train_dataset = []
        _TestMethod._render_call_step = []

    @classmethod
    def install(cls):
        _TestMethod._install_called_count += 1

    def get_info(self) -> MethodInfo:
        return MethodInfo(
            loaded_step=None,
            supported_camera_models=frozenset(
                (
                    CameraModel.PINHOLE,
                    CameraModel.OPENCV,
                    CameraModel.OPENCV_FISHEYE,
                    CameraModel.FULL_OPENCV,
                )
            ),
            num_iterations=13,
        )

    def render(self, cameras: Cameras, progress_callback=None) -> Iterable[RenderOutput]:
        _TestMethod._render_call_step.append(_TestMethod._last_step)
        for i in range(len(cameras)):
            cam = cameras[i]
            yield {
                "color": np.zeros([cam.image_sizes[1], cam.image_sizes[0], 3], dtype=np.float32),
            }
            if progress_callback is not None:
                progress_callback(
                    CurrentProgress(
                        i=i + 1,
                        total=len(cameras),
                        image_i=i + 1,
                        image_total=len(cameras),
                    )
                )

    def setup_train(self, train_dataset, *, num_iterations: int):
        self._setup_train_dataset.append(train_dataset)

    def train_iteration(self, step: int):
        _TestMethod._last_step = step
        return {"loss": 0.1}

    def save(self, path: Path):
        self._save_paths.append(str(path))


@pytest.fixture
def wandb_init_run():
    import wandb.sdk.wandb_run

    mock_run = mock.Mock(wandb.sdk.wandb_run.Run)
    with mock.patch.object(wandb, "init", mock.Mock(wandb.init, return_value=mock_run)), mock.patch.object(wandb, "run", mock_run):
        yield


@pytest.mark.parametrize("vis", ["none", "wandb", "tensorboard", "wandb+tensorboard"])
def test_train_command(tmp_path, wandb_init_run, vis):
    from nerfbaselines.train import train_command
    from nerfbaselines.registry import registry, MethodSpec
    import wandb

    _ns_prefix_backup = os.environ.get("NS_PREFIX", None)
    try:
        os.environ["NS_PREFIX"] = str(tmp_path / "prefix")
        registry["_test"] = MethodSpec(method=_TestMethod, conda=CondaMethod.wrap(_TestMethod, conda_name="_test", python_version="3.10", install_script=""))

        # train_command.callback(method, checkpoint, data, output, verbose, backend, eval_single_iters, eval_all_iters, vis="none")
        make_dataset(tmp_path / "data")
        (tmp_path / "output").mkdir()
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), True, "python", Indices.every_iters(5), Indices([-1]), vis=vis)

        # Test if model was saved at the end
        assert len(_TestMethod._save_paths) > 0
        assert "13" in _TestMethod._save_paths[-1]
        assert _TestMethod._last_step == 12
        assert _TestMethod._install_called_count == 1
        assert _TestMethod._setup_train_dataset is not None
        assert (tmp_path / "output" / "checkpoint-13").exists()
        assert _TestMethod._render_call_step == [4, 9, 12]

        wandb_init_mock: mock.Mock = wandb.init
        wandb_mock: mock.Mock = wandb.run
        if "wandb" not in vis:
            wandb_init_mock.assert_not_called()
            wandb_mock.log.assert_not_called()
        else:
            wandb_init_mock.assert_called_once()
            wandb_mock.log.assert_called()

            # Last log is the final evaluation
            assert wandb_mock.log.call_args[1]["step"] == 13
            last_event = wandb_mock.log.call_args[0][0]
            assert "eval-all-images/color" in last_event

            eval_single_calls = []
            eval_all_calls = []
            train_calls = []
            must_have = {"train/loss", "eval-single-image/psnr", "eval-all-images/psnr"}
            print(wandb_mock.log.call_args_list)
            for args, kwargs in wandb_mock.log.call_args_list:
                if "eval-single-image/color" in args[0]:
                    eval_single_calls.append(kwargs["step"])
                if "eval-all-images/color" in args[0]:
                    eval_all_calls.append(kwargs["step"])
                if "train/loss" in args[0]:
                    train_calls.append(kwargs["step"])
                must_have.difference_update(args[0].keys())
            assert eval_single_calls == [5, 10]
            assert eval_all_calls == [13]
            assert train_calls == [10, 13]
            assert len(must_have) == 0

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
        _TestMethod._reset()
        if _ns_prefix_backup is not None:
            os.environ["NS_PREFIX"] = _ns_prefix_backup
        else:
            os.environ.pop("NS_PREFIX", None)
        registry.pop("_test", None)


@pytest.mark.extras
def test_train_command_extras(tmp_path):
    from nerfbaselines.train import train_command
    from nerfbaselines.registry import registry, MethodSpec

    try:
        registry["_test"] = MethodSpec(method=_TestMethod, conda=CondaMethod.wrap(_TestMethod, conda_name="_test", python_version="3.10", install_script=""))

        # train_command.callback(method, checkpoint, data, output, no_wandb, verbose, backend, eval_single_iters, eval_all_iters)
        make_dataset(tmp_path / "data")
        (tmp_path / "output").mkdir()
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), True, "python", Indices.every_iters(5), Indices([-1]), vis="tensorboard")

        # By default, the model should render all images at the end
        print(os.listdir(tmp_path / "output"))
        assert (tmp_path / "output" / "checkpoint-13").exists()
        assert (tmp_path / "output" / "predictions-13.tar.gz").exists()
        assert (tmp_path / "output" / "results-13.json").exists()
        assert (tmp_path / "output" / "tensorboard").exists()
        # LPIPS should be computed by default
        with open(tmp_path / "output" / "results-13.json", "r") as f:
            results = json.load(f)
            assert "lpips" in results["metrics"], "lpips should be in results"
        assert (tmp_path / "output" / "output.zip").exists(), "output artifact should be generated by default"

        _TestMethod._reset()
        shutil.rmtree(tmp_path / "output")
        (tmp_path / "output").mkdir()
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), True, "python", Indices.every_iters(5), Indices([-1]), disable_extra_metrics=True, vis="tensorboard")
        print(os.listdir(tmp_path / "output"))
        assert (tmp_path / "output" / "checkpoint-13").exists()
        assert (tmp_path / "output" / "predictions-13.tar.gz").exists()
        assert (tmp_path / "output" / "results-13.json").exists()
        assert (tmp_path / "output" / "tensorboard").exists()
        # LPIPS should be computed by default
        with open(tmp_path / "output" / "results-13.json", "r") as f:
            results = json.load(f)
            assert "lpips" not in results["metrics"], "lpips should not be in results"
        assert not (tmp_path / "output" / "output.zip").exists(), "output artifact should not be generated without extra metrics"

        _TestMethod._reset()
        shutil.rmtree(tmp_path / "output")
        (tmp_path / "output").mkdir()
        train_command.callback(
            "_test", None, str(tmp_path / "data"), str(tmp_path / "output"), True, "python", Indices.every_iters(5), Indices([-1]), generate_output_artifact=False, vis="tensorboard"
        )
        print(os.listdir(tmp_path / "output"))
        assert (tmp_path / "output" / "checkpoint-13").exists()
        assert (tmp_path / "output" / "predictions-13.tar.gz").exists()
        assert (tmp_path / "output" / "results-13.json").exists()
        assert (tmp_path / "output" / "tensorboard").exists()
        # LPIPS should be computed by default
        with open(tmp_path / "output" / "results-13.json", "r") as f:
            results = json.load(f)
            assert "lpips" in results["metrics"], "lpips should be in results"
        assert not (tmp_path / "output" / "output.zip").exists(), "output artifact should not be generated without extra metrics"

        # Test if output artifact is generated
        _TestMethod._reset()
        shutil.rmtree(tmp_path / "output")
        (tmp_path / "output").mkdir()
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), True, "python", Indices.every_iters(5), Indices([-1]), generate_output_artifact=True, vis="tensorboard")
        assert (tmp_path / "output" / "checkpoint-13").exists()
        assert (tmp_path / "output" / "predictions-13.tar.gz").exists()
        assert (tmp_path / "output" / "results-13.json").exists()
        assert (tmp_path / "output" / "tensorboard").exists()
        assert (tmp_path / "output" / "output.zip").exists(), "output artifact should not be generated without extra metrics"

    finally:
        _TestMethod._reset()
        registry.pop("_test", None)


def test_train_command_no_extras(tmp_path):
    from nerfbaselines.train import train_command
    from nerfbaselines.registry import registry, MethodSpec

    try:
        registry["_test"] = MethodSpec(method=_TestMethod, conda=CondaMethod.wrap(_TestMethod, conda_name="_test", python_version="3.10", install_script=""))

        # train_command.callback(method, checkpoint, data, output, no_wandb, verbose, backend, eval_single_iters, eval_all_iters)
        make_dataset(tmp_path / "data")
        (tmp_path / "output").mkdir()
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), True, "python", Indices.every_iters(5), Indices([-1]), vis="none")

        # By default, the model should render all images at the end
        print(os.listdir(tmp_path / "output"))
        assert (tmp_path / "output" / "checkpoint-13").exists()
        assert (tmp_path / "output" / "predictions-13.tar.gz").exists()
        assert (tmp_path / "output" / "results-13.json").exists()
        with open(tmp_path / "output" / "results-13.json", "r") as f:
            results = json.load(f)
            assert "lpips" not in results["metrics"], "lpips should not be in results"

        # LPIPS should not be computed without extras, output artifact should not be produced
        assert not (tmp_path / "output" / "output.zip").exists()

    finally:
        _TestMethod._reset()
        registry.pop("_test", None)


def test_train_command_undistort(tmp_path, wandb_init_run):
    from nerfbaselines.train import train_command
    from nerfbaselines.registry import registry, MethodSpec

    render_was_called = False
    setup_data_was_called = False

    class _Method(_TestMethod):
        def get_info(self) -> MethodInfo:
            info = super().get_info()
            info.supported_camera_models = frozenset((CameraModel.PINHOLE,))
            return info

        def setup_train(self, train_dataset, *args, **kwargs):
            nonlocal setup_data_was_called
            setup_data_was_called = True
            assert all(train_dataset.cameras.camera_types == 0)
            return super().setup_train(train_dataset, *args, **kwargs)

        def render(self, cameras, *args, **kwargs):
            nonlocal render_was_called
            render_was_called = True
            assert all(cameras.camera_types == 0)
            return super().render(cameras, *args, **kwargs)

    _ns_prefix_backup = os.environ.get("NS_PREFIX", None)
    try:
        os.environ["NS_PREFIX"] = str(tmp_path / "prefix")
        registry["_test"] = MethodSpec(method=_Method, conda=CondaMethod.wrap(_Method, conda_name="_test", python_version="3.10", install_script=""))

        # train_command.callback(method, checkpoint, data, output, no_wandb, verbose, backend, eval_single_iters, eval_all_iters)
        make_dataset(tmp_path / "data")
        (tmp_path / "output").mkdir()
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), True, "python", Indices([1]), Indices([]), vis="none")
        assert render_was_called
    finally:
        _TestMethod._reset()
        if _ns_prefix_backup is not None:
            os.environ["NS_PREFIX"] = _ns_prefix_backup
        else:
            os.environ.pop("NS_PREFIX", None)
        registry.pop("_test", None)


@pytest.mark.parametrize("output_type", ["folder", "tar"])
def test_render_command(tmp_path, output_type):
    from nerfbaselines.train import train_command
    from nerfbaselines.render import render_command
    from nerfbaselines.registry import registry, MethodSpec

    _ns_prefix_backup = os.environ.get("NS_PREFIX", None)
    try:
        os.environ["NS_PREFIX"] = str(tmp_path / "prefix")
        registry["_test"] = MethodSpec(method=_TestMethod, conda=CondaMethod.wrap(_TestMethod, conda_name="_test", python_version="3.10", install_script=""))

        make_dataset(tmp_path / "data")
        (tmp_path / "output").mkdir()
        # Generate checkpoint
        train_command.callback("_test", None, str(tmp_path / "data"), str(tmp_path / "output"), True, "python", Indices([]), Indices([]), vis="none")

        # render_command(checkpoint, data, output, split, verbose, backend)
        if output_type == "folder":
            output = tmp_path / "output2"
            (tmp_path / "output2").mkdir()
        else:
            output = tmp_path / "output2.tar.gz"
        render_command.callback(str(tmp_path / "output" / "checkpoint-13"), str(tmp_path / "data"), output, "train", True, "python")

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
        _TestMethod._reset()
        if _ns_prefix_backup is not None:
            os.environ["NS_PREFIX"] = _ns_prefix_backup
        else:
            os.environ.pop("NS_PREFIX", None)
        registry.pop("_test", None)
