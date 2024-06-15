from pathlib import Path
from collections import defaultdict
import os
from dataclasses import dataclass, field
from typing import Any, cast
import enum
import pytest
import sys
from unittest import mock


@pytest.fixture
def mock_nerfstudio(mock_torch):
    trainer = mock.MagicMock()
    torch = mock_torch
    config = None

    class ConfigMock(mock.MagicMock):
        def setup(self):
            nonlocal config
            config = self
            return trainer

    class DataManager:
        pass

    class DataParser:
        def __init__(self, config):
            self.config = config

        def get_dataparser_outputs(self, split="train", **kwargs):
            return self._generate_dataparser_outputs(split, **kwargs)  # type: ignore

    class InputDataset:
        def __init__(self, dataparser_outputs):
            self._dataparser_outputs = dataparser_outputs

    class CameraType(enum.Enum):
        PERSPECTIVE = 1
        FISHEYE = 2

    class Model(mock.MagicMock):
        def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle):
            input_device = camera_ray_bundle.directions.device
            num_rays_per_chunk = self.config.eval_num_rays_per_chunk
            image_height, image_width = camera_ray_bundle.origins.shape[:2]
            ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(0, num_rays_per_chunk)
            # move the chunk inputs to the model device
            outputs_lists = defaultdict(list)
            num_rays = len(camera_ray_bundle)
            for i in range(0, num_rays, num_rays_per_chunk):
                start_idx = i
                end_idx = i + num_rays_per_chunk
                ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
                # move the chunk inputs to the model device
                ray_bundle = ray_bundle.to(self.device)
                outputs = self.forward(ray_bundle=ray_bundle)
                for output_name, output in outputs.items():  # type: ignore
                    if not isinstance(output, torch.Tensor):
                        continue
                    # move the chunk outputs from the model device back to the device of the inputs.
                    outputs_lists[output_name].append(output.to(input_device))
            outputs = {}
            for output_name, outputs_list in outputs_lists.items():
                outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
            return outputs

        def get_rgba_image(self, outputs):
            return outputs["color"]

        def forward(self, ray_bundle):
            return {
                "color": torch.zeros((ray_bundle.shape[0], 3)),
                "accumulation": torch.zeros((ray_bundle.shape[0], 1)),
                "depth": torch.zeros((ray_bundle.shape[0], 1)),
            }

    @dataclass
    class NSCameras:
        camera_to_worlds: Any
        fx: Any
        fy: Any
        cx: Any
        cy: Any
        width: Any = None
        height: Any = None
        distortion_params: Any = None
        camera_type: Any = None
        times: Any = None
        metadata: Any = None

        def generate_rays(self, camera_indices, keep_shape):
            out = mock.MagicMock()
            w = self.width[camera_indices]
            h = self.height[camera_indices]
            out.origins = torch.zeros((h, w, 3))
            out.__len__ = lambda self: h * w
            t = w * h
            out.get_row_major_sliced_ray_bundle = lambda s, e: torch.zeros((min(e, t) - s, 8))
            return out

        def to(self, *args, **kwargs):
            return self

    @dataclass
    class DataparserOutputs:
        image_filenames: Any
        cameras: Any
        alpha_color: Any
        scene_box: Any
        mask_filenames: Any = None
        metadata: Any = field(default_factory=dict)
        dataparser_transform: Any = field(default_factory=lambda: torch.eye(4)[:3, :])
        dataparser_scale: float = 1.0

    oconfig = ConfigMock()
    oconfig.pipeline = mock.MagicMock()
    oconfig.pipeline.datamanager = mock.MagicMock()
    oconfig.pipeline.datamanager.train_num_rays_per_batch = torch.tensor(128)
    oconfig.pipeline.model.config.max_num_iterations = 12
    oconfig.pipeline.model.config.eval_num_rays_per_chunk = 12
    oconfig.pipeline.datamanager.dataparser = mock.MagicMock()
    oconfig.pipeline.datamanager._target = DataManager
    oconfig.eval_num_rays_per_chunk = 128
    oconfig.max_num_iterations = 12
    oconfig.relative_model_dir = "test"
    trainer.pipeline = mock.MagicMock()
    trainer.pipeline.model = Model()
    trainer.pipeline.model.config = oconfig

    def setup():
        assert config is not None
        dataparser_config = config.pipeline.datamanager.dataparser
        dataparser = dataparser_config._target(dataparser_config)
        train_dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
        _ = dataparser.get_dataparser_outputs(split="test")
        datamanager = config.pipeline.datamanager._target()
        _dataset = datamanager.dataset_type(train_dataparser_outputs)

    trainer.setup = mock.Mock(side_effect=setup)

    def train_iteration(step):
        return (torch.tensor(0.1, dtype=torch.float32), {"loss": torch.tensor(0.1, dtype=torch.float32)}, {"psnr": torch.tensor(11.0, dtype=torch.float32)})

    trainer.train_iteration = train_iteration

    with mock.patch.dict(
        sys.modules,
        {
            "nerfstudio": mock.MagicMock(),
            "nerfstudio.configs": mock.MagicMock(),
            "nerfstudio.configs.dataparser_configs": mock.MagicMock(),
            "nerfstudio.cameras": mock.MagicMock(),
            "nerfstudio.cameras.cameras": mock.MagicMock(),
            "nerfstudio.models.base_model": mock.MagicMock(),
            "nerfstudio.models": mock.MagicMock(),
            "nerfstudio.data.dataparsers.base_dataparser": mock.MagicMock(),
            "nerfstudio.data.datamanagers.base_datamanager": mock.MagicMock(),
            "nerfstudio.data.scene_box": mock.MagicMock(),
            "nerfstudio.engine.trainer": mock.MagicMock(),
            "nerfstudio.configs.method_configs": mock.MagicMock(),
            "nerfstudio.utils.colors": mock.MagicMock(),
            "yaml": mock.MagicMock(),
        },
    ):
        default_methods = ["nerfacto", "nerfacto-big", "nerfacto-huge", "tetra-nerf", "tetra-nerf-original"]
        method_configs = cast(mock.MagicMock, sys.modules["nerfstudio.configs.method_configs"])
        method_configs.all_methods = {k: oconfig for k in default_methods}
        cast(Any, sys.modules["nerfstudio.data.datamanagers.base_datamanager"]).DataManager = DataManager
        cast(Any, sys.modules["nerfstudio.data.datamanagers.base_datamanager"]).InputDataset = InputDataset
        cast(Any, sys.modules["nerfstudio.data.dataparsers.base_dataparser"]).DataParser = DataParser
        cast(Any, sys.modules["nerfstudio.data.dataparsers.base_dataparser"]).DataparserOutputs = DataparserOutputs
        cast(Any, sys.modules["nerfstudio.cameras.cameras"]).CameraType = CameraType
        cast(Any, sys.modules["nerfstudio.cameras.cameras"]).Cameras = NSCameras
        cast(Any, sys.modules["nerfstudio.cameras"]).camera_utils = camera_utils = mock.MagicMock()
        cast(Any, sys.modules["nerfstudio.models.base_model"]).Model = Model
        cast(Any, sys.modules["nerfstudio.engine.trainer"]).Trainer = type(trainer)
        cast(Any, sys.modules["yaml"]).dump.return_value = "test"
        cast(Any, sys.modules["yaml"]).load.return_value = oconfig
        camera_utils.auto_orient_and_center_poses = lambda poses, *args, **kwargs: (poses, torch.eye(4)[:3])

        from nerfbaselines.methods.nerfstudio import NerfStudio

        old_save = NerfStudio.save

        def new_save(self, path):
            old_save(self, path)
            os.makedirs(Path(path) / "test" / f"ckpt-{self.step-1}.pth")

        with mock.patch.object(NerfStudio, "save", new_save):
            yield None


@pytest.mark.parametrize(
    "method_name",
    [pytest.param(k, marks=[pytest.mark.method(k)]) for k in ["nerfacto", "nerfacto:big", "nerfacto:huge"]],
)
def test_train_nerfstudio_mocked(mock_nerfstudio, run_test_train, method_name):
    run_test_train()


@pytest.mark.apptainer
@pytest.mark.method("nerfacto")
def test_train_nerfstudio_apptainer(run_test_train):
    run_test_train()


@pytest.mark.docker
@pytest.mark.method("nerfacto")
def test_train_nerfstudio_docker(run_test_train):
    run_test_train()


@pytest.mark.conda
@pytest.mark.method("nerfacto")
def test_train_nerfstudio_conda(run_test_train):
    run_test_train()
