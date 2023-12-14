import os
from dataclasses import dataclass, field
from typing import Any
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
            return self._generate_dataparser_outputs(split, **kwargs)

    class InputDataset:
        def __init__(self, dataparser_outputs):
            self._dataparser_outputs = dataparser_outputs

    class CameraType(enum.Enum):
        PERSPECTIVE = 1
        FISHEYE = 2

    class Model(mock.MagicMock):
        def get_outputs_for_camera_ray_bundle(*args, **kwargs):
            return super().get_outputs_for_camera_ray_bundle(*args, **kwargs)

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
    oconfig.pipeline.datamanager.dataparser = mock.MagicMock()
    oconfig.pipeline.datamanager._target = DataManager
    oconfig.eval_num_rays_per_chunk = 128
    oconfig.relative_model_dir = "test"
    trainer.pipeline = mock.MagicMock()
    trainer.pipeline.model = Model()
    trainer.pipeline.model.config = oconfig

    def setup():
        dataparser_config = config.pipeline.datamanager.dataparser
        dataparser = dataparser_config._target(dataparser_config)
        train_dataparser_outputs = dataparser.get_dataparser_outputs(split="train")
        _ = dataparser.get_dataparser_outputs(split="test")
        datamanager = config.pipeline.datamanager._target()
        _dataset = datamanager.dataset_type(train_dataparser_outputs)
        # if len(train_dataparser_outputs.image_filenames) > 0:
        #     for i in range(3):
        #         img = dataset.get_image(i)
        #         assert img is not None
        #         assert len(img.shape) == 3
        #         assert img.dtype == torch.float32

    trainer.setup = mock.Mock(side_effect=setup)

    def train_iteration(step):
        return (torch.tensor(0.1, dtype=torch.float32), {"loss": torch.tensor(0.1, dtype=torch.float32)}, {"psnr": torch.tensor(11.0, dtype=torch.float32)})

    trainer.train_iteration = train_iteration

    with mock.patch.dict(
        sys.modules,
        {
            "nerfstudio": mock.MagicMock(),
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
        method_configs = sys.modules["nerfstudio.configs.method_configs"]
        method_configs.all_methods = {k: oconfig for k in default_methods}
        sys.modules["nerfstudio.data.datamanagers.base_datamanager"].DataManager = DataManager
        sys.modules["nerfstudio.data.datamanagers.base_datamanager"].InputDataset = InputDataset
        sys.modules["nerfstudio.data.dataparsers.base_dataparser"].DataParser = DataParser
        sys.modules["nerfstudio.data.dataparsers.base_dataparser"].DataparserOutputs = DataparserOutputs
        sys.modules["nerfstudio.cameras.cameras"].CameraType = CameraType
        sys.modules["nerfstudio.cameras.cameras"].Cameras = NSCameras
        sys.modules["nerfstudio.cameras"].camera_utils = camera_utils = mock.MagicMock()
        sys.modules["nerfstudio.models.base_model"].Model = Model
        sys.modules["nerfstudio.engine.trainer"].Trainer = type(trainer)
        sys.modules["yaml"].dump.return_value = "test"
        sys.modules["yaml"].load.return_value = oconfig
        camera_utils.auto_orient_and_center_poses = lambda poses, *args, **kwargs: (poses, torch.eye(4)[:3])

        from nerfbaselines.methods._impl.nerfstudio import NerfStudio

        old_save = NerfStudio.save

        def new_save(self, path):
            old_save(self, path)
            os.makedirs(path / "test" / f"ckpt-{self.step-1}.pth")

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
