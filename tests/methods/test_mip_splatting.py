import pytest
from collections import namedtuple
import numpy as np
import sys
from unittest import mock


METHOD_NAME = "mip-splatting"


@pytest.fixture
def mock_mip_splatting(mock_torch):
    torch = mock_torch
    with mock.patch.dict(
        sys.modules,
        {
            "arguments": mock.MagicMock(),
            "gaussian_renderer": mock.MagicMock(),
            "scene": mock.MagicMock(),
            "scene.cameras": mock.MagicMock(),
            "scene.dataset_readers": mock.MagicMock(),
            "utils": mock.MagicMock(),
            "utils.general_utils": mock.MagicMock(),
            "utils.image_utils": mock.MagicMock(),
            "utils.loss_utils": mock.MagicMock(),
            "utils.sh_utils": mock.MagicMock(),
            "train": mock.MagicMock(),
        },
    ):
        # from arguments import ModelParams, PipelineParams, OptimizationParams
        arguments = sys.modules["arguments"]

        def setup_model_args(parser):
            parser.add_argument("--source_path")
            parser.add_argument("--sh_degree", default=3)
            parser.add_argument("--densify_until_iter", default=7)
            parser.add_argument("--densify_from_iter", default=1)
            parser.add_argument("--densify_grad_threshold", default=0.1)
            parser.add_argument("--densification_interval", default=2)
            parser.add_argument("--opacity_reset_interval", default=2)
            parser.add_argument("--lambda_dssim", default=0.1)
            parser.add_argument("--white_background", action="store_true")
            parser.add_argument("--sample_more_highres", action="store_true")
            parser.add_argument("--resample_gt_image", action="store_true")
            parser.add_argument("--ray_jitter", action="store_true")
            parser.add_argument("--kernel_size", "-kernel_size", default=0.1, type=float)
            out = mock.MagicMock()
            out.extract = lambda args: args
            return out

        arguments.ModelParams = setup_model_args

        def setup_opt_args(parser):
            parser.add_argument("--iterations", type=int)
            out = mock.MagicMock()
            out.extract = lambda args: args
            return out

        arguments.OptimizationParams = setup_opt_args

        def raise_error():
            raise NotImplementedError()

        sys.modules["scene"].sceneLoadTypeCallbacks = sceneLoadTypeCallbacks = {
            "Colmap": raise_error,
        }
        class Camera:
            def __init__(self,  **kwargs):
                self.image_width = kwargs["image"].shape[1]
                self.image_height = kwargs["image"].shape[0]
                self.original_image = kwargs["image"]
                for k, v in kwargs.items():
                    setattr(self, k, v)

        def Scene(opt, gaussians, load_iteration):
            scene_info = sceneLoadTypeCallbacks["Colmap"]()
            scene = mock.MagicMock()

            def loadCam(cam_info):
                dct = cam_info._asdict()
                dct.pop("image_path")
                dct.pop("width")
                dct.pop("height")
                dct["FoVx"] = dct.pop("FovX")
                dct["FoVy"] = dct.pop("FovY")
                dct["image"] = torch.tensor(np.array(cam_info.image) / 255.0)
                dct['colmap_id'] = 0
                dct['gt_alpha_mask'] = None
                dct['data_device'] = "cuda"
                return Camera(**dct)

            scene.getTrainCameras = lambda: list(map(loadCam, scene_info.train_cameras))
            return scene

        sys.modules["scene"].Scene = Scene

        def GaussianModel(*args, **kwargs):
            out = mock.MagicMock()
            out.capture.return_value = None
            return out

        sys.modules["scene"].GaussianModel = GaussianModel
        sys.modules["scene.cameras"].Camera = Camera
        sys.modules["scene.dataset_readers"].CameraInfo = namedtuple("CameraInfo", ["uid", "R", "T", "FovY", "FovX", "image", "image_path", "image_name", "width", "height"])
        sys.modules["scene.dataset_readers"].SceneInfo = namedtuple("SceneInfo", ["point_cloud", "train_cameras", "test_cameras", "nerf_normalization", "ply_path"])
        sys.modules["utils.loss_utils"].l1_loss = lambda a, b: torch.abs(a - b).sum()
        sys.modules["utils.loss_utils"].ssim = lambda a, b: torch.abs(a - b).sum()

        def render(viewpoint, gaussians, pipe, background, kernel_size: float, scaling_modifier = 1.0, override_color = None, subpixel_offset=None):
            shape = viewpoint.image.shape
            return dict(
                render=torch.zeros(shape, dtype=torch.float32),
                viewspace_points=torch.zeros((124, 2), dtype=torch.float32),
                visibility_filter=torch.zeros((124, 2), dtype=bool),
                radii=torch.zeros((124, 2), dtype=torch.float32),
            )

        sys.modules["gaussian_renderer"].render = render
        yield None


@pytest.mark.method(METHOD_NAME)
def test_train_mip_splatting_mocked(mock_mip_splatting, run_test_train):
    _ = mock_mip_splatting
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.apptainer
def test_train_mip_splatting_apptainer(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.docker
def test_train_mip_splatting_docker(run_test_train):
    run_test_train()
