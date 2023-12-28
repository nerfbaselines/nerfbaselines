import numpy as np
import logging
import time
import tempfile
import torch
from pathlib import Path
from typing import Callable

from ..types import Method, Dataset
from ..cameras import Cameras, CameraModel
from ..datasets._colmap_utils import qvec2rotmat, rotmat2qvec
from ..utils import convert_image_dtype
from ..datasets import load_dataset

from nerfstudio.configs.base_config import ViewerConfig, LoggingConfig, LocalWriterConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.data.datamanagers.base_datamanager import DataManager
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.data.datasets.base_dataset import InputDataset as NSDataset, DataparserOutputs
from nerfstudio.utils import writer
from nerfstudio.cameras.cameras import Cameras as NSCameras
from nerfstudio.viewer.server.viewer_state import ViewerState
from nerfstudio.cameras.rays import RayBundle


def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w


def get_position_quaternion(c2s):
    position = c2s[..., :3, 3]
    wxyz = np.stack([rotmat2qvec(x) for x in c2s[..., :3, :3].reshape(-1, 3, 3)], 0)
    wxyz = wxyz.reshape(c2s.shape[:-2] + (4,))
    return position, wxyz


def _map_distortion_parameters(distortion_parameters):
    distortion_parameters = np.concatenate(
        (
            distortion_parameters[..., :6],
            np.zeros((*distortion_parameters.shape[:-1], 6 - min(6, distortion_parameters.shape[-1])), dtype=distortion_parameters.dtype),
        ),
        -1,
    )

    distortion_parameters = distortion_parameters[..., [0, 1, 4, 5, 2, 3]]
    return distortion_parameters


def _get_dataparser_outputs(dataset: Dataset, transform):
    aabb_scale = 1.0
    scene_box = SceneBox(aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32))

    c2w = pad_poses(dataset.cameras.poses)
    c2w = transform @ c2w
    c2w = c2w[..., :3, :4]
    th_poses = torch.from_numpy(c2w).float()
    # Transform?
    distortion_parameters = torch.from_numpy(_map_distortion_parameters(dataset.cameras.distortion_parameters))
    camera_types_map = {
        CameraModel.PINHOLE.value: 0,
        CameraModel.OPENCV.value: 0,
        CameraModel.FULL_OPENCV.value: 0,
        CameraModel.OPENCV_FISHEYE.value: 1,
    }
    cameras = NSCameras(
        camera_to_worlds=th_poses,
        fx=torch.from_numpy(dataset.cameras.intrinsics[..., 0]).contiguous(),
        fy=torch.from_numpy(dataset.cameras.intrinsics[..., 1]).contiguous(),
        cx=torch.from_numpy(dataset.cameras.intrinsics[..., 2]).contiguous(),
        cy=torch.from_numpy(dataset.cameras.intrinsics[..., 3]).contiguous(),
        distortion_params=distortion_parameters.contiguous(),
        width=torch.from_numpy(dataset.cameras.image_sizes[..., 0]).long().contiguous(),
        height=torch.from_numpy(dataset.cameras.image_sizes[..., 1]).long().contiguous(),
        camera_type=torch.tensor([camera_types_map.get(x, 0) for x in dataset.cameras.camera_types.tolist()], dtype=torch.long),
    )
    metadata = {}
    transform_matrix = torch.eye(4, dtype=torch.float32)
    scale_factor = 1.0
    return DataparserOutputs(
        dataset.file_paths,
        cameras,
        None,
        scene_box,
        None,
        metadata,
        dataparser_transform=transform_matrix[..., :3, :].contiguous(),  # pylint: disable=protected-access
        dataparser_scale=scale_factor,
    )


def pad_poses(poses):
    if poses.shape[-2] == 3:
        poses = np.concatenate([poses, np.tile(np.eye(4, dtype=np.float32)[-1:], (*poses.shape[:-2], 1, 1))], -2)
    return poses


class FakeViewerState(ViewerState):
    def __init__(self, *args, transform, **kwargs):
        self.transform = transform
        super().__init__(*args, **kwargs)


class FakeDataManager(DataManager):
    def __init__(self):
        super().__init__()
        self.includes_time = False


class FakeModel(torch.nn.Module):
    def __init__(self, method: Method, get_viewer_state: Callable[[], ViewerState], transform):
        super().__init__()
        self.method = method
        self.get_viewer_state = get_viewer_state

        self.transform = transform if transform is not None else np.eye(4, dtype=np.float32)
        self._inv_transform = np.linalg.inv(self.transform)

    @property
    def device(self):
        return torch.device("cpu")

    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle):
        h, w, _ = camera_ray_bundle.origins.shape

        # Generate image using the standard pipeline
        camera = self.get_viewer_state().get_camera(h, w)
        assert camera is not None, "Camera is not initialized"

        c2w = camera.camera_to_worlds.numpy()
        c2w = pad_poses(c2w)
        c2w = self._inv_transform @ c2w
        c2w = c2w[..., :3, :4]

        out = next(
            iter(
                self.method.render(
                    Cameras(
                        poses=c2w,
                        normalized_intrinsics=np.concatenate(
                            (
                                camera.fx.numpy() / w,
                                camera.fy.numpy() / w,
                                camera.cx.numpy() / w,
                                camera.cy.numpy() / w,
                            ),
                            -1,
                        ),
                        image_sizes=np.array([[w, h]], dtype=np.int32),
                        distortion_parameters=(np.zeros((1, 6), dtype=np.float32) if camera.distortion_params is None else _map_distortion_parameters(camera.distortion_params.numpy())),
                        camera_types=(
                            np.array([camera.camera_type.item() + 1], dtype=np.int32)
                            if camera.distortion_params is not None and not torch.allclose(camera.distortion_params, torch.zeros_like(camera.distortion_params))
                            else np.zeros((1,), dtype=np.float32)
                        ),
                        nears_fars=None,
                    )
                )
            )
        )
        assert out["color"].shape[:-1] == (h, w)
        out_th = {"rgb": torch.from_numpy(convert_image_dtype(out["color"], np.float32))}
        return out_th


class FakePipeline(Pipeline):
    def __init__(self, method, get_viewer_state, transform):
        super().__init__()
        self.method = method
        self.datamanager = FakeDataManager()
        self.model = FakeModel(method, get_viewer_state, transform)

    @property
    def device(self):
        return self.model.device


class FakeDataset(NSDataset):
    def __init__(self, dataset: Dataset, transform):
        super().__init__(dataparser_outputs=_get_dataparser_outputs(dataset, transform))


class NerfstudioViewer:
    def __init__(self, method: Method, port, transform=None):
        self.port = port
        self.method = method
        self.transform = transform if transform is not None else np.eye(4, dtype=np.float32)

        self._dataset_views = {}
        self.viewer_state = None

    def run(self):
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                config = ViewerConfig()
                viewer_log_path = tmp_path / "viewer-log.txt"
                logging.info(f"Logging viewer to {viewer_log_path}")
                pipeline = FakePipeline(self.method, lambda: viewer_state, transform=self.transform)
                viewer_state = FakeViewerState(
                    config,
                    log_filename=viewer_log_path,
                    datapath=tmp_path / "data",
                    pipeline=pipeline,
                    transform=self.transform,
                )
                self.viewer_state = viewer_state
                self._update_scene()

                writer.setup_local_writer(
                    LoggingConfig(
                        local_writer=LocalWriterConfig(enable=False),
                    ),
                    max_iter=1,
                    banner_messages=[],
                )

                viewer_state.viser_server.set_training_state("completed")
                viewer_state.update_scene(step=0)
                while True:
                    time.sleep(0.01)
        finally:
            self.viewer_state = None

    def _update_scene(self):
        if self.viewer_state is None:
            return
        if "test" not in self._dataset_views or "train" not in self._dataset_views:
            return

        self.viewer_state.init_scene(
            train_dataset=FakeDataset(self._dataset_views["train"], transform=self.transform),
            train_state="completed",
            eval_dataset=FakeDataset(self._dataset_views["test"], transform=self.transform),
        )
        self._dataset_views = {}
        self.viewer_state.viser_server.set_training_state("completed")
        self.viewer_state.update_scene(step=0)

    def add_dataset_views(self, dataset: Dataset, split: str):
        self._dataset_views[split] = dataset
        self._update_scene()


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (np.random.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=a.dtype,
    )
    return np.eye(3, dtype=a.dtype) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def get_orientation_transform(poses):
    origins = poses[..., :3, 3]
    mean_origin = np.mean(origins, 0)
    translation = mean_origin
    up = np.mean(poses[:, :3, 1], 0)
    up = up / np.linalg.norm(up)

    rotation = rotation_matrix(up, np.array([0, 0, 1], dtype=up.dtype))
    transform = np.concatenate([rotation, rotation @ -translation[..., None]], -1)
    transform = np.concatenate([transform, np.array([[0, 0, 0, 1]], dtype=transform.dtype)], 0)
    return transform


def run_nerfstudio_viewer(method: Method, data: str, port=6006):
    if data is not None:
        features = frozenset({"color"})
        train_dataset = load_dataset(data, split="test", features=features)
        test_dataset = load_dataset(data, split="train", features=features)
        # transform = get_orientation_transform(train_dataset.cameras.poses)
        transform = None
        # NOTE: transform is not supported at the moment
        # The reason is that we cannot undo the transform inside the trajectory export panel
        server = NerfstudioViewer(method, port=port, transform=transform)

        train_dataset.load_features(features)
        server.add_dataset_views(train_dataset, "train")
        test_dataset.load_features(features)
        server.add_dataset_views(test_dataset, "test")
    else:
        server = NerfstudioViewer(method, port=port)
    server.run()
