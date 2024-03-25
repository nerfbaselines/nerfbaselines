import contextlib
from pathlib import Path
from collections import deque
from time import perf_counter
from typing import Optional

import numpy as np
import viser

from ..types import Method, Dataset, FrozenSet, DatasetFeature, Literal
from ..cameras import Cameras
from ..datasets._colmap_utils import qvec2rotmat, rotmat2qvec
from ..utils import CancellationToken, CancelledException, cancellable, assert_not_none
from ..pose_utils import apply_transform, get_transform_and_scale, invert_transform
from ..datasets import load_dataset
from ..backends._rpc import EventCancellationToken


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


ControlType = Literal["object-centric", "default"]


class ViserViewer:
    def __init__(self, 
                 method: Optional[Method], 
                 port, 
                 transform=None, 
                 initial_pose=None,
                 control_type: ControlType = "default"):
        self.port = port
        self.method = method
        self.transform = transform if transform is not None else np.eye(4, dtype=np.float32)
        self._inv_transform = np.linalg.inv(self.transform)

        self._render_state = {}

        self._render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)
        self.server.world_axes.visible = True

        self.need_update = False
        self.resolution_slider = self.server.add_gui_slider("Resolution", min=30, max=4096, step=2, initial_value=1024)

        @self.resolution_slider.on_update
        def _(_):
            self._reset_render()

        self.c2ws = []
        self.camera_infos = []

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_camera_handle: viser.CameraHandle):
                self._reset_render()

            if initial_pose is not None:
                pos, quat = get_position_quaternion(
                    apply_transform(self.transform, initial_pose)
                )
                client.camera.position = pos
                client.camera.wxyz = quat
                client.camera.up_direction = np.array([0, 0, 1], dtype=np.float32)

                # For object-centric scenes, we look at the origin
                if True:
                    client.camera.look_at = np.array([0, 0, 0], dtype=np.float32)

        self._cancellation_token = None
        self._setup_gui()

    def _reset_render(self):
        self._render_state = {}
        if self._cancellation_token is not None:
            self._cancellation_token.cancel()
            self._cancellation_token = None

    def _setup_gui(self):
        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

    def run(self):
        while True:
            if self.method is not None:
                try:
                    self._update()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"Error: {e}")

    
    def add_initial_point_cloud(self, points, colors):
        transform, scale = get_transform_and_scale(self.transform)
        points = np.concatenate([points, np.ones((len(points), 1))], -1) @ transform.T
        points = points[..., :-1] / points[..., -1:]
        points *= scale
        self.server.add_point_cloud(
            "/initial-point-cloud",
            points=points,
            colors=colors,
            point_size=0.005,
            point_shape="circle",
        )

    def add_dataset_views(self, dataset: Dataset, split: str):
        def _set_view_to_camera(handle: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]):
            with handle.client.atomic():
                handle.client.camera.position = handle.target.position
                handle.client.camera.wxyz = handle.target.wxyz
            self._reset_render()

        downsample_factor = 2
        for i, (cam, path) in enumerate(zip(dataset.cameras, dataset.file_paths)):
            assert cam.image_sizes is not None, "dataset.image_sizes must be set"
            image = None
            if dataset.images is not None:
                image = dataset.images[i]
            if str(path).startswith("/undistorted/"):
                path = str(path)[len("/undistorted/") :]
            else:
                path = str(Path(path).relative_to(Path(dataset.file_paths_root or "")))
            c2w = apply_transform(self.transform, cam.poses)
            pos, quat = get_position_quaternion(c2w)
            W, H = cam.image_sizes.tolist()
            fy = cam.intrinsics[1]
            handle = self.server.add_camera_frustum(
                f"/dataset-{split}/{path}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.03,
                position=pos,
                wxyz=quat,
                image=image[::downsample_factor, ::downsample_factor] if image is not None else None,
            )
            handle.on_click(_set_view_to_camera)

    def _update(self):
        for client in self.server.get_clients().values():
            render_state = self._render_state.get(client.client_id, 0)
            if render_state < 2:
                start = perf_counter()
                camera = client.camera

                w_total = self.resolution_slider.value
                h_total = int(self.resolution_slider.value / camera.aspect)
                focal = h_total / 2 / np.tan(camera.fov / 2)

                num_rays_total = num_rays = w_total * h_total
                w, h = w_total, h_total

                # In state 0, we render a low resolution image to display it fast
                if render_state == 0:
                    target_fps = 24
                    fps = 1 if not self._render_times else 1.0 / np.mean(self._render_times)
                    if fps >= target_fps:
                        render_state = 1
                    else:
                        h = (fps * num_rays_total / target_fps / camera.aspect) ** 0.5
                        h = int(round(h, -1))
                        h = max(min(h_total, h), 30)
                        w = int(h * camera.aspect)
                        if w > w_total:
                            w = w_total
                            h = int(w / camera.aspect)
                        num_rays = w * h
                self._render_state[client.client_id] = render_state + 1
                cancellation_token = self._cancellation_token = None
                if render_state == 1:
                    cancellation_token = EventCancellationToken()
                    self._cancellation_token = cancellation_token

                c2w = get_c2w(camera)
                c2w = self._inv_transform @ c2w
                camera = Cameras(
                    poses=c2w[None, :3, :4],
                    intrinsics=np.array([[focal, focal, w_total / 2, h_total / 2]], dtype=np.float32),
                    camera_types=np.array([0], dtype=np.int32),
                    distortion_parameters=np.zeros((1, 8), dtype=np.float32),
                    image_sizes=np.array([[w, h]], dtype=np.int32),
                    nears_fars=None,
                )
                try:
                    with self._cancellation_token or contextlib.nullcontext():
                        for outputs in self.method.render(camera):
                            pass
                except CancelledException:
                    # if we got interrupted, don't send the output to the viewer
                    self._render_state[client.client_id] = 0
                    continue
                interval = perf_counter() - start
                image = outputs["color"]
                client.set_background_image(image, format="jpeg")
                self._render_state[client.client_id] = min(self._render_state.get(client.client_id, 0), render_state + 1)

                if render_state == 1 or len(self._render_times) < assert_not_none(self._render_times.maxlen):
                    self._render_times.append(interval / num_rays * num_rays_total)
                self.fps.value = f"{1.0 / np.mean(self._render_times):.3g}"


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
    poses = poses.copy()

    # Convert from OpenCV to OpenGL coordinate system
    poses[..., 0:3, 1:3] *= -1
    origins = poses[..., :3, 3]
    mean_origin = np.mean(origins, 0)
    translation = mean_origin
    up = np.mean(poses[:, :3, 1], 0)
    up = up / np.linalg.norm(up)

    rotation = rotation_matrix(up, np.array([0, 0, 1], dtype=up.dtype))
    transform = np.concatenate([rotation, rotation @ -translation[..., None]], -1)
    transform = np.concatenate([transform, np.array([[0, 0, 0, 1]], dtype=transform.dtype)], 0)
    return transform


def run_viser_viewer(method: Optional[Method] = None, 
                     data=None, 
                     max_num_views: Optional[int] = 100,
                     port=6006,
                     nb_info=None):
    def build_server(dataset_metadata=None, **kwargs):
        transform = initial_pose = None
        control_type = "default"
        if dataset_metadata is not None:
            transform = dataset_metadata.get("viewer_transform")
            initial_pose = dataset_metadata.get("viewer_initial_pose")
            control_type = "object-centric" if dataset_metadata.get("type") == "object-centric" else "default"
            initial_pose = apply_transform(invert_transform(transform), initial_pose)
        return ViserViewer(**kwargs, 
                           port=port, 
                           method=method,
                           transform=transform,
                           initial_pose=initial_pose,
                           control_type=control_type)

    if data is not None:
        features: FrozenSet[DatasetFeature] = frozenset({"color", "points3D_xyz"})
        train_dataset = load_dataset(data, split="test", features=features)
        server = build_server(dataset_metadata=train_dataset.metadata)

        if max_num_views is not None and len(train_dataset) > max_num_views:
            train_dataset = train_dataset[np.random.choice(len(train_dataset), 100)]

        server.add_initial_point_cloud(train_dataset.points3D_xyz, train_dataset.points3D_rgb)

        test_dataset = load_dataset(data, split="train", features=features)
        if max_num_views is not None and len(test_dataset) > max_num_views:
            test_dataset = test_dataset[np.random.choice(len(test_dataset), 100)]

        train_dataset.load_features(features)
        server.add_dataset_views(train_dataset, "train")

        test_dataset.load_features(features)
        server.add_dataset_views(test_dataset, "test")

    elif nb_info is not None:
        dataset_metadata = nb_info.get("dataset_metadata")
        server = build_server(dataset_metadata=dataset_metadata)
    else:
        server = build_server()
    server.run()
