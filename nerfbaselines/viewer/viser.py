from pathlib import Path
from collections import deque
from time import perf_counter

import numpy as np
import viser

from ..types import Method, Dataset
from ..cameras import Cameras
from ..datasets._colmap_utils import qvec2rotmat, rotmat2qvec
from ..utils import CancellationToken, CancelledException, cancellable
from ..datasets import load_dataset


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


class ViserViewer:
    def __init__(self, method: Method, port, transform=None):
        self.port = port
        self.method = method
        self.transform = transform if transform is not None else np.eye(4, dtype=np.float32)
        self._inv_transform = np.linalg.inv(self.transform)

        self._render_state = {}

        self._render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)

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

        self._cancellation_token = None
        self._setup_gui()

    def _reset_render(self):
        self._render_state = {}
        if self._cancellation_token is not None:
            self._cancellation_token.cancel()

    def _setup_gui(self):
        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

    def run(self):
        while True:
            self._update()

    def add_dataset_views(self, dataset: Dataset, split: str):
        def _set_view_to_camera(handle: viser.SceneNodePointerEvent[viser.ImageHandle]):
            with handle.client.atomic():
                handle.client.camera.position = handle.target.position
                handle.client.camera.wxyz = handle.target.wxyz
            self._reset_render()

        downsample_factor = 2
        for i, (cam, image, path) in enumerate(zip(dataset.cameras, dataset.images, dataset.file_paths)):
            if str(path).startswith("/undistorted/"):
                path = str(path)[len("/undistorted/") :]
            else:
                path = str(Path(path).relative_to(Path(dataset.file_paths_root)))
            c2w = cam.poses
            assert len(c2w.shape) == 2
            if c2w.shape[0] == 3:
                c2w = np.concatenate([c2w, np.eye(4, dtype=np.float32)[-1:]], 0)
            c2w = self.transform @ c2w
            c2w[0:3, 1:3] *= -1
            pos, quat = get_position_quaternion(c2w)
            W, H = cam.image_sizes.tolist()
            fy = cam.intrinsics[1]
            handle = self.server.add_camera_frustum(
                f"/dataset-{split}/{path}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.15,
                position=pos,
                wxyz=quat,
                image=image[::downsample_factor, ::downsample_factor],
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
                    cancellation_token = CancellationToken()
                    self._cancellation_token = cancellation_token

                c2w = get_c2w(camera)
                c2w[0:3, 1:3] *= -1
                c2w = self._inv_transform @ c2w
                camera = Cameras(
                    poses=c2w[None, :3, :4],
                    normalized_intrinsics=np.array([[focal, focal, w_total / 2, h_total / 2]], dtype=np.float32) / w_total,
                    camera_types=np.array([0], dtype=np.int32),
                    distortion_parameters=np.zeros((1, 8), dtype=np.float32),
                    image_sizes=np.array([[w, h]], dtype=np.int32),
                    nears_fars=None,
                )
                try:
                    outputs = next(iter(cancellable(self.method.render)(camera, cancellation_token=self._cancellation_token)))
                except CancelledException:
                    # if we got interrupted, don't send the output to the viewer
                    self._render_state[client.client_id] = 0
                    continue
                interval = perf_counter() - start
                image = outputs["color"]
                client.set_background_image(image, format="jpeg")
                self._render_state[client.client_id] = min(self._render_state.get(client.client_id, 0), render_state + 1)

                if render_state == 1 or len(self._render_times) < self._render_times.maxlen:
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
    origins = poses[..., :3, 3]
    mean_origin = np.mean(origins, 0)
    translation = mean_origin
    up = np.mean(poses[:, :3, 1], 0)
    up = up / np.linalg.norm(up)

    rotation = rotation_matrix(up, np.array([0, 0, 1], dtype=up.dtype))
    transform = np.concatenate([rotation, rotation @ -translation[..., None]], -1)
    transform = np.concatenate([transform, np.array([[0, 0, 0, 1]], dtype=transform.dtype)], 0)
    return transform


def run_viser_viewer(method: Method, data, port=6006):
    if data is not None:
        features = frozenset({"color"})
        train_dataset = load_dataset(data, split="test", features=features)
        test_dataset = load_dataset(data, split="train", features=features)
        server = ViserViewer(method, port=port, transform=get_orientation_transform(train_dataset.cameras.poses))

        train_dataset.load_features(features)
        server.add_dataset_views(train_dataset, "train")
        test_dataset.load_features(features)
        server.add_dataset_views(test_dataset, "test")
    else:
        server = ViserViewer(method, port=port)
    server.run()
