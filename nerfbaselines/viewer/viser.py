import inspect
import dataclasses
from dataclasses import dataclass
from typing import Callable, Tuple, Any, TypeVar, TYPE_CHECKING
from pathlib import Path
from collections import deque
from time import perf_counter

import numpy as np
import viser
from viser import ViserServer
from viser import transforms as vtf

from ..types import Method, Dataset, Optional
from ..cameras import Cameras
from ..datasets._colmap_utils import qvec2rotmat, rotmat2qvec
from ..utils import CancellationToken, CancelledException, cancellable
from ..datasets import load_dataset
from .viser_render_panel import populate_render_tab


VISER_SCALE_RATIO = 10.0
T = TypeVar("T")


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


@dataclass(frozen=True, eq=True)
class ViewerState:
    resolution: int = 512
    background_color: Tuple[int, int, int] = (38, 42, 55)
    output_type: Optional[str] = None
    output_type_options: Tuple[str, ...] = ()
    composite_depth: bool = False

    output_split: bool = False
    split_percentage: float = 0.5
    show_train_cameras: bool = True
    show_test_cameras: bool = True
    fps: str = ""


if TYPE_CHECKING:
    _ViewerState = ViewerState
else:
    _ViewerState = object


class BindableState(_ViewerState):
    def __init__(self, initial_state, get_value=None, update_value=None):
        self._initial_state = initial_state
        self._get_value = get_value
        self._update_value = update_value

    def __getattr__(self, __name: str) -> Any:
        def _update_value(state, value):
            current_state = self.get_value(state)
            return self.update_value(state, dataclasses.replace(current_state, **{__name: value}))

        def _get_value(state):
            return getattr(self.get_value(state), __name)

        return BindableState(self._initial_state, _get_value, _update_value)

    def update_value(self, state, value):
        if self._update_value is None:
            return value
        return self._update_value(state, value)

    def get_value(self, state):
        if self._get_value is None:
            return state
        return self._get_value(state)

    def initial_value(self):
        return self.get_value(self._initial_state)

    def with_default(self, default):
        return BindableState(self._initial_state, lambda state: self.get_value(state) if self.get_value(state) is not None else default, lambda state, value: self.update_value(state, value))

    def map(self, fn):
        def set(*args, **kwargs):
            raise ValueError("Cannot update a mapped state")

        return BindableState(self._initial_state, lambda state: fn(self.get_value(state)), set)

    def __not__(self):
        return self.map(lambda x: not x)


class BindableViserServer(ViserServer):
    def __init__(self, server: ViserServer, update_state: Callable[[T], T], on_update_state: Callable[[Callable[[T], None]], None]):
        for name, value in inspect.getmembers(server):
            if name.startswith("add_gui_"):
                setattr(self, name, self._bindable_add_gui(value))
            elif not name.startswith("__"):
                setattr(self, name, value)
        self.update_state = update_state
        self._update_fn = []
        self.server = server
        on_update_state(self._set_new_state)

    def _bindable_add_gui(self, add_gui):
        signature = inspect.signature(add_gui)
        arg_names = list(signature.parameters.keys())

        def _add_gui(*args, **kwargs):
            prop_bindings = {}

            def map_arg(name, value):
                if isinstance(value, BindableState):
                    prop_bindings[name] = value
                    return value.initial_value()
                return value

            args = list(args)
            for i, arg in enumerate(args):
                args[i] = map_arg(arg_names[i], arg)
            for name, value in kwargs.items():
                kwargs[name] = map_arg(name, value)
            component = add_gui(*args, **kwargs)
            for name, binding in prop_bindings.items():
                if name == "initial_value":
                    name = "value"
                self._update_fn.append(lambda state: setattr(component, name, binding.get_value(state)))
                if name == "value":
                    component.on_update(lambda e: self.update_state(lambda x: binding.update_value(x, e.target.value)))
            return component

        return _add_gui

    def _set_new_state(self, state: T):
        for fn in self._update_fn:
            fn(state)


def add_control_panel(state: BindableState, server: BindableViserServer):
    with server.add_gui_folder("Render Options"):
        server.add_gui_slider(
            "Max res",
            64,
            2048,
            100,
            state.resolution,
            hint="Maximum resolution to render in viewport",
        )
        server.add_gui_dropdown(
            "Output type",
            state.output_type_options.map(lambda x: x or ("not set",)),
            hint="The output to render",
        )
        server.add_gui_checkbox(
            "Composite depth",
            state.composite_depth,
            hint="Allow NeRF to occlude 3D browser objects",
        )
        server.add_gui_rgb("Background color", state.background_color, hint="Color of the background")

    # split options
    with server.add_gui_folder("Split Screen"):
        server.add_gui_checkbox(
            "Enable",
            False,
            state.output_split,
            hint="Render two outputs",
        )
        server.add_gui_slider("Split percentage", initial_value=state.split_percentage, min=0.0, max=1.0, step=0.01, hint="Where to split")
        server.add_gui_dropdown(
            "Output render split",
            options=state.output_type_options.map(lambda x: x or ("not set",)),
            hint="The second output",
        )

    def _reset_camera_cb(_) -> None:
        for client in server.get_clients().values():
            client.camera.up_direction = vtf.SO3(client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

    server.add_gui_button(
        label="Reset Up Direction",
        icon=viser.Icon.ARROW_BIG_UP_LINES,
        color="gray",
        hint="Set the up direction of the camera orbit controls to the camera's current up direction.",
    ).on_click(_reset_camera_cb)


def add_gui(state: BindableState, server: BindableViserServer):
    buttons = (
        viser.theme.TitlebarButton(
            text="Getting Started",
            icon=None,
            href="https://nerf.studio",
        ),
        viser.theme.TitlebarButton(
            text="Github",
            icon="GitHub",
            href="https://github.com/nerfstudio-project/nerfstudio",
        ),
        viser.theme.TitlebarButton(
            text="Documentation",
            icon="Description",
            href="https://docs.nerf.studio",
        ),
    )
    image = viser.theme.TitlebarImage(
        image_url_light="https://docs.nerf.studio/_static/imgs/logo.png",
        image_url_dark="https://docs.nerf.studio/_static/imgs/logo-dark.png",
        image_alt="NerfStudio Logo",
        href="https://docs.nerf.studio/",
    )
    titlebar_theme = viser.theme.TitlebarConfig(buttons=buttons, image=image)
    server.configure_theme(
        titlebar_content=titlebar_theme,
        control_layout="collapsible",
        dark_mode=True,
        brand_color=(255, 211, 105),
    )

    def make_stats_markdown(state: ViewerState) -> str:
        # if either are None, read it from the current stats_markdown content
        step = 0
        res = state.resolution
        return f"Step: {step}  \nResolution: {res}"

    server.add_gui_markdown(state.map(make_stats_markdown))
    server.add_gui_text("FPS", initial_value=state.fps, disabled=True)

    tabs = server.add_gui_tab_group()
    with tabs.add_tab("Control", viser.Icon.SETTINGS):
        add_control_panel(state, server)

        # Add toggles to show/hide cameras
        server.add_gui_checkbox(label="Show train cams", value=state.show_train_cameras)
        server.add_gui_checkbox(label="Show test cams", value=state.show_test_cameras)

    with tabs.add_tab("Render", viser.Icon.CAMERA):
        populate_render_tab(server.server)


class ViserViewer:
    def __init__(self, method: Method, port, transform=None):
        self.port = port
        self.method = method
        self.transform = transform if transform is not None else np.eye(4, dtype=np.float32)
        self._inv_transform = np.linalg.inv(self.transform)

        self._render_state = {}
        self._viewer_state = ViewerState()
        self._update_state_callbacks = []
        self._train_camera_handles = []
        self._test_camera_handles = []

        self._render_times = deque(maxlen=3)
        self.server = viser.ViserServer(port=self.port)

        def _update_state(state: ViewerState) -> ViewerState:
            if self._viewer_state != state:
                if self._viewer_state.show_train_cameras != state.show_train_cameras:
                    for handle in self._train_camera_handles:
                        handle.visible = state.show_train_cameras
                if self._viewer_state.show_test_cameras != state.show_test_cameras:
                    for handle in self._test_camera_handles:
                        handle.visible = state.show_test_cameras
                self._viewer_state = state
                self._reset_render()

        add_gui(BindableState(self._viewer_state), BindableViserServer(self.server, lambda call: _update_state(call(self._viewer_state)), lambda fn: self._update_state_callbacks.append(fn)))

        self.c2ws = []
        self.camera_infos = []

        @self.server.on_client_connect
        def _(client: viser.ClientHandle):
            @client.camera.on_update
            def _(_camera_handle: viser.CameraHandle):
                self._render_state.pop(client.client_id, None)
                if self._cancellation_token is not None:
                    self._cancellation_token.cancel()

        self._cancellation_token = None

    def _notify_view_state_updated(self):
        for callback in self._update_state_callbacks:
            callback(self._viewer_state)

    def _reset_render(self):
        self._render_state = {}
        if self._cancellation_token is not None:
            self._cancellation_token.cancel()

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
            if split == "train":
                self._train_camera_handles.append(handle)
            elif split == "test":
                self._test_camera_handles.append(handle)

    def _update(self):
        for client in self.server.get_clients().values():
            render_state = self._render_state.get(client.client_id, 0)
            if render_state < 2:
                start = perf_counter()
                camera = client.camera

                resolution = self._viewer_state.resolution
                w_total = resolution
                h_total = int(resolution / camera.aspect)
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

                self._viewer_state = dataclasses.replace(self._viewer_state, output_type_options=tuple(outputs.keys()), fps=f"{1.0 / np.mean(self._render_times):.3g}")
                self._notify_view_state_updated()


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
        train_dataset = load_dataset(data, split="train", features=features)
        test_dataset = load_dataset(data, split="test", features=features)
        server = ViserViewer(method, port=port, transform=get_orientation_transform(train_dataset.cameras.poses))

        train_dataset.load_features(features)
        server.add_dataset_views(train_dataset, "train")
        test_dataset.load_features(features)
        server.add_dataset_views(test_dataset, "test")
    else:
        server = ViserViewer(method, port=port)
    server.run()
