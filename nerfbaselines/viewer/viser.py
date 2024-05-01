import inspect
from dataclasses import dataclass
import dataclasses
import contextlib
from pathlib import Path
from collections import deque
from time import perf_counter
from typing import Optional, Tuple, TYPE_CHECKING, Any, Callable

import numpy as np
import viser
from viser import ViserServer

from ..types import Method, Dataset, FrozenSet, DatasetFeature, Literal, TypeVar
from ..types import Cameras, new_cameras
from ..datasets import dataset_load_features, dataset_index_select
from ..datasets._colmap_utils import qvec2rotmat, rotmat2qvec
from ..utils import CancelledException, assert_not_none
from ..pose_utils import apply_transform, get_transform_and_scale, invert_transform
from ..datasets import load_dataset
from ..backends._rpc import EventCancellationToken
from .viser_render_panel import populate_render_tab
from ..utils import image_to_srgb, visualize_depth


ControlType = Literal["object-centric", "default"]
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
    split_output_type: Optional[str] = None

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
        server.add_gui_rgb("Background color", state.background_color, hint="Color of the background")

    # split options
    with server.add_gui_folder("Split Screen", visible=state.output_type_options.map(lambda x: len(x) > 1)):
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
            initial_value=state.split_output_type,
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
        server.add_gui_checkbox(label="Show train cams", initial_value=state.show_train_cameras)
        server.add_gui_checkbox(label="Show test cams", initial_value=state.show_test_cameras)

    with tabs.add_tab("Render", viser.Icon.CAMERA):
        populate_render_tab(server.server, Path("."), Path("."), state.is_time_enabled)


class ViserViewer:
    def __init__(self, 
                 method: Optional[Method], 
                 port, 
                 transform=None, 
                 initial_pose=None,
                 state=None,
                 control_type: ControlType = "default"):
        self.port = port
        self.method = method
        self.transform = transform if transform is not None else np.eye(4, dtype=np.float32)
        self._inv_transform = np.linalg.inv(self.transform)

        self._render_state = {}
        self._viewer_state = state or ViewerState()
        self._update_state_callbacks = []
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
        init_bg_image = np.array(self._viewer_state.background_color, dtype=np.uint8).reshape((1, 1, 3))
        self._initial_background_color = self._viewer_state.background_color
        self.server.set_background_image(init_bg_image)

        def set_viewer_state(update_state):
            old_state = self._viewer_state
            self._viewer_state = update_state(old_state)
            self._update_gui(old_state)

        bindable_state = BindableState(self._viewer_state)
        bindable_server = BindableViserServer(self.server, set_viewer_state, self._update_state_callbacks.append)
        add_gui(bindable_state, bindable_server)

        self._cancellation_token = None
        self._setup_gui()

        self._camera_frustrum_handles = {}

    def _reset_render(self):
        self._render_state = {}
        if self._cancellation_token is not None:
            self._cancellation_token.cancel()
            self._cancellation_token = None

    def _setup_gui(self):
        self.fps = self.server.add_gui_text("FPS", initial_value="-1", disabled=True)

    def _update_gui(self, old_state):
        state = self._viewer_state
        if state.show_train_cameras != old_state.show_train_cameras and "train" in self._camera_frustrum_handles:
            for handle in self._camera_frustrum_handles["train"]:
                handle.visible = state.show_train_cameras
        if state.show_test_cameras != old_state.show_test_cameras and "test" in self._camera_frustrum_handles:
            for handle in self._camera_frustrum_handles["test"]:
                handle.visible = state.show_test_cameras

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

        max_img_size = 128
        cam: Cameras
        self._camera_frustrum_handles[split] = self._camera_frustrum_handles.get(split, [])
        for i, (cam, path) in enumerate(zip(dataset["cameras"], dataset["file_paths"])):
            assert cam.image_sizes is not None, "dataset.image_sizes must be set"
            image = None
            if dataset["images"] is not None:
                image = dataset["images"][i]
            if str(path).startswith("/undistorted/"):
                path = str(path)[len("/undistorted/") :]
            else:
                path = str(Path(path).relative_to(Path(dataset.get("file_paths_root") or "")))
            c2w = apply_transform(self.transform, cam.poses)
            pos, quat = get_position_quaternion(c2w)
            W, H = cam.image_sizes.tolist()
            fy = cam.intrinsics[1]
            downsample_factor = max(1, min(W//max_img_size, H//max_img_size))
            image = image[::downsample_factor, ::downsample_factor]
            image = image_to_srgb(image, dtype=np.uint8, 
                                  color_space="srgb", 
                                  background_color=np.array(self._initial_background_color, dtype=np.uint8))
            handle = self.server.add_camera_frustum(
                f"/dataset-{split}/{path}/frustum",
                fov=2 * np.arctan2(H / 2, fy),
                aspect=W / H,
                scale=0.03,
                position=pos,
                wxyz=quat,
                image=image,
            )
            handle.on_click(_set_view_to_camera)
            self._camera_frustrum_handles[split].append(handle)

    def _update(self):
        if self.method is None:
            # No need to render anything
            return
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
                nb_camera = new_cameras(
                    poses=c2w[None, :3, :4],
                    intrinsics=np.array([[focal, focal, w_total / 2, h_total / 2]], dtype=np.float32),
                    camera_types=np.array([0], dtype=np.int32),
                    distortion_parameters=np.zeros((1, 8), dtype=np.float32),
                    image_sizes=np.array([[w, h]], dtype=np.int32),
                    nears_fars=None,
                )
                outputs = None
                try:
                    with self._cancellation_token or contextlib.nullcontext():
                        for outputs in self.method.render(nb_camera):
                            pass
                except CancelledException:
                    # if we got interrupted, don't send the output to the viewer
                    self._render_state[client.client_id] = 0
                    continue
                assert outputs is not None, "Method did not return any outputs"
                interval = perf_counter() - start

                def render_single(name):
                    name = name or "color"
                    image = outputs[name]
                    if name == "color":
                        bg_color = np.array(self._viewer_state.background_color, dtype=np.uint8)
                        render = image_to_srgb(image, np.uint8, color_space="srgb", allow_alpha=False, background_color=bg_color)
                    elif name == "depth":
                        # Blend depth with correct color pallete
                        render = visualize_depth(image)
                    else:
                        render = image
                    return render
                render = render_single(self._viewer_state.output_type)
                if self._viewer_state.output_split:
                    split_render = render_single(self._viewer_state.split_output_type)
                    assert render.shape == split_render.shape
                    split_point = int(render.shape[1] * self._viewer_state.split_percentage)
                    render[:, :split_point] = split_render[:, split_point:]
                    

                client.set_background_image(render, format="jpeg")
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
    state = ViewerState()
    if nb_info is not None:
        if nb_info.get("dataset_background_color") is not None:
            bg_color = nb_info["dataset_background_color"]
            bg_color = tuple(int(x) for x in bg_color)
            state = dataclasses.replace(state, background_color=bg_color)

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
        train_dataset = load_dataset(data, split="test", features=features, load_features=False)
        server = build_server(dataset_metadata=train_dataset["metadata"])

        # Get background color
        bg_color = train_dataset["metadata"].get("background_color", None)
        if bg_color is not None:
            bg_color = tuple(int(x) for x in bg_color)
            state = dataclasses.replace(state, background_color=bg_color)

        if max_num_views is not None and len(train_dataset) > max_num_views:
            train_dataset = dataset_index_select(train_dataset, np.random.choice(len(train_dataset), 100))

        if train_dataset.get("points3D_xyz") is not None:
            server.add_initial_point_cloud(train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])

        test_dataset = load_dataset(data, split="train", features=features, load_features=False)
        if max_num_views is not None and len(test_dataset) > max_num_views:
            test_dataset = dataset_index_select(test_dataset, np.random.choice(len(test_dataset), 100))

        server.add_dataset_views(dataset_load_features(train_dataset, features), "train")
        server.add_dataset_views(dataset_load_features(test_dataset, features), "test")

    elif nb_info is not None:
        dataset_metadata = nb_info.get("dataset_metadata")
        server = build_server(dataset_metadata=dataset_metadata)
    else:
        server = build_server()
    server.run()
