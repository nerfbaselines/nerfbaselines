import os
import inspect
from dataclasses import dataclass
import dataclasses
import contextlib
from pathlib import Path
from collections import deque
from time import perf_counter
from typing import Optional, Tuple, TYPE_CHECKING, Any, Callable, Dict, cast, List

import numpy as np
import viser
import viser.theme
import viser.transforms as vtf
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


class BindableSource:
    def __init__(self, getter, update, on_update):
        self.get = getter
        self.update = update
        self.on_update = on_update

    def __getattr__(self, __name: str) -> Any:
        last_value = object()

        def _get():
            nonlocal last_value
            value = self.get()
            if isinstance(value, dict):
                out = value[__name]
            out = getattr(value, __name)
            last_value = out
            return out

        def _update(value=None, **changes):
            nonlocal last_value
            if value is None:
                old = self.get()
                if isinstance(old, dict):
                    value = old.copy()
                    value.update(changes)
                elif isinstance(old, dataclasses):
                    value = dataclasses.replace(old, **changes)
                elif hasattr(old, "update"):
                    value = old.update(**changes)
                else:
                    raise ValueError("Cannot update value")
            else:
                assert not changes, "Cannot provide both value and changes"
            last_value = value
            self.update(**{__name: value})

        def _on_update(callback):
            nonlocal last_value
            def wrapped(state):
                nonlocal last_value
                new = state[__name] if isinstance(state, dict) else getattr(state, __name)
                if new == last_value:
                    return
                last_value = new
                return callback(new)
            self.on_update(wrapped)

        return BindableSource(_get, _update, _on_update)

    def with_default(self, default):
        def _get():
            value = self.get()
            return value if value is not None else default
        def _on_update(callback):
            self.on_update(lambda state: callback(state) if state is not None else callback(default))
        return BindableSource(_get, self.update, _on_update)

    def map(self: Any, fn):
        def _set(*args, **kwargs):
            raise ValueError("Cannot update a mapped state")
    
        def _on_update(callback):
            def wrapped(state):
                if fn(self.get()) != fn(state):
                    callback(fn(state))
            self.on_update(wrapped)
        return BindableSource(lambda: fn(self.get()), _set, _on_update)

    def __not__(self):
        out = self.map(lambda x: not x)
        def _set(value):
            self.update(value=not value)
        out.update = _set
        return out


@dataclass(eq=True)
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
    has_input_points: bool = False
    show_input_points: bool = True
    fps: str = ""

    _update_callbacks: List = dataclasses.field(default_factory=list)

    def get(self):
        return self

    def on_update(self, callback):
        self._update_callbacks.append(callback)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            object.__setattr__(self, key, value)
        for callback in self._update_callbacks:
            callback(self)

    def __setattr__(self, name, value):
        if hasattr(self, name) and getattr(self, name) == value:
            return
        self.update(**{name: value})

    @property
    def b(self) -> 'ViewerState':
        return cast(ViewerState, BindableSource(lambda: self, self.update, self.on_update))


class BindableViserServer(ViserServer):
    def __init__(self, server: ViserServer):
        for name, value in inspect.getmembers(server):
            if name.startswith("add_gui_"):
                setattr(self, name, self._bindable_add_gui(value))
            elif not name.startswith("__"):
                setattr(self, name, value)
        self.server = server

    def _bindable_add_gui(self, add_gui):
        signature = inspect.signature(add_gui)
        arg_names = list(signature.parameters.keys())

        def _add_gui(*args, **kwargs):
            prop_bindings: Dict[str, BindableSource] = {}
            for i, arg in enumerate(args):
                name = arg_names[i]
                if isinstance(arg, BindableSource):
                    prop_bindings[name] = arg
            for name, arg in kwargs.items():
                if isinstance(arg, BindableSource):
                    prop_bindings[name] = arg

            def _current_args():
                _args = list(args)
                _kwargs = kwargs.copy()
                for i in range(len(_args)):
                    name = arg_names[i]
                    if name in prop_bindings:
                        _args[i] = prop_bindings[name].get()
                for name in kwargs.keys():
                    if name in prop_bindings:
                        _kwargs[name] = prop_bindings[name].get()
                return _args, _kwargs

            handle = None

            def _build_component():
                nonlocal handle
                _args, _kwargs = _current_args()
                handle = add_gui(*_args, **_kwargs)

            _build_component()
            assert handle is not None, "Failed to build component"
            for name, binding in prop_bindings.items():
                if name == "initial_value":
                    name = "value"
                if hasattr(handle, name):
                    binding.on_update(lambda value: setattr(handle, name, value) if getattr(handle, name) != value else None)
                else:
                    binding.on_update(lambda _: _build_component())
                if name == "value":
                    handle.on_update(lambda e: binding.update(e.target.value))
            return handle

        return _add_gui


def add_control_panel(state: ViewerState, server: BindableViserServer):
    with server.add_gui_folder("Render Options"):
        server.add_gui_slider(
            "Max res",
            64,
            2048,
            100,
            state.b.resolution,
            hint="Maximum resolution to render in viewport",
        )
        server.add_gui_dropdown(
            "Output type",
            state.b.output_type_options.map(lambda x: x or ("not set",)),
            hint="The output to render",
        )
        server.add_gui_rgb("Background color", state.b.background_color, hint="Color of the background")

    # split options
    with server.add_gui_folder("Split Screen", visible=state.b.output_type_options.map(lambda x: len(x) > 1)):
        server.add_gui_checkbox(
            "Enable",
            False,
            state.b.output_split,
            hint="Render two outputs",
        )
        server.add_gui_slider("Split percentage", initial_value=state.b.split_percentage, min=0.0, max=1.0, step=0.01, hint="Where to split")
        server.add_gui_dropdown(
            "Output render split",
            options=state.b.output_type_options.map(lambda x: x or ("not set",)),
            initial_value=state.b.split_output_type,
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


def add_gui(state: ViewerState, server: BindableViserServer):
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

    server.add_gui_markdown(state.b.map(make_stats_markdown))
    server.add_gui_text("FPS", initial_value=state.b.fps, disabled=True)

    tabs = server.add_gui_tab_group()
    with tabs.add_tab("Control", viser.Icon.SETTINGS):
        add_control_panel(state, server)

        # Add toggles to show/hide cameras
        server.add_gui_checkbox(label="Show train cams", initial_value=state.b.show_train_cameras)
        server.add_gui_checkbox(label="Show test cams", initial_value=state.b.show_test_cameras)
        server.add_gui_checkbox(label="Show input PC", initial_value=state.b.show_input_points)

    with tabs.add_tab("Render", viser.Icon.CAMERA):
        populate_render_tab(server.server, Path("."), Path("."), state.b.is_time_enabled)


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
        self.server = BindableViserServer(viser.ViserServer(port=self.port))
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

        self._cancellation_token = None

        self._camera_frustrum_handles = {}
        self._render_context = {}
        self._build_gui()


    def _build_gui(self):
        add_gui(self._viewer_state, self.server)

        def _update_handles_visibility(split, visible):
            for handle in self._camera_frustrum_handles.get(split, []):
                handle.visible = visible
        self._viewer_state.b.show_train_cameras.on_update(lambda x: _update_handles_visibility("train", x))
        self._viewer_state.b.show_test_cameras.on_update(lambda x: _update_handles_visibility("test", x))
        self._viewer_state.on_update(lambda _: self._reset_render())

    def _update_render_context(self, context):
        self._render_context = context
        self._reset_render()

    def _reset_render(self):
        self._render_state = {}
        if self._cancellation_token is not None:
            self._cancellation_token.cancel()
            self._cancellation_token = None

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
        pc = self.server.add_point_cloud(
            "/initial-point-cloud",
            points=points,
            colors=colors,
            point_size=0.001,
            point_shape="circle",
        )
        self._viewer_state.b.show_input_points.on_update(lambda x: setattr(pc, "visible", x))
        self._viewer_state.b.has_input_points.update(True)

    def add_dataset_views(self, dataset: Dataset, split: str):
        def _set_view_to_camera(handle: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]):
            with handle.client.atomic():
                handle.client.camera.position = handle.target.position
                handle.client.camera.wxyz = handle.target.wxyz
            self._reset_render()

        if split == "train" and self.method is not None:
            try:
                if self.method.get_train_embedding(0) is not None:
                    options = ["none"] + [os.path.relpath(x, dataset.get("file_paths_root") or "") for x in dataset.get("file_paths")]
                    dropdown = self.server.add_gui_dropdown(
                        "Train embedding images",
                        options,
                        "none",
                        hint="Select images to visualize embeddings for",
                    )
                    @dropdown.on_update
                    def _(_):
                        selected_image = dropdown.value
                        train_id = options.index(selected_image)
                        self._update_render_context({
                            "embedding": self.method.get_train_embedding(train_id-1) if train_id > 0 else None,
                            "embedding_train_index": train_id-1 if train_id > 0 else None,
                        })
            except (AttributeError, NotImplementedError):
                pass
        return

        max_img_size = 64
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
            if image is not None:
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
                    intrinsics=np.array([[focal * w/w_total, focal* h/h_total, w / 2, h / 2]], dtype=np.float32),
                    camera_types=np.array([0], dtype=np.int32),
                    distortion_parameters=np.zeros((1, 8), dtype=np.float32),
                    image_sizes=np.array([[w, h]], dtype=np.int32),
                    nears_fars=None,
                )
                outputs = None
                try:
                    with self._cancellation_token or contextlib.nullcontext():
                        for outputs in self.method.render(nb_camera, embeddings=[self._render_context.get("embedding")]):
                            pass
                except CancelledException:
                    # if we got interrupted, don't send the output to the viewer
                    self._render_state[client.client_id] = 0
                    continue
                assert outputs is not None, "Method did not return any outputs"
                interval = perf_counter() - start

                def render_single(name):
                    assert outputs is not None, "Method did not return any outputs"
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
                self._viewer_state.fps = f"{1.0 / np.mean(self._render_times):.3g}"


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

        # if max_num_views is not None and len(train_dataset) > max_num_views:
        #     train_dataset = dataset_index_select(train_dataset, np.random.choice(len(train_dataset), 100))

        if train_dataset.get("points3D_xyz") is not None:
            server.add_initial_point_cloud(train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])

        test_dataset = load_dataset(data, split="train", features=features, load_features=False)
        # if max_num_views is not None and len(test_dataset) > max_num_views:
        #     test_dataset = dataset_index_select(test_dataset, np.random.choice(len(test_dataset), 100))

        server.add_dataset_views(dataset_load_features(test_dataset, features), "test")
        server.add_dataset_views(dataset_load_features(train_dataset, features), "train")

    elif nb_info is not None:
        dataset_metadata = nb_info.get("dataset_metadata")
        server = build_server(dataset_metadata=dataset_metadata)
    else:
        server = build_server()
    server.run()
