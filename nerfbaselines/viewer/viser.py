import os
import inspect
from dataclasses import dataclass
import dataclasses
import contextlib
from pathlib import Path
from collections import deque
from time import perf_counter
from typing import Optional, Tuple, Any, Dict, cast, List

import numpy as np
import viser
import viser.theme
import viser.transforms as vtf
from viser import ViserServer
import colorsys
import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import splines
import splines.quaternion
import viser
import viser.transforms as tf
from ..pose_utils import apply_transform, get_transform_and_scale, invert_transform
from scipy import interpolate

from ..types import Method, Dataset, FrozenSet, DatasetFeature, Literal, TypeVar
from ..types import new_cameras
from ..datasets import dataset_load_features, dataset_index_select
from ..datasets._colmap_utils import qvec2rotmat, rotmat2qvec
from ..utils import CancelledException, assert_not_none
from ..pose_utils import apply_transform, get_transform_and_scale, invert_transform, pad_poses
from ..datasets import load_dataset
from ..backends._rpc import EventCancellationToken
from ..utils import image_to_srgb, visualize_depth


ControlType = Literal["object-centric", "default"]
VISER_SCALE_RATIO = 10.0
T = TypeVar("T")


def transform_points(transform, points):
    transform, scale = get_transform_and_scale(transform)
    points = np.concatenate([points, np.ones((len(points), 1))], -1) @ transform.T
    points = points[..., :-1] / points[..., -1:]
    points *= scale
    return points


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
                if isinstance(new, np.ndarray):
                    if id(last_value) == id(new):
                        return
                elif new == last_value:
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


def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    See:
        https://github.com/nerfstudio-project/nerfstudio/blob/1aba4ea7a29b05e86f5d223245a573e7dcd86caa/nerfstudio/viewer_legacy/server/utils.py#L52
    """
    if fov is None:
        print("Warning: fov is None, using default value")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length


@dataclasses.dataclass
class Keyframe:
    position: np.ndarray
    wxyz: np.ndarray
    override_fov_enabled: bool
    override_fov_rad: float
    override_time_enabled: bool
    override_time_val: float
    aspect: float
    override_transition_enabled: bool
    override_transition_sec: Optional[float]

    @staticmethod
    def from_camera(camera: viser.CameraHandle, aspect: float) -> "Keyframe":
        return Keyframe(
            camera.position,
            camera.wxyz,
            override_fov_enabled=False,
            override_fov_rad=camera.fov,
            override_time_enabled=False,
            override_time_val=0.0,
            aspect=aspect,
            override_transition_enabled=False,
            override_transition_sec=None,
        )


class CameraPath:
    def __init__(
        self, server: viser.ViserServer, duration_element: viser.GuiInputHandle[float], time_enabled: bool = False,
        transform=None,
    ):
        self._server = server
        self._keyframes: Dict[int, Tuple[Keyframe, viser.CameraFrustumHandle]] = {}
        self._keyframe_counter: int = 0
        self._spline_nodes: List[viser.SceneNodeHandle] = []
        self._camera_edit_panel: Optional[viser.Gui3dContainerHandle] = None

        self._orientation_spline: Optional[splines.quaternion.KochanekBartels] = None
        self._position_spline: Optional[splines.KochanekBartels] = None
        self._fov_spline: Optional[splines.KochanekBartels] = None
        self._keyframes_visible: bool = True

        self._duration_element = duration_element

        # These parameters should be overridden externally.
        self.loop: bool = False
        self.framerate: float = 30.0
        self.tension: float = 0.5  # Tension / alpha term.
        self.default_fov: float = 0.0
        self.time_enabled = time_enabled
        self.default_render_time: float = 0.0
        self.default_transition_sec: float = 0.0
        self.show_spline: bool = True
        self.transform = transform

    def set_keyframes_visible(self, visible: bool) -> None:
        self._keyframes_visible = visible
        for keyframe in self._keyframes.values():
            keyframe[1].visible = visible

    def add_camera(self, keyframe: Keyframe, keyframe_index: Optional[int] = None) -> None:
        """Add a new camera, or replace an old one if `keyframe_index` is passed in."""
        server = self._server

        # Add a keyframe if we aren't replacing an existing one.
        if keyframe_index is None:
            keyframe_index = self._keyframe_counter
            self._keyframe_counter += 1

        frustum_handle = server.add_camera_frustum(
            f"/render_cameras/{keyframe_index}",
            fov=keyframe.override_fov_rad if keyframe.override_fov_enabled else self.default_fov,
            aspect=keyframe.aspect,
            scale=0.1,
            color=(200, 10, 30),
            wxyz=keyframe.wxyz,
            position=keyframe.position,
            visible=self._keyframes_visible,
        )
        self._server.add_icosphere(
            f"/render_cameras/{keyframe_index}/sphere",
            radius=0.03,
            color=(200, 10, 30),
        )

        @frustum_handle.on_click
        def _(_) -> None:
            if self._camera_edit_panel is not None:
                self._camera_edit_panel.remove()
                self._camera_edit_panel = None

            with server.add_3d_gui_container(
                "/camera_edit_panel",
                position=keyframe.position,
            ) as camera_edit_panel:
                self._camera_edit_panel = camera_edit_panel
                override_fov = server.add_gui_checkbox("Override FOV", initial_value=keyframe.override_fov_enabled)
                override_fov_degrees = server.add_gui_slider(
                    "Override FOV (degrees)",
                    5.0,
                    175.0,
                    step=0.1,
                    initial_value=keyframe.override_fov_rad * 180.0 / np.pi,
                    disabled=not keyframe.override_fov_enabled,
                )
                if self.time_enabled:
                    override_time = server.add_gui_checkbox(
                        "Override Time", initial_value=keyframe.override_time_enabled
                    )
                    override_time_val = server.add_gui_slider(
                        "Override Time",
                        0.0,
                        1.0,
                        step=0.01,
                        initial_value=keyframe.override_time_val,
                        disabled=not keyframe.override_time_enabled,
                    )

                    @override_time.on_update
                    def _(_) -> None:
                        keyframe.override_time_enabled = override_time.value
                        override_time_val.disabled = not override_time.value
                        self.add_camera(keyframe, keyframe_index)

                    @override_time_val.on_update
                    def _(_) -> None:
                        keyframe.override_time_val = override_time_val.value
                        self.add_camera(keyframe, keyframe_index)

                delete_button = server.add_gui_button("Delete", color="red", icon=viser.Icon.TRASH)
                go_to_button = server.add_gui_button("Go to")
                close_button = server.add_gui_button("Close")

            @override_fov.on_update
            def _(_) -> None:
                keyframe.override_fov_enabled = override_fov.value
                override_fov_degrees.disabled = not override_fov.value
                self.add_camera(keyframe, keyframe_index)

            @override_fov_degrees.on_update
            def _(_) -> None:
                keyframe.override_fov_rad = override_fov_degrees.value / 180.0 * np.pi
                self.add_camera(keyframe, keyframe_index)

            @delete_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                with event.client.add_gui_modal("Confirm") as modal:
                    event.client.add_gui_markdown("Delete keyframe?")
                    confirm_button = event.client.add_gui_button("Yes", color="red", icon=viser.Icon.TRASH)
                    exit_button = event.client.add_gui_button("Cancel")

                    @confirm_button.on_click
                    def _(_) -> None:
                        assert camera_edit_panel is not None

                        keyframe_id = None
                        for i, keyframe_tuple in self._keyframes.items():
                            if keyframe_tuple[1] is frustum_handle:
                                keyframe_id = i
                                break
                        assert keyframe_id is not None

                        self._keyframes.pop(keyframe_id)
                        frustum_handle.remove()
                        camera_edit_panel.remove()
                        self._camera_edit_panel = None
                        modal.close()
                        self.update_spline()

                    @exit_button.on_click
                    def _(_) -> None:
                        modal.close()

            @go_to_button.on_click
            def _(event: viser.GuiEvent) -> None:
                assert event.client is not None
                client = event.client
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )
                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(keyframe.wxyz), keyframe.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(10):
                    T_world_set = T_world_current @ tf.SE3.exp(T_current_target.log() * j / 9.0)

                    # Important bit: we atomically set both the orientation and the position
                    # of the camera.
                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                    time.sleep(1.0 / 30.0)

            @close_button.on_click
            def _(_) -> None:
                assert camera_edit_panel is not None
                camera_edit_panel.remove()
                self._camera_edit_panel = None

        self._keyframes[keyframe_index] = (keyframe, frustum_handle)

    def update_aspect(self, aspect: float) -> None:
        for keyframe_index, frame in self._keyframes.items():
            frame = dataclasses.replace(frame[0], aspect=aspect)
            self.add_camera(frame, keyframe_index=keyframe_index)

    def get_aspect(self) -> float:
        """Get W/H aspect ratio, which is shared across all keyframes."""
        assert len(self._keyframes) > 0
        return next(iter(self._keyframes.values()))[0].aspect

    def reset(self) -> None:
        for frame in self._keyframes.values():
            frame[1].remove()
        self._keyframes.clear()
        self.update_spline()

    def spline_t_from_t_sec(self, time: np.ndarray) -> np.ndarray:
        """From a time value in seconds, compute a t value for our geometric
        spline interpolation. An increment of 1 for the latter will move the
        camera forward by one keyframe.

        We use a PCHIP spline here to guarantee monotonicity.
        """
        transition_times_cumsum = self.compute_transition_times_cumsum()
        spline_indices = np.arange(transition_times_cumsum.shape[0])

        if self.loop:
            # In the case of a loop, we pad the spline to match the start/end
            # slopes.
            interpolator = interpolate.PchipInterpolator(
                x=np.concatenate(
                    [
                        [-(transition_times_cumsum[-1] - transition_times_cumsum[-2])],
                        transition_times_cumsum,
                        transition_times_cumsum[-1:] + transition_times_cumsum[1:2],
                    ],
                    axis=0,
                ),
                y=np.concatenate([[-1], spline_indices, [spline_indices[-1] + 1]], axis=0),
            )
        else:
            interpolator = interpolate.PchipInterpolator(x=transition_times_cumsum, y=spline_indices)

        # Clip to account for floating point error.
        return np.clip(interpolator(time), 0, spline_indices[-1])

    def interpolate_pose_and_fov_rad(
        self, normalized_t: float
    ) -> Optional[Union[Tuple[tf.SE3, float], Tuple[tf.SE3, float, float]]]:
        if len(self._keyframes) < 2:
            return None

        self._fov_spline = splines.KochanekBartels(
            [
                keyframe[0].override_fov_rad if keyframe[0].override_fov_enabled else self.default_fov
                for keyframe in self._keyframes.values()
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        self._time_spline = splines.KochanekBartels(
            [
                keyframe[0].override_time_val if keyframe[0].override_time_enabled else self.default_render_time
                for keyframe in self._keyframes.values()
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        assert self._orientation_spline is not None
        assert self._position_spline is not None
        assert self._fov_spline is not None
        if self.time_enabled:
            assert self._time_spline is not None
        max_t = self.compute_duration()
        t = max_t * normalized_t
        spline_t = float(self.spline_t_from_t_sec(np.array(t)))

        quat = self._orientation_spline.evaluate(spline_t)
        assert isinstance(quat, splines.quaternion.UnitQuaternion)
        if self.time_enabled:
            return (
                tf.SE3.from_rotation_and_translation(
                    tf.SO3(np.array([quat.scalar, *quat.vector])),
                    self._position_spline.evaluate(spline_t),
                ),
                float(self._fov_spline.evaluate(spline_t)),
                float(self._time_spline.evaluate(spline_t)),
            )
        else:
            return (
                tf.SE3.from_rotation_and_translation(
                    tf.SO3(np.array([quat.scalar, *quat.vector])),
                    self._position_spline.evaluate(spline_t),
                ),
                float(self._fov_spline.evaluate(spline_t)),
            )

    def update_spline(self) -> None:
        num_frames = int(self.compute_duration() * self.framerate)
        keyframes = list(self._keyframes.values())

        if num_frames <= 0 or not self.show_spline or len(keyframes) < 2:
            for node in self._spline_nodes:
                node.remove()
            self._spline_nodes.clear()
            return

        transition_times_cumsum = self.compute_transition_times_cumsum()

        self._orientation_spline = splines.quaternion.KochanekBartels(
            [
                splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(keyframe[0].wxyz, shift=-1))
                for keyframe in keyframes
            ],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )
        self._position_spline = splines.KochanekBartels(
            [keyframe[0].position for keyframe in keyframes],
            tcb=(self.tension, 0.0, 0.0),
            endconditions="closed" if self.loop else "natural",
        )

        # Update visualized spline.
        points_array = self._position_spline.evaluate(
            self.spline_t_from_t_sec(np.linspace(0, transition_times_cumsum[-1], num_frames))
        )
        colors_array = np.array([colorsys.hls_to_rgb(h, 0.5, 1.0) for h in np.linspace(0.0, 1.0, len(points_array))])

        # Clear prior spline nodes.
        for node in self._spline_nodes:
            node.remove()
        self._spline_nodes.clear()

        self._spline_nodes.append(
            self._server.add_spline_catmull_rom(
                "/render_camera_spline",
                positions=points_array,
                color=(220, 220, 220),
                closed=self.loop,
                line_width=1.0,
                segments=points_array.shape[0] + 1,
            )
        )
        self._spline_nodes.append(
            self._server.add_point_cloud(
                "/render_camera_spline/points",
                points=points_array,
                colors=colors_array,
                point_size=0.04,
            )
        )

        def make_transition_handle(i: int) -> None:
            assert self._position_spline is not None
            transition_pos = self._position_spline.evaluate(
                float(
                    self.spline_t_from_t_sec(
                        (transition_times_cumsum[i] + transition_times_cumsum[i + 1]) / 2.0,
                    )
                )
            )
            transition_sphere = self._server.add_icosphere(
                f"/render_camera_spline/transition_{i}",
                radius=0.04,
                color=(255, 0, 0),
                position=transition_pos,
            )
            self._spline_nodes.append(transition_sphere)

            @transition_sphere.on_click
            def _(_) -> None:
                server = self._server

                if self._camera_edit_panel is not None:
                    self._camera_edit_panel.remove()
                    self._camera_edit_panel = None

                keyframe_index = (i + 1) % len(self._keyframes)
                keyframe = keyframes[keyframe_index][0]

                with server.add_3d_gui_container(
                    "/camera_edit_panel",
                    position=transition_pos,
                ) as camera_edit_panel:
                    self._camera_edit_panel = camera_edit_panel
                    override_transition_enabled = server.add_gui_checkbox(
                        "Override transition",
                        initial_value=keyframe.override_transition_enabled,
                    )
                    override_transition_sec = server.add_gui_number(
                        "Override transition (sec)",
                        initial_value=keyframe.override_transition_sec
                        if keyframe.override_transition_sec is not None
                        else self.default_transition_sec,
                        min=0.001,
                        max=30.0,
                        step=0.001,
                        disabled=not override_transition_enabled.value,
                    )
                    close_button = server.add_gui_button("Close")

                @override_transition_enabled.on_update
                def _(_) -> None:
                    keyframe.override_transition_enabled = override_transition_enabled.value
                    override_transition_sec.disabled = not override_transition_enabled.value
                    self._duration_element.value = self.compute_duration()

                @override_transition_sec.on_update
                def _(_) -> None:
                    keyframe.override_transition_sec = override_transition_sec.value
                    self._duration_element.value = self.compute_duration()

                @close_button.on_click
                def _(_) -> None:
                    assert camera_edit_panel is not None
                    camera_edit_panel.remove()
                    self._camera_edit_panel = None

        (num_transitions_plus_1,) = transition_times_cumsum.shape
        for i in range(num_transitions_plus_1 - 1):
            make_transition_handle(i)

        # for i in range(transition_times.shape[0])

    def compute_duration(self) -> float:
        """Compute the total duration of the trajectory."""
        total = 0.0
        for i, (keyframe, frustum) in enumerate(self._keyframes.values()):
            if i == 0 and not self.loop:
                continue
            del frustum
            total += (
                keyframe.override_transition_sec
                if keyframe.override_transition_enabled and keyframe.override_transition_sec is not None
                else self.default_transition_sec
            )
        return total

    def compute_transition_times_cumsum(self) -> np.ndarray:
        """Compute the total duration of the trajectory."""
        total = 0.0
        out = [0.0]
        for i, (keyframe, frustum) in enumerate(self._keyframes.values()):
            if i == 0:
                continue
            del frustum
            total += (
                keyframe.override_transition_sec
                if keyframe.override_transition_enabled and keyframe.override_transition_sec is not None
                else self.default_transition_sec
            )
            out.append(total)

        if self.loop:
            keyframe = next(iter(self._keyframes.values()))[0]
            total += (
                keyframe.override_transition_sec
                if keyframe.override_transition_enabled and keyframe.override_transition_sec is not None
                else self.default_transition_sec
            )
            out.append(total)

        return np.array(out)


@dataclass(eq=True)
class ViewerState:
    resolution: int = 512
    background_color: Tuple[int, int, int] = (38, 42, 55)
    output_type: Optional[str] = None
    output_type_options: Tuple[str, ...] = ()
    composite_depth: bool = False
    is_time_enabled: bool = False

    output_split: bool = False
    split_percentage: float = 0.5
    split_output_type: Optional[str] = None

    show_train_cameras: bool = False
    show_test_cameras: bool = False
    show_input_points: bool = True
    fps: str = ""

    preview_render: bool = False
    preview_fov: float = 0.0
    preview_time: float = 0.0
    preview_aspect: float = 1.0

    input_points: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None
    camera_frustums_train: Optional[Any] = None
    camera_frustums_test: Optional[Any] = None

    _update_callbacks: List = dataclasses.field(default_factory=list)

    def get(self):
        return self

    def on_update(self, callback):
        self._update_callbacks.append(callback)

    def update(self, **kwargs):
        has_change = False
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, np.ndarray):
                    if id(value) != id(getattr(self, key)):
                        has_change = True
                elif value != getattr(self, key):
                    has_change = True
            object.__setattr__(self, key, value)
        if has_change:
            for callback in self._update_callbacks:
                callback(self)

    def __setattr__(self, name, value):
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


class ViserViewer:
    def __init__(self, 
                 method: Optional[Method], 
                 port, 
                 dataset_metadata=None,
                 state=None):

        self.transform = self._inv_transform = np.eye(4, dtype=np.float32)
        self.initial_pose = None

        control_type = "default"
        if dataset_metadata is not None:
            self.transform = dataset_metadata.get("viewer_transform").copy()
            self.initial_pose = dataset_metadata.get("viewer_initial_pose").copy()
            control_type = "object-centric" if dataset_metadata.get("type") == "object-centric" else "default"
            self.initial_pose[:3, 3] *= VISER_SCALE_RATIO

        self.transform[:3, :] *= VISER_SCALE_RATIO
        self._inv_transform = invert_transform(self.transform, True)

        self.port = port
        self.method = method

        self._render_state = {}
        self.state = state or ViewerState()

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

            if self.initial_pose is not None:
                pos, quat = get_position_quaternion(
                    self.initial_pose,
                )
                client.camera.position = pos
                client.camera.wxyz = quat
                client.camera.up_direction = np.array([0, 0, 1], dtype=np.float32)
                if control_type == "object-centric":
                    # For object-centric scenes, we look at the origin
                    client.camera.look_at = np.array([0, 0, 0], dtype=np.float32)

        self._cancellation_token = None
        init_bg_image = np.array(self.state.background_color, dtype=np.uint8).reshape((1, 1, 3))
        self._initial_background_color = self.state.background_color
        self.server.set_background_image(init_bg_image)

        self._cancellation_token = None

        self._camera_frustrum_handles = {}
        self._render_context = {}
        self._build_gui()
        self._add_state_handlers()

    def _build_render_tab(self):
        server = self.server

        fov_degrees = server.add_gui_slider(
            "Default FOV",
            initial_value=75.0,
            min=0.1,
            max=175.0,
            step=0.01,
            hint="Field-of-view for rendering, which can also be overridden on a per-keyframe basis.",
        )

        render_time = None
        if self.state.is_time_enabled:
            render_time = server.add_gui_slider(
                "Default Time",
                initial_value=0.0,
                min=0.0,
                max=1.0,
                step=0.01,
                hint="Rendering time step, which can also be overridden on a per-keyframe basis.",
            )

            @render_time.on_update
            def _(_) -> None:
                camera_path.default_render_time = render_time.value

        @fov_degrees.on_update
        def _(_) -> None:
            fov_radians = fov_degrees.value / 180.0 * np.pi
            for client in server.get_clients().values():
                client.camera.fov = fov_radians
            camera_path.default_fov = fov_radians

            # Updating the aspect ratio will also re-render the camera frustums.
            # Could rethink this.
            camera_path.update_aspect(resolution.value[0] / resolution.value[1])
            compute_and_update_preview_camera_state()

        resolution = server.add_gui_vector2(
            "Resolution",
            initial_value=(1920, 1080),
            min=(50, 50),
            max=(10_000, 10_000),
            step=1,
            hint="Render output resolution in pixels.",
        )

        @resolution.on_update
        def _(_) -> None:
            camera_path.update_aspect(resolution.value[0] / resolution.value[1])
            compute_and_update_preview_camera_state()

        add_button = server.add_gui_button(
            "Add Keyframe",
            icon=viser.Icon.PLUS,
            hint="Add a new keyframe at the current pose.",
        )

        @add_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            camera = server.get_clients()[event.client_id].camera

            # Add this camera to the path.
            camera_path.add_camera(
                Keyframe.from_camera(
                    camera,
                    aspect=resolution.value[0] / resolution.value[1],
                ),
            )
            duration_number.value = camera_path.compute_duration()
            camera_path.update_spline()

        clear_keyframes_button = server.add_gui_button(
            "Clear Keyframes",
            icon=viser.Icon.TRASH,
            hint="Remove all keyframes from the render path.",
        )

        @clear_keyframes_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            client = server.get_clients()[event.client_id]
            with client.atomic(), client.add_gui_modal("Confirm") as modal:
                client.add_gui_markdown("Clear all keyframes?")
                confirm_button = client.add_gui_button("Yes", color="red", icon=viser.Icon.TRASH)
                exit_button = client.add_gui_button("Cancel")

                @confirm_button.on_click
                def _(_) -> None:
                    camera_path.reset()
                    modal.close()

                    duration_number.value = camera_path.compute_duration()

                    # Clear move handles.
                    if len(transform_controls) > 0:
                        for t in transform_controls:
                            t.remove()
                        transform_controls.clear()
                        return

                @exit_button.on_click
                def _(_) -> None:
                    modal.close()

        loop = server.add_gui_checkbox("Loop", False, hint="Add a segment between the first and last keyframes.")

        @loop.on_update
        def _(_) -> None:
            camera_path.loop = loop.value
            duration_number.value = camera_path.compute_duration()

        tension_slider = server.add_gui_slider(
            "Spline tension",
            min=0.0,
            max=1.0,
            initial_value=0.0,
            step=0.01,
            hint="Tension parameter for adjusting smoothness of spline interpolation.",
        )

        @tension_slider.on_update
        def _(_) -> None:
            camera_path.tension = tension_slider.value
            camera_path.update_spline()

        move_checkbox = server.add_gui_checkbox(
            "Move keyframes",
            initial_value=False,
            hint="Toggle move handles for keyframes in the scene.",
        )

        transform_controls: List[viser.SceneNodeHandle] = []

        @move_checkbox.on_update
        def _(event: viser.GuiEvent) -> None:
            # Clear move handles when toggled off.
            if move_checkbox.value is False:
                for t in transform_controls:
                    t.remove()
                transform_controls.clear()
                return

            def _make_transform_controls_callback(
                keyframe: Tuple[Keyframe, viser.SceneNodeHandle],
                controls: viser.TransformControlsHandle,
            ) -> None:
                @controls.on_update
                def _(_) -> None:
                    keyframe[0].wxyz = controls.wxyz
                    keyframe[0].position = controls.position

                    keyframe[1].wxyz = controls.wxyz
                    keyframe[1].position = controls.position

                    camera_path.update_spline()

            # Show move handles.
            assert event.client is not None
            for keyframe_index, keyframe in camera_path._keyframes.items():
                controls = event.client.add_transform_controls(
                    f"/keyframe_move/{keyframe_index}",
                    scale=0.4,
                    wxyz=keyframe[0].wxyz,
                    position=keyframe[0].position,
                )
                transform_controls.append(controls)
                _make_transform_controls_callback(keyframe, controls)

        show_keyframe_checkbox = server.add_gui_checkbox(
            "Show keyframes",
            initial_value=True,
            hint="Show keyframes in the scene.",
        )

        @show_keyframe_checkbox.on_update
        def _(_: viser.GuiEvent) -> None:
            camera_path.set_keyframes_visible(show_keyframe_checkbox.value)

        show_spline_checkbox = server.add_gui_checkbox(
            "Show spline",
            initial_value=True,
            hint="Show camera path spline in the scene.",
        )

        @show_spline_checkbox.on_update
        def _(_) -> None:
            camera_path.show_spline = show_spline_checkbox.value
            camera_path.update_spline()

        playback_folder = server.add_gui_folder("Playback")
        with playback_folder:
            play_button = server.add_gui_button("Play", icon=viser.Icon.PLAYER_PLAY)
            pause_button = server.add_gui_button("Pause", icon=viser.Icon.PLAYER_PAUSE, visible=False)
            preview_render_button = server.add_gui_button(
                "Preview Render", hint="Show a preview of the render in the viewport."
            )
            preview_render_stop_button = server.add_gui_button("Exit Render Preview", color="red", visible=False)

            transition_sec_number = server.add_gui_number(
                "Transition (sec)",
                min=0.001,
                max=30.0,
                step=0.001,
                initial_value=2.0,
                hint="Time in seconds between each keyframe, which can also be overridden on a per-transition basis.",
            )
            framerate_number = server.add_gui_number("FPS", min=0.1, max=240.0, step=1e-2, initial_value=30.0)
            framerate_buttons = server.add_gui_button_group("", ("24", "30", "60"))
            duration_number = server.add_gui_number(
                "Duration (sec)",
                min=0.0,
                max=1e8,
                step=0.001,
                initial_value=0.0,
                disabled=True,
            )

            @framerate_buttons.on_click
            def _(_) -> None:
                framerate_number.value = float(framerate_buttons.value)

        @transition_sec_number.on_update
        def _(_) -> None:
            camera_path.default_transition_sec = transition_sec_number.value
            duration_number.value = camera_path.compute_duration()

        def get_max_frame_index() -> int:
            return max(1, int(framerate_number.value * duration_number.value) - 1)

        preview_camera_handle: Optional[viser.SceneNodeHandle] = None

        def remove_preview_camera() -> None:
            nonlocal preview_camera_handle
            if preview_camera_handle is not None:
                preview_camera_handle.remove()
                preview_camera_handle = None

        def compute_and_update_preview_camera_state() -> Optional[Union[Tuple[tf.SE3, float], Tuple[tf.SE3, float, float]]]:
            """Update the render tab state with the current preview camera pose.
            Returns current camera pose + FOV if available."""

            if preview_frame_slider is None:
                return
            maybe_pose_and_fov_rad = camera_path.interpolate_pose_and_fov_rad(
                preview_frame_slider.value / get_max_frame_index()
            )
            if maybe_pose_and_fov_rad is None:
                remove_preview_camera()
                return
            time = None
            if len(maybe_pose_and_fov_rad) == 3:  # Time is enabled.
                pose, fov_rad, time = maybe_pose_and_fov_rad
                self.state.preview_time = time
            else:
                pose, fov_rad = maybe_pose_and_fov_rad
            self.state.preview_fov = fov_rad
            self.state.preview_aspect = camera_path.get_aspect()

            if time is not None:
                return pose, fov_rad, time
            else:
                return pose, fov_rad

        def add_preview_frame_slider() -> Optional[viser.GuiInputHandle[int]]:
            """Helper for creating the current frame # slider. This is removed and
            re-added anytime the `max` value changes."""

            with playback_folder:
                preview_frame_slider = server.add_gui_slider(
                    "Preview frame",
                    min=0,
                    max=get_max_frame_index(),
                    step=1,
                    initial_value=0,
                    # Place right after the pause button.
                    order=preview_render_stop_button.order + 0.01,
                    disabled=get_max_frame_index() == 1,
                )
                play_button.disabled = preview_frame_slider.disabled
                preview_render_button.disabled = preview_frame_slider.disabled

            @preview_frame_slider.on_update
            def _(_) -> None:
                nonlocal preview_camera_handle
                maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
                if maybe_pose_and_fov_rad is None:
                    return
                if len(maybe_pose_and_fov_rad) == 3:  # Time is enabled.
                    pose, fov_rad, time = maybe_pose_and_fov_rad
                else:
                    pose, fov_rad = maybe_pose_and_fov_rad

                preview_camera_handle = server.add_camera_frustum(
                    "/preview_camera",
                    fov=fov_rad,
                    aspect=resolution.value[0] / resolution.value[1],
                    scale=0.35,
                    wxyz=pose.rotation().wxyz,
                    position=pose.translation(),
                    color=(10, 200, 30),
                )
                if self.state.preview_render:
                    for client in server.get_clients().values():
                        client.camera.wxyz = pose.rotation().wxyz
                        client.camera.position = pose.translation()

            return preview_frame_slider

        # We back up the camera poses before and after we start previewing renders.
        camera_pose_backup_from_id: Dict[int, tuple] = {}

        @preview_render_button.on_click
        def _(_) -> None:
            self.state.preview_render = True
            preview_render_button.visible = False
            preview_render_stop_button.visible = True

            maybe_pose_and_fov_rad = compute_and_update_preview_camera_state()
            if maybe_pose_and_fov_rad is None:
                remove_preview_camera()
                return
            if len(maybe_pose_and_fov_rad) == 3:  # Time is enabled.
                pose, fov, time = maybe_pose_and_fov_rad
            else:
                pose, fov = maybe_pose_and_fov_rad
            del fov

            # Hide all scene nodes when we're previewing the render.
            server.set_global_scene_node_visibility(False)

            # Back up and then set camera poses.
            for client in server.get_clients().values():
                camera_pose_backup_from_id[client.client_id] = (
                    client.camera.position,
                    client.camera.look_at,
                    client.camera.up_direction,
                )
                client.camera.wxyz = pose.rotation().wxyz
                client.camera.position = pose.translation()

        @preview_render_stop_button.on_click
        def _(_) -> None:
            self.state.preview_render = False
            preview_render_button.visible = True
            preview_render_stop_button.visible = False

            # Revert camera poses.
            for client in server.get_clients().values():
                if client.client_id not in camera_pose_backup_from_id:
                    continue
                cam_position, cam_look_at, cam_up = camera_pose_backup_from_id.pop(client.client_id)
                client.camera.position = cam_position
                client.camera.look_at = cam_look_at
                client.camera.up_direction = cam_up
                client.flush()

            # Un-hide scene nodes.
            server.set_global_scene_node_visibility(True)

        preview_frame_slider = add_preview_frame_slider()

        # Update the # of frames.
        @duration_number.on_update
        @framerate_number.on_update
        def _(_) -> None:
            remove_preview_camera()  # Will be re-added when slider is updated.

            nonlocal preview_frame_slider
            old = preview_frame_slider
            assert old is not None

            preview_frame_slider = add_preview_frame_slider()
            if preview_frame_slider is not None:
                old.remove()
            else:
                preview_frame_slider = old

            camera_path.framerate = framerate_number.value
            camera_path.update_spline()

        # Play the camera trajectory when the play button is pressed.
        @play_button.on_click
        def _(_) -> None:
            play_button.visible = False
            pause_button.visible = True

            def play() -> None:
                while not play_button.visible:
                    max_frame = int(framerate_number.value * duration_number.value)
                    if max_frame > 0:
                        assert preview_frame_slider is not None
                        preview_frame_slider.value = (preview_frame_slider.value + 1) % max_frame
                    time.sleep(1.0 / framerate_number.value)

            threading.Thread(target=play).start()

        # Play the camera trajectory when the play button is pressed.
        @pause_button.on_click
        def _(_) -> None:
            play_button.visible = True
            pause_button.visible = False

        # add button for loading existing path
        load_camera_path_button = server.add_gui_upload_button(
            "Load trajectory", icon=viser.Icon.FILE_UPLOAD, hint="Load an existing camera path."
        )

        @load_camera_path_button.on_upload
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None

            data = event.target.value.content
            
            # Read trajectory json
            json_data = json.loads(data.decode("utf-8"))

            keyframes = json_data["keyframes"]
            camera_path.reset()
            for i in range(len(keyframes)):
                frame = keyframes[i]
                pose_np = np.array(frame["matrix"]).reshape(-1, 4)
                pose_np = apply_transform(self.transform, pose_np)
                pose_np = pad_poses(pose_np)
                pose = tf.SE3.from_matrix(pose_np)
                # apply the x rotation by 180 deg
                pose = tf.SE3.from_rotation_and_translation(
                    pose.rotation(),
                    pose.translation(),
                )
                camera_path.add_camera(
                    Keyframe(
                        position=pose.translation(),
                        wxyz=pose.rotation().wxyz,
                        # There are some floating point conversions between degrees and radians, so the fov and
                        # default_Fov values will not be exactly matched.
                        override_fov_enabled=abs(frame["fov"] - json_data.get("default_fov", 0.0)) > 1e-3,
                        override_fov_rad=frame["fov"] / 180.0 * np.pi,
                        override_time_enabled=frame.get("override_time_enabled", False),
                        override_time_val=frame.get("render_time", None),
                        aspect=frame["aspect"],
                        override_transition_enabled=frame.get("override_transition_enabled", None),
                        override_transition_sec=frame.get("override_transition_sec", None),
                    ),
                )

            transition_sec_number.value = json_data.get("default_transition_sec", 0.5)

            # update the render name
            camera_path.update_spline()


        export_button = server.add_gui_button(
            "Export trajectory",
            color="green",
            icon=viser.Icon.FILE_EXPORT,
            hint="Export trajectory file.",
        )

        reset_up_button = server.add_gui_button(
            "Reset up direction",
            icon=viser.Icon.ARROW_BIG_UP_LINES,
            color="gray",
            hint="Set the up direction of the camera orbit controls to the camera's current up direction.",
        )

        def get_trajectory():
            num_frames = int(framerate_number.value * duration_number.value)
            json_data = {}
            # json data has the properties:
            # keyframes: list of keyframes with
            #     matrix : flattened 4x4 matrix
            #     fov: float in degrees
            #     aspect: float
            # camera_type: string of camera type
            # image_size: [int, int]
            # fps: int
            # seconds: float
            # is_cycle: bool
            # smoothness_value: float
            # camera_path: list of frames with properties
            #     pose: flattened 4x4 matrix
            #     intrinsics: [fx: float, fy: float, cx: float, cy: float]
            keyframes = []
            for keyframe, dummy in camera_path._keyframes.values():
                pose = tf.SE3.from_rotation_and_translation(
                    tf.SO3(keyframe.wxyz),
                    keyframe.position,
                ).as_matrix()
                pose = apply_transform(self._inv_transform, pose)
                keyframe_dict = {
                    "matrix": pose.flatten().tolist(),
                    "fov": np.rad2deg(keyframe.override_fov_rad) if keyframe.override_fov_enabled else fov_degrees.value,
                    "aspect": keyframe.aspect,
                    "override_transition_enabled": keyframe.override_transition_enabled,
                    "override_transition_sec": keyframe.override_transition_sec,
                }
                if render_time is not None:
                    keyframe_dict["render_time"] = (
                        keyframe.override_time_val if keyframe.override_time_enabled else render_time.value
                    )
                    keyframe_dict["override_time_enabled"] = keyframe.override_time_enabled
                keyframes.append(keyframe_dict)
            json_data["default_fov"] = fov_degrees.value
            if render_time is not None:
                json_data["default_time"] = render_time.value if render_time is not None else None
            json_data["format"] = "nerfbaselines"
            json_data["version"] = 1
            json_data["default_transition_sec"] = transition_sec_number.value
            json_data["keyframes"] = keyframes
            json_data["camera_type"] = "pinhole"
            json_data["image_size"] = w, h = int(resolution.value[0]), int(resolution.value[1])
            json_data["fps"] = framerate_number.value
            json_data["seconds"] = duration_number.value
            json_data["is_cycle"] = loop.value
            json_data["smoothness_value"] = tension_slider.value
            # now populate the camera path:
            camera_path_list = []
            for i in range(num_frames):
                maybe_pose_and_fov = camera_path.interpolate_pose_and_fov_rad(i / num_frames)
                if maybe_pose_and_fov is None:
                    return
                time = None
                if len(maybe_pose_and_fov) == 3:  # Time is enabled.
                    pose, fov, time = maybe_pose_and_fov
                else:
                    pose, fov = maybe_pose_and_fov
                # rotate the axis of the camera 180 about x axis
                pose = tf.SE3.from_rotation_and_translation(
                    pose.rotation(),
                    pose.translation(),
                ).as_matrix()
                pose = apply_transform(self._inv_transform, pose)

                fov_deg = np.rad2deg(keyframe.override_fov_rad) if keyframe.override_fov_enabled else fov_degrees.value
                focal_length = three_js_perspective_camera_focal_length(fov_deg, h)
                camera_path_list_dict = {
                    "pose": pose.flatten().tolist(),
                    "intrinsics": [focal_length, focal_length, w / 2, h / 2],
                }
                if time is not None:
                    camera_path_list_dict["render_time"] = time
                camera_path_list.append(camera_path_list_dict)
            json_data["camera_path"] = camera_path_list
            return json_data

        @reset_up_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            event.client.camera.up_direction = tf.SO3(event.client.camera.wxyz) @ np.array([0.0, -1.0, 0.0])

        @export_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            json_data = get_trajectory()

            # now write the json file
            data = json.dumps(json_data, indent=2).encode("utf-8")
            self.server.send_file_download("trajectory.json", data)

        camera_path = CameraPath(server, duration_number, self.state.is_time_enabled, self.transform)
        camera_path.default_fov = fov_degrees.value / 180.0 * np.pi
        camera_path.default_transition_sec = transition_sec_number.value

    def _build_control_panel(self):
        state = self.state
        server = self.server
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

    def _build_gui(self):
        buttons = (
            viser.theme.TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://jkulhanek.com/nerfbaselines",
            ),
            viser.theme.TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/jkulhanek/nerfbaselines",
            ),
        )
        image = viser.theme.TitlebarImage(
            image_url_light="https://docs.nerf.studio/_static/imgs/logo.png",
            image_url_dark="https://docs.nerf.studio/_static/imgs/logo-dark.png",
            image_alt="Nerfbaselines Logo",
            href="https://jkulhanek.com/nerfbaselines/",
        )
        titlebar_theme = viser.theme.TitlebarConfig(buttons=buttons, image=image)
        self.server.configure_theme(
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

        self.server.add_gui_markdown(self.state.b.map(make_stats_markdown))
        self.server.add_gui_text("FPS", initial_value=self.state.b.fps, disabled=True)

        tabs = self.server.add_gui_tab_group()
        with tabs.add_tab("Control", viser.Icon.SETTINGS):
            self._build_control_panel()

            # Add toggles to show/hide cameras
            self.server.add_gui_checkbox(label="Show train cams", 
                                         initial_value=self.state.b.show_train_cameras)
            self.server.add_gui_checkbox(label="Show test cams", 
                                         initial_value=self.state.b.show_test_cameras)
            self.server.add_gui_checkbox(label="Show input PC", 
                                         initial_value=self.state.b.show_input_points)

        with tabs.add_tab("Render", viser.Icon.CAMERA):
            self._build_render_tab()

    def _add_state_handlers(self):
        # Add bindings
        def _update_handles_visibility(split, visible):
            for handle in self._camera_frustrum_handles.get(split, []):
                handle.visible = visible
        self.state.b.show_train_cameras.on_update(lambda x: _update_handles_visibility("train", x))
        self.state.b.show_test_cameras.on_update(lambda x: _update_handles_visibility("test", x))
        self.state.on_update(lambda _: self._reset_render())

        pc = None
        pc_points = None
        def _update_points_3D(_):
            nonlocal pc, pc_points
            if not pc_points is self.state.input_points:
                if pc is not None:
                    pc.remove()
                    pc = None
                pc_points = None
            if self.state.input_points is None:
                return
            if not self.state.show_input_points:
                if pc is not None:
                    pc.visible = False
                return
            if pc is None:
                points, rgb = self.state.input_points
                transform, scale = get_transform_and_scale(self.transform)
                points = np.concatenate([points, np.ones((len(points), 1))], -1) @ transform.T
                points = points[..., :-1] / points[..., -1:]
                points *= scale
                pc = self.server.add_point_cloud(
                    "/initial-point-cloud",
                     points=points,
                     colors=rgb,
                     point_size=0.01,
                     point_shape="circle",
                )
            else:
                pc.visible = True

        # Add state handlers and update the state
        self.state.b.input_points.on_update(_update_points_3D)
        self.state.b.show_input_points.on_update(_update_points_3D)

        # Add frustums handlers
        frustums = {}
        old_camimgs = {}

        def _set_view_to_camera(handle: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]):
            with handle.client.atomic():
                handle.client.camera.position = handle.target.position
                handle.client.camera.wxyz = handle.target.wxyz
            self._reset_render()

        def _update_frustums(split, _):
            camimgs = getattr(self.state, f"camera_frustums_{split}")
            if camimgs is None or not old_camimgs.get(split) is camimgs:
                if split in frustums:
                    for handle in frustums[split]:
                        handle.remove()
                    del frustums[split]
            if not getattr(self.state, f"show_{split}_cameras"):
                if split in frustums:
                    for handle in frustums[split]:
                        handle.visible = False
                return
            if camimgs is None:
                return
            if frustums.get(split) is None:
                cams, images, paths = camimgs
                handles = []
                for cam, img, path in zip(cams, images, paths):
                    c2w = apply_transform(self.transform, cam.poses)
                    pos, quat = get_position_quaternion(c2w)
                    W, H = cam.image_sizes.tolist()
                    fy = cam.intrinsics[1]
                    handle = self.server.add_camera_frustum(
                        f"/dataset-{split}/{path}/frustum",
                        fov=2 * np.arctan2(H / 2, fy),
                        aspect=W / H,
                        scale=0.3,
                        position=pos,
                        wxyz=quat,
                        image=img,
                    )
                    handle.on_click(_set_view_to_camera)
                    handles.append(handle)
                frustums[split] = handles
            else:
                for handle in frustums[split]:
                    handle.visible = True

        self.state.b.show_train_cameras.on_update(lambda x: _update_frustums("train", x))
        self.state.b.show_test_cameras.on_update(lambda x: _update_frustums("test", x))
        self.state.b.camera_frustums_train.on_update(lambda _: _update_frustums("train", True))
        self.state.b.camera_frustums_test.on_update(lambda _: _update_frustums("test", True))

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
        self.state.input_points = points, colors

    def add_dataset_views(self, dataset: Dataset, split: str):
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

        max_img_size = 64
        images = []
        paths = []
        for i, (cam, path) in enumerate(zip(dataset["cameras"], dataset["file_paths"])):
            assert cam.image_sizes is not None, "dataset.image_sizes must be set"
            image = None
            if dataset["images"] is not None:
                image = dataset["images"][i]
            if str(path).startswith("/undistorted/"):
                path = str(path)[len("/undistorted/") :]
            else:
                path = str(Path(path).relative_to(Path(dataset.get("file_paths_root") or "")))
            paths.append(path)
            W, H = cam.image_sizes.tolist()
            downsample_factor = max(1, min(W//max_img_size, H//max_img_size))
            if image is not None:
                image = image[::downsample_factor, ::downsample_factor]
                image = image_to_srgb(image, dtype=np.uint8, 
                                      color_space="srgb", 
                                      background_color=np.array(self._initial_background_color, dtype=np.uint8))
            images.append(image)
        setattr(self.state, f"camera_frustums_{split}", (dataset["cameras"], images, paths))

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
                c2w = apply_transform(self._inv_transform, c2w)
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
                        bg_color = np.array(self.state.background_color, dtype=np.uint8)
                        render = image_to_srgb(image, np.uint8, color_space="srgb", allow_alpha=False, background_color=bg_color)
                    elif name == "depth":
                        # Blend depth with correct color pallete
                        render = visualize_depth(image)
                    else:
                        render = image
                    return render
                render = render_single(self.state.output_type)
                if self.state.output_split:
                    split_render = render_single(self.state.split_output_type)
                    assert render.shape == split_render.shape
                    split_point = int(render.shape[1] * self.state.split_percentage)
                    render[:, :split_point] = split_render[:, split_point:]
                    

                client.set_background_image(render, format="jpeg")
                self._render_state[client.client_id] = min(self._render_state.get(client.client_id, 0), render_state + 1)

                if render_state == 1 or len(self._render_times) < assert_not_none(self._render_times.maxlen):
                    self._render_times.append(interval / num_rays * num_rays_total)
                self.state.fps = f"{1.0 / np.mean(self._render_times):.3g}"


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
            state.background_color = bg_color

    def build_server(dataset_metadata=None, **kwargs):
        return ViserViewer(**kwargs, 
                           port=port, 
                           method=method,
                           dataset_metadata=dataset_metadata)

    if data is not None:
        features: FrozenSet[DatasetFeature] = frozenset({"color", "points3D_xyz"})
        train_dataset = load_dataset(data, split="train", features=features, load_features=False)
        server = build_server(dataset_metadata=train_dataset["metadata"])

        # Get background color
        bg_color = train_dataset["metadata"].get("background_color", None)
        if bg_color is not None:
            bg_color = tuple(int(x) for x in bg_color)
            state.background_color = bg_color

        # if max_num_views is not None and len(train_dataset) > max_num_views:
        #     train_dataset = dataset_index_select(train_dataset, np.random.choice(len(train_dataset), 100))

        if train_dataset.get("points3D_xyz") is not None:
            server.add_initial_point_cloud(train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])

        test_dataset = load_dataset(data, split="test", features=features, load_features=False)
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
