from functools import reduce
import logging
import io
import math
import os
from functools import partial, wraps
import inspect
from dataclasses import dataclass
import dataclasses
import contextlib
from pathlib import Path
from collections import deque
from time import perf_counter
from typing import Optional, Tuple, Any, Dict, cast, List, Callable, Union

import numpy as np
import viser
import viser.theme
import viser.transforms as vtf
from viser import ViserServer
import colorsys
import dataclasses
import threading
import time
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import splines
import splines.quaternion
import viser
import viser.transforms as tf
from ..pose_utils import apply_transform, get_transform_and_scale, invert_transform
from scipy import interpolate

from ..types import Method, Dataset, FrozenSet, DatasetFeature, Literal, TypeVar
from ..types import new_cameras
from ..types import TrajectoryFrameAppearance, TrajectoryFrame, TrajectoryKeyframe, Trajectory, TrajectoryInterpolationSource
from ..types import KochanekBartelsInterpolationSource
from ..datasets import dataset_load_features, dataset_index_select
from ..datasets._colmap_utils import qvec2rotmat, rotmat2qvec
from ..utils import CancelledException, assert_not_none
from ..pose_utils import apply_transform, get_transform_and_scale, invert_transform, pad_poses
from ..datasets import load_dataset
from ..backends._rpc import EventCancellationToken
from ..utils import image_to_srgb, visualize_depth, apply_colormap
from ..io import load_trajectory, save_trajectory


ControlType = Literal["object-centric", "default"]
VISER_SCALE_RATIO = 10.0
T = TypeVar("T")


def simple_cache(fn):
    cache_key_val = None

    @wraps(fn)
    def inner(*args, **kwargs):
        nonlocal cache_key_val
        if cache_key_val is not None and safe_eq(cache_key_val[0], (args, kwargs)):
            return cache_key_val[1]
        cache_key_val = (args, kwargs), fn(*args, **kwargs)
        return cache_key_val[1]
    return inner


def pad_to_aspect_ratio(img, aspect_ratio):
    h, w = img.shape[:2]
    newh, neww = h, w
    aspect = w / h
    if aspect_ratio == aspect:
        return img
    if aspect_ratio > aspect:
        neww = int(h * aspect_ratio)
        newimg = np.zeros((h, neww, 3), dtype=img.dtype)
        pad = (neww - w) // 2
        newimg[:, pad:pad + w] = img
    else:
        newh = int(w / aspect_ratio)
        newimg = np.zeros((newh, w, 3), dtype=img.dtype)
        pad = (newh - h) // 2
        newimg[pad:pad + h] = img
    return newimg


def transform_points(transform, points):
    transform, scale = get_transform_and_scale(transform)
    points = np.concatenate([points, np.ones((len(points), 1))], -1) @ transform.T
    points = points[..., :-1] / points[..., -1:]
    points *= scale
    return points


def get_c2w(position, wxyz):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(wxyz)
    c2w[:3, 3] = position
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

        def _update(value: Any = None, **changes):
            nonlocal last_value
            if value is None:
                old = self.get()
                if isinstance(old, dict):
                    value = old.copy()
                    value.update(changes)
                elif hasattr(old, "update"):
                    value = old.update(**changes)
                elif dataclasses.is_dataclass(old):
                    value = dataclasses.replace(old, **changes)  # type: ignore
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
            return self.on_update(wrapped)

        return BindableSource(_get, _update, _on_update)

    def with_default(self, default):
        def _get():
            value = self.get()
            return value if value is not None else default
        def _on_update(callback):
            return self.on_update(lambda state: callback(state) if state is not None else callback(default))
        return BindableSource(_get, self.update, _on_update)

    def map(self: Any, fn, fn_back=None):
        if isinstance(fn, (list, tuple)):
            def fntuple(value):
                if isinstance(value, dict):
                    out = tuple(value[n] for n in fn)
                else:
                    out = tuple(getattr(value, n) for n in fn)
                return out
            return self.map(fntuple)

        def _get():
            return fn(self.get())

        def _set(*args, **kwargs):
            if fn_back is None:
                raise ValueError("Cannot update a mapped state")
            self.update(fn_back(*args, **kwargs))
    
        def _on_update(callback):
            last_value = object()

            def wrapped(state):
                nonlocal last_value
                nval = fn(state)
                if safe_eq(nval, last_value):
                    return
                last_value = nval
                callback(nval)
            return self.on_update(wrapped)
        return BindableSource(_get, _set, _on_update)

    def __not__(self):
        out = self.map(lambda x: not x)
        def _set(value):
            self.update(value=not value)
        out.update = _set
        return out


def autobind(fn) -> Callable[[Union[BindableSource, 'ViewerState']], Any]:
    signature = inspect.signature(fn)
    names = list(signature.parameters.keys())

    def wrapped(state):
        if isinstance(state, BindableSource):
            inner = lambda args: fn(*args)
            return state.map(names).map(inner)
        else:
            return wrapped(state.b).get()
    return wrapped


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


def safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays"""
    if type(a) != type(b):
        return False
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(safe_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and safe_eq(v, b[k]) for k, v in a.items())
    if hasattr(a, "__eq__"):
        return a == b
    return False


@dataclasses.dataclass(frozen=True, eq=False)
class Keyframe:
    position: np.ndarray
    wxyz: np.ndarray
    fov: Optional[float] = None
    transition_duration: Optional[float] = None
    appearance_train_index: Optional[int] = None

    def __eq__(self, other):
       if self is other:
            return True
       if self.__class__ is not other.__class__:
           return NotImplemented  # better than False
       t1 = dataclasses.astuple(self)
       t2 = dataclasses.astuple(other)
       return all(safe_eq(a1, a2) for a1, a2 in zip(t1, t2))


@autobind
@simple_cache
def state_compute_duration(camera_path_loop, camera_path_interpolation, camera_path_keyframes, camera_path_default_transition_duration) -> float:
    if camera_path_interpolation == "none":
        return len(camera_path_keyframes) * camera_path_default_transition_duration
    kf = camera_path_keyframes
    if not camera_path_loop:
        kf = kf[1:]
    return sum(
        k.transition_duration
        if k.transition_duration is not None
        else camera_path_default_transition_duration
        for k in kf
    )


@dataclass(eq=True)
class ViewerState:
    resolution: int = 512
    background_color: Tuple[int, int, int] = (38, 42, 55)
    output_type: Optional[str] = "color"
    output_type_options: Tuple[str, ...] = ("color",)
    composite_depth: bool = False

    output_split: bool = False
    split_percentage: float = 0.5
    split_output_type: Optional[str] = None

    show_train_cameras: bool = False
    show_test_cameras: bool = False
    show_input_points: bool = True
    fps: str = ""

    preview_render: bool = False
    preview_time: float = 0.0
    preview_current_frame: int = 0
    preview_is_playing: bool = False
    render_resolution: Tuple[int, int] = 1920, 1080
    render_fov: float = 75.0
    render_appearance_train_index: Optional[int] = None
    _temporary_appearance_train_index: Optional[int] = None

    camera_path_interpolation: str = "kochanek-bartels"
    camera_path_loop: bool = False
    camera_path_tension: float = 0.0
    camera_path_keyframes: Tuple[Keyframe,...] = ()
    camera_path_default_transition_duration: float = 2.0
    camera_path_framerate: float = 30.0
    camera_path_show_keyframes: bool = True
    camera_path_move_keyframes: bool = False
    camera_path_show_spline: bool = True

    input_points: Optional[Tuple[np.ndarray, Optional[np.ndarray]]] = None
    camera_frustums_train: Optional[Any] = None
    camera_frustums_test: Optional[Any] = None
    image_names_train: Tuple[str,...] = ()

    supports_appearance_from_train_images: bool = False

    _update_callbacks: List = dataclasses.field(default_factory=list)

    def get(self):
        return self

    def on_update(self, callback):
        self._update_callbacks.append(callback)
        return lambda: self._update_callbacks.remove(callback)

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
    def b(self) -> 'BindableSource':
        return BindableSource(lambda: self, self.update, self.on_update)

    def load_trajectory(self, trajectory: Trajectory, transform) -> None:
        if trajectory.get("camera_model", None) != "pinhole":
            raise RuntimeError("Only pinhole camera model is supported")
        if trajectory.get("source") is None:
            raise RuntimeError("Trajectory does not contain 'source'. It is not editable.")
        source = trajectory.get("source")
        assert source is not None  # pyright legacy
        if source.get("type") != "interpolation" or source.get("interpolation") not in {"none", "kochanek-bartels", "ellipse"}:
            raise RuntimeError("The viewer only supports 'kochanek-bartels', 'none' interpolation for the camera trajectory")
        def validate_appearance(appearance):
            if appearance and not appearance.get("embedding_train_index"):
                raise RuntimeError("Setting appearance is only supported through embedding_train_index")
            return appearance
        validate_appearance(source.get("default_appearance"))
        self.camera_path_interpolation = source["interpolation"]
        self.render_resolution = trajectory["image_size"]
        if source["interpolation"] in {"kochanek-bartels"}:
            source = cast(KochanekBartelsInterpolationSource, source)
            self.camera_path_framerate = trajectory["fps"]
            self.camera_path_tension = source["tension"]
            self.camera_path_loop = source["is_cycle"]
        self.render_fov = source["default_fov"]
        def_app = source.get("default_appearance")
        if def_app:
            self.render_appearance_train_index = def_app.get("embedding_train_index", None)
        if "default_transition_duration" in source:
            self.camera_path_default_transition_duration = source["default_transition_duration"]
        keyframes = []
        for k in source["keyframes"]:
            pose_np = apply_transform(transform, k["pose"])
            pose_np = pad_poses(pose_np)
            pose = tf.SE3.from_matrix(pose_np)
            pos, wxyz = pose.translation(), pose.rotation().wxyz
            appearance = validate_appearance(k.get("appearance"))
            appearance_train_index = appearance.get("embedding_train_index") if appearance else None
            keyframes.append(Keyframe(pos, wxyz, k["fov"], k.get("transition_duration"), appearance_train_index))
        self.camera_path_keyframes = tuple(keyframes)

    def get_trajectory(self, inv_transform) -> Trajectory:
        w, h = int(self.render_resolution[0]), int(self.render_resolution[1])
        appearances: List[TrajectoryFrameAppearance] = []
        keyframes: List[TrajectoryKeyframe] = []
        supports_transition_duration = (
            self.camera_path_interpolation == "kochanek-bartels"
        )
        for keyframe in self.camera_path_keyframes:
            pose = tf.SE3.from_rotation_and_translation(
                tf.SO3(keyframe.wxyz),
                keyframe.position,
            ).as_matrix()
            pose = apply_transform(inv_transform, pose)
            appearance: Optional[TrajectoryFrameAppearance] = None
            keyframe_dict: TrajectoryKeyframe = {
                "pose": pose[:3, :],
                "fov": keyframe.fov,
            }
            if supports_transition_duration:
                keyframe_dict["transition_duration"] = keyframe.transition_duration
            if keyframe.appearance_train_index is not None:
                appearance = {
                    "embedding_train_index": keyframe.appearance_train_index,
                }
                keyframe_dict["appearance"] = appearance
            keyframes.append(keyframe_dict)
            if appearance is not None:
                appearances.append(appearance)

        if len(appearances) != 0 and len(appearances) != len(keyframes):
            raise RuntimeError("Appearances must be set for all keyframes or none")
        # now populate the camera path:
        frames: List[TrajectoryFrame] = []
        trajectory_frames = _compute_camera_path_splines(self)
        if trajectory_frames is not None:
            frames = []
            for pos, wxyz, fov, weights in zip(*trajectory_frames):
                pose = tf.SE3.from_rotation_and_translation(
                    tf.SO3(wxyz),
                    pos,
                ).as_matrix()
                pose = apply_transform(inv_transform, pose)
                focal_length = three_js_perspective_camera_focal_length(fov, h)
                intrinsics = np.array([focal_length, focal_length, w / 2, h / 2], dtype=np.float32)
                frames.append(TrajectoryFrame({
                    "pose": pose[:3, :],
                    "intrinsics": intrinsics,
                    "appearance_weights": weights,
                }))
        source: Dict = {
            "type": "interpolation",
            "interpolation": self.camera_path_interpolation,
            "keyframes": keyframes,
            "default_fov": self.render_fov,
            "default_appearance": None if self.render_appearance_train_index is None else {
                "embedding_train_index": self.render_appearance_train_index,
            },
        }
        fps = self.camera_path_framerate
        if source["interpolation"] == "kochanek-bartels":
            source["is_cycle"] = self.camera_path_loop
            source["tension"] = self.camera_path_tension
            source["default_transition_duration"] = self.camera_path_default_transition_duration
        if source["interpolation"] == "none" or source["interpolation"] == "ellipse":
            source["default_transition_duration"] = self.camera_path_default_transition_duration
            fps = 1.0 / self.camera_path_default_transition_duration
        data: Trajectory = {
            "camera_model": "pinhole",
            "image_size": (w, h),
            "fps": fps,
            "source": cast(KochanekBartelsInterpolationSource, source),
            "frames": frames,
        }
        if len(appearances) != 0:
            data["appearances"] = appearances
        return data


_camera_edit_panel = None

def _add_keypoint_onclick_callback(server: ViserServer, state, index, handle):
    global _camera_edit_panel

    @handle.on_click
    def _(_):
        global _camera_edit_panel
        keyframe = state.camera_path_keyframes[index]

        if _camera_edit_panel is not None:
            _camera_edit_panel.remove()
            _camera_edit_panel = None


        state._temporary_appearance_train_index = keyframe.appearance_train_index

        with server.add_3d_gui_container(
            "/camera_edit_panel",
            position=keyframe.position,
        ) as camera_edit_panel:
            _camera_edit_panel = camera_edit_panel
            override_fov = server.add_gui_checkbox("Override FOV", initial_value=keyframe.fov is not None)
            override_fov_degrees = server.add_gui_slider(
                "FOV (degrees)",
                5.0,
                175.0,
                step=0.1,
                initial_value=keyframe.fov if keyframe.fov is not None else state.render_fov,
                disabled=keyframe.fov is None,
            )
            if state.camera_path_interpolation not in {"none", "ellipse"}:
                def _override_transition_changed(_) -> None:
                    state.camera_path_keyframes = tuple(
                        key if i != index else dataclasses.replace(key, transition_duration=None if not override_transition_enabled.value else override_transition_sec.value)
                        for i, key in enumerate(state.camera_path_keyframes)
                    )
                    override_transition_sec.disabled = not override_transition_enabled.value

                override_transition_enabled = server.add_gui_checkbox(
                    "Override transition",
                    initial_value=keyframe.transition_duration is not None,
                )
                override_transition_sec = server.add_gui_number(
                    "Transition (sec)",
                    initial_value=keyframe.transition_duration
                    if keyframe.transition_duration is not None
                    else state.camera_path_default_transition_duration,
                    min=0.001,
                    max=30.0,
                    step=0.001,
                    disabled=not override_transition_enabled.value,
                )
                override_transition_sec.on_update(_override_transition_changed)
                override_transition_enabled.on_update(_override_transition_changed)

            if state.supports_appearance_from_train_images:
                # TODO: fix this
                # Note, we do not do bindable here, because remove() does not remove it at the moment!
                train_image_names = state.b.image_names_train.get()
                init_val = "none"
                options = train_image_names
                if keyframe.appearance_train_index is not None:
                    if keyframe.appearance_train_index < len(train_image_names):
                        init_val = train_image_names[keyframe.appearance_train_index]
                    else:
                        init_val = "unknown #" + str(keyframe.appearance_train_index)
                        options = (init_val,) + train_image_names
                else:
                    options = (init_val,) + train_image_names

                train_embed_dropdown = server.add_gui_dropdown(
                    "Appearance from train image",
                    options=options,
                    initial_value=init_val,
                    hint="Select images to visualize embeddings for",
                    disabled=state.b.image_names_train.map(lambda x: len(x) == 0).get(),
                )

                @train_embed_dropdown.on_update
                def _(_) -> None:
                    val = None
                    if train_embed_dropdown.value != "none" and train_embed_dropdown.value in state.image_names_train:
                        val = state.image_names_train.index(train_embed_dropdown.value)
                    state.camera_path_keyframes = tuple(
                        dataclasses.replace(key, appearance_train_index=val if key.appearance_train_index is None or i == index else key.appearance_train_index)
                        for i, key in enumerate(state.camera_path_keyframes)
                    )
                    state._temporary_appearance_train_index = val

            delete_button = server.add_gui_button("Delete", color="red", icon=viser.Icon.TRASH)
            go_to_button = server.add_gui_button("Go to")
            close_button = server.add_gui_button("Close")


        @override_fov.on_update
        def _(_) -> None:
            override_fov_degrees.disabled = not override_fov.value
            state.camera_path_keyframes = tuple(
                key if i != index else dataclasses.replace(key, fov=None if not override_fov.value else key.fov)
                for i, key in enumerate(state.camera_path_keyframes)
            )

        @override_fov_degrees.on_update
        def _(_) -> None:
            fov = override_fov_degrees.value
            state.camera_path_keyframes = tuple(
                key if i != index else dataclasses.replace(key, fov=None if not override_fov.value else fov)
                for i, key in enumerate(state.camera_path_keyframes)
            )

        @delete_button.on_click
        def _(event: viser.GuiEvent) -> None:
            global _camera_edit_panel
            assert event.client is not None
            with event.client.add_gui_modal("Confirm") as modal:
                event.client.add_gui_markdown("Delete keyframe?")
                confirm_button = event.client.add_gui_button("Yes", color="red", icon=viser.Icon.TRASH)
                exit_button = event.client.add_gui_button("Cancel")

                @confirm_button.on_click
                def _(_) -> None:
                    global _camera_edit_panel
                    assert camera_edit_panel is not None

                    state._temporary_appearance_train_index = None
                    camera_edit_panel.remove()
                    _camera_edit_panel = None
                    modal.close()
                    state.camera_path_keyframes = tuple(key for i, key in enumerate(state.camera_path_keyframes) if i != index)

                @exit_button.on_click
                def _(_) -> None:
                    modal.close()

        @go_to_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            client = event.client
            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(keyframe.wxyz), keyframe.position
            )
            # Important bit: we atomically set both the orientation and the position
            # of the camera.
            with client.atomic():
                client.camera.wxyz = T_world_target.rotation().wxyz
                client.camera.position = T_world_target.translation()

        @close_button.on_click
        def _(_) -> None:
            global _camera_edit_panel
            assert _camera_edit_panel is not None
            state._temporary_appearance_train_index = None
            _camera_edit_panel.remove()
            _camera_edit_panel = None


def _add_keyframe_camera_frustum(server: ViserServer, index: int, keyframe: Keyframe, default_fov: float, aspect: float, visible: bool, move_handle_visible: bool):
    frustum_handle = server.add_camera_frustum(
        f"/render_cameras/{index}",
        fov=(keyframe.fov if keyframe.fov is not None else default_fov) / 180.0 * math.pi,
        aspect=aspect,
        scale=0.1,
        color=(200, 10, 30),
        wxyz=keyframe.wxyz,
        position=keyframe.position,
        visible=visible,
    )
    server.add_icosphere(
        f"/render_cameras/{index}/sphere",
        radius=0.03,
        color=(200, 10, 30),
    )
    controls = None
    if move_handle_visible:
        controls = server.add_transform_controls(
            f"/keyframe_move/{index}",
            scale=0.4,
            wxyz=keyframe.wxyz,
            position=keyframe.position,
        )
    return frustum_handle, controls


def _add_keypoint_move_callback(state: ViewerState, index: int, handle):
    if handle is None:
        return

    @handle.on_update
    def _(_) -> None:
        state.camera_path_keyframes = tuple(
            k if i != index else dataclasses.replace(k, wxyz=handle.wxyz, position=handle.position)
            for i, k in enumerate(state.camera_path_keyframes)
        )


def _add_keyframe_frustums_to_camera_path(server: ViserServer, state: ViewerState):
    keyframe_handles = []

    def update_keyframes(args):
        nonlocal keyframe_handles
        keyframes, render_fov, render_resolution, camera_path_move_keyframes = args

        aspect = render_resolution[0] / render_resolution[1]
        handles_to_remove = []
        handles_to_add = []
        for i, keyframe in enumerate(keyframes):
            cachekey = (keyframe, render_fov, aspect, camera_path_move_keyframes)
            if i < len(keyframe_handles):
                if keyframe_handles[i][0] == cachekey:
                    continue
                cachekey_diffpos = (dataclasses.replace(keyframe_handles[i][0][0],
                                                       position=keyframe.position,
                                                       wxyz=keyframe.wxyz),) + keyframe_handles[i][0][1:]
                if cachekey == cachekey_diffpos:
                    if keyframe_handles[i][1] is not None:
                        keyframe_handles[i][1].position = keyframe.position
                        keyframe_handles[i][1].wxyz = keyframe.wxyz
                    continue
                if keyframe_handles[i][1] is not None:
                    handles_to_remove.append(keyframe_handles[i][1])
                if keyframe_handles[i][2] is not None:
                    handles_to_remove.append(keyframe_handles[i][2])
            else:
                keyframe_handles.append((cachekey, None, None))
            keyframe_handles[i] = cachekey, None, None
            handles_to_add.append((i, keyframe, cachekey))

        while len(keyframes) < len(keyframe_handles):
            _, h1, h2 = keyframe_handles.pop(-1)
            if h1 is not None:
                handles_to_remove.append(h1)
            if h2 is not None:
                handles_to_remove.append(h2)

        for handle in handles_to_remove:
            handle.remove()

        for i, keyframe, cachekey in handles_to_add:
            if i >= len(keyframe_handles) or keyframe_handles[i][0] != cachekey:
                # Newer update has occurred, remove this handle.
                continue
            handle1, handle2 = _add_keyframe_camera_frustum(server, i, keyframe, 
                                                            render_fov, 
                                                            aspect, 
                                                            state.camera_path_show_keyframes,
                                                            camera_path_move_keyframes)
            if i >= len(keyframe_handles) or keyframe_handles[i][0] != cachekey:
                # Newer update has occurred, remove this handle.
                if handle1 is not None:
                    handle1.remove()
                if handle2 is not None:
                    handle2.remove()
                continue
            keyframe_handles[i] = cachekey, handle1, handle2
            _add_keypoint_onclick_callback(server, state, i, handle1)
            _add_keypoint_move_callback(state, i, handle2)


    binding = state.b.map(["camera_path_keyframes", "render_fov", "render_resolution", "camera_path_move_keyframes"])
    binding.on_update(update_keyframes)
    state.b.camera_path_show_keyframes.on_update(lambda show: [setattr(h[1], "visible", show) for h in keyframe_handles if h[1] is not None])
    update_keyframes(binding.get())


def _onehot(index, length):
    out = np.zeros(length, dtype=np.float32)
    out[index] = 1.0
    return out


def _interpolate_ellipse(camera_path_keyframes, num_frames: int, render_fov: float):
    # Compute transition times cumsum
    if num_frames <= 0 or len(camera_path_keyframes) < 3:
        return None

    points = np.stack([k.position for k in camera_path_keyframes], axis=0)
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(centered_points)
    normal_vector = Vt[-1]  # The normal vector to the plane is the last row of Vt

    # Project the points onto the plane
    projection_matrix = np.eye(3) - np.outer(normal_vector, normal_vector)
    projected_points = centered_points @ projection_matrix

    # Now, we have points in a 2D plane, fit a circle in 2D
    A = np.c_[2*projected_points[:,0], 2*projected_points[:,1], np.ones(projected_points.shape[0])]
    b = np.sum(projected_points**2, axis=1)
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    center_2d = x[:2]
    radius = np.sqrt(x[2] + np.sum(center_2d**2))

    # Reproject the center back to 3D
    angles = np.linspace(0, 2*np.pi, int(num_frames), endpoint=False)
    positions = np.stack([center_2d[0] + radius * np.cos(angles), center_2d[1] + radius * np.sin(angles)], axis=-1)
    points_array = positions @ projection_matrix[:2, :2].T
    points_array = np.concatenate([points_array, np.zeros((num_frames, 1))], axis=-1)
    points_array += centroid

    # Convert wxyz to rotation matrices
    poses = np.stack([get_c2w(k.position, k.wxyz) for k in camera_path_keyframes], axis=0)
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
    mt_m = np.transpose(m, [0, 2, 1]) @ m
    focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]

    # Compute camera orientation
    dirz = (focus_pt - points_array).astype(np.float32)
    dirz /= np.linalg.norm(dirz, axis=-1, keepdims=True)
    oriented_normal_vector = normal_vector if np.dot(normal_vector, dirz[0]) > 0 else -normal_vector
    dirx = np.cross(dirz, -oriented_normal_vector)
    diry = np.cross(dirz, dirx)
    R = np.stack([dirx, diry, dirz], axis=-1)
    orientation_array = np.stack([rotmat2qvec(r) for r in R], axis=0)

    # TODO: implement rest
    fovs = np.full(num_frames, render_fov, dtype=np.float32)
    weights = _onehot(0, len(camera_path_keyframes))[np.newaxis].repeat(num_frames, axis=0)
    return points_array, orientation_array, fovs, weights


@autobind
@simple_cache
def _compute_camera_path_splines(camera_path_keyframes, 
                                 camera_path_loop, 
                                 camera_path_tension, 
                                 render_fov,
                                 camera_path_default_transition_duration, 
                                 camera_path_interpolation,
                                 camera_path_framerate):
    if len(camera_path_keyframes) < 2:
        return None

    # For none interpolation, we just return the keyframes.
    if camera_path_interpolation == "none":
        return (
            np.array([k.position for k in camera_path_keyframes], dtype=np.float32),
            np.array([k.wxyz for k in camera_path_keyframes], dtype=np.float32),
            np.array([k.fov if k.fov is not None else render_fov for k in camera_path_keyframes], dtype=np.float32),
            np.array([_onehot(i, len(camera_path_keyframes)) for i in range(len(camera_path_keyframes))],
            dtype=np.float32),
        )

    if camera_path_interpolation == "ellipse":
        num_frames = int(camera_path_default_transition_duration * camera_path_framerate)
        return _interpolate_ellipse(camera_path_keyframes, num_frames, render_fov)


    # Compute transition times cumsum
    times = np.array([
        k.transition_duration if k.transition_duration is not None else camera_path_default_transition_duration 
        for k in camera_path_keyframes], dtype=np.float32)
    transition_times_cumsum = np.cumsum(np.roll(times, -1)) if camera_path_loop else np.cumsum(times[:-1])
    transition_times_cumsum = np.insert(transition_times_cumsum, 0, 0.0)
    del times

    num_frames = int(transition_times_cumsum[-1] * camera_path_framerate)
    if num_frames <= 0 or len(camera_path_keyframes) < 2:
        return None

    orientation_spline = splines.quaternion.KochanekBartels(
        [
            splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(k.wxyz, shift=-1))
            for k in camera_path_keyframes
        ],
        tcb=(camera_path_tension, 0.0, 0.0),
        endconditions="closed" if camera_path_loop else "natural",
    )
    position_spline = splines.KochanekBartels(
        [k.position for k in camera_path_keyframes],
        tcb=(camera_path_tension, 0.0, 0.0),
        endconditions="closed" if camera_path_loop else "natural",
    )
    fov_spline = splines.KochanekBartels(
        [
            k.fov if k.fov is not None else render_fov
            for k in camera_path_keyframes
        ],
        tcb=(camera_path_tension, 0.0, 0.0),
        endconditions="closed" if camera_path_loop else "natural",
    )
    weight_spline = splines.KochanekBartels(
        [
            _onehot(i, len(camera_path_keyframes)) for i in range(len(camera_path_keyframes))
        ],
        tcb=(camera_path_tension, 0.0, 0.0),
        endconditions="closed" if camera_path_loop else "natural",
    )

    # Get time splines
    spline_indices = np.arange(transition_times_cumsum.shape[0])
    gtime = np.linspace(0, transition_times_cumsum[-1], num_frames)
    if camera_path_loop:
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
    gtime_splines = np.clip(interpolator(gtime), 0, spline_indices[-1])

    # Update visualized spline.
    points_array = position_spline.evaluate(gtime_splines)
    orientation_array = orientation_spline.evaluate(gtime_splines)
    orientation_array = np.stack([np.array([quat.scalar, *quat.vector]) for quat in orientation_array])
    fovs = fov_spline.evaluate(gtime_splines)
    weights = weight_spline.evaluate(gtime_splines)
    return points_array, orientation_array, fovs, weights


def _add_spline_to_camera_path(server: ViserServer, state: ViewerState, interpolated_camera_path):
    # Clear prior spline nodes.
    spline_nodes = []

    def add_path_spline(args):
        for node in spline_nodes:
            node.remove()
        spline_nodes.clear()
        if args is None or not state.camera_path_show_spline:
            return
        if state.camera_path_interpolation == "none":
            return
        points = args[0]
        colors = np.array([colorsys.hls_to_rgb(h, 0.5, 1.0) for h in np.linspace(0.0, 1.0, len(points))])
        spline_nodes.append(
            server.add_spline_catmull_rom(
                "/render_camera_spline",
                positions=points,
                color=(220, 220, 220),
                closed=state.camera_path_loop,
                line_width=1.0,
                segments=points.shape[0] + 1,
            )
        )
        spline_nodes.append(
            server.add_point_cloud(
                "/render_camera_spline/points",
                points=points,
                colors=colors,
                point_size=0.04,
            )
        )

    interpolated_camera_path.on_update(add_path_spline)
    state.b.camera_path_show_spline.on_update(lambda _: add_path_spline(interpolated_camera_path.get()))
    add_path_spline(interpolated_camera_path.get())


def _make_train_image_embedding_dropdown(server: ViserServer, state: ViewerState, binding: Optional[BindableSource] = None):
    if binding is None:
        binding = state.b.render_appearance_train_index
    assert binding is not None
    value_binding = binding.map(
        lambda x: ("none" if x is None else (state.image_names_train[x] 
                                            if state.image_names_train is not None and x < len(state.image_names_train) 
                                            else "none")),
        lambda x: None if x == "none" else state.image_names_train.index(x) if x in state.image_names_train else None,
    )
    server.add_gui_dropdown(
        "Appearance from train image",
        options=state.b.image_names_train.map(lambda x: ("none",) + (x or ())),
        initial_value=value_binding,
        hint="Select images to visualize embeddings for",
        disabled=state.b.image_names_train.map(lambda x: len(x) == 0),
    )


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
            handle_update = None
            updatable = ("value", "initial_value", "disabled", "enabled", "visible")

            # Capture container_id
            gui = getattr(self.server, "gui", self.server)
            container_id = gui._get_container_id()

            def _update_component(name, value):
                nonlocal handle
                if name is not None:
                    if name == "initial_value":
                        name = "value"
                    if handle is not None and hasattr(handle, name) and name in updatable:
                        if getattr(handle, name) != value:
                            setattr(handle, name, value)
                        logging.debug("Updating component", add_gui, name, "->", value)
                        return
                if handle is not None:
                    handle.remove()
                    handle = None
                _args, _kwargs = _current_args()
                old_container_id = gui._get_container_id()
                try:
                    # Mock the container_id
                    gui._set_container_id(container_id)
                    handle = add_gui(*_args, **_kwargs)
                finally:
                    gui._set_container_id(old_container_id)
                logging.debug("Creating component", add_gui, name, "->", value)
                if handle_update is not None:
                    handle.on_update(handle_update)
                return handle

            def _update_binding(binding, event):
                binding.update(event.target.value)

            bindings_to_remove = []
            for name, binding in prop_bindings.items():
                bindings_to_remove.append(binding.on_update(partial(_update_component, name)))
                if name in ("value", "initial_value"):
                    handle_update = partial(_update_binding, binding)
            _update_component(None, None)
            assert handle is not None, "Failed to build component"
            if "order" not in kwargs:
                kwargs["order"] = handle.order

            if not bindings_to_remove:
                return handle

            class ProxyHandle:
                def __getattr__(self, name):
                    return getattr(handle, name)

                def __setattr__(self, name, value):
                    setattr(handle, name, value)

                def __enter__(self):
                    assert handle is not None
                    out = handle.__enter__()
                    if out == handle:
                        return self
                    return out

                def __exit__(self, *args):
                    assert handle is not None
                    return handle.__exit__(*args)

                def remove(self):
                    nonlocal handle
                    for remove in bindings_to_remove:
                        remove()
                    bindings_to_remove.clear()
                    if handle is not None:
                        handle.remove()
                        handle = None

            return ProxyHandle()

        return _add_gui


class ViserViewer:
    def __init__(self, 
                 method: Optional[Method], 
                 port, 
                 dataset_metadata=None,
                 state=None):

        self.transform = self._inv_transform = np.eye(4, dtype=np.float32)
        self.initial_pose = None
        self._dataset_metadata = dataset_metadata

        control_type = "default"
        self._expected_depth_scale = 0.5
        if dataset_metadata is not None:
            self.transform = dataset_metadata.get("viewer_transform").copy()
            self.initial_pose = dataset_metadata.get("viewer_initial_pose").copy()
            control_type = "object-centric" if dataset_metadata.get("type") == "object-centric" else "default"
            self.initial_pose[:3, 3] *= VISER_SCALE_RATIO
            self._expected_depth_scale = dataset_metadata.get("expected_scene_scale", 0.5)

        self.transform[:3, :] *= VISER_SCALE_RATIO
        self._inv_transform = invert_transform(self.transform, True)

        self.port = port
        self.method = method

        self.state = state or ViewerState()
        if self.method is not None:
            try:
                self.state.supports_appearance_from_train_images = self.method.get_train_embedding(0) is not None
            except (AttributeError, NotImplementedError):
                pass
            if self.state.supports_appearance_from_train_images:
                logging.info("Supports appearance from train embeddings")
            else:
                logging.info("Does not support appearance from train embeddings")
        self._render_state = {}
        self._last_poses = {}
        self._update_state_callbacks = []
        self._render_times = deque(maxlen=3)
        self._preview_camera: Any = None
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
                self._render_state.pop(client.client_id, None)
                self._reset_render(False)

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

        self._current_embedding = None

        self._camera_frustrum_handles = {}
        self._camera_path_binding = _compute_camera_path_splines(self.state.b)

        def _fix_current_frame(camera_path):
            max_frame = camera_path[0].shape[0] - 1 if camera_path is not None else 0
            if self.state.preview_current_frame > max_frame:
                self.state.preview_current_frame = max_frame
        self._camera_path_binding.on_update(_fix_current_frame)
        self._build_gui()
        self._start_preview_timer()

    def _build_render_tab(self):
        server: ViserServer = self.server

        server.add_gui_slider(
            "Default FOV",
            initial_value=self.state.b.render_fov,
            min=0.1,
            max=175.0,
            step=0.01,
            hint="Field-of-view for rendering, which can also be overridden on a per-keyframe basis.",
        )

        server.add_gui_vector2(
            "Resolution",
            initial_value=self.state.b.render_resolution,
            min=(50, 50),
            max=(10_000, 10_000),
            step=1,
            hint="Render output resolution in pixels.",
        )

        if self.state.supports_appearance_from_train_images:
            _make_train_image_embedding_dropdown(server, self.state, self.state.b.render_appearance_train_index)

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
            apperance_train_index = self.state.render_appearance_train_index
            if apperance_train_index is not None and len(self.state.camera_path_keyframes) > 0:
                apperance_train_index = self.state.camera_path_keyframes[-1].appearance_train_index
            self.state.camera_path_keyframes = self.state.camera_path_keyframes + (Keyframe(
                position=camera.position,
                wxyz=camera.wxyz,
                appearance_train_index=apperance_train_index,
            ),)

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
                    self.state.camera_path_keyframes = ()
                    modal.close()

                @exit_button.on_click
                def _(_) -> None:
                    modal.close()

        server.add_gui_dropdown(
            "Interpolation",
            initial_value=self.state.b.camera_path_interpolation,
            options=("kochanek-bartels", "none", "ellipse"),
            hint="Camera path interpolation.")

        server.add_gui_checkbox(
            "Loop",
            initial_value=self.state.b.camera_path_loop,
            visible=self.state.b.camera_path_interpolation.map(lambda x: x == "kochanek-bartels"),
            hint="Add a segment between the first and last keyframes.")

        server.add_gui_slider(
            "Spline tension",
            min=0.0,
            max=1.0,
            initial_value=self.state.b.camera_path_tension,
            visible=self.state.b.camera_path_interpolation.map(lambda x: x == "kochanek-bartels"),
            step=0.01,
            hint="Tension parameter for adjusting smoothness of spline interpolation.",
        )

        server.add_gui_checkbox(
            "Move keyframes",
            initial_value=self.state.b.camera_path_move_keyframes,
            hint="Toggle move handles for keyframes in the scene.",
        )

        server.add_gui_checkbox(
            "Show keyframes",
            initial_value=self.state.b.camera_path_show_keyframes,
            hint="Show keyframes in the scene.",
        )

        server.add_gui_checkbox(
            "Show spline",
            initial_value=self.state.b.camera_path_show_spline,
            visible=self.state.b.camera_path_interpolation.map(lambda x: x == "kochanek-bartels"),
            hint="Show camera path spline in the scene.",
        )

        playback_folder = server.add_gui_folder("Playback")
        preview_disabled_b = self._camera_path_binding.map(lambda x: x is None or x[0].shape[0] <= 1)
        with playback_folder:
            play_button = server.add_gui_button(
                "Play", 
                icon=viser.Icon.PLAYER_PLAY,
                disabled=preview_disabled_b,
                visible=self.state.b.preview_is_playing.map(lambda x: not x))
            play_button.on_click(lambda _: setattr(self.state, "preview_is_playing", True))
            pause_button = server.add_gui_button("Pause", 
                icon=viser.Icon.PLAYER_PAUSE, 
                visible=self.state.b.preview_is_playing)
            pause_button.on_click(lambda _: setattr(self.state, "preview_is_playing", False))
            preview_render_button = server.add_gui_button(
                "Preview Render", 
                hint="Show a preview of the render in the viewport.",
                disabled=preview_disabled_b,
                visible=self.state.b.preview_render.map(lambda x: not x),
            )
            preview_render_button.on_click(lambda _: setattr(self.state, "preview_render", True))
            preview_render_stop_button = server.add_gui_button(
                "Exit Render Preview", 
                color="red", 
                visible=self.state.b.preview_render)
            preview_render_stop_button.on_click(lambda _: setattr(self.state, "preview_render", False))
            server.add_gui_number(
                "Transition (sec)",
                min=0.001,
                max=30.0,
                step=0.001,
                initial_value=self.state.b.camera_path_default_transition_duration,
                hint="Time in seconds between each keyframe, which can also be overridden on a per-transition basis.",
            )
            server.add_gui_number(
                "FPS", min=0.1, max=240.0, step=1e-2, 
                visible=self.state.b.camera_path_interpolation.map(lambda x: x != "none"),
                initial_value=self.state.b.camera_path_framerate)
            framerate_buttons = server.add_gui_button_group(
                "",
                ("24", "30", "60"),
                visible=self.state.b.camera_path_interpolation.map(lambda x: x != "none"))
            framerate_buttons.on_click(lambda _: self.state.b.camera_path_framerate.update(float(framerate_buttons.value)))
            server.add_gui_number(
                "Duration (sec)",
                min=0.0,
                max=1e8,
                step=0.001,
                disabled=True,
                initial_value=state_compute_duration(self.state.b),
            )

            server.add_gui_slider(
                "Preview frame",
                min=0,
                step=1,
                initial_value=self.state.b.preview_current_frame,
                # Place right after the pause button.
                order=preview_render_stop_button.order + 0.01,
                disabled=preview_disabled_b,
                max=self._camera_path_binding.map(lambda x: x[0].shape[0] - 1 if x is not None else 1),
            )

        # add button for loading existing path
        load_camera_path_button = server.add_gui_upload_button(
            "Load trajectory", icon=viser.Icon.FILE_UPLOAD, hint="Load an existing camera path."
        )

        @load_camera_path_button.on_upload
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None

            data = event.target.value.content
            
            # Read trajectory json
            with io.TextIOWrapper(io.BytesIO(data), encoding="utf8") as file:
                trajectory = load_trajectory(file)
            self.state.load_trajectory(trajectory, self.transform)

        export_button = server.add_gui_button(
            "Export trajectory",
            color="green",
            icon=viser.Icon.FILE_EXPORT,
            hint="Export trajectory file.",
        )

        @export_button.on_click
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None

            trajectory = self.state.get_trajectory(self._inv_transform)
            with io.BytesIO() as file, io.TextIOWrapper(file, encoding="utf8") as textfile:
                save_trajectory(trajectory, textfile)
                textfile.flush()
                data = file.getvalue()

            # now write the json file
            self.server.send_file_download("trajectory.json", data)

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
                initial_value=state.b.output_type,
                hint="The output to render",
            )
            server.add_gui_rgb("Background color", state.b.background_color, hint="Color of the background")

        # split options
        with server.add_gui_folder("Split Screen"):#, visible=state.b.output_type_options.map(lambda x: len(x) > 1)):
            server.add_gui_checkbox(
                "Enable",
                initial_value=state.b.output_split,
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
        self.server.configure_theme(
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )
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

        _add_spline_to_camera_path(self.server, self.state, self._camera_path_binding)
        self._add_state_handlers()

    def _start_preview_timer(self):
        # Add preview_is_playing timer
        max_frame_b = self._camera_path_binding.map(lambda x: x[0].shape[0] - 1 if x is not None else 1)
        max_fps = 30

        def play() -> None:
            start = None
            start_frame = None
            i = 0
            # while self.state is not None:
            while True:
                i+=1
                max_frame = max_frame_b.get()
                target_fps = self.state.camera_path_framerate
                if self.state.camera_path_interpolation == "none":
                    target_fps = 1. / self.state.camera_path_default_transition_duration
                if not self.state.preview_is_playing:
                    start = None
                else:
                    if start is None:
                        start = time.time()
                        start_frame = self.state.preview_current_frame 
                    if max_frame > 0:
                        assert start_frame is not None
                        frame = int((time.time() - start) * target_fps)
                        self.state.preview_current_frame = (start_frame + frame) % (max_frame + 1)
                wait = 1.0 / min(target_fps, max_fps)
                if start is not None:
                    wait = max(wait / 2, (start - time.time()) % wait)
                time.sleep(wait)

        threading.Thread(target=play, daemon=True).start()

    def _add_state_handlers(self):
        # Add bindings
        def _update_handles_visibility(split, visible):
            for handle in self._camera_frustrum_handles.get(split, []):
                handle.visible = visible
        self.state.b.show_train_cameras.on_update(lambda x: _update_handles_visibility("train", x))
        self.state.b.show_test_cameras.on_update(lambda x: _update_handles_visibility("test", x))

        # Add handler to update render on render panel change
        self.state.b.render_fov.on_update(lambda _: self._reset_render())
        self.state.b.output_type.on_update(lambda _: self._reset_render())
        self.state.b.resolution.on_update(lambda _: self._reset_render())
        self.state.b.output_split.on_update(lambda _: self._reset_render())
        self.state.b.split_output_type.on_update(lambda _: self._reset_render())
        self.state.b.background_color.on_update(lambda _: self._reset_render())

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
                if rgb is None:
                    rgb = np.full((len(points), 3), 255, dtype=np.uint8)
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

        # Add preview camera handler
        preview_camera_handle = None
        
        def _update_preview_handle(_) -> None:
            nonlocal preview_camera_handle
            trajectory = self._camera_path_binding.get()
            if trajectory is None:
                if preview_camera_handle is not None:
                    preview_camera_handle.remove()
                    preview_camera_handle = None
                return
            poses, quats, fovs, _ = trajectory
            pos = poses[self.state.preview_current_frame % len(poses)]
            quat = quats[self.state.preview_current_frame % len(quats)]
            fov = fovs[self.state.preview_current_frame % len(fovs)]
            if preview_camera_handle is None:
                preview_camera_handle = self.server.add_camera_frustum(
                    "/preview_camera",
                    fov=fov / 180 * math.pi,
                    aspect=self.state.render_resolution[0] / self.state.render_resolution[1],
                    scale=0.35,
                    wxyz=quat,
                    position=pos,
                    color=(10, 200, 30),
                )
            else:
                preview_camera_handle.position = pos
                preview_camera_handle.wxyz = quat
                preview_camera_handle.fov = fov / 180 * math.pi
        self._camera_path_binding.on_update(_update_preview_handle)
        self.state.b.preview_current_frame.on_update(_update_preview_handle)
        self.state.b.render_resolution.on_update(_update_preview_handle)

        # Add preview render handler
        # It hides all scene nodes and backup/restores the camera states before/after preview
        camera_pose_backup = {}

        def _update_preview(_) -> None:
            if self.state.preview_render:
                # Back up and then set camera poses.
                points, quats, fovs, weights, *_ = self._camera_path_binding.get()
                num_points = len(points)
                position = points[self.state.preview_current_frame % num_points]
                wxyz = quats[self.state.preview_current_frame % num_points]
                fov = fovs[self.state.preview_current_frame % num_points]
                kweight = weights[self.state.preview_current_frame % num_points]
                embedding = None
                for i, val in enumerate(kweight):
                    if val > 1e-5 and self.method is not None and self.state.camera_path_keyframes[i].appearance_train_index is not None:
                        if embedding is None:
                            embedding = { "weights": [], "appearances": []}
                        embedding["weights"].append(val)
                        embedding["appearances"].append({
                            "embedding_train_index": self.state.camera_path_keyframes[i].appearance_train_index,
                        })
                self._preview_camera = (
                    position,
                    wxyz,
                    fov / 180 * math.pi,
                    embedding,
                )
                for client in self.server.get_clients().values():
                    if client.client_id not in camera_pose_backup:
                        with client.atomic():
                            camera_pose_backup[client.client_id] = (
                                client.camera.wxyz,
                                client.camera.position,
                                client.camera.fov,
                            )

                # Preview camera changed, reset render
                self._reset_render()

                    # Important bit: we atomically set both the orientation and the position
                    # of the camera.
                    # if not self.state.preview_is_playing:
                    #     with client.atomic():
                    #         client.camera.fov = fov / 180 * math.pi
                    #         client.camera.wxyz = wxyz
                    #         client.camera.position = position

            elif camera_pose_backup:
                self._preview_camera = None
                # Revert camera poses.
                for client in self.server.get_clients().values():
                    if client.client_id not in camera_pose_backup:
                        continue
                    with client.atomic():
                        (
                            client.camera.wxyz,
                            client.camera.position,
                            client.camera.fov,
                        ) = camera_pose_backup.pop(client.client_id)
                        client.flush()
                        camera_pose_backup.pop(client.client_id, None)
            else:
                self._preview_camera = None

        def _show_hide_preview_render(preview_render):
            # Hide all scene nodes when we're previewing the render.
            self.server.set_global_scene_node_visibility(not preview_render)

        self.state.b.preview_render.on_update(_show_hide_preview_render)
        _show_hide_preview_render(self.state.b.preview_render.get())

        self._camera_path_binding.on_update(_update_preview)
        self.state.b.preview_render.on_update(_update_preview)
        self.state.b.preview_current_frame.on_update(_update_preview)
        self.state.b.preview_is_playing.on_update(_update_preview)

        # Add keypoint handles
        _add_keyframe_frustums_to_camera_path(self.server, self.state)

        # if self.state.preview_render:
        #     for client in server.get_clients().values():
        #         client.camera.wxyz = pose.rotation().wxyz
        #         client.camera.position = pose.translation()

        # return preview_frame_slider

        def _update_fov(fov):
            if not self.state.preview_render:
                for client in self.server.get_clients().values():
                    client.camera.fov = fov / 180 * math.pi
        self.state.b.render_fov.on_update(_update_fov)

        def _update_render_appearance_train_index(args):
            index, temp_index = args
            if self.method is None:
                self._current_embedding = None
                return

            if temp_index is not None:
                self._current_embedding = {
                    "embedding_train_index": temp_index,
                }
            elif index is not None:
                self._current_embedding = {
                    "embedding_train_index": index,
                }
            else:
                self._current_embedding = None
            self._reset_render()
        self.state.b.map(("render_appearance_train_index", "_temporary_appearance_train_index")).on_update(_update_render_appearance_train_index)

    def _reset_render(self, reset_state=True):
        if reset_state:
            self._render_state = {}
        token, self._cancellation_token = self._cancellation_token, None
        if token is not None:
            token.cancel()

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
        if split == "train":
            self.state.image_names_train = tuple([os.path.relpath(x, dataset.get("image_paths_root") or "") for x in dataset.get("image_paths")])

        max_img_size = 64
        images = []
        paths = []
        for i, (cam, path) in enumerate(zip(dataset["cameras"], dataset["image_paths"])):
            assert cam.image_sizes is not None, "dataset.image_sizes must be set"
            image = None
            if dataset["images"] is not None:
                image = dataset["images"][i]
            if str(path).startswith("/undistorted/"):
                path = str(path)[len("/undistorted/") :]
            else:
                path = str(Path(path).relative_to(Path(dataset.get("image_paths_root") or "")))
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

                if self._preview_camera is not None:
                    cam_pos, cam_wxyz, cam_fov, cam_app = self._preview_camera
                    cam_aspect = self.state.render_resolution[0] / self.state.render_resolution[1]
                else:
                    camera = client.camera
                    cam_pos, cam_wxyz, cam_fov = camera.position, camera.wxyz, camera.fov
                    cam_aspect = camera.aspect
                    cam_app = self._current_embedding

                cam_embedding = None
                if cam_app is not None:
                    def _get_embedding(app) -> Optional[np.ndarray]:
                        if self.method is None:
                            return None
                        if "embedding_train_index" in app:
                            return self.method.get_train_embedding(app["embedding_train_index"])
                        elif "weights" in app and "appearances" in app:
                            embeddings = list(map(_get_embedding, app["appearances"]))
                            if not embeddings:
                                return None
                            return reduce(lambda acc, x: (acc + x[0] * x[1] if x[1] is not None and acc is not None else None),
                                          zip(app["weights"], embeddings),
                                          (np.zeros(embeddings[0].shape, dtype=np.float32) 
                                           if len(embeddings) > 0 and embeddings[0] is not None else None))
                        else:
                            raise ValueError(f"Invalid appearance: {app}")
                    cam_embedding = _get_embedding(cam_app)

                w_total = self.resolution_slider.value
                h_total = int(self.resolution_slider.value / cam_aspect)
                focal = h_total / 2 / np.tan(cam_fov / 2)

                num_rays_total = num_rays = w_total * h_total
                w, h = w_total, h_total

                c2w = get_c2w(cam_pos, cam_wxyz)
                c2w = apply_transform(self._inv_transform, c2w)
                outputs = None

                # In state 0, we render a low resolution image to display it fast
                if render_state == 0:
                    target_fps = 12
                    fps = 1 if not self._render_times else 1.0 / np.mean(self._render_times)
                    if fps >= target_fps:
                        render_state = 1
                    else:
                        h = (fps * num_rays_total / target_fps / cam_aspect) ** 0.5
                        h = int(round(h, -1))
                        h = max(min(h_total, h), 30)
                        w = int(h * cam_aspect)
                        if w > w_total:
                            w = w_total
                            h = int(w / cam_aspect)
                        num_rays = w * h
                self._render_state[client.client_id] = render_state + 1

                nb_camera = new_cameras(
                    poses=c2w[None, :3, :4],
                    intrinsics=np.array([[focal * w/w_total, focal* h/h_total, w / 2, h / 2]], dtype=np.float32),
                    camera_types=np.array([0], dtype=np.int32),
                    distortion_parameters=np.zeros((1, 8), dtype=np.float32),
                    image_sizes=np.array([[w, h]], dtype=np.int32),
                    nears_fars=None,
                )

                try:
                    if render_state == 1:
                        scope = self._cancellation_token = EventCancellationToken()
                    else:
                        scope = contextlib.nullcontext()
                    with scope:
                        for outputs in self.method.render(nb_camera, embeddings=[cam_embedding] if cam_embedding is not None else None):
                            pass
                    self._cancellation_token = None
                    
                except CancelledException:
                    # if we got interrupted, don't send the output to the viewer
                    self._render_state.pop(client.client_id, None)
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
                        render = visualize_depth(image, expected_scale=self._expected_depth_scale)
                    elif name == "accumulation":
                        render = apply_colormap(image, pallete="coolwarm")
                    else:
                        render = image
                    return render

                # Update output options
                if set(self.state.output_type_options) != set(outputs.keys()):
                    self.state.output_type_options = tuple(sorted(outputs.keys()))
                    if self.state.split_output_type is None:
                        self.state.split_output_type = next(
                            (x for x in self.state.output_type_options if x != self.state.output_type), 
                        self.state.output_type)

                render = render_single(self.state.output_type)
                if self.state.output_split:
                    split_render = render_single(self.state.split_output_type)
                    assert render.shape == split_render.shape
                    split_point = int(render.shape[1] * self.state.split_percentage)
                    render[:, split_point:] = split_render[:, split_point:]
                    
                if self._preview_camera is not None:
                    # If the preview camera is set, we correct the aspect ratio to match client's viewport
                    render = pad_to_aspect_ratio(render, client.camera.aspect)

                client.set_background_image(render, format="jpeg")
                self._render_state[client.client_id] = min(self._render_state.get(client.client_id, 0), render_state + 1)

                if render_state == 1 or len(self._render_times) < assert_not_none(self._render_times.maxlen):
                    self._render_times.append(interval / num_rays * num_rays_total)
                self.state.fps = f"{1.0 / np.mean(self._render_times):.3g}"
                del outputs


def run_viser_viewer(method: Optional[Method] = None, 
                     data=None, 
                     port=6006,
                     nb_info=None):
    state = ViewerState()
    if nb_info is not None:
        if nb_info.get("dataset_background_color") is not None:
            bg_color = nb_info["dataset_background_color"]
            bg_color = tuple(int(x) for x in bg_color)
            state.background_color = cast(Tuple[int, int, int], bg_color)

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
            state.background_color = cast(Tuple[int, int, int], bg_color)

        if train_dataset.get("points3D_xyz") is not None:
            server.add_initial_point_cloud(train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])

        test_dataset = load_dataset(data, split="test", features=features, load_features=False)

        server.add_dataset_views(dataset_load_features(test_dataset, features), "test")
        server.add_dataset_views(dataset_load_features(train_dataset, features), "train")

    elif nb_info is not None:
        dataset_metadata = nb_info.get("dataset_metadata")
        server = build_server(dataset_metadata=dataset_metadata)
    else:
        server = build_server()
    server.run()
