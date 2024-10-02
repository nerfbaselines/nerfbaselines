import sys
import threading
import ast
import tempfile
from functools import reduce
import logging
import io
import math
import os
from functools import partial, wraps
import inspect
from dataclasses import dataclass
from collections import deque
import dataclasses
import contextlib
from pathlib import Path
from time import perf_counter
from typing import Optional, Tuple, Any, Dict, cast, List, Callable, Union, FrozenSet, TypeVar, ClassVar

import numpy as np
import viser
import viser.theme
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
from scipy import interpolate
try:
    from typing import get_args, Literal, TypeVar
except ImportError:
    from typing_extensions import get_args, Literal, TypeVar

from nerfbaselines import (
    Method, Dataset, DatasetFeature, CameraModel,
    new_cameras,
    TrajectoryFrameAppearance, TrajectoryFrame, TrajectoryKeyframe, Trajectory,
    KochanekBartelsInterpolationSource, RenderOptions,
)
from nerfbaselines.datasets import dataset_load_features, load_dataset
from nerfbaselines.datasets._colmap_utils import qvec2rotmat, rotmat2qvec
from nerfbaselines.utils import apply_transform, get_transform_and_scale, invert_transform
from nerfbaselines.utils import CancelledException, CancellationToken
from nerfbaselines.utils import apply_transform, get_transform_and_scale, invert_transform
from nerfbaselines.utils import image_to_srgb, visualize_depth, apply_colormap
from nerfbaselines.io import load_trajectory, save_trajectory
from nerfbaselines.evaluation import render_frames, trajectory_get_embeddings, trajectory_get_cameras
from ._viser_viewmodel import ViewerState, Keyframe


ControlType = Literal["object-centric", "default"]
VISER_SCALE_RATIO = 10.0
T = TypeVar("T")
TCallable = TypeVar("TCallable", bound=Callable)


def _handle_gui_error(server):
    def wrap(fn):
        def wrapped(*args, **kwargs):
            gui = server
            if isinstance(args[0], viser.GuiEvent):
                gui = server.get_clients()[args[0].client_id]
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                modal = gui.add_gui_modal("Error occured")
                with modal:
                    gui.add_gui_markdown(f"An error occured: \n{e}")
                    gui.add_gui_button("Close").on_click(lambda _: modal.close())
                logging.exception(e)
                return None
        return wrapped
    return wrap


def _pad_to_aspect_ratio(img, aspect_ratio):
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


def _get_c2w(position, wxyz):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(wxyz)
    c2w[:3, 3] = position
    return c2w


def _get_position_quaternion(c2s):
    position = c2s[..., :3, 3]
    wxyz = np.stack([rotmat2qvec(x) for x in c2s[..., :3, :3].reshape(-1, 3, 3)], 0)
    wxyz = wxyz.reshape(c2s.shape[:-2] + (4,))
    return position, wxyz


_camera_edit_panel = None


def _add_keypoint_onclick_callback(server: ViserServer, state, index, handle):
    global _camera_edit_panel

    @handle.on_click
    @_handle_gui_error(server)
    def _(_):
        global _camera_edit_panel
        keyframe = state.camera_path_keyframes[index]

        if _camera_edit_panel is not None:
            _camera_edit_panel.remove()
            _camera_edit_panel = None


        state.temporary_appearance_train_index = keyframe.appearance_train_index

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
                @_handle_gui_error(server)
                def _(_) -> None:
                    val = None
                    if train_embed_dropdown.value != "none" and train_embed_dropdown.value in state.image_names_train:
                        val = state.image_names_train.index(train_embed_dropdown.value)
                    state.camera_path_keyframes = tuple(
                        dataclasses.replace(key, appearance_train_index=val if key.appearance_train_index is None or i == index else key.appearance_train_index)
                        for i, key in enumerate(state.camera_path_keyframes)
                    )
                    state.temporary_appearance_train_index = val

            delete_button = server.add_gui_button("Delete", color="red", icon=viser.Icon.TRASH)
            go_to_button = server.add_gui_button("Go to")
            close_button = server.add_gui_button("Close")


        @override_fov.on_update
        @_handle_gui_error(server)
        def _(_) -> None:
            override_fov_degrees.disabled = not override_fov.value
            state.camera_path_keyframes = tuple(
                key if i != index else dataclasses.replace(key, fov=None if not override_fov.value else key.fov)
                for i, key in enumerate(state.camera_path_keyframes)
            )

        @override_fov_degrees.on_update
        @_handle_gui_error(server)
        def _(_) -> None:
            fov = override_fov_degrees.value
            state.camera_path_keyframes = tuple(
                key if i != index else dataclasses.replace(key, fov=None if not override_fov.value else fov)
                for i, key in enumerate(state.camera_path_keyframes)
            )

        @delete_button.on_click
        @_handle_gui_error(server)
        def _(event: viser.GuiEvent) -> None:
            global _camera_edit_panel
            assert event.client is not None
            with event.client.add_gui_modal("Confirm") as modal:
                event.client.add_gui_markdown("Delete keyframe?")
                confirm_button = event.client.add_gui_button("Yes", color="red", icon=viser.Icon.TRASH)
                exit_button = event.client.add_gui_button("Cancel")

                @confirm_button.on_click
                @_handle_gui_error(server)
                def _(_) -> None:
                    global _camera_edit_panel
                    assert camera_edit_panel is not None

                    state.temporary_appearance_train_index = None
                    camera_edit_panel.remove()
                    _camera_edit_panel = None
                    modal.close()
                    state.camera_path_keyframes = tuple(key for i, key in enumerate(state.camera_path_keyframes) if i != index)

                @exit_button.on_click
                def _(_) -> None:
                    modal.close()

        @go_to_button.on_click
        @_handle_gui_error(server)
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
            state.temporary_appearance_train_index = None
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


    args = ["camera_path_keyframes", "render_fov", "render_resolution", "camera_path_move_keyframes"]
    state.on_update(lambda _: update_keyframes([getattr(state, x) for x in args]), tuple(args))
    state.on_update(lambda _: [setattr(h[1], "visible", state.camera_path_show_keyframes) for h in keyframe_handles if h[1] is not None], ("camera_path_show_keyframes",))
    update_keyframes([getattr(state, x) for x in args])


def _add_spline_to_camera_path(server: ViserServer, state: ViewerState):
    # Clear prior spline nodes.
    spline_nodes = []

    def add_path_spline(_):
        for node in spline_nodes:
            node.remove()
        spline_nodes.clear()
        if state.camera_path_splines is None or not state.camera_path_show_spline:
            return
        if state.camera_path_interpolation == "none":
            return
        points = state.camera_path_splines[0]
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

    state.on_update(add_path_spline, ("camera_path_splines", "camera_path_show_spline"))
    add_path_spline(state.camera_path_splines)


BindableSource = Any
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


class ViewerRenderer:
    def __init__(self,
                 method: Optional[Method],
                 expected_depth_scale=0.5):
        self.method = method
        self._expected_depth_scale = expected_depth_scale
        self._cancellation_token = None
        self._output_type_options = ()
        self._task_queue = []
        if self.method is not None:
            method_info = self.method.get_info()
            self._output_types = {
                (self.method, x if isinstance(x, str) else x["name"]): {"name": x, "type": x} if isinstance(x, str) else x for x in method_info.get("supported_outputs", ("color",))}
        else:
            self._output_types = {}
    def update(self):
        if not self._task_queue:
            return
        
        task = self._task_queue.pop(0)
        task_type = task.pop("type")
        error_callback = task.pop("error_callback", None)
        try:
            if task_type == "render_video":
                self._render_video(**task)
            else:
                raise ValueError(f"Unknown task type: {task['type']}")
        except Exception as e:
            if error_callback is not None:
                error_callback(e)
            else:
                raise

    def _render_video(self, trajectory, callback):
        if self.method is None:
            raise ValueError("No method to render video")
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "video.mp4")
            # Embed the appearance
            embeddings = trajectory_get_embeddings(self.method, trajectory)
            cameras = trajectory_get_cameras(trajectory)
            render_frames(self.method, 
                          cameras, 
                          embeddings=embeddings, 
                          output=output, 
                          output_names=("color",), 
                          nb_info={}, 
                          fps=trajectory["fps"])
            logging.info(f"Output saved to {output}")
            with open(output, "rb") as file:
                data = file.read()
        callback(data)

    def add_render_video_task(self, trajectory: Trajectory, callback, error_callback=None):
        self._task_queue.append({
            "type": "render_video",
            "trajectory": trajectory,
            "callback": callback,
            "error_callback": error_callback,
        })

    @property
    def output_type_options(self):
        return self._output_type_options

    def render(self, 
               camera, *, 
               embedding=None, 
               allow_cancel=False,
               output_type=None, 
               background_color=None,
               split_output_type: Optional[str] = None,
               split_percentage: Optional[float] = 0.5,
               output_aspect_ratio: Optional[float] = None):
        if self.method is None:
            # No need to render anything
            return None

        try:
            if allow_cancel:
                scope = self._cancellation_token = CancellationToken()
            else:
                scope = contextlib.nullcontext()
            with scope:
                options: RenderOptions = { "output_type_dtypes": { "color": "uint8" }, 
                                           "embedding": embedding }
                outputs = self.method.render(camera, options=options)
            self._cancellation_token = None
            
        except CancelledException:
            # if we got interrupted, don't send the output to the viewer
            return None
        assert outputs is not None, "Method did not return any outputs"

        def render_single(name):
            assert outputs is not None, "Method did not return any outputs"
            assert self.method is not None, "No method to render"
            name = name or "color"
            rtype_spec = self._output_types.get((self.method, name))
            if rtype_spec is None:
                raise ValueError(f"Unknown output type: {name}")
            rtype = rtype_spec.get("type", name)
            image = outputs[name]
            if rtype == "color":
                bg_color = np.array(background_color, dtype=np.uint8)
                render = image_to_srgb(image, np.uint8, color_space="srgb", allow_alpha=False, background_color=bg_color)
            elif rtype == "depth":
                # Blend depth with correct color pallete
                render = visualize_depth(image, expected_scale=self._expected_depth_scale)
            elif rtype == "accumulation":
                render = apply_colormap(image, pallete="coolwarm")
            else:
                render = image
            return render

        render = render_single(output_type)
        if split_output_type is not None:
            split_render = render_single(split_output_type)
            assert render.shape == split_render.shape, f"Output shapes do not match: {render.shape} vs {split_render.shape}"
            split_percentage_ = split_percentage if split_percentage is not None else 0.5
            split_point = int(render.shape[1] * split_percentage_)
            render[:, split_point:] = split_render[:, split_point:]

        if output_aspect_ratio is not None:
            # If the preview camera is set, we correct the aspect ratio to match client's viewport
            render = _pad_to_aspect_ratio(render, output_aspect_ratio)

        return render


class ViserViewer:
    def __init__(self, 
                 method: Optional[Method], 
                 port, 
                 dataset_metadata=None,
                 state=None):

        self.transform = np.eye(4, dtype=np.float32)
        self.initial_pose = None
        self._dataset_metadata = dataset_metadata

        control_type = "default"
        expected_depth_scale = 0.5
        if dataset_metadata is not None:
            self.transform = dataset_metadata.get("viewer_transform").copy()
            self.initial_pose = dataset_metadata.get("viewer_initial_pose").copy()
            control_type = "object-centric" if dataset_metadata.get("type") == "object-centric" else "default"
            self.initial_pose[:3, 3] *= VISER_SCALE_RATIO
            expected_depth_scale = dataset_metadata.get("expected_scene_scale", 0.5)

        self.transform[:3, :] *= VISER_SCALE_RATIO
        self._inv_transform = invert_transform(self.transform, True)

        self.port = port
        self.method = method
        self.renderer = ViewerRenderer(method, expected_depth_scale=expected_depth_scale)

        self.state = state or ViewerState()
        if self.method is not None:
            method_info = self.method.get_info()
            self.state.output_type_options = tuple(sorted([
                x if isinstance(x, str) else x["name"]
                for x in method_info.get("supported_outputs", ("color",))
            ]))
            if "color" not in self.state.output_type_options:
                self.state.output_type = next(iter(self.state.output_type_options))
            else:
                self.state.output_type = "color"
            self.state.split_output_type = next(
                (x for x in self.state.output_type_options if x != self.state.output_type), 
                self.state.output_type)
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
            def _(_: viser.CameraHandle):
                self._render_state.pop(client.client_id, None)
                self._reset_render(False)

            if self.initial_pose is not None:
                pos, quat = _get_position_quaternion(
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

        def _fix_current_frame(_):
            camera_path = self.state.camera_path_splines
            max_frame = camera_path[0].shape[0] - 1 if camera_path is not None else 0
            if self.state.preview_current_frame > max_frame:
                self.state.preview_current_frame = max_frame
        # TODO:
        self.state.on_update(_fix_current_frame, ("camera_path_splines",))
        self._build_gui()
        self._start_preview_timer()

    def _build_gui(self):
        self.server.configure_theme(
            control_layout="collapsible",
            dark_mode=True,
            brand_color=(255, 211, 105),
        )

        _build_viser_gui(self.server, self.state)
        _add_spline_to_camera_path(self.server, self.state)
        self._add_state_handlers()

    def _start_preview_timer(self):
        # Add preview_is_playing timer
        max_fps = 30

        def play() -> None:
            start = None
            start_frame = None
            i = 0
            # while self.state is not None:
            while True:
                i+=1
                max_frame = (
                    1 if self.state.camera_path_splines is None 
                    else self.state.camera_path_splines[0].shape[0] - 1
                )
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
        self.state.on_update(lambda _: _update_handles_visibility("train", self.state.show_train_cameras), 
            ("show_train_cameras",))
        self.state.on_update(lambda _: _update_handles_visibility("test", self.state.show_test_cameras),
            ("show_test_cameras",))

        # Add handler to update render on render panel change
        self.state.on_update(lambda _: self._reset_render(), 
            ("render_fov", "output_type", "resolution", "output_split", "split_output_type", "background_color"))

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
                if points.shape[-1] == 4:
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
        self.state.on_update(_update_points_3D, ("input_points", "show_input_points"))

        # Add frustums handlers
        frustums = {}
        old_camimgs = {}

        @_handle_gui_error(self.server)
        def _set_view_to_camera(handle: viser.SceneNodePointerEvent[viser.CameraFrustumHandle]):
            with handle.client.atomic():
                handle.client.camera.position = handle.target.position
                handle.client.camera.wxyz = handle.target.wxyz
            self._reset_render()

        @_handle_gui_error(self.server)
        def _update_frustums(split):
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
                    pos, quat = _get_position_quaternion(c2w)
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

        self.state.on_update(lambda _: _update_frustums("train"), ("show_train_cameras", "camera_frustums_train"))
        self.state.on_update(lambda _: _update_frustums("test"), ("show_test_cameras", "camera_frustums_test"))

        # Add preview camera handler
        preview_camera_handle = None
        
        def _update_preview_handle(_) -> None:
            nonlocal preview_camera_handle
            trajectory = self.state.camera_path_splines
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
        self.state.on_update(_update_preview_handle, ("camera_path_splines", "preview_current_frame", "render_resolution"))

        # Add preview render handler
        # It hides all scene nodes and backup/restores the camera states before/after preview
        camera_pose_backup = {}

        def _update_preview(_) -> None:
            if self.state.preview_render:
                # Back up and then set camera poses.
                points, quats, fovs, weights, *_ = self.state.camera_path_splines
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

        self.state.on_update(
            lambda _: self.server.set_global_scene_node_visibility(not self.state.preview_render), 
            ("preview_render",))
        self.server.set_global_scene_node_visibility(not self.state.preview_render)
        self.state.on_update(_update_preview, ("camera_path_splines", "preview_render", "preview_current_frame", "preview_is_playing"))

        # Add keypoint handles
        _add_keyframe_frustums_to_camera_path(self.server, self.state)

        def _update_fov(fov):
            if not self.state.preview_render:
                for client in self.server.get_clients().values():
                    client.camera.fov = fov / 180 * math.pi
        self.state.on_update(lambda _: _update_fov(self.state.render_fov), ("render_fov",))

        def _update_render_appearance_train_index(_):
            index = self.state.render_appearance_train_index
            temp_index = self.state.temporary_appearance_train_index
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
        self.state.on_update(_update_render_appearance_train_index, ("render_appearance_train_index", "temporary_appearance_train_index"))

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
        self.renderer.update()

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

                c2w = _get_c2w(cam_pos, cam_wxyz)
                c2w = apply_transform(self._inv_transform, c2w)

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
                    camera_models=np.array([0], dtype=np.int32),
                    distortion_parameters=np.zeros((1, 8), dtype=np.float32),
                    image_sizes=np.array([[w, h]], dtype=np.int32),
                    nears_fars=None,
                )
                    
                output_aspect = None
                if self._preview_camera is not None:
                    # If the preview camera is set, we correct the aspect ratio to match client's viewport
                    output_aspect = client.camera.aspect

                render = self.renderer.render(
                    nb_camera,
                    embedding=cam_embedding,
                    background_color=self.state.background_color,
                    allow_cancel=render_state == 1,
                    output_type=self.state.output_type, 
                    split_output_type=self.state.split_output_type if self.state.output_split else None,
                    split_percentage=self.state.split_percentage if self.state.output_split else None,
                    output_aspect_ratio=output_aspect)

                # if we got interrupted, don't send the output to the viewer
                if render is None:
                    self._render_state.pop(client.client_id, None)
                    continue

                interval = perf_counter() - start
                client.set_background_image(render, format="jpeg")
                self._render_state[client.client_id] = min(self._render_state.get(client.client_id, 0), render_state + 1)

                assert self._render_times.maxlen is not None, "Render times should have a maximum length"
                if render_state == 1 or len(self._render_times) < self._render_times.maxlen:
                    self._render_times.append(interval / num_rays * num_rays_total)
                del render

        # Update FPS and output options
        self.state.fps = f"{1.0 / np.mean(self._render_times):.3g}"


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

        supported_camera_models = frozenset(get_args(CameraModel))
        server.add_dataset_views(dataset_load_features(test_dataset, features, supported_camera_models=supported_camera_models), "test")
        server.add_dataset_views(dataset_load_features(train_dataset, features, supported_camera_models=supported_camera_models), "train")

    elif nb_info is not None:
        dataset_metadata = nb_info.get("dataset_metadata")
        server = build_server(dataset_metadata=dataset_metadata)
    else:
        server = build_server()
    server.run()


@dataclasses.dataclass(frozen=True)
class Bindable:
    path: str
    twoway: bool = False


def _wrap_add_component(state: ViewerState, add_gui: TCallable) -> TCallable:
    def _add_gui(*args, **kwargs):
        self = getattr(add_gui, "__self__", None)
        if self is None and len(args) > 0:
            self, args = args[0], args[1:]
        assert len(args) == 0, "Positional arguments are not supported"

        prop_bindings: Dict[str, Bindable] = {}
        del args
        for name, arg in kwargs.items():
            if isinstance(arg, Bindable):
                prop_bindings[name] = arg

        def _current_kwargs():
            _kwargs = kwargs.copy()
            for name in kwargs.keys():
                if name in prop_bindings:
                    _kwargs[name] = getattr(state, prop_bindings[name].path)
            return _kwargs

        handle = None
        handle_update = None
        orig_kwargs = _current_kwargs()

        # Capture container_id
        gui = getattr(self, "gui", self)
        container_id = gui._get_container_id()

        def _update_component(names):
            nonlocal handle
            # We call the special set_prop method only if there is a single updatable property
            changed_props = None
            if names is not None:
                changed_props = set(
                    x for x, y in prop_bindings.items() 
                    if y.path in names and getattr(state, y.path) != getattr(handle, "value" if x == "initial_value" else x, orig_kwargs[x]))
                if not changed_props:
                    return

            def _handle_supports_update(p):
                # This should really be supported in viser>0.2.0
                if handle is None:
                    return False
                if p == "initial_value":
                    p = "value"
                proptype = getattr(type(handle), p, None)
                if isinstance(proptype, property) and proptype.fset is not None:
                    return True
                return False

            if (names is not None and
                    all(_handle_supports_update(x) for x in changed_props)):
                for name in changed_props:
                    prop_name = "value" if name == "initial_value" else name
                    value = getattr(state, prop_bindings[name].path)
                    setattr(handle, prop_name, value)
                    logging.debug("Updating component", add_gui, "value", "->", value)
                return
            if handle is not None:
                handle.remove()
                handle = None
            old_container_id = gui._get_container_id()
            try:
                # Mock the container_id
                gui._set_container_id(container_id)
                handle = add_gui(**_current_kwargs())
            finally:
                gui._set_container_id(old_container_id)
            # Add callbacks
            for name, callbacks in _collected_callbacks.items():
                for callback in callbacks:
                    getattr(handle, name)(callback)
            logging.debug("Creating component", add_gui, "changed names:", names)
            if handle_update is not None:
                handle.on_update(handle_update)
            return handle

        _update_callback = None
        _collected_callbacks = {}
        if any(prop_bindings):
            _update_callback = state.on_update(_update_component, tuple(x.path for x in prop_bindings.values()))
        update_value_binding = next((y for x, y in prop_bindings.items() if y.twoway and x in {"value", "initial_value"}), None)
        if update_value_binding is not None:
            handle_update = lambda _: setattr(state, update_value_binding.path, handle.value)
        _update_component(None)
        assert handle is not None, "Failed to build component"
        if "order" not in kwargs and hasattr(handle, "order"):
            kwargs["order"] = handle.order

        class ProxyHandle:
            def __getattribute__(self, name):
                out = object.__getattribute__(handle, name)
                if name.startswith("on_"):
                    if name not in _collected_callbacks:
                        _collected_callbacks[name] = []
                    def _add_callback(callback):
                        _collected_callbacks[name].append(callback)
                        return out(callback)
                    return _add_callback
                return out

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
                nonlocal _update_callback
                if _update_callback is not None:
                    _update_callback.remove()
                    _update_callback = None
                if handle is not None:
                    handle.remove()
                    handle = None

        return ProxyHandle()
    return _add_gui


def _wrap_state_bindable(state: ViewerState, twoway=False) -> ViewerState:
    del state
    class _:
        def __getattr__(self, name):
            return Bindable(name, twoway=twoway)
    return _()


def _build_viser_gui(server: viser.ViserServer, state: ViewerState):
    _c = partial(_wrap_add_component, state)
    _b = _wrap_state_bindable(state)
    _b2 = _wrap_state_bindable(state, twoway=True)

    _c(server.add_gui_text)(label="FPS", initial_value=_b.fps, disabled=True)
    tabs = server.add_gui_tab_group()
    with tabs.add_tab("Control", viser.Icon.SETTINGS):
        with server.add_gui_folder("Render Options"):
            _c(server.add_gui_slider)(
                label="Max res",
                min=64,
                max=2048,
                step=100,
                initial_value=_b2.resolution,
                hint="Maximum resolution to render in viewport",
            )
            _c(server.add_gui_dropdown)(
                label="Output type",
                options=_b.output_type_options,
                initial_value=_b2.output_type,
                hint="The output to render",
            )
            _c(server.add_gui_rgb)(
                label="Background color",
                initial_value=_b2.background_color,
                hint="Color of the background",
            )

        with server.add_gui_folder("Split Screen"):
            _c(server.add_gui_checkbox)(
                label="Enable", initial_value=_b2.output_split, hint="Render two outputs"
            )
            _c(server.add_gui_slider)(
                label="Split percentage",
                initial_value=_b2.split_percentage,
                min=0.0,
                max=1.0,
                step=0.01,
                hint="Where to split",
            )
            _c(server.add_gui_dropdown)(
                label="Output render split",
                options=_b.output_type_options,
                initial_value=_b2.split_output_type,
                hint="The second output",
            )

        _c(server.add_gui_checkbox)(
            label="Show train cams",
            initial_value=_b2.show_train_cameras,
            disabled=_b.show_train_cameras_disabled,
        )
        _c(server.add_gui_checkbox)(
            label="Show test cams",
            initial_value=_b2.show_test_cameras,
            disabled=_b.show_test_cameras_disabled,
        )
        _c(server.add_gui_checkbox)(
            label="Show input PC",
            initial_value=_b2.show_input_points,
            disabled=_b.show_input_points_disabled,
        )

    with tabs.add_tab("Trajectory", viser.Icon.CAMERA):
        with _c(server.add_gui_folder)(
            label="Selected Camera",
            visible=_b.camera_path_has_selected_camera):
            _c(server.add_gui_button)(
                label="Remove",
                icon=viser.Icon.TRASH,
                color="red",
                hint="Remove the selected camera from the path.",
            ).on_click(lambda _: setattr(state, "camera_path_selected_camera", None))

        _c(server.add_gui_slider)(
            label="Default FOV",
            initial_value=_b2.render_fov,
            min=0.1,
            max=175.0,
            step=0.01,
            hint="Field-of-view for rendering, which can also be overridden on a per-keyframe basis.",
        )

        _c(server.add_gui_vector2)(
            label="Resolution",
            initial_value=_b2.render_resolution,
            min=(50, 50),
            max=(10_000, 10_000),
            step=1,
            hint="Render output resolution in pixels.",
        )

        # TODO: ...
        # if self.state.supports_appearance_from_train_images:
        #     _make_train_image_embedding_dropdown(server, self.state, self.state.b.render_appearance_train_index)

        add_button = server.add_gui_button(
            label="Add Keyframe",
            icon=viser.Icon.PLUS,
            hint="Add a new keyframe at the current pose.",
        )

        @add_button.on_click
        @_handle_gui_error(server)
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            camera = server.get_clients()[event.client_id].camera

            # Add this camera to the path.
            apperance_train_index = state.render_appearance_train_index
            if apperance_train_index is not None and len(state.camera_path_keyframes) > 0:
                apperance_train_index = state.camera_path_keyframes[-1].appearance_train_index
            state.camera_path_keyframes = state.camera_path_keyframes + (Keyframe(
                position=camera.position,
                wxyz=camera.wxyz,
                appearance_train_index=apperance_train_index,
            ),)

        clear_keyframes_button = server.add_gui_button(
            label="Clear Keyframes",
            icon=viser.Icon.TRASH,
            hint="Remove all keyframes from the render path.",
        )

        @clear_keyframes_button.on_click
        @_handle_gui_error(server)
        def _(event: viser.GuiEvent) -> None:
            assert event.client_id is not None
            client = server.get_clients()[event.client_id]
            with client.atomic(), client.add_gui_modal("Confirm") as modal:
                client.add_gui_markdown("Clear all keyframes?")
                confirm_button = client.add_gui_button("Yes", color="red", icon=viser.Icon.TRASH)
                exit_button = client.add_gui_button("Cancel")

                @confirm_button.on_click
                def _(_) -> None:
                    state.camera_path_keyframes = ()
                    modal.close()

                @exit_button.on_click
                def _(_) -> None:
                    modal.close()

        _c(server.add_gui_dropdown)(
            label="Interpolation",
            initial_value=_b2.camera_path_interpolation,
            options=("kochanek-bartels", "none", "ellipse"),
            hint="Camera path interpolation.")

        _c(server.add_gui_checkbox)(
            label="Loop",
            initial_value=_b2.camera_path_loop,
            visible=_b.camera_path_loop_visible,
            hint="Add a segment between the first and last keyframes.")

        _c(server.add_gui_slider)(
            label="Spline tension",
            min=0.0,
            max=1.0,
            initial_value=_b2.camera_path_tension,
            visible=_b.camera_path_tension_visible,
            step=0.01,
            hint="Tension parameter for adjusting smoothness of spline interpolation.",
        )
        _c(server.add_gui_checkbox)(
            label="Move keyframes",
            initial_value=_b2.camera_path_move_keyframes,
            hint="Toggle move handles for keyframes in the scene.",
        )
        _c(server.add_gui_checkbox)(
            label="Show keyframes",
            initial_value=_b2.camera_path_show_keyframes,
            hint="Show keyframes in the scene.",
        )
        _c(server.add_gui_checkbox)(
            label="Show spline",
            initial_value=_b2.camera_path_show_spline,
            visible=_b.camera_path_show_spline_visible,
            hint="Show camera path spline in the scene.",
        )

        with server.add_gui_folder("Playback"):
            _c(server.add_gui_button)(
                label="Play", 
                icon=viser.Icon.PLAYER_PLAY,
                disabled=_b.preview_disabled,
                visible=_b.preview_is_not_playing
            ).on_click(lambda _: setattr(state, "preview_is_playing", True))
            _c(server.add_gui_button)(
                label="Pause", 
                icon=viser.Icon.PLAYER_PAUSE, 
                visible=_b.preview_is_playing
            ).on_click(lambda _: setattr(state, "preview_is_playing", False))
            _c(server.add_gui_button)(
                label="Preview Render", 
                hint="Show a preview of the render in the viewport.",
                disabled=_b.preview_disabled,
                visible=_b.preview_render_inverted,
            ).on_click(lambda _: setattr(state, "preview_render", True))
            _c(server.add_gui_button)(
                label="Exit Render Preview", 
                color="red", 
                visible=_b.preview_render,
            ).on_click(lambda _: setattr(state, "preview_render", False))

            _c(server.add_gui_slider)(
                label="Preview frame",
                min=0,
                step=1,
                initial_value=_b2.preview_current_frame,
                disabled=_b.preview_disabled,
                max=_b.camera_path_num_frames,
            )

            _c(server.add_gui_number)(
                label="Transition (sec)",
                min=0.001,
                max=30.0,
                step=0.001,
                initial_value=_b2.camera_path_default_transition_duration,
                hint="Time in seconds between each keyframe, which can also be overridden on a per-transition basis.",
            )
            _c(server.add_gui_number)(
                label="FPS", min=0.1, max=240.0, step=1e-2, 
                visible=_b.camera_path_framerate_visible,
                initial_value=_b2.camera_path_framerate)
            framerate_buttons = _c(server.add_gui_button_group)(
                label="",
                options=("24", "30", "60"),
                visible=_b.camera_path_framerate_visible)
            framerate_buttons.on_click(lambda _: setattr(state, "camera_path_framerate", float(framerate_buttons.value)))
            _c(server.add_gui_number)(
                label="Duration (sec)",
                min=0.0,
                max=1e8,
                step=0.001,
                disabled=True,
                initial_value=_b.camera_path_duration,
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
            state.load_trajectory(trajectory, self.transform)

        export_button = server.add_gui_button(
            "Export trajectory",
            icon=viser.Icon.FILE_EXPORT,
            hint="Export trajectory file.",
        )

        @export_button.on_click
        @_handle_gui_error(server)
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None

            inv_transform = invert_transform(self.transform, True)
            trajectory = state.get_trajectory(inv_transform)
            with io.BytesIO() as file, io.TextIOWrapper(file, encoding="utf8") as textfile:
                save_trajectory(trajectory, textfile)
                textfile.flush()
                data = file.getvalue()

            # now write the json file
            server.send_file_download("trajectory.json", data)

        render_button = server.add_gui_button(
            label="Render video",
            color="green",
            icon=viser.Icon.CAMERA,
            hint="Render the scene.",
        )

        @render_button.on_click
        @_handle_gui_error(server)
        def _(event: viser.GuiEvent) -> None:
            assert event.client is not None
            assert event.client_id is not None

            gui = server.get_clients()[event.client_id]
            modal = gui.add_gui_modal("Rendering video")
            inv_transform = invert_transform(self.transform, True)
            trajectory = state.get_trajectory(inv_transform)
            def _send_video(data):
                modal.close()
                server.send_file_download("video.mp4", data)

            @_handle_gui_error(server)
            def _error(error):
                modal.close()
                raise error
            self.renderer.add_render_video_task(trajectory, callback=_send_video, error_callback=_error)


server = viser.ViserServer()
state = ViewerState()
_build_viser_gui(server, state)
# _add_state_handlers(server, state)
