import contextlib
import numpy as np
import dataclasses
from typing import Any, Dict, List, Optional, Tuple, cast
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
from nerfbaselines.utils import apply_transform, get_transform_and_scale, invert_transform, pad_poses
from nerfbaselines.utils import image_to_srgb, visualize_depth, apply_colormap
from nerfbaselines.io import load_trajectory, save_trajectory
from nerfbaselines.evaluation import render_frames, trajectory_get_embeddings, trajectory_get_cameras
from scipy import interpolate
import splines
import splines.quaternion
import viser.transforms as tf


def _three_js_perspective_camera_focal_length(fov: float, image_height: int):
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




def _get_c2w(position, wxyz):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(wxyz)
    c2w[:3, 3] = position
    return c2w


def _safe_eq(a, b) -> bool:
    """Check if a and b are equal, even if they are numpy arrays"""
    if type(a) != type(b):
        return False
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a.shape == b.shape and (a == b).all()
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_safe_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _safe_eq(v, b[k]) for k, v in a.items())
    if hasattr(a, "__eq__"):
        return a == b
    return False


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
    del U, S
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
    poses = np.stack([_get_c2w(k.position, k.wxyz) for k in camera_path_keyframes], axis=0)
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
       return all(_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))


class ViewModel:
    def __init__(self):
        self._changed_properties = None
        self._update_callbacks = []

    def on_update(self, callback, dependencies):
        self._update_callbacks.append((set(dependencies), callback))

    @contextlib.contextmanager
    def atomic(self):
        is_outer = False
        if self._changed_properties is None:
            self._changed_properties = set()
            is_outer = True
        yield
        if is_outer:
            for callback in self._update_callbacks:
                callback(self)

    def __setattr__(self, __name, __value):
        with self.atomic():
            (self._changed_properties or set()).add(__name)
            object.__setattr__(self, __name, __value)


@dataclasses.dataclass(eq=True)
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
    render_resolution: Tuple[int, int] = (1920, 1080)
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
            if appearance and appearance.get("embedding_train_index", None) is None:
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
        trajectory_frames = self.camera_path_splines(self)
        if trajectory_frames is not None:
            frames = []
            for pos, wxyz, fov, weights in zip(*trajectory_frames):
                pose = tf.SE3.from_rotation_and_translation(
                    tf.SO3(wxyz),
                    pos,
                ).as_matrix()
                pose = apply_transform(inv_transform, pose)
                focal_length = _three_js_perspective_camera_focal_length(fov, h)
                intrinsics = np.array([focal_length, focal_length, w / 2, h / 2], dtype=np.float32)
                frames.append(TrajectoryFrame({
                    "pose": pose[:3, :].astype(np.float32),
                    "intrinsics": intrinsics,
                    "appearance_weights": weights.astype(np.float32),
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
            if source["interpolation"] == "none":
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
                if _safe_eq(nval, last_value):
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
       return all(_safe_eq(a1, a2) for a1, a2 in zip(t1, t2))


@dataclasses.dataclass(eq=True)
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
    render_resolution: Tuple[int, int] = (1920, 1080)
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
            if appearance and appearance.get("embedding_train_index", None) is None:
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
        trajectory_frames = self.camera_path_splines(self)
        if trajectory_frames is not None:
            frames = []
            for pos, wxyz, fov, weights in zip(*trajectory_frames):
                pose = tf.SE3.from_rotation_and_translation(
                    tf.SO3(wxyz),
                    pos,
                ).as_matrix()
                pose = apply_transform(inv_transform, pose)
                focal_length = _three_js_perspective_camera_focal_length(fov, h)
                intrinsics = np.array([focal_length, focal_length, w / 2, h / 2], dtype=np.float32)
                frames.append(TrajectoryFrame({
                    "pose": pose[:3, :].astype(np.float32),
                    "intrinsics": intrinsics,
                    "appearance_weights": weights.astype(np.float32),
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
            if source["interpolation"] == "none":
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

    # Computed properties
    @computed_property(cache=True)
    def camera_path_duration(self) -> float:
        if self.camera_path_interpolation == "none":
            return len(self.camera_path_keyframes) * self.camera_path_default_transition_duration
        kf = self.camera_path_keyframes
        if not self.camera_path_loop:
            kf = kf[1:]
        return sum(
            k.transition_duration
            if k.transition_duration is not None
            else self.camera_path_default_transition_duration
            for k in kf
        )

    @computed_property(cache=True)
    def camera_path_splines(self):
        if len(self.camera_path_keyframes) < 2:
            return None

        # For none interpolation, we just return the keyframes.
        if self.camera_path_interpolation == "none":
            return (
                np.array([k.position for k in self.camera_path_keyframes], dtype=np.float32),
                np.array([k.wxyz for k in self.camera_path_keyframes], dtype=np.float32),
                np.array([k.fov if k.fov is not None else self.render_fov for k in self.camera_path_keyframes], dtype=np.float32),
                np.array([_onehot(i, len(self.camera_path_keyframes)) for i in range(len(self.camera_path_keyframes))],
                dtype=np.float32),
            )

        if self.camera_path_interpolation == "ellipse":
            num_frames = int(self.camera_path_default_transition_duration * self.camera_path_framerate)
            return _interpolate_ellipse(self.camera_path_keyframes, num_frames, self.render_fov)


        # Compute transition times cumsum
        times = np.array([
            k.transition_duration if k.transition_duration is not None else self.camera_path_default_transition_duration 
            for k in self.camera_path_keyframes], dtype=np.float32)
        transition_times_cumsum = np.cumsum(np.roll(times, -1)) if self.camera_path_loop else np.cumsum(times[:-1])
        transition_times_cumsum = np.insert(transition_times_cumsum, 0, 0.0)
        del times

        num_frames = int(transition_times_cumsum[-1] * self.camera_path_framerate)
        if num_frames <= 0 or len(self.camera_path_keyframes) < 2:
            return None

        orientation_spline = splines.quaternion.KochanekBartels(
            [
                splines.quaternion.UnitQuaternion.from_unit_xyzw(np.roll(k.wxyz, shift=-1))
                for k in self.camera_path_keyframes
            ],
            tcb=(self.camera_path_tension, 0.0, 0.0),
            endconditions="closed" if self.camera_path_loop else "natural",
        )
        position_spline = splines.KochanekBartels(
            [k.position for k in self.camera_path_keyframes],
            tcb=(self.camera_path_tension, 0.0, 0.0),
            endconditions="closed" if self.camera_path_loop else "natural",
        )
        fov_spline = splines.KochanekBartels(
            [
                k.fov if k.fov is not None else self.render_fov
                for k in self.camera_path_keyframes
            ],
            tcb=(self.camera_path_tension, 0.0, 0.0),
            endconditions="closed" if self.camera_path_loop else "natural",
        )
        weight_spline = splines.KochanekBartels(
            [
                _onehot(i, len(self.camera_path_keyframes)) for i in range(len(self.camera_path_keyframes))
            ],
            tcb=(self.camera_path_tension, 0.0, 0.0),
            endconditions="closed" if self.camera_path_loop else "natural",
        )

        # Get time splines
        spline_indices = np.arange(transition_times_cumsum.shape[0])
        gtime = np.linspace(0, transition_times_cumsum[-1], num_frames)
        if self.camera_path_loop:
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
