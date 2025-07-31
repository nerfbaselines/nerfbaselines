import io
import base64
import hashlib
import gzip
import msgpack
import warnings
import pprint
import logging
import struct
import math
import json
import os
from pathlib import Path
from typing import Optional, Union, cast
try:
    from typing import get_origin, get_args
except ImportError:
    from typing_extensions import get_origin, get_args

import tempfile
import numpy as np
from PIL import Image, ImageOps
from nerfbaselines import (
    Dataset, Method, MethodInfo, ModelInfo, RenderOutput,
    Cameras, camera_model_to_int
)
from nerfbaselines.utils import pad_poses, unpad_poses


def numpy_to_base64(array: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, array)
        return base64.b64encode(f.getvalue()).decode("ascii")


def numpy_from_base64(data: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(data)) as f:
        return np.load(f)


def cast_value(tp, value):
    origin = get_origin(tp)
    if origin is Union:
        for t in get_args(tp):
            try:
                return cast_value(t, value)
            except ValueError:
                pass
            except TypeError:
                pass
        raise TypeError(f"Value {value} is not in {tp}")
    if tp is type(None):
        if str(value).lower() == "none":
            return None
        else:
            raise TypeError(f"Value {value} is not None")
    if tp is bool:
        if str(value).lower() in {"true", "1", "yes"}:
            return True
        elif str(value).lower() in {"false", "0", "no"}:
            return False
        else:
            raise TypeError(f"Value {value} is not a bool")
    if tp in {int, float, bool, str}:
        return tp(value)
    if isinstance(value, tp):
        return value
    raise TypeError(f"Cannot cast value {value} to type {tp}")


def flatten_hparams(hparams, *, separator: str = "/", _prefix: str = ""):
    flat = {}
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if isinstance(v, dict):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator).items())
        else:
            flat[k] = v
    return flat


def rotmat(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # handle exception for the opposite direction input
    if c < -1 + 1e-10:
        return rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2 + 1e-10))


def closest_point_2_lines(oa, da, ob, db):  # returns point closest to both rays of form o+t*d, and a weight factor that goes to 0 if the lines are parallel
    da = da / np.linalg.norm(da)
    db = db / np.linalg.norm(db)
    c = np.cross(da, db)
    denom = np.linalg.norm(c) ** 2
    t = ob - oa
    ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
    tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
    if ta > 0:
        ta = 0
    if tb > 0:
        tb = 0
    return (oa + ta * da + ob + tb * db) * 0.5, denom


def srgb_to_linear(img):
    limit = 0.04045
    return np.where(img > limit, np.power((img + 0.055) / 1.055, 2.4), img / 12.92)


def linear_to_srgb(img):
    limit = 0.0031308
    return np.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)


def get_transforms(dataset: Dataset, dataparser_transform=None, dataparser_scale=None, aabb_scale=None, keep_coords=None, **kwargs):
    frames = []
    up = np.zeros(3)

    for i in range(len(dataset["image_paths"])):
        camera = {}
        camera["w"] = int(dataset["cameras"].image_sizes[i, 0])
        camera["h"] = int(dataset["cameras"].image_sizes[i, 1])
        camera["fl_x"] = float(dataset["cameras"].intrinsics[i, 0])
        camera["fl_y"] = float(dataset["cameras"].intrinsics[i, 1])
        camera["cx"] = dataset["cameras"].intrinsics[i, 2].item()
        camera["cy"] = dataset["cameras"].intrinsics[i, 3].item()
        camera["k1"] = 0
        camera["k2"] = 0
        camera["p1"] = 0
        camera["p2"] = 0
        camera["k3"] = 0
        camera["k4"] = 0
        camera["is_fisheye"] = False
        cam_type = dataset["cameras"].camera_models[i]
        if cam_type == camera_model_to_int("pinhole"):
            pass
        elif cam_type in {camera_model_to_int("opencv"), camera_model_to_int("opencv_fisheye")}:
            camera["k1"] = dataset["cameras"].distortion_parameters[i, 0].item()
            camera["k2"] = dataset["cameras"].distortion_parameters[i, 1].item()
            camera["p1"] = dataset["cameras"].distortion_parameters[i, 2].item()
            camera["p2"] = dataset["cameras"].distortion_parameters[i, 3].item()
            camera["k3"] = dataset["cameras"].distortion_parameters[i, 4].item()
            camera["k4"] = dataset["cameras"].distortion_parameters[i, 5].item()
            if cam_type == camera_model_to_int("opencv_fisheye"):
                camera["is_fisheye"] = True
        else:
            raise NotImplementedError(f"Camera model {cam_type} not supported")
        # fl = 0.5 * w / tan(0.5 * angle_x);
        camera["camera_angle_x"] = math.atan(camera["w"] / (camera["fl_x"] * 2)) * 2
        camera["camera_angle_y"] = math.atan(camera["h"] / (camera["fl_y"] * 2)) * 2
        # camera["fovx"] = camera["camera_angle_x"] * 180 / math.pi
        # camera["fovy"] = camera["camera_angle_y"] * 180 / math.pi
        frame = camera.copy()
        bottom = np.array([0.0, 0.0, 0.0, 1.0]).reshape([1, 4])
        name = str(dataset["image_paths"][i])
        # b = sharpness(name) if os.path.exists(name) else 1.0
        c2w = dataset["cameras"].poses[i, :3, :4]

        # Convert from Opencv to OpenGL coordinate system
        c2w = c2w.copy()
        c2w[..., 0:3, 1:3] *= -1

        c2w = np.concatenate([c2w[..., 0:3, 0:4], bottom], 0)

        if not keep_coords:
            c2w = c2w[[1, 0, 2, 3], :]
            c2w[2, :] *= -1  # flip whole world upside down

        up += c2w[0:3, 1]

        frame["file_path"] = name
        # Adding sharpness triggers removal in ingp code if the file doesn't exist
        # frame["sharpness"] = b
        frame["transform_matrix"] = c2w
        frames.append(frame)

    nframes = len(frames)
    if dataparser_transform is None and not keep_coords:
        # don't keep colmap coords - reorient the scene to be easier to work with
        up = up / np.linalg.norm(up)
        R = rotmat(up, [0, 0, 1])  # rotate up vector to [0,0,1]
        dataparser_transform = np.eye(4, dtype=np.float32)
        dataparser_transform[..., :3, :3] = R
        poses = [np.matmul(dataparser_transform, pad_poses(f["transform_matrix"])) for f in frames]

        # find a central point they are all looking at
        totw = 0.0
        totp = np.array([0.0, 0.0, 0.0])
        for p in poses:
            mf = p[0:3, :]
            for g in poses:
                mg = g[0:3, :]
                p, w = closest_point_2_lines(mf[:, 3], mf[:, 2], mg[:, 3], mg[:, 2])
                if w > 0.00001:
                    totp += p * w
                    totw += w
        if totw > 0.0:
            totp /= totw
        dataparser_transform[..., :3, 3] = -totp
    elif dataparser_transform is None:
        dataparser_transform = np.eye(4, dtype=np.float32)
        # dataparser_transform[..., 0:3, 1:3] *= -1

    # Compute scale
    if dataparser_scale is None and not keep_coords:
        avglen = 0.0
        for f in frames:
            pose = np.matmul(dataparser_transform, pad_poses(f["transform_matrix"]))
            avglen += np.linalg.norm(pose[0:3, 3])
        avglen /= nframes
        dataparser_scale = float(4.0 / avglen)  # scale to "nerf sized"
    elif dataparser_scale is None:
        dataparser_scale = 1.0

    for f in frames:
        f["transform_matrix"] = unpad_poses(np.matmul(dataparser_transform, pad_poses(f["transform_matrix"])))
        f["transform_matrix"][0:3, 3] *= dataparser_scale
        f["transform_matrix"] = f["transform_matrix"].tolist()

    out = {"frames": frames}
    if aabb_scale is not None:
        out["aabb_scale"] = aabb_scale
    return out, dict(dataparser_transform=dataparser_transform, 
                     dataparser_scale=dataparser_scale, 
                     aabb_scale=aabb_scale, 
                     keep_coords=keep_coords, 
                     **kwargs)


def _config_overrides_fix_types(config_overrides):
    out = {}
    for k, v in config_overrides.items():
        if isinstance(v, str):
            if v.lower() == "true":
                v = True
            elif k == "testbed.background_color":
                v = [float(x) for x in v.split(",")]
            elif v.lower() == "false":
                v = False
            elif v.lower() in {"none", "null"}:
                v = None
            else:
                try:
                    v = int(v)
                except ValueError:
                    try:
                        v = float(v)
                    except ValueError:
                        pass
        out[k] = v
    return out


class InstantNGP(Method):
    def __init__(self, *,
                 checkpoint: Optional[str] = None, 
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self._train_transforms = None
        self.n_steps = 35_000
        self._config_overrides = None
        self._base_config = None
        self._loaded_step = None
        self._tempdir = None

        # Fix older checkpoints
        if config_overrides is not None:
            new_cfg_overrides = {k.replace("/", "."): v for k, v in config_overrides.items()}
            config_overrides.clear()
            config_overrides.update(new_cfg_overrides)

        self._setup(train_dataset, config_overrides)

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",  # Will be set by the registry
            required_features=frozenset(("color",)), 
            supported_camera_models=frozenset(("pinhole", "opencv", "opencv_fisheye")),
            supported_outputs=("color", "accumulation"),
            viewer_default_resolution=(128, 512),
        )

    def get_info(self) -> ModelInfo:
        hparams = flatten_hparams(self._base_config or {}, separator=".")
        if self._config_overrides is not None:
            hparams.update(self._config_overrides)
        return ModelInfo(
            num_iterations=self.n_steps, 
            loaded_step=self._loaded_step,
            loaded_checkpoint=str(self.checkpoint) if self.checkpoint is not None else None,
            hparams=hparams,
            **self.get_method_info(),
        )

    def _setup_testbed(self, config_overrides=None):
        # Importing pyngp triggers error when no GPU is available
        import pyngp as ngp  # type: ignore

        self.RenderMode = ngp.RenderMode
        testbed = ngp.Testbed()
        testbed.root_dir = tempfile.gettempdir()

        # Get config path
        config_path = None
        if self.checkpoint is not None and os.path.exists(self.checkpoint / "base_config.json"):
            config_path = self.checkpoint / "base_config.json"
            # Ignore config_overrides if we have a checkpoint
            # TODO: consider writing test here
            self._config_overrides = json.loads((self.checkpoint / "config_overrides.json").read_text())
        else:
            if self.checkpoint is not None:
                warnings.warn(f"Checkpoint {self.checkpoint} does not contain base_config.json. We will use the default config.")
            package_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(ngp.__file__))))
            config_path = package_root / "configs" / "nerf" / "base.json"
            self._config_overrides = config_overrides or {}
        self._base_config = json.loads(config_path.read_text())

        # Load checkpoint and configure testbed
        if self.checkpoint is not None:
            testbed.load_snapshot(str(self.checkpoint / "checkpoint.ingp"))
        else:
            testbed.reload_network_from_file(str(config_path))

            # Update with config overrides
            self._config_overrides = config_overrides or {}

        self.testbed = testbed
        self._set_overrides()
        self._train_params_backup = {
            "snap_to_pixel_centers": self.testbed.snap_to_pixel_centers,
            "render_min_transmittance": self.testbed.nerf.render_min_transmittance,
        }
        self._current_mode = True

    def _set_overrides(self):
        import pyngp as ngp  # type: ignore
        def set_param(k, v):
            parts = k.split(".")
            testbedobj = self.testbed
            if parts[0] == "testbed":
                for part in parts[1:-1]:
                    testbedobj = getattr(testbedobj, part)
                if parts[-1] == "color_space":
                    v = getattr(ngp.ColorSpace, v)
                setattr(testbedobj, parts[-1], v)
            if k == "aabb_scale":
                self.dataparser_params["aabb_scale"] = v
            if k == "keep_coords":
                self.dataparser_params["keep_coords"] = v
            print(f"Setting {k} to {v}")

        # Default parameters from scripts/run.py
        self.testbed.exposure = 0.0
        self.testbed.nerf.sharpen = 0.0
        self.testbed.nerf.render_with_lens_distortion = True

        for k, v in (self._config_overrides or {}).items():
            set_param(k, v)
        print("Config overrides:")
        pprint.pprint(self._config_overrides or {})

        self.testbed.shall_train = True


    def _validate_config(self, train_dataset):
        # Verify blender config
        if train_dataset["metadata"].get("id") == "blender":
            if self.testbed.color_space.name != "SRGB":
                warnings.warn("Blender dataset is expected to have 'testbed.color_space=SRGB' in config_overrides.")
            if self.testbed.nerf.cone_angle_constant != 0:
                warnings.warn("Blender dataset is expected to have 'testbed.nerf.cone_angle_constant=0' in config_overrides.")
            if self.testbed.nerf.training.random_bg_color:
                warnings.warn("Blender dataset is expected to have 'testbed.nerf.training.random_bg_color=False' in config_overrides.")
            if self.testbed.background_color.tolist() != [1.0,1.0,1.0,1.0]:
                # Blender uses background_color = [255, 255, 255, 255]
                # NOTE: this is different from ingp official implementation
                warnings.warn("Blender dataset is expected to have 'testbed.background_color=1.0,1.0,1.0,1.0' in config_overrides.")

            if self.dataparser_params["aabb_scale"] is not None:
                warnings.warn("Blender dataset is expected to have 'aabb_scale=None' in config_overrides.")
            if self.dataparser_params["keep_coords"] is not True:
                warnings.warn("Blender dataset is expected to have 'keep_coords=True' in config_overrides.")

    def _write_images(self, dataset: Dataset, tmpdir: str):
        from tqdm import tqdm

        linked, copied = 0, 0

        with tqdm(dataset["image_paths"], desc="caching images", dynamic_ncols=True) as progress:
            for i, impath_source in enumerate(progress):
                impath_source = Path(impath_source)
                impath_target = Path(tmpdir) / str(impath_source.relative_to(dataset["image_paths_root"])).replace("/", "__")
                dataset["image_paths"][i] = str(impath_target)
                impath_target.parent.mkdir(parents=True, exist_ok=True)
                if impath_target.exists():
                    continue
                width, height = dataset["cameras"].image_sizes[i]
                if impath_source.exists():
                    impath_target.resolve().symlink_to(impath_source)
                    logging.debug(f"symlinked {impath_source} to {impath_target}")
                    linked += 1
                else:
                    img = dataset["images"][i][:height, :width]
                    if dataset["metadata"]["color_space"] == "srgb":
                        impath_target = impath_target.with_suffix(".png")
                        dataset["image_paths"][i] = str(impath_target)
                        image = Image.fromarray(img)
                        image.save(str(impath_target))
                    elif dataset["metadata"]["color_space"] == "linear":
                        impath_target = impath_target.with_suffix(".bin")
                        dataset["image_paths"][i] = str(impath_target)
                        if img.shape[2] < 4:
                            img = np.dstack((img, np.ones([img.shape[0], img.shape[1], 4 - img.shape[2]])))
                        with open(str(impath_target), "wb") as f:
                            f.write(struct.pack("<ii", img.shape[0], img.shape[1]))
                            f.write(img.astype(np.float16).tobytes())
                    logging.debug(f"copied {impath_source} to {impath_target}")
                    copied += 1
                progress.set_postfix(linked=linked, copied=copied)
                if dataset["masks"] is not None:
                    mask_paths = dataset.get("mask_paths", None)
                    if mask_paths is None:
                        dataset["mask_paths"] = mask_paths = [""] * len(dataset["image_paths"])
                    mask = dataset["masks"][i]
                    mask = Image.fromarray(mask[:height, :width], mode="L")
                    mask = ImageOps.invert(mask)
                    maskname = impath_target.with_name(f"dynamic_mask_{impath_target.name}")
                    mask_paths[i] = str(maskname)
                    mask.save(str(maskname))
        dataset["image_paths_root"] = str(tmpdir)

    def _setup(self, train_dataset: Optional[Dataset]=None, config_overrides=None):
        config_overrides = _config_overrides_fix_types(config_overrides or {}).copy()

        current_step = 0
        if self.checkpoint is not None:
            with (self.checkpoint / "meta.json").open() as f:
                meta = json.load(f)
                self.dataparser_params = meta["dataparser_params"]
                if "dataparser_transform_base64" in self.dataparser_params:
                    self.dataparser_params["dataparser_transform"] = numpy_from_base64(self.dataparser_params["dataparser_transform_base64"])
                    del self.dataparser_params["dataparser_transform_base64"]
                else:
                    warnings.warn("dataparser_transform_base64 not found in checkpoint. This may degrade performance.")
                    self.dataparser_params["dataparser_transform"] = np.array(self.dataparser_params["dataparser_transform"], dtype=np.float32)
                current_step = meta["step"]
                self._loaded_step = meta["step"]
                self.n_steps = meta.get("n_steps", self.n_steps)
        else:
            self.dataparser_params = {}
            aabb_scale = cast_value(Optional[int], config_overrides.get("aabb_scale", 32))
            keep_coords = cast_value(bool, config_overrides.get("keep_coords", False))
            self.dataparser_params["aabb_scale"] = aabb_scale
            self.dataparser_params["keep_coords"] = keep_coords
            self.dataparser_params["color_space"] = (train_dataset or {}).get("metadata", {}).get("color_space", "srgb")

        self._setup_testbed(config_overrides)

        # Load training data if available
        if train_dataset is not None:
            # Write training images
            self._tempdir = tempfile.TemporaryDirectory()
            self._tempdir.__enter__()
            tmpdir = self._tempdir.name
            self._write_images(train_dataset, tmpdir)

            # Get train transforms
            self._train_transforms, self.dataparser_params = get_transforms(
                train_dataset, 
                **self.dataparser_params
            )
            with (Path(tmpdir) / "transforms.json").open("w") as f:
                json.dump(self._train_transforms, f)

            # Load training data
            self.testbed.load_training_data(str(Path(tmpdir) / "transforms.json"))

            # Loading training data might have changed some parameters
            self._set_overrides()

        # Validate config
        if train_dataset is not None:
            self._validate_config(train_dataset)
        assert self.testbed.training_step == current_step, "Training step mismatch"

    def train_iteration(self, step: int):
        assert self._tempdir is not None, "Tempdir is not set"
        self._set_mode_and_data(training=True)
        assert self.testbed.shall_train, "Training is disabled"
        current_frame = self.testbed.training_step
        if step < self.n_steps:
            deadline = 100
            while current_frame < step + 1:
                if not self.testbed.frame():
                    raise RuntimeError("Training failed")
                current_frame = self.testbed.training_step
                deadline -= 1
                if deadline < 0:
                    raise RuntimeError("Training failed")
        if step == self.n_steps - 1:
            # Release the tempdir
            self._tempdir.cleanup()
            self._tempdir = None

        return {
            "loss": self.testbed.loss,
        }

    def save(self, path: str):
        _path: Path = Path(path)
        _path.mkdir(parents=True, exist_ok=True)
        with (_path / "meta.json").open("w") as f:
            out = self.dataparser_params.copy()
            dataparser_transform  = out.get("dataparser_transform")
            assert dataparser_transform is not None, "dataparser_transform is None"
            out["dataparser_transform_base64"] = numpy_to_base64(dataparser_transform)
            out["dataparser_transform"] = dataparser_transform.tolist()
            json.dump(
                {
                    "dataparser_params": out,
                    "step": self.testbed.training_step,
                    "n_steps": self.n_steps,
                },
                f,
                indent=2,
            )
        with (_path / "base_config.json").open("w") as f:
            json.dump(self._base_config, f, indent=2)
        with (_path / "config_overrides.json").open("w") as f:
            json.dump(self._config_overrides, f, indent=2)
        self.testbed.save_snapshot(str(_path / "checkpoint.ingp"), False)

        # In the ingp impl. there is a bug which clears the two following parameters on load
        # They have to be preserved in order to obtain consistent checkpoints.
        # m_nerf.training.counters_rgb.rays_per_batch = 1 << 12;
        # m_nerf.training.counters_rgb.measured_batch_size_before_compaction = 0;
        with gzip.open(_path / "checkpoint.ingp", "r") as f:
            checkpoint_data = msgpack.load(f)

            sha = hashlib.sha256()
            def _update(obj):
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        if k in {"rays_per_batch", "measured_batch_size_before_compaction"}:
                            continue
                        if k == "paths":
                            v = [os.path.basename(p) for p in v]
                        _update(k)
                        _update(v)
                elif isinstance(obj, (list, tuple)):
                    for v in obj:
                        _update(v)
                elif isinstance(obj, (str, int, float, bool)):
                    sha.update(str(obj).encode())
                elif isinstance(obj, bytes):
                    sha.update(obj)
            _update(checkpoint_data)
            sha = hashlib.sha256()
            pass
        with (_path / "checkpoint.ingp.sha256").open("w") as f:
            f.write(sha.hexdigest())

    def _set_mode_and_data(self, training=True, camera=None):
        if training and self._current_mode:
            # Already in training mode
            return
        elif training:
            # Switch back to training mode - we have to reload train transforms after
            # it was replaced by eval mode.
            data = self._train_params_backup
            self.testbed.snap_to_pixel_centers = data["snap_to_pixel_centers"]
            self.testbed.nerf.render_min_transmittance = data["render_min_transmittance"]
            self.testbed.shall_train = True
            if self._tempdir is not None:
                with (Path(self._tempdir.name) / "transforms.json").open("w") as f:
                    json.dump(self._train_transforms, f)
                self.testbed.load_training_data(str(Path(self._tempdir.name) / "transforms.json"))
        else:  # Eval mode
            # Switch to eval mode and load the eval data
            assert camera is not None, "Camera is required in eval mode"
            camera = camera.item()

            with tempfile.TemporaryDirectory() as tmpdir:
                w, h = camera.image_sizes
                dataset = cast(Dataset, dict(
                    points3D_xyz=None,
                    points3D_rgb=None,
                    images_points3D_indices=None,
                    cameras=camera[None],
                    mask_paths=None,
                    mask_paths_root=None,
                    masks=None,
                    images=[np.zeros((h, w, 3), dtype=np.uint8)],
                    image_paths_root="eval",
                    image_paths=[f"eval/{0:06d}.png"],
                    metadata={
                        "color_space": self.dataparser_params["color_space"],
                    }
                ))
                self._write_images(dataset, tmpdir)
                with (Path(tmpdir) / "transforms.json").open("w") as f:
                    json.dump(get_transforms(dataset, **self.dataparser_params)[0], f)

                self.testbed.load_training_data(os.path.join(tmpdir, "transforms.json"))

            # Prior nerf papers don't typically do multi-sample anti aliasing.
            # So snap all pixels to the pixel centers.
            self.testbed.snap_to_pixel_centers = True

            self.testbed.nerf.render_min_transmittance = 1e-4

            self.testbed.shall_train = False

        self._current_mode = training

    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        del options
        camera = camera.item()
        self._set_mode_and_data(training=False, camera=camera)
        assert self.testbed.nerf.training.dataset.n_images == 1, "Only one image is supported in eval mode"

        spp = 8
        # print(testbed.nerf.training.dataset.metadata[i].resolution)
        resolution = self.testbed.nerf.training.dataset.metadata[0].resolution
        self.testbed.set_camera_to_training_view(0)
        self.testbed.render_mode = self.RenderMode.Shade
        image = self.testbed.render(resolution[0], resolution[1], spp, True)

        # Unmultiply by alpha
        image[..., 0:3] = np.divide(image[..., 0:3], image[..., 3:4], out=np.zeros_like(image[..., 0:3]), where=image[..., 3:4] != 0)

        # old_render_mode = self.testbed.render_mode
        ## testbed.render_mode = ngp.RenderMode.Depth
        ## depth = np.copy(testbed.render(resolution[0], resolution[1], spp, True))
        ## [ H, W, 4]

        # testbed.render_mode = ngp.RenderMode.Normals
        # normals = testbed.render(resolution[0], resolution[1], spp, True)
        # self.testbed.render_mode = old_render_mode

        # If input color was in sRGB, we map back to sRGB here
        if self.dataparser_params["color_space"] == "srgb":
            image = linear_to_srgb(image)
        return {
            "color": image,
            ## "depth": depth,
            "accumulation": image[..., 3],
        }

    def close(self):
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None
