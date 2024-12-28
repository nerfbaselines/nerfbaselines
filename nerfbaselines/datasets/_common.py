from datetime import datetime
import time
import re
import json
import importlib
import shutil
from pathlib import Path
import warnings
import gc
import functools
import logging
import os
import struct
import tempfile
import numpy as np
import PIL.Image
import PIL.ExifTags
from tqdm import tqdm
import requests
from typing import (
    Optional, TypeVar, Tuple, Union, List, Dict, Any, 
    FrozenSet, Iterable,
    overload, cast,
)
from nerfbaselines import (
    Dataset, UnloadedDataset, DatasetFeature,
    Cameras, CameraModel,
    camera_model_to_int, DatasetNotFoundError,
    get_dataset_spec,
    get_supported_datasets,
    get_dataset_loader_spec,
    get_supported_dataset_loaders,
    NB_PREFIX,
)
from .. import cameras
from ..utils import padded_stack, pad_poses, unpad_poses, apply_transform
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


TDataset = TypeVar("TDataset", bound=Union[Dataset, UnloadedDataset])
logger = logging.getLogger("nerfbaselines.datasets")


def _import_type(name: str) -> Any:
    package, name = name.split(":")
    obj: Any = importlib.import_module(package)
    for p in name.split("."):
        obj = getattr(obj, p)
    return obj


def experimental_parse_dataset_path(path: str) -> Tuple[str, Dict[str, Any]]:
    # NOTE: This is an experimental feature likely to change
    kwargs: Dict[str, Any] = {}
    if "#" in path:
        path, urlquery = path.split("#", 1)
        kwargs = dict(x.split("=") for x in urlquery.split("&"))
    return path, kwargs


def single(xs):
    out = None
    for x in xs:
        if out is not None:
            raise ValueError("Expected single value, got multiple")
        out = x
    if out is None:
        raise ValueError("Expected single value, got none")
    return out


def get_transform_poses_pca(poses):
    t = poses[:, :3, 3]
    t_mean = t.mean(axis=0)
    t = t - t_mean

    eigval, eigvec = np.linalg.eig(t.T @ t)
    # Sort eigenvectors in order of largest to smallest eigenvalue.
    inds = np.argsort(eigval)[::-1]
    eigvec = eigvec[:, inds]
    rot = eigvec.T
    if np.linalg.det(rot) < 0:
        rot = np.diag(np.array([1, 1, -1])) @ rot

    transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
    poses_recentered = unpad_poses(transform @ pad_poses(poses))
    transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)

    # Flip coordinate system if z component of y-axis is positive
    if poses_recentered.mean(axis=0)[2, 1] > 0:
        transform = np.diag(np.array([1, -1, -1, 1])) @ transform

    # Just make sure it's it in the [-1, 1]^3 cube
    scale_factor = 1.0 / np.max(np.abs(poses_recentered[:, :3, 3]))
    transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform
    return transform


def focus_point_fn(poses, xnp = np):
    """Calculate nearest point to all focal axes in poses."""
    directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
    m = xnp.eye(3) - directions * xnp.transpose(directions, [0, 2, 1])
    mt_m = xnp.transpose(m, [0, 2, 1]) @ m
    focus_pt = xnp.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
    return focus_pt


def make_rotation_matrix(a, b):
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
        return make_rotation_matrix(a + eps, b)
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


def _viewmatrix(lookdir, up, position):
    def normalize(x):
        return x / np.linalg.norm(x)
    z = normalize(lookdir)
    x = normalize(np.cross(z, up))
    y = normalize(np.cross(z, x))
    m = np.stack([x, y, z, position], axis=1)
    return m


def get_default_viewer_transform(poses, dataset_type: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the default viewer transform and initial pose for a dataset.
    The default viewer transform assumes z=up, y=forward, x=right in the world coordinate system.

    Args:
        poses: The camera poses.
        dataset_type: The type of dataset. If None, the dataset type is unknown.
    """
    if dataset_type == "object-centric":
        transform = get_transform_poses_pca(poses)

        poses = apply_transform(transform, poses)
        lookat = focus_point_fn(poses)

        poses[:, :3, 3] -= lookat
        transform[:3, 3] -= lookat
        return transform, poses[0][..., :3, :4]

    elif dataset_type == "forward-facing":
        # First, we compute the average pose
        position = poses[:, :3, 3].mean(0)
        lookdir = poses[:, :3, 2].mean(0)
        up = -poses[:, :3, 1].mean(0)
        avg_pose = _viewmatrix(lookdir, up, position)

        # Then, we compute the transform that moves the average pose to the origin
        transform = np.linalg.inv(pad_poses(avg_pose))

        # Currently we assume (z=forward, y=down, x=right)
        # We need to convert to (z=up, y=forward, x=right)
        # Swap z, y
        transform[:3, [1, 2]] = transform[:3, [2, 1]]
        # Invert y
        transform[:3, 1] = -transform[:3, 1]

        # Finally, we fix the scale
        # Scale so that cameras fit in a 2x2x2 cube centered at the origin
        mean_origin = apply_transform(transform, poses)[..., :3, 3].mean(0)
        maxlen = np.quantile(np.abs(poses[..., 0:3, 3] - mean_origin[None]).max(-1), 0.95) * 1.1
        dataparser_scale = float(1 / maxlen)
        transform = np.diag([dataparser_scale, dataparser_scale, dataparser_scale, 1]) @ transform

        initial_pose = apply_transform(transform, poses[1])
        return transform, initial_pose[..., :3, :4]
    elif dataset_type is None:
        # Unknown dataset type
        # We move all center the scene on the mean of the camera origins
        # and reorient the scene so that the average camera up is up
        origins = poses[..., :3, 3]
        mean_origin = np.mean(origins, 0)
        translation = mean_origin
        up = np.mean(poses[:, :3, 1], 0)
        up = -up / np.linalg.norm(up)

        rotation = make_rotation_matrix(up, np.array([0, 0, 1], dtype=up.dtype))
        transform = np.concatenate([rotation, rotation @ -translation[..., None]], -1)
        transform = np.concatenate([transform, np.array([[0, 0, 0, 1]], dtype=transform.dtype)], 0)

        # Scale so that cameras fit in a 2x2x2 cube centered at the origin
        maxlen = np.quantile(np.abs(poses[..., 0:3, 3] - mean_origin[None]).max(-1), 0.95) * 1.1
        dataparser_scale = float(1 / maxlen)
        transform = np.diag([dataparser_scale, dataparser_scale, dataparser_scale, 1]) @ transform

        camera = apply_transform(transform, poses[0])
        return transform, camera[..., :3, :4]
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported")


def _dataset_undistort_unsupported(dataset: Dataset, supported_camera_models):
    assert dataset["images"] is not None, "Images must be loaded"
    supported_models_int = set(camera_model_to_int(x) for x in supported_camera_models)
    undistort_tasks = []
    for i, camera in enumerate(dataset["cameras"]):
        if camera.camera_models.item() in supported_models_int:
            continue
        undistort_tasks.append((i, camera))
    if len(undistort_tasks) == 0:
        return False

    was_list = isinstance(dataset["images"], list)
    new_images = list(dataset["images"])
    new_sampling_masks = (
        list(dataset["sampling_masks"]) if dataset["sampling_masks"] is not None else None
    )
    dataset["images"] = new_images
    dataset["sampling_masks"] = new_sampling_masks

    # Release memory here
    gc.collect()

    for i, camera in tqdm(undistort_tasks, desc="undistorting images", dynamic_ncols=True):
        undistorted_camera = cameras.undistort_camera(camera)
        ow, oh = camera.image_sizes
        if dataset["image_paths"] is not None:
            dataset["image_paths"][i] = os.path.join(
                "/undistorted", os.path.split(dataset["image_paths"][i])[-1]
            )
        if dataset["sampling_mask_paths"] is not None:
            dataset["sampling_mask_paths"][i] = os.path.join(
                "/undistorted-masks", os.path.split(dataset["sampling_mask_paths"][i])[-1]
            )
        warped = cameras.warp_image_between_cameras(
            camera, undistorted_camera, new_images[i][:oh, :ow]
        )
        new_images[i] = warped
        if new_sampling_masks is not None:
            warped = cameras.warp_image_between_cameras(camera, undistorted_camera, new_sampling_masks[i][:oh, :ow])
            new_sampling_masks[i] = warped
        # IMPORTANT: camera is modified in-place
        dataset["cameras"][i] = undistorted_camera
    if not was_list:
        dataset["images"] = padded_stack(new_images)
        dataset["sampling_masks"] = (
            padded_stack(new_sampling_masks) if new_sampling_masks is not None else None
        )
    dataset["image_paths_root"] = "/undistorted"
    return True


METADATA_COLUMNS = ["exposure"]
DatasetType = Literal["object-centric", "forward-facing"]


def get_scene_scale(cameras: Cameras, dataset_type: Optional[DatasetType]):
    if dataset_type == "object-centric":
        return float(np.percentile(np.linalg.norm(cameras.poses[..., :3, 3] - cameras.poses[..., :3, 3].mean(), axis=-1), 90))

    elif dataset_type == "forward-facing":
        assert cameras.nears_fars is not None, "Forward-facing dataset must set z-near and z-far"
        return float(cameras.nears_fars.mean())

    elif dataset_type is None:
        return float(np.percentile(np.linalg.norm(cameras.poses[..., :3, 3] - cameras.poses[..., :3, 3].mean(), axis=-1), 90))
    
    else:
        raise ValueError(f"Dataset type {dataset_type} is not supported")


def get_image_metadata(image: PIL.Image.Image):
    # Metadata format: [ exposure, ]
    values = {}
    try:
        exif_pil = image.getexif()
    except AttributeError:
        exif_pil = image._getexif()  # type: ignore
    if exif_pil is not None:
        exif = {PIL.ExifTags.TAGS[k]: v for k, v in exif_pil.items() if k in PIL.ExifTags.TAGS}
        if "ExposureTime" in exif and "ISOSpeedRatings" in exif:
            shutters = exif["ExposureTime"]
            isos = exif["ISOSpeedRatings"]
            exposure = shutters * isos / 1000.0
            values["exposure"] = exposure
    return np.array([values.get(c, np.nan) for c in METADATA_COLUMNS], dtype=np.float32)


def _dataset_rescale_intrinsics(dataset: Dataset, image_sizes: np.ndarray):
    cameras = dataset["cameras"]
    if np.any(cameras.image_sizes != image_sizes):
        logger.info("Image sizes do not match camera sizes. Resizing cameras to match image sizes.")

        if np.any(cameras.image_sizes % image_sizes != 0):
            warnings.warn("Downscaled image sizes are not a multiple of camera sizes.")

        multx, multy = np.moveaxis(
            image_sizes.astype(np.float64) / cameras.image_sizes.astype(np.float64), -1, 0)

        if "downscale_factor" in dataset["metadata"]:
            # Downscale factor is passed, we will use it for focal lengths
            # Not for the center of the image, because there could have been rounding
            # which would move the center from the center of the image
            downscale_factor = dataset["metadata"]["downscale_factor"]
            low = np.floor(cameras.image_sizes * np.stack([multx, multy], -1))
            high = np.ceil(cameras.image_sizes * np.stack([multx, multy], -1))
            if np.any(image_sizes < low) or np.any(image_sizes > high):
                raise RuntimeError(f"Downscaled image sizes do not match the downscale_factor of {downscale_factor}.")

        # NOTE: In previous versions of NerfBaselines, we scaled the parameters differently
        # We used:
        #   cx <- cx * multx,  cy <- cy * multy
        #   fx <- fx * multx,  fy <- fy * multx
        # This renders changes slightly the results on the MipNeRF 360 dataset

        multipliers = np.stack([multx, multy, multx, multy], -1)
        dataset["cameras"] = cameras.replace(
            image_sizes=image_sizes, 
            intrinsics=(cameras.intrinsics * multipliers).astype(cameras.intrinsics.dtype))


def dataset_load_features(
    dataset: UnloadedDataset, features=None, supported_camera_models=None, show_progress=True
) -> Dataset:
    if features is None:
        features = frozenset(("color",))
    images: List[np.ndarray] = []
    image_sizes = []
    all_metadata = []
    resize = dataset["metadata"].get("downscale_loaded_factor")
    if resize == 1:
        resize = None

    image_paths_root = dataset.get("image_paths_root")
    if image_paths_root is not None:
        logger.info(f"Loading images from {image_paths_root}")

    i = 0
    for p in tqdm(dataset["image_paths"], desc="loading images", dynamic_ncols=True, disable=not show_progress):
        if str(p).endswith(".bin"):
            assert dataset["metadata"]["color_space"] == "linear"
            with open(p, "rb") as f:
                data_bytes = f.read()
                h, w = struct.unpack("<II", data_bytes[:8])
                image = (
                    np.frombuffer(
                        data_bytes, dtype=np.float16, count=h * w * 4, offset=8
                    )
                    .astype(np.float32)
                    .reshape([h, w, 4])
                )
            metadata = np.array(
                [np.nan for _ in range(len(METADATA_COLUMNS))], dtype=np.float32
            )
        else:
            assert dataset["metadata"]["color_space"] == "srgb"
            pil_image = PIL.Image.open(p)
            metadata = get_image_metadata(pil_image)
            if resize is not None:
                w, h = pil_image.size
                new_size = round(w/resize), round(h/resize)
                pil_image = pil_image.resize(new_size, PIL.Image.Resampling.BICUBIC)
                warnings.warn(f"Resized image with a factor of {resize}")

            image = np.array(pil_image, dtype=np.uint8)
        images.append(image)
        image_sizes.append([image.shape[1], image.shape[0]])
        all_metadata.append(metadata)
        i += 1

    logger.debug(f"Loaded {len(images)} images")

    if dataset["sampling_mask_paths"] is not None:
        sampling_masks = []
        for p in tqdm(dataset["sampling_mask_paths"], desc="loading sampling masks", dynamic_ncols=True):
            sampling_mask = PIL.Image.open(p).convert("L")
            if resize is not None:
                w, h = sampling_mask.size
                new_size = round(w*resize), round(h*resize)
                sampling_mask = sampling_mask.resize(new_size, PIL.Image.Resampling.NEAREST)
                warnings.warn(f"Resized sampling mask with a factor of {resize}")

            sampling_masks.append(np.array(sampling_mask, dtype=np.uint8).astype(bool))
        dataset["sampling_masks"] = sampling_masks  # padded_stack(sampling_masks)
        logger.debug(f"Loaded {len(sampling_masks)} sampling masks")

    if resize is not None:
        # Replace all paths with the resized paths
        dataset["image_paths"] = [
            os.path.join("/resized", os.path.relpath(p, dataset["image_paths_root"])) 
            for p in dataset["image_paths"]]
        dataset["image_paths_root"] = "/resized"
        if dataset["sampling_mask_paths"] is not None:
            dataset["sampling_mask_paths"] = [
                os.path.join("/resized-sampling-masks", os.path.relpath(p, dataset["sampling_mask_paths_root"])) 
                for p in dataset["sampling_mask_paths"]]
            dataset["sampling_mask_paths_root"] = "/resized-sampling-masks"

    dataset["images"] = images

    # Replace image sizes and metadata
    image_sizes = np.array(image_sizes, dtype=np.int32)

    _dataset_rescale_intrinsics(cast(Dataset, dataset), image_sizes)

    if supported_camera_models is not None:
        if _dataset_undistort_unsupported(cast(Dataset, dataset), supported_camera_models):
            logger.warning(
                "Some cameras models are not supported by the method. Images have been undistorted. Make sure to use the undistorted images for training."
            )
    return cast(Dataset, dataset)


class MultiDatasetError(DatasetNotFoundError):
    def __init__(self, errors, message):
        self.errors = errors
        self.message = message
        super().__init__(message + "\n" + "".join(f"\n  {name}: {error}" for name, error in errors.items()))

    def write_to_logger(self, color=True, terminal_width=None):
        if terminal_width is None:
            terminal_width = 120
            try:
                terminal_width = min(os.get_terminal_size().columns, 120)
            except OSError:
                pass
        message = self.message
        if color:
            message = "\33[0m\33[31m" + message + "\33[0m"
        for name, error in self.errors.items():
            prefix = f"   {name}: "
            mlen = terminal_width - len(prefix)
            prefixlen = len(prefix)
            if color:
                prefix = f"\33[96m{prefix}\33[0m"
            rows = [error[i : i + mlen] for i in range(0, len(error), mlen)]
            mdetail = f'\n{" "*prefixlen}'.join(rows)
            message += f"\n{prefix}{mdetail}"
        logger.error(message)


def dataset_index_select(dataset: TDataset, i: Union[slice, list, np.ndarray]) -> TDataset:
    assert isinstance(i, (slice, list, np.ndarray))
    dataset_len = len(dataset["image_paths"])
    if isinstance(i, np.ndarray):
        is_bool_mask = (i.shape == (dataset_len,) and i.dtype == bool)
        is_int_indices = (i.ndim == 1 and np.issubdtype(i.dtype, np.integer))
        if not is_bool_mask and not is_int_indices:
            raise ValueError("Expected boolean mask or integer indices")

    def index(key, obj):
        if obj is None:
            return None
        if key == "cameras":
            return obj[i]
        if isinstance(obj, np.ndarray):
            return obj[i]
        if isinstance(obj, list):
            indices = np.arange(dataset_len)[i]
            return [obj[i] for i in indices]
        raise ValueError(f"Cannot index object of type {type(obj)} at key {key}")

    _dataset = cast(Dict, dataset.copy())
    _dataset.update({k: index(k, v) for k, v in dataset.items() if k not in {
        "image_paths_root", 
        "sampling_mask_paths_root", 
        "points3D_xyz", 
        "points3D_rgb", 
        "metadata"}})
    return cast(TDataset, _dataset)


def download_dataset(path: str, output: Union[str, Path]):
    if not path.startswith("external://"):
        raise ValueError("Only external datasets can be downloaded (path must start with 'external://')")
    path = path[len("external://") :]
    output = Path(output)
    errors = {}
    for name in get_supported_datasets():
        dataset_spec = get_dataset_spec(name)
        download_fn = _import_type(dataset_spec["download_dataset_function"])
        try:
            download_fn(path, str(output))
            logger.info(f"Downloaded {name} dataset with path {path}")
            return
        except DatasetNotFoundError as e:
            logger.debug(e)
            logger.debug(f"Path {path} is not supported by {name} dataset")
            errors[name] = str(e)
    raise MultiDatasetError(errors, f"No supported dataset found for path {path}")


@overload
def load_dataset(
        path: Union[Path, str], 
        split: str, 
        features: Optional[FrozenSet[DatasetFeature]] = ...,
        supported_camera_models: Optional[FrozenSet[CameraModel]] = ...,
        load_features: Literal[True] = ...,
        **kwargs) -> Dataset:
    ...


@overload
def load_dataset(
        path: Union[Path, str], 
        split: str, 
        features: Optional[FrozenSet[DatasetFeature]] = ...,
        supported_camera_models: Optional[FrozenSet[CameraModel]] = ...,
        load_features: Literal[False] = ...,
        **kwargs) -> UnloadedDataset:
    ...


def _resolve_loader(loader):
    if loader in get_supported_dataset_loaders():
        loader = get_dataset_loader_spec(loader)["load_dataset_function"]
    elif ":" not in loader:
        raise ValueError(f"Unknown dataset loader {loader}")
    return _import_type(loader)


def _load_unknown_dataset(path, **kwargs):
    logger.info(f"Detecting dataset format from path: {path}")
    if path.startswith(os.path.join(NB_PREFIX, "datasets")):
        raise RuntimeError("Dataset is an external dataset, but it does not have nb-info.json metadata. "
                           "This might have been caused by an older version of NerfBaselines. "
                           f"Please remove the directory {path} and try again.")
    from .mipnerf360 import SCENES as MIPNERF360_SCENES
    from .blender import SCENES as BLENDER_SCENES
    from .tanksandtemples import SCENES as TANKSANDTEMPLES_SCENES
    from .phototourism import SCENES as PHOTOTOURISM_SCENES
    from .llff import SCENES as LLFF_SCENES

    # Validate for common mistakes (e.g., not specifying loader for known datasets)
    def _give_warning(dataset_id, dataset_name):
        logger.warning(f"Detected {dataset_name} dataset, but no loader was specified. "
                       f"If the dataset really is {dataset_name}, please use the official NerfBaselines downloader "
                       f"e.g., by specifying `--data external://{dataset_id}/{scene}`. "
                       "Otherwise, the dataset will be loaded as a generic dataset. "
                       f"If the dataset is not a {dataset_name} dataset, you can ignore the message. "
                       "In order to suppress this message, please specify the loader by adding a nb-info.json file.")
    def _simplify(x):
        return re.sub("[^a-z0-9]", "", x.lower())
    scene = _simplify(os.path.split(path)[-1].lower())
    spath = _simplify(path)
    if "360" in spath and any(scene == _simplify(x) for x in MIPNERF360_SCENES):
        _give_warning("mipnerf360", "MipNeRF 360")
    elif (("blender" in spath or ("nerf" in spath and "synthetic" in spath)) and
          any(scene == _simplify(x) for x in BLENDER_SCENES)):
        _give_warning("blender", "Blender")
    elif ("tanks" in spath and "temples" in spath and 
        any(scene == _simplify(x) for x in TANKSANDTEMPLES_SCENES)):
        _give_warning("tanksandtemples", "Tanks and Temples")
    elif ("phototourism" in spath and
        any(scene == _simplify(x) for x in PHOTOTOURISM_SCENES)):
        _give_warning("phototourism", "Phototourism")
    elif ("llff" in spath and
        any(scene == _simplify(x) for x in LLFF_SCENES)):
        _give_warning("llff", "LLFF")

    # Now, we gave all the warnings, we can try to detect and load the dataset.
    loaders = get_supported_dataset_loaders()
    loader_results = {}
    for name in loaders:
        spec = get_dataset_loader_spec(name)
        load_fn = _import_type(spec["load_dataset_function"])
        try:
            dataset = load_fn(path, **kwargs)
            loader_results[name] = dataset, None
        except Exception as e:
            loader_results[name] = None, e
    if len(loader_results) == 0:
        raise DatasetNotFoundError(f"No supported dataset found in path {path}")
    num_success = sum(1 for _, exc in loader_results.values() if exc is None)
    if num_success == 0:
        raise MultiDatasetError({
            name: str(exc) for name, (_, exc) in loader_results.items() if exc is not None
        }, f"No supported dataset found in path {path}")
    elif num_success > 1:
        # Raise an error about detecting more than one dataset
        loaders = [name for name, (_, exc) in loader_results.items() if exc is None]
        raise RuntimeError(f"There are multiple loaders which can load the dataset stored in path: {path}. "
                           "The loaders are {', '.join(loaders)}. "
                           "Please specify the loader by adding a nb-info.json file to the dataset directory "
                           f"with contents '{{\"loader\": \"<loader>\"}}' where '<loader>' is one of the loaders ({', '.join(loaders)}).")
    else:
        # Return the correct dataset
        loader, dataset = next(((k, d) for k, (d, exc) in loader_results.items() if exc is None))
    return loader, dataset


def download_dataset_wrapper(all_scenes: Iterable[str], dataset_name: str):
    """
    Wraps a function which downloads a single scene into a function which downloads multiple scenes.

    Args:
        all_scenes: Supported scenes.
    """
    all_scenes = list(all_scenes)

    def wrap(fn):
        @functools.wraps(fn)
        def download_dataset(path, output, **kwargs):
            if "/" in path:
                if not path.startswith(dataset_name + "/"):
                    raise DatasetNotFoundError(f"Dataset {path} does not start with {dataset_name}/")

                # Download multiple scenes
                return fn(path, output, **kwargs)
            else:
                if path != dataset_name:
                    raise DatasetNotFoundError(f"Dataset {path} does not start with {dataset_name}")
                for scene in all_scenes:
                    fn(f"{path}/{scene}", os.path.join(output, scene), **kwargs)
        return download_dataset
    return wrap


def download_archive_dataset(url: str,
                             output: str,
                             *,
                             archive_prefix: Optional[str],
                             nb_info: Dict[str, Any],
                             callback=None,
                             file_type = None):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc=f"Downloading {url.split('/')[-1]}", 
        dynamic_ncols=True,
    )
    with tempfile.TemporaryFile("rb+") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
        file.flush()
        file.seek(0)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:  # noqa: PLR1714
            logger.error(
                f"Failed to download dataset. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes."
            )

        has_any = False
        if file_type is None:
            if url.split("?")[0].split("#")[0].endswith(".tar.gz"):
                file_type = "tar.gz"
            elif url.split("?")[0].split("#")[0].endswith(".zip"):
                file_type = "zip"
            else:
                raise RuntimeError(f"Unknown file type for {url}")
        output_tmp = output + ".tmp"
        os.makedirs(output_tmp, exist_ok=True)
        if file_type == "tar.gz":
            import tarfile
            with tarfile.open(fileobj=file, mode="r:gz") as z:
                def members(tf):
                    nonlocal has_any
                    nonlocal archive_prefix
                    if archive_prefix is None:
                        # We estimate archive prefix as the common prefix of all files
                        archive_prefix = os.path.commonprefix([member.path for member in tf.getmembers() if not member.isdir()])
                    for member in tf.getmembers():
                        if not member.path.startswith(archive_prefix):
                            continue
                        has_any = True
                        member.path = member.path[len(archive_prefix):]
                        yield member

                z.extractall(output_tmp, members=members(z))
        elif file_type == "zip":
            import zipfile
            with zipfile.ZipFile(file, "r") as z:
                if archive_prefix is None:
                    # We estimate archive prefix as the common prefix of all files
                    archive_prefix = os.path.commonprefix([member.filename for member in z.infolist() if not member.is_dir()])
                for member in z.infolist():
                    if not member.filename.startswith(archive_prefix):
                        continue
                    relname = member.filename[len(archive_prefix):]
                    target = os.path.join(output_tmp, relname)
                    if member.is_dir():
                        os.makedirs(target, exist_ok=True)
                    else:
                        os.makedirs(os.path.dirname(target), exist_ok=True)
                        member.filename = relname
                        z.extract(member, output_tmp)
                        has_any = True

                        # Fix mtime
                        date_time = datetime(*member.date_time)
                        mtime = time.mktime(date_time.timetuple())
                        os.utime(target, (mtime, mtime))
        else:
            raise RuntimeError(f"Unknown file type {file_type}")
        if not has_any:
            raise RuntimeError(f"Prefix '{archive_prefix}' not found in {url}.")

        with open(os.path.join(str(output_tmp), "nb-info.json"), "w", encoding="utf8") as f2:
            json.dump(nb_info, f2)

        if callback is not None:
            callback(output_tmp)

        shutil.rmtree(output, ignore_errors=True)
        shutil.move(str(output_tmp), str(output))


def _parse_metadata(meta):
    meta = meta.copy()
    if "background_color" in meta and isinstance(meta["background_color"], (list, tuple)):
        meta["background_color"] = np.array(meta["background_color"], dtype=np.uint8)
    return meta


def load_dataset(
        path: Union[Path, str], 
        split: str, 
        features: Optional[FrozenSet[DatasetFeature]] = None,
        supported_camera_models: Optional[FrozenSet[CameraModel]] = None,
        load_features: bool = True,
        **kwargs,
        ) -> Union[Dataset, UnloadedDataset]:
    path = str(path)
    path, _kwargs = experimental_parse_dataset_path(path)
    _kwargs.update(kwargs)
    kwargs = _kwargs
    if features is None:
        features = frozenset(("color",))
    kwargs["features"] = features
    del features

    # If path is and external path, we download the dataset first
    if path.startswith("external://"):
        dataset = path.split("://", 1)[1]
        path = Path(NB_PREFIX) / "datasets" / dataset
        if not path.exists():
            download_dataset("external://"+dataset, path)
        path = str(path)

    # Try loading info if exists
    loader = None
    meta = {}
    info_fname = "nb-info.json"
    if (os.path.exists(os.path.join(path, "info.json")) and 
        not os.path.exists(os.path.join(path, info_fname))):
        logger.warning("Using 'info.json' instead of 'nb-info.json'. Please update the dataset.")
        info_fname = "info.json"
    if os.path.exists(os.path.join(path, info_fname)):
        logger.info(f"Loading dataset metadata from {os.path.join(path, info_fname)}")
        with open(os.path.join(path, info_fname), "r") as f:
            meta = json.load(f)
            if meta.get("name") is not None and meta.get("id") is None:
                logger.warning("Using 'name' field as 'id' field in metadata (nerfbaselines version <1.1.0)")
                meta["id"] = meta["name"]
        loader = meta.pop("loader", None)
        loader_kwargs = meta.pop("loader_kwargs", None)
        if loader is None and loader_kwargs is not None:
            logger.warning("Ignoring nb-info.json loader_kwargs because loader is not specified")
            loader_kwargs = None
        for k, v in (loader_kwargs or {}).items():
            if k not in kwargs:
                kwargs[k] = v

    # Add split back to kwargs
    kwargs["split"] = split

    # If loader is None, try detecting a dataset
    if loader is None:
        loader, dataset_instance = _load_unknown_dataset(path, **kwargs)
    else:
        # Load the dataset using the specified loader
        load_fn = _resolve_loader(loader)
        dataset_instance = load_fn(path, **kwargs)

    name = dataset_instance["metadata"].get("id", None)
    logger.info(f"Loaded {name or 'unknown'} dataset from path {path} using loader {loader}")

    # Dataset loaded successfully
    # Now we apply the postprocessing
    # We update the metadata from nb-info.json
    meta = _parse_metadata(meta)
    dataset_instance["metadata"].update(meta)
    if split == "train":
        # For train split, we compute additional metadata
        dataset_type = dataset_instance["metadata"].get("type", None)
        viewer_transform, viewer_pose = get_default_viewer_transform(
            dataset_instance["cameras"].poses, dataset_type)
        # And we set the viewer transform and initial pose if missing
        if dataset_instance["metadata"].get("viewer_transform") is None:
            dataset_instance["metadata"]["viewer_transform"] = viewer_transform
        if dataset_instance["metadata"].get("viewer_initial_pose") is None:
            dataset_instance["metadata"]["viewer_initial_pose"] = viewer_pose
        if "expected_scene_scale" not in  dataset_instance["metadata"]:
            dataset_instance["metadata"]["expected_scene_scale"] = \
                get_scene_scale(dataset_instance["cameras"], dataset_type)

    # Set correct eval protocol
    if name is not None:
        spec_ = get_dataset_spec(name)
        if spec_ is not None:
            eval_protocol = spec_.get("evaluation_protocol", "default")
            if dataset_instance["metadata"].get("evaluation_protocol", "default") != eval_protocol:
                raise RuntimeError(f"Evaluation protocol mismatch: {dataset_instance['metadata']['evaluation_protocol']} != {eval_protocol}")
            dataset_instance["metadata"]["evaluation_protocol"] = eval_protocol

    if load_features:
        return dataset_load_features(dataset_instance, features=kwargs["features"], supported_camera_models=supported_camera_models)
    return dataset_instance
