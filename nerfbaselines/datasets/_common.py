import struct
import os
import dataclasses
from collections import OrderedDict
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import PIL.Image
from tqdm import tqdm
from ..types import Dataset, DatasetFeature, FrozenSet
from ..utils import Indices
from ..distortion import Distortions, CameraModel
from ._colmap_utils import read_cameras_binary, read_images_binary, read_points3D_binary, qvec2rotmat
from ._colmap_utils import read_cameras_text, read_images_text, read_points3D_text, Image, Camera


def single(xs):
    out = None
    for x in xs:
        if out is not None:
            raise ValueError("Expected single value, got multiple")
        out = x
    if out is None:
        raise ValueError("Expected single value, got none")
    return out


def padded_stack(images: List[np.ndarray]) -> np.ndarray:
    h, w, _ = tuple(max(s) for s in zip(*[img.shape for img in images]))
    out_images = []
    for image in images:
        pad_h = h - image.shape[0]
        pad_w = w - image.shape[1]
        out_images.append(np.pad(image, ((0, pad_w), (0, pad_h), (0, 0))))
    return np.stack(out_images, 0)


def dataset_load_features(dataset: Dataset, required_features):
    images = []
    image_sizes = []
    for p in tqdm(dataset.file_paths, desc="loading images"):
        if str(p).endswith(".bin"):
            assert dataset.color_space is None or dataset.color_space == "linear"
            with open(p, "rb") as f:
                data_bytes = f.read()
                h, w = struct.unpack("ii", data_bytes[:8])
                image = np.frombuffer(data_bytes, dtype=np.float16, count=h*w*4, offset=8).astype(np.float32).reshape([h, w, 4])
            dataset.color_space = "linear"
        else:
            assert dataset.color_space is None or dataset.color_space == "srgb"
            image = np.array(PIL.Image.open(p).convert("RGB"), dtype=np.uint8)
            dataset.color_space = "srgb"
        images.append(image)
        image_sizes.append([image.shape[1], image.shape[0]])

    dataset.images = padded_stack(images)
    dataset.image_sizes = np.array(image_sizes, dtype=np.int32)

    if "sampling_masks" in required_features and dataset.sampling_mask_paths is not None:
        images = []
        for p in tqdm(dataset.file_paths, desc="loading masks"):
            image = np.array(PIL.Image.open(p).convert("L"), dtype=np.float32)
            images.append(image)
        dataset.sampling_masks = padded_stack(images)
    return dataset


class DatasetNotFoundError(Exception):
    pass


def _parse_colmap_camera_params(camera: Camera) -> Tuple[np.ndarray, CameraModel, Tuple[int, int]]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    out = OrderedDict() # Default in Python 3.7+
    camera_params = camera.params
    intrinsics = []
    if camera.model == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        camera_model = CameraModel.PINHOLE
    elif camera.model == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        camera_model = CameraModel.PINHOLE
    elif camera.model == "SIMPLE_RADIAL":
        # f, cx, cy, k

        # r2 = u**2 + v**2;
        # radial = k * r2
        # du = u * radial
        # dv = u * radial
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "RADIAL":
        # f, cx, cy, k1, k2

        # r2 = u**2 + v**2;
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial
        # dv = v * radial
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2

        # uv = u * v;
        # r2 = u**2 + v**2
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial + 2 * p1 * u*v + p2 * (r2 + 2 * u**2)
        # dv = v * radial + 2 * p2 * u*v + p1 * (r2 + 2 * v**2)
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV
    elif camera.model == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4

        # r = sqrt(u**2 + v**2)

        # if r > eps:
        #    theta = atan(r)
        #    theta2 = theta ** 2
        #    theta4 = theta2 ** 2
        #    theta6 = theta4 * theta2
        #    theta8 = theta4 ** 2
        #    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        #    du = u * thetad / r - u;
        #    dv = v * thetad / r - v;
        # else:
        #    du = dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

        # u2 = u ** 2
        # uv = u * v
        # v2 = v ** 2
        # r2 = u2 + v2
        # r4 = r2 * r2
        # r6 = r4 * r2
        # radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) /
        #          (1 + k4 * r2 + k5 * r4 + k6 * r6)
        # du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        # dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "FOV":
        # fx, fy, cx, cy, omega
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")
    elif camera.model == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     thetad = theta * (1 + k * theta2)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     theta4 = theta2 ** 2
        #     thetad = theta * (1 + k * theta2)
        #     thetad = theta * (1 + k1 * theta2 + k2 * theta4)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0.
        out["k4"] = 0.
        camera_model = CameraModel.OPENCV_FISHEYE
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    image_width = camera.width
    image_height = camera.height
    intrinsics = np.array([fl_x, fl_y, cx, cy], dtype=np.float32) / float(image_width)
    distortion = Distortions(np.array([camera_model.value], dtype=np.int32), np.array([list(out.values())], dtype=np.float32))
    return intrinsics, distortion, (image_width, image_height)


def load_colmap_dataset(path: Path,
                        images_path: Optional[Path] = None,
                        split: Optional[str] = None,
                        test_indices: Optional[Indices] = Indices.every_iters(8),
                        features: Optional[FrozenSet[DatasetFeature]] = None):
    if not features:
        features = {}
    load_points = "points3D_xyz" in features or "points3D_rgb" in features
    if split:
        assert split in {"train", "test"}
    # Load COLMAP dataset
    colmap_path = path / "sparse" / "0"
    if images_path is None:
        images_path = Path("images")
    images_path = path / images_path
    if not colmap_path.exists():
        raise DatasetNotFoundError("Missing 'sparse/0' folder in COLMAP dataset")
    if not (colmap_path / "cameras.bin").exists() and not (colmap_path / "cameras.txt").exists():
        raise DatasetNotFoundError("Missing 'sparse/0/cameras.{bin,txt}' file in COLMAP dataset")
    if not images_path.exists():
        raise DatasetNotFoundError("Missing 'images' folder in COLMAP dataset")

    if (colmap_path / "cameras.bin").exists():
        cameras = read_cameras_binary(colmap_path / "cameras.bin")
    elif (colmap_path / "cameras.txt").exists():
        cameras = read_cameras_text(colmap_path / "cameras.txt")
    else:
        raise DatasetNotFoundError("Missing 'sparse/0/cameras.{bin,txt}' file in COLMAP dataset")

    if not (colmap_path / "images.bin").exists() and not (colmap_path / "images.txt").exists():
        raise DatasetNotFoundError("Missing 'sparse/0/images.{bin,txt}' file in COLMAP dataset")
    if (colmap_path / "images.bin").exists():
        images = read_images_binary(colmap_path / "images.bin")
    elif (colmap_path / "images.txt").exists():
        images = read_images_text(colmap_path / "images.txt")
    else:
        raise DatasetNotFoundError("Missing 'sparse/0/images.{bin,txt}' file in COLMAP dataset")

    if load_points:
        if not (colmap_path / "points3D.bin").exists() and not (colmap_path / "points3D.txt").exists():
            raise DatasetNotFoundError("Missing 'sparse/0/points3D.{bin,txt}' file in COLMAP dataset")
        if (colmap_path / "points3D.bin").exists():
            points3D = read_points3D_binary(colmap_path / "points3D.bin")
        elif (colmap_path / "points3D.txt").exists():
            points3D = read_points3D_text(colmap_path / "points3D.txt")
        else:
            raise DatasetNotFoundError("Missing 'sparse/0/points3D.{bin,txt}' file in COLMAP dataset")

    # Convert to tensors
    camera_intrinsics = []
    camera_poses = []
    camera_distortions = []
    image_paths = []
    camera_sizes = []

    image: Image
    i = 0
    for image in images.values():

        camera: Camera = cameras[image.camera_id]
        intrinsics, distortion, (w, h) = _parse_colmap_camera_params(camera)
        camera_sizes.append(np.array((w, h), dtype=np.int32))
        camera_intrinsics.append(intrinsics)
        camera_distortions.append(distortion)
        image_paths.append(images_path / image.name)

        rotation = qvec2rotmat(image.qvec).astype(np.float32)

        translation = image.tvec.reshape(3, 1).astype(np.float32)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]], dtype=w2c.dtype)], 0)
        c2w = np.linalg.inv(w2c)

        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        # c2w = c2w[np.array([1, 0, 2, 3]), :]
        # c2w[2, :] *= -1
        camera_poses.append(c2w[0:3, :])
        i += 1

    # Estimate nears fars
    near = 0.01
    far = c2w[:3, -1]
    far = float(np.percentile((far - np.mean(far, keepdims=True)) * 3, 90))
    nears_fars = np.array([[near, far]] * len(camera_poses), dtype=np.float32)

    # Load points
    points3D_xyz = None
    points3D_rgb = None
    if load_points:
        points3D_xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float32)
        points3D_rgb = np.array([p.rgb for p in points3D.values()], dtype=np.uint8)

    # camera_ids=torch.tensor(camera_ids, dtype=torch.int32),
    dataset = Dataset(
        camera_intrinsics_normalized=np.stack(camera_intrinsics, 0).astype(np.float32),
        camera_poses=np.stack(camera_poses, 0).astype(np.float32),
        camera_distortions=Distortions.cat(camera_distortions),
        image_sizes=np.stack(camera_sizes, 0).astype(np.int32),
        nears_fars=nears_fars,
        file_paths=image_paths,
        points3D_xyz=points3D_xyz,
        points3D_rgb=points3D_rgb,
        sampling_mask_paths=None,
        file_paths_root=images_path,
    )
    if split is not None:
        test_indices.total = len(dataset)
        test_indices = np.array([i for i in range(len(dataset)) if i in test_indices], dtype=bool)
        if split == "train":
            indices = np.logical_not(test_indices)
        else:
            indices = test_indices
        dataset = dataset[indices]
    dataset.metadata["type"] = "colmap"
    return dataset

def load_mipnerf360_dataset(path: Path, split: str, **kwargs):
    if split:
        assert split in {"train", "test"}
    scenes360_res = {
        "bicycle": 4, "flowers": 4, "garden": 4, "stump": 4, "treehill": 4, 
        "bonsai": 2, "counter": 2, "kitchen": 2, "room": 2, 
    }
    if "360" not in str(path) or not any(s in str(path) for s in scenes360_res):
        raise DatasetNotFoundError(f"360 and {set(scenes360_res.keys())} is missing from the dataset path: {path}")

    # Load MipNerf360 dataset
    scene = single(res for res in scenes360_res if str(res) in path.name)
    res = scenes360_res[scene]
    images_path = Path(f"images_{res}")

    # Use split=None to load all images
    # We then select the same images as in the LLFF multinerf dataset loader
    dataset: Dataset = load_colmap_dataset(path, images_path=images_path, split=None, **kwargs)
    dataset.metadata["type"] = "mipnerf360"

    image_names = dataset.file_paths
    inds = np.argsort(image_names)

    all_indices = np.arange(len(dataset))
    llffhold = 8
    if split == "train":
        indices = all_indices % llffhold != 0
    else:
        indices = all_indices % llffhold == 0
    indices = inds[indices]
    return dataset[indices]


SUPPORTED_DATASETS = {
    "mipnerf360": load_mipnerf360_dataset,
    "colmap": load_colmap_dataset,
}


def load_dataset(path: Path, split: str, features: FrozenSet[DatasetFeature]) -> Dataset:
    errors = {}
    for name, load_fn in SUPPORTED_DATASETS.items():
        try:
            dataset = load_fn(path, split=split, features=features)
            logging.info(f"loaded {name} dataset from path {path}")
            return dataset
        except DatasetNotFoundError as e:
            logging.debug(e)
            logging.debug(f"{name} dataset not found in path {path}")
            errors[name] = str(e)
    raise DatasetNotFoundError(f"no supported dataset found in path {path}:"
                               "".join(f"\n  {name}: {error}" for name, error in errors.items()))
