import warnings
import typing
from collections import OrderedDict
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union, FrozenSet
import numpy as np
from nerfbaselines import DatasetFeature, CameraModel, camera_model_to_int, new_cameras, DatasetNotFoundError, new_dataset, Cameras
from ..utils import Indices
from . import _colmap_utils as colmap_utils
from ._common import padded_stack, dataset_index_select


def camera_to_colmap_camera(camera: Cameras, id) -> colmap_utils.Camera:
    camera = camera.item()

    fx, fy, cx, cy = camera.intrinsics
    width, height = camera.image_sizes
    ds = camera.distortion_parameters
    if camera.camera_models == camera_model_to_int("pinhole"):
        return colmap_utils.Camera(
            id=id, model="PINHOLE", width=width, height=height,
            params=np.array([fx, fy, cx, cy], dtype=np.float64)
        )
    elif camera.camera_models == camera_model_to_int("opencv"):
        return colmap_utils.Camera(
            id=id, model="OPENCV", width=width, height=height,
            params=np.array([fx, fy, cx, cy, ds[0], ds[1], ds[2], ds[3]], dtype=np.float64)
        )
    elif camera.camera_models == camera_model_to_int("opencv_fisheye"):
        return colmap_utils.Camera(
            id=id, model="OPENCV_FISHEYE", width=width, height=height,
            params=np.array([fx, fy, cx, cy, ds[0], ds[1], ds[4], ds[5]], dtype=np.float64)
        )
    elif camera.camera_models == camera_model_to_int("full_opencv"):
        return colmap_utils.Camera(
            id=id, model="FULL_OPENCV", width=width, height=height,
            params=np.array([fx, fy, cx, cy, ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], ds[6], ds[7]], dtype=np.float64)
        )
    else:
        raise NotImplementedError(f"Camera model {camera.camera_models} is not supported for COLMAP export!")


def _parse_colmap_camera_params(camera: colmap_utils.Camera) -> Tuple[np.ndarray, int, np.ndarray, Tuple[int, int]]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    out = OrderedDict()  # Default in Python 3.7+
    camera_params = camera.params
    camera_model: CameraModel
    if camera.model == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[0])
        cx = float(camera_params[1])
        cy = float(camera_params[2])
        camera_model = "pinhole"
    elif camera.model == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        fl_x = float(camera_params[0])
        fl_y = float(camera_params[1])
        cx = float(camera_params[2])
        cy = float(camera_params[3])
        camera_model = "pinhole"
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
        camera_model = "opencv"
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
        camera_model = "opencv"
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
        camera_model = "opencv"
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
        camera_model = "opencv_fisheye"
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
        camera_model = "opencv_fisheye"
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
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = "opencv_fisheye"
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model} camera model is not supported yet!")

    image_width: int = camera.width
    image_height: int = camera.height
    intrinsics = np.array([fl_x, fl_y, cx, cy], dtype=np.float32)
    attributes = ["k1", "k2", "p1", "p2", "k3", "k4"]
    if "k5" in out or "k6" in out:
        attributes.extend(("k5", "k6"))
    distortion_params = np.array([out.get(k, 0.0) for k in attributes], dtype=np.float32)
    return intrinsics, camera_model_to_int(camera_model), distortion_params, (image_width, image_height)


def _select_indices_llff(image_names, llffhold=8):
    inds = np.argsort(image_names)
    all_indices = np.arange(len(image_names))
    indices_train = inds[all_indices % llffhold != 0]
    indices_test = inds[all_indices % llffhold == 0]
    return indices_train, indices_test


def load_colmap_dataset(path: Union[Path, str],
        split: Optional[str] = None, 
        *,
        test_indices: Optional[Indices] = None,
        features: Optional[FrozenSet[DatasetFeature]] = None,
        images_path: Optional[Union[Path, str]] = None, 
        colmap_path: Optional[Union[Path, str]] = None,
        masks_path: Optional[Union[Path, str]] = None,
        sampling_masks_path: Optional[Union[Path, str]] = None):
    if sampling_masks_path is not None:
        # Obsolete
        warnings.warn("sampling_masks_path is deprecated and will be removed in the future. Use masks_path instead.")
        masks_path = sampling_masks_path
    path = Path(path)
    if images_path is not None:
        images_path = Path(images_path)
    if colmap_path is not None:
        colmap_path = Path(colmap_path)
    if features is None:
        features = typing.cast(FrozenSet[DatasetFeature], frozenset())
    load_points = "points3D_xyz" in features or "points3D_rgb" in features
    if split:
        assert split in {"train", "test"}
    # Load COLMAP dataset
    if colmap_path is None:
        colmap_path = "sparse/0"
        if not (path / colmap_path).exists():
            colmap_path = "sparse"
    rel_colmap_path = colmap_path
    colmap_path = (path / colmap_path).resolve()
    if images_path is None:
        images_path = "images"
    rel_images_path = images_path
    images_path = (path / images_path).resolve()
    if masks_path is None:
        masks_path = "masks"
        masks_path = (path / masks_path).resolve()
        if not masks_path.exists():
            masks_path = None
    else:
        masks_path = Path(masks_path)
        masks_path = (path / masks_path).resolve()
    if not colmap_path.exists():
        raise DatasetNotFoundError(f"Missing '{rel_colmap_path}' folder in COLMAP dataset")
    if not (colmap_path / "cameras.bin").exists() and not (colmap_path / "cameras.txt").exists():
        raise DatasetNotFoundError(f"Missing '{rel_colmap_path}/cameras.{{bin,txt}}' file in COLMAP dataset")
    if not images_path.exists():
        raise DatasetNotFoundError(f"Missing '{rel_images_path}' folder in COLMAP dataset")

    if (colmap_path / "cameras.bin").exists():
        colmap_cameras = colmap_utils.read_cameras_binary(colmap_path / "cameras.bin")
    elif (colmap_path / "cameras.txt").exists():
        colmap_cameras = colmap_utils.read_cameras_text(colmap_path / "cameras.txt")
    else:
        raise DatasetNotFoundError("Missing 'sparse/0/cameras.{bin,txt}' file in COLMAP dataset")

    if not (colmap_path / "images.bin").exists() and not (colmap_path / "images.txt").exists():
        raise DatasetNotFoundError("Missing 'sparse/0/images.{bin,txt}' file in COLMAP dataset")
    if (colmap_path / "images.bin").exists():
        images = colmap_utils.read_images_binary(colmap_path / "images.bin")
    elif (colmap_path / "images.txt").exists():
        images = colmap_utils.read_images_text(colmap_path / "images.txt")
    else:
        raise DatasetNotFoundError("Missing 'sparse/0/images.{bin,txt}' file in COLMAP dataset")

    points3D: Optional[Dict[int, colmap_utils.Point3D]] = None
    if load_points:
        if not (colmap_path / "points3D.bin").exists() and not (colmap_path / "points3D.txt").exists():
            raise DatasetNotFoundError("Missing 'sparse/0/points3D.{bin,txt}' file in COLMAP dataset")
        if (colmap_path / "points3D.bin").exists():
            points3D = colmap_utils.read_points3D_binary(colmap_path / "points3D.bin")
        elif (colmap_path / "points3D.txt").exists():
            points3D = colmap_utils.read_points3D_text(colmap_path / "points3D.txt")
        else:
            raise DatasetNotFoundError("Missing 'sparse/0/points3D.{bin,txt}' file in COLMAP dataset")

    # Convert to tensors
    camera_intrinsics = []
    camera_poses = []
    camera_models = []
    camera_distortion_params = []
    image_paths: List[str] = []
    image_names = []
    mask_paths: Optional[List[str]] = None if not masks_path is not None else []
    camera_sizes = []

    image: colmap_utils.Image
    i = 0
    c2w: np.ndarray
    images_points3D_ids = []
    images_points2D_xy = []
    for image in images.values():
        camera: colmap_utils.Camera = colmap_cameras[image.camera_id]
        intrinsics, camera_model, distortion_params, (w, h) = _parse_colmap_camera_params(camera)
        camera_sizes.append(np.array((w, h), dtype=np.int32))
        camera_intrinsics.append(intrinsics)
        camera_models.append(camera_model)
        camera_distortion_params.append(distortion_params)
        image_names.append(image.name)
        image_paths.append(str(images_path / image.name))
        if mask_paths is not None:
            assert masks_path is not None, "masks_path is None"
            mask_paths.append(str(masks_path / Path(image.name).with_suffix(".png")))

        # rotation = qvec2rotmat(image.qvec).astype(np.float32)
        # translation = image.tvec.reshape(3, 1).astype(np.float32)
        # w2c = np.concatenate([rotation, translation], 1)
        # w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]], dtype=w2c.dtype)], 0)
        # c2w = np.linalg.inv(w2c)[:3, :4]

        # w2c
        R = colmap_utils.qvec2rotmat(image.qvec.astype(np.float64))
        t = image.tvec.reshape(3, 1).astype(np.float64)
        # c2w
        c2w = np.concatenate([R.T, -np.matmul(R.T, t)], axis=-1)
        camera_poses.append(c2w)

        i += 1

        if "images_points2D_xy" in features:
            images_points2D_xy.append(image.xys[image.point3D_ids >= 0].astype(np.float32))

        if "images_points3D_indices" in features:
            images_points3D_ids.append(image.point3D_ids[image.point3D_ids >= 0])



    # Estimate nears fars
    # near = 0.01
    # far = np.stack([x[:3, -1] for x in camera_poses], 0)
    # far = float(np.percentile(np.linalg.norm(far - np.mean(far, keepdims=True, axis=0), axis=-1), 90, axis=0))
    # nears_fars = np.array([[near, far]] * len(camera_poses), dtype=np.float32)
    nears_fars = None

    # Load points
    points3D_xyz = None
    points3D_rgb = None
    points3D_error = None
    images_points3D_indices = None
    if load_points:
        assert points3D is not None, "3D points have not been loaded"
        points3D_xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float32)
        points3D_rgb = np.array([p.rgb for p in points3D.values()], dtype=np.uint8)
        if "points3D_error" in features:
            points3D_error = np.array([p.error for p in points3D.values()], dtype=np.float32)
        if "images_points3D_indices" in features:
            images_points3D_indices = []
            ptmap = {point3D_id: i for i, point3D_id in enumerate(points3D.keys())}
            for ids in images_points3D_ids:
                indices3D = np.array([
                    ptmap[point3D_id] for point3D_id in ids if point3D_id != -1
                ], dtype=np.int32)
                images_points3D_indices.append(indices3D)

    # camera_ids=torch.tensor(camera_ids, dtype=torch.int32),
    all_cameras = new_cameras(
        poses=np.stack(camera_poses, 0).astype(np.float32),
        intrinsics=np.stack(camera_intrinsics, 0).astype(np.float32),
        camera_models=np.array(camera_models, dtype=np.int32),
        distortion_parameters=padded_stack(camera_distortion_params).astype(np.float32),
        image_sizes=np.stack(camera_sizes, 0).astype(np.int32),
        nears_fars=nears_fars,
    )
    indices = None
    train_indices = np.arange(len(image_paths))
    if split is not None:
        if test_indices is None and ((path / "train_list.txt").exists() or (path / "test_list.txt").exists()):
            logging.info(f"Colmap dataloader is loading split data from {path / f'{split}_list.txt'}")
            train_indices = None
            for split in ("train", split):
                split_image_names = set((path / f"{split}_list.txt").read_text().splitlines())
                indices = np.array([name in split_image_names for name in image_names], dtype=bool)
                if indices.sum() == 0:
                    raise DatasetNotFoundError(f"no images found for split {split} in {path / f'{split}_list.txt'}")
                if indices.sum() < len(split_image_names):
                    logging.warning(f"Only {indices.sum()} images found for split {split} in {path / f'{split}_list.txt'}")
                if split == "train":
                    train_indices = indices
            assert train_indices is not None
            logging.info(f"Colmap dataloader is using LLFF split with {train_indices.sum()} training images")
        elif test_indices is None:
            train_indices, test_indices_array = _select_indices_llff(image_names)
            indices = train_indices if split == "train" else test_indices_array
            logging.info(f"Colmap dataloader is using LLFF split with {len(train_indices)} training and {len(test_indices_array)} test images")
        else:
            dataset_len = len(image_paths)
            test_indices.total = dataset_len
            test_indices_array: np.ndarray = np.array([i in test_indices for i in range(dataset_len)], dtype=bool)
            train_indices = np.logical_not(test_indices_array)
            indices = train_indices if split == "train" else test_indices_array

            # Apply indices to sorted image names (COLMAP order is arbitrary)
            indices = np.argsort(image_names)[indices]
            logging.info(f"Colmap dataloader is using LLFF split with {train_indices.sum()} training and {test_indices_array.sum()} test images")

    dataset = new_dataset(
        cameras=all_cameras,
        image_paths=image_paths,
        image_paths_root=str(images_path),
        mask_paths=mask_paths,
        mask_paths_root=str(masks_path) if mask_paths is not None else None,
        points3D_xyz=points3D_xyz,
        points3D_rgb=points3D_rgb,
        points3D_error=points3D_error,
        images_points2D_xy=images_points2D_xy if "images_points2D_xy" in features else None,
        images_points3D_indices=images_points3D_indices if "images_points3D_indices" in features else None,
        metadata={
            "id": None,
            "color_space": "srgb",
            "evaluation_protocol": "default",
        })
    if indices is not None:
        dataset = dataset_index_select(dataset, indices)

    return dataset


__all__ = ["load_colmap_dataset"]
