import os
import typing
import logging
import numpy as np
from glob import glob
from tqdm import trange
from typing import Optional, List
from nerfbaselines.types import camera_model_to_int, new_cameras, FrozenSet, DatasetFeature
from nerfbaselines.datasets import DatasetNotFoundError, new_dataset, dataset_index_select
from nerfbaselines.datasets._common import get_default_viewer_transform
from PIL import Image


LOADER_NAME = "bundler"


def get_split_and_train_indices(image_names, path, split):
    num_images = len(image_names)
    indices = None
    train_indices = np.arange(num_images)
    if split is not None:
        if os.path.exists(os.path.join(path, "train_list.txt")) or os.path.exists(os.path.join(path, "test_list.txt")):
            logging.info(f"{LOADER_NAME} dataloader is loading split data from {os.path.join(path, f'{split}_list.txt')}")
            train_indices = None
            for split in ("train", split):
                with open(os.path.join(path, f"{split}_list.txt"), "r") as f:
                    split_image_names = set(f.read().splitlines())
                indices = np.array([name in split_image_names for name in image_names], dtype=bool)
                if indices.sum() == 0:
                    raise DatasetNotFoundError(f"no images found for split {split} in {os.path.join(path, f'{split}_list.txt')}")
                if indices.sum() < len(split_image_names):
                    logging.warning(f"only {indices.sum()} images found for split {split} in {os.path.join(path, f'{split}_list.txt')}")
                if split == "train":
                    train_indices = indices
            assert train_indices is not None
        else:
            dataset_len = num_images
            test_indices = list(range(0, num_images, 8))
            test_indices_array: np.ndarray = np.array([i in test_indices for i in range(dataset_len)], dtype=bool)
            train_indices = np.logical_not(test_indices_array)
            indices = train_indices if split == "train" else test_indices_array
    return indices, train_indices


def load_bundler_file(path: str, image_list: List[str]):
    with open(path, "r") as f:
        lines = f.readlines()
    assert lines[0].strip() == "# Bundle file v0.3"

    # Prepare size cache
    _parent, _pathname = os.path.split(path)
    cache_path = os.path.join(_parent, ".cache." + _pathname + ".imsizes")
    sizes_cache = dict()
    if os.path.exists(os.path.join(_parent, ".cache." + _pathname + ".imsizes")):
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    name, w, h = line.split()
                    sizes_cache[name] = (int(w), int(h))

    # Read header
    num_images, num_points = map(int, lines[1].split())
    image_lines = lines[2:2+num_images*5]
    point_lines = lines[2+num_images*5:2+num_images*5 + num_points*3]
    assert len(image_lines) == num_images*5
    assert len(point_lines) == num_points*3
    assert len(image_list) == num_images, f"Expected {num_images} images, got {len(image_list)}"

    # Read images
    intrinsics = np.zeros((num_images, 4), dtype=np.float32)
    distortion_params = np.zeros((num_images, 6), dtype=np.float32)
    camera_types = np.full((num_images,), camera_model_to_int("pinhole"), dtype=np.uint8)
    c2ws = np.zeros((num_images, 3, 4), dtype=np.float32)
    image_sizes = np.zeros((num_images, 2), dtype=np.int32)
    for img in trange(0, num_images, desc="Loading bundler file"):
        focal, k1, k2 = map(float, image_lines[5*img].split())
        if k1 != 0 or k2 != 0:
            camera_types[img] = np.full((num_images,), camera_model_to_int("opencv_fisheye"), dtype=np.uint8)

        # We need to read the images to get the w and h
        if image_list[img] not in sizes_cache:
            img_pil = Image.open(image_list[img])
            w, h = img_pil.size
            img_pil.close()
            sizes_cache[image_list[img]] = (w, h)
        w, h = sizes_cache[image_list[img]]
        intrinsics[img] = np.array([focal, focal, w/2, h/2], dtype=intrinsics.dtype)
        distortion_params[img] = np.array([k1, k2, 0, 0, 0, 0], dtype=distortion_params.dtype)
        image_sizes[img] = np.array([w, h], dtype=image_sizes.dtype)
        
        # Read R, t
        Rt = np.fromstring(" ".join(image_lines[5*img+1:5*img+5]), dtype=c2ws.dtype, sep=" ").reshape(4, 3)
        R_w2c = Rt[:3]
        t_w2c = Rt[3]

        # Convert to c2w matrix
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        c2ws[img] = np.concatenate([R_c2w, t_c2w[:, None]], 1)

    # Switch from OpenGL (bundler default) to OpenCV
    c2ws[...,0:3, 1:3] *= -1
    xyz = np.zeros((num_points, 3), dtype=np.float32)
    rgb = np.zeros((num_points, 3), dtype=np.uint8)
    for point in range(0, num_points):
        xyz[point] = np.fromstring(point_lines[point*3], dtype=xyz.dtype, sep=" ")
        rgb[point] = np.fromstring(point_lines[point*3+1], dtype=rgb.dtype, sep=" ")
    cameras = new_cameras(
        poses=c2ws,
        intrinsics=intrinsics,
        camera_types=camera_types,
        distortion_parameters=distortion_params,
        image_sizes=image_sizes,
        nears_fars=None,
    )
    # Store image sizes cache
    with open(cache_path, "w") as f:
        for name, (w, h) in sizes_cache.items():
            f.write(f"{name} {w} {h}\n")
    return cameras, (xyz, rgb)


def load_bundler_dataset(path: str, 
                         split: Optional[str] = None, 
                         images_path: Optional[str] = None,
                         features: Optional[FrozenSet[DatasetFeature]] = None,
                         sampling_masks_path: Optional[str] = None):
    if features is None:
        features = typing.cast(FrozenSet[DatasetFeature], {})
    if not os.path.exists(os.path.join(path, "cameras.out")):
        raise DatasetNotFoundError(f"Could not find cameras.out in {path}")
    load_points = "points3D_xyz" in features or "points3D_rgb" in features
    if split:
        assert split in {"train", "test"}

    if "images_points3D_indices" in features:
        # TODO: Implement this feature
        raise NotImplementedError("images_points3D_indices is not implemented for the bundler loader")

    images_root = os.path.join(path, "images") if images_path is None else os.path.join(path, images_path)
    images_root = os.path.realpath(images_root)

    images_masks_root = os.path.join(path, "sampling_masks") if sampling_masks_path is None else os.path.join(path, sampling_masks_path)
    images_masks_root = os.path.realpath(images_masks_root)

    image_names = [
        os.path.relpath(x, images_root) for x in glob(os.path.join(images_root, "**/*"), recursive=True)
        if x.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    indices, train_indices = get_split_and_train_indices(image_names, path, split)
    abs_paths = [os.path.join(images_root, x) for x in image_names]
    sampling_mask_paths = None if not os.path.exists(images_masks_root) else [os.path.join(images_masks_root, x) for x in image_names]
    all_cameras, (points3D_xyz, points3D_rgb) = load_bundler_file(os.path.join(path, "cameras.out"), abs_paths)
    viewer_transform, viewer_pose = get_default_viewer_transform(all_cameras[train_indices].poses, None)
    dataset = new_dataset(
        cameras=all_cameras,
        image_paths=abs_paths,
        image_paths_root=images_root,
        sampling_mask_paths=sampling_mask_paths,
        sampling_mask_paths_root=images_masks_root,
        points3D_xyz=points3D_xyz if load_points else None,
        points3D_rgb=points3D_rgb if load_points else None,
        metadata={
            "name": "colmap",
            "color_space": "srgb",
            "evaluation_protocol": "default",
            "viewer_transform": viewer_transform,
            "viewer_initial_pose": viewer_pose,
        })
    if indices is not None:
        dataset = dataset_index_select(dataset, indices)

    return dataset


if __name__ == "__main__":
    import sys
    dataset = load_bundler_dataset(sys.argv[1], split="train")
    print(len(dataset["image_paths"]))
