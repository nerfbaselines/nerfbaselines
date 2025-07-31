import os
import typing
import logging
import numpy as np
from glob import glob
from tqdm import trange
from typing import Optional, List, Tuple, Union, cast, FrozenSet
from nerfbaselines import camera_model_to_int, new_cameras, DatasetFeature, DatasetNotFoundError, new_dataset
from nerfbaselines.datasets import dataset_index_select
from PIL import Image
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


LOADER_NAME = "bundler"
KNOWN_ATTRIBUTES = "w h f fx fy cx cy k1 k2 k3 k4 p1 p2 delta_cx delta_cy filename".split()
CoordinateSystem = Literal["opengl", "opencv"]


def _select_indices_llff(image_names, llffhold=8):
    inds = np.argsort(image_names)
    all_indices = np.arange(len(image_names))
    indices_train = inds[all_indices % llffhold != 0]
    indices_test = inds[all_indices % llffhold == 0]
    return indices_train, indices_test


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
            train_indices, test_indices = _select_indices_llff(image_names)
            indices = train_indices if split == "train" else test_indices
    return indices, train_indices


def load_bundler_file(path: str, 
                      images_path: str,
                      attributes, 
                      camera_model=None,
                      coordinate_system="opengl"):
    with open(path, "r") as f:
        lines = f.readlines()
    assert lines[0].strip().startswith("# Bundle file v0.3")

    # Read header
    num_images, num_points = map(int, lines[1].split())
    image_lines = lines[2:2+num_images*5]
    point_lines = lines[2+num_images*5:2+num_images*5 + num_points*3]
    assert len(image_lines) == num_images*5
    assert len(point_lines) == num_points*3
    if "filename" not in attributes:
        image_list = [
            os.path.relpath(x, images_path) for x in glob(os.path.join(images_path, "**/*"), recursive=True)
            if x.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        assert len(image_list) == num_images, f"Expected {num_images} images, got {len(image_list)}"
    else:
        image_list = [None] * num_images

    # Read images
    intrinsics = np.zeros((num_images, 4), dtype=np.float32)
    distortion_params = np.zeros((num_images, 6), dtype=np.float32)
    camera_models = np.full((num_images,), camera_model_to_int("pinhole"), dtype=np.uint8)
    c2ws = np.zeros((num_images, 3, 4), dtype=np.float32)
    image_sizes = np.zeros((num_images, 2), dtype=np.int32)
    for img in trange(0, num_images, desc="Loading bundler file"):
        props = {}

        attr_values = image_lines[5*img].split()
        assert len(attr_values) == len(attributes), f"Expected {len(attributes)} attributes, got {len(attr_values)}"
        for attr, value in zip(attributes, attr_values):
            if attr not in KNOWN_ATTRIBUTES:
                raise RuntimeError(f"Unknown attribute {attr}")
            if attr == "filename":
                pass
            elif attr == "w" or attr == "h":
                value = int(value)
            else:
                value = float(value)
            props[attr] = value

        camera_model_ = camera_model_to_int(camera_model or "opencv")
        if camera_model == "pinhole":
            for attr in "k1 k2 k3 k4 p1 p2".split():
                assert props.get(attr) == 0, f"Expected {attr} to be 0 for pinhole camera model"
        elif camera_model == "opencv" or camera_model is None:
            if all((props.get(x, 0.0) == 0.0) for x in "k1 k2 k3 k4 p1 p2".split()):
                camera_model_ = camera_model_to_int("pinhole")
                logging.debug(f"Camera {img} has all distortion parameters set to 0, switching to pinhole camera model")
        camera_models[img] = camera_model_

        if props.get("filename") is not None:
            image_list[img] = filename = props.pop("filename")
        else:
            filename = image_list[img]

        # We need to read the images to get the w and h if not available
        if props.get("w") is None or props.get("h") is None:
            assert filename is not None, f"Missing filename for image {img}"
            img_pil = Image.open(os.path.join(images_path, filename))
            props["w"], props["h"] = img_pil.size
            img_pil.close()

        # Fill missing fx, fy
        if props.get("fx") is None or props.get("fy") is None:
            props["fx"] = props["fy"] = props.pop("f")

        # Fill missing cx, cy
        if props.get("cx") is None:
            if props.get("delta_cx") is not None:
                props["cx"] = props["w"]/2 + props.pop("delta_cx")
            else:
                props["cx"] = props["w"] / 2
        if props.get("cy") is None:
            if props.get("delta_cy") is not None:
                props["cy"] = props["h"]/2 + props.pop("delta_cy")
            else:
                props["cy"] = props["h"] / 2

        # Fill missing distortion parameters
        for attr in "k1 k2 k3 k4 p1 p2".split():
            props[attr] = 0.0 if props.get(attr) is None else props[attr]

        intrinsics[img] = np.array([props["fx"], props["fy"], props["cx"], props["cy"]], dtype=intrinsics.dtype)
        distortion_params[img] = np.array([props[a] for a in "k1 k2 p1 p2 k3 k4".split()], dtype=distortion_params.dtype)
        image_sizes[img] = np.array([int(props["w"]), int(props["h"])], dtype=image_sizes.dtype)
        
        # Read R, t
        Rt = np.fromstring(" ".join(image_lines[5*img+1:5*img+5]), dtype=c2ws.dtype, sep=" ").reshape(4, 3)

        R_w2c = Rt[:3]
        t_w2c = Rt[3]

        # Convert to c2w matrix
        R_c2w = R_w2c.T
        t_c2w = -R_c2w @ t_w2c
        c2ws[img] = np.concatenate([R_c2w, t_c2w[:, None]], 1)

    # Switch from OpenGL (bundler default) to OpenCV
    if coordinate_system == "opengl":
        c2ws[...,0:3, 1:3] *= -1
    elif coordinate_system == "opencv":
        pass
    else:
        raise ValueError(f"Unknown coordinate system {coordinate_system}")

    xyz = np.zeros((num_points, 3), dtype=np.float32)
    rgb = np.zeros((num_points, 3), dtype=np.uint8)
    for point in range(0, num_points):
        xyz[point] = np.fromstring(point_lines[point*3], dtype=xyz.dtype, sep=" ")
        rgb[point] = np.fromstring(point_lines[point*3+1], dtype=rgb.dtype, sep=" ")

    cameras = new_cameras(
        poses=c2ws,
        intrinsics=intrinsics,
        camera_models=camera_models,
        distortion_parameters=distortion_params,
        image_sizes=image_sizes,
        nears_fars=None,
    )
    return cameras, (xyz, rgb), cast(List[str], image_list)


def load_bundler_dataset(path: str, 
                         split: Optional[str] = None, 
                         images_path: Optional[str] = None,
                         features: Optional[FrozenSet[DatasetFeature]] = None,
                         bundler_file: str = "cameras.out",
                         attributes: Union[str, Tuple[str, ...]] = tuple("f,cx,cy".split(",")),
                         camera_model: Optional[str] = None,
                         coordinate_system: CoordinateSystem = "opengl",
                         masks_path: Optional[str] = None):
    if isinstance(attributes, str):
        attributes = tuple(attributes.split(","))
    if features is None:
        features = typing.cast(FrozenSet[DatasetFeature], {})
    if not os.path.exists(os.path.join(path, bundler_file)):
        raise DatasetNotFoundError(f"Could not find {bundler_file} in {path}")
    load_points = "points3D_xyz" in features or "points3D_rgb" in features
    if split:
        assert split in {"train", "test"}

    if "images_points3D_indices" in features:
        # TODO: Implement this feature
        raise NotImplementedError("images_points3D_indices is not implemented for the bundler loader")

    images_root = os.path.join(path, "images") if images_path is None else os.path.join(path, images_path)
    images_root = os.path.realpath(images_root)

    images_masks_root = os.path.join(path, "masks") if masks_path is None else os.path.join(path, masks_path)
    images_masks_root = os.path.realpath(images_masks_root)

    all_cameras, (points3D_xyz, points3D_rgb), image_names = load_bundler_file(os.path.join(path, bundler_file), 
                                                                               images_root,
                                                                               attributes=attributes,
                                                                               coordinate_system=coordinate_system,
                                                                               camera_model=camera_model)
    indices, train_indices = get_split_and_train_indices(image_names, path, split)
    dataset = new_dataset(
        cameras=all_cameras,
        image_paths=[os.path.join(images_root, x) for x in image_names],
        image_paths_root=images_root,
        mask_paths=None if not os.path.exists(images_masks_root) else [os.path.join(images_masks_root, x) for x in image_names],
        mask_paths_root=images_masks_root,
        points3D_xyz=points3D_xyz if load_points else None,
        points3D_rgb=points3D_rgb if load_points else None,
        metadata={
            "id": None,
            "color_space": "srgb",
            "evaluation_protocol": "default",
        })
    if indices is not None:
        dataset = dataset_index_select(dataset, indices)

    return dataset


__all__ = ["load_bundler_dataset"]


if __name__ == "__main__":
    import sys
    dataset = load_bundler_dataset(sys.argv[1], split="train")
    print(len(dataset["image_paths"]))
