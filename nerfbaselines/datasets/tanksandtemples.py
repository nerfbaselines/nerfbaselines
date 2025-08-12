import shutil
from functools import partial
import json
import os
import logging
from typing import TypeVar
import numpy as np
from nerfbaselines import DatasetNotFoundError
from nerfbaselines.datasets import dataset_index_select
from nerfbaselines.datasets.colmap import load_colmap_dataset
from nerfbaselines._constants import DATASETS_REPOSITORY
from nerfbaselines.datasets import _colmap_utils as colmap_utils
from ._common import download_dataset_wrapper, download_archive_dataset


T = TypeVar("T")
DATASET_NAME = "tanksandtemples"
BASE_URL = f"https://{DATASETS_REPOSITORY}/resolve/main/tanksandtemples"
VERSION = "1"
_URL = f"{BASE_URL}/{{scene}}.tar.gz"
del _URL
_URL2DOWN = f"{BASE_URL}/{{scene}}_2down.tar.gz"
SCENES = {
    # advanced
    "auditorium": True,
    "ballroom": True,
    "courtroom": True,
    "museum": True,
    "palace": True,
    "temple": True,

    # intermediate
    "family": True,
    "francis": True,
    "horse": True,
    "lighthouse": True,
    "m60": True,
    "panther": True,
    "playground": True,
    "train": True,

    # training
    "barn": True,
    "caterpillar": True,
    "church": True,
    "courthouse": True,
    "ignatius": True,
    "meetingroom": True,
    "truck": True,
}


def _select_indices_llff(image_names, llffhold=8):
    inds = np.argsort(image_names)
    all_indices = np.arange(len(image_names))
    indices_train = inds[all_indices % llffhold != 0]
    indices_test = inds[all_indices % llffhold == 0]
    return indices_train, indices_test


def _div_round_half_up(x, a):
    q, r = divmod(x, a)
    if 2 * r >= a:
        q += 1
    return q


def _downscale_cameras_v1(cameras_path, output_cameras_path, downscale_factor: int):
    cameras = colmap_utils.read_cameras_binary(cameras_path)
    new_cameras = {}
    for k, v in cameras.items():
        assert v.model == "PINHOLE", f"Expected PINHOLE camera model, got {v.model}."
        params = v.params
        oldw, oldh = v.width, v.height
        w = _div_round_half_up(v.width, downscale_factor)
        h = _div_round_half_up(v.height, downscale_factor)
        multx, multy = np.array([w, h], dtype=np.float64) / np.array([oldw, oldh], dtype=np.float64)
        multipliers = np.stack([multx, multy, multx, multy], -1)
        params = params * multipliers
        new_camera = colmap_utils.Camera(
            id=v.id,
            model=v.model,
            width=w,
            height=h,
            params=params,
        )
        new_cameras[k] = new_camera

    # Write output
    os.makedirs(os.path.dirname(output_cameras_path), exist_ok=True)
    colmap_utils.write_cameras_binary(new_cameras, output_cameras_path)


def _downscale_cameras_v2(cameras_path, output_cameras_path, downscale_factor: int):
    cameras = colmap_utils.read_cameras_binary(cameras_path)
    new_cameras = {}
    for k, v in cameras.items():
        assert v.model == "PINHOLE", f"Expected PINHOLE camera model, got {v.model}."
        params = v.params
        oldw, oldh = v.width, v.height
        fx, fy, cx, cy = params.tolist()
        fx = fx / downscale_factor
        fy = fy / downscale_factor
        w = _div_round_half_up(v.width, downscale_factor)
        h = _div_round_half_up(v.height, downscale_factor)
        cx = (cx - oldw/2) / downscale_factor + w/2
        cy = (cy - oldh/2) / downscale_factor + h/2
        new_camera = colmap_utils.Camera(
            id=v.id,
            model=v.model,
            width=w,
            height=h,
            params=np.array([fx, fy, cx, cy], dtype=np.float64),
        )
        new_cameras[k] = new_camera

    # Write output
    os.makedirs(os.path.dirname(output_cameras_path), exist_ok=True)
    colmap_utils.write_cameras_binary(new_cameras, output_cameras_path)

    # Write output
    os.makedirs(os.path.dirname(output_cameras_path), exist_ok=True)
    colmap_utils.write_cameras_binary(new_cameras, output_cameras_path)


def _finish_dataset(downscale_factor, output):
    # Write sparse_downscale info
    shutil.copytree(
        os.path.join(output, "sparse"),
        os.path.join(output, f"sparse_{downscale_factor}/0"),
        dirs_exist_ok=True,
    )
    _downscale_cameras_v1(
        os.path.join(output, f"sparse_{downscale_factor}/0/cameras.bin"),
        os.path.join(output, f"sparse_{downscale_factor}/0/cameras.bin"),
        downscale_factor,
    )

    # Write splits
    with open(os.path.join(output, "nb-info.json"), "r", encoding="utf8") as f:
        dataset_info = json.load(f)
    try:
        colmap_dataset = load_colmap_dataset(output, split=None, **dataset_info["loader_kwargs"])
    except DatasetNotFoundError as e:
        raise RuntimeError(f"Failed to load dataset {output} after downloading.") from e
    indices_train, indices_test = _select_indices_llff(colmap_dataset["image_paths"])
    with open(os.path.join(str(output), "train_list.txt"), "w", encoding="utf8") as f:
        for img_name in dataset_index_select(colmap_dataset, indices_train)["image_paths"]:
            f.write(os.path.relpath(img_name, colmap_dataset["image_paths_root"]) + "\n")
    with open(os.path.join(str(output), "test_list.txt"), "w", encoding="utf8") as f:
        for img_name in dataset_index_select(colmap_dataset, indices_test)["image_paths"]:
            f.write(os.path.relpath(img_name, colmap_dataset["image_paths_root"]) + "\n")


@download_dataset_wrapper(SCENES, DATASET_NAME)
def download_tanksandtemples_dataset(path: str, output: str) -> None:
    dataset_name, scene = path.split("/", 1)
    if SCENES.get(scene) is None:
        raise RuntimeError(f"Unknown scene {scene}")
    url = _URL2DOWN.format(scene=scene)
    downscale_factor = 2
    prefix = scene + "/"
    nb_info = {
        "id": dataset_name,
        "scene": scene,
        "loader": "colmap",
        "evaluation_protocol": "default",
        "type": "object-centric",
        "downscale_factor": downscale_factor,
        "loader_kwargs": {
            "images_path": f"images_{downscale_factor}",
            "colmap_path": f"sparse_{downscale_factor}/0",
        },
        "version": VERSION,
    }
    download_archive_dataset(url, output, 
                             archive_prefix=prefix, 
                             nb_info=nb_info,
                             callback=partial(_finish_dataset, downscale_factor),
                             file_type="tar.gz")
    logging.info(f"Downloaded {DATASET_NAME}/{scene} to {output}")


def load_tanksandtemples_dataset(path, *args, **kwargs):
    del args, kwargs
    raise RuntimeError(f"The dataset was likely downloaded with an older version of NerfBaselines. Please remove `{path}` and try again.")


download_tanksandtemples_dataset.version = VERSION  # type: ignore
__all__ = ["download_tanksandtemples_dataset"]
