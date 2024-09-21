import json
import os
import logging
from pathlib import Path
from typing import Union, TypeVar
import numpy as np
from nerfbaselines import DatasetNotFoundError
from nerfbaselines.datasets import dataset_index_select
from nerfbaselines.datasets.colmap import load_colmap_dataset
from nerfbaselines._constants import DATASETS_REPOSITORY
from ._common import download_dataset_wrapper, download_archive_dataset


T = TypeVar("T")
DATASET_NAME = "tanksandtemples"
BASE_URL = f"https://{DATASETS_REPOSITORY}/resolve/main/tanksandtemples"
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


def _write_splits(output):
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
        },
    }
    download_archive_dataset(url, output, 
                             archive_prefix=prefix, 
                             nb_info=nb_info,
                             callback=_write_splits,
                             file_type="tar.gz")
    logging.info(f"Downloaded {DATASET_NAME}/{scene} to {output}")


def load_tanksandtemples_dataset(path, *args, **kwargs):
    del args, kwargs
    raise RuntimeError(f"The dataset was likely downloaded with an older version of NerfBaselines. Please remove `{path}` and try again.")


__all__ = ["download_tanksandtemples_dataset"]
