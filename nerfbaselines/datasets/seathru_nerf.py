import json
import zipfile
import sys
import logging
import os
import numpy as np
from typing import Optional
from contextlib import contextmanager
from nerfbaselines import NB_PREFIX, DatasetNotFoundError
from nerfbaselines.datasets import dataset_index_select
from nerfbaselines.datasets.colmap import load_colmap_dataset


DATASET_NAME = "seathru-nerf"
SEATHRUNERF_GDRIVE_ID = "1RzojBFvBWjUUhuJb95xJPSNP3nJwZWaT"
SCENES = {
    "panama": "Panama",
    "curasao": "Curasao",
    "iui3": "IUI3-RedSea",
    "japanese-gradens": "JapaneseGradens-RedSea",
}


@contextmanager
def _open_seathrunerf_zip():
    # Download the official seathrunerf dataset and cache it locally
    os.makedirs(os.path.join(NB_PREFIX, "cache"), exist_ok=True)
    download_path = os.path.join(NB_PREFIX, "cache", f"dataset-{DATASET_NAME}-{SEATHRUNERF_GDRIVE_ID}.zip")
    if not os.path.exists(download_path):
        logging.info(f"Downloading the official SeaThru-NeRF dataset to {download_path}.")
        try:
            os.remove(download_path + ".tmp")
        except OSError:
            pass
        try:
            import gdown
        except ImportError:
            logging.fatal("Please install gdown: pip install gdown")
            sys.exit(2)
        try:
            os.remove(download_path)
        except OSError:
            pass
        gdown.download(id=SEATHRUNERF_GDRIVE_ID, output=str(download_path) + ".tmp")
        if not os.path.exists(download_path):
            os.rename(str(download_path) + ".tmp", download_path)
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        yield zip_ref


def _select_indices_llff(image_names, llffhold=8):
    inds = np.argsort(image_names)
    all_indices = np.arange(len(image_names))
    indices_train = inds[all_indices % llffhold != 0]
    indices_test = inds[all_indices % llffhold == 0]
    return indices_train, indices_test


def load_seathru_nerf_dataset(path: str, split: Optional[str], **kwargs):
    import numpy as np
    images_path = "images_wb"
    dataset = load_colmap_dataset(path, split=None, images_path=images_path, **kwargs) 

    # Load bounds
    poses_bounds = np.load(os.path.join(path, "poses_bounds.npy"))
    nears_fars = poses_bounds[:, -2:]
    assert len(nears_fars) == len(dataset["cameras"]), f"Expected {len(dataset['cameras'])} near-far pairs, got {len(nears_fars)}."
    dataset["cameras"] = dataset["cameras"].replace(nears_fars=nears_fars)

    # Set dataset metadata
    dataset["metadata"]["type"] = "forward-facing"

    # Apply train/test split
    train_indices, test_indices = _select_indices_llff(dataset["image_paths"], llffhold=8)
    if split is not None:
        assert split in {"train", "test"}
        indices = train_indices if split == "train" else test_indices
        dataset = dataset_index_select(dataset, indices)

    return dataset


def download_seathru_nerf_dataset(path: str, output: str):
    output = str(output)
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError(f"Dataset path must be equal to '{DATASET_NAME}' or must start with '{DATASET_NAME}/'.")
    if path == DATASET_NAME:
        for x in SCENES:
            download_seathru_nerf_dataset(f"{DATASET_NAME}/{x}", os.path.join(output, x))
        return
    capture_name = path[len(f"{DATASET_NAME}/") :]
    if capture_name not in SCENES:
        raise DatasetNotFoundError(f"Scene '{capture_name}' not in the list of valid scenes: {','.join(SCENES.keys())}.")
    with _open_seathrunerf_zip() as zip_ref:
        zip_prefix = f"SeathruNeRF_dataset/{SCENES[capture_name]}"
        for zip_info in zip_ref.infolist():
            if zip_info.filename.startswith(zip_prefix + "/"):
                if zip_info.is_dir():
                    continue
                zip_info.filename = zip_info.filename[len(zip_prefix) + 1 :]
                if zip_info.filename.startswith("Images_wb/"):
                    zip_info.filename = "images_wb/" + zip_info.filename[len("Images_wb/") :]
                zip_ref.extract(zip_info, output)
    with open(os.path.join(str(output), "nb-info.json"), "w", encoding="utf8") as f2:
        json.dump({
            "loader": load_seathru_nerf_dataset.__module__ + ":" + load_seathru_nerf_dataset.__name__,
            "loader_kwargs": {},
            "id": DATASET_NAME,
            "scene": capture_name,
            "evaluation_protocol": "default",
            "type": "forward-facing",
        }, f2)
    logging.info(f"Extracted {path} to {output}")


__all__ = ["load_seathru_nerf_dataset", "download_seathru_nerf_dataset"]
