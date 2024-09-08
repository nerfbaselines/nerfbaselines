import json
import os
import warnings
from typing import List, Tuple, Union
import logging
from itertools import groupby
import shutil
import requests
from pathlib import Path
import numpy as np
import zipfile
from tqdm import tqdm
import tempfile
from nerfbaselines import DatasetNotFoundError
from ._common import single, dataset_index_select
from .colmap import load_colmap_dataset


DATASET_NAME = "mipnerf360"
_scenes360_res = {
    "bicycle": 4,
    "flowers": 4,
    "garden": 4,
    "stump": 4,
    "treehill": 4,
    "bonsai": 2,
    "counter": 2,
    "kitchen": 2,
    "room": 2,
}
SCENES = set(_scenes360_res.keys())


def load_mipnerf360_dataset(path: Union[Path, str], split: str, resize_full_image: bool = False, downscale_factor=None, **kwargs):
    path = Path(path)
    if split:
        assert split in {"train", "test"}
    if "360" not in str(path) or not any(s in str(path) for s in _scenes360_res):
        raise DatasetNotFoundError(f"360 and {set(_scenes360_res.keys())} is missing from the dataset path: {path}")

    # Load MipNerf360 dataset
    scene = single(res for res in _scenes360_res if str(res) in path.name)
    res = _scenes360_res[scene]

    if downscale_factor is not None and downscale_factor != res:
        warnings.warn(f"downscale_factor {downscale_factor} was specified, overriding the default downscale_factor for the scene {res}.")
        res = downscale_factor

    if resize_full_image or res == 1:
        images_path = f"images"
    else:
        images_path = f"images_{res}"

    # Use split=None to load all images
    # We then select the same images as in the LLFF multinerf dataset loader
    dataset = load_colmap_dataset(path, images_path=images_path, split=None, **kwargs)
    dataset["metadata"]["id"] = DATASET_NAME
    dataset["metadata"]["scene"] = scene
    dataset["metadata"]["downscale_factor"] = res
    if resize_full_image:
        dataset["metadata"]["downscale_loaded_factor"] = res
    dataset["metadata"]["color_space"] = "srgb"

    image_names = dataset["image_paths"]
    inds = np.argsort(image_names)

    all_indices = np.arange(len(dataset["image_paths"]))
    llffhold = 8
    if split == "train":
        indices = all_indices % llffhold != 0
    else:
        indices = all_indices % llffhold == 0
    indices = inds[indices]
    return dataset_index_select(dataset, indices)


def download_mipnerf360_dataset(path: str, output: Union[Path, str]):
    url_extra = "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
    url_base = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    output = Path(output)
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError(f"Dataset path must be equal to '{DATASET_NAME}' or must start with '{DATASET_NAME}/'.")

    captures: List[Tuple[str, Path]] = []
    if path == DATASET_NAME:
        # We will download all faster here
        for x in _scenes360_res:
            captures.append((x, output / x))
    else:
        captures = [(path[len(f"{DATASET_NAME}/") :], output)]
    captures_to_download: List[Tuple[str, str, Path]] = []
    for capture_name, output in captures:
        if capture_name not in _scenes360_res:
            raise DatasetNotFoundError(f"Capture '{capture_name}' not a valid {DATASET_NAME} scene.")
        url = url_extra if capture_name in {"flowers", "treehill"} else url_base
        captures_to_download.append((url, capture_name, output))
    captures_to_download.sort(key=lambda x: x[0])
    for url, _captures in groupby(captures_to_download, key=lambda x: x[0]):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Downloading {url.split('/')[-1]}", dynamic_ncols=True)
        with tempfile.TemporaryFile("rb+") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
            file.flush()
            file.seek(0)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                logging.error(f"Failed to download dataset. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes.")

            has_any = False
            with zipfile.ZipFile(file) as z:
                for _, capture_name, output in _captures:
                    output_tmp = output.with_suffix(".tmp")
                    output_tmp.mkdir(exist_ok=True, parents=True)
                    for info in z.infolist():
                        if not info.filename.startswith(capture_name + "/"):
                            continue
                        # z.extract(name, output_tmp)
                        has_any = True
                        relname = info.filename[len(capture_name) + 1 :]
                        target = output_tmp / relname
                        target.parent.mkdir(exist_ok=True, parents=True)
                        if info.is_dir():
                            target.mkdir(exist_ok=True, parents=True)
                        else:
                            with z.open(info) as source, open(target, "wb") as target:
                                shutil.copyfileobj(source, target)
                    if not has_any:
                        raise RuntimeError(f"Capture '{capture_name}' not found in {url}.")
                    with open(os.path.join(str(output_tmp), "nb-info.json"), "w", encoding="utf8") as f:
                        json.dump({
                            "loader": load_mipnerf360_dataset.__module__ + ":" + load_mipnerf360_dataset.__name__,
                            "loader_kwargs": {},
                            "id": DATASET_NAME,
                            "scene": capture_name,
                            "evaluation_protocol": "nerf",
                            "type": "object-centric",
                        }, f)
                    shutil.rmtree(output, ignore_errors=True)
                    shutil.move(str(output_tmp), str(output))
                    logging.info(f"Downloaded {DATASET_NAME}/{capture_name} to {output}")


__all__ = ["load_mipnerf360_dataset", "download_mipnerf360_dataset"]
