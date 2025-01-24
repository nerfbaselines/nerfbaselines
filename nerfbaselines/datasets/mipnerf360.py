import json
import os
from typing import List, Tuple, Union
import logging
from itertools import groupby
import shutil
from pathlib import Path
import numpy as np
import zipfile
import tempfile
from nerfbaselines import DatasetNotFoundError
from nerfbaselines.io import wget
from ._common import dataset_index_select
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


def load_mipnerf360_dataset(path, *args, **kwargs):
    del args, kwargs
    raise RuntimeError(f"The dataset was likely downloaded with an older version of NerfBaselines. Please remove `{path}` and try again.")


def _save_colmap_splits(output):
    with open(os.path.join(output, "nb-info.json"), "r", encoding="utf8") as f:
        dataset_info = json.load(f)

    dataset = load_colmap_dataset(output, **dataset_info["loader_kwargs"])
    image_names = dataset["image_paths"]
    inds = np.argsort(image_names)

    all_indices = np.arange(len(dataset["image_paths"]))
    llffhold = 8
    indices = {}
    indices["train"] = inds[all_indices % llffhold != 0]
    indices["test"] = inds[all_indices % llffhold == 0]
    for split in indices:
        with open(os.path.join(output, f"{split}_list.txt"), "w", encoding="utf8") as f:
            for img_name in dataset_index_select(dataset, indices[split])["image_paths"]:
                f.write(os.path.relpath(img_name, dataset["image_paths_root"]) + "\n")


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
    for scene, output in captures:
        if scene not in _scenes360_res:
            raise DatasetNotFoundError(f"Capture '{scene}' not a valid {DATASET_NAME} scene.")
        url = url_extra if scene in {"flowers", "treehill"} else url_base
        captures_to_download.append((url, scene, output))
    captures_to_download.sort(key=lambda x: x[0])
    for url, _captures in groupby(captures_to_download, key=lambda x: x[0]):
        with tempfile.TemporaryFile("rb+") as file:
            wget(url, file, desc=f"Downloading {url.split('/')[-1]}")
            file.seek(0)
            has_any = False
            with zipfile.ZipFile(file) as z:
                for _, scene, output in _captures:
                    output_tmp = output.with_suffix(".tmp")
                    output_tmp.mkdir(exist_ok=True, parents=True)
                    for info in z.infolist():
                        if not info.filename.startswith(scene + "/"):
                            continue
                        # z.extract(name, output_tmp)
                        has_any = True
                        relname = info.filename[len(scene) + 1 :]
                        target = output_tmp / relname
                        target.parent.mkdir(exist_ok=True, parents=True)
                        if info.is_dir():
                            target.mkdir(exist_ok=True, parents=True)
                        else:
                            with z.open(info) as source, open(target, "wb") as target:
                                shutil.copyfileobj(source, target)
                    if not has_any:
                        raise RuntimeError(f"Capture '{scene}' not found in {url}.")

                    res = _scenes360_res[scene]
                    images_path = "images" if res == 1 else f"images_{res}"

                    with open(os.path.join(str(output_tmp), "nb-info.json"), "w", encoding="utf8") as f:
                        json.dump({
                            "loader": "colmap",
                            "loader_kwargs": {
                                "images_path": images_path,
                            },
                            "id": DATASET_NAME,
                            "scene": scene,
                            "downscale_factor": res,
                            "evaluation_protocol": "nerf",
                            "type": "object-centric",
                        }, f)

                    # Generate split files
                    _save_colmap_splits(output_tmp)

                    shutil.rmtree(output, ignore_errors=True)
                    shutil.move(str(output_tmp), str(output))
                    logging.info(f"Downloaded {DATASET_NAME}/{scene} to {output}")


__all__ = ["download_mipnerf360_dataset"]
