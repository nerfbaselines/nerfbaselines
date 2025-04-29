import json
import os
from typing import Union
import logging
import shutil
from pathlib import Path
import numpy as np
import zipfile
import tempfile
from nerfbaselines import DatasetNotFoundError
from nerfbaselines.io import wget
from ._common import dataset_index_select
from .colmap import load_colmap_dataset


DATASET_NAME = "zipnerf"
_scenes_links = {
    "alameda": "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/alameda.zip",
    "berlin": "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/berlin.zip",
    "london": "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/london.zip",
    "nyc": "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/nyc.zip",
}
SCENES = set(_scenes_links.keys())


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


def download_zipnerf_dataset(path: str, output: Union[Path, str]):
    output = Path(output)
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError(f"Dataset path must be equal to '{DATASET_NAME}' or must start with '{DATASET_NAME}/'.")

    if path == DATASET_NAME:
        # We will download all faster here
        for x in _scenes_links:
            download_zipnerf_dataset(f"{DATASET_NAME}/{x}", output / x)

    scene_name = path.split("/")[-1]
    if scene_name not in SCENES:
        raise DatasetNotFoundError(f"Unknown scene '{scene_name}'. Available scenes: {', '.join(SCENES)}")

    with tempfile.TemporaryFile("rb+") as file:
        url = _scenes_links[path.split("/")[-1]]
        wget(url, file, desc=f"Downloading {url.split('/')[-1]}")
        file.seek(0)
        has_any = False
        with zipfile.ZipFile(file) as z:
            output_tmp = output.with_suffix(".tmp")
            output_tmp.mkdir(exist_ok=True, parents=True)
            for info in z.infolist():
                if not info.filename.startswith(scene_name + "/"):
                    continue
                # z.extract(name, output_tmp)
                has_any = True
                relname = info.filename[len(scene_name) + 1 :]
                target = output_tmp / relname
                target.parent.mkdir(exist_ok=True, parents=True)
                if info.is_dir():
                    target.mkdir(exist_ok=True, parents=True)
                else:
                    with z.open(info) as source, open(target, "wb") as target:
                        shutil.copyfileobj(source, target)
            if not has_any:
                raise RuntimeError(f"Capture '{scene_name}' not found in {url}.")

            res = 2
            images_path = "images" if res == 1 else f"images_{res}"

            with open(os.path.join(str(output_tmp), "nb-info.json"), "w", encoding="utf8") as f:
                json.dump({
                    "loader": "colmap",
                    "loader_kwargs": {
                        "images_path": images_path,
                    },
                    "id": DATASET_NAME,
                    "scene": scene_name,
                    "downscale_factor": res,
                    "evaluation_protocol": "nerf",
                }, f)

            # Generate split files
            _save_colmap_splits(output_tmp)

            shutil.rmtree(output, ignore_errors=True)
            shutil.move(str(output_tmp), str(output))
            logging.info(f"Downloaded {DATASET_NAME}/{scene_name} to {output}")


__all__ = ["download_zipnerf_dataset"]
