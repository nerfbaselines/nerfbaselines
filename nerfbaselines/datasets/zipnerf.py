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
from nerfbaselines.datasets import _colmap_utils as colmap_utils
from ._common import dataset_index_select, atomic_output
from .colmap import load_colmap_dataset


DATASET_NAME = "zipnerf"
_scenes_links = {
    "alameda": "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/alameda.zip",
    "berlin": "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/berlin.zip",
    "london": "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/london.zip",
    "nyc": "https://storage.googleapis.com/gresearch/refraw360/zipnerf-undistorted/nyc.zip",
}
SCENES = set(_scenes_links.keys())
VERSION = "1"


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


def _downscale_cameras_v1(cameras_path, output_cameras_path, downscale_factor: int):
    cameras = colmap_utils.read_cameras_binary(cameras_path)
    new_cameras = {}
    for k, v in cameras.items():
        assert v.model == "PINHOLE", f"Expected PINHOLE camera model, got {v.model}."
        params = v.params
        oldw, oldh = v.width, v.height
        w = v.width // downscale_factor
        h = v.height // downscale_factor
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
        w = v.width // downscale_factor
        h = v.height // downscale_factor
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
            with atomic_output(output) as output_tmp:
                output_tmp.mkdir(exist_ok=True, parents=True)
                for info in z.infolist():
                    if not info.filename.startswith(scene_name + "/"):
                        continue
                    if (not info.filename.startswith(scene_name + "/sparse") and
                        not info.filename.startswith(scene_name + f"/images_2")):
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

                with open(os.path.join(str(output_tmp), "nb-info.json"), "w", encoding="utf8") as f:
                    json.dump({
                        "loader": "colmap",
                        "loader_kwargs": {
                            "images_path": "images_2",
                            "colmap_path": "sparse_2/0",
                        },
                        "id": DATASET_NAME,
                        "scene": scene_name,
                        "downscale_factor": 2,
                        "evaluation_protocol": "nerf",
                        "version": VERSION,
                    }, f)

                # Generate downscaled COLMAP model
                shutil.move(
                    os.path.join(str(output_tmp), "sparse"),
                    os.path.join(str(output_tmp), "sparse_2")
                )
                _downscale_cameras_v1(
                    os.path.join(str(output_tmp), "sparse_2", "0", "cameras.bin"),
                    os.path.join(str(output_tmp), "sparse_2", "0", "cameras.bin"),
                    2
                )

                # Generate split files
                _save_colmap_splits(output_tmp)
            logging.info(f"Downloaded {DATASET_NAME}/{scene_name} to {output}")


download_zipnerf_dataset.version = VERSION  # type: ignore
__all__ = ["download_zipnerf_dataset"]
