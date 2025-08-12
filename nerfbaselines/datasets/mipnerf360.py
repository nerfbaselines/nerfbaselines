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
from ._common import dataset_index_select, atomic_output
from .colmap import load_colmap_dataset
from . import _colmap_utils as colmap_utils


VERSION = "1"
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
                    with atomic_output(output) as output_tmp:
                        output_tmp = Path(output_tmp)
                        res = _scenes360_res[scene]
                        images_path = "images" if res == 1 else f"images_{res}"
                        for info in z.infolist():
                            if not info.filename.startswith(scene + "/"):
                                continue
                            if (not info.filename.startswith(scene + "/sparse") and
                                not info.filename.startswith(scene + f"/{images_path}")):
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

                        if res != 1:
                            shutil.move(
                                os.path.join(str(output_tmp), "sparse"),
                                os.path.join(str(output_tmp), f"sparse_{res}"))
                            # Downscale cameras
                            # NOTE: Switching to v2 version is a breaking change
                            _downscale_cameras_v1(
                                os.path.join(str(output_tmp), f"sparse_{res}", "0", "cameras.bin"),
                                os.path.join(str(output_tmp), f"sparse_{res}", "0", "cameras.bin"),
                                res
                            )

                        with open(os.path.join(str(output_tmp), "nb-info.json"), "w", encoding="utf8") as f:
                            json.dump({
                                "loader": "colmap",
                                "loader_kwargs": {
                                    "images_path": images_path,
                                    "colmap_path": f"sparse_{res}/0" if res != 1 else "sparse/0",
                                },
                                "id": DATASET_NAME,
                                "scene": scene,
                                "downscale_factor": res,
                                "evaluation_protocol": "nerf",
                                "type": "object-centric",
                                "version": VERSION,
                            }, f)

                        # Generate split files
                        _save_colmap_splits(output_tmp)
                    logging.info(f"Downloaded {DATASET_NAME}/{scene} to {output}")


download_mipnerf360_dataset.version = VERSION  # type: ignore
__all__ = ["download_mipnerf360_dataset"]
