import os
import time
from datetime import datetime
import zipfile
import requests
import shutil
import tempfile
import logging
import json
from tqdm import tqdm
from pathlib import Path
from typing import Union
import numpy as np
from nerfbaselines import camera_model_to_int, new_cameras, new_dataset, DatasetNotFoundError


DATASET_NAME = "blender"
SCENES = {"lego", "ship", "drums", "hotdog", "materials", "mic", "chair", "ficus"}
SPLITS = {"train", "test"}
_URL = "https://huggingface.co/datasets/jkulhanek/nerfbaselines-data/resolve/main/blender/{scene}.zip"


def load_blender_dataset(path: str, split: str, **kwargs):
    """
    Load a Blender dataset (scenes: lego, ship, drums, hotdog, materials, mic, chair, ficus).

    Args:
        path: Path to the dataset directory.
        split: The split to load, either 'train' or 'test'.

    Returns:
        Unloaded dataset dictionary.
    """
    del kwargs
    _path = Path(path)

    assert split in SPLITS, "split must be one of 'train' or 'test'"

    with (_path / f"transforms_{split}.json").open("r", encoding="utf8") as fp:
        meta = json.load(fp)

    cams = []
    image_paths = []
    for _, frame in enumerate(meta["frames"]):
        fprefix = _path / frame["file_path"]
        image_paths.append(str(fprefix) + ".png")
        cams.append(np.array(frame["transform_matrix"], dtype=np.float32))

    w = h = 800
    image_sizes = np.array([w, h], dtype=np.int32)[None].repeat(len(cams), axis=0)
    nears_fars = np.array([2, 6], dtype=np.float32)[None].repeat(len(cams), axis=0)
    fx = fy = 0.5 * w / np.tan(0.5 * float(meta["camera_angle_x"]))
    cx = cy = 0.5 * w
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)[None].repeat(len(cams), axis=0)
    c2w = np.stack(cams)[:, :3, :4]

    # Convert from OpenGL to OpenCV coordinate system
    c2w[..., 0:3, 1:3] *= -1

    return new_dataset(
        cameras=new_cameras(
            poses=c2w,
            intrinsics=intrinsics,
            camera_models=np.full(len(cams), camera_model_to_int("pinhole"), dtype=np.int32),
            distortion_parameters=np.zeros((len(cams), 0), dtype=np.float32),
            image_sizes=image_sizes,
            nears_fars=nears_fars,
        ),
        image_paths_root=path,
        image_paths=image_paths,
        sampling_mask_paths=None,
        metadata={
            "color_space": "srgb",
            "expected_scene_scale": 4,
            "background_color": np.array([255, 255, 255], dtype=np.uint8),
        },
    )


def download_blender_dataset(path: str, output: Union[Path, str]) -> None:
    output = Path(output)
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError("Dataset path must be equal to 'blender' or must start with 'blender/'.")

    if path == DATASET_NAME:
        for scene in SCENES:
            download_blender_dataset(f"{DATASET_NAME}/{scene}", output/scene)
        return

    scene = path.split("/")[-1]
    if scene not in SCENES:
        raise RuntimeError(f"Unknown scene {scene}, supported scenes: {SCENES}")
    url = _URL.format(scene=scene)
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

        with zipfile.ZipFile(file, mode="r") as z:
            output_tmp = output.with_suffix(".tmp")
            output_tmp.mkdir(exist_ok=True, parents=True)

            for info in z.infolist():
                if not info.filename.startswith(scene + "/"):
                    continue
                relname = info.filename[len(scene) + 1 :]
                target = output_tmp / relname
                target.parent.mkdir(exist_ok=True, parents=True)
                if info.is_dir():
                    target.mkdir(exist_ok=True, parents=True)
                else:
                    info.filename = relname
                    z.extract(info, output_tmp)

                    # Fix mtime
                    date_time = datetime(*info.date_time)
                    mtime = time.mktime(date_time.timetuple())
                    os.utime(target, (mtime, mtime))

            with open(os.path.join(str(output_tmp), "nb-info.json"), "w", encoding="utf8") as f2:
                json.dump({
                    "loader": load_blender_dataset.__module__ + ":" + load_blender_dataset.__name__,
                    "id": DATASET_NAME,
                    "scene": scene,
                    "evaluation_protocol": "nerf",
                    "type": "object-centric",
                }, f2)
            shutil.rmtree(output, ignore_errors=True)
            shutil.move(str(output_tmp), str(output))
            logging.info(f"Downloaded {DATASET_NAME}/{scene} to {output}")


__all__ = ["load_blender_dataset", "download_blender_dataset"]
