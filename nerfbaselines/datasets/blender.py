import hashlib
import shutil
import os
import tempfile
import zipfile
import logging
import sys
import json
from pathlib import Path
from typing import Union
import numpy as np
from ..types import camera_model_to_int, new_cameras
from ._common import DatasetNotFoundError, get_default_viewer_transform, new_dataset


BLENDER_SCENES = {"lego", "ship", "drums", "hotdog", "materials", "mic", "chair", "ficus"}
BLENDER_SPLITS = {"train", "test"}


def load_blender_dataset(path: Union[Path, str], split: str, **kwargs):
    assert isinstance(path, (Path, str)), "path must be a pathlib.Path or str"
    path = Path(path)

    scene = path.name
    if scene not in BLENDER_SCENES:
        raise DatasetNotFoundError(f"Scene {scene} not found in nerf_synthetic dataset. Supported scenes: {BLENDER_SCENES}.")
    for dsplit in BLENDER_SPLITS:
        if not (path / f"transforms_{dsplit}.json").exists():
            raise DatasetNotFoundError(f"Path {path} does not contain a blender dataset. Missing file: {path / f'transforms_{dsplit}.json'}")

    assert split in BLENDER_SPLITS, "split must be one of 'train' or 'test'"

    with (path / f"transforms_{split}.json").open("r", encoding="utf8") as fp:
        meta = json.load(fp)

    cams = []
    image_paths = []
    for _, frame in enumerate(meta["frames"]):
        fprefix = path / frame["file_path"]
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

    viewer_transform, viewer_pose = get_default_viewer_transform(c2w, "object-centric")

    return new_dataset(
        cameras=new_cameras(
            poses=c2w,
            intrinsics=intrinsics,
            camera_types=np.full(len(cams), camera_model_to_int("pinhole"), dtype=np.int32),
            distortion_parameters=np.zeros((len(cams), 0), dtype=np.float32),
            image_sizes=image_sizes,
            nears_fars=nears_fars,
        ),
        image_paths_root=str(path),
        image_paths=image_paths,
        sampling_mask_paths=None,
        metadata={
            "name": "blender",
            "scene": scene,
            "color_space": "srgb",
            "type": "object-centric",
            "evaluation_protocol": "nerf",
            "expected_scene_scale": 4,
            "viewer_transform": viewer_transform,
            "viewer_initial_pose": viewer_pose,
            "background_color": np.array([255, 255, 255], dtype=np.uint8),
        },
    )


def download_blender_dataset(path: str, output: Path):
    if path == "blender":
        extract_prefix = "nerf_synthetic/"
    elif path.startswith("blender/") and len(path) > len("blender/"):
        scene_name = path[len("blender/") :]
        if scene_name not in BLENDER_SCENES:
            raise DatasetNotFoundError(f"Scene {scene_name} not found in nerf_synthetic dataset. Supported scenes: {BLENDER_SCENES}.")
        extract_prefix = f"nerf_synthetic/{scene_name}/"
    else:
        raise DatasetNotFoundError(f"Dataset path must be equal to 'blender' or must start with 'blender/'. It was {path}")

    # https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
    blender_file_id = "18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"
    file_sha256 = "f01fd1b4ab045b0d453917346f26f898657bb5bec4834b95fdad1f361826e45e"
    try:
        import gdown
    except ImportError:
        logging.fatal("Please install gdown: pip install gdown")
        sys.exit(2)

    url = f"https://drive.google.com/uc?id={blender_file_id}"
    output_tmp = str(output) + ".tmp"
    if os.path.exists(output_tmp):
        shutil.rmtree(output_tmp)
    os.makedirs(output_tmp)
    has_member = False
    with tempfile.TemporaryDirectory() as tmpdir:
        gdown.download(url, output=tmpdir + "/blender_data.zip")

        # Verify hash
        with open(tmpdir + "/blender_data.zip", "rb") as f:
            hasher = hashlib.sha256()
            for blk in iter(lambda: f.read(4096), b""):
                hasher.update(blk)
            if hasher.hexdigest() != file_sha256:
                raise RuntimeError(f"Hash of {tmpdir + '/blender_data.zip'} does not match {file_sha256}")

        # Extract files
        logging.info("Blender dataset downloaded and verified")
        logging.info(f"Extracting blender dataset: {tmpdir + '/blender_data.zip'}")
        with zipfile.ZipFile(tmpdir + "/blender_data.zip", "r") as zip_ref:
            for member in zip_ref.infolist():
                if member.filename.startswith(extract_prefix) and len(member.filename) > len(extract_prefix):
                    member.filename = member.filename[len(extract_prefix) :]
                    zip_ref.extract(member, output_tmp)
                    has_member = True
    if not has_member:
        raise RuntimeError(f"Path {path} not found in nerf_synthetic dataset.")
    if os.path.exists(str(output)):
        shutil.rmtree(str(output))
    os.rename(str(output) + ".tmp", str(output))
    logging.info(f"Downloaded {path} to {output}")
