import hashlib
import shutil
import os
import tempfile
import zipfile
import logging
import sys
from typing import Union
from pathlib import Path
import numpy as np
from ..types import camera_model_to_int, new_cameras
from ._common import DatasetNotFoundError, new_dataset


gdrive_id = "16VnMcF1KJYxN9QId6TClMsZRahHNMW5g"
LLFF_SCENES = "fern flower fortress horns leaves orchids room trex".split()


def load_llff_dataset(path: Union[Path, str], split: str, downscale_factor: int = 4, **_):
    assert isinstance(path, (Path, str)), "path must be a pathlib.Path or str"
    path = Path(path)

    hold_every: int = 8
    scene = path.name
    if scene not in LLFF_SCENES:
        raise DatasetNotFoundError(f"Scene {scene} not found in LLFF dataset. Supported scenes: {LLFF_SCENES}.")
    for file in ("poses_bounds.npy", "simplices.npy", "sparse", "trimesh.png", "database.db", "images", "images_4", "images_8"):
        if not (path / file).exists():
            raise DatasetNotFoundError(f"Path {path} does not contain a LLFF dataset. Missing file: {path / file}")
    assert split in ["train", "test"], "split must be one of 'train', 'test'"

    poses_bounds = np.load(path / "poses_bounds.npy")
    image_paths = sorted(path.glob(f"images_{downscale_factor}/*"))
    assert len(image_paths) > 0, f"No images found in {path / f'images_{downscale_factor}'}"
    assert len(poses_bounds) == len(image_paths), "Mismatch between number of images and number of poses! Please rerun COLMAP!"

    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    near_fars = poses_bounds[:, -2:]  # (N_images, 2)

    # Step 1: rescale focal length according to training resolution
    H, W, focal = np.moveaxis(poses[:, :, -1], -1, 0)  # original intrinsics
    img_wh = np.array([W / downscale_factor, H / downscale_factor]).astype(np.int32)
    focal = [focal * img_wh[0] / W, focal * img_wh[1] / H]

    # Step 2: correct poses
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
    # (N_images, 3, 4) exclude H, W, focal

    intrinsics = np.stack([focal[0], focal[1], 0.5 * img_wh[0], 0.5 * img_wh[1]], -1)
    assert intrinsics.shape[0] == len(image_paths), "Intrinsics shape does not match number of images"

    i_test = np.arange(0, poses.shape[0], hold_every)
    indices = i_test if split != "train" else np.array(list(set(np.arange(len(poses))) - set(i_test)))

    c2w = poses[indices, :3, :4]

    # Convert from OpenGL to OpenCV coordinate system
    c2w = c2w.copy()
    c2w[..., 0:3, 1:3] *= -1

    return new_dataset(
        cameras=new_cameras(
            poses=c2w,
            intrinsics=(intrinsics)[indices],
            camera_types=np.full(len(indices), camera_model_to_int("pinhole"), dtype=np.int32),
            distortion_parameters=np.zeros((len(indices), 0), dtype=np.float32),
            image_sizes=img_wh.T[indices],
            nears_fars=near_fars[indices],
        ),
        image_paths_root=str(path),
        image_paths=[str(x) for i, x in enumerate(image_paths) if i in indices],
        sampling_mask_paths=None,
        sampling_mask_paths_root=None,
        metadata={
            "name": "llff",
            "scene": scene,
            "downscale_factor": downscale_factor,
            "color_space": "srgb",
            "evaluation_protocol": "nerf",
            "expected_scene_scale": 0.5,
            "type": "forward-facing",
        },
    )


def download_llff_dataset(path: str, output: Union[Path, str]):
    output = Path(output)
    if path == "llff":
        extract_prefix = "nerf_llff_data/"
    elif path.startswith("llff/") and len(path) > len("llff/"):
        scene_name = path[len("llff/") :]
        if scene_name not in LLFF_SCENES:
            raise DatasetNotFoundError(f"Scene {scene_name} not found in LLFF dataset. Supported scenes: {LLFF_SCENES}.")
        extract_prefix = f"nerf_llff_data/{scene_name}/"
    else:
        raise DatasetNotFoundError(f"Dataset path must be equal to 'llff' or must start with 'llff/'. It was {path}")

    file_sha256 = "5794b432feaf4f25bcd603addc6ad0270cec588fed6a364b7952001f07466635"
    try:
        import gdown
    except ImportError:
        logging.fatal("Please install gdown: pip install gdown")
        sys.exit(2)

    url = f"https://drive.google.com/uc?id={gdrive_id}"
    output_tmp = str(output) + ".tmp"
    if os.path.exists(output_tmp):
        shutil.rmtree(output_tmp)
    os.makedirs(output_tmp)
    has_member = False
    with tempfile.TemporaryDirectory() as tmpdir:
        gdown.download(url, output=tmpdir + "/llff_data.zip")

        # Verify hash
        with open(tmpdir + "/llff_data.zip", "rb") as f:
            hasher = hashlib.sha256()
            for blk in iter(lambda: f.read(4096), b""):
                hasher.update(blk)
            if hasher.hexdigest() != file_sha256:
                raise RuntimeError(f"Hash of {tmpdir + '/llff_data.zip'} ({hasher.hexdigest()}) does not match {file_sha256}")

        # Extract files
        logging.info("LLFF dataset downloaded and verified")
        logging.info(f"Extracting LLFF dataset: {tmpdir + '/llff_data.zip'}")
        with zipfile.ZipFile(tmpdir + "/llff_data.zip", "r") as zip_ref:
            for member in zip_ref.infolist():
                if member.filename.startswith(extract_prefix) and len(member.filename) > len(extract_prefix):
                    member.filename = member.filename[len(extract_prefix) :]
                    zip_ref.extract(member, output_tmp)
                    has_member = True
    if not has_member:
        raise RuntimeError(f"Path {path} not found in LLFF dataset.")
    if os.path.exists(str(output)):
        shutil.rmtree(str(output))
    os.rename(str(output) + ".tmp", str(output))
    logging.info(f"Downloaded {path} to {output}")
