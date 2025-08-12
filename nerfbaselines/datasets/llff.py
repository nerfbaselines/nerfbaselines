import logging
from typing import Union
from pathlib import Path
import numpy as np
from nerfbaselines import camera_model_to_int, new_cameras, DatasetNotFoundError, new_dataset
from ._common import download_dataset_wrapper, download_archive_dataset
from nerfbaselines._constants import DATASETS_REPOSITORY


SCENES = "fern flower fortress horns leaves orchids room trex".split()
DATASET_NAME = "llff"


def load_llff_dataset(path: Union[Path, str], split: str, features):
    del features
    assert isinstance(path, (Path, str)), "path must be a pathlib.Path or str"
    downscale_factor = 4
    path = Path(path)

    hold_every: int = 8
    for file in ("poses_bounds.npy", "sparse", f"images_{downscale_factor}"):
        if not (path / file).exists():
            raise DatasetNotFoundError(f"Path {path} does not contain a LLFF dataset. Missing file: {path / file}")
    assert split in ["train", "test"], "split must be one of 'train', 'test'"

    poses_bounds = np.load(path / "poses_bounds.npy")
    image_paths = sorted(path.glob(f"images_{downscale_factor}/*.png"))
    assert len(image_paths) > 0, f"No images found in {path / f'images_{downscale_factor}'}"
    assert len(poses_bounds) == len(image_paths), f"Mismatch between number of images ({len(image_paths)}) and number of poses ({len(poses_bounds)})!"

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
            camera_models=np.full(len(indices), camera_model_to_int("pinhole"), dtype=np.int32),
            distortion_parameters=np.zeros((len(indices), 0), dtype=np.float32),
            image_sizes=img_wh.T[indices],
            nears_fars=near_fars[indices],
        ),
        image_paths_root=str(path),
        image_paths=[str(x) for i, x in enumerate(image_paths) if i in indices],
        mask_paths=None,
        mask_paths_root=None,
        metadata={
            "downscale_factor": downscale_factor,
            "color_space": "srgb",
            "type": "forward-facing",
            "evaluation_protocol": "nerf",
            "expected_scene_scale": 0.5,
        },
    )


@download_dataset_wrapper(SCENES, DATASET_NAME)
def download_llff_dataset(path: str, output: str) -> None:
    dataset_name, scene = path.split("/", 1)
    if scene not in SCENES:
        raise RuntimeError(f"Unknown scene {scene}, supported scenes: {SCENES}")
    url = f"https://{DATASETS_REPOSITORY}/resolve/main/llff/{{scene}}.zip".format(scene=scene)
    extract_prefix = f"{scene}/"
    nb_info = {
        "id": dataset_name,
        "scene": scene,
        "loader": load_llff_dataset.__module__ + ":" + load_llff_dataset.__name__,
        "downscale_factor": 4,
        "evaluation_protocol": "nerf",
        "type": "forward-facing",
    }
    def filter(path):
        return path.startswith("images_4/") or path == "poses_bounds.npy" or path.startswith("sparse/")

    download_archive_dataset(url, str(output), 
                             archive_prefix=extract_prefix, 
                             nb_info=nb_info,
                             filter=filter,
                             file_type="zip")
    logging.info(f"Downloaded {path} to {output}")


__all__ = ["load_llff_dataset", "download_llff_dataset"]
