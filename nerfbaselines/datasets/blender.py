import logging
import json
from pathlib import Path
import numpy as np
from nerfbaselines import camera_model_to_int, new_cameras, new_dataset
from nerfbaselines._constants import DATASETS_REPOSITORY
from ._common import download_dataset_wrapper, download_archive_dataset


DATASET_NAME = "blender"
SCENES = {"lego", "ship", "drums", "hotdog", "materials", "mic", "chair", "ficus"}
SPLITS = {"train", "test"}
_URL = f"https://{DATASETS_REPOSITORY}/resolve/main/blender/{{scene}}.zip"


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

    w = meta.get("w", 800)
    h = meta.get("h", 800)
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
        mask_paths=None,
        metadata={
            "color_space": "srgb",
            "expected_scene_scale": 4,
            "depth_range": [2, 6],
            "background_color": np.array([255, 255, 255], dtype=np.uint8),
        },
    )


@download_dataset_wrapper(SCENES, DATASET_NAME)
def download_blender_dataset(path: str, output: str) -> None:
    dataset_name, scene = path.split("/", 1)
    if scene not in SCENES:
        raise RuntimeError(f"Unknown scene {scene}, supported scenes: {SCENES}")

    url = _URL.format(scene=scene)
    prefix = f"{scene}/"
    nb_info = {
        "loader": load_blender_dataset.__module__ + ":" + load_blender_dataset.__name__,
        "id": dataset_name,
        "scene": scene,
        "evaluation_protocol": "nerf",
        "type": "object-centric",
    }
    download_archive_dataset(url, output, 
                             archive_prefix=prefix, 
                             nb_info=nb_info, 
                             file_type="zip")
    logging.info(f"Downloaded {DATASET_NAME}/{scene} to {output}")


__all__ = ["load_blender_dataset", "download_blender_dataset"]
