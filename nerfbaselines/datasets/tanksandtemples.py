import logging
import shutil
import requests
from pathlib import Path
from typing import Union
import tarfile
from tqdm import tqdm
import tempfile
import numpy as np
from ..utils import assert_not_none
from ..types import UnloadedDataset
from ._common import DatasetNotFoundError, get_scene_scale, get_default_viewer_transform, dataset_index_select
from .colmap import load_colmap_dataset


DATASET_NAME = "tanksandtemples"
BASE_URL = "https://huggingface.co/datasets/jkulhanek/nerfbaselines-data/resolve/main/tanksandtemples"
_URL = f"{BASE_URL}/{{scene}}.tar.gz"
_URL2DOWN = f"{BASE_URL}/{{scene}}_2down.tar.gz"
SCENES = {
    # advanced
    "auditorium": True,
    "ballroom": True,
    "courtroom": True,
    "museum": True,
    "palace": True,
    "temple": True,

    # intermediate
    "family": True,
    "francis": True,
    "horse": True,
    "lighthouse": True,
    "m60": True,
    "panther": True,
    "playground": True,
    "train": True,

    # training
    "barn": True,
    "caterpillar": True,
    "church": True,
    "courthouse": True,
    "ignatius": True,
    "meetingroom": True,
    "truck": True,
}


def _select_indices_llff(image_names, llffhold=8):
    inds = np.argsort(image_names)
    all_indices = np.arange(len(image_names))
    indices_train = inds[all_indices % llffhold != 0]
    indices_test = inds[all_indices % llffhold == 0]
    return indices_train, indices_test


def load_tanksandtemples_dataset(path: Union[Path, str], split: str, downscale_factor: int = 2, **kwargs) -> UnloadedDataset:
    path = Path(path)
    if split:
        assert split in {"train", "test"}
    if DATASET_NAME not in str(path) or not any(s in str(path).lower() for s in SCENES):
        raise DatasetNotFoundError(f"{DATASET_NAME} and {set(SCENES.keys())} is missing from the dataset path: {path}")

    # Load TT dataset
    images_path = "images" if downscale_factor == 1 else f"images_{downscale_factor}"
    scene = next((x for x in SCENES if x in str(path)), None)
    assert scene is not None, f"Scene not found in path {path}"

    dataset = load_colmap_dataset(path, images_path=images_path, split=None, **kwargs)
    dataset["metadata"]["name"] = DATASET_NAME
    dataset["metadata"]["scene"] = scene
    dataset["metadata"]["downscale_factor"] = downscale_factor
    dataset["metadata"]["expected_scene_scale"] = get_scene_scale(dataset["cameras"], None)
    dataset["metadata"]["type"] = None
    viewer_transform, viewer_pose = get_default_viewer_transform(dataset["cameras"].poses, None)
    dataset["metadata"]["viewer_transform"] = viewer_transform
    dataset["metadata"]["viewer_initial_pose"] = viewer_pose
    dataset["metadata"]["evaluation_protocol"] = "default"
    indices_train, indices_test = _select_indices_llff(dataset["image_paths"])
    indices = indices_train if split == "train" else indices_test
    return dataset_index_select(dataset, indices)


def download_tanksandtemples_dataset(path: str, output: Union[Path, str]) -> None:
    output = Path(output)
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError("Dataset path must be equal to 'tanksandtemples' or must start with 'tanksandtemples/'.")

    if path == DATASET_NAME:
        for scene in SCENES:
            download_tanksandtemples_dataset(f"{DATASET_NAME}/{scene}", output/scene)
        return

    scene = path.split("/")[-1]
    if SCENES.get(scene) is None:
        raise RuntimeError(f"Unknown scene {scene}")
    if SCENES[scene] is False:
        raise DatasetNotFoundError(f"Scene {scene} is not available in current release of the tanksandtemples dataset.")
    url = _URL2DOWN.format(scene=scene)
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

        with tarfile.open(fileobj=file, mode="r:gz") as z:
            output_tmp = output.with_suffix(".tmp")
            output_tmp.mkdir(exist_ok=True, parents=True)
            for info in z.getmembers():
                if not info.name.startswith(scene + "/"):
                    continue
                relname = info.name[len(scene) + 1 :]
                target = output_tmp / relname
                target.parent.mkdir(exist_ok=True, parents=True)
                if info.isdir():
                    target.mkdir(exist_ok=True, parents=True)
                else:
                    with assert_not_none(z.extractfile(info)) as source, open(target, "wb") as target:
                        shutil.copyfileobj(source, target)

            shutil.rmtree(output, ignore_errors=True)
            shutil.move(str(output_tmp), str(output))
            logging.info(f"Downloaded {DATASET_NAME}/{scene} to {output}")
