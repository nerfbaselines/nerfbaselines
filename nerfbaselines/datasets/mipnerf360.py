import warnings
from typing import List, Tuple, Union
import logging
from itertools import groupby
import shutil
import requests
from pathlib import Path
import numpy as np
import zipfile
from tqdm import tqdm
import tempfile
from ._common import DatasetNotFoundError, single, get_scene_scale, get_default_viewer_transform, dataset_index_select
from .colmap import load_colmap_dataset


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


def load_mipnerf360_dataset(path: Union[Path, str], split: str, resize_full_image: bool = False, downscale_factor=None, **kwargs):
    path = Path(path)
    if split:
        assert split in {"train", "test"}
    if "360" not in str(path) or not any(s in str(path) for s in _scenes360_res):
        raise DatasetNotFoundError(f"360 and {set(_scenes360_res.keys())} is missing from the dataset path: {path}")

    # Load MipNerf360 dataset
    scene = single(res for res in _scenes360_res if str(res) in path.name)
    res = _scenes360_res[scene]

    if downscale_factor is not None and downscale_factor != res:
        warnings.warn(f"downscale_factor {downscale_factor} was specified, overriding the default downscale_factor for the scene {res}.")
        res = downscale_factor

    if resize_full_image or res == 1:
        images_path = f"images"
    else:
        images_path = f"images_{res}"

    # Use split=None to load all images
    # We then select the same images as in the LLFF multinerf dataset loader
    dataset = load_colmap_dataset(path, images_path=images_path, split=None, **kwargs)
    dataset["metadata"]["name"] = "mipnerf360"
    dataset["metadata"]["scene"] = scene
    dataset["metadata"]["downscale_factor"] = res
    if resize_full_image:
        dataset["metadata"]["downscale_loaded_factor"] = res
    dataset["metadata"]["expected_scene_scale"] = get_scene_scale(dataset["cameras"], "object-centric")
    dataset["metadata"]["type"] = "object-centric"
    dataset["metadata"]["color_space"] = "srgb"
    dataset["metadata"]["evaluation_protocol"] = "nerf"

    viewer_transform, viewer_pose = get_default_viewer_transform(dataset["cameras"].poses, "object-centric")
    dataset["metadata"]["viewer_transform"] = viewer_transform
    dataset["metadata"]["viewer_initial_pose"] = viewer_pose

    image_names = dataset["image_paths"]
    inds = np.argsort(image_names)

    all_indices = np.arange(len(dataset["image_paths"]))
    llffhold = 8
    if split == "train":
        indices = all_indices % llffhold != 0
    else:
        indices = all_indices % llffhold == 0
    indices = inds[indices]
    return dataset_index_select(dataset, indices)


def download_mipnerf360_dataset(path: str, output: Union[Path, str]):
    url_extra = "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
    url_base = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    output = Path(output)
    if not path.startswith("mipnerf360/") and path != "mipnerf360":
        raise DatasetNotFoundError("Dataset path must be equal to 'mipnerf360' or must start with 'mipnerf360/'.")

    captures: List[Tuple[str, Path]] = []
    if path == "mipnerf360":
        # We will download all faster here
        for x in _scenes360_res:
            captures.append((x, output / x))
    else:
        captures = [(path[len("nerfstudio/") :], output)]
    captures_to_download: List[Tuple[str, str, Path]] = []
    for capture_name, output in captures:
        if capture_name not in _scenes360_res:
            raise DatasetNotFoundError(f"Capture '{capture_name}' not a valid mipnerf360 scene.")
        url = url_extra if capture_name in {"flowers", "treehill"} else url_base
        captures_to_download.append((url, capture_name, output))
    captures_to_download.sort(key=lambda x: x[0])
    for url, _captures in groupby(captures_to_download, key=lambda x: x[0]):
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

            has_any = False
            with zipfile.ZipFile(file) as z:
                for _, capture_name, output in _captures:
                    output_tmp = output.with_suffix(".tmp")
                    output_tmp.mkdir(exist_ok=True, parents=True)
                    for info in z.infolist():
                        if not info.filename.startswith(capture_name + "/"):
                            continue
                        # z.extract(name, output_tmp)
                        has_any = True
                        relname = info.filename[len(capture_name) + 1 :]
                        target = output_tmp / relname
                        target.parent.mkdir(exist_ok=True, parents=True)
                        if info.is_dir():
                            target.mkdir(exist_ok=True, parents=True)
                        else:
                            with z.open(info) as source, open(target, "wb") as target:
                                shutil.copyfileobj(source, target)

                    shutil.rmtree(output, ignore_errors=True)
                    if not has_any:
                        raise RuntimeError(f"Capture '{capture_name}' not found in {url}.")
                    shutil.move(str(output_tmp), str(output))
                    logging.info(f"Downloaded mipnerf360/{capture_name} to {output}")
