import logging
from itertools import groupby
import shutil
import requests
from pathlib import Path
import numpy as np
import zipfile
from tqdm import tqdm
import tempfile
from ..types import Dataset
from ._common import DatasetNotFoundError, single
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


def load_mipnerf360_dataset(path: Path, split: str, **kwargs):
    if split:
        assert split in {"train", "test"}
    if "360" not in str(path) or not any(s in str(path) for s in _scenes360_res):
        raise DatasetNotFoundError(f"360 and {set(_scenes360_res.keys())} is missing from the dataset path: {path}")

    # Load MipNerf360 dataset
    scene = single(res for res in _scenes360_res if str(res) in path.name)
    res = _scenes360_res[scene]
    images_path = Path(f"images_{res}")

    # Use split=None to load all images
    # We then select the same images as in the LLFF multinerf dataset loader
    dataset: Dataset = load_colmap_dataset(path, images_path=images_path, split=None, **kwargs)
    dataset.metadata["type"] = "mipnerf360"
    dataset.metadata["scene"] = scene

    image_names = dataset.file_paths
    inds = np.argsort(image_names)

    all_indices = np.arange(len(dataset))
    llffhold = 8
    if split == "train":
        indices = all_indices % llffhold != 0
    else:
        indices = all_indices % llffhold == 0
    indices = inds[indices]
    return dataset[indices]


def download_mipnerf360_dataset(path: str, output: Path):
    url_extra = "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip"
    url_base = "http://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
    output = Path(output)
    if not path.startswith("mipnerf360/") and path != "mipnerf360":
        raise DatasetNotFoundError("Dataset path must be equal to 'mipnerf360' or must start with 'mipnerf360/'.")

    captures = []
    if path == "mipnerf360":
        # We will download all faster here
        for x in _scenes360_res:
            captures.append((x, output / x))
    else:
        captures = [(path[len("nerfstudio/") :], output)]
    captures_to_download = []
    for capture_name, output in captures:
        if capture_name not in _scenes360_res:
            raise DatasetNotFoundError(f"Capture '{capture_name}' not a valid mipnerf360 scene.")
        url = url_extra if capture_name in {"flowers", "treehill"} else url_base
        captures_to_download.append((url, capture_name, output))
    captures_to_download.sort(key=lambda x: x[0])
    for url, captures in groupby(captures_to_download, key=lambda x: x[0]):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Downloading {url.split('/')[-1]}")
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
                for _, capture_name, output in captures:
                    output_tmp = output.with_suffix(".tmp")
                    output_tmp.mkdir(exist_ok=True, parents=True)
                    for name in z.namelist():
                        if name.startswith(capture_name):
                            z.extract(name, output_tmp)
                            has_any = True
                    shutil.rmtree(output, ignore_errors=True)
                    if not has_any:
                        raise RuntimeError(f"Capture '{capture_name}' not found in {url}.")
                    shutil.move(output_tmp, output)
                    logging.info(f"Downloaded mipnerf360/{capture_name} to {output}")
