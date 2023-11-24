from pathlib import Path
import numpy as np
from ..types import Dataset
from ._common import DatasetNotFoundError, single
from .colmap import load_colmap_dataset


def load_mipnerf360_dataset(path: Path, split: str, **kwargs):
    if split:
        assert split in {"train", "test"}
    scenes360_res = {
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
    if "360" not in str(path) or not any(s in str(path) for s in scenes360_res):
        raise DatasetNotFoundError(f"360 and {set(scenes360_res.keys())} is missing from the dataset path: {path}")

    # Load MipNerf360 dataset
    scene = single(res for res in scenes360_res if str(res) in path.name)
    res = scenes360_res[scene]
    images_path = Path(f"images_{res}")

    # Use split=None to load all images
    # We then select the same images as in the LLFF multinerf dataset loader
    dataset: Dataset = load_colmap_dataset(path, images_path=images_path, split=None, **kwargs)
    dataset.metadata["type"] = "mipnerf360"

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
