from typing import Union
from importlib import import_module
import logging
from pathlib import Path
from ..types import Dataset, DatasetFeature, FrozenSet, NB_PREFIX
from ._common import DatasetNotFoundError, MultiDatasetError


SUPPORTED_DATASETS = [
    "mipnerf360",
    "colmap",
    "nerfstudio",
    "blender",
]


def download_dataset(path: str, output: Path):
    output = Path(output)
    errors = {}
    for name in SUPPORTED_DATASETS:
        try:
            module = import_module(f".{name}", __package__)
            if not hasattr(module, f"download_{name}_dataset"):
                continue
            download_fn = getattr(module, f"download_{name}_dataset")
            download_fn(path, output)
            logging.info(f"downloaded {name} dataset with path {path}")
            return
        except DatasetNotFoundError as e:
            logging.debug(e)
            logging.debug(f"path {path} is not supported by {name} dataset")
            errors[name] = str(e)
    raise MultiDatasetError(errors, f"no supported dataset found for path {path}")


def load_dataset(path: Union[Path, str], split: str, features: FrozenSet[DatasetFeature]) -> Dataset:
    # If path is and external path, we download the dataset first
    if isinstance(path, str) and path.startswith("external://"):
        dataset = path.split("://", 1)[1]
        path = Path(NB_PREFIX) / "datasets" / dataset
        if not path.exists():
            download_dataset(dataset, path)

    path = Path(path)
    errors = {}
    for name in SUPPORTED_DATASETS:
        try:
            module = import_module(f".{name}", __package__)
            load_fn = getattr(module, f"load_{name}_dataset")
            dataset = load_fn(path, split=split, features=features)
            logging.info(f"loaded {name} dataset from path {path}")
            return dataset
        except DatasetNotFoundError as e:
            logging.debug(e)
            logging.debug(f"{name} dataset not found in path {path}")
            errors[name] = str(e)
    raise MultiDatasetError(errors, f"no supported dataset found in path {path}")
