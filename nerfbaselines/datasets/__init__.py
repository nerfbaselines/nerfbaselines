from importlib import import_module
import logging
from pathlib import Path
from ..types import Dataset, DatasetFeature, FrozenSet
from ._common import DatasetNotFoundError


SUPPORTED_DATASETS = [
    "mipnerf360",
    "colmap",
]


def load_dataset(path: Path, split: str, features: FrozenSet[DatasetFeature]) -> Dataset:
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
    raise DatasetNotFoundError(f"no supported dataset found in path {path}:" "".join(f"\n  {name}: {error}" for name, error in errors.items()))
