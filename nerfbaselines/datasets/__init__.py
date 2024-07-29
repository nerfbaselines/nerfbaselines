import json
import os
from typing import Union, Optional, overload
import logging
from pathlib import Path
from ..types import Dataset, DatasetFeature, CameraModel, FrozenSet, NB_PREFIX
from ._common import dataset_load_features as dataset_load_features
from ._common import dataset_index_select as dataset_index_select
from ._common import new_dataset as new_dataset
from ._common import DatasetNotFoundError, MultiDatasetError
from ._common import experimental_parse_dataset_path
from ..types import UnloadedDataset, Literal


def download_dataset(path: str, output: Union[str, Path]):
    from ..registry import get_dataset_downloaders

    output = Path(output)
    errors = {}
    for name, download_fn in get_dataset_downloaders().items():
        try:
            download_fn(path, str(output))
            logging.info(f"Downloaded {name} dataset with path {path}")
            return
        except DatasetNotFoundError as e:
            logging.debug(e)
            logging.debug(f"Path {path} is not supported by {name} dataset")
            errors[name] = str(e)
    raise MultiDatasetError(errors, f"No supported dataset found for path {path}")


@overload
def load_dataset(
        path: Union[Path, str], 
        split: str, 
        features: Optional[FrozenSet[DatasetFeature]] = ...,
        supported_camera_models: Optional[FrozenSet[CameraModel]] = ...,
        load_features: Literal[True] = ...,
        **kwargs) -> Dataset:
    ...


@overload
def load_dataset(
        path: Union[Path, str], 
        split: str, 
        features: Optional[FrozenSet[DatasetFeature]] = ...,
        supported_camera_models: Optional[FrozenSet[CameraModel]] = ...,
        load_features: Literal[False] = ...,
        **kwargs) -> UnloadedDataset:
    ...



def load_dataset(
        path: Union[Path, str], 
        split: str, 
        features: Optional[FrozenSet[DatasetFeature]] = None,
        supported_camera_models: Optional[FrozenSet[CameraModel]] = None,
        load_features: bool = True,
        **kwargs,
        ) -> Union[Dataset, UnloadedDataset]:
    from ..registry import get_dataset_loaders, get_dataset_spec

    path = str(path)
    path, _kwargs = experimental_parse_dataset_path(path)
    _kwargs.update(kwargs)
    kwargs = _kwargs
    if features is None:
        features = frozenset(("color",))
    kwargs["features"] = features
    del features
    if supported_camera_models is None:
        supported_camera_models = frozenset(("pinhole",))
    # If path is and external path, we download the dataset first
    if path.startswith("external://"):
        dataset = path.split("://", 1)[1]
        path = Path(NB_PREFIX) / "datasets" / dataset
        if not path.exists():
            download_dataset(dataset, path)
        path = str(path)

    loaders = list(get_dataset_loaders())
    loaders_override = False
    loader = None
    if "://" in path:
        # We assume the 
        loader, path = path.split("://", 1)
        if loader not in dict(loaders):
            raise ValueError(f"Unknown dataset loader {loader}")
        loaders_override = True
        loaders = [(loader, dict(loaders)[loader])]

    # Try loading info if exists
    meta = {}
    info_fname = "nb-info.json"
    if (os.path.exists(os.path.join(path, "info.json")) and 
        not os.path.exists(os.path.join(path, info_fname))):
        info_fname = "info.json"
    if os.path.exists(os.path.join(path, info_fname)):
        logging.info(f"Loading dataset metadata from {os.path.join(path, info_fname)}")
        with open(os.path.join(path, info_fname), "r") as f:
            meta = json.load(f)
        loader_ = meta.pop("loader", None)
        if loader is None:
            loader = loader_
            if loader is not None and not loaders_override:
                loaders = [(loader, dict(loaders)[loader])]
        if loader_ is None or loader == loader_:
            for k, v in meta.pop("loader_kwargs", {}).items():
                if k not in kwargs:
                    kwargs[k] = v
        else:
            logging.warning(f"Not using loader_kwargs because dataset's loader: {loader_} does not match specified loader: {loader}")

    errors = {}
    dataset_instance = None
    for name, load_fn in loaders:
        try:
            dataset_instance = load_fn(path, split=split, **kwargs)
            logging.info(f"Loaded {name} dataset from path {path} using loader {name}")
            dataset_instance["metadata"].update(meta)
            break
        except DatasetNotFoundError as e:
            logging.debug(e)
            logging.debug(f"{name} dataset not found in path {path}")
            errors[name] = str(e)
    else:
        raise MultiDatasetError(errors, f"No supported dataset found in path {path}")

    # Set correct eval protocol
    eval_protocol = get_dataset_spec(name).get("evaluation_protocol", "default")
    if dataset_instance["metadata"].get("evaluation_protocol", "default") != eval_protocol:
        raise RuntimeError(f"Evaluation protocol mismatch: {dataset_instance['metadata']['evaluation_protocol']} != {eval_protocol}")
    dataset_instance["metadata"]["evaluation_protocol"] = eval_protocol

    if load_features:
        return dataset_load_features(dataset_instance, features=kwargs["features"], supported_camera_models=supported_camera_models)
    return dataset_instance
