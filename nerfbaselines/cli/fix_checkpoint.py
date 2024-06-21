from datetime import datetime
import time
import shutil
import logging
import json
import warnings
from pathlib import Path
import os
import click
from typing import cast
from nerfbaselines.utils import setup_logging, handle_cli_error, SetParamOptionType
from nerfbaselines import backends, registry
from nerfbaselines.types import Method, Dataset
from nerfbaselines.datasets import load_dataset
from nerfbaselines.io import open_any_directory, deserialize_nb_info, serialize_nb_info
from nerfbaselines.io import new_nb_info
from ._common import ChangesTracker


def _fix_sha_keys(obj):
    for k in list(obj.keys()):
        if k.endswith("_sha256"):
            obj[:-3] = obj.pop(k)
        elif isinstance(obj[k], dict):
            _fix_sha_keys(obj[k])


def update_nb_info(nb_info, new_nb_info):
    _fix_sha_keys(nb_info)
    for k in list(nb_info.keys()):
        if k not in new_nb_info:
            nb_info.pop(k)
    new_nb_info = new_nb_info.copy()
    if "nb_version" in nb_info:
        new_nb_info["nb_version"] = nb_info.pop("nb_version")
    if "total_train_time" in nb_info:
        new_nb_info.pop("total_train_time", None)
    new_nb_info.pop("resources_utilization", None)
    if nb_info.get("resources_utilization") is not None and "gpu_name" not in nb_info["resources_utilization"]:
        nb_info["resources_utilization"]["gpu_name"] = "NVIDIA A100-SXM4-40GB"
    if nb_info.get("resources_utilization") is not None and "gpu_name" in nb_info["resources_utilization"]:
        nb_info["resources_utilization"]["gpu_name"] = ",".join(x.strip() for x in nb_info["resources_utilization"]["gpu_name"].split(","))
    new_nb_info.pop("datetime", None)
    nb_info.update(new_nb_info)


def fix_checkpoint(checkpoint_path, new_checkpoint, load_train_dataset_fn, backend_name, method_name=None, force=False, config_overrides=None, changes_tracker=None):
    if config_overrides is None:
        config_overrides = {}

    # Read method nb-info
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f"checkpoint path {checkpoint_path} does not exist"
    # assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
    nb_info = None
    if not checkpoint_path.joinpath("nb-info.json").exists():
        if method_name is None:
            raise RuntimeError(f"Checkpoint path {checkpoint_path} does not contain nb-info.json and was not produced by nerfbaselines. Please specify --method argument.")
        warnings.warn(f"Checkpoint path {checkpoint_path} does not contain nb-info.json and was not produced by nerfbaselines. Not all methods support this.")
    else:
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)
        # Get nb-info creation time as "%Y-%m-%dT%H:%M:%S%z"
        if "datetime" not in nb_info:
            dm = datetime(*time.gmtime(os.path.getmtime(checkpoint_path / "nb-info.json"))[:6]).isoformat(timespec="seconds")
            nb_info["datetime"] = dm
        if method_name is not None:
            if nb_info["method"] != method_name:
                if force:
                    warnings.warn(f"Method name in nb-info.json {nb_info['method']} does not match the provided method name {method_name}. Forcing the method name {method_name}.")
                else:
                    raise RuntimeError(f"Method name in nb-info.json {nb_info['method']} does not match the provided method name {method_name}")
        else:
            method_name = nb_info["method"]

    with registry.build_method(method_name, backend=backend_name) as method_cls:
        method_info = method_cls.get_method_info()
        train_dataset = load_train_dataset_fn(
             features=method_info.get("required_features"),
             supported_camera_models=method_info.get("supported_camera_models"))

        dataset_overrides = registry.get_dataset_overrides(method_name, train_dataset["metadata"])
        if train_dataset["metadata"].get("name") is None:
            logging.warning("Dataset name not specified, dataset-specific config overrides may not be applied")
        if dataset_overrides is not None:
            dataset_overrides = dataset_overrides.copy()
            dataset_overrides.update(config_overrides or {})
            config_overrides = dataset_overrides
        del dataset_overrides

        method: Method = method_cls(
            checkpoint=str(checkpoint_path), 
            train_dataset=cast(Dataset, train_dataset),
            config_overrides=config_overrides)

        nb_info = (nb_info or {}).copy()
        update_nb_info(nb_info, new_nb_info(
            train_dataset["metadata"],
            method,
            config_overrides=config_overrides,
        ))
        try:
            with open_any_directory(new_checkpoint, mode="w") as _new_checkpoint:
                method.save(_new_checkpoint)
                with open(os.path.join(_new_checkpoint, "nb-info.json"), mode="w+", encoding="utf8") as f:
                    json.dump(serialize_nb_info(nb_info), f, indent=2)

                if changes_tracker is not None:
                    changes_tracker.add_dir_changes((), checkpoint_path, _new_checkpoint)

            # Validate the new checkpoint
            logging.info("Validating the new checkpoint")
            with open_any_directory(new_checkpoint, mode="r") as _new_checkpoint:
                method_new: Method = method_cls(checkpoint=_new_checkpoint)

                # Test if we can render
                out = list(method_new.render(train_dataset["cameras"][:1]))
                assert len(out) == 1, f"Rendering failed: {out}"
        except Exception as e:
            if os.path.isfile(new_checkpoint):
                os.remove(new_checkpoint)
            elif os.path.isdir(new_checkpoint):
                shutil.rmtree(new_checkpoint)
            raise e


@click.command("fix-checkpoint")
@click.option("--checkpoint", type=str, default=None, required=True)
@click.option("--data", type=str, default=None, required=True)
@click.option("--method", "method_name", type=str, default=None, required=False)
@click.option("--new-checkpoint", type=str, required=True, help="Path to save the new checkpoint")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--set", "config_overrides", help="Override a parameter in the method.", type=SetParamOptionType(), multiple=True, default=None)
@handle_cli_error
def main(checkpoint: str, data: str, method_name: str, verbose: bool, backend_name, new_checkpoint: str, config_overrides=None):
    setup_logging(verbose)
    if os.path.exists(new_checkpoint):
        raise RuntimeError(f"New checkpoint path {new_checkpoint} already exists")

    if config_overrides is not None and isinstance(config_overrides, (list, tuple)):
        config_overrides = dict(config_overrides)

    # Read method nb-info
    with open_any_directory(checkpoint, mode="r") as _checkpoint_path:
        backends.mount(_checkpoint_path, _checkpoint_path)

        def load_train_dataset_fn(features=None, supported_camera_models=None):
            return load_dataset(data, 
                                split="train", 
                                features=features,
                                supported_camera_models=supported_camera_models)
        changes_tracker = ChangesTracker()
        fix_checkpoint(_checkpoint_path, new_checkpoint, load_train_dataset_fn, backend_name, method_name=method_name, config_overrides=config_overrides, changes_tracker=changes_tracker)

        # Print changes
        print()
        print("Changes:")
        changes_tracker.print_changes()
        print()
        logging.info(f"New checkpoint is stored at {new_checkpoint}")
