import shutil
import logging
import pprint
import json
import warnings
from pathlib import Path
import os
import click
from nerfbaselines.utils import setup_logging, handle_cli_error, SetParamOptionType
from nerfbaselines import backends, registry
from nerfbaselines.datasets import load_dataset
from nerfbaselines.io import open_any_directory, deserialize_nb_info, serialize_nb_info
from nerfbaselines.types import Method
from nerfbaselines.train import get_nb_info


@click.command("fix-checkpoint")
@click.option("--checkpoint", type=str, default=None, required=True)
@click.option("--data", type=str, default=None, required=True)
@click.option("--method", "method_name", type=str, default=None, required=False)
@click.option("--new-checkpoint", type=str, required=True, help="Path to save the new checkpoint")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--set", "config_overrides", help="Override a parameter in the method.", type=SetParamOptionType(), multiple=True, default=None)
@handle_cli_error
def fix_checkpoint_command(checkpoint: str, data: str, method_name: str, verbose: bool, backend_name, new_checkpoint: str, config_overrides=None):
    setup_logging(verbose)
    if os.path.exists(new_checkpoint):
        raise RuntimeError(f"New checkpoint path {new_checkpoint} already exists")

    if config_overrides is not None and isinstance(config_overrides, (list, tuple)):
        config_overrides = dict(config_overrides)

    # Read method nb-info
    with open_any_directory(checkpoint, mode="r") as _checkpoint_path:
        backends.mount(_checkpoint_path, _checkpoint_path)
        checkpoint_path = Path(_checkpoint_path)
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        # assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        nb_info = None
        if not checkpoint_path.joinpath("nb-info.json").exists():
            if method_name is None:
                raise RuntimeError(f"Checkpoint path {checkpoint} does not contain nb-info.json and was not produced by nerfbaselines. Please specify --method argument.")
            warnings.warn(f"Checkpoint path {checkpoint} does not contain nb-info.json and was not produced by nerfbaselines. Not all methods support this.")
        else:
            with (checkpoint_path / "nb-info.json").open("r") as f:
                nb_info = json.load(f)
            nb_info = deserialize_nb_info(nb_info)
            method_name = nb_info["method"]


        old_nb_info = nb_info
        with registry.build_method(method_name, backend=backend_name) as method_cls:
            method_info = method_cls.get_method_info()
            train_dataset = load_dataset(data, 
                                         split="train", 
                                         load_features=False,
                                         features=method_info.get("required_features"),
                                         supported_camera_models=method_info.get("supported_camera_models"))
            method: Method = method_cls(
                checkpoint=str(checkpoint_path), 
                train_dataset=train_dataset,
                config_overrides=config_overrides)

            # TODO: merge nb_info and old_nb_info
            warnings.warn("Merging of nb_info is not implemented yet")
            nb_info = dict(**nb_info, **get_nb_info(
                train_dataset["metadata"],
                method,
                config_overrides=config_overrides,
            ))
            try:
                with open_any_directory(new_checkpoint, mode="w") as _new_checkpoint:
                    method.save(_new_checkpoint)
                    with open(os.path.join(_new_checkpoint, "nb-info.json"), mode="w+", encoding="utf8") as f:
                        json.dump(serialize_nb_info(nb_info), f, indent=2)

                # Validate the new checkpoint
                logging.info("Validating the new checkpoint")
                with open_any_directory(new_checkpoint, mode="r") as _new_checkpoint:
                    method_new: Method = method_cls(checkpoint=_new_checkpoint)

                    # Test if we can render
                    out = list(method_new.render(train_dataset["cameras"][:1]))
                    assert len(out) == 1, f"Rendering failed: {out}"
                    logging.info(f"New checkpoint is stored at {new_checkpoint}")
            except Exception as e:
                if os.path.isfile(new_checkpoint):
                    os.remove(new_checkpoint)
                elif os.path.isdir(new_checkpoint):
                    shutil.rmtree(new_checkpoint)
                raise e

