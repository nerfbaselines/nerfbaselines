import click
import typing
from typing import Any
from pathlib import Path
import os
import json
import logging
from .utils import setup_logging
from .io import open_any_directory, deserialize_nb_info
from . import registry


@click.command("export-demo")
@click.option("--checkpoint", type=click.Path(file_okay=True, dir_okay=True, path_type=Path), required=True)
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False, path_type=Path), required=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
def main(checkpoint, output, backend, verbose=False):
    checkpoint = str(checkpoint)
    setup_logging(verbose)

    # Read method nb-info
    with open_any_directory(checkpoint, mode="r") as checkpoint_path:
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
        method_spec = registry.get(method_name)
        method_cls, backend = method_spec.build(backend=backend, checkpoint=Path(os.path.abspath(str(checkpoint))))
        logging.info(f"Using backend: {backend}")

        if hasattr(method_cls, "install"):
            method_cls.install()

        method = method_cls()
        try:
            dataset_info = nb_info["dataset_metadata"]
            method.export_demo(
                path=Path(output),
                viewer_transform=dataset_info["viewer_transform"], 
                viewer_initial_pose=dataset_info["viewer_initial_pose"])

        finally:
            if hasattr(method, "close"):
                typing.cast(Any, method).close()


if __name__ == "__main__":
    main()