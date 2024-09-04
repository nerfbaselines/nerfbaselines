import json
import typing
from typing import Optional, Any
import logging
from pathlib import Path
import click
from nerfbaselines import get_method_spec, Method, build_method_class
from nerfbaselines import backends
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from ._common import handle_cli_error, click_backend_option, setup_logging


@click.command("viewer")
@click.option("--checkpoint", default=None, required=False)
@click.option("--data", type=str, default=None, required=False)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--port", type=int, default=6006)
@click_backend_option()
@handle_cli_error
def viewer_command(checkpoint: str, data, verbose, backend, port=6006):
    setup_logging(verbose)

    def run_viewer(method: Optional[Method] = None, nb_info=None):
        try:
            from nerfbaselines.viewer import run_viser_viewer

            run_viser_viewer(method, port=port, data=data, nb_info=nb_info)
        finally:
            if hasattr(method, "close"):
                typing.cast(Any, method).close()

    # Read method nb-info
    if checkpoint is not None:
        logging.info(f"Loading checkpoint {checkpoint}")
        with open_any_directory(checkpoint) as _checkpoint_path:
            checkpoint_path = Path(_checkpoint_path)
            assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
            assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
            with (checkpoint_path / "nb-info.json").open("r") as f:
                nb_info = json.load(f)
            nb_info = deserialize_nb_info(nb_info)

            method_name = nb_info["method"]
            backends.mount(checkpoint_path, checkpoint_path)
            method_spec = get_method_spec(method_name)
            with build_method_class(method_spec, backend=backend) as method_cls:
                method = method_cls(checkpoint=str(checkpoint_path))
                run_viewer(method, nb_info=nb_info)
    else:
        logging.info("Starting viewer without method")
        run_viewer()

    
if __name__ == "__main__":
    viewer_command()
