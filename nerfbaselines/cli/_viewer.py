import json
from contextlib import ExitStack
import logging
from pathlib import Path
import click

from nerfbaselines.viewer import run_viser_viewer
from nerfbaselines import get_method_spec, build_method_class
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
def viewer_command(checkpoint: str, data, verbose, backend_name, port=6006):
    setup_logging(verbose)

    with ExitStack() as stack:
        nb_info = None
        method = None
        if checkpoint is not None:
            # Forward port
            stack.enter_context(backends.forward_port(port, port))

            # Load checkpoint directory
            logging.info(f"Loading checkpoint {checkpoint}")
            _checkpoint_path = stack.enter_context(open_any_directory(checkpoint))
            stack.enter_context(backends.mount(_checkpoint_path, _checkpoint_path))

            # Read method nb-info
            checkpoint_path = Path(_checkpoint_path)
            assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
            assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
            with (checkpoint_path / "nb-info.json").open("r") as f:
                nb_info = json.load(f)
            nb_info = deserialize_nb_info(nb_info)

            # Build the method
            method_name = nb_info["method"]
            method_spec = get_method_spec(method_name)
            method_cls = stack.enter_context(build_method_class(method_spec, backend=backend_name))
            method = method_cls(checkpoint=str(checkpoint_path))
        else:
            logging.info("Starting viewer without method")

        # Start the viewer
        run_viser_viewer(method, port=port, data=data, nb_info=nb_info)


if __name__ == "__main__":
    viewer_command()
