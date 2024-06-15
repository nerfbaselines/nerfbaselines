import click
from pathlib import Path
import os
import json
from nerfbaselines.backends import ALL_BACKENDS
from nerfbaselines.utils import setup_logging
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from nerfbaselines import registry
from nerfbaselines import backends


@click.command("export-demo")
@click.option("--checkpoint", type=click.Path(file_okay=True, dir_okay=True, path_type=str), required=True)
@click.option("--output", "-o", type=click.Path(file_okay=True, dir_okay=False, path_type=str), required=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
def main(checkpoint: str, output: str, backend_name, verbose=False):
    checkpoint = str(checkpoint)
    output = str(output)
    setup_logging(verbose)

    # Read method nb-info
    with open_any_directory(checkpoint, mode="r") as _checkpoint_path:
        checkpoint_path = Path(_checkpoint_path)
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
        backends.mount(checkpoint_path, checkpoint_path)
        with registry.build_method(method_name, backend=backend_name) as method_cls:
            method = method_cls(checkpoint=str(checkpoint_path))
            dataset_info = nb_info["dataset_metadata"]
            method_export_demo = getattr(method, "export_demo", None)
            if method_export_demo is None:
                raise NotImplementedError(f"Method {method_name} does not support export_demo")
            method_export_demo(
                path=output,
                viewer_transform=dataset_info["viewer_transform"], 
                viewer_initial_pose=dataset_info["viewer_initial_pose"])


if __name__ == "__main__":
    main()
