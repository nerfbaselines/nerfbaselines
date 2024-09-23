from contextlib import ExitStack
import logging
import click
from pathlib import Path
import json
from nerfbaselines import backends
from nerfbaselines import (
    get_method_spec, build_method_class,
)
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from nerfbaselines.datasets import load_dataset
from ._common import click_backend_option
from ._common import setup_logging, SetParamOptionType


@click.command("export-mesh", short_help="Export mesh from model checkpoint", help=(
    "Export a mesh from a trained model. "
    "This usually involves rendering all training cameras and fusing the depth maps and normals into a mesh. "
    "The mesh will be saved in the specified output directory, usually as `mesh.ply` file, but different methods can export other files or file formats. "
    "Note, only some methods support this feature."
))
@click.option("--checkpoint", default=None, required=False, type=str, help=(
    "Path to the checkpoint directory. It can also be a remote path (starting with `http(s)://`) or be a path inside a zip file."
))
@click.option("--output", "-o", type=str, required=True, help="Path to a directory to save the mesh (usually as a `mesh.ply` file).")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--data", type=str, default=None, required=False, help=(
    "A path to the dataset to load which matches the dataset used to train the model. "
    "The dataset can be either an external dataset (e.g., a path starting with `external://{dataset}/{scene}`) or a local path to a dataset. If the dataset is an external dataset, the dataset will be downloaded and cached locally. "
    "If the dataset is a local path, the dataset will be loaded directly from the specified path. "
    "While required for most, some methods do not require the dataset to export a mesh (e.g. `colmap`)."))
@click.option("--set", "options", type=SetParamOptionType(), multiple=True, default=None, help=(
    "Set a parameter for mesh export. " 
    "The argument should be in the form of `--set key=value` and can be used multiple times to set multiple parameters. "
    "The parameters will override the parameters stored in the checkpoint. "
    "The parameters are specific to each method."))
@click_backend_option()
def export_mesh_command(*, checkpoint: str, output: str, backend_name, data=None, options, verbose=False):
    checkpoint = str(checkpoint)
    output = str(output)
    setup_logging(verbose)
    options = dict(options or [])

    with ExitStack() as stack:
        # Read method nb-info
        checkpoint_path = Path(stack.enter_context(open_any_directory(checkpoint, mode="r")))
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
        backends.mount(checkpoint_path, checkpoint_path)
        method_spec = get_method_spec(method_name)
        method_cls = stack.enter_context(build_method_class(method_spec, backend=backend_name))
        method = method_cls(checkpoint=str(checkpoint_path))
        dataset_metadata = nb_info.get("dataset_metadata")
        dataset = None
        if data is not None:
            # We need to load features at least to fix image resolutions
            method_info = method.get_method_info()
            dataset = load_dataset(data, split="train", 
                                   load_features=True, 
                                   supported_camera_models=method_info.get("supported_camera_models"),
                                   features=method_info.get("required_features"))
            if dataset_metadata is not None:
                logging.warning("Overwriting dataset metadata stored in the checkpoint")
            dataset_metadata = dataset["metadata"]
        if dataset_metadata is None:
            logging.warning("No dataset metadata found in the checkpoint and no dataset provided as input. Some methods may require dataset metadata to export a demo. Please provide a dataset using the --data option.")
        try:
            method_export_mesh = method.export_mesh  # type: ignore
        except AttributeError:
            raise NotImplementedError(f"Method {method_name} does not support export_mesh")
        method_export_mesh(
            path=output,
            train_dataset=dataset,
            options=dict(**options, 
                         dataset_metadata=dataset_metadata)
        )
