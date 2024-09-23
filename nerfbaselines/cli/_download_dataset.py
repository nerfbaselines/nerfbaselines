from pathlib import Path
import logging
import click
from ._common import NerfBaselinesCliCommand
from nerfbaselines import NB_PREFIX
from nerfbaselines.datasets import download_dataset


@click.command("download-dataset", cls=NerfBaselinesCliCommand, short_help="Download a dataset", help=(
    "Download a dataset either to a specified directory or to the default dataset directory. "
    "The dataset must have format `external://{dataset}/{scene}`, where `{dataset}` is one of the registered datasets and `{scene}` is the scene id supported by the dataset. "
))
@click.argument("dataset", type=str, required=True)
@click.option("--output", "-o", type=click.Path(file_okay=False, dir_okay=True, path_type=str), required=False, default=None, help=(
    f"Output directory to download the dataset to. If not provided, the dataset will be downloaded to the default dataset directory ({NB_PREFIX})."))
def download_dataset_command(dataset: str, output: str):
    logging.basicConfig(level=logging.INFO)
    if output is None:
        _out_dataset = dataset
        if _out_dataset.startswith("external://"):
            _out_dataset = _out_dataset[len("external://") :]
        output = str(Path(NB_PREFIX) / "datasets" / _out_dataset)
    download_dataset(dataset, output)
    logging.info(f"Dataset {dataset} downloaded to {output}")


