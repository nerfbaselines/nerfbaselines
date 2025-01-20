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
from ._common import SetParamOptionType, NerfBaselinesCliCommand
from nerfbaselines._export_demo import export_demo


@click.command("export-demo", cls=NerfBaselinesCliCommand, help=(
    "Export a demo from a trained model. "
    "The interactive demo will be a website (index.html) that can be opened in the browser. "
    "Only some methods support this feature."))
@click.option("--checkpoint", default=None, required=False, type=str, help=(
    "Path to the checkpoint directory. It can also be a remote path (starting with `http(s)://`) or be a path inside a zip file."
))
@click.option("--output", "-o", type=str, required=True, help="Output directory for the demo.")
@click.option("--data", type=str, default=None, required=True, help=(
    "A path to the dataset to load which matches the dataset used to train the model. "
    "The dataset can be either an external dataset (e.g., a path starting with `external://{dataset}/{scene}`) or a local path to a dataset. If the dataset is an external dataset, the dataset will be downloaded and cached locally. "
    "If the dataset is a local path, the dataset will be loaded directly from the specified path. "))
@click.option("--train-embedding", type=str, default=None, help="Select the train embedding index to use for the demo (if the method supports appearance modelling. A comma-separated list of indices can be provided to select multiple embeddings.")
@click.option("--set", "options", type=SetParamOptionType(), multiple=True, default=None, help=(
    "Set a parameter for demo export. " 
    "The argument should be in the form of `--set key=value` and can be used multiple times to set multiple parameters. "
    "The parameters are specific to each method."))
@click_backend_option()
def main(*, checkpoint: str, output: str, backend_name, data: str, train_embedding=None, options):
    checkpoint = str(checkpoint)
    output = str(output)
    options = dict(options or [])

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
        method_spec = get_method_spec(method_name)
        with build_method_class(method_spec, backend=backend_name) as method_cls:
            method = method_cls(checkpoint=str(checkpoint_path))
            info = method.get_info()
            train_dataset = load_dataset(data, split="train", load_features=True, features=info.get("required_features"))
            test_dataset = load_dataset(data, split="train", load_features=True, features=info.get("required_features"))

            export_demo(output, 
                        method, 
                        [int(x) if x.lower() != "none" else None for x in train_embedding.split(",")] if train_embedding else None,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset)


if __name__ == "__main__":
    main()
