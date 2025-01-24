import json
import pprint
import logging
import sys
import os
import click
import nerfbaselines
from contextlib import ExitStack
from nerfbaselines import (
    build_method_class,
)
from nerfbaselines.datasets import load_dataset
from nerfbaselines.training import (
    Trainer, Indices, build_logger,
    get_presets_and_config_overrides,
)
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from nerfbaselines import backends
from ._common import SetParamOptionType, TupleClickType, IndicesClickType, click_backend_option, NerfBaselinesCliCommand


@click.command("train", cls=NerfBaselinesCliCommand, help=(
    "Train a model of a specified method on a dataset. The method is specified by the `--method` argument, and the dataset is specified by the `--data` argument. The training will periodically save checkpoints, evaluate the intermediate model on a few images, and evaluate the final model on all images. The training progress will be logged to the console and optionally to TensorBoard, Weights & Biases, or another supported logger. The final model and predictions can be saved as an output artifact which can be uploaded to the web benchmark. The training can be resumed from a checkpoint by specifying the `--checkpoint` argument. The method's parameters can be overridden using the `--set` argument, and the method's presets can be applied using the `--presets` argument. The `--set` and `--presets` arguments can be used multiple times to apply multiple overrides and presets and are specific to each method."
), short_help="Train a model")
@click.option("--method", "method_name", type=click.Choice(sorted(nerfbaselines.get_supported_methods())), required=True, help="Method to use")
@click.option("--checkpoint", type=click.Path(path_type=str), default=None, help="Path to a checkpoint to resume training from.")
@click.option("--data", type=str, required=True, help=(
    "A path to the dataset to train on. The dataset can be either an external dataset (e.g., a path starting with `external://{dataset}/{scene}`) or a local path to a dataset. If the dataset is an external dataset, the dataset will be downloaded and cached locally. If the dataset is a local path, the dataset will be loaded directly from the specified path."))
@click.option("--output", type=str, default=".", help="Output directory to save the training results", show_default=True)
@click.option("--logger", type=click.Choice(["none", "wandb", "tensorboard", "wandb,tensorboard"]), default="tensorboard", help="Logger to use.", show_default=True)
@click.option("--save-iters", type=IndicesClickType(), default=Indices([-1]), help="When to save the model", show_default=True)
@click.option("--eval-few-iters", type=IndicesClickType(), default=Indices.every_iters(2_000), help="When to evaluate on few images", show_default=True)
@click.option("--eval-all-iters", type=IndicesClickType(), default=Indices([-1]), help="When to evaluate all images", show_default=True)
@click.option("--disable-output-artifact", "generate_output_artifact", help="Disable producing output artifact containing final model and predictions.", default=None, flag_value=False, is_flag=True)
@click.option("--force-output-artifact", "generate_output_artifact", help="Force producing output artifact containing final model and predictions.", default=None, flag_value=True, is_flag=True)
@click.option("--set", "config_overrides", type=SetParamOptionType(), multiple=True, default=None, help=(
    "Override a parameter in the method. The argument should be in the form of `--set key=value`. This argument can be used multiple times to override multiple parameters. And it is specific to each method."))
@click.option("--presets", type=TupleClickType(), default=None, help=(
    "Apply a comma-separated list of preset to the method. If no `--presets` is supplied, or if a special `@auto` preset is present (default if no presets are specified),"
    " the method's default presets are applied (based on the dataset metadata)."))
@click_backend_option()
def train_command(
    method_name,
    checkpoint,
    data,
    output,
    backend_name,
    save_iters,
    eval_few_iters,
    eval_all_iters,
    generate_output_artifact=None,
    logger="none",
    config_overrides=None,
    presets=None,
):
    if config_overrides is None:
        config_overrides = {}
    _loggers = frozenset((x for x in logger.split(",") if x != "none"))
    del logger

    if config_overrides is not None and isinstance(config_overrides, (list, tuple)):
        config_overrides = dict(config_overrides)

    if method_name is None and checkpoint is None:
        logging.error("Either --method or --checkpoint must be specified")
        sys.exit(1)

    with ExitStack() as stack:
        checkpoint_path = None
        if checkpoint is not None:
            checkpoint_path = stack.enter_context(open_any_directory(checkpoint))
            stack.enter_context(backends.mount(checkpoint_path, checkpoint_path))
            with open(os.path.join(checkpoint_path, "nb-info.json"), "r", encoding="utf8") as f:
                info = json.load(f)
            info = deserialize_nb_info(info)
            if method_name is not None and method_name != info["method"]:
                logging.error(f"Argument --method={method_name} is in conflict with the checkpoint's method {info['method']}.")
                sys.exit(1)
            method_name = info["method"]

        # Print started training
        logging.info(f"Started training, version: {nerfbaselines.__version__}, method: {method_name}")

        # Prepare the output directory
        output_path = stack.enter_context(open_any_directory(output, "w"))
        stack.enter_context(backends.mount(output_path, output_path))

        # Make paths absolute
        _data = data
        if "://" not in _data:
            _data = os.path.abspath(_data)
            stack.enter_context(backends.mount(_data, _data))

        # Change working directory to output
        os.chdir(str(output_path))

        # Build the method
        method_spec = nerfbaselines.get_method_spec(method_name)
        method_cls = stack.enter_context(build_method_class(method_spec, backend_name))

        # Load train dataset
        logging.info("Loading train dataset")
        method_info = method_cls.get_method_info()
        required_features = method_info.get("required_features", frozenset())
        supported_camera_models = method_info.get("supported_camera_models", frozenset(("pinhole",)))
        train_dataset = load_dataset(_data, 
                                     split="train", 
                                     features=required_features, 
                                     supported_camera_models=supported_camera_models, 
                                     load_features=True)
        assert train_dataset["cameras"].image_sizes is not None, "image sizes must be specified"

        # Load eval dataset
        logging.info("Loading eval dataset")
        test_dataset = load_dataset(_data, 
                                    split="test", 
                                    features=required_features, 
                                    supported_camera_models=supported_camera_models, 
                                    load_features=True)
        test_dataset["metadata"]["expected_scene_scale"] = train_dataset["metadata"].get("expected_scene_scale")

        # Apply config overrides for the train dataset
        _presets, _config_overrides = get_presets_and_config_overrides(
            method_spec, train_dataset["metadata"], presets=presets, config_overrides=config_overrides)
        # Log the current set of config overrides
        logging.info(f"Active presets: {', '.join(_presets)}")
        logging.info(f"Using config overrides: {pprint.pformat(_config_overrides)}")

        # Build the method
        method = method_cls(
            checkpoint=os.path.abspath(checkpoint_path) if checkpoint_path else None,
            train_dataset=train_dataset,
            config_overrides=_config_overrides,
        )

        # Train the method
        trainer = Trainer(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            method=method,
            output=output_path,
            save_iters=save_iters,
            eval_all_iters=eval_all_iters,
            eval_few_iters=eval_few_iters,
            logger=build_logger(_loggers),
            generate_output_artifact=generate_output_artifact,
            config_overrides=_config_overrides,
            applied_presets=frozenset(_presets),
        )
        trainer.train()


if __name__ == "__main__":
    train_command()  # pylint: disable=no-value-for-parameter
