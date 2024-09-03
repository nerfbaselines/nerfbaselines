import json
import pprint
import logging
import sys
import os
import click
import nerfbaselines
from nerfbaselines import (
    build_method_class,
)
from nerfbaselines.datasets import load_dataset
from nerfbaselines.training import (
    Trainer, Indices, build_logger,
    get_presets_to_apply, get_config_overrides_from_presets,
)
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from nerfbaselines import backends
from ._common import SetParamOptionType, TupleClickType, IndicesClickType, handle_cli_error, click_backend_option, setup_logging


@click.command("train")
@click.option("--method", "method_name", type=click.Choice(sorted(nerfbaselines.get_supported_methods())), required=True, help="Method to use")
@click.option("--checkpoint", type=click.Path(path_type=str), default=None)
@click.option("--data", type=str, required=True)
@click.option("--output", type=str, default=".")
@click.option("--logger", type=click.Choice(["none", "wandb", "tensorboard", "wandb,tensorboard"]), default="tensorboard", help="Logger to use. Defaults to tensorboard.")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--save-iters", type=IndicesClickType(), default=Indices([-1]), help="When to save the model")
@click.option("--eval-few-iters", type=IndicesClickType(), default=Indices.every_iters(2_000), help="When to evaluate on few images")
@click.option("--eval-all-iters", type=IndicesClickType(), default=Indices([-1]), help="When to evaluate all images")
@click.option("--disable-output-artifact", "generate_output_artifact", help="Disable producing output artifact containing final model and predictions.", default=None, flag_value=False, is_flag=True)
@click.option("--force-output-artifact", "generate_output_artifact", help="Force producing output artifact containing final model and predictions.", default=None, flag_value=True, is_flag=True)
@click.option("--set", "config_overrides", help="Override a parameter in the method.", type=SetParamOptionType(), multiple=True, default=None)
@click.option("--presets", type=TupleClickType(), default=None, help=(
    "Apply a comma-separated list of preset to the method. If no `--presets` is supplied, or if a special `@auto` preset is present,"
    " the method's default presets are applied (based on the dataset metadata)."))
@click_backend_option()
@handle_cli_error
def train_command(
    method_name,
    checkpoint,
    data,
    output,
    verbose,
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

    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    if config_overrides is not None and isinstance(config_overrides, (list, tuple)):
        config_overrides = dict(config_overrides)

    if method_name is None and checkpoint is None:
        logging.error("Either --method or --checkpoint must be specified")
        sys.exit(1)

    def _train(checkpoint_path=None):
        nonlocal config_overrides
        # Make paths absolute
        _data = data
        if "://" not in _data:
            _data = os.path.abspath(_data)
            backends.mount(_data, _data)
        if checkpoint_path is not None:
            backends.mount(checkpoint_path, checkpoint_path)
        with open_any_directory(output, "w") as output_path, \
                    backends.mount(output_path, output_path):

            # change working directory to output
            os.chdir(str(output_path))

            method_spec = nerfbaselines.get_method_spec(method_name)
            with build_method_class(method_spec, backend_name) as method_cls:
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
                _presets = get_presets_to_apply(method_spec, train_dataset["metadata"], presets)
                dataset_overrides = get_config_overrides_from_presets(
                    method_spec,
                    _presets,
                )
                if train_dataset["metadata"].get("name") is None:
                    logging.warning("Dataset name not specified, dataset-specific config overrides may not be applied")
                if dataset_overrides is not None:
                    dataset_overrides = dataset_overrides.copy()
                    dataset_overrides.update(config_overrides or {})
                    config_overrides = dataset_overrides
                del dataset_overrides

                # Log the current set of config overrides
                logging.info(f"Active presets: {', '.join(_presets)}")
                logging.info(f"Using config overrides: {pprint.pformat(config_overrides)}")

                # Build the method
                method = method_cls(
                    checkpoint=os.path.abspath(checkpoint_path) if checkpoint_path else None,
                    train_dataset=train_dataset,
                    config_overrides=config_overrides,
                )

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
                    config_overrides=config_overrides,
                    applied_presets=frozenset(_presets),
                )
                trainer.train()

    if checkpoint is not None:
        with open_any_directory(checkpoint) as checkpoint_path:
            with open(os.path.join(checkpoint_path, "nb-info.json"), "r", encoding="utf8") as f:
                info = json.load(f)
            info = deserialize_nb_info(info)
            if method_name is not None and method_name != info["method"]:
                logging.error(f"Argument --method={method_name} is in conflict with the checkpoint's method {info['method']}.")
                sys.exit(1)
            method_name = info["method"]
            _train(checkpoint_path)
    else:
        _train(None)


if __name__ == "__main__":
    train_command()  # pylint: disable=no-value-for-parameter
