import shutil
import struct
import sys
import json
import hashlib
import time
import os
import math
import logging
from pathlib import Path
from typing import Optional, Union, List, Any, Dict, Tuple, cast
from tqdm import tqdm
import numpy as np
import click
from .io import open_any_directory, deserialize_nb_info, serialize_nb_info
from .datasets import load_dataset, Dataset
from .utils import Indices, setup_logging, image_to_srgb, visualize_depth, handle_cli_error
from .utils import remap_error, get_resources_utilization_info, assert_not_none
from .utils import IndicesClickType, SetParamOptionType
from .utils import make_image_grid
from .cameras import CameraModel
from .types import Method, Literal, FrozenSet
from .render import render_all_images, build_update_progress
from .evaluate import EvaluationProtocol
from .upload_results import prepare_results_for_upload
from .logging import TensorboardLogger, WandbLogger, ConcatLogger, Logger
from . import backends
from . import __version__
from . import registry
from . import evaluate


def compute_exponential_gamma(num_iters: int, initial_lr: float, final_lr: float) -> float:
    gamma = (math.log(final_lr) - math.log(initial_lr)) / num_iters
    return math.exp(gamma)


def log_metrics(logger: Logger, metrics, *, prefix: str = "", step: int):
    with logger.add_event(step) as event:
        for k, val in metrics.items():
            tag = f"{prefix}{k}"
            if isinstance(val, (int, float)):
                event.add_scalar(tag, val)
            elif isinstance(val, str):
                event.add_text(tag, val)
            else:
                raise ValueError(f"Unknown metric type for {tag}: {val}")


def eval_few(method: Method, logger: Logger, dataset: Dataset, *, split: str, step, evaluation_protocol: EvaluationProtocol):
    rand_number, = struct.unpack("L", hashlib.sha1(str(step).encode("utf8")).digest()[:8])

    idx = rand_number % len(dataset)
    dataset_slice = dataset[idx : idx + 1]

    expected_scene_scale: Optional[float] = dataset_slice.metadata.get("expected_scene_scale")

    assert dataset_slice.images is not None, f"{split} dataset must have images loaded"
    assert dataset_slice.cameras.image_sizes is not None, f"{split} dataset must have image_sizes specified"

    start = time.perf_counter()
    # Pseudo-randomly select an image based on the step
    total_rays = 0
    with tqdm(desc=f"rendering single image at step={step}") as pbar:
        predictions = None
        for predictions in evaluation_protocol.render(method, dataset_slice, progress_callback=build_update_progress(pbar, simple=True)):
            pass
        assert predictions is not None, "render failed to compute predictions"
    elapsed = time.perf_counter() - start

    # Log to wandb
    if logger:
        logging.debug(f"logging image to {logger}")

        # Log to wandb
        metrics = {}
        for _metrics in evaluation_protocol.evaluate([predictions], dataset_slice):
            metrics = cast(Dict[str, Union[str, int, float]], _metrics)

        w, h = dataset_slice.cameras.image_sizes[0]
        gt = dataset_slice.images[0][:h, :w]
        color = predictions["color"]

        background_color = dataset_slice.metadata.get("background_color", None)
        color_srgb = image_to_srgb(color, np.uint8, color_space=dataset_slice.color_space, background_color=background_color)
        gt_srgb = image_to_srgb(gt, np.uint8, color_space=dataset_slice.color_space, background_color=background_color)

        image_path = dataset_slice.file_paths[0]
        if dataset_slice.file_paths_root is not None:
            if str(image_path).startswith(str(dataset_slice.file_paths_root)):
                image_path = str(Path(image_path).relative_to(dataset_slice.file_paths_root))

        metrics["image-path"] = image_path
        metrics["fps"] = 1 / elapsed
        metrics["rays-per-second"] = total_rays / elapsed
        metrics["time"] = elapsed

        depth = None
        if "depth" in predictions:
            near_far = dataset_slice.cameras.nears_fars[0] if dataset_slice.cameras.nears_fars is not None else None
            depth = visualize_depth(predictions["depth"], expected_scale=expected_scene_scale, near_far=near_far)
        log_metrics(logger, metrics, prefix=f"eval-few-{split}/", step=step)

        color_vis = make_image_grid(gt_srgb, color_srgb)
        with logger.add_event(step) as event:
            event.add_image(
                f"eval-few-{split}/color",
                color_vis,
                display_name=image_path,
                description="left: gt, right: prediction",
            )
            if depth is not None:
                event.add_image(
                    f"eval-few-{split}/depth",
                    depth,
                    display_name=image_path,
                    description="depth",
                )


def eval_all(method: Method, logger: Logger, dataset: Dataset, *, output: Union[str, Path], step: int, evaluation_protocol: EvaluationProtocol, split: str, nb_info):
    assert dataset.images is not None, "test dataset must have images loaded"
    total_rays = 0
    metrics: Optional[Dict[str, float]] = {} if logger else None
    expected_scene_scale = dataset.metadata.get("expected_scene_scale")

    # Store predictions, compute metrics, etc.
    prefix = dataset.file_paths_root
    if prefix is None:
        prefix = Path(os.path.commonpath(dataset.file_paths))

    output = Path(output)
    if split != "test":
        output_metrics = output / f"results-{step}-{split}.json"
        output = output / f"predictions-{step}-{split}.tar.gz"
    else:
        output_metrics = output / f"results-{step}.json"
        output = output / f"predictions-{step}.tar.gz"

    if output.exists():
        if output.is_file():
            os.unlink(str(output))
        else:
            shutil.rmtree(str(output))
        logging.warning(f"removed existing predictions at {output}")

    if output_metrics.exists():
        os.unlink(str(output_metrics))
        logging.warning(f"removed existing results at {output_metrics}")

    start = time.perf_counter()
    num_vis_images = 16
    vis_images: List[Tuple[np.ndarray, np.ndarray]] = []
    vis_depth: List[np.ndarray] = []
    for (i, gt), pred, (w, h) in zip(
        enumerate(dataset.images),
        render_all_images(
            method,
            dataset,
            output=output,
            description=f"rendering all images at step={step}",
            nb_info=nb_info,
            evaluation_protocol=evaluation_protocol,
        ),
        assert_not_none(dataset.cameras.image_sizes),
    ):
        if len(vis_images) < num_vis_images:
            color = pred["color"]
            background_color = dataset.metadata.get("background_color", None)
            color_srgb = image_to_srgb(color, np.uint8, color_space=dataset.color_space, background_color=background_color)
            gt_srgb = image_to_srgb(gt[:h, :w], np.uint8, color_space=dataset.color_space, background_color=background_color)
            vis_images.append((gt_srgb, color_srgb))
            if "depth" in pred:
                near_far = dataset.cameras.nears_fars[i] if dataset.cameras.nears_fars is not None else None
                vis_depth.append(visualize_depth(pred["depth"], expected_scale=expected_scene_scale, near_far=near_far))
    elapsed = time.perf_counter() - start

    # Compute metrics
    info = evaluate.evaluate(
        output, 
        output_metrics, 
        evaluation_protocol=evaluation_protocol,
        description=f"evaluating all images at step={step}")
    metrics = info["metrics"]

    if logger:
        assert metrics is not None, "metrics must be computed"
        logging.debug(f"logging metrics to {logger}")
        metrics["fps"] = len(dataset) / elapsed
        metrics["rays-per-second"] = total_rays / elapsed
        metrics["time"] = elapsed
        log_metrics(logger, metrics, prefix=f"eval-all-{split}/", step=step)

        num_cols = int(math.sqrt(len(vis_images)))

        color_vis = make_image_grid(
            make_image_grid(*[x[0] for x in vis_images], ncol=num_cols),
            make_image_grid(*[x[1] for x in vis_images], ncol=num_cols),
        )

        logger.add_image(f"eval-all-{split}/color", 
                         color_vis, 
                         display_name="color", 
                         description="left: gt, right: prediction", 
                         step=step)



MetricAccumulationMode = Literal["average", "last", "sum"]


class MetricsAccumulator:
    def __init__(
        self,
        options: Optional[Dict[str, MetricAccumulationMode]] = None,
    ):
        self.options = options or {}
        self._state = None

    def update(self, metrics: Dict[str, Union[int, float]]) -> None:
        if self._state is None:
            self._state = {}
        state = self._state
        n_iters_since_update = state["n_iters_since_update"] = state.get("n_iters_since_update", {})
        for k, v in metrics.items():
            accumulation_mode = self.options.get(k, "average")
            n_iters_since_update[k] = n = n_iters_since_update.get(k, 0) + 1
            if k not in state:
                state[k] = 0
            if accumulation_mode == "last":
                state[k] = v
            elif accumulation_mode == "average":
                state[k] = state[k] * ((n - 1) / n) + v / n
            elif accumulation_mode == "sum":
                state[k] += v
            else:
                raise ValueError(f"Unknown accumulation mode {accumulation_mode}")

    def pop(self) -> Dict[str, Union[int, float]]:
        if self._state is None:
            return {}
        state = self._state
        self._state = None
        state.pop("n_iters_since_update", None)
        return state


Visualization = Literal["none", "wandb", "tensorboard"]


class Trainer:
    def __init__(
        self,
        *,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        method: Method,
        output: Path = Path("."),
        save_iters: Indices = Indices.every_iters(10_000, zero=True),
        eval_few_iters: Indices = Indices.every_iters(2_000),
        eval_all_iters: Indices = Indices([-1]),
        loggers: FrozenSet[Visualization] = frozenset(),
        generate_output_artifact: Optional[bool] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.method = method
        self.model_info = self.method.get_info()
        self.test_dataset: Optional[Dataset] = test_dataset

        self.step = self.model_info.get("loaded_step") or 0
        self.num_iterations = self.model_info["num_iterations"]
        if self.num_iterations is None:
            raise RuntimeError(f"Method {self.model_info['name']} must specify the default number of iterations")

        self.output = output
        self.save_iters = save_iters

        self.eval_few_iters = eval_few_iters
        self.eval_all_iters = eval_all_iters
        self.loggers = loggers
        self.generate_output_artifact = generate_output_artifact
        self.config_overrides = config_overrides

        self._logger: Optional[Logger] = None
        self._average_image_size = None
        self._dataset_metadata = None
        self._total_train_time = 0
        self._resources_utilization_info = None
        self._train_dataset_for_eval = None
        self._acc_metrics = MetricsAccumulator({
            "total-train-time": "last",
            "learning-rate": "last",
        })

        # Restore checkpoint if specified
        loaded_checkpoint = self.model_info.get("loaded_checkpoint")
        if loaded_checkpoint is not None:
            with Path(loaded_checkpoint).joinpath("nb-info.json").open("r", encoding="utf8") as f:
                info = json.load(f)
                info = deserialize_nb_info(info)
                self._total_train_time = info["total_train_time"]
                self._resources_utilization_info = info["resources_utilization"]

        # Fix total for indices
        for v in vars(self).values():
            if isinstance(v, Indices):
                v.total = self.num_iterations + 1

        # Validate and setup datasets
        self._setup_data(train_dataset, test_dataset)

    def _setup_data(self, train_dataset, test_dataset):
        # Validate and setup datasets
        # Store a slice of train dataset used for eval_few
        self._average_image_size = train_dataset.cameras.image_sizes.prod(-1).astype(np.float32).mean()
        dataset_background_color = train_dataset.metadata.get("background_color")
        if dataset_background_color is not None:
            assert isinstance(dataset_background_color, np.ndarray), "Dataset background color must be a numpy array"
            assert dataset_background_color.dtype == np.uint8, "Dataset background color must be an uint8 array"
        train_dataset_indices = np.linspace(0, len(train_dataset) - 1, 16, dtype=int)
        self._train_dataset_for_eval = train_dataset[train_dataset_indices]

        assert train_dataset.color_space is not None
        color_space = train_dataset.color_space
        self._dataset_metadata = train_dataset.metadata.copy()

        # Setup test dataset dataset
        self.test_dataset = test_dataset
        if test_dataset is not None:
            if test_dataset.color_space != color_space:
                raise RuntimeError(f"train dataset color space {color_space} != test dataset color space {test_dataset.color_space}")
            test_background_color = test_dataset.metadata.get("background_color")
            if test_background_color is not None:
                assert isinstance(test_background_color, np.ndarray), "Dataset's background_color must be a numpy array"
            if not (
                (test_background_color is None and dataset_background_color is None) or 
                (
                    test_background_color is not None and 
                    dataset_background_color is not None and
                    np.array_equal(test_background_color, dataset_background_color)
                )
            ):
                raise RuntimeError(f"train dataset color space {dataset_background_color} != test dataset color space {test_dataset.metadata.get('background_color')}")

        self._validate_output_artifact()
        test_dataset_name = self.test_dataset.metadata.get("name") if self.test_dataset is not None else train_dataset.metadata.get("name")
        self._evaluation_protocol = evaluate.get_evaluation_protocol(test_dataset_name)

    def _validate_output_artifact(self):
        # Validate generate output artifact
        if self.generate_output_artifact is None or self.generate_output_artifact:
            messages = []
            # Model is saved automatically at the end!
            # if self.num_iterations not in self.save_iters:
            #     messages.append(f"num_iterations ({self.num_iterations}) must be in save_iters: {self.save_iters}")
            if self.num_iterations not in self.eval_all_iters:
                messages.append(f"num_iterations ({self.num_iterations}) must be in eval_all_iters: {self.eval_all_iters}")
            if "tensorboard" not in self.loggers:
                messages.append("Tensorboard logger must be enabled. Please add `--vis tensorboard` to the command line arguments.")

            if self.generate_output_artifact is None and messages:
                logging.warning("Disabling output artifact generation due to the following problems:")
                for message in messages:
                    logging.warning(message)
                self.generate_output_artifact = False
            elif messages:
                logging.error("Cannot generate output artifact due to the following problems:")
                for message in messages:
                    logging.error(message)
                sys.exit(1)
            else:
                self.generate_output_artifact = True

    def _get_nb_info(self):
        assert self._dataset_metadata is not None, "dataset_metadata must be set"
        dataset_metadata = self._dataset_metadata.copy()
        expected_scene_scale = self._dataset_metadata.get("expected_scene_scale", None)
        dataset_metadata["expected_scene_scale"] = round(expected_scene_scale, 5) if expected_scene_scale is not None else None
        dataset_metadata["background_color"] = dataset_metadata["background_color"].tolist() if dataset_metadata.get("background_color") is not None else None
        return {
            "method": self.model_info["name"],
            "nb_version": __version__,
            "num_iterations": self.model_info["num_iterations"],
            "total_train_time": round(self._total_train_time, 5),
            "resources_utilization": self._resources_utilization_info,
            # Date time in ISO format
            "datetime": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "config_overrides": self.config_overrides,
            "dataset_metadata": self._dataset_metadata,
            "evaluation_protocol": self._evaluation_protocol.get_name(),

            "color_space": self._dataset_metadata.get("color_space"),  # TODO: remove
            "expected_scene_scale": expected_scene_scale,  # TODO: remove
            "dataset_background_color": dataset_metadata["background_color"],  # TODO: remove
        }

    def save(self):
        path = os.path.join(str(self.output), f"checkpoint-{self.step}")  # pyright: ignore[reportCallIssue]
        os.makedirs(os.path.join(str(self.output), f"checkpoint-{self.step}"), exist_ok=True)
        self.method.save(Path(path))
        with open(os.path.join(path, "nb-info.json"), mode="w+", encoding="utf8") as f:
            json.dump(serialize_nb_info(self._get_nb_info()), f)
        logging.info(f"checkpoint saved at step={self.step}")

    def train_iteration(self):
        start = time.perf_counter()
        metrics = self.method.train_iteration(self.step)

        elapsed = time.perf_counter() - start
        self._total_train_time += elapsed

        # Replace underscores with dashes for in metrics
        metrics = {k.replace("_", "-"): v for k, v in metrics.items()}
        metrics["time"] = elapsed
        metrics["total-train-time"] = self._total_train_time
        if "num_rays" in metrics:
            batch_size = metrics.pop("num_rays")
            metrics["rays-per-second"] = batch_size / elapsed
            if self._average_image_size is not None:
                metrics["fps"] = batch_size / elapsed / self._average_image_size
        return metrics

    def get_logger(self) -> Logger:
        if self._logger is None:
            loggers = []
            for logger in self.loggers:
                if "tensorboard" == logger:
                    loggers.append(TensorboardLogger(str(self.output / "tensorboard")))
                elif "wandb" == logger:
                    loggers.append(WandbLogger(str(self.output)))
                else:
                    raise ValueError(f"Unknown logger {logger}")
            self._logger = ConcatLogger(loggers)
            logging.info("initialized loggers: " + ",".join(self.loggers))
        return self._logger

    def _update_resource_utilization_info(self):
        update = False
        util: Dict[str, Union[int, float]] = {}
        if self._resources_utilization_info is None:
            update = True
        elif self.step % 1000 == 11:
            update = True
            util = self._resources_utilization_info
        if update:
            logging.debug(f"computing resource utilization at step={self.step}")
            new_util = cast(Dict[str, int], get_resources_utilization_info())
            for k, v in new_util.items():
                if k not in util:
                    util[k] = 0
                util[k] = max(util[k], v)
            self._resources_utilization_info = util

    @remap_error
    def train(self):
        assert self.num_iterations is not None, "num_iterations must be set"
        assert self._average_image_size is not None, "dataset not set"
        if self.step == 0 and self.step in self.save_iters:
            self.save()

        # Initialize loggers before training loop for better tqdm output
        logger = self.get_logger()

        update_frequency = 100
        with tqdm(total=self.num_iterations, initial=self.step, desc="training") as pbar:
            for i in range(self.step, self.num_iterations):
                self.step = i
                metrics = self.train_iteration()
                # Checkpoint changed, reset sha
                self._checkpoint_sha = None
                self.step = i + 1
                pbar.update()

                # Update accumulated metrics
                self._acc_metrics.update(metrics)

                # Update resource utilization info
                self._update_resource_utilization_info()

                # Log metrics and update progress bar
                if self.step % update_frequency == 0 or self.step == self.num_iterations:
                    acc_metrics = self._acc_metrics.pop()
                    pbar.set_postfix(
                        {
                            "train/loss": f'{acc_metrics["loss"]:.4f}',
                        }
                    )
                    log_metrics(logger, acc_metrics, prefix="train/", step=self.step)

                # Visualize and save
                if self.step in self.save_iters:
                    self.save()
                if self.step in self.eval_few_iters:
                    self.eval_few()
                if self.step in self.eval_all_iters:
                    self.eval_all()

        # Save if not saved by default
        if self.step not in self.save_iters:
            self.save()

        # Generate output artifact if enabled
        if self.generate_output_artifact:
            prepare_results_for_upload(
                self.output / f"checkpoint-{self.step}",
                self.output / f"predictions-{self.step}.tar.gz",
                self.output / f"results-{self.step}.json",
                self.output / "tensorboard",
                self.output / "output.zip",
                validate=False,
            )

    def eval_all(self):
        if self.test_dataset is None:
            logging.warning("skipping eval_all on test dataset - no test dataset")
            return
        logger = self.get_logger()
        nb_info = self._get_nb_info()
        eval_all(self.method, logger, self.test_dataset, 
                 step=self.step, evaluation_protocol=self._evaluation_protocol,
                 split="test", nb_info=nb_info, output=self.output)

    def eval_few(self):
        logger = self.get_logger()

        assert self._train_dataset_for_eval is not None, "train_dataset_for_eval must be set"
        rand_number, = struct.unpack("L", hashlib.sha1(str(self.step).encode("utf8")).digest()[:8])

        idx = rand_number % len(self._train_dataset_for_eval)
        dataset_slice = self._train_dataset_for_eval[idx : idx + 1]

        eval_few(self.method, logger, dataset_slice, split="train", step=self.step, evaluation_protocol=self._evaluation_protocol)
        
        if self.test_dataset is None:
            logging.warning("skipping eval_few on test dataset - no eval dataset")
            return

        idx = rand_number % len(self.test_dataset)
        dataset_slice = self.test_dataset[idx : idx + 1]
        eval_few(self.method, logger, dataset_slice, split="test", step=self.step, evaluation_protocol=self._evaluation_protocol)


@click.command("train")
@click.option("--method", "method_name", type=click.Choice(sorted(registry.supported_methods())), required=True, help="Method to use")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--data", type=str, required=True)
@click.option("--output", type=str, default=".")
@click.option("--vis", type=click.Choice(["none", "wandb", "tensorboard", "wandb+tensorboard"]), default="tensorboard", help="Logger to use. Defaults to tensorboard.")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--save-iters", type=IndicesClickType(), default=Indices.every_iters(10_000), help="When to save the model")
@click.option("--eval-few-iters", type=IndicesClickType(), default=Indices.every_iters(2_000), help="When to evaluate on few images")
@click.option("--eval-all-iters", type=IndicesClickType(), default=Indices([-1]), help="When to evaluate all images")
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--disable-output-artifact", "generate_output_artifact", help="Disable producing output artifact containing final model and predictions.", default=None, flag_value=False, is_flag=True)
@click.option("--force-output-artifact", "generate_output_artifact", help="Force producing output artifact containing final model and predictions.", default=None, flag_value=True, is_flag=True)
@click.option("--set", "config_overrides", help="Override a parameter in the method.", type=SetParamOptionType(), multiple=True, default=None)
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
    vis="none",
    config_overrides=None,
):
    loggers: FrozenSet[Visualization]
    if vis == "wandb":
        loggers = frozenset(("wandb",))
    elif vis == "tensorboard":
        loggers = frozenset(("tensorboard",))
    elif vis in {"wandb+tensorboard", "tensorboard+wandb"}:
        loggers = frozenset(("wandb", "tensorboard"))
    elif vis == "none":
        loggers = frozenset()
    else:
        raise ValueError(f"unknown visualization tool {vis}")

    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    if config_overrides is not None and isinstance(config_overrides, (list, tuple)):
        config_overrides = dict(config_overrides)

    if method_name is None and checkpoint is None:
        logging.error("Either --method or --checkpoint must be specified")
        sys.exit(1)

    def _train(checkpoint_path=None):
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

            with registry.build_method(method_name, backend_name) as method_cls:

                # Load train dataset
                logging.info("loading train dataset")
                method_info = method_cls.get_method_info()
                required_features = method_info.get("required_features", frozenset())
                supported_camera_models = method_info.get("supported_camera_models", frozenset((CameraModel.PINHOLE,)))
                train_dataset = load_dataset(_data, split="train", features=required_features)
                train_dataset.load_features(required_features, supported_camera_models)
                assert train_dataset.cameras.image_sizes is not None, "image sizes must be specified"

                # Load eval dataset
                logging.info("loading eval dataset")
                test_dataset = load_dataset(_data, split="test", features=required_features)
                test_dataset.metadata["expected_scene_scale"] = train_dataset.metadata.get("expected_scene_scale")
                test_dataset.load_features(required_features.union({"color"}))

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
                    output=Path(output),
                    save_iters=save_iters,
                    eval_all_iters=eval_all_iters,
                    eval_few_iters=eval_few_iters,
                    loggers=frozenset(loggers),
                    generate_output_artifact=generate_output_artifact,
                    config_overrides=config_overrides,
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
