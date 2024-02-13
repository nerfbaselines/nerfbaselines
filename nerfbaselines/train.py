import shutil
import struct
import sys
import json
import hashlib
from functools import partial
import time
import os
import math
import logging
from pathlib import Path
import typing
from typing import Callable, Optional, Union, Type, List, Any, Dict, Tuple
from tqdm import tqdm
import numpy as np
import click
from .io import open_any_directory
from .datasets import load_dataset, Dataset
from .utils import Indices, setup_logging, partialclass, image_to_srgb, visualize_depth, handle_cli_error
from .utils import remap_error, get_resources_utilization_info, assert_not_none
from .utils import IndicesClickType, SetParamOptionType
from .utils import make_image_grid
from .types import Method, Literal, FrozenSet
from .render import render_all_images, build_update_progress
from .evaluate import EvaluationProtocol, get_extra_metrics_available
from .upload_results import prepare_results_for_upload
from .logging import TensorboardLogger, WandbLogger, ConcatLogger, Logger
from . import __version__
from . import registry
from . import evaluate


def compute_exponential_gamma(num_iters: int, initial_lr: float, final_lr: float) -> float:
    gamma = (math.log(final_lr) - math.log(initial_lr)) / num_iters
    return math.exp(gamma)


def method_get_resources_utilization_info(method):
    if hasattr(method, "call"):
        return method.call(f"{get_resources_utilization_info.__module__}.{get_resources_utilization_info.__name__}")
    return get_resources_utilization_info()


def log_metrics(logger: Logger, metrics, *, prefix: str = "", step: int):
    with logger.add_event(step) as event:
        for k, val in metrics.items():
            tag = f"{prefix}{k}"
            if isinstance(val, (int, float)):
                event.add_scalar(tag, val)
            elif isinstance(val, str):
                event.add_text(tag, val)


def eval_few(method: Method, logger: Logger, dataset: Dataset, *, split: str, step, evaluation_protocol: EvaluationProtocol):
    rand_number, = struct.unpack("L", hashlib.sha1(str(step).encode("utf8")).digest()[:8])

    idx = rand_number % len(dataset)
    dataset_slice = dataset[idx : idx + 1]

    expected_scene_scale: Optional[float] = dataset_slice.metadata["expected_scene_scale"]

    assert dataset_slice.images is not None, f"{split} dataset must have images loaded"
    assert dataset_slice.cameras.image_sizes is not None, f"{split} dataset must have image_sizes specified"

    start = time.perf_counter()
    # Pseudo-randomly select an image based on the step
    total_rays = 0
    with tqdm(desc=f"rendering single image at step={step}") as pbar:
        predictions = next(
            iter(
                evaluation_protocol.render(method, dataset_slice, progress_callback=build_update_progress(pbar, simple=True))
            )
        )
    elapsed = time.perf_counter() - start

    # Log to wandb
    if logger:
        logging.debug(f"logging image to {logger}")

        # Log to wandb
        metrics = next(iter(evaluation_protocol.evaluate([predictions], dataset_slice)))

        w, h = dataset_slice.cameras.image_sizes[0]
        gt = dataset_slice.images[0][:h, :w]
        color = predictions["color"]

        background_color = dataset_slice.metadata.get("background_color", None)
        color_srgb = image_to_srgb(color, np.uint8, color_space=dataset_slice.color_space, background_color=background_color)
        gt_srgb = image_to_srgb(gt, np.uint8, color_space=dataset_slice.color_space, background_color=background_color)

        image_path = dataset_slice.file_paths[0]
        if dataset_slice.file_paths_root is not None:
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


def eval_all(method: Method, logger: Logger, dataset: Dataset, *, output: Union[str, Path], step: int, evaluation_protocol: EvaluationProtocol, split: str, ns_info):
    assert dataset.images is not None, "test dataset must have images loaded"
    total_rays = 0
    metrics: Optional[Dict[str, float]] = {} if logger else None
    expected_scene_scale = dataset.metadata["expected_scene_scale"]

    # Store predictions, compute metrics, etc.
    prefix = dataset.file_paths_root
    if prefix is None:
        prefix = Path(os.path.commonpath(dataset.file_paths))

    if split != "test":
        output_metrics = output / f"results-{step}-{split}.json"
        output = output / f"predictions-{step}-{split}.tar.gz"
    else:
        output_metrics = output / f"results-{step}.json"
        output = output / f"predictions-{step}.tar.gz"

    if output.exists():
        shutil.rmtree(str(output))
        logging.warning(f"removed existing predictions at {output}")

    if output_metrics.exists():
        shutil.rmtree(str(output))
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
            ns_info=ns_info,
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
    info = evaluate.evaluate(output, output_metrics, description=f"evaluating all images at step={step}")
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
        n_iters_since_update = self._state.pop("n_iters_since_update", 0) + 1
        state = self._state
        for k, v in metrics.items():
            accumulation_mode = self.options.get(k, "average")
            if k not in state:
                state[k] = 0
            if accumulation_mode == "last":
                state[k] = v
            elif accumulation_mode == "average":
                state[k] = state[k] * ((n_iters_since_update - 1) / n_iters_since_update) + v / n_iters_since_update
            elif accumulation_mode == "sum":
                state[k] += v
            else:
                raise ValueError(f"Unknown accumulation mode {accumulation_mode}")
        state["n_iters_since_update"] = n_iters_since_update
        self._state = state

    def pop(self) -> Dict[str, Union[int, float]]:
        if self._state is None:
            return {}
        state = self._state
        self._state = None
        state.pop("n_iters_since_update", 0)
        return state


Visualization = Literal["none", "wandb", "tensorboard"]


class Trainer:
    def __init__(
        self,
        *,
        train_dataset: Union[str, Path, Callable[[], Dataset]],
        test_dataset: Union[None, str, Path, Callable[[], Dataset]] = None,
        method: Type[Method],
        output: Path = Path("."),
        num_iterations: Optional[int] = None,
        save_iters: Indices = Indices.every_iters(10_000, zero=True),
        eval_few_iters: Indices = Indices.every_iters(2_000),
        eval_all_iters: Indices = Indices([-1]),
        loggers: FrozenSet[Visualization] = frozenset(),
        run_extra_metrics: bool = False,
        method_name: Optional[str] = None,
        generate_output_artifact: Optional[bool] = None,
        checkpoint: Union[str, Path, None] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        self.method_name = method_name or method.__name__
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.method = method(**({"checkpoint": checkpoint} if checkpoint is not None else {}))
        method_info = self.method.get_info()
        if isinstance(train_dataset, (Path, str)):
            if test_dataset is None:
                test_dataset = train_dataset
            train_dataset_path = train_dataset
            train_dataset = partial(load_dataset, train_dataset, split="train", features=method_info.required_features)

            # Allow direct access to the stored images if needed for docker remote
            if hasattr(self.method, "mounts"):
                typing.cast(Any, self.method).mounts.append((str(train_dataset_path), str(train_dataset_path)))
        if isinstance(test_dataset, (Path, str)):
            test_dataset = partial(load_dataset, test_dataset, split="test", features=method_info.required_features)
        assert test_dataset is not None, "test dataset must be specified"
        self._train_dataset_fn: Callable[[], Dataset] = train_dataset
        self._test_dataset_fn: Callable[[], Dataset] = test_dataset
        self.test_dataset: Optional[Dataset] = None

        self.step = method_info.loaded_step or 0
        self.output = output
        self.num_iterations = num_iterations
        self.save_iters = save_iters

        self.eval_few_iters = eval_few_iters
        self.eval_all_iters = eval_all_iters
        self.loggers = loggers
        self.run_extra_metrics = run_extra_metrics
        self.generate_output_artifact = generate_output_artifact
        self.config_overrides = config_overrides

        self._logger: Optional[Logger] = None
        self._average_image_size = None
        self._dataset_metadata = None
        self._method_info = method_info
        self._total_train_time = 0
        self._resources_utilization_info = None
        self._train_dataset_for_eval = None
        self._acc_metrics = MetricsAccumulator({
            "total-train-time": "last",
            "learning-rate": "last",
        })

        # Restore checkpoint if specified
        if self.checkpoint is not None:
            with open(self.checkpoint / "nb-info.json", mode="r", encoding="utf8") as f:
                info = json.load(f)
                self._total_train_time = info["total_train_time"]
                self._resources_utilization_info = info["resources_utilization"]

    def _setup_num_iterations(self):
        if self.num_iterations is None:
            self.num_iterations = self._method_info.num_iterations
        if self.num_iterations is None:
            raise RuntimeError(f"Method {self.method_name} must specify the default number of iterations")
        for v in vars(self).values():
            if isinstance(v, Indices):
                v.total = self.num_iterations + 1

    def _validate_output_artifact(self):
        # Validate generate output artifact
        if self.generate_output_artifact is None or self.generate_output_artifact:
            messages = []
            # Model is saved automatically at the end!
            # if self.num_iterations not in self.save_iters:
            #     messages.append(f"num_iterations ({self.num_iterations}) must be in save_iters: {self.save_iters}")
            if self.num_iterations not in self.eval_all_iters:
                messages.append(f"num_iterations ({self.num_iterations}) must be in eval_all_iters: {self.eval_all_iters}")
            if not self.run_extra_metrics:
                messages.append("Extra metrics must be enabled. Please verify they are installed and not disabled by --disable-extra-metrics flag.")
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

    @remap_error
    def setup_data(self):
        logging.info("loading train dataset")
        train_dataset: Dataset = self._train_dataset_fn()
        logging.info("loading eval dataset")
        method_info = self.method.get_info()

        train_dataset.load_features(method_info.required_features, method_info.supported_camera_models)
        assert train_dataset.cameras.image_sizes is not None, "image sizes must be specified"
        self._average_image_size = train_dataset.cameras.image_sizes.prod(-1).astype(np.float32).mean()
        dataset_background_color = train_dataset.metadata.get("background_color")
        if dataset_background_color is not None:
            assert isinstance(dataset_background_color, np.ndarray), "Dataset background color must be a numpy array"
            assert dataset_background_color.dtype == np.uint8, "Dataset background color must be an uint8 array"

        # Store a slice of train dataset used for eval_few
        train_dataset_indices = np.linspace(0, len(train_dataset) - 1, 16, dtype=int)
        self._train_dataset_for_eval = train_dataset[train_dataset_indices]

        self.method.setup_train(train_dataset, num_iterations=self.num_iterations, **({"config_overrides": self.config_overrides} if self.config_overrides is not None else {}))

        assert train_dataset.color_space is not None
        color_space = train_dataset.color_space
        self._dataset_metadata = train_dataset.metadata.copy()

        self.test_dataset = self._test_dataset_fn()
        self.test_dataset.metadata["expected_scene_scale"] = self._dataset_metadata["expected_scene_scale"]
        self.test_dataset.load_features(method_info.required_features.union({"color"}))
        if self.test_dataset.color_space != color_space:
            raise RuntimeError(f"train dataset color space {color_space} != test dataset color space {self.test_dataset.color_space}")
        test_background_color = self.test_dataset.metadata.get("background_color")
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
            raise RuntimeError(f"train dataset color space {dataset_background_color} != test dataset color space {self.test_dataset.metadata.get('background_color')}")

        self._method_info = self.method.get_info()
        self._setup_num_iterations()
        self._validate_output_artifact()
        test_dataset_name = self.test_dataset.metadata.get("name") if self.test_dataset.metadata is not None else train_dataset.metadata.get("name")
        self._evaluation_protocol = evaluate.get_evaluation_protocol(test_dataset_name, run_extra_metrics=self.run_extra_metrics)

    def _get_ns_info(self):
        dataset_metadata = self._dataset_metadata.copy()
        expected_scene_scale = self._dataset_metadata.get("expected_scene_scale", None)
        dataset_metadata["expected_scene_scale"] = round(expected_scene_scale, 5) if expected_scene_scale is not None else None
        dataset_metadata["background_color"] = dataset_metadata["background_color"].tolist() if dataset_metadata.get("background_color") is not None else None
        return {
            "method": self.method_name,
            "nb_version": __version__,
            "num_iterations": self._method_info.num_iterations,
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
            json.dump(self._get_ns_info(), f)
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
            if "tensorboard" in self.loggers:
                loggers.append(TensorboardLogger(str(self.output / "tensorboard")))
            if "wandb" in self.loggers:
                loggers.append(WandbLogger(str(self.output)))
            self._logger = ConcatLogger(loggers)
            logging.info("initialized loggers: " + ",".join(self.loggers))
        return self._logger

    def _update_resource_utilization_info(self):
        update = False
        util = {}
        if self._resources_utilization_info is None:
            update = True
        elif self.step % 1000 == 11:
            update = True
            util = self._resources_utilization_info
        if update:
            logging.debug(f"computing resource utilization at step={self.step}")
            new_util = method_get_resources_utilization_info(self.method)
            for k, v in new_util.items():
                if k not in util:
                    util[k] = 0
                util[k] = max(util[k], v)
            self._resources_utilization_info = util

    @remap_error
    def train(self):
        assert self.num_iterations is not None, "num_iterations must be set"
        if self._average_image_size is None:
            self.setup_data()
        if self.step == 0 and self.step in self.save_iters:
            self.save()

        # Initialize loggers before training loop for better tqdm output
        logger = self.get_logger()

        update_frequency = 100
        with tqdm(total=self.num_iterations, initial=self.step, desc="training") as pbar:
            last_update_i = self.step
            for i in range(self.step, self.num_iterations):
                self.step = i
                metrics = self.train_iteration()
                # Checkpoint changed, reset sha
                self._checkpoint_sha = None
                self.step = i + 1

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
                    pbar.update(self.step - last_update_i)
                    last_update_i = self.step
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
        logger = self.get_logger()
        ns_info = self._get_ns_info()
        eval_all(self.method, logger, self.test_dataset, 
                 step=self.step, evaluation_protocol=self._evaluation_protocol,
                 split="test", ns_info=ns_info, output=self.output)

    def close(self):
        if self.method is not None and hasattr(self.method, "close"):
            typing.cast(Any, self.method).close()

    def eval_few(self):
        assert self._train_dataset_for_eval is not None, "train_dataset_for_eval must be set"
        rand_number, = struct.unpack("L", hashlib.sha1(str(self.step).encode("utf8")).digest()[:8])

        idx = rand_number % len(self._train_dataset_for_eval)
        dataset_slice = self._train_dataset_for_eval[idx : idx + 1]

        eval_few(self.method, self.logger, dataset_slice, split="train", step=self.step, evaluation_protocol=self._evaluation_protocol)
        
        if self.test_dataset is None:
            logging.warning("skipping eval_few on test dataset - no eval dataset")
            return

        idx = rand_number % len(self.test_dataset)
        dataset_slice = self.test_dataset[idx : idx + 1]
        eval_few(self.method, self.logger, dataset_slice, split="test", step=self.step, evaluation_protocol=self._evaluation_protocol)


@click.command("train")
@click.option("--method", type=click.Choice(sorted(registry.supported_methods())), required=True, help="Method to use")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--data", type=str, required=True)
@click.option("--output", type=str, default=".")
@click.option("--vis", type=click.Choice(["none", "wandb", "tensorboard", "wandb+tensorboard"]), default="tensorboard", help="Logger to use. Defaults to tensorboard.")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--eval-few-iters", type=IndicesClickType(), default=Indices.every_iters(2_000), help="When to evaluate on few images")
@click.option("--eval-all-iters", type=IndicesClickType(), default=Indices([-1]), help="When to evaluate all images")
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--disable-extra-metrics", help="Disable extra metrics which need additional dependencies.", is_flag=True)
@click.option("--disable-output-artifact", "generate_output_artifact", help="Disable producing output artifact containing final model and predictions.", default=None, flag_value=False, is_flag=True)
@click.option("--force-output-artifact", "generate_output_artifact", help="Force producing output artifact containing final model and predictions.", default=None, flag_value=True, is_flag=True)
@click.option("--num-iterations", type=int, help="Number of training iterations.", default=None)
@click.option("--set", "config_overrides", help="Override a parameter in the method.", type=SetParamOptionType(), multiple=True, default=None)
@handle_cli_error
def train_command(
    method,
    checkpoint,
    data,
    output,
    verbose,
    backend,
    eval_few_iters,
    eval_all_iters,
    num_iterations=None,
    disable_extra_metrics=None,
    generate_output_artifact=None,
    vis="none",
    config_overrides=None,
):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    if config_overrides is not None and isinstance(config_overrides, (list, tuple)):
        config_overrides = dict(config_overrides)

    if disable_extra_metrics is None:
        disable_extra_metrics = not get_extra_metrics_available()

    if method is None and checkpoint is None:
        logging.error("Either --method or --checkpoint must be specified")
        sys.exit(1)

    def _train(checkpoint_path=None):
        with open_any_directory(output, "w") as output_path:
            # Make paths absolute, and change working directory to output
            _data = data
            if "://" not in _data:
                _data = os.path.abspath(_data)
            os.chdir(str(output_path))

            method_spec = registry.get(method)
            _method, _backend = method_spec.build(backend=backend, checkpoint=Path(os.path.abspath(checkpoint_path)) if checkpoint_path else None)
            logging.info(f"Using method: {method}, backend: {_backend}")

            # Enable direct memory access to images and if supported by the backend
            if _backend in {"docker", "apptainer"} and "://" not in _data:
                _method = partialclass(_method, mounts=[(_data, _data)])
            if hasattr(_method, "install"):
                _method.install()

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

            trainer = Trainer(
                train_dataset=_data,
                output=Path(output),
                method=_method,
                eval_all_iters=eval_all_iters,
                eval_few_iters=eval_few_iters,
                loggers=frozenset(loggers),
                num_iterations=num_iterations,
                run_extra_metrics=not disable_extra_metrics,
                generate_output_artifact=generate_output_artifact,
                method_name=method,
                config_overrides=config_overrides,
            )
            try:
                trainer.setup_data()
                trainer.train()
            finally:
                trainer.close()

    if checkpoint is not None:
        with open_any_directory(checkpoint) as checkpoint_path:
            with open(os.path.join(checkpoint_path, "nb-info.json"), "r", encoding="utf8") as f:
                info = json.load(f)
            if method is not None and method != info["method"]:
                logging.error(f"Argument --method={method} is in conflict with the checkpoint's method {info['method']}.")
                sys.exit(1)
            method = info["method"]
            _train(checkpoint_path)
    else:
        _train(None)


if __name__ == "__main__":
    train_command()  # pylint: disable=no-value-for-parameter
