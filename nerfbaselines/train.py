import io
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
from typing import TYPE_CHECKING
from tqdm import tqdm
import numpy as np
from PIL import Image
import click
from .io import open_any_directory
from .datasets import load_dataset, Dataset
from .utils import Indices, setup_logging, partialclass, image_to_srgb, visualize_depth, handle_cli_error
from .utils import remap_error, convert_image_dtype, get_resources_utilization_info
from .types import Method, CurrentProgress, ColorSpace, Literal, FrozenSet
from .render import render_all_images, with_supported_camera_models
from .upload_results import prepare_results_for_upload
from . import __version__
from . import registry
from . import evaluate

if TYPE_CHECKING:
    import wandb.sdk.wandb_run

    _wandb_type = type(wandb)


def make_grid(*images: np.ndarray, ncol=None, padding=2, max_width=1920, background=1.0):
    if ncol is None:
        ncol = len(images)
    nrow = int(math.ceil(len(images) / ncol))
    scale_factor = 1
    height, width = tuple(map(int, np.max([x.shape[:2] for x in images], axis=0).tolist()))
    dtype = images[0].dtype
    if max_width is not None:
        scale_factor = int(min(1, (max_width + ncol * images[0].shape[-2] - 1) // (ncol * images[0].shape[-2])))
    if scale_factor != 1:

        def interpolate(image) -> np.ndarray:
            img = Image.fromarray(image)
            img = img.resize((int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)), Image.Resampling.NEAREST)
            return np.array(img)

        images = tuple(map(interpolate, images))
        height = (height + scale_factor - 1) // scale_factor
        width = (width + scale_factor - 1) // scale_factor
    grid: np.ndarray = np.ndarray(
        (height * nrow + padding * (nrow - 1), width * ncol + padding * (ncol - 1), images[0].shape[2]),
        dtype=dtype,
    )
    background = convert_image_dtype(np.array(background, dtype=np.float32 if isinstance(background, float) else np.uint8), dtype).item()
    grid.fill(background)
    for i, image in enumerate(images):
        x = i % ncol
        y = i // ncol
        h, w = image.shape[:2]
        grid[y * (height + padding) : y * (height + padding) + h, x * (width + padding) : x * (width + padding) + w] = image
    return grid


def compute_exponential_gamma(num_iters: int, initial_lr: float, final_lr: float) -> float:
    gamma = (math.log(final_lr) - math.log(initial_lr)) / num_iters
    return math.exp(gamma)


def method_get_resources_utilization_info(method):
    if hasattr(method, "call"):
        return method.call(f"{get_resources_utilization_info.__module__}.{get_resources_utilization_info.__name__}")
    return get_resources_utilization_info()


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
        eval_single_iters: Indices = Indices.every_iters(2_000),
        eval_all_iters: Indices = Indices([-1]),
        loggers: FrozenSet[Visualization] = {},
        color_space: Optional[ColorSpace] = None,
        run_extra_metrics: bool = False,
        method_name: Optional[str] = None,
        generate_output_artifact: Optional[bool] = None,
        checkpoint: Union[str, Path, None] = None,
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
        self.num_iterations = num_iterations or method_info.num_iterations or 100_000
        self.save_iters = save_iters

        self.eval_single_iters = eval_single_iters
        self.eval_all_iters = eval_all_iters
        self.loggers = loggers
        self.run_extra_metrics = run_extra_metrics
        self.generate_output_artifact = generate_output_artifact
        for v in vars(self).values():
            if isinstance(v, Indices):
                v.total = self.num_iterations + 1
        self._wandb_run: Union["wandb.sdk.wandb_run.Run", None] = None
        self._tensorboard_writer = None
        self._average_image_size = None
        self._color_space = color_space
        self._expected_scene_scale = None
        self._method_info = method_info
        self._total_train_time = 0
        self._resources_utilization_info = None

        # Validate generate output artifact
        if self.generate_output_artifact is None or self.generate_output_artifact:
            messages = self._get_generate_output_artifact_problems()
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

        # Restore checkpoint if specified
        if self.checkpoint is not None:
            with open(self.checkpoint / "nb-info.json", mode="r", encoding="utf8") as f:
                info = json.load(f)
                self._total_train_time = info["total_train_time"]
                self._resources_utilization_info = info["resources_utilization"]

    def _get_generate_output_artifact_problems(self):
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
        return messages

    @remap_error
    def setup_data(self):
        logging.info("loading train dataset")
        train_dataset: Dataset = self._train_dataset_fn()
        logging.info("loading eval dataset")
        self.test_dataset = self._test_dataset_fn()
        method_info = self.method.get_info()

        train_dataset.load_features(method_info.required_features, method_info.supported_camera_models)
        assert train_dataset.cameras.image_sizes is not None, "image sizes must be specified"
        self._average_image_size = train_dataset.cameras.image_sizes.prod(-1).astype(np.float32).mean()
        self._expected_scene_scale = train_dataset.expected_scene_scale

        self.test_dataset.load_features(method_info.required_features.union({"color"}))
        self.method.setup_train(train_dataset, num_iterations=self.num_iterations)

        assert train_dataset.color_space is not None
        if self._color_space is not None and self._color_space != train_dataset.color_space:
            raise RuntimeError(f"train dataset color space {train_dataset.color_space} != {self._color_space}")
        self._color_space = train_dataset.color_space
        if self.test_dataset.color_space != self._color_space:
            raise RuntimeError(f"train dataset color space {self._color_space} != test dataset color space {self.test_dataset.color_space}")
        self._method_info = method_info

    def _get_ns_info(self):
        return {
            "method": self.method_name,
            "nb_version": __version__,
            "color_space": self._color_space,
            "expected_scene_scale": round(self._expected_scene_scale, 5),
            "total_train_time": round(self._total_train_time, 5),
            "resources_utilization": self._resources_utilization_info,
        }

    def save(self):
        path = os.path.join(str(self.output), f"checkpoint-{self.step}")
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

    def ensure_loggers_initialized(self):
        loggers_init = False
        if "wandb" in self.loggers and self._wandb_run is None:
            import wandb  # pylint: disable=import-outside-toplevel

            if not TYPE_CHECKING:
                wandb_run: "wandb.sdk.wandb_run.Run" = wandb.init(dir=self.output)
                self._wandb_run = wandb_run

            loggers_init = True

        if "tensorboard" in self.loggers and self._tensorboard_writer is None:
            from tensorboard.summary.writer.event_file_writer import EventFileWriter

            self._tensorboard_writer = EventFileWriter(str(self.output / "tensorboard"))
            loggers_init = True
        if loggers_init:
            logging.info("initialized loggers: " + ",".join(self.loggers))

    def _update_accumulated_metrics(self, acc_metrics, metrics, n_iters_since_update):
        for k, v in metrics.items():
            assert "_" not in k, "metrics must not contain underscores"
            if k not in acc_metrics:
                acc_metrics[k] = 0
            if k in {"total-train-time", "learning-rate"}:
                acc_metrics[k] = v
            else:
                acc_metrics[k] = (acc_metrics[k] * (n_iters_since_update - 1) + v) / n_iters_since_update

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
        if self._average_image_size is None:
            self.setup_data()
        if self.step == 0 and self.step in self.save_iters:
            self.save()

        # Initialize loggers before training loop for better tqdm output
        self.ensure_loggers_initialized()

        update_frequency = 10
        with tqdm(total=self.num_iterations, initial=self.step, desc="training") as pbar:
            last_update_i = self.step
            acc_metrics = {}
            for i in range(self.step, self.num_iterations):
                self.step = i
                metrics = self.train_iteration()
                # Checkpoint changed, reset sha
                self._checkpoint_sha = None
                self.step = i + 1

                # Update accumulated metrics
                self._update_accumulated_metrics(acc_metrics, metrics, self.step - last_update_i)

                # Update resource utilization info
                self._update_resource_utilization_info()

                # Log metrics and update progress bar
                if self.step % update_frequency == 0 or self.step == self.num_iterations:
                    pbar.set_postfix(
                        {
                            "train/loss": f'{acc_metrics["loss"]:.4f}',
                        }
                    )
                    pbar.update(self.step - last_update_i)
                    last_update_i = self.step
                    self.log_metrics(acc_metrics, "train/")

                # Visualize and save
                if self.step in self.save_iters:
                    self.save()
                if self.step in self.eval_single_iters:
                    self.eval_single()
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

    def log_metrics(self, metrics, prefix: str = ""):
        self.ensure_loggers_initialized()
        if self._wandb_run is not None:
            self._wandb_run.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=self.step)
        if self._tensorboard_writer is not None:
            from tensorboard.compat.proto.summary_pb2 import Summary
            from tensorboard.compat.proto.event_pb2 import Event

            summaries = []
            for k, val in metrics.items():
                summaries.append(Summary.Value(tag=f"{prefix}{k}", simple_value=val))
            summary = Summary(value=summaries)
            self._tensorboard_writer.add_event(Event(summary=summary, step=self.step))

    def eval_all(self):
        if self.test_dataset is None:
            logging.debug("skipping eval_all, no eval dataset")
            return
        assert self.test_dataset.images is not None, "test dataset must have images loaded"
        total_rays = 0

        self.ensure_loggers_initialized()
        metrics: Optional[Dict[str, float]] = {} if self._wandb_run is not None else None

        # Store predictions, compute metrics, etc.
        prefix = self.test_dataset.file_paths_root
        if prefix is None:
            prefix = Path(os.path.commonpath(self.test_dataset.file_paths))

        output = self.output / f"predictions-{self.step}.tar.gz"
        if output.exists():
            output.unlink()
            logging.warning(f"removed existing predictions at {output}")
        output_metrics = self.output / f"results-{self.step}.json"
        if output_metrics.exists():
            output_metrics.unlink()
            logging.warning(f"removed existing results at {output_metrics}")

        start = time.perf_counter()
        num_vis_images = 16
        vis_images: List[Tuple[np.ndarray, np.ndarray]] = []
        vis_depth: List[np.ndarray] = []
        for (i, gt), name, pred, (w, h) in zip(
            enumerate(self.test_dataset.images),
            self.test_dataset.file_paths,
            render_all_images(
                self.method,
                self.test_dataset,
                output=output,
                description=f"rendering all images at step={self.step}",
                ns_info=self._get_ns_info(),
            ),
            self.test_dataset.cameras.image_sizes,
        ):
            name = str(Path(name).relative_to(prefix).with_suffix(""))
            color = pred["color"]
            background_color = self.test_dataset.metadata.get("background_color", None)
            color_srgb = image_to_srgb(color, np.uint8, color_space=self._color_space, background_color=background_color)
            gt_srgb = image_to_srgb(gt[:h, :w], np.uint8, color_space=self._color_space, background_color=background_color)
            if len(vis_images) < num_vis_images:
                vis_images.append((gt_srgb, color_srgb))
                if "depth" in pred:
                    near_far = self.test_dataset.cameras.nears_fars[i] if self.test_dataset.cameras.nears_fars is not None else None
                    vis_depth.append(visualize_depth(pred["depth"], expected_scale=self._expected_scene_scale, near_far=near_far))
        elapsed = time.perf_counter() - start

        # Compute metrics
        info = evaluate.evaluate(output, output_metrics, disable_extra_metrics=not self.run_extra_metrics, description=f"evaluating all images at step={self.step}")
        metrics = info["metrics"]

        if len(self.loggers) > 0:
            assert metrics is not None, "metrics must be computed"
            logging.debug("logging metrics to " + ",".join(self.loggers))
            metrics["fps"] = len(self.test_dataset) / elapsed
            metrics["rays-per-second"] = total_rays / elapsed
            metrics["time"] = elapsed
            self.log_metrics(metrics, "eval-all-images/")

            num_cols = int(math.sqrt(len(vis_images)))

            color_vis = make_grid(
                make_grid(*[x[0] for x in vis_images], ncol=num_cols),
                make_grid(*[x[1] for x in vis_images], ncol=num_cols),
            )

            if self._wandb_run is not None:
                import wandb  # pylint: disable=import-outside-toplevel

                log_image = {
                    "eval-all-images/color": [wandb.Image(color_vis, caption="left: gt, right: prediction")],
                }
                if vis_depth:
                    log_image["eval-all-images/depth"] = [wandb.Image(make_grid(*vis_depth, ncol=num_cols), caption="depth")]
                self._wandb_run.log(log_image, step=self.step)
            if self._tensorboard_writer is not None:
                from tensorboard.compat.proto.summary_pb2 import Summary
                from tensorboard.compat.proto.event_pb2 import Event

                summaries = []
                with io.BytesIO() as simg:
                    Image.fromarray(color_vis).save(simg, format="png")
                    summaries.append(Summary.Value(tag="eval-all-images/color", image=Summary.Image(encoded_image_string=simg.getvalue(), height=color_vis.shape[0], width=color_vis.shape[1])))
                self._tensorboard_writer.add_event(Event(summary=Summary(value=summaries), step=self.step))

    def close(self):
        if self.method is not None and hasattr(self.method, "close"):
            typing.cast(Any, self.method).close()

    def eval_single(self):
        if self.test_dataset is None:
            logging.debug("skipping eval_single, no eval dataset")
            return
        assert self.test_dataset.images is not None, "test dataset must have images loaded"
        start = time.perf_counter()
        # Pseudo-randomly select an image based on the step
        idx = hashlib.sha1(str(self.step).encode("utf8")).digest()[0] % len(self.test_dataset)
        dataset_slice = self.test_dataset[idx : idx + 1]
        total_rays = 0
        with tqdm(desc=f"rendering single image at step={self.step}") as pbar:

            def update_progress(stat: CurrentProgress):
                nonlocal total_rays
                total_rays = stat.total
                if pbar.total != stat.total:
                    pbar.reset(total=stat.total)
                pbar.update(stat.i - pbar.n)

            predictions = next(
                iter(
                    with_supported_camera_models(self._method_info.supported_camera_models)(self.method.render)(
                        cameras=dataset_slice.cameras,
                        progress_callback=update_progress,
                    )
                )
            )
        elapsed = time.perf_counter() - start

        # Log to wandb
        self.ensure_loggers_initialized()
        if len(self.loggers) > 0:
            logging.debug("logging image to " + ",".join(self.loggers))
            metrics = {}
            w, h = self.test_dataset.cameras.image_sizes[idx]
            gt = self.test_dataset.images[idx][:h, :w]
            color = predictions["color"]

            background_color = self.test_dataset.metadata.get("background_color", None)
            color_srgb = image_to_srgb(color, np.uint8, color_space=self._color_space, background_color=background_color)
            gt_srgb = image_to_srgb(gt, np.uint8, color_space=self._color_space, background_color=background_color)
            metrics = evaluate.compute_metrics(color_srgb, gt_srgb, run_extras=self.run_extra_metrics)
            image_path = dataset_slice.file_paths[0]
            if dataset_slice.file_paths_root is not None:
                image_path = str(Path(image_path).relative_to(dataset_slice.file_paths_root))

            metrics["image-id"] = idx
            metrics["fps"] = 1 / elapsed
            metrics["rays-per-second"] = total_rays / elapsed
            metrics["time"] = elapsed

            if "depth" in predictions:
                depth = visualize_depth(predictions["depth"], expected_scale=self._expected_scene_scale)
            self.log_metrics(metrics, "eval-single-image/")

            if self._wandb_run is not None:
                import wandb  # pylint: disable=import-outside-toplevel

                log_image = {
                    "eval-single-image/color": [
                        wandb.Image(
                            make_grid(
                                gt_srgb,
                                color_srgb,
                            ),
                            caption=f"{image_path}: left: gt, right: prediction",
                        )
                    ],
                }
                if "depth" in predictions:
                    log_image["eval-single-image/depth"] = [wandb.Image(depth, caption=f"{image_path}: depth")]
                self._wandb_run.log(log_image, step=self.step)
            if self._tensorboard_writer is not None:
                from tensorboard.compat.proto.summary_pb2 import Summary
                from tensorboard.compat.proto.event_pb2 import Event
                from tensorboard.plugins.image.metadata import create_summary_metadata

                summaries = []
                color_vis = make_grid(gt_srgb, color_srgb)
                metadata = create_summary_metadata(
                    display_name=image_path,
                    description="left: gt, right: prediction",
                )
                with io.BytesIO() as simg:
                    Image.fromarray(color_vis).save(simg, format="png")
                    summaries.append(
                        Summary.Value(tag="eval-single-images/color", metadata=metadata, image=Summary.Image(encoded_image_string=simg.getvalue(), height=color_vis.shape[0], width=color_vis.shape[1]))
                    )
                if "depth" in predictions:
                    metadata = create_summary_metadata(
                        display_name=image_path,
                        description="depth",
                    )
                    with io.BytesIO() as simg:
                        Image.fromarray(depth).save(simg, format="png")
                        summaries.append(
                            Summary.Value(tag="eval-single-images/depth", metadata=metadata, image=Summary.Image(encoded_image_string=simg.getvalue(), height=depth.shape[0], width=depth.shape[1]))
                        )
                self._tensorboard_writer.add_event(Event(summary=Summary(value=summaries), step=self.step))


class IndicesClickType(click.ParamType):
    name = "indices"

    def convert(self, value, param, ctx):
        if value is None:
            return None
        if isinstance(value, Indices):
            return value
        if ":" in value:
            parts = [int(x) if x else None for x in value.split(":")]
            assert len(parts) <= 3, "too many parts in slice"
            return Indices(slice(*parts))
        return Indices([int(x) for x in value.split(",")])


@click.command("train")
@click.option("--method", type=click.Choice(sorted(registry.supported_methods())), required=True, help="Method to use")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--data", type=str, required=True)
@click.option("--output", type=str, default=".")
@click.option("--vis", type=click.Choice(["none", "wandb", "tensorboard", "wandb+tensorboard"]), default="tensorboard", help="Logger to use. Defaults to tensorboard.")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--eval-single-iters", type=IndicesClickType(), default=Indices.every_iters(2_000), help="When to evaluate a single image")
@click.option("--eval-all-iters", type=IndicesClickType(), default=Indices([-1]), help="When to evaluate all images")
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NB_BACKEND", None))
@click.option("--disable-extra-metrics", help="Disable extra metrics which need additional dependencies.", is_flag=True)
@click.option("--disable-output-artifact", "generate_output_artifact", help="Disable producing output artifact containing final model and predictions.", default=None, flag_value=False, is_flag=True)
@click.option("--force-output-artifact", "generate_output_artifact", help="Force producing output artifact containing final model and predictions.", default=None, flag_value=True, is_flag=True)
@handle_cli_error
def train_command(method, checkpoint, data, output, verbose, backend, eval_single_iters, eval_all_iters, num_iterations=None, disable_extra_metrics=False, generate_output_artifact=None, vis="none"):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    if not disable_extra_metrics:
        try:
            evaluate.test_extra_metrics()
        except ImportError as exc:
            logging.error(exc)
            logging.error("Extra metrics are not available and will be disabled. Please install the required dependencies by running `pip install nerfbaselines[extras]`.")
            disable_extra_metrics = True

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

            loggers = set()
            if vis == "wandb":
                loggers.add("wandb")
            elif vis == "tensorboard":
                loggers.add("tensorboard")
            elif vis == "wandb+tensorboard" or vis == "tensorboard+wandb":
                loggers = frozenset({"wandb", "tensorboard"})
            elif vis == "none":
                pass
            else:
                raise ValueError(f"unknown visualization tool {vis}")

            trainer = Trainer(
                train_dataset=_data,
                output=Path(output),
                method=_method,
                eval_all_iters=eval_all_iters,
                eval_single_iters=eval_single_iters,
                loggers=frozenset(loggers),
                num_iterations=num_iterations,
                run_extra_metrics=not disable_extra_metrics,
                generate_output_artifact=generate_output_artifact,
                method_name=method,
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
