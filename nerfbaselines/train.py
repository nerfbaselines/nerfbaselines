import struct
import sys
import json
import hashlib
from functools import partial
import time
import io
import os
import math
import logging
import tarfile
from pathlib import Path
import typing
from typing import Callable, Optional, Union, Type, List, Any, Dict, Tuple
from typing import TYPE_CHECKING
from tqdm import tqdm
import numpy as np
from PIL import Image
import click
from .datasets import load_dataset, Dataset
from .utils import Indices, setup_logging, partialclass, convert_image_dtype, linear_to_srgb
from .types import Method, CurrentProgress, ColorSpace

try:
    from typing import Annotated
except ImportError:
    from typing_extensions import Annotated  # type: ignore
from . import registry

if TYPE_CHECKING:
    import wandb.sdk.wandb_run

    _wandb_type = type(wandb)


def make_grid(*images: np.ndarray, ncol=None, padding=2, max_width=1920, background=1.0):
    if ncol is None:
        ncol = len(images)
    nrow = math.ceil(len(images) / ncol)
    scale_factor = 1
    if max_width is not None:
        scale_factor = min(1, max_width / (ncol * images[0].shape[-2]))
    if scale_factor != 1:

        def interpolate(image) -> np.ndarray:
            img = Image.fromarray(image)
            img = img.resize((int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)), Image.Resampling.NEAREST)
            return np.array(img)

        images = tuple(map(interpolate, images))
    height, width = images[0].shape[0], images[0].shape[1]
    grid: np.ndarray = np.ndarray(
        (height * nrow + padding * (nrow - 1), width * ncol + padding * (ncol - 1), images[0].shape[2]),
        dtype=images[0].dtype,
    )
    grid.fill(background)
    for i, image in enumerate(images):
        x = i % ncol
        y = i // ncol
        grid[y * (height + padding) : y * (height + padding) + height, x * (width + padding) : x * (width + padding) + width] = image
    return grid


def compute_exponential_gamma(num_iters: int, initial_lr: float, final_lr: float) -> float:
    gamma = (math.log(final_lr) - math.log(initial_lr)) / num_iters
    return math.exp(gamma)


def compute_image_metrics(pred, gt):
    # NOTE: we blend with black background here!
    pred = pred[..., : gt.shape[-1]]
    pred = convert_image_dtype(pred, np.float32)
    gt = convert_image_dtype(gt, np.float32)
    mse = ((pred - gt) ** 2).mean()
    return {
        "mse": mse,
        "psnr": -10 * math.log10(mse),
        "mae": np.abs(pred - gt).mean(),
    }


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
        use_wandb: bool = True,
        color_space: Optional[ColorSpace] = None,
    ):
        self.method_name = next(iter(getattr(method, "__metadata__", tuple())), None) or method.__name__
        self.method = method()
        self.method.install()
        if isinstance(train_dataset, (Path, str)):
            if test_dataset is None:
                test_dataset = train_dataset
            train_dataset_path = train_dataset
            train_dataset = partial(load_dataset, Path(train_dataset), split="train", features=self.method.info.required_features)

            # Allow direct access to the stored images if needed for docker remote
            if hasattr(self.method, "mounts"):
                typing.cast(Any, self.method).mounts.append((str(train_dataset_path), str(train_dataset_path)))
        if isinstance(test_dataset, (Path, str)):
            test_dataset = partial(load_dataset, Path(test_dataset), split="test", features=self.method.info.required_features)
        assert test_dataset is not None, "test dataset must be specified"
        self._train_dataset_fn: Callable[[], Dataset] = train_dataset
        self._test_dataset_fn: Callable[[], Dataset] = test_dataset
        self.test_dataset: Optional[Dataset] = None

        self.step = self.method.info.loaded_step or 0
        self.output = output
        self.num_iterations = num_iterations or self.method.info.num_iterations or 100_000
        self.save_iters = save_iters

        self.eval_single_iters = eval_single_iters
        self.eval_all_iters = eval_all_iters
        self.use_wandb = use_wandb
        for v in vars(self).values():
            if isinstance(v, Indices):
                v.total = self.num_iterations + 1
        self._wandb_run: Union["wandb.sdk.wandb_run.Run", None] = None
        self._average_image_size = None
        self._installed = False
        self._color_space = color_space

    def install(self):
        if self._installed:
            return
        if hasattr(self.method, "install"):
            self.method.install()
        self._installed = True

    def setup_data(self):
        logging.info("loading train dataset")
        train_dataset: Dataset = self._train_dataset_fn()
        logging.info("loading eval dataset")
        self.test_dataset = self._test_dataset_fn()

        train_dataset.load_features(self.method.info.required_features)
        assert train_dataset.cameras.image_sizes is not None, "image sizes must be specified"
        self._average_image_size = train_dataset.cameras.image_sizes.prod(-1).astype(np.float32).mean()

        self.test_dataset.load_features(self.method.info.required_features.union({"images"}))
        self.method.setup_train(train_dataset, num_iterations=self.num_iterations)

        assert train_dataset.color_space is not None
        if self._color_space is not None and self._color_space != train_dataset.color_space:
            raise RuntimeError(f"train dataset color space {train_dataset.color_space} != {self._color_space}")
        self._color_space = train_dataset.color_space
        if self.test_dataset.color_space != self._color_space:
            raise RuntimeError(f"train dataset color space {self._color_space} != test dataset color space {self.test_dataset.color_space}")

    def save(self):
        path = os.path.join(str(self.output), f"checkpoint-{self.step}")
        os.makedirs(os.path.join(str(self.output), f"checkpoint-{self.step}"), exist_ok=True)
        self.method.save(path)
        with open(os.path.join(path, "nb-info.json"), mode="w+", encoding="utf8") as f:
            json.dump(
                {
                    "method": self.method_name,
                    "color_space": self._color_space,
                },
                f,
            )
        logging.info(f"checkpoint saved at step={self.step}")

    def train_iteration(self):
        start = time.perf_counter()
        metrics = self.method.train_iteration(self.step)
        elapsed = time.perf_counter() - start

        metrics["time"] = elapsed
        if "num_rays" in metrics:
            batch_size = metrics.pop("num_rays")
            metrics["rays-per-second"] = batch_size / elapsed
            if self._average_image_size is not None:
                metrics["fps"] = batch_size / elapsed / self._average_image_size
        return metrics

    def ensure_loggers_initialized(self):
        if self.use_wandb and self._wandb_run is None:
            import wandb  # pylint: disable=import-outside-toplevel

            if not TYPE_CHECKING:
                wandb_run: "wandb.sdk.wandb_run.Run" = wandb.init(dir=self.output)
                self._wandb_run = wandb_run

    def train(self):
        if self._average_image_size is None:
            self.setup_data()
        if self.step == 0 and self.step in self.save_iters:
            self.save()

        with tqdm(total=self.num_iterations, initial=self.step, desc="training") as pbar:
            for i in range(self.step, self.num_iterations):
                self.step = i
                metrics = self.train_iteration()
                self.log_metrics(metrics, "train/")
                self.step = i + 1

                # Visualize and save
                if self.step in self.save_iters:
                    self.save()
                if self.step in self.eval_single_iters:
                    self.eval_single()
                if self.step in self.eval_all_iters:
                    self.eval_all()
                pbar.set_postfix(
                    {
                        "train/loss": f'{metrics["loss"]:.4f}',
                    }
                )
                pbar.update(1)

        # Save if not saved by default
        if self.step not in self.save_iters:
            self.save()

    def log_metrics(self, metrics, prefix: str = ""):
        self.ensure_loggers_initialized()
        if self._wandb_run is not None:
            self._wandb_run.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=self.step)

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
        with tarfile.open(output, "w:gz") as tar, tqdm(desc=f"rendering all images at step={self.step}") as pbar:

            def _save_image(path, tensor):
                with io.BytesIO() as f:
                    if str(path).endswith(".bin"):
                        if tensor.shape[2] < 4:
                            tensor = np.dstack((tensor, np.ones([tensor.shape[0], tensor.shape[1], 4 - tensor.shape[2]])))
                        f.write(struct.pack("ii", tensor.shape[0], tensor.shape[1]))
                        f.write(tensor.astype(np.float16).tobytes())
                    else:
                        tensor = convert_image_dtype(tensor, np.uint8)
                        image = Image.fromarray(tensor)
                        image.save(f, format="png")
                    f.seek(0)
                    tar.addfile(tarfile.TarInfo(path), f)

            def update_progress(stat: CurrentProgress):
                nonlocal total_rays
                total_rays = stat.total
                if pbar.total != stat.total:
                    pbar.reset(total=stat.total)
                pbar.update(stat.i - pbar.n)
                pbar.set_postfix({"image": f"{stat.image_i+1}/{stat.image_total}"})

            start = time.perf_counter()
            num_vis_images = 16
            vis_images: List[Tuple[np.ndarray, np.ndarray]] = []
            predictions = self.method.render(cameras=self.test_dataset.cameras, progress_callback=update_progress)
            for gt, name, pred in zip(self.test_dataset.images, self.test_dataset.file_paths, predictions):
                name = str(Path(name).relative_to(prefix).with_suffix(""))
                color = pred["color"]

                color_srgb = self._get_srgb(color, np.uint8)
                gt_srgb = self._get_srgb(gt, np.uint8)

                _save_image(f"gt/{name}.png", gt_srgb)
                _save_image(f"color/{name}.png", color_srgb)
                if self._color_space == "linear":
                    _save_image(f"gt/{name}.bin", gt)
                    _save_image(f"color/{name}.bin", color)

                if metrics is not None:
                    for k, v in compute_image_metrics(color_srgb, gt_srgb).items():
                        if k not in metrics:
                            metrics[k] = 0
                        metrics[k] += v
                if len(vis_images) < num_vis_images:
                    vis_images.append((gt_srgb, color_srgb))
            elapsed = time.perf_counter() - start

        # Log to wandb
        if self._wandb_run is not None:
            assert metrics is not None, "metrics must be computed"
            import wandb  # pylint: disable=import-outside-toplevel

            logging.debug("logging images to wandb")
            metrics["fps"] = len(self.test_dataset) / elapsed
            metrics["rays-per-second"] = total_rays / elapsed
            metrics["time"] = elapsed
            self.log_metrics(metrics, "eval-all-images/")
            num_cols = int(math.sqrt(len(vis_images)))

            color_vis = make_grid(
                make_grid(*[x[0] for x in vis_images], ncol=num_cols),
                make_grid(*[x[1] for x in vis_images], ncol=num_cols),
            )
            self._wandb_run.log(
                {
                    "eval-all-images/color": [wandb.Image(color_vis, caption="left: gt, right: prediction")],
                },
                step=self.step,
            )

    def close(self):
        if self.method is not None and hasattr(self.method, "close"):
            typing.cast(Any, self.method).close()

    def _get_srgb(self, tensor, dtype):
        # NOTE: here we blend with black background
        tensor = tensor[..., :3]

        if self._color_space == "linear":
            tensor = convert_image_dtype(tensor, np.float32)
            tensor = linear_to_srgb(tensor)

        tensor = convert_image_dtype(tensor, np.uint8)
        tensor = convert_image_dtype(tensor, dtype)
        return tensor

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
                    self.method.render(
                        cameras=dataset_slice.cameras,
                        progress_callback=update_progress,
                    )
                )
            )
        elapsed = time.perf_counter() - start

        # Log to wandb
        self.ensure_loggers_initialized()
        if self._wandb_run is not None:
            import wandb  # pylint: disable=import-outside-toplevel

            logging.debug("logging image to wandb")
            metrics = {}
            gt = self.test_dataset.images[idx]
            color = predictions["color"]

            color_srgb = self._get_srgb(color, np.uint8)
            gt_srgb = self._get_srgb(gt, np.uint8)
            metrics = compute_image_metrics(color_srgb, gt_srgb)
            image_path = dataset_slice.file_paths[0]
            if dataset_slice.file_paths_root is not None:
                image_path = str(Path(image_path).relative_to(dataset_slice.file_paths_root))

            metrics["image-id"] = idx
            metrics["fps"] = 1 / elapsed
            metrics["rays-per-second"] = total_rays / elapsed
            metrics["time"] = elapsed
            self.log_metrics(metrics, "eval-single-image/")
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
                # TODO: map depth to RGB
                log_image["eval-single-image/depth"] = [wandb.Image(predictions["depth"], caption=f"{image_path}: depth")]

            self._wandb_run.log(log_image, step=self.step)


@click.command("train")
@click.option("--method", type=click.Choice(sorted(registry.supported_methods())), required=True, help="Method to use")
@click.option("--checkpoint", type=str, default=None)
@click.option("--data", type=str, required=True)
@click.option("--output", type=str, default=".")
@click.option("--no-wandb", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NB_DEFAULT_BACKEND", None))
def train_command(method, checkpoint, data, output, no_wandb, verbose, backend):
    logging.basicConfig(level=logging.INFO)
    setup_logging(verbose)

    # Make paths absolute, and change working directory to output
    data = os.path.abspath(data)
    output = os.path.abspath(output)
    os.chdir(output)

    if method is None and checkpoint is None:
        logging.error("Either --method or --checkpoint must be specified")
        sys.exit(1)

    if checkpoint is not None:
        with open(os.path.join(checkpoint, "nb-info.json"), "r", encoding="utf8") as f:
            info = json.load(f)
        if method is not None and method != info["method_name"]:
            logging.error(f"Argument --method={method} is in conflict with the checkpoint's method {info['method_name']}.")
            sys.exit(1)
        method = info["method_name"]

    method_spec = registry.get(method)
    _method, backend = method_spec.build(backend=backend, checkpoint=os.path.abspath(checkpoint) if checkpoint else None)
    logging.info(f"Using method: {method}, backend: {backend}")

    # Enable direct memory access to images and if supported by the backend
    if backend in {"docker", "apptainer"}:
        _method = partialclass(_method, mounts=[(data, data)])

    trainer = Trainer(
        train_dataset=Path(data),
        output=Path(output),
        method=Annotated[_method, method],
        use_wandb=not no_wandb,
    )
    try:
        trainer.setup_data()
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    train_command()  # pylint: disable=no-value-for-parameter
