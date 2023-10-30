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
from typing import Callable, Optional, Union, Type, Tuple
from tqdm import tqdm
import numpy as np
from PIL import Image
from .datasets import load_dataset, Dataset
from .utils import Indices, setup_logging
from .types import Method, CurrentProgress
from . import registry


def make_grid(*images, ncol=None, padding=2, max_width=1920, background=1.):
    if ncol is None:
        ncol = len(images)
    nrow = math.ceil(len(images) / ncol)
    scale_factor = 1
    if max_width is not None:
        scale_factor = min(1, max_width / (ncol * images[0].shape[-2]))
    if scale_factor != 1:
        def interpolate(image):
            img = Image.fromarray(image)
            img = img.resize((int(image.shape[1] * scale_factor), int(image.shape[0] * scale_factor)), Image.Resampling.NEAREST)
            return np.array(img)
        images = list(map(interpolate, images))
    height, width = images[0].shape[0], images[0].shape[1]
    grid = np.ndarray(
        (height * nrow + padding * (nrow - 1), width * ncol + padding * (ncol - 1), images[0].shape[2]),
        dtype=images[0].dtype,
    )
    grid.fill(background)
    for i, image in enumerate(images):
        x = i % ncol
        y = i // ncol
        grid[y * (height + padding):y * (height + padding) + height, x * (width + padding):x * (width + padding) + width] = image
    return grid


def compute_exponential_gamma(num_iters: int, initial_lr: float, final_lr: float) -> float:
    gamma = (math.log(final_lr) - math.log(initial_lr)) / num_iters
    return math.exp(gamma)


def compute_image_metrics(pred, gt):
    mse = ((pred - gt)**2).mean()
    return {
        "mse": mse,
        "psnr": -10 * math.log10(mse),
        "mae": np.abs(pred - gt).mean(),
    }


class Trainer:
    def __init__(self,
                 *,
                 train_dataset: Union[str, Path, Callable[[], Dataset]],
                 test_dataset: Union[None, str, Path, Callable[[], Dataset]] = None,
                 method: Type[Method],
                 output: Path = Path("."),
                 num_iterations: Optional[int] = None,
                 save_iters: Indices = Indices.every_iters(1),# Indices.every_iters(10_000),
                 eval_single_iters: Indices = Indices.every_iters(1_000),
                 eval_all_iters: Indices = Indices([-1]),
                 use_wandb: bool = True,
                 ):
        self.method_name = method.method_name
        self.method = method()
        if isinstance(train_dataset, (Path, str)):
            if test_dataset is None:
                test_dataset = train_dataset
            train_dataset_path  = train_dataset
            train_dataset = partial(load_dataset, Path(train_dataset), split="train")

            # Allow direct access to the stored images if needed for docker remote
            if hasattr(self.method, "mounts"):
                self.method.mounts.append((str(train_dataset_path), str(train_dataset_path)))
        if isinstance(test_dataset, (Path, str)):
            test_dataset = partial(load_dataset, Path(test_dataset), split="test")
        self.train_dataset_fn = train_dataset

        self.test_dataset_fn = test_dataset
        self.test_dataset: Optional[Dataset] = None
        self.test_images = None

        self.step = self.method.info.loaded_step or 0
        self.output = output
        self.num_iterations = num_iterations or self.method.info.num_iterations or 100_000
        self.save_iters = save_iters

        self.eval_single_iters = eval_single_iters
        self.eval_all_iters = eval_all_iters
        self.use_wandb = use_wandb
        for v in vars(self).values():
            if isinstance(v, Indices):
                v.total = self.num_iterations

        self._wandb_run = None
        self._wandb = None
        self._average_image_size = None
        self._installed = False

    def install(self):
        if self._installed:
            return
        if hasattr(self.method, "install"):
            self.method.install()
        self._installed = True

    def setup_data(self):
        logging.info("loading train dataset")
        train_dataset: Dataset = self.train_dataset_fn()
        logging.info("loading eval dataset")
        self.test_dataset: Dataset = self.test_dataset_fn()
        self.test_images = None

        logging.info("loading images")
        train_images, train_dataset.image_sizes = train_dataset.load_images()
        self._average_image_size = train_dataset.image_sizes.prod(-1).astype(np.float32).mean()
        self.test_images, self.test_dataset.image_sizes = self.test_dataset.load_images()
        train_sampling_masks = None
        if train_dataset.sampling_mask_paths is not None:
            train_sampling_masks = train_dataset.load_sampling_masks()

        self.method.setup_train(
            poses=train_dataset.camera_poses,
            intrinsics=train_dataset.camera_intrinsics * train_dataset.image_sizes[..., :1].astype(np.float32),
            sizes=train_dataset.image_sizes,
            nears_fars=train_dataset.nears_fars,
            images=train_images,
            num_iterations=self.num_iterations,
            sampling_masks=train_sampling_masks,
            distortions=train_dataset.camera_distortions,
        )

    def save(self):
        path = os.path.join(self.output, f"checkpoint-{self.step}")
        os.makedirs(os.path.join(self.output, f"checkpoint-{self.step}"), exist_ok=True)
        self.method.save(path)
        with open(os.path.join(path, "nb-info.json"), mode="w+", encoding="utf8") as f:
            json.dump({
                "method": self.method_name,
            }, f)
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
        if self.use_wandb and self._wandb is None:
            import wandb  # pylint: disable=import-outside-toplevel
            self._wandb_run = wandb.init(dir=self.output/"wandb")
            self._wandb = wandb

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
                pbar.set_postfix({
                    "train/loss": f'{metrics["loss"]:.4f}',
                })
                pbar.update(1)

        # Save if not saved by default
        if self.step not in self.save_iters:
            self.save()

    def log_metrics(self, metrics, prefix: str = ""):
        self.ensure_loggers_initialized()
        if self._wandb_run is not None:
            self._wandb_run.log({
                f"{prefix}{k}": v for k, v in metrics.items()
            }, step=self.step)

    def eval_all(self):
        if self.test_dataset is None:
            logging.debug("skipping eval_all, no eval dataset")
            return
        total_rays = 0

        self.ensure_loggers_initialized()
        metrics = {} if self._wandb is not None else None

        # Store predictions, compute metrics, etc.
        prefix = self.test_dataset.file_paths_root
        if prefix is None:
            prefix = Path(os.path.commonpath(self.test_dataset.file_paths))

        output = self.output/f"predictions-{self.step}.tar.gz"
        if output.exists():
            output.unlink()
            logging.warning(f"removed existing predictions at {output}")
        with tarfile.open(output, "w:gz") as tar, \
            tqdm(desc=f"rendering all images at step={self.step}") as pbar:

            def save_image(path, tensor):
                if tensor.dtype != np.uint8:
                    tensor = (tensor * 255.).astype(np.uint8)
                image = Image.fromarray(tensor.numpy())
                with io.BytesIO() as f:
                    image.save(f, format="png")
                    f.seek(0)
                    tar.addfile(tarfile.TarInfo(path), f)

            def update_progress(stat: CurrentProgress):
                nonlocal total_rays
                total_rays = stat.total
                if pbar.total != stat.total:
                    pbar.reset(total=stat.total)
                pbar.update(stat.i - pbar.n)
                pbar.set_postfix({"image": f'{stat.image_i+1}/{stat.image_total}'})

            start = time.perf_counter()
            num_vis_images = 16
            vis_images = []
            predictions = self.method.render(
                poses=self.test_dataset.camera_poses,
                intrinsics=self.test_dataset.camera_intrinsics * self.test_dataset.image_sizes[..., :1].astype(np.float32),
                sizes=self.test_dataset.image_sizes,
                nears_fars=self.test_dataset.nears_fars,
                distortions=self.test_dataset.camera_distortions,
                progress_callback=update_progress,
            )
            for gt, name, pred in zip(self.test_images, self.test_dataset.file_paths, predictions):
                name = str(Path(name).relative_to(prefix).with_suffix(""))
                color_f = pred["color"]
                assert color_f.dtype == np.float32
                color = np.clip(color_f * 255., 0, 255).astype(np.uint8)

                save_image(f"gt/{name}.png", gt)
                save_image(f"color/{name}.png", color)

                if metrics is not None:
                    for k, v in compute_image_metrics(color_f, gt.astype(np.float32) / 255.).items():
                        if k not in metrics:
                            metrics[k] = 0
                        metrics[k] += v
                if len(vis_images) < num_vis_images:
                    vis_images.append((gt, color))
            elapsed = time.perf_counter() - start

        # Log to wandb
        if self._wandb is not None:
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
            self._wandb.log({
                "eval-all-images/color": [
                    self._wandb.Image(color_vis.numpy(), caption="left: gt, right: prediction")
                ],
            }, step=self.step)

    def close(self):
        if self.method is not None and hasattr(self.method, "close"):
            self.method.close()

    def eval_single(self):
        if self.test_dataset is None:
            logging.debug("skipping eval_single, no eval dataset")
            return
        start = time.perf_counter()
        # Pseudo-randomly select an image based on the step
        idx = hashlib.sha1(str(self.step).encode("utf8")).digest()[0] % len(self.test_dataset)
        dataset_slice = self.test_dataset[idx:idx+1]
        total_rays = 0
        with tqdm(desc=f"rendering single image at step={self.step}") as pbar:
            def update_progress(stat: CurrentProgress):
                nonlocal total_rays
                total_rays = stat.total
                if pbar.total != stat.total:
                    pbar.reset(total=stat.total)
                pbar.update(stat.i - pbar.n)
            predictions = next(iter(self.method.render(
                poses=dataset_slice.camera_poses,
                intrinsics=dataset_slice.camera_intrinsics * dataset_slice.image_sizes[..., :1].astype(np.float32),
                sizes=dataset_slice.image_sizes,
                nears_fars=dataset_slice.nears_fars,
                distortions=dataset_slice.camera_distortions,
                progress_callback=update_progress,
            )))
        elapsed = time.perf_counter() - start

        # Log to wandb
        self.ensure_loggers_initialized()
        if self._wandb is not None:
            logging.debug("logging image to wandb")
            metrics = {}
            gt = self.test_images[idx]
            gt_f = gt.astype(np.float32) / 255.
            color_f = predictions["color"]
            metrics = compute_image_metrics(color_f, gt_f)
            image_path = dataset_slice.file_paths[0]
            if dataset_slice.file_paths_root is not None:
                image_path = Path(image_path).relative_to(dataset_slice.file_paths_root)

            color = np.clip(color_f * 255., 0, 255).astype(np.uint8)
            metrics["image-id"] = idx
            metrics["fps"] = 1 / elapsed
            metrics["rays-per-second"] = total_rays / elapsed
            metrics["time"] = elapsed
            self.log_metrics(metrics, "eval-single-image/")

            self._wandb.log({
                "eval-single-image/color": [
                    self._wandb.Image(make_grid(
                        gt,
                        color,
                    ), caption=f"{image_path}: left: gt, right: prediction")
                ],
            }, step=self.step)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(registry.supported_methods()), default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=".")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--backend", choices=registry.ALL_BACKENDS, default=None)
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.method is None and args.checkpoint is None:
        logging.error("Either --method or --checkpoint must be specified")
        sys.exit(1)

    if args.checkpoint is not None:
        with open(os.path.join(args.checkpoint, "nb-info.json"), "r", encoding="utf8") as f:
            info = json.load(f)
        if args.method is not None and args.method != info["method_name"]:
            logging.error(f"Argument --method={args.method} is in conflict with the checkpoint's method {info['method_name']}.")
            sys.exit(1)
        args.method = info["method_name"]

    method_spec = registry.get(args.method)
    method, backend = method_spec.build(
        backend=args.backend,
        checkpoint=os.path.abspath(args.checkpoint) if args.checkpoint else None)
    method.method_name = args.method
    logging.info(f"Using method: {method.method_name}, backend: {backend}")

    trainer = Trainer(
        train_dataset=Path(args.data) if args.data is not None else None,
        output=Path(args.output),
        method=method,
        use_wandb=not args.no_wandb,
    )
    try:
        trainer.install()
        trainer.setup_data()
        trainer.train()
    finally:
        trainer.close()
