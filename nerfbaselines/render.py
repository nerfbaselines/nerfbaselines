import contextlib
import time
import io
import os
import logging
import tarfile
from typing import Optional
from pathlib import Path
import json
import click
from tqdm import tqdm
import numpy as np
import typing
from typing import Any
from .datasets import load_dataset, Dataset
from .utils import setup_logging, image_to_srgb, save_image, save_depth, visualize_depth
from .types import Method, CurrentProgress
from . import registry


def render_all_images(method: Method, dataset: Dataset, output: Path, color_space: Optional[str] = None, expected_scene_scale: Optional[float] = None, description: str = "rendering all images"):
    allow_transparency = True
    if color_space is None:
        color_space = dataset.color_space
    if expected_scene_scale is None:
        expected_scene_scale = dataset.expected_scene_scale

    def _predict_all(open_fn):
        assert dataset.images is not None, "dataset must have images loaded"
        with tqdm(desc=description) as pbar:

            def update_progress(progress: CurrentProgress):
                if pbar.total != progress.total:
                    pbar.reset(total=progress.total)
                pbar.set_postfix({"image": f"{min(progress.image_i+1, progress.image_total)}/{progress.image_total}"})
                pbar.update(progress.i - pbar.n)

            predictions = method.render(dataset.cameras, progress_callback=update_progress)
            for i, pred in enumerate(predictions):
                gt_image = image_to_srgb(dataset.images[i], np.uint8, color_space=color_space, allow_alpha=allow_transparency)
                pred_image = image_to_srgb(pred["color"], np.uint8, color_space=color_space, allow_alpha=allow_transparency)
                assert gt_image.shape[:-1] == pred_image.shape[:-1], f"gt size {gt_image.shape[:-1]} != pred size {pred_image.shape[:-1]}"
                relative_name = Path(dataset.file_paths[i])
                if dataset.file_paths_root is not None:
                    relative_name = relative_name.relative_to(Path(dataset.file_paths_root))
                with open_fn(f"gt-color/{relative_name.with_suffix('.png')}") as f:
                    save_image(f, gt_image)
                with open_fn(f"color/{relative_name.with_suffix('.png')}") as f:
                    save_image(f, pred_image)
                if "depth" in pred:
                    with open_fn(f"depth/{relative_name.with_suffix('.bin')}") as f:
                        save_depth(f, pred["depth"])
                    depth_rgb = visualize_depth(pred["depth"], near_far=dataset.cameras.nears_fars[i] if dataset.cameras.nears_fars is not None else None, expected_scale=expected_scene_scale)
                    with open_fn(f"depth-rgb/{relative_name.with_suffix('.png')}") as f:
                        save_image(f, depth_rgb)
                if color_space == "linear":
                    # Store the raw linear image as well
                    with open_fn(f"gt-color-linear/{relative_name.with_suffix('.bin')}") as f:
                        save_image(f, dataset.images[i])
                    with open_fn(f"color-linear/{relative_name.with_suffix('.bin')}") as f:
                        save_image(f, pred["color"])
                yield pred

    if str(output).endswith(".tar.gz"):
        with tarfile.open(output, "w:gz") as tar:

            @contextlib.contextmanager
            def open_fn(path):
                rel_path = path
                path = output / path
                tarinfo = tarfile.TarInfo(name=rel_path)
                tarinfo.mtime = int(time.time())
                with io.BytesIO() as f:
                    f.name = path
                    yield f
                    tarinfo.size = f.tell()
                    f.seek(0)
                    tar.addfile(tarinfo=tarinfo, fileobj=f)

            yield from _predict_all(open_fn)
    else:

        def open_fn(path):
            path = output / path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return open(path, "wb")

        yield from _predict_all(open_fn)


@click.command("render")
@click.option("--checkpoint", type=Path, default=None, required=True)
@click.option("--data", type=Path, default=None, required=True)
@click.option("--output", type=Path, default="predictions", help="output directory or tar.gz file")
@click.option("--split", type=str, default="test")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NB_DEFAULT_BACKEND", None))
def render_command(checkpoint, data, output, split, verbose, backend):
    setup_logging(verbose)

    # Read method nb-info
    checkpoint = Path(checkpoint)
    assert checkpoint.exists(), f"checkpoint path {checkpoint} does not exist"
    assert (checkpoint / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
    with (checkpoint / "nb-info.json").open("r") as f:
        ns_info = json.load(f)

    method_spec = registry.get(ns_info["method"])
    method_cls, backend = method_spec.build(backend=backend, checkpoint=os.path.abspath(str(checkpoint)))
    logging.info(f"Using backend: {backend}")

    if hasattr(method_cls, "install"):
        method_cls.install()

    method = method_cls()
    try:
        method_info = method.get_info()
        dataset = load_dataset(Path(data), split=split, features=method_info.required_features)
        dataset.load_features(method_info.required_features)
        if dataset.color_space != ns_info["color_space"]:
            raise RuntimeError(f"Dataset color space {dataset.color_space} != method color space {ns_info['color_space']}")
        for _ in render_all_images(method, dataset, output=Path(output), color_space=dataset.color_space, expected_scene_scale=ns_info["expected_scene_scale"]):
            pass
    finally:
        if hasattr(method, "close"):
            typing.cast(Any, method).close()
