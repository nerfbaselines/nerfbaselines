import time
import io
import os
import logging
import tarfile
from pathlib import Path
import json
import click
from tqdm import tqdm
import numpy as np
from PIL import Image
from .datasets import load_dataset, Dataset, dataset_load_features
from .utils import setup_logging, convert_image_dtype
from .types import Method, CurrentProgress
from . import registry


def render_all_images(method: Method, dataset: Dataset, output: Path):
    def _predict_all(save):
        with tqdm("rendering all images") as pbar:
            def update_progress(progress: CurrentProgress):
                if pbar.total != progress.total:
                    pbar.reset(total=progress.total)
                pbar.set_postfix({"image": f'{min(progress.image_i+1, progress.image_total)}/{progress.image_total}'})
                pbar.update(progress.i - pbar.n)

            predictions = method.render(dataset.camera_poses,
                                        dataset.camera_intrinsics,
                                        dataset.image_sizes,
                                        dataset.nears_fars,
                                        dataset.camera_distortions,
                                        progress_callback=update_progress)
            for i, pred in enumerate(predictions):
                gt_image = Image.fromarray(convert_image_dtype(dataset.images[i], np.uint8))
                pred_image = Image.fromarray(convert_image_dtype(pred["color"], np.uint8))
                assert gt_image.size == pred_image.size, f"gt size {gt_image.size} != pred size {pred_image.size}"
                relative_name = Path(dataset.file_paths[i]).relative_to(Path(dataset.file_paths_root))
                save(f"gt-color/{relative_name.with_suffix('.png')}", gt_image)
                save(f"color/{relative_name.with_suffix('.png')}", pred_image)

    if str(output).endswith(".tar.gz"):
        with tarfile.open(output, "w:gz") as tar:
            def write_single(path, image: Image.Image):
                rel_path = path
                path = output / path
                tarinfo = tarfile.TarInfo(name=rel_path)
                tarinfo.mtime = time.time()
                with io.BytesIO() as f:
                    f.name = path
                    image.save(f)
                    tarinfo.size = f.tell()
                    f.seek(0)
                    tar.addfile(tarinfo=tarinfo, fileobj=f)
            _predict_all(write_single)
    else:
        def write_single(path, image):
            path = output / path
            Path(path).mkdir(parents=True, exist_ok=True)
            image.save(path)
        _predict_all(write_single)


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
    method_cls, backend = method_spec.build(
        backend=backend,
        checkpoint=os.path.abspath(str(checkpoint)))
    logging.info(f"Using backend: {backend}")

    method = method_cls()
    try:
        dataset = load_dataset(Path(data), split=split, features=method.info.required_features)
        if dataset.color_space != ns_info["color_space"]:
            raise RuntimeError(f"Dataset color space {dataset.color_space} != method color space {ns_info['color_space']}")
        dataset_load_features(dataset, method.info.required_features)
        render_all_images(method, dataset, output=Path(output))
    finally:
        method.close()
