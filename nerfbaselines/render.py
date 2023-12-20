import hashlib
import tempfile
from functools import wraps
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
from typing import Any, Iterable
from .datasets import load_dataset, Dataset
from .utils import setup_logging, image_to_srgb, save_image, save_depth, visualize_depth, handle_cli_error
from .types import Method, CurrentProgress, RenderOutput
from . import cameras as _cameras
from . import registry
from . import __version__


def get_checkpoint_sha(path: Path) -> str:
    path = Path(path)
    if str(path).endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tar, tempfile.TemporaryDirectory() as tmpdir:
            tar.extractall(tmpdir)
            return get_checkpoint_sha(Path(tmpdir))

    b = bytearray(128 * 1024)
    mv = memoryview(b)

    files = list(f for f in path.glob("**/*") if f.is_file())
    files.sort()
    sha = hashlib.sha256()
    for f in files:
        if f.name == "nb-info.json":
            continue

        with open(f, "rb", buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                sha.update(mv[:n])
    return sha.hexdigest()


def with_supported_camera_models(supported_camera_models):
    supported_cam_models = set(x.value for x in supported_camera_models)

    def decorator(render):
        @wraps(render)
        def _render(cameras: _cameras.Cameras, *args, **kwargs):
            original_cameras = cameras
            needs_distort = []
            undistorted_cameras_list = []
            for cam in cameras:
                if cam.camera_types.item() not in supported_cam_models:
                    needs_distort.append(True)
                    undistorted_cameras_list.append(_cameras.undistort_camera(cam)[None])
                else:
                    needs_distort.append(False)
                    undistorted_cameras_list.append(cam[None])
            undistorted_cameras = _cameras.Cameras.cat(undistorted_cameras_list)
            for x, distort, cam, ucam in zip(render(undistorted_cameras, *args, **kwargs), needs_distort, original_cameras, undistorted_cameras):
                if not distort:
                    yield x
                else:
                    yield {k: _cameras.warp_image_between_cameras(ucam, cam, v) for k, v in x.items()}

        return _render

    return decorator


def render_all_images(
    method: Method,
    dataset: Dataset,
    output: Path,
    description: str = "rendering all images",
    ns_info: Optional[dict] = None,
) -> Iterable[RenderOutput]:
    info = method.get_info()
    render = with_supported_camera_models(info.supported_camera_models)(method.render)
    allow_transparency = True
    background_color = dataset.metadata.get("background_color", None)
    if ns_info is None:
        ns_info = {}
    else:
        assert dataset.color_space == ns_info.get("color_space", "srgb"), f"Dataset color space {dataset.color_space} != method color space {ns_info['color_space']}"
    color_space = dataset.color_space
    expected_scene_scale = ns_info.get("expected_scene_scale", dataset.expected_scene_scale)

    with tempfile.TemporaryDirectory() as tmpdir:
        method.save(Path(tmpdir))
        checkpoint_sha = get_checkpoint_sha(Path(tmpdir))

    def _predict_all(open_fn):
        assert dataset.images is not None, "dataset must have images loaded"
        with tqdm(desc=description) as pbar:

            def update_progress(progress: CurrentProgress):
                if pbar.total != progress.total:
                    pbar.reset(total=progress.total)
                pbar.set_postfix({"image": f"{min(progress.image_i+1, progress.image_total)}/{progress.image_total}"})
                pbar.update(progress.i - pbar.n)

            predictions = render(dataset.cameras, progress_callback=update_progress)
            for i, (pred, (w, h)) in enumerate(zip(predictions, dataset.cameras.image_sizes)):
                gt_image = image_to_srgb(dataset.images[i][:h, :w], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
                pred_image = image_to_srgb(pred["color"], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
                assert gt_image.shape[:-1] == pred_image.shape[:-1], f"gt size {gt_image.shape[:-1]} != pred size {pred_image.shape[:-1]}"
                relative_name = Path(dataset.file_paths[i])
                if dataset.file_paths_root is not None:
                    if str(relative_name).startswith("/undistorted/"):
                        relative_name = Path(str(relative_name)[len("/undistorted/") :])
                    else:
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
                        save_image(f, dataset.images[i][:h, :w])
                    with open_fn(f"color-linear/{relative_name.with_suffix('.bin')}") as f:
                        save_image(f, pred["color"])
                yield pred

    def write_metadata(open_fn):
        with open_fn("info.json") as fp:
            background_color = dataset.metadata.get("background_color", None)
            if isinstance(background_color, np.ndarray):
                background_color = background_color.tolist()
            fp.write(
                json.dumps(
                    {
                        **ns_info,
                        "nb_version": __version__,
                        "checkpoint_sha256": checkpoint_sha,
                        "dataset_type": dataset.metadata.get("type", None),
                        "dataset_scene": dataset.metadata.get("scene", None),
                        "dataset_background_color": background_color,
                        "expected_scene_scale": round(expected_scene_scale, 5),
                        "color_space": color_space,
                    },
                    indent=4,
                ).encode("utf-8")
            )

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

            write_metadata(open_fn)
            yield from _predict_all(open_fn)
    else:

        def open_fn(path):
            path = output / path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return open(path, "wb")

        write_metadata(open_fn)
        yield from _predict_all(open_fn)


@click.command("render")
@click.option("--checkpoint", type=Path, default=None, required=True)
@click.option("--data", type=str, default=None, required=True)
@click.option("--output", type=Path, default="predictions", help="output directory or tar.gz file")
@click.option("--split", type=str, default="test")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NB_BACKEND", None))
@handle_cli_error
def render_command(checkpoint, data, output, split, verbose, backend):
    setup_logging(verbose)

    # Read method nb-info
    checkpoint = Path(checkpoint)
    assert checkpoint.exists(), f"checkpoint path {checkpoint} does not exist"
    assert (checkpoint / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
    with (checkpoint / "nb-info.json").open("r") as f:
        ns_info = json.load(f)

    method_name = ns_info["method"]
    method_spec = registry.get(method_name)
    method_cls, backend = method_spec.build(backend=backend, checkpoint=Path(os.path.abspath(str(checkpoint))))
    logging.info(f"Using backend: {backend}")

    if hasattr(method_cls, "install"):
        method_cls.install()

    method = method_cls()
    try:
        method_info = method.get_info()
        dataset = load_dataset(data, split=split, features=method_info.required_features)
        dataset.load_features(method_info.required_features)
        if dataset.color_space != ns_info["color_space"]:
            raise RuntimeError(f"Dataset color space {dataset.color_space} != method color space {ns_info['color_space']}")
        for _ in render_all_images(method, dataset, output=Path(output), ns_info=ns_info):
            pass
    finally:
        if hasattr(method, "close"):
            typing.cast(Any, method).close()
