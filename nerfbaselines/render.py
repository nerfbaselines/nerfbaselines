import hashlib
import tempfile
from functools import wraps
import contextlib
import time
import io
import os
import logging
import tarfile
from typing import Optional, Union, TypeVar
from pathlib import Path
import json
import click
from tqdm import tqdm
import numpy as np
import typing
from typing import Any, Iterable, cast
from .datasets import load_dataset, Dataset
from .utils import setup_logging, image_to_srgb, save_image, save_depth, visualize_depth, handle_cli_error, convert_image_dtype, assert_not_none
from .types import Method, CurrentProgress, RenderOutput, EvaluationProtocol
from .io import open_any_directory, serialize_nb_info, deserialize_nb_info
from . import cameras as _cameras
from . import registry
from . import __version__


TRender = TypeVar("TRender", bound=typing.Callable[..., Iterable[RenderOutput]])


def build_update_progress(pbar: tqdm, simple=False):
    old_image_i = -1

    def update_progress(progress: CurrentProgress):
        nonlocal old_image_i

        report_update = False
        if pbar.total != progress.total:
            pbar.reset(total=progress.total)
            report_update = True
        if progress.image_i != old_image_i:
            report_update = True
            old_image_i = progress.image_i
        elif progress.i % 10 == 0:
            report_update = True

        if report_update:
            if not simple:
                pbar.set_postfix({"image": f"{min(progress.image_i+1, progress.image_total)}/{progress.image_total}"})
            pbar.update(progress.i - pbar.n)
    return update_progress


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

        with open(f, "rb", buffering=0) as fio:
            for n in iter(lambda: fio.readinto(mv), 0):
                sha.update(mv[:n])
    return sha.hexdigest()


def get_method_sha(method: Method) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        method.save(Path(tmpdir))
        return get_checkpoint_sha(Path(tmpdir))


def with_supported_camera_models(supported_camera_models):
    supported_cam_models = set(x.value for x in supported_camera_models)

    def decorator(render: TRender) -> TRender:
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
                if distort:
                    x = cast(RenderOutput, 
                             {k: _cameras.warp_image_between_cameras(ucam, cam, cast(np.ndarray, v)) for k, v in x.items()})
                yield x
        return cast(TRender, _render)

    return decorator


def store_predictions(output: Path, predictions: Iterable[RenderOutput], dataset: Dataset, *, nb_info=None) -> Iterable[RenderOutput]:
    background_color =  dataset.metadata.get("background_color", None)
    assert background_color is None or background_color.dtype == np.uint8, "background_color must be None or uint8"
    color_space = dataset.color_space
    expected_scene_scale = dataset.expected_scene_scale
    allow_transparency = True

    def _predict_all(open_fn) -> Iterable[RenderOutput]:
        assert dataset.images is not None, "dataset must have images loaded"
        for i, (pred, (w, h)) in enumerate(zip(predictions, assert_not_none(dataset.cameras.image_sizes))):
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
                    serialize_nb_info(
                        {
                            **(nb_info or {}),
                            "nb_version": __version__,
                            "dataset_metadata": dataset.metadata,
                        }),
                    indent=4,
                ).encode("utf-8")
            )

    if str(output).endswith(".tar.gz"):
        with tarfile.open(output, "w:gz") as tar:

            @contextlib.contextmanager
            def open_fn_tar(path):
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

            write_metadata(open_fn_tar)
            yield from _predict_all(open_fn_tar)
    else:

        def open_fn_fs(path):
            path = output / path
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return open(path, "wb")

        write_metadata(open_fn_fs)
        yield from _predict_all(open_fn_fs)


def render_all_images(
    method: Method,
    dataset: Dataset,
    output: Path,
    description: str = "rendering all images",
    nb_info: Optional[dict] = None,
    evaluation_protocol: Optional[EvaluationProtocol] = None,
) -> Iterable[RenderOutput]:
    if evaluation_protocol is None:
        from .evaluate import get_evaluation_protocol

        evaluation_protocol = get_evaluation_protocol(dataset_name=dataset.metadata.get("name"))
    background_color =  dataset.metadata.get("background_color", None)
    if background_color is not None:
        background_color = convert_image_dtype(background_color, np.uint8)
    if nb_info is None:
        nb_info = {}
    else:
        nb_info = nb_info.copy()
        assert dataset.color_space == nb_info.get("color_space", "srgb"), \
            f"Dataset color space {dataset.color_space} != method color space {nb_info['color_space']}"
        if "dataset_background_color" in nb_info:
            info_background_color = nb_info.get("dataset_background_color")
            if info_background_color is not None:
                info_background_color = np.array(info_background_color, np.uint8)
            assert info_background_color is None or (background_color is not None and np.array_equal(info_background_color, background_color)), \
                f"Dataset background color {background_color} != method background color {info_background_color}"
    nb_info["checkpoint_sha256"] = get_method_sha(method)
    nb_info["evaluation_protocol"] = evaluation_protocol.get_name()

    with tqdm(desc=description) as pbar:
        yield from store_predictions(
            output,
            evaluation_protocol.render(method, dataset, progress_callback=build_update_progress(pbar)),
            dataset=dataset,
            nb_info=nb_info)


@click.command("render")
@click.option("--checkpoint", type=str, default=None, required=True)
@click.option("--data", type=str, default=None, required=True)
@click.option("--output", type=Path, default="predictions", help="output directory or tar.gz file")
@click.option("--split", type=str, default="test")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@handle_cli_error
def render_command(checkpoint: Union[str, Path], data, output, split, verbose, backend):
    checkpoint = str(checkpoint)
    setup_logging(verbose)

    # Read method nb-info
    with open_any_directory(checkpoint, mode="r") as checkpoint_path:
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
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
            if dataset.color_space != nb_info["color_space"]:
                raise RuntimeError(f"Dataset color space {dataset.color_space} != method color space {nb_info['color_space']}")
            for _ in render_all_images(method, dataset, output=Path(output), nb_info=nb_info):
                pass
        finally:
            if hasattr(method, "close"):
                typing.cast(Any, method).close()