import logging
import hashlib
import tempfile
from functools import wraps
import contextlib
import time
import io
import os
import tarfile
from typing import Optional, TypeVar
from pathlib import Path
import json
import click
from tqdm import tqdm
import numpy as np
import typing
from typing import Iterable, cast, List
from .datasets import load_dataset, Dataset
from .utils import setup_logging, image_to_srgb, save_image, save_depth, visualize_depth, handle_cli_error, convert_image_dtype, assert_not_none
from .types import Method, RenderOutput, EvaluationProtocol, Cameras
from .io import open_any_directory, serialize_nb_info, deserialize_nb_info
from . import backends
from . import cameras as _cameras
from . import registry
from . import __version__


TRender = TypeVar("TRender", bound=typing.Callable[..., Iterable[RenderOutput]])


def get_checkpoint_sha(path: str) -> str:
    if path.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tar, tempfile.TemporaryDirectory() as tmpdir:
            tar.extractall(tmpdir)
            return get_checkpoint_sha(tmpdir)

    b = bytearray(128 * 1024)
    mv = memoryview(b)

    files = list(f for f in Path(path).glob("**/*") if f.is_file())
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
        method.save(tmpdir)
        return get_checkpoint_sha(tmpdir)


def with_supported_camera_models(supported_camera_models):
    supported_cam_models = set(_cameras.camera_model_to_int(x) for x in supported_camera_models)

    def decorator(render: TRender) -> TRender:
        @wraps(render)
        def _render(cameras: Cameras, *args, **kwargs):
            assert len(cameras) > 0, "No cameras"
            original_cameras = cameras
            needs_distort = []
            undistorted_cameras_list: List[Cameras] = []
            for cam in cameras:
                if cam.camera_types.item() not in supported_cam_models:
                    needs_distort.append(True)
                    undistorted_cameras_list.append(_cameras.undistort_camera(cam)[None])
                else:
                    needs_distort.append(False)
                    undistorted_cameras_list.append(cam[None])
            undistorted_cameras = undistorted_cameras_list[0].cat(undistorted_cameras_list)
            for x, distort, cam, ucam in zip(render(undistorted_cameras, *args, **kwargs), needs_distort, original_cameras, undistorted_cameras):
                if distort:
                    x = cast(RenderOutput, 
                             {k: _cameras.warp_image_between_cameras(ucam, cam, cast(np.ndarray, v)) for k, v in x.items()})
                yield x
        return cast(TRender, _render)

    return decorator


def store_predictions(output: str, predictions: Iterable[RenderOutput], dataset: Dataset, *, nb_info=None) -> Iterable[RenderOutput]:
    background_color =  dataset["metadata"].get("background_color", None)
    assert background_color is None or background_color.dtype == np.uint8, "background_color must be None or uint8"
    color_space = dataset["metadata"]["color_space"]
    expected_scene_scale = dataset["metadata"].get("expected_scene_scale")
    allow_transparency = True

    def _predict_all(open_fn) -> Iterable[RenderOutput]:
        for i, (pred, (w, h)) in enumerate(zip(predictions, assert_not_none(dataset["cameras"].image_sizes))):
            gt_image = image_to_srgb(dataset["images"][i][:h, :w], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
            pred_image = image_to_srgb(pred["color"], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
            assert gt_image.shape[:-1] == pred_image.shape[:-1], f"gt size {gt_image.shape[:-1]} != pred size {pred_image.shape[:-1]}"
            relative_name = Path(dataset["file_paths"][i])
            if dataset["file_paths_root"] is not None:
                if str(relative_name).startswith("/undistorted/"):
                    relative_name = Path(str(relative_name)[len("/undistorted/") :])
                else:
                    relative_name = relative_name.relative_to(Path(dataset["file_paths_root"]))
            with open_fn(f"gt-color/{relative_name.with_suffix('.png')}") as f:
                save_image(f, gt_image)
            with open_fn(f"color/{relative_name.with_suffix('.png')}") as f:
                save_image(f, pred_image)
            if "depth" in pred:
                with open_fn(f"depth/{relative_name.with_suffix('.bin')}") as f:
                    save_depth(f, pred["depth"])
                depth_rgb = visualize_depth(pred["depth"], near_far=dataset["cameras"].nears_fars[i] if dataset["cameras"].nears_fars is not None else None, expected_scale=expected_scene_scale)
                with open_fn(f"depth-rgb/{relative_name.with_suffix('.png')}") as f:
                    save_image(f, depth_rgb)
            if color_space == "linear":
                # Store the raw linear image as well
                with open_fn(f"gt-color-linear/{relative_name.with_suffix('.bin')}") as f:
                    save_image(f, dataset["images"][i][:h, :w])
                with open_fn(f"color-linear/{relative_name.with_suffix('.bin')}") as f:
                    save_image(f, pred["color"])
            yield pred

    def write_metadata(open_fn):
        with open_fn("info.json") as fp:
            background_color = dataset["metadata"].get("background_color", None)
            if isinstance(background_color, np.ndarray):
                background_color = background_color.tolist()
            fp.write(
                json.dumps(
                    serialize_nb_info(
                        {
                            **(nb_info or {}),
                            "nb_version": __version__,
                            "dataset_metadata": dataset["metadata"],
                        }),
                    indent=4,
                ).encode("utf-8")
            )

    if str(output).endswith(".tar.gz"):
        with tarfile.open(output, "w:gz") as tar:

            @contextlib.contextmanager
            def open_fn_tar(path):
                rel_path = path
                path = os.path.join(output, path)
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
            path = os.path.join(output, path)
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            return open(path, "wb")

        write_metadata(open_fn_fs)
        yield from _predict_all(open_fn_fs)


def render_all_images(
    method: Method,
    dataset: Dataset,
    output: str,
    description: str = "rendering all images",
    nb_info: Optional[dict] = None,
    evaluation_protocol: Optional[EvaluationProtocol] = None,
) -> Iterable[RenderOutput]:
    if evaluation_protocol is None:
        from .evaluate import get_evaluation_protocol

        evaluation_protocol = get_evaluation_protocol(dataset_name=dataset["metadata"].get("name"))
    logging.info(f"Rendering images with evaluation protocol {evaluation_protocol.get_name()}")
    background_color =  dataset["metadata"].get("background_color", None)
    if background_color is not None:
        background_color = convert_image_dtype(background_color, np.uint8)
    if nb_info is None:
        nb_info = {}
    else:
        nb_info = nb_info.copy()
        dataset_colorspace = dataset["metadata"].get("color_space", "srgb")
        assert dataset_colorspace == nb_info.get("color_space", "srgb"), \
            f"Dataset color space {dataset_colorspace} != method color space {nb_info['color_space']}"
        if "dataset_background_color" in nb_info:
            info_background_color = nb_info.get("dataset_background_color")
            if info_background_color is not None:
                info_background_color = np.array(info_background_color, np.uint8)
            assert info_background_color is None or (background_color is not None and np.array_equal(info_background_color, background_color)), \
                f"Dataset background color {background_color} != method background color {info_background_color}"
    nb_info["checkpoint_sha256"] = get_method_sha(method)
    nb_info["evaluation_protocol"] = evaluation_protocol.get_name()

    iterator = store_predictions(
        output,
        evaluation_protocol.render(method, dataset),
        dataset=dataset,
        nb_info=nb_info)
    yield from tqdm(iterator, desc=description, total=len(dataset["file_paths"]), dynamic_ncols=True)

@click.command("render")
@click.option("--checkpoint", type=str, default=None, required=True)
@click.option("--data", type=str, default=None, required=True)
@click.option("--output", type=str, default="predictions", help="output directory or tar.gz file")
@click.option("--split", type=str, default="test")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@handle_cli_error
def render_command(checkpoint: str, data: str, output: str, split: str, verbose: bool, backend_name):
    checkpoint = str(checkpoint)
    setup_logging(verbose)

    # Read method nb-info
    with open_any_directory(checkpoint, mode="r") as _checkpoint_path:
        checkpoint_path = Path(_checkpoint_path)
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
        backends.mount(checkpoint_path, checkpoint_path)
        with registry.build_method(method_name, backend=backend_name) as method_cls:
            method: Method = method_cls(checkpoint=str(checkpoint_path))
            method_info = method.get_info()
            dataset = load_dataset(data, 
                                   split=split, 
                                   features=method_info.get("required_features", None), 
                                   supported_camera_models=method_info.get("supported_camera_models", None))
            dataset_colorspace = dataset["metadata"].get("color_space", "srgb")
            if dataset_colorspace != nb_info.get("color_space", "srgb"):
                raise RuntimeError(f"Dataset color space {dataset_colorspace} != method color space {nb_info.get('color_space', 'srgb')}")
            for _ in render_all_images(method, dataset, output=output, nb_info=nb_info):
                pass
