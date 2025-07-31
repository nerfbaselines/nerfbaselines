import importlib
from contextlib import contextmanager
import zipfile
import tarfile
import time
import io
from functools import wraps
import logging
import os
import typing
from typing import Dict, Union, Iterable, TypeVar, Optional, cast, List, Tuple, BinaryIO, Any
import numpy as np
import json
from pathlib import Path

from tqdm import tqdm


import nerfbaselines
from .utils import (
    apply_colormap,
    image_to_srgb,
    visualize_depth,
    convert_image_dtype, 
)
from .backends import run_on_host, zero_copy
from . import (
    Dataset,
    RenderOutput, 
    EvaluationProtocol, 
    Cameras,
    Method,
    Trajectory,
    RenderOptions,
    camera_model_to_int,
    new_cameras,
    new_dataset,
)
from .io import (
    open_any_directory, 
    deserialize_nb_info, 
    get_predictions_sha,
    get_method_sha,
    save_evaluation_results,
    _save_predictions_iterate,
    save_image,
    read_image, 
    read_mask,
    save_mask,
)
from .datasets import dataset_index_select
from . import metrics
from . import cameras as _cameras
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
try:
    from typeguard import suppress_type_checks  # type: ignore
except ImportError:
    from contextlib import nullcontext as suppress_type_checks


OutputType = Literal["color", "depth"]
T = TypeVar("T")


def _assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


def _import_type(name: str) -> Any:
    package, name = name.split(":")
    obj: Any = importlib.import_module(package)
    for p in name.split("."):
        obj = getattr(obj, p)
    return obj


def build_evaluation_protocol(id: str) -> 'EvaluationProtocol':
    spec = nerfbaselines.get_evaluation_protocol_spec(id)
    if spec is None:
        raise RuntimeError(f"Could not find evaluation protocol {id} in registry. Supported protocols: {','.join(nerfbaselines.get_supported_evaluation_protocols())}")
    return cast('EvaluationProtocol', _import_type(spec["evaluation_protocol_class"])())



@typing.overload
def compute_metrics(pred: np.ndarray, gt: np.ndarray, *, mask=None, reduce: Literal[True] = True, run_lpips_vgg: bool = ...) -> Dict[str, float]:
    ...


@typing.overload
def compute_metrics(pred: np.ndarray, gt: np.ndarray, *, mask=None, reduce: Literal[False], run_lpips_vgg: bool = ...) -> Dict[str, np.ndarray]:
    ...


@run_on_host()
def compute_metrics(pred, gt, *, mask=None, reduce: bool = True, run_lpips_vgg: bool = False):
    def reduction(x):
        if reduce:
            return x.mean().item()
        else:
            return x

    pred = pred[..., : gt.shape[-1]]
    pred = convert_image_dtype(pred, np.float32)
    gt = convert_image_dtype(gt, np.float32)
    if mask is not None:
        mask = convert_image_dtype(mask, np.float32)[..., None]
        pred = pred * mask
        gt = gt * mask
    mse = metrics.mse(pred, gt)
    out = {
        "psnr": reduction(metrics.psnr(mse)),
        "ssim": reduction(metrics.ssim(gt, pred)),
        "mae": reduction(metrics.mae(gt, pred)),
        "mse": reduction(mse),
        "lpips": reduction(metrics.lpips(gt, pred)),
    }
    if run_lpips_vgg:
        out["lpips_vgg"] = reduction(metrics.lpips_vgg(gt, pred))
    return out


def path_is_video(path: str) -> bool:
    return (path.endswith(".mp4") or 
            path.endswith(".avi") or 
            path.endswith(".gif") or 
            path.endswith(".webp") or
            path.endswith(".mov"))


def evaluate(predictions: str, 
             output: str, 
             description: str = "evaluating", 
             evaluation_protocol: Optional[EvaluationProtocol] = None):
    """
    Evaluate a set of predictions.

    Args:
        predictions: Path to a directory containing the predictions.
        output: Path to a json file where the results will be written.
        description: Description of the evaluation, used for progress bar.
        evaluation_protocol: The evaluation protocol to use. If None, the protocol from info.json will be used.
    Returns:
        A dictionary containing the results.
    """
    if os.path.exists(output):
        raise FileExistsError(f"{output} already exists")

    with open_any_directory(predictions, "r") as _predictions_path:
        predictions_path = Path(_predictions_path)
        with open(predictions_path / "info.json", "r", encoding="utf8") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        if evaluation_protocol is None:
            evaluation_protocol = build_evaluation_protocol(nb_info["evaluation_protocol"])
        logging.info(f"Using evaluation protocol {evaluation_protocol.get_name()}")

        # Run the evaluation
        metrics_lists = {}
        relpaths = [str(x.relative_to(predictions_path / "color")) for x in (predictions_path / "color").glob("**/*") if x.is_file()]
        relpaths.sort()

        def read_predictions() -> Iterable[RenderOutput]:
            # Load the prediction
            for relname in relpaths:
                yield {
                    "color": read_image(predictions_path / "color" / relname)
                }

        gt_image_paths = [
            str(predictions_path / "gt-color" / x) for x in relpaths
        ]
        gt_images = list(map(read_image, gt_image_paths))
        if (predictions_path / "mask").exists():
            gt_masks_root = str(predictions_path / "mask")
            gt_mask_paths = [
                str(Path(gt_masks_root).joinpath(x).with_suffix(".png")) for x in relpaths
            ]
            gt_masks = list(map(read_mask, gt_mask_paths))
        else:
            gt_masks_root = None
            gt_masks = None
            gt_mask_paths = None
        with suppress_type_checks():
            from pprint import pprint
            pprint(nb_info)
            dataset = new_dataset(
                cameras=typing.cast(Cameras, None),
                image_paths=relpaths,
                image_paths_root=str(predictions_path / "gt-color"),
                mask_paths=gt_mask_paths,
                mask_paths_root=gt_masks_root,
                metadata=typing.cast(Dict, nb_info.get("render_dataset_metadata", nb_info.get("dataset_metadata", {}))),
                images=gt_images,
                masks=gt_masks)

            # Evaluate the prediction
            with tqdm(desc=description, dynamic_ncols=True, total=len(relpaths)) as progress:
                def collect_metrics_lists():
                    for i, pred in enumerate(read_predictions()):
                        dataset_slice = dataset_index_select(dataset, [i])
                        metrics = evaluation_protocol.evaluate(pred, dataset_slice)
                        for k, v in metrics.items():
                            if k not in metrics_lists:
                                metrics_lists[k] = []
                            metrics_lists[k].append(v)
                        progress.update(1)
                        if "psnr" in metrics_lists:
                            psnr_val = np.mean(metrics_lists["psnr"][-1])
                            progress.set_postfix(psnr=f"{psnr_val:.4f}")
                        yield metrics

                metrics = evaluation_protocol.accumulate_metrics(collect_metrics_lists())


        predictions_sha, ground_truth_sha = get_predictions_sha(str(predictions_path))

        # If output is specified, write the results to a file
        if os.path.exists(str(output)):
            raise FileExistsError(f"{output} already exists")

        out = save_evaluation_results(str(output),
                                      metrics=metrics, 
                                      metrics_lists=metrics_lists, 
                                      predictions_sha=predictions_sha,
                                      ground_truth_sha=ground_truth_sha,
                                      evaluation_protocol=evaluation_protocol.get_name(),
                                      nb_info=nb_info)
        return out


class DefaultEvaluationProtocol(EvaluationProtocol):
    _name = "default"
    _lpips_vgg = False

    def __init__(self):
        pass

    def render(self, method: Method, dataset: Dataset, *, options=None) -> RenderOutput:
        dataset["cameras"].item()  # Assert there is only one camera
        info = method.get_info()
        supported_camera_models = info.get("supported_camera_models", frozenset(("pinhole",)))
        render = with_supported_camera_models(supported_camera_models)(method.render)
        return render(dataset["cameras"].item(), options=options)

    def get_name(self):
        return self._name

    def evaluate(self, predictions: RenderOutput, dataset: Dataset) -> Dict[str, Union[float, int]]:
        assert len(dataset["images"]) == 1, "There must be exactly one image in the dataset"
        background_color = dataset["metadata"].get("background_color")
        color_space = dataset["metadata"]["color_space"]
        pred = predictions["color"]
        gt = dataset["images"][0]
        pred = image_to_srgb(pred, np.uint8, color_space=color_space, background_color=background_color)
        gt = image_to_srgb(gt, np.uint8, color_space=color_space, background_color=background_color)
        pred_f = convert_image_dtype(pred, np.float32)
        gt_f = convert_image_dtype(gt, np.float32)
        mask = None
        if dataset.get("masks") is not None:
            assert dataset["masks"] is not None  # pyright issues
            mask = convert_image_dtype(dataset["masks"][0], np.float32)
        return compute_metrics(pred_f[None], gt_f[None], run_lpips_vgg=self._lpips_vgg, mask=mask, 
                               reduce=True)

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        acc = {}
        for i, data in enumerate(metrics):
            for k, v in data.items():
                # acc[k] = (acc.get(k, 0) * i + v) / (i + 1)
                acc[k] = acc.get(k, 0) * (i / (i + 1)) + v / (i + 1)
        return acc


class NerfEvaluationProtocol(DefaultEvaluationProtocol):
    _name = "nerf"
    _lpips_vgg = True


TRender = TypeVar("TRender", bound=typing.Callable[..., RenderOutput])


def with_supported_camera_models(supported_camera_models):
    supported_cam_models = set(_cameras.camera_model_to_int(x) for x in supported_camera_models)

    def decorator(render: TRender) -> TRender:
        @wraps(render)
        def _render(camera: Cameras, *args, **kwargs):
            cam = camera.item()  # Assert there is only one camera
            original_camera = None
            if cam.camera_models.item() not in supported_cam_models:
                original_camera = cam
                cam = _cameras.undistort_camera(cam)

            out = render(cam, *args, **kwargs)
            if original_camera is not None:
                out = cast(RenderOutput, {
                    k: _cameras.warp_image_between_cameras(cam, original_camera, cast(np.ndarray, v)) for k, v in out.items()})
            return out
        return cast(TRender, _render)
    return decorator


def render_all_images(
    method: Method,
    dataset: Dataset,
    output: str,
    description: Optional[str] = "rendering all images",
    nb_info: Optional[dict] = None,
    evaluation_protocol: Optional[EvaluationProtocol] = None,
) -> Iterable[RenderOutput]:
    if evaluation_protocol is None:
        evaluation_protocol = build_evaluation_protocol(dataset["metadata"]["evaluation_protocol"])
    logging.info(f"Rendering images with evaluation protocol {evaluation_protocol.get_name()}")
    background_color =  dataset["metadata"].get("background_color", None)
    if background_color is not None:
        background_color = convert_image_dtype(np.array(background_color), np.uint8)
    if nb_info is None:
        nb_info = {}
    else:
        nb_info = nb_info.copy()
        dataset_colorspace = dataset["metadata"].get("color_space", "srgb")
        assert dataset_colorspace == nb_info.get("color_space", "srgb"), \
            f"Dataset color space {dataset_colorspace} != method color space {nb_info.get('color_space')}"
        if "dataset_background_color" in nb_info:
            info_background_color = nb_info.get("dataset_background_color")
            if info_background_color is not None:
                info_background_color = np.array(info_background_color, np.uint8)
            assert info_background_color is None or (background_color is not None and np.array_equal(info_background_color, background_color)), \
                f"Dataset background color {background_color} != method background color {info_background_color}"
    nb_info["checkpoint_sha256"] = get_method_sha(method)
    nb_info["evaluation_protocol"] = evaluation_protocol.get_name()

    def _render_all():
        for i in range(len(dataset["cameras"])):
            yield evaluation_protocol.render(method, dataset_index_select(dataset, [i]))

    with tqdm(desc=description or "", 
              disable=description is None, 
              total=len(dataset["cameras"]), 
              dynamic_ncols=True) as progress:
        for val in _save_predictions_iterate(output,
                                             _render_all(),
                                             dataset=dataset,
                                             nb_info=nb_info):
            progress.update(1)
            yield val


def render_frames(
    method: Method,
    cameras: Cameras,
    output: Union[str, Path],
    fps: float,
    embeddings: Optional[List[np.ndarray]] = None,
    description: str = "rendering frames",
    output_names: Tuple[str, ...] = ("color",),
    nb_info: Optional[dict] = None,
) -> None:
    output = str(output) if isinstance(output, Path) else output
    assert cameras.image_sizes is not None, "cameras.image_sizes must be set"
    info = method.get_info()
    render = with_supported_camera_models(info.get("supported_camera_models", frozenset(("pinhole",))))(method.render)
    color_space = "srgb"
    background_color = nb_info.get("background_color") if nb_info is not None else None
    expected_scene_scale = nb_info.get("expected_scene_scale") if nb_info is not None else None
    output_types_map = {
        (x if isinstance(x, str) else x["name"]): (x if isinstance(x, str) else x.get("type", x["name"]))
        for x in info.get("supported_outputs", ["color"])
    }
    for output_name in output_names:
        if output_name not in output_types_map:
            raise ValueError(f"Output type {output_name} not supported by method. Supported types: {list(output_types_map.keys())}")

    def _predict_all(allow_transparency=True):
        for i in tqdm(range(len(cameras)), desc=description, total=len(cameras), dynamic_ncols=True):
            options: RenderOptions = {
                "output_type_dtypes": {"color": "uint8"},
                "embedding": (embeddings[i] if embeddings is not None else None),
            }
            pred = render(cameras[i], options=options)
            out = {}
            for output_name in output_names:
                output_type = output_types_map[output_name]
                if output_type == "color":
                    pred_image = image_to_srgb(pred[output_name], np.uint8, 
                                               color_space=color_space, 
                                               allow_alpha=allow_transparency, 
                                               background_color=background_color)
                    out[output_name] = pred_image
                elif output_type == "depth":
                    depth_rgb = visualize_depth(
                            pred[output_name], 
                            near_far=cameras.nears_fars[i] if cameras.nears_fars is not None else None, 
                            expected_scale=expected_scene_scale)
                    out[output_name] = convert_image_dtype(depth_rgb, np.uint8)
                elif output_type == "accumulation":
                    out[output_name] = convert_image_dtype(apply_colormap(pred[output_name], pallete="coolwarm"), np.uint8)
                else:
                    raise RuntimeError(f"Output type {output_type} is not supported.")
            yield out

    @contextmanager
    def _zip_writer(output):
        with zipfile.ZipFile(output, "w") as zip:
            i = 0
            def _write_frame(frame):
                nonlocal i
                rel_path = f"{i:05d}.png"
                framedata = frame if isinstance(frame, dict) else {None: frame}
                for key, image in framedata.items():
                    lpath = f"{key}/{rel_path}" if key is not None else rel_path

                    date_time = time.localtime(time.time())[:6]
                    zinfo = zipfile.ZipInfo(lpath, date_time)
                    zinfo.compress_type = zip.compression
                    if hasattr(zip, "_compresslevel"):
                        zinfo._compresslevel = zip.compresslevel  # type: ignore
                    zinfo.external_attr = 0o600 << 16     # ?rw-------

                    with zip.open(zinfo, 'w') as dest:
                        if not hasattr(dest, "name"):
                            # For older versions of Python
                            dest.name = lpath  # type: ignore
                        assert dest.name == lpath
                        save_image(cast(BinaryIO, dest), image)
                i += 1
            yield _write_frame

    @contextmanager
    def _targz_writer(output):
        with tarfile.open(output, "w:gz") as tar:
            i = 0
            def _write_frame(frame):
                nonlocal i
                rel_path = f"{i:05d}.png"
                framedata = frame if isinstance(frame, dict) else {None: frame}
                for key, image in framedata.items():
                    lpath = f"{key}/{rel_path}" if key is not None else rel_path
                    tarinfo = tarfile.TarInfo(name=lpath)
                    tarinfo.mtime = int(time.time())
                    with io.BytesIO() as f:
                        f.name = lpath  # type: ignore
                        save_image(f, image)
                        tarinfo.size = f.tell()
                        f.seek(0)
                        tar.addfile(tarinfo=tarinfo, fileobj=f)
                i += 1
            yield _write_frame

    vidwriter = 0

    @contextmanager
    def _video_writer(output):
        nonlocal vidwriter
        # Handle video
        try:
            import mediapy
        except ImportError:
            raise ImportError("mediapy is required to write videos. Install it with `pip install mediapy`")

        codec = 'gif' if output.endswith(".gif") else "h264"
        writer = None
        try:
            def _add_frame(frame):
                nonlocal writer
                if isinstance(frame, dict):
                    frame = np.concatenate(list(frame.values()), axis=1)
                if writer is None:
                    h, w = frame.shape[:-1]

                    writer = mediapy.VideoWriter(output, (h, w), fps=fps, codec=codec)
                    writer.__enter__()
                writer.add_image(frame)
            yield _add_frame
        finally:
            if writer is not None:
                writer.__exit__(None, None, None)
                writer = None

    @contextmanager
    def _folder_writer(output):
        os.makedirs(output, exist_ok=True)
        i = 0
        def _add_frame(frame):
            nonlocal i
            rel_path = f"{i:05d}.png"
            if isinstance(frame, dict):
                for key, image in frame.items():
                    os.makedirs(os.path.join(output, key), exist_ok=True)
                    with open(os.path.join(output, key, rel_path), "wb") as f:
                        save_image(f, image)
            else:
                with open(os.path.join(output, rel_path), "wb") as f:
                    save_image(f, frame)
            i += 1
        yield _add_frame

    writers = {}
    try:
        for output_name in output_names:
            loutput = output.format(output=output_name)
            if loutput not in writers:
                if path_is_video(loutput):
                    writer_obj = _video_writer(loutput)
                elif loutput.endswith(".zip"):
                    writer_obj = _zip_writer(loutput)
                elif loutput.endswith(".tar.gz"):
                    writer_obj = _targz_writer(loutput)
                else:
                    writer_obj = _folder_writer(loutput)
                writers[loutput] = (writer_obj, writer_obj.__enter__(), (output_name,))
            else:
                writer_obj, writer, _outs = writers[loutput]
                writers[loutput] = (writer_obj, writer, _outs + (output_name,))

        with zero_copy():
            for frame in _predict_all():
                for _, writer, _outs in writers.values():
                    if len(_outs) == 1:
                        writer(frame[_outs[0]])
                    else:
                        writer({name: frame[name] for name in _outs})
    finally:
        # Release all writers
        for writer_obj, _, _ in reversed(list(writers.values())):
            writer_obj.__exit__(None, None, None)


def trajectory_get_cameras(trajectory: Trajectory) -> Cameras:
    if trajectory["camera_model"] != "pinhole":
        raise NotImplementedError("Only pinhole camera model is supported")
    poses = np.stack([x["pose"] for x in trajectory["frames"]])
    intrinsics = np.stack([x["intrinsics"] for x in trajectory["frames"]])
    camera_models = np.array([camera_model_to_int(trajectory["camera_model"])]*len(poses), dtype=np.int32)
    image_sizes = np.array([list(trajectory["image_size"])]*len(poses), dtype=np.int32)
    return new_cameras(poses=poses, 
                       intrinsics=intrinsics, 
                       camera_models=camera_models, 
                       image_sizes=image_sizes,
                       distortion_parameters=np.zeros((len(poses), 0), dtype=np.float32),
                       nears_fars=None, 
                       metadata=None)


def trajectory_get_embeddings(method: Method, trajectory: Trajectory) -> Optional[List[np.ndarray]]:
    appearances = list(trajectory.get("appearances") or [])
    appearance_embeddings: List[Optional[np.ndarray]] = [None] * len(appearances)

    # Fill in embedding images
    for i, appearance in enumerate(appearances):
        if appearance.get("embedding") is not None:
            appearance_embeddings[i] = appearance.get("embedding")
        elif appearance.get("embedding_train_index") is not None:
            appearance_embeddings[i] = method.get_train_embedding(_assert_not_none(appearance.get("embedding_train_index")))
    if all(x is None for x in appearance_embeddings):
        return None
    if not all(x is not None for x in appearance_embeddings):
        raise ValueError("Either all embeddings must be provided or all must be missing")
    if all(x.get("appearance_weights") is None for x in trajectory["frames"]):
        return None
    if not all(x.get("appearance_weights") is not None for x in trajectory["frames"]):
        raise ValueError("Either all appearance weights must be provided or all must be missing")
    appearance_embeddings_np = np.stack(cast(List[np.ndarray], appearance_embeddings))

    # Interpolate embeddings
    out = []
    for frame in trajectory["frames"]:
        embedding = (frame.get("appearance_weights") @ appearance_embeddings_np).astype(appearance_embeddings_np.dtype)
        out.append(embedding)
    return out


@contextmanager
def run_inside_eval_container(backend_name: Optional[str] = None):
    """
    Ensures PyTorch is available to compute extra metrics (lpips)
    """
    from .backends import get_backend
    try:
        import torch as _
        yield None
        return
    except ImportError:
        pass

    logging.warning("PyTorch is not available in the current environment, we will create a new environment to compute extra metrics (lpips)")
    if backend_name is None:
        backend_name = os.environ.get("NERFBASELINES_BACKEND", None)
    backend = get_backend({
        "id": "metrics",
        "method_class": "base",
        "conda": {
            "environment_name": "_metrics", 
            "python_version": "3.10",
            "install_script": """
# Install dependencies
pip install \
    opencv-python==4.9.0.80 \
    torch==2.2.0 \
    torchvision==0.17.0 \
    'numpy<2.0.0' \
    --extra-index-url https://download.pytorch.org/whl/cu118
"""
        }}, backend=backend_name)
    with backend:
        backend.install()
        yield None

