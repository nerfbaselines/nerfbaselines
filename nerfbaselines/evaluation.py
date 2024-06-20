import tarfile
import time
import io
from functools import wraps
import logging
import os
import typing
from typing import Dict, Union, Iterable, TypeVar, Optional, cast, List
import numpy as np
import json
from pathlib import Path

from tqdm import tqdm

from .datasets import new_dataset
from .utils import (
    read_image, 
    convert_image_dtype, 
    run_on_host,
    image_to_srgb,
    save_image,
    visualize_depth,
    assert_not_none,
)
from .types import (
    Literal, 
    Dataset,
    RenderOutput, 
    EvaluationProtocol, 
    Cameras,
    Method,
    Trajectory,
    camera_model_to_int,
    new_cameras,
)
from .registry import build_evaluation_protocol
from .io import (
    open_any_directory, 
    deserialize_nb_info, 
    get_predictions_sha,
    get_method_sha,
    save_evaluation_results,
    save_predictions,
)
from . import metrics
from . import cameras as _cameras
try:
    from typeguard import suppress_type_checks
except ImportError:
    from contextlib import nullcontext as suppress_type_checks


OutputType = Literal["color", "depth"]
T = TypeVar("T")


@typing.overload
def compute_metrics(pred: np.ndarray, gt: np.ndarray, *, reduce: Literal[True] = True, run_lpips_vgg: bool = ...) -> Dict[str, float]:
    ...


@typing.overload
def compute_metrics(pred: np.ndarray, gt: np.ndarray, *, reduce: Literal[False], run_lpips_vgg: bool = ...) -> Dict[str, np.ndarray]:
    ...


@run_on_host()
def compute_metrics(pred, gt, *, reduce: bool = True, run_lpips_vgg: bool = False):
    # NOTE: we blend with black background here!
    def reduction(x):
        if reduce:
            return x.mean().item()
        else:
            return x

    pred = pred[..., : gt.shape[-1]]
    pred = convert_image_dtype(pred, np.float32)
    gt = convert_image_dtype(gt, np.float32)
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

        gt_images = [
            read_image(predictions_path / "gt-color" / name) for name in relpaths
        ]
        with suppress_type_checks():
            from pprint import pprint
            pprint(nb_info)
            dataset = new_dataset(
                cameras=typing.cast(Cameras, None),
                image_paths=relpaths,
                image_paths_root=str(predictions_path / "color"),
                metadata=typing.cast(Dict, nb_info.get("render_dataset_metadata", nb_info.get("dataset_metadata", {}))),
                images=gt_images)

            # Evaluate the prediction
            with tqdm(desc=description, dynamic_ncols=True, total=len(relpaths)) as progress:
                def collect_metrics_lists(iterable: Iterable[Dict[str, T]]) -> Iterable[Dict[str, T]]:
                    for data in iterable:
                        for k, v in data.items():
                            if k not in metrics_lists:
                                metrics_lists[k] = []
                            metrics_lists[k].append(v)
                        progress.update(1)
                        if "psnr" in metrics_lists:
                            psnr_val = np.mean(metrics_lists["psnr"][-1])
                            progress.set_postfix(psnr=f"{psnr_val:.4f}")
                        yield data

                metrics = evaluation_protocol.accumulate_metrics(
                    collect_metrics_lists(evaluation_protocol.evaluate(read_predictions(), dataset))
                )

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

    def render(self, method: Method, dataset: Dataset) -> Iterable[RenderOutput]:
        info = method.get_info()
        supported_camera_models = info.get("supported_camera_models", frozenset(("pinhole",)))
        render = with_supported_camera_models(supported_camera_models)(method.render)
        yield from render(dataset["cameras"])

    def get_name(self):
        return self._name

    def evaluate(self, predictions: Iterable[RenderOutput], dataset: Dataset) -> Iterable[Dict[str, Union[float, int]]]:
        background_color = dataset["metadata"].get("background_color")
        color_space = dataset["metadata"]["color_space"]
        for i, prediction in enumerate(predictions):
            pred = prediction["color"]
            gt = dataset["images"][i]
            pred = image_to_srgb(pred, np.uint8, color_space=color_space, background_color=background_color)
            gt = image_to_srgb(gt, np.uint8, color_space=color_space, background_color=background_color)
            pred_f = convert_image_dtype(pred, np.float32)
            gt_f = convert_image_dtype(gt, np.float32)
            yield compute_metrics(pred_f[None], gt_f[None], run_lpips_vgg=self._lpips_vgg, reduce=True)

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


TRender = TypeVar("TRender", bound=typing.Callable[..., Iterable[RenderOutput]])


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


def render_all_images(
    method: Method,
    dataset: Dataset,
    output: str,
    description: str = "rendering all images",
    nb_info: Optional[dict] = None,
    evaluation_protocol: Optional[EvaluationProtocol] = None,
) -> Iterable[RenderOutput]:
    if evaluation_protocol is None:
        evaluation_protocol = build_evaluation_protocol(dataset["metadata"]["evaluation_protocol"])
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

    with tqdm(desc=description, total=len(dataset["image_paths"]), dynamic_ncols=True) as progress:
        for val in save_predictions(output,
                                    evaluation_protocol.render(method, dataset),
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
    output_type: OutputType = "color",
    nb_info: Optional[dict] = None,
) -> None:
    output = Path(output)
    assert cameras.image_sizes is not None, "cameras.image_sizes must be set"
    info = method.get_info()
    render = with_supported_camera_models(info.get("supported_camera_models", frozenset(("pinhole",))))(method.render)
    color_space = "srgb"
    background_color = nb_info.get("background_color") if nb_info is not None else None
    expected_scene_scale = nb_info.get("expected_scene_scale") if nb_info is not None else None

    def _predict_all(allow_transparency=True):
        predictions = render(cameras, embeddings=embeddings)
        for i, pred in enumerate(tqdm(predictions, desc=description, total=len(cameras), dynamic_ncols=True)):
            pred_image = image_to_srgb(pred["color"], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
            if output_type == "color":
                yield pred_image
            elif output_type == "depth":
                assert "depth" in pred, "Method does not output depth"
                depth_rgb = visualize_depth(pred["depth"], near_far=cameras.nears_fars[i] if cameras.nears_fars is not None else None, expected_scale=expected_scene_scale)
                yield convert_image_dtype(depth_rgb, np.uint8)
            else:
                raise RuntimeError(f"Output type {output_type} is not supported.")

    if str(output).endswith(".tar.gz"):
        with tarfile.open(output, "w:gz") as tar:
            for i, frame in enumerate(_predict_all()):
                rel_path = f"{i:05d}.png"
                tarinfo = tarfile.TarInfo(name=rel_path)
                tarinfo.mtime = int(time.time())
                with io.BytesIO() as f:
                    f.name = rel_path
                    tarinfo.size = f.tell()
                    f.seek(0)
                    save_image(f, frame)
                    tar.addfile(tarinfo=tarinfo, fileobj=f)
    elif str(output).endswith(".mp4") or str(output).endswith(".gif"):
        # Handle video
        import mediapy

        w, h = cameras.image_sizes[0]
        codec = 'h264'
        if str(output).endswith(".gif"):
            codec = "gif"
        with mediapy.VideoWriter(output, (h, w), metadata=mediapy.VideoMetadata(len(cameras), (h, w), fps, bps=None), fps=fps, codec=codec) as writer:
            for i, frame in enumerate(_predict_all(allow_transparency=False)):
                writer.add_image(frame)
    else:
        os.makedirs(output, exist_ok=True)
        for i, frame in enumerate(_predict_all()):
            rel_path = f"{i:05d}.png"
            with open(os.path.join(output, rel_path), "wb") as f:
                save_image(f, frame)


def trajectory_get_cameras(trajectory: Trajectory) -> Cameras:
    if trajectory["camera_model"] != "pinhole":
        raise NotImplementedError("Only pinhole camera model is supported")
    poses = np.stack([x["pose"] for x in trajectory["frames"]])
    intrinsics = np.stack([x["intrinsics"] for x in trajectory["frames"]])
    camera_types = np.array([camera_model_to_int(trajectory["camera_model"])]*len(poses), dtype=np.int32)
    image_sizes = np.array([list(trajectory["image_size"])]*len(poses), dtype=np.int32)
    return new_cameras(poses=poses, 
                       intrinsics=intrinsics, 
                       camera_types=camera_types, 
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
            appearance_embeddings[i] = method.get_train_embedding(assert_not_none(appearance.get("embedding_train_index")))
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
        embedding = frame.get("appearance_weights") @ appearance_embeddings_np
        out.append(embedding)
    return out

