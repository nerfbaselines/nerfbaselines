import contextlib
import logging
import os
import struct
import hashlib
import base64
import typing
from typing import List, Dict, Union, Iterable, TypeVar
import numpy as np
import json
from pathlib import Path
import tarfile
import tempfile

from tqdm import tqdm

from .cameras import CameraModel
from .utils import read_image, convert_image_dtype, run_on_host
from .types import Optional, Literal, Dataset, ProgressCallback, RenderOutput, EvaluationProtocol, Cameras, MethodInfo
from .render import image_to_srgb, Method, with_supported_camera_models
from .io import open_any_directory, deserialize_nb_info, serialize_nb_info
from .backends import get_backend
from . import metrics


T = TypeVar("T")


@typing.overload
def compute_metrics(pred: np.ndarray, gt: np.ndarray, *, reduce: Literal[True] = True) -> Dict[str, float]:
    ...


@typing.overload
def compute_metrics(pred: np.ndarray, gt: np.ndarray, *, reduce: Literal[False]) -> Dict[str, np.ndarray]:
    ...


@run_on_host()
def compute_metrics(pred, gt, *, reduce: bool = True):
    # NOTE: we blend with black background here!
    run_extras = True
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
    }
    if run_extras:
        out["lpips"] = reduction(metrics.lpips(gt, pred))
    return out


def _encode_values(values: List[float]) -> str:
    return base64.b64encode(b"".join(struct.pack("f", v) for v in values)).decode("ascii")


def get_predictions_hashes(predictions: Path, description: str = "hashing predictions"):
    b = bytearray(128 * 1024)
    mv = memoryview(b)

    def sha256_update(sha, filename):
        with open(filename, "rb", buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                sha.update(mv[:n])

    if str(predictions).endswith(".tar.gz"):
        with tarfile.open(predictions, "r:gz") as tar, tempfile.TemporaryDirectory() as tmpdir:
            tar.extractall(tmpdir)
            return get_predictions_hashes(Path(tmpdir))
    predictions_sha = hashlib.sha256()
    gt_sha = hashlib.sha256()
    relpaths = [x.relative_to(predictions / "color") for x in (predictions / "color").glob("**/*") if x.is_file()]
    relpaths.sort()
    for relname in tqdm(relpaths, desc=description):
        sha256_update(predictions_sha, predictions / "color" / relname)
        sha256_update(gt_sha, predictions / "gt-color" / relname)
    return (
        predictions_sha.hexdigest(),
        gt_sha.hexdigest(),
    )


def _get_metrics_hash(metrics_lists):
    metrics_sha = hashlib.sha256()
    for k in sorted(metrics_lists.keys()):
        metrics_sha.update(k.lower().encode("utf8"))
        values = sorted(metrics_lists[k])
        metrics_sha.update(_encode_values(values).encode("ascii"))
        metrics_sha.update(b"\n")
    return metrics_sha.hexdigest()


def evaluate(predictions: Union[str, Path], 
             output: Path, 
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

    with open_any_directory(str(predictions), "r") as predictions_path:
        with open(str(predictions_path / "info.json"), "r", encoding="utf8") as f:
            info = json.load(f)
        info = deserialize_nb_info(info)

        if evaluation_protocol is None:
            evaluation_protocol = get_evaluation_protocol(name=info["evaluation_protocol"])

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
        dataset = Dataset(
            cameras=typing.cast(Cameras, None),
            file_paths=relpaths,
            file_paths_root=str(predictions_path / "color"),
            metadata=typing.cast(Dict, info.get("dataset_metadata", {})),
            images=gt_images)

        def collect_metrics_lists(iterable: Iterable[Dict[str, T]]) -> Iterable[Dict[str, T]]:
            for data in iterable:
                for k, v in data.items():
                    if k not in metrics_lists:
                        metrics_lists[k] = []
                    metrics_lists[k].append(v)
                yield data

        # Evaluate the prediction
        metrics = evaluation_protocol.accumulate_metrics(
            collect_metrics_lists(
                tqdm(evaluation_protocol.evaluate(read_predictions(), dataset), desc=description)
            )
        )

        predictions_sha, ground_truth_sha = get_predictions_hashes(predictions_path)
        precision = 5
        out = {
            "info": serialize_nb_info(info),
            "metrics": {k: round(v, precision) for k, v in metrics.items()},
            "metrics_raw": {k: _encode_values(metrics_lists[k]) for k in metrics_lists},
            "metrics_sha256": _get_metrics_hash(metrics_lists),
            "predictions_sha256": predictions_sha,
            "ground_truth_sha256": ground_truth_sha,
            "evaluation_protocol": evaluation_protocol.get_name(),
        }

        # If output is specified, write the results to a file
        if os.path.exists(str(output)):
            raise FileExistsError(f"{output} already exists")
        with open(str(output), "w", encoding="utf8") as f:
            json.dump(out, f, indent=2)
        return out


class DefaultEvaluationProtocol(EvaluationProtocol):
    def __init__(self, **kwargs):
        pass

    def render(self, method: Method, dataset: Dataset, progress_callback: Optional[ProgressCallback] = None) -> Iterable[RenderOutput]:
        info = method.get_info()
        supported_camera_models = info.get("supported_camera_models", frozenset((CameraModel.PINHOLE,)))
        render = with_supported_camera_models(supported_camera_models)(method.render)
        yield from render(dataset.cameras, progress_callback=progress_callback)

    def get_name(self):
        return "default"

    def evaluate(self, predictions: Iterable[RenderOutput], dataset: Dataset) -> Iterable[Dict[str, Union[float, int]]]:
        assert dataset.images is not None, "dataset.images must be set"
        background_color = dataset.metadata.get("background_color")
        for i, prediction in enumerate(predictions):
            pred = prediction["color"]
            gt = dataset.images[i]
            pred = image_to_srgb(pred, np.uint8, color_space=dataset.color_space, background_color=background_color)
            gt = image_to_srgb(gt, np.uint8, color_space=dataset.color_space, background_color=background_color)
            pred_f = convert_image_dtype(pred, np.float32)
            gt_f = convert_image_dtype(gt, np.float32)
            yield compute_metrics(pred_f[None], gt_f[None], reduce=True)

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        acc = {}
        for i, data in enumerate(metrics):
            for k, v in data.items():
                # acc[k] = (acc.get(k, 0) * i + v) / (i + 1)
                acc[k] = acc.get(k, 0) * (i / (i + 1)) + v / (i + 1)
        return acc


def get_evaluation_protocol(name: Optional[str] = None, dataset_name: Optional[str] = None, **kwargs) -> EvaluationProtocol:
    return DefaultEvaluationProtocol(**kwargs)


@contextlib.contextmanager
def run_inside_eval_container():
    """
    Ensures PyTorch is available to compute extra metrics (lpips)
    """
    try:
        import torch as _
        yield None
        return
    except ImportError:
        pass

    logging.warning("PyTorch is not available in the current environment, we will create a new environment to compute extra metrics (lpips)")
    backend = get_backend({
        "method": "base",
        "conda": {
            "environment_name": "_metrics", 
            "install_script": ""
        }}, os.environ.get("NERFBASELINES_BACKEND", None))
    with backend:
        backend.install()
        yield None
