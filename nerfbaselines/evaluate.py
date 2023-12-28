import logging
import os
import struct
import hashlib
import base64
import typing
from typing import List, Dict, Union
import numpy as np
import json
from pathlib import Path
import tarfile
import tempfile

from tqdm import tqdm

from .utils import read_image, convert_image_dtype
from .types import Optional
from .render import image_to_srgb
from .io import open_any_directory
from . import metrics


def test_extra_metrics():
    """
    Test if the extra metrics are available.

    """
    a = np.zeros((1, 48, 56, 3), dtype=np.float32)
    b = np.ones((1, 48, 56, 3), dtype=np.float32)
    compute_metrics(a, b, run_extras=True)


@typing.overload
def compute_image_metrics(pred: np.ndarray, gt: np.ndarray, *, run_extras: bool = ..., reduce: bool = True) -> Dict[str, float]:
    ...


@typing.overload
def compute_image_metrics(pred: np.ndarray, gt: np.ndarray, *, run_extras: bool = ..., reduce: bool = False) -> Dict[str, np.ndarray]:
    ...


def compute_metrics(pred, gt, *, run_extras: bool = False, reduce: bool = True):
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


def evaluate(predictions: Union[str, Path], output: Path, disable_extra_metrics: Optional[bool] = None, description: str = "evaluating"):
    """
    Evaluate a set of predictions.

    Args:
        predictions: Path to a directory containing the predictions.
        output: Path to a json file where the results will be written.
        disable_extra_metrics: If True, skip the evaluation of metrics requiring extra dependencies.
        description: Description of the evaluation, used for progress bar.
    Returns:
        A dictionary containing the results.
    """

    with open_any_directory(str(predictions), "r") as predictions_path:
        if disable_extra_metrics is None:
            disable_extra_metrics = False
            try:
                test_extra_metrics()
            except ImportError as exc:
                logging.error(exc)
                logging.error("Extra metrics are not available and will be disabled. Please install torch and jax and other required dependencies by running `pip install nerfbaselines[extras]`.")
                disable_extra_metrics = True

        with open(str(predictions_path / "info.json"), "r", encoding="utf8") as f:
            info = json.load(f)
        color_space = info.get("color_space")
        assert color_space is not None, "Color space must be specified in info.json"
        expected_scene_scale = info.get("expected_scene_scale")
        assert expected_scene_scale is not None, "Expected scene scale must be specified in info.json"
        background_color = info["dataset_background_color"]
        if background_color is not None:
            background_color = np.array(background_color)
            if background_color.dtype.kind == "f":
                background_color = background_color.astype(np.float32)
            else:
                background_color = background_color.astype(np.uint8)

        # Run the evaluation
        metrics_lists = {}
        relpaths = [x.relative_to(predictions_path / "color") for x in (predictions_path / "color").glob("**/*") if x.is_file()]
        relpaths.sort()

        for relname in tqdm(relpaths, desc=description):
            # Load the prediction
            assert color_space == "srgb", "Only srgb color space is supported for now"
            gt = read_image(predictions_path / "gt-color" / relname)
            pred = read_image(predictions_path / "color" / relname)

            gt_f = image_to_srgb(gt, np.float32, color_space=color_space, background_color=background_color)
            pred_f = image_to_srgb(pred, np.float32, color_space=color_space, background_color=background_color)

            # Evaluate the prediction
            for k, v in compute_metrics(gt_f[None], pred_f[None], run_extras=not disable_extra_metrics, reduce=True).items():
                if k not in metrics_lists:
                    metrics_lists[f"{k}"] = []
                metrics_lists[f"{k}"].append(v)

        predictions_sha, ground_truth_sha = get_predictions_hashes(predictions_path)
        precision = 5
        out = {
            "info": info,
            "metrics": {k: round(np.mean(metrics_lists[k]).item(), precision) for k in metrics_lists},
            "metrics_raw": {k: _encode_values(metrics_lists[k]) for k in metrics_lists},
            "metrics_sha256": _get_metrics_hash(metrics_lists),
            "predictions_sha256": predictions_sha,
            "ground_truth_sha256": ground_truth_sha,
        }

        # If output is specified, write the results to a file
        if os.path.exists(str(output)):
            raise FileExistsError(f"{output} already exists")
        with open(str(output), "w", encoding="utf8") as f:
            json.dump(out, f, indent=2)
        return out
