from typing import Optional
import zipfile
import urllib.parse
from tqdm import tqdm
import requests
import hashlib
import logging
import shutil
import tarfile
import tempfile
import json
from pathlib import Path
from .render import get_checkpoint_sha
from .evaluate import get_predictions_hashes


def _upload_fileio_single(path: Path):
    # Size limit is 2GB, therefore, we need to split the file into chunks
    path = Path(path)
    url = "https://file.io/"

    logging.info("uploading " + str(path))
    with open(path, "rb") as f:
        response = requests.post(
            url,
            files={"file": f},
            data={
                "expires": "7d",
                "maxDownloads": "1",
                "autoDelete": "true",
            },
        )
        response.raise_for_status()
        res = response.json()
        return res["link"]


def _upload_fileio(path: Path):
    # Size limit is 2GB, therefore, we need to split the file into chunks
    path = Path(path)

    b = bytearray(128 * 1024)
    mv = memoryview(b)

    # limit = 2 * 1000 * 1024 * 1024
    limit = 100 * 1024 * 1024
    total_size = path.stat().st_size
    if total_size <= limit:
        return _upload_fileio_single(path)
    parts = []
    nparts = (total_size + limit - 1) // limit
    with open(path, "rb", buffering=0) as f, tempfile.TemporaryDirectory() as td, tqdm(total=nparts, desc="Uploading") as pbar:
        for i in range(nparts):
            with open(Path(td) / (f"part_{i}" + path.suffix), "wb") as fp:
                current_size = 0
                for n in iter(lambda: f.readinto(mv), 0):
                    current_size += n
                    fp.write(mv[:n])
                    if current_size >= limit:
                        break
                fp.flush()

            # Commit current file
            parts.append(_upload_fileio_single(Path(td) / (f"part_{i}" + path.suffix)))
            pbar.update()
    return parts


def _create_github_update_link(results):
    """Creates a link to a GitHub issue with the results."""
    # Create a GitHub issue
    method = results["info"]["method"]
    if method is None:
        raise ValueError("method must be set")
    dataset_type = results["info"]["dataset_type"]
    if dataset_type is None:
        raise ValueError("dataset_type must be set")
    scene = results["info"]["dataset_scene"]
    if scene is None:
        raise ValueError("dataset_scene must be set")
    value = urllib.parse.quote_plus(json.dumps(results, indent=2))
    url = f"https://github.com/jkulhanek/nerfbaselines/new/main?filename=results/{method}/{dataset_type}/{scene}.json&value={value}"
    return url


def _zip_add_dir(zip: zipfile.ZipFile, dirpath: Path, arcname: Optional[str] = None):
    for name in dirpath.glob("**/*"):
        rel_name = name.relative_to(dirpath)
        if arcname is not None:
            rel_name = Path(arcname) / rel_name
        if name.is_dir():
            pass
        elif name.is_file():
            zip.write(str(name), str(rel_name))
        else:
            raise ValueError(f"unknown file type: {name}")


def prepare_results_for_upload(model_path: Path, predictions_path: Path, metrics_path: Path, tensorboard_path: Path, output_path: Path, validate: bool = True):
    """Prepares artifacts for upload to the NeRF benchmark.

    Args:
        model_path: Path to the model directory.
        predictions_path: Path to the predictions directory/file.
        metrics_path: Path to the metrics file.
        tensorboard_path: Path to the tensorboard events file.
    """
    # Convert to Path objects (if strs)
    model_path = Path(model_path)
    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)
    tensorboard_path = Path(tensorboard_path)
    assert model_path.exists(), f"{model_path} does not exist"
    assert predictions_path.exists(), f"{predictions_path} does not exist"
    assert metrics_path.exists(), f"{metrics_path} does not exist"
    assert tensorboard_path.exists(), f"{tensorboard_path} does not exist"

    # Load metrics
    with metrics_path.open("r", encoding="utf8") as f:
        metrics = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Decompress model if necessary
        if str(model_path).endswith(".tar.gz"):
            (tmpdir / "checkpoint").mkdir()
            with tarfile.open(model_path, "r:gz") as tar:
                tar.extractall(tmpdir / "checkpoint")
            model_path = tmpdir / "checkpoint"

        # Decompress predictions if necessary
        if str(predictions_path).endswith(".tar.gz"):
            (tmpdir / "predictions").mkdir()
            with tarfile.open(predictions_path, "r:gz") as tar:
                tar.extractall(tmpdir / "predictions")
            predictions_path = tmpdir / "predictions"

        # Verify all signatures
        if validate:
            checkpoint_sha = get_checkpoint_sha(model_path)
            predictions_sha, ground_truth_sha = get_predictions_hashes(predictions_path)
            if metrics["predictions_sha256"] != predictions_sha:
                raise ValueError("Predictions SHA mismatch")
            if metrics["ground_truth_sha256"] != ground_truth_sha:
                raise ValueError("Ground truth SHA mismatch")
            if metrics["info"]["checkpoint_sha256"] != checkpoint_sha:
                raise ValueError("Checkpoint SHA mismatch")

        # Prepare artifact
        # with tarfile.open(tmpdir/"artifact.tar.gz", "w") as zip:
        #     tar.add(metrics_path, arcname="results.json")
        #     tar.add(model_path, arcname="checkpoint")
        #     tar.add(predictions_path, arcname="predictions")
        artifact_path = tmpdir / "artifact.zip"
        with zipfile.ZipFile(artifact_path, "w") as zip:
            zip.write(metrics_path, "results.json")
            _zip_add_dir(zip, model_path, arcname="checkpoint")
            _zip_add_dir(zip, predictions_path, arcname="predictions")
            _zip_add_dir(zip, tensorboard_path, arcname="tensorboard")

        # Get the artifact SHA
        logging.info("computing output artifact SHA")
        b = bytearray(128 * 1024)
        mv = memoryview(b)
        sha = hashlib.sha256()
        with open(artifact_path, "rb", buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                sha.update(mv[:n])
        shutil.move(artifact_path, output_path)
        logging.info(f"artifact {output_path} generated, sha: " + sha.hexdigest())
