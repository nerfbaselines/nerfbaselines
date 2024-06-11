import json
import numpy as np
import time
import tarfile
import os
from typing import Union, Iterator, IO, Any
import zipfile
import contextlib
from pathlib import Path
from typing import BinaryIO
import tempfile
import logging
from tqdm import tqdm
import requests
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from .types import Trajectory
from .utils import assert_not_none


OpenMode = Literal["r", "w"]


def wget(url: str, output: Union[str, Path]):
    output = Path(output)
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(
        total=total_size_in_bytes, unit="iB", unit_scale=True, desc="Downloading"
    )
    with open(output, "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        logging.error(
            f"Failed to download {url}. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes."
        )


@contextlib.contextmanager
def open_any(
    path: Union[str, Path, BinaryIO], mode: OpenMode = "r"
) -> Iterator[IO[bytes]]:
    if not isinstance(path, (str, Path)):
        yield path
        return

    path = str(path)
    components = path.split("/")
    zip_parts = [i for i, c in enumerate(components[:-1]) if c.endswith(".zip")]
    if zip_parts:
        with open_any("/".join(components[: zip_parts[-1] + 1]), mode=mode) as f:
            if components[zip_parts[-1]].endswith(".tar.gz"):
                # Extract from tar.gz
                rest = "/".join(components[zip_parts[-1] + 1 :])
                with tarfile.open(fileobj=f, mode=mode + ":gz") as tar:
                    if mode == "r":
                        with assert_not_none(tar.extractfile(rest)) as f:
                            yield f
                    elif mode == "w":
                        _, extension = os.path.split(rest)
                        with tempfile.TemporaryFile("wb", suffix=extension) as tmp:
                            yield tmp
                            tmp.flush()
                            tmp.seek(0)
                            tarinfo = tarfile.TarInfo(name=rest)
                            tarinfo.mtime = int(time.time())
                            tarinfo.mode = 0o644
                            tarinfo.size = tmp.tell()
                            tar.addfile(
                                tarinfo=tarinfo,
                                fileobj=tmp,
                            )

            else:
                # Extract from zip
                with zipfile.ZipFile(f, mode=mode) as zip, zip.open(
                    "/".join(components[zip_parts[-1] + 1 :]), mode=mode
                ) as f:
                    yield f
        return

    # Download from url
    if path.startswith("http://") or path.startswith("https://"):
        assert mode == "r", "Only reading from remote files is supported."
        response = requests.get(path, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True, desc="Downloading"
        )
        name = path.split("/")[-1]
        with tempfile.TemporaryFile("rb+", suffix=name) as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
            file.flush()
            file.seek(0)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                logging.error(
                    f"Failed to download {path}. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes."
                )
            yield file
        return

    # Normal file
    if mode == "w":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode=mode + "b") as f:
        yield f


@contextlib.contextmanager
def open_any_directory(path: Union[str, Path], mode: OpenMode = "r") -> Iterator[str]:
    path = str(path)
    path = os.path.abspath(path)

    components = path.split("/")
    compressed_parts = [
        i
        for i, c in enumerate(components)
        if c.endswith(".zip") or c.endswith(".tar.gz")
    ]
    if compressed_parts:
        with open_any(
            "/".join(components[: compressed_parts[-1] + 1]), mode=mode
        ) as f, tempfile.TemporaryDirectory() as tmpdir:
            rest = "/".join(components[compressed_parts[-1] + 1 :])
            if components[compressed_parts[-1]].endswith(".tar.gz"):
                with tarfile.open(fileobj=f, mode=mode + ":gz") as tar:
                    if mode == "r":
                        for member in tar.getmembers():
                            if not member.name.startswith(rest):
                                continue
                            if member.isdir():
                                os.makedirs(
                                    os.path.join(tmpdir, member.name), exist_ok=True
                                )
                            else:
                                tar.extract(member, tmpdir)
                        yield os.path.join(tmpdir, rest)
                    elif mode == "w":
                        tmp_path = Path(tmpdir) / rest
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        yield os.path.join(tmpdir, rest)

                        for root, dirs, files in os.walk(tmp_path):
                            for dir in dirs:
                                tar.add(
                                    os.path.join(root, dir),
                                    arcname=os.path.relpath(
                                        os.path.join(root, dir), tmp_path
                                    ),
                                )
                            for file in files:
                                tar.add(
                                    os.path.join(root, file),
                                    arcname=os.path.relpath(
                                        os.path.join(root, file), tmp_path
                                    ),
                                )
                    else:
                        raise RuntimeError(f"Unsupported mode {mode} for tar.gz files.")
            else:
                with zipfile.ZipFile(f, mode=mode) as zip:
                    # Extract from zip
                    if mode == "r":
                        for member in zip.infolist():
                            if not member.filename.startswith(rest):
                                continue
                            if member.is_dir():
                                os.makedirs(
                                    os.path.join(tmpdir, member.filename), exist_ok=True
                                )
                            else:
                                zip.extract(member, tmpdir)
                        yield os.path.join(tmpdir, rest)
                    elif mode == "w":
                        tmp_path = Path(tmpdir) / rest
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        yield os.path.join(tmpdir, rest)

                        for root, dirs, files in os.walk(tmp_path):
                            for dir in dirs:
                                zip.write(
                                    os.path.join(root, dir),
                                    arcname=os.path.relpath(
                                        os.path.join(root, dir), tmp_path
                                    ),
                                )
                            for file in files:
                                zip.write(
                                    os.path.join(root, file),
                                    arcname=os.path.relpath(
                                        os.path.join(root, file), tmp_path
                                    ),
                                )
                    else:
                        raise RuntimeError(f"Unsupported mode {mode} for zip files.")
        return

    if path.startswith("http://") or path.startswith("https://"):
        raise RuntimeError(
            "Only tar.gz and zip files are supported for remote directories."
        )

    # Normal file
    Path(path).mkdir(parents=True, exist_ok=True)
    yield str(Path(path).absolute())
    return


def serialize_nb_info(info: dict) -> dict:
    info = info.copy()
    if "dataset_metadata" in info:
        info["dataset_metadata"] = dm = info["dataset_metadata"].copy()
        if isinstance(dm.get("background_color"), np.ndarray):
            dm["background_color"] = dm["background_color"].tolist()
        if "viewer_initial_pose" in dm:
            dm["viewer_initial_pose"] = np.round(dm["viewer_initial_pose"], 5).tolist()
        if "viewer_transform" in dm:
            dm["viewer_transform"] = np.round(dm["viewer_transform"], 5).tolist()

    def ts(x):
        _ = info
        if isinstance(x, np.ndarray):
            raise NotImplementedError("Numpy arrays are not supported in nb-info")
        if isinstance(x, dict):
            return {k: ts(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return [ts(v) for v in x]
        else:
            return x
    ts(info)
    return info


def deserialize_nb_info(info: dict) -> dict:
    info = info.copy()
    if "dataset_metadata" in info:
        info["dataset_metadata"] = dm = info["dataset_metadata"].copy()
        if dm.get("background_color") is not None:
            dm["background_color"] = np.array(dm["background_color"], dtype=np.uint8)
        if "viewer_initial_pose" in dm:
            dm["viewer_initial_pose"] = np.array(dm["viewer_initial_pose"], dtype=np.float32)
        if "viewer_transform" in dm:
            dm["viewer_transform"] = np.array(dm["viewer_transform"], dtype=np.float32)
    return info


def save_trajectory(trajectory: Trajectory, file) -> None:
    data: Any = trajectory.copy()
    data["format"] = "nerfbaselines-v1"

    # Replace arrays with flat lists
    def _fix_appearance(appearance):
        if not appearance:
            return appearance
        appearance = appearance.copy()
        if appearance.get("embedding") is not None:
            appearance["embedding"] = appearance["embedding"].tolist()
        return appearance

    if data.get("source"):
        data["source"] = data["source"].copy()
        data["source"]["keyframes"] = data["source"]["keyframes"].copy()
        for i, kf in enumerate(data["source"].get("keyframes", [])):
            kf = data["source"]["keyframes"][i] = kf.copy()
            kf["pose"] = kf["pose"].flatten().tolist()
            if "appearance" in kf:
                kf["appearance"] = _fix_appearance(kf["appearance"])
        if data["source"]["default_appearance"]:
            data["source"]["default_appearance"] = _fix_appearance(data["source"]["default_appearance"])
    if data.get("frames"):
        data["frames"] = data["frames"].copy()
        for i, frame in enumerate(data["frames"]):
            frame = data["frames"][i] = frame.copy()
            frame["pose"] = frame["pose"].flatten().tolist()
            frame["intrinsics"] = frame["intrinsics"].tolist()
            frame["appearance_weights"] = frame["appearance_weights"].tolist()
    if data.get("appearances"):
        data["appearances"] = list(map(_fix_appearance, data["appearances"]))
    json.dump(data, file, indent=2)


def load_trajectory(file) -> Trajectory:
    data = json.load(file)
    if data.pop("format", None) != "nerfbaselines-v1":
        raise RuntimeError("Trajectory format is not supported")
    
    # Fix np arrays
    def _fix_appearance(appearance):
        if not appearance:
            return appearance
        appearance = appearance.copy()
        if appearance.get("embedding") is not None:
            appearance["embedding"] = np.array(appearance["embedding"], dtype=np.float32)
        return appearance

    data["image_size"] = tuple(data["image_size"])

    if data.get("source"):
        data["source"] = data["source"].copy()
        data["source"]["keyframes"] = data["source"]["keyframes"].copy()
        for i, kf in enumerate(data["source"].get("keyframes", [])):
            kf = data["source"]["keyframes"][i] = kf.copy()
            kf["pose"] = np.array(kf["pose"], dtype=np.float32).reshape(-1, 4)
            if "appearance" in kf:
                kf["appearance"] = _fix_appearance(kf["appearance"])
        if data["source"]["default_appearance"]:
            data["source"]["default_appearance"] = _fix_appearance(data["source"]["default_appearance"])
    if data.get("frames"):
        data["frames"] = data["frames"].copy()
        for i, frame in enumerate(data["frames"]):
            frame = data["frames"][i] = frame.copy()
            frame["pose"] = np.array(frame["pose"], dtype=np.float32).reshape(-1, 4)
            frame["intrinsics"] = np.array(frame["intrinsics"], dtype=np.float32)
            frame["appearance_weights"] = np.array(frame["appearance_weights"], dtype=np.float32)
    if data.get("appearances"):
        data["appearances"] = list(map(_fix_appearance, data["appearances"]))
    return data
