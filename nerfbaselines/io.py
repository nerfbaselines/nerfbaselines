import zlib
import io
import struct
import base64
import hashlib
import json
from contextlib import ExitStack
from datetime import datetime
import numpy as np
import time
import tarfile
import os
from typing import (
    Union, Iterator, Any, Dict, List, Iterable, Optional, TypeVar, overload,
    ContextManager, IO
)
import zipfile
import contextlib
from pathlib import Path
import tempfile
import logging
import shutil
from tqdm import tqdm
import urllib.request
from . import (
    Trajectory, 
    Method,
    Dataset,
    RenderOutput,
)
from .utils import (
    convert_image_dtype,
    visualize_depth,
    image_to_srgb,
)
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from . import __version__


OpenMode = Literal["r", "w"]
T = TypeVar("T")


def _assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


@overload
def wget(url: str, output: Literal[None] = None, *, desc: Optional[str] = ...) -> ContextManager[IO[bytes]]:
    ...

@overload
def wget(url: str, output: Union[str, Path, IO[bytes]], *, desc: Optional[str] = ...) -> None:
    ...


def wget(url: str, output: Union[str, Path, None, IO[bytes]] = None, *, desc: Optional[str] = None) -> Union[ContextManager[IO[bytes]], None]:
    if output is None:
        @contextlib.contextmanager
        def _wget():
            with io.BytesIO() as f:
                wget(url, f, desc=desc)
                f.seek(0)
                yield f
        return _wget()

    if isinstance(output, (str, Path)):
        output = str(output)
        with open_any(output, "w") as f:
            wget(url, f, desc=desc)
        return

    with contextlib.ExitStack() as stack:
        request = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Encoding": "gzip",
            "Accept": "*/*",
        })
        response = stack.enter_context(urllib.request.urlopen(request))
        if response.getcode() != 200:
            raise RuntimeError(f"Failed to download {url}.")
        total_size_in_bytes = int(response.getheader("Content-Length", 0))
        block_size = 1024 * 32
        rfile = response
        if desc is not None:
            rfile = stack.enter_context(tqdm.wrapattr(
                rfile, "read", total=total_size_in_bytes,
                unit="iB", unit_scale=True, desc=desc, dynamic_ncols=True))
        decompressor = None
        if response.getheader("Content-Encoding", "") == "gzip":
            decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
        while True:
            data = rfile.read(block_size)
            if not data:
                break
            if decompressor is not None:
                data = decompressor.decompress(data)
            output.write(data)
        if decompressor is not None:
            data = decompressor.flush()
            output.write(data)
        output.flush()


@contextlib.contextmanager
def open_any(
    path: Union[str, Path, IO[bytes]], mode: OpenMode = "r"
) -> Iterator[IO[bytes]]:
    if not isinstance(path, (str, Path)):
        yield path
        return

    path = str(path)
    components = path.split(os.path.sep)
    zip_parts = [i for i, c in enumerate(components[:-1]) if c.endswith(".zip")]
    if zip_parts:
        with open_any(os.path.sep.join(components[: zip_parts[-1] + 1]), mode=mode) as f:
            if components[zip_parts[-1]].endswith(".tar.gz"):
                # Extract from tar.gz
                rest = "/".join(components[zip_parts[-1] + 1 :])
                with tarfile.open(fileobj=f, mode=mode + ":gz") as tar:
                    if mode == "r":
                        with _assert_not_none(tar.extractfile(rest)) as f:
                            yield f
                    elif mode == "w":
                        _, extension = os.path.split(rest.replace("/", os.path.sep))
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
        name = path.split("/")[-1]
        with tempfile.TemporaryFile("w+b", suffix=name) as file:
            file = getattr(file, "file", file)  # Fix typeguard on windows
            wget(path, file, desc="Downloading")
            file.seek(0)
            yield file
        return

    # Normal file
    if mode == "w":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, mode=mode + "b") as f:
        yield f


@contextlib.contextmanager
def open_any_directory(path: Union[str, Path], mode: OpenMode = "r") -> Iterator[str]:
    path = str(path)
    if "://" not in path:
        path = os.path.abspath(path)

    components = path.split(os.path.sep)
    compressed_parts = [
        i
        for i, c in enumerate(components)
        if c.endswith(".zip") or c.endswith(".tar.gz")
    ]
    if compressed_parts:
        with open_any(
            os.path.sep.join(components[: compressed_parts[-1] + 1]), mode=mode
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
                        yield os.path.join(tmpdir, rest.replace("/", os.path.sep))
                    elif mode == "w":
                        tmp_path = Path(tmpdir) / rest.replace("/", os.path.sep)
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        yield str(tmp_path)

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
                                # Fix mtime
                                extracted_path = os.path.join(tmpdir, member.filename)
                                date_time = datetime(*member.date_time)
                                mtime = time.mktime(date_time.timetuple())
                                os.utime(extracted_path, (mtime, mtime))

                        yield os.path.join(tmpdir, rest.replace("/", os.path.sep))
                    elif mode == "w":
                        tmp_path = Path(tmpdir) / rest.replace("/", os.path.sep)
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        yield str(tmp_path)

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

    def fix_dm(dm):
        if dm is None:
            return None
        dm = dm.copy()
        if isinstance(dm.get("background_color"), np.ndarray):
            dm["background_color"] = dm["background_color"].tolist()
        if "viewer_initial_pose" in dm:
            dm["viewer_initial_pose"] = np.round(dm["viewer_initial_pose"][:3, :4].astype(np.float64), 6).flatten().tolist()
        if "viewer_transform" in dm:
            dm["viewer_transform"] = np.round(dm["viewer_transform"][:3, :4].astype(np.float64), 6).flatten().tolist()
        if dm.get("expected_scene_scale") is not None:
            dm["expected_scene_scale"] = round(dm["expected_scene_scale"], 6)
        return dm

    if "dataset_metadata" in info:
        info["dataset_metadata"] = fix_dm(info["dataset_metadata"])
    if "render_dataset_metadata" in info:
        info["render_dataset_metadata"] = fix_dm(info["render_dataset_metadata"])

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
    def fix_dm(dm):
        if dm is None:
            return None
        dm = dm.copy()
        if dm.get("background_color") is not None:
            dm["background_color"] = np.array(dm["background_color"], dtype=np.uint8)
        if "viewer_initial_pose" in dm:
            viewer_initial_pose = np.array(dm["viewer_initial_pose"], dtype=np.float32)
            # Fix legacy format
            if viewer_initial_pose.shape[0] == 4:
                viewer_initial_pose = viewer_initial_pose[:3]
            viewer_initial_pose = viewer_initial_pose.reshape(3, 4)
            dm["viewer_initial_pose"] = viewer_initial_pose
        if "viewer_transform" in dm:
            viewer_transform = np.array(dm["viewer_transform"], dtype=np.float32)
            # Fix legacy format
            if viewer_transform.shape[0] == 4:
                viewer_transform = viewer_transform[:3]
            dm["viewer_transform"] = viewer_transform.reshape(3, 4)
        return dm
    if "dataset_metadata" in info:
        info["dataset_metadata"] = fix_dm(info["dataset_metadata"])
    if "render_dataset_metadata" in info:
        info["render_dataset_metadata"] = fix_dm(info["render_dataset_metadata"])
    return info


def new_nb_info(train_dataset_metadata, 
                method: Method, 
                config_overrides, 
                evaluation_protocol=None,
                resources_utilization_info=None,
                total_train_time=None,
                applied_presets=None):
    dataset_metadata = train_dataset_metadata.copy()
    model_info = method.get_info()

    if evaluation_protocol is None:
        evaluation_protocol = "default"
        evaluation_protocol = dataset_metadata.get("evaluation_protocol", evaluation_protocol)
    if not isinstance(evaluation_protocol, str):
        evaluation_protocol = evaluation_protocol.get_name()
    return {
        "method": model_info["method_id"],
        "nb_version": __version__,
        "num_iterations": model_info["num_iterations"],
        "total_train_time": round(total_train_time, 5) if total_train_time is not None else None,
        "resources_utilization": resources_utilization_info,
        # Date time in ISO 8601 format
        "datetime": datetime.utcnow().isoformat(timespec="seconds"),
        "config_overrides": config_overrides,
        "applied_presets": tuple(sorted(applied_presets)) if applied_presets is not None else None,
        "dataset_metadata": dataset_metadata,
        "evaluation_protocol": evaluation_protocol,

        # Store hparams
        "hparams": method.get_info().get("hparams"),
    }


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


def get_predictions_sha(predictions: str, description: str = "hashing predictions"):
    b = bytearray(128 * 1024)
    mv = memoryview(b)

    def sha256_update(sha, filename):
        with open(filename, "rb", buffering=0) as f:
            for n in iter(lambda: f.readinto(mv), 0):
                sha.update(mv[:n])

    predictions_sha = hashlib.sha256()
    gt_sha = hashlib.sha256()
    with open_any_directory(predictions, "r") as predictions:
        relpaths = [x.relative_to(Path(predictions) / "color") for x in (Path(predictions) / "color").glob("**/*") if x.is_file()]
        relpaths.sort()
        for relname in tqdm(relpaths, desc=description, dynamic_ncols=True):
            sha256_update(predictions_sha, Path(predictions) / "color" / relname)
            sha256_update(gt_sha, Path(predictions) / "gt-color" / relname)
        return (
            predictions_sha.hexdigest(),
            gt_sha.hexdigest(),
        )


def _encode_values(values: List[float]) -> str:
    return base64.b64encode(b"".join(struct.pack("<f", v) for v in values)).decode("ascii")


def numpy_to_base64(array: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, array)
        return base64.b64encode(f.getvalue()).decode("ascii")


def numpy_from_base64(data: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(data)) as f:
        return np.load(f)


def get_metrics_hash(metrics_lists):
    metrics_sha = hashlib.sha256()
    for k in sorted(metrics_lists.keys()):
        metrics_sha.update(k.lower().encode("utf8"))
        values = sorted(metrics_lists[k])
        metrics_sha.update(_encode_values(values).encode("ascii"))
        metrics_sha.update(b"\n")
    return metrics_sha.hexdigest()


def get_checkpoint_sha(path: str) -> str:
    print("===========")
    print(f"Calculating SHA256 for {path}")
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
        if f.name.endswith(".sha") or f.name.endswith(".sha256"):
            continue

        rf = str(f.relative_to(path))

        if os.path.exists(str(f) + ".sha256"):
            with open(str(f) + ".sha256", "rb") as fio:
                sha.update(fio.read().strip())
        elif os.path.exists(str(f) + ".sha"):
            with open(str(f) + ".sha", "rb") as fio:
                sha.update(fio.read().strip())
        else:
            with open(f, "rb", buffering=0) as fio:
                for n in iter(lambda: fio.readinto(mv), 0):
                    sha.update(mv[:n])
            sha_ = hashlib.sha256()
            with open(f, "rb", buffering=0) as fio:
                for n in iter(lambda: fio.readinto(mv), 0):
                    sha_.update(mv[:n])
    return sha.hexdigest()


def get_method_sha(method: Method) -> str:
    with tempfile.TemporaryDirectory() as tmpdir:
        method.save(tmpdir)
        return get_checkpoint_sha(tmpdir)




def serialize_evaluation_results(metrics: Dict, 
                                 metrics_lists, 
                                 predictions_sha: str,
                                 ground_truth_sha: str,
                                 evaluation_protocol: str, 
                                 nb_info: Dict):
    precision = 5
    nb_info = serialize_nb_info(nb_info)
    out = {}
    render_datetime = nb_info.pop("render_datetime", None)
    if render_datetime is not None:
        out["render_datetime"] = render_datetime
    render_version = nb_info.pop("render_version", None)
    if render_version is not None:
        out["render_version"] = render_version
    render_dataset_metadata = nb_info.pop("render_dataset_metadata", None)
    if render_dataset_metadata is not None:
        out["render_dataset_metadata"] = render_dataset_metadata
    out.update({
        "nb_info": nb_info,
        "evaluate_datetime": datetime.utcnow().isoformat(timespec="seconds"),
        "evaluate_version": __version__,
        "metrics": {k: round(v, precision) for k, v in metrics.items()},
        "metrics_raw": {k: _encode_values(metrics_lists[k]) for k in metrics_lists},
        "metrics_sha256": get_metrics_hash(metrics_lists),
        "predictions_sha256": predictions_sha,
        "ground_truth_sha256": ground_truth_sha,
        "evaluation_protocol": evaluation_protocol,
    })
    return out


def save_evaluation_results(file,
                            metrics: Dict, 
                            metrics_lists, 
                            predictions_sha: str,
                            ground_truth_sha: str,
                            evaluation_protocol: str, 
                            nb_info: Dict):
    if isinstance(file, str):
        if os.path.exists(file):
            raise FileExistsError(f"{file} already exists")
        with open(file, "w", encoding="utf8") as f:
            return save_evaluation_results(f, metrics, metrics_lists, predictions_sha, ground_truth_sha, evaluation_protocol, nb_info)

    else:
        out = serialize_evaluation_results(metrics, metrics_lists, predictions_sha, ground_truth_sha, evaluation_protocol, nb_info)
        json.dump(out, file, indent=2)
        return out


def _save_predictions_iterate(output: str, predictions: Iterable[RenderOutput], dataset: Dataset, *, nb_info=None):
    background_color =  dataset["metadata"].get("background_color", None)
    assert background_color is None or background_color.dtype == np.uint8, "background_color must be None or uint8"
    color_space = dataset["metadata"].get("color_space", "srgb")
    expected_scene_scale = dataset["metadata"].get("expected_scene_scale")
    allow_transparency = True

    with ExitStack() as stack:
        if str(output).endswith(".tar.gz"):
            tar = stack.enter_context(tarfile.open(output, "w:gz"))

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
            open_fn = open_fn_tar
        else:
            def open_fn_fs(path):
                path = os.path.join(output, path)
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                return open(path, "wb")

            open_fn = open_fn_fs

        # Write metadata
        from pprint import pprint
        pprint(nb_info)
        with open_fn("info.json") as fp:
            _background_color = background_color
            if isinstance(_background_color, np.ndarray):
                _background_color = _background_color.tolist()
            fp.write(
                json.dumps(
                    serialize_nb_info(
                        {
                            **(nb_info or {}),
                            "render_version": __version__,
                            "render_datetime": datetime.utcnow().isoformat(timespec="seconds"),
                            "render_dataset_metadata": dataset["metadata"],
                        }),
                    indent=2,
                ).encode("utf-8")
            )


        for i, (pred, (w, h)) in enumerate(zip(predictions, _assert_not_none(dataset["cameras"].image_sizes))):
            gt_image = image_to_srgb(dataset["images"][i][:h, :w], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
            pred_image = image_to_srgb(pred["color"], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
            assert gt_image.shape[:-1] == pred_image.shape[:-1], f"gt size {gt_image.shape[:-1]} != pred size {pred_image.shape[:-1]}"
            relative_name = Path(dataset["image_paths"][i])
            if dataset["image_paths_root"] is not None:
                relative_name = relative_name.relative_to(Path(dataset["image_paths_root"]))
            with open_fn(f"gt-color/{relative_name.with_suffix('.png')}") as f:
                save_image(f, gt_image)
            if dataset["masks"] is not None:
                with open_fn(f"mask/{relative_name.with_suffix('.png')}") as f:
                    gt_mask = convert_image_dtype(dataset["masks"][i][:h, :w], np.uint8)
                    save_mask(f, gt_mask)
            with open_fn(f"color/{relative_name.with_suffix('.png')}") as f:
                save_image(f, pred_image)

            with open_fn(f"cameras/{relative_name.with_suffix('.npz')}") as f:
                save_cameras_npz(f, dataset["cameras"][i])
            # with open_fn(f"gt-color/{relative_name.with_suffix('.npy')}") as f:
            #     np.save(f, dataset["images"][i][:h, :w])
            # with open_fn(f"color/{relative_name.with_suffix('.npy')}") as f:
            #     np.save(f, pred["color"])
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


def save_predictions(output: str, predictions: Iterable[RenderOutput], dataset: Dataset, *, nb_info=None):
    for _ in _save_predictions_iterate(output, predictions, dataset, nb_info=nb_info):
        pass


def save_output_artifact(model_path: Union[str, Path], predictions_path: Union[str, Path], metrics_path: Union[str, Path], tensorboard_path: Union[str, Path], output_path: Union[str, Path], validate: bool = True):
    """Prepares artifacts for upload to the NeRF benchmark.

    Args:
        model_path: Path to the model directory.
        predictions_path: Path to the predictions directory/file.
        metrics_path: Path to the metrics file.
        tensorboard_path: Path to the tensorboard events file.
    """
    def _zip_add_dir(zip: zipfile.ZipFile, dirpath: Path, arcname: Optional[str] = None):
        for name in dirpath.glob("**/*"):
            rel_name = name.relative_to(dirpath)
            if arcname is not None:
                rel_name = Path(arcname) / rel_name
            if str(rel_name).startswith("predictions/depth-rgb"):
                continue
            if name.is_dir():
                pass
            elif name.is_file():
                zip.write(str(name), str(rel_name))
            else:
                raise ValueError(f"unknown file type: {name}")

    # Convert to Path objects (if strs)
    model_path = Path(model_path)
    predictions_path = Path(predictions_path)
    metrics_path = Path(metrics_path)
    tensorboard_path = Path(tensorboard_path)
    output_path = Path(output_path)
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
            checkpoint_sha = get_checkpoint_sha(str(model_path))
            predictions_sha, ground_truth_sha = get_predictions_sha(str(predictions_path))
            if metrics.get("predictions_sha256") != predictions_sha:
                raise ValueError("Predictions SHA mismatch")
            if metrics.get("ground_truth_sha256") != ground_truth_sha:
                raise ValueError("Ground truth SHA mismatch")
            if metrics["info"].get("checkpoint_sha256") != checkpoint_sha:
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
                if n is None:
                    break
                sha.update(mv[:n])
        shutil.move(str(artifact_path), str(output_path))
        logging.info(f"artifact {output_path} generated, sha: " + sha.hexdigest())


def save_cameras_npz(file, cameras):
    numpy_arrays = {}
    def extract_array(arr, name):
        numpy_arrays[name] = arr
        return arr
    cameras.apply(extract_array)
    np.savez(file, **numpy_arrays)


def get_torch_checkpoint_sha(checkpoint_data):
    sha = hashlib.sha256()
    def update(d):
        if type(d).__name__ == "Tensor":
            sha.update(d.cpu().numpy().tobytes())
        elif isinstance(d, dict):
            items = sorted(d.items(), key=lambda x: x[0])
            for k, v in items:
                update(k)
                update(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                update(v)
        elif isinstance(d, (int, float)):
            sha.update(struct.pack("<f", d))
        elif isinstance(d, str):
            sha.update(d.encode("utf8"))
        elif d is None:
            sha.update("(None)".encode("utf8"))
        else:
            raise ValueError(f"Unsupported type {type(d)}")
    update(checkpoint_data)
    return sha.hexdigest()


def save_image(file: Union[IO[bytes], str, Path], tensor: np.ndarray):
    if isinstance(file, (str, Path)):
        with open(file, "wb") as f:
            return save_image(f, tensor)
    path = Path(file.name)
    if str(path).endswith(".bin"):
        if tensor.shape[2] < 4:
            tensor = np.dstack((tensor, np.ones([tensor.shape[0], tensor.shape[1], 4 - tensor.shape[2]])))
        file.write(struct.pack("<ii", tensor.shape[0], tensor.shape[1]))
        file.write(tensor.astype(np.float16).tobytes())
    else:
        from PIL import Image

        tensor = convert_image_dtype(tensor, np.uint8)
        image = Image.fromarray(tensor)
        image.save(file, format="png")


def save_mask(file: Union[IO[bytes], str, Path], tensor: np.ndarray):
    if isinstance(file, (str, Path)):
        with open(file, "wb") as f:
            return save_image(f, tensor)
    path = Path(file.name)
    if str(path).endswith(".bin"):
        assert tensor.ndim == 2
        file.write(struct.pack("<ii", tensor.shape[0], tensor.shape[1]))
        file.write(tensor.astype(np.float16).tobytes())
    else:
        assert tensor.ndim == 2
        from PIL import Image

        tensor = convert_image_dtype(tensor, np.uint8)
        image = Image.fromarray(tensor)
        image.save(file, format="png")


def read_image(file: Union[IO[bytes], str, Path]) -> np.ndarray:
    if isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            return read_image(f)
    path = Path(file.name)
    if str(path).endswith(".bin"):
        h, w = struct.unpack("<II", file.read(8))
        itemsize = 2
        img = np.frombuffer(file.read(h * w * 4 * itemsize), dtype=np.float16, count=h * w * 4, offset=8).reshape([h, w, 4])
        assert img.itemsize == itemsize
        return img.astype(np.float32)
    else:
        from PIL import Image

        return np.array(Image.open(file))


def read_mask(file: Union[IO[bytes], str, Path]) -> np.ndarray:
    if isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            return read_image(f)
    path = Path(file.name)
    if str(path).endswith(".bin"):
        h, w = struct.unpack("<II", file.read(8))
        itemsize = 2
        img = np.frombuffer(file.read(h * w * itemsize), dtype=np.float16, count=h * w, offset=8).reshape([h, w])
        assert img.itemsize == itemsize
        return img.astype(np.float32)
    else:
        from PIL import Image

        with Image.open(file) as img:
            w, h = img.size
            array = np.array(img)
            return convert_image_dtype(array, np.float32).reshape((h, w))


def save_depth(file: Union[IO[bytes], str, Path], tensor: np.ndarray):
    if isinstance(file, (str, Path)):
        with open(file, "wb") as f:
            return save_depth(f, tensor)
    path = Path(file.name)
    assert str(path).endswith(".bin")
    file.write(struct.pack("<ii", tensor.shape[0], tensor.shape[1]))
    file.write(tensor.astype(np.float16).tobytes())
