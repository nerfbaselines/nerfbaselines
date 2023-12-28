import time
import tarfile
import os
from typing import Union, Iterator
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


OpenMode = Literal["r", "w"]


@contextlib.contextmanager
def open_any(path: Union[str, Path, BinaryIO], mode: OpenMode = "r") -> Iterator[BinaryIO]:
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
                        with tar.extractfile(rest) as f:
                            yield f
                    elif mode == "w":
                        _, extension = os.path.split(rest)
                        with tempfile.TemporaryFile("wb", suffix=extension) as tmp:
                            yield tmp
                            tmp.flush()
                            tmp.seek(0)
                            tar.addfile(
                                tarinfo=tarfile.TarInfo(
                                    name=rest,
                                    mtime=int(time.time()),
                                    mode=0o644,
                                    size=tmp.tell(),
                                ),
                                fileobj=tmp,
                            )

            else:
                # Extract from zip
                with zipfile.ZipFile(f, mode=mode) as zip, zip.open("/".join(components[zip_parts[-1] + 1 :]), mode=mode) as f:
                    yield f
        return

    # Download from url
    if path.startswith("http://") or path.startswith("https://"):
        assert mode == "r", "Only reading from remote files is supported."
        response = requests.get(path, stream=True)
        response.raise_for_status()
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Downloading {path}")
        name = path.split("/")[-1]
        with tempfile.TemporaryFile("rb+", suffix=name) as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
            file.flush()
            file.seek(0)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                logging.error(f"Failed to download {path}. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes.")
            yield file
        return

    # Normal file
    if mode == "w":
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, mode=mode + "b") as f:
        yield f


@contextlib.contextmanager
def open_any_directory(path: Union[str, Path], mode: OpenMode = "r") -> Iterator[Path]:
    path = str(path)

    components = path.split("/")
    compressed_parts = [i for i, c in enumerate(components) if c.endswith(".zip") or c.endswith(".tar.gz")]
    if compressed_parts:
        with open_any("/".join(components[: compressed_parts[-1] + 1]), mode=mode) as f, tempfile.TemporaryDirectory() as tmpdir:
            rest = "/".join(components[compressed_parts[-1] + 1 :])
            if components[compressed_parts[-1]].endswith(".tar.gz"):
                with tarfile.open(fileobj=f, mode=mode + ":gz") as tar:
                    if mode == "r":
                        for member in tar.getmembers():
                            if not member.name.startswith(rest):
                                continue
                            if member.isdir():
                                os.makedirs(os.path.join(tmpdir, member.name), exist_ok=True)
                            else:
                                tar.extract(member, tmpdir)
                        yield Path(tmpdir) / rest
                    elif mode == "w":
                        tmp_path = Path(tmpdir) / rest
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        yield Path(tmpdir) / rest

                        for root, dirs, files in os.walk(tmp_path):
                            for dir in dirs:
                                tar.add(os.path.join(root, dir), arcname=os.path.relpath(os.path.join(root, dir), tmp_path))
                            for file in files:
                                tar.add(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), tmp_path))
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
                                os.makedirs(os.path.join(tmpdir, member.filename), exist_ok=True)
                            else:
                                zip.extract(member, tmpdir)
                        yield Path(tmpdir) / rest
                    elif mode == "w":
                        tmp_path = Path(tmpdir) / rest
                        tmp_path.mkdir(parents=True, exist_ok=True)
                        yield Path(tmpdir) / rest

                        for root, dirs, files in os.walk(tmp_path):
                            for dir in dirs:
                                zip.write(os.path.join(root, dir), arcname=os.path.relpath(os.path.join(root, dir), tmp_path))
                            for file in files:
                                zip.write(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), tmp_path))
                    else:
                        raise RuntimeError(f"Unsupported mode {mode} for zip files.")
        return

    if path.startswith("http://") or path.startswith("https://"):
        raise RuntimeError("Only tar.gz and zip files are supported for remote directories.")

    # Normal file
    Path(path).mkdir(parents=True, exist_ok=True)
    yield Path(path)
    return
