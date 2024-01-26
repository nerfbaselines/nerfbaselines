import logging
import shutil
import requests
from pathlib import Path
import zipfile
from tqdm import tqdm
import tempfile
from ..types import Dataset
from ._common import DatasetNotFoundError, single
from .colmap import load_colmap_dataset


DATASET_NAME = "tanksandtemples"
BASE_URL = "https://data.ciirc.cvut.cz/public/projects/2023NerfBaselines/datasets/tanksandtemples"
SCENES = {
    "train": f"{BASE_URL}/train_2down.zip",
    "truck": f"{BASE_URL}/truck_2down.zip",
}


def load_tanksandtemples_dataset(path: Path, split: str, downscale_factor: int = 2, **kwargs) -> Dataset:
    if split:
        assert split in {"train", "test"}
    if DATASET_NAME not in str(path) or not any(s in str(path) for s in SCENES):
        raise DatasetNotFoundError(f"{DATASET_NAME} and {set(SCENES.keys())} is missing from the dataset path: {path}")

    # Load TT dataset
    images_path = Path("images") if downscale_factor == 1 else Path(f"images_{downscale_factor}")
    scene = single(x for x in SCENES if x in str(path))

    dataset: Dataset = load_colmap_dataset(path, images_path=images_path, split=None, **kwargs)
    dataset.metadata["name"] = DATASET_NAME
    dataset.metadata["scene"] = scene
    return dataset


def download_tanksandtemples_dataset(path: str, output: Path) -> None:
    output = Path(output)
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError("Dataset path must be equal to 'tanksandtemples' or must start with 'tanksandtemples/'.")

    if path == DATASET_NAME:
        for scene in SCENES:
            download_tanksandtemples_dataset(f"{DATASET_NAME}/{scene}", output/scene)
        return

    scene = path.split("/")[-1]
    if scene not in SCENES:
        raise RuntimeError(f"Unknown scene {scene}")
    url = SCENES[scene]
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Downloading {url.split('/')[-1]}")
    with tempfile.TemporaryFile("rb+") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
        file.flush()
        file.seek(0)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            logging.error(f"Failed to download dataset. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes.")

        with zipfile.ZipFile(file) as z:
            output_tmp = output.with_suffix(".tmp")
            output_tmp.mkdir(exist_ok=True, parents=True)
            for info in z.infolist():
                if not info.filename.startswith(scene + "/"):
                    continue
                relname = info.filename[len(scene) + 1 :]
                target = output_tmp / relname
                target.parent.mkdir(exist_ok=True, parents=True)
                if info.is_dir():
                    target.mkdir(exist_ok=True, parents=True)
                else:
                    with z.open(info) as source, open(target, "wb") as target:
                        shutil.copyfileobj(source, target)

            shutil.rmtree(output, ignore_errors=True)
            shutil.move(output_tmp, output)
            logging.info(f"Downloaded {DATASET_NAME}/{scene} to {output}")
