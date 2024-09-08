import json
import os
import csv
import logging
from pathlib import Path
import shutil
import tempfile
from typing import Union, cast, Dict, Iterable
import tarfile

import numpy as np
import requests
from tqdm import tqdm

from nerfbaselines import Dataset, EvaluationProtocol, Method, RenderOutput, DatasetNotFoundError
from ..utils import image_to_srgb
from ._common import dataset_index_select
from .colmap import load_colmap_dataset

DATASET_NAME = "phototourism"


def load_phototourism_dataset(path: Union[Path, str], split: str, use_nerfw_split=None, **kwargs):
    path = Path(path)
    use_nerfw_split = use_nerfw_split if use_nerfw_split is not None else True

    # Load phototourism dataset
    images_path = "images"
    split_list = None
    if use_nerfw_split:
        if (path / "nerfw_split.csv").exists():
            with (path / "nerfw_split.csv").open() as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)
                split_list = [x[0] for x in reader if x[1] and x[2] == split]
                assert len(split_list) > 0, f"{split} list is empty"
        else:
            logging.warning(
                f"NeRF-W test list not found for. Using a standard COLMAP split!"
            )

    # We then select the same images as in the LLFF multinerf dataset loader
    dataset = load_colmap_dataset(
        path, 
        images_path=images_path,
        colmap_path="sparse",
        split=None, **kwargs
    )

    dataset_len = len(dataset["image_paths"])
    if split_list is not None:
        indices = np.array(
            [i for i, x in enumerate(dataset["image_paths"]) if Path(x).name in split_list]
        )
        assert len(indices) > 0, f"No images found in {split} list"
        logging.info(f"Using {len(indices)}/{dataset_len} images from the NeRF-W {split} list")
    else:
        all_indices = np.arange(dataset_len)
        llffhold = 8
        if split == "train":
            indices = all_indices % llffhold != 0
        else:
            indices = all_indices % llffhold == 0
        logging.info(f"Using {len(indices)}/{dataset_len} images using LLFF-hold of {llffhold}")
    return dataset_index_select(dataset, indices)


# https://www.cs.ubc.ca/%7Ekmyi/imw2020/data.html
# We further removed the hagia_sophia_interior, westminster_abbey in 2022 due to data inconsistencies.
# We removed the prague_old_town in 2021 due to data inconsistencies.

_phototourism_downloads = {
    "brandenburg-gate": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/brandenburg_gate.tar.gz",
    "buckingham-palace": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/buckingham_palace.tar.gz",
    "colosseum-exterior": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/colosseum_exterior.tar.gz",
    "grand-palace-brussels": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/grand_place_brussels.tar.gz",
    "notre-dame-facade": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/notre_dame_front_facade.tar.gz",
    "westminster-palace": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/palace_of_westminster.tar.gz",
    "pantheon-exterior": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/pantheon_exterior.tar.gz",
    "taj-mahal": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/taj_mahal.tar.gz",
    "temple-nara": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/temple_nara_japan.tar.gz",
    "trevi-fountain": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/trevi_fountain.tar.gz",
    "sacre-coeur": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/sacre_coeur.tar.gz",
    # "prague-old-town": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/prague_old_town.tar.gz",
    "hagia-sophia": "https://www.cs.ubc.ca/research/kmyi_data/imw2020/TrainingData/hagia_sophia.tar.gz",
}
SCENES = list(_phototourism_downloads.keys())

_split_lists = {
    "brandenburg-gate": "https://nerf-w.github.io/data/selected_images/brandenburg.tsv",
    "trevi-fountain": "https://nerf-w.github.io/data/selected_images/trevi.tsv",
    "sacre-coeur": "https://nerf-w.github.io/data/selected_images/sacre.tsv",
    # "prague-old-town": "https://nerf-w.github.io/data/selected_images/prague.tsv",
    "hagia-sophia": "https://nerf-w.github.io/data/selected_images/hagia.tsv",
    "taj-mahal": "https://nerf-w.github.io/data/selected_images/taj_mahal.tsv",
}


def download_phototourism_dataset(path: str, output: Union[Path, str]):
    output = Path(output)
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError(
            f"Dataset path must be equal to '{DATASET_NAME}' or must start with '{DATASET_NAME}/'."
        )

    if path == "phototourism":
        for x in _split_lists:
            download_phototourism_dataset(f"phototourism/{x}", output / x)
        return

    capture_name = path.split("/")[1]
    if capture_name not in _phototourism_downloads:
        raise DatasetNotFoundError(
            f"Capture '{capture_name}' not a valid {DATASET_NAME} scene."
        )

    if output.exists():
        logging.info(f"Dataset {DATASET_NAME}/{capture_name} already exists in {output}")
        return

    url = _phototourism_downloads[capture_name]
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(
        total=total_size_in_bytes,
        unit="iB",
        unit_scale=True,
        desc=f"Downloading {url.split('/')[-1]}", 
        dynamic_ncols=True,
    )
    with tempfile.TemporaryFile("rb+") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
        file.flush()
        file.seek(0)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:  # noqa: PLR1714
            logging.error(
                f"Failed to download dataset. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes."
            )

        has_any = False
        prefix = url.split("/")[-1].split(".")[0] + "/dense"
        with tarfile.open(fileobj=file, mode="r:gz") as z:
            output_tmp = output.with_suffix(".tmp")
            output_tmp.mkdir(exist_ok=True, parents=True)
            def members(tf):
                nonlocal has_any
                for member in tf.getmembers():
                    if not member.path.startswith(prefix + "/"):
                        continue
                    has_any = True
                    member.path = member.path[len(prefix) + 1 :]
                    yield member

            z.extractall(output_tmp, members=members(z))
            with open(os.path.join(str(output_tmp), "nb-info.json"), "w", encoding="utf8") as f2:
                json.dump({
                    "loader": load_phototourism_dataset.__module__ + ":" + load_phototourism_dataset.__name__,
                    "id": DATASET_NAME,
                    "scene": capture_name,
                    "type": None,
                    "evaluation_protocol": "nerfw",
                }, f2)
            shutil.rmtree(output, ignore_errors=True)
            if not has_any:
                raise RuntimeError(f"Capture '{capture_name}' not found in {url}.")
            shutil.move(str(output_tmp), str(output))

    # Download test list if available
    if capture_name in _split_lists:
        split_list_url = _split_lists[capture_name]
        response = requests.get(split_list_url)
        response.raise_for_status()
        with open(output / "nerfw_split.csv", "w") as f:
            f.write(response.text)

    logging.info(f"Downloaded {DATASET_NAME}/{capture_name} to {output}")


def horizontal_half_dataset(dataset: Dataset, left: bool = True) -> Dataset:
    intrinsics = dataset["cameras"].intrinsics.copy()
    image_sizes = dataset["cameras"].image_sizes.copy()
    image_sizes[:, 0] //= 2
    if left:
        image_sizes[:, 0] = dataset["cameras"].image_sizes[:, 0] - image_sizes[:, 0]
    if not left:
        intrinsics[:, 2] -= image_sizes[:, 0]
    def get_slice(img, w):
        if left:
            return img[:, :w]
        else:
            return img[:, -w:]
    dataset = dataset.copy()
    dataset.update(cast(Dataset, dict(
        cameras=dataset["cameras"].replace(
            intrinsics=intrinsics,
            image_sizes=image_sizes),
        images=[get_slice(img, w) for img, w in zip(dataset["images"], image_sizes[:, 0])],
        sampling_masks=[get_slice(mask, w) for mask, w in zip(dataset["sampling_masks"], image_sizes[:, 0])] if dataset["sampling_masks"] is not None else None,
    )))
    return dataset


class NerfWEvaluationProtocol(EvaluationProtocol):
    def __init__(self):
        from nerfbaselines.evaluation import compute_metrics
        self._compute_metrics = compute_metrics

    def get_name(self):
        return "nerfw"

    def render(self, method: Method, dataset: Dataset, *, options=None) -> Iterable[RenderOutput]:
        optimization_dataset = horizontal_half_dataset(dataset, left=True)
        optim_iterator = method.optimize_embeddings(optimization_dataset)
        if optim_iterator is None:
            # Method does not support optimization
            for pred in method.render(dataset["cameras"], options=options):
                yield pred
            return

        for i, optim_result in enumerate(optim_iterator):
            # Render with the optimzied result
            for pred in method.render(dataset["cameras"][i:i+1], embeddings=[optim_result["embedding"]], options=options):
                yield pred

    def evaluate(self, predictions: Iterable[RenderOutput], dataset: Dataset) -> Iterable[Dict[str, Union[float, int]]]:
        for i, prediction in enumerate(predictions):
            gt = dataset["images"][i]
            color = prediction["color"]

            background_color = dataset["metadata"].get("background_color", None)
            color_srgb = image_to_srgb(color, np.uint8, color_space="srgb", background_color=background_color)
            gt_srgb = image_to_srgb(gt, np.uint8, color_space="srgb", background_color=background_color)
            w = gt_srgb.shape[1]
            metrics = self._compute_metrics(color_srgb[:, (w//2):], gt_srgb[:, (w//2):])
            yield metrics

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        acc = {}
        for i, data in enumerate(metrics):
            for k, v in data.items():
                acc[k] = (acc.get(k, 0) * i + v) / (i + 1)
        return acc

__all__ = ["load_phototourism_dataset", "download_phototourism_dataset", "NerfWEvaluationProtocol"]
