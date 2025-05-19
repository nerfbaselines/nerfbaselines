import io
import os
import csv
import logging
from pathlib import Path
from typing import Union, cast, Dict, Iterable
from functools import partial

import numpy as np

from nerfbaselines.io import wget
from nerfbaselines import (
    Dataset, EvaluationProtocol, Method, 
    RenderOutput, DatasetNotFoundError, RenderOptions
)
from ..utils import image_to_srgb
from ._common import dataset_index_select, download_dataset_wrapper, download_archive_dataset
from .colmap import load_colmap_dataset


DATASET_NAME = "phototourism"


def load_phototourism_dataset(path, *args, **kwargs):
    del args, kwargs
    raise RuntimeError(f"The dataset was likely downloaded with an older version of NerfBaselines. Please remove `{path}` and try again.")


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


def _add_split_lists(output, scene):
    # Download test list if available
    if scene in _split_lists:
        split_list_url = _split_lists[scene]
        with wget(split_list_url) as response:
            with open(os.path.join(output, "nerfw_split.csv"), "w", encoding="utf8") as f:
                with io.TextIOWrapper(response) as fin:
                    f.write(fin.read())

        split_lists = {}
        for split in ["train", "test"]:
            with open(os.path.join(output, "nerfw_split.csv"), "r", encoding="utf8") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)
                split_lists[split] = [x[0] for x in reader if x[1] and x[2] == split]
                assert len(split_lists[split]) > 0, f"{split} list is empty"

        # Load phototourism dataset
        # We then select the same images as in the LLFF multinerf dataset loader
        dataset = load_colmap_dataset(output, images_path="images", colmap_path="sparse", split=None)
        dataset_len = len(dataset["image_paths"])
        indices = {}
        for split in split_lists:
            indices[split] = np.array(
                [i for i, x in enumerate(dataset["image_paths"]) if Path(x).name in split_lists[split]]
            )
            assert len(indices[split]) > 0, f"No images found in {split} list"
            logging.info(f"Using {len(indices)}/{dataset_len} images from the NeRF-W {split} list")

            # Save the lists
            with open(os.path.join(output, f"{split}_list.txt"), "w", encoding="utf8") as f:
                for img_name in dataset_index_select(dataset, indices[split])["image_paths"]:
                    f.write(os.path.relpath(img_name, dataset["image_paths_root"]) + "\n")
    else:
        logging.warning(f"Split list not found for {scene}, will use LLFF-hold split")


@download_dataset_wrapper(_split_lists.keys(), DATASET_NAME)
def download_phototourism_dataset(path: str, output: str):
    assert "/" in path

    if os.path.exists(output):
        logging.info(f"Dataset {path} already exists in {output}")
        return

    dataset_name, scene = path.split("/")
    if scene not in _phototourism_downloads:
        raise DatasetNotFoundError(
            f"Capture '{scene}' not a valid {dataset_name} scene."
        )

    url = _phototourism_downloads[scene]
    archive_prefix = url.split("/")[-1].split(".")[0] + "/dense/"
    nb_info = {
        "loader": "colmap",
        "loader_kwargs": {
            "images_path": "images",
            "colmap_path": "sparse",
        },
        "id": dataset_name,
        "scene": scene,
        "type": None,
        "evaluation_protocol": "nerfw",
    }
    download_archive_dataset(url, output, 
                             archive_prefix=archive_prefix, 
                             nb_info=nb_info, 
                             callback=partial(_add_split_lists, scene=scene),
                             file_type="tar.gz")
    logging.info(f"Downloaded {path} to {output}")


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
        masks=[get_slice(mask, w) for mask, w in zip(dataset["masks"], image_sizes[:, 0])] if dataset["masks"] is not None else None,
    )))
    return dataset


class NerfWEvaluationProtocol(EvaluationProtocol):
    def __init__(self):
        from nerfbaselines.evaluation import compute_metrics
        self._compute_metrics = compute_metrics

    def get_name(self):
        return "nerfw"

    def render(self, method: Method, dataset: Dataset, *, options=None) -> RenderOutput:
        dataset["cameras"].item()  # Assert single camera
        optimization_dataset = horizontal_half_dataset(dataset, left=True)
        embedding = (options or {}).get("embedding", None)
        optim_result = None
        try:
            optim_result = method.optimize_embedding(optimization_dataset, embedding=embedding)
            embedding = optim_result["embedding"]
        except NotImplementedError as e:
            logging.debug(e)
            method_id = method.get_method_info()["method_id"]
            logging.warning(f"Method {method_id} does not support camera embedding optimization.")

        new_options: RenderOptions = {
            **(options or {}),
            "embedding": embedding,
        }
        return method.render(dataset["cameras"], options=new_options)

    def evaluate(self, predictions: RenderOutput, dataset: Dataset) -> Dict[str, Union[float, int]]:
        assert len(dataset["images"]) == 1, "EvaluationProtocol.evaluate must be run on individual samples (a dataset with a single image)"
        gt = dataset["images"][0]
        color = predictions["color"]

        background_color = dataset["metadata"].get("background_color", None)
        color_srgb = image_to_srgb(color, np.uint8, color_space="srgb", background_color=background_color)
        gt_srgb = image_to_srgb(gt, np.uint8, color_space="srgb", background_color=background_color)
        w = gt_srgb.shape[1]
        metrics = self._compute_metrics(color_srgb[:, (w//2):], gt_srgb[:, (w//2):])
        return metrics

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        acc = {}
        for i, data in enumerate(metrics):
            for k, v in data.items():
                acc[k] = (acc.get(k, 0) * i + v) / (i + 1)
        return acc

__all__ = ["download_phototourism_dataset", "NerfWEvaluationProtocol"]
