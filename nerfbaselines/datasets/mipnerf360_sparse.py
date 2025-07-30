import json
import os
from typing import List, Tuple, Union
import logging
from itertools import groupby
import shutil
from pathlib import Path
import numpy as np
import zipfile
import tempfile
from nerfbaselines import DatasetNotFoundError
from nerfbaselines.io import wget
from nerfbaselines import NB_PREFIX
import tempfile
from ._common import dataset_index_select
from .colmap import load_colmap_dataset


DATASET_NAME = "mipnerf360-sparse"


def _save_colmap_splits(output):
    with open(os.path.join(output, "nb-info.json"), "r", encoding="utf8") as f:
        dataset_info = json.load(f)

    dataset = load_colmap_dataset(output, **dataset_info["loader_kwargs"])
    image_names = dataset["image_paths"]
    inds = np.argsort(image_names)

    all_indices = np.arange(len(dataset["image_paths"]))
    llffhold = 8
    indices = {}
    indices["train"] = inds[all_indices % llffhold != 0]
    indices["test"] = inds[all_indices % llffhold == 0]
    for split in indices:
        with open(os.path.join(output, f"{split}_list.txt"), "w", encoding="utf8") as f:
            for img_name in dataset_index_select(dataset, indices[split])["image_paths"]:
                f.write(os.path.relpath(img_name, dataset["image_paths_root"]) + "\n")


def download_mipnerf360_sparse_dataset(path: str, output: Union[Path, str]):
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError(f"Dataset path must be equal to '{DATASET_NAME}' or must start with '{DATASET_NAME}/'.")
    output = Path(output)

    # First we download mipnerf360 dataset
    from .mipnerf360 import download_mipnerf360_dataset, SCENES
    mipnerf360_path = str(Path(NB_PREFIX) / "datasets" / "mipnerf360")

    if path == DATASET_NAME:
        if not os.path.exists(mipnerf360_path):
            download_mipnerf360_dataset("mipnerf360", mipnerf360_path)

    if "/" in path:
        scenes = [(path.split("/")[-1], output)]
        if not path.endswith("-n12") and not path.endswith("-n24"):
            raise DatasetNotFoundError(f"Invalid scene path: {path}. It should have format '{{scene}}-n12' or '{{scene}}-n24'.")

    else:
        scenes = (
            [(f"{s}-n12", output/(f"{s}-n12")) for s in SCENES] + 
            [(f"{s}-n24", output/(f"{s}-n24")) for s in SCENES]
        )
    for scene, output in scenes:
        mipnerf360_scene = scene.split("-")[0]
        num_views = int(scene.split("-")[1][1:])
        if not os.path.exists(os.path.join(mipnerf360_path, mipnerf360_scene)):
            download_mipnerf360_dataset(f"mipnerf360/{mipnerf360_scene}", os.path.join(mipnerf360_path, mipnerf360_scene))

        # We have mipnerf360 dataset ready, now we simply copy the relevant part
        os.makedirs(output, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=os.path.dirname(output)) as tmpdir:
            tmpoutput = os.path.join(tmpdir, "output")
            source = os.path.join(mipnerf360_path, mipnerf360_scene)

            # Now we copy sparse directory
            shutil.copytree(os.path.join(source, "sparse"), os.path.join(tmpoutput, "sparse"), dirs_exist_ok=True)
            shutil.copy(os.path.join(source, "test_list.txt"), os.path.join(tmpoutput, "test_list.txt"))
            # Build train list
            train_images = [x.strip() for x in open(os.path.join(source, "train_list.txt"), "r", encoding="utf8")]
            train_images.sort()  # Sort to ensure consistent order
            # Select subset of images
            indices = np.linspace(0, len(train_images) - 1, num_views, dtype=int)
            train_images = [train_images[i] for i in indices]
            with open(os.path.join(tmpoutput, "train_list.txt"), "w", encoding="utf8") as f:
                for img_name in train_images:
                    f.write(img_name + "\n")
            all_images = train_images + [x.strip() for x in open(os.path.join(source, "test_list.txt"), "r", encoding="utf8")]
            
            # Copy all images
            images_dirs = [x for x in os.listdir(source) if x.startswith("images")]
            for img_name in all_images:
                for images_dir in images_dirs:
                    src_image_path = os.path.join(source, images_dir, img_name)
                    os.makedirs(os.path.join(tmpoutput, images_dir), exist_ok=True)
                    shutil.copy(src_image_path, os.path.join(tmpoutput, images_dir, img_name))

            # Write the nb-info.json
            with open(os.path.join(source, "nb-info.json"), "r", encoding="utf8") as fsource, open(os.path.join(tmpoutput, "nb-info.json"), "w", encoding="utf8") as fout:
                nb_info = json.load(fsource)
                nb_info["id"] = DATASET_NAME
                nb_info["scene"] = scene
                json.dump(nb_info, fout, indent=2, ensure_ascii=False)

            # Write the output
            if os.path.exists(output):
                shutil.rmtree(output)
            shutil.move(tmpoutput, output)


__all__ = ["download_mipnerf360_sparse_dataset"]
