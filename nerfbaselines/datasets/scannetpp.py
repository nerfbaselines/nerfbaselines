import zipfile
import json
import tempfile
import shutil
import sys
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
from ._common import dataset_index_select, download_dataset_wrapper, download_archive_dataset, atomic_output
from .colmap import load_colmap_dataset


DATASET_NAME = "scannetpp"
root_url = "https://kaldir.vc.in.tum.de/scannetpp/download/v2?token={token}&file={filepath}"
scene_ids = [
]
download_assets = ["dslr_train_test_lists_path", "dslr_resized_dir", "dslr_resized_mask_dir", "dslr_colmap_dir"]
zipped_assets = ["dslr_resized_dir", "dslr_resized_mask_dir", "dslr_colmap_dir"]


def run_with_scannet_token(fn):
    # Write dataset token
    token_path = os.path.abspath(__file__) + ".token"
    if "SCANNETPP_TOKEN" in os.environ:
        token = os.environ["SCANNETPP_TOKEN"]
        print("Using SCANNETPP_TOKEN from environment variable")
        if os.path.exists(token_path):
            try:
                with open(token_path, "w") as f:
                    f.write(token)
            except IOError as e:
                logging.warning(f"Failed to cache token to {token_path}: {e}")
    elif os.path.exists(token_path):
        with open(token_path, "r") as f:
            token = f.read().strip()
        print(f"Using cached SCANNETPP_TOKEN from {token_path}")
    else:
        # This could be the first time using the dataset. We need to inform the user about the way
        # to obtain a token.
        print(f"""======== ScanNet++ dataset ========
To use the ScanNet++ dataset, you need accept the terms of use and obtain a token.
The token can be obtained from the ScanNet++ website:
https://kaldir.vc.in.tum.de/scannetpp

After obtaining the token, please set the environment variable SCANNETPP_TOKEN to the value of the token and run the script again.
""")
        sys.exit(1)

    def new_fn(*args, **kwargs):
        return fn(token=token, *args, **kwargs)
    return new_fn


@download_dataset_wrapper(scene_ids, DATASET_NAME)
@run_with_scannet_token
def download_scannetpp_dataset(token: str, path: str, output: str):
    assert "/" in path

    if os.path.exists(output):
        logging.info(f"Dataset {path} already exists in {output}")
        return

    dataset_name, scene = path.split("/")
    if scene not in scene_ids:
        raise DatasetNotFoundError(
            f"Capture '{scene}' not a valid {dataset_name} scene."
        )

    def download_scene(output):
        # Download the images
        wget(
            root_url.format(token=token, filepath=f"data/{scene}/dslr/resized_images.zip"),
            os.path.join(output, "resized_images.zip")
        )
        # Download the masks
        wget(
            root_url.format(token=token, filepath=f"data/{scene}/dslr/resized_anon_masks.zip"),
            os.path.join(output, "resized_anon_masks.zip")
        )
        # Download the colmap directory
        wget(
            root_url.format(token=token, filepath=f"data/{scene}/dslr/colmap.zip"),
            os.path.join(output, "colmap.zip")
        )
        # Download the train/test lists
        wget(
            root_url.format(token=token, filepath=f"data/{scene}/dslr/train_test_lists.json"),
            os.path.join(output, "train_test_lists.json")
        )

        # Extract the downloaded files
        os.makedirs(os.path.join(output, "dslr"), exist_ok=True)
        with zipfile.ZipFile(os.path.join(output, "resized_images.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output, "dslr", "resized_images"))
        with zipfile.ZipFile(os.path.join(output, "resized_anon_masks.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output, "dslr", "resized_anon_masks"))
        with zipfile.ZipFile(os.path.join(output, "colmap.zip"), 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output, "dslr", "colmap"))
        with open(os.path.join(output, "train_test_lists.json"), "r", encoding="utf8") as f:
            train_test_lists = json.load(f)

        # Build test/train list
        ...

        # We can remove used files
        os.remove(os.path.join(output, "resized_images.zip"))
        os.remove(os.path.join(output, "resized_anon_masks.zip"))
        os.remove(os.path.join(output, "colmap.zip"))
        os.remove(os.path.join(output, "train_test_lists.json"))

        # Download the dataset
        nb_info = {
            "loader": "colmap",
            "loader_kwargs": {
                "images_path": "images",
                "colmap_path": "sparse",
            },
            "id": dataset_name,
            "scene": scene,
            "type": None,
            "evaluation_protocol": "default",
        }
        with open(os.path.join(output, "nb-info.json"), "w", encoding="utf8") as f:
            json.dump(nb_info, f, indent=2)

    atomic_output(output)(download_scene)
    logging.info(f"Downloaded {path} to {output}")

