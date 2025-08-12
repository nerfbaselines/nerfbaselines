import json
import os
from typing import Union
import shutil
from pathlib import Path
import numpy as np
from nerfbaselines import DatasetNotFoundError
from nerfbaselines import NB_PREFIX
from .mipnerf360 import download_mipnerf360_dataset, SCENES, VERSION, atomic_output


DATASET_NAME = "mipnerf360-sparse"


def download_mipnerf360_sparse_dataset(path: str, output: Union[Path, str]):
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError(f"Dataset path must be equal to '{DATASET_NAME}' or must start with '{DATASET_NAME}/'.")
    output = Path(output)

    # First we download mipnerf360 dataset
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
        with atomic_output(output) as tmpoutput:
            source = os.path.join(mipnerf360_path, mipnerf360_scene)

            # Now we copy sparse directory
            for x in os.listdir(source):
                if x.startswith("sparse"):
                    shutil.copytree(os.path.join(source, x), os.path.join(tmpoutput, x), dirs_exist_ok=True)
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


download_mipnerf360_sparse_dataset.version = VERSION  # type: ignore
__all__ = ["download_mipnerf360_sparse_dataset"]
