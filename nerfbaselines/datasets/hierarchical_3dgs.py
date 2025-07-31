import tqdm
import json
import os
from typing import Union
import logging
import shutil
from pathlib import Path
import zipfile
import tempfile
from nerfbaselines import DatasetNotFoundError
from nerfbaselines.io import wget
from nerfbaselines.evaluation import NerfEvaluationProtocol
from PIL import Image


DATASET_NAME = "hierarchical-3dgs"
_scenes_links = {
    "smallcity": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/standalone_chunks/small_city.zip",
    "campus": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/datasets/standalone_chunks/campus.zip",
}
_scenes_prefix = {
    "smallcity": "small_city/",
    "campus": "campus/",
}
SCENES = set(_scenes_links.keys())


def _noop(*args, **kwargs): del args, kwargs


def download_hierarchical_3dgs_dataset(path: str, output: Union[Path, str]):
    if path == DATASET_NAME:
        # We will download all faster here
        for x in _scenes_links:
            download_hierarchical_3dgs_dataset(f"{DATASET_NAME}/{x}", Path(output) / x)

    output = Path(output)
    if not path.startswith(f"{DATASET_NAME}/") and path != DATASET_NAME:
        raise DatasetNotFoundError(f"Dataset path must be equal to '{DATASET_NAME}' or must start with '{DATASET_NAME}/'.")

    scene_name = path.split("/")[-1]
    if scene_name not in SCENES:
        raise DatasetNotFoundError(f"Unknown scene '{scene_name}'. Available scenes: {', '.join(SCENES)}")

    output.parent.mkdir(exist_ok=True, parents=True)
    tmpdir = tempfile.TemporaryDirectory(dir=output.parent)
    with tmpdir, tempfile.TemporaryFile("rb+") as file:
        output_tmp = Path(tmpdir.name)
        url = _scenes_links[path.split("/")[-1]]
        wget(url, file, desc=f"Downloading {scene_name}")
        file.seek(0)
        all_images = []
        with zipfile.ZipFile(file) as z:
            prefix = _scenes_prefix.get(scene_name, scene_name + "/")
            for info in tqdm.tqdm(z.infolist(), desc=f"Extracting {scene_name}"):
                if not info.filename.startswith(prefix):
                    continue
                # z.extract(name, output_tmp)
                relname = info.filename[len(prefix):]
                if (not relname.startswith("images") and
                    not relname.startswith("sparse")):
                    continue
                if relname == "sparse/0/test.txt":
                    relname = "test_list.txt"
                
                target = output_tmp / relname
                target.parent.mkdir(exist_ok=True, parents=True)
                if info.is_dir():
                    target.mkdir(exist_ok=True, parents=True)
                else:
                    if relname.startswith("images/"):
                        imgname = relname[len("images/"):]
                        all_images.append(imgname)

                        target_alpha = (output_tmp / ("masks/" + imgname)).with_suffix(".png")
                        target_alpha.parent.mkdir(exist_ok=True, parents=True)

                        # Export image as mask and RGB
                        with z.open(info) as source, Image.open(source) as img:
                            # Get alpha channel and save it
                            assert img.mode == "RGBA", f"Image {imgname} is not RGBA, but {img.mode}"
                            alpha = img.getchannel(3)
                            alpha.save(target_alpha)
                            img.convert("RGB").save(target)

                    else:
                        with z.open(info) as source, open(target, "wb") as target:
                            shutil.copyfileobj(source, target)
            if not all_images:
                raise RuntimeError(f"No images found in {url}.")

            with open(os.path.join(str(output_tmp), "nb-info.json"), "w", encoding="utf8") as f:
                json.dump({
                    "loader": "colmap",
                    "loader_kwargs": {
                        "colmap_path": "sparse/0",
                        "images_path": "images",
                        "masks_path": "masks",
                    },
                    "id": DATASET_NAME,
                    "scene": scene_name,
                    "evaluation_protocol": "nerf",
                }, f)

            # Generate split files
            test_set = set([x.strip() for x in open(str(output_tmp / "test_list.txt")).readlines()])
            with open(str(output_tmp / "train_list.txt"), "w") as f:
                for x in sorted(set(all_images).difference(test_set)):
                    f.write(x + "\n")

            if os.path.exists(output):
                logging.warning(f"Output folder {output} already exists. Removing it.")
                # First, we move the existing folder to a temporary location to move the new folder
                # in as quickly as possible
                with tempfile.TemporaryDirectory(dir=output.parent) as tmp_to_del:
                    # NOTE: The following two lines should be atomic, but they are not
                    shutil.move(str(output), os.path.join(tmp_to_del, "old"))
                    shutil.move(str(output_tmp), str(output))
                    tmpdir.__exit__ = _noop  # Prevent deletion of the temporary directory
            else:
                shutil.move(str(output_tmp), str(output))
            logging.info(f"Downloaded {DATASET_NAME}/{scene_name} to {output}")


__all__ = ["download_hierarchical_3dgs_dataset"]
