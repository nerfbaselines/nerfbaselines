import logging
from tqdm import tqdm
import tempfile
import requests
import numpy as np
from .nerfstudio import NerfStudio
from ...datasets import Dataset


def download_pointcloud(url):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Downloading {url.split('/')[-1]}")
    with tempfile.TemporaryFile("rb+", suffix=".npz") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
        file.flush()
        file.seek(0)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            logging.error(f"Failed to download dataset. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes.")

        data = np.load(file)
        return data["xyz"], data["rgb"]


class TetraNeRF(NerfStudio):
    def _apply_config_patch_for_dataset(self, config, dataset: Dataset):
        # We do not allow config to be overriden by the dataset
        pass

    def _patch_dataparser_params(self):
        self.dataparser_params["alpha_color"] = "white"
        return super()._patch_dataparser_params()

    def setup_train(self, train_dataset: Dataset, *, num_iterations: int):
        dataset = train_dataset
        dataset_name = dataset.metadata["type"]
        if dataset_name == "blender":
            # We use the official PC for the Blender dataset
            scene = dataset.metadata["scene"]
            url = f"https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/pointnerf-blender/{scene}.npz"
            logging.info(f"Downloading official point cloud for {dataset_name}/{scene} from {url}")
            dataset.points3D_xyz, dataset.points3D_rgb = download_pointcloud(url)
        elif dataset_name == "mipnerf360":
            # We use the official PC for the MipNerf360 dataset
            scene = dataset.metadata["scene"]
            url = f"https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/mipnerf360/{scene}.npz"
            logging.info(f"Downloading official point cloud for {dataset_name}/{scene} from {url}")
            dataset.points3D_xyz, dataset.points3D_rgb = download_pointcloud(url)
        return super().setup_train(dataset, num_iterations=num_iterations)
