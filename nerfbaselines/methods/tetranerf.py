import logging
from tqdm import tqdm
import tempfile
import urllib.request
import numpy as np
from .nerfstudio import NerfStudio


def download_pointcloud(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        if response.getcode() != 200:
            raise RuntimeError(f"Failed to download {url} (HTTP {response.getcode()})")
        total_size_in_bytes = int(response.getheader("Content-Length", 0))
        block_size = 1024
        with tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True, desc=f"Downloading point cloud", dynamic_ncols=True) as progress_bar:
            with tempfile.TemporaryFile("rb+", suffix=".npz") as file:
                while True:
                    data = response.read(block_size)
                    if not data:
                        break
                    file.write(data)
                    progress_bar.update(len(data))
                file.flush()
                file.seek(0)
                progress_bar.close()
                if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                    logging.error(f"Failed to download dataset. {progress_bar.n} bytes downloaded out of {total_size_in_bytes} bytes.")

                data = np.load(file)
                return data["xyz"], data["rgb"]


class TetraNeRF(NerfStudio):
    _require_points3D = True
    _default_nerfstudio_name = "tetra-nerf-original"

    def _patch_config(self, config, *args, **kwargs):
        cfg = config.pipeline.datamanager.dataparser
        if hasattr(cfg, "alpha_color"):
            cfg.alpha_color = "white"
        return super()._patch_config(config, *args, **kwargs)

    def _patch_model(self, model_cls, *args, **kwargs):
        model_cls = super()._patch_model(model_cls, *args, **kwargs)

        class M(model_cls):
            def __init__(self, config, *args, **kwargs):
                # Patch loading (we can supply the point cloud and override buffer sizes ourselves)
                config.num_tetrahedra_cells = config.num_tetrahedra_cells or 0
                config.num_tetrahedra_vertices = config.num_tetrahedra_vertices or 0
                super().__init__(config, *args, **kwargs)
        return M

    def _patch_datamanager(self, datamanager_cls, **kwargs):
        datamanager_cls = super()._patch_datamanager(datamanager_cls, **kwargs)
        class DM(datamanager_cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                from torch import nn
                self.train_camera_optimizer = nn.Module()
        return DM

    def _setup(self, train_dataset=None, *args, **kwargs):
        if train_dataset is not None:
            dataset_name = train_dataset["metadata"]["id"]
            if dataset_name == "blender":
                # We use the official PC for the Blender dataset
                scene = train_dataset["metadata"]["scene"]
                url = f"https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/pointnerf-blender/{scene}.npz"
                logging.info(f"Downloading official point cloud for {dataset_name}/{scene} from {url}")
                train_dataset["points3D_xyz"], train_dataset["points3D_rgb"] = download_pointcloud(url)
            elif dataset_name == "mipnerf360":
                # We use the official PC for the MipNerf360 dataset
                scene = train_dataset["metadata"]["scene"]
                url = f"https://data.ciirc.cvut.cz/public/projects/2023TetraNeRF/assets/mipnerf360/{scene}.npz"
                logging.info(f"Downloading official point cloud for {dataset_name}/{scene} from {url}")
                train_dataset["points3D_xyz"], train_dataset["points3D_rgb"] = download_pointcloud(url)
        return super()._setup(train_dataset, *args, **kwargs)


class TetraNeRFLatest(NerfStudio):
    _default_nerfstudio_name = "tetra-nerf"
