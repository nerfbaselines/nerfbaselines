from unittest.mock import patch
import sys
import importlib
import numpy as np
import copy
import argparse
import os
import pytest
try:
    import sklearn.neighbors
except ImportError:
    pass
from nerfbaselines._registry import collect_register_calls

METHOD_ID = "3dgrut"


class SplatRaster:
    def __init__(self, conf):
        self.conf = conf

    def trace(self, frame_id, n_active_features, particle_density, particle_radiance, ray_ori, ray_dir, *args, **kwargs):
        import torch
        ray_radiance_density = torch.ones((*ray_dir.shape[1:-1], 4), dtype=ray_dir.dtype, device=ray_dir.device) * 0.1
        ray_radiance_density[..., 3] = 0.5
        ray_radiance_density = ray_radiance_density + 0 * ray_dir[0, ..., :1]  # Enable grad
        ray_hit_distance = ray_dir[0, ..., :]
        ray_hit_count = ray_dir[..., 0].long()
        mog_visibility = None
        return ray_radiance_density, ray_hit_distance, ray_hit_count, mog_visibility

    def trace_bwd(self, frame_id,
                  n_active_features,
                  particle_density,
                  particle_radiance,
                  ray_ori,
                  ray_dir,
                  ray_time,
                  sensor_params,
                  sensor_poses_1,
                  sensor_poses_2,
                  sensor_poses_3,
                  sensor_poses_4,
                  ray_radiance_density,
                  ray_radiance_density_grd,
                  ray_hit_distance,
                  ray_hit_distance_grd):

        return particle_density, particle_radiance

    def collect_times(self):
        return {}


@pytest.fixture
def method_source_code(load_source_code, mock_module):
    sys.modules.pop('fused_ssim', None)
    mock_module("fused_ssim").fused_ssim = lambda x, y, **kwargs: (x - y).sum()
    with collect_register_calls([]):
        from nerfbaselines.methods.threedgrut_spec import GIT_REF, GIT_REPOSITORY
        load_source_code(GIT_REPOSITORY, GIT_REF)


@pytest.fixture
def colmap_dataset(colmap_dataset_path):
    from nerfbaselines.datasets import load_dataset
    return load_dataset(str(colmap_dataset_path), split="train", features=frozenset(("points3D_xyz",)))


@pytest.fixture
def method_module(method_source_code, mock_module):
    del method_source_code
    from omegaconf import OmegaConf
    lib3dgrt_cc = mock_module("threedgrt_tracer.lib3dgrt_cc")
    lib3dgrt_cc.SplatRaster = SplatRaster
    lib3dgut_cc = mock_module("threedgut_tracer.lib3dgut_cc")
    lib3dgut_cc.SplatRaster = SplatRaster
    kornia = mock_module("kornia")
    polyscope = mock_module("polyscope")
    polyscope.imgui = mock_module("polyscope.imgui")
    pandas = mock_module("pandas")
    class DataFrame: pass
    pandas.DataFrame = DataFrame
    torchmetrics = mock_module("torchmetrics")
    torchmetrics.image = mock_module("torchmetrics.image")
    torchmetrics.image.lpip = mock_module("torchmetrics.image.lpip")
    import torch
    torch_load = torch.load
    with patch.object(torch, "load", lambda *args, weights_only=True, **kwargs: torch_load(*args, **kwargs, weights_only=False)):
        yield importlib.import_module(f"nerfbaselines.methods.threedgrut")
    OmegaConf.clear_resolvers()


def _test_method(method_module, colmap_dataset, tmp_path):
    dataset = copy.deepcopy(colmap_dataset)
    # Only pinhole
    dataset["cameras"] = dataset["cameras"].replace(camera_models=dataset["cameras"].camera_models*0)
    Method = method_module.ThreeDGRUT
    model = Method(train_dataset=dataset, config_overrides={'num_workers': 0})

    # Test train iteration
    out = model.train_iteration(0)
    assert isinstance(out, dict)
    assert isinstance(out.get("loss"), float)
    assert isinstance(out.get("l1_loss"), float)

    # Test get_info
    assert isinstance(model.get_info(), dict)
    assert isinstance(Method.get_method_info(), dict)

    # Test save
    model.save(str(tmp_path/"test"))
    assert os.path.exists(tmp_path/"test")

    # Test render
    render = model.render(dataset["cameras"][0])
    assert isinstance(render, dict)
    assert "color" in render
    assert isinstance(render["color"], np.ndarray)

    # Test load + dataset
    model3 = Method(checkpoint=str(tmp_path/"test"), train_dataset=dataset)
    del model3
    model2 = Method(checkpoint=str(tmp_path/"test"))
    render = model2.render(dataset["cameras"][0])
    assert isinstance(render, dict)
    assert "color" in render
    assert isinstance(render["color"], np.ndarray)


@pytest.mark.extras
@pytest.mark.method("3dgut")
@pytest.mark.method("3dgrt")
def test_method_torch(torch_cpu, isolated_modules, method_module, colmap_dataset, tmp_path):
    del torch_cpu, isolated_modules
    _test_method(method_module, colmap_dataset, tmp_path)


@pytest.mark.extras
@pytest.mark.method("3dgrt")
@pytest.mark.method("3dgut")
def test_train_method_cpu(torch_cpu, isolated_modules, method_module, run_test_train):
    del torch_cpu, isolated_modules, method_module
    run_test_train(config_overrides={'num_workers':0})


@pytest.mark.method("3dgrt")
@pytest.mark.method("3dgut")
@pytest.mark.apptainer
def test_train_method_apptainer(run_test_train):
    run_test_train(config_overrides={'num_workers':0})


@pytest.mark.method("3dgrt")
@pytest.mark.method("3dgut")
@pytest.mark.docker
def test_train_method_docker(run_test_train):
    run_test_train()


@pytest.mark.method("3dgrt")
@pytest.mark.method("3dgut")
@pytest.mark.conda
def test_train_method_conda(run_test_train):
    run_test_train()


# Fix test names
for name, obj in list(globals().items()):
    if callable(obj) and name.startswith("test_"):
        new_name = name.replace("_method_", f"_{METHOD_ID}_")
        globals()[new_name] = obj
        globals()[new_name].__name__ = new_name
        del globals()[name]
