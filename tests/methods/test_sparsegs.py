from unittest import mock
import importlib
import numpy as np
import copy
import argparse
import os
import pytest
from PIL import Image
# Ensure matplotlib is imported before mocking other modules
import matplotlib.pyplot
from nerfbaselines._registry import collect_register_calls

METHOD_ID = "sparsegs"


@pytest.fixture
def method_source_code(load_source_code):
    with collect_register_calls([]):
        from nerfbaselines.methods.sparsegs_spec import GIT_COMMIT, GIT_REPOSITORY
        load_source_code(GIT_REPOSITORY, GIT_COMMIT)


@pytest.fixture
def colmap_dataset(colmap_dataset_path):
    from nerfbaselines.datasets import load_dataset
    return load_dataset(str(colmap_dataset_path), split="train", features=frozenset(("points3D_xyz",)))


class Rasterizer:
    def __init__(self, raster_settings):
        self.raster_settings = raster_settings

    def __call__(self, means3D, means2D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp, var_loss):
        import torch
        keep_grad = 0
        if means3D is not None: keep_grad += means3D.sum()
        if means2D is not None: keep_grad += means2D.sum()
        if shs is not None: keep_grad += shs.sum()
        if colors_precomp is not None: keep_grad += colors_precomp.sum()
        if opacities is not None: keep_grad += opacities.sum()
        if scales is not None: keep_grad += scales.sum()
        if rotations is not None: keep_grad += rotations.sum()
        if cov3D_precomp is not None: keep_grad += cov3D_precomp.sum()
        if var_loss is not None: keep_grad += var_loss.sum()
        keep_grad = keep_grad * 0.0
        num_points = means3D.shape[0]
        width = self.raster_settings.image_width
        height = self.raster_settings.image_height
        rendered_image = torch.zeros((3, height, width), dtype=torch.float32) + keep_grad
        radii = torch.zeros((num_points,), dtype=torch.float32) + keep_grad
        depth = 1 + 10 * torch.rand((height, width,), dtype=torch.float32) + keep_grad
        mode_id = torch.stack((
            1 + torch.zeros((height, width,), dtype=torch.int32),
            3 + torch.ones((height, width,), dtype=torch.int32),
        ), 0).int()
        modes = torch.full((height, width,), 0.5, dtype=torch.float32) + keep_grad
        point_list = torch.zeros((num_points,), dtype=torch.float32) + keep_grad
        means2D = torch.zeros((num_points, 2), dtype=torch.float32) + keep_grad
        conic_opacity = torch.zeros((num_points,3), dtype=torch.float32) + keep_grad
        return rendered_image, radii, depth, num_points, depth, mode_id, modes, point_list, means2D, conic_opacity


def prepare_gt_depth(input_folder, save_folder):
    for fname in os.listdir(input_folder):
        w, h = Image.open(os.path.join(input_folder, fname)).size
        depth = np.random.rand(h, w).astype(np.float32)
        np.save(os.path.join(save_folder, fname.replace(".png", ".npy")), depth)


@pytest.fixture
def method_module(method_source_code, mock_module):
    del method_source_code
    diff_gaussian_rasterization = mock_module("diff_gaussian_rasterization")
    def distCUDA2(x):
        return x.norm(dim=-1)
    mock_module("diptest")
    mock_module("icecream")
    try:
        import transformers  # type: ignore
    except ImportError:
        mock_module("transformers")
    BoostingMonocularDepth = mock_module("BoostingMonocularDepth")
    BoostingMonocularDepth.prepare_depth = mock_module("BoostingMonocularDepth.prepare_depth")
    BoostingMonocularDepth.prepare_depth.prepare_gt_depth = prepare_gt_depth

    guidance = mock_module("guidance")
    guidance.sd_utils = mock_module("guidance.sd_utils")
    guidance.sd_utils.StableDiffusion = mock.MagicMock()

    try:
        import scipy.signal
    except ImportError:
        scipy = mock_module("scipy")
        scipy.signal = mock_module("scipy.signal")
        scipy.interpolate = mock_module("scipy.interpolate")

    diptest = mock_module("diptest")
    diptest.dipstat = lambda x: x.mean()
    diffusers = mock_module("diffusers")
    diffusers.utils = mock_module("diffusers.utils")
    diffusers.utils.import_utils = mock_module("diffusers.utils.import_utils")
    mock_module("simple_knn._C").distCUDA2 = distCUDA2
    diff_gaussian_rasterization.GaussianRasterizer = Rasterizer
    diff_gaussian_rasterization.GaussianRasterizationSettings = argparse.Namespace

    yield importlib.import_module(f"nerfbaselines.methods.{METHOD_ID}")


def _test_method(method_module, colmap_dataset, tmp_path):
    dataset = copy.deepcopy(colmap_dataset)
    # Only pinhole
    dataset["cameras"] = dataset["cameras"].replace(camera_models=dataset["cameras"].camera_models*0)
    Method = method_module.SparseGS
    model = Method(train_dataset=dataset)

    # Test train iteration
    out = model.train_iteration(0)
    assert isinstance(out, dict)
    assert isinstance(out.get("loss"), float)
    assert isinstance(out.get("psnr"), float)
    assert isinstance(out.get("l1_loss"), float)
    assert isinstance(out.get("num_points"), int)

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

    # Test export demo
    splats = model.export_gaussian_splats()
    assert isinstance(splats, dict)
    assert "means" in splats
    assert "opacities" in splats
    assert "scales" in splats


@pytest.mark.extras
@pytest.mark.method(METHOD_ID)
def test_method_torch(torch_cpu, isolated_modules, method_module, colmap_dataset, tmp_path):
    del torch_cpu, isolated_modules
    _test_method(method_module, colmap_dataset, tmp_path)


@pytest.mark.extras
@pytest.mark.method(METHOD_ID)
def test_train_method_cpu(torch_cpu, isolated_modules, method_module, run_test_train):
    del torch_cpu, isolated_modules, method_module
    run_test_train(config_overrides=dict(box_p=3, prune_sched=[2]))


@pytest.mark.method(METHOD_ID)
@pytest.mark.apptainer
def test_train_method_apptainer(run_test_train):
    run_test_train(config_overrides=dict(box_p=3))


@pytest.mark.method(METHOD_ID)
@pytest.mark.docker
def test_train_method_docker(run_test_train):
    run_test_train(config_overrides=dict(box_p=3))


@pytest.mark.method(METHOD_ID)
@pytest.mark.conda
def test_train_method_conda(run_test_train):
    run_test_train(config_overrides=dict(box_p=3))


# Fix test names
for name, obj in list(globals().items()):
    if callable(obj) and name.startswith("test_"):
        new_name = name.replace("_method_", f"_{METHOD_ID}_")
        globals()[new_name] = obj
        globals()[new_name].__name__ = new_name


        del globals()[name]
