import importlib
import numpy as np
import copy
import argparse
import sys
import os
import pytest


METHOD_NAME = "taming-3dgs"


@pytest.fixture
def taming_source_code(load_source_code):
    load_source_code("https://github.com/humansensinglab/taming-3dgs.git", "446f2c0d50d082e660e5b899d304da5931351dec")


@pytest.fixture
def colmap_dataset(colmap_dataset_path):
    from nerfbaselines.datasets import load_dataset
    return load_dataset(str(colmap_dataset_path), split="train", features=frozenset(("points3D_xyz",)))


class Rasterizer:
    def __init__(self, raster_settings):
        self.raster_settings = raster_settings

    def __call__(self, means3D, means2D, dc, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp):
        import torch
        keep_grad = 0
        if means3D is not None: keep_grad += means3D.sum()
        if means2D is not None: keep_grad += means2D.sum()
        if dc is not None: keep_grad += dc.sum()
        if shs is not None: keep_grad += shs.sum()
        if colors_precomp is not None: keep_grad += colors_precomp.sum()
        if opacities is not None: keep_grad += opacities.sum()
        if scales is not None: keep_grad += scales.sum()
        if rotations is not None: keep_grad += rotations.sum()
        if cov3D_precomp is not None: keep_grad += cov3D_precomp.sum()
        num_points = means3D.shape[0]
        width = self.raster_settings.image_width
        height = self.raster_settings.image_height
        rendered_image = torch.zeros((3, height, width), dtype=torch.float32) + keep_grad
        radii = torch.zeros((num_points,), dtype=torch.float32) + keep_grad
        counts = torch.zeros((num_points,), dtype=torch.int32)
        lists = None
        listsRender = None
        listsDistance = None
        centers = torch.zeros((num_points, 3), dtype=torch.float32) + keep_grad
        depths = torch.zeros((num_points,), dtype=torch.float32) + keep_grad
        my_radii = torch.zeros((num_points,), dtype=torch.float32) + keep_grad
        accum_weights = torch.zeros((height, width), dtype=torch.float32) + keep_grad
        accum_count = torch.zeros((height, width), dtype=torch.int32) + keep_grad
        accum_blend = torch.zeros((height, width, 3), dtype=torch.float32) + keep_grad
        accum_dist = torch.zeros((height, width), dtype=torch.float32) + keep_grad
        return rendered_image, radii, counts, lists, listsRender, listsDistance, centers, depths, my_radii, accum_weights, accum_count, accum_blend, accum_dist


@pytest.fixture
def method_module(taming_source_code, mock_module):
    del taming_source_code
    diff_gaussian_rasterization = mock_module("diff_gaussian_rasterization")
    def distCUDA2(x):
        return x.norm(dim=-1)
    mock_module("websockets")
    mock_module("fused_ssim").fused_ssim = lambda x, y: (x - y).sum()
    mock_module("simple_knn._C").distCUDA2 = distCUDA2
    diff_gaussian_rasterization.GaussianRasterizer = Rasterizer
    diff_gaussian_rasterization.GaussianRasterizationSettings = argparse.Namespace

    yield importlib.import_module("nerfbaselines.methods.taming_3dgs")


def _test_taming_3dgs(method_module, colmap_dataset, tmp_path):
    taming_3dgs = method_module

    dataset = copy.deepcopy(colmap_dataset)
    # Only pinhole
    dataset["cameras"] = dataset["cameras"].replace(camera_models=dataset["cameras"].camera_models*0)
    Method = taming_3dgs.Taming3DGS
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
@pytest.mark.method(METHOD_NAME)
def test_taming_3dgs_torch(torch_cpu, isolated_modules, method_module, colmap_dataset, tmp_path):
    del torch_cpu, isolated_modules
    _test_taming_3dgs(method_module, colmap_dataset, tmp_path)


@pytest.mark.method(METHOD_NAME)
def test_taming_3dgs_mocked(isolated_modules, mock_torch, method_module, colmap_dataset, tmp_path):
    del mock_torch, isolated_modules
    _test_taming_3dgs(method_module, colmap_dataset, tmp_path)


@pytest.mark.extras
@pytest.mark.method(METHOD_NAME)
def test_train_taming_3dgs_cpu(torch_cpu, isolated_modules, method_module, run_test_train):
    del torch_cpu, isolated_modules, method_module
    run_test_train()


@pytest.mark.method(METHOD_NAME)
def test_train_taming_3dgs_mocked(isolated_modules, mock_torch, method_module, run_test_train):
    del mock_torch, isolated_modules, method_module
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.apptainer
def test_train_taming_3dgs_apptainer(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.docker
def test_train_taming_3dgs_docker(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.conda
def test_train_taming_3dgs_conda(run_test_train):
    run_test_train()

