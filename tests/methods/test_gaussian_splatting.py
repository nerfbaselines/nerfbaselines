import argparse
import pytest
import cv2

# Needs to be loaded
del cv2  


class Rasterizer:
    def __init__(self, raster_settings):
        self.raster_settings = raster_settings

    def __call__(self, means3D, means2D, shs, colors_precomp, opacities, scales, rotations, cov3D_precomp):
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
        num_points = means3D.shape[0]
        width = self.raster_settings.image_width
        height = self.raster_settings.image_height
        rendered_image = torch.zeros((3, height, width), dtype=torch.float32) + keep_grad
        radii = torch.zeros((num_points,), dtype=torch.float32) + keep_grad
        return rendered_image, radii


@pytest.fixture
def mock_gaussian_splatting(load_source_code, mock_module):
    load_source_code("https://github.com/graphdeco-inria/gaussian-splatting", "2eee0e26d2d5fd00ec462df47752223952f6bf4e")
    def distCUDA2(x):
        return x.norm(dim=-1)
    mock_module("open3d")
    mock_module("simple_knn._C").distCUDA2 = distCUDA2
    diff_gaussian_rasterization = mock_module("diff_gaussian_rasterization")
    diff_gaussian_rasterization.GaussianRasterizer = Rasterizer
    diff_gaussian_rasterization.GaussianRasterizationSettings = argparse.Namespace


METHOD_NAME = "gaussian-splatting"


@pytest.mark.extras
@pytest.mark.method(METHOD_NAME)
def test_train_mip_splatting_cpu(torch_cpu, isolated_modules, mock_gaussian_splatting, run_test_train):
    del torch_cpu, isolated_modules, mock_gaussian_splatting
    run_test_train()


@pytest.mark.method(METHOD_NAME)
def test_train_mip_splatting_mocked(isolated_modules, mock_torch, mock_gaussian_splatting, run_test_train):
    del mock_torch, isolated_modules, mock_gaussian_splatting
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.apptainer
def test_train_gaussian_splatting_apptainer(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.docker
def test_train_gaussian_splatting_docker(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.conda
def test_train_gaussian_splatting_conda(run_test_train):
    run_test_train()

