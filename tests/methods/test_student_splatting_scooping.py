import sys
import importlib
import numpy as np
import copy
import argparse
import os
import pytest


METHOD_NAME = "student-splatting-scooping"


@pytest.fixture
def method_source_code(load_source_code):
    load_source_code("https://github.com/realcrane/3D-student-splating-and-scooping.git", "bf6d26f0aa54663d9084c24d6121eab7705b9fdd")


@pytest.fixture
def colmap_dataset(colmap_dataset_path):
    from nerfbaselines.datasets import load_dataset
    return load_dataset(str(colmap_dataset_path), split="train", features=frozenset(("points3D_xyz",)))


class Rasterizer:
    def __init__(self, raster_settings):
        self.raster_settings = raster_settings

    def __call__(self, *, means3D, means2D, shs, colors_precomp, 
                 opacities, scales, rotations, cov3D_precomp,
                 nu_degree, cover, negative_value):
        del nu_degree, cover, negative_value
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
def method_module(method_source_code, mock_module):
    del method_source_code
    renderer = mock_module("diff_t_rasterization")
    def distCUDA2(x):
        return x.norm(dim=-1)
    mock_module("simple_knn._C").distCUDA2 = distCUDA2
    renderer.TRasterizer = Rasterizer
    renderer.TRasterizationSettings = argparse.Namespace
    pytorch3d = mock_module("pytorch3d")
    pytorch3d.transforms = mock_module("pytorch3d.transforms")
    def quaternion_to_matrix(q):
        import torch
        shape = q.shape[:-1] + (3, 3)
        return torch.zeros(shape, dtype=q.dtype, device=q.device)
    pytorch3d.transforms.quaternion_to_matrix = quaternion_to_matrix
    yield importlib.import_module(f"nerfbaselines.methods.{METHOD_NAME.replace('-', '_')}")


def _test_method(method_module, colmap_dataset, tmp_path):
    dataset = copy.deepcopy(colmap_dataset)
    # Only pinhole
    dataset["cameras"] = dataset["cameras"].replace(camera_models=dataset["cameras"].camera_models*0)
    Method = method_module.StudentSplattingScooping
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


@pytest.mark.extras
@pytest.mark.method(METHOD_NAME)
def test_student_splatting_scooping_torch(torch_cpu, isolated_modules, method_module, colmap_dataset, tmp_path):
    del torch_cpu, isolated_modules
    _test_method(method_module, colmap_dataset, tmp_path)


@pytest.mark.extras
@pytest.mark.method(METHOD_NAME)
def test_train_student_splatting_scooping_cpu(torch_cpu, isolated_modules, method_module, run_test_train):
    del torch_cpu, isolated_modules, method_module
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.apptainer
def test_train_student_splatting_scooping_apptainer(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.docker
def test_train_student_splatting_scooping_docker(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.conda
def test_train_student_splatting_scooping_conda(run_test_train):
    run_test_train()

