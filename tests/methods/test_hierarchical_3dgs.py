import unittest.mock
import importlib
import numpy as np
import copy
import argparse
import os
import pytest


METHOD_NAME = "hierarchical-3dgs"


@pytest.fixture
def dataloader_noworkers():
    from torch.utils.data import DataLoader
    old_init = DataLoader.__init__
    def dl_init(self, *args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["prefetch_factor"] = None
        kwargs["persistent_workers"] = False
        old_init(self, *args, **kwargs)
    with unittest.mock.patch.object(DataLoader, "__init__", dl_init):
        yield DataLoader


@pytest.fixture
def method_source_code(load_source_code):
    load_source_code("https://github.com/graphdeco-inria/hierarchical-3d-gaussians.git", "85777b143010dedb7bc370a4591de3498fe878bb")


@pytest.fixture
def colmap_dataset(colmap_dataset_path):
    from nerfbaselines.datasets import load_dataset
    from nerfbaselines import get_method_spec
    from nerfbaselines.results import get_method_info_from_spec
    info = get_method_info_from_spec(get_method_spec("hierarchical-3dgs"))
    return load_dataset(str(colmap_dataset_path), split="train", features=info.get("required_features"))


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
        depth_image = torch.zeros((1, height, width,), dtype=torch.float32) + keep_grad
        return rendered_image, radii, depth_image


@pytest.fixture
def method_module(method_source_code, mock_module, dataloader_noworkers, tmp_path):
    del method_source_code, dataloader_noworkers
    diff_gaussian_rasterization = mock_module("diff_gaussian_rasterization")
    def distCUDA2(x):
        return x.norm(dim=-1)
    import torch.optim
    from torch import nn
    class PatchedAdam(torch.optim.Adam):
        def step(self, relevant, closure=None):  # type: ignore
            del relevant
            super().step(closure=closure)
    mock_module("simple_knn._C").distCUDA2 = distCUDA2
    mock_module("gaussian_hierarchy._C")
    mock_module("scene.OurAdam").Adam = PatchedAdam

    # Mock depth anything
    depth_anything_v2 = mock_module("depth_anything_v2")
    depth_anything_v2.dpt = mock_module("depth_anything_v2.dpt")
    depth_anything_v2.dpt.__file__ = str(tmp_path / "dpt" / "__init__.py")
    class DepthAnythingV2(nn.Module):
        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__()
        def infer_image(self, img, *args, **kwargs):
            del args, kwargs
            return (img + 0.1)[:, :, 0]
    depth_anything_v2.dpt.DepthAnythingV2 = DepthAnythingV2
    
    # Mock DPT
    mock_module("dpt").__file__ = ""
    trans = mock_module("dpt.transforms")
    class DPTDepthModel(nn.Module):
        def __init__(self, *args, **kwargs):
            del args, kwargs
            super().__init__()
        def forward(self, x):
            return x[..., 0]
    trans.PrepareForNet = lambda: lambda x: x
    trans.NormalizeImage = lambda mean, std: lambda x: x
    trans.Resize = lambda *args, **kwargs: lambda x: x
    mock_module("dpt.models").DPTDepthModel = DPTDepthModel

    diff_gaussian_rasterization.GaussianRasterizer = Rasterizer
    diff_gaussian_rasterization.GaussianRasterizationSettings = argparse.Namespace

    # Create a mock of depth anything checkpoint
    import torch
    (tmp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({}, str(tmp_path / "checkpoints" / "depth_anything_v2_vitl.pth"))

    yield importlib.import_module("nerfbaselines.methods.hierarchical_3dgs")


def _test_hierarchical_3dgs(method_module, colmap_dataset, tmp_path):
    dataset = copy.deepcopy(colmap_dataset)
    # Only pinhole
    dataset["cameras"] = dataset["cameras"].replace(camera_models=dataset["cameras"].camera_models*0)
    Method = method_module.Hierarchical3DGS
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
def test_hierarchical_3dgs_torch(torch_cpu, isolated_modules, method_module, colmap_dataset, tmp_path):
    del torch_cpu, isolated_modules
    _test_hierarchical_3dgs(method_module, colmap_dataset, tmp_path)


@pytest.mark.method(METHOD_NAME)
def test_hierarchical_3dgs_mocked(isolated_modules, mock_torch, method_module, colmap_dataset, tmp_path):
    del mock_torch, isolated_modules
    _test_hierarchical_3dgs(method_module, colmap_dataset, tmp_path)


@pytest.mark.extras
@pytest.mark.method(METHOD_NAME)
@pytest.mark.parametrize('run_test_train', ['colmap'], indirect=True)
def test_train_hierarchical_3dgs_cpu(torch_cpu, isolated_modules, method_module, run_test_train):
    del torch_cpu, isolated_modules, method_module
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.parametrize('run_test_train', ['colmap'], indirect=True)
def test_train_hierarchical_3dgs_mocked(isolated_modules, mock_torch, method_module, run_test_train):
    del mock_torch, isolated_modules, method_module
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.apptainer
@pytest.mark.parametrize('run_test_train', ['colmap'], indirect=True)
def test_train_hierarchical_3dgs_apptainer(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.docker
@pytest.mark.parametrize('run_test_train', ['colmap'], indirect=True)
def test_train_hierarchical_3dgs_docker(run_test_train):
    run_test_train()


@pytest.mark.method(METHOD_NAME)
@pytest.mark.conda
@pytest.mark.parametrize('run_test_train', ['colmap'], indirect=True)
def test_train_hierarchical_3dgs_conda(run_test_train):
    run_test_train()
