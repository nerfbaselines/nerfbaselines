import pytest
from nerfbaselines.registry import get_supported_methods
from nerfbaselines import registry
from nerfbaselines.datasets import load_dataset
from nerfbaselines.types import Method
from nerfbaselines.backends import Backend
from nerfbaselines.utils import NoGPUError
import tempfile


def test_supported_methods():
    methods = get_supported_methods()
    assert len(methods) > 0
    assert "nerf" in methods
    assert "mipnerf360" in methods
    assert "instant-ngp" in methods
    assert "gaussian-splatting" in methods
    assert "nerfacto" in methods
    assert "tetra-nerf" in methods
    assert "zipnerf" in methods
    assert "mip-splatting" in methods


## ## TODO: Run this test inside containers to test the methods
## @pytest.mark.python_backend
## @pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods()])
## def test_method_python(blender_dataset_path, method_name):
##     try:
##         with registry.build_method(method_name, backend="python") as method_cls:
##             info = method_cls.get_method_info()
##             dataset = load_dataset(blender_dataset_path, "train", 
##                                    features=info.get("required_features"), 
##                                    supported_camera_models=info.get("supported_camera_models"))
##             assert Backend.current is not None
##             assert Backend.current.name == "python"
##             assert method_cls.get_method_info()["name"] == method_name
##             with tempfile.TemporaryDirectory() as tmpdir:
##                 dataset_overrides = registry.get_dataset_overrides(method_name, dataset["metadata"])
##                 method = method_cls(train_dataset=dataset, config_overrides=dataset_overrides)
## 
##                 # Try training step
##                 method.train_iteration(0)
## 
##                 assert isinstance(method, Method)  # type: ignore
##                 method.save(tmpdir)
## 
##                 method = method_cls(checkpoint=tmpdir)
##                 out = None
##                 for out in method.render(dataset["cameras"][:1]):
##                     assert "color" in out
##                 assert out is not None
##     except NoGPUError:
##         pytest.skip("No GPU available")


@pytest.mark.conda
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("conda")])
def test_method_conda(blender_dataset_path, method_name):
    try:
        with registry.build_method(method_name, backend="conda") as method_cls:
            info = method_cls.get_method_info()
            dataset = load_dataset(blender_dataset_path, "train", features=info.get("required_features"), supported_camera_models=info.get("supported_camera_models"))
            assert Backend.current is not None
            assert Backend.current.name == "conda"
            assert method_cls.get_method_info()["name"] == method_name
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_overrides = registry.get_dataset_overrides(method_name, dataset["metadata"])
                method = method_cls(train_dataset=dataset, config_overrides=dataset_overrides)
                assert isinstance(method, Method)  # type: ignore
                method.save(tmpdir)

                method = method_cls(checkpoint=tmpdir)
    except NoGPUError:
        pytest.skip("No GPU available")



@pytest.mark.docker
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("docker")])
def test_method_docker(blender_dataset_path, method_name):
    try:
        with registry.build_method(method_name, backend="docker") as method_cls:
            info = method_cls.get_method_info()
            dataset = load_dataset(blender_dataset_path, "train", features=info.get("required_features"), supported_camera_models=info.get("supported_camera_models"))
            assert Backend.current is not None
            assert Backend.current.name == "docker"
            assert method_cls.get_method_info()["name"] == method_name
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_overrides = registry.get_dataset_overrides(method_name, dataset["metadata"])
                method = method_cls(train_dataset=dataset, config_overrides=dataset_overrides)
                assert isinstance(method, Method)  # type: ignore
                method.save(tmpdir)

                method = method_cls(checkpoint=tmpdir)
    except NoGPUError:
        pytest.skip("No GPU available")


@pytest.mark.apptainer
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("apptainer")])
def test_method_apptainer(blender_dataset_path, method_name):
    try:
        with registry.build_method(method_name, backend="apptainer") as method_cls:
            info = method_cls.get_method_info()
            dataset = load_dataset(blender_dataset_path, "train", features=info.get("required_features"), supported_camera_models=info.get("supported_camera_models"))
            assert Backend.current is not None
            assert Backend.current.name == "apptainer"
            assert method_cls.get_method_info()["name"] == method_name
            with tempfile.TemporaryDirectory() as tmpdir:
                dataset_overrides = registry.get_dataset_overrides(method_name, dataset["metadata"])
                method = method_cls(train_dataset=dataset, config_overrides=dataset_overrides)
                assert isinstance(method, Method)  # type: ignore
                method.save(tmpdir)

                method = method_cls(checkpoint=tmpdir)
    except NoGPUError:
        pytest.skip("No GPU available")
