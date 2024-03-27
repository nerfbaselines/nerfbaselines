import pytest
from nerfbaselines.registry import supported_methods
from nerfbaselines import registry
from nerfbaselines.types import Method
from nerfbaselines.backends import Backend


def test_supported_methods():
    methods = supported_methods()
    assert len(methods) > 0
    assert "mipnerf360" in methods
    assert "instant-ngp" in methods
    assert "gaussian-splatting" in methods
    assert "nerfacto" in methods
    assert "tetra-nerf" in methods
    assert "zipnerf" in methods
    assert "mip-splatting" in methods


@pytest.mark.docker
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in supported_methods()])
def test_method_docker(method_name):
    with registry.build_method(method_name, backend="docker") as method_cls:
        assert Backend.current is not None
        assert Backend.current.name == "docker"
        assert method_cls.get_method_info()["name"] == method_name
        method = method_cls()
        assert isinstance(method, Method)  # type: ignore


@pytest.mark.apptainer
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in supported_methods()])
def test_method_apptainer(method_name):
    with registry.build_method(method_name, backend="apptainer") as method_cls:
        assert Backend.current is not None
        assert Backend.current.name == "apptainer"
        assert method_cls.get_method_info()["name"] == method_name
        method = method_cls()
        assert isinstance(method, Method)  # type: ignore
