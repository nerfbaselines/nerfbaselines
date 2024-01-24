import pytest
from nerfbaselines.registry import supported_methods
from nerfbaselines import registry
from nerfbaselines.types import Method


def test_supported_methods():
    methods = supported_methods()
    assert len(methods) > 0
    assert "mipnerf360" in methods
    assert "instant-ngp" in methods
    assert "gaussian-splatting" in methods
    assert "nerfacto" in methods
    assert "tetra-nerf" in methods
    assert "zipnerf" in methods


@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in supported_methods()])
@pytest.mark.parametrize("backend", ["docker", "apptainer"])
def test_method(method_name, backend):
    spec = registry.get(method_name)
    method_cls, backend_real = spec.build(backend=backend)
    assert backend_real == backend
    assert issubclass(method_cls, Method)


@pytest.mark.docker
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in supported_methods()])
def test_method_docker(method_name):
    spec = registry.get(method_name)
    method_cls, backend_real = spec.build(backend="docker")
    method_cls.install()
    assert backend_real == "docker"
    method = method_cls()
    method.get_info()
    assert isinstance(method, Method)


@pytest.mark.apptainer
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in supported_methods()])
def test_method_apptainer(method_name):
    spec = registry.get(method_name)
    method_cls, backend_real = spec.build(backend="apptainer")
    method_cls.install()
    assert backend_real == "apptainer"
    method = method_cls()
    method.get_info()
    assert isinstance(method, Method)
