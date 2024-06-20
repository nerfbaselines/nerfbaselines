import pytest
from nerfbaselines.registry import get_supported_methods, get_method_spec
from nerfbaselines.utils import assert_not_none


@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("docker")])
def test_docker_get_dockerfile(method_name):
    from nerfbaselines.backends._docker import docker_get_dockerfile, get_docker_spec
    spec = get_method_spec(method_name)
    dockerfile = docker_get_dockerfile(assert_not_none(get_docker_spec(spec)))
    assert isinstance(dockerfile, str)
    assert len(dockerfile) > 0
