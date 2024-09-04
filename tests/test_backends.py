import pytest
from nerfbaselines import get_supported_methods, get_method_spec


@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("docker")])
def test_docker_get_dockerfile(method_name):
    from nerfbaselines.backends._docker import docker_get_dockerfile, get_docker_spec
    spec = get_method_spec(method_name)
    docker_spec = get_docker_spec(spec)
    assert docker_spec is not None
    dockerfile = docker_get_dockerfile(docker_spec)
    assert isinstance(dockerfile, str)
    assert len(dockerfile) > 0


def test_get_package_dependencies():
    from nerfbaselines.backends._common import get_package_dependencies

    dependencies = get_package_dependencies()
    assert len(dependencies) > 0
    assert any(x.startswith("click") for x in dependencies)
