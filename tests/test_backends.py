import os
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


@pytest.fixture()
def dependencies():
    def get(py_version=None):
        try: import tomllib
        except ModuleNotFoundError: import pip._vendor.tomli as tomllib
        pyproject_path = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
        with open(pyproject_path, "rb") as f:
            dependencies = tomllib.load(f)["project"]["dependencies"]
        dependencies = [x.split(";")[0].split(" ")[0].split("<")[0].split("=")[0].split(">")[0] for x in dependencies]
        if 'importlib-resources' in dependencies:
            dependencies.remove('importlib-resources')
        if 'importlib-metadata' in dependencies:
            dependencies.remove('importlib-metadata')
        assert len(dependencies) > 0
        dependencies.append("torch")
        dependencies.append("ffmpeg")
        dependencies.append("wandb")
        dependencies.append("mediapy")
        dependencies.append("pytest")
        if py_version in ("3.6", "3.7", "3.8"):
            dependencies.append("importlib-metadata")
            dependencies.append("importlib-resources")
        return dependencies
    return get


@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("docker")])
def test_dockerfile_has_dependencies(method_name, dependencies):
    from nerfbaselines.backends._docker import docker_get_dockerfile, get_docker_spec
    spec = get_method_spec(method_name)
    docker_spec = get_docker_spec(spec)
    assert docker_spec is not None
    dockerfile = docker_get_dockerfile(docker_spec).lower()
    for dep in dependencies():
        dep = dep.lower()
        if (dep.replace("_", "-") in dockerfile) or (dep.replace("-", "_") in dockerfile):
            continue
        raise AssertionError(f"Dependency {dep} not found in dockerfile")


@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("conda")])
def test_conda_installscript_has_dependencies(method_name, dependencies):
    from nerfbaselines.backends._conda import conda_get_install_script
    spec = get_method_spec(method_name)
    conda_spec = spec.get("conda")
    assert conda_spec is not None
    install_script = conda_get_install_script(conda_spec).lower()
    for dep in dependencies(conda_spec.get("python_version")):
        dep = dep.lower()
        if (dep.replace("_", "-") in install_script) or (dep.replace("-", "_") in install_script):
            continue
        raise AssertionError(f"Dependency {dep} not found in dockerfile")
