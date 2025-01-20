import os
from unittest import mock
import contextlib
import numpy as np
import pytest
from nerfbaselines import (
    get_supported_methods,
    get_method_spec,
)
from nerfbaselines.training import (
    get_presets_and_config_overrides,
)
from nerfbaselines import build_method_class
from nerfbaselines.datasets import load_dataset
from nerfbaselines import Method
from nerfbaselines.backends import Backend
import tempfile


def test_supported_methods():
    allowed_methods = os.environ.get("NERFBASELINES_ALLOWED_METHODS")
    try:
        os.environ.pop("NERFBASELINES_ALLOWED_METHODS", None)
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
    finally:
        if allowed_methods is not None:
            os.environ["NERFBASELINES_ALLOWED_METHODS"] = allowed_methods


## ## TODO: Run this test inside containers to test the methods
## @pytest.mark.python_backend
## @pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods()])
## def test_method_python(blender_dataset_path, method_name):
##     try:
##         spec = registry.get_method_spec(method_name)
##         with build_method_class(spec, backend="python") as method_cls:
##             info = method_cls.get_method_info()
##             dataset = load_dataset(blender_dataset_path, "train", 
##                                    features=info.get("required_features"), 
##                                    supported_camera_models=info.get("supported_camera_models"))
##             assert Backend.current is not None
##             assert Backend.current.name == "python"
##             assert method_cls.get_method_info()["name"] == method_name
##             with tempfile.TemporaryDirectory() as tmpdir:
##                 dataset_overrides = registry.get_config_overrides_from_presets(
##                     spec, registry.get_presets_to_apply(spec, dataset["metadata"]))
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
##    except Exception as e:
##        if is_gpu_error(e):
##            pytest.skip("No GPU available")
##        raise


@pytest.mark.conda
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("conda")])
def test_method_conda(blender_dataset_path, method_name, is_gpu_error):
    from nerfbaselines import get_method_spec
    try:
        spec = get_method_spec(method_name)
        with build_method_class(spec, backend="conda") as method_cls:
            info = method_cls.get_method_info()
            dataset = load_dataset(blender_dataset_path, "train", features=info.get("required_features"), supported_camera_models=info.get("supported_camera_models"))
            assert Backend.current is not None
            assert Backend.current.name == "conda"
            assert method_cls.get_method_info()["method_id"] == method_name
            with tempfile.TemporaryDirectory() as tmpdir:
                _, dataset_overrides = get_presets_and_config_overrides(spec, dataset["metadata"])
                method = method_cls(train_dataset=dataset, config_overrides=dataset_overrides)
                assert isinstance(method, Method)  # type: ignore
                method.save(tmpdir)

                method = method_cls(checkpoint=tmpdir)

                # Test metrics computation inside the container
                from nerfbaselines.evaluation import compute_metrics
                _ = compute_metrics(*np.random.rand(2, 1, 50, 50, 3), run_lpips_vgg=True)
    except Exception as e:
        if is_gpu_error(e):
            pytest.skip("No GPU available")
        raise


@pytest.mark.conda
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("conda")])
def test_metrics_conda(method_name):
    from nerfbaselines import get_method_spec
    spec = get_method_spec(method_name)
    with build_method_class(spec, backend="conda"):
        # Test metrics computation inside the container
        from nerfbaselines.evaluation import compute_metrics
        _ = compute_metrics(*np.random.rand(2, 1, 50, 50, 3), run_lpips_vgg=True)


@pytest.mark.conda
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("conda")])
def test_shell_conda(method_name):
    backend = "conda"
    from nerfbaselines import backends, get_method_spec
    spec = get_method_spec(method_name)
    with capture_execvpe():
        with backends.get_backend(spec, backend) as backend:
            backend.install()
            backend.shell(("nerfbaselines", "train", "--help"))


@pytest.mark.docker
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("docker")])
def test_method_docker(blender_dataset_path, method_name, is_gpu_error):
    from nerfbaselines import get_method_spec
    try:
        spec = get_method_spec(method_name)
        with build_method_class(spec, backend="docker") as method_cls:
            info = method_cls.get_method_info()
            dataset = load_dataset(blender_dataset_path, "train", features=info.get("required_features"), supported_camera_models=info.get("supported_camera_models"))
            assert Backend.current is not None
            assert Backend.current.name == "docker"
            assert method_cls.get_method_info()["method_id"] == method_name
            with tempfile.TemporaryDirectory() as tmpdir:
                _, dataset_overrides = get_presets_and_config_overrides(spec, dataset["metadata"])
                method = method_cls(train_dataset=dataset, config_overrides=dataset_overrides)
                assert isinstance(method, Method)  # type: ignore
                method.save(tmpdir)

                method = method_cls(checkpoint=tmpdir)

                # Test metrics computation inside the container
                from nerfbaselines.evaluation import compute_metrics
                _ = compute_metrics(*np.random.rand(2, 1, 50, 50, 3), run_lpips_vgg=True)
    except Exception as e:
        if is_gpu_error(e):
            pytest.skip("No GPU available")
        raise


@pytest.mark.docker
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("docker")])
def test_metrics_docker(method_name):
    from nerfbaselines import get_method_spec
    spec = get_method_spec(method_name)
    with build_method_class(spec, backend="docker"):
        # Test metrics computation inside the container
        from nerfbaselines.evaluation import compute_metrics
        _ = compute_metrics(*np.random.rand(2, 1, 50, 50, 3), run_lpips_vgg=True)


@contextlib.contextmanager
def capture_execvpe():
    import argparse
    obj = argparse.Namespace()
    def new_execvpe(argv0, argv, env):
        assert argv0 == argv[0]
        import subprocess
        process = subprocess.run(argv, env=env, text=True)
        process.check_returncode()
        obj.stdout = process.stdout
        obj.stderr = process.stderr

    with mock.patch("os.execvpe", new_execvpe):
        yield obj


@pytest.mark.docker
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("docker")])
def test_shell_docker(method_name):
    backend = "docker"
    from nerfbaselines import backends, get_method_spec
    spec = get_method_spec(method_name)
    with capture_execvpe():
        with backends.get_backend(spec, backend) as backend:
            backend.install()
            backend.shell(("nerfbaselines", "train", "--help"))


@pytest.mark.apptainer
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("apptainer")])
def test_method_apptainer(blender_dataset_path, method_name, is_gpu_error):
    try:
        spec = get_method_spec(method_name)
        with build_method_class(spec, backend="apptainer") as method_cls:
            info = method_cls.get_method_info()
            dataset = load_dataset(blender_dataset_path, "train", features=info.get("required_features"), supported_camera_models=info.get("supported_camera_models"))
            assert Backend.current is not None
            assert Backend.current.name == "apptainer"
            assert method_cls.get_method_info()["method_id"] == method_name
            with tempfile.TemporaryDirectory() as tmpdir:
                _, dataset_overrides = get_presets_and_config_overrides(spec, dataset["metadata"])
                method = method_cls(train_dataset=dataset, config_overrides=dataset_overrides)
                assert isinstance(method, Method)  # type: ignore
                method.save(tmpdir)

                method = method_cls(checkpoint=tmpdir)

                # Test metrics computation inside the container
                from nerfbaselines.evaluation import compute_metrics
                _ = compute_metrics(*np.random.rand(2, 1, 50, 50, 3), run_lpips_vgg=True)
    except Exception as e:
        if is_gpu_error(e):
            pytest.skip("No GPU available")
        raise


@pytest.mark.apptainer
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("apptainer")])
def test_metrics_apptainer(method_name):
    spec = get_method_spec(method_name)
    with build_method_class(spec, backend="apptainer"):
        # Test metrics computation inside the container
        from nerfbaselines.evaluation import compute_metrics
        _ = compute_metrics(*np.random.rand(2, 1, 50, 50, 3), run_lpips_vgg=True)


@pytest.mark.apptainer
@pytest.mark.parametrize("method_name", [pytest.param(k, marks=[pytest.mark.method(k)]) for k in get_supported_methods("apptainer")])
def test_shell_apptainer(method_name):
    backend = "apptainer"
    from nerfbaselines import backends, get_method_spec
    spec = get_method_spec(method_name)
    with capture_execvpe():
        with backends.get_backend(spec, backend) as backend:
            backend.install()
            backend.shell(("nerfbaselines", "train", "--help"))
