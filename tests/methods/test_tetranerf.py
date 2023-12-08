import pytest
from .test_nerfstudio import mock_nerfstudio  # noqa: F401


@pytest.mark.method("tetra-nerf")
def test_train_tetranerf_mocked(mock_nerfstudio, run_test_train):  # noqa: F811
    if run_test_train.dataset_type == "blender":
        pytest.skip("Blender dataset is not supported by tetra-nerf")
    run_test_train()


@pytest.mark.apptainer
@pytest.mark.method("tetra-nerf")
def test_train_tetranerf_apptainer(run_test_train):
    if run_test_train.dataset_type == "blender":
        pytest.skip("Blender dataset is not supported by tetra-nerf")
    run_test_train()


@pytest.mark.docker
@pytest.mark.method("tetra-nerf")
def test_train_tetranerf_docker(run_test_train):
    if run_test_train.dataset_type == "blender":
        pytest.skip("Blender dataset is not supported by tetra-nerf")
    run_test_train()
