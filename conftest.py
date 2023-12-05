from typing import List
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-docker", action="store_true", default=False, help="run docker tests")
    parser.addoption("--run-apptainer", action="store_true", default=False, help="run apptainer tests")
    parser.addoption("--run-extras", action="store_true", default=False, help="run extras tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "docker: mark test as requiring docker")
    config.addinivalue_line("markers", "apptainer: mark test as requiring apptainer")
    config.addinivalue_line("markers", "extras: mark test as requiring other dependencies")


def pytest_collection_modifyitems(config, items: List[pytest.Item]):
    if not config.getoption("--run-docker"):
        skip_slow = pytest.mark.skip(reason="need --run-docker option to run")
        for item in items:
            if "docker" in item.keywords:
                item.add_marker(skip_slow)
    if not config.getoption("--run-apptainer"):
        skip_slow = pytest.mark.skip(reason="need --run-apptainer option to run")
        for item in items:
            if "apptainer" in item.keywords:
                item.add_marker(skip_slow)

    if not config.getoption("--run-extras"):
        for item in items:
            if "extras" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="need --run-extras option to run"))
