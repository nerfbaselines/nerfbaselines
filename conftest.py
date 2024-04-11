import re
from typing import List
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-docker", action="store_true", default=False, help="run docker tests")
    parser.addoption("--run-conda", action="store_true", default=False, help="run conda tests")
    parser.addoption("--run-apptainer", action="store_true", default=False, help="run apptainer tests")
    parser.addoption("--run-extras", action="store_true", default=False, help="run extras tests")
    parser.addoption("--method", action="append", default=[], help="run only these methods")
    parser.addoption("--method-regex", default=None, help="run only methods matching regex")


def pytest_configure(config):
    config.addinivalue_line("markers", "conda: mark test as requiring conda")
    config.addinivalue_line("markers", "docker: mark test as requiring docker")
    config.addinivalue_line("markers", "apptainer: mark test as requiring apptainer")
    config.addinivalue_line("markers", "extras: mark test as requiring other dependencies")
    config.addinivalue_line("markers", "method: mark test as running only a specific method")


def pytest_collection_modifyitems(config, items: List[pytest.Item]):
    for i, item in reversed(list(enumerate(items))):
        if "method" in item.keywords:
            method_name = item.keywords["method"].args[0]
            if isinstance(method_name, (list, tuple)):
                items.pop(i)
                for m in method_name:
                    function = pytest.Item.from_parent(
                        item,
                        name=item.name + "[" + m + "]",
                    )
                    function.keywords["method"] = pytest.Mark(name="method", args=(m,), kwargs={})
                    items.append(function)

    if not config.getoption("--run-conda"):
        skip_slow = pytest.mark.skip(reason="need --run-conda option to run")
        for item in items:
            if "conda" in item.keywords:
                item.add_marker(skip_slow)
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

    methods = config.getoption("--method")
    method_regex = config.getoption("--method-regex")
    if methods and method_regex:
        raise ValueError("cannot specify both --method and --method-regex")
    if methods:
        method_regex = "(" + "|".join(re.escape(x) for x in methods) + ")"
    if method_regex:
        method_re = re.compile("^" + method_regex + "$")
        for i, item in reversed(list(enumerate(items))):
            if "method" in item.keywords:
                method_name = item.keywords["method"].args[0]
                assert isinstance(method_name, str), "method name must be a string"
                if method_re.match(method_name) is None:
                    item.add_marker(pytest.mark.skip(reason="method not enabled"))
