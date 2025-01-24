import pytest
import os
import sys
import contextlib
from unittest import mock
from nerfbaselines import Method, MethodInfo, ModelInfo
import numpy as np


@pytest.fixture
def clear_allowed_methods():
    allowed_methods = os.environ.get("NERFBASELINES_ALLOWED_METHODS", None)
    try:
        os.environ.pop("NERFBASELINES_ALLOWED_METHODS", None)
        yield
    finally:
        if allowed_methods is not None:
            os.environ["NERFBASELINES_ALLOWED_METHODS"] = allowed_methods


class _TestMethod(Method):
    _test = 1

    def __init__(self):
        self.test = self._test

    @classmethod
    def get_method_info(cls):
        info: MethodInfo = {
            "method_id": "test",
            "supported_outputs": ("color", "depth", {"name": "anoutput", "type": "color"})
        }
        return info

    def get_info(self):
        info: ModelInfo = {
            **self.get_method_info(),
            "num_iterations": 1
        }
        return info

    def render(self, camera, **kwargs):
        del camera, kwargs
        return {
            "color": np.zeros((23, 30, 3), dtype=np.float32),
            "depth": np.zeros((23, 30), dtype=np.float32),
            "anoutput": np.zeros((23, 30), dtype=np.float32),
            "anoutput2": np.zeros((23, 30), dtype=np.float32),
        }

    def train_iteration(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError()

    def save(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError()


def test_registry_build_method():
    from nerfbaselines import build_method_class, MethodSpec, get_method_spec
    from nerfbaselines._registry import methods_registry as registry

    spec_dict: MethodSpec = {
        "id": "test",
        "method_class": _TestMethod.__module__ + ":_TestMethod",
        "conda": {
            "environment_name": "test",
            "install_script": "",
        },
    }
    with mock.patch.dict(registry, test=spec_dict):
        spec = get_method_spec("test")
        with build_method_class(spec, backend="python") as method_cls:
            method = method_cls()
            assert isinstance(method, _TestMethod)
    
    spec_dict: MethodSpec = {
        "id": "test",
        "method_class": _TestMethod.__module__ + ":_TestMethod",
        "conda": {
            "environment_name": "test",
            "install_script": "",
        },
    }

    with mock.patch.dict(registry, test=spec_dict):
        spec = get_method_spec("test")
        with build_method_class(spec, backend="python") as method_cls:
            method = method_cls()
            assert isinstance(method, _TestMethod)


def test_register_spec():
    from nerfbaselines import register, build_method_class
    from nerfbaselines import _registry as registry

    with mock.patch.object(registry, "methods_registry", {}):
        register({
            "method_class": test_register_spec.__module__ + ":_TestMethod",
            "conda": {
                "environment_name": "test",
                "install_script": "",
            },
            "id": ("_test_" + test_register_spec.__name__),
        })
        method_spec = registry.get_method_spec("_test_" + test_register_spec.__name__)
        with build_method_class(method_spec, backend="python") as method_cls:
            method = method_cls()
            assert isinstance(method, _TestMethod)
            assert method.test == 1


def test_get_presets_to_apply():
    from nerfbaselines import MethodSpec
    from nerfbaselines.training import get_presets_and_config_overrides

    spec: MethodSpec = {
        "method_class": test_register_spec.__module__ + ":_TestMethod",
        "conda": {
            "environment_name": "test",
            "install_script": "",
        },
        "id": "test",
        "presets": {
            "p1": { "@apply": [{ "dataset": "test-dataset" }] },
            "p2": {},
            "p3": { "@apply": [{ "dataset": "test-dataset", "scene": "test-scene-2" }] },
            "p4": { "@apply": [
                { "dataset": "test-dataset-2", "scene": "test-scene-3" },
                { "dataset": "test-dataset-3", "scene": "test-scene-2" },
            ] },
        },
    }
    dataset_metadata = {
        "id": "test-dataset",
        "scene": "test-scene",
    }

    presets = None
    presets, _ = get_presets_and_config_overrides(spec, dataset_metadata, presets=presets)
    assert presets == set(("p1",))

    presets = []
    presets, _ = get_presets_and_config_overrides(spec, dataset_metadata, presets=presets)
    assert presets == set(())

    presets = ["p2"]
    presets, _ = get_presets_and_config_overrides(spec, dataset_metadata, presets=presets)
    assert presets == set(("p2",))

    presets = ["p2", "@auto"]
    presets, _ = get_presets_and_config_overrides(spec, dataset_metadata, presets=presets)
    assert presets == set(("p2", "p1"))

    # Test union conditions
    presets = None
    presets, _ = get_presets_and_config_overrides(spec, {
        "id": "test-dataset-2",
        "scene": "test-scene-3",
    }, presets=presets)
    assert presets == set(("p4",))


def test_get_config_overrides_from_presets():
    from nerfbaselines import MethodSpec
    from nerfbaselines.training import get_presets_and_config_overrides

    spec: MethodSpec = {
        "method_class": "TestMethod",
        "conda": { "environment_name": "test", "install_script": "" },
        "id": "test",
        "presets": {
            "p1": { 
               "@apply": [{ "dataset": "test-dataset" }],
               "@description": "Test preset 1",
                "key1": "value1",
                "key2": "value2",
            },
            "p2": { 
               "@apply": [{ "dataset": "test-dataset" }],
                "key1": "value3",
                "key3": "value3",
            },
            "p3": { 
                "key4": "value4",
            },
        },
    }
    dataset_metadata = {
        "id": "test-dataset",
        "scene": "test-scene",
    }

    # Simple test
    _, o = get_presets_and_config_overrides(spec, dataset_metadata, presets=["p1"])
    assert o == {
        "key1": "value1",
        "key2": "value2",
    }

    # Passed config overrides override the presets
    _, o = get_presets_and_config_overrides(spec, dataset_metadata, presets=["p1"], config_overrides={"key1": "value3", "key4": "value4"})
    assert o == {
        "key1": "value3",
        "key2": "value2",
        "key4": "value4",
    }

    # Test override previous preset
    _, o = get_presets_and_config_overrides(spec, dataset_metadata, presets=["p1", "p2"])
    assert o == {
        "key1": "value3",
        "key2": "value2",
        "key3": "value3",
    }
    _, o = get_presets_and_config_overrides(spec, dataset_metadata, presets=["p2", "p1"])
    assert o == {
        "key1": "value3",
        "key2": "value2",
        "key3": "value3",
    }

    # Test override previous preset
    _, o = get_presets_and_config_overrides(spec, dataset_metadata, presets=["p1", "p2", "p3"])
    assert o == {
        "key1": "value3",
        "key2": "value2",
        "key3": "value3",
        "key4": "value4",
    }


def test_method_autocast_render():
    from nerfbaselines import build_method_class
    from nerfbaselines import new_cameras
    with build_method_class({ "id": "test", "method_class": _TestMethod.__module__ + ":_TestMethod" }, backend="python") as method_cls:
        method = method_cls()
        num_called = 0
        cameras = new_cameras(
            poses=np.zeros((2, 4, 4)), 
            intrinsics=np.zeros((2, 4)), 
            image_sizes=np.zeros((2, 2), dtype=np.int32),
            camera_models=np.zeros(2, dtype=np.int32),
            distortion_parameters=np.zeros((2, 6)),
        )
        for camera in cameras:
            out = method.render(camera)
            assert out["color"].dtype == np.float32
            assert out["depth"].dtype == np.float32
            assert out["anoutput"].dtype == np.float32
            assert out["anoutput2"].dtype == np.float32
            assert out["color"].shape == (23, 30, 3)
            assert out["depth"].shape == (23, 30)
            assert out["anoutput"].shape == (23, 30)
            assert out["anoutput2"].shape == (23, 30)
            num_called += 1
        assert num_called == 2

        num_called = 0
        for camera in cameras:
            out = method.render(camera, options={"output_type_dtypes": {"color": "uint8"}})
            assert out["color"].dtype == np.uint8
            assert out["depth"].dtype == np.float32
            assert out["anoutput"].dtype == np.uint8
            assert out["anoutput2"].dtype == np.float32
            assert out["color"].shape == (23, 30, 3)
            assert out["depth"].shape == (23, 30)
            assert out["anoutput"].shape == (23, 30)
            assert out["anoutput2"].shape == (23, 30)
            num_called += 1
        assert num_called == 2


def test_register_environment_variable(tmp_path):
    from nerfbaselines._registry import _discover_specs

    (tmp_path / "test1.py").write_text(f"""
from nerfbaselines import register
register({{
    "id": "test1",
    "method_class": "{_TestMethod.__module__}:{_TestMethod.__name__}",
    "backends_order": ["python"],
    "metadata": {{ "test": 1 }},
}})
register({{
    "id": "test2",
    "method_class": "{_TestMethod.__module__}:{_TestMethod.__name__}",
    "backends_order": ["python"],
    "metadata": {{ "test": 2 }},
}})
""")

    environ = dict(os.environ)
    environ["NERFBASELINES_REGISTER"] = str(tmp_path/"test1.py")
    with mock.patch("os.environ", environ):
        print(os.environ)
        specs = _discover_specs()
        assert len(specs) == 2
        assert specs[0]["metadata"]["test"] == 1
        assert specs[1]["metadata"]["test"] == 2


@contextlib.contextmanager
def _patch_registry():
    with contextlib.ExitStack() as stack:
        from nerfbaselines import _registry as registry
        stack.enter_context(mock.patch.object(registry, "methods_registry", {}))
        stack.enter_context(mock.patch.object(registry, "datasets_registry", {}))
        stack.enter_context(mock.patch.object(registry, "dataset_loaders_registry", {}))
        stack.enter_context(mock.patch.object(registry, "evaluation_protocols_registry", {}))
        stack.enter_context(mock.patch.object(registry, "loggers_registry", {}))
        stack.enter_context(mock.patch.object(registry, "_auto_register_completed", False))
        yield


def test_register_environment_variable_does_not_override_default(tmp_path):
    from nerfbaselines import _registry as registry
    (tmp_path / "test1.py").write_text(f"""
from nerfbaselines import register
register({{
    "id": "nerf",
    "method_class": "{_TestMethod.__module__}:{_TestMethod.__name__}",
    "backends_order": ["python"],
    "metadata": {{ "@@test": 1 }},
}})
register({{
    "id": "@@unique-nerf",
    "method_class": "{_TestMethod.__module__}:{_TestMethod.__name__}",
    "backends_order": ["python"],
    "metadata": {{ "@@test": 1 }},
}})
""")
    environ = dict(os.environ)
    environ["NERFBASELINES_REGISTER"] = str(tmp_path/"test1.py")
    with mock.patch("os.environ", environ), _patch_registry():
        registry._auto_register()
        assert "@@unique-nerf" in registry.methods_registry
        assert "nerf" in registry.methods_registry
        assert registry.methods_registry["nerf"].get("metadata", {}).get("@@test") is None
        assert registry.methods_registry["@@unique-nerf"].get("metadata", {}).get("@@test") == 1


def test_register_from_entrypoints(tmp_path):
    from nerfbaselines import _registry as registry

    (tmp_path / "test1.py").write_text(f"""
from nerfbaselines import register
register({{
    "id": "test1",
    "method_class": "{_TestMethod.__module__}:{_TestMethod.__name__}",
    "backends_order": ["python"],
    "metadata": {{ "test": 1 }},
}})
""")
    if sys.version_info < (3, 10):
        import importlib_metadata
    else:
        from importlib import metadata as importlib_metadata

    def entry_points(group=None):
        assert group is not None, "group should not be None"
        return importlib_metadata.EntryPoints(
            [
                importlib_metadata.EntryPoint(
                    name="test1", value="test1", group="nerfbaselines.specs"
                )
            ]
        ).select(group=group)

    with contextlib.ExitStack() as stack:
        stack.enter_context(_patch_registry())
        stack.enter_context(mock.patch("nerfbaselines._registry.entry_points", entry_points))
        stack.enter_context(mock.patch("sys.modules", sys.modules.copy()))
        # Add tmp_path to sys.path
        stack.enter_context(mock.patch("sys.path", sys.path + [str(tmp_path)]))

        # Check if the entrypoint was registered
        specs = registry._discover_specs()
        assert len(specs) == 1
        assert specs[0]["metadata"]["test"] == 1


def test_install_method_spec(clear_allowed_methods, tmp_path):
    del clear_allowed_methods
    from nerfbaselines import _registry as registry

    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "test3.py").write_text(f"""
from nerfbaselines import register
register({{
    "id": "test3",
    "method_class": "{_TestMethod.__module__}:{_TestMethod.__name__}",
    "backends_order": ["python"],
    "metadata": {{ "test": 1 }},
}})
register({{
    "id": "test3i",
    "method_class": "{_TestMethod.__module__}:{_TestMethod.__name__}",
    "backends_order": ["python"],
    "metadata": {{ "test": 2 }},
}})
""")

    command = "nerfbaselines install --spec {tmp_path}test3.py"
    command = [x.replace("{tmp_path}", str(tmp_path / "input") + os.path.sep) for x in command.split()]

    # Patch sys.argv
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patch_registry())
        stack.enter_context(mock.patch("sys.argv", command))
        stack.enter_context(mock.patch("nerfbaselines._registry.METHOD_SPECS_PATH", tmp_path / "output"))
        ex = stack.enter_context(pytest.raises(SystemExit))
        from nerfbaselines import __main__
        __main__.main()
        assert ex.value.code == 0

        # Test if spec exists
        assert (tmp_path / "output" / "method-test3.py").exists()
        assert (tmp_path / "output" / "method-test3i.py").exists()

    # Can the spec be imported now?
    with contextlib.ExitStack() as stack:
        stack.enter_context(_patch_registry())
        stack.enter_context(mock.patch("nerfbaselines._registry.METHOD_SPECS_PATH", tmp_path / "output"))

        assert "test3" in registry.get_supported_methods()
        assert "test3i" in registry.get_supported_methods()
        assert registry.get_method_spec("test3").get("metadata", {}).get("test") == 1
        assert registry.get_method_spec("test3i").get("metadata", {}).get("test") == 2
