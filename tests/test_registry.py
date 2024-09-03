from unittest import mock
from nerfbaselines import Method, MethodInfo, ModelInfo
import numpy as np


class _TestMethod(Method):
    _test = 1

    def __init__(self):
        self.test = self._test

    def optimize_embeddings(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError()

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

    def render(self, cameras, **kwargs):
        del kwargs
        for _ in cameras:
            yield {
                "color": np.zeros((23, 30, 3), dtype=np.float32),
                "depth": np.zeros((23, 30), dtype=np.float32),
                "anoutput": np.zeros((23, 30), dtype=np.float32),
                "anoutput2": np.zeros((23, 30), dtype=np.float32),
            }

    def train_iteration(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, *args, **kwargs):
        raise NotImplementedError()

    def get_train_embedding(self, *args, **kwargs):
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
    from nerfbaselines.training import get_presets_to_apply

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
        "name": "test-dataset",
        "scene": "test-scene",
    }

    presets = None
    presets = get_presets_to_apply(spec, dataset_metadata, presets)
    assert presets == set(("p1",))

    presets = []
    presets = get_presets_to_apply(spec, dataset_metadata, presets)
    assert presets == set(())

    presets = ["p2"]
    presets = get_presets_to_apply(spec, dataset_metadata, presets)
    assert presets == set(("p2",))

    presets = ["p2", "@auto"]
    presets = get_presets_to_apply(spec, dataset_metadata, presets)
    assert presets == set(("p2", "p1"))

    # Test union conditions
    presets = None
    presets = get_presets_to_apply(spec, {
        "name": "test-dataset-2",
        "scene": "test-scene-3",
    }, presets)
    assert presets == set(("p4",))


def test_get_config_overrides_from_presets():
    from nerfbaselines import MethodSpec
    from nerfbaselines.training import get_config_overrides_from_presets

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

    # Simple test
    o = get_config_overrides_from_presets(spec, ["p1"])
    assert o == {
        "key1": "value1",
        "key2": "value2",
    }

    # Test override previous preset
    o = get_config_overrides_from_presets(spec, ["p1", "p2"])
    assert o == {
        "key1": "value3",
        "key2": "value2",
        "key3": "value3",
    }
    o = get_config_overrides_from_presets(spec, ["p2", "p1"])
    assert o == {
        "key1": "value3",
        "key2": "value2",
        "key3": "value3",
    }

    # Test override previous preset
    o = get_config_overrides_from_presets(spec, ["p1", "p2", "p3"])
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
            camera_types=np.zeros(2, dtype=np.int32),
            distortion_parameters=np.zeros((2, 6)),
        )
        for out in method.render(cameras):
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
        for out in method.render(cameras, options={"output_type_dtypes": {"color": "uint8"}}):
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
