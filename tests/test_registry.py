from unittest import mock
from nerfbaselines.types import Method

class _TestMethod(Method):
    _method_name = None
    _test = 1

    def __init__(self):
        assert self._method_name is not None, "Method name not set"
        self.test = self._test

    def optimize_embeddings(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def get_method_info(cls):
        raise NotImplementedError()

    def get_info(self):
        raise NotImplementedError()

    def render(self, *args, **kwargs):
        raise NotImplementedError()

    def train_iteration(self, *args, **kwargs):
        raise NotImplementedError()

    def save(self, *args, **kwargs):
        raise NotImplementedError()

    def get_train_embedding(self, *args, **kwargs):
        raise NotImplementedError()



def test_registry_build_method():
    from nerfbaselines.registry import build_method, MethodSpec, methods_registry as registry

    spec_dict: MethodSpec = {
        "method": _TestMethod.__module__ + ":_TestMethod",
        "conda": {
            "environment_name": "test",
            "install_script": "",
        },
        "kwargs": {
            "test": 2,
        }
    }
    with mock.patch.dict(registry, test=spec_dict):
        with build_method("test", backend="python") as method_cls:
            method = method_cls()
            assert isinstance(method, _TestMethod)
            assert method.test == 2
    
    spec_dict: MethodSpec = {
        "method": _TestMethod.__module__ + ":_TestMethod",
        "conda": {
            "environment_name": "test",
            "install_script": "",
        },
        "kwargs": {
            "test": 2,
        }
    }

    # def start(self):
    #      self._rpc_backend = SimpleBackend()
    # mock.patch.object(CondaBackend, "install", lambda *_: None), \
    # mock.patch.object(CondaBackend, "_ensure_started", start):
    with mock.patch.dict(registry, test=spec_dict):
        with build_method("test", backend="python") as method_cls:
            method = method_cls()
            assert isinstance(method, _TestMethod)
            assert method.test == 2

        # with build_method("test", backend="conda") as method_cls:
        #     assert issubclass(method_cls, Method)


def test_register_spec():
    from nerfbaselines.registry import register, build_method
    from nerfbaselines import registry

    with mock.patch.object(registry, "methods_registry", {}):
        register({
            "method": test_register_spec.__module__ + ":_TestMethod",
            "conda": {
                "environment_name": "test",
                "install_script": "",
            }
        }, name="_test_" + test_register_spec.__name__)
        with build_method("_test_" + test_register_spec.__name__, backend="python") as method_cls:
            method = method_cls()
            assert isinstance(method, _TestMethod)
            assert method.test == 1
