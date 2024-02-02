from nerfbaselines.backends import CondaMethod

class _TestMethod:
    def __init__(self, test=1):
        self.test = test


def test_convert_spec_dict_to_spec():
    from nerfbaselines.registry import convert_spec_dict_to_spec, MethodSpecDict


    spec_dict: MethodSpecDict = {
        "method": ".:_TestMethod",
        "conda": {
            "conda_name": "test",
            "build_code": "",
        }
    }
    spec = convert_spec_dict_to_spec(spec_dict)
    method_class, _ = spec.build(backend="python")
    method = method_class()
    assert isinstance(method, _TestMethod)
    assert method.test == 1
    
    spec_dict: MethodSpecDict = {
        "method": _TestMethod.__module__ + ":_TestMethod",
        "conda": {
            "conda_name": "test",
            "build_code": "",
        }
    }
    spec = convert_spec_dict_to_spec(spec_dict)
    method_class, _ = spec.build(backend="python")
    method = method_class()
    assert isinstance(method, _TestMethod)
    assert method.test == 1

    conda_method = spec.build(backend="conda")[0]()
    assert isinstance(conda_method, CondaMethod)


def test_convert_spec_dict_to_spec_method_args():
    from nerfbaselines.registry import convert_spec_dict_to_spec, MethodSpecDict


    spec_dict: MethodSpecDict = {
        "method": ".:_TestMethod",
        "conda": {
            "conda_name": "test",
            "build_code": "",
        },
        "kwargs": {
            "test": 2,
        }
    }
    spec = convert_spec_dict_to_spec(spec_dict)
    method_class, _ = spec.build(backend="python")
    method = method_class()
    assert isinstance(method, _TestMethod)
    assert method.test == 2
    
    spec_dict: MethodSpecDict = {
        "method": _TestMethod.__module__ + ":_TestMethod",
        "conda": {
            "conda_name": "test",
            "build_code": "",
        },
        "kwargs": {
            "test": 2,
        }
    }
    spec = convert_spec_dict_to_spec(spec_dict)
    method_class, _ = spec.build(backend="python")
    method = method_class()
    assert isinstance(method, _TestMethod)
    assert method.test == 2

    conda_method = spec.build(backend="conda")[0]()
    assert isinstance(conda_method, CondaMethod)


def test_register_spec():
    from nerfbaselines.registry import register, get


    register({
        "method": ".:_TestMethod",
        "conda": {
            "conda_name": "test",
            "build_code": "",
        }
    }, "_test_" + test_register_spec.__name__)
    method_class, _ = get("_test_" + test_register_spec.__name__).build(backend="python")
    method = method_class()
    assert isinstance(method, _TestMethod)
    assert method.test == 1
