import json
import sys
import pytest
from typing import Any, cast
from unittest import mock
from nerfbaselines import get_supported_methods, get_supported_datasets
from nerfbaselines.results import render_markdown_dataset_results_table


def mock_results(results_path, datasets, methods):
    from nerfbaselines import get_method_spec, get_dataset_spec
    from nerfbaselines.io import _encode_values

    results_path.mkdir(parents=True, exist_ok=True)
    for method in methods:
        spec = get_method_spec(method)
        for dataset in datasets:
            dataset_spec = get_dataset_spec(dataset)
            dinfo = cast(Any, dataset_spec.get("metadata", {}).copy())
            for scene_data in dinfo["scenes"]:
                scene = scene_data["id"]
                if spec.get("output_artifacts", {}).get(f"{dataset}/{scene}"):
                    continue

                # Create results
                results_path.joinpath(method, dataset).mkdir(parents=True, exist_ok=True)
                results_path.joinpath(method, dataset, scene + ".json").write_text(
                    json.dumps(
                        {
                            "metrics_raw": {k: _encode_values([0.5, 0.1]) for k in dinfo["metrics"]},
                            "nb_info": {"method": method },
                            "render_dataset_metadata": {
                                "name": dataset,
                                "scene": scene
                            }
                        }
                    )
                )


def test_get_supported_datasets():
    datasets = get_supported_datasets()
    assert "blender" in datasets
    assert "mipnerf360" in datasets


def assert_compile_dataset_results_correct(results, dataset):
    assert "methods" in results
    assert "id" in results
    assert results["id"] == dataset

    def assert_nonempty_string(x):
        assert isinstance(x, str)
        assert len(x) > 0

    assert_nonempty_string(results.get("name"))
    assert_nonempty_string(results.get("description"))
    assert_nonempty_string(results.get("default_metric"))
    assert len(results["methods"]) > 0
    assert len(results["scenes"]) > 0
    assert len(results["metrics"]) > 0
    for metric in results["metrics"]:
        assert_nonempty_string(metric.get("id"))
        assert_nonempty_string(metric.get("name"))
        assert_nonempty_string(metric.get("description"))
        assert_nonempty_string(metric.get("link"))
        assert metric.get("ascending") in {True, False}

    for scene in results["scenes"]:
        assert_nonempty_string(scene.get("id"))
        assert_nonempty_string(scene.get("name"))

    for method in results["methods"]:
        assert "scenes" in method
        assert "name" in method
        assert_nonempty_string(method.get("id"))
        assert_nonempty_string(method.get("name"))
        assert_nonempty_string(method.get("description"))
        assert isinstance(method.get("psnr"), float)
        assert len(method["scenes"]) > 0
        fscene = next(iter(method["scenes"].values()))
        assert isinstance(fscene.get("psnr"), float)


@pytest.mark.parametrize("method", list(get_supported_methods()))
def test_get_method_info_from_spec(method):
    from nerfbaselines import MethodInfo, get_method_spec
    from nerfbaselines.results import get_method_info_from_spec

    spec = get_method_spec(method)
    method_info: MethodInfo = get_method_info_from_spec(spec)
    # assert isinstance(method_info, MethodInfo)
    assert isinstance(method_info, dict)
    # Assert all fields are standard
    def assert_ok_type(x):
        if x is None: return
        if type(x) in (str, bytes, int, float, bool): return
        if type(x) in (list, tuple, frozenset, set):
            for y in x:
                assert_ok_type(y)
            return
        if type(x) is dict:
            for k, v in x.items():
                assert_ok_type(k)
                assert_ok_type(v)
            return
        raise ValueError(f"Unexpected type {type(x)}")

    assert_ok_type(method_info)
    assert method_info["method_id"] == method


@pytest.mark.parametrize("dataset", get_supported_datasets())
def test_compile_dataset_results(tmp_path, dataset):
    from nerfbaselines.results import compile_dataset_results
    mock_results(tmp_path, [dataset], list(get_supported_methods()))

    # Mock
    results = compile_dataset_results(tmp_path, dataset)
    assert_compile_dataset_results_correct(results, dataset)


@pytest.mark.parametrize("dataset", get_supported_datasets())
def test_render_dataset_results_json_capture_command(tmp_path, capsys, dataset):
    with mock.patch.object(sys, "argv", ["nerfbaselines", "generate-dataset-results", "--output-type", "json", "--results", str(tmp_path / "results"), "--dataset", dataset]):
        mock_results(tmp_path.joinpath("results"), [dataset], list(get_supported_methods()))

        import nerfbaselines.cli

        with pytest.raises(SystemExit) as excinfo:
            nerfbaselines.cli.main()
            assert excinfo.value.code == 0

        out, stderr = capsys.readouterr()
        print(stderr, file=sys.stderr)
        results = json.loads(out)
        assert_compile_dataset_results_correct(results, dataset)


@pytest.mark.parametrize("dataset", get_supported_datasets())
def test_render_dataset_results_json_command(tmp_path, dataset):
    with mock.patch.object(
        sys, "argv", ["nerfbaselines", "generate-dataset-results", "--output-type", "json", "--results", str(tmp_path / "results"), "--dataset", dataset, "--output", str(tmp_path / "results.json")]
    ):
        mock_results(tmp_path.joinpath("results"), [dataset], list(get_supported_methods()))

        import nerfbaselines.cli

        with pytest.raises(SystemExit) as excinfo:
            nerfbaselines.cli.main()
            assert excinfo.value.code == 0

        results = json.loads(tmp_path.joinpath("results.json").read_text(encoding="utf8"))
        assert_compile_dataset_results_correct(results, dataset)


@pytest.fixture
def dataset_results():
    return json.loads(
        """{
  "id": "mipnerf360",
  "name": "Mip-NeRF 360",
  "description": "Mip-NeRF 360 is a novel neural radiance field (NeRF) based method for view synthesis of 360° scenes. It is based on the idea of mip-mapping, which is a technique for efficiently storing and rendering textures at multiple resolutions.",
  "paper_link": "https://arxiv.org/abs/2104.00677",
  "link": "https://arxiv.org/abs/2104.00677",
  "default_metric": "psnr",
  "scenes": [{
    "id": "garden",
    "name": "garden"
  }, {
    "id": "bonsai",
    "name": "bonsai"
  }],
  "metrics": [{
    "id": "psnr",
    "name": "PSNR",
    "ascending": true,
    "description": "Peak Signal-to-Noise-Ratio"
  }, {
    "id": "ssim",
    "name": "SSIM",
    "ascending": true,
    "description": "Structural Similarity Index"
  }, {
    "id": "lpips",
    "name": "LPIPS",
    "ascending": false,
    "description": "Learned Perceptual Image Patch Similarity"
  }],
  "methods": [{
    "id": "mipnerf360",
    "name": "mipnerf360",
    "psnr": 15.0,
    "ssim": 0.89,
    "lpips": 0.01,
    "total_train_time": 100,
    "scenes": {
      "garden": {
        "psnr": 24.5,
        "ssim": 0.8234,
        "lpips": 0.005,
        "total_train_time": 100
      },
      "bonsai": {
        "psnr": 24.5,
        "ssim": 0.8234,
        "lpips": 0.005,
        "total_train_time": 100
      }
    }
  }, {
    "id": "gaussian-splatting",
    "name": "Gaussian Splatting",
    "psnr": 21.0,
    "ssim": null,
    "lpips": 0.02,
    "total_train_time": 100,
    "gpu_memory": 23500,
    "scenes": {
      "garden": {
        "psnr": 23.5,
        "ssim": 0.7234,
        "lpips": 0.005,
        "total_train_time": 100,
        "gpu_memory": 23500
      },
      "bonsai": {
        "psnr": 12.5,
        "ssim": 0.3234,
        "lpips": 0.005,
        "total_train_time": 100,
        "gpu_memory": 23500
      }
    }
  }]
}
"""
    )


def test_render_markdown_dataset_results_table(dataset_results):
    table = render_markdown_dataset_results_table(dataset_results)
    lines = table.splitlines()
    print(table)
    assert len(lines) == 4
    for line in lines:
        assert len(line) == len(lines[0])
