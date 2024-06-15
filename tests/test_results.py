import json
import sys
import pytest
from pathlib import Path
from typing import Any, cast
from unittest import mock
from nerfbaselines.results import get_benchmark_datasets, render_markdown_dataset_results_table
from nerfbaselines.registry import methods_registry as registry


def mock_results(results_path, datasets, methods):
    import nerfbaselines.datasets
    from nerfbaselines import registry
    from nerfbaselines.io import _encode_values

    root = Path(nerfbaselines.__file__).absolute().parent
    for method in methods:
        for dataset in datasets:
            dinfo = cast(Any, registry.datasets_registry[dataset].get("metadata", {}).copy())
            for scene_data in dinfo["scenes"]:
                scene = scene_data["id"]

                # Create results
                results_path.joinpath(method, dataset).mkdir(parents=True, exist_ok=True)
                results_path.joinpath(method, dataset, scene + ".json").write_text(
                    json.dumps(
                        {
                            "metrics_raw": {k: _encode_values([0.5, 0.1]) for k in dinfo["metrics"]},
                        }
                    )
                )


def test_get_benchmark_datasets():
    datasets = get_benchmark_datasets()
    assert "blender" in datasets
    assert "mipnerf360" in datasets


def assert_compile_dataset_results_correct(results, dataset):
    assert "methods" in results
    assert "scenes" in results["methods"][0]
    assert "name" in results["methods"][0]
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
    assert_nonempty_string(results["metrics"][0].get("id"))
    assert_nonempty_string(results["metrics"][0].get("name"))
    assert_nonempty_string(results["metrics"][0].get("description"))
    assert_nonempty_string(results["metrics"][0].get("link"))
    assert results["metrics"][0].get("ascending") in {True, False}
    assert_nonempty_string(results["scenes"][0].get("id"))
    assert_nonempty_string(results["scenes"][0].get("name"))
    method = results["methods"][0]
    assert_nonempty_string(method.get("id"))
    assert_nonempty_string(method.get("name"))
    assert_nonempty_string(method.get("description"))
    assert isinstance(method.get("psnr"), float)
    assert len(method["scenes"]) > 0
    fscene = next(iter(method["scenes"].values()))
    assert isinstance(fscene.get("psnr"), float)


@pytest.mark.parametrize("method", list(registry.keys()))
@pytest.mark.parametrize("dataset", get_benchmark_datasets())
def test_compile_dataset_results(tmp_path, dataset, method):
    from nerfbaselines.results import compile_dataset_results

    mock_results(tmp_path, [dataset], [method])

    # Mock
    results = compile_dataset_results(tmp_path, dataset)
    print(json.dumps(results, indent=2))
    assert_compile_dataset_results_correct(results, dataset)


def test_render_dataset_results_json_capture_command(tmp_path, capsys):
    dataset = next(iter(get_benchmark_datasets()))
    method = next(iter(registry.keys()))

    with mock.patch.object(sys, "argv", ["nerfbaselines", "generate-dataset-results", "--output-type", "json", "--results", str(tmp_path / "results"), "--dataset", dataset]):
        mock_results(tmp_path.joinpath("results"), [dataset], [method])

        import nerfbaselines.cli

        with pytest.raises(SystemExit) as excinfo:
            nerfbaselines.cli.main()
            assert excinfo.value.code == 0

        out, _ = capsys.readouterr()
        results = json.loads(out)
        assert_compile_dataset_results_correct(results, dataset)


def test_render_dataset_results_json_command(tmp_path):
    dataset = next(iter(get_benchmark_datasets()))
    method = next(iter(registry.keys()))

    with mock.patch.object(
        sys, "argv", ["nerfbaselines", "generate-dataset-results", "--output-type", "json", "--results", str(tmp_path / "results"), "--dataset", dataset, "--output", str(tmp_path / "results.json")]
    ):
        mock_results(tmp_path.joinpath("results"), [dataset], [method])

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
