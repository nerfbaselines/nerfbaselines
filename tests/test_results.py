import sys
import pytest
import json
from pathlib import Path
from unittest import mock
from nerfbaselines.results import get_benchmark_datasets
from nerfbaselines.registry import registry


def mock_results(results_path, datasets, methods):
    import nerfbaselines.datasets
    from nerfbaselines.evaluate import _encode_values

    root = Path(nerfbaselines.__file__).absolute().parent
    for method in methods:
        for dataset in datasets:
            dinfo = json.loads((root / "datasets" / f"{dataset}.json").read_text(encoding="utf8"))
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


def test_compile_dataset_results_command(tmp_path):
    dataset = next(iter(get_benchmark_datasets()))
    method = next(iter(registry.keys()))

    with mock.patch.object(sys, "argv", ["nerfbaselines", "compile-dataset-results", "--results", str(tmp_path / "results"), "--dataset", dataset, "--output", str(tmp_path / "results.json")]):
        mock_results(tmp_path.joinpath("results"), [dataset], [method])

        import nerfbaselines.cli

        with pytest.raises(SystemExit) as excinfo:
            nerfbaselines.cli.main()
            assert excinfo.value.code == 0

        results = json.loads(tmp_path.joinpath("results.json").read_text(encoding="utf8"))
        assert_compile_dataset_results_correct(results, dataset)
