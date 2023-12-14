import sys
from unittest import mock
import pytest
import os
import json
import numpy as np
from pathlib import Path
import tarfile
from PIL import Image


from nerfbaselines.evaluate import evaluate


def _generate_predictions(path, background_color=None):
    path = Path(path)
    for i in range(100):
        (path / "color").mkdir(parents=True, exist_ok=True)
        (path / "gt-color").mkdir(parents=True, exist_ok=True)
        img = np.random.randint(0, 255, (64, 34, 3), dtype=np.uint8)
        Image.fromarray(img).save(path / "color" / f"{i}.png")

        img = np.random.randint(0, 255, (64, 34, 3), dtype=np.uint8)
        Image.fromarray(img).save(path / "gt-color" / f"{i}.png")
    with open(path / "info.json", "w") as fp:
        json.dump(
            {
                "method": "test",
                "nb_version": "0.0.0",
                "color_space": "srgb",
                "expected_scene_scale": 1.0,
                "dataset_type": "test",
                "dataset_scene": "test",
                "dataset_background_color": background_color,
            },
            fp,
            indent=4,
        )


@pytest.mark.parametrize("background_color", [None, [0, 0, 0]])
@pytest.mark.filterwarnings("ignore::UserWarning:torchvision")
def test_evaluate_folder(tmp_path, background_color):
    _generate_predictions(tmp_path / "predictions", background_color)
    evaluate(tmp_path / "predictions", output=tmp_path / "results.json", disable_extra_metrics=True)
    assert os.path.exists(tmp_path / "results.json")

    with pytest.raises(FileExistsError):
        evaluate(tmp_path / "predictions", output=tmp_path / "results.json")


@pytest.mark.extras
@pytest.mark.filterwarnings("ignore::UserWarning:torchvision")
def test_evaluate_folder_extras(tmp_path):
    _generate_predictions(tmp_path / "predictions")
    evaluate(tmp_path / "predictions", output=tmp_path / "results.json", disable_extra_metrics=False)
    assert os.path.exists(tmp_path / "results.json")
    with (tmp_path / "results.json").open("r", encoding="utf8") as f:
        results = json.load(f)
        assert "lpips" in results["metrics"]

    with pytest.raises(FileExistsError):
        evaluate(tmp_path / "predictions", output=tmp_path / "results.json")


def test_evaluate_targz(tmp_path):
    _generate_predictions(tmp_path / "predictions")
    with tarfile.open(tmp_path / "predictions.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "predictions", arcname="")
    _results = evaluate(tmp_path / "predictions.tar.gz", output=tmp_path / "results.json")
    assert os.path.exists(tmp_path / "results.json")

    with pytest.raises(FileExistsError):
        evaluate(tmp_path / "predictions.tar.gz", output=tmp_path / "results.json")


@pytest.mark.extras
@pytest.mark.filterwarnings("ignore::UserWarning:torchvision")
def test_evaluate_targz_extras(tmp_path):
    _generate_predictions(tmp_path / "predictions")
    with tarfile.open(tmp_path / "predictions.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "predictions", arcname="")
    results = evaluate(tmp_path / "predictions.tar.gz", output=tmp_path / "results.json")
    assert os.path.exists(tmp_path / "results.json")
    with (tmp_path / "results.json").open("r", encoding="utf8") as f:
        results = json.load(f)
        assert "lpips" in results["metrics"]

    with pytest.raises(FileExistsError):
        evaluate(tmp_path / "predictions.tar.gz", output=tmp_path / "results.json")


@pytest.mark.filterwarnings("ignore::UserWarning:torchvision")
def test_evaluate_command(tmp_path):
    _generate_predictions(tmp_path / "predictions")
    args = ["nerfbaselines", "evaluate", str(tmp_path / "predictions"), "-o", str(tmp_path / "results.json"), "--disable-extra-metrics"]

    with mock.patch.object(sys, "argv", args):
        import nerfbaselines.cli

        with pytest.raises(SystemExit) as excinfo:
            nerfbaselines.cli.main()
            assert excinfo.value.code == 0
    assert os.path.exists(tmp_path / "results.json")
    with (tmp_path / "results.json").open("r", encoding="utf8") as f:
        results = json.load(f)
        assert "ssim" in results["metrics"]


@pytest.mark.extras
@pytest.mark.filterwarnings("ignore::UserWarning:torchvision")
def test_evaluate_command_extras(tmp_path):
    _generate_predictions(tmp_path / "predictions")
    args = ["nerfbaselines", "evaluate", str(tmp_path / "predictions"), "-o", str(tmp_path / "results.json")]

    with mock.patch.object(sys, "argv", args):
        import nerfbaselines.cli

        with pytest.raises(SystemExit) as excinfo:
            nerfbaselines.cli.main()
            assert excinfo.value.code == 0
    assert os.path.exists(tmp_path / "results.json")
    with (tmp_path / "results.json").open("r", encoding="utf8") as f:
        results = json.load(f)
        assert "lpips" in results["metrics"]
