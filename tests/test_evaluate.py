import sys
from unittest import mock
import pytest
import os
import json
import numpy as np
from pathlib import Path
import tarfile
from PIL import Image


from nerfbaselines.evaluation import evaluate


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
                "dataset_metadata": {
                    "name": "test",
                    "scene": "test",
                    "background_color": [0, 0, 0],
                    "color_space": "srgb",
                    "expected_scene_scale": 1.0,
                },
                "evaluation_protocol": "default",
            },
            fp,
            indent=4,
        )


def test_evaluate_folder(tmp_path, mock_torch):
    _generate_predictions(tmp_path / "predictions")
    evaluate(str(tmp_path / "predictions"), output=str(tmp_path / "results.json"))
    assert os.path.exists(tmp_path / "results.json")
    with (tmp_path / "results.json").open("r", encoding="utf8") as f:
        results = json.load(f)
        assert "lpips" in results["metrics"]

    with pytest.raises(FileExistsError):
        evaluate(str(tmp_path / "predictions"), output=str(tmp_path / "results.json"))


def test_evaluate_targz(tmp_path, mock_torch):
    _generate_predictions(tmp_path / "predictions")
    with tarfile.open(tmp_path / "predictions.tar.gz", "w:gz") as tar:
        tar.add(tmp_path / "predictions", arcname="")
    results = evaluate(str(tmp_path / "predictions.tar.gz"), output=str(tmp_path / "results.json"))
    assert os.path.exists(tmp_path / "results.json")
    with (tmp_path / "results.json").open("r", encoding="utf8") as f:
        results = json.load(f)
        assert "lpips" in results["metrics"]

    with pytest.raises(FileExistsError):
        evaluate(str(tmp_path / "predictions.tar.gz"), output=str(tmp_path / "results.json"))


def test_evaluate_command_extras(tmp_path, mock_torch):
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
        assert "ssim" in results["metrics"]
        assert "lpips" in results["metrics"]
