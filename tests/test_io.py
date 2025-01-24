import json
from pathlib import Path


def test_open_any(tmp_path):
    from nerfbaselines.io import open_any

    # Test nested archives
    with open_any(tmp_path / "tests"/"data.zip"/"obj.tar.gz"/"test"/"test.zip"/"nerf_synthetic"/"data.txt", "w") as f:
        f.write(b"Hello world")

    assert (tmp_path / "tests"/"data.zip").exists()
    assert not (tmp_path / "tests"/"data.zip"/"obj.tar.gz").exists()
    assert (tmp_path / "tests"/"data.zip").is_file()

    with open_any(tmp_path / "tests"/"data.zip"/"obj.tar.gz"/"test"/"test.zip"/"nerf_synthetic"/"data.txt", "r") as f:
        data = f.read().decode("utf-8")
        assert data == "Hello world"

    # Test simple file
    with open_any(tmp_path / "data.txt", "w") as f:
        f.write(b"Hello world")
    with open_any(tmp_path / "data.txt", "r") as f:
        data = f.read().decode("utf-8")
        assert data == "Hello world"


def test_open_any_directory(tmp_path):
    from nerfbaselines.io import open_any_directory, open_any

    with open_any_directory(tmp_path / "data.zip/obj.tar.gz/test/test.zip/ok/pass.zip", "w") as _path:
        path = Path(_path)
        (path / "data.txt").write_text("Hello world2")
        (path / "test" / "ok").mkdir(parents=True, exist_ok=True)
        (path / "test" / "ok2").mkdir(parents=True, exist_ok=True)
        (path / "test" / "ok" / "data.txt").write_text("Hello world")

    assert (tmp_path / "data.zip").exists()
    assert not (tmp_path / "data.zip/obj.tar.gz").exists()
    assert (tmp_path / "data.zip").is_file()

    with open_any_directory(tmp_path / "data.zip/obj.tar.gz/test/test.zip/ok/pass.zip", "r") as _path:
        path = Path(_path)
        assert (path / "data.txt").read_text() == "Hello world2"
        assert (path / "test" / "ok" / "data.txt").read_text() == "Hello world"
        assert (path / "test" / "ok2").exists()
        assert (path / "test" / "ok2").is_dir()

    with open_any(tmp_path / "data.zip/obj.tar.gz/test/test.zip/ok/pass.zip/data.txt", "r") as f:
        assert f.read() == b"Hello world2"


def test_load_and_save_trajectory(tmp_path):
    from nerfbaselines import io

    pose = [0.568, -0.102, 0.816, -11.05, 0.178, 0.983, -0.001, 0.059, -0.802, 0.146, 0.577, -7.812]
    frame = {
        "pose": pose,
        "intrinsics": [623.53, 623.53, 640.0, 360.0],
        "appearance_weights": []
    }

    trajectory = orig = {
      "format": "nerfbaselines-v1",
      "camera_model": "pinhole",
      "image_size": [ 1280, 720 ],
      "fps": 3,
      "source": {
        "type": "interpolation",
        "interpolation": "kochanek-bartels",
        "keyframes": [
          { "pose": pose },
          { "pose": pose }
        ],
        "default_fov": 60,
        "duration": 1,
        "default_transition_duration": 2,
        "time_interpolation": "velocity",
        "is_cycle": False,
        "tension": 0,
        "continuity": 0,
        "bias": 0,
        "distance_alpha": 1
      },
      "frames": [frame, frame, frame]
    }

    with open(tmp_path / "trajectory.json", "w") as file:
        json.dump(trajectory, file, indent=2)

    with open(tmp_path / "trajectory.json", "r") as file:
        trajectory = io.load_trajectory(file)

    assert len(trajectory["frames"]) == 3

    with open(tmp_path / "trajectory2.json", "w") as file:
        io.save_trajectory(trajectory, file)

    assert (tmp_path / "trajectory2.json").exists()
    data2 = json.loads((tmp_path / "trajectory2.json").read_text())
    def round_floats(o):
        if isinstance(o, (float, int)): return round(o, 4)
        if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
        return o
    assert (
        json.dumps(round_floats(data2), indent=2, sort_keys=True) == 
        json.dumps(round_floats(orig), indent=2, sort_keys=True)
    )
