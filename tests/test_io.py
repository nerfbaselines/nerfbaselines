from pathlib import Path


def test_open_any(tmp_path):
    from nerfbaselines.io import open_any

    # Test nested archives
    with open_any(tmp_path / "tests/data.zip/obj.tar.gz/test/test.zip/nerf_synthetic/data.txt", "w") as f:
        f.write(b"Hello world")

    assert (tmp_path / "tests/data.zip").exists()
    assert not (tmp_path / "tests/data.zip/obj.tar.gz").exists()
    assert (tmp_path / "tests/data.zip").is_file()

    with open_any(tmp_path / "tests/data.zip/obj.tar.gz/test/test.zip/nerf_synthetic/data.txt", "r") as f:
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
