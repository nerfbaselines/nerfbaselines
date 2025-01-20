import numpy as np
import pytest
import platform
import plyfile
import urllib.request


def test_viewer_simple_http_server():
    # Skip test on windows
    if platform.system() == "Windows":
        pytest.skip("Windows needs to be tested first.")

    from nerfbaselines.viewer import Viewer

    dataset = {
        "points3D_xyz" : np.random.rand(100, 3),
        "points3D_rgb" : np.random.rand(100, 3).astype(np.uint8),
    }

    with Viewer(train_dataset=dataset) as viewer:
        assert viewer.port is not None

        # Test index.html
        host = f"http://localhost:{viewer.port}"
        with urllib.request.urlopen(host) as response:
            assert response.getcode() == 200
            assert response.read().decode("utf-8").startswith("<!DOCTYPE html>")

        # Test get dataset pointcloud
        with urllib.request.urlopen(f"{host}/dataset/pointcloud.ply") as response:
            assert response.getcode() == 200
            plydata = plyfile.PlyData.read(response)
            assert plydata["vertex"].count == len(dataset["points3D_xyz"])

        with Viewer(port=viewer.port) as viewer2:
            assert viewer2.port is not None
            assert viewer.port is not None
            assert viewer2.port > viewer.port and viewer2.port < viewer.port + 10
            del viewer2
            pass

    # Assert server is closed
    with pytest.raises(Exception):
        with urllib.request.urlopen(host) as response:
            assert response.getcode() != 200


def test_build_static_viewer(tmp_path):
    from nerfbaselines.viewer import build_static_viewer

    build_static_viewer(tmp_path/"v1")
    build_static_viewer(str(tmp_path/"v2"))
    build_static_viewer(str(tmp_path/"v3"), {"test": "passed"})
    assert (tmp_path/"v1"/"index.html").exists()
    assert (tmp_path/"v1"/"viewer.js").exists()
    assert (tmp_path/"v1"/"third-party"/"three.module.js").exists()
