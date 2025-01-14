import io
import numpy as np
import pytest
import requests
import platform
import plyfile


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
        response = requests.get(host)
        assert response.status_code == 200
        assert response.text.startswith("<!DOCTYPE html>")

        # Test get dataset pointcloud
        response = requests.get(f"{host}/dataset/pointcloud.ply")
        assert response.status_code == 200
        with io.BytesIO(response.content) as f:
            plydata = plyfile.PlyData.read(f)
        assert plydata["vertex"].count == len(dataset["points3D_xyz"])

        with Viewer(port=viewer.port) as viewer2:
            assert viewer2.port == viewer.port+1
            del viewer2
            pass

    # Assert server is closed
    with pytest.raises(Exception):
        response = requests.get(host)
        assert response.status_code != 200
