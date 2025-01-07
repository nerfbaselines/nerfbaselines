import pytest
import requests
import platform


def test_viewer_simple_http_server():
    # Skip test on windows
    if platform.system() == "Windows":
        pytest.skip("Windows needs to be tested first.")

    from nerfbaselines.viewer import Viewer

    with Viewer() as viewer:
        assert viewer.port is not None

        # Test index.html
        host = f"http://localhost:{viewer.port}"
        response = requests.get(host)
        assert response.status_code == 200
        assert response.text.startswith("<!DOCTYPE html>")

        with Viewer(port=viewer.port) as viewer2:
            assert viewer2.port == viewer.port+1
            del viewer2
            pass

    # Assert server is closed
    with pytest.raises(Exception):
        response = requests.get(host)
        assert response.status_code != 200
