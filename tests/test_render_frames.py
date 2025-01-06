import pytest
import os
import numpy as np
from typing import Optional, TypeVar
from unittest import mock
from unittest.mock import MagicMock
from nerfbaselines import Method, new_cameras
from nerfbaselines import evaluation
from PIL import Image
import tarfile
import zipfile
try:
    from typeguard import suppress_type_checks  # type: ignore
except ImportError:
    from contextlib import nullcontext as suppress_type_checks


T = TypeVar("T")


def _assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


num_cams = 2


class FakeMethod(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, spec=Method)

    @staticmethod
    def get_method_info():
        return dict(
            supported_camera_models=("pinhole",),
            supported_outputs=("color", 
                               "depth", 
                               "accumulation",
                               dict(name="color-custom", type="color"))
        )

    def get_info(self):
        return self.get_method_info()

    def render(self, camera, options=None):
        del options
        camera = camera.item()
        def get_shape(w, h, output):
            output_type = output.get("type", output["name"]) if isinstance(output, dict) else output
            if output_type == "color":
                return (h, w, 3)
            else:
                return (h, w)
        w, h = camera.image_sizes
        return {
            (k["name"] if isinstance(k, dict) else k): np.zeros(get_shape(w, h, k)) for k in self.get_info()["supported_outputs"]
        }


def _mock_cameras(num_cameras=2):
    w,h = 20, 30
    return new_cameras(
        poses=np.zeros((num_cameras, 3, 4),),
        intrinsics=np.zeros((num_cameras, 4),),
        camera_models=np.zeros((num_cameras, ), dtype=np.uint8),
        distortion_parameters=np.zeros((num_cameras, 0),),
        image_sizes=np.array([[w,h]]*num_cameras, dtype=np.uint32))


def _mock_trajectory(num_cams):
    return {
        "format": "nerfbaselines-v1",
        "image_size": [20, 30],
        "camera_model": "pinhole",
        "fps": 1,
        "frames": [
            {
                "pose": np.zeros((3, 4)).tolist(),
                "intrinsics": np.zeros((4,)).tolist(),
                "appearance_weights": np.zeros((0,)).tolist()
            } for _ in range(num_cams)
        ],
    }


def _test_render_trajectory_command(tmp_path, out, *args):
    (tmp_path/"test-checkpoint").mkdir()
    with open(tmp_path/"test-checkpoint"/"nb-info.json", "w") as f:
        f.write('{"method": "test-method"}')
    with open(tmp_path/"trajectory.json", "w") as f:
        import json
        json.dump(_mock_trajectory(num_cams), f)
    
    command = [x.replace("{tmp_path}", str(tmp_path)).replace("{out}", out) for x in "nerfbaselines render-trajectory --checkpoint {tmp_path}/test-checkpoint --trajectory {tmp_path}/trajectory.json --output {tmp_path}/{out} --backend python".split()]
    # Patch sys.argv
    test_registry = {"test-method": {
        "id": "test-method",
        "method_class": (__package__ or "") + "." + os.path.splitext(os.path.basename(__file__))[0] + ":" + FakeMethod.__name__,
    }}
    with mock.patch("sys.argv", command + list(args)), \
         mock.patch("nerfbaselines._registry.methods_registry", test_registry), \
         mock.patch("nerfbaselines._registry._auto_register_completed", True), \
         suppress_type_checks(), \
         pytest.raises(SystemExit) as ex:
        from nerfbaselines import __main__
        __main__.main()
        assert ex.value.code == 0


def _verify_targz_single(tmp_path, num_cams):
    assert os.path.exists(tmp_path/"out.tar.gz")
    with tarfile.open(str(tmp_path/"out.tar.gz")) as tar:
        members = list(tar.getnames())
        assert len(members) == num_cams
        for i in range(num_cams):
            path = f"{i:05d}.png"
            assert f"{i:05d}.png" in members
            member = tar.getmember(path)
            with _assert_not_none(tar.extractfile(member)) as f:
                img = Image.open(f)
                assert img.size == (20, 30)


def test_render_frames_targz_single(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)
    evaluation.render_frames(method, cameras, 
                             output=tmp_path/"out.tar.gz",
                             fps=1,
                             output_names=("color",))

    # Single output
    _verify_targz_single(tmp_path, num_cams)


def test_render_frames_targz_single_command(tmp_path):
    _test_render_trajectory_command(tmp_path, "out.tar.gz")
    _verify_targz_single(tmp_path, num_cams)


def _verify_targz_multi(tmp_path, num_cams, all_output_names):
    assert os.path.exists(tmp_path/"out-multi.tar.gz")
    with tarfile.open(str(tmp_path/"out-multi.tar.gz")) as tar:
        members = list(tar.getnames())
        assert len(members) == num_cams * len(all_output_names)
        for out in all_output_names:
            for i in range(num_cams):
                p = f"{out}/{i:05d}.png"
                assert f"{out}/{i:05d}.png" in members
                with _assert_not_none(tar.extractfile(tar.getmember(p))) as f:
                    img = Image.open(f)
                    assert img.size == (20, 30)


def test_render_frames_targz_multi(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)
    
    # Multiple outputs
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in method.get_info()["supported_outputs"])
    evaluation.render_frames(method, cameras, 
                             output=tmp_path/"out-multi.tar.gz",
                             fps=1,
                             output_names=all_output_names)
    _verify_targz_multi(tmp_path, num_cams, all_output_names)


def test_render_frames_targz_multi_command(tmp_path):
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in FakeMethod().get_info()["supported_outputs"])
    _test_render_trajectory_command(tmp_path, "out-multi.tar.gz", "--output-names", ",".join(all_output_names))
    _verify_targz_multi(tmp_path, num_cams, all_output_names)


def _verify_targz_multi_format(path, all_output_names, num_cams):
    for out in all_output_names:
        assert os.path.exists(path.format(output=out))
        with _assert_not_none(tarfile.open(path.format(output=out))) as tar:
            members = list(tar.getnames())
            assert len(members) == num_cams
            for i in range(num_cams):
                p = f"{i:05d}.png"
                assert p in members
                with _assert_not_none(tar.extractfile(tar.getmember(p))) as f:
                    img = Image.open(f)
                    assert img.size == (20, 30)


def test_render_frames_targz_multi_format(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)

    # Multiple outputs with format "{output}"
    path = str(tmp_path/"{output}.tar.gz")
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in method.get_info()["supported_outputs"])
    evaluation.render_frames(method, cameras, 
                             output=path,
                             fps=1,
                             output_names=all_output_names)
    _verify_targz_multi_format(path, all_output_names, num_cams)


def test_render_frames_targz_multi_format_command(tmp_path):
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in FakeMethod().get_info()["supported_outputs"])
    _test_render_trajectory_command(tmp_path, "{output}.tar.gz", "--output-names", ",".join(all_output_names))
    path = str(tmp_path/"{output}.tar.gz")
    _verify_targz_multi_format(path, all_output_names, num_cams)


def _verify_zip_single(tmp_path, num_cams):
    assert os.path.exists(tmp_path/"out.zip")
    with zipfile.ZipFile(str(tmp_path/"out.zip"), mode="r") as zip:
        members = [x.filename for x in zip.infolist() if not x.is_dir()]
        assert len(members) == num_cams
        for i in range(num_cams):
            path = f"{i:05d}.png"
            assert f"{i:05d}.png" in members
            with zip.open(path) as f:
                img = Image.open(f)
                assert img.size == (20, 30)


def test_render_frames_zip_single(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)

    # Single output
    evaluation.render_frames(method, cameras, 
                             output=tmp_path/"out.zip",
                             fps=1,
                             output_names=("color",))
    _verify_zip_single(tmp_path, num_cams)


def test_render_frames_zip_single_command(tmp_path):
    _test_render_trajectory_command(tmp_path, "out.zip")
    _verify_zip_single(tmp_path, num_cams)


def _verify_zip_multi(tmp_path, num_cams, all_output_names):
    assert os.path.exists(tmp_path/"out-multi.zip")
    with zipfile.ZipFile(str(tmp_path/"out-multi.zip"), mode="r") as zip:
        members = [x.filename for x in zip.infolist() if not x.is_dir()]
        assert len(members) == num_cams * len(all_output_names)
        for out in all_output_names:
            for i in range(num_cams):
                path = f"{out}/{i:05d}.png"
                assert f"{out}/{i:05d}.png" in members
                with zip.open(path) as f:
                    img = Image.open(f)
                    assert img.size == (20, 30)


def test_render_frames_zip_multi(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)

    # Multiple outputs
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in method.get_info()["supported_outputs"])
    evaluation.render_frames(method, cameras, 
                             output=tmp_path/"out-multi.zip",
                             fps=1,
                             output_names=all_output_names)
    _verify_zip_multi(tmp_path, num_cams, all_output_names)


def test_render_frames_zip_multi_command(tmp_path):
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in FakeMethod().get_info()["supported_outputs"])
    _test_render_trajectory_command(tmp_path, "out-multi.zip", "--output-names", ",".join(all_output_names))
    _verify_zip_multi(tmp_path, num_cams, all_output_names)


def _verify_zip_multi_format(path, all_output_names, num_cams):
    for out in all_output_names:
        assert os.path.exists(path.format(output=out))
        with zipfile.ZipFile(path.format(output=out), mode="r") as zip:
            members = [x.filename for x in zip.infolist() if not x.is_dir()]
            assert len(members) == num_cams
            for i in range(num_cams):
                p = f"{i:05d}.png"
                assert p in members
                with zip.open(p) as f:
                    img = Image.open(f)
                    assert img.size == (20, 30)


def test_render_frames_zip_multi_format(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)

    # Multiple outputs with format "{output}"
    path = str(tmp_path/"{output}.zip")
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in method.get_info()["supported_outputs"])
    evaluation.render_frames(method, cameras, 
                             output=path,
                             fps=1,
                             output_names=all_output_names)
    _verify_zip_multi_format(path, all_output_names, num_cams)


def test_render_frames_zip_multi_format_command(tmp_path):
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in FakeMethod().get_info()["supported_outputs"])
    _test_render_trajectory_command(tmp_path, "{output}.zip", "--output-names", ",".join(all_output_names))
    path = str(tmp_path/"{output}.zip")
    _verify_zip_multi_format(path, all_output_names, num_cams)


def _verify_folder_single(tmp_path, num_cams):
    assert os.path.exists(tmp_path/"out")
    members = os.listdir(tmp_path/"out")
    assert len(members) == num_cams
    for i in range(num_cams):
        path = f"{i:05d}.png"
        assert f"{i:05d}.png" in members
        img = Image.open(tmp_path/"out"/path)
        assert img.size == (20, 30)
        img.close()


def test_render_frames_folder_single(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)
    evaluation.render_frames(method, cameras, 
                             output=tmp_path/"out",
                             fps=1,
                             output_names=("color",))
    _verify_folder_single(tmp_path, num_cams)


def test_render_frames_folder_single_command(tmp_path):
    _test_render_trajectory_command(tmp_path, "out")
    _verify_folder_single(tmp_path, num_cams)


def _verify_folder_multi(tmp_path, num_cams, all_output_names):
    assert os.path.exists(tmp_path/"out-multi")
    members = [os.path.relpath(os.path.join(root, x), tmp_path/"out-multi") for root, _, files in os.walk(tmp_path/"out-multi") for x in files]
    assert len(members) == num_cams * len(all_output_names)
    for out in all_output_names:
        for i in range(num_cams):
            assert os.path.exists(tmp_path/"out-multi"/out/f"{i:05d}.png")
            img = Image.open(tmp_path/"out-multi"/out/f"{i:05d}.png")
            assert img.size == (20, 30)


def test_render_frames_folder_multi(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)

    # Multiple outputs
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in method.get_info()["supported_outputs"])
    evaluation.render_frames(method, cameras, 
                             output=tmp_path/"out-multi",
                             fps=1,
                             output_names=all_output_names)
    _verify_folder_multi(tmp_path, num_cams, all_output_names)


def test_render_frames_folder_multi_command(tmp_path):
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in FakeMethod().get_info()["supported_outputs"])
    _test_render_trajectory_command(tmp_path, "out-multi", "--output-names", ",".join(all_output_names))
    _verify_folder_multi(tmp_path, num_cams, all_output_names)


def _verify_folder_multi_format(path, all_output_names, num_cams):
    for out in all_output_names:
        assert os.path.exists(path.format(output=out))
        members = os.listdir(path.format(output=out))
        assert len(members) == num_cams
        for i in range(num_cams):
            p = f"{i:05d}.png"
            assert p in members
            img = Image.open(os.path.join(path.format(output=out), p))
            assert img.size == (20, 30)


def test_render_frames_folder_multi_format(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)

    # Multiple outputs with format "{output}"
    path = str(tmp_path/"{output}")
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in method.get_info()["supported_outputs"])
    evaluation.render_frames(method, cameras, 
                             output=path,
                             fps=1,
                             output_names=all_output_names)
    _verify_folder_multi_format(path, all_output_names, num_cams)


def test_render_frames_folder_multi_format_command(tmp_path):
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in FakeMethod().get_info()["supported_outputs"])
    _test_render_trajectory_command(tmp_path, "{output}", "--output-names", ",".join(all_output_names))
    _verify_folder_multi_format(str(tmp_path/"{output}"), all_output_names, num_cams)


def _verify_mp4_single(tmp_path, num_cams):
    assert os.path.exists(tmp_path/"out.mp4")
    import mediapy

    try:
        video = mediapy.read_video(tmp_path/"out.mp4")
        assert len(video) == num_cams
        for frame in video:
            assert frame.shape == (30, 20, 3)
    except RuntimeError as e:
        # Skip because there is incompatibility with the mediapy library and ffmpeg
        if "Unable to find frames in video" in str(e):
            pytest.skip("Skip because of incompatibility between mediapy and ffmpeg")
        raise e


@pytest.mark.extras
def test_render_frames_mp4_single(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)

    # Single output
    evaluation.render_frames(method, cameras, 
                             output=tmp_path/"out.mp4",
                             fps=1,
                             output_names=("color",))
    _verify_mp4_single(tmp_path, num_cams)


@pytest.mark.extras
def test_render_frames_mp4_single_command(tmp_path):
    _test_render_trajectory_command(tmp_path, "out.mp4")
    _verify_mp4_single(tmp_path, num_cams)


def _verify_mp4_multi(tmp_path, num_cams, all_output_names):
    assert os.path.exists(tmp_path/"out-multi.mp4")
    import mediapy

    try:
        video = mediapy.read_video(tmp_path/"out-multi.mp4")
        assert len(video) == num_cams
        for frame in video:
            assert frame.shape == (30, 20*len(all_output_names), 3)
    except RuntimeError as e:
        # Skip because there is incompatibility with the mediapy library and ffmpeg
        if "Unable to find frames in video" in str(e):
            pytest.skip("Skip because of incompatibility between mediapy and ffmpeg")
        raise e


@pytest.mark.extras
def test_render_frames_mp4_multi(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)

    # Support multivideo by stacking
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in method.get_info()["supported_outputs"])
    evaluation.render_frames(method, cameras, 
                             output=tmp_path/"out-multi.mp4",
                             fps=1,
                             output_names=all_output_names)
    _verify_mp4_multi(tmp_path, num_cams, all_output_names)


@pytest.mark.extras
def test_render_frames_mp4_multi_command(tmp_path):
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in FakeMethod().get_info()["supported_outputs"])
    _test_render_trajectory_command(tmp_path, "out-multi.mp4", "--output-names", ",".join(all_output_names))
    _verify_mp4_multi(tmp_path, num_cams, all_output_names)


def _verify_mp4_multi_format(path, all_output_names, num_cams):
    import mediapy
    try:
        for out in all_output_names:
            assert os.path.exists(path.format(output=out))

            video = mediapy.read_video(path.format(output=out))
            assert len(video) == num_cams
            for frame in video:
                assert frame.shape == (30, 20, 3)
    except RuntimeError as e:
        # Skip because there is incompatibility with the mediapy library and ffmpeg
        if "Unable to find frames in video" in str(e):
            pytest.skip("Skip because of incompatibility between mediapy and ffmpeg")
        raise e

@pytest.mark.extras
def test_render_frames_mp4_multi_format(tmp_path):
    method = FakeMethod()
    cameras = _mock_cameras(num_cams)

    # Multiple outputs with format "{output}"
    path = str(tmp_path/"{output}.mp4")
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in method.get_info()["supported_outputs"])
    evaluation.render_frames(method, cameras, 
                             output=path,
                             fps=1,
                             output_names=all_output_names)

    # Test if output is correct
    _verify_mp4_multi_format(path, all_output_names, num_cams)


@pytest.mark.extras
def test_render_frames_mp4_multi_format_command(tmp_path):
    all_output_names = tuple(x if isinstance(x, str) else x["name"] for x in FakeMethod().get_info()["supported_outputs"])
    _test_render_trajectory_command(tmp_path, "{output}.mp4", "--output-names", ",".join(all_output_names))
    path = str(tmp_path/"{output}.mp4")
    _verify_mp4_multi_format(path, all_output_names, num_cams)
