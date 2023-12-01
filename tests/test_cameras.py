import pytest
import numpy as np
from nerfbaselines.cameras import Cameras, CameraModel
from nerfbaselines import cameras


@pytest.mark.parametrize("camera_type", CameraModel)
def test_distortions(camera_type):
    num_samples = 10000
    camera_types = np.full((num_samples,), camera_type.value, dtype=np.int32)
    params = np.random.rand(num_samples, 8) * 0.01
    xy = np.random.rand(num_samples, 2) * 2 - 1
    xy *= 0.8

    xy_distorted = cameras._distort(camera_types, params, xy)
    if camera_type != CameraModel.PINHOLE:
        assert not np.allclose(xy, xy_distorted, atol=5e-4, rtol=0), "distorted image should not be equal to original image"
    new_xy = cameras._undistort(camera_types, params, xy_distorted)
    np.testing.assert_allclose(xy, new_xy, atol=5e-4, rtol=0)


@pytest.mark.parametrize("camera_type", CameraModel)
def test_camera(camera_type):
    np.random.seed(42)
    num_cam = 10
    rotation = np.stack([np.linalg.qr(x)[0] for x in np.random.randn(num_cam, 3, 3)], 0)
    # rotation = np.eye(3)[None].repeat(num_cam, 0)
    pose = np.random.randn(num_cam, 3, 1)
    poses = np.concatenate([rotation, pose], -1)
    cameras = Cameras(
        poses=poses,
        normalized_intrinsics=np.random.randn(num_cam, 4) * 0.2 + np.array([1.0, 1.0, 0.5, 0.5]),
        camera_types=np.full((num_cam,), camera_type.value, dtype=np.int32),
        distortion_parameters=np.random.rand(num_cam, 8).astype(np.float32) * 0.01,
        image_sizes=np.random.randint(200, 300, (num_cam, 2), dtype=np.int32),
        nears_fars=None,
    )

    xy = np.stack(np.meshgrid(np.arange(0, 100), np.arange(0, 200), indexing="ij"), -1).reshape(1, -1, 2).astype(np.float32)
    origins, dirs = cameras[:, None].unproject(xy)
    xy_new = cameras[:, None].project(origins + dirs)
    xy = np.broadcast_to(xy, xy_new.shape)
    np.testing.assert_allclose(xy, xy_new, atol=5e-4, rtol=0)


def _build_camera(camera_model, intrinsics, distortion_parameters):
    image_sizes = np.array([800, 600], dtype=np.int32)
    return Cameras(
        poses=np.eye(4)[:3, :4],
        normalized_intrinsics=intrinsics / image_sizes[0],
        image_sizes=image_sizes,
        camera_types=np.array(camera_model.value, dtype=np.int32),
        distortion_parameters=distortion_parameters,
        nears_fars=None,
    )


@pytest.mark.parametrize("camera_type", CameraModel)
def test_camera_undistort(camera_type):
    np.random.seed(42)
    num_cam = 3
    rotation = np.stack([np.linalg.qr(x)[0] for x in np.random.randn(num_cam, 3, 3)], 0)
    # rotation = np.eye(3)[None].repeat(num_cam, 0)
    pose = np.random.randn(num_cam, 3, 1)
    poses = np.concatenate([rotation, pose], -1)
    cams = Cameras(
        poses=poses,
        normalized_intrinsics=np.random.randn(num_cam, 4) * 0.1 + np.array([2.0, 2.0, 0.5, 0.5]),
        camera_types=np.full((num_cam,), camera_type.value, dtype=np.int32),
        distortion_parameters=np.random.randn(num_cam, 8) * 0.001,
        image_sizes=np.random.randint(100, 200, (num_cam, 2), dtype=np.int32),
        nears_fars=None,
    )
    undistorted_cams = cameras.undistort_camera(cams)
    images = np.random.rand(num_cam, 200, 100, 3)
    cameras.warp_image_between_cameras(cams, undistorted_cams, images)


#
# # @pytest.mark.parametrize("camera_type", [CameraModel.OPENCV, CameraModel.PINHOLE, CameraModel.OPENCV_FISHEYE])
# # def test_camera_undistort(camera_type):
# #     np.random.seed(42)
# #     num_cam = 3
# #     rotation = np.stack([np.linalg.qr(x)[0] for x in np.random.randn(num_cam, 3, 3)], 0)
# #     # rotation = np.eye(3)[None].repeat(num_cam, 0)
# #     pose = np.random.randn(num_cam, 3, 1)
# #     poses = np.concatenate([rotation, pose], -1)
# #     print(poses.shape)
# #     cameras = Cameras(
# #         poses=poses,
# #         normalized_intrinsics=np.random.rand(num_cam, 4),
# #         camera_types=np.full((num_cam,), camera_type.value, dtype=np.int32),
# #         distortion_parameters=np.random.randn(num_cam, 6) * 0.1,
# #         image_sizes=np.random.randint(100, 200, (num_cam, 2), dtype=np.int32),
# #         nears_fars=None)
# #
# #     images = np.random.rand(num_cam, 200, 100, 3)
# #     nimg = undistort_images(cameras, images)
# #     assert nimg.shape == images.shape
# #
# #     nimg = undistort_images(dataclasses.replace(cameras, distortion_parameters=np.zeros_like(cameras.distortion_parameters)), images)
# #     assert nimg.shape == images.shape
# #     np.testing.assert_allclose(nimg, images, atol=1e-5, rtol=0)


def _test_cam_to_cam_from_img(camera_model, intrinsics, params, uvw0):
    image_size = np.array([800, 600], dtype=np.int32)
    cam = Cameras(np.eye(4)[:3, :4], intrinsics / image_size[0], image_sizes=image_size, camera_types=np.array(camera_model.value, dtype=np.int32), distortion_parameters=params, nears_fars=None)[None]
    np.testing.assert_allclose(cam.intrinsics[0], intrinsics, atol=1e-4, rtol=0)

    xy = cam.project(uvw0)
    _, uvw = cam.unproject(xy)
    uv0 = uvw0[..., :2] / uvw0[..., 2:]
    uv = uvw[..., :2] / uvw[..., 2:]
    np.testing.assert_allclose(uv, uv0, atol=1e-6, rtol=0)


def _test_cam_from_img_to_img(camera_model, intrinsics, params, xy0):
    image_size = np.array([800, 600], dtype=np.int32)
    cam = Cameras(np.eye(4)[:3, :4], intrinsics / image_size[0], image_sizes=image_size, camera_types=np.array(camera_model.value, dtype=np.int32), distortion_parameters=params, nears_fars=None)[None]
    np.testing.assert_allclose(cam.intrinsics[0], intrinsics, atol=1e-4, rtol=0)

    _, directions = cam.unproject(xy0)
    xy = cam.project(directions)
    np.testing.assert_allclose(xy, xy0, atol=1e-6, rtol=0)


def _test_model(camera_model, intrinsics, params):
    uv = (
        np.stack(np.meshgrid(np.linspace(-0.5, 0.5, 11, endpoint=True, dtype=np.float64), np.linspace(-0.5, 0.5, 11, endpoint=True, dtype=np.float64), indexing="ij"), -1)
        .reshape(-1, 2)
        .astype(np.float64)
    )
    uvw = np.concatenate((np.concatenate((uv, np.full_like(uv[..., :1], 1, dtype=uv.dtype)), -1), np.concatenate((uv, np.full_like(uv[..., :1], 2, dtype=uv.dtype)), -1)), 0)
    _test_cam_to_cam_from_img(camera_model, intrinsics, params, uvw)

    xy = np.stack(np.meshgrid(np.linspace(0, 800, 17, endpoint=True, dtype=np.float64), np.linspace(0, 800, 17, endpoint=True, dtype=np.float64), indexing="ij"), -1).reshape(-1, 2).astype(np.float64)
    _test_cam_from_img_to_img(camera_model, intrinsics, params, xy)
    intrinsics = intrinsics.copy()
    xyw = np.array([[intrinsics[2], intrinsics[3]]], dtype=intrinsics.dtype)
    _test_cam_from_img_to_img(camera_model, intrinsics, params, xyw)


# TEST(SimplePinhole, Nominal) {
#   TestModel<SimplePinholeCameraModel>({655.123, 386.123, 511.123});
# }


def test_pinhole():
    camera_model = CameraModel.PINHOLE
    params = np.array([651.123, 655.123, 386.123, 511.123], dtype=np.float64)
    intrinsics = params[:4]
    distortion = params[4:]
    _test_model(camera_model, intrinsics, distortion)


# TEST(SimpleRadial, Nominal) {
#   TestModel<SimpleRadialCameraModel>({651.123, 386.123, 511.123, 0});
#   TestModel<SimpleRadialCameraModel>({651.123, 386.123, 511.123, 0.1});
# }

# TEST(Radial, Nominal) {
#   TestModel<RadialCameraModel>({651.123, 386.123, 511.123, 0, 0});
#   TestModel<RadialCameraModel>({651.123, 386.123, 511.123, 0.1, 0});
#   TestModel<RadialCameraModel>({651.123, 386.123, 511.12, 0, 0.05});
#   TestModel<RadialCameraModel>({651.123, 386.123, 511.123, 0.05, 0.03});
# }


def test_opencv():
    camera_model = CameraModel.OPENCV
    params = np.array([651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001], dtype=np.float64)
    intrinsics = params[:4]
    distortion = params[4:]
    _test_model(camera_model, intrinsics, distortion)


def test_opencv_fisheye():
    camera_model = CameraModel.OPENCV_FISHEYE
    params = np.array([651.123, 655.123, 386.123, 511.123, -0.471, 0.223, 0.0, 0.0, -0.001, 0.001], dtype=np.float64)
    intrinsics = params[:4]
    distortion = params[4:]
    _test_model(camera_model, intrinsics, distortion)


def test_full_opencv():
    camera_model = CameraModel.FULL_OPENCV
    params = np.array([651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001, 0.001, 0.02, -0.02, 0.001], dtype=np.float64)
    intrinsics = params[:4]
    distortion = params[4:]
    _test_model(camera_model, intrinsics, distortion)


def test_camera_undistort_opencv():
    cam = _build_camera(CameraModel.OPENCV, np.array([536.07343019, 536.01634475, 342.37038789, 235.53685636]), np.array([-0.27864655, 0.06717323, 0.00182394, -0.00034344]))
    ucam = cameras.undistort_camera(cam[None])[0]
    assert np.all(ucam.image_sizes > 500), "undistorted camera is wrong"
