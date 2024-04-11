import pytest
import numpy as np
from nerfbaselines import cameras
from nerfbaselines.types import get_args, camera_model_to_int, CameraModel
from nerfbaselines.types import new_cameras


@pytest.mark.parametrize("camera_type", get_args(CameraModel))
def test_distortions(camera_type: CameraModel):
    num_samples = 10000
    camera_types = np.full((num_samples,), camera_model_to_int(camera_type), dtype=np.int32)
    params = np.random.rand(num_samples, 8) * 0.01
    xy = np.random.rand(num_samples, 2) * 2 - 1
    xy *= 0.8

    xy_distorted = cameras._distort(camera_types, params, xy)
    if camera_type != "pinhole":
        assert not np.allclose(xy, xy_distorted, atol=5e-4, rtol=0), "distorted image should not be equal to original image"
    new_xy = cameras._undistort(camera_types, params, xy_distorted)
    np.testing.assert_allclose(xy, new_xy, atol=5e-4, rtol=0)


@pytest.mark.parametrize("camera_type", get_args(CameraModel))
def test_camera(camera_type):
    np.random.seed(42)
    num_cam = 10
    rotation = np.stack([np.linalg.qr(x)[0] for x in np.random.randn(num_cam, 3, 3)], 0)
    # rotation = np.eye(3)[None].repeat(num_cam, 0)
    pose = np.random.randn(num_cam, 3, 1)
    poses = np.concatenate([rotation, pose], -1)
    sizes = np.random.randint(200, 300, (num_cam, 2), dtype=np.int32)
    all_cameras = new_cameras(
        poses=poses,
        intrinsics=(np.random.randn(num_cam, 4) * 0.2 + np.array([1.0, 1.0, 0.5, 0.5])) * sizes[..., :1],
        camera_types=np.full((num_cam,), camera_model_to_int(camera_type), dtype=np.int32),
        distortion_parameters=np.random.rand(num_cam, 8).astype(np.float32) * 0.01,
        image_sizes=sizes,
        nears_fars=None,
    )

    xy = cameras.get_image_pixels(np.array([100, 200])).reshape(1, -1, 2).astype(np.float32)
    origins, dirs = cameras.unproject(all_cameras[:, None], xy)
    xy_new = cameras.project(all_cameras[:, None], origins + dirs)
    xy = np.broadcast_to(xy, xy_new.shape)
    np.testing.assert_allclose(xy, xy_new, atol=5e-4, rtol=0)


def _build_camera(camera_model: CameraModel, intrinsics, distortion_parameters):
    image_sizes = np.array([800, 600], dtype=np.int32)
    return new_cameras(
        poses=np.eye(4)[:3, :4],
        intrinsics=intrinsics,
        image_sizes=image_sizes,
        camera_types=np.array(camera_model_to_int(camera_model), dtype=np.int32),
        distortion_parameters=distortion_parameters,
        nears_fars=None,
    )


@pytest.mark.parametrize("camera_type", get_args(CameraModel))
def test_camera_undistort(camera_type):
    np.random.seed(42)
    num_cam = 3
    rotation = np.stack([np.linalg.qr(x)[0] for x in np.random.randn(num_cam, 3, 3)], 0)
    # rotation = np.eye(3)[None].repeat(num_cam, 0)
    pose = np.random.randn(num_cam, 3, 1)
    poses = np.concatenate([rotation, pose], -1)
    image_sizes=np.random.randint(100, 200, (num_cam, 2), dtype=np.int32)
    cams = new_cameras(
        poses=poses,
        intrinsics=(np.random.randn(num_cam, 4) * 0.1 + np.array([2.0, 2.0, 0.5, 0.5])) * image_sizes[..., :1],
        camera_types=np.full((num_cam,), camera_model_to_int(camera_type), dtype=np.int32),
        distortion_parameters=np.random.randn(num_cam, 8) * 0.001,
        image_sizes=image_sizes,
        nears_fars=None,
    )
    undistorted_cams = cameras.undistort_camera(cams)
    images = np.random.rand(num_cam, 200, 100, 3)
    cameras.warp_image_between_cameras(cams, undistorted_cams, images)


# # @pytest.mark.parametrize("camera_type", get_args(CameraModel))
# # def test_camera_undistort(camera_type):
# #     np.random.seed(42)
# #     num_cam = 3
# #     rotation = np.stack([np.linalg.qr(x)[0] for x in np.random.randn(num_cam, 3, 3)], 0)
# #     # rotation = np.eye(3)[None].repeat(num_cam, 0)
# #     pose = np.random.randn(num_cam, 3, 1)
# #     poses = np.concatenate([rotation, pose], -1)
# #     print(poses.shape)
# #     cameras = new_cameras(
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
    cam = new_cameras(poses=np.eye(4)[:3, :4], intrinsics=intrinsics, image_sizes=image_size, camera_types=np.array(camera_model_to_int(camera_model), dtype=np.int32), distortion_parameters=params, nears_fars=None)[None]
    np.testing.assert_allclose(cam.intrinsics[0], intrinsics, atol=1e-4, rtol=0)

    xy = cameras.project(cam, uvw0)
    _, uvw = cameras.unproject(cam, xy)
    uv0 = uvw0[..., :2] / uvw0[..., 2:]
    uv = uvw[..., :2] / uvw[..., 2:]
    np.testing.assert_allclose(uv, uv0, atol=1e-6, rtol=0)


def _test_cam_from_img_to_img(camera_model, intrinsics, params, xy0):
    image_size = np.array([800, 600], dtype=np.int32)
    cam = new_cameras(poses=np.eye(4)[:3, :4], intrinsics=intrinsics, image_sizes=image_size, camera_types=np.array(camera_model_to_int(camera_model), dtype=np.int32), distortion_parameters=params, nears_fars=None)[None]
    np.testing.assert_allclose(cam.intrinsics[0], intrinsics, atol=1e-4, rtol=0)

    _, directions = cameras.unproject(cam, xy0)
    xy = cameras.project(cam, directions)
    np.testing.assert_allclose(xy, xy0, atol=1e-6, rtol=0)


def _test_model(camera_model: CameraModel, intrinsics, params):
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
    camera_model = "pinhole"
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
    camera_model = "opencv"
    params = np.array([651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001], dtype=np.float64)
    intrinsics = params[:4]
    distortion = params[4:]
    _test_model(camera_model, intrinsics, distortion)


def test_opencv_fisheye():
    camera_model = "opencv_fisheye"
    params = np.array([651.123, 655.123, 386.123, 511.123, -0.471, 0.223, 0.0, 0.0, -0.001, 0.001], dtype=np.float64)
    intrinsics = params[:4]
    distortion = params[4:]
    _test_model(camera_model, intrinsics, distortion)


def test_full_opencv():
    camera_model = "full_opencv"
    params = np.array([651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001, 0.001, 0.02, -0.02, 0.001], dtype=np.float64)
    intrinsics = params[:4]
    distortion = params[4:]
    _test_model(camera_model, intrinsics, distortion)


def test_camera_undistort_opencv():
    cam = _build_camera("opencv", np.array([536.07343019, 536.01634475, 342.37038789, 235.53685636]), np.array([-0.27864655, 0.06717323, 0.00182394, -0.00034344]))
    ucam = cameras.undistort_camera(cam[None])[0]
    assert ucam.image_sizes is not None
    assert np.all(ucam.image_sizes > 500), "undistorted camera is wrong"
