import sys
from typing import Tuple, Dict, cast, Any, TYPE_CHECKING
import numpy as np
from .utils import padded_stack, is_broadcastable, convert_image_dtype
from .types import Protocol, runtime_checkable, CameraModel, camera_model_to_int
from .types import Cameras, GenericCameras, TTensor, _get_xnp


def _xnp_astype(tensor: TTensor, dtype, xnp: Any) -> TTensor:
    if xnp.__name__ == "torch":
        return tensor.to(dtype)  # type: ignore
    return tensor.astype(dtype)  # type: ignore


def _xnp_copy(tensor: TTensor, xnp: Any = np) -> TTensor:
    if xnp.__name__ == "torch":
        return tensor.clone()  # type: ignore
    return xnp.copy(tensor)  # type: ignore


@runtime_checkable
class _DistortionFunction(Protocol):
    def __call__(self, distortion_params: TTensor, uv: TTensor, **kwargs) -> TTensor:
        ...


def _iterative_undistortion(distortion: _DistortionFunction, uv: TTensor, params: TTensor, num_iterations: int = 100, **kwargs) -> TTensor:
    xnp = _get_xnp(uv) if TYPE_CHECKING else kwargs["xnp"]
    # Source: https://github.com/colmap/colmap/blob/a6352b20a04ff8b426e9f591c31f5c3e8a46fa3f/src/colmap/sensor/models.h#L547
    # Parameters for Newton iteration using numerical differentiation with
    # central differences, 100 iterations should be enough even for complex
    # camera models with higher order terms.
    max_step_norm = 1e-10
    rel_step_size = 1e-6

    eps = float(xnp.finfo(params.dtype).eps)  # type: ignore
    assert len(uv.shape) == len(params.shape), "uv and params must have the same number of dimensions"
    new_uv_shape = tuple(map(max, uv.shape[:-1], params.shape[:-1])) + (2,)
    if uv.shape != new_uv_shape:
        uv = xnp.broadcast_to(uv, new_uv_shape)
    x = _xnp_copy(uv, xnp=xnp)
    # xout, mask = x, None

    for i in range(num_iterations):
        step = xnp.abs(rel_step_size * x).clip(eps, None)
        dx = distortion(params, x, xnp=xnp)
        dx_0b = distortion(params, xnp.stack((x[..., 0] - step[..., 0], x[..., 1]), -1), xnp=xnp)
        dx_0f = distortion(params, xnp.stack((x[..., 0] + step[..., 0], x[..., 1]), -1), xnp=xnp)
        dx_1b = distortion(params, xnp.stack((x[..., 0], x[..., 1] - step[..., 1]), -1), xnp=xnp)
        dx_1f = distortion(params, xnp.stack((x[..., 0], x[..., 1] + step[..., 1]), -1), xnp=xnp)
        J = xnp.stack(
            (
                1 + (dx_0f[..., 0] - dx_0b[..., 0]) / (2 * step[..., 0]),
                (dx_1f[..., 0] - dx_1b[..., 0]) / (2 * step[..., 1]),
                (dx_0f[..., 1] - dx_0b[..., 1]) / (2 * step[..., 0]),
                1 + (dx_1f[..., 1] - dx_1b[..., 1]) / (2 * step[..., 1]),
            ),
            -1,
        ).reshape((*dx.shape, 2))
        step_x = np.linalg.solve(J, (x + dx - uv)[..., None]).squeeze(-1)
        x -= step_x
        local_mask = (step_x**2).sum(-1) >= max_step_norm

        if not xnp.any(local_mask):
            break

    # NOTE: We might consider adding this speedup in future.
    #     # Mask here is to speedup the computation by only computing on
    #     # unconverged points. The gather/scatter is costly so we do it
    #     # only once every 10 steps.
    #     # This optimization can be removed without loss of correctness
    #     if i % 10 == 0 and i >= 30 and not xnp.all(local_mask):
    #         # Update mask once every 10 steps

    #         # Write out old values
    #         if mask is not None:
    #             xout[mask] = x

    #         # Update mask
    #         if mask is None:
    #             mask = local_mask
    #         else:
    #             mask[mask] = local_mask
    #
    #         # Params can be broadcastable to uv
    #         # We want to keep it that way so that we do not
    #         # expand used memory
    #         if params.size == params.shape[-1]:
    #             params = xnp.reshape(params, (1, params.shape[-1]))
    #         elif len(params.shape) > 2:
    #             params = xnp.broadcast_to(params, (*x.shape[:-1], *params.shape[-1:]))
    #             params = params[local_mask]
    #         else:
    #             params = params[local_mask]
    #         uv = uv[local_mask]
    #         x = x[local_mask]
    # if mask is not None:
    #     xout[mask] = x
    # return xout
    return cast(TTensor, x)


def _distort_opencv(distortion_params: TTensor, uv: TTensor, **kwargs) -> TTensor:
    xnp = _get_xnp(uv) if TYPE_CHECKING else kwargs["xnp"]
    u = uv[..., 0]
    v = uv[..., 1]
    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    p1 = distortion_params[..., 2]
    p2 = distortion_params[..., 3]

    u2 = u * u
    uv = u * v
    v2 = v * v
    r2 = u2 + v2
    radial = k1 * r2 + k2 * r2 * r2
    du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2)
    dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2)
    return xnp.stack((du, dv), -1)


def _distort_opencv_fisheye(distortion_params: TTensor, uv: TTensor, **kwargs) -> TTensor:
    xnp = _get_xnp(uv) if TYPE_CHECKING else kwargs["xnp"]
    eps = xnp.finfo(uv.dtype).eps
    u = uv[..., 0]
    v = uv[..., 1]
    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    # p1 = distortion_params[..., 2]
    # p2 = distortion_params[..., 3]
    k3 = distortion_params[..., 4]
    k4 = distortion_params[..., 5]
    r = xnp.sqrt(u * u + v * v)
    theta = xnp.arctan(r)
    theta2 = theta * theta
    theta4 = theta2 * theta2
    theta6 = theta4 * theta2
    theta8 = theta4 * theta4
    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
    du = xnp.where(r > eps, u * thetad / r.clip(eps, None) - u, xnp.zeros_like(u))
    dv = xnp.where(r > eps, v * thetad / r.clip(eps, None) - v, xnp.zeros_like(v))
    return xnp.stack((du, dv), -1)


def _distort_full_opencv(distortion_params: TTensor, uv: TTensor, **kwargs) -> TTensor:
    xnp = _get_xnp(uv) if TYPE_CHECKING else kwargs["xnp"]
    u = uv[..., 0]
    v = uv[..., 1]
    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    p1 = distortion_params[..., 2]
    p2 = distortion_params[..., 3]
    k3 = distortion_params[..., 4]
    k4 = distortion_params[..., 5]
    k5 = distortion_params[..., 6]
    k6 = distortion_params[..., 7]

    u2 = u * u
    uv = u * v
    v2 = v * v
    r2 = u2 + v2
    r4 = r2 * r2
    r6 = r4 * r2
    radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
    du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
    dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
    return xnp.stack((du, dv), -1)


_DISTORTIONS: Dict[CameraModel, _DistortionFunction] = {
    "opencv": _distort_opencv,
    "opencv_fisheye": _distort_opencv_fisheye,
    "full_opencv": _distort_full_opencv,
}


def _distort(camera_types, distortion_params, uv, xnp: Any = np):
    """
    Distorts OpenCV points according to the distortion parameters.

    Args:
        camera_types: [batch]
        distortion_params: [batch, num_params]
        uv: [batch, ..., 2]
        xnp: The numpy module to use.
    """
    pinhole_mask = camera_types == camera_model_to_int("pinhole")
    if xnp.all(pinhole_mask):
        return uv
    out = None
    for cam, distortion in _DISTORTIONS.items():
        mask = camera_types == camera_model_to_int(cam)
        if xnp.any(mask):
            if xnp.all(mask):
                return uv + distortion(distortion_params, uv, xnp=xnp)
            else:
                if out is None:
                    out = _xnp_copy(uv, xnp=xnp)
                out[mask] = uv[mask] + distortion(distortion_params[mask], uv[mask], xnp=xnp)
    if out is None:
        out = uv
    return out


def _undistort(camera_types: TTensor, distortion_params: TTensor, uv: TTensor, xnp: Any = np, **kwargs) -> TTensor:
    pinhole_mask = camera_types == camera_model_to_int("pinhole")
    if xnp.all(pinhole_mask):
        return uv
    out = None
    for cam, distortion in _DISTORTIONS.items():
        mask = camera_types == camera_model_to_int(cam)
        if xnp.any(mask):
            if xnp.all(mask):
                return _iterative_undistortion(distortion, uv, distortion_params, xnp=xnp, **kwargs)
            else:
                if out is None:
                    out = _xnp_copy(uv, xnp=xnp)
                out[mask] = cast(TTensor, _iterative_undistortion(distortion, uv[mask], distortion_params[mask], xnp=xnp, **kwargs))
    if out is None:
        out = uv
    return out


def get_image_pixels(image_sizes: TTensor) -> TTensor:
    xnp = _get_xnp(image_sizes)
    options = {}
    if len(image_sizes.shape) == 1:
        w, h = image_sizes
        if xnp.__name__ == "torch" and not TYPE_CHECKING:
            options = {"device": image_sizes.device}
        return xnp.stack(xnp.meshgrid(xnp.arange(w, **options), xnp.arange(h, **options), indexing="xy"), -1).reshape(-1, 2)
    return xnp.concatenate([get_image_pixels(s) for s in image_sizes])


def get_rays(cameras: GenericCameras[TTensor], xy: TTensor) -> Tuple[TTensor, TTensor]:
    xnp = _get_xnp(xy)
    assert xy.shape[-1] == 2
    assert xy.shape[0] == len(cameras)
    kind = getattr(xy.dtype, "kind", None)
    if kind is not None:
        assert kind in {"i", "u"}, "xy must be integer"

    xy = _xnp_astype(xy, xnp.float32, xnp=xnp) + 0.5
    return unproject(cameras, xy)


def unproject(cameras: GenericCameras[TTensor], xy: TTensor) -> Tuple[TTensor, TTensor]:
    xnp = _get_xnp(xy)
    assert xy.shape[-1] == 2
    assert is_broadcastable(xy.shape[:-1], cameras.poses.shape[:-2]), \
        "xy must be broadcastable with poses, shapes: {}, {}".format(xy.shape[:-1], cameras.poses.shape[:-2])
    if hasattr(xy.dtype, "kind"):
        if not TYPE_CHECKING:
            assert xy.dtype.kind == "f"
    fx: TTensor
    fy: TTensor
    cx: TTensor
    cy: TTensor
    fx, fy, cx, cy = xnp.moveaxis(cameras.intrinsics, -1, 0)
    x = xy[..., 0]
    y = xy[..., 1]
    u = (x - cx) / fx
    v = (y - cy) / fy

    uv = xnp.stack((u, v), -1)
    uv = _undistort(cameras.camera_types, cameras.distortion_parameters, uv, xnp=xnp)
    directions = xnp.concatenate((uv, xnp.ones_like(uv[..., :1])), -1)

    rotation = cameras.poses[..., :3, :3]  # (..., 3, 3)
    directions = (directions[..., None, :] * rotation).sum(-1)
    origins = xnp.broadcast_to(cameras.poses[..., :3, 3], directions.shape)
    return origins, directions


def project(cameras: GenericCameras[TTensor], xyz: TTensor) -> TTensor:
    xnp = _get_xnp(xyz)
    eps = xnp.finfo(xyz.dtype).eps  # type: ignore
    assert xyz.shape[-1] == 3
    assert is_broadcastable(xyz.shape[:-1], cameras.poses.shape[:-2]), \
        "xyz must be broadcastable with poses, shapes: {}, {}".format(xyz.shape[:-1], cameras.poses.shape[:-2])

    # World -> Camera
    origins = cameras.poses[..., :3, 3]
    rotation = cameras.poses[..., :3, :3]
    # Rotation and translation
    uvw = xyz - origins
    uvw = (rotation * uvw[..., :, None]).sum(-2)

    # Camera -> Camera distorted
    uv = xnp.where(uvw[..., 2:] > eps, uvw[..., :2] / uvw[..., 2:], xnp.zeros_like(uvw[..., :2]))

    uv = _distort(cameras.camera_types, cameras.distortion_parameters, uv, xnp=xnp)
    x, y = xnp.moveaxis(uv, -1, 0)

    # Transform to image coordinates
    # Camera distorted -> Image
    fx, fy, cx, cy = xnp.moveaxis(cameras.intrinsics, -1, 0)
    x = fx * x + cx
    y = fy * y + cy
    return xnp.stack((x, y), -1)


def interpolate_bilinear(image: TTensor, xy: TTensor) -> TTensor:
    xnp = _get_xnp(image)
    if xnp.__name__ == "torch":
        if not sys.modules["torch"].is_floating_point(xy):
            xy = xy.float()  # type: ignore
    elif getattr(xy.dtype, "kind", None) != "f":
        xy = _xnp_astype(xy, xnp.float32, xnp=xnp)
    original_shape = xy.shape
    xy = xnp.reshape(xy, (-1, 2))
    x, y = xy[..., 0], xy[..., 1]
    height, width = image.shape[:2]

    x0 = _xnp_astype(xnp.floor(x), xnp.int32, xnp=xnp)
    x1 = x0 + 1
    y0 = _xnp_astype(xnp.floor(y), xnp.int32, xnp=xnp)
    y1 = y0 + 1

    image = _xnp_astype(image, xnp.float32, xnp=xnp)
    mask = xnp.logical_and(
        xnp.logical_and(
            x0 >= 0,
            x1 < width,
        ),
        xnp.logical_and(
            y0 >= 0,
            y1 < height,
        ),
    )
    x0 = x0[mask]
    x1 = x1[mask]
    y0 = y0[mask]
    y1 = y1[mask]
    x = x[mask]
    y = y[mask]

    dx = x - x0
    dy = y - y0
    dx_1 = 1 - dx
    dy_1 = 1 - dy

    # Top row, column-wise linear interpolation.
    shape = x.shape + tuple(1 for _ in image.shape[2:])
    v0 = xnp.reshape(dx_1, shape) * image[y0, x0] + xnp.reshape(dx, shape) * image[y0, x1]
    # Bottom row, column-wise linear interpolation.
    v1 = xnp.reshape(dx_1, shape) * image[y1, x0] + xnp.reshape(dx, shape) * image[y1, x1]

    # Row-wise linear interpolation.
    kwargs = {}
    if xnp.__name__ == "torch" and not TYPE_CHECKING:
        kwargs = {"device": image.device}
    output = xnp.zeros(original_shape[:-1] + image.shape[2:], dtype=cast(Any, image.dtype), **kwargs)
    output_slice = xnp.reshape(dy_1, shape) * v0 + xnp.reshape(dy, shape) * v1
    output[mask] = _xnp_astype(output_slice, image.dtype, xnp=xnp)
    return output


def warp_image_between_cameras(cameras1: GenericCameras[np.ndarray], 
                               cameras2: GenericCameras[np.ndarray],
                               images: np.ndarray):
    xnp = _get_xnp(images)
    assert cameras1.image_sizes is not None, "cameras1 must have image sizes"
    assert cameras2.image_sizes is not None, "cameras2 must have image sizes"
    assert cameras1.image_sizes.shape == cameras2.image_sizes.shape, "Camera shapes must be the same"

    if len(cameras1.intrinsics.shape) == 2:
        out_image_list = []
        for cam1, cam2, image in zip(cameras1, cameras2, images):
            out_image_list.append(warp_image_between_cameras(cam1, cam2, image))
        return padded_stack(out_image_list)

    cam1 = cameras1
    cam2 = cameras2

    # NOTE: pyright workaround
    assert cam1.image_sizes is not None
    assert cam2.image_sizes is not None

    image = images

    # TODO: Fix aliasing issue
    # To avoid aliasing we rescale the output camera to input resolution and rescale images later
    # new_size = cam2.image_sizes
    # cam2 = dataclasses.replace(cam2, image_sizes=cam1.image_sizes)

    w, h = cam2.image_sizes
    xy = get_image_pixels(cam2.image_sizes)
    xy = xy.astype(xnp.float32) + 0.5

    # Camera models assume that the upper left pixel center is (0.5, 0.5).
    empty_poses = xnp.eye(4, dtype=cam2.poses.dtype)[:3, :4]
    _, cam_point = unproject(cam2.replace(poses=empty_poses)[None], xy)
    source_point = project(cam1.replace(poses=empty_poses)[None], cam_point)
    # Undo 0.5 offset
    source_point -= 0.5

    # Interpolate bilinear to obtain the image
    out_image = interpolate_bilinear(convert_image_dtype(image, xnp.float32), source_point)
    out_image = xnp.reshape(out_image, (h, w, -1))

    # TODO: Resize image

    # Cast image to original dtype
    out_image = convert_image_dtype(out_image, images.dtype)
    return out_image


def undistort_camera(camera: Cameras):
    assert camera.image_sizes is not None, "Camera must have image sizes"
    # xnp = _get_xnp(camera.image_sizes)
    original_camera = camera

    mask = camera.camera_types != camera_model_to_int("pinhole")
    if not np.any(mask):
        return camera

    camera = camera[mask]
    # NOTE: the following is a pyright workaround for the the 
    # fact that we cannot propagate not-null checks through the index operations
    assert camera.image_sizes is not None, "camera must have image sizes"
    assert original_camera.image_sizes is not None, "camera must have image sizes"

    # Scale the image such the the boundary of the undistorted image.
    empty_poses = np.eye(4, dtype=camera.poses.dtype)[None].repeat(len(camera), axis=0)[..., :3, :4]
    camera_empty = camera.replace(poses=empty_poses)
    camera_empty_undistorted = camera.replace(poses=empty_poses, camera_types=np.zeros_like(camera.camera_types), distortion_parameters=np.zeros_like(camera.distortion_parameters))
    assert len(camera_empty) == len(camera)

    # Determine min/max coordinates along top / bottom image border.
    w, h = camera.image_sizes.max(0)
    hrange = np.linspace(0.5, h + 0.5, h, endpoint=False, dtype=np.float32)[None].repeat(len(camera), axis=0)
    hmask = np.arange(h)[None, :] < camera.image_sizes[..., 1, None]
    _, point1_in_cam = unproject(camera_empty[:, None], np.stack((np.full_like(hrange, 0.5), hrange), -1))
    undistorted_point1 = project(camera_empty_undistorted[:, None], point1_in_cam)[..., 0]
    left_min_x = np.min(undistorted_point1, axis=1, where=hmask, initial=np.inf)
    left_max_x = np.max(undistorted_point1, axis=1, where=hmask, initial=-np.inf)

    _, point2_in_cam = unproject(camera_empty[:, None], np.stack((np.broadcast_to(camera.image_sizes[..., :1] - 0.5, hrange.shape), hrange), -1))
    undistorted_point2 = project(camera_empty_undistorted[:, None], point2_in_cam)[..., 0]
    right_min_x = np.min(undistorted_point2, axis=1, where=hmask, initial=np.inf)
    right_max_x = np.max(undistorted_point2, axis=1, where=hmask, initial=-np.inf)

    # Determine min, max coordinates along left / right image border.
    # Top border.
    wrange = np.linspace(0.5, w + 0.5, w, endpoint=False, dtype=np.float32)[None].repeat(len(camera), axis=0)
    wmask = np.arange(w)[None, :] < camera.image_sizes[..., 0, None]
    _, point1_in_cam = unproject(camera_empty[:, None], np.stack((wrange, np.full_like(wrange, 0.5)), -1))
    undistorted_point1 = project(camera_empty_undistorted[:, None], point1_in_cam)[..., 1]
    top_min_y = np.min(undistorted_point1, axis=1, where=wmask, initial=np.inf)
    top_max_y = np.max(undistorted_point1, axis=1, where=wmask, initial=-np.inf)
    # Bottom border.
    _, point2_in_cam = unproject(camera_empty[:, None], np.stack((wrange, np.broadcast_to(camera.image_sizes[..., 1:] - 0.5, wrange.shape)), -1))
    undistorted_point2 = project(camera_empty_undistorted[:, None], point2_in_cam)[..., 1]
    bottom_min_y = np.min(undistorted_point2, axis=1, where=wmask, initial=np.inf)
    bottom_max_y = np.max(undistorted_point2, axis=1, where=wmask, initial=-np.inf)

    fx, fy, cx, cy = np.moveaxis(camera.intrinsics, -1, 0)
    w, h = camera.image_sizes[..., 0], camera.image_sizes[..., 1]

    # Scale such that undistorted image contains all pixels of distorted image.
    min_scale_x = np.minimum(cx / (cx - left_min_x), (w - 0.5 - cx) / (right_max_x - cx))
    min_scale_y = np.minimum(cy / (cy - top_min_y), (h - 0.5 - cy) / (bottom_max_y - cy))

    # Scale such that there are no blank pixels in undistorted image.
    max_scale_x = np.maximum(cx / (cx - left_max_x), (w - 0.5 - cx) / (right_min_x - cx))
    max_scale_y = np.maximum(cy / (cy - top_max_y), (h - 0.5 - cy) / (bottom_min_y - cy))

    # Interpolate scale according to blank_pixels.
    blank_pixels = 0.0
    scale_x = 1.0 / (min_scale_x * blank_pixels + max_scale_x * (1.0 - blank_pixels))
    scale_y = 1.0 / (min_scale_y * blank_pixels + max_scale_y * (1.0 - blank_pixels))
    # Minimum and maximum scale change of camera used to satisfy the blank
    # pixel constraint.
    min_scale = 0.2
    max_scale = 2.0

    # Clip the scaling factors.
    scale_x = np.clip(scale_x, min_scale, max_scale)
    scale_y = np.clip(scale_y, min_scale, max_scale)

    # Scale undistorted camera dimensions.
    orig_undistorted_camera_width = w
    orig_undistorted_camera_height = h
    w = np.clip(scale_x * w, 1.0, None)
    h = np.clip(scale_y * h, 1.0, None)

    # Scale the principal point according to the new dimensions of the camera.
    cx = cx * w / orig_undistorted_camera_width
    cy = cy * h / orig_undistorted_camera_height

    undistorted_image_sizes = np.stack((w, h), -1)
    undistorted_intrinsics = np.stack((fx, fy, cx, cy), -1)

    # Get output
    if mask is None:
        intrinsics = undistorted_intrinsics
        image_sizes = undistorted_image_sizes
    else:
        intrinsics = np.copy(original_camera.intrinsics)
        image_sizes = np.copy(original_camera.image_sizes)
        intrinsics[mask] = undistorted_intrinsics
        image_sizes[mask] = undistorted_image_sizes

    out = vars(original_camera)
    out.update(dict(
        camera_types=np.full_like(original_camera.camera_types, camera_model_to_int("pinhole")),
        distortion_parameters=np.zeros_like(original_camera.distortion_parameters),
        intrinsics=intrinsics,
        image_sizes=image_sizes,
    ))
    return type(original_camera)(**out)
