from typing import Optional, Sequence, Tuple
from enum import Enum
from dataclasses import dataclass
import dataclasses
import numpy as np
from .utils import cached_property, padded_stack, is_broadcastable, convert_image_dtype
from .types import Protocol, runtime_checkable


class CameraModel(Enum):
    PINHOLE = 0
    OPENCV = 1
    OPENCV_FISHEYE = 2
    FULL_OPENCV = 3


@runtime_checkable
class _DistortionFunction(Protocol):
    def __call__(self, params: np.ndarray, uv: np.ndarray, xnp=np) -> np.ndarray:
        ...


def _iterative_undistortion(distortion: _DistortionFunction, uv: np.ndarray, params: np.ndarray, num_iterations: int = 100, xnp=np):
    # Source: https://github.com/colmap/colmap/blob/a6352b20a04ff8b426e9f591c31f5c3e8a46fa3f/src/colmap/sensor/models.h#L547
    # Parameters for Newton iteration using numerical differentiation with
    # central differences, 100 iterations should be enough even for complex
    # camera models with higher order terms.
    max_step_norm = 1e-10
    rel_step_size = 1e-6

    eps = xnp.finfo(params.dtype).eps
    assert len(uv.shape) == len(params.shape), "uv and params must have the same number of dimensions"
    new_uv_shape = tuple(map(max, uv.shape[:-1], params.shape[:-1])) + (2,)
    if uv.shape != new_uv_shape:
        uv = xnp.broadcast_to(uv, new_uv_shape)
    x = xnp.copy(uv)
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
        step_x = xnp.linalg.solve(J, x + dx - uv)
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
    return x


# def _compute_residual_and_jacobian(x, y, xd, yd, distortion_params):
#     """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
#     Adapted from MultiNeRF:
#     https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474
#
#     Args:
#         x: The updated x coordinates.
#         y: The updated y coordinates.
#         xd: The distorted x coordinates.
#         yd: The distorted y coordinates.
#         distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].
#
#     Returns:
#         The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
#     """
#
#     k1 = distortion_params[..., 0]
#     k2 = distortion_params[..., 1]
#     p1 = distortion_params[..., 2]
#     p2 = distortion_params[..., 3]
#     k3 = distortion_params[..., 4]
#     k4 = distortion_params[..., 5]
#
#     # let r(x, y) = x^2 + y^2;
#     #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
#     #                   k4 * r(x, y)^4;
#     r = x * x + y * y
#     d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))
#
#     # The perfect projection is:
#     # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
#     # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
#     #
#     # Let's define
#     #
#     # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
#     # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
#     #
#     # We are looking for a solution that satisfies
#     # fx(x, y) = fy(x, y) = 0;
#     fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
#     fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd
#
#     # Compute derivative of d over [x, y]
#     d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
#     d_x = 2.0 * x * d_r
#     d_y = 2.0 * y * d_r
#
#     # Compute derivative of fx over x and y.
#     fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
#     fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y
#
#     # Compute derivative of fy over x and y.
#     fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
#     fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y
#
#     return fx, fy, fx_x, fx_y, fy_x, fy_y
#
#
#
#
# def _radial_and_tangential_undistort(distortion_params: np.ndarray, uv: np.ndarray, eps: float = 1e-3, max_iterations: int = 10, xnp=np):
#     """Computes undistorted coords given opencv distortion parameters.
#     Adapted from MultiNeRF
#     https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509
#
#     Args:
#         coords: The distorted coordinates.
#         distortion_params: The distortion parameters [k1, k2, p1, p2, k3, k4].
#         eps: The epsilon for the convergence.
#         max_iterations: The maximum number of iterations to perform.
#
#     Returns:
#         The undistorted coordinates.
#     """
#
#     # Initialize from the distorted point.
#     distortion_params = distortion_params.reshape((-1,) + (1,) * (len(uv.shape) - 2) + (distortion_params.shape[-1],))
#     xd, yd = uv[..., 0], uv[..., 1]
#     x = xnp.copy(xd)
#     y = xnp.copy(yd)
#
#     for _ in range(max_iterations):
#         fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(x=x, y=y, xd=xd, yd=yd, distortion_params=distortion_params)
#         denominator = fy_x * fx_y - fx_x * fy_y
#         x_numerator = fx * fy_y - fy * fx_y
#         y_numerator = fy * fx_x - fx * fy_x
#         step_x = xnp.where(xnp.abs(denominator) > eps, x_numerator / denominator, xnp.zeros_like(denominator))
#         step_y = xnp.where(xnp.abs(denominator) > eps, y_numerator / denominator, xnp.zeros_like(denominator))
#
#         x = x + step_x
#         y = y + step_y
#
#     return xnp.stack([x, y], -1)
#
#
# def _undistort(camera_types, distortion_params, uv, xnp=np):
#     """
#     Undistorts OpenCV directions according to the distortion parameters.
#
#     Args:
#         uv: [batch, ..., 2]
#         camera_types: [batch]
#         distortion_params: [batch, num_params]
#         xnp: The numpy module to use.
#     """
#     directions = xnp.concatenate(
#         (uv, xnp.ones_like(uv[..., :1])),
#         -1,
#     )
#     mask = xnp.logical_or(camera_types == CameraModel.OPENCV_FISHEYE.value, camera_types == CameraModel.OPENCV.value)
#     if xnp.any(mask):
#         is_all_distorted = xnp.all(mask)
#         if is_all_distorted:
#             ldistortion_params = distortion_params
#             dl = directions
#             dcamera_types = camera_types
#         else:
#             ldistortion_params = distortion_params[mask]
#             dl = directions[mask]
#             dcamera_types = camera_types[mask]
#         dl[..., :2] = _radial_and_tangential_undistort(
#             ldistortion_params,
#             dl,
#             xnp=xnp,
#         )
#
#         fisheye_mask = dcamera_types == CameraModel.OPENCV_FISHEYE.value
#         if xnp.any(fisheye_mask):
#             is_all_fisheye = xnp.all(fisheye_mask)
#             if is_all_fisheye:
#                 dll = dl
#             else:
#                 dll = dl[mask]
#             theta = xnp.sqrt(xnp.sum(xnp.square(dll[..., :2]), axis=-1, keepdims=True))
#             theta = xnp.minimum(xnp.pi, theta)
#             sin_theta_over_theta = xnp.sin(theta) / theta
#             if is_all_fisheye:
#                 dl[..., 2:] *= xnp.cos(theta)
#                 dl[..., :2] *= sin_theta_over_theta
#             else:
#                 dl[mask, 2:] *= xnp.cos(theta)
#                 dl[mask, :2] *= sin_theta_over_theta
#         if is_all_distorted:
#             directions = dl
#         else:
#             directions = xnp.copy(directions)
#             directions[mask, :] = dl
#     return directions[..., :2] / directions[..., 2:]


def _distort_opencv(distortion_params, uv, xnp=np):
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


def _distort_opencv_fisheye(distortion_params, uv, xnp=np):
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


def _distort_full_opencv(distortion_params, uv, xnp=np):
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


_DISTORTIONS = {
    CameraModel.OPENCV: _distort_opencv,
    CameraModel.OPENCV_FISHEYE: _distort_opencv_fisheye,
    CameraModel.FULL_OPENCV: _distort_full_opencv,
}


def _distort(camera_types, distortion_params, uv, xnp=np):
    """
    Distorts OpenCV points according to the distortion parameters.

    Args:
        camera_types: [batch]
        distortion_params: [batch, num_params]
        uv: [batch, ..., 2]
        xnp: The numpy module to use.
    """
    pinhole_mask = camera_types == CameraModel.PINHOLE.value
    if xnp.all(pinhole_mask):
        return uv
    out = None
    for cam, distortion in _DISTORTIONS.items():
        mask = camera_types == cam.value
        if xnp.any(mask):
            if xnp.all(mask):
                return uv + distortion(distortion_params, uv, xnp=xnp)
            else:
                if out is None:
                    out = xnp.copy(uv)
                out[mask] = uv[mask] + distortion(distortion_params[mask], uv[mask], xnp=xnp)
    return out


def _undistort(camera_types: np.ndarray, distortion_params: np.ndarray, uv: np.ndarray, xnp=np, **kwargs):
    pinhole_mask = camera_types == CameraModel.PINHOLE.value
    if xnp.all(pinhole_mask):
        return uv
    out = None
    for cam, distortion in _DISTORTIONS.items():
        mask = camera_types == cam.value
        if xnp.any(mask):
            if xnp.all(mask):
                return _iterative_undistortion(distortion, uv, distortion_params, xnp=xnp, **kwargs)
            else:
                if out is None:
                    out = xnp.copy(uv)
                out[mask] = _iterative_undistortion(distortion, uv[mask], distortion_params[mask], xnp=xnp, **kwargs)


@dataclass(frozen=True)
class Cameras:
    poses: np.ndarray  # [N, (R, t)]
    normalized_intrinsics: np.ndarray  # [N, (fx,fy,cx,cy)]

    # Distortions
    camera_types: np.ndarray  # [N]
    distortion_parameters: np.ndarray  # [N, num_params]

    image_sizes: Optional[np.ndarray]  # [N, 2]
    nears_fars: Optional[np.ndarray]  # [N, 2]

    @cached_property
    def intrinsics(self):
        assert self.image_sizes is not None
        assert self.normalized_intrinsics.shape[:-1] == self.image_sizes.shape[:-1], "normalized_intrinsics and image_sizes must be broadcastable"
        return self.normalized_intrinsics * self.image_sizes[..., :1]

    def __len__(self):
        if len(self.poses.shape) == 2:
            return 1
        return len(self.poses)

    def item(self):
        assert len(self) == 1, "Cameras must have exactly one element to be converted to a single camera"
        if len(self.poses.shape) == 2:
            return self
        return self[0]

    def __getitem__(self, index):
        return type(self)(
            poses=self.poses[index],
            normalized_intrinsics=self.normalized_intrinsics[index],
            camera_types=self.camera_types[index],
            distortion_parameters=self.distortion_parameters[index],
            image_sizes=self.image_sizes[index] if self.image_sizes is not None else None,
            nears_fars=self.nears_fars[index] if self.nears_fars is not None else None,
        )

    def __setitem__(self, index, value):
        assert (self.image_sizes is None) == (value.image_sizes is None), "Either both or none of the cameras must have image sizes"
        assert (self.nears_fars is None) == (value.nears_fars is None), "Either both or none of the cameras must have nears and fars"
        self.poses[index] = value.poses
        self.normalized_intrinsics[index] = value.normalized_intrinsics
        self.camera_types[index] = value.camera_types
        self.distortion_parameters[index] = value.distortion_parameters
        if self.image_sizes is not None:
            self.image_sizes[index] = value.image_sizes
        if self.nears_fars is not None:
            self.nears_fars[index] = value.nears_fars

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_rays(self, xy: np.ndarray, xnp=np) -> Tuple[np.ndarray, np.ndarray]:
        assert xy.shape[-1] == 2
        assert xy.shape[0] == len(self)
        assert xy.dtype.kind in {"i", "u"}, "xy must be integer"

        xy = xy.astype(xnp.float32) + 0.5
        return self.unproject(xy, xnp=xnp)

    def unproject(self, xy: np.ndarray, xnp=np) -> Tuple[np.ndarray, np.ndarray]:
        assert xy.shape[-1] == 2
        assert is_broadcastable(xy.shape[:-1], self.poses.shape[:-2]), "xy must be broadcastable with poses, shapes: {}, {}".format(xy.shape[:-1], self.poses.shape[:-2])
        assert xy.dtype.kind == "f"
        fx, fy, cx, cy = xnp.moveaxis(self.intrinsics, -1, 0)
        x = xy[..., 0]
        y = xy[..., 1]
        u = (x - cx) / fx
        v = (y - cy) / fy

        uv = xnp.stack((u, v), -1)
        uv = _undistort(self.camera_types, self.distortion_parameters, uv, xnp=xnp)
        directions = xnp.concatenate((uv, xnp.ones_like(uv[..., :1])), -1)

        # Switch from OpenCV to OpenGL coordinate system
        directions[..., 1:] *= -1

        rotation = self.poses[..., :3, :3]  # (..., 3, 3)
        directions = (directions[..., None, :] * rotation).sum(-1)
        origins = xnp.broadcast_to(self.poses[..., :3, 3], directions.shape)
        return origins, directions

    def project(self, xyz: np.ndarray, xnp=np) -> np.ndarray:
        eps = xnp.finfo(xyz.dtype).eps
        assert xyz.shape[-1] == 3
        assert is_broadcastable(xyz.shape[:-1], self.poses.shape[:-2]), "xyz must be broadcastable with poses, shapes: {}, {}".format(xyz.shape[:-1], self.poses.shape[:-2])

        # World -> Camera
        origins = self.poses[..., :3, 3]
        rotation = self.poses[..., :3, :3]
        # Rotation and translation
        uvw = xyz - origins
        uvw = (rotation * uvw[..., :, None]).sum(-2)
        # Switch from OpenGL to OpenCV coordinate system
        uvw[..., 1:] *= -1

        # Camera -> Camera distorted
        uv = xnp.divide(uvw[..., :2], uvw[..., 2:], out=xnp.zeros_like(uvw[..., :2]), where=xnp.abs(uvw[..., 2:]) > eps)

        uv = _distort(self.camera_types, self.distortion_parameters, uv, xnp=xnp)
        x, y = xnp.moveaxis(uv, -1, 0)

        # Transform to image coordinates
        # Camera distorted -> Image
        fx, fy, cx, cy = xnp.moveaxis(self.intrinsics, -1, 0)
        x = fx * x + cx
        y = fy * y + cy
        return xnp.stack((x, y), -1)

    @classmethod
    def cat(cls, values: Sequence["Cameras"]) -> "Cameras":
        return cls(
            poses=np.concatenate([v.poses for v in values]),
            normalized_intrinsics=np.concatenate([v.normalized_intrinsics for v in values]),
            camera_types=np.concatenate([v.camera_types for v in values]),
            distortion_parameters=np.concatenate([v.distortion_parameters for v in values]),
            image_sizes=np.concatenate([v.image_sizes for v in values]) if any(v.image_sizes is not None for v in values) else None,
            nears_fars=np.concatenate([v.nears_fars for v in values]) if any(v.nears_fars is not None for v in values) else None,
        )

    def with_image_sizes(self, image_sizes: np.ndarray) -> "Cameras":
        return dataclasses.replace(self, image_sizes=image_sizes)


# def undistort_images(cameras: Cameras, images: np.ndarray) -> np.ndarray:
#     """
#     Undistort images
#
#     Args:
#         cameras: Original distorted cameras [N_cams]
#         images: Images to undistort [N_cams, H, W, C]
#     Returns:
#         undistorted images: [N_cams, H, W, C]
#     """
#     assert len(cameras) == len(images), "Number of cameras and images must be the same"
#     outputs = []
#     undistorted_cameras = cameras.undistort()
#     for cam, cam_target, img in zip(cameras, undistorted_cameras, images):
#         if cam.camera_types.item() != CameraModel.PINHOLE.value:
#             # Undistort
#             w, h = cam.image_sizes
#             outputs.append(warp_image_between_cameras(cam, cam_target, img[:h, :w]))
#         else:
#             outputs.append(img)
#     return padded_stack(outputs)


def interpolate_bilinear(image: np.ndarray, xy: np.ndarray, xnp=np) -> np.ndarray:
    if xy.dtype.kind != "f":
        xy = xy.astype(xnp.float32)
    original_shape = xy.shape
    xy = xnp.reshape(xy, (-1, 2))
    x, y = xy[..., 0], xy[..., 1]
    height, width = image.shape[:2]

    x0 = xnp.floor(x).astype(xnp.int32)
    x1 = x0 + 1
    y0 = xnp.floor(y).astype(xnp.int32)
    y1 = y0 + 1

    image = image.astype(np.float32)
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
    output = xnp.zeros(original_shape[:-1] + image.shape[2:], dtype=image.dtype)
    output[mask] = (xnp.reshape(dy_1, shape) * v0 + xnp.reshape(dy, shape) * v1).astype(image.dtype)
    return output


def warp_image_between_cameras(cameras1: Cameras, cameras2: Cameras, images: np.ndarray):
    xnp = np
    assert cameras1.image_sizes.shape == cameras2.image_sizes.shape, "Camera shapes must be the same"

    if len(cameras1.normalized_intrinsics.shape) == 2:
        out_image_list = []
        for cam1, cam2, image in zip(cameras1, cameras2, images):
            out_image_list.append(warp_image_between_cameras(cam1, cam2, image))
        return padded_stack(out_image_list)

    cam1 = cameras1
    cam2 = cameras2
    image = images

    # TODO: Fix aliasing issue
    # To avoid aliasing we rescale the output camera to input resolution and rescale images later
    # new_size = cam2.image_sizes
    # cam2 = dataclasses.replace(cam2, image_sizes=cam1.image_sizes)

    w, h = cam2.image_sizes
    xy = xnp.stack(xnp.meshgrid(xnp.arange(w), xnp.arange(h), indexing="xy"), -1).reshape(-1, 2)
    xy = xy.astype(xnp.float32) + 0.5

    # Camera models assume that the upper left pixel center is (0.5, 0.5).
    empty_poses = xnp.eye(4, dtype=cam2.poses.dtype)[:3, :4]
    _, cam_point = dataclasses.replace(cam2, poses=empty_poses)[None].unproject(xy, xnp=xnp)
    source_point = dataclasses.replace(cam1, poses=empty_poses)[None].project(cam_point, xnp=xnp)
    # Undo 0.5 offset
    source_point -= 0.5

    # Interpolate bilinear to obtain the image
    out_image = interpolate_bilinear(image, source_point, xnp=xnp)
    out_image = xnp.reshape(out_image, (h, w, -1))

    # TODO: Resize image

    # Cast image to original dtype
    out_image = convert_image_dtype(out_image, images.dtype)
    return out_image


def undistort_camera(camera: Cameras, xnp=np):
    original_camera = camera

    mask = camera.camera_types != CameraModel.PINHOLE.value
    if not np.any(mask):
        return camera

    camera = camera[mask]

    # Scale the image such the the boundary of the undistorted image.
    empty_poses = np.eye(4, dtype=camera.poses.dtype)[None].repeat(len(camera), axis=0)[..., :3, :4]
    camera_empty = dataclasses.replace(camera, poses=empty_poses)
    camera_empty_undistorted = dataclasses.replace(camera, poses=empty_poses, camera_types=np.zeros_like(camera.camera_types), distortion_parameters=np.zeros_like(camera.distortion_parameters))
    assert len(camera_empty) == len(camera)

    # Determine min/max coordinates along top / bottom image border.
    w, h = camera.image_sizes.max(0)
    hrange = np.linspace(0.5, h + 0.5, h, endpoint=False, dtype=np.float32)[None].repeat(len(camera), axis=0)
    hmask = np.arange(h)[None, :] < camera.image_sizes[..., 1, None]
    _, point1_in_cam = camera_empty[:, None].unproject(np.stack((np.full_like(hrange, 0.5), hrange), -1), xnp=xnp)
    undistorted_point1 = camera_empty_undistorted[:, None].project(point1_in_cam, xnp=xnp)[..., 0]
    left_min_x = np.min(undistorted_point1, axis=1, where=hmask, initial=np.inf)
    left_max_x = np.max(undistorted_point1, axis=1, where=hmask, initial=-np.inf)

    _, point2_in_cam = camera_empty[:, None].unproject(np.stack((np.broadcast_to(camera.image_sizes[..., :1] - 0.5, hrange.shape), hrange), -1), xnp=xnp)
    undistorted_point2 = camera_empty_undistorted[:, None].project(point2_in_cam, xnp=xnp)[..., 0]
    right_min_x = np.min(undistorted_point2, axis=1, where=hmask, initial=np.inf)
    right_max_x = np.max(undistorted_point2, axis=1, where=hmask, initial=-np.inf)

    # Determine min, max coordinates along left / right image border.
    # Top border.
    wrange = np.linspace(0.5, w + 0.5, w, endpoint=False, dtype=np.float32)[None].repeat(len(camera), axis=0)
    wmask = np.arange(w)[None, :] < camera.image_sizes[..., 0, None]
    _, point1_in_cam = camera_empty[:, None].unproject(np.stack((wrange, np.full_like(wrange, 0.5)), -1), xnp=xnp)
    undistorted_point1 = camera_empty_undistorted[:, None].project(point1_in_cam, xnp=xnp)[..., 1]
    top_min_y = np.min(undistorted_point1, axis=1, where=wmask, initial=np.inf)
    top_max_y = np.max(undistorted_point1, axis=1, where=wmask, initial=-np.inf)
    # Bottom border.
    _, point2_in_cam = camera_empty[:, None].unproject(np.stack((wrange, np.broadcast_to(camera.image_sizes[..., 1:] - 0.5, wrange.shape)), -1), xnp=xnp)
    undistorted_point2 = camera_empty_undistorted[:, None].project(point2_in_cam, xnp=xnp)[..., 1]
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
    undistorted_normalized_intrinsics = np.stack((fx, fy, cx, cy), -1) / w[:, None]

    # Get output
    if mask is None:
        normalized_intrinsics = undistorted_normalized_intrinsics
        image_sizes = undistorted_image_sizes
    else:
        normalized_intrinsics = np.copy(original_camera.normalized_intrinsics)
        image_sizes = np.copy(original_camera.image_sizes)
        normalized_intrinsics[mask] = undistorted_normalized_intrinsics
        image_sizes[mask] = undistorted_image_sizes
    return dataclasses.replace(
        original_camera,
        camera_types=np.full_like(original_camera.camera_types, CameraModel.PINHOLE.value),
        distortion_parameters=np.zeros_like(original_camera.distortion_parameters),
        normalized_intrinsics=normalized_intrinsics,
        image_sizes=image_sizes,
    )
