from typing import Optional, Sequence, Any, Tuple, TYPE_CHECKING
from enum import Enum
from dataclasses import dataclass
import dataclasses
import numpy as np
from .utils import cached_property


class CameraModel(Enum):
    PINHOLE = 0
    OPENCV = 1
    OPENCV_FISHEYE = 2


def _compute_residual_and_jacobian(x, y, xd, yd, distortion_params):
    """Auxiliary function of radial_and_tangential_undistort() that computes residuals and jacobians.
    Adapted from MultiNeRF:
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L427-L474

    Args:
        x: The updated x coordinates.
        y: The updated y coordinates.
        xd: The distorted x coordinates.
        yd: The distorted y coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].

    Returns:
        The residuals (fx, fy) and jacobians (fx_x, fx_y, fy_x, fy_y).
    """

    k1 = distortion_params[..., 0]
    k2 = distortion_params[..., 1]
    k3 = distortion_params[..., 2]
    k4 = distortion_params[..., 3]
    p1 = distortion_params[..., 4]
    p2 = distortion_params[..., 5]

    # let r(x, y) = x^2 + y^2;
    #     d(x, y) = 1 + k1 * r(x, y) + k2 * r(x, y) ^2 + k3 * r(x, y)^3 +
    #                   k4 * r(x, y)^4;
    r = x * x + y * y
    d = 1.0 + r * (k1 + r * (k2 + r * (k3 + r * k4)))

    # The perfect projection is:
    # xd = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2);
    # yd = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2);
    #
    # Let's define
    #
    # fx(x, y) = x * d(x, y) + 2 * p1 * x * y + p2 * (r(x, y) + 2 * x^2) - xd;
    # fy(x, y) = y * d(x, y) + 2 * p2 * x * y + p1 * (r(x, y) + 2 * y^2) - yd;
    #
    # We are looking for a solution that satisfies
    # fx(x, y) = fy(x, y) = 0;
    fx = d * x + 2 * p1 * x * y + p2 * (r + 2 * x * x) - xd
    fy = d * y + 2 * p2 * x * y + p1 * (r + 2 * y * y) - yd

    # Compute derivative of d over [x, y]
    d_r = k1 + r * (2.0 * k2 + r * (3.0 * k3 + r * 4.0 * k4))
    d_x = 2.0 * x * d_r
    d_y = 2.0 * y * d_r

    # Compute derivative of fx over x and y.
    fx_x = d + d_x * x + 2.0 * p1 * y + 6.0 * p2 * x
    fx_y = d_y * x + 2.0 * p1 * x + 2.0 * p2 * y

    # Compute derivative of fy over x and y.
    fy_x = d_x * y + 2.0 * p2 * y + 2.0 * p1 * x
    fy_y = d + d_y * y + 2.0 * p2 * x + 6.0 * p1 * y

    return fx, fy, fx_x, fx_y, fy_x, fy_y


def _radial_and_tangential_undistort(coords, distortion_params: np.ndarray, eps: float = 1e-3, max_iterations: int = 10, xnp=np):
    """Computes undistorted coords given opencv distortion parameters.
    Adapted from MultiNeRF
    https://github.com/google-research/multinerf/blob/b02228160d3179300c7d499dca28cb9ca3677f32/internal/camera_utils.py#L477-L509

    Args:
        coords: The distorted coordinates.
        distortion_params: The distortion parameters [k1, k2, k3, k4, p1, p2].
        eps: The epsilon for the convergence.
        max_iterations: The maximum number of iterations to perform.

    Returns:
        The undistorted coordinates.
    """

    # Initialize from the distorted point.
    distortion_params = distortion_params.reshape((-1,) + (1,) * (len(coords.shape) - 2) + (distortion_params.shape[-1],))
    xd, yd = coords[..., 0], coords[..., 1]
    x = xnp.copy(xd)
    y = xnp.copy(yd)

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(x=x, y=y, xd=xd, yd=yd, distortion_params=distortion_params)
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = xnp.where(xnp.abs(denominator) > eps, x_numerator / denominator, xnp.zeros_like(denominator))
        step_y = xnp.where(xnp.abs(denominator) > eps, y_numerator / denominator, xnp.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return xnp.stack([x, y], -1)


def _distort_ray_directions(directions, camera_types, distortion_params, xnp=np):
    """
    Distorts OpenCV directions according to the distortion parameters.

    Args:
        directions: [batch, ..., 3]
        camera_types: [batch]
        distortion_params: [batch, num_params]
        xnp: The numpy module to use.
    """
    mask = xnp.logical_or(camera_types == CameraModel.OPENCV_FISHEYE.value, camera_types == CameraModel.OPENCV.value)
    if xnp.any(mask):
        is_all_distorted = xnp.all(mask)
        mask = mask.expand(len(directions))
        if is_all_distorted:
            ldistortion_params = distortion_params
            dl = xnp.copy(directions)
            dcamera_types = camera_types
        else:
            ldistortion_params = distortion_params[mask]
            dl = directions[mask]
            dcamera_types = camera_types[mask]
        dl[..., :2] = _radial_and_tangential_undistort(
            dl,
            ldistortion_params,
            xnp=xnp,
        )

        fisheye_mask = dcamera_types == CameraModel.OPENCV_FISHEYE.value
        if xnp.any(fisheye_mask):
            is_all_fisheye = xnp.all(fisheye_mask)
            if is_all_fisheye:
                dll = dl
            else:
                dll = dl[mask]
            theta = xnp.sqrt(xnp.sum(xnp.square(dll[..., :2]), axis=-1))
            theta = xnp.minimum(xnp.pi, theta)
            sin_theta_over_theta = xnp.sin(theta) / theta
            if is_all_fisheye:
                dl[..., 2:] *= xnp.cos(theta)
                dl[..., :2] *= sin_theta_over_theta
            else:
                dl[mask, 2:] *= xnp.cos(theta)
                dl[mask, :2] *= sin_theta_over_theta
        if is_all_distorted:
            directions = dl
        else:
            directions = xnp.copy(directions)
            directions[mask, :] = dl
    return directions


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
        return self.normalized_intrinsics * self.image_sizes[..., :1]

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, index):
        return type(self)(
            poses=self.poses[index],
            normalized_intrinsics=self.normalized_intrinsics[index],
            camera_types=self.camera_types[index],
            distortion_parameters=self.distortion_parameters[index],
            image_sizes=self.image_sizes[index] if self.image_sizes is not None else None,
            nears_fars=self.nears_fars[index] if self.nears_fars is not None else None,
        )

    def get_rays(self, xy: np.ndarray, xnp: Any = np) -> Tuple[np.ndarray, np.ndarray]:
        if TYPE_CHECKING:
            onp = np
        else:
            onp = xnp
        assert xy.shape[-1] == 2
        assert xy.shape[0] == len(self)

        xy = xy.astype(onp.float32)
        fx, fy, cx, cy = onp.moveaxis(self.intrinsics, -1, 0)
        x = xy[..., 0]
        y = xy[..., 1]
        directions = onp.stack(((x + 0.5 - cx) / fx, (y + 0.5 - cy) / fy, onp.ones_like(xy[..., 0])), -1)

        directions = _distort_ray_directions(directions, self.camera_types, self.distortion_parameters, xnp=xnp)

        # Switch from OpenCV to OpenGL coordinate system
        directions[..., 1:] *= -1

        out_shape = xy.shape[:-1]
        rotation = self.poses[..., :3, :3]  # (..., 3, 3)
        directions = (directions[..., None, :] * rotation).sum(-1)
        origins = self.poses[..., :3, 3].expand(out_shape + (3,))
        return origins, directions

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
