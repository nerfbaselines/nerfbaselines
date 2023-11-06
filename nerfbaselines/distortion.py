import math
from enum import Enum
from dataclasses import dataclass
import numpy as np


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


def _radial_and_tangential_undistort(coords, distortion_params, eps: float = 1e-3, max_iterations: int = 10):
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
    x = coords[..., 0]
    y = coords[..., 1]

    for _ in range(max_iterations):
        fx, fy, fx_x, fx_y, fy_x, fy_y = _compute_residual_and_jacobian(
            x=x, y=y, xd=coords[..., 0], yd=coords[..., 1], distortion_params=distortion_params
        )
        denominator = fy_x * fx_y - fx_x * fy_y
        x_numerator = fx * fy_y - fy * fx_y
        y_numerator = fy * fx_x - fx * fy_x
        step_x = np.where(np.abs(denominator) > eps, x_numerator / denominator, np.zeros_like(denominator))
        step_y = np.where(np.abs(denominator) > eps, y_numerator / denominator, np.zeros_like(denominator))

        x = x + step_x
        y = y + step_y

    return np.stack([x, y], -1)


@dataclass(frozen=True)
class Distortions:
    camera_types: np.ndarray # [batch]
    distortion_params: np.ndarray # [batch num_params]

    def __len__(self):
        return len(self.camera_types)

    def __getitem__(self, index):
        return Distortions(
            camera_types=self.camera_types[index],
            distortion_params=self.distortion_params[index],
        )

    def distort(self, directions):
        mask = np.logical_or(
            self.camera_types == CameraModel.OPENCV_FISHEYE.value,
            self.camera_types == CameraModel.OPENCV.value)
        if mask.any():
            ldistortion_params = self.distortion_params[mask, :]
            mask = mask.expand(len(directions))
            dl = directions[mask, :]
            dl[..., :2] = _radial_and_tangential_undistort(
                dl,
                ldistortion_params,
            )

            dcamera_types = self.camera_types.expand(len(directions))[mask]
            fisheye_mask = dcamera_types == CameraModel.OPENCV_FISHEYE.value
            if fisheye_mask.any():
                dll = dl[mask, :2]

                theta = np.linalg.norm(dll, 2, axis=-1, keepdims=True)
                theta = np.clip(theta, 0.0, math.pi)

                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)
                dll = dll * sin_theta / theta
                dl[mask, 2:] *= cos_theta
                dl[mask, :2] = dll

        if mask.any():
            directions[mask, :] = dl
        return directions

    @classmethod
    def cat(cls, values):
        return cls(
            camera_types=np.concatenate([v.camera_types for v in values]),
            distortion_params=np.concatenate([v.distortion_params for v in values]),
        )
