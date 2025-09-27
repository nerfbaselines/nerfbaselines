import numpy as np


def pad_poses(p):
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.0], p[..., :1, :4].shape)
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)


def unpad_poses(p):
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]


def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (np.random.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    s = np.linalg.norm(v)
    skew_sym_mat = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ],
        dtype=a.dtype,
    )
    return np.eye(3, dtype=a.dtype) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def viewmatrix(
    lookdir,
    up,
    position,
    lock_up = False,
):
    """Construct lookat view matrix."""
    def normalize(x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def orthogonal_dir(a, b): 
        return normalize(np.cross(a, b))

    vecs = [None, normalize(up), normalize(lookdir)]
    # x-axis is always the normalized cross product of `lookdir` and `up`.
    vecs[0] = orthogonal_dir(vecs[1], vecs[2])
    # Default is to lock `lookdir` vector, if lock_up is True lock `up` instead.
    ax = 2 if lock_up else 1
    # Set the not-locked axis to be orthogonal to the other two.
    vecs[ax] = orthogonal_dir(vecs[(ax + 1) % 3], vecs[(ax + 2) % 3])
    m = np.stack(vecs + [position], axis=1)
    return m


def get_transform_and_scale(transform):
    assert len(transform.shape) == 2, "Transform should be a 4x4 or a 3x4 matrix."
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    assert np.allclose(scale, scale[0], rtol=1e-3, atol=0)
    scale = float(np.mean(scale).item())
    transform = transform.copy()
    transform[:3, :] /= scale
    return transform, scale


def apply_transform(transform, poses):
    transform, scale = get_transform_and_scale(transform)
    poses = unpad_poses(transform @ pad_poses(poses))
    poses[..., :3, 3] *= scale
    return poses


def invert_transform(transform, has_scale=False):
    scale = None
    if has_scale:
        transform, scale = get_transform_and_scale(transform)
    else:
        transform = transform.copy()
    R = transform[..., :3, :3]
    t = transform[..., :3, 3]
    transform[..., :3, :] = np.concatenate([R.T, -np.matmul(R.T, t[..., None])], axis=-1)
    if scale is not None:
        transform[..., :3, :3] *= 1/scale
    return transform

