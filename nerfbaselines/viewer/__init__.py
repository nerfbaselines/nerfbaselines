from pathlib import Path
import os
import logging
import json
from typing import Any, Optional
import typing

import click
import numpy as np

from ..datasets._colmap_utils import qvec2rotmat, rotmat2qvec
from ..io import open_any_directory, deserialize_nb_info
from ..utils import setup_logging, handle_cli_error
from ..types import Method
from .. import backends
from .. import registry


def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = qvec2rotmat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w


def get_position_quaternion(c2s):
    position = c2s[..., :3, 3]
    wxyz = np.stack([rotmat2qvec(x) for x in c2s[..., :3, :3].reshape(-1, 3, 3)], 0)
    wxyz = wxyz.reshape(c2s.shape[:-2] + (4,))
    return position, wxyz


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


def get_orientation_transform(poses):
    origins = poses[..., :3, 3]
    mean_origin = np.mean(origins, 0)
    translation = mean_origin
    up = np.mean(poses[:, :3, 1], 0)
    up = up / np.linalg.norm(up)

    rotation = rotation_matrix(up, np.array([0, 0, 1], dtype=up.dtype))
    transform = np.concatenate([rotation, rotation @ -translation[..., None]], -1)
    transform = np.concatenate([transform, np.array([[0, 0, 0, 1]], dtype=transform.dtype)], 0)
    return transform


@click.command("viewer")
@click.option("--checkpoint", default=None, required=False)
@click.option("--data", type=str, default=None, required=False)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@click.option("--port", type=int, default=6006)
@handle_cli_error
def main(checkpoint: str, data, verbose, backend, viewer="viser", port=6006):
    setup_logging(verbose)

    def run_viewer(method: Optional[Method] = None, nb_info=None):
        try:
            from .viser import run_viser_viewer

            run_viser_viewer(method, port=port, data=data, nb_info=nb_info)
        finally:
            if hasattr(method, "close"):
                typing.cast(Any, method).close()

    # Read method nb-info
    if checkpoint is not None:
        logging.info(f"Loading checkpoint {checkpoint}")
        with open_any_directory(checkpoint) as _checkpoint_path:
            checkpoint_path = Path(_checkpoint_path)
            assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
            assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
            with (checkpoint_path / "nb-info.json").open("r") as f:
                nb_info = json.load(f)
            nb_info = deserialize_nb_info(nb_info)

            method_name = nb_info["method"]
            backends.mount(checkpoint_path, checkpoint_path)
            with registry.build_method(method_name, backend=backend) as method_cls:
                method = method_cls(checkpoint=str(checkpoint_path))
                run_viewer(method, nb_info=nb_info)
    else:
        logging.info("Starting viewer without method")
        run_viewer()

    


if __name__ == "__main__":
    main()
