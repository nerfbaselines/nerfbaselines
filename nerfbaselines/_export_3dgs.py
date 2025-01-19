import shutil
import shlex
import subprocess
import tempfile
import os
from typing import Union
import numpy as np
try:
    from typing import get_origin, get_args
except ImportError:
    from typing_extensions import get_origin, get_args


def _inverse_sigmoid(x):
    return np.log(x) - np.log(1 - x)


def _cast_value(tp, value):
    origin = get_origin(tp)
    if origin is Union:
        for t in get_args(tp):
            try:
                return _cast_value(t, value)
            except ValueError:
                pass
            except TypeError:
                pass
        raise TypeError(f"Value {value} is not in {tp}")
    if tp is type(None):
        if str(value).lower() == "none":
            return None
        else:
            raise TypeError(f"Value {value} is not None")
    if tp is bool:
        if str(value).lower() in {"true", "1", "yes"}:
            return True
        elif str(value).lower() in {"false", "0", "no"}:
            return False
        else:
            raise TypeError(f"Value {value} is not a bool")
    if tp in {int, float, bool, str}:
        return tp(value)
    if isinstance(value, tp):
        return value
    raise TypeError(f"Cannot cast value {value} to type {tp}")


def generate_ksplat_file(path: str,
                         means: np.ndarray,
                         scales: np.ndarray,
                         opacities: np.ndarray,
                         quaternions: np.ndarray,
                         spherical_harmonics: np.ndarray):
    from plyfile import PlyElement, PlyData  # type: ignore
    attributes = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    attributes.extend([f'f_dc_{i}' for i in range(3)])
    attributes.extend([f'f_rest_{i}' for i in range(3*(spherical_harmonics.shape[-1]-1))])
    attributes.append('opacity')
    attributes.extend([f'scale_{i}' for i in range(scales.shape[-1])])
    attributes.extend([f'rot_{i}' for i in range(4)])

    if len(opacities.shape) == 1:
        opacities = opacities[:, None]

    with tempfile.TemporaryDirectory() as tmpdirname:
        dtype_full = [(attribute, 'f4') for attribute in attributes]
        elements = np.empty(means.shape[0], dtype=dtype_full)
        f_dc = spherical_harmonics[..., 0]
        f_rest = spherical_harmonics[..., 1:].reshape(f_dc.shape[0], -1)
        attributes = np.concatenate((
            means, np.zeros_like(means), 
            f_dc, f_rest, 
            _inverse_sigmoid(opacities), 
            np.log(scales), 
            quaternions), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        ply_file = os.path.join(tmpdirname, "splat.ply")
        out_file = os.path.join(tmpdirname, "scene.ksplat")
        ply_data = PlyData([el])
        ply_data.write(ply_file)

        # Convert to ksplat format
        subprocess.check_call(["bash", "-c", f"""
if [ ! -e /tmp/gaussian-splats-3d ]; then
rm -rf "/tmp/gaussian-splats-3d-tmp"
git clone https://github.com/mkkellogg/GaussianSplats3D.git "/tmp/gaussian-splats-3d-tmp"
cd /tmp/gaussian-splats-3d-tmp
npm install
npm run build
cd "$PWD"
mv /tmp/gaussian-splats-3d-tmp /tmp/gaussian-splats-3d
fi
node /tmp/gaussian-splats-3d/util/create-ksplat.js {shlex.quote(ply_file)} {shlex.quote(out_file)}
"""])
        shutil.move(out_file, path)
