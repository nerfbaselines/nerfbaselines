import json
import shutil
import logging
import shlex
import subprocess
import tempfile
import os
from typing import Optional, Union
import numpy as np
from nerfbaselines.utils import (
    get_transform_and_scale, 
    rotation_matrix_to_quaternion,
    convert_image_dtype,
)
from nerfbaselines.io import wget
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


def export_generic_demo(path: str, *,
                        options):
    os.makedirs(path, exist_ok=True)

    # Parse options
    options = (options or {}).copy()
    if "dataset_metadata" not in options:
        raise ValueError("dataset_metadata is required for export_demo")
    dataset_metadata = options.pop("dataset_metadata")
    viewer_transform = dataset_metadata.get("viewer_transform")
    assert viewer_transform is not None, "viewer_transform is required for export_demo and it is missing in dataset_metadata"
    viewer_initial_pose = dataset_metadata.get("viewer_initial_pose")
    assert viewer_initial_pose is not None, "viewer_initial_pose is required for export_demo and it is missing in dataset_metadata"
    mock_cors = _cast_value(Optional[bool], options.pop("mock_cors", False)) or False
    enable_shared_memory = _cast_value(Optional[bool], options.pop("enable_shared_memory", False)) or False
    antialiased = options.pop("antialiased", False)
    kernel2DSize = options.pop("kernel_2D_size", 0.3)
    splatRenderMode = "TwoD" if options.pop("is_2DGS", False) else "ThreeD"
    if options:
        logging.warning(f"Unused options: {', '.join(options.keys())}")
    del options

    # Generate html demo
    with open(__file__ + ".html", "r") as ftemplate:
        index = ftemplate.read()

    scale = 1
    offset = np.zeros(3, dtype=np.float32)
    rotation = np.zeros(4, dtype=np.float32)
    rotation[0] = 1

    offset = np.zeros(3, dtype=np.float32)
    rotation = np.array([1,0,0,0], dtype=np.float32)
    scale = 1
    cameraUp = np.array([0,0,1], dtype=np.float32)
    initialCameraPosition = np.array([1,0,0], dtype=np.float32)
    initialCameraLookAt = np.array([0,0,0], dtype=np.float32)
    if viewer_transform is not None:
        _transform, scale = get_transform_and_scale(viewer_transform)
        rotation = rotation_matrix_to_quaternion(_transform[:3, :3])
        offset = _transform[:3, 3]*scale
        initialCameraPosition = viewer_initial_pose[:3, 3]
        if dataset_metadata.get("type") != "object-centric":
            initialCameraLookAt = initialCameraPosition + viewer_initial_pose[:3, 2]
    background_color = None
    if dataset_metadata.get("background_color") is not None:
        # Convert (255, 255, 255) to #ffffff
        background_color = "#" + "".join([hex(x)[2:] for x in convert_image_dtype(dataset_metadata["background_color"], np.uint8)])
    with open(os.path.join(path, "params.json"), "w", encoding="utf8") as f:
        json.dump({
            "type": "gaussian-splatting",
            "backgroundColor": background_color,
            "initialCameraPosition": initialCameraPosition.tolist(),
            "initialCameraLookAt": initialCameraLookAt.tolist(),
            "cameraUp": cameraUp.tolist(),
            "scale": float(scale),
            "rotation": rotation[[1, 2, 3, 0]].tolist(),
            "antialiased": antialiased,
            "kernel2DSize": kernel2DSize,
            "splatRenderMode": splatRenderMode,
            "offset": offset.tolist(),
            "sceneUri": "scene.ksplat",
        }, f, indent=2)
    if enable_shared_memory:
        assert index.index("const enableSharedMemory = false;") >= 0, "Could not set shared memory"
        index = index.replace("const enableSharedMemory = false;", "const enableSharedMemory = true;")
    if mock_cors:
        index = index.replace("<head>", '<head><script type="text/javascript" src="coi-serviceworker.min.js"></script>')
        wget("https://raw.githubusercontent.com/gzuidhof/coi-serviceworker/7b1d2a092d0d2dd2b7270b6f12f13605de26f214/coi-serviceworker.min.js", 
            os.path.join(path, "coi-serviceworker.min.js"))
    with open(os.path.join(path, "index.html"), "w", encoding="utf8") as f:
        f.write(index)
    # NOTE: the viewer is my fork of the Kellogg's viewer with some modifications to 
    # support methods like MipSplatting, etc.
    wget(
        "https://gist.githubusercontent.com/jkulhanek/8792b41dc4a8af77f9883c7f1b846cb4/raw/03c6155d1e358c2562e08172495f79ba07b77876/gaussian-splats-3d.module.min.js",
        os.path.join(path, "gaussian-splats-3d.module.min.js"))
    wget(
        "https://gist.githubusercontent.com/jkulhanek/8792b41dc4a8af77f9883c7f1b846cb4/raw/da860114d02c3fdd6ed13c8121330474c93289e9/three.module.min.js",
        os.path.join(path, "three.module.min.js"))
    logging.info(f"Demo exported to {path}")


def generate_ksplat_file(path: str,
                         xyz: np.ndarray,
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
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        f_dc = spherical_harmonics[..., 0]
        f_rest = spherical_harmonics[..., 1:].reshape(f_dc.shape[0], -1)
        attributes = np.concatenate((
            xyz, np.zeros_like(xyz), 
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


def export_demo(path: str, *,
                xyz: np.ndarray,
                scales: np.ndarray,
                opacities: np.ndarray,
                quaternions: np.ndarray,
                spherical_harmonics: np.ndarray,
                options):
    os.makedirs(path, exist_ok=True)
    generate_ksplat_file(os.path.join(path, "scene.ksplat"), xyz, scales, opacities, quaternions, spherical_harmonics)
    export_generic_demo(path, options=options)
