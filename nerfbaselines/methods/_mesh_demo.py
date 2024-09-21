import json
import logging
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
    background_color = None
    if dataset_metadata.get("background_color") is not None:
        # Convert (255, 255, 255) to #ffffff
        background_color = "#" + "".join([hex(x)[2:] for x in convert_image_dtype(dataset_metadata["background_color"], np.uint8)])
    with open(os.path.join(path, "params.json"), "w", encoding="utf8") as f:
        json.dump({
            "type": "mesh",
            "backgroundColor": background_color,
            "initialCameraPosition": initialCameraPosition.tolist(),
            "initialCameraLookAt": initialCameraLookAt.tolist(),
            "cameraUp": cameraUp.tolist(),
            "scale": float(scale),
            "rotation": rotation[[1, 2, 3, 0]].tolist(),
            "offset": offset.tolist(),
            "meshUri": "mesh.ply",
        }, f, indent=2)
    if mock_cors:
        index = index.replace("<head>", '<head><script type="text/javascript" src="coi-serviceworker.min.js"></script>')
        wget("https://raw.githubusercontent.com/gzuidhof/coi-serviceworker/7b1d2a092d0d2dd2b7270b6f12f13605de26f214/coi-serviceworker.min.js", 
            os.path.join(path, "coi-serviceworker.min.js"))
    with open(os.path.join(path, "index.html"), "w", encoding="utf8") as f:
        f.write(index)
    logging.info(f"Demo exported to {path}")
