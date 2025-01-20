import os
import logging
from pathlib import Path
import json
import shutil
import shlex
import subprocess
import tempfile
import os
from typing import Union, Dict
import numpy as np
from nerfbaselines.utils import apply_transform, invert_transform, convert_image_dtype
from nerfbaselines import viewer
from nerfbaselines import Dataset, Method
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


def _get_float_background_color(background_color):
    if background_color is None:
        return None
    if hasattr(background_color, "dtype"):
        background_color = convert_image_dtype(background_color, "float32")
        return background_color.tolist()
    if isinstance(background_color, (tuple, list)):
        if all(isinstance(x, int) for x in background_color):
            return [x / 255 for x in background_color]
        return list(background_color)
    raise ValueError(f"Invalid background color {background_color}")


def try_export_gaussian_splats(output, method, train_embedding, dataset_metadata):
    # Try export gaussian splats
    output_name = "scene.ksplat"
    if train_embedding is not None:
        output_name = f"scene-{train_embedding}.ksplat"
    export_splats = None
    try:
        export_splats = method.export_gaussian_splats  # type: ignore
    except AttributeError:
        return None

    camera_pose = None
    if "viewer_transform" in dataset_metadata and "viewer_initial_pose" in dataset_metadata:
        viewer_initial_pose_ws = apply_transform(
            invert_transform(dataset_metadata["viewer_transform"], has_scale=True), 
            dataset_metadata["viewer_initial_pose"])
        camera_pose = viewer_initial_pose_ws[:3, :4]

    # If train embedding is enabled, select train_embedding
    embedding = None
    if train_embedding is not None:
        embedding = method.get_train_embedding(train_embedding)
        if train_embedding is None:
            logging.error(f"Train embedding {train_embedding} not found or not supported by the method.")
    splats = export_splats(
        options=dict(
            embedding=embedding,
            camera_pose=camera_pose))

    # Export ksplat and params
    generate_ksplat_file(
        os.path.join(output, output_name),
        means=splats["means"],
        scales=splats["scales"],
        opacities=splats["opacities"],
        quaternions=splats["quaternions"],
        spherical_harmonics=splats["spherical_harmonics"])
    params = {}
    if splats.get("transform") is not None:
        params["transform"] = splats["transform"][:3, :4].flatten().tolist()
    if splats.get("is_2DGS"):
        params["is_2DGS"] = True
    if splats.get("antialias_2D_kernel_size") is not None:
        params["antialias_2D_kernel_size"] = splats["antialias_2D_kernel_size"]
    with open(os.path.join(output, f"{output_name}.json"), "w", encoding="utf8") as f:
        json.dump(params, f, indent=2)
    params["scene_url"] = f"./{output_name}"
    params["type"] = "3dgs"
    background_color = _get_float_background_color(
        dataset_metadata.get("background_color", None))
    if background_color is not None:
        params["background_color"] = background_color
    return params


def try_export_mesh(output, method, train_embedding, train_dataset: Dataset):
    output_name = "mesh.ply"
    if train_embedding is not None:
        output_name = f"mesh-{train_embedding}.ply"
    try:
        export_mesh = method.export_mesh  # type: ignore
    except AttributeError:
        return None

    if train_dataset is None:
        raise ValueError("Training dataset is required for export_mesh. Please pass --data option.")

    # If train embedding is enabled, select train_embedding
    embedding = None
    if train_embedding is not None:
        embedding = method.get_train_embedding(train_embedding)
        if train_embedding is None:
            logging.error(f"Train embedding {train_embedding} not found or not supported by the method.")
    dataset_metadata = train_dataset["metadata"]
    export_mesh(
        output,
        train_dataset=train_dataset,
        options=dict(
            embedding=embedding,
            dataset_metadata=dataset_metadata))
    background_color = _get_float_background_color(
        dataset_metadata.get("background_color", None))
    # Rename mesh.ply to output_name
    os.rename(os.path.join(output, "mesh.ply"), os.path.join(output, output_name))

    # Exported as mesh.ply
    out: Dict = {
        "type": "mesh",
        "mesh_url": f"./{output_name}",
    }
    if background_color is not None:
        out["background_color"] = background_color
    return out


def _export_demo(output, method, train_embedding, train_dataset):
    dataset_metadata = train_dataset["metadata"]
    rparams = try_export_gaussian_splats(output, method, train_embedding, dataset_metadata)

    # Try export mesh
    if rparams is None:
        rparams = try_export_mesh(output, method, train_embedding, train_dataset)
    if rparams is None:
        raise NotImplementedError(f"Method {method.__class__.__name__} does not support demo (supported demos: mesh,3dgs).")
    return rparams


def export_demo(output: Union[str, Path],
                method: Method, 
                train_embedding, 
                train_dataset: Dataset,
                test_dataset: Dataset):
    # Try export gaussian splats
    os.makedirs(output, exist_ok=True)

    if isinstance(train_embedding, int):
        train_embedding = [train_embedding]
    if isinstance(train_embedding, (list, tuple)):
        rparams = None
        for i in train_embedding:
            single_params = _export_demo(output, method, i, train_dataset)
            if rparams is None:
                rparams = single_params
            if i is not None:
                if "scene_url" in single_params:
                    rparams.setdefault("scene_url_per_appearance", {})[str(i)] = single_params["scene_url"]
                elif "mesh_url" in single_params:
                    rparams.setdefault("mesh_url_per_appearance", {})[str(i)] = single_params["mesh_url"]
    else:
        rparams = _export_demo(output, method, None, train_dataset)
    with open(os.path.join(output, "params.json"), "w") as f:
        f.write(json.dumps(rparams, indent=2))

    # Export dataset
    dataset: Dict = viewer.export_viewer_dataset(train_dataset, test_dataset)

    # Generate ply file
    if train_dataset.get("points3D_xyz") is not None:
        with open(os.path.join(output, "dataset-pointcloud.ply"), "wb") as f:
            viewer.write_dataset_pointcloud(f, train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])

        dataset["pointcloud_url"] = "./dataset-pointcloud.ply"
    with open(os.path.join(output, "dataset.json"), "w") as f:
        f.write(json.dumps(dataset, indent=2))

    # Export the viewer
    dataset_metadata = train_dataset["metadata"]
    params = viewer.merge_viewer_params({
        "renderer": rparams,
        "dataset": { "url": "./dataset.json" }
    }, viewer.get_viewer_params_from_dataset_metadata(dataset_metadata))
    viewer.build_static_viewer(output, params)


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
