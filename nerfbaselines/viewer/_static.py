import base64
import os
import json
import io
from typing import Dict, cast
import nerfbaselines.viewer
from pathlib import Path
import numpy as np
from PIL import Image
from nerfbaselines.utils import (
    image_to_srgb, convert_image_dtype, get_supported_palette_names, get_palette
)


def _yield_files(path, _path=None):
    if path.is_file():
        if _path == None:
            yield path.name, path
        else:
            yield _path, path
    elif path.is_dir():
        for p in path.iterdir():
            yield from _yield_files(p, p.name if _path == None else f'{_path}/{p.name}')


def get_palettes_js():
    out = "const palettes = {\n"
    for name in get_supported_palette_names():
        palette = get_palette(name)
        palette = convert_image_dtype(palette, np.uint8).reshape(-1)
        palette_vals = ",".join(map(str, palette.tolist()))
        out += f"{json.dumps(name)}:[{palette_vals}],\n"
    out += "};\nexport default palettes;\n"
    return out


def build_static_viewer(output, params=None):
    """
    Builds the viewer in the given path.

    Args:
        output: The path to write the viewer to.
    """
    params = params or {}
    path = Path(nerfbaselines.viewer.__file__).absolute().parent
    for p, file in _yield_files(path/"static"):
        dirname = (Path(output) / p).parent
        dirname.mkdir(parents=True, exist_ok=True)
        with open(Path(output) / p, "wb") as f:
            f.write(file.read_bytes())
    # Add template
    index = (path/"static"/"index.html").read_text()
    with open(Path(output) / "index.html", "w") as f2:
        index = index.replace("{{ data|safe }}", json.dumps(params))
        f2.write(index)
    # Add palettes.js
    with open(Path(output) / "palettes.js", "w") as f:
        f.write(get_palettes_js())


def get_image_thumbnail_url(image, dataset):
    max_img_size = 96
    W, H = image.shape[:2]
    downsample_factor = max(1, min(W//int(max_img_size), H//int(max_img_size)))
    image = image[::downsample_factor, ::downsample_factor]
    image = image_to_srgb(image, 
                          dtype=np.uint8, 
                          color_space="srgb", 
                          background_color=(dataset.get("metadata") or {}).get("background_color"))
    with io.BytesIO() as output:
        Image.fromarray(image).save(output, format="JPEG")
        output.seek(0)
        out = output.getvalue()
    return f"data:image/jpeg;base64,{base64.b64encode(out).decode()}"


def export_viewer_dataset(train_dataset=None, test_dataset=None) -> Dict:
    def _get_thumbnail_url(image, dataset):
        if image is None: return None
        return get_image_thumbnail_url(image, dataset)

    def _make_split(dataset):
        nb_cameras = dataset.get("cameras")
        root_path = dataset.get("image_paths_root")
        if root_path is None:
            root_path = os.path.commonpath(dataset["image_paths"])
        cameras = [{
            "pose": nb_cameras.poses[i][:3, :4].flatten().tolist(),
            "intrinsics": nb_cameras.intrinsics[i].tolist(),
            "image_size": nb_cameras.image_sizes[i].tolist(),
            "image_name": os.path.relpath(dataset["image_paths"][i], root_path),
            "thumbnail_url": _get_thumbnail_url((
                dataset["images"][i] if dataset.get("images") is not None 
                else dataset["image_paths"][i]
            ), dataset),
        } for i in range(len(nb_cameras.poses))]
        return { "cameras": cameras }

    metadata = ((train_dataset and train_dataset.get("metadata")) or {}).copy()
    def fix_val(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [fix_val(x) for x in v]
        if isinstance(v, dict):
            return {k: fix_val(v) for k, v in v.items()}
        return v
    if isinstance(metadata.get("viewer_initial_pose"), np.ndarray):
        metadata["viewer_initial_pose"] = metadata["viewer_initial_pose"][:3, :4].flatten().tolist()
    if isinstance(metadata.get("viewer_transform"), np.ndarray):
        metadata["viewer_transform"] = metadata["viewer_transform"][:3, :4].flatten().tolist()
    dataset = {
        "train": _make_split(train_dataset) if train_dataset else None,
        "test": _make_split(test_dataset) if test_dataset else None,
        "metadata": metadata,
    }
    dataset = cast(Dict, fix_val(dataset))
    return dataset
        

def build_viewer_dataset_params(output, train_dataset=None, test_dataset=None):
    """
    Builds the viewer dataset params in the given path.
    It will generate "params.json" file and all other required files in the given path.

    Args:
        output: The path to write the params to.
    """
    outpath = Path(output)
    params = {}
    def _get_thumbnail_url(image, dataset):
        if not image: return None
        return get_image_thumbnail_url(image, dataset)

    def _make_split(dataset):
        nb_cameras = dataset.get("cameras")
        root_path = dataset.get("image_paths_root")
        if root_path is None:
            root_path = os.path.commonpath(dataset["image_paths"])
        cameras = [{
            "pose": nb_cameras.poses[i][:3, :4].flatten().tolist(),
            "intrinsics": nb_cameras.intrinsics[i].tolist(),
            "image_size": nb_cameras.image_size[i].tolist(),
            "image_name": os.path.relpath(dataset["image_paths"][i], root_path),
            "thumbnail_url": _get_thumbnail_url((
                dataset["images"][i] if dataset.get("images") is not None 
                else dataset["image_paths"][i]
            ), dataset),
        } for i in range(len(nb_cameras.poses))]
        return { "cameras": cameras }
    dataset = {
        "train": _make_split(train_dataset) if train_dataset else None,
        "test": _make_split(test_dataset) if test_dataset else None,
    }
    with open(outpath / "dataset.json", "w") as f:
        json.dump(dataset, f)
    with open(outpath / "params.json", "w") as f:
        json.dump(params, f)
