import numpy as np
import os
import logging
import click
from pathlib import Path
import json
from nerfbaselines import backends
from nerfbaselines import (
    get_method_spec, build_method_class,
)
from nerfbaselines.io import open_any_directory, deserialize_nb_info
from nerfbaselines.datasets import load_dataset, dataset_load_features
from nerfbaselines.utils import apply_transform, invert_transform, convert_image_dtype
from nerfbaselines import viewer
from ._common import click_backend_option
from ._common import SetParamOptionType, NerfBaselinesCliCommand


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
    from nerfbaselines._export_3dgs import generate_ksplat_file
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


def try_export_mesh(output, method, train_embedding, train_dataset):
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

    train_dataset = dataset_load_features(train_dataset)
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
    out = {
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


def export_demo(output, method, 
                train_embedding, 
                train_dataset,
                test_dataset):
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
    dataset = viewer.export_viewer_dataset(train_dataset, test_dataset)

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


@click.command("export-demo", cls=NerfBaselinesCliCommand, help=(
    "Export a demo from a trained model. "
    "The interactive demo will be a website (index.html) that can be opened in the browser. "
    "Only some methods support this feature."))
@click.option("--checkpoint", default=None, required=False, type=str, help=(
    "Path to the checkpoint directory. It can also be a remote path (starting with `http(s)://`) or be a path inside a zip file."
))
@click.option("--output", "-o", type=str, required=True, help="Output directory for the demo.")
@click.option("--data", type=str, default=None, required=True, help=(
    "A path to the dataset to load which matches the dataset used to train the model. "
    "The dataset can be either an external dataset (e.g., a path starting with `external://{dataset}/{scene}`) or a local path to a dataset. If the dataset is an external dataset, the dataset will be downloaded and cached locally. "
    "If the dataset is a local path, the dataset will be loaded directly from the specified path. "))
@click.option("--train-embedding", type=str, default=None, help="Select the train embedding index to use for the demo (if the method supports appearance modelling. A comma-separated list of indices can be provided to select multiple embeddings.")
@click.option("--set", "options", type=SetParamOptionType(), multiple=True, default=None, help=(
    "Set a parameter for demo export. " 
    "The argument should be in the form of `--set key=value` and can be used multiple times to set multiple parameters. "
    "The parameters are specific to each method."))
@click_backend_option()
def main(*, checkpoint: str, output: str, backend_name, data=None, train_embedding=None, options):
    checkpoint = str(checkpoint)
    output = str(output)
    options = dict(options or [])

    # Read method nb-info
    with open_any_directory(checkpoint, mode="r") as _checkpoint_path:
        checkpoint_path = Path(_checkpoint_path)
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
        backends.mount(checkpoint_path, checkpoint_path)
        method_spec = get_method_spec(method_name)
        with build_method_class(method_spec, backend=backend_name) as method_cls:
            method = method_cls(checkpoint=str(checkpoint_path))
            info = method.get_info()
            train_dataset = load_dataset(data, split="train", load_features=True, features=info["required_features"])
            test_dataset = load_dataset(data, split="train", load_features=True, features=info["required_features"])

            export_demo(output, 
                        method, 
                        [int(x) if x.lower() != "none" else None for x in train_embedding.split(",")] if train_embedding else None,
                        train_dataset=train_dataset,
                        test_dataset=test_dataset)


if __name__ == "__main__":
    main()
