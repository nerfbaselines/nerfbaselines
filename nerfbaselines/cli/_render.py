import sys
from contextlib import ExitStack
from typing import Union, Tuple
import logging
from pathlib import Path
import os
import json

import click
from nerfbaselines.datasets import load_dataset
from nerfbaselines import Method, get_method_spec, build_method_class
from nerfbaselines import backends
from nerfbaselines.evaluation import render_all_images, render_frames, trajectory_get_embeddings, trajectory_get_cameras
from nerfbaselines.io import open_any_directory, deserialize_nb_info, load_trajectory, open_any
from ._common import click_backend_option, NerfBaselinesCliCommand


@click.command("render", cls=NerfBaselinesCliCommand, short_help="Render images from a trained model", help=(
    "Render images from a trained model for all cameras in the dataset. "
    "The images will be saved in the specified output directory. "
    "The correct evaluation protocol will be used to render the images. "
    "E.g., for Photo Tourism, appearance embeddings will be optimized before the images are rendered."
))
@click.option("--checkpoint", default=None, required=True, type=str, help=(
    "Path to the checkpoint directory. It can also be a remote path (starting with `http(s)://`) or be a path inside a zip file."
))
@click.option("--data", type=str, required=True, help=(
    "A path to the dataset to render the cameras from. The dataset can be either an external dataset (e.g., a path starting with `external://{dataset}/{scene}`) or a local path to a dataset. If the dataset is an external dataset, the dataset will be downloaded and cached locally. If the dataset is a local path, the dataset will be loaded directly from the specified path."))
@click.option("--output", type=str, default="predictions", help="Output directory or tar.gz/zip file.")
@click.option("--split", type=str, default="test", show_default=True, help="Dataset split to render.")
@click_backend_option()
def render_command(checkpoint: str, data: str, output: str, split: str, backend_name):
    checkpoint = str(checkpoint)

    if os.path.exists(output):
        logging.critical("Output path already exists")
        sys.exit(1)

    with ExitStack() as stack:
        # Open checkpoint directory
        _checkpoint_path = stack.enter_context(open_any_directory(checkpoint, mode="r"))
        stack.enter_context(backends.mount(_checkpoint_path, _checkpoint_path))
        checkpoint_path = Path(_checkpoint_path)

        # Read method nb-info
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        # Prepare method backend
        method_name = nb_info["method"]
        method_spec = get_method_spec(method_name)
        method_cls = stack.enter_context(build_method_class(method_spec, backend=backend_name))

        # Prepare method 
        method: Method = method_cls(checkpoint=str(checkpoint_path))

        # Load the dataset
        method_info = method.get_info()
        dataset = load_dataset(data, 
                               split=split, 
                               features=method_info.get("required_features", None), 
                               supported_camera_models=method_info.get("supported_camera_models", None))
        dataset_colorspace = dataset["metadata"].get("color_space", "srgb")
        if dataset_colorspace != nb_info.get("color_space", "srgb"):
            raise RuntimeError(f"Dataset color space {dataset_colorspace} != method color space {nb_info.get('color_space', 'srgb')}")

        # Render all images
        for _ in render_all_images(method, dataset, output=output, nb_info=nb_info):
            pass


@click.command("render-trajectory", cls=NerfBaselinesCliCommand, short_help="Render images from a trained model for a trajectory", help=(
    "Render images from a trained model for a trajectory. "
    "The trajectory should be a JSON file obtained from the viewer (`nerfbaselines viewer`). "
    "The output can be a video or a directory with images. "
    "Furthermore, one or multiple outputs can be rendered (e.g., color, depth, accumulation)."
    "If multiple outputs are specified, the `--output` argument should contain a placeholder `{output}` which will be replaced with the output name and multiple files will be generated (unless the output is a directory or an archive, where all outputs can be placed to a single archive)."))
@click.option("--checkpoint", default=None, required=True, type=str, help=(
    "Path to the checkpoint directory. It can also be a remote path (starting with `http(s)://`) or be a path inside a zip file."
))
@click.option("--trajectory", type=str, required=True, help="Path to the trajectory JSON file.")
@click.option("--output", type=click.Path(path_type=str), default=None, help="Output a mp4/directory/tar.gz file. Use '{output}' as a placeholder for output name.")
@click.option("--resolution", type=str, default=None, help="Override the resolution of the output. Use 'widthxheight' format (e.g., 1920x1080). If one of the dimensions is negative, the aspect ratio will be preserved and the dimension will be rounded to the nearest multiple of the absolute value of the dimension.")
@click.option("--output-names", type=str, default="color", help="Comma separated list of output types (e.g. color,depth,accumulation). See the method's `get_info()['supported_outputs']` for supported outputs.")
@click_backend_option()
def render_trajectory_command(checkpoint: Union[str, Path], 
                              trajectory: str, 
                              output: Union[str, Path], 
                              output_names: Union[Tuple[str, ...], str], 
                              backend_name, 
                              resolution=None):
    checkpoint = str(checkpoint)
    output = str(output)
    if isinstance(output_names, str):
        output_names = tuple(output_names.split(","))

    for output_name in output_names:
        loutput = output.format(output=output_name)
        if os.path.exists(loutput):
            logging.critical(f"Output path {loutput} already exists")
            sys.exit(1)

    # Parse trajectory
    with open_any(trajectory, "r") as f:
        _trajectory = load_trajectory(f)
    cameras = trajectory_get_cameras(_trajectory)

    # Override resolution
    if resolution is not None:
        w, h = tuple(map(int, resolution.split("x")))
        aspect = _trajectory["image_size"][0] / _trajectory["image_size"][1]
        if w < 0:
            assert h > 0, "Either width or height must be positive"
            w = ((int(h * aspect) + abs(w) - 1) // abs(w)) * abs(w)
        elif h < 0:
            assert w > 0, "Either width or height must be positive"
            h = ((int(w / aspect) + abs(h) - 1) // abs(h)) * abs(h)
        logging.info(f"Resizing to {w}x{h}")

        # Rescale cameras
        oldw = cameras.image_sizes[..., 0]
        oldh = cameras.image_sizes[..., 1]
        cameras.intrinsics[..., 0] *= w / oldw
        cameras.intrinsics[..., 1] *= h / oldh
        cameras.intrinsics[..., 2] *= w / oldw
        cameras.intrinsics[..., 3] *= h / oldh
        cameras.image_sizes[..., 0] = w
        cameras.image_sizes[..., 1] = h

    logging.info(f"Loading checkpoint {checkpoint}")
    with ExitStack() as stack:
        # Load the checkpoint
        _checkpoint_path = stack.enter_context(open_any_directory(checkpoint, mode="r"))
        stack.enter_context(backends.mount(_checkpoint_path, _checkpoint_path))
        checkpoint_path = Path(_checkpoint_path)
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        # Read method nb-info
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        # Load the method
        method_name = nb_info["method"]
        method_spec = get_method_spec(method_name)
        method_cls = stack.enter_context(build_method_class(method_spec, backend=backend_name))
        method = method_cls(checkpoint=str(checkpoint_path))

        # Embed the appearance
        embeddings = trajectory_get_embeddings(method, _trajectory)

        # Render the frames
        render_frames(method, cameras, 
                      embeddings=embeddings, 
                      output=output, 
                      output_names=output_names, 
                      nb_info=nb_info, 
                      fps=_trajectory["fps"])
        logging.info(f"Output saved to {output}")

