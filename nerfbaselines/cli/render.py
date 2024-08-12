import sys
from typing import Union, Tuple
import logging
from pathlib import Path
import os
import json

import click
from nerfbaselines.datasets import load_dataset
from nerfbaselines.types import Method
from nerfbaselines import backends
from nerfbaselines import registry
from nerfbaselines.evaluation import render_all_images, render_frames, trajectory_get_embeddings, trajectory_get_cameras, path_is_video
from nerfbaselines.utils import setup_logging, handle_cli_error
from nerfbaselines.io import open_any_directory, deserialize_nb_info, load_trajectory, open_any


@click.command("render")
@click.option("--checkpoint", type=str, default=None, required=True)
@click.option("--data", type=str, default=None, required=True)
@click.option("--output", type=str, default="predictions", help="output directory or tar.gz file")
@click.option("--split", type=str, default="test")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@handle_cli_error
def render_command(checkpoint: str, data: str, output: str, split: str, verbose: bool, backend_name):
    checkpoint = str(checkpoint)
    setup_logging(verbose)

    if os.path.exists(output):
        logging.critical("Output path already exists")
        sys.exit(1)

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
        with registry.build_method(method_name, backend=backend_name) as method_cls:
            method: Method = method_cls(checkpoint=str(checkpoint_path))
            method_info = method.get_info()
            dataset = load_dataset(data, 
                                   split=split, 
                                   features=method_info.get("required_features", None), 
                                   supported_camera_models=method_info.get("supported_camera_models", None))
            dataset_colorspace = dataset["metadata"].get("color_space", "srgb")

            if dataset_colorspace != nb_info.get("color_space", "srgb"):
                raise RuntimeError(f"Dataset color space {dataset_colorspace} != method color space {nb_info.get('color_space', 'srgb')}")
            for _ in render_all_images(method, dataset, output=output, nb_info=nb_info):
                pass


@click.command("render-trajectory")
@click.option("--checkpoint", type=str, required=True)
@click.option("--trajectory", type=str, required=True)
@click.option("--output", type=click.Path(path_type=str), default=None, help="Output a mp4/directory/tar.gz file. Use '{output}' as a placeholder for output name.")
@click.option("--resolution", type=str, default=None, help="Override the resolution of the output")
@click.option("--output-names", type=str, default="color", help="Comma separated list of output types (e.g. color,depth,accumulation)")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(backends.ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@handle_cli_error
def render_trajectory_command(checkpoint: Union[str, Path], 
                              trajectory: str, 
                              output: Union[str, Path], 
                              output_names: Union[Tuple[str, ...], str], 
                              verbose, 
                              backend_name, 
                              resolution=None):
    checkpoint = str(checkpoint)
    output = str(output)
    setup_logging(verbose)
    if isinstance(output_names, str):
        output_names = tuple(output_names.split(","))

    for output_name in output_names:
        loutput = output.format(output=output_name)
        if os.path.exists(loutput):
            logging.critical("Output path {loutput} already exists")
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

    # Read method nb-info
    logging.info(f"Loading checkpoint {checkpoint}")
    with open_any_directory(checkpoint, mode="r") as _checkpoint_path:
        checkpoint_path = Path(_checkpoint_path)
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            nb_info = json.load(f)
        nb_info = deserialize_nb_info(nb_info)

        method_name = nb_info["method"]
        backends.mount(checkpoint_path, checkpoint_path)
        with registry.build_method(method_name, backend=backend_name) as method_cls:
            method = method_cls(checkpoint=str(checkpoint_path))

            # Embed the appearance
            embeddings = trajectory_get_embeddings(method, _trajectory)

            render_frames(method, cameras, embeddings=embeddings, output=output, output_names=output_names, nb_info=nb_info, fps=_trajectory["fps"])
            logging.info(f"Output saved to {output}")

