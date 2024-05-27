import sys
from dataclasses import dataclass
from typing import Union, Dict, IO, Optional, Any, TypedDict, List, cast
import time
import io
import os
import logging
import tarfile
from pathlib import Path
import json
import click
from tqdm import tqdm
import numpy as np

try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args

from .utils import setup_logging, image_to_srgb, save_image, visualize_depth, handle_cli_error
from .render import with_supported_camera_models
from .utils import convert_image_dtype
from .types import Method, Literal, Cameras, CameraModel, camera_model_to_int, new_cameras, Trajectory
from .backends import ALL_BACKENDS

from .io import open_any_directory, deserialize_nb_info, load_trajectory
from . import registry
from . import backends


OutputType = Literal["color", "depth"]


def render_frames(
    method: Method,
    cameras: Cameras,
    output: Union[str, Path],
    fps: float,
    embeddings: Optional[List[np.ndarray]] = None,
    description: str = "rendering frames",
    output_type: OutputType = "color",
    nb_info: Optional[dict] = None,
) -> None:
    output = Path(output)
    assert cameras.image_sizes is not None, "cameras.image_sizes must be set"
    info = method.get_info()
    render = with_supported_camera_models(info.get("supported_camera_models", frozenset(("pinhole",))))(method.render)
    color_space = "srgb"
    background_color = nb_info.get("background_color") if nb_info is not None else None
    expected_scene_scale = nb_info.get("expected_scene_scale") if nb_info is not None else None
    allow_transparency = True

    def _predict_all():
        predictions = render(cameras, embeddings=embeddings)
        for i, pred in enumerate(tqdm(predictions, desc=description, total=len(cameras), dynamic_ncols=True)):
            pred_image = image_to_srgb(pred["color"], np.uint8, color_space=color_space, allow_alpha=allow_transparency, background_color=background_color)
            if output_type == "color":
                yield pred_image
            elif output_type == "depth":
                assert "depth" in pred, "Method does not output depth"
                depth_rgb = visualize_depth(pred["depth"], near_far=cameras.nears_fars[i] if cameras.nears_fars is not None else None, expected_scale=expected_scene_scale)
                yield convert_image_dtype(depth_rgb, np.uint8)
            else:
                raise RuntimeError(f"Output type {output_type} is not supported.")

    if str(output).endswith(".tar.gz"):
        with tarfile.open(output, "w:gz") as tar:
            for i, frame in enumerate(_predict_all()):
                rel_path = f"{i:05d}.png"
                tarinfo = tarfile.TarInfo(name=rel_path)
                tarinfo.mtime = int(time.time())
                with io.BytesIO() as f:
                    f.name = rel_path
                    tarinfo.size = f.tell()
                    f.seek(0)
                    save_image(f, frame)
                    tar.addfile(tarinfo=tarinfo, fileobj=f)
    elif str(output).endswith(".mp4"):
        # Handle video
        import mediapy

        w, h = cameras.image_sizes[0]
        with mediapy.VideoWriter(output, (h, w), metadata=mediapy.VideoMetadata(len(cameras), (h, w), fps, bps=None)) as writer:
            for i, frame in enumerate(_predict_all()):
                writer.add_image(frame)
    else:
        os.makedirs(output, exist_ok=True)
        for i, frame in enumerate(_predict_all()):
            rel_path = f"{i:05d}.png"
            with open(os.path.join(output, rel_path), "wb") as f:
                save_image(f, frame)


def trajectory_get_cameras(trajectory: Trajectory) -> Cameras:
    if trajectory["camera_model"] != "pinhole":
        raise NotImplementedError("Only pinhole camera model is supported")
    poses = np.stack([x["pose"] for x in trajectory["frames"]])
    intrinsics = np.stack([x["intrinsics"] for x in trajectory["frames"]])
    camera_types = np.array([camera_model_to_int(trajectory["camera_model"])]*len(poses), dtype=np.int32)
    image_sizes = np.array([list(trajectory["image_size"])]*len(poses), dtype=np.int32)
    return new_cameras(poses=poses, 
                       intrinsics=intrinsics, 
                       camera_types=camera_types, 
                       image_sizes=image_sizes,
                       distortion_parameters=np.zeros((len(poses), 0), dtype=np.float32),
                       nears_fars=None, 
                       metadata=None)


def trajectory_get_embeddings(method: Method, trajectory: Trajectory) -> Optional[List[np.ndarray]]:
    appearances = list(trajectory.get("appearances") or [])
    appearance_embeddings = [None] * len(appearances)

    # Fill in embedding images
    for i, appearance in enumerate(appearances):
        if appearance.get("embedding") is not None:
            appearance_embeddings[i] = appearance.get("embedding")
        elif appearance.get("embedding_train_index") is not None:
            appearance_embeddings[i] = method.get_train_embedding(appearance.get("embedding_train_index"))
    if all(x is None for x in appearance_embeddings):
        return [None] * len(appearances)
    if not all(x is not None for x in appearance_embeddings):
        raise ValueError("Either all embeddings must be provided or all must be missing")
    if all(x.get("appearance_weights") is None for x in trajectory["frames"]):
        return [None] * len(appearances)
    if not all(x.get("appearance_weights") is not None for x in trajectory["frames"]):
        raise ValueError("Either all appearance weights must be provided or all must be missing")
    appearance_embeddings = np.stack(appearance_embeddings)

    # Interpolate embeddings
    out = []
    for frame in trajectory["frames"]:
        embedding = frame.get("appearance_weights") @ appearance_embeddings
        out.append(embedding)
    return out


@click.command("render-trajectory")
@click.option("--checkpoint", type=str, required=True)
@click.option("--trajectory", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=str), default=None, help="output a mp4/directory/tar.gz file")
@click.option("--output-type", type=click.Choice(get_args(OutputType)), default="color", help="output type")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", "backend_name", type=click.Choice(ALL_BACKENDS), default=os.environ.get("NERFBASELINES_BACKEND", None))
@handle_cli_error
def main(checkpoint: Union[str, Path], trajectory: Path, output: Union[str, Path], output_type: OutputType, verbose, backend_name):
    checkpoint = str(checkpoint)
    setup_logging(verbose)

    if os.path.exists(output):
        logging.critical("Output path already exists")
        sys.exit(1)

    # Parse trajectory
    with trajectory.open("r") as f:
        _trajectory = load_trajectory(f)
    cameras = trajectory_get_cameras(_trajectory)

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

            render_frames(method, cameras, embeddings=embeddings, output=output, output_type=output_type, nb_info=nb_info, fps=_trajectory["fps"])
            logging.info(f"Output saved to {output}")
