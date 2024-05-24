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
from .types import Method, Literal, Cameras, CameraModel, camera_model_to_int, new_cameras
from .backends import ALL_BACKENDS

from .io import open_any_directory, deserialize_nb_info
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


def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    See:
        https://github.com/nerfstudio-project/nerfstudio/blob/1aba4ea7a29b05e86f5d223245a573e7dcd86caa/nerfstudio/viewer_legacy/server/utils.py#L52
    """
    if fov is None:
        print("Warning: fov is None, using default value")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length


def read_nerfstudio_trajectory(data: Dict[str, Any]) -> "Trajectory":
    if "seconds" in data:
        seconds = data["seconds"]
        fps = len(data["camera_path"]) / seconds
    elif "fps" in data:
        fps = data["fps"]
    else:
        raise RuntimeError("Either fps or seconds must be included in the trajectory file")

    w, h = data["image_size"]
    camera_type = data["camera_type"]
    if camera_type not in get_args(CameraModel):
        raise RuntimeError(f"Unsupported camera type {camera_type}.")

    poses = []
    appearances = []
    intrinsics = []
    for camera in data["camera_path"]:
        # pose
        poses.append(np.array(camera["pose"], dtype=np.float32).reshape(-1, 4)[:3])
        intrinsics.append(np.array(camera["intrinsics"], dtype=np.float32))
        appearances.append(camera.get("appearance") or {})

    poses=np.stack(poses, 0)
    intrinsics = np.stack(intrinsics, 0)
    return Trajectory(
        cameras=new_cameras(
            poses=poses,
            intrinsics=np.stack(intrinsics, 0),
            image_sizes=np.array((w, h), dtype=np.int32)[None].repeat(len(poses), 0),
            camera_types=np.array([camera_model_to_int(camera_type)] * len(poses), dtype=np.int32),
            distortion_parameters=np.zeros((len(poses), 0), dtype=np.float32),
            nears_fars=None,
        ),
        appearances=appearances,
        fps=fps,
    )


class TrajectoryFrameAppearance(TypedDict, total=False):
    embedding: Optional[np.ndarray]
    embedding_train_index: Optional[int]


@dataclass
class Trajectory:
    cameras: Cameras
    appearances: List[TrajectoryFrameAppearance]
    fps: float

    @classmethod
    def from_json(cls, data: Union[Dict[str, Any], IO, str, Path]) -> "Trajectory":
        """
        Trajectory format is similar to nerfstudio with the following differences:
          - "render_width" and "render_height" are replaced with "image_size"
          - "camera_type" are different "perspective" -> "pinhole" ("opencv"), ...
          - "camera_to_world" is replaced with "pose"
          - "intrinsics" is added
          - removed "fov" and "aspect_ratio"
          - "appearance" is added
          - coordinate system changed from OpenGL to OpenCV
        """
        if not isinstance(data, dict):
            # Load the data from IO
            if isinstance(data, (str, Path)):
                with open(data, "r", encoding="utf8") as f:
                    return cls.from_json(f)
            else:
                return cls.from_json(json.load(data))
        return read_nerfstudio_trajectory(data)


def trajectory_get_embeddings(method: Method, trajectory: Trajectory) -> Optional[List[np.ndarray]]:
    appearances = list(trajectory.appearances)

    # Fill in embedding images
    for i, appearance in enumerate(appearances):
        if appearance.get("embedding") is not None:
            continue
        embedding_train_index = appearance.get("embedding_train_index")
        if embedding_train_index is not None:
            appearances[i] = TrajectoryFrameAppearance(
                embedding=method.get_train_embedding(embedding_train_index), 
                **{k: cast(Any, v) for k, v in appearance.items() if k not in {"embedding", "embedding_train_index"}})

    # Interpolate embeddings
    steps = []
    embeddings = []
    for i, appearance in enumerate(appearances):
        if appearance.get("embedding") is not None:
            embeddings.append(appearance.get("embedding"))
            steps.append(i)
    if not steps:
        return None
    steps.append(steps[0] + len(appearances))
    embeddings.append(embeddings[0])

    all_embeddings = [None] * len(appearances)
    for i, (step, embedding) in enumerate(zip(steps[:-1], embeddings[:-1])):
        next_step = steps[i + 1]
        next_embedding = embeddings[i + 1]
        for j in range(step, next_step):
            if step == next_step:
                alpha = 0.5
            else:
                alpha = (j - step) / (next_step - step)
            all_embeddings[j%len(appearances)] = alpha * next_embedding + (1 - alpha) * embedding
    return cast(List[np.ndarray], all_embeddings)


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
    _trajectory = Trajectory.from_json(trajectory)

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

            render_frames(method, _trajectory.cameras, embeddings=embeddings, output=output, output_type=output_type, nb_info=nb_info, fps=_trajectory.fps)
            logging.info(f"Output saved to {output}")
