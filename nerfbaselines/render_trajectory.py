import sys
from dataclasses import dataclass
import typing
from typing import Literal, Union, Dict, IO, Optional, Any, Iterable
import time
import io
import os
import logging
import tarfile
from pathlib import Path
import json
import mediapy
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
from .types import Method, CurrentProgress, RenderOutput
from .cameras import Cameras, CameraModel

from .io import open_any_directory
from . import registry
from . import __version__


OutputType = Literal["color", "depth"]


def render_frames(
    method: Method,
    cameras: Cameras,
    output: Path,
    fps: float,
    description: str = "rendering frames",
    output_type: OutputType = "color",
    ns_info: Optional[dict] = None,
) -> Iterable[RenderOutput]:
    info = method.get_info()
    render = with_supported_camera_models(info.supported_camera_models)(method.render)
    color_space = "srgb"
    background_color = ns_info.get("background_color") if ns_info is not None else None
    expected_scene_scale = ns_info.get("expected_scene_scale") if ns_info is not None else None
    allow_transparency = True
    
    def _predict_all():
        with tqdm(desc=description) as pbar:
            def update_progress(progress: CurrentProgress):
                if pbar.total != progress.total:
                    pbar.reset(total=progress.total)
                pbar.set_postfix({"image": f"{min(progress.image_i+1, progress.image_total)}/{progress.image_total}"})
                pbar.update(progress.i - pbar.n)

            predictions = render(cameras, progress_callback=update_progress)
            for i, pred in enumerate(predictions):
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

        w, h = cameras.image_sizes[0]
        with mediapy.VideoWriter(output,
                                 (h, w),
                                 metadata=mediapy.VideoMetadata(len(cameras), (h, w), fps, bps=None)
                                ) as writer:
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


def read_nerfstudio_trajectory(data: Dict[str, Any]) -> 'Trajectory':
    if "seconds" in data:
        seconds = data["seconds"]
        fps = len(data["camera_path"]) / seconds
    elif "fps" in data:
        fps = data["fps"]
    else:
        raise RuntimeError("Either fps or seconds must be included in the trajectory file")

    h = data["render_height"]
    w = data["render_width"]

    if "camera_type" not in data:
        camera_type = CameraModel.PINHOLE
    else:
        camera_type_name = data["camera_type"].upper().replace("-", "_")
        if camera_type_name == "PERSPECTIVE":
            camera_type = CameraModel.PINHOLE
        elif camera_type_name == "FISHEYE":
            camera_type = CameraModel.OPENCV_FISHEYE
        elif camera_type_name in CameraModel:
            camera_type = CameraModel[camera_type_name]
        else:
            raise RuntimeError(f"Unsupported camera type {data['camera_type']}.")

    c2ws = []
    fxs = []
    fys = []
    cxs = []
    cys = []
    for camera in data["camera_path"]:
        # pose
        c2w = np.array(camera["camera_to_world"], dtype=np.float32).reshape(4, 4)[:3]
        c2ws.append(c2w)

        # field of view
        fov = camera["fov"]
        focal_length = three_js_perspective_camera_focal_length(fov, h)
        fxs.append(focal_length)
        fys.append(focal_length)
        cxs.append(w / 2)
        cys.append(h/ 2)

    camera_to_worlds = np.stack(c2ws, 0)
    fx = np.array(fxs, dtype=np.float32)
    fy = np.array(fys, dtype=np.float32)
    cx = np.array(cxs, dtype=np.float32)
    cy = np.array(cys, dtype=np.float32)
    intrinsics = np.stack([fx, fy, cx, cy], -1)
    return Trajectory(
        cameras=Cameras(
            poses=camera_to_worlds,
            normalized_intrinsics=intrinsics/w,
            image_sizes=np.array((w, h), dtype=np.int32)[None].repeat(len(camera_to_worlds), 0),
            camera_types=np.array([camera_type.value] * len(camera_to_worlds), dtype=np.int32),
            distortion_parameters=np.zeros((len(camera_to_worlds), 0), dtype=np.float32),
            nears_fars=None),
        fps=fps)


@dataclass
class Trajectory:
    cameras: Cameras
    fps: float

    @classmethod
    def from_json(cls, data: Union[Dict[str, Any], IO]) -> 'Trajectory':
        if not isinstance(data, dict):
            # Load the data from IO
            if isinstance(data, (str, Path)):
                with open(data, "r", encoding="utf8") as f:
                    return cls.from_json(f)
            else:
                return cls.from_json(json.load(data))
        return read_nerfstudio_trajectory(data)


@click.command("render-trajectory")
@click.option("--checkpoint", type=str, required=True)
@click.option("--trajectory", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None, help="output a mp4/directory/tar.gz file")
@click.option("--output-type", type=click.Choice(get_args(OutputType)), default="color", help="output type")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--backend", type=click.Choice(registry.ALL_BACKENDS), default=os.environ.get("NB_BACKEND", None))
@handle_cli_error
def main(checkpoint: Union[str, Path], trajectory: Path, output, output_type: OutputType, verbose, backend):
    checkpoint = str(checkpoint)
    setup_logging(verbose)

    if os.path.exists(output):
        logging.critical("Output path already exists")
        sys.exit(1)

    # Parse trajectory
    _trajectory = Trajectory.from_json(trajectory)

    # Read method nb-info
    logging.info(f"Loading checkpoint {checkpoint}")
    with open_any_directory(checkpoint, mode="r") as checkpoint_path:
        assert checkpoint_path.exists(), f"checkpoint path {checkpoint} does not exist"
        assert (checkpoint_path / "nb-info.json").exists(), f"checkpoint path {checkpoint} does not contain nb-info.json"
        with (checkpoint_path / "nb-info.json").open("r") as f:
            ns_info = json.load(f)

        method_name = ns_info["method"]
        method_spec = registry.get(method_name)
        method_cls, backend = method_spec.build(backend=backend, checkpoint=Path(os.path.abspath(str(checkpoint_path))))
        logging.info(f"Using backend: {backend}")

        if hasattr(method_cls, "install"):
            method_cls.install()

        method = method_cls()
        try:
            render_frames(method, _trajectory.cameras, output, output_type=output_type, ns_info=ns_info, fps=_trajectory.fps)
            logging.info(f"Output saved to {output}")
        finally:
            if hasattr(method, "close"):
                typing.cast(Any, method).close()
