# NOTE: This code modifies taming-3DGS:
# 1) Adds support for cx, cy not in the center of the image
# 2) Adds support for masks
import sys
import logging
import copy
import warnings
import itertools
import shlex
from typing import Optional, Any
import os

from nerfbaselines import (
    Method, MethodInfo, ModelInfo, RenderOutput, Cameras, camera_model_to_int, Dataset
)
import torch
import numpy as np
from PIL import Image

from .taming_3dgs_patch import import_context
with import_context:
    from scene.dataset_readers import CameraInfo  # type: ignore
    from utils.camera_utils import loadCam  # type: ignore
    from scene import Scene  # type: ignore
    from gaussian_renderer import render # type: ignore
    from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
    from train import train_iteration  # type: ignore
    from scene.dataset_readers import blender_create_pcd  # type: ignore
    from scene.gaussian_model import BasicPointCloud  # type: ignore
    import train as _train  # type: ignore


def _build_caminfo(idx, pose, intrinsics, image_name, image_size, image=None, 
                  image_path=None, mask=None, scale_coords=None):
    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
    pose = np.linalg.inv(pose)
    R = pose[:3, :3]
    T = pose[:3, 3]
    if scale_coords is not None:
        T = T * scale_coords
    R = np.transpose(R)

    width, height = image_size
    fx, fy, cx, cy = intrinsics
    if image is None:
        image = Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8))
    return CameraInfo(
        uid=idx, R=R, T=T, 
        FovX=focal2fov(float(fx), float(width)),
        FovY=focal2fov(float(fy), float(height)),
        image=image, image_path=image_path, image_name=image_name, 
        width=int(width), height=int(height),
        mask=mask,
        cx=cx, cy=cy)


def _config_overrides_to_args_list(args_list, config_overrides):
    for k, v in config_overrides.items():
        if str(v).lower() == "true":
            v = True
        if str(v).lower() == "false":
            v = False
        if isinstance(v, bool):
            if v:
                if f'--no-{k}' in args_list:
                    args_list.remove(f'--no-{k}')
                if f'--{k}' not in args_list:
                    args_list.append(f'--{k}')
            else:
                if f'--{k}' in args_list:
                    args_list.remove(f'--{k}')
                else:
                    args_list.append(f"--no-{k}")
        elif f'--{k}' in args_list:
            args_list[args_list.index(f'--{k}') + 1] = str(v)
        else:
            args_list.append(f"--{k}")
            args_list.append(str(v))


def _convert_dataset_to_scene_info(dataset: Optional[Dataset], white_background: bool = False, scale_coords=None):
    if dataset is None:
        return SceneInfo(None, [], [], nerf_normalization=dict(radius=None, translate=None), ply_path=None)
    assert np.all(dataset["cameras"].camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

    cam_infos = []
    for idx, extr in enumerate(dataset["cameras"].poses):
        del extr
        intrinsics = dataset["cameras"].intrinsics[idx]
        pose = dataset["cameras"].poses[idx]
        image_path = dataset["image_paths"][idx] if dataset["image_paths"] is not None else f"{idx:06d}.png"
        image_name = (
            os.path.relpath(str(dataset["image_paths"][idx]), str(dataset["image_paths_root"])) if dataset["image_paths"] is not None and dataset["image_paths_root"] is not None else os.path.basename(image_path)
        )

        w, h = dataset["cameras"].image_sizes[idx]
        im_data = dataset["images"][idx][:h, :w]
        assert im_data.dtype == np.uint8, "Gaussian Splatting supports images as uint8"
        if white_background and im_data.shape[-1] == 4:
            bg = np.array([1, 1, 1])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + (1 - norm_data[:, :, 3:4]) * bg
            im_data = np.array(arr * 255.0, dtype=np.uint8)
        if not white_background and dataset["metadata"].get("id") == "blender":
            warnings.warn("Blender scenes are expected to have white background. If the background is not white, please set white_background=True in the dataset loader.")
        elif white_background and dataset["metadata"].get("id") != "blender":
            warnings.warn("white_background=True is set, but the dataset is not a blender scene. The background may not be white.")
        image = Image.fromarray(im_data)
        mask = None
        if dataset["masks"] is not None:
            mask = Image.fromarray((dataset["masks"][idx] * 255).astype(np.uint8))

        cam_info = _build_caminfo(
            idx, pose, intrinsics, 
            image_name=image_name, 
            image_path=image_path,
            image_size=(w, h),
            image=image,
            mask=mask,
            scale_coords=scale_coords,
        )
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    points3D_xyz = dataset["points3D_xyz"]
    if scale_coords is not None:
        points3D_xyz = points3D_xyz * scale_coords
    points3D_rgb = dataset["points3D_rgb"]
    if points3D_xyz is None and dataset["metadata"].get("id", None) == "blender":
        pcd = blender_create_pcd()
    else:
        assert points3D_xyz is not None, "points3D_xyz is required"
        if points3D_rgb is None:
            points3D_rgb = np.random.rand(*points3D_xyz.shape).astype(np.float32)*255.
        pcd = BasicPointCloud(points3D_xyz, points3D_rgb/255., np.zeros_like(points3D_xyz))

    return SceneInfo(point_cloud=pcd, 
                     train_cameras=cam_infos, 
                     test_cameras=[], 
                     ply_path=None,
                     nerf_normalization=nerf_normalization)


class Taming3DGS(Method):
    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.step = 0
        self._gaussians: Any = None
        self._dataset: Any = None
        self._opt: Any = None
        self._pipe: Any = None
        self._background: Any = None
        self._args: Any = None

        # Setup parameters
        self._args_list = ["--source_path", "<empty>", "--resolution", "1", "--eval"]
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            with open(os.path.join(checkpoint, "args.txt"), "r", encoding="utf8") as f:
                self._args_list = shlex.split(f.read())
            self._loaded_step = sorted(
                int(x[x.find("_") + 1:]) for x in os.listdir(os.path.join(str(checkpoint), "point_cloud")) if x.startswith("iteration_"))[-1]

            self.step = self._loaded_step

        if config_overrides is not None:
            if checkpoint is not None:
                logging.warning("Checkpoint hyperparameters are being overriden by the provided config_overrides")
            _config_overrides_to_args_list(self._args_list, config_overrides)

        self._load_config()
        self._setup(train_dataset)

    def _load_config(self):
        parser = _train.get_argparser(self)
        parser.add_argument("--scale_coords", type=float, default=None)
        args = parser.parse_args(self._args_list)
        self._args = args
        return args

    def _setup(self, train_dataset):
        def build_scene(dataset, gaussians):
            dataset = copy.deepcopy(dataset)
            scene_info =  _convert_dataset_to_scene_info(
                train_dataset, 
                white_background=dataset.white_background, 
                scale_coords=self._args.scale_coords)
            dataset.model_path = self.checkpoint
            # Patch GaussianModel using incorrect render mode
            if self._loaded_step is not None and self._loaded_step >= self._args.ho_iteration:
                gaussians.render_mode = "abs"
                gaussians.setup_functions()
            return Scene(scene_info=scene_info, 
                         args=dataset, 
                         gaussians=gaussians, 
                         load_iteration=(
                str(self._loaded_step) 
                if self._loaded_step is not None 
                else None))
        oldstdout = sys.stdout
        try:
            _train.setup_train(self, self._args, build_scene)
        finally:
            sys.stdout = oldstdout

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color",),
            can_resume_training=False,
        )

    def get_info(self) -> ModelInfo:
        hparams = vars(self._args)
        for k in ("source_path", "resolution", "eval", "images", "model_path", "data_device",
                  "ip", "port", "debug_from", "detect_anomaly", "test_iterations", "save_iterations", "quiet", "checkpoint_iterations",
                  "start_checkpoint", "websockets", "benchmark_dir", "debug", "compute_conv3D_python", "convert_SHs_python"):
            hparams.pop(k, None)
        return ModelInfo(
            num_iterations=self._opt.iterations,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=hparams,
            **self.get_method_info(),
        )

    def _format_output(self, output, options):
        del options
        return {
            k: v.cpu().numpy() for k, v in output.items()
        }

    @torch.no_grad()
    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        camera = camera.item()
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"

        viewpoint_cam = _build_caminfo(
            0, camera.poses, camera.intrinsics, f"{0:06d}.png", camera.image_sizes, scale_coords=self._args.scale_coords)
        render_pkg = render(
            loadCam(self._dataset, 0, viewpoint_cam, 1.0), 
            self._gaussians, self._pipe, self._background)
        return self._format_output({
            "color": render_pkg["render"].clamp(0, 1).detach().permute(1, 2, 0),
        }, options)

    def train_iteration(self, step):
        if self._loaded_step is not None:
            method_id = self.get_method_info()["method_id"]
            raise RuntimeError(f"Method {method_id} was loaded from checkpoint and training cannot be resumed.")
        self.step = step
        metrics = train_iteration(self, step+1)
        self.step = step+1
        return metrics

    def save(self, path: str):
        self._gaussians.save_ply(os.path.join(str(path), "point_cloud", f"iteration_{self.step}", "point_cloud.ply"))
        # There is a bug in Taming 3DGS that prevents loading checkpoint anyway
        # We just drop it to save space
        # torch.save((self._gaussians.capture(), self.step), str(path) + f"/chkpnt-{self.step}.pth")
        with open(str(path) + "/args.txt", "w", encoding="utf8") as f:
            f.write(" ".join(shlex.quote(x) for x in self._args_list))

    def export_gaussian_splats(self, *, options=None):
        del options
        return dict(
            means=self._gaussians.get_xyz.detach().cpu().numpy(),
            scales=self._gaussians.get_scaling.detach().cpu().numpy(),
            opacities=self._gaussians.get_opacity.detach().cpu().numpy(),
            quaternions=self._gaussians.get_rotation.detach().cpu().numpy(),
            spherical_harmonics=self._gaussians.get_features.transpose(1, 2).detach().cpu().numpy())
