#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import subprocess
import random
from pathlib import Path
import shlex
import logging
import copy
from typing import Optional, Iterable
import os
import tempfile
import numpy as np
from PIL import Image
from random import randint
from argparse import ArgumentParser

try:
    from shlex import join as shlex_join
except ImportError:

    def shlex_join(split_command):
        """Return a shelshlex.ped string from *split_command*."""
        return " ".join(shlex.quote(arg) for arg in split_command)


import torch

from ...types import Method, MethodInfo, CurrentProgress, ProgressCallback, RenderOutput
from ...datasets import Dataset
from ...cameras import CameraModel, Cameras
from ...utils import cached_property
from ...pose_utils import get_transform_and_scale
from ...math_utils import rotate_spherical_harmonics, rotation_matrix_to_quaternion
from ...io import wget

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import GaussianModel
import scene.dataset_readers
from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov
from scene.dataset_readers import storePly, fetchPly
from scene.gaussian_model import inverse_sigmoid, build_rotation, PlyData, PlyElement  # noqa: E402
from scene.dataset_readers import CameraInfo as _old_CameraInfo
from utils.general_utils import safe_state
from utils.graphics_utils import fov2focal  # noqa: E402
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import SH2RGB
from scene import Scene, sceneLoadTypeCallbacks
from train import create_offset_gt
from utils import camera_utils
from utils.general_utils import PILtoTorch

def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
    z_sign = 1.0
    P = torch.zeros((4, 4))
    P[0, 0] = 2.0 * fx / w
    P[1, 1] = 2.0 * fy / h
    P[0, 2] = (2.0 * cx - w) / w
    P[1, 2] = (2.0 * cy - h) / h
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

#
# Patch Gaussian Splatting to include sampling masks
# Also, fix cx, cy (ignored in mip-splatting)
#
# Patch loadCam to include sampling mask
_old_loadCam = camera_utils.loadCam
def loadCam(args, id, cam_info, resolution_scale):
    camera = _old_loadCam(args, id, cam_info, resolution_scale)

    sampling_mask = None
    if cam_info.sampling_mask is not None:
        sampling_mask = PILtoTorch(cam_info.sampling_mask, (camera.image_width, camera.image_height))
    setattr(camera, "sampling_mask", sampling_mask)
    setattr(camera, "_patched", True)

    # Fix cx, cy (ignored in mip-splatting)
    camera.focal_x = fov2focal(cam_info.FovX, camera.image_width)
    camera.focal_y = fov2focal(cam_info.FovY, camera.image_height)
    camera.cx = cam_info.cx
    camera.cy = cam_info.cy
    camera.projection_matrix = getProjectionMatrixFromOpenCV(
        camera.image_width, 
        camera.image_height, 
        camera.focal_x, 
        camera.focal_y, 
        camera.cx, 
        camera.cy, 
        camera.znear, 
        camera.zfar).transpose(0, 1).cuda()

    return camera
camera_utils.loadCam = loadCam


# Patch CameraInfo to add sampling mask
class CameraInfo(_old_CameraInfo):
    def __new__(cls, *args, sampling_mask=None, cx, cy, **kwargs):
        self = super(CameraInfo, cls).__new__(cls, *args, **kwargs)
        self.sampling_mask = sampling_mask
        self.cx = cx
        self.cy = cy
        return self
scene.dataset_readers.CameraInfo = CameraInfo


def normalize_ssim(ssim_value, mask_percentage):
    # SSIM is in the range [-1, 1], 1 is perfect match
    # Move to [0, 1]
    ssim_value = ssim_value / 2.0 + 0.5
    # Invert, 0 is perfect match
    ssim_value = 1.0 - ssim_value
    # Normalize by mask percentage
    ssim_value = ssim_value / mask_percentage
    # Undo inversion, 1 is perfect match
    ssim_value = 1.0 - ssim_value
    # Move to [-1, 1]
    ssim_value = ssim_value * 2.0 - 1.0
    return ssim_value


def _load_caminfo(idx, pose, intrinsics, image_name, image_size, image=None, image_path=None, sampling_mask=None):
    pose = np.copy(pose)
    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
    pose = np.linalg.inv(pose)
    R = pose[:3, :3]
    T = pose[:3, 3]
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
        sampling_mask=sampling_mask,
        cx=cx, cy=cy)


def _convert_dataset_to_gaussian_splatting(dataset: Optional[Dataset], tempdir: str, white_background: bool = False):
    if dataset is None:
        return SceneInfo(None, [], [], nerf_normalization=dict(radius=None), ply_path=None)
    assert np.all(dataset.cameras.camera_types == CameraModel.PINHOLE.value), "Only pinhole cameras supported"

    cam_infos = []
    for idx, extr in enumerate(dataset.cameras.poses):
        intrinsics = dataset.cameras.intrinsics[idx]
        width, height = dataset.cameras.image_sizes[idx]
        pose = dataset.cameras.poses[idx]
        image_path = dataset.file_paths[idx] if dataset.file_paths is not None else f"{idx:06d}.png"
        image_name = (
            os.path.relpath(str(dataset.file_paths[idx]), str(dataset.file_paths_root)) if dataset.file_paths is not None and dataset.file_paths_root is not None else os.path.basename(image_path)
        )

        w, h = dataset.cameras.image_sizes[idx]
        im_data = dataset.images[idx][:h, :w]
        assert im_data.dtype == np.uint8, "Gaussian Splatting supports images as uint8"
        if dataset.metadata.get("name", None) == "blender":
            assert white_background, "white_background must be set for blender scenes"
            assert im_data.shape[-1] == 4
            bg = np.array([1, 1, 1])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + (1 - norm_data[:, :, 3:4]) * bg
            im_data = np.array(arr * 255.0, dtype=np.uint8)
        else:
            assert not white_background, "white_background is only supported for blender scenes"
        image = Image.fromarray(im_data)
        sampling_mask = None
        if dataset.sampling_masks is not None:
            sampling_mask = Image.fromarray((dataset.sampling_masks[idx] * 255).astype(np.uint8))

        cam_info = _load_caminfo(
            idx, pose, intrinsics, 
            image_name=image_name, 
            image_path=image_path,
            image_size=(w, h),
            image=image,
            sampling_mask=sampling_mask,
        )
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    points3D_xyz = dataset.points3D_xyz
    points3D_rgb = dataset.points3D_rgb
    if points3D_xyz is None and dataset.metadata.get("name", None) == "blender":
        # https://github.com/graphdeco-inria/gaussian-splatting/blob/2eee0e26d2d5fd00ec462df47752223952f6bf4e/scene/dataset_readers.py#L221C4-L221C4
        num_pts = 100_000
        logging.info(f"generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        points3D_xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        points3D_rgb = (SH2RGB(shs) * 255).astype(np.uint8)

    storePly(os.path.join(tempdir, "scene.ply"), points3D_xyz, points3D_rgb)
    pcd = fetchPly(os.path.join(tempdir, "scene.ply"))
    scene_info = SceneInfo(point_cloud=pcd, train_cameras=cam_infos, test_cameras=[], nerf_normalization=nerf_normalization, ply_path=os.path.join(tempdir, "scene.ply"))
    return scene_info


class MipSplatting(Method):
    config_overrides: Optional[dict] = None

    def __init__(self, checkpoint: Optional[Path] = None, config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.gaussians = None
        self.background = None
        self.step = 0

        self.scene = None

        # Setup parameters
        self._args_list = ["--source_path", "<empty>"]
        if checkpoint is not None:
            with open(os.path.join(checkpoint, "args.txt"), "r", encoding="utf8") as f:
                self._args_list = shlex.split(f.read())
        self._load_config()

        self._viewpoint_stack = []
        self._input_points = None

        self.trainCameras = None
        self.highresolution_index = None
        self.config_overrides = self.config_overrides or {}
        self.config_overrides.update(config_overrides or {})

    def _load_config(self):
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args(self._args_list)
        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

    def setup_train(self, train_dataset, *, num_iterations: Optional[int] = None, config_overrides=None):
        # Initialize system state (RNG)
        safe_state(False)

        if self.checkpoint is None:
            if train_dataset.metadata.get("name") == "blender":
                # Blender scenes have white background
                self._args_list.append("--white_background")
                logging.info("overriding default background color to white for blender dataset")

            assert "--iterations" not in self._args_list, "iterations should not be specified when loading from checkpoint"
            if num_iterations is not None:
                self._load_config()
                self._args_list.extend(("--iterations", str(num_iterations)))
                iter_factor = num_iterations / 30_000
                self._args_list.extend(("--densify_from_iter", str(int(self.opt.densify_from_iter * iter_factor))))
                self._args_list.extend(("--densify_until_iter", str(int(self.opt.densify_until_iter * iter_factor))))
                self._args_list.extend(("--densification_interval", str(int(self.opt.densification_interval * iter_factor))))
                self._args_list.extend(("--opacity_reset_interval", str(int(self.opt.opacity_reset_interval * iter_factor))))
                self._args_list.extend(("--position_lr_max_steps", str(int(self.opt.position_lr_max_steps * iter_factor))))

            config_overrides, _config_overrides = (self.config_overrides or {}), config_overrides
            config_overrides.update(_config_overrides or {})
            for k, v in config_overrides.items():
                self._args_list.append(f"--{k}")
                self._args_list.append(str(v))
            self._load_config()

        if self.checkpoint is None:
            # Verify parameters are set correctly
            if train_dataset.metadata.get("name") == "blender":
                assert self.dataset.white_background, "white_background should be True for blender dataset"

        # Setup model
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = self._build_scene(train_dataset)
        self.gaussians.training_setup(self.opt)
        if self.checkpoint:
            info = self.get_info()
            (model_params, self.step) = torch.load(str(self.checkpoint) + f"/chkpnt-{info.loaded_step}.pth")
            self.gaussians.restore(model_params, self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._input_points = (train_dataset.points3D_xyz, train_dataset.points3D_rgb)
        self._viewpoint_stack = []

        self.trainCameras = self.scene.getTrainCameras().copy()
        if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack):
            raise RuntimeError("could not patch loadCam!")

        # highresolution index
        self.highresolution_index = []
        for index, camera in enumerate(self.trainCameras):
            if camera.image_width >= 800:
                self.highresolution_index.append(index)

        self.gaussians.compute_3D_filter(cameras=self.trainCameras)

    def _eval_setup(self):
        # Initialize system state (RNG)
        safe_state(False)

        # Setup model
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = self._build_scene(None)
        info = self.get_info()
        (model_params, self.step) = torch.load(str(self.checkpoint) + f"/chkpnt-{info.loaded_step}.pth")
        self.gaussians.restore(model_params, self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    @cached_property
    def _loaded_step(self):
        loaded_step = None
        if self.checkpoint is not None:
            if not os.path.exists(self.checkpoint):
                raise RuntimeError(f"Model directory {self.checkpoint} does not exist")
            loaded_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(str(self.checkpoint)) if x.startswith("chkpnt-"))[-1]
        return loaded_step

    def get_info(self) -> MethodInfo:
        info = MethodInfo(
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset((CameraModel.PINHOLE,)),
            num_iterations=self.opt.iterations,
            loaded_step=self._loaded_step,
        )
        return info

    def _build_scene(self, dataset):
        opt = copy.copy(self.dataset)
        with tempfile.TemporaryDirectory() as td:
            os.mkdir(td + "/sparse")
            opt.source_path = td  # To trigger colmap loader
            opt.model_path = td if dataset is not None else str(self.checkpoint)
            backup = sceneLoadTypeCallbacks["Colmap"]
            try:
                info = self.get_info()
                sceneLoadTypeCallbacks["Colmap"] = lambda *args, **kwargs: _convert_dataset_to_gaussian_splatting(dataset, td, white_background=self.dataset.white_background)
                return Scene(opt, self.gaussians, load_iteration=info.loaded_step if dataset is None else None)
            finally:
                sceneLoadTypeCallbacks["Colmap"] = backup

    def render(self, cameras: Cameras, progress_callback: Optional[ProgressCallback] = None) -> Iterable[RenderOutput]:
        if self.scene is None:
            self._eval_setup()
        assert np.all(cameras.camera_types == CameraModel.PINHOLE.value), "Only pinhole cameras supported"
        sizes = cameras.image_sizes
        poses = cameras.poses
        intrinsics = cameras.intrinsics

        with torch.no_grad():
            global_total = int(sizes.prod(-1).sum())
            global_i = 0
            if progress_callback:
                progress_callback(CurrentProgress(global_i, global_total, 0, len(poses)))
            for i, pose in enumerate(poses):
                viewpoint_cam = _load_caminfo(i, pose, intrinsics[i], f"{i:06d}.png", sizes[i])
                viewpoint = loadCam(self.dataset, i, viewpoint_cam, 1.0)
                image = torch.clamp(render(viewpoint, self.gaussians, self.pipe, self.background, kernel_size=self.dataset.kernel_size)["render"], 0.0, 1.0)
                global_i += int(sizes[i].prod(-1))
                if progress_callback:
                    progress_callback(CurrentProgress(global_i, global_total, i + 1, len(poses)))
                color = image.detach().permute(1, 2, 0).cpu().numpy()
                yield {
                    "color": color,
                }

    def train_iteration(self, step):
        self.step = step
        iteration = step + 1  # Gaussian Splatting is 1-indexed
        del step

        self.gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self._viewpoint_stack:
            self._viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self._viewpoint_stack.pop(randint(0, len(self._viewpoint_stack) - 1))

        # Pick a random high resolution camera
        if random.random() < 0.3 and self.dataset.sample_more_highres:
            viewpoint_cam = self.trainCameras[self.highresolution_index[randint(0, len(self.highresolution_index) - 1)]]
            if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack):
                raise RuntimeError("could not patch loadCam!")

        # Render
        # NOTE: random background color is not supported
        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if self.dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None
        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, background, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        sampling_mask = viewpoint_cam.sampling_mask.cuda() if viewpoint_cam.sampling_mask is not None else None 

        # sample gt_image with subpixel offset
        if self.dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)
            sampling_mask = create_offset_gt(sampling_mask, subpixel_offset) if sampling_mask is not None else None

        # Apply mask
        mask_percentage = 1.0
        if sampling_mask is not None:
            image *= sampling_mask
            gt_image *= sampling_mask
            mask_percentage = sampling_mask.mean()

        Ll1 = l1_loss(image, gt_image) / mask_percentage
        ssim_value = ssim(image, gt_image)
        if sampling_mask is not None:
            ssim_value = normalize_ssim(ssim_value, mask_percentage)

        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()

        with torch.no_grad():
            psnr_value = 10 * torch.log10(1 / torch.mean((image - gt_image) ** 2) / mask_percentage)
            metrics = {"l1_loss": Ll1.detach().cpu().item(), "loss": loss.detach().cpu().item(), "psnr": psnr_value.detach().cpu().item()}

            # Densification
            if iteration < self.opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    self.gaussians.compute_3D_filter(cameras=self.trainCameras)

                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > self.opt.densify_until_iter:
                if iteration < self.opt.iterations - 100:
                    # don't update in the end of training
                    self.gaussians.compute_3D_filter(cameras=self.trainCameras)

            # Optimizer step
            if iteration < self.opt.iterations:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.step = self.step + 1
        return metrics

    def save(self, path):
        if self.scene is None:
            self._eval_setup()
        self.gaussians.save_ply(os.path.join(str(path), f"point_cloud/iteration_{self.step}", "point_cloud.ply"))
        torch.save((self.gaussians.capture(), self.step), str(path) + f"/chkpnt-{self.step}.pth")
        with open(str(path) + "/args.txt", "w", encoding="utf8") as f:
            f.write(shlex_join(self._args_list))

    def export_demo(self, path: Path, *, viewer_transform, viewer_initial_pose):
        model: GaussianModel = self.gaussians
        transform, scale = get_transform_and_scale(viewer_transform)
        R, t = transform[..., :3, :3], transform[..., :3, 3]
        xyz = model._xyz.detach().cpu().numpy()
        xyz = (xyz @ R.T + t[None, :]) * scale
        normals = np.zeros_like(xyz)

        f_dc = model._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = model._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()

        # Rotate sh using Winger's group on SO3
        features = rotate_spherical_harmonics(R, np.stack((f_dc, f_rest), axis=-1))
        f_dc, f_rest = features[..., :f_dc.shape[-1]], features[..., f_dc.shape[-1]:]

        # fuse opacity and scale
        current_opacity_with_filter = model.get_opacity_with_3D_filter
        opacities = inverse_sigmoid(current_opacity_with_filter).detach().cpu().numpy()
        gs_scale = model.scaling_inverse_activation(model.get_scaling_with_3D_filter * scale).detach().cpu().numpy()
        
        rotmat = build_rotation(model.get_rotation.detach().cpu().numpy())
        rotmat = R @ rotmat
        rotation = rotation_matrix_to_quaternion(rotmat)

        dtype_full = [(attribute, 'f4') for attribute in model.construct_list_of_attributes(exclude_filter=True)]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, gs_scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        with tempfile.TemporaryDirectory() as tmpdirname:
            ply_file = os.path.join(tmpdirname, "gaussian_splat.ply")
            out_file = os.path.join(tmpdirname, "gaussian_splat.ksplat")
            ply_data = PlyData([el])
            ply_data.write(ply_file)

            # Convert to ksplat format
            subprocess.check_call("bash", "-c", f"""
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
""")
            output = Path(path)
            os.rename(out_file, output / "scene.ksplat")
            wget(
                "https://raw.githubusercontent.com/gzuidhof/coi-serviceworker/7b1d2a092d0d2dd2b7270b6f12f13605de26f214/coi-serviceworker.min.js", 
                output / "coi-serviceworker.min.js")
            wget(
                "https://raw.githubusercontent.com/jkulhanek/nerfbaselines/bd328ea7d68942eea76037baed50501daa3a2425/web/public/three.module.min.js",
                output / "three.module.min.js")
            wget(
                "https://raw.githubusercontent.com/jkulhanek/nerfbaselines/bd328ea7d68942eea76037baed50501daa3a2425/web/public/gaussian-splats-3d.module.min.js",
                output / "gaussian-splats-3d.module.min.js")
            format_vector = lambda v: "[" + ",".join(f'{x:.3f}' for x in v) + "]"  # noqa: E731
            with (output / "index.html").open("w", encoding="utf8") as f, \
                open(Path(__file__).parent / "gaussian_splatting_demo.html", "r", encoding="utf8") as template:
                f.write(template.read().replace("{up}", format_vector(viewer_initial_pose.reshape(-1))))
