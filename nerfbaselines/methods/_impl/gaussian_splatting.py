from pathlib import Path
import shlex
import logging
import copy
from typing import Optional, Iterable
import os
import tempfile
import numpy as np
from PIL import Image
from ...types import Method, MethodInfo, CurrentProgress, ProgressCallback, RenderOutput
from ...datasets import Dataset
from ...cameras import CameraModel, Cameras
from ...utils import cached_property

try:
    from shlex import join as shlex_join
except ImportError:

    def shlex_join(split_command):
        """Return a shelshlex.ped string from *split_command*."""
        return " ".join(shlex.quote(arg) for arg in split_command)


from argparse import ArgumentParser

import torch
from random import randint

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene import GaussianModel
from scene.cameras import Camera
from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov, CameraInfo
from scene.dataset_readers import storePly, fetchPly
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import SH2RGB
from scene import Scene, sceneLoadTypeCallbacks


def _load_cam(pose, intrinsics, camera_id, image_name, image_size, device=None):
    pose = np.copy(pose)

    # Convert from OpenGL to COLMAP's camera coordinate system (OpenCV)
    pose[0:3, 1:3] *= -1

    pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
    pose = np.linalg.inv(pose)
    R = pose[:3, :3]
    T = pose[:3, 3]
    R = np.transpose(R)

    width, height = image_size
    fx, fy, cx, cy = intrinsics
    return Camera(
        colmap_id=int(camera_id),
        R=R,
        T=T,
        FoVx=focal2fov(float(fx), float(width)),
        FoVy=focal2fov(float(fy), float(height)),
        image=torch.zeros((3, height, width), dtype=torch.float32),
        gt_alpha_mask=None,
        image_name=image_name,
        uid=id,
        data_device=device,
    )


def _convert_dataset_to_gaussian_splatting(dataset: Optional[Dataset], tempdir: str, white_background: bool = False):
    if dataset is None:
        return SceneInfo(None, [], [], nerf_normalization=dict(radius=None), ply_path=None)
    assert np.all(dataset.cameras.camera_types == CameraModel.PINHOLE.value), "Only pinhole cameras supported"

    cam_infos = []
    for idx, extr in enumerate(dataset.cameras.poses):
        intr = dataset.cameras.intrinsics[idx]
        width, height = dataset.cameras.image_sizes[idx]
        focal_length_x, focal_length_y, cx, cy = intr
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)

        extr = np.copy(extr)

        # Convert from OpenGL to COLMAP's camera coordinate system (OpenCV)
        extr[0:3, 1:3] *= -1
        extr = np.concatenate([extr, np.array([[0, 0, 0, 1]], dtype=extr.dtype)], axis=0)
        extr = np.linalg.inv(extr)
        R, T = extr[:3, :3], extr[:3, 3]
        R = np.transpose(R)
        image_path = dataset.file_paths[idx] if dataset.file_paths is not None else f"{idx:06d}.png"
        image_name = (
            os.path.relpath(str(dataset.file_paths[idx]), str(dataset.file_paths_root)) if dataset.file_paths is not None and dataset.file_paths_root is not None else os.path.basename(image_path)
        )

        w, h = dataset.cameras.image_sizes[idx]
        im_data = dataset.images[idx][:h, :w]
        assert im_data.dtype == np.uint8, "Gaussian Splatting supports images as uint8"
        if dataset.metadata.get("type", None) == "blender":
            assert white_background, "white_background must be set for blender scenes"
            assert im_data.shape[-1] == 4
            bg = np.array([1, 1, 1])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + (1 - norm_data[:, :, 3:4]) * bg
            im_data = np.array(arr * 255.0, dtype=np.uint8)
        else:
            assert not white_background, "white_background is only supported for blender scenes"
        image = Image.fromarray(im_data)

        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=float(FovY), FovX=float(FovX), image=image, image_path=image_path, image_name=image_name, width=int(width), height=int(height))
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    points3D_xyz = dataset.points3D_xyz
    points3D_rgb = dataset.points3D_rgb
    if points3D_xyz is None and dataset.metadata.get("type", None) == "blender":
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


class GaussianSplatting(Method):
    def __init__(self, checkpoint: Optional[Path] = None):
        self.checkpoint = checkpoint
        self.gaussians = None
        self.background = None
        self.step = 0

        self.scene = None

        # Setup parameters
        self._args_list = ["--source_path", "<empty>"]
        if checkpoint is not None:
            with open(os.path.join(checkpoint, "args.txt"), "r") as f:
                self._args_list = shlex.split(f.read())
        self._load_config()

        self._viewpoint_stack = []
        self._input_points = None

    def _load_config(self):
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args(self._args_list)
        self.dataset = lp.extract(args)
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

    def setup_train(self, train_dataset, *, num_iterations: int):
        # Initialize system state (RNG)
        safe_state(True)

        if self.checkpoint is None:
            if train_dataset.metadata.get("type") == "blender":
                # Blender scenes have white background
                self._args_list.append("--white_background")
                logging.info("overriding default background color to white for blender dataset")

            assert "--iterations" not in self._args_list, "iterations should not be specified when loading from checkpoint"
            self._args_list.extend(("--iterations", str(num_iterations)))
            self._load_config()

            # Verify parameters are set correctly
            assert self.opt.iterations == num_iterations, "iterations should be equal to num_iterations"
            if train_dataset.metadata.get("type") == "blender":
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

    def _eval_setup(self):
        # Initialize system state (RNG)
        safe_state(True)

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
                viewpoint = _load_cam(pose, intrinsics[i], i, f"{i:06d}.png", sizes[i], device="cuda")
                image = torch.clamp(render(viewpoint, self.gaussians, self.pipe, self.background)["render"], 0.0, 1.0)
                global_i += int(sizes[i].prod(-1))
                if progress_callback:
                    progress_callback(CurrentProgress(global_i, global_total, i + 1, len(poses)))
                color = image.detach().permute(1, 2, 0).cpu().numpy()
                yield {
                    "color": color,
                }

    def train_iteration(self, step):
        self.step = step
        # iteration = step + 1 # Gaussian Splatting is 1-indexed
        # But we do not do it here since the evaluation is usually done
        # at round numbers of iterations, and we do not want to evaluate
        # after the density was resetted. Therefore we increase the iteration
        # by 2 here
        iteration = step + 2
        del step

        self.gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self._viewpoint_stack:
            self._viewpoint_stack = self.scene.getTrainCameras().copy()
        viewpoint_cam = self._viewpoint_stack.pop(randint(0, len(self._viewpoint_stack) - 1))

        # Render
        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background

        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()
        with torch.no_grad():
            metrics = {"l1_loss": Ll1.detach().cpu().item(), "loss": loss.detach().cpu().item(), "psnr": psnr(image, gt_image).mean().detach().cpu().item()}

            # Densification
            if iteration < self.opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)

                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if iteration < self.opt.iterations + 1:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.step = self.step + 1
        return metrics

    def save(self, path):
        if self.scene is None:
            self._eval_setup()
        self.gaussians.save_ply(os.path.join(str(path), f"point_cloud/iteration_{self.step}", "point_cloud.ply"))
        torch.save((self.gaussians.capture(), self.step), str(path) + f"/chkpnt-{self.step}.pth")
        with open(str(path) + "/args.txt", "w") as f:
            f.write(shlex_join(self._args_list))
