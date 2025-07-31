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

import shutil
import contextlib
import dataclasses
import warnings
import itertools
import random
import shlex
import logging
import copy
from typing import Optional
import os
import tempfile
import numpy as np
from PIL import Image
from random import randint
from argparse import ArgumentParser
import shlex

import torch

from nerfbaselines import (
    Method, 
    MethodInfo, 
    RenderOutput, 
    ModelInfo,
    Cameras, 
    camera_model_to_int,
    Dataset,
)

from arguments import ModelParams, PipelineParams, OptimizationParams  # type: ignore
from gaussian_renderer import render  # type: ignore
from scene import GaussianModel  # type: ignore
import scene.dataset_readers  # type: ignore
from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
from scene.dataset_readers import storePly, fetchPly  # type: ignore
from scene.dataset_readers import CameraInfo as _old_CameraInfo  # type: ignore
from utils.general_utils import safe_state  # type: ignore
from utils.graphics_utils import fov2focal  # type: ignore
from utils.loss_utils import l1_loss, ssim  # type: ignore
from utils.sh_utils import SH2RGB  # type: ignore
from scene import Scene, sceneLoadTypeCallbacks  # type: ignore
from train import create_offset_gt, get_edge_aware_distortion_map, L1_loss_appearance  # type: ignore
from utils import camera_utils  # type: ignore
from utils.general_utils import PILtoTorch  # type: ignore
from utils.depth_utils import depth_to_normal  # type: ignore


def flatten_hparams(hparams, *, separator: str = "/", _prefix: str = ""):
    flat = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)}
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator).items())
        else:
            flat[k] = v
    return flat


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
# Patch Gaussian Splatting to include masks
# Also, fix cx, cy (ignored in gof)
#
# Patch loadCam to include mask
_old_loadCam = camera_utils.loadCam
def loadCam(args, id, cam_info, resolution_scale):
    camera = _old_loadCam(args, id, cam_info, resolution_scale)

    mask = None
    if cam_info.mask is not None:
        mask = PILtoTorch(cam_info.mask, (camera.image_width, camera.image_height))
    setattr(camera, "mask", mask)
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
    camera.full_proj_transform = (camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0))).squeeze(0)

    return camera
camera_utils.loadCam = loadCam


# Patch CameraInfo to add mask
class CameraInfo(_old_CameraInfo):
    def __new__(cls, *args, mask=None, cx, cy, **kwargs):
        self = super(CameraInfo, cls).__new__(cls, *args, **kwargs)
        self.mask = mask
        self.cx = cx
        self.cy = cy
        return self
scene.dataset_readers.CameraInfo = CameraInfo


def _load_caminfo(idx, pose, intrinsics, image_name, image_size, image=None, image_path=None, mask=None, scale_coords=None):
    pose = np.copy(pose)
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


def _convert_dataset_to_gaussian_splatting(dataset: Optional[Dataset], tempdir: str, white_background: bool = False, scale_coords=None):
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
        if im_data.shape[-1] == 4:
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
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

        cam_info = _load_caminfo(
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


def _config_overrides_to_args_list(args_list, config_overrides):
    for k, v in config_overrides.items():
        if str(v).lower() == "true":
            v = True
        if str(v).lower() == "false":
            v = False
        if isinstance(v, bool):
            if v:
                if f'--{k}' not in args_list:
                    args_list.append(f'--{k}')
            else:
                if f'--{k}' in args_list:
                    args_list.remove(f'--{k}')
        elif f'--{k}' in args_list:
            args_list[args_list.index(f'--{k}') + 1] = str(v)
        else:
            args_list.append(f"--{k}")
            args_list.append(str(v))


class GaussianOpacityFields(Method):
    def __init__(self, *,
                 checkpoint: Optional[str] = None, 
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.step = 0

        # Setup parameters
        self._args_list = ["--source_path", "<empty>", "--resolution", "1", "--eval"]
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            with open(os.path.join(checkpoint, "args.txt"), "r", encoding="utf8") as f:
                self._args_list = shlex.split(f.read())
            self._loaded_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(str(checkpoint)) if x.startswith("chkpnt-"))[-1]

        if self.checkpoint is None and config_overrides is not None:
            _config_overrides_to_args_list(self._args_list, config_overrides)

        self._load_config()

        self._setup(train_dataset)

    def _load_config(self):
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument("--scale_coords", type=float, default=None, help="Scale the coords")
        args = parser.parse_args(self._args_list)
        self.dataset = lp.extract(args)
        self.dataset.scale_coords = args.scale_coords
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)

    def _setup(self, train_dataset):
        # Initialize system state (RNG)
        safe_state(False)

        # Setup model
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = self._build_scene(train_dataset)
        if train_dataset is not None:
            self.gaussians.training_setup(self.opt)
        filter_3D = None
        if train_dataset is None or self.checkpoint:
            info = self.get_info()
            (model_params, filter_3D, self.step) = torch.load(str(self.checkpoint) + f"/chkpnt-{info.get('loaded_step')}.pth")
            self.gaussians.restore(model_params, self.opt)
            # NOTE: this is not handled in the original code
            self.gaussians.filter_3D = filter_3D

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._viewpoint_stack = []
        self._input_points = None
        self.trainCameras = None
        self.highresolution_index = None
        if train_dataset is not None:
            self._input_points = (train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])
            self.trainCameras = self.scene.getTrainCameras().copy()
            if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack):
                raise RuntimeError("could not patch loadCam!")

            # highresolution index
            self.highresolution_index = []
            for index, camera in enumerate(self.trainCameras):
                if camera.image_width >= 800:
                    self.highresolution_index.append(index)

        if filter_3D is None:
            self.gaussians.compute_3D_filter(cameras=self.trainCameras)

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",  # Will be set by the registry
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color", "normal", "depth", "accumulation", "distortion_map"),
            viewer_default_resolution=768,
        )

    def get_info(self) -> ModelInfo:
        hparams = flatten_hparams(dict(itertools.chain(vars(self.dataset).items(), vars(self.opt).items(), vars(self.pipe).items()))) 
        for k in ("source_path", "resolution", "eval", "images", "model_path", "data_device"):
            hparams.pop(k, None)
        return ModelInfo(
            num_iterations=self.opt.iterations,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=hparams,
            **self.get_method_info(),
        )

    def _build_scene(self, dataset):
        opt = copy.copy(self.dataset)
        with tempfile.TemporaryDirectory() as td:
            os.mkdir(td + "/sparse")
            opt.source_path = td  # To trigger colmap loader
            opt.model_path = td if dataset is not None else str(self.checkpoint)
            backup = sceneLoadTypeCallbacks["Colmap"]
            try:
                info = self.get_info()
                def colmap_loader(*args, **kwargs):
                    del args, kwargs
                    return _convert_dataset_to_gaussian_splatting(dataset, td, white_background=self.dataset.white_background, scale_coords=self.dataset.scale_coords)
                sceneLoadTypeCallbacks["Colmap"] = colmap_loader
                loaded_step = info.get("loaded_step")
                assert dataset is not None or loaded_step is not None, "Either dataset or loaded_step must be provided"
                scene = Scene(opt, self.gaussians, load_iteration=str(loaded_step) if dataset is None else None)
                # NOTE: This is a hack to match the RNG state of GS on 360 scenes
                _tmp = list(range((len(next(iter(scene.train_cameras.values()))) + 6) // 7))
                random.shuffle(_tmp)
                return scene
            finally:
                sceneLoadTypeCallbacks["Colmap"] = backup

    def _format_output(self, output, options):
        del options
        return {
            k: v.cpu().numpy() for k, v in output.items()
        }

    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        camera = camera.item()
        assert np.all(camera.camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

        with torch.no_grad():
            viewpoint_cam = _load_caminfo(0, camera.poses, camera.intrinsics, f"{0:06d}.png", camera.image_sizes, scale_coords=self.dataset.scale_coords)
            viewpoint = loadCam(self.dataset, 0, viewpoint_cam, 1.0)

            rendering = render(viewpoint, self.gaussians, self.pipe, self.background, kernel_size=self.dataset.kernel_size)["render"]
            image = rendering[:3, :, :]
            embedding_np = (options or {}).get("embedding", None)
            embedding = torch.from_numpy(embedding_np).to(device="cuda") if embedding_np is not None else None
            del embedding_np
            if self.dataset.use_decoupled_appearance and embedding is not None:
                max_idx = self.gaussians._appearance_embeddings.shape[0] - 1
                oldemb = self.gaussians._appearance_embeddings[max_idx]
                self.gaussians._appearance_embeddings.data[max_idx] = embedding
                image = L1_loss_appearance(image, viewpoint.original_image.cuda(), self.gaussians, max_idx, return_transformed_image=True)
                self.gaussians._appearance_embeddings.data[max_idx] = oldemb

            normal = rendering[3:6, :, :]
            normal = torch.nn.functional.normalize(normal, p=2, dim=0)

            # transform to world space
            c2w = (viewpoint.world_view_transform.T).inverse()
            normal2 = c2w[:3, :3] @ normal.reshape(3, -1)
            normal = normal2.reshape(3, *normal.shape[1:])
            normal = normal.permute(1, 2, 0)

            depth = rendering[6, :, :]
            # depth_normal, _ = depth_to_normal(viewpoint, depth[None, ...])
            # depth_normal = (depth_normal + 1.) / 2.
            # depth_normal = depth_normal.permute(2, 0, 1)

            accumlated_alpha = rendering[7, :, :]
            distortion_map = rendering[8, :, :]

            return self._format_output({
                "color": image.clamp(0, 1).detach().permute(1, 2, 0),
                "normal": normal,
                "depth": depth,
                "accumulation": accumlated_alpha,
                "distortion_map": distortion_map,
            }, options)

    def train_iteration(self, step):
        assert self.trainCameras is not None, "Method not initialized with training dataset"
        assert self.highresolution_index is not None, "Method not initialized with training dataset"
        self.step = step
        iteration = step + 1  # Gaussian Splatting is 1-indexed
        del step

        self.gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self._viewpoint_stack:
            loadCam.was_called = False  # type: ignore
            self._viewpoint_stack = self.scene.getTrainCameras().copy()
            if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack):
                raise RuntimeError("could not patch loadCam!")
        viewpoint_cam = self._viewpoint_stack.pop(randint(0, len(self._viewpoint_stack) - 1))

        # Pick a random high resolution camera
        if random.random() < 0.3 and self.dataset.sample_more_highres:
            viewpoint_cam = self.trainCameras[self.highresolution_index[randint(0, len(self.highresolution_index) - 1)]]
            if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack):
                raise RuntimeError("could not patch loadCam!")

        # Render
        bg = torch.rand((3), device="cuda") if getattr(self.opt, 'random_background', False) else self.background

        if self.dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None

        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg, kernel_size=self.dataset.kernel_size, subpixel_offset=subpixel_offset)
        rendering, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        image = rendering[:3, :, :]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.mask.cuda() if viewpoint_cam.mask is not None else None 

        # sample gt_image with subpixel offset
        if self.dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)
            mask = create_offset_gt(mask, subpixel_offset) if mask is not None else None

        # Apply mask
        if mask is not None:
            image = image * mask + (1.0 - mask) * image.detach()

        Ll1 = l1_loss(image, gt_image)
        # use L1 loss for the transformed image if using decoupled appearance
        if self.dataset.use_decoupled_appearance:
            Ll1 = L1_loss_appearance(image, gt_image, self.gaussians, viewpoint_cam.uid)

        rgb_loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        # depth distortion regularization
        distortion_map = rendering[8, :, :]
        distortion_map = get_edge_aware_distortion_map(gt_image, distortion_map)
        distortion_loss = distortion_map.mean()

        # depth normal consistency
        depth = rendering[6, :, :]
        depth_normal, _ = depth_to_normal(viewpoint_cam, depth[None, ...])
        depth_normal = depth_normal.permute(2, 0, 1)

        render_normal = rendering[3:6, :, :]
        render_normal = torch.nn.functional.normalize(render_normal, p=2, dim=0)

        c2w = (viewpoint_cam.world_view_transform.T).inverse()
        normal2 = c2w[:3, :3] @ render_normal.reshape(3, -1)
        render_normal_world = normal2.reshape(3, *render_normal.shape[1:])

        normal_error = 1 - (render_normal_world * depth_normal).sum(dim=0)
        depth_normal_loss = normal_error.mean()

        lambda_distortion = self.opt.lambda_distortion if iteration >= self.opt.distortion_from_iter else 0.0
        lambda_depth_normal = self.opt.lambda_depth_normal if iteration >= self.opt.depth_normal_from_iter else 0.0

        # Final loss
        loss = rgb_loss + depth_normal_loss * lambda_depth_normal + distortion_loss * lambda_distortion
        loss.backward()

        with torch.no_grad():
            psnr_value = 10 * torch.log10(1 / torch.mean((image - gt_image) ** 2))
            metrics = {
                "l1_loss": Ll1.detach().cpu().item(), 
                "loss": loss.detach().cpu().item(), 
                "psnr": psnr_value.detach().cpu().item(),
            }

            # Densification
            if iteration < self.opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.05, self.scene.cameras_extent, size_threshold)
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

    def save(self, path: str):
        self.gaussians.save_ply(os.path.join(str(path), f"point_cloud/iteration_{self.step}", "point_cloud.ply"))
        torch.save((self.gaussians.capture(), self.gaussians.filter_3D, self.step), str(path) + f"/chkpnt-{self.step}.pth")
        with open(str(path) + "/args.txt", "w", encoding="utf8") as f:
            f.write(" ".join(shlex.quote(x) for x in self._args_list))

    def export_gaussian_splats(self, *, options=None):
        del options
        return dict(
            antialias_2D_kernel_size=self.dataset.kernel_size,
            means=self.gaussians.get_xyz.detach().cpu().numpy(),
            scales=self.gaussians.get_scaling_with_3D_filter.detach().cpu().numpy(),
            opacities=self.gaussians.get_opacity_with_3D_filter.detach().cpu().numpy(),
            quaternions=self.gaussians.get_rotation.detach().cpu().numpy(),
            spherical_harmonics=self.gaussians.get_features.transpose(1, 2).detach().cpu().numpy())

    def export_mesh(self, path: str, train_dataset=None, options=None, **kwargs):
        del kwargs
        assert train_dataset is not None, "train_dataset is required for export_mesh. Please add --data option to the command."

        with temp_seed(0), torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
            from extract_mesh import marching_tetrahedra_with_binary_search  # type: ignore

            # Load cameras
            dataset_args = copy.deepcopy(self.dataset)
            dataset_args.data_device = "cpu"
            cams = [
                loadCam(dataset_args, 0, 
                    _load_caminfo(0, camera.poses, camera.intrinsics, 
                                  f"{0:06d}.png", camera.image_sizes, 
                                  scale_coords=self.dataset.scale_coords), 1.0)
                for camera in train_dataset["cameras"]]

            kernel_size = self.dataset.kernel_size
            filter_mesh = (options or {}).get("filter_mesh", False)
            texture_mesh = (options or {}).get("texture_mesh", True)
            marching_tetrahedra_with_binary_search(tmpdir, "test", 0, cams, self.gaussians, self.pipe, self.background, kernel_size, filter_mesh, texture_mesh)

            # Move resulting mesh to the output path
            render_path = os.path.join(tmpdir, "test", "ours_0", "fusion")
            _meshes = [f for f in os.listdir(render_path) if f.startswith("mesh_binary_search_") and f.endswith(".ply")]
            _meshes.sort(key=lambda x: int(x[len("mesh_binary_search_") : -len(".ply")]))
            os.makedirs(path, exist_ok=True)
            shutil.move(os.path.join(render_path, _meshes[-1]), os.path.join(path, "mesh.ply"))


@contextlib.contextmanager
def temp_seed(seed):
    npstate = np.random.get_state()
    rstate = random.getstate()
    torchstate = torch.random.get_rng_state() 
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        with torch.random.fork_rng(devices=["cuda:0"]):
            torch.random.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(rstate)
        np.random.set_state(npstate)
        torch.random.set_rng_state(torchstate)
