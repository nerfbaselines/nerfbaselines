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

import json
import hashlib
import pickle
import warnings
import itertools
import logging
import copy
from typing import Optional, Iterable, Sequence
import os
import tempfile
import numpy as np
from PIL import Image
from nerfbaselines.types import Method, MethodInfo, ModelInfo, OptimizeEmbeddingsOutput, RenderOutput
from nerfbaselines.types import Cameras, camera_model_to_int, Dataset
from nerfbaselines.utils import cached_property, flatten_hparams, remap_error, convert_image_dtype
from argparse import ArgumentParser

import torch
from random import randint

from utils.general_utils import PILtoTorch
from arguments import ModelParams, PipelineParams, OptimizationParams, args_init # noqa: E402
from gaussian_renderer import render # noqa: E402
from scene import GaussianModel # noqa: E402
import scene.dataset_readers
from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # noqa: E402
from scene.dataset_readers import CameraInfo as _old_CameraInfo
from scene.dataset_readers import storePly, fetchPly  # noqa: E402
from utils.general_utils import safe_state  # noqa: E402
from utils.graphics_utils import fov2focal  # noqa: E402
from utils.loss_utils import l1_loss, ssim  # noqa: E402
from utils.sh_utils import SH2RGB  # noqa: E402
from scene import Scene, sceneLoadTypeCallbacks  # noqa: E402
from utils import camera_utils  # noqa: E402
from utils.image_utils import psnr  # type: ignore
import lpips  # type: ignore



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
# Also, fix cx, cy (ignored in gaussian-splatting)
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

    # Fix cx, cy (ignored in gaussian-splatting)
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


# Patch CameraInfo to add sampling mask
class CameraInfo(_old_CameraInfo):
    def __new__(cls, *args, sampling_mask=None, cx, cy, **kwargs):
        self = super(CameraInfo, cls).__new__(cls, *args, **kwargs)
        self.sampling_mask = sampling_mask
        self.cx = cx
        self.cy = cy
        return self
scene.dataset_readers.CameraInfo = CameraInfo


def _load_caminfo(idx, pose, intrinsics, image_name, image_size, image=None, image_path=None, sampling_mask=None, scale_coords=None):
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
        sampling_mask=sampling_mask,
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


def _convert_dataset_to_gaussian_splatting(dataset: Optional[Dataset], tempdir: str, white_background: bool = False, scale_coords=None):
    if dataset is None:
        return SceneInfo(None, [], [], nerf_normalization=dict(radius=None, translate=None), ply_path=None)
    assert np.all(dataset["cameras"].camera_types == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

    cam_infos = []
    for idx in range(len(dataset["cameras"].poses)):
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
        if not white_background and dataset["metadata"].get("name") == "blender":
            warnings.warn("Blender scenes are expected to have white background. If the background is not white, please set white_background=True in the dataset loader.")
        elif white_background and dataset["metadata"].get("name") != "blender":
            warnings.warn("white_background=True is set, but the dataset is not a blender scene. The background may not be white.")
        image = Image.fromarray(im_data)
        sampling_mask = None
        if dataset["sampling_masks"] is not None:
            sampling_mask = Image.fromarray(convert_image_dtype(dataset["sampling_masks"][idx], np.uint8))

        cam_info = _load_caminfo(
            idx, pose, intrinsics, 
            image_name=image_name, 
            image_path=image_path,
            image_size=(w, h),
            image=image,
            sampling_mask=sampling_mask,
            scale_coords=scale_coords,
        )
        cam_infos.append(cam_info)

    cam_infos = sorted(cam_infos.copy(), key=lambda x: x.image_name)
    nerf_normalization = getNerfppNorm(cam_infos)

    points3D_xyz = dataset["points3D_xyz"]
    if scale_coords is not None:
        points3D_xyz = points3D_xyz * scale_coords
    points3D_rgb = dataset["points3D_rgb"]
    if points3D_xyz is None and dataset["metadata"].get("name", None) == "blender":
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


class GaussianSplattingWild(Method):
    _method_name: str = "gaussian-splatting-wild"

    @remap_error
    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.gaussians = None
        self.background = None
        self.step = 0
        self.scene = None
        self.lpips_criteria = None

        # Setup parameters
        self._args_list = ["--source_path", "<empty>", "--resolution", "1", "--eval"]
        if self.checkpoint is None and config_overrides is not None:
            _config_overrides_to_args_list(self._args_list, config_overrides)

        self._default_embedding = None
        if self.checkpoint is not None and os.path.exists(os.path.join(self.checkpoint, "default_embedding.npy")):
            self._default_embedding = np.load(os.path.join(self.checkpoint, "default_embedding.npy"))

        self._load_config()
        self._setup(train_dataset)

        # Needed for get_train_embedding
        self._train_dataset_cache = None
        self._train_dataset_link = None
        if self.checkpoint is not None:
            with open(os.path.join(self.checkpoint, "train_dataset_link.json"), "r", encoding="utf8") as file:
                link = json.load(file)
                self._train_dataset_link = link["link"], link["image_names_sha"]
        if train_dataset is not None:
            if (
                train_dataset["metadata"].get("name") == "phototourism" and
                train_dataset["metadata"].get("scene") in {"sacre-coeur", "brandenburg-gate", "trevi-fountain"}):
                scene = train_dataset["metadata"]["scene"]
                image_names_sha = hashlib.sha256("".join([os.path.split(x)[-1] for x in train_dataset["image_paths"]]).encode()).hexdigest()
                self._train_dataset_link = f"external://phototourism/{scene}", image_names_sha
            else:
                warnings.warn("Train dataset is not a phototourism scene supported by nerfbaselines. Obtaining train embeddings will be disabled for the method.")

    @property
    def _train_dataset(self):
        if self._train_dataset_cache is None and self._train_dataset_link is not None:
            logging.info(f"Loading train dataset from {self._train_dataset_link[0]}")
            from nerfbaselines.datasets import load_dataset
            features = self.get_method_info()["required_features"]
            supported_camera_models = self.get_method_info()["supported_camera_models"]
            train_dataset = load_dataset(self._train_dataset_link[0], split="train", features=features, supported_camera_models=supported_camera_models)
            image_names_sha = hashlib.sha256("".join([os.path.split(x)[-1] for x in train_dataset["image_paths"]]).encode()).hexdigest()
            if self._train_dataset_link[1] != image_names_sha:
                raise RuntimeError(f"Image names in train dataset do not match the expected image names '{image_names_sha}' != '{self._train_dataset_link[1]}'. Method wasn't trained on this dataset.")
            self._train_dataset_cache = train_dataset
        return self._train_dataset_cache

    def _load_config(self):
        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument("--scale_coords", type=float, default=None, help="Scale the coords")
        parser.add_argument("--data_perturb", nargs="+", type=str, default=[])#for lego ["color","occ"]
        if self.checkpoint is None:
            args = parser.parse_args(self._args_list)
            args = args_init.argument_init(args)
        else:
            with open(os.path.join(self.checkpoint, "cfg_arg.pkl"), "rb") as file:
                args = pickle.load(file)
        self.dataset = lp.extract(args)
        self.dataset.scale_coords = args.scale_coords
        op.position_lr_max_steps = op.iterations
        self.opt = op.extract(args)
        self.pipe = pp.extract(args)
        self.args = args

    def _setup(self, train_dataset):
        # Initialize system state (RNG)
        safe_state(False)

        # Setup model
        logging.info(f"Experiment Configuration: {self.args}")
        logging.info(f"Model initialization and Data reading....")

        # Setup model
        self.gaussians = GaussianModel(self.dataset.sh_degree, self.args)
        self.scene = self._build_scene(train_dataset)
        if train_dataset is not None:
            self.gaussians.training_setup(self.opt)
        if train_dataset is None or self.checkpoint:
            info = self.get_info()
            loaded_step = info["loaded_step"]
            ckpt_path = os.path.join(self.checkpoint,
                                     "ckpts_point_cloud",
                                     "iteration_" + str(loaded_step),
                                     "point_cloud.ply")
            logging.info(f"Loading checkpoint data from {ckpt_path}")
            self.gaussians.load_ckpt_ply(ckpt_path)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._viewpoint_stack = []
        self._input_points = None
        if train_dataset is not None:
            self._input_points = (train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])

        logging.info(f"Start trainning....")

        if self.args.warm_up_iter > (self._loaded_step or 0):
            self.gaussians.set_learning_rate("box_coord",0.0)
            
        if train_dataset is not None and self.args.use_lpips_loss:#vgg alex
            self.lpips_criteria = lpips.LPIPS(net='vgg').to("cuda:0")

    @cached_property
    def _loaded_step(self):
        loaded_step = None
        if self.checkpoint is not None:
            if not os.path.exists(self.checkpoint):
                raise RuntimeError(f"Model directory {self.checkpoint} does not exist")
            loaded_step = sorted(int(x[x.find("_") + 1:]) for x in os.listdir(os.path.join(self.checkpoint, "ckpts_point_cloud")) if x.startswith("iteration_"))[-1]
        return loaded_step

    @classmethod
    def get_method_info(cls):
        assert cls._method_name is not None, "Method was not properly registered"
        return MethodInfo(
            name=cls._method_name,
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color",),
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
                scene = Scene(opt, self.gaussians, load_iteration=info["loaded_step"] if dataset is None else None)
                return scene
            finally:
                sceneLoadTypeCallbacks["Colmap"] = backup

    def train_iteration(self, step):
        self.step = step
        iteration = step + 1  # Gaussian Splatting is 1-indexed
        del step

        self.gaussians.update_learning_rate(iteration, self.args.warm_up_iter)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Pick a random Camera
        if not self._viewpoint_stack:
            loadCam.was_called = False
            self._viewpoint_stack = self.scene.getTrainCameras().copy()
            if any(not getattr(cam, "_patched", False) for cam in self._viewpoint_stack):
                raise RuntimeError("could not patch loadCam!")
        viewpoint_cam = self._viewpoint_stack.pop(randint(0, len(self._viewpoint_stack) - 1))

        # Render
        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg, store_cache=True)
        if viewpoint_cam.colmap_id == 0 or self._default_embedding is None:
            self._default_embedding = self.gaussians.color_net.cache_outd.detach().cpu().numpy().reshape(-1)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Apply mask
        # NOTE: Not included in the original code
        if viewpoint_cam.sampling_mask is not None:
            sampling_mask = viewpoint_cam.sampling_mask.cuda()
            image = image * sampling_mask + (1.0 - sampling_mask) * image.detach()

        gt_image = viewpoint_cam.original_image.cuda()
      
        if self.args.use_features_mask and iteration>self.args.features_mask_iters:#2500
            mask=self.gaussians.features_mask
            mask=torch.nn.functional.interpolate(mask,size=(image.shape[-2:]))
            Ll1 = l1_loss(image*mask, gt_image*mask)                        
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image*mask, gt_image*mask))
            
        else:
            Ll1 = l1_loss(image, gt_image)                     
            loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if self.args.use_features_mask and iteration>self.args.features_mask_iters:#2500
            loss+=(torch.square(1-self.gaussians.features_mask)).mean()*self.args.features_mask_loss_coef

        if self.args.use_scaling_loss :
            loss+=torch.abs(self.gaussians.get_scaling).mean()*self.args.scaling_loss_coef
        if self.args.use_lpips_loss: 
            loss+=self.lpips_criteria(image,gt_image).mean()*self.args.lpips_loss_coef

        if ( self.gaussians.use_kmap_pjmap or self.gaussians.use_okmap) and self.args.use_box_coord_loss:
            loss+=torch.relu(torch.abs(self.gaussians.map_pts_norm)-1).mean()*self.args.box_coord_loss_coef
        psnr_ = psnr(image,gt_image).mean().double()       
        loss.backward()
 
        with torch.no_grad():
            metrics = {
                "l1_loss": Ll1.detach().cpu().item(), 
                "loss": loss.detach().cpu().item(), 
                "psnr": psnr_.detach().cpu().item(),
                "num_gaussians": self.gaussians._xyz.shape[0],
            }

            # Densification
            if iteration < self.opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > self.opt.densify_from_iter and iteration % self.opt.densification_interval == 0:
                    size_threshold = 20 if iteration > self.opt.opacity_reset_interval else None
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)
                    self._default_embedding = None

                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if iteration < self.opt.iterations + 1:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        self.step = self.step + 1
        return metrics

    def save(self, path: str):
        os.makedirs(str(path), exist_ok=True)

        #save args
        with open(os.path.join(path,'cfg_arg.pkl'), 'wb') as file:
            pickle.dump(self.args, file)

        old_model_path = self.scene.model_path
        try:
            self.scene.model_path = str(path)
            self.scene.save(self.step)
        finally:
            self.scene.model_path = old_model_path

        # Save default embedding
        if self._default_embedding is not None:
            np.save(os.path.join(path, "default_embedding.npy"), self._default_embedding)
        if self._train_dataset_link is not None:
            with open(os.path.join(path, "train_dataset_link.json"), "w", encoding="utf8") as file:
                json.dump(dict(
                    link=self._train_dataset_link[0],
                    image_names_sha=self._train_dataset_link[1],
                ), file)

    def render(self, cameras: Cameras, *, embeddings=None, options=None, _gt_images=None, _store_cache=False) -> Iterable[RenderOutput]:
        del options
        assert np.all(cameras.camera_types == camera_model_to_int("pinhole")), "Only pinhole cameras supported"
        sizes = cameras.image_sizes
        poses = cameras.poses
        intrinsics = cameras.intrinsics

        with torch.no_grad():
            for i, pose in enumerate(poses):
                viewpoint_cam = _load_caminfo(i, pose, intrinsics[i], f"{i:06d}.png", 
                                              sizes[i], 
                                              image=Image.fromarray(_gt_images[i]) if _gt_images is not None else None,
                                              scale_coords=self.dataset.scale_coords)
                viewpoint = loadCam(self.dataset, i, viewpoint_cam, 1.0)
                emb = embeddings[i] if embeddings is not None and embeddings[i] is not None else self._default_embedding

                use_cache = False
                if emb is not None and not _store_cache:
                    num_gaussians = self.gaussians._xyz.shape[0]
                    self.gaussians.color_net.cache_outd = torch.tensor(emb, dtype=torch.float32, device="cuda").view(num_gaussians, -1)
                    use_cache = True
                if use_cache:
                    # Bug in the current code where this attribute is not set
                    self.gaussians._opacity_dealed = self.gaussians._opacity
                rendering = render(viewpoint, self.gaussians, self.pipe, self.background, use_cache=use_cache, store_cache=_store_cache)
                color = torch.clamp(rendering["render"], 0.0, 1.0).detach().permute(1, 2, 0).cpu().numpy()
                yield {
                    "color": color,
                }

    def optimize_embeddings(
        self, 
        dataset: Dataset,
        embeddings: Optional[Sequence[np.ndarray]] = None
    ) -> Iterable[OptimizeEmbeddingsOutput]:
        """
        Optimize embeddings for each image in the dataset.

        Args:
            dataset: Dataset.
            embeddings: Optional initial embeddings.
        """
        for i in range(len(dataset["cameras"].poses)):
            emb = None
            if embeddings is not None and embeddings[i] is not None:
                emb = embeddings[i]
            else:
                for _ in self.render(dataset["cameras"][i:i+1], embeddings=embeddings, _gt_images=dataset["images"][i:i+1], _store_cache=True):
                    pass
                emb = self.gaussians.color_net.cache_outd.detach().cpu().numpy().reshape(-1)
            assert emb is not None, "Embedding is None"
            yield {
                "embedding": emb,
                "metrics": {},
                "render_output": {},
            }

    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        """
        Get the embedding for a training image.

        Args:
            index: Index of the image.
        """
        if self._train_dataset is None:
            raise NotImplementedError("Method supports optimizing embeddings, but train dataset is required to infer the embeddings.")

        i = index
        for _ in self.render(self._train_dataset["cameras"][i:i+1], embeddings=None, _gt_images=self._train_dataset["images"][i:i+1], _store_cache=True):
            pass
        return self.gaussians.color_net.cache_outd.detach().cpu().numpy().reshape(-1)
