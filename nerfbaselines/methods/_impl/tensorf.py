import math
import dataclasses
import numpy as np
import json
import logging
import os
import shlex
import torch
from pathlib import Path
from typing import Any, Dict, Iterable

try:
    from typing import Optional
except ImportError:
    from typing_extensions import Optional

from nerfbaselines.types import Dataset, CurrentProgress, RenderOutput, MethodInfo, ModelInfo, ProgressCallback
from nerfbaselines import Cameras, CameraModel
from nerfbaselines import Method

import configargparse
import opt
from opt import config_parser
from renderer import OctreeRender_trilinear_fast
from utils import N_to_reso, cal_n_samples, TVLoss
from train import SimpleSampler
from models import tensoRF as models
from dataLoader.ray_utils import ndc_rays_blender
from dataLoader.llff import average_poses


def get_rays_and_indices(camera: Cameras):
    w, h = camera.image_sizes
    xy = np.stack(np.meshgrid(np.arange(w), np.arange(h), indexing="xy"), -1).reshape(-1, 2)
    origins, directions = camera[None].get_rays(xy[None])
    return origins[0], directions[0], xy


def compute_scene_bbox_and_far(camera_centers):
    mult = 1.5
    min_bounds = np.percentile(camera_centers, 5.0, axis=0)
    max_bounds = np.percentile(camera_centers, 95.0, axis=0)
    center = (min_bounds + max_bounds) / 2
    sizes = max_bounds - min_bounds
    scene_bbox = np.stack([center - mult * sizes / 2, center + mult * sizes / 2], axis=0)
    far = float(np.linalg.norm(sizes) * mult)
    return scene_bbox, far


def get_llff_transform(poses, near_fars):
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    transform = np.linalg.inv(pose_avg_homo)

    # Step 3: correct scale so that the nearest depth is at a little more than 1.0
    # See https://github.com/bmild/nerf/issues/34
    near_original = near_fars.min()
    scale = 1 / (near_original * 0.75)  # 0.75 is the default parameter
    # the nearest depth is at 1/0.75=1.33

    transform[:3, :] *= scale
    return transform


def get_transform_and_scale(transform):
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    assert np.allclose(scale, scale[0])
    scale = float(scale[0])
    transform = transform.copy()
    transform[:3, :] /= scale
    return transform, scale


def apply_transform(poses, transform):
    transform, scale = get_transform_and_scale(transform)
    if poses.shape[-2] < 4:
        shape = poses.shape[:-2]
        poses = poses.reshape((-1, *poses.shape[-2:]))
        poses = np.concatenate((poses, np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))), -2)
        poses = poses.reshape((*shape, 4, 4))
    poses = transform @ poses
    poses = poses[..., :3, :]
    poses[:, :3, 3] *= scale
    return poses


class TensoRFDataset:
    def __init__(self, dataset: Dataset, transform=None, is_stack=False):
        self.is_stack = is_stack
        self.scene_bbox = torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        self.white_bg = True
        self.near_far = [0.1, 100.0]

        self.transform = np.eye(4)

        poses = dataset.cameras.poses.copy()

        if dataset.metadata.get("name") == "blender":
            self.white_bg = True
            self.near_far = [2.0, 6.0]
        elif dataset.metadata.get("name") == "llff":
            self.white_bg = False
            assert dataset.metadata.get("type") == "forward-facing"
            assert dataset.cameras.nears_fars is not None

            if transform is None:
                transform = get_llff_transform(poses, dataset.cameras.nears_fars)
            poses = apply_transform(poses, transform)

            dataset = dataclasses.replace(
                dataset,
                cameras=dataclasses.replace(
                    dataset.cameras,
                    poses=poses,
                    nears_fars=np.array([[0.0, 1.0]] * len(poses), dtype=np.float32),
                ),
            )

            self.near_far = [0.0, 1.0]
            self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
            self.transform = transform
        else:
            camera_centers = poses[:, :3, 3]
            scene_bbox, far = compute_scene_bbox_and_far(camera_centers)
            self.scene_bbox = torch.tensor(scene_bbox)
            self.near_far = [0.1, far]

        self.center = torch.mean(self.scene_bbox, axis=0).float().view(1, 1, 3)
        self.radius = (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self._setup(dataset)

    def _setup(self, dataset: Dataset):
        self.all_rays = []
        self.all_rgbs = []

        for i, cam in enumerate(dataset.cameras):
            if dataset.metadata.get("type") == "forward-facing":
                origins, directions, xy = get_rays_and_indices(cam)
                origins = origins.copy()
                directions = directions.copy()
                # Directions and origins to openGL
                origins[..., 1:3] *= -1
                directions[..., 1:3] *= -1
                origins = torch.tensor(origins)
                directions = torch.tensor(directions)

                W, H = cam.image_sizes
                fx, *_ = cam.intrinsics
                origins, directions = ndc_rays_blender(H, W, fx, 1.0, origins, directions)
            else:
                origins, directions, xy = get_rays_and_indices(cam)
                origins = torch.tensor(origins)
                directions = torch.tensor(directions)
                directions = torch.nn.functional.normalize(directions, 2, dim=-1)
            self.all_rays.append(torch.cat([origins, directions], -1).float())

            if dataset.images is not None:
                rgbs = dataset.images[i][xy[..., 1], xy[..., 0]]
                if rgbs.dtype == np.uint8:
                    rgbs = rgbs.astype(np.float32) / 255.0

                # RGBA is blended with white background
                if rgbs.shape[1] == 4:
                    rgbs = rgbs[:, :3] * rgbs[:, -1:] + (1 - rgbs[:, -1:])
                self.all_rgbs.append(torch.from_numpy(rgbs))


        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)  # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)  # (len(self.meta['frames])*h*w, 3)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):
        return {"rays": self.all_rays[idx], "rgbs": self.all_rgbs[idx]}


class TensoRF(Method):
    _method_name: str = "tensorf"

    def __init__(self, *,
                 checkpoint: Optional[Path] = None, 
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.renderer = OctreeRender_trilinear_fast

        self.args = None
        self.metadata = {}
        self._arg_list = ()
        if checkpoint is not None:
            with open(os.path.join(checkpoint, "metadata.json"), "r") as f:
                self.metadata = json.load(f)
                self.metadata["dataset_transform"] = np.array(self.metadata["dataset_transform"], dtype=np.float32)
            self._arg_list = shlex.split(self.metadata["args"])
        self.nSamples = None
        self.tensorf = None
        self.step = 0

        self._load_config()
        if train_dataset is not None:
            self._setup_train(train_dataset, config_overrides=config_overrides)
        else:
            self._setup_eval()

    def _load_config(self):
        self.args = config_parser(shlex.join(self._arg_list))

    @classmethod
    def get_method_info(cls) -> MethodInfo:
        assert cls._method_name is not None, "Method was not properly registered"
        return MethodInfo(
            name=cls._method_name,
            required_features=frozenset(("color",)),
            supported_camera_models=frozenset(CameraModel.__members__.values()),
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            name=self._method_name,
            num_iterations=self.args.n_iters,
            supported_camera_models=frozenset(CameraModel.__members__.values()),
            loaded_step=self.metadata.get("step"),
            batch_size=self.args.batch_size,
            eval_batch_size=self.args.batch_size,
            hparams=vars(self.args) if self.args else {},
        )

    def save(self, path: Path):
        if self.tensorf is None:
            self._setup_eval()
        with open(str(path) + "/args.txt", "w") as f:
            f.write(shlex.join(self._arg_list))
        self.tensorf.save(str(path / "tensorf.th"))
        self.metadata["args"] = shlex.join(self._arg_list)
        self.metadata["step"] = self.step
        metadata = self.metadata.copy()
        metadata["dataset_transform"] = metadata["dataset_transform"].tolist()
        with (path / "metadata.json").open("w") as f:
            json.dump(metadata, f)

    @property
    def white_bg(self):
        return self.metadata.get("white_bg", False)

    @white_bg.setter
    def white_bg(self, value: bool):
        self.metadata["white_bg"] = value

    def _setup_eval(self):
        torch.set_default_dtype(torch.float32)
        torch.manual_seed(20211202)
        np.random.seed(20211202)

        ckpt = torch.load(os.path.join(self.checkpoint, "tensorf.th"), map_location=self.device)
        kwargs = ckpt["kwargs"]
        kwargs.update({"device": self.device})
        self.tensorf = getattr(models, self.args.model_name)(**kwargs)
        self.tensorf.load(ckpt)

    def _setup_train(self, train_dataset: Dataset, *, config_overrides: Optional[Dict[str, Any]] = None):
        config_overrides = (config_overrides or {}).copy()
        if self.checkpoint is not None:
            raise NotImplementedError("Loading from checkpoint is not supported for TensoRF")

        self.metadata["dataset_metadata"] = {
            "type": train_dataset.metadata.get("type"),
            "name": train_dataset.metadata.get("name"),
        }

        # Load dataset-specific config
        dataset_name = train_dataset.metadata.get("name")
        config_name = "your_own_data.txt"
        if dataset_name == "blender":
            config_name = "lego.txt"
        elif dataset_name == "llff":
            config_name = "flower.txt"
        config_file = Path(opt.__file__).absolute().parent.joinpath("configs", config_name)
        logging.info(f"Loading config from {config_file}")
        with config_file.open("r", encoding="utf8") as f:
            config_overrides.update(configargparse.DefaultConfigFileParser().parse(f))

        # config_overrides["n_iters"] = str(num_iterations)
        for k, v in config_overrides.items():
            if isinstance(v, list):
                for vs in v:
                    self._arg_list += (f"--{k}", str(vs))
            elif isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                for vs in v[1:-1].split(","):
                    self._arg_list += (f"--{k}", str(vs))
            else:
                self._arg_list += (f"--{k}", str(v))
        logging.info("Using arguments: " + shlex.join(self._arg_list))
        self._load_config()

        torch.set_default_dtype(torch.float32)
        torch.manual_seed(20211202)
        np.random.seed(20211202)

        # init dataset
        train_dataset = TensoRFDataset(train_dataset, transform=self.metadata.get("dataset_transform"), is_stack=False)
        self.metadata["dataset_transform"] = train_dataset.transform

        self.white_bg = train_dataset.white_bg
        near_far = train_dataset.near_far

        # init resolution
        upsamp_list = self.args.upsamp_list
        self.update_AlphaMask_list = self.args.update_AlphaMask_list
        n_lamb_sigma = self.args.n_lamb_sigma
        n_lamb_sh = self.args.n_lamb_sh

        # init parameters
        # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
        aabb = train_dataset.scene_bbox.to(self.device)
        self.reso_cur = N_to_reso(self.args.N_voxel_init, aabb)
        self.nSamples = min(self.args.nSamples, cal_n_samples(self.reso_cur, self.args.step_ratio))

        tensorf = getattr(models, self.args.model_name)(
            aabb,
            self.reso_cur,
            self.device,
            density_n_comp=n_lamb_sigma,
            appearance_n_comp=n_lamb_sh,
            app_dim=self.args.data_dim_color,
            near_far=near_far,
            shadingMode=self.args.shadingMode,
            alphaMask_thres=self.args.alpha_mask_thre,
            density_shift=self.args.density_shift,
            distance_scale=self.args.distance_scale,
            pos_pe=self.args.pos_pe,
            view_pe=self.args.view_pe,
            fea_pe=self.args.fea_pe,
            featureC=self.args.featureC,
            step_ratio=self.args.step_ratio,
            fea2denseAct=self.args.fea2denseAct,
        )

        grad_vars = tensorf.get_optparam_groups(self.args.lr_init, self.args.lr_basis)
        if self.args.lr_decay_iters > 0:
            self.lr_factor = self.args.lr_decay_target_ratio ** (1 / self.args.lr_decay_iters)
        else:
            self.args.lr_decay_iters = self.args.n_iters
            self.lr_factor = self.args.lr_decay_target_ratio ** (1 / self.args.n_iters)

        logging.info(f"lr decay {self.args.lr_decay_target_ratio}, {self.args.lr_decay_iters}")

        self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        upsamp_list = self.args.upsamp_list
        self.tensorf = tensorf
        self.N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(self.args.N_voxel_init), np.log(self.args.N_voxel_final), len(upsamp_list) + 1))).long()).tolist()[1:]

        torch.cuda.empty_cache()

        self.allrays, self.allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
        if not self.args.ndc_ray:
            self.allrays, self.allrgbs = self.tensorf.filtering_rays(self.allrays, self.allrgbs, bbox_only=True)
        self.trainingSampler = SimpleSampler(self.allrays.shape[0], self.args.batch_size)

        self.Ortho_reg_weight = self.args.Ortho_weight
        self.L1_reg_weight = self.args.L1_weight_inital
        self.TV_weight_density, self.TV_weight_app = self.args.TV_weight_density, self.args.TV_weight_app
        self.tvreg = TVLoss()

    def train_iteration(self, step: int):
        iteration = step
        ray_idx = self.trainingSampler.nextids()
        rays_train, rgb_train = self.allrays[ray_idx], self.allrgbs[ray_idx].to(self.device)

        # rgb_map, alphas_map, depth_map, weights, uncertainty
        ndc_ray = self.args.ndc_ray
        rgb_map, alphas_map, depth_map, weights, uncertainty = self.renderer(
            rays_train, self.tensorf, chunk=self.args.batch_size, 
            N_samples=self.nSamples, white_bg=self.white_bg, ndc_ray=ndc_ray, 
            device=self.device, is_train=True,
        )

        loss = torch.mean((rgb_map - rgb_train) ** 2)

        output = {}
        output["mse"] = mse = loss.detach().item()
        output["psnr"] = -10 * math.log10(mse)

        # loss
        total_loss = loss
        if self.Ortho_reg_weight > 0:
            loss_reg = self.tensorf.vector_comp_diffs()
            total_loss += self.Ortho_reg_weight * loss_reg
            output["reg"] = loss_reg.detach().item()
        if self.L1_reg_weight > 0:
            loss_reg_L1 = self.tensorf.density_L1()
            total_loss += self.L1_reg_weight * loss_reg_L1
            output["reg_l1"] = loss_reg_L1.detach().item()

        if self.TV_weight_density > 0:
            self.TV_weight_density *= self.lr_factor
            loss_tv = self.tensorf.TV_loss_density(self.tvreg) * self.TV_weight_density
            total_loss = total_loss + loss_tv
            output["reg_tv_density"] = loss_tv.detach().item()
        if self.TV_weight_app > 0:
            self.TV_weight_app *= self.lr_factor
            loss_tv = self.tensorf.TV_loss_app(self.tvreg) * self.TV_weight_app
            total_loss = total_loss + loss_tv
            output["train/reg_tv_app"] = loss_tv.detach().item()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        loss = loss.detach().item()
        output["loss"] = loss

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * self.lr_factor

        if iteration in self.update_AlphaMask_list:
            if self.reso_cur[0] * self.reso_cur[1] * self.reso_cur[2] < 256**3:  # update volume resolution
                reso_mask = self.reso_cur
            new_aabb = self.tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == self.update_AlphaMask_list[0]:
                self.tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                self.L1_reg_weight = self.args.L1_weight_rest
                logging.info("continuing L1_reg_weight %.6f", self.L1_reg_weight)

            if not self.args.ndc_ray and iteration == self.update_AlphaMask_list[1]:
                # filter rays outside the bbox
                self.allrays, self.allrgbs = self.tensorf.filtering_rays(
                    self.allrays, self.allrgbs,
                )
                self.trainingSampler = SimpleSampler(self.allrgbs.shape[0], self.args.batch_size)

        upsamp_list = self.args.upsamp_list
        if iteration in upsamp_list:
            n_voxels = self.N_voxel_list.pop(0)
            self.reso_cur = N_to_reso(n_voxels, self.tensorf.aabb)
            self.nSamples = min(self.args.nSamples, cal_n_samples(self.reso_cur, self.args.step_ratio))
            self.tensorf.upsample_volume_grid(self.reso_cur)

            if self.args.lr_upsample_reset:
                logging.info("reset lr to initial")
                lr_scale = 1  # 0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = self.args.lr_decay_target_ratio ** (iteration / self.args.n_iters)
            grad_vars = self.tensorf.get_optparam_groups(self.args.lr_init * lr_scale, self.args.lr_basis * lr_scale)
            self.optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))
        self.step = step + 1
        return output

    @torch.no_grad()
    def render(self, cameras: Cameras, progress_callback: Optional[ProgressCallback] = None) -> Iterable[RenderOutput]:
        assert self.metadata.get("dataset_metadata") is not None, "Missing dataset_metadata"
        assert self.metadata.get("dataset_transform") is not None, "Missing dataset_transform"
        test_dataset = TensoRFDataset(
            Dataset(
                cameras=cameras,
                file_paths=[f"{i:06d}.png" for i in range(len(cameras))],
                metadata=self.metadata["dataset_metadata"],
            ),
            transform=self.metadata.get("dataset_transform"),
            is_stack=True,
        )
        idx = 0
        if progress_callback is not None:
            progress_callback(CurrentProgress(idx, len(test_dataset), idx, len(test_dataset)))
        for idx, samples in enumerate(test_dataset.all_rays):
            W, H = cameras.image_sizes[idx]
            rays = samples.view(-1, samples.shape[-1])

            rgb_map, _, depth_map, _, _ = self.renderer(rays, self.tensorf, chunk=4096, N_samples=-1, ndc_ray=self.args.ndc_ray, white_bg=self.white_bg, device=self.device)

            rgb_map = rgb_map.clamp(0.0, 1.0)
            rgb_map, depth_map = rgb_map.reshape(H, W, 3).cpu(), depth_map.reshape(H, W).cpu()
            if progress_callback is not None:
                progress_callback(CurrentProgress(idx + 1, len(test_dataset), idx + 1, len(test_dataset)))

            yield {
                "color": rgb_map.detach().numpy(),
                "depth": depth_map.detach().numpy(),
            }
