# NOTE: This code modifies 3DGS-mcmc:
# 1) Adds support for cx, cy not in the center of the image
# 2) Adds support for masks
import copy
import json
import sys
from argparse import ArgumentParser
import warnings
import itertools
import types
from typing import Optional
import os
import numpy as np
import tempfile
from tqdm import tqdm
from PIL import Image
from nerfbaselines import (
    Method, MethodInfo, ModelInfo, RenderOutput, Cameras, camera_model_to_int, Dataset
)
from nerfbaselines.utils import convert_image_dtype
from importlib import import_module

import_context = import_module(".sparsegs_patch", __package__).import_context
with import_context:
    from scene.dataset_readers import CameraInfo  # type: ignore
    from utils.camera_utils import loadCam  # type: ignore
    from scene import Scene  # type: ignore

    import torch
    from arguments import ModelParams, PipelineParams, OptimizationParams #  type: ignore
    from gaussian_renderer import render # type: ignore
    from scene import GaussianModel # type: ignore
    from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
    from utils.general_utils import safe_state  # type: ignore
    import train  # type: ignore
    from train import train_iteration  # type: ignore
    from scene.dataset_readers import blender_create_pcd  # type: ignore
    from scene.gaussian_model import BasicPointCloud  # type: ignore
    from guidance.sd_utils import StableDiffusion  # type: ignore


def _build_caminfo(idx, pose, intrinsics, image_name, image_size, image=None, depth=None,
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
        depth=depth,
        image=image, image_path=image_path, image_name=image_name, 
        width=int(width), height=int(height),
        mask=mask,
        cx=cx, cy=cy)


def _apply_config_overrides(parser, config_overrides):
    actions_map = {action.dest: action for action in parser._actions}
    for key, value in config_overrides.items():
        if key not in actions_map:
            raise ValueError(f"Unknown config override key: {key}")

        # Convert value to correct type
        action = actions_map[key]
        target_type = str
        if action.type is not None:
            target_type = action.type
        if isinstance(action.const, bool):
            target_type = bool
        if action.nargs == "+":
            if isinstance(value, str):
                value = value.split(",")
            if not isinstance(value, list):
                raise ValueError(f"Expected a list or comma separated list for {key}, got {type(value)}")
            value = [target_type(v) for v in value]
        else:
            if target_type is bool:
                if isinstance(value, str):
                    assert value.lower() in ("true", "false", "1", "0", "yes", "no"), \
                        f"Invalid boolean value for {key}: {value}. Expected 'true', 'false', '1', '0', 'yes', or 'no'."
                elif not isinstance(value, bool):
                    raise ValueError(f"Expected a boolean for {key}, got {type(value)}")
            else:
                value = target_type(value)
        parser.set_defaults(**{key: value})



def _(args_list, config_overrides):
    for k, v in config_overrides.items():
        if str(v).lower() == "true":
            v = True
        if str(v).lower() == "false":
            v = False
        if isinstance(v, list):
            for val in enumerate(v):
                args_list.append(f"--{k}")
                args_list.append(str(val))
        elif isinstance(v, bool):
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


def _write_images(dataset: Dataset, temp_dir: str):
    img_paths = []
    for i, image in tqdm(enumerate(dataset["images"]), desc="preparing images", total=len(dataset["images"])):
        width, height = dataset["cameras"].image_sizes[i]
        image = image[:height, :width]
        image = convert_image_dtype(image, "uint8")
        Image.fromarray(image).save(os.path.join(temp_dir, f"{i:06d}.png"))
        img_paths.append(os.path.join(temp_dir, f"{i:06d}.png"))
    return img_paths


def _extract_depths(dataset: Dataset):
    with tempfile.TemporaryDirectory() as temp_dir:
        input_folder = os.path.join(temp_dir, "input")
        os.makedirs(input_folder, exist_ok=True)
        output_folder = os.path.join(temp_dir, "output")
        os.makedirs(output_folder, exist_ok=True)
        old_sys_argv = sys.argv
        old_sys_path = sys.path
        old_cwd = os.getcwd()
        try:
            # Patch sys argv because of bug in sparsegs
            with import_context:
                root = os.path.dirname(os.path.abspath(train.__file__))
                sys.path = sys.path + [os.path.join(root, "BoostingMonocularDepth")]
                import BoostingMonocularDepth.prepare_depth  # type: ignore
                sys.argv = ["python", "code.py"]
                os.chdir(root)

            # Write images temporarily
            _write_images(dataset, input_folder)

            # Generate depths
            BoostingMonocularDepth.prepare_depth.prepare_gt_depth(
                input_folder=input_folder,
                save_folder=output_folder,
            )

            # Read depth maps
            depths = []
            for i in range(len(dataset["images"])):
                depth_path = os.path.join(output_folder, f"{i:06d}.npy")
                if not os.path.exists(depth_path):
                    raise RuntimeError(f"Depth map {depth_path} does not exist")
                depth = np.load(depth_path)
                depths.append(depth)
        finally:
            sys.path = old_sys_path
            sys.argv = old_sys_argv
            os.chdir(old_cwd)
    return depths


def _convert_dataset_to_scene_info(dataset: Optional[Dataset], 
                                   *,
                                   white_background: bool = False, 
                                   scale_coords=None, 
                                   init_type="sfm",
                                   depths):
    if dataset is None:
        image=types.SimpleNamespace()
        image.size = (10, 12)
        image.resize = lambda *_: image
        image.__array__ = lambda: np.zeros((10, 12, 3), dtype=np.uint8)
        fake_train_cams = [
            CameraInfo(
                uid=i, 
                R=np.random.rand(3, 3).astype(np.float32), 
                T=np.random.rand(3).astype(np.float32), 
                FovX=12, FovY=10,
                depth=None, image=image, image_path=None, image_name=None, 
                width=12, height=10, mask=None, cx=6, cy=5)
            for i in range(4)
        ]
        return SceneInfo(None, fake_train_cams, [], hold_cameras=[], nerf_normalization=dict(radius=None, translate=None), ply_path=None)
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
            depth=depths[idx],
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
        if init_type == "sfm":
            method_id = "3dgs-mcmc"
            assert points3D_xyz is not None and points3D_rgb is not None, \
                (f"points3D_xyz and points3D_rgb are required for {method_id} when init_type is sfm, "
                 "set init_type=random to generate random point cloud.")
            num_pts = points3D_xyz.shape[0]
            pcd = BasicPointCloud(points3D_xyz, points3D_rgb/255., np.zeros_like(points3D_xyz))
        elif init_type == "random":
            num_pts=100000
            print(f"Generating random point cloud ({num_pts})...")
            xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)
            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(points=xyz, colors=shs, normals=np.zeros((num_pts, 3)))
        else:
            raise ValueError(f"Invalid init_type: {init_type}")

    return SceneInfo(point_cloud=pcd, 
                     train_cameras=cam_infos, 
                     test_cameras=[], 
                     hold_cameras=[],
                     ply_path=None,
                     nerf_normalization=nerf_normalization)


class SparseGS(Method):
    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.step = 0

        # Setup parameters
        self._config_overrides = config_overrides if config_overrides is not None else {}
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            with open(os.path.join(checkpoint, "config_overrides.json"), "r", encoding="utf8") as f:
                _config_overrides = json.load(f)
                _config_overrides.update(self._config_overrides)
                self._config_overrides = _config_overrides
            self._loaded_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(str(checkpoint)) if x.startswith("chkpnt-"))[-1]
        self._load_config()
        self._setup(train_dataset)

    def _load_config(self):
        parser = ArgumentParser(description="Training script parameters")
        parser.add_argument("--init_type", type=str, default="sfm", choices=["sfm", "random"], help="Initialization type for the point cloud")

        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        parser.add_argument("--scale_coords", type=float, default=None, help="Scale the coords")
        parser.add_argument("--prune_sched", nargs="+", type=int, default=[])
        # Set defaults from the README
        parser.set_defaults(
            source_path="<empty>",
            resolution=1,
            eval=True,
            beta=5.0,
            lambda_pearson=0.05,
            lambda_local_pearson=0.15,
            box_p=128,
            p_corr=0.5,
            lambda_diffusion=0.001,
            SDS_freq=0.1,
            step_ratio=0.99,
            lambda_reg=0.1,
            prune_sched=[20000],
            prune_perc=0.98,
            prune_exp=7.5,
            iterations=30000,
        )
        _apply_config_overrides(parser, self._config_overrides)
        args = parser.parse_args([])
        self._dataset = lp.extract(args)
        self._dataset.init_type = args.init_type
        self._dataset.scale_coords = args.scale_coords
        self._opt = op.extract(args)
        self._pipe = pp.extract(args)
        # This is needed because globals are accessed in the training step
        self._args = args

    def _setup(self, train_dataset):
        # First extract the depth maps
        depths = None
        if train_dataset is not None: 
            depths = _extract_depths(train_dataset)

        # Initialize system state (RNG)
        safe_state(False)

        # Setup model
        self._gaussians = GaussianModel(self._dataset.sh_degree)
        scene_info =  _convert_dataset_to_scene_info(
            train_dataset, 
            white_background=self._dataset.white_background, 
            depths=depths,
            scale_coords=self._dataset.scale_coords,
            init_type=self._dataset.init_type)
        self._dataset.model_path = self.checkpoint or ""
        self._scene = Scene(scene_info, args=self._dataset, gaussians=self._gaussians, 
                            load_iteration=str(self._loaded_step) if train_dataset is None else None,
                            mode="train" if train_dataset is not None else "eval")
        if train_dataset is not None:
            self._gaussians.training_setup(self._opt)
        if train_dataset is None or self.checkpoint:
            info = self.get_info()
            loaded_step = info.get("loaded_step")
            assert loaded_step is not None, "Could not infer loaded step"
            (model_params, self.step) = torch.load(str(self.checkpoint) + f"/chkpnt-{loaded_step}.pth", weights_only=False)
            self._gaussians.restore(model_params, self._opt)

        bg_color = [1, 1, 1] if self._dataset.white_background else [0, 0, 0]
        self._background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._viewpoint_stack = None
        self._last_prune_iter = None
        self._warp_cam_stack = None
        self._prune_sched = self._args.prune_sched
        if train_dataset is not None:
            if self._dataset.lambda_diffusion:
                self._guidance_sd = StableDiffusion(device="cuda")
                self._guidance_sd.get_text_embeds([""], [""])
                print(f"[INFO] loaded SD!")

            self._diff_cam = copy.deepcopy(self._scene.getTrainCameras()[0])

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color", "depth", "accumulation", "normal"),
            viewer_default_resolution=768,
        )

    def get_info(self) -> ModelInfo:
        hparams = dict(itertools.chain(vars(self._dataset).items(), vars(self._opt).items(), vars(self._pipe).items()))
        for k in ("source_path", "resolution", "eval", "images", "model_path", "data_device"):
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
            0, camera.poses, camera.intrinsics, f"{0:06d}.png", camera.image_sizes, scale_coords=self._dataset.scale_coords)
        render_pkg = render(loadCam(self._dataset, 0, viewpoint_cam, 1.0), 
                            self._gaussians, self._pipe, self._background)
        return self._format_output({
            "color": render_pkg["render"].clamp(0, 1).detach().permute(1, 2, 0),
        }, options)

    def train_iteration(self, step):
        self.step = step
        metrics = train_iteration(self, step+1)
        self.step = step+1
        return metrics

    def save(self, path: str):
        self._gaussians.save_ply(os.path.join(str(path), f"point_cloud/iteration_{self.step}", "point_cloud.ply"))
        torch.save((self._gaussians.capture(), self.step), str(path) + f"/chkpnt-{self.step}.pth")
        with open(os.path.join(str(path), "config_overrides.json"), "w", encoding="utf8") as f:
            json.dump(self._config_overrides, f, indent=2, ensure_ascii=False)

    def export_gaussian_splats(self, *, options=None):
        del options
        return dict(
            means=self._gaussians.get_xyz.detach().cpu().numpy(),
            scales=self._gaussians.get_scaling.detach().cpu().numpy(),
            opacities=self._gaussians.get_opacity.detach().cpu().numpy(),
            quaternions=self._gaussians.get_rotation.detach().cpu().numpy(),
            spherical_harmonics=self._gaussians.get_features.transpose(1, 2).detach().cpu().numpy())
