import json
import logging
import copy
from typing import Optional, Any
from argparse import Namespace
import os

from nerfbaselines import (
    Method, MethodInfo, ModelInfo, 
    RenderOutput, Cameras, camera_model_to_int, Dataset,
)
import torch
import numpy as np
from PIL import Image
import hashlib

from .octree_gs_patch import import_context
with import_context:
    from scene import Scene  # type: ignore
    from scene.cameras import Camera  # type: ignore
    from gaussian_renderer import render, prefilter_voxel # type: ignore
    from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
    from scene.gaussian_model import BasicPointCloud  # type: ignore
    import train as _train  # type: ignore


_disabled_config_keys = (
    "debug_from", "detect_anomaly", "quiet", "ip", "start_checkpoint",
    "source_path", "resolution", "eval", "images", "model_path", "data_device",
    "ip", "port", "debug_from", "detect_anomaly", "test_iterations", "save_iterations",
    "quiet", "checkpoint_iterations", "start_checkpoint", "websockets", "benchmark_dir", 
    "debug", "compute_conv3D_python", "convert_SHs_python"
)


def get_module_hash(module):
    out = hashlib.sha256()
    for param in module.parameters():
        out.update(param.data.cpu().numpy().tobytes())
    return out.hexdigest()


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


def get_scene_info(train_dataset, args):
    images_root = train_dataset["image_paths_root"]
    masks = train_dataset.get("masks", None)
    gargs = args

    def get_caminfo(i):
        args = copy.copy(gargs)
        camera = train_dataset["cameras"][i]
        pose = camera.poses
        R = pose[:3, :3]
        T = pose[:3, 3]
        T = -R.T @ T
        w, h = camera.image_sizes
        image_name = os.path.relpath(train_dataset["image_paths"][i], images_root)
        image = train_dataset["images"][i]
        if image.shape[-1] == 4:
            # Strip alpha channel
            alpha = image[..., 3]
            image = image[..., :3]
            if args.white_background:
                image = image.astype(np.float32) / 255.0
                alpha = alpha.astype(np.float32) / 255.0
                image = image * alpha[..., None] + (1 - alpha[..., None])
                image = (image.clip(0, 1) * 255).astype(np.uint8)
        mask = None
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if masks is not None:
            mask = masks[i]
            if mask.dtype == np.uint8:
                mask = mask.astype(np.float32) / 255.0
            mask = torch.from_numpy(mask).float()
        return Namespace(
            global_id=i,
            uid=i,
            camera=camera,
            R=R,
            T=T,
            width=w,
            height=h,
            image_name=image_name,
            image=Image.fromarray(image),
            image_path=image_name,
            mask=mask,
            id=i,
            args=args,
            FovX=focal2fov(camera.intrinsics[0], w),
            FovY=focal2fov(camera.intrinsics[1], h),
            cx=camera.intrinsics[2],
            cy=camera.intrinsics[3],
        )
    train_cam_infos = [
        get_caminfo(i)
        for i in range(len(train_dataset["cameras"]))
    ]
    nerf_normalization = getNerfppNorm(train_cam_infos)
    positions = train_dataset.get("points3D_xyz")
    if positions is None:
        logging.warning("No 3D points found in the dataset. Using random points.")
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        # We create random points inside the bounds of the synthetic Blender scenes
        positions = np.random.random((num_pts, 3)).astype(np.float32) * 2.6 - 1.3
    colors = train_dataset.get("points3D_rgb")
    if colors is None:
        logging.warning("No colors found in the dataset. Using random colors.")
        colors = np.random.random((len(positions), 3)).astype(positions.dtype)
    elif colors.dtype == np.uint8:
        colors = colors.astype(positions.dtype) / 255.0
    normals = np.zeros_like(positions)
    pcd = BasicPointCloud(positions, colors, normals)
    return SceneInfo(point_cloud=pcd,
                     train_cameras=train_cam_infos,
                     test_cameras=[],
                     nerf_normalization=nerf_normalization,
                     ply_path=None)


def camera_to_minicam(camera):
    camera = camera.item()
    width, height = camera.image_sizes
    fx, fy, cx, cy = camera.intrinsics
    fovy = focal2fov(fy, height)
    fovx = focal2fov(fx, width)
    pose = camera.poses
    R = pose[:3, :3]
    T = pose[:3, 3]
    T = -R.T @ T
    def noop_ret(self, *args, **kwargs):
        del args, kwargs
        return self

    class Tensor:
        clamp = noop_ret
        to = noop_ret
        __mul__ = noop_ret

        @property
        def shape(self):
            return (3, height, width)
    return Camera(
        None, cx, cy,
        0, R, T, fovx, fovy,
        Tensor(), None,
        "", 1, 0,
        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
        data_device="cuda")


class OctreeGS(Method):
    module = _train

    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.config_overrides = config_overrides or {}
        self.train_dataset = train_dataset
        self.checkpoint = checkpoint
        self.step = 0

        self._gaussians: Any = None
        self._dataset: Any = None
        self._opt: Any = None
        self._pipe: Any = None
        self._background: Any = None
        self._args: Any = None
        self._scene: Any = None
        self._app_model: Any = None

        # Setup parameters
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            self._loaded_step = sorted(
                int(x[x.find("_") + 1:]) for x in os.listdir(os.path.join(str(checkpoint), "point_cloud")) if x.startswith("iteration_"))[-1]
            with open(os.path.join(checkpoint, "config-overrides.json"), "r") as f:
                self.config_overrides = json.load(f)
                self.config_overrides.update(config_overrides or {})
            self.step = self._loaded_step

        self._load_config()
        self._setup(train_dataset)

    def _load_config(self):
        args = self._get_args(self.config_overrides or {}, self)
        self._args = self._opt = self._dataset = self._pipe = args
        return args

    def _setup(self, train_dataset):
        self.module.make_gaussians(self)
        if train_dataset is not None:
            scene_info = get_scene_info(
                train_dataset, 
                args=self._args)
        else:
            scene_info = Namespace(
                train_cameras=[],
                test_cameras=[],
                nerf_normalization={'radius': None},
                point_cloud=Namespace(points=[]),
            )
        self._args.model_path = self.checkpoint
        self._scene = Scene(
            args=self._args, 
            gaussians=self._gaussians, 
            scene_info=scene_info,
            shuffle=False, 
            logger=logging.getLogger(), 
            resolution_scales=self._args.resolution_scales,
            load_iteration=(
                str(self._loaded_step) 
                if self._loaded_step is not None
                else None))
        self._args.model_path = None
        self._scene.model_path = None
        if self.checkpoint is None:
            self._viewpoint_stack = None
            self._debug_from = None
            self.module.setup_train(self)

    @classmethod
    def _get_args(cls, config_overrides, store=None):
        args_list = ["--source_path", "<empty>"]
        _config_overrides_to_args_list(args_list, config_overrides)
        parser = cls.module.get_argparser(store or Namespace())
        parser.set_defaults(
            eval=True,
            ratio=1,
            resolution=-1,
            appearance_dim=0,
            fork=2,
            base_layer=10,
            visible_threshold=-1,
            dist2level="round",
            update_ratio=0.2,
            progressive=True,
            dist_ratio=0.999,
            levels=-1,
            init_level=-1,
            extra_ratio=0.25,
            extra_up=0.01,
        )
        args = parser.parse_args(args_list)
        return args

    @classmethod
    def get_hparams(cls, config_overrides):
        args = cls._get_args(config_overrides)
        hparams = vars(args)
        for k in ("source_path", "resolution", "eval", "images", "model_path", "data_device", "hierarchy",
                  "ip", "port", "debug_from", "detect_anomaly", "test_iterations", "save_iterations", "quiet", "checkpoint_iterations",
                  "start_checkpoint", "websockets", "benchmark_dir", "debug", "compute_conv3D_python", "convert_SHs_python"):
            hparams.pop(k, None)
        return hparams

    def _format_output(self, output, options):
        del options
        return {
            k: v.cpu().numpy() for k, v in output.items()
        }

    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        """
        Get the embedding for a training image.

        Args:
            index: Index of the image.
        """
        if self._app_model is None:
            return None
        return self._app_model[index].detach().cpu().numpy()

    def _render(self, camera: Cameras, *, options=None):
        del options
        training = self._gaussians.mlp_opacity.training
        self._gaussians.eval()
        camera = camera.item()
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"
        view = camera_to_minicam(camera)
        if self._args.white_background:
            background = torch.ones((3,), dtype=torch.float32, device='cuda')
        else:
            background = torch.zeros((3,), dtype=torch.float32, device='cuda')
        self._args.debug = False
        self._gaussians.set_anchor_mask(view.camera_center, self.step, view.resolution_scale)
        voxel_visible_mask = prefilter_voxel(view, self._gaussians, self._args, background)
        # TODO: Handle appearance
        render_pkg = render(view, self._gaussians, self._args, background, visible_mask=voxel_visible_mask, ape_code=-1)
        color = render_pkg["render"].clamp(0, 1).detach().permute(1, 2, 0)
        if training:
            self._gaussians.train()
        return {
            "color": color,
        }

    @torch.no_grad()
    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        return self._format_output(self._render(camera, options=options), options)

    def train_iteration(self, step):
        self.step = step
        self._args.debug = False
        metrics = self.module.train_iteration(self, step+1)
        self.step = step+1
        return metrics

    def save(self, path: str):
        old_model_path = self._scene.model_path
        try:
            self._scene.model_path = path
            self._scene.save(self.step)
        finally:
            self._scene.model_path = old_model_path
        with open(os.path.join(path, "config-overrides.json"), "w") as f:
            json.dump(self.config_overrides, f, indent=2)
        for net in ('cov_mlp', 'color_mlp', 'opacity_mlp'):
            hash_ = get_module_hash(getattr(self._gaussians, "get_" + net))
            with open(os.path.join(path, "point_cloud", f"iteration_{self.step}", net + ".pt.sha256"), "w") as f:
                f.write(hash_)

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",  # Will be set by the registry
            required_features=frozenset(("color", "points3D_xyz", "points3D_rgb")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color", "normal", "depth", "accumulation"),
            viewer_default_resolution=768,
            can_resume_training=False,
        )

    def get_info(self) -> ModelInfo:
        hparams = vars(self._args)
        for k in _disabled_config_keys:
            hparams.pop(k, None)
        return ModelInfo(
            num_iterations=self._opt.iterations,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=hparams,
            **self.get_method_info(),
        )
