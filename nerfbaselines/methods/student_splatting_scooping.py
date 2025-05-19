import json
import math
import logging
import copy
from typing import Optional, Any
from argparse import Namespace
import os

from nerfbaselines import (
    Method, MethodInfo, ModelInfo, 
    RenderOutput, Cameras, camera_model_to_int, Dataset,
    OptimizeEmbeddingOutput
)
import torch
import numpy as np
from PIL import Image

from .student_splatting_scooping_patch import import_context
with import_context:
    from scene import Scene_nt  # type: ignore
    from scene.nt_model import NTModel  # type: ignore
    from scene.cameras import Camera  # type: ignore
    from scene.nt_model import BasicPointCloud # type: ignore
    from t_renderer import render  # type: ignore
    from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
    import train as _train  # type: ignore


_disabled_config_keys = (
    "debug_from", "detect_anomaly", "quiet", "ip", "start_checkpoint",
    "source_path", "resolution", "eval", "images", "model_path", "data_device",
    "ip", "port", "debug_from", "detect_anomaly", "test_iterations", "save_iterations",
    "quiet", "checkpoint_iterations", "start_checkpoint", "websockets", "benchmark_dir", 
    "debug", "compute_conv3D_python", "convert_SHs_python"
)


def _compute_weighted_affine_mapping(image, gt_image, mask=None):
    # Normalize sampling weights
    if mask is None:
        mask = torch.ones_like(image)
    mask = mask / torch.sum(mask)
    
    # Compute weighted means
    x_mean = torch.sum(mask * image)
    y_mean = torch.sum(mask * gt_image)
    
    # Compute weighted covariance and variance
    cov_xy = torch.sum(mask * (image - x_mean) * (gt_image - y_mean))
    var_x = torch.sum(mask * (image - x_mean) ** 2)
    
    # Compute coefficients
    a = cov_xy / var_x
    b = y_mean - a * x_mean
    return a.detach().cpu().item(), b.detach().cpu().item()


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
        image_name="", 
        uid=0,
        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
        data_device="cuda")


class StudentSplattingScooping(Method):
    module = _train

    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.config_overrides = config_overrides or {}
        self.train_dataset = train_dataset
        self.checkpoint = checkpoint
        self.step = 0

        self._primitives: Any = None
        self._dataset: Any = None
        self._opt: Any = None
        self._pipe: Any = None
        self._background: Any = None
        self._args: Any = None
        self._scene: Any = None
        self._app_model: Any = None
        self._debug_from = None

        # Setup parameters
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            self._loaded_step = sorted(
                int(x[x.find("_") + 1:]) for x in os.listdir(os.path.join(str(checkpoint), "point_cloud")) if x.startswith("iteration_"))[-1]
            self.step = self._loaded_step

            # Load config overrides
            if os.path.exists(os.path.join(checkpoint, "info.json")):
                with open(os.path.join(checkpoint, "info.json"), "r") as f:
                    info = json.load(f)
                    self.config_overrides = info["config_overrides"]
                    self.config_overrides.update(config_overrides or {})
            else:
                logging.warning(f"Could not find info.json in {checkpoint}.")

        self._load_config()
        self._setup(train_dataset)

    def _load_config(self):
        args = self._get_args(self.config_overrides or {}, self)
        self._args = args
        self._dataset = args
        self._opt = args
        self._pipe = args
        return args

    def _setup(self, train_dataset):
        if train_dataset is not None:
            scene_info = get_scene_info(
                train_dataset, 
                args=self._args)
        else:
            scene_info = Namespace(
                train_cameras=[],
                test_cameras=[],
                nerf_normalization={'radius': None},
            )
        self._primitives = NTModel(self._args.sh_degree, self._args.nu_degree)
        args = copy.copy(self._args)
        args.model_path = self.checkpoint
        self._scene = Scene_nt(
            scene_info=scene_info, 
            args=args, 
            primitives=self._primitives, 
            shuffle=False,
            load_iteration=(
                str(self._loaded_step) 
                if self._loaded_step is not None
                else None))
        if self.checkpoint is None:
            self.module.setup_train(self, self._args)

    @classmethod
    def _get_args(cls, config_overrides, store=None):
        args_list = ["--source_path", "<empty>", "--resolution", "1", "--eval"]
        _config_overrides_to_args_list(args_list, config_overrides)
        parser = cls.module.get_argparser(store or Namespace())
        # Pick sensible defaults
        parser.set_defaults(
            cap_max=1300000,
            nu_degree=100,
            C_burnin=5e5,
            burnin_iterations=7000,
            iterations=40000,
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

    def optimize_embedding(self, 
                           dataset: Dataset, *, 
                           embedding: Optional[np.ndarray] = None) -> OptimizeEmbeddingOutput:
        """
        Optimize embeddings on a single image (passed as a dataset).

        Args:
            dataset: Dataset (single image).
            embedding: Optional initial embedding.
        """
        del embedding
        camera = dataset["cameras"].item()
        device = torch.device("cuda")
        color = self._render(camera)["color"]
        color = color.to(device)
        # Load gt_color
        gt_color = torch.from_numpy(dataset["images"][0])
        if gt_color.dtype == torch.uint8:
            gt_color = gt_color.float().div(255).to(device)
        # Apply alpha for blender dataset
        if gt_color.shape[-1] == 4:
            alpha = gt_color[..., 3]
            gt_color = gt_color[..., :3]
            if self._args.white_background:
                gt_color = gt_color * alpha[..., None] + (1 - alpha[..., None])
        # Load masks
        masks = dataset.get("masks", None)
        mask = None
        if masks is not None:
            mask = torch.from_numpy(masks[0])
            if mask.dtype == torch.uint8:
                mask = mask.float().div(255).to(device)
        # Find optimal embedding
        assert color.shape == gt_color.shape, "Color and gt_color must have the same shape"
        scale, offset = _compute_weighted_affine_mapping(color, gt_color, mask)
        return {
            "embedding": np.array([math.log(scale), offset], dtype=np.float32)
        }

    def _render(self, camera: Cameras, *, options=None):
        del options
        camera = camera.item()
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"
        viewpoint_cam = camera_to_minicam(camera)
        if self._args.white_background:
            background = torch.ones((3,), dtype=torch.float32, device='cuda')
        else:
            background = torch.zeros((3,), dtype=torch.float32, device='cuda')

        self._args.debug = False
        render_pkg = render(viewpoint_cam, self._primitives, self._args, background) 
        color = render_pkg["render"].clamp(0, 1).detach().permute(1, 2, 0)
        return {
            "color": color,
        }

    @torch.no_grad()
    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        return self._format_output(self._render(camera, options=options), options)

    def train_iteration(self, step):
        self.step = step
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
        # Save info.json
        with open(os.path.join(path, "info.json"), "w") as f:
            json.dump({
                "config_overrides": self.config_overrides,
            }, f, indent=2)


    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",  # Will be set by the registry
            required_features=frozenset(("color", "points3D_xyz", "points3D_rgb")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color",),
            viewer_default_resolution=768,
            can_resume_training=False,
        )

    def get_info(self) -> ModelInfo:
        hparams = copy.copy(vars(self._args))
        for k in _disabled_config_keys:
            hparams.pop(k, None)
        return ModelInfo(
            num_iterations=self._opt.iterations,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=hparams,
            **self.get_method_info(),
        )
