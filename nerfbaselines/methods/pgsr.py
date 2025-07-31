import tempfile
import shutil
import math
import logging
import sys
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

from .pgsr_patch import import_context
with import_context:
    from scene import Scene  # type: ignore
    from scene.cameras import Camera  # type: ignore
    from gaussian_renderer import render # type: ignore
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
            image=image,
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
    return Camera(
        None, None, cx, cy,
        0, R, T, fovx, fovy,
        width, height,
        "", "", 0,
        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
        ncc_scale=1.0,
        preload_img=False, data_device="cuda")


class PGSR(Method):
    module = _train

    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.config_overrides = config_overrides
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
            self.step = self._loaded_step

        self._load_config()
        self._setup(train_dataset)

    def _load_config(self):
        args = self._get_args(self.config_overrides or {}, self)
        self._args = args
        return args

    def _setup(self, train_dataset):
        def build_scene(dataset, gaussians, *args, **kwargs):
            dataset = copy.deepcopy(dataset)
            dataset.model_path = self.checkpoint
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
                kwargs["resolution_scales"] = []
            scene = Scene(scene_info=scene_info, 
                         args=dataset, 
                         gaussians=gaussians, 
                         *args,
                         load_iteration=(
                             str(self._loaded_step) 
                             if self._loaded_step is not None
                             else None), 
                          **kwargs)
            return scene
        oldstdout = sys.stdout
        try:
            with import_context:
                self.module.setup_train(self, self._args, build_scene)
                if self.checkpoint is not None:
                    self._app_model.load_weights(self.checkpoint)
        finally:
            sys.stdout = oldstdout

    @classmethod
    def _get_args(cls, config_overrides, store=None):
        args_list = ["--source_path", "<empty>", "--resolution", "1", "--eval"]
        _config_overrides_to_args_list(args_list, config_overrides)
        parser = cls.module.get_argparser(store or Namespace())
        parser.add_argument("--max_depth", default=5.0, type=float)
        parser.add_argument("--voxel_size", default=0.002, type=float)
        parser.add_argument("--num_images", default=1600, type=int)
        parser.add_argument("--num_cluster", default=1, type=int)
        parser.add_argument("--use_depth_filter", action="store_true")
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
        camera = camera.item()
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"
        viewpoint_cam = camera_to_minicam(camera)
        if self._args.white_background:
            background = torch.ones((3,), dtype=torch.float32, device='cuda')
        else:
            background = torch.zeros((3,), dtype=torch.float32, device='cuda')
        render_pkg = render(viewpoint_cam, self._gaussians, self._pipe, 
                            background, app_model=self._app_model)
        color = render_pkg["render"].clamp(0, 1).detach().permute(1, 2, 0)
        emb = (options or {}).get("embedding", None)
        if self._app_model is not None and emb is not None:
            log_scale, offset = float(emb[0]), float(emb[1])
            color = (math.exp(log_scale) * color + offset).clip(0, 1)
        return {
            "color": color,
            "depth": render_pkg["rendered_distance"].squeeze(0).detach(),
            "normal": render_pkg["rendered_normal"].detach().permute(1, 2, 0),
            "accumulation": render_pkg["accumulation"].squeeze(0).detach(),
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
            self._app_model.save_weights(
                self._scene.model_path, 
                self._opt.iterations
            )
        finally:
            self._scene.model_path = old_model_path

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

    def export_gaussian_splats(self, *, options=None):
        # TODO: Add appearance
        del options
        return dict(
            means=self._gaussians.get_xyz.detach().cpu().numpy(),
            scales=self._gaussians.get_scaling.detach().cpu().numpy(),
            opacities=self._gaussians.get_opacity.detach().cpu().numpy(),
            quaternions=self._gaussians.get_rotation.detach().cpu().numpy(),
            spherical_harmonics=self._gaussians.get_features.transpose(1, 2).detach().cpu().numpy())

    def export_mesh(self, path: str, train_dataset=None, options=None, **kwargs):
        import render  # type: ignore
        del kwargs
        assert train_dataset is not None, "train_dataset is required for export_mesh. Please add --data option to the command."

        self = PGSR(checkpoint=self.checkpoint,
                    train_dataset=train_dataset,
                    config_overrides=self.config_overrides)
        backup_scene = render.Scene
        backup_gaussian_model = render.GaussianModel
        backup_model_path = self._args.model_path
        with torch.no_grad(), tempfile.TemporaryDirectory() as tmpdir:
            try:
                render.Scene = lambda *args, **kwargs: self._scene
                render.GaussianModel = lambda *args, **kwargs: self._gaussians
                self._args.model_path = tmpdir
                # Hack to avoid a bug in PGSR
                for cam in self._scene.train_cameras[1.0]:
                    cam.image_name = cam.image_name.replace("/", "__")
                render.render_sets(self._args, self.step, self._args, False, True, self._args.max_depth, self._args.voxel_size, self._args.num_cluster, self._args.use_depth_filter)

                os.makedirs(path, exist_ok=True)
                shutil.move(os.path.join(tmpdir, "mesh", "tsdf_fusion_post.ply"), os.path.join(path, "mesh.ply"))
            finally:
                render.Scene = backup_scene
                render.GaussianModel = backup_gaussian_model
                self._args.model_path = backup_model_path
