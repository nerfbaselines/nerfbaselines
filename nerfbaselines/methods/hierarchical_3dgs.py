import math
import contextlib
import tqdm
import tempfile
import gc
import json
import sys
import logging
import copy
import subprocess
from typing import Optional, Any
from argparse import Namespace
import os

from nerfbaselines import (
    Method, MethodInfo, ModelInfo, RenderOutput, Cameras, camera_model_to_int, Dataset
)
from nerfbaselines.datasets import _colmap_utils as colmap_utils
from nerfbaselines.datasets import dataset_index_select
import torch
import numpy as np
from PIL import Image

from .hierarchical_3dgs_patch import import_context
with import_context:
    import scene  # type: ignore
    import utils.camera_utils  # type: ignore
    from scene import Scene  # type: ignore
    from scene.cameras import getWorld2View2, getProjectionMatrix, MiniCam  # type: ignore
    from gaussian_renderer import render, render_post # type: ignore
    from scene.dataset_readers import SceneInfo, getNerfppNorm, focal2fov  # type: ignore
    from scene.gaussian_model import BasicPointCloud  # type: ignore
    from preprocess import make_depth_scale  # type: ignore
    from preprocess import make_chunk  # type: ignore
    import train_single as _train_single  # type: ignore
    import train_post as _train_post  # type: ignore
    import train_coarse as _train_coarse  # type: ignore
    from gaussian_hierarchy._C import expand_to_size, get_interpolation_weights  # type: ignore


_method_id = "hierarchical-3dgs"


def _noop(*args, **kwargs):
    del args, kwargs
    pass


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


class CameraDataset(utils.camera_utils.CameraDataset):
    def __getitem__(self, index):
        # Select sample
        info = self.list_cam_infos[index]
        args = info.args
        assert args.resolution == 1, "Resolution must be 1"
        width, height = info.camera.image_sizes
        fx, fy, cx, cy = info.camera.intrinsics
        FovY = focal2fov(fy, height)
        FovX = focal2fov(fx, width)
        return utils.camera_utils.Camera((width, height), 
                  colmap_id=info.id, R=info.R, T=info.T, 
                  FoVx=FovX, FoVy=FovY, depth_params=info.depth_params,
                  primx=float(cx) / width, primy=float(cy) / height,
                  image=info.image, alpha_mask=None, invdepthmap=info.invdepthmap,
                  image_name=info.image_name, uid=id, data_device="cpu", 
                  train_test_exp=args.train_test_exp, is_test_dataset=self.is_test, is_test_view=self.is_test)


scene.CameraDataset = CameraDataset
utils.camera_utils.CameraDataset = CameraDataset


def get_scene_info(train_dataset, args):
    images_root = train_dataset["image_paths_root"]
    masks = train_dataset.get("masks", None)
    gargs = args

    def get_caminfo(i):
        camera = train_dataset["cameras"][i]
        pose = camera.poses
        R = pose[:3, :3]
        T = pose[:3, 3]
        T = -R.T @ T
        image_name = os.path.relpath(train_dataset["image_paths"][i], images_root)
        image = train_dataset["images"][i]
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if masks is not None:
            alpha_mask = masks[i]
            if alpha_mask.dtype != np.uint8:
                alpha_mask = (alpha_mask * 255).astype(np.uint8)
            image = np.concatenate([image, alpha_mask[..., None]], axis=-1)
        args = copy.copy(gargs)
        invdepths = train_dataset.get("invdepths")
        depth_params = train_dataset.get("invdepth_params")
        return Namespace(
            camera=camera,
            R=R,
            T=T,
            image_name=image_name,
            invdepthmap=invdepths[i] if invdepths is not None else None,
            depth_params=depth_params[i] if depth_params is not None else None,
            image=Image.fromarray(image),
            id=i,
            args=args,
        )
    train_cam_infos = [
        get_caminfo(i)
        for i in range(len(train_dataset["cameras"]))
    ]
    nerf_normalization = getNerfppNorm(train_cam_infos)
    positions = train_dataset["points3D_xyz"]
    colors = train_dataset.get("points3D_rgb")
    if colors is None:
        colors = np.ones_like(positions) * 0.5
    elif colors.dtype == np.uint8:
        colors = colors.astype(positions.dtype) / 255.0
    normals = np.zeros_like(positions)
    pcd = BasicPointCloud(positions, colors, normals)
    return SceneInfo(point_cloud=pcd,
                     train_cameras=train_cam_infos,
                     test_cameras=[],
                     nerf_normalization=nerf_normalization,
                     ply_path=None)


def camera_to_minicam(camera, device):
    camera = camera.item()
    zfar = 100.0
    znear = 0.01
    width, height = camera.image_sizes
    fx, fy, cx, cy = camera.intrinsics
    fovy = focal2fov(fy, height)
    fovx = focal2fov(fx, width)
    primx = float(cx) / width
    primy = float(cy) / height
    pose = camera.poses
    R = pose[:3, :3]
    T = pose[:3, 3]
    T = -R.T @ T
    trans = np.zeros((3,), dtype=np.float32)
    scale = 1.0
    world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(device)
    projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=fovx, fovY=fovy, primx=primx, primy=primy).transpose(0,1).to(device)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0).to(device)
    return MiniCam(width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform)


@torch.no_grad()
def generate_invdepths(dataset, mode):
    if mode == "depth_anything_v2":
        logging.info(f"Generating depths using mode {mode}")
        import depth_anything_v2.dpt  # type: ignore
        from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore
        input_size=518
        DEVICE = 'cuda'
        depth_anything = DepthAnythingV2(**{'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]})
        basepath = os.path.dirname(os.path.dirname(os.path.abspath(depth_anything_v2.dpt.__file__)))
        depth_anything.load_state_dict(torch.load(f'{basepath}/checkpoints/depth_anything_v2_vitl.pth', map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()

        dataset["invdepths"] = []
        for image in tqdm.tqdm(dataset["images"], desc="Generating depths"):
            depth = depth_anything.infer_image(image, input_size)
            dataset["invdepths"].append(depth)

        # Clear memory
        del depth_anything
        gc.collect()
        torch.cuda.empty_cache()
        return dataset
    elif mode == "dpt":
        logging.info(f"Generating depths using mode {mode}")
        import cv2
        from torchvision.transforms import Compose
        import dpt  # type: ignore
        from dpt.models import DPTDepthModel  # type: ignore
        from dpt.transforms import Resize, NormalizeImage, PrepareForNet  # type: ignore

        dataset["invdepths"] = []
        net_w = net_h = 384
        basepath = os.path.dirname(os.path.dirname(os.path.abspath(dpt.__file__)))
        model_path = f"{basepath}/weights/dpt_large-midas-2f21e586.pt"
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
            enable_attention_hooks=False,
        )
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform = Compose([
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet()])
        model.eval()
        model = model.to(memory_format=torch.channels_last)
        model = model.half()
        model.cuda()

        # compute
        for img in tqdm.tqdm(dataset["images"], desc="Generating depths"):
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.
            img_input = transform({"image": img})["image"]
            sample = torch.from_numpy(img_input).cuda().unsqueeze(0)
            sample = sample.to(memory_format=torch.channels_last)  # type: ignore
            sample = sample.half()
            prediction = model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.float().unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .float()
                .cpu()
                .numpy()
            )
            dataset["invdepths"].append(prediction)

        # Clear memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return dataset
    else:
        raise ValueError(f"Unknown mode {mode}")


def compute_depth_params(dataset):
    cameras = dataset["cameras"]
    points3D = dataset["points3D_xyz"]
    if "invdepth_params" in dataset:
        return dataset
    dataset["invdepth_params"] = []
    for i in tqdm.tqdm(range(len(cameras)), desc="Computing depth params"):
        pose = cameras.poses[i]
        qvec = colmap_utils.rotmat2qvec(pose[:3, :3].T)
        tvec = -np.matmul(pose[:3, :3].T, pose[:3, 3]).reshape(3)
        xys = dataset["images_points2D_xy"][i]
        point3D_ids = dataset["images_points3D_indices"][i]
        img = Namespace(
            camera_id=0,
            name=f'image_{i}.png',
            qvec=qvec,
            tvec=tvec,
            xys=xys,
            point3D_ids=point3D_ids,
        )
        w, h = cameras.image_sizes[i]
        cam = Namespace(height=h, width=w)
        invdepth = dataset["invdepths"][i]
        params = make_depth_scale.get_scales(0, [cam], [img], points3D, None, invdepth, [img])
        dataset["invdepth_params"].append(params)
    # Fix bug in H3DGS
    med_scale = np.median([param["scale"] for param in dataset["invdepth_params"]])
    for param in dataset["invdepth_params"]:
        param["med_scale"] = med_scale
    return dataset


def split_dataset_into_chunks(dataset, config_overrides):
    argparser = make_chunk.get_argparser()
    arg_list = []
    supported_args = ["chunk_size", "min_padd", "lapla_thresh", "min_n_cams", "max_n_cams", "add_far_cams"]
    _config_overrides_to_args_list(arg_list, {
        k: v for k, v in config_overrides.items() if k in supported_args
    })
    args = argparser.parse_args(arg_list)

    cams = {}
    imgs = {}
    points3D = {}
    for i in range(len(dataset["cameras"])):
        cams[i+1] = Namespace(
            id=i+1,
            model="PINHOLE",
        )
        qvec = colmap_utils.rotmat2qvec(dataset["cameras"][i].poses[:3, :3].T)
        tvec = -np.matmul(dataset["cameras"][i].poses[:3, :3].T, dataset["cameras"][i].poses[:3, 3]).reshape(3)
        imgs[i+1] = Namespace(
            id=i+1,
            camera_id=i+1,
            qvec=qvec,
            tvec=tvec,
            name=os.path.relpath(dataset["image_paths"][i], dataset["image_paths_root"]),
            point3D_ids=dataset["images_points3D_indices"][i]+1,
        )
    rgbs = dataset.get("points3D_rgb")

    # We need to invert images_points3D_indices
    points3D_image_indices = [[] for _ in range(len(dataset["points3D_xyz"]))]
    for i, image_ids in enumerate(dataset["images_points3D_indices"]):
        for j in image_ids:
            points3D_image_indices[j].append(i)
    for i in range(len(dataset["points3D_xyz"])):
        points3D[i+1] = Namespace(
            id=i+1,
            xyz=dataset["points3D_xyz"][i],
            rgb=rgbs[i] if rgbs is not None else np.zeros(3, dtype=np.uint8),
            error=dataset["points3D_error"][i],
            image_ids=np.array(points3D_image_indices[i], dtype=np.int32)+1,
        )
    chunk_datasets = []
    for chunk in tqdm.tqdm(
            make_chunk.generate_chunks(cams, imgs, points3D, dataset["images"], args), desc="Generating chunks"):
        # Ignored chunk
        if not chunk: continue
        # We need to
        # 1) Update images_points3D_indices
        # 2) Remove other images
        # 3) Remove other points3D
        image_indices = np.array([x-1 for x in chunk["images"].keys()], dtype=np.int32)
        point_indices = np.array([x-1 for x in chunk["points3D"].keys()], dtype=np.int32)
        inverse_point_indices = {k: i for i, k in enumerate(point_indices)}
        chunk_dataset = dataset_index_select(dataset, image_indices)
        for i, indices_map in enumerate(chunk_dataset["images_points3D_indices"]):
            chunk_dataset["images_points3D_indices"][i] = \
                np.array([inverse_point_indices[k] for k in indices_map], dtype=np.int32)
        for k in chunk_dataset.keys():
            if k.startswith("points3D_"):
                chunk_dataset[k] = chunk_dataset[k][point_indices]
        chunk_datasets.append(chunk_dataset)
    return chunk_datasets


class SingleHierarchical3DGS:
    module = _train_single


    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.step = 0
        self._gaussians: Any = None
        self._scene: Any = None
        self._dataset: Any = None
        self._opt: Any = None
        self._pipe: Any = None
        self._background: Any = None
        self._args: Any = None
        self._training_generator: Any = None

        # Setup parameters
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            iters = []
            if os.path.exists(os.path.join(checkpoint, "point_cloud")):
                iters = sorted(
                    int(x[x.find("_") + 1:]) for x in os.listdir(os.path.join(str(checkpoint), "point_cloud")) if x.startswith("iteration_"))
            if iters:
                self._loaded_step = iters[-1]

        self.step = self._loaded_step
        self.config_overrides = config_overrides
        self._load_config()
        self.train_dataset = copy.copy(train_dataset)
        self._setup(train_dataset)

    def _load_config(self):
        args = self._get_args(self.config_overrides, self)
        if not args.hierarchy and self.checkpoint is not None:
            if os.path.exists(os.path.join(self.checkpoint, "hierarchy.hier_opt")):
                 args.hierarchy = os.path.join(self.checkpoint, "hierarchy.hier_opt")
            elif os.path.exists(os.path.join(self.checkpoint, "hierarchy.hier")):
                 args.hierarchy = os.path.join(self.checkpoint, "hierarchy.hier")
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
            scene = Scene(scene_info=scene_info, 
                         args=dataset, 
                         gaussians=gaussians, 
                         *args,
                         load_iteration=(
                             str(self._loaded_step) 
                             if self._loaded_step is not None and not self._args.hierarchy
                             else None), 
                          **kwargs)
            # Fix exposure_mapping not being loaded
            if getattr(gaussians, 'exposure_mapping', None) is None and self.checkpoint is not None:
                gaussians.exposure_mapping = {}
                exposure = []
                with open(os.path.join(self.checkpoint, "exposure.json"), "r") as f:
                    for i, (k, v) in enumerate(json.load(f).items()):
                        gaussians.exposure_mapping[k] = i
                        exposure.append(torch.tensor(v, dtype=torch.float32))
                if getattr(gaussians, '_exposure', None) is None:
                    gaussians._exposure = torch.stack(exposure)
            return scene
        oldstdout = sys.stdout
        training_setup = self.module.GaussianModel.training_setup
        try:
            # Disable training setup for inference
            if train_dataset is None:
                self.module.GaussianModel.training_setup = _noop
            with import_context:
                self.module.setup_train(self, self._args, build_scene)
        finally:
            sys.stdout = oldstdout
            self.module.GaussianModel.training_setup = training_setup

    def _next_viewpoint_cam(self):
        if getattr(self, '_train_cam_generator', None) is None:
            def iter_cameras():
                while True:
                    for viewpoint_batch in self._training_generator:
                        for viewpoint_cam in viewpoint_batch:
                            yield viewpoint_cam
            self._train_cam_generator = iter(iter_cameras())
        return next(self._train_cam_generator)

    @classmethod
    def _get_args(cls, config_overrides, store=None):
        args_list = ["--source_path", "<empty>"]
        _config_overrides_to_args_list(args_list, config_overrides)
        parser = cls.module.get_argparser(store or Namespace())
        parser.add_argument("--depth_mode", choices=["depth_anything_v2", "dpt", "none"], default="dpt")
        parser.set_defaults(
            exposure_lr_init=0.0, 
            resolution=1,
            eval=True,
            exposure_lr_final=0.0)
        args = parser.parse_args(args_list)
        args.depths = "<provided>" if args.depth_mode == "none" else None
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

    @torch.no_grad()
    def render(self, camera: Cameras, *, options=None):
        del options
        camera = camera.item()
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"
        viewpoint_cam = camera_to_minicam(camera, 'cuda')
        render_pkg = render(viewpoint_cam, self._gaussians, self._pipe, 
                            torch.zeros((3,), dtype=torch.float32, device='cuda'),
                            indices=None, use_trained_exp=False)
        return {
            "color": render_pkg["render"].clamp(0, 1).detach().permute(1, 2, 0),
            "depth": (1/render_pkg["depth"]).squeeze(0),
        }

    def train_iteration(self, step):
        assert self.train_dataset is not None, "train_dataset must be set"
        if step == 0:
            # Generate invdepths
            if self.train_dataset.get("invdepths") is None and getattr(self._args, "depth_mode", "none") != "none":
                generate_invdepths(self.train_dataset, mode=self._args.depth_mode)
            if self.train_dataset.get("invdepths") is not None:
                compute_depth_params(self.train_dataset)
        self.step = step
        viewpoint_cam = self._next_viewpoint_cam()
        metrics = self.module.train_iteration(self, viewpoint_cam, step+1)
        self.step = step+1
        return metrics

    def save(self, path: str):
        old_model_path = self._scene.model_path
        _empty = object()
        old_hierarchy = getattr(self._gaussians, "hierarchy_path", _empty)
        try:
            self._scene.model_path = path
            if old_hierarchy is not _empty:
                self._gaussians.hierarchy_path = os.path.join(path, "hierarchy.hier")
            self._scene.save(self.step)
        finally:
            self._scene.model_path = old_model_path
            if old_hierarchy is not _empty:
                self._gaussians.hierarchy_path = old_hierarchy

    def export_gaussian_splats(self, options=None):
        options = (options or {}).copy()
        return {
            "antialias_2D_kernel_size": 0.3,
            "means": self._gaussians.get_xyz.detach().cpu().numpy(),
            "scales": self._gaussians.get_scaling.detach().cpu().numpy(),
            "opacities": self._gaussians.get_opacity.detach().cpu().numpy(),
            "quaternions": self._gaussians.get_rotation.detach().cpu().numpy(),
            "spherical_harmonics": self._gaussians.get_features.transpose(1, 2).detach().cpu().numpy(),
        }



def write_colmap_dataset(path, dataset):
    cameras = dataset["cameras"]
    colmap_images = {}
    colmap_cameras = {}
    for i, cam in enumerate(cameras):
        width, height = cam.image_sizes
        fx, fy, cx, cy = cam.intrinsics
        params = [fx, fy, cx, cy]
        colmap_cameras[i+1] = colmap_utils.Camera(i+1, "PINHOLE", width, height, params)
        name = os.path.relpath(dataset["image_paths"][i], dataset["image_paths_root"])
        image_id = i+1
        R = cam.poses[:3, :3].T
        qvec = colmap_utils.rotmat2qvec(R)
        tvec = -np.matmul(R, cam.poses[:3, 3]).reshape(3)
        point3D_ids = np.array([], dtype=np.int64)
        xys = np.zeros((0, 2), dtype=np.float32)
        images_points3D_indices = dataset.get("images_points3D_indices")
        if images_points3D_indices is not None:
            point3D_ids = images_points3D_indices[i] + 1
            xys = np.zeros((len(point3D_ids), 2), dtype=np.float32)
        colmap_images[image_id] = colmap_utils.Image(image_id, qvec, tvec, i+1, name, xys, point3D_ids)

    os.makedirs(os.path.join(path, "sparse", "0"), exist_ok=True)
    colmap_utils.write_cameras_binary(colmap_cameras, os.path.join(path, "sparse", "0", "cameras.bin"))
    colmap_utils.write_images_binary(colmap_images, os.path.join(path, "sparse", "0", "images.bin"))


class CoarseHierarchical3DGS(SingleHierarchical3DGS):
    module = _train_coarse

    @classmethod
    def _get_args(cls, config_overrides, store=None):
        args_list = ["--source_path", "<empty>", "--resolution", "1", "--eval"]
        _config_overrides_to_args_list(args_list, config_overrides)
        parser = cls.module.get_argparser(store or Namespace())

        parser.set_defaults(skybox_num=100000,
                            position_lr_init=0.00016, position_lr_final=0.0000016, 
                            exposure_lr_init=0.0, exposure_lr_final=0.0)
        args = parser.parse_args(args_list)
        return args



class PostHierarchical3DGS(SingleHierarchical3DGS):
    module = _train_post
    _render_indices = None
    _parent_indices = None
    _nodes_for_render_indices = None
    _interpolation_weights = None
    _num_siblings = None

    def __init__(self, *,
                 checkpoint,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        # If there is no hierarchy, we need to convert the checkpoint to a hierarchy
        assert checkpoint is not None, "Checkpoint must be provided"
        if (not os.path.exists(os.path.join(checkpoint, "hierarchy.hier_opt")) and
            not os.path.exists(os.path.join(checkpoint, "hierarchy.hier"))):
            self.generate_hierarchy(checkpoint, train_dataset)
        super().__init__(
            checkpoint=checkpoint, 
            train_dataset=train_dataset, 
            config_overrides=config_overrides)

    @classmethod
    def _get_args(cls, config_overrides, store=None):
        args_list = ["--source_path", "<empty>", "--resolution", "1", "--eval"]
        _config_overrides_to_args_list(args_list, config_overrides)
        parser = cls.module.get_argparser(store or Namespace())
        parser.add_argument("--tau", type=float, default=6.0)
        parser.set_defaults(iterations=15000, feature_lr=0.0005, 
                            opacity_lr=0.01, scaling_lr=0.001, 
                            exposure_lr_init=0.0, exposure_lr_final=0.0)
        args = parser.parse_args(args_list)
        return args

    def _prepare_buffers(self):
        if self._render_indices is None or self._render_indices.size(0) != self._gaussians._xyz.size(0):
            self._render_indices = torch.zeros(self._gaussians._xyz.size(0)).int().cuda()
            self._parent_indices = torch.zeros(self._gaussians._xyz.size(0)).int().cuda()
            self._nodes_for_render_indices = torch.zeros(self._gaussians._xyz.size(0)).int().cuda()
            self._interpolation_weights = torch.zeros(self._gaussians._xyz.size(0)).float().cuda()
            self._num_siblings = torch.zeros(self._gaussians._xyz.size(0)).int().cuda()

    @torch.no_grad()
    def render(self, camera: Cameras, *, options=None):
        self._prepare_buffers()
        # Pyright assertions
        assert self._render_indices is not None, "Buffers not prepared"
        assert self._parent_indices is not None, "Buffers not prepared"
        assert self._nodes_for_render_indices is not None, "Buffers not prepared"
        assert self._interpolation_weights is not None, "Buffers not prepared"
        assert self._num_siblings is not None, "Buffers not prepared"

        camera = camera.item()
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"
        options = options or {}
        tau = options.get("tau", self._args.tau)
        viewpoint_cam = camera_to_minicam(camera, 'cuda')
        tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
        threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * viewpoint_cam.image_width)
        to_render = expand_to_size(
            self._gaussians.nodes,
            self._gaussians.boxes,
            threshold,
            viewpoint_cam.camera_center,
            torch.zeros((3)),
            self._render_indices,
            self._parent_indices,
            self._nodes_for_render_indices)
        indices = self._render_indices[:to_render].int().contiguous()
        node_indices = self._nodes_for_render_indices[:to_render].contiguous()
        get_interpolation_weights(
            node_indices,
            threshold,
            self._gaussians.nodes,
            self._gaussians.boxes,
            viewpoint_cam.camera_center.cpu(),
            torch.zeros((3)),
            self._interpolation_weights,
            self._num_siblings
        )
        render_pkg = render_post(
            viewpoint_cam, 
            self._gaussians, 
            self._args, 
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), 
            render_indices=indices,
            parent_indices = self._parent_indices,
            interpolation_weights = self._interpolation_weights,
            num_node_kids = self._num_siblings, 
            use_trained_exp=self._args.train_test_exp)
        return {
            "color": render_pkg["render"].clamp(0, 1).detach().permute(1, 2, 0),
        }

    def generate_hierarchy(self, checkpoint, train_dataset):
        assert train_dataset is not None, "train_dataset must be provided to generate hierarchy"
        loaded_step = max(int(x[x.find("_") + 1:]) for x in os.listdir(os.path.join(str(checkpoint), "point_cloud")) if x.startswith("iteration_"))
        point_cloud = os.path.join(checkpoint, "point_cloud", f"iteration_{loaded_step}", "point_cloud.ply")
        with tempfile.TemporaryDirectory() as tmpdir:
            write_colmap_dataset(tmpdir, train_dataset)
            subprocess.check_call([
                "GaussianHierarchyCreator",
                point_cloud,
                tmpdir,
                checkpoint])

    def export_gaussian_splats(self, options=None):
        options = (options or {}).copy()
        # We select all leafs for the demo
        indices = self._gaussians.nodes[:, -1] == 0
        return {
            "antialias_2D_kernel_size": 0.3,
            "means": self._gaussians.get_xyz[indices].detach().cpu().numpy(),
            "scales": self._gaussians.get_scaling[indices].detach().cpu().numpy(),
            "opacities": self._gaussians.get_opacity[indices].detach().cpu().numpy(),
            "quaternions": self._gaussians.get_rotation[indices].detach().cpu().numpy(),
            "spherical_harmonics": self._gaussians.get_features[indices].transpose(1, 2).detach().cpu().numpy(),
        }



class Hierarchical3DGS(Method):
    tempdir = None

    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self._config_overrides = config_overrides or {}
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            self._config_overrides = self._config_overrides.copy()

            with open(os.path.join(checkpoint, "h3dgs-info.json"), "r") as f:
                meta = json.load(f)

            self._config_overrides = meta["config_overrides"] or {}
            self._config_overrides.update(config_overrides or {})
            self._loaded_step = meta["iteration"]

        config_overrides_ = self._config_overrides.copy()
        mode = config_overrides_.pop("mode", "single")
        assert mode in ("per-chunk", "single"), f"Unknown mode {mode}"
        configs_per_stage = {
            name: {
                k.split(".")[-1]: v for k, v in (config_overrides_ or {}).items()
                if k.startswith(f"{name}.") or "." not in k}
                for name in ("single", "post", "coarse", "generate_chunks")
        }
            
        if mode == "single":
            if "skip_scale_big_gauss" not in configs_per_stage["single"]:
                configs_per_stage["single"]["skip_scale_big_gauss"] = True
            stages = [
                ('single', SingleHierarchical3DGS, configs_per_stage['single'], train_dataset),
                ('post', PostHierarchical3DGS, configs_per_stage['post'], train_dataset),
            ]
        elif mode == "per-chunk":
            self.tempdir = tempfile.TemporaryDirectory()
            train_datasets = split_dataset_into_chunks(train_dataset, configs_per_stage["generate_chunks"])
            stages = []
            stages.append(('coarse', CoarseHierarchical3DGS, configs_per_stage['coarse'], train_dataset))
            # Now, we process all chunks
            for i, dataset in enumerate(train_datasets):
                stages.append((f'single-{i}', SingleHierarchical3DGS, configs_per_stage['single'], dataset))
                stages.append((f'post-{i}', PostHierarchical3DGS, configs_per_stage['post'], dataset))
        else:
            raise ValueError(f"Unknown mode {mode}")

        self.stages = stages
        self._stages_cumsum = np.cumsum(
            [0] + [x[1].get_hparams(x[2])["iterations"] for x in self.stages])
        self.train_dataset = train_dataset
        self.checkpoint = checkpoint
        current_stage_idx = self._get_current_stage_idx(self._loaded_step or 0)
        _, cls, co, dataset = self.stages[current_stage_idx]
        self.current_stage = cls(
            checkpoint=checkpoint, 
            train_dataset=dataset, 
            config_overrides=co)

    def _get_current_stage_idx(self, step):
        out = min(np.searchsorted(self._stages_cumsum, step, side="right")-1, len(self.stages)-1)
        self._current_stage_idx = out
        return out

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",
            required_features=frozenset((
                "color", "points3D_xyz",
                "images_points3D_indices", "images_points2D_xy",
                "points3D_error", "points3D_rgb",
            )),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color",),
            can_resume_training=False,
        )

    def get_info(self) -> ModelInfo:
        hparams = {}
        for name, cls, co, *_ in self.stages:
            hparams.update({f"{name}.{k}": v for k, v in cls.get_hparams(co).items()})
        return ModelInfo(
            num_iterations=int(self._stages_cumsum[-1]),
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
        out = self.current_stage.render(camera, options=options)
        return self._format_output(out, options)

    def train_iteration(self, step):
        if self.checkpoint is not None:
            raise RuntimeError(f"Method {_method_id} was loaded from checkpoint and training cannot be resumed.")
        current_stage = self._get_current_stage_idx(step)
        stage_change = step > 0 and self._get_current_stage_idx(step-1) != current_stage
        step_offset = self._stages_cumsum[current_stage]
        if stage_change:
            fromstage = self.stages[current_stage-1][0]
            tostage = self.stages[current_stage][0]
            logging.info("Switching stage {} -> {}".format(
                fromstage, tostage))
            with contextlib.ExitStack() as stack:
                if self.tempdir is not None:
                    tempdir = self.tempdir.name
                else:
                    tempdir = stack.enter_context(tempfile.TemporaryDirectory())
                self.current_stage.save(os.path.join(tempdir, fromstage))
                del self.current_stage
                gc.collect()
                torch.cuda.empty_cache()
                _, cls, co, dataset = self.stages[current_stage]
                checkpoint = None
                if fromstage.startswith("single") and tostage.startswith("post"):
                    checkpoint = os.path.join(tempdir, fromstage)
                self.current_stage = cls(
                    checkpoint=checkpoint,
                    train_dataset=dataset, 
                    config_overrides=co)
        return self.current_stage.train_iteration(step - step_offset)

    def save(self, path: str):
        self.current_stage.save(path)
        step_offset = self._stages_cumsum[self._current_stage_idx]
        with open(os.path.join(path, "h3dgs-info.json"), "w") as f:
            json.dump({
                "config_overrides": self._config_overrides,
                "iteration": int(step_offset + self.current_stage.step),
            }, f)

        # Add missing exposure saving for post stage
        scene = self.current_stage._scene
        if not os.path.exists(os.path.join(path, "exposure.json")):
            exposure_dict = {
                image_name: scene.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
                for image_name in scene.gaussians.exposure_mapping
            }
            with open(os.path.join(path, "exposure.json"), "w") as f:
                json.dump(exposure_dict, f, indent=2)

    def export_gaussian_splats(self, options=None):
        return self.current_stage.export_gaussian_splats(options=options)
