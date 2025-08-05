"""
We patched 3DGRUT with support for OPENCV and OPENCV_FULL camera models.
"""
import shutil
import ast
import sys
import os
import contextlib
from typing import Optional, Dict
from nerfbaselines import Method, Dataset, MethodInfo, ModelInfo, RenderOutput, Cameras
from nerfbaselines.datasets import _colmap_utils as colmap_utils
from nerfbaselines.datasets import colmap as colmap_dataset
from nerfbaselines.utils import convert_image_dtype, image_to_srgb
from nerfbaselines import cameras as nb_cameras
from unittest.mock import patch
from types import SimpleNamespace
from PIL import Image
import numpy as np

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, DictConfig

from .threedgrut_patch import import_context
with import_context:
    from threedgrut.datasets import dataset_colmap  # type: ignore
    from threedgrut.utils.timer import CudaTimer  # type: ignore
    import threedgrut


OmegaConf.register_new_resolver("int_list", lambda l: [int(x) for x in l], replace=True)


def _get_train_iteration(Trainer):
    module = sys.modules[Trainer.__module__]
    source_path = os.path.abspath(module.__file__ or "")
    with open(source_path, 'r') as f:
        ast_tree = ast.parse(f.read(), filename=source_path)
    Trainer_ast = next((x for x in ast_tree.body if isinstance(x, ast.ClassDef) and x.name == Trainer.__name__), None)
    assert Trainer_ast is not None, f"Class {Trainer.__name__} not found in {source_path}"
    run_train_pass_ast = next((x for x in Trainer_ast.body if isinstance(x, ast.FunctionDef) and x.name == "run_train_pass"), None)
    assert run_train_pass_ast is not None, f"Method run_train_pass not found in {Trainer.__name__} class"
    for_ast = run_train_pass_ast.body[-2]
    assert isinstance(for_ast, ast.For), "Expected a for loop in the run_train_pass method"
    train_iteration_body = for_ast.body[:-6]
    train_iteration_body.append(ast.Return(value=ast.Name(id='batch_metrics', ctx=ast.Load())))
    train_iteration_ast = ast.FunctionDef(
        name="train_iteration",
        args=ast.arguments(
            args=[
                ast.arg(arg='self', annotation=None), 
                ast.arg(arg='conf', annotation=None),
                ast.arg(arg='global_step', annotation=None),
                ast.arg(arg='model', annotation=None),
                ast.arg(arg='metrics', annotation=None),
                ast.arg(arg='profilers', annotation=None),
                ast.arg(arg='iter', annotation=None),
                ast.arg(arg='batch', annotation=None)
            ], vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[], posonlyargs=[]),
        body=train_iteration_body,
        decorator_list=[],
        returns=None
    )
    train_iteration_ast = ast.fix_missing_locations(train_iteration_ast)
    _globals = vars(module).copy()
    exec(compile(ast.fix_missing_locations(ast.Module(
        body=[train_iteration_ast],
        type_ignores=[]
    )), filename=source_path, mode='exec'), _globals)
    return _globals["train_iteration"]


def _get_gpu_batch(camera, device):
    with patch.object(dataset_colmap.logger, "track", lambda x, *args, **kwargs: x):
        dataset = _Dataset({
            "cameras": camera.item()[None],
            "image_paths": ["/dummy/0.png"],
            "image_paths_root": "/dummy",
            "images": [np.zeros((1, 1, 3), dtype=np.uint8)],
        }, device=device)
        batch = dataset[0]
        batch["data"] = batch["data"].unsqueeze(0)
        batch["pose"] = batch["pose"].unsqueeze(0)
        batch["intr"] = torch.tensor([batch["intr"]], dtype=torch.int64)
        out = dataset.get_gpu_batch_with_intrinsics(batch)
    return out


class _Dataset(dataset_colmap.ColmapDataset):
    def __init__(self, dataset, bg_color=None, **kwargs):
        self._dataset = dataset
        self._inverse_map = {self._dataset["image_paths"][i]: i for i in range(len(self._dataset["image_paths"]))}
        self._inverse_masks_map = {}
        if dataset.get("masks"):
            self._inverse_masks_map = {
                "/masks/" + self._dataset["image_paths"][i]: i
                for i in range(len(self._dataset["image_paths"]))
            }
        path = dataset["image_paths_root"]
        self.bg_color = bg_color
        super().__init__(path, split="train", downsample_factor=1, test_split_interval=1, **kwargs)

    def _to_colmap_intrinsics(self, i):
        return colmap_dataset.camera_to_colmap_camera(self._dataset["cameras"][i], i+1)

    def get_images_folder(self):
        return "~~~~~"

    def _to_colmap_extrinsics(self, i):
        # c2w was obtained as follows...
        c2w = self._dataset["cameras"][i].poses
        # Now we need to invert it to obtain both t and R
        R = c2w[:3, :3].T
        t = -np.matmul(R, c2w[:3, 3])

        return SimpleNamespace(
            id=i+1,
            camera_id=i+1,
            tvec=t,
            qvec=colmap_utils.rotmat2qvec(R.astype(np.float64)),
            name=self._dataset["image_paths"][i])

    def reload(self):
        @contextlib.contextmanager
        def _open_image(path):
            i = self._inverse_map[path]
            yield SimpleNamespace(size=(self._dataset["cameras"].image_sizes[i]))
        with patch.object(dataset_colmap, "Image", SimpleNamespace(open=_open_image)):
            self.intrinsics = {}
            self.cam_intrinsics = {(i+1): self._to_colmap_intrinsics(i) for i in range(len(self._dataset["cameras"]))}
            self.cam_extrinsics = [self._to_colmap_extrinsics(i) for i in range(len(self._dataset["cameras"]))]
            self.n_frames = len(self.cam_extrinsics)
            self.load_camera_data()
            if self._dataset.get("masks"):
                self.mask_paths = np.stack(["/masks/" + path for path in self._dataset["image_paths"]], dtype=str)
            self.center, self.length_scale, self.scene_bbox = self.compute_spatial_extents()
            self._worker_gpu_cache.clear()

    def _get_bg_color(self):
        if self.bg_color is None:
            return None
        elif self.bg_color == "black":
            return np.zeros((3,), dtype=np.float32)
        elif self.bg_color == "white":
            return np.ones((3,), dtype=np.float32)
        else:
            raise ValueError(f"Unknown background color: {self.bg_color}")


    def __getitem__(self, idx):
        def _open_image(path):
            i = self._inverse_map.get(path)
            if i is not None:
                image = self._dataset["images"][i]
                image = image_to_srgb(image, dtype=np.uint8, allow_alpha=False, background_color=self._get_bg_color())
                return Image.fromarray(image)
            i = self._inverse_masks_map.get(path)
            if i is not None:
                image = self._dataset["masks"][i]
                return Image.fromarray(image)
            raise ValueError(f"Image path {path} not found in dataset.")

        with patch.object(dataset_colmap, "Image", SimpleNamespace(open=_open_image)):
            return super().__getitem__(idx)


class ThreeDGRUT(Method):
    _default_config = "apps/colmap_3dgut.yaml"

    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = checkpoint
        self.train_dataset = train_dataset
        config_overrides = (config_overrides or {}).copy()

        self._loaded_step = None

        with import_context:
            from threedgrut.trainer import Trainer3DGRUT  # type: ignore
        self.trainer = None
        self._train_iteration = _get_train_iteration(Trainer3DGRUT)
        self._train_iterator = None
        self._metrics = None
        self._profilers = None

        def _init_from_colmap(model, _, observer_pts):
            assert train_dataset is not None, "train_dataset must be provided"
            points3D_xyz = train_dataset.get("points3D_xyz")
            assert points3D_xyz is not None, "points3D_xyz must be provided in the train_dataset"
            file_pts = points3D_xyz.astype(np.float32)
            file_rgb = train_dataset.get("points3D_rgb")
            if file_rgb is None:
                file_rgb = np.ones_like(file_pts, dtype=np.uint8) * 255
            else:
                file_rgb = convert_image_dtype(file_rgb, np.uint8)
            file_pts = torch.tensor(file_pts, dtype=torch.float32, device=model.device)
            file_rgb = torch.tensor(file_rgb, dtype=torch.uint8, device=model.device)
            model.default_initialize_from_points(file_pts, observer_pts, file_rgb, 
                                                 use_observer_pts=model.conf.initialization.use_observation_points)

        with patch("threedgrut.datasets.make", 
               lambda name, config, ray_jitter: (
                   _Dataset(train_dataset, ray_jitter=ray_jitter, bg_color=config.model.background.color),
                   _Dataset(train_dataset, ray_jitter=ray_jitter, bg_color=config.model.background.color))), \
               patch("threedgrut.model.model.MixtureOfGaussians.init_from_colmap", _init_from_colmap), \
               patch.object(Trainer3DGRUT, "init_experiments_tracking", lambda *args, **kwargs: None):
            if self.checkpoint is None:
                configs_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(threedgrut.__file__))), "configs")
                with initialize_config_dir(version_base=None, config_dir=configs_path):
                    config_name = config_overrides.pop("config", self._default_config)
                    self.config = compose(config_name=config_name, overrides=(
                        ["path=", "enable_writer=false"] + [f"{k}={v}" for k, v in (config_overrides or {}).items()]))
                self.trainer = Trainer3DGRUT(self.config)
                self.model = self.trainer.model
            else:
                # Load from checkpoint
                config = OmegaConf.load(os.path.join(self.checkpoint, "config.yaml"))
                config_overrides.pop("config", None)
                config = OmegaConf.merge(config, OmegaConf.create(config_overrides))
                self.config = DictConfig(config)
                if train_dataset is not None:
                    # Setup for training
                    self.trainer = Trainer3DGRUT.create_from_checkpoint(
                        os.path.join(self.checkpoint, "ckpt_last.pt"), self.config)
                    self._loaded_step = self.trainer.global_step
                    self.model = self.trainer.model
                else:
                    self._loaded_step, self.model = self._load_model_without_dataset(
                        os.path.join(self.checkpoint, "ckpt_last.pt"), self.config)
            if self.trainer is not None:
                self.trainer.tracking = SimpleNamespace(
                    output_dir=None,
                    run_name="experiment",
                    object_name="experiment",
                    writer=None)

    def _load_model_without_dataset(self, checkpoint_path, conf):
        with import_context:
            from threedgrut.model.model import MixtureOfGaussians  # type: ignore
        checkpoint = torch.load(checkpoint_path)
        global_step = checkpoint["global_step"]
        # overrides
        if conf["render"]["method"] == "3dgrt":
            conf["render"]["particle_kernel_density_clamping"] = True
            conf["render"]["min_transmittance"] = 0.03
        # Initialize the model and the optix context
        model = MixtureOfGaussians(conf)
        # Initialize the parameters from checkpoint
        model.init_from_checkpoint(checkpoint)
        model.build_acc()
        return global_step, model

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",
            required_features=frozenset(("color", "points3D_xyz", "points3D_rgb")),
            supported_camera_models=frozenset(("pinhole", "opencv_fisheye", "opencv", "full_opencv")),
            supported_outputs=("color", "distance", "accumulation", "normal", "depth"),
            can_resume_training=True,
            viewer_default_resolution=768,
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            num_iterations=self.config.n_iterations,
            loaded_step=self._loaded_step,  # TODO:
            loaded_checkpoint=self.checkpoint,
            hparams={},  # TODO
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
        model = self.model
        # Get the GPU-cached batch
        gpu_batch = _get_gpu_batch(camera, model.device)
        # Compute the outputs of a single batch
        outputs = model(gpu_batch)
        out = {
            "color": outputs["pred_rgb"].squeeze(0),
            "distance": outputs["pred_dist"].squeeze((0, -1)),
            "normal": outputs["pred_rgb"].squeeze(0),
            "accumulation": outputs["pred_opacity"].squeeze((0, -1)),
        }
        if "depth" in (options or {}).get("outputs", []):
            distance = outputs["pred_dist"].squeeze((0, -1))
            # Map distance to depth
            torch_camera = camera[None].apply(lambda x, _: torch.from_numpy(x).squeeze(0).to(model.device))
            xy = nb_cameras.get_image_pixels(torch_camera.image_sizes)
            _, ray_dirs = nb_cameras.get_rays(torch_camera, xy)
            dist_scaling = torch.linalg.norm(ray_dirs, dim=-1)
            out["depth"] = dist_scaling.view(distance.shape) * distance
        return self._format_output(out, options)

    def _next_train_iteration(self):
        assert self.trainer is not None, "Method is not initialized for training"
        def _reset():
            self._metrics = []
            self._profilers = {
                "inference": CudaTimer(enabled=self.config.enable_frame_timings),
                "backward": CudaTimer(enabled=self.config.enable_frame_timings),
                "build_as": CudaTimer(enabled=self.config.enable_frame_timings),
            }
            # Pyright issue
            assert self.trainer is not None, "Method is not initialized for training"
            self._train_iterator = iter(enumerate(self.trainer.train_dataloader))
        if self._train_iterator is None:
            _reset()
        # Pyright issue
        assert self._train_iterator is not None, "Train iterator is not initialized"
        try:
            return next(self._train_iterator)
        except StopIteration:
            _reset()
            return next(self._train_iterator)

    def train_iteration(self, step: int) -> Dict[str, float]:
        assert self.trainer is not None, "Method is not initialized for training"
        self.trainer.global_step = step
        epoch_iter, batch = self._next_train_iteration()
        batch_metrics = self._train_iteration(
            self.trainer, 
            self.config, 
            self.trainer.global_step, 
            self.trainer.model, 
            [], 
            self._profilers, 
            epoch_iter, 
            batch,
        )
        out = batch_metrics.get("losses", {})
        out["loss"] = out.get("total_loss", 0.0)
        return out

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        if self.trainer is not None:
            self.trainer.tracking.output_dir = path
            self.trainer.save_checkpoint(last_checkpoint=True)
            os.rmdir(os.path.join(path, f"ours_{self.trainer.global_step}"))
        else:
            assert self.checkpoint is not None, "Checkpoint is not set for saving"
            # We just copy the checkpoint to the target directory
            shutil.copyfile(
                os.path.join(self.checkpoint, "ckpt_last.pt"),
                os.path.join(path, "ckpt_last.pt")
            )
        # Save config
        config_path = os.path.join(path, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(self.config, f)


class ThreeDGUT(ThreeDGRUT):
    _default_config = "apps/colmap_3dgut.yaml"


class ThreeDGRT(ThreeDGRUT):
    _default_config = "apps/colmap_3dgrt.yaml"

