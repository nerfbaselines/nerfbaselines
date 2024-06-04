import argparse
import importlib.util
import logging
import os
import pprint
import sys
from typing import List, Dict, Any
import numpy as np
import torch
import torch.utils.data
from plenoxels.main import init_trainer
from plenoxels.runners import video_trainer
from plenoxels.runners import phototourism_trainer
from plenoxels.runners import static_trainer
from plenoxels.utils.create_rendering import render_to_path, decompose_space_time
from plenoxels.utils.parse_args import parse_optfloat
from nerfbaselines.types import Method
from nerfbaselines.utils import remap_error


def load_data(model_type: str, data_downsample, data_dirs, validate_only: bool, render_only: bool, **kwargs):
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)

    if model_type == "video":
        return video_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs)
    elif model_type == "phototourism":
        return phototourism_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs
        )
    else:
        return static_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs)


def save_config(config):
    log_dir = os.path.join(config['logdir'], config['expname'])
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, 'config.py'), 'wt') as out:
        out.write('config = ' + pprint.pformat(config))

    with open(os.path.join(log_dir, 'config.csv'), 'w') as f:
        for key in config.keys():
            f.write("%s\t%s\n" % (key, config[key]))


def main():
    p = argparse.ArgumentParser(description="")

    p.add_argument('--render-only', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--spacetime-only', action='store_true')
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('override', nargs=argparse.REMAINDER)

    args = p.parse_args()

    # Import config
    spec = importlib.util.spec_from_file_location(os.path.basename(args.config_path), args.config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    config: Dict[str, Any] = cfg.config
    # Process overrides from argparse into config
    # overrides can be passed from the command line as key=value pairs. E.g.
    # python plenoxels/main.py --config-path plenoxels/config/cfg.py max_ts_frames=200
    # note that all values are strings, so code should assume incorrect data-types for anything
    # that's derived from config - and should not a string.
    overrides: List[str] = args.override
    overrides_dict = {ovr.split("=")[0]: ovr.split("=")[1] for ovr in overrides}
    config.update(overrides_dict)
    if "keyframes" in config:
        model_type = "video"
    elif "appearance_embedding_dim" in config:
        model_type = "phototourism"
    else:
        model_type = "static"
    validate_only = args.validate_only
    render_only = args.render_only
    spacetime_only = args.spacetime_only
    if validate_only and render_only:
        raise ValueError("render_only and validate_only are mutually exclusive.")
    if render_only and spacetime_only:
        raise ValueError("render_only and spacetime_only are mutually exclusive.")
    if validate_only and spacetime_only:
        raise ValueError("validate_only and spacetime_only are mutually exclusive.")

    pprint.pprint(config)
    if validate_only or render_only:
        assert args.log_dir is not None and os.path.isdir(args.log_dir)
    else:
        save_config(config)

    data = load_data(model_type, validate_only=validate_only, render_only=render_only or spacetime_only, **config)
    trainer = init_trainer(model_type, **config, **data)
    if args.log_dir is not None:
        checkpoint_path = os.path.join(args.log_dir, "model.pth")
        training_needed = not (validate_only or render_only or spacetime_only)
        trainer.load_model(torch.load(checkpoint_path), training_needed=training_needed)

    if validate_only:
        trainer.validate()
    elif render_only:
        render_to_path(trainer, extra_name="")
    elif spacetime_only:
        decompose_space_time(trainer, extra_name="")
    else:
        trainer.train()


class KPlanes(Method):
    _method_name: str = "kplanes"

    @remap_error
    def __init__(self, 
                 *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.gaussians = None
        self.background = None
        self.step = 0

        self.scene = None

        # Setup parameters
        self._args_list = ["--source_path", "<empty>"]
        if checkpoint is not None:
            with open(os.path.join(checkpoint, "args.txt"), "r", encoding="utf8") as f:
                self._args_list = shlex.split(f.read())

        self._viewpoint_stack = []
        self._input_points = None

        # Setup config
        if self.checkpoint is None:
            if train_dataset["metadata"].get("name") == "blender":
                # Blender scenes have white background
                self._args_list.append("--white_background")
                logging.info("overriding default background color to white for blender dataset")

        if config_overrides is not None:
            for k, v in config_overrides.items():
                if f'--{k}' in self._args_list:
                    self._args_list[self._args_list.index(f'--{k}') + 1] = str(v)
                else:
                    self._args_list.append(f"--{k}")
                    self._args_list.append(str(v))

        self._load_config()

        if self.checkpoint is None:
            # Verify parameters are set correctly
            if train_dataset["metadata"].get("name") == "blender":
                assert self.dataset.white_background, "white_background should be True for blender dataset"

        if train_dataset is not None:
            self._setup_train(train_dataset)
        else:
            self._setup_eval()

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

    def _setup_train(self, train_dataset: Dataset):
        # Set random seed
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Get config path for the dataset
        config_path = ...

        # Import config
        spec = importlib.util.spec_from_file_location(os.path.basename(config_path), config_path)
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        config: Dict[str, Any] = cfg.config
        # Process overrides from argparse into config
        # overrides can be passed from the command line as key=value pairs. E.g.
        # python plenoxels/main.py --config-path plenoxels/config/cfg.py max_ts_frames=200
        # note that all values are strings, so code should assume incorrect data-types for anything
        # that's derived from config - and should not a string.
        overrides: List[str] = args.override
        overrides_dict = {ovr.split("=")[0]: ovr.split("=")[1] for ovr in overrides}
        config.update(overrides_dict)
        if "keyframes" in config:
            model_type = "video"
        elif "appearance_embedding_dim" in config:
            model_type = "phototourism"
        else:
            model_type = "static"

        pprint.pprint(config)
        if validate_only or render_only:
            assert args.log_dir is not None and os.path.isdir(args.log_dir)
        else:
            save_config(config)

        data = load_data(model_type, validate_only=validate_only, render_only=render_only or spacetime_only, **config)
        trainer = init_trainer(model_type, **config, **data)
        if args.log_dir is not None:
            checkpoint_path = os.path.join(args.log_dir, "model.pth")
            training_needed = not (validate_only or render_only or spacetime_only)
            trainer.load_model(torch.load(checkpoint_path), training_needed=training_needed)

        if validate_only:
            trainer.validate()
        elif render_only:
            render_to_path(trainer, extra_name="")
        elif spacetime_only:
            decompose_space_time(trainer, extra_name="")
        else:
            # trainer.train()







        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._input_points = (train_dataset["points3D_xyz"], train_dataset["points3D_rgb"])
        self._viewpoint_stack = []

    def train_iteration(self, step):
        if self.batch_iter is None:
            self.trainer.pre_epoch()
            self.batch_iter = iter(self.trainer.train_data_loader)
        self.trainer.timer.reset()
        self.trainer.model.step_before_iter(step)
        self.trainer.global_step += 1
        self.trainer.timer.check("step-before-iter")
        try:
            data = next(self.batch_iter)
            self.trainer.timer.check("dloader-next")
        except StopIteration:
            self.trainer.pre_epoch()
            self.batch_iter = iter(self.trainer.train_data_loader)
            data = next(self.batch_iter)
            logging.info("Reset data-iterator")

        try:
            step_successful = self.trainer.train_step(data)
        except StopIteration:
            self.trainer.pre_epoch()
            self.batch_iter = iter(self.trainer.train_data_loader)
            logging.info("Reset data-iterator")
            step_successful = True

        if step_successful and self.scheduler is not None:
            self.trainer.scheduler.step()
        for r in self.trainer.regularizers:
            r.step(self.global_step)
        self.trainer.model.step_after_iter(self.global_step)
        self.trainer.timer.check("after-step")

    def _setup_eval(self):
        # Initialize system state (RNG)
        safe_state(False)

        # Setup model
        self.gaussians = GaussianModel(self.dataset.sh_degree)
        self.scene = self._build_scene(None)
        info = self.get_info()
        loaded_step = info["loaded_step"]
        (model_params, self.step) = torch.load(str(self.checkpoint) + f"/chkpnt-{loaded_step}.pth")
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

    @classmethod
    def get_method_info(cls):
        assert cls._method_name is not None, "Method was not properly registered"
        return MethodInfo(
            name=cls._method_name,
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(("pinhole",)),
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            num_iterations=self.opt.iterations,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=(
                flatten_hparams(dict(itertools.chain(vars(self.dataset).items(), vars(self.opt).items(), vars(self.pipe).items()))) 
                if self.dataset is not None else {}),
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
                    return _convert_dataset_to_gaussian_splatting(dataset, td, white_background=self.dataset.white_background, scale_coords=self.dataset.scale_coords)
                sceneLoadTypeCallbacks["Colmap"] = colmap_loader
                scene =  Scene(opt, self.gaussians, load_iteration=info["loaded_step"] if dataset is None else None)
                # NOTE: This is a hack to match the RNG state of GS on 360 scenes
                _tmp = list(range((len(next(iter(scene.train_cameras.values()))) + 6) // 7))
                random.shuffle(_tmp)
                return scene
            finally:
                sceneLoadTypeCallbacks["Colmap"] = backup

    def render(self, cameras: Cameras, embeddings=None) -> Iterable[RenderOutput]:
        if embeddings is not None:
            raise NotImplementedError(f"Optimizing embeddings is not supported for method {self.get_method_info()['name']}")
        assert np.all(cameras.camera_types == camera_model_to_int("pinhole")), "Only pinhole cameras supported"
        sizes = cameras.image_sizes
        poses = cameras.poses
        intrinsics = cameras.intrinsics

        with torch.no_grad():
            for i, pose in enumerate(poses):
                viewpoint_cam = _load_caminfo(i, pose, intrinsics[i], f"{i:06d}.png", sizes[i], scale_coords=self.dataset.scale_coords)
                viewpoint = loadCam(self.dataset, i, viewpoint_cam, 1.0)
                image = torch.clamp(render(viewpoint, self.gaussians, self.pipe, self.background)["render"], 0.0, 1.0)
                color = image.detach().permute(1, 2, 0).cpu().numpy()

                yield {
                    "color": color,
                }

    def train_iteration(self, step):
        self.step = step
        iteration = step + 1  # Gaussian Splatting is 1-indexed
        del step

        self.gaussians.update_learning_rate(iteration)

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

        render_pkg = render(viewpoint_cam, self.gaussians, self.pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        sampling_mask = viewpoint_cam.sampling_mask.cuda() if viewpoint_cam.sampling_mask is not None else None

        # Apply mask
        mask_percentage = 1.0
        if sampling_mask is not None:
            image *= sampling_mask
            gt_image *= sampling_mask
            mask_percentage = sampling_mask.mean()

        Ll1 = l1_loss(image, gt_image) / mask_percentage
        ssim_value = ssim(image, gt_image)
        if sampling_mask is not None:
            ssim_value = normalize_ssim(ssim_value, mask_percentage)

        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)
        loss.backward()
        with torch.no_grad():
            psnr_value = 10 * torch.log10(1 / torch.mean((image - gt_image) ** 2) / mask_percentage)
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
                    self.gaussians.densify_and_prune(self.opt.densify_grad_threshold, 0.005, self.scene.cameras_extent, size_threshold)

                if iteration % self.opt.opacity_reset_interval == 0 or (self.dataset.white_background and iteration == self.opt.densify_from_iter):
                    self.gaussians.reset_opacity()

            # Optimizer step
            if iteration < self.opt.iterations + 1:
                self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)
        self.step = self.step + 1
        return metrics

    def save(self, path: str):
        self.config['logdir'] = path
        self.config['expname'] = ""
        self.trainer.save_model()
        save_config(self.config)

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
        return None

    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        """
        Get the embedding for a training image.

        Args:
            index: Index of the image.
        """
        return None
