import shlex
from argparse import ArgumentParser
import os
from functools import cached_property
from typing import Optional, Iterable, Sequence
from nerfbaselines.utils import remap_error
from nerfbaselines.types import Method, Dataset, Cameras, RenderOutput, OptimizeEmbeddingsOutput, ModelInfo, MethodInfo
from pytorch_lightning import Trainer
import numpy as np
import train


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


def get_opts(args):
    parse_args = ArgumentParser.parse_args
    try:
        ArgumentParser.parse_args = lambda self: self
        parser = train.get_opts()
    except Exception as e:
        print(f"Failed to load options: {e}")
        print("KKKK")
        raise RuntimeError(f"Failed to load options: {e}")
    finally:
        ArgumentParser.parse_args = parse_args
    return parser.parse_args(args)


class NeRFWReimpl(Method):
    _method_name: str = "nerfw-reimpl"

    @remap_error
    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.trainer = None
        self.hparams = None
        self.checkpoint = checkpoint
        self.gaussians = None
        self.background = None
        self.step = 0

        self.scene = None

        # Setup parameters
        self._args_list = ["--root_dir", "<empty>"]
        if checkpoint is not None:
            with open(os.path.join(checkpoint, "args.txt"), "r", encoding="utf8") as f:
                self._args_list = shlex.split(f.read())

        if self.checkpoint is None and config_overrides is not None:
            _config_overrides_to_args_list(self._args_list, config_overrides)

        self._load_config()

        if self.checkpoint is None:
            # Verify parameters are set correctly
            if train_dataset["metadata"].get("name") == "blender":
                assert self.dataset.white_background, "white_background should be True for blender dataset"

        self._setup(train_dataset)

    def _load_config(self):
        self.hparams = get_opts(self._args_list)

    def _setup(self, train_dataset):
        system = train.NeRFSystem(self.hparams)
        trainer = Trainer(max_epochs=self.hparams.num_epochs,
                          resume_from_checkpoint=self.checkpoint,
                          weights_summary=None,
                          gpus=self.hparams.num_gpus,
                          accelerator='ddp' if self.hparams.num_gpus>1 else None,
                          num_sanity_val_steps=1,
                          benchmark=True,
                          profiler=None)
        class CException(Exception):
            pass

        old_ts = trainer.fit_loop.on_train_start

        def _interrupt_init(self):
            old_ts()
            raise CException()

        trainer.fit_loop.on_train_start = _interrupt_init
        try:
            trainer.fit(system)
        except CException:
            pass
        self.trainer = trainer

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
                scene = Scene(opt, self.gaussians, load_iteration=info["loaded_step"] if dataset is None else None)
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
        if sampling_mask is not None:
            image = image * sampling_mask + (1.0 - sampling_mask) * image.detach()

        Ll1 = l1_loss(image, gt_image)
        ssim_value = ssim(image, gt_image)
        loss = (1.0 - self.opt.lambda_dssim) * Ll1 + self.opt.lambda_dssim * (1.0 - ssim_value)
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
        # checkpoint_callback = \
        #     ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}',
        #                                            '{epoch:d}'),
        #                 monitor='val/psnr',
        #                 mode='max',
        #                 save_top_k=-1)

        with open(str(path) + "/args.txt", "w", encoding="utf8") as f:
            f.write(shlex_join(self._args_list))

    def export_demo(self, path: str, *, viewer_transform, viewer_initial_pose):
        model: GaussianModel = self.gaussians
        transform, scale = get_transform_and_scale(viewer_transform)
        R, t = transform[..., :3, :3], transform[..., :3, 3]
        xyz = model._xyz.detach().cpu().numpy()
        xyz = (xyz @ R.T + t[None, :]) * scale
        normals = np.zeros_like(xyz)

        f_dc = model._features_dc.detach().cpu().transpose(2, 1).numpy()
        f_rest = model._features_rest.detach().cpu().transpose(2, 1).numpy()

        # Rotate sh using Winger's group on SO3
        features = rotate_spherical_harmonics(R, np.concatenate((f_dc, f_rest), axis=-1))
        features = features.reshape(features.shape[0], -1)
        f_dc, f_rest = features[..., :f_dc.shape[-1]], features[..., f_dc.shape[-1]:]

        # fuse opacity and scale
        opacities = model.get_opacity.detach().cpu().numpy()
        gs_scale = model.scaling_inverse_activation(model.get_scaling * scale).detach().cpu().numpy()
        
        rotation = model.get_rotation.detach().cpu().numpy()
        rotation_update = rotation_matrix_to_quaternion(R)
        rotation = quaternion_multiply(rotation_update, rotation)

        dtype_full = [(attribute, 'f4') for attribute in model.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, gs_scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')

        with tempfile.TemporaryDirectory() as tmpdirname:
            ply_file = os.path.join(tmpdirname, "gaussian_splat.ply")
            out_file = os.path.join(tmpdirname, "gaussian_splat.ksplat")
            ply_data = PlyData([el])
            ply_data.write(ply_file)
            logging.info(f"Converting to ksplat format: {ply_file} -> {out_file}")

            # Convert to ksplat format
            subprocess.check_call(["bash", "-c", f"""
if [ ! -e /tmp/gaussian-splats-3d ]; then
    rm -rf "/tmp/gaussian-splats-3d-tmp"
    git clone https://github.com/mkkellogg/GaussianSplats3D.git "/tmp/gaussian-splats-3d-tmp"
    cd /tmp/gaussian-splats-3d-tmp
    npm install
    npm run build
    cd "$PWD"
    mv /tmp/gaussian-splats-3d-tmp /tmp/gaussian-splats-3d
fi
node /tmp/gaussian-splats-3d/util/create-ksplat.js {shlex.quote(ply_file)} {shlex.quote(out_file)}
"""])
            output = Path(path)
            os.rename(out_file, output / "scene.ksplat")
            wget(
                "https://raw.githubusercontent.com/gzuidhof/coi-serviceworker/7b1d2a092d0d2dd2b7270b6f12f13605de26f214/coi-serviceworker.min.js", 
                output / "coi-serviceworker.min.js")
            wget(
                "https://raw.githubusercontent.com/jkulhanek/nerfbaselines/bd328ea7d68942eea76037baed50501daa3a2425/web/public/three.module.min.js",
                output / "three.module.min.js")
            wget(
                "https://raw.githubusercontent.com/jkulhanek/nerfbaselines/bd328ea7d68942eea76037baed50501daa3a2425/web/public/gaussian-splats-3d.module.min.js",
                output / "gaussian-splats-3d.module.min.js")
            format_vector = lambda v: "[" + ",".join(f'{x:.3f}' for x in v) + "]"  # noqa: E731
            with (output / "index.html").open("w", encoding="utf8") as f, \
                open(Path(__file__).parent / "gaussian_splatting_demo.html", "r", encoding="utf8") as template:
                f.write(template.read().replace("{up}", format_vector(viewer_initial_pose.reshape(-1))))

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
