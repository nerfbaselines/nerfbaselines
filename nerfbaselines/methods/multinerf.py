import json
import warnings
import os
from typing import Optional, Iterable, Sequence
from pathlib import Path
import numpy as np
import functools
import gc
from nerfbaselines.types import Method, MethodInfo, ModelInfo, Dataset, OptimizeEmbeddingsOutput
from nerfbaselines.types import Cameras, camera_model_to_int
from nerfbaselines.utils import padded_stack
from nerfbaselines.io import numpy_to_base64, numpy_from_base64

try:
    # We need to import torch before jax to load correct CUDA libraries
    import torch
except ImportError:
    torch = None
import gin
import jax
from jax import random
import jax.numpy as jnp
import flax
from flax.training import checkpoints
from internal.datasets import Dataset as MNDataset
from internal import camera_utils
from internal import configs
from internal import models
from internal import train_utils  # pylint: disable=unused-import
from internal import utils
from internal import raw_utils


def patch_multinerf_with_multicam():
    """Add support to variable image sizes and camera intrinsics at runtime"""

    def pixels_to_rays(pix_x_int, pix_y_int, pixtocams, camtoworlds, *, distortion_params, pixtocam_ndc=None, camtype, xnp=np):
        # Must add half pixel offset to shoot rays through pixel centers.
        def pix_to_dir(x, y):
            return xnp.stack([x + .5, y + .5, xnp.ones_like(x)], axis=-1)
        # We need the dx and dy rays to calculate ray radii for mip-NeRF cones.
        pixel_dirs_stacked = xnp.stack([
            pix_to_dir(pix_x_int, pix_y_int),
            pix_to_dir(pix_x_int + 1, pix_y_int),
            pix_to_dir(pix_x_int, pix_y_int + 1)
        ], axis=0)
        # For jax, need to specify high-precision matmul.
        matmul = camera_utils.math.matmul if xnp == jnp else xnp.matmul
        mat_vec_mul = lambda A, b: matmul(A, b[..., None])[..., 0]
        # Apply inverse intrinsic matrices.
        camera_dirs_stacked = mat_vec_mul(pixtocams, pixel_dirs_stacked)

        mask = camtype > 0
        if xnp.any(mask):
            is_uniform = xnp.all(mask)
            if is_uniform:
                ldistortion_params = distortion_params
                dl = camera_dirs_stacked
            else:
                ldistortion_params = distortion_params[mask, :]
                dl = camera_dirs_stacked[:, mask, :]

            # Correct for distortion.
            dist_dict = dict(zip(
                ["k1", "k2", "k3", "k4", "p1", "p2"],
                xnp.moveaxis(ldistortion_params, -1, 0)))
            x, y = camera_utils._radial_and_tangential_undistort(
              dl[..., 0],
              dl[..., 1],
              **dist_dict,
              xnp=xnp)
            dl = xnp.stack([x, y, xnp.ones_like(x)], -1)
            dcamera_types = camtype[mask]

            fisheye_mask = dcamera_types == 2
            if fisheye_mask.any():
                is_all_fisheye = xnp.all(fisheye_mask)
                if is_all_fisheye:
                    dll = dl
                else:
                    dll = dl[:, mask, :2]
                theta = xnp.sqrt(xnp.sum(xnp.square(dll[..., :2]), axis=-1, keepdims=True))
                theta = xnp.minimum(xnp.pi, theta)
                sin_theta_over_theta = xnp.sin(theta) / theta
          
                if is_all_fisheye:
                    dl[..., :2] *= sin_theta_over_theta
                    dl[..., 2:] *= xnp.cos(theta)
                else:
                    dl[:, mask, :2] *= sin_theta_over_theta
                    dl[:, mask, 2:] *= xnp.cos(theta)

        if mask.any():
            if is_uniform:
                camera_dirs_stacked = dl
            else:
                camera_dirs_stacked[:, mask, :] = dl

        # Flip from OpenCV to OpenGL coordinate system.
        camera_dirs_stacked = matmul(camera_dirs_stacked, xnp.diag(xnp.array([1., -1., -1.])))
        # Extract 2D image plane (x, y) coordinates.
        imageplane = camera_dirs_stacked[0, ..., :2]
        # Apply camera rotation matrices.
        directions_stacked = mat_vec_mul(camtoworlds[..., :3, :3], camera_dirs_stacked)
        # Extract the offset rays.
        directions, dx, dy = directions_stacked
        origins = xnp.broadcast_to(camtoworlds[..., :3, -1], directions.shape)
        viewdirs = directions / xnp.linalg.norm(directions, axis=-1, keepdims=True)
        if pixtocam_ndc is None:
            # Distance from each unit-norm direction vector to its neighbors.
            dx_norm = xnp.linalg.norm(dx - directions, axis=-1)
            dy_norm = xnp.linalg.norm(dy - directions, axis=-1)
        else:
            # Convert ray origins and directions into projective NDC space.
            origins_dx, _ = camera_utils.convert_to_ndc(origins, dx, pixtocam_ndc, xnp=xnp)
            origins_dy, _ = camera_utils.convert_to_ndc(origins, dy, pixtocam_ndc, xnp=xnp)
            origins, directions = camera_utils.convert_to_ndc(origins, directions, pixtocam_ndc, xnp=xnp)
            # In NDC space, we use the offset between origins instead of directions.
            dx_norm = xnp.linalg.norm(origins_dx - origins, axis=-1)
            dy_norm = xnp.linalg.norm(origins_dy - origins, axis=-1)
        # Cut the distance in half, multiply it to match the variance of a uniform
        # distribution the size of a pixel (1/12, see the original mipnerf paper).
        radii = (0.5 * (dx_norm + dy_norm))[..., None] * 2 / xnp.sqrt(12)
        return origins, directions, viewdirs, radii, imageplane


    def cast_ray_batch(cameras, pixels, camtype, xnp=np):
        pixtocams, camtoworlds, distortion_params, pixtocam_ndc, camtype = cameras
        # pixels.cam_idx has shape [..., 1], remove this hanging dimension.
        cam_idx = pixels.cam_idx[..., 0]
        batch_index = lambda arr: arr if arr.ndim == 2 else arr[cam_idx]
        # Compute rays from pixel coordinates.
        origins, directions, viewdirs, radii, imageplane = camera_utils.pixels_to_rays(
            pixels.pix_x_int,
            pixels.pix_y_int,
            batch_index(pixtocams),
            batch_index(camtoworlds),
            distortion_params=distortion_params[cam_idx],
            pixtocam_ndc=pixtocam_ndc[cam_idx] if pixtocam_ndc is not None else None,
            camtype=camtype[cam_idx],
            xnp=xnp)
        # Create Rays data structure.
        return utils.Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            imageplane=imageplane,
            lossmult=pixels.lossmult,
            near=pixels.near,
            far=pixels.far,
            cam_idx=pixels.cam_idx,
            exposure_idx=pixels.exposure_idx,
            exposure_values=pixels.exposure_values,
        )

    # Patch functions
    camera_utils.pixels_to_rays = pixels_to_rays
    camera_utils.cast_ray_batch = cast_ray_batch


# Patch the functions at runtime
patch_multinerf_with_multicam()


def gin_config_to_dict(config_str: str):
    cfg = {}
    def format_value(v):
        if v in {"True", "False"}:
            return v == "True"  # bool
        if len(v) > 1 and v[0] == v[-1] == "'" or v[0] == v[-1] == '"':
            return v[1:-1]  # str
        if len(v) > 1 and v[0] == "(" and v[-1] == ")":
            return v
            # return tuple(format_value(x.strip()) for x in v[1:-1].split(",") if x.strip())  # tuple
        if len(v) > 1 and v[0] == "{" and v[-1] == "}":
            return v
            # return {format_value(x.split(":", 1)[0].strip()): format_value(x.split(":", 1)[1].strip()) for x in v[1:-1].split(",") if x.strip()}  # dict
        if v.startswith("@"):
            return v
        if v == "None":
            return None
        if "." in v or "e" in v:
            return float(v)  # float
        return int(v)  # int

    lines = config_str.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        line = line.strip()
        if line.startswith('#'):
            continue
        if not line:
            continue
        if "=" not in line:
            warnings.warn(f"Unsupported line in gin config: {line}")
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v == "\\":
            # Continuation
            while i < len(lines) and lines[i].startswith(" ") or lines[i].startswith("\t"):
                v += lines[i].strip()
                i += 1
        v = format_value(v)
        cfg[k] = v
    return cfg


class NBDataset(MNDataset):
    def __init__(self, dataset, config, eval=False, dataparser_transform=None):
        self.dataset: Dataset = dataset
        self._config = config
        self._eval = eval
        self._dataparser_transform = dataparser_transform
        super().__init__("train" if not eval else "test", None, config)

    def __setattr__(self, name, value):
        if name == "cameras" and len(value) == 4:
            # NOTE: Hack to save 5th camera type parameter
            # (pixtocams, camtoworlds, distortion_params, pixtocam_ndc, camtype)
            value = (*value, self.camtype)
        super().__setattr__(name, value)

    @property
    def dataparser_transform(self):
        return self._dataparser_transform

    def start(self):
        if not self._eval:
            return super().start()

    def _next_train(self):
        if not self._eval:
            return super()._next_train()

    def _next_test(self):
        if not self._eval:
            return super()._next_test()

    def __iter__(self):
        if not self._eval:
            return super().__iter__()
        return self._iter_eval()

    def _iter_eval(self):
        for i in range(self._n_examples):
            yield self.generate_ray_batch(i)

    def generate_ray_batch(self, cam_idx: int) -> utils.Batch:
        """Generate ray batch for a specified camera in the dataset."""
        assert not self._render_spherical, "Spherical rendering is not supported."
        # Generate rays for all pixels in the image.
        pix_x_int, pix_y_int = camera_utils.pixel_coordinates(
            self.width[cam_idx], self.height[cam_idx])
        return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx)

    def _load_renderings(self, config):
        cameras = self.dataset["cameras"]
        camtype_map = {
            camera_model_to_int("pinhole"): 0,
            camera_model_to_int("opencv"): 1,
            camera_model_to_int("opencv_fisheye"): 2,
        }
        self.camtype = np.array([camtype_map[i] for i in cameras.camera_types], np.int32)
        dataset_len = len(self.dataset["image_paths"])
        distortion_params = np.zeros((dataset_len, 6), dtype=np.float32)
        dist_shape = min(cameras.distortion_parameters.shape[1], 6)
        distortion_params[:, :cameras.distortion_parameters.shape[1]] = cameras.distortion_parameters[:, :dist_shape]
        distortion_params = np.where(self.camtype[..., None] == 0, np.zeros_like(distortion_params), distortion_params)
        # Permute k1,k2,p1,p2,k3,k4->k1,k2,k3,k4,p1,p2
        self.distortion_params = distortion_params[:, [0, 1, 4, 5, 2, 3]]

        fx, fy, cx, cy = np.moveaxis(cameras.intrinsics, -1, 0)
        pixtocam = np.linalg.inv(np.stack([camera_utils.intrinsic_matrix(fx[i], fy[i], cx[i], cy[i]) for i in range(len(fx))], axis=0))
        self.pixtocams = pixtocam.astype(np.float32)
        self.focal = 1.0 / self.pixtocams[..., 0, 0]
        self.colmap_to_world_transform = np.eye(4)

        # TODO: handle rawnerf and FF scenes

        # Rotate/scale poses to align ground with xy plane and fit to unit cube.
        poses = self.dataset["cameras"].poses.copy()

        # Convert from Opencv to OpenGL coordinate system
        poses[..., 0:3, 1:3] *= -1

        if self._dataparser_transform is not None:
            transform = self._dataparser_transform
            scale = np.linalg.norm(transform[:3, :3], ord=2, axis=-2)
            poses = camera_utils.unpad_poses(transform @ camera_utils.pad_poses(poses))
            poses[..., :3, :3] = np.diag(1 / scale) @ poses[..., :3, :3]
        elif config.dataset_loader == "blender":
            transform = np.eye(4)
            self._dataparser_transform = transform
        else:
            # test_poses = poses.copy()
            poses, transform = camera_utils.transform_poses_pca(poses)
            self._dataparser_transform = transform

            # Test if transform work correctly
            # scale = np.linalg.norm(transform[:3, :3], ord=2, axis=-2)
            # test_poses = camera_utils.unpad_poses(transform @ camera_utils.pad_poses(test_poses))
            # test_poses[..., :3, :3] = np.diag(1/scale) @ test_poses[..., :3, :3]
            # np.testing.assert_allclose(test_poses, poses, atol=1e-5)
        self.colmap_to_world_transform = transform
        self.poses = poses
        if not self._eval:
            images = self.dataset["images"]
            if isinstance(images, list):
                images = padded_stack(images)
            self.images = images.astype(np.float32) / 255.0
            if config.dataset_loader == "blender" and not self._eval:
                # Blender renders images in sRGB space, so convert to linear.
                self.images = self.images[..., :3] * self.images[..., 3:] + (1 - self.images[..., 3:])
        else:
            self.images = self.dataset["images"]
        self.camtoworlds = poses
        self.width, self.height = np.moveaxis(self.dataset["cameras"].image_sizes, -1, 0)
        self.near = np.full((len(self.dataset["cameras"]),), self._config.near, dtype=np.float32)
        self.far = np.full((len(self.dataset["cameras"]),), self._config.far, dtype=np.float32)

    def _next_train(self):
        # NOTE: Only patched for multicam support!
        # We assume all images in the dataset are the same resolution, so we can use
        # the same width/height for sampling all pixels coordinates in the batch.
        # Batch/patch sampling parameters.
        num_patches = self._batch_size // self._patch_size ** 2
        lower_border = self._num_border_pixels_to_mask
        upper_border = self._num_border_pixels_to_mask + self._patch_size - 1

        # Random camera indices.
        if self._batching == utils.BatchingMethod.ALL_IMAGES:
            cam_idx = np.random.randint(0, self._n_examples, (num_patches, 1, 1))
        else:
            cam_idx = np.random.randint(0, self._n_examples, (1,))

        # Random pixel patch x-coordinates.
        pix_x_int = np.random.randint(lower_border, self.width[cam_idx] - upper_border, (num_patches, 1, 1))
        # Random pixel patch y-coordinates.
        pix_y_int = np.random.randint(lower_border, self.height[cam_idx] - upper_border, (num_patches, 1, 1))
        # Add patch coordinate offsets.
        # Shape will broadcast to (num_patches, _patch_size, _patch_size).
        patch_dx_int, patch_dy_int = camera_utils.pixel_coordinates(self._patch_size, self._patch_size)
        pix_x_int = pix_x_int + patch_dx_int
        pix_y_int = pix_y_int + patch_dy_int

        lossmult = None
        if self._apply_bayer_mask:
            # Compute the Bayer mosaic mask for each pixel in the batch.
            lossmult = raw_utils.pixels_to_bayer_mask(pix_x_int, pix_y_int)
        return self._make_ray_batch(pix_x_int, pix_y_int, cam_idx, lossmult=lossmult)

    def _make_ray_batch(self, pix_x_int, pix_y_int, cam_idx, lossmult=None):
        # NOTE: Only patched for multicam support!
        idx = 0 if self.render_path else cam_idx
        backup = (self.near, self.far)
        try:
            self.near = self.near[idx]
            self.far = self.far[idx]
            return super()._make_ray_batch(pix_x_int, pix_y_int, cam_idx, lossmult=lossmult)
        finally:
            self.near, self.far = backup


class MultiNeRF(Method):
    _method_name: str = "multinerf"

    def __init__(self, *,
                 checkpoint=None, 
                 train_dataset: Optional[Dataset] = None,
                 config_overrides=None):
        self.checkpoint = str(checkpoint) if checkpoint is not None else None

        self._loaded_step = None
        self.pdataset_iter = None
        self.lr_fn = None
        self.train_pstep = None
        self.render_eval_pfn = None
        self.rngs = None
        self.step = 0
        self.state = None
        self.cameras = None
        self.loss_threshold = None
        self.dataset = None
        self.config = None
        self.model = None
        self._config_str = None
        self._dataparser_transform = None
        if checkpoint is not None:
            if os.path.exists(os.path.join(checkpoint, "dataparser_transform.json")):
                with open(os.path.join(checkpoint, "dataparser_transform.json"), "r") as f:
                    meta = json.load(f)
                    self._dataparser_transform = numpy_from_base64(meta["colmap_to_world_transform_base64"])
            elif os.path.exists(os.path.join(checkpoint, "dataparser_transform.txt")):
                warnings.warn("Using deprecated text format for dataparser_transform. Please upgrade the checkpoint.")
                with open(os.path.join(checkpoint, "dataparser_transform.txt"), "r") as f:
                    self._dataparser_transform = np.loadtxt(f)
            else:
                raise ValueError("Could not find dataparser_transform.{txt,json} in the checkpoint.")
            self.step = self._loaded_step = int(next(iter((x for x in os.listdir(checkpoint) if x.startswith("checkpoint_")))).split("_")[1])
            self._config_str = (Path(checkpoint) / "config.gin").read_text()
            self.config = self._load_config()
        else:
            self.config = self._load_config(config_overrides=config_overrides)

        if train_dataset is not None:
            self._setup_train(train_dataset)
        else:
            self._setup_eval()

    def _load_config(self, config_overrides=None):
        if self.checkpoint is None:
            # Find the config files root
            import train

            configs_path = str(Path(train.__file__).absolute().parent / "configs")
            config_overrides = (config_overrides or {}).copy()
            config_path = config_overrides.pop("base_config", None) or "360.gin"
            config_path = os.path.join(configs_path, config_path)
            gin.unlock_config()
            gin.config.clear_config(clear_constants=True)
            gin.parse_config_file(config_path, skip_unknown=True)
            gin.parse_config([
                f'{k} = {v}' for k, v in (config_overrides or {}).items()
            ])
            # gin.bind_parameter("Config.max_steps", num_iterations)
        else:
            assert self._config_str is not None, "Config string must be set when loading from checkpoint"
            gin.unlock_config()
            gin.config.clear_config(clear_constants=True)
            gin.parse_config(self._config_str, skip_unknown=False)
        gin.finalize()
        config = configs.Config()
        return config

    @classmethod
    def get_method_info(cls) -> MethodInfo:
        assert cls._method_name is not None, "Method was not properly registered"
        return MethodInfo(
            name=cls._method_name,
            required_features=frozenset(("color",)),
            supported_camera_models=frozenset(("pinhole", "opencv", "opencv_fisheye")),
        )

    def get_info(self):
        return ModelInfo(
            num_iterations=self.config.max_steps,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=gin_config_to_dict(self._config_str or ""),
            **self.get_method_info()
        )

    def _setup_eval(self):
        rng = random.PRNGKey(20200823)
        np.random.seed(20201473 + jax.process_index())
        rng, key = random.split(rng)

        dummy_rays = utils.dummy_rays(include_exposure_idx=self.config.rawnerf_mode, include_exposure_values=True)
        self.model, variables = models.construct_model(rng, dummy_rays, self.config)

        state, self.lr_fn = train_utils.create_optimizer(self.config, variables)
        self.render_eval_pfn = train_utils.create_render_fn(self.model)

        if self.checkpoint is not None:
            state = checkpoints.restore_checkpoint(self.checkpoint, state)
        # Resume training at the step of the last checkpoint.
        state = flax.jax_utils.replicate(state)
        self.loss_threshold = 1.0

        # Prefetch_buffer_size = 3 x batch_size.
        rng = rng + jax.process_index()  # Make random seed separate across hosts.
        self.rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
        self.state = state

    def _setup_train(self, train_dataset: Dataset):
        rng = random.PRNGKey(20200823)
        # Shift the numpy random seed by process_index() to shuffle data loaded by different
        # hosts.
        np.random.seed(20201473 + jax.process_index())

        if self.config.batch_size % jax.device_count() != 0:
            raise ValueError("Batch size must be divisible by the number of devices.")

        dataset = NBDataset(train_dataset, self.config, eval=False, dataparser_transform=self._dataparser_transform)
        self._dataparser_transform = dataset.dataparser_transform
        assert self._dataparser_transform is not None

        # Fail on CI if no GPU is available to avoid expensive CPU training.
        if os.environ.get("CI", "") == "" and jax.device_count() == 0:
            raise ValueError("Found no NVIDIA driver on your system.")

        def np_to_jax(x):
            return jnp.array(x) if isinstance(x, np.ndarray) else x

        self.cameras = tuple(np_to_jax(x) for x in dataset.cameras)

        rng, key = random.split(rng)
        setup = train_utils.setup_model(self.config, key, dataset=dataset)
        self.model, state, self.render_eval_pfn, train_pstep, self.lr_fn = setup

        variables = state.params
        num_params = jax.tree_util.tree_reduce(lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0)
        print(f"Number of parameters being optimized: {num_params}")

        if dataset.size > self.model.num_glo_embeddings and self.model.num_glo_features > 0:
            raise ValueError(f"Number of glo embeddings {self.model.num_glo_embeddings} " f"must be at least equal to number of train images " f"{dataset.size}")

        if self.checkpoint is not None:
            state = checkpoints.restore_checkpoint(self.checkpoint, state)
        # Resume training at the step of the last checkpoint.
        state = flax.jax_utils.replicate(state)
        self.loss_threshold = 1.0

        # Prefetch_buffer_size = 3 x batch_size.
        pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
        rng = rng + jax.process_index()  # Make random seed separate across hosts.
        self.rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
        gc.disable()  # Disable automatic garbage collection for efficiency.
        self.train_pstep = train_pstep
        self.pdataset_iter = iter(pdataset)
        self.state = state
        self.dataset = dataset
        if self._config_str is None:
            self._config_str = gin.operative_config_str()

    @property
    def train_frac(self):
        return jnp.clip((self.step - 1) / (self.config.max_steps - 1), 0, 1)

    def train_iteration(self, step: int):
        self.step = step
        batch = next(self.pdataset_iter)

        learning_rate = self.lr_fn(step)

        self.state, stats, self.rngs = self.train_pstep(
            self.rngs,
            self.state,
            batch,
            self.cameras,
            self.train_frac,
            self.loss_threshold,
        )
        if self.config.enable_robustnerf_loss:
            self.loss_threshold = jnp.mean(stats["loss_threshold"])

        if self.step % self.config.gc_every == 0:
            gc.collect()  # Disable automatic garbage collection for efficiency.

        # Log training summaries. This is put behind a process_index check because in
        # multi-host evaluation, all hosts need to run inference even though we
        # only use host 0 to record results.
        if jax.process_index() == 0:
            stats = flax.jax_utils.unreplicate(stats)

        # Transpose and stack stats_buffer along axis 0.
        fstats = flax.traverse_util.flatten_dict(stats, sep="/")
        fstats["learning_rate"] = learning_rate
        self.step = step + 1

        # Remap important stats
        out = {
            "psnr": float(fstats["psnr"]),
            "loss": float(fstats["loss"]),
            "learning_rate": float(fstats["learning_rate"]),
        }
        if "losses/distortion" in fstats:
            out["loss_distortion"] = float(fstats["losses/distortion"])
        if "losses/interlevel" in fstats:
            out["loss_interlevel"] = float(fstats["losses/interlevel"])
        if "losses/data" in fstats:
            out["loss_data"] = float(fstats["losses/data"])
        if self.config.enable_robustnerf_loss:
            out["loss_threshold"] = float(fstats["loss_threshold"])
        return out

    def save(self, path: str):
        path = os.path.abspath(str(path))
        if jax.process_index() == 0:
            state_to_save = jax.device_get(flax.jax_utils.unreplicate(self.state))
            checkpoints.save_checkpoint(path, state_to_save, int(self.step), keep=100)
            # np.savetxt(Path(path) / "dataparser_transform.txt", self._dataparser_transform)
            with Path(path).joinpath("dataparser_transform.json").open("w+") as fp:
                fp.write(
                    json.dumps(
                        {
                            "colmap_to_world_transform": self._dataparser_transform.tolist(),
                            "colmap_to_world_transform_base64": numpy_to_base64(self._dataparser_transform),
                        }
                    )
                )
            with (Path(path) / "config.gin").open("w+") as f:
                f.write(self._config_str)

    def render(self, cameras: Cameras, embeddings=None):
        if embeddings is not None:
            raise NotImplementedError(f"Optimizing embeddings is not supported for method {self.get_method_info()['name']}")
        # Test-set evaluation.
        # We reuse the same random number generator from the optimization step
        # here on purpose so that the visualization matches what happened in
        # training.
        xnp = jnp
        sizes = cameras.image_sizes
        poses = cameras.poses
        eval_variables = flax.jax_utils.unreplicate(self.state).params
        mwidth, mheight = sizes.max(0)
        assert self._dataparser_transform is not None
        test_dataset = NBDataset(
            dict(
                cameras=cameras,
                image_paths=[f"{i:06d}.png" for i in range(len(poses))],
                images=np.zeros((len(sizes), mheight, mwidth), dtype=np.uint8),
            ),
            self.config,
            eval=True,
            dataparser_transform=self._dataparser_transform,
        )

        for i, test_case in enumerate(test_dataset):
            rendering = models.render_image(functools.partial(self.render_eval_pfn, eval_variables, self.train_frac), test_case.rays, self.rngs[0], self.config, verbose=False)

            # TODO: handle rawnerf color space
            # if config.rawnerf_mode:
            #     postprocess_fn = test_dataset["metadata"]['postprocess_fn']
            # else:
            accumulation = rendering["acc"]
            eps = np.finfo(accumulation.dtype).eps
            color = rendering["rgb"]
            if not self.model.opaque_background:
                color = xnp.concatenate(
                    (
                        # Unmultiply alpha.
                        xnp.where(accumulation[..., None] > eps, xnp.divide(color, xnp.clip(accumulation[..., None], eps, None)), xnp.zeros_like(rendering["rgb"])),
                        accumulation[..., None],
                    ),
                    -1,
                )
            depth = np.array(rendering["distance_mean"], dtype=np.float32)
            assert len(accumulation.shape) == 2
            assert len(depth.shape) == 2
            yield {
                "color": np.array(color, dtype=np.float32),
                "depth": np.array(depth, dtype=np.float32),
                "accumulation": np.array(accumulation, dtype=np.float32),
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
        raise NotImplementedError()

    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        """
        Get the embedding for a training image.

        Args:
            index: Index of the image.
        """
        return None

