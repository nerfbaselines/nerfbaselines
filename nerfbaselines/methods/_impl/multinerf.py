import os
from typing import Optional
from pathlib import Path
import numpy as np
import functools
import gc
from ...types import Method, MethodInfo, Dataset, CurrentProgress
from ...cameras import Cameras, CameraModel
from ...utils import padded_stack

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


class NBDataset(MNDataset):
    def __init__(self, dataset, config, eval=False, dataparser_transform=None):
        self.dataset: Dataset = dataset
        self._config = config
        self._eval = eval
        self._dataparser_transform = dataparser_transform
        super().__init__("train" if not eval else "test", None, config)

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

    def _load_renderings(self, config):
        """Load images and poses from disk.

        Args:
            config: utils.Config, user-specified config parameters.
            In inherited classes, this method must set the following public attributes:
                images: [N, height, width, 3] array for RGB images.
                disp_images: [N, height, width] array for depth data (optional).
                normal_images: [N, height, width, 3] array for normals (optional).
                camtoworlds: [N, 3, 4] array of extrinsic pose matrices.
                poses: [..., 3, 4] array of auxiliary pose data (optional).
                pixtocams: [N, 3, 4] array of inverse intrinsic matrices.
                distortion_params: dict, camera lens distortion model parameters.
                height: int, height of images.
                width: int, width of images.
                focal: float, focal length to use for ideal pinhole rendering.
        """
        camtype_map = [
            camera_utils.ProjectionType.PERSPECTIVE,
            camera_utils.ProjectionType.PERSPECTIVE,
            camera_utils.ProjectionType.FISHEYE,
        ]
        self.camtype = [camtype_map[i] for i in self.dataset.cameras.camera_types]
        self.distortion_params = [
            dict(zip(["k1", "k2", "p1", "p2", "k3", "k4"], self.dataset.cameras.distortion_parameters[i])) if self.dataset.cameras.camera_types[i] > 0 else None for i in range(len(self.dataset))
        ]

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        fx, fy, cx, cy = np.moveaxis(self.dataset.cameras.intrinsics, -1, 0)
        pixtocam = np.linalg.inv(np.stack([camera_utils.intrinsic_matrix(fx[i], fy[i], cx[i], cy[i]) for i in range(len(fx))], axis=0))
        self.pixtocams = pixtocam.astype(np.float32)
        self.focal = 1.0 / self.pixtocams[..., 0, 0]

        self.colmap_to_world_transform = np.eye(4)

        # TODO: handle rawnerf and FF scenes

        # Rotate/scale poses to align ground with xy plane and fit to unit cube.
        poses = self.dataset.cameras.poses
        if self._dataparser_transform is not None:
            transform = self._dataparser_transform
            scale = np.linalg.norm(transform[:3, :3], ord=2, axis=-2)
            poses = camera_utils.unpad_poses(transform @ camera_utils.pad_poses(poses))
            poses[..., :3, :3] = np.diag(1 / scale) @ poses[..., :3, :3]
        elif self.dataset.metadata.get("type") == "blender":
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
            images = self.dataset.images
            if isinstance(images, list):
                images = padded_stack(images)
            self.images = images.astype(np.float32) / 255.0
            if self.dataset.metadata.get("type") == "blender" and not self._eval:
                # Blender renders images in sRGB space, so convert to linear.
                self.images = self.images[..., :3] * self.images[..., 3:] + (1 - self.images[..., 3:])
        else:
            self.images = self.dataset.images
        self.camtoworlds = poses
        self.width, self.height = np.moveaxis(self.dataset.cameras.image_sizes, -1, 0)
        # if self.dataset.cameras.nears_fars is None:
        self.near = np.full((len(self.dataset),), self._config.near, dtype=np.float32)
        self.far = np.full((len(self.dataset),), self._config.far, dtype=np.float32)
        # else:
        #     self.near, self.far = np.moveaxis(self.dataset.cameras.nears_fars, -1, 0)


class MultiNeRF(Method):
    batch_size: int = 16384
    num_iterations: int = 250_000
    learning_rate_multiplier: float = 1.0

    def __init__(self, checkpoint=None, batch_size: Optional[int] = None, num_iterations: Optional[int] = None, learning_rate_multiplier: Optional[float] = None):
        super().__init__()
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.batch_size = self.batch_size if batch_size is None else batch_size
        self.num_iterations = self.num_iterations if num_iterations is None else num_iterations
        self.learning_rate_multiplier = self.learning_rate_multiplier if learning_rate_multiplier is None else learning_rate_multiplier

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
            self._dataparser_transform = np.loadtxt(Path(checkpoint) / "dataparser_transform.txt")
            self.step = self._loaded_step = int(next(iter((x for x in os.listdir(checkpoint) if x.startswith("checkpoint_")))).split("_")[1])
            self._config_str = (Path(checkpoint) / "config.gin").read_text()
            self.config = self._load_config()

    def _load_config(self, dataset_type=None):
        if self.checkpoint is None:
            # Find the config files root
            import train

            configs_path = str(Path(train.__file__).absolute().parent / "configs")
            config_path = f"{configs_path}/360.gin"
            if dataset_type == "blender":
                config_path = f"{configs_path}/blender_256.gin"
            gin.parse_config_file(config_path, skip_unknown=True)
        else:
            assert self._config_str is not None, "Config string must be set when loading from checkpoint"
            gin.parse_config(self._config_str, skip_unknown=False)
        gin.bind_parameter("Config.batch_size", self.batch_size)
        gin.bind_parameter("Config.max_steps", self.num_iterations)
        gin.finalize()
        config = configs.Config()
        config.lr_init *= self.learning_rate_multiplier
        config.lr_final *= self.learning_rate_multiplier
        return config

    def get_info(self):
        return MethodInfo(
            num_iterations=self.num_iterations,
            loaded_step=self._loaded_step,
            supported_camera_models=frozenset((CameraModel.PINHOLE, CameraModel.OPENCV, CameraModel.OPENCV_FISHEYE)),
        )

    def _setup_eval(self):
        rng = random.PRNGKey(20200823)
        np.random.seed(20201473 + jax.host_id())
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
        rng = rng + jax.host_id()  # Make random seed separate across hosts.
        self.rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
        self.state = state

    def setup_train(self, train_dataset: Dataset, *, num_iterations):
        if self.config is None:
            self.config = self._load_config(train_dataset.metadata.get("type"))

        rng = random.PRNGKey(20200823)
        # Shift the numpy random seed by host_id() to shuffle data loaded by different
        # hosts.
        np.random.seed(20201473 + jax.host_id())

        if self.config.batch_size % jax.device_count() != 0:
            raise ValueError("Batch size must be divisible by the number of devices.")

        dataset = NBDataset(train_dataset, self.config, eval=False, dataparser_transform=self._dataparser_transform)
        self._dataparser_transform = dataset.dataparser_transform
        assert self._dataparser_transform is not None

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
        rng = rng + jax.host_id()  # Make random seed separate across hosts.
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

        # Log training summaries. This is put behind a host_id check because in
        # multi-host evaluation, all hosts need to run inference even though we
        # only use host 0 to record results.
        if jax.host_id() == 0:
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

    def save(self, path):
        if self.render_eval_pfn is None:
            self._setup_eval()
        if jax.host_id() == 0:
            state_to_save = jax.device_get(flax.jax_utils.unreplicate(self.state))
            checkpoints.save_checkpoint(str(path), state_to_save, int(self.step), keep=100)
            np.savetxt(Path(path) / "dataparser_transform.txt", self._dataparser_transform)
            with (Path(path) / "config.gin").open("w+") as f:
                f.write(self._config_str)

    def render(self, cameras: Cameras, progress_callback=None):
        if self.render_eval_pfn is None:
            self._setup_eval()
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
            Dataset(
                cameras=cameras,
                file_paths=[f"{i:06d}.png" for i in range(len(poses))],
                images=np.zeros((len(sizes), mheight, mwidth), dtype=np.uint8),
            ),
            self.config,
            eval=True,
            dataparser_transform=self._dataparser_transform,
        )
        if progress_callback:
            progress_callback(CurrentProgress(0, len(poses), 0, len(poses)))

        for i, test_case in enumerate(test_dataset):
            rendering = models.render_image(functools.partial(self.render_eval_pfn, eval_variables, self.train_frac), test_case.rays, self.rngs[0], self.config, verbose=False)
            if progress_callback:
                progress_callback(CurrentProgress(i + 1, len(poses), i + 1, len(poses)))

            # TODO: handle rawnerf color space
            # if config.rawnerf_mode:
            #     postprocess_fn = test_dataset.metadata['postprocess_fn']
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
