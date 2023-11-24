from typing import Optional
import numpy as np
import functools
import gc
from ...types import Method, MethodInfo, Dataset, CurrentProgress
from ...cameras import Cameras

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


def build_dataset(dataset: Dataset, config, eval=False):
    class NBDataset(MNDataset):
        def __init__(self, dataset, config):
            self.dataset: Dataset = dataset
            super().__init__("train" if not eval else "test", None, config)

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
                dict(zip(["k1", "k2", "k3", "k4", "p1", "p2"], self.distortion_params[i])) if self.dataset.cameras.camera_types[i] > 0 else None for i in range(len(self.dataset))
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
            poses, transform = camera_utils.transform_poses_pca(poses)
            self.colmap_to_world_transform = transform
            self.poses = poses
            self.images = self.dataset.images
            self.camtoworlds = poses
            self.width, self.height = np.moveaxis(self.dataset.cameras.image_sizes, -1, 0)
            if self.dataset.cameras.nears_fars is None:
                self.near = np.full((len(self.dataset),), self.config.near, dtype=np.float32)
                self.far = np.full((len(self.dataset),), self.config.far, dtype=np.float32)
            else:
                self.near, self.far = np.moveaxis(self.dataset.cameras.nears_fars, -1, 0)

    if eval:
        # Speedup by not creating thread and pushing into the queue
        NBDataset.start = lambda self: None
        NBDataset._next_train = lambda self: None
        NBDataset._next_test = lambda self: None

        def iter_eval(self):
            for i in range(self._n_examples):
                yield self.generate_ray_batch(i)

        NBDataset.__iter__ = iter_eval
    return NBDataset(dataset, config)


class MultiNeRF(Method):
    batch_size: int = 16384
    num_iterations: int = 250_000

    def __init__(self, checkpoint=None, batch_size: Optional[int] = None, num_iterations: Optional[int] = None):
        super().__init__()
        self.checkpoint = checkpoint
        self.batch_size = self.batch_size if batch_size is None else batch_size
        self.num_iterations = self.num_iterations if num_iterations is None else num_iterations

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
        self.config = self._load_config()

    def _load_config(self):
        config_paths = []
        gin.parse_config_files_and_bindings(
            config_paths,
            [
                "Config.batch_size = %d" % self.batch_size,
                "Config.max_steps = %d" % self.num_iterations,
            ],
            skip_unknown=True,
        )
        return configs.Config()

    @property
    def info(self):
        return MethodInfo(
            num_iterations=self.num_iterations,
        )

    def setup_train(self, train_dataset: Dataset, *, num_iterations):
        rng = random.PRNGKey(20200823)
        # Shift the numpy random seed by host_id() to shuffle data loaded by different
        # hosts.
        np.random.seed(20201473 + jax.host_id())

        if self.config.batch_size % jax.device_count() != 0:
            raise ValueError("Batch size must be divisible by the number of devices.")

        dataset = build_dataset(train_dataset, self.config)

        def np_to_jax(x):
            return jnp.array(x) if isinstance(x, np.ndarray) else x

        self.cameras = tuple(np_to_jax(x) for x in dataset.cameras)

        rng, key = random.split(rng)
        setup = train_utils.setup_model(self.config, key, dataset=dataset)
        model, state, self.render_eval_pfn, train_pstep, self.lr_fn = setup

        variables = state.params
        num_params = jax.tree_util.tree_reduce(lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0)
        print(f"Number of parameters being optimized: {num_params}")

        if dataset.size > model.num_glo_embeddings and model.num_glo_features > 0:
            raise ValueError(f"Number of glo embeddings {model.num_glo_embeddings} " f"must be at least equal to number of train images " f"{dataset.size}")

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
            "loss_distortion": float(fstats["losses/distortion"]),
            "loss_intelevel": float(fstats["losses/interlevel"]),
            "loss_data": float(fstats["losses/data"]),
        }
        if self.config.enable_robustnerf_loss:
            out["loss_threshold"] = float(fstats["loss_threshold"])
        return out

    def save(self, path):
        state_to_save = jax.device_get(flax.jax_utils.unreplicate(self.state))
        checkpoints.save_checkpoint(str(path), state_to_save, int(self.step), keep=100)

    def render(self, cameras: Cameras, progress_callback=None):
        # Test-set evaluation.
        # We reuse the same random number generator from the optimization step
        # here on purpose so that the visualization matches what happened in
        # training.
        sizes = cameras.image_sizes
        poses = cameras.poses
        eval_variables = flax.jax_utils.unreplicate(self.state).params
        mwidth, mheight = sizes.max(0)
        test_dataset = build_dataset(
            Dataset(
                cameras=cameras,
                file_paths=[f"{i:06d}.png" for i in range(len(poses))],
                images=np.zeros((len(sizes), mheight, mwidth), dtype=np.uint8),
            ),
            self.config,
            eval=True,
        )
        if progress_callback:
            progress_callback(CurrentProgress(0, len(poses), 0, len(poses)))
        for i, test_case in enumerate(test_dataset):
            rendering = models.render_image(functools.partial(self.render_eval_pfn, eval_variables, self.train_frac), test_case.rays, self.rngs[0], self.config)
            if progress_callback:
                progress_callback(CurrentProgress(i + 1, len(poses), i + 1, len(poses)))

            # TODO: handle rawnerf color space
            # if config.rawnerf_mode:
            #     postprocess_fn = test_dataset.metadata['postprocess_fn']
            # else:
            yield {
                "color": np.array(rendering["rgb"]),
            }
