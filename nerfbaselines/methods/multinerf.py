import warnings
import os
from typing import Optional, Iterable, Sequence
from pathlib import Path
import numpy as np
import functools
import gc
from nerfbaselines.types import Method, MethodInfo, ModelInfo, Dataset, OptimizeEmbeddingsOutput
from nerfbaselines.types import Cameras
from nerfbaselines.utils import padded_stack

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
        self.camtype = [camtype_map[i] for i in self.dataset["cameras"].camera_types]
        dataset_len = len(self.dataset["image_paths"])
        self.distortion_params = [
            dict(zip(["k1", "k2", "p1", "p2", "k3", "k4"], self.dataset["cameras"].distortion_parameters[i])) if self.dataset["cameras"].camera_types[i] > 0 else None for i in range(dataset_len)
        ]

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        fx, fy, cx, cy = np.moveaxis(self.dataset["cameras"].intrinsics, -1, 0)
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
        elif self.dataset["metadata"].get("name") == "blender":
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
            if self.dataset["metadata"].get("name") == "blender" and not self._eval:
                # Blender renders images in sRGB space, so convert to linear.
                self.images = self.images[..., :3] * self.images[..., 3:] + (1 - self.images[..., 3:])
        else:
            self.images = self.dataset["images"]
        self.camtoworlds = poses
        self.width, self.height = np.moveaxis(self.dataset["cameras"].image_sizes, -1, 0)
        # if self.dataset["cameras"].nears_fars is None:
        self.near = np.full((len(self.dataset),), self._config.near, dtype=np.float32)
        self.far = np.full((len(self.dataset),), self._config.far, dtype=np.float32)
        # else:
        #     self.near, self.far = np.moveaxis(self.dataset["cameras"].nears_fars, -1, 0)


class MultiNeRF(Method):
    _method_name: str = "multinerf"

    def __init__(self, *,
                 checkpoint=None, 
                 train_dataset: Optional[Dataset] = None,
                 config_overrides=None):
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self._config_overrides = config_overrides

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
        else:
            self.config = self._load_config(train_dataset["metadata"].get("name"), config_overrides=config_overrides)

        if train_dataset is not None:
            self._setup_train(train_dataset)
        else:
            self._setup_eval()

    def _load_config(self, dataset_name=None, num_iterations=None, config_overrides=None):
        if self.checkpoint is None:
            # Find the config files root
            import train

            configs_path = str(Path(train.__file__).absolute().parent / "configs")
            config_path = f"{configs_path}/360.gin"
            if dataset_name == "blender":
                config_path = f"{configs_path}/blender_256.gin"
            gin.unlock_config()
            gin.config.clear_config(clear_constants=True)
            gin.parse_config_file(config_path, skip_unknown=True)
            config_overrides, _config_overrides = (self._config_overrides or {}).copy(), (config_overrides or {})
            config_overrides.update(_config_overrides)
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

    def _setup_train(self, train_dataset: Dataset, *, config_overrides=None):
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
            np.savetxt(Path(path) / "dataparser_transform.txt", self._dataparser_transform)
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

