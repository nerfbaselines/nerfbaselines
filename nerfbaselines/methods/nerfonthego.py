from itertools import chain
import gc
import warnings
import os
from pathlib import Path
from typing import Optional
from nerfbaselines.types import Dataset, Method, MethodInfo, ModelInfo, Cameras, camera_model_to_int
from nerfbaselines.utils import convert_image_dtype

from flax.training import checkpoints
import flax
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import gin
import cv2
from internal import configs
from internal import utils
from internal import models
from internal import train_utils
from internal import camera_utils
from internal.datasets import Dataset as GoBaseDataset


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


class GoDataset(GoBaseDataset):
    def __init__(self, dataset: Dataset, config, eval=False):
        self.dataset = dataset
        self._rendering_mode = eval
        super().__init__("train", None, config)

    def load_feat(self, path, feat_rate, factor):
        image_dir = f'images_{factor}' if f'images_{factor}' in path else 'images'
        format = path[-4:]
        feat_path = path.replace(image_dir, f'features_{feat_rate}').replace(format, '.npy')
        feat = np.load(feat_path)
        return feat

    def _load_renderings(self, config):
        """Load images from disk."""
        dataset: Dataset = self.dataset
        # Set up scaling factor.

        # Validate cameras first
        assert (
            np.all(dataset["cameras"].camera_types[:1] == dataset["cameras"].camera_types) and
            np.all(dataset["cameras"].image_sizes[:1] == dataset["cameras"].image_sizes) and
            np.all(dataset["cameras"].distortion_parameters[:1] == dataset["cameras"].distortion_parameters) and
            np.all(dataset["cameras"].intrinsics[:1] == dataset["cameras"].intrinsics)), "All cameras must be the same"

        # factor = config.factor
        # NOTE: config.factor is ignored because the images are provided already
        poses = dataset["cameras"].poses.copy()

        # Convert from OpenCV to OpenGL coordinate system
        poses[..., 0:3, 1:3] *= -1
        
        fx, fy, cx, cy = dataset["cameras"].intrinsics[0]
        pixtocam = np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy))
        coeffs = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4']
        distortion_params = None
        if dataset["cameras"].camera_types[0].item() in (camera_model_to_int("opencv"), camera_model_to_int("opencv_fisheye")):
            distortion_params = dict(zip(coeffs, chain(dataset["cameras"].distortion_parameters[0], [0]*6)))
        camtype = camera_utils.ProjectionType.PERSPECTIVE

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        self.pixtocams = pixtocam.astype(np.float32)
        self.focal = 1. / self.pixtocams[0, 0]
        self.distortion_params = distortion_params
        self.camtype = camtype

        # Load images.
        images = np.stack([convert_image_dtype(x, np.uint8) for x in dataset["images"]])

        # read in features
        # TODO: fix features
        features = None
        if not self._rendering_mode:
            features = [self.load_feat(x, config.feat_rate, config.factor) for x in dataset["images"]]
            features = np.stack(features, axis=0)

        # create assignment
        # assignment is for super-pixel setting, not used in this project
        patch_H, patch_W = config.H // config.feat_rate, config.W //config.feat_rate
        patch_H = patch_H // config.feat_ds * config.feat_ds
        patch_W = patch_W // config.feat_ds * config.feat_ds
        i, j = np.meshgrid(np.arange(patch_H), np.arange(patch_W), indexing='ij')
        assignment_data = i // config.feat_ds * patch_W // config.feat_ds + j // config.feat_ds
        assignment_data_resized = cv2.resize(assignment_data,
                                             (images.shape[2], images.shape[1]),
                                             interpolation=cv2.INTER_NEAREST).astype(np.int64)
        assignments = assignment_data_resized[None,...].repeat(images.shape[0], axis=0)

        self.colmap_to_world_transform = np.eye(4)

        # Separate out 360 versus forward facing scenes.
        assert not config.forward_facing, "Forward facing scenes not supported."

        # Rotate/scale poses to align ground with xy plane and fit to unit cube.
        poses, transform = camera_utils.transform_poses_pca(poses)
        self.colmap_to_world_transform = transform

        self.poses = poses
        self.images = images
        self.camtoworlds = poses
        self.height, self.width = images.shape[1:3]

        self.features = features
        self.assignments = assignments


class NeRFOnthego(Method):
    _method_name: str = "nerfonthego"

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

        # Setup config
        self.config = self._load_config(config_overrides=config_overrides)

        if checkpoint is not None:
            self._dataparser_transform = np.loadtxt(Path(checkpoint) / "dataparser_transform.txt")
            self.step = self._loaded_step = int(next(iter((x for x in os.listdir(checkpoint) if x.startswith("checkpoint_")))).split("_")[1])

        self._setup(train_dataset)

    def _load_config(self, config_overrides=None):
        if self.checkpoint is None:
            # Find the config files root
            import train

            configs_path = str(Path(train.__file__).absolute().parent / "configs")
            config_path = "360_dino.gin"
            if (config_overrides or {}).get("base_config") is not None:
                config_path = config_overrides["base_config"]
            config_path = os.path.join(configs_path, config_path)
            gin.unlock_config()
            gin.config.clear_config(clear_constants=True)
            gin.parse_config_file(config_path, skip_unknown=True)
            gin.parse_config([
                f'{k} = {v}' for k, v in (config_overrides or {}).items() if k != "base_config"
            ])
            gin.finalize()
            self._config_str = gin.operative_config_str()
        else:
            self._config_str = (Path(self.checkpoint) / "config.gin").read_text()
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

    def _setup(self, train_dataset: Dataset, *, config_overrides=None):
        rng = random.PRNGKey(20200823)
        # Shift the numpy random seed by process_index() to shuffle data loaded by different
        # hosts.
        np.random.seed(20201473 + jax.process_index())
        rng, key = random.split(rng)

        # Fail on CI if no GPU is available to avoid expensive CPU training.
        if os.environ.get("CI", "") == "" and jax.device_count() == 0:
            raise RuntimeError("Found no NVIDIA driver on your system.")

        if self.config.batch_size % jax.device_count() != 0:
            raise ValueError("Batch size must be divisible by the number of devices.")

        if train_dataset is not None:
            self.dataset = dataset = GoDataset(train_dataset, self.config, eval=False, dataparser_transform=self._dataparser_transform)
            self._dataparser_transform = dataset.dataparser_transform
            assert self._dataparser_transform is not None

            def np_to_jax(x):
                return jnp.array(x) if isinstance(x, np.ndarray) else x

            self.cameras = tuple(np_to_jax(x) for x in dataset.cameras)

            self.model, self.state, self.render_eval_pfn, self.train_pstep, self.lr_fn = \
                train_utils.setup_model(self.config, key, dataset=dataset)
            variables = self.state.params

            if dataset.size > self.model.num_glo_embeddings and self.model.num_glo_features > 0:
                raise ValueError(f"Number of glo embeddings {self.model.num_glo_embeddings} " f"must be at least equal to number of train images " f"{dataset.size}")

            pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
            self.pdataset_iter = iter(pdataset)
            gc.disable()  # Disable automatic garbage collection for efficiency.
        else:
            dummy_rays = utils.dummy_rays(include_exposure_idx=self.config.rawnerf_mode, include_exposure_values=True)
            self.model, variables = models.construct_model(rng, dummy_rays, self.config)
            self.state, self.lr_fn = train_utils.create_optimizer(self.config, variables)
            self.render_eval_pfn = train_utils.create_render_fn(self.model)

        num_params = jax.tree_util.tree_reduce(lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0)
        print(f"Number of parameters being optimized: {num_params}")

        if self.checkpoint is not None:
            self.state = checkpoints.restore_checkpoint(self.checkpoint, self.state)
        # Resume training at the step of the last checkpoint.
        self.state = flax.jax_utils.replicate(self.state)
        self.loss_threshold = 1.0

        # Prefetch_buffer_size = 3 x batch_size.
        rng = rng + jax.process_index()  # Make random seed separate across hosts.
        self.rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.

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
        test_dataset = GoDataset(
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

