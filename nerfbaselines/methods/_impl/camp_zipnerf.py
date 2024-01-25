from collections import namedtuple
import json
import logging
import os
from typing import Optional
from pathlib import Path
import numpy as np
import functools
import gc
from ...types import Method, MethodInfo, Dataset, CurrentProgress
from ...cameras import Cameras, CameraModel

import gin
import chex
import jax
from jax import random
import jax.numpy as jnp
import flax
from flax.training import checkpoints
from internal import configs
from internal import models
from internal import train_utils  # pylint: disable=unused-import
from internal import utils
from internal import datasets
from internal import camera_utils
from internal.camera_utils import pad_poses, unpad_poses


os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.65"

# configs.define_common_flags()
# jax.config.parse_flags_with_absl()

TIME_PRECISION = 1000  # Internally represent integer times in milliseconds.


def np_to_jax(x):
    return jnp.array(x) if isinstance(x, np.ndarray) else x


def flatten_data(images):
    """Flattens list of variable-resolution images into an array of pixels."""

    def flatten_and_concat(values, n):
        return np.concatenate([np.array(z).reshape(-1, n) for z in values])

    def index_array(i, w, h):
        x, y = camera_utils.pixel_coordinates(w, h)
        i = np.full((h, w), i)
        return np.stack([i, x, y], axis=-1)

    height = np.array([z.shape[0] for z in images])
    width = np.array([z.shape[1] for z in images])
    indices = [index_array(i, w, h) for i, (w, h) in enumerate(zip(width, height))]
    indices = flatten_and_concat(indices, 3)
    pixels = flatten_and_concat(images, 3)
    return pixels, indices


def convert_posedata(dataset: Dataset):
    camera_types = dataset.cameras.camera_types
    assert np.all(camera_types == camera_types[:1]), "Currently, all camera types must be the same for the ZipNeRF method"
    camtype = {
        CameraModel.PINHOLE.value: camera_utils.ProjectionType.PERSPECTIVE,
        CameraModel.OPENCV.value: camera_utils.ProjectionType.PERSPECTIVE,
        CameraModel.OPENCV_FISHEYE.value: camera_utils.ProjectionType.FISHEYE,
    }[camera_types[0]]

    names = []
    camtoworlds = []
    pixtocams = []
    for i in range(len(dataset)):
        names.append(dataset.file_paths[i])
        camtoworlds.append(dataset.cameras.poses[i])
        fx, fy, cx, cy = dataset.cameras.intrinsics[i]
        pixtocams.append(np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy)))
    camtoworlds = np.stack(camtoworlds, axis=0).astype(np.float32)

    pixtocams = np.stack(pixtocams, axis=0)
    distortion_params = None
    if dataset.cameras.distortion_parameters is not None:
        distortion_params = dict(zip(["k1", "k2", "p1", "p2", "k3", "k4"], np.moveaxis(dataset.cameras.distortion_parameters, -1, 0)))
    return camtoworlds, pixtocams, distortion_params, camtype


def get_transform_and_scale(transform):
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    assert np.allclose(scale, scale[0])
    scale = float(scale[0])
    transform = transform.copy()
    transform[:3, :] /= scale
    return transform, scale


class MNDataset(datasets.Dataset):
    def __init__(self, dataset: Dataset, config, split, dataparser_transform=None, verbose=True):
        self.split = split
        self.dataset = dataset
        self.dataparser_transform = dataparser_transform
        if split != "train":
            assert dataparser_transform is not None, "Must provide dataparser_transform when not training"

        self.verbose = verbose
        super().__init__(split, None, config)

    @staticmethod
    def _get_scene_bbox(config: configs.Config):
        if isinstance(config.scene_bbox, float):
            b = config.scene_bbox
            return np.array(((-b,) * 3, (b,) * 3))
        elif config.scene_bbox is not None:
            return np.array(config.scene_bbox)
        else:
            return None

    def _load_renderings(self, config: configs.Config):
        assert not config.rawnerf_mode, "RawNeRF mode is not supported for the ZipNeRF method yet"
        assert not config.forward_facing, "Forward facing scenes are not supported for the ZipNeRF method yet"
        poses, pixtocams, distortion_params, camtype = convert_posedata(self.dataset)
        self.points = self.dataset.points3D_xyz
        if self.verbose:
            print(f"*** Loaded camera parameters for {len(poses)} images")

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        pixtocams = pixtocams.astype(np.float32)
        self.camtype = camtype

        images = []
        for img in self.dataset.images:
            img = img[..., :3] / 255.0
            if self.dataset.metadata.get("type") == "blender" and img.shape[-1] == 4:
                # Blend with white background.
                img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])
            images.append(img)

        # TODO: load exif data
        self.exifs = None
        self.exposures = None
        self.max_exposure = None

        self.colmap_to_world_transform = np.eye(4)

        if self.dataset.metadata.get("type") == "blender":
            self.dataparser_transform = (None, np.eye(4))
            meters_per_colmap = self.dataparser_transform[0]
        elif self.dataparser_transform is None:
            meters_per_colmap = camera_utils.get_meters_per_colmap_from_calibration_images(config, poses, [x.name for x in self.dataset.file_paths])

            # Rotate/scale poses to align ground with xy plane and fit to unit cube.
            if config.transform_poses_fn is None:
                transform_poses_fn = camera_utils.transform_poses_pca
            else:
                transform_poses_fn = config.transform_poses_fn
            test_poses = poses.copy()
            poses, transform = transform_poses_fn(poses)
            self.colmap_to_world_transform = transform
            if self.verbose:
                print("*** Constructed COLMAP-to-world transform.")

            self.dataparser_transform = (meters_per_colmap, self.colmap_to_world_transform)

            # Test if everything is working
            transform, scale = get_transform_and_scale(self.colmap_to_world_transform)
            test_poses = unpad_poses(transform @ pad_poses(test_poses))
            test_poses[:, :3, 3] *= scale
            np.testing.assert_allclose(test_poses, poses)
        else:
            self.colmap_to_world_transform = self.dataparser_transform[1]
            meters_per_colmap = self.dataparser_transform[0]

            transform, scale = get_transform_and_scale(self.colmap_to_world_transform)
            poses = unpad_poses(transform @ pad_poses(poses))
            poses[:, :3, 3] *= scale

        self.scene_metadata = {"meters_per_colmap": self.dataparser_transform[0]}
        self.poses = poses

        indices = np.arange(len(poses))
        # All per-image quantities must be re-indexed using the split indices.
        images = [z for i, z in enumerate(images) if i in indices]
        poses, self.pixtocams, self.distortion_params = camera_utils.gather_cameras((poses, pixtocams, distortion_params), indices)
        if self.exposures is not None:
            self.exposures = self.exposures[indices]
        if config.rawnerf_mode:
            for key in ["exposure_idx", "exposure_values"]:
                self.metadata[key] = self.metadata[key][indices]

        if config.multiscale_train_factors is not None:
            all_images = images
            all_pixtocams = [self.pixtocams]
            lcm = np.lcm.reduce(config.multiscale_train_factors)
            if self.verbose:
                print(f"*** Cropping images to a multiple of {lcm}")

            def crop(z):
                sh = z.shape
                return z[: (sh[0] // lcm) * lcm, : (sh[1] // lcm) * lcm]

            def downsample(z, factor):
                down_sh = tuple(np.array(z.shape[:-1]) // factor) + z.shape[-1:]
                return np.array(jax.image.resize(z, down_sh, "bicubic"))

            images = [crop(z) for z in images]
            lossmult = [1.0] * len(images)
            # Warning: we use box filter downsampling here, for now.
            for factor in config.multiscale_train_factors:
                if self.verbose:
                    print(f"*** Downsampling by factor of {factor}x")
                all_images += [downsample(z, factor) for z in images]
                all_pixtocams.append(self.pixtocams @ np.diag([factor, factor, 1.0]))
                # Weight by the scale factor. In mip-NeRF I think we weighted by the
                # pixel area (factor**2) but empirically this seems to weight coarser
                # scales too heavily.
                lossmult += [factor] * len(images)

            n_copies = 1 + len(config.multiscale_train_factors)
            copy_inds = np.concatenate([np.arange(len(poses))] * n_copies, axis=0)
            _, poses, self.distortion_params = camera_utils.gather_cameras((self.pixtocams, poses, self.distortion_params), copy_inds)
            self.lossmult = np.array(lossmult, dtype=np.float32)
            if self.exposures is not None:
                self.exposures = np.concatenate([self.exposures] * n_copies, axis=0)

            images = all_images
            self.pixtocams = np.concatenate(all_pixtocams, axis=0).astype(np.float32)

        widths, heights = np.moveaxis(self.dataset.cameras.image_sizes, -1, 0)
        const_height = np.all(np.array(heights) == heights[0])
        const_width = np.all(np.array(widths) == widths[0])
        if const_height and const_width:
            images = np.stack(images, axis=0)
        else:
            self.images_flattened, self.indices_flattened = flatten_data(images)
            self.heights = heights
            self.widths = widths
            self._flattened = True
            if self.verbose:
                print(f"*** Flattened images into f{len(self.images_flattened)} pixels")

        self.images = images
        self.camtoworlds = poses
        self.height, self.width = images[0].shape[:2]
        if self.verbose:
            print("*** LLFF successfully loaded!")
            print(f"*** split={self.split}")
            print(f"*** #images/poses/exposures={len(images)}")
            print(f"*** #camtoworlds={len(self.camtoworlds)}")
            print(f"*** resolution={(self.height, self.width)}")


class CamP_ZipNeRF(Method):
    batch_size: int = 8192
    num_iterations: int = 200_000
    learning_rate_multiplier: float = 1.0
    camp: bool = False

    def __init__(self, checkpoint=None, batch_size: Optional[int] = None, num_iterations: Optional[int] = None, learning_rate_multiplier: Optional[float] = None, camp: Optional[bool] = None):
        super().__init__()
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.batch_size = self.batch_size if batch_size is None else batch_size
        self.num_iterations = self.num_iterations if num_iterations is None else num_iterations
        self.learning_rate_multiplier = self.learning_rate_multiplier if learning_rate_multiplier is None else learning_rate_multiplier
        self.camp = self.camp if camp is None else camp

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
        self.opaque_background = True
        self._config_str = None
        self._dataparser_transform = None
        self._camera_type = None
        if checkpoint is not None:
            with Path(checkpoint).joinpath("calibration.json").open("r") as fp:
                meta = json.load(fp)
            self._dataparser_transform = meta["meters_per_colmap"], np.array(meta["colmap_to_world_transform"]).astype(np.float32)
            self._camera_type = camera_utils.ProjectionType[meta["camera_type"]]
            self.step = self._loaded_step = int(next(iter((x for x in os.listdir(checkpoint) if x.startswith("checkpoint_")))).split("_")[1])
            self._config_str = (Path(checkpoint) / "config.gin").read_text()
            self.config = self._load_config()

    def _load_config(self, dataset_type=None):
        if self.checkpoint is None:
            # Find the config files root
            import train

            configs_path = str(Path(train.__file__).absolute().parent / "configs")
            config_path = f"{configs_path}/zipnerf/360.gin"
            if dataset_type == "blender":
                config_path = f"{configs_path}/zipnerf/blender.gin"
            gin.parse_config_file(config_path, skip_unknown=True)
            if self.camp:
                gin.parse_config_file(f"{configs_path}/camp/camera_optim.gin", skip_unknown=True)
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
        np.random.seed(self.config.np_rng_seed + jax.process_index())

        if self.config.disable_pmap_and_jit:
            chex.fake_pmap_and_jit().start()

        dummy_rays = utils.dummy_rays(include_exposure_idx=self.config.rawnerf_mode, include_exposure_values=True)
        self.model, variables = models.construct_model(rng, dummy_rays, self.config)

        fake_dataset = namedtuple("FakeDataset", ["camtype", "scene_bbox"])(self._camera_type, MNDataset._get_scene_bbox(self.config))
        state, self.lr_fn = train_utils.create_optimizer(self.config, variables)
        self.render_eval_pfn = train_utils.create_render_fn(self.model, fake_dataset)

        if self.checkpoint is not None:
            state = checkpoints.restore_checkpoint(self.checkpoint, state)
        # Resume training at the step of the last checkpoint.
        state = flax.jax_utils.replicate(state)
        self.loss_threshold = 1.0

        # Prefetch_buffer_size = 3 x batch_size.
        rng = rng + jax.process_index()  # Make random seed separate across hosts.
        self.rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
        self.state = state

    def setup_train(self, train_dataset: Dataset, *, num_iterations):
        if self.config is None:
            self.config = self._load_config(train_dataset.metadata.get("type"))

        rng = random.PRNGKey(self.config.jax_rng_seed)
        np.random.seed(self.config.np_rng_seed + jax.process_index())

        if self.config.disable_pmap_and_jit:
            chex.fake_pmap_and_jit().start()

        if self.config.batch_size % jax.device_count() != 0:
            raise ValueError("Batch size must be divisible by the number of devices.")

        dataset = MNDataset(train_dataset, self.config, "train", dataparser_transform=self._dataparser_transform)
        self._dataparser_transform = dataset.dataparser_transform
        self._camera_type = dataset.camtype
        assert self._dataparser_transform is not None

        self.cameras = jax.tree_util.tree_map(np_to_jax, dataset.cameras)
        self.cameras_replicated = flax.jax_utils.replicate(self.cameras)

        rng, key = random.split(rng)
        setup = train_utils.setup_model(self.config, key, dataset=dataset)
        self.model, state, self.render_eval_pfn, train_pstep, self.lr_fn = setup

        def fn(x):
            return x.shape if isinstance(x, jnp.ndarray) else train_utils.tree_len(x)

        param_summary = train_utils.summarize_tree(fn, state.params["params"])
        num_chars = max([len(x) for x in param_summary])
        logging.info("Optimization parameter sizes/counts:")
        for k, v in param_summary.items():
            logging.info("%s %s", k.ljust(num_chars), str(v))

        if dataset.size > self.model.num_glo_embeddings and self.model.num_glo_features > 0:
            raise ValueError(f"Number of glo embeddings {self.model.num_glo_embeddings} " f"must be at least equal to number of train images " f"{dataset.size}")

        if self.checkpoint is not None:
            state = checkpoints.restore_checkpoint(self.checkpoint, state)

        # Resume training at the step of the last checkpoint.
        state = flax.jax_utils.replicate(state)
        self.loss_threshold = 1.0

        # Prefetch_buffer_size = 3 x batch_size.
        raybatcher = datasets.RayBatcher(dataset)
        self.p_raybatcher = flax.jax_utils.prefetch_to_device(raybatcher, 3)
        rng = rng + jax.process_index()  # Make random seed separate across hosts.
        self.rngs = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
        gc.disable()  # Disable automatic garbage collection for efficiency.

        self.train_pstep = train_pstep
        self.state = state
        self.dataset = dataset
        if self._config_str is None:
            self._config_str = gin.operative_config_str()

    @property
    def train_frac(self):
        return jnp.clip((self.step - 1) / (self.config.max_steps - 1), 0, 1)

    def train_iteration(self, step: int):
        self.step = step

        with jax.profiler.StepTraceAnnotation("train", step_num=step):
            batch = next(self.p_raybatcher)
            learning_rate = self.lr_fn(step)

            self.state, stats, self.rngs = self.train_pstep(self.rngs, self.state, batch, self.cameras, self.train_frac)  # pytype: disable=wrong-arg-types  # jnp-type

        if step % self.config.gc_every == 0:
            gc.collect()  # Disable automatic garbage collection for efficiency.

        # Log training summaries. This is put behind a host_id check because in
        # multi-host evaluation, all hosts need to run inference even though we
        # only use host 0 to record results.
        out = {}
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
            for x in fstats:
                if not x.startswith("losses/"):
                    continue
                out[x[7:]] = float(fstats[x])
        return out

    def save(self, path):
        if self.render_eval_pfn is None:
            self._setup_eval()

        checkpoints.save_checkpoint_multiprocess(str(path), jax.device_get(flax.jax_utils.unreplicate(self.state)), int(self.step), keep=self.config.checkpoint_keep)

        if jax.process_index() == 0:
            with Path(path).joinpath("calibration.json").open("w+") as fp:
                meters_per_colmap, colmap_to_world_transform = self._dataparser_transform
                fp.write(
                    json.dumps(
                        {
                            "meters_per_colmap": meters_per_colmap,
                            "colmap_to_world_transform": colmap_to_world_transform.tolist(),
                            "camera_type": self._camera_type.name,
                        }
                    )
                )
            with (Path(path) / "config.gin").open("w+") as f:
                f.write(self._config_str)

    def render(self, cameras: Cameras, progress_callback=None):
        if self.render_eval_pfn is None:
            self._setup_eval()
        # Test-set evaluation.
        # We reuse the same random number generator from the optimization step
        # here on purpose so that the visualization matches what happened in
        # training.

        eval_variables = self.state.params  # Do not unreplicate
        sizes = cameras.image_sizes
        xnp = jnp
        poses = cameras.poses
        mwidth, mheight = sizes.max(0)
        assert self._dataparser_transform is not None
        test_dataset = MNDataset(
            Dataset(
                cameras=cameras,
                file_paths=[f"{i:06d}.png" for i in range(len(poses))],
                images=np.zeros((len(sizes), mheight, mwidth, 3), dtype=np.uint8),
            ),
            self.config,
            split="test",
            dataparser_transform=self._dataparser_transform,
            verbose=False,
        )
        cameras = jax.tree_util.tree_map(np_to_jax, test_dataset.cameras)
        cameras_replicated = flax.jax_utils.replicate(cameras)

        if progress_callback:
            progress_callback(CurrentProgress(0, len(poses), 0, len(poses)))

        for i in range(len(poses)):
            rays = test_dataset.generate_ray_batch(i).rays
            rendering = models.render_image(
                functools.partial(
                    self.render_eval_pfn,
                    eval_variables,
                    1.0,
                    cameras_replicated,
                ),
                rays=rays,
                rng=self.rngs[0],
                config=self.config,
            )
            if progress_callback:
                progress_callback(CurrentProgress(i + 1, len(poses), i + 1, len(poses)))

            # TODO: handle rawnerf color space
            # if config.rawnerf_mode:
            #     postprocess_fn = test_dataset.metadata['postprocess_fn']
            # else:
            accumulation = rendering["acc"]
            eps = np.finfo(accumulation.dtype).eps
            color = rendering["rgb"]
            if not self.opaque_background:
                color = xnp.concatenate(
                    (
                        # Unmultiply alpha.
                        xnp.where(accumulation[..., None] > eps, xnp.divide(color, xnp.clip(accumulation[..., None], eps, None)), xnp.zeros_like(rendering["rgb"])),
                        accumulation[..., None],
                    ),
                    -1,
                )
            depth = rendering["distance_mean"]
            assert len(accumulation.shape) == 2
            assert len(depth.shape) == 2
            yield {
                "color": np.array(color, dtype=np.float32),
                "depth": np.array(depth, dtype=np.float32),
                "accumulation": np.array(accumulation, dtype=np.float32),
            }
