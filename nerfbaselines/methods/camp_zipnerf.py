import shutil
import re
import warnings
from collections import namedtuple
import json
import logging
import os
from typing import Optional, Iterable, Sequence
from pathlib import Path
import numpy as np
import functools
import gc
from nerfbaselines.types import Method, MethodInfo, ModelInfo, Dataset, OptimizeEmbeddingsOutput
from nerfbaselines.types import Cameras, camera_model_to_int
from nerfbaselines.io import numpy_to_base64, numpy_from_base64
try:
    # We need to import torch before jax to load correct CUDA libraries
    import torch
except ImportError:
    torch = None

import gin
import gin.config
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
    camera_types = dataset["cameras"].camera_types
    assert np.all(camera_types == camera_types[:1]), "Currently, all camera types must be the same for the ZipNeRF method"
    camtype = {
        camera_model_to_int("pinhole"): camera_utils.ProjectionType.PERSPECTIVE,
        camera_model_to_int("opencv"): camera_utils.ProjectionType.PERSPECTIVE,
        camera_model_to_int("opencv_fisheye"): camera_utils.ProjectionType.FISHEYE,
    }[camera_types[0]]

    names = []
    camtoworlds = []
    pixtocams = []
    for i in range(len(dataset["image_paths"])):
        names.append(dataset["image_paths"][i])
        camtoworlds.append(dataset["cameras"].poses[i])
        fx, fy, cx, cy = dataset["cameras"].intrinsics[i]
        pixtocams.append(np.linalg.inv(camera_utils.intrinsic_matrix(fx, fy, cx, cy)))
    camtoworlds = np.stack(camtoworlds, axis=0).astype(np.float32).copy()

    # Convert from Opencv to OpenGL coordinate system
    camtoworlds[..., 0:3, 1:3] *= -1

    pixtocams = np.stack(pixtocams, axis=0)
    distortion_params = None
    if dataset["cameras"].distortion_parameters is not None:
        distortion_params = dict(zip(["k1", "k2", "p1", "p2", "k3", "k4"], np.moveaxis(dataset["cameras"].distortion_parameters, -1, 0)))
    return camtoworlds, pixtocams, distortion_params, camtype


def get_transform_and_scale(transform):
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    assert np.allclose(scale, scale[0])
    scale = float(scale[0])
    transform = transform.copy()
    transform[:3, :] /= scale
    return transform, scale


def gin_config_to_dict(config_str: str):
    cfg = {}
    float_re = r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?"
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
        if re.match(float_re, v):
            return float(v)  # float
        elif v.isdigit() or (v[0] in "+-" and v[1:].isdigit()):
            return int(v)  # int
        return str(v)  # int

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
        if self.verbose:
            print(f"*** Loaded camera parameters for {len(poses)} images")

        # Scale the inverse intrinsics matrix by the image downsampling factor.
        pixtocams = pixtocams.astype(np.float32)
        self.camtype = camtype

        images = []
        for img in self.dataset["images"]:
            img = img.astype(np.float32) / 255.0
            if config.dataset_loader == 'blender' and img.shape[-1] == 4:
                # Blend with white background.
                img = img[..., :3] * img[..., 3:] + (1 - img[..., 3:])
            images.append(img)

        # TODO: load exif data
        self.exifs = None
        self.exposures = None
        self.max_exposure = None

        self.colmap_to_world_transform = np.eye(4)

        if self.dataparser_transform is not None:
            self.colmap_to_world_transform = self.dataparser_transform[1]
            meters_per_colmap = self.dataparser_transform[0]

            transform, scale = get_transform_and_scale(self.colmap_to_world_transform)
            poses = unpad_poses(transform @ pad_poses(poses))
            poses[:, :3, 3] *= scale
        elif config.dataset_loader == 'blender':
            self.dataparser_transform = (None, np.eye(4))
            meters_per_colmap = self.dataparser_transform[0]
        elif self.dataparser_transform is None:
            meters_per_colmap = camera_utils.get_meters_per_colmap_from_calibration_images(config, poses, [os.path.split(x)[-1] for x in self.dataset["image_paths"]])

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

        widths, heights = np.moveaxis(self.dataset["cameras"].image_sizes, -1, 0)
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
    _method_name: str = "camp_zipnerf"
    _camp: bool = False

    def __init__(self, 
                 *,
                 checkpoint=None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides=None):
        super().__init__()
        self._config_overrides = config_overrides or {}

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
        self.opaque_background = True
        self._config_str = None
        self._dataparser_transform = None
        self._camera_type = None
        if checkpoint is not None:
            if os.path.exists(os.path.join(checkpoint, "calibration.json")):
                # NOTE: old checkpoints have calibration.json, new ones have dataparser_transform.json
                warnings.warn("calibration.json is deprecated, please re-save the checkpoint.")
                with Path(checkpoint).joinpath("calibration.json").open("r") as fp:
                    meta = json.load(fp)
            elif os.path.exists(os.path.join(checkpoint, "dataparser_transform.json")):
                with Path(checkpoint).joinpath("dataparser_transform.json").open("r") as fp:
                    meta = json.load(fp)
            else:
                raise RuntimeError("dataparser_transform.json not found in metadata")
            if "colmap_to_world_transform_base64" in meta:
                colmap_to_world_transform = numpy_from_base64(meta["colmap_to_world_transform_base64"])
            else:
                warnings.warn("colmap_to_world_transform_base64 not found in metadata, falling back to colmap_to_world_transform. Please re-save the checkpoint.")
                colmap_to_world_transform = np.array(meta["colmap_to_world_transform"]).astype(np.float32)
            self._dataparser_transform = meta["meters_per_colmap"], colmap_to_world_transform
            self._camera_type = camera_utils.ProjectionType[meta["camera_type"]]
            self.step = self._loaded_step = int(next(iter((x for x in os.listdir(checkpoint) if x.startswith("checkpoint_")))).split("_")[1])
            self._config_str = (Path(checkpoint) / "config.gin").read_text()
            self.config = self._load_config()
            self._setup_eval()
        elif train_dataset is not None:
            self.config = self._load_config(config_overrides)
            self._setup_train(train_dataset)
        else:
            raise ValueError("Either checkpoint or train_dataset must be provided")

    def _load_config(self, config_overrides=None):
        # Find the config files root
        import train

        configs_path = str(Path(train.__file__).absolute().parent)
        gin.config.clear_config(clear_constants=True)
        gin.add_config_file_search_path(configs_path)

        config_overrides = (config_overrides or {}).copy()
        base_config = config_overrides.pop("base_config", "zipnerf/360")

        # Fix a bug in gin
        gin.config._FILE_READERS = gin.config._FILE_READERS[:1]
        if self.checkpoint is None:
            config_path = f"configs/{base_config}.gin"
            gin.parse_config_file(config_path, skip_unknown=True)
            if self._camp:
                gin.parse_config_file("configs/camp/camera_optim.gin", skip_unknown=True)
            config = configs.Config()
            # gin.bind_parameter("Config.max_steps", num_iterations)
        else:
            assert self._config_str is not None, "Config string must be set when loading from checkpoint"
            gin.parse_config(self._config_str, skip_unknown=False)
        if config_overrides:
            gin.parse_config([f"{k} = {v}" for k, v in config_overrides.items()])
        gin.finalize()
        config = configs.Config()
        return config

    @classmethod
    def get_method_info(cls):
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
            loaded_checkpoint=str(self.checkpoint) if self.checkpoint is not None else None,
            hparams=gin_config_to_dict(self._config_str or ""),
            **self.get_method_info()
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

    def _setup_train(self, train_dataset: Dataset):
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

        # Log training summaries. This is put behind a process_index check because in
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

    def save(self, path: str):
        path = os.path.abspath(str(path))
        if self.render_eval_pfn is None:
            self._setup_eval()

        if os.path.exists(os.path.join(path, f"checkpoint_{self.step}")):
            # If the checkpoint already exists, we will remove it
            shutil.rmtree(os.path.join(path, f"checkpoint_{self.step}"))

        checkpoints.save_checkpoint_multiprocess(path, jax.device_get(flax.jax_utils.unreplicate(self.state)), int(self.step), keep=self.config.checkpoint_keep)

        if jax.process_index() == 0:
            with Path(path).joinpath("dataparser_transform.json").open("w+") as fp:
                meters_per_colmap, colmap_to_world_transform = self._dataparser_transform
                fp.write(
                    json.dumps(
                        {
                            "meters_per_colmap": meters_per_colmap,
                            "colmap_to_world_transform": colmap_to_world_transform.tolist(),
                            "colmap_to_world_transform_base64": numpy_to_base64(colmap_to_world_transform),
                            "camera_type": self._camera_type.name,
                        }
                    )
                )
            with (Path(path) / "config.gin").open("w+") as f:
                f.write(self._config_str)

    def render(self, cameras: Cameras, embeddings=None):
        if self.render_eval_pfn is None:
            self._setup_eval()
        if embeddings is not None:
            raise NotImplementedError(f"Optimizing embeddings is not supported for method {self.get_method_info()['name']}")
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
            dict(
                cameras=cameras,
                image_paths=[f"{i:06d}.png" for i in range(len(poses))],
                images=np.zeros((len(sizes), mheight, mwidth, 3), dtype=np.uint8),
            ),
            self.config,
            split="test",
            dataparser_transform=self._dataparser_transform,
            verbose=False,
        )
        cameras = jax.tree_util.tree_map(np_to_jax, test_dataset.cameras)
        cameras_replicated = flax.jax_utils.replicate(cameras)

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
                verbose=False,
            )

            # TODO: handle rawnerf color space
            # if config.rawnerf_mode:
            #     postprocess_fn = test_dataset["metadata"]['postprocess_fn']
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
