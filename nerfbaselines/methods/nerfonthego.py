import base64
import io
import json
import functools
from itertools import chain
import gc
import warnings
import os
from pathlib import Path
from typing import Optional, Union, Dict, cast
from nerfbaselines import (
    Dataset, Method, MethodInfo, ModelInfo, Cameras, camera_model_to_int, RenderOutput,
)
from nerfbaselines.utils import convert_image_dtype

from flax.training import checkpoints  # type: ignore
import flax  # type: ignore
import jax  # type: ignore
from jax import random  # type: ignore
import jax.numpy as jnp  # type: ignore
import numpy as np
import gin  # type: ignore
import cv2  # type: ignore
from PIL import Image
from tqdm import tqdm
from internal import configs  # type: ignore
from internal import models  # type: ignore
from internal import train_utils  # type: ignore
from internal import camera_utils  # type: ignore
from internal.datasets import Dataset as GoBaseDataset  # type: ignore


def apply_transform(transform, poses):
    # Get transform and scale
    assert len(transform.shape) == 2, "Transform should be a 4x4 or a 3x4 matrix."
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    assert np.allclose(scale, scale[0], rtol=1e-3, atol=0)
    scale = float(np.mean(scale).item())
    transform = transform.copy()
    transform[:3, :] /= scale

    # Pad poses
    bottom = np.broadcast_to([0, 0, 0, 1.0], poses[..., :1, :4].shape)
    poses = np.concatenate([poses[..., :3, :4], bottom], axis=-2)

    # Apply transform
    poses = transform @ poses

    # Unpad poses
    poses = poses[..., :3, :4]

    # Scale translation
    poses[..., :3, 3] *= scale
    return poses


def numpy_from_base64(data: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(data)) as f:
        return np.load(f)


def get_features(images, rate=1):
    import torch  # type: ignore
    from torchvision import transforms as T  # type: ignore

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    dinov2_vits14.to(device)
    extractor = dinov2_vits14

    IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

    all_features = []
    for img in tqdm(images, desc='Extracting features'):
        H, W = img.shape[:2]
        RESIZE_H = (H // rate) // 14 * 14
        RESIZE_W = (W // rate) // 14 * 14
        pil_img = Image.fromarray(img).convert('RGB')
        transform = T.Compose([
            T.Resize((RESIZE_H, RESIZE_W)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        img = transform(pil_img)[:3].unsqueeze(0)
        with torch.no_grad():
            features_dict = extractor.forward_features(img.cuda())
            features = features_dict['x_norm_patchtokens'].view(RESIZE_H // 14, RESIZE_W // 14, -1)
        all_features.append(features.detach().cpu().numpy())
    return all_features


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
        try:
            return int(v, 0)  # int
        except ValueError:
            pass
        try:
            return float(v)  # float
        except ValueError:
            pass
        return str(v)

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
    def __init__(self, dataset: Union[None, Dataset, Dict], config, eval=False, dataparser_transform=None):
        self.dataset = dataset
        self._rendering_mode = eval
        self.dataparser_transform = dataparser_transform
        is_render = config.is_render
        try:
            config.is_render = eval
            super().__init__("train" if not eval else "test", None, config)
        finally:
            config.is_render = is_render

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
            np.all(dataset["cameras"].camera_models[:1] == dataset["cameras"].camera_models) and
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
        if dataset["cameras"].camera_models[0].item() in (camera_model_to_int("opencv"), camera_model_to_int("opencv_fisheye")):
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
        features = np.zeros((0, 384), dtype=np.float32)
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

        if self.dataparser_transform is not None:
            self.colmap_to_world_transform = transform = self.dataparser_transform
            poses = apply_transform(self.dataparser_transform, poses)
        else:
            # Rotate/scale poses to align ground with xy plane and fit to unit cube.
            poses, transform = camera_utils.transform_poses_pca(poses)
            self.colmap_to_world_transform = transform[:3, :4]

        self.poses = poses
        self.images = images
        self.camtoworlds = poses
        self.height, self.width = images.shape[1:3]

        self.features = features
        self.assignments = assignments

    def start(self):
        if self.split == "train":
            return super().start()

    def __iter__(self):
        if self.split == "train":
            return super().__iter__()
        return self._iter_eval()

    def _iter_eval(self):
        for i in range(self._n_examples):
            yield self.generate_ray_batch(i)


class NeRFOnthego(Method):
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
        self.model = None
        self._config_str = None
        self._dataparser_transform = None

        # Setup config
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

        self._setup(train_dataset)

    def _load_config(self, config_overrides=None):
        if self.checkpoint is None:
            # Find the config files root
            import train  # type: ignore

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
    def get_method_info(cls):
        return MethodInfo(
            required_features=frozenset(("color",)),
            supported_camera_models=frozenset(("pinhole", "opencv", "opencv_fisheye")),
            supported_outputs=("color", "depth", "accumulation"),
        )

    def get_info(self):
        return ModelInfo(
            num_iterations=self.config.max_steps,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=gin_config_to_dict(self._config_str or ""),
            **self.get_method_info()
        )

    def _setup(self, train_dataset: Optional[Dataset]):
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
            _, state, self.render_eval_pfn, _, _ = train_utils.setup_model(self.config, rng)
            state = checkpoints.restore_checkpoint(self.config.checkpoint_dir, state)
            step = int(state.step)
            assert step == self.get_info().get("loaded_step", step), f"Loaded step {step} does not match expected step {self.get_info().get('loaded_step', step)}"

            variables = state.params
            state, self.lr_fn = train_utils.create_optimizer(self.config, variables)
            self.state = state

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

    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        del options
        camera = camera.item()
        cameras = camera[None]
        # Test-set evaluation.
        # We reuse the same random number generator from the optimization step
        # here on purpose so that the visualization matches what happened in
        # training.
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

        test_case = next(iter(test_dataset))
        rendering = models.render_image(functools.partial(self.render_eval_pfn, eval_variables, self.train_frac), test_case.rays, None, self.config, verbose=False)

        accumulation = rendering["acc"]
        color = rendering["rgb"]
        depth = np.array(rendering["distance_mean"], dtype=np.float32)
        assert len(accumulation.shape) == 2
        assert len(depth.shape) == 2
        return cast(RenderOutput, {
            "color": np.array(color, dtype=np.float32),
            "depth": np.array(depth, dtype=np.float32),
            "accumulation": np.array(accumulation, dtype=np.float32),
        })

