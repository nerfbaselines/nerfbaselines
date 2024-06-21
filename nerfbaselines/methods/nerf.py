# NOTE: We extended NeRF to work with variable image sizes and intrinsics. Also, we support different camera models and varying image sizes.
# TODO: write transforms for custom datasets
# TODO: rewrite code to allow different intrinsics and camera sizes per image
import warnings
import shlex
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence
import logging
try:
    import torch as _
except ImportError:
    pass
import os
import configargparse
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf
import numpy as np
import imageio
import run_nerf
from run_nerf_helpers import img2mse, mse2psnr, to8b
from run_nerf import render, get_rays_np, config_parser, create_nerf
from load_llff import load_llff_data, spherify_poses, poses_avg
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
import tempfile
from argparse import ArgumentParser

from nerfbaselines.types import Dataset, OptimizeEmbeddingsOutput, RenderOutput, MethodInfo, ModelInfo
from nerfbaselines.types import Cameras, CameraModel, get_args
from nerfbaselines import Method
from nerfbaselines.types import Optional
from nerfbaselines.utils import convert_image_dtype
from nerfbaselines.pose_utils import pad_poses, apply_transform, unpad_poses, invert_transform
from nerfbaselines import cameras as _cameras

# Setup TF GPUs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.compat.v1.enable_eager_execution()


def shlex_join(split_command):
    """Return a shell-escaped string from *split_command*."""
    return ' '.join(shlex.quote(arg) for arg in split_command)


def transform_images(args, images):
    # Transform images
    images = [convert_image_dtype(img, np.float32) for img in images]
    return [
        img[..., :3]*img[..., -1:] + (1.-img[..., -1:]) if args.white_bkgd and img.shape[-1] == 4 
        else img[..., :3] 
        for img in images]


def transform_cameras(args, cameras, transform_args, verbose=False):
    poses = cameras.poses.copy()

    near, far = 0, 1
    spherify = False
    if transform_args is not None:
        transform = np.array(transform_args["transform"], dtype=np.float32)
        near, far = transform_args["near_far"]
        poses = apply_transform(transform, poses)
        spherify = transform_args["spherify"]
        if spherify:
            poses, _, _ = spherify_poses(poses, 1)
    elif args.dataset_type == 'blender':
        sc = 1
        near = cameras.nears_fars[..., 0].min()
        far = cameras.nears_fars[..., 1].max()
        transform = np.eye(4, dtype=np.float32)
    elif args.dataset_type == 'llff':
        recenter=True
        spherify=args.spherify

        bds = cameras.nears_fars
        print('Loaded', bds.min(), bds.max())
        
        # Rescale if bd_factor is provided
        near_original = cameras.nears_fars.min()
        bd_factor=.75  # 0.75 is the default parameter
        sc = 1 / (near_original * bd_factor)
        poses[:,:3,3] *= sc
        bds *= sc

        transform = np.eye(4, dtype=np.float32)
        
        if recenter:
            transform = np.linalg.inv(pad_poses(poses_avg(poses)))
            poses = apply_transform(transform, poses)
        if spherify:
            poses, _, bds = spherify_poses(poses, bds)
        if args.no_ndc:
            near = np.min(bds) * .9
            far = np.max(bds) * 1.
        else:
            near = 0.
            far = 1.
        # Pre-multiply by sc
        transform[:3, :3] *= sc
    else:
        raise RuntimeError('Unsupported dataset type', args.dataset_type)

    transform_args = {
        "transform": transform[:3, :4].tolist(),
        "near_far": [float(near), float(far)],
        "spherify": spherify,
    }

    # Replace cameras
    cameras = cameras.replace(
        poses=poses,
        nears_fars=np.stack([np.full(len(poses), near), np.full(len(poses), far)], -1),
    )

    if verbose:
        print('Data:')
        print(poses.shape)
        print('DEFINING BOUNDS')
        print('NEAR FAR', near, far)
    return cameras, transform_args


def get_rays(cameras, images=None):
    def _load_image(w, h, i):
        if images is None:
            return np.zeros((h, w, 3), dtype=np.float32)
        img = images[i]
        return img
    rays = [np.concatenate(
        _cameras.unproject(cameras[i], np.stack(np.meshgrid(
            np.arange(w, dtype=np.float32), 
            np.arange(h, dtype=np.float32), 
            indexing="xy"), -1)
        ) + (_load_image(w, h, i),), -1).reshape(-1, 3, 3) for i, (w, h) in enumerate(cameras.image_sizes)]
    return np.concatenate(rays, axis=0)


class NeRF(Method):
    _method_name: str = "nerf"

    def __init__(self, *,
                 checkpoint: Optional[str] = None, 
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.args = None
        self.transform_args = None
        self._arg_list = None
        if checkpoint is not None:
            if not os.path.exists(os.path.join(checkpoint, "transforms.json")):
                if train_dataset is None:
                    raise RuntimeError("Could not find transforms.json in checkpoint, please provide a train dataset to infer transforms (if your checkpoint was not trained with nerfbaselines).")
                warnings.warn(f"Could not find transforms.json in {checkpoint}. " 
                              "This should only happen when fixing a checkpoint not trained with nefbaselines.")

            else:
                with open(os.path.join(checkpoint, "transforms.json"), "r") as f:
                    self.transform_args = json.load(f)
        self.step = 0

        self._setup(train_dataset, config_overrides=config_overrides)

    @classmethod
    def get_method_info(cls) -> MethodInfo:
        assert cls._method_name is not None, "Method was not properly registered"
        return MethodInfo(
            name=cls._method_name,
            required_features=frozenset(("color",)),
            supported_camera_models=frozenset(get_args(CameraModel)),
        )

    def get_info(self) -> ModelInfo:
        N_iters = 1000000

        loaded_step = None
        if self.checkpoint is not None:
            ckpts = [os.path.join(self.checkpoint, f) for f in sorted(os.listdir(os.path.join(self.checkpoint))) if
                     (f.startswith("model_") and 'fine' not in f and f.endswith('.npy'))]
            if len(ckpts) > 0:
                ft_weights = ckpts[-1]
                loaded_step = int(os.path.split(ft_weights)[-1][6:-4])

        hparams = vars(self.args).copy() if self.args else {}
        hparams.pop("config", None)
        hparams.pop("basedir", None)
        hparams.pop("expname", None)
        hparams.pop("ft_path", None)
        return ModelInfo(
            **self.get_method_info(),
            num_iterations=N_iters,
            loaded_step=loaded_step,
            loaded_checkpoint=self.checkpoint,
            batch_size=self.args.N_rand,
            eval_batch_size=self.args.N_rand,
            hparams=hparams,
        )

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        def save_weights(net, prefix, i):
            mpath = os.path.join(path, '{}_{:06d}.npy'.format(prefix, i))
            if type(net).__name__ == "Adam" and not hasattr(net, "get_weights"):
                # Fix missing get_weights function in newer versions of TF
                weights = [w.numpy() for w in net._variables]
            else:
                weights = net.get_weights()
            np.save(mpath, np.array(weights, dtype=object))
            print('saved weights at', path)
        for k in self.models:
            save_weights(self.models[k], k, self.step)

        with open(os.path.join(path, "args.txt"), 'w') as file:
            for arg in sorted(self._arg_list):
                file.write('{} = {}\n'.format(arg, self._arg_list[arg]))

        with open(os.path.join(path, "config.txt"), 'w') as file:
            file.write(self._base_config_text)

        with (Path(path) / "transforms.json").open("w") as f:
            print(self.transform_args)
            json.dump(self.transform_args, f)

    def _setup(self, train_dataset: Dataset, *, config_overrides: Optional[Dict[str, Any]] = None):
        config_overrides = (config_overrides or {}).copy()
        if self.checkpoint is not None:
            config_overrides.pop("config", None)
            config_file = os.path.join(self.checkpoint, "config.txt")
            self._base_config_text = open(config_file, 'r', encoding='utf8').read()
            with Path(self.checkpoint).joinpath("args.txt").open("r", encoding="utf8") as f:
                config_overrides.update(configargparse.DefaultConfigFileParser().parse(f))
        elif train_dataset is not None:
            # Load dataset-specific config
            config_file = config_overrides.pop("config", None)
            if config_file is not None:
                config_file = Path(run_nerf.__file__).absolute().parent.joinpath("paper_configs", config_file)
            else:
                # TODO: Add default config for object-centric datasets
                raise RuntimeError(f"Could not find config file, please specify --set config=...")

            logging.info(f"Loading config from {config_file}")
            with config_file.open("r", encoding="utf8") as f:
                old_overrides = config_overrides.copy()
                config_overrides.update(configargparse.DefaultConfigFileParser().parse(f))
                config_overrides.update(old_overrides)
            self._base_config_text = open(config_file, 'r', encoding='utf8').read()
        else:
            raise RuntimeError("Either train_dataset or checkpoint must be provided")

        with tempfile.NamedTemporaryFile("w") as f:
            self._arg_list = config_overrides
            for k, v in config_overrides.items():
                f.write(f"{k} = {v}\n")
            f.flush()
            f.seek(0)
            parser: ArgumentParser = config_parser()
            self.args = parser.parse_args(["--config", f.name])
        logging.info("Using arguments: \n" + "\n".join(f"   {k} = {v}" for k, v in vars(self.args).items()))

        if self.args.random_seed is not None:
            print('Fixing random seed', self.args.random_seed)
            np.random.seed(self.args.random_seed)
            tf.compat.v1.set_random_seed(self.args.random_seed)

        # Create nerf model
        with tempfile.TemporaryDirectory() as basedir:
            self.args.basedir, self.args.expname = os.path.split(basedir)
            if self.checkpoint is not None:
                step = self.get_info().get("loaded_step")
                assert step is not None, f"Could not find valid checkpoint in path {self.checkpoint}"
                self.args.ft_path = os.path.join(self.checkpoint, f"model_{step:06d}.npy")

            # NOTE: There is a bug in the original code which doesn't allow the model to load the checkpoint
            # We patched it here
            old_no_reload = self.args.no_reload
            try:
                self.args.no_reload = True
                render_kwargs_train, render_kwargs_test, start, self.grad_vars, self.models = create_nerf(
                    self.args)
            finally:
                self.args.no_reload = old_no_reload

            if self.checkpoint and not self.args.no_reload:
                ckpts = [os.path.join(self.checkpoint, f) for f in sorted(os.listdir(os.path.join(self.checkpoint))) if f.startswith("model_") and f.endswith('.npy')]
                for ckpt in ckpts:
                    if os.path.split(ckpt)[-1].startswith('model_fine_'):
                        print('Reloading fine from', ckpt)
                        model = self.models['model_fine']
                    elif os.path.split(ckpt)[-1].startswith('model_'):
                        step = int(os.path.split(ckpt)[-1][6:-4])
                        print('Resetting step to', step)
                        start = step + 1
                        print('Reloading from', ckpt)
                        model = self.models['model']
                    else:
                        raise RuntimeError('Invalid checkpoint name', ckpt)
                    model.set_weights(np.load(ckpt, allow_pickle=True))
                if len(ckpts) < 2:
                    raise RuntimeError('Invalid checkpoint (missing files)', self.checkpoint)

            self.args.basedir = self.args.exp = None

        # Create optimizer
        lrate = self.args.lrate
        if self.args.lrate_decay > 0:
            lrate = tf.keras.optimizers.schedules.ExponentialDecay(
                lrate,
                decay_steps=self.args.lrate_decay * 1000, 
                decay_rate=0.1)
        self.optimizer = tf.keras.optimizers.Adam(lrate)
        self.models['optimizer'] = self.optimizer
        if self.checkpoint is not None:
            step = self.get_info().get("loaded_step")
            optimpath = os.path.join(self.checkpoint, f"optimizer_{step:06d}.npy")
            optimweights = np.load(optimpath, allow_pickle=True)
            if len(optimweights) > 0:
                self.optimizer.apply_gradients(zip([tf.zeros_like(x) for x in self.grad_vars], self.grad_vars))
                self.models["optimizer"].set_weights(optimweights)
                logging.info(f"Loaded optimizer state from {optimpath}")
            else:
                logging.warning(f"Could not load optimizer state from {optimpath}, length was 0")

        self.step = start - 1
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.global_step.assign(self.step)

        # Prepare raybatch tensor if batching random rays
        use_batching = not self.args.no_batching
        if train_dataset is not None:
            self._train_images = transform_images(self.args, train_dataset["images"])
            self._train_cameras, self.transform_args = transform_cameras(self.args, train_dataset["cameras"], self.transform_args, verbose=True)
            self._train_rays_rgb = None
            self._train_i_batch = None

            # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
            # interpreted as,
            #   axis=0: ray origin in world space
            #   axis=1: ray direction in world space
            #   axis=2: observed RGB color of pixel
            logging.debug('get rays')
            # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
            # for each pixel in the image. This stack() adds a new dimension.
            self._train_rays_rgb = get_rays(self._train_cameras, self._train_images).astype(np.float32)
            logging.debug('done, concats')

            if use_batching:
                logging.debug('shuffle rays')
                np.random.shuffle(self._train_rays_rgb)
                logging.debug('done')

                self._train_i_batch = 0
            else:
                self._train_rays_cumsum = np.cumsum([0] + [w*h for w, h in self._train_cameras.image_sizes])
                assert self._train_rays_cumsum[-1] == self._train_rays_rgb.shape[0], "Invalid rays shape"
            logging.info("Train rays cached")

        # Setup kwargs
        near, far = self.transform_args["near_far"]
        bds_dict = {
            'near': tf.cast(near, tf.float32),
            'far': tf.cast(far, tf.float32),
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test

        logging.debug('Begin')


    def train_iteration(self, step: int):
        self.step = step
        self.global_step.assign(self.step)
        # Sample random ray batch

        use_batching = not self.args.no_batching
        N_rand = self.args.N_rand
        if use_batching:
            # Random over all images
            batch = self._train_rays_rgb[self._train_i_batch:self._train_i_batch+N_rand]  # [B, 2+1, 3*?]

            self._train_i_batch += N_rand
            if self._train_i_batch >= self._train_rays_rgb.shape[0]:
                np.random.shuffle(self._train_rays_rgb)
                self._train_i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(list(range(len(self._train_cameras))))
            W, H = self._train_cameras.image_sizes[img_i]
            batch = self._train_rays_rgb[self._train_rays_cumsum[img_i]:self._train_rays_cumsum[img_i+1]]

            if N_rand is not None:
                if step < self.args.precrop_iters:
                    dH = int(H//2 * self.args.precrop_frac)
                    dW = int(W//2 * self.args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH), 
                        tf.range(W//2 - dW, W//2 + dW), 
                        indexing='ij'), -1)
                    if step < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                    batch = tf.gather_nd(batch, coords)
                select_inds = np.random.choice(
                    batch.shape[0], size=[N_rand], replace=False)
                batch = tf.gather_nd(batch, select_inds[:, tf.newaxis])

        # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
        # target_s[n, rgb] = example_id, observed color.
        rays_o, rays_d, target_s = np.moveaxis(batch, 1, 0)
        batch_rays = rays_o, rays_d
        #####  Core optimization loop  #####

        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            # TODO: implement NDC rays
            rgb, disp, acc, extras = render(
                None, None, None, chunk=self.args.chunk, rays=batch_rays,
                verbose=step < 10, retraw=True, **self.render_kwargs_train)

            # Compute MSE loss between predicted and true RGB.
            img_loss = img2mse(rgb, target_s)
            # trans = extras['raw'][..., -1]
            loss = img_loss
            psnr = mse2psnr(img_loss)

            # Add MSE loss for coarse-grained model
            if 'rgb0' in extras:
                img_loss0 = img2mse(extras['rgb0'], target_s)
                loss += img_loss0
                psnr0 = mse2psnr(img_loss0)

        gradients = tape.gradient(loss, self.grad_vars)
        self.optimizer.apply_gradients(zip(gradients, self.grad_vars))
        self.step = step + 1
        self.global_step.assign(self.step)
        return {
            "loss": loss.numpy().item(),
            "psnr": psnr.numpy().item(),
            "mse": img_loss.numpy().item(),
            "psnr0": psnr0.numpy().item(),
            "mse0": img_loss0.numpy().item(),
        }

    def render(self, cameras: Cameras, embeddings=None) -> Iterable[RenderOutput]:
        if embeddings is not None:
            raise NotImplementedError(f"Optimizing embeddings is not supported for method {self.get_method_info()['name']}")
        cameras, _ = transform_cameras(self.args, cameras, self.transform_args)
        for idx in range(len(cameras.poses)):
            W, H = cameras.image_sizes[idx]
            batch_rays = get_rays(cameras[idx:idx+1])
            rays_o, rays_d = batch_rays[:, 0], batch_rays[:, 1]
            rgb, disp, acc, extras = render(
                None, None, None, chunk=self.args.chunk, rays=(rays_o, rays_d), **self.render_kwargs_test)
            rgb = np.clip(rgb.numpy(), 0.0, 1.0)
            yield {
                "color": rgb.reshape(H, W, 3),
                "accumulation": acc.numpy().reshape(H, W),
                "depth": extras["depth"].numpy().reshape(H, W)
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

