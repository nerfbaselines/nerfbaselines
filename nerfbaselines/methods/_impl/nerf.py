import shlex
import json
from pathlib import Path
from typing import Any, Dict, Iterable
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
from run_nerf import get_rays, render, get_rays_np, config_parser, create_nerf
from load_llff import load_llff_data, spherify_poses, poses_avg
from load_deepvoxels import load_dv_data
from load_blender import load_blender_data
import tempfile
from argparse import ArgumentParser

from nerfbaselines.types import Dataset, CurrentProgress, RenderOutput, MethodInfo, ModelInfo, ProgressCallback
from nerfbaselines import Cameras, CameraModel
from nerfbaselines import Method
from nerfbaselines.types import Optional
from nerfbaselines.utils import padded_stack, convert_image_dtype
from nerfbaselines.pose_utils import pad_poses, apply_transform, unpad_poses, invert_transform


tf.compat.v1.enable_eager_execution()
def shlex_join(split_command):
    """Return a shell-escaped string from *split_command*."""
    return ' '.join(shlex.quote(arg) for arg in split_command)


def load_dataset(args, dataset: Dataset, transform_args=None):
    poses = dataset.cameras.poses.copy()
    imgs = None
    if dataset.images is not None:
        imgs = np.stack(dataset.images, 0)
        imgs = convert_image_dtype(imgs, np.float32)

    # Convert from OpenCV to OpenGL coordinate system
    poses[..., 1:3] *= -1
    poses = poses.astype(np.float32)
    W, H = dataset.cameras.image_sizes[0]
    focal = dataset.cameras.intrinsics[0, 0]
    if args.dataset_type == "blender":
        assert (
            np.all(dataset.cameras.image_sizes[..., 0] == W) and
            np.all(dataset.cameras.image_sizes[..., 1] == H) and
            np.all(dataset.cameras.intrinsics[..., 0] == focal) and
            np.all(dataset.cameras.intrinsics[..., 1] == focal)
        ), "All images must have the same width, height, and focal lenghts"
        cx, cy = W / 2, H / 2
        assert (
            np.all(dataset.cameras.intrinsics[..., 2] == cx) and
            np.all(dataset.cameras.intrinsics[..., 3] == cy)
        ), "All images must have the same principal point in the center of the image"

        near = 2.
        far = 6.

        if imgs is not None:
            if args.white_bkgd:
                imgs = imgs[..., :3]*imgs[..., -1:] + (1.-imgs[..., -1:])
            else:
                imgs = imgs[..., :3]
        transform_args = {
            "transform": np.eye(4, dtype=np.float32),
            "hwfnearfarscale": (int(H), int(W), float(focal), near, far, 1.)
        }
        return imgs, poses[:, :3, :4], transform_args
    # Load data
    elif args.dataset_type == 'llff':
        if transform_args is None:
            recenter=True
            spherify=args.spherify

            bds = dataset.cameras.nears_fars
            print('Loaded', bds.min(), bds.max())
            
            # Rescale if bd_factor is provided
            near_original = dataset.cameras.nears_fars.min()
            bd_factor=.75  # 0.75 is the default parameter
            sc = 1 / (near_original * bd_factor)
            poses[:,:3,3] *= sc
            bds *= sc

            transform = np.eye(4, dtype=np.float32)
            
            if recenter:
                transform = np.linalg.inv(pad_poses(poses_avg(poses)))
                poses = apply_transform(transform, poses)
                
            if spherify:
                poses, render_poses, bds = spherify_poses(poses, bds)
            else:
                # Find a reasonable "focus depth" for this dataset
                close_depth, inf_depth = bds.min()*.9, bds.max()*5.
                dt = .75
                mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
                focal = mean_dz

            if args.no_ndc:
                near = np.min(bds) * .9
                far = np.max(bds) * 1.
            else:
                near = 0.
                far = 1.
        else:
            transform = transform_args["transform"]
            H, W, focal, near, far, sc = transform_args["hwfnearfarscale"]
            poses[:,:3,3] *= sc
            poses = apply_transform(transform, poses)
            if spherify:
                poses, _, _ = spherify_poses(poses, 1)
        transform_args = {
            "transform": transform,
            "hwfnearfarscale": (int(H), int(W), float(focal), near, far, float(sc))
        }
        print('Data:')
        print(poses.shape)
        if imgs is not None:
            print(imgs.shape, imgs.min(), imgs.max())
        print('Loaded llff', imgs.shape, (H, W, focal))
        print('DEFINING BOUNDS')
        print('NEAR FAR', near, far)
        return imgs, poses[:, :3, :4], transform_args
    else:
        raise RuntimeError('Unsupported dataset type', args.dataset_type)


class NeRF(Method):
    _method_name: str = "nerf"

    def __init__(self, *,
                 checkpoint: Optional[str] = None, 
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.args = None
        self.metadata = {}
        self._arg_list = ()
        if checkpoint is not None:
            with open(os.path.join(checkpoint, "metadata.json"), "r") as f:
                self.metadata = json.load(f)
                self.metadata["transform_args"]["transform"] = np.array(self.metadata["transform_args"]["transform"], dtype=np.float32)
            self._arg_list = shlex.split(self.metadata["args"])
        print(train_dataset, checkpoint)
        self.step = 0

        self._load_config()
        self._setup(train_dataset, config_overrides=config_overrides)

    def _load_config(self):
        parser: ArgumentParser = config_parser()
        self.args = parser.parse_args(self._arg_list)

    @classmethod
    def get_method_info(cls) -> MethodInfo:
        assert cls._method_name is not None, "Method was not properly registered"
        return MethodInfo(
            name=cls._method_name,
            required_features=frozenset(("color",)),
            supported_camera_models=frozenset(CameraModel.__members__.values()),
        )

    def get_info(self) -> ModelInfo:
        N_iters = 1000000

        loaded_step = None
        if self.checkpoint is not None:
            ckpts = [os.path.join(self.checkpoint, f) for f in sorted(os.listdir(os.path.join(self.checkpoint))) if
                        ('model_' in f and 'fine' not in f and 'optimizer' not in f)]
            if len(ckpts) > 0:
                ft_weights = ckpts[-1]
                loaded_step = int(ft_weights[-10:-4])

        return ModelInfo(
            name=self._method_name,
            num_iterations=N_iters,
            supported_camera_models=frozenset(CameraModel.__members__.values()),
            loaded_step=loaded_step,
            loaded_checkpoint=self.checkpoint,
            batch_size=self.args.N_rand,
            eval_batch_size=self.args.N_rand,
            hparams=vars(self.args) if self.args else {},
        )

    def save(self, path: str):
        with open(str(path) + "/args.txt", "w") as f:
            f.write(shlex_join(self._arg_list))
        def save_weights(net, prefix, i):
            mpath = os.path.join(path, '{}_{:06d}.npy'.format(prefix, i))
            np.save(mpath, net.get_weights())
            print('saved weights at', path)
        for k in self.models:
            save_weights(self.models[k], k, self.step)

        self.metadata["args"] = shlex_join(self._arg_list)
        metadata = self.metadata.copy()
        metadata["transform_args"]["transform"] = metadata["transform_args"]["transform"].tolist()
        with (Path(path) / "metadata.json").open("w") as f:
            json.dump(metadata, f)

    def _setup(self, train_dataset: Dataset, *, config_overrides: Optional[Dict[str, Any]] = None):
        if train_dataset is not None:
            config_overrides = (config_overrides or {}).copy()

            self.metadata["dataset_metadata"] = {
                "type": train_dataset.metadata.get("type"),
                "name": train_dataset.metadata.get("name"),
            }

            # Load dataset-specific config
            dataset_name = train_dataset.metadata.get("name")
            if dataset_name == "blender":
                config_name = "blender_config.txt"
            elif dataset_name == "llff":
                config_name = "llff_config.txt"
            else:
                raise RuntimeError(f"Unsupported dataset {dataset_name}")
            config_file = Path(run_nerf.__file__).absolute().parent.joinpath("paper_configs", config_name)
            logging.info(f"Loading config from {config_file}")
            with config_file.open("r", encoding="utf8") as f:
                config_overrides.update(configargparse.DefaultConfigFileParser().parse(f))

            for k, v in config_overrides.items():
                if isinstance(v, list):
                    for vs in v:
                        self._arg_list += (f"--{k}", str(vs))
                elif v == "True":
                    if v:
                        self._arg_list += (f"--{k}",)
                elif isinstance(v, str) and v.startswith("[") and v.endswith("]"):
                    for vs in v[1:-1].split(","):
                        self._arg_list += (f"--{k}", str(vs))
                else:
                    self._arg_list += (f"--{k}", str(v))
            logging.info("Using arguments: " + shlex_join(self._arg_list))
        self._load_config()

        if self.args.random_seed is not None:
            print('Fixing random seed', self.args.random_seed)
            np.random.seed(self.args.random_seed)
            tf.compat.v1.set_random_seed(self.args.random_seed)

        # Load data
        if train_dataset is not None:
            images, poses, self.metadata["transform_args"] = load_dataset(self.args, train_dataset, self.metadata.get("transform_args"))
        (H, W, focal, near, far, sc) = self.metadata["transform_args"]["hwfnearfarscale"]

        # Cast intrinsics to right types
        H, W = int(H), int(W)
        self.focal = focal

        # Create nerf model
        with tempfile.TemporaryDirectory() as basedir:
            self.args.basedir, self.args.expname = os.path.split(basedir)
            if self.checkpoint is not None:
                step = self.get_info().get("loaded_step")
                assert step is not None, f"Could not find valid checkpoint in path {self.checkpoint}"
                self.args.ft_path = os.path.join(self.checkpoint, f"model_{step:06d}.npy")
            render_kwargs_train, render_kwargs_test, start, self.grad_vars, self.models = create_nerf(
                self.args)
            self.args.basedir = self.args.exp = None

        bds_dict = {
            'near': tf.cast(near, tf.float32),
            'far': tf.cast(far, tf.float32),
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test

        # Create optimizer
        lrate = self.args.lrate
        if self.args.lrate_decay > 0:
            lrate = tf.keras.optimizers.schedules.ExponentialDecay(
                lrate,
                decay_steps=self.args.lrate_decay * 1000, 
                decay_rate=0.1)
        self.optimizer = tf.keras.optimizers.Adam(lrate)
        self.models['optimizer'] = self.optimizer

        self.step = start
        self.global_step = tf.compat.v1.train.get_or_create_global_step()
        self.global_step.assign(self.step)

        # Prepare raybatch tensor if batching random rays
        use_batching = not self.args.no_batching
        if train_dataset is not None:
            self.images = images
            self.poses = poses
            self.rays_rgb = None
            self.i_batch = None
            if use_batching:
                # For random ray batching.
                #
                # Constructs an array 'rays_rgb' of shape [N*H*W, 3, 3] where axis=1 is
                # interpreted as,
                #   axis=0: ray origin in world space
                #   axis=1: ray direction in world space
                #   axis=2: observed RGB color of pixel
                logging.debug('get rays')
                # get_rays_np() returns rays_origin=[H, W, 3], rays_direction=[H, W, 3]
                # for each pixel in the image. This stack() adds a new dimension.
                rays = [get_rays_np(H, W, focal, p) for p in poses[:, :3, :4]]
                rays = np.stack(rays, axis=0)  # [N, ro+rd, H, W, 3]
                logging.debug('done, concats')
                # [N, ro+rd+rgb, H, W, 3]
                rays_rgb = np.concatenate([rays, images[:, None, ...]], 1)
                # [N, H, W, ro+rd+rgb, 3]
                rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])
                rays_rgb = np.stack(rays_rgb, axis=0)  # train images only
                # [(N-1)*H*W, ro+rd+rgb, 3]
                rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])
                rays_rgb = rays_rgb.astype(np.float32)
                logging.debug('shuffle rays')
                np.random.shuffle(rays_rgb)
                logging.debug('done')

                self.i_batch = 0
                self.rays_rgb = rays_rgb
        logging.debug('Begin')

    def train_iteration(self, step: int):
        self.step = step
        self.global_step.assign(self.step)
        # Sample random ray batch

        use_batching = not self.args.no_batching
        N_rand = self.args.N_rand
        if use_batching:
            # Random over all images
            batch = self.rays_rgb[self.i_batch:self.i_batch+N_rand]  # [B, 2+1, 3*?]
            batch = tf.transpose(batch, [1, 0, 2])

            # batch_rays[i, n, xyz] = ray origin or direction, example_id, 3D position
            # target_s[n, rgb] = example_id, observed color.
            batch_rays, target_s = batch[:2], batch[2]

            self.i_batch += N_rand
            if self.i_batch >= self.rays_rgb.shape[0]:
                np.random.shuffle(self.rays_rgb)
                self.i_batch = 0

        else:
            # Random from one image
            img_i = np.random.choice(list(range(len(self.poses))))
            target = self.images[img_i]
            H, W, _ = target.shape
            pose = self.poses[img_i, :3, :4]

            if N_rand is not None:
                rays_o, rays_d = get_rays(H, W, self.focal, pose)
                if step < self.args.precrop_iters:
                    dH = int(H//2 * self.args.precrop_frac)
                    dW = int(W//2 * self.args.precrop_frac)
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H//2 - dH, H//2 + dH), 
                        tf.range(W//2 - dW, W//2 + dW), 
                        indexing='ij'), -1)
                    if step < 10:
                        print('precrop', dH, dW, coords[0,0], coords[-1,-1])
                else:
                    coords = tf.stack(tf.meshgrid(
                        tf.range(H), tf.range(W), indexing='ij'), -1)
                coords = tf.reshape(coords, [-1, 2])
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)
                select_inds = tf.gather_nd(coords, select_inds[:, tf.newaxis])
                rays_o = tf.gather_nd(rays_o, select_inds)
                rays_d = tf.gather_nd(rays_d, select_inds)
                batch_rays = tf.stack([rays_o, rays_d], 0)
                target_s = tf.gather_nd(target, select_inds)

        #####  Core optimization loop  #####

        with tf.GradientTape() as tape:

            # Make predictions for color, disparity, accumulated opacity.
            rgb, disp, acc, extras = render(
                H, W, self.focal, chunk=self.args.chunk, rays=batch_rays,
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

    def render(self, cameras: Cameras, progress_callback: Optional[ProgressCallback] = None) -> Iterable[RenderOutput]:
        _, poses, _ = load_dataset(self.args,
            Dataset(
                cameras=cameras,
                file_paths=[f"{i:06d}.png" for i in range(len(cameras))],
                metadata=self.metadata["dataset_metadata"],
            ),
            transform_args=self.metadata["transform_args"]
        )
        idx = 0
        if progress_callback is not None:
            progress_callback(CurrentProgress(idx, len(cameras), idx, len(cameras)))
        for idx, pose in enumerate(poses):
            W, H = cameras.image_sizes[idx]
            focal, *_ = cameras.intrinsics[idx]
            pose = cameras.poses[idx, :3, :4]
            rgb, disp, acc, extras = render(
                H, W, focal, chunk=self.args.chunk, c2w=pose, **self.render_kwargs_test)
            if progress_callback is not None:
                progress_callback(CurrentProgress(idx + 1, len(cameras), idx + 1, len(cameras)))
            rgb = np.clip(rgb.numpy(), 0.0, 1.0)
            yield {
                "color": rgb,
                "accumulation": acc.numpy(),
                "depth": extras["depth"].numpy(),
            }
