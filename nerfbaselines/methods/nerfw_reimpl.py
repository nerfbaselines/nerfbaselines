# TODO: remove points3D requirement
import copy
import dataclasses
import tempfile
import sys
from collections import defaultdict
import contextlib
import os
from queue import Queue
from threading import Thread
import pickle
import tqdm
import math
from contextlib import contextmanager
import warnings
import glob
import logging
import numpy as np
from argparse import ArgumentParser
from functools import partial
from typing import Optional, Any
from nerfbaselines import (
    Method, Dataset, Cameras, RenderOutput,
    OptimizeEmbeddingOutput, 
    ModelInfo, MethodInfo, CameraModel,
)
from nerfbaselines.utils import invert_transform, pad_poses, convert_image_dtype
from nerfbaselines import cameras as _cameras
try:
    from typing import get_args
except ImportError:
    from typing_extensions import get_args

import pytorch_lightning as pl  # type: ignore
from pytorch_lightning import Trainer  # type: ignore
import torch  # type: ignore


from pytorch_lightning import loggers as _loggers  # type: ignore
try:
    from pytorch_lightning.loggers import TestTubeLogger as _  # type: ignore
    del _
except ImportError:
    _loggers.TestTubeLogger = _loggers.TensorBoardLogger  # type: ignore

from datasets.phototourism import PhototourismDataset as _PhototourismDataset  # type: ignore
from models.rendering import render_rays  # type: ignore
import train  # type: ignore


logger = logging.getLogger("nerfw-reimpl")
warnings.filterwarnings("ignore", ".*does not have many workers.*")


# We copy the Trainer class to remove the global_step property
class ETrainer(Trainer):
    global_step = 0
    current_epoch = 0


def flatten_hparams(hparams, *, separator: str = "/", _prefix: str = ""):
    flat = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)}
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator).items())
        else:
            flat[k] = v
    return flat


class CallbackIterator:
    def __init__(self, run):
        self.run_callback = run
        self._queue = Queue(maxsize=1)
        self._thread = None
        self._pause_sentinel = object()
        self._end_sentinel = object()
        self._exception_sentinel = object()
        self._result_sentinel = object()

    def _run(self):
        try:
            def _block(obj=None):
                self._queue.put((self._result_sentinel, obj))
                self._queue.put((self._pause_sentinel, None))
                self._queue.put((self._pause_sentinel, None))
            self.run_callback(_block)
            self._queue.put((self._end_sentinel, None))
        except Exception as e:
            self._queue.put((self._exception_sentinel, e))

    def __iter__(self):
        assert self._thread is None, "Iterator is already running"
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __next__(self):
        if self._thread is None:
            raise RuntimeError("Iterator is not running. Call iter() first")
        sentinel = self._pause_sentinel
        out = None
        while sentinel == self._pause_sentinel:
            sentinel, out = self._queue.get()
        if sentinel is self._end_sentinel:
            raise StopIteration()
        elif sentinel is self._exception_sentinel:
            assert isinstance(out, BaseException), "Invalid exception type"
            raise out
        elif sentinel is self._result_sentinel:
            return out
        else:
            raise RuntimeError("Invalid sentinel")



_patched = False

def patch_nerfw_rempl():
    global _patched
    if _patched:
        return
    _patched = True
    # Fix bug in the nerf_pl codebase
    # From the README:
    #   There is a difference between the paper: I didn't add the appearance embedding in the coarse model while it should. 
    #   Please change this line to self.encode_appearance = encode_appearance to align with the paper.
    from models.nerf import NeRF  # type: ignore
    if getattr(type(NeRF), "__name__", None) == "MagicMock":
        # Skip in tests
        return
    old_init = NeRF.__init__
    def new_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.encode_appearance = self.encode_appearance
    NeRF.__init__ = new_init

    # All memory is in cache, using more workers is just a waste
    from train import NeRFSystem  # type: ignore
    old_train_dataloader = NeRFSystem.train_dataloader
    def _train_dataloader(self):
        from train import DataLoader  # type: ignore
        old_Dataloader = DataLoader.__init__
        try:
            def _init(*args, **kwargs):
                kwargs["num_workers"] = 0
                kwargs.pop("prefetch_factor", None)
                return old_Dataloader(*args, **kwargs)
            DataLoader.__init__ = _init
            dl = old_train_dataloader(self)
        finally:
            DataLoader.__init__ = old_Dataloader
        assert dl.num_workers == 0, f"Failed to patch DataLoader. num_workers ({dl.num_workers}) != 0"
        logger.info("Patched DataLoader to num_workers=0 to avoid OOM")
        return dl
    NeRFSystem.train_dataloader = _train_dataloader

    try:
        # Patch for newer PL
        del train.NeRFSystem.validation_epoch_end
    except AttributeError:
        pass

    # Patch nerf_pl for newer PL
    @train.NeRFSystem.hparams.setter
    def hparams(self, hparams):
        self._set_hparams(hparams)
        # make a deep copy so there are no other runtime changes reflected
        self._hparams_initial = copy.deepcopy(self._hparams)

    train.NeRFSystem.hparams = hparams


patch_nerfw_rempl()


class CameraTransformer:
    def __init__(self, scale_factor, points3D_xyz=None):
        if points3D_xyz is not None:
            points3D_xyz = torch.from_numpy(points3D_xyz).float()
        self.scale_factor = scale_factor
        self.points3D_xyz = points3D_xyz
        self._nears_fars_cache: Any = None

    @staticmethod
    @torch.no_grad()
    def _get_nears_fars(points3D_xyz, cameras):
        # Compute near and far bounds for each image individually
        xyz_world = points3D_xyz
        xyz_world_h = torch.cat([xyz_world, torch.ones_like(xyz_world[..., :1])], -1)
        nears, fars = [], [] # {id_: distance}
        w2c_mats = [torch.from_numpy(pad_poses(invert_transform(cameras.poses[i]))).float() for i in range(len(cameras))]
        for i in range(len(cameras)):
            xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
            xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
            nears.append(torch.quantile(xyz_cam_i[:, 2], 0.01*0.1))
            fars.append(torch.quantile(xyz_cam_i[:, 2], 0.01*99.9))
        nears = torch.stack(nears).cpu().float().numpy()
        fars = torch.stack(fars).cpu().float().numpy()
        return nears, fars

    def save(self, checkpoint_data):
        checkpoint_data["scale_factor"] = self.scale_factor
        checkpoint_data["points3D_xyz"] = None
        if self.points3D_xyz is not None:
            checkpoint_data["points3D_xyz"] = self.points3D_xyz.cpu().numpy()

    @staticmethod
    def build_from_dataset(dataset):
        cameras = dataset["cameras"]
        points3D_xyz = torch.from_numpy(dataset["points3D_xyz"]).float()
        nears, fars = CameraTransformer._get_nears_fars(points3D_xyz, cameras)

        min_near = nears.min()
        max_far = fars.max()
        scale_factor = max_far/5 # so that the max far is scaled to 5
        logger.info(f"Min far: {min_near:.6f}, Max far: {max_far:.6f}, scale_factor: {scale_factor:.6f}")

        out = CameraTransformer(scale_factor, dataset["points3D_xyz"].copy())
        out._nears_fars_cache = (cameras.poses, (nears, fars))
        return out

    def get_rays(self, cameras, include_ts=False):
        device = torch.device("cpu")
        if self._nears_fars_cache is not None and np.array_equal(self._nears_fars_cache[0], cameras.poses):
            # Speedup: reuse the cached nears and fars for the training dataset
            nears, fars = self._nears_fars_cache[1]
        else:
            nears, fars = self._get_nears_fars(self.points3D_xyz, cameras)
        self._nears_fars_cache = None

        nears /= self.scale_factor
        fars /= self.scale_factor

        # Create rays
        num_rays = cameras.image_sizes.prod(-1).sum()
        all_rays = torch.zeros((num_rays, 8 if not include_ts else 9), dtype=torch.float32)
        offset = 0
        cameras_th = cameras.apply(lambda x, _: torch.from_numpy(x).to(device))
        for i in range(len(cameras)):
            xy = _cameras.get_image_pixels(cameras_th.image_sizes[i]).to(torch.float32)
            # We do not add +0.5 offset to pixel coords to be consistent with nerf_pl codebase
            # https://github.com/kwea123/nerf_pl/blob/52aeb387da64a9ad9a0f914ea9b049ffc598b20c/datasets/ray_utils.py#L5
            rays_o, rays_d = _cameras.unproject(cameras_th[i:i+1], xy)
            # Normalize directions as in the nerf_pl codebase
            rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
            rays_o = rays_o / float(self.scale_factor)
            rays_o, rays_d = rays_o.cpu(), rays_d.cpu()
            nrays_ = len(rays_o)
            # Save memory by reusing the same tensor
            all_rays[offset:offset+nrays_, 0:3] = rays_o
            all_rays[offset:offset+nrays_, 3:6] = rays_d
            all_rays[offset:offset+nrays_, 6] = float(nears[i])
            all_rays[offset:offset+nrays_, 7] = float(fars[i])
            if include_ts:
                all_rays[offset:offset+nrays_, 8] = i
            offset += nrays_
        assert offset == num_rays, f"Offset mismatch {offset} != {num_rays}"
        return all_rays


class PhototourismDataset(_PhototourismDataset):
    def __init__(self, dataset: Dataset, camera_transformer: CameraTransformer, *, split, **kwargs):
        if split != "train":
            return None
        self.dataset = dataset
        self.camera_transformer = camera_transformer
        self.dataset_images = None
        if dataset is not None:
            self.dataset_images = dataset["images"]
        super().__init__(**kwargs)
        assert self.img_downscale == 1, "img_downscale should be 1"

    def read_meta(self):
        if self.dataset is None:
            return
        cameras = self.dataset["cameras"]
        self.all_rays = None

        # Compress rays to save memory
        num_rays = cameras.image_sizes.prod(-1).sum()
        self.all_rays_d = torch.zeros((num_rays, 3), dtype=torch.float32)
        self.sparse_rays_onf = torch.zeros((len(cameras), 3 + 2), dtype=torch.float32)
        self.sparse_rays_i = torch.zeros((num_rays,), dtype=torch.long)

        # Needed here to fix __len__
        self.all_rays = self.all_rays_d

        offset = 0
        for i in range(len(cameras)):
            rays = self.camera_transformer.get_rays(cameras[i:i+1])
            self.all_rays_d[offset:offset+len(rays), :] = rays[:, 3:6]
            # Test if the assumptions are valid
            assert torch.allclose(rays[:, 6], rays[:1, 6]), f"Near mismatch {rays[:, 6]} != {rays[0, 6]}"
            assert torch.allclose(rays[:, 7], rays[:1, 7]), f"Far mismatch {rays[:, 7]} != {rays[0, 7]}"
            assert torch.allclose(rays[:, :3], rays[:1, :3]), f"Origins mismatch {rays[:, :3]} != {rays[:1, :3]}"
            self.sparse_rays_onf[i, :3] = rays[0, :3]
            self.sparse_rays_onf[i, 3:] = rays[0, 6:]
            self.sparse_rays_i[offset:offset+len(rays)] = i
            offset += len(rays)
        assert offset == num_rays, f"Offset mismatch {offset} != {num_rays}"

        self.N_images_train = len(cameras)

        self.all_rgbs = torch.zeros((num_rays, 3), dtype=torch.uint8)
        if self.dataset_images is not None:
            self.all_rgbs = torch.cat([torch.from_numpy(x).view(-1, 3) for x in self.dataset_images], 0)
        assert len(self.all_rgbs) == num_rays, f"RGBs and rays size mismatch {len(self.all_rgbs)} != {num_rays}"
        assert self.all_rgbs.dtype == torch.uint8, f"RGBs should be uint8, got {self.all_rgbs.dtype}"

        # Save some memory
        del self.dataset_images
        self.dataset.pop("images", None)  # type: ignore

        logger.info(f"Loaded {self.N_images_train} images for training")
        logger.info(f"Cached {num_rays} rays")

    def __getitem__(self, idx):
        rays_d = self.all_rays_d[idx]
        sparse_idx = self.sparse_rays_i[idx]
        rays_o = self.sparse_rays_onf[sparse_idx, :3]
        near_far = self.sparse_rays_onf[sparse_idx, 3:]
        rays = torch.cat([rays_o, rays_d, near_far], 0)
        return {
            'rays': rays,
            'ts': sparse_idx,
            'rgbs': self.all_rgbs[idx].float() / 255.0
        }


def _override_train_iteration(model, old, cb, *args, **kwargs):
    cb(None)
    metrics = {}
    def log(name, value, *args, **kwargs):
        del args, kwargs
        metrics[name] = value
    model.log, old_log = log, model.log
    try:
        loss = old(*args, **kwargs)
    finally:
        model.log = old_log
    cb((loss, metrics))
    return loss


def _system_setup(old_setup, train_dataset, camera_transformer, *args, **kwargs):
    # Patch datasets
    old_train_datasets = train.dataset_dict.copy()
    try:
        train.dataset_dict.clear()
        train.dataset_dict["phototourism"] = partial(PhototourismDataset, train_dataset, camera_transformer)
        old_setup(*args, **kwargs)
    finally:
        train.dataset_dict.clear()
        train.dataset_dict.update(old_train_datasets)


def get_opts(config_overrides):
    parse_args = ArgumentParser.parse_args
    try:
        ArgumentParser.parse_args = lambda self: self  # type: ignore
        parser = train.get_opts()
    except Exception as e:
        print(f"Failed to load options: {e}")
        raise RuntimeError(f"Failed to load options: {e}")
    finally:
        ArgumentParser.parse_args = parse_args

    # Set default values
    parser.add_argument("--steps_per_epoch", type=int, default=15000)
    parser.add_argument("--appearance_optim_finetune_tau", type=bool, default=True)
    parser.add_argument("--appearance_optim_steps", type=int, default=1000)
    parser.add_argument("--appearance_optim_lr", type=float, default=0.01)
    parser.set_defaults(
        dataset_name="phototourism",
        N_samples=256,
        N_importance=256,
        N_a = 48,
        N_tau = 16,
        chunk=16384,
        batch_size=1024,
        beta_min = 0.1,
        num_epochs=20,
        num_gpus=8,
        img_downscale=1,
        encode_a=True,
        encode_t=True,
        appearance_optim_finetune_tau=True,
    )
    hparams = parser.parse_args(["--root_dir", "<empty>"])
    type_registry = {x.option_strings[-1].lstrip("-"): x.type for x in parser._actions}
    if config_overrides is not None:
        for k, v in config_overrides.items():
            vtype = type_registry.get(k, None)
            if vtype is not None:
                v = vtype(v)
            setattr(hparams, k, v)
    return hparams


class NeRFWReimpl(Method):
    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        self._train_iterator = None
        self.checkpoint = checkpoint

        self._loaded_step = None
        self._num_pixels = None
        self._setup(train_dataset, config_overrides)

    def _setup(self, train_dataset, config_overrides):
        # Validate config overrides
        if config_overrides is not None:
            assert "N_vocab" not in config_overrides, "N_vocab cannot be overridden"

        # Setup parameters and camera transformer
        ckpt_file = None
        self._num_pixels = None
        ckpt_data = None
        if self.checkpoint is not None:
            # Get checkpoint path
            ckpts = glob.glob(os.path.join(self.checkpoint, "*.ckpt"))
            assert len(ckpts) == 1, f"Expected one checkpoint file, got {len(ckpts)}"
            ckpt_file = ckpts[0]
            ckpt_data = torch.load(ckpt_file, map_location="cpu")
            hparams = get_opts(config_overrides)
            for k, v in ckpt_data["hyper_parameters"].items():
                setattr(hparams, k, v)
            self._loaded_step = int(ckpt_data["global_step"])
            self._num_pixels = ckpt_data.get("num_pixels", 10)
            camera_transformer_data = ckpt_data.get("camera_transformer", None)
            if camera_transformer_data is not None:
                self.camera_transformer = CameraTransformer(**camera_transformer_data)
            else:
                if train_dataset is not None:
                    warnings.warn("Camera transformer not found in the checkpoint, trying to reconstruct it from the dataset")
                    self.camera_transformer = CameraTransformer.build_from_dataset(train_dataset)
                else:
                    raise ValueError("Camera transformer not found in the checkpoint and no dataset provided")
        else:
            # Add default parameters from the train dataset
            hparams = get_opts(config_overrides)
            hparams.N_vocab = len(train_dataset["images"])
            self.camera_transformer = CameraTransformer.build_from_dataset(train_dataset)
            self._num_pixels = sum(a*b for a, b in train_dataset["cameras"].image_sizes)
        self.hparams = hparams

        # Load PL module
        system = train.NeRFSystem(self.hparams)
        system.setup = partial(_system_setup, system.setup, train_dataset, self.camera_transformer)
        system.val_dataloader = pl.LightningModule.val_dataloader
        system.validation_step = pl.LightningModule.validation_step
        self.system = system
        if ckpt_data is not None:
            system.load_state_dict(ckpt_data["state_dict"])

        # Patch PL trying to detect slurm
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_JOB_NAME", None)

        # Setup trainer
        trainer = (Trainer if train_dataset is not None else ETrainer)(
            max_epochs=self.hparams.num_epochs,
            devices=self.hparams.num_gpus if train_dataset is not None else 1,
            accelerator="cuda",
            strategy="ddp",
            barebones=True,
            num_sanity_val_steps=0,
            limit_train_batches=self.hparams.steps_per_epoch,
            enable_progress_bar=False,
            benchmark=True,
            profiler=None)

        # Setup training
        if train_dataset is not None:
            with tempfile.NamedTemporaryFile() as tmpfile:
                pickle.dump({
                    "train_dataset": train_dataset,
                    "config_overrides": config_overrides,
                    "checkpoint": self.checkpoint,
                }, tmpfile)
                tmpfile.flush()
                tmpfile.seek(0)
                sys.argv = [os.path.abspath(__file__), tmpfile.name]
                os.environ["_NERFBASELINES_DISABLE_LOGGING"] = "1"
                def _run_training(yield_cb):
                    system.training_step = partial(_override_train_iteration, system, system.training_step, yield_cb)
                    # Patch yield_cb to pause the training after each step
                    trainer.fit(system, ckpt_path=ckpt_file)

                # Fix PL adding barrier after save.
                old_barrier = trainer.strategy.barrier
                def _barrier(name, *args, **kwargs):
                    if name != "Trainer.save_checkpoint":
                        # print("Barrier", name, os.environ.get("LOCAL_RANK", None))
                        return old_barrier(name, *args, **kwargs)
                    else:
                        logger.info(f"Skipping barrier for {name}")
                trainer.strategy.barrier = _barrier

                # Run training stopping after each training step
                self._train_iterator = iter(CallbackIterator(_run_training))
                next(self._train_iterator)
                logger.info(f"Model initialized for training")
        else:
            assert ckpt_data is not None, "Either train_dataset or checkpoint must be provided"
            trainer.strategy._lightning_module = trainer.strategy.model = self.system.to("cuda")
            trainer.model.setup("eval")
            trainer._checkpoint_connector._restore_modules_and_callbacks(ckpt_file)
            # We need to fix the trainer.global_step and trainer.current_epoch
            trainer.global_step = ckpt_data["global_step"]
            if "current_epoch" in ckpt_data:
                trainer.current_epoch = ckpt_data["current_epoch"]
            logger.info(f"Model initialized for evaluation")
        self.trainer = trainer

    @property
    def _model(self):
        model = self.trainer.model
        if hasattr(model, "module"):
            # Unwrap DDP
            model = model.module
        return model

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",  # Will be provided by the registry
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(get_args(CameraModel)),
            supported_outputs=("color", "depth"),
            viewer_default_resolution=(64, 256),
        )

    def get_info(self) -> ModelInfo:
        num_iterations = self.hparams.num_epochs * min(15000, ((self._num_pixels + self.hparams.batch_size - 1) // self.hparams.batch_size))
        num_iterations = int(num_iterations)
        hparamsflat = flatten_hparams(vars(self.hparams), separator=".")
        hparamsflat.pop("root_dir", None)
        hparamsflat.pop("prefixes_to_ignore", None)
        hparamsflat.pop("refresh_every", None)
        hparamsflat.pop("exp_name", None)
        hparamsflat.pop("img_downscale", None)
        hparamsflat.pop("img_wh", None)
        hparamsflat.pop("use_cache", None)
        return ModelInfo(
            num_iterations=num_iterations,
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=hparamsflat,
            **self.get_method_info(),
        )

    @contextmanager 
    def _with_eval_embedding(self, embedding_a, embedding_t):
        torch.autograd.set_detect_anomaly(True)
        model = self.trainer.model
        context = contextlib.nullcontext
        if hasattr(model, "module"):
            # Unwrap DDP
            context = model.no_sync
            model = model.module
        model = self._model
        device = next(iter(model.parameters())).device
        is_training = model.training
        old_requires_grad = [x.requires_grad for x in model.parameters()]
        with context():
            def render(rays, ts, output_device=None):
                if isinstance(embedding_a, np.ndarray):
                    embedding_a_ = torch.tensor(embedding_a, device=device)
                elif embedding_a is None:
                    embedding_a_ = self._model.embedding_a.weight.data.mean(0).to(device)
                else:
                    embedding_a_ = embedding_a.to(device)
                if isinstance(embedding_t, np.ndarray):
                    embedding_t_ = torch.tensor(embedding_t, device=device)
                elif embedding_t is None:
                    embedding_t_ = self._model.embedding_t.weight.data.mean(0).to(device)
                else:
                    embedding_t_ = embedding_t.to(device)
                def expand_embedding(embedding, ts):
                    emb = embedding.view(*([1] * len(ts.shape) + [-1]))
                    emb = emb.expand(list(ts.shape) + [-1])
                    return emb
                B = rays.shape[0]
                results = defaultdict(list)
                for i in range(0, B, self.hparams.chunk):
                    tspart = ts[i:i+self.hparams.chunk]
                    rendered_ray_chunks = \
                        render_rays(model.models,
                                    model.embeddings,
                                    rays[i:i+self.hparams.chunk],
                                    tspart,
                                    self.hparams.N_samples,
                                    self.hparams.use_disp,
                                    0,
                                    0,
                                    self.hparams.N_importance,
                                    self.hparams.chunk, # chunk size is effective in val mode
                                    model.train_dataset.white_back,
                                    a_embedded=expand_embedding(embedding_a_, tspart),
                                    t_embedded=expand_embedding(embedding_t_, tspart),
                                    test_time=True)
                    for k, v in rendered_ray_chunks.items():
                        if output_device is not None:
                            v = v.to(output_device)
                        results[k] += [v]
                return {
                    k: torch.cat(v, 0) for k, v in results.items()}
            try:
                for p in model.parameters():
                    p.requires_grad = False
                    pass
                model.eval()
                yield render
            finally:
                model.train(is_training)
                for p, r in zip(model.parameters(), old_requires_grad):
                    p.requires_grad = r

    @torch.no_grad()
    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        camera = camera.item()  # Ensure it is a single camera
        model = self._model
        device = next(iter(model.parameters())).device
        assert device.type == "cuda", "Model is not on GPU"

        rays = self.camera_transformer.get_rays(camera[None]).to(device) # (H*W, 8)
        ts = torch.zeros((len(rays),), dtype=torch.long, device=device)

        embedding = (options or {}).get("embedding", None)
        na = self.hparams.N_a
        embedding_a, embedding_t = (embedding[..., :na], embedding[..., na:]) if embedding is not None else (None, None)
        with self._with_eval_embedding(embedding_a, embedding_t) as model:
            results = model(rays, ts, output_device=torch.device("cpu"))

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        W, H = camera.image_sizes
        img = results[f'rgb_{typ}'].view(H, W, 3) # (3, H, W)
        depth = results[f'depth_{typ}'].view(H, W) # (3, H, W)
        return {
            "color": img.detach().cpu().numpy(),
            "depth": depth.detach().cpu().numpy(),
        }

    @property
    def _is_rank0(self):
        return self.trainer.global_rank == 0

    def _finish_training(self):
        self._train_iterator = None
        # Lightning moves everything to CPU, we need to move it back to GPU
        model = self._model.to("cuda").eval()
        self.trainer.strategy._lightning_module = self.trainer.strategy.model = model
        logger.info(f"Model initialized for evaluation")

    def train_iteration(self, step):
        assert self._train_iterator is not None, "Method not initialized for training"
        del step
        out = None
        while out is None:
            out = next(self._train_iterator)
        # Backward pass after train_iterator return
        try:
            next(self._train_iterator)
        except StopIteration:  # Last step
            logging.info("Training stopped")
            self._finish_training()
        loss, metrics = out
        def _get(value):
            if isinstance(value, torch.Tensor):
                return value.detach().cpu().item()
            return value
        metrics = {
            (k[len("train/"):] if k.startswith("train/") else k): _get(v) for k, v in metrics.items()
        }
        metrics["loss"] = _get(loss)
        return metrics

    def save(self, path: str):
        if not self._is_rank0:
            logging.debug("Skipping save on non-rank0 process")
            return
        os.makedirs(path, exist_ok=True)
        weights_only = False
        ckpt_data = self.trainer._checkpoint_connector.dump_checkpoint(weights_only)

        # Add camera transformer
        ckpt_data["camera_transformer"] = {}
        self.camera_transformer.save(ckpt_data["camera_transformer"])
        ckpt_data["num_pixels"] = self._num_pixels
        filepath = os.path.join(path, f"checkpoint.{self.trainer.global_step}.ckpt")
        self.trainer.strategy.save_checkpoint(ckpt_data, filepath)

    def _get_default_embedding(self):
        if self.hparams.encode_a:
            return self._model.embedding_a.weight.data.mean(0).cpu().numpy()

    def optimize_embedding(self, dataset: Dataset, *, embedding: Optional[np.ndarray] = None) -> OptimizeEmbeddingOutput:
        torch.cuda.empty_cache()
        camera = dataset["cameras"].item()  # Ensure it is a single camera
        model = self._model
        device = next(iter(model.parameters())).device
        assert device.type == "cuda", "Model is not on GPU"

        # TODO: nondefault embedding
        if embedding is not None:
            embedding_th = torch.from_numpy(embedding).to(device=device)
            embedding_a = embedding_th[..., :self.hparams.N_a]
            embedding_t = embedding_th[..., self.hparams.N_a:]
        else:
            embedding_a = self._model.embedding_a.weight.mean(0)
            embedding_t = self._model.embedding_t.weight.mean(0)

        param_a = torch.nn.Parameter(embedding_a.clone().detach().to(device=device).requires_grad_(True), requires_grad=True)
        param_t = torch.nn.Parameter(embedding_t.clone().detach().to(device=device).requires_grad_(True), requires_grad=True)
        params = [param_a, param_t] if self.hparams.appearance_optim_finetune_tau else [param_a]
        optim = torch.optim.Adam(params, lr=self.hparams.appearance_optim_lr)
        num_steps = self.hparams.appearance_optim_steps
        lr_sched = torch.optim.lr_scheduler.StepLR(optim, step_size=num_steps//3, gamma=0.1)
        mses = []
        psnrs = []
        all_rays = self.camera_transformer.get_rays(camera[None]) # (H*W, 8)
        w, h = camera.image_sizes
        img_data = dataset["images"][0]
        all_rgbs = torch.from_numpy(convert_image_dtype(img_data[:h, :w], np.float32).reshape(-1, img_data.shape[-1])) # (H*W, 3)
        assert all_rays.size(0) == all_rgbs.size(0), f"Rays and images size mismatch {all_rays.size(0)} != {all_rgbs.size(0)}"
        with self._with_eval_embedding(param_a, param_t) as model, \
             tqdm.tqdm(total=num_steps, desc=f"Optimizing image embedding") as pbar:
            for _ in range(num_steps):
                optim.zero_grad()
                local_indices = torch.randperm(len(all_rays))[:self.hparams.batch_size]
                rays = all_rays[local_indices].to(device)
                rgbs = all_rgbs[local_indices].to(device)
                ts = torch.zeros((len(rays),), dtype=torch.long, device=device)

                results = model(rays, ts)
                typ = 'fine' if 'rgb_fine' in results else 'coarse'
                img = results[f'rgb_{typ}']
                mse = torch.nn.functional.mse_loss(img, rgbs).mean()
                mse.backward()
                assert param_a.grad is not None and torch.isfinite(param_a.grad).all()
                assert param_t.grad is not None and torch.isfinite(param_t.grad).all()
                optim.step()
                lr_sched.step()
                mses.append(mse.item())
                psnrs.append(10 * math.log10(1 / mse.item()))
                pbar.set_postfix({"mse": f"{mse.item():.4f}", "psnr": f"{psnrs[-1]:.3f}"})
                pbar.update()

        appearance_embedding = np.concatenate((param_a.detach().cpu().numpy(), param_t.detach().cpu().numpy()), -1)
        return {
            "embedding": appearance_embedding,
            "metrics": {
                "psnr": psnrs,
                "mse": mses,
                "loss": mses,
            }
        }

    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        """
        Get the embedding for a training image.

        Args:
            index: Index of the image.
        """
        model = self._model
        out = []
        if self.hparams.encode_a:
            out.append(model.embedding_a.weight.data[index].cpu().numpy())
        if self.hparams.encode_t:
            out.append(model.embedding_t.weight.data[index].cpu().numpy())
        return np.concatenate(out, -1) if out else None


if __name__ == "__main__":
    # Run DDP loop
    logger = logging.getLogger("slave")
    logger.setLevel(logging.INFO)
    logger.info("Launching slave process")
    with open(sys.argv[1], "rb") as f:
        kwargs = pickle.load(f)

    method = NeRFWReimpl(**kwargs)  # type: ignore
    del kwargs
    info = method.get_info()
    for i in range(info.get("loaded_step") or 0, info["num_iterations"]):
        method.train_iteration(i)
