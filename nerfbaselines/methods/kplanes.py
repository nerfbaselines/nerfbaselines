"""
NOTE: there is slight difference from K-Planes official implementation.
(Photo Tourism): In the official implementation, the closest camera bounds are used for rendering test images.
                 Here, we make it more stable by using the top 5 closest cameras.
"""
import hashlib
import struct
import shutil
import glob
import tqdm
import dataclasses
import pprint
import warnings
import importlib.util
import logging
import os
import pprint
from contextlib import contextmanager
from typing import Optional
import urllib.request
import numpy as np
import torch
import torch.utils.data
from functools import partial
from nerfbaselines import (
    Dataset, RenderOutput, OptimizeEmbeddingOutput,
    Method, MethodInfo, ModelInfo, Cameras, camera_model_to_int,
)
from nerfbaselines.utils import convert_image_dtype
import tempfile


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


def get_torch_checkpoint_sha(checkpoint_data):
    sha = hashlib.sha256()
    def update(d):
        if type(d).__name__ == "Tensor":
            sha.update(d.cpu().numpy().tobytes())
        elif isinstance(d, dict):
            items = sorted(d.items(), key=lambda x: x[0])
            for k, v in items:
                update(k)
                update(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                update(v)
        elif isinstance(d, (int, float)):
            sha.update(struct.pack("<f", d))
        elif isinstance(d, str):
            sha.update(d.encode("utf8"))
        elif d is None:
            sha.update("(None)".encode("utf8"))
        else:
            raise ValueError(f"Unsupported type {type(d)}")
    update(checkpoint_data)
    return sha.hexdigest()


def _noop(out=None, *args, **kwargs):
    del args, kwargs
    return out


_kplanes_patched = False


def _patch_kplanes():
    global _kplanes_patched
    if _kplanes_patched:
        return
    _kplanes_patched = True
    # Patch resource.setrlimit
    import resource
    _backup = resource.setrlimit

    try:
        resource.setrlimit = _noop
        from plenoxels.runners import base_trainer  # type: ignore
    finally:
        resource.setrlimit = _backup
    del _backup

    # Patch SummaryWriter
    class NoopWritter:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            pass
        def __getattr__(self, name):
            del name
            return _noop

    base_trainer.SummaryWriter = NoopWritter


class LambdaModule(torch.nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


@contextmanager
def _patch_kplanes_phototourism_dataset(dataset, camera_bounds_index):
    from plenoxels.datasets import phototourism_dataset as ptdataset  # type: ignore
    # Patch kplanes dataset
    with tempfile.TemporaryDirectory(suffix=dataset["metadata"].get("scene") if dataset is not None else None) as tmpdir:
        ptinit_backup = ptdataset.PhotoTourismDataset.__init__
        pt_getidsforsplit = ptdataset.get_ids_for_split
        pt_loadcamerametadata = ptdataset.load_camera_metadata
        pt_readpng = ptdataset.read_png
        pt_getnumtrainimages = ptdataset.PhotoTourismDataset.get_num_train_images
        pt_renderposes = ptdataset.pt_render_poses
        torchload_backup = torch.load
        torchsave_backup = torch.save
        cached_data = None
        poses = kinvs = res = bounds = None
        if dataset is not None:
            cameras = dataset["cameras"]
            poses, kinvs, res = transform_cameras(cameras)
            bounds = camera_bounds_index.bounds
            assert len(poses) == len(bounds), f"Expected {len(poses)} == {len(bounds)}"
        def pt_init(self, datadir, *args, **kwargs):
            del datadir
            if dataset is None:
                kwargs["split"] = "render"
            ptinit_backup(self, tmpdir, *args, **kwargs)
        def pt_getidsforsplit(datadir, split):
            del datadir, split
            return list(range(len(dataset["cameras"]))), dataset["image_paths"]
        def pt_loadcamerametadata(datadir, idx):
            del datadir
            assert (
                poses is not None and 
                kinvs is not None and 
                bounds is not None and 
                res is not None
            ), "Dataset is required to load camera metadata"
            return poses[idx], kinvs[idx], bounds[idx], res[idx]
        def pt_readpng(impath):
            imgid = dataset["image_paths"].index(impath)
            if dataset.get("images") is None:
                w, h = dataset["cameras"].image_sizes[imgid]
                img = np.zeros((h, w, 3), dtype=np.uint8)
            else:
                img = dataset["images"][imgid]
            img = torch.from_numpy(convert_image_dtype(img, np.float32))
            return img  # H, W, C
        def pt_renderposes(self, *args, **kwargs):
            del self, args, kwargs
            return None, None, None, None
        def torchload(path, *args, **kwargs):
            del path, args, kwargs
            return cached_data
        def torchsave(obj, path, *args, **kwargs):
            del path, args, kwargs
            nonlocal cached_data
            cached_data = obj
        try:
            torch.load = torchload
            torch.save = torchsave
            ptdataset.PhotoTourismDataset.__init__ = pt_init
            ptdataset.get_ids_for_split = pt_getidsforsplit
            ptdataset.load_camera_metadata = pt_loadcamerametadata
            ptdataset.read_png = pt_readpng
            ptdataset.pt_render_poses = pt_renderposes
            num_images = len(dataset["cameras"]) if dataset is not None else 0
            ptdataset.PhotoTourismDataset.get_num_train_images = partial(_noop, num_images)
            yield None
        finally:
            torch.load = torchload_backup
            torch.save = torchsave_backup
            ptdataset.PhotoTourismDataset.__init__ = ptinit_backup
            ptdataset.get_ids_for_split = pt_getidsforsplit
            ptdataset.load_camera_metadata = pt_loadcamerametadata
            ptdataset.read_png = pt_readpng
            ptdataset.pt_render_poses = pt_renderposes
            ptdataset.PhotoTourismDataset.get_num_train_images = pt_getnumtrainimages


@contextmanager
def _patch_kplanes_static_datasets(dataset, camera_bounds_index):
    import plenoxels.datasets.synthetic_nerf_dataset as sndataset  # type: ignore
    from plenoxels.datasets.intrinsics import Intrinsics  # type: ignore
    del camera_bounds_index
    # Patch kplanes dataset
    # TODO:!!!
    with tempfile.TemporaryDirectory(suffix=dataset["metadata"].get("scene") if dataset is not None else None) as tmpdir:
        init_backup = sndataset.SyntheticNerfDataset.__init__
        load_360_images_backup = sndataset.load_360_images
        load_360_frames_backup = sndataset.load_360_frames
        load_360_intrinsics_backup = sndataset.load_360_intrinsics
        def init(self, datadir, split, *args, **kwargs):
            del datadir
            if dataset is None:
                split = "render"
            init_backup(self, tmpdir, split, *args, **kwargs)

        def load_360_frames(datadir, split, max_frames):
            del datadir, split, max_frames
            return None, None

        def load_360_images(frames, datadir, split, downsample):
            del frames, datadir, split
            if dataset is None:
                return torch.zeros((1, 8, 8, 3,), dtype=torch.float32), torch.zeros((1, 3, 4), dtype=torch.float32)
            dataset_downscale = dataset['metadata'].get('downsample_factor', 1.0)
            if downsample != dataset_downscale:
                warnings.warn(f"Downsample factor {downsample} does not match dataset downsample factor {dataset_downscale}")
            imgs = torch.from_numpy(convert_image_dtype(np.stack(dataset["images"], 0), dtype=np.float32))
            poses = np.stack(dataset["cameras"].poses[:, :3, :], 0)
            # Convert from OpenCV to OpenGL coordinate system
            poses[..., 0:3, 1:3] *= -1
            return imgs, torch.from_numpy(poses)

        def load_360_intrinsics(*args, **kwargs):
            del args, kwargs
            if dataset is None:
                return Intrinsics(height=8, width=8, focal_x=8, focal_y=8, center_x=4, center_y=4)
            height, width = dataset["images"][0].shape[:2]
            fl_x, fl_y, cx, cy = dataset["cameras"].intrinsics.mean(0)
            return Intrinsics(height=height, width=width, focal_x=fl_x, focal_y=fl_y, center_x=cx, center_y=cy)
        try:
            sndataset.SyntheticNerfDataset.__init__ = init
            sndataset.load_360_images = load_360_images
            sndataset.load_360_frames = load_360_frames
            sndataset.load_360_intrinsics = load_360_intrinsics
            yield None
        finally:
            sndataset.SyntheticNerfDataset.__init__ = init_backup
            sndataset.load_360_images = load_360_images_backup
            sndataset.load_360_frames = load_360_frames_backup
            sndataset.load_360_intrinsics = load_360_intrinsics_backup


def load_data(model_type: str, data_downsample, data_dirs, validate_only: bool, render_only: bool, dataset, camera_bounds_index, **kwargs):
    from plenoxels.utils.parse_args import parse_optfloat  # type: ignore
    from plenoxels.runners import video_trainer  # type: ignore
    from plenoxels.runners import phototourism_trainer  # type: ignore
    from plenoxels.runners import static_trainer  # type: ignore
    data_downsample = parse_optfloat(data_downsample, default_val=1.0)

    if model_type == "video":
        return video_trainer.load_data(
            data_downsample, data_dirs, validate_only=validate_only,
            render_only=render_only, **kwargs)
    elif model_type == "phototourism":
        with _patch_kplanes_phototourism_dataset(dataset, camera_bounds_index):
            return phototourism_trainer.load_data(
                data_downsample, data_dirs, validate_only=validate_only,
                render_only=render_only, **kwargs
            )
    else:
        with _patch_kplanes_static_datasets(dataset, camera_bounds_index):
            return static_trainer.load_data(
                data_downsample, data_dirs, validate_only=validate_only,
                render_only=render_only, **kwargs)


def save_config(path, config):
    with open(os.path.join(path, 'config.py'), 'wt') as out:
        out.write('config = ' + pprint.pformat(config))

    with open(os.path.join(path, 'config.csv'), 'w') as f:
        for key in sorted(config.keys()):
            f.write("%s\t%s\n" % (key, config[key]))


# The kplanes data was extracted from https://drive.google.com/drive/folders/1SVHKRQXiRb98q4KHVEbj8eoWxjNS2QLW
# using the following code:
# dataset = load_colmap_dataset("external://phototourism/{scene}", split=None
# bds = np.load("{path to downloaded bds")
# files = sorted(dataset["image_paths"])
# with open("phototourism-{scene}-bounds.txt", "w") as f:
#     f.write("image min_bound max_bound P00 P01 P02 P03 P10 P11 P12 P13 P20 P21 P22 P23\n")
#     for i, bd in enumerate(bds):
#         ps = " ".join(map(str, dataset["cameras"].poses[i][:3, :].flatten()))
#         f.write(f"{os.path.split(os.path.splitext(files[i])[0])[-1]} {bd[0]} {bd[1]} {ps}\n")


def transform_cameras(cameras):
    intrinsics = cameras.intrinsics.copy()
    poses = cameras.poses.copy()

    # Transform coordinates
    # intrinsics[..., 1:] *= -1
    # poses[..., :, 1:3] *= -1

    K = np.zeros((*intrinsics.shape[:-1], 3, 3))
    K[..., 0, 0] = intrinsics[..., 0]
    K[..., 1, 1] = intrinsics[..., 1]
    K[..., 0, 2] = intrinsics[..., 2]
    K[..., 1, 2] = intrinsics[..., 3]
    K[..., 2, 2] = 1
    kinvs = np.linalg.inv(K)
    return poses, kinvs, cameras.image_sizes


class CameraBoundsIndex:
    def __init__(self, poses, bounds, offsets, query_top=5):
        # NOTE: !! in the official code query_top = 1, but that fails in lots of cases
        # If having issues, try setting query_top = 5 or larger
        self.poses = poses[:, :3, :].copy()
        self.bounds = bounds.copy()
        self.offsets = offsets.copy()
        self.query_top = query_top

    def save(self, checkpoint):
        np.savez(os.path.join(checkpoint, "bounds.npz"), poses=self.poses, bounds=self.bounds, offsets=self.offsets)

    @staticmethod
    def load(checkpoint):
        data = np.load(os.path.join(checkpoint, "bounds.npz"))
        return CameraBoundsIndex(data["poses"], data["bounds"], data["offsets"])

    @staticmethod
    def build(dataset, config):
        # Data was extracted from: https://drive.google.com/drive/folders/1SVHKRQXiRb98q4KHVEbj8eoWxjNS2QLW
        _phototourism_bounds = {
            "trevi-fountain": "https://gist.githubusercontent.com/jkulhanek/45cc7a94fe543237683b563660d6e590/raw/244c06c1b67588ca9f6f1c8ff02e63f044ab72e7/trevi-fountain.txt",
            "brandenburg-gate": "https://gist.githubusercontent.com/jkulhanek/45cc7a94fe543237683b563660d6e590/raw/244c06c1b67588ca9f6f1c8ff02e63f044ab72e7/brandenburg-gate.txt",
            "sacre-coeur": "https://gist.githubusercontent.com/jkulhanek/45cc7a94fe543237683b563660d6e590/raw/244c06c1b67588ca9f6f1c8ff02e63f044ab72e7/sacre-coeur.txt",
        }
        dataset_name = scene = None
        pref = config["data_dirs"][0].replace("_", "-").split("/")
        if len(pref) > 1:
            dataset_name, scene = pref[-2:]

        if dataset is None:
            raise RuntimeError("Dataset is required to estimate camera bounds")

        if dataset_name == "phototourism" and scene in _phototourism_bounds:
            logging.info(f"Using official K-planes pre-computed camera bounds for scene {scene}")
            assert scene is not None, "Scene is required"  # pyright not clever enough
            request = urllib.request.Request(_phototourism_bounds[scene], headers={
                "User-Agent": "Mozilla/5.0",
            })
            with urllib.request.urlopen(request) as r:
                if r.getcode() != 200:
                    raise RuntimeError(f"Failed to download camera bounds for scene {scene} (HTTP {r.getcode()})")
                names, bounds_data = [], []
                for line in r.read().decode("utf-8").splitlines()[1:]:
                    name, l1, h1 = line.split()
                    names.append(name)
                    bounds_data.append(np.array([float(l1), float(h1)], dtype=np.float32))
                bounds_map = dict(zip(names, bounds_data))
                bounds = [bounds_map[os.path.split(os.path.splitext(x)[0])[-1]] for x in dataset["image_paths"]]
                if scene in ("trevi-fountain", "brandenburg-gate"):
                    offsets = np.array([0.01, 0.0])
                elif scene == "sacre-coeur":
                    offsets = np.array([0.05, 0.0])
                else:
                    raise RuntimeError(f"Unknown scene {scene}")
                return CameraBoundsIndex(dataset["cameras"].poses[:, :3, :], np.stack(bounds, 0), offsets)
        elif dataset_name == "nerf-synthetic":
            return CameraBoundsIndex(
                np.zeros((1, 3, 4), dtype=np.float32), 
                np.array([[2.0, 6.0]], dtype=np.float32), 
                offsets=np.array([0.0, 0.0], dtype=np.float32))
        elif dataset_name == "phototourism":
            logging.warning(f"Could not load camera bounds for scene {scene}")

        # Estimate bounds from the dataset
        if dataset.get("images_points3D_indices") is None or dataset.get("points3D_xyz") is None:
            raise RuntimeError(f"Dataset (config {dataset_name}) does not contain points3D_xyz and images_points3D_indices. Cannot estimate bounds.")

        bounds = []
        poses = dataset["cameras"].poses
        for i in tqdm.trange(len(dataset["cameras"])):
            inds = dataset["images_points3D_indices"][i]
            zvals = dataset["points3D_xyz"][inds] - poses[i, None, :3, 3]
            zvals = (zvals * poses[i, None, :3, 2]).sum(-1)
            zvals = np.abs(zvals)
            dl1, dh1 = np.percentile(zvals, .1, axis=-1), np.percentile(zvals, 99.9, axis=-1)
            dl1 = max(dl1, 0.0001)
            bounds.append(np.array([dl1, dh1], dtype=np.float32))
        bounds = np.stack(bounds, 0)
        poses = poses[:, :3, :]
        return CameraBoundsIndex(poses, bounds, offsets=np.array([0.0, 0.0], dtype=np.float32))

    def query(self, poses):
        # Find the closest cam
        closest_cam_idx = np.argpartition(np.linalg.norm(
            self.poses.reshape((1, -1, 12)) - poses.reshape((-1, 1, 12)), axis=-1
        ), self.query_top, axis=-1)

        bmin = self.bounds[closest_cam_idx, 0].min(-1)
        bmax = self.bounds[closest_cam_idx, 1].max(-1)
        bounds = np.stack([bmin, bmax], -1)
        if self.offsets is not None:
            bounds = bounds + self.offsets
        return bounds


class KPlanes(Method):
    def __init__(self, 
                 *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None,
                 config_overrides: Optional[dict] = None):
        _patch_kplanes()
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.background = None
        self.step = 0

        self.scene = None

        # Setup parameters
        if checkpoint is not None:
            pass

        # Setup config
        import plenoxels.configs as cfg_package  # type: ignore
        config_root = os.path.join(os.path.dirname(os.path.abspath(cfg_package.__file__)), "final")
        self._loaded_step = None
        if self.checkpoint is None:  #  or not os.path.exists(os.path.join(self.checkpoint, "config.py")):
            # Load config
            config_path = (config_overrides or {}).copy().pop("config", None)
            if config_path is None:
                config_path = "NeRF/nerf_hybrid.py"
            config_path = os.path.join(config_root, config_path)
        else:
            # Load config from checkpoint
            config_path = os.path.join(self.checkpoint, "config.py")
            if not os.path.exists(self.checkpoint):
                raise RuntimeError(f"Model directory {self.checkpoint} does not exist")
            ckpt = torch.load(str(self.checkpoint) + f"/model.pth")
            self._loaded_step = int(ckpt["global_step"]) + 1

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpcpath = os.path.join(tmpdir, os.path.basename(config_path))
            shutil.copy(config_path, tmpcpath)
            spec = importlib.util.spec_from_file_location(
                os.path.basename(config_path), tmpcpath)
            assert spec is not None, f"Could not load config from {config_path}"
            assert spec.loader is not None, f"Could not load config from {config_path}"
            logging.info(f"Loading config from {config_path}")
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)
        config = cfg.config
        config["logdir"] = "<empty>"
        config["expname"] = ""
        if self.checkpoint is not None:
            config.update(config_overrides or {})
        self.config = config

        # Setup trainer
        self.batch_iter = None
        self._setup(train_dataset)

    def _setup(self, train_dataset: Optional[Dataset]):
        from plenoxels.main import init_trainer  # type: ignore
        # Set random seed
        np.random.seed(self.config.get("seed", 0))
        torch.manual_seed(self.config.get("seed", 0))

        if "keyframes" in self.config:
            model_type = "video"
        elif "appearance_embedding_dim" in self.config:
            model_type = "phototourism"
        else:
            model_type = "static"

        pprint.pprint(self.config)
        camera_bounds_index = None
        if self.checkpoint is not None:
            try:
                camera_bounds_index = CameraBoundsIndex.load(self.checkpoint)
            except FileNotFoundError as e:
                if train_dataset is None:
                    raise RuntimeError("Could not load camera bounds from checkpoint."
                                       "If the checkpoint is not official NerfBaselines checkpoint,"
                                       "you can try reconstructing the necessary bounds by supplying"
                                       "also the train dataset into the constructor.") from e
                else:
                    logging.warning(f"Could not load camera bounds from {self.checkpoint}")
        if camera_bounds_index is None:
            logging.info("Building camera bounds from dataset")
            camera_bounds_index = CameraBoundsIndex.build(train_dataset, self.config)
        self.camera_bounds_index = camera_bounds_index
        data = load_data(model_type, 
                         validate_only=False, 
                         render_only=False, 
                         **self.config, 
                         dataset=train_dataset, 
                         camera_bounds_index=self.camera_bounds_index)
        kwargs = dict(**self.config, **data)
        kwargs["logdir"] = tempfile.gettempdir()
        trainer = init_trainer(model_type, **kwargs)
        self.trainer = trainer
        self._patch_model()

        if self.checkpoint is not None:
            checkpoint_path = os.path.join(self.checkpoint, "model.pth")
            trainer.load_model(
                torch.load(checkpoint_path, map_location="cpu"), 
                training_needed=True)
        trainer.post_step = _noop
        self.data = data

    def _patch_model(self):
        self = self.trainer.model
        old_load_state_dict = self.load_state_dict
        if getattr(old_load_state_dict, "__patched__", False):
            return

        def load_state_dict(state_dict, *args, **kwargs):
            # Try fixing shapes
            for name, buf in self.named_buffers():
                if name in state_dict and state_dict[name].shape != buf.shape:
                    buf.resize_(*state_dict[name].shape)
            for name, par in self.named_parameters():
                if name in state_dict and state_dict[name].shape != par.shape:
                    par.data = state_dict[name].to(par.device)
            old_load_state_dict(state_dict, *args, **kwargs)

        load_state_dict.__patched__ = True  # type: ignore
        self.load_state_dict = load_state_dict

    def train_iteration(self, step):
        if self.batch_iter is None:
            self.trainer.pre_epoch()
            self.batch_iter = iter(self.trainer.train_data_loader)
        self.trainer.timer.reset()
        self.trainer.model.step_before_iter(step)
        self.trainer.global_step = step
        self.trainer.timer.check("step-before-iter")
        try:
            data = next(self.batch_iter)
            self.trainer.timer.check("dloader-next")
        except StopIteration:
            self.trainer.pre_epoch()
            self.batch_iter = iter(self.trainer.train_data_loader)
            data = next(self.batch_iter)
            logging.info("Reset data-iterator")

        metrics = {"loss": 0.0}
        try:
            # Patch gscaler.scale(loss) to extract the loss value to report as a metric
            old_scale = self.trainer.gscaler.scale
            try:
                def scale(loss):
                    metrics["loss"] = float(loss)
                    return old_scale(loss)
                self.trainer.gscaler.scale = scale
                self.trainer.init_epoch_info()
                step_successful = self.trainer.train_step(data)
                metrics.update({k: v.value for k, v in self.trainer.loss_info.items()})
            finally:
                self.trainer.gscaler.scale = old_scale
        except StopIteration:
            self.trainer.pre_epoch()
            self.batch_iter = iter(self.trainer.train_data_loader)
            logging.info("Reset data-iterator")
            step_successful = True

        if step_successful and self.trainer.scheduler is not None:
            self.trainer.scheduler.step()
        for r in self.trainer.regularizers:
            r.step(self.trainer.global_step)
        self.trainer.model.step_after_iter(self.trainer.global_step)
        self.trainer.timer.check("after-step")
        return metrics

    @classmethod
    def get_method_info(cls):
        return MethodInfo(
            method_id="",  # Will be set by the registry
            required_features=frozenset(("color", "points3D_xyz", "images_points3D_indices")),
            supported_camera_models=frozenset(("pinhole",)),
            supported_outputs=("color", "depth"),
            viewer_default_resolution=(128, 512),
        )

    def get_info(self) -> ModelInfo:
        hparams = flatten_hparams(self.config, separator=".")
        hparams.pop("expname", None)
        hparams.pop("logdir", None)
        hparams.pop("device", None)
        hparams.pop("valid_every", None)
        hparams.pop("save_every", None)
        return ModelInfo(
            num_iterations=self.config["num_steps"],
            loaded_step=self._loaded_step,
            loaded_checkpoint=self.checkpoint,
            hparams=hparams,
            **self.get_method_info(),
        )

    def _get_eval_data(self, cameras):
        if type(self.data["ts_dset"]).__name__ == "PhotoTourismDataset":
            from plenoxels.datasets import phototourism_dataset as ptdataset  # type: ignore
            poses, kinvs, res = transform_cameras(cameras)
            bounds = self.camera_bounds_index.query(cameras.poses)
            poses, kinvs, bounds = ptdataset.scale_cam_metadata(poses, kinvs, bounds, scale=0.05)
            poses = torch.from_numpy(poses).float()
            bounds = torch.from_numpy(bounds).float()
            kinvs = torch.from_numpy(kinvs).float()

            for i, pose in enumerate(poses):
                frame_w, frame_h = res[i]
                rays_o, rays_d = ptdataset.get_rays_tourism(frame_h, frame_w, kinvs[i], pose)
                rays_o = rays_o.view(-1, 3)
                rays_d = rays_d.view(-1, 3)
                yield {
                    "rays_o": rays_o,
                    "rays_d": rays_d,
                    "rays_o_left": rays_o,
                    "rays_d_left": rays_d,
                    "near_fars": bounds[i].expand(rays_o.shape[0], 2),
                    "timestamps": torch.tensor([i], dtype=torch.long).expand(rays_o.shape[0], 1),
                    "bg_color": torch.ones((1, 3), dtype=torch.float32),
                }
        elif type(self.data["ts_dset"]).__name__ == "SyntheticNerfDataset":
            import plenoxels.datasets.synthetic_nerf_dataset as sndataset  # type: ignore
            for i, pose in enumerate(cameras.poses):
                frame_w, frame_h = cameras.image_sizes[i]
                # Convert from OpenCV to OpenGL coordinate system
                pose = pose.copy()
                pose[..., 0:3, 1:3] *= -1
                intrinsics = sndataset.Intrinsics(
                    height=frame_h, width=frame_w, 
                    focal_x=cameras.intrinsics[i, 0].item(), focal_y=cameras.intrinsics[i, 1].item(),
                    center_x=cameras.intrinsics[i, 2].item(), center_y=cameras.intrinsics[i, 3].item())
                rays_o, rays_d, _ = sndataset.create_360_rays(
                    None, torch.from_numpy(pose[None]), merge_all=False, intrinsics=intrinsics)
                yield {
                    "rays_o": rays_o.view(-1, 3),
                    "rays_d": rays_d.view(-1, 3),
                    "timestamps": torch.tensor([i], dtype=torch.long).expand(rays_o.shape[0], 1),
                    "bg_color": torch.ones((1, 3), dtype=torch.float32),
                    "near_fars": torch.tensor([[2.0, 6.0]])
                }
        else:
            raise NotImplementedError("Only phototourism, nerfsynthetic datasets are supported")

    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        camera = camera.item()
        assert np.all(camera.camera_models == camera_model_to_int("pinhole")), "Only pinhole cameras supported"

        def load_embedding(indices):
            _embnp = (options or {}).get("embedding")
            if _embnp is None:
                # Fix bug in kplanes
                embed = self.trainer.model.field.appearance_embedding
                embed = old_appearance_embedding.weight.mean(0)
            else:
                embed = torch.from_numpy(_embnp)
            embed = embed.to(self.trainer.model.field.appearance_embedding.weight.device)
            return embed.view(*([1] * len(indices.shape)), -1).expand(*indices.shape, -1)

        with torch.no_grad():
            data = next(self._get_eval_data(camera[None]))

            old_train = self.trainer.model.training
            old_test_appearance_embedding = getattr(self.trainer.model.field, 'test_appearance_embedding', None)
            old_appearance_embedding = self.trainer.model.field.appearance_embedding
            
            try:
                self.trainer.model.eval()
                if self.trainer.model.field.use_appearance_embedding:
                    self.trainer.model.field.test_appearance_embedding = LambdaModule(lambda indices: load_embedding(indices))
                out = self.trainer.eval_step(data)
            finally:
                self.trainer.model.train(old_train)
                if hasattr(self.trainer.model.field, 'test_appearance_embedding'):
                    del self.trainer.model.field.test_appearance_embedding
                if old_test_appearance_embedding is not None:
                    self.trainer.model.field.test_appearance_embedding = old_test_appearance_embedding
            w, h = camera.image_sizes
            return {
                "color": out["rgb"].view(h, w, -1).cpu().numpy(),
                "depth": out["depth"].view(h, w).cpu().numpy(),
            }

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.trainer.log_dir = path
        self.trainer.save_model()
        save_config(path, self.config)
        self.camera_bounds_index.save(path)

        # Note, since the torch checkpoint does not have deterministic SHA, we compute the SHA here.
        for fname in glob.glob(os.path.join(path, "**/*.pth"), recursive=True):
            ckpt = torch.load(fname, map_location="cpu")
            sha = get_torch_checkpoint_sha(ckpt)
            with open(fname + ".sha256", "w", encoding="utf8") as f:
                f.write(sha)

    def optimize_embedding(self, dataset: Dataset, *, embedding: Optional[np.ndarray] = None) -> OptimizeEmbeddingOutput:
        """
        Optimize embedding of a single image (passed as dataset).

        Args:
            dataset: Dataset (containing single image).
            embedding: Optional initial embedding.
        """
        old_test_appearance_embedding = getattr(self.trainer.model.field, 'test_appearance_embedding', None)
        param_trainable = {}
        for pn, p in self.trainer.model.named_parameters():
            param_trainable[pn] = p.requires_grad
        is_train = self.trainer.model.training
        old_add_scalar = self.trainer.writer.add_scalar
        try:
            # 1. Initialize test appearance code
            tst_embedding = torch.nn.Embedding(
                1, self.trainer.model.field.appearance_embedding_dim
            ).to(self.trainer.device)
            self.trainer.model.field.test_appearance_embedding = tst_embedding
            self.trainer.model.field.test_appearance_embedding.requires_grad_(True)

            # 2. Setup parameter trainability
            self.trainer.model.eval()
            for pn, p in self.trainer.model.named_parameters():
                p.requires_grad_(False)
            self.trainer.model.field.test_appearance_embedding.requires_grad_(True)

            losses = []
            # Patch to grab losses
            def add_scalar(tag, value, step):
                assert tag.startswith("appearance_loss_"), f"Unexpected tag {tag}"
                losses.append(value)
                return old_add_scalar(tag, value, step)
            self.trainer.writer.add_scalar = add_scalar

            # 3. Optimize
            camera = dataset["cameras"].item()
            data = next(self._get_eval_data(camera[None]))
            imgs_left = torch.from_numpy(convert_image_dtype(dataset["images"][0], np.float32))
            data["imgs_left"] = imgs_left.view(-1, imgs_left.shape[-1])
            with torch.autograd.no_grad():
                if embedding is not None:
                    tst_embedding.weight.data.copy_(torch.from_numpy(embedding)[None])
                else:
                    tst_embedding.weight.data.copy_(
                        self.trainer.model.field.appearance_embedding.weight
                            .detach()
                            .mean(dim=0, keepdim=True)
                    )

            # Optimize
            self.trainer.optimize_appearance_step(data, 0)
            appearance_embedding = tst_embedding.weight.data.cpu().numpy().copy()
            return {
                "embedding": appearance_embedding,
                "metrics": {
                    "loss": losses.copy(),
                }
            }

        finally:
            self.trainer.model.field.test_appearance_embedding.requires_grad_(False)
            del self.trainer.model.field.test_appearance_embedding

            # 4. Reset parameter trainability
            if old_test_appearance_embedding is not None:
                self.trainer.model.field.test_appearance_embedding = old_test_appearance_embedding
            for pn, p in self.trainer.model.named_parameters():
                p.requires_grad_(param_trainable[pn])
            self.trainer.model.train(is_train)
            self.trainer.writer.add_scalar = old_add_scalar

    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        """
        Get the embedding for a training image.

        Args:
            index: Index of the image.
        """
        app_emb = self.trainer.model.field.appearance_embedding
        if app_emb is None:
            return None
        try:
            embed = app_emb(torch.tensor(index, dtype=torch.long, device=app_emb.weight.device)).detach().cpu().numpy()
            return embed
        except Exception as e:
            logging.error(f"Failed to get embedding for image {index}: {e}")
            return None
