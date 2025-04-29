# NOTE: WaterSplatting wrapper is just copied nerfstudio.py
# <nerfstudio.py version=1.1.4>
# pylint: disable=protected-access
import struct
import io
import hashlib
import base64
import glob
import warnings
import pprint
import json
import enum
import os
import dataclasses
from functools import partial
import logging
from dataclasses import fields
from pathlib import Path
import copy
import tempfile
from typing import Optional, TypeVar, Union
from typing_extensions import Literal, get_origin, get_args
import numpy as np
from nerfbaselines import (
    Method, MethodInfo, ModelInfo,
    Dataset, RenderOutput,
    Cameras, camera_model_from_int,
)
from nerfbaselines.utils import convert_image_dtype
from nerfbaselines.io import get_torch_checkpoint_sha

import yaml
import torch  # type: ignore

# Patch wrong viser version
import sys
from unittest.mock import MagicMock
sys.modules["nerfstudio.viewer_legacy.server.viewer_state"] = MagicMock()

from nerfstudio.cameras import camera_utils  # type: ignore
from nerfstudio.cameras.cameras import Cameras as NSCameras, CameraType as NPCameraType  # type: ignore
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataparserOutputs  # type: ignore
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, InputDataset  # type: ignore
from nerfstudio.data.scene_box import SceneBox  # type: ignore
from nerfstudio.engine.trainer import TrainingCallbackLocation  # type: ignore
from nerfstudio.engine.trainer import Trainer  # type: ignore
from nerfstudio.configs.method_configs import all_methods  # type: ignore
from nerfstudio.utils.colors import COLORS_DICT  # type: ignore


T = TypeVar("T")


def numpy_to_base64(array: np.ndarray) -> str:
    with io.BytesIO() as f:
        np.save(f, array)
        return base64.b64encode(f.getvalue()).decode("ascii")


def numpy_from_base64(data: str) -> np.ndarray:
    with io.BytesIO(base64.b64decode(data)) as f:
        return np.load(f)


def cast_value(tp, value):
    origin = get_origin(tp)
    if origin is Literal:
        for val in get_args(tp):
            try:
                value_casted = cast_value(type(val), value)
                if val == value_casted:
                    return value_casted
            except ValueError:
                pass
            except TypeError:
                pass
        raise TypeError(f"Value {value} is not in {get_args(tp)}")
            
    if origin is Union:
        for t in get_args(tp):
            try:
                return cast_value(t, value)
            except ValueError:
                pass
            except TypeError:
                pass
        raise TypeError(f"Value {value} is not in {tp}")
    if tp is type(None):
        if str(value).lower() == "none":
            return None
        else:
            raise TypeError(f"Value {value} is not None")
    if tp is bool:
        if str(value).lower() in {"true", "1", "yes"}:
            return True
        elif str(value).lower() in {"false", "0", "no"}:
            return False
        else:
            raise TypeError(f"Value {value} is not a bool")
    if tp in {int, float, bool, str}:
        return tp(value)
    if isinstance(value, tp):
        return value
    raise TypeError(f"Cannot cast value {value} to type {tp}")


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
            sha.update(struct.pack("f", d))
        elif isinstance(d, str):
            sha.update(d.encode("utf8"))
        elif d is None:
            sha.update("(None)".encode("utf8"))
        else:
            raise ValueError(f"Unsupported type {type(d)}")
    update(checkpoint_data)
    return sha.hexdigest()


def _map_distortion_parameters(distortion_parameters):
    distortion_parameters = np.concatenate(
        (
            distortion_parameters[..., :6],
            np.zeros((*distortion_parameters.shape[:-1], 6 - min(6, distortion_parameters.shape[-1])), dtype=distortion_parameters.dtype),
        ),
        -1,
    )
    distortion_parameters = distortion_parameters[..., [0, 1, 4, 5, 2, 3]]
    return distortion_parameters


def _config_safe_set(config, path, value, autocast=False):
    path = path.split(".")
    for p in path[:-1]:
        if not hasattr(config, p):
            return False
        config = getattr(config, p)
    p = path[-1]
    if hasattr(config, p):
        if autocast:
            assert dataclasses.is_dataclass(config)
            value = cast_value(dataclasses.fields(config)[p].type, value)
        setattr(config, p, value)
        return True
    return False


def flatten_hparams(hparams, *, separator: str = "/", _prefix: str = ""):
    def format_value(v, only_simple_types=True):
        if isinstance(v, (str, float, int, bool, bytes, type(None))):
            return v
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            return format_value(v.tolist(), only_simple_types=only_simple_types)
        if isinstance(v, (list, tuple)):
            # If list of simple types, convert to string
            if not only_simple_types:
                return type(v)([format_value(x, only_simple_types=False) for x in v])
            formatted = [format_value(x) for x in v]
            if all(isinstance(x, (str, float, int, bool, bytes, type(None))) for x in formatted):
                return ",".join(str(x) for x in formatted)
            return ",".join(pprint.pformat(x) for x in formatted)
        if isinstance(v, dict):
            if not only_simple_types:
                return {k: format_value(v, only_simple_types=False) for k, v in v.items()}
            return pprint.pformat(format_value(v, only_simple_types=False))
        if isinstance(v, type):
            return v.__module__ + ":" + v.__name__
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, enum.Enum):
            return v.name
        if dataclasses.is_dataclass(v):
            return format_value({
                f.name: getattr(v, f.name) for f in dataclasses.fields(v)
            }, only_simple_types=only_simple_types)
        if callable(v):
            return v.__module__ + ":" + v.__name__
        return repr(v)

    flat = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)}
    _blacklist = set((
        "output_dir",
        "pipeline.datamanager.dataparser._target",
        "pipeline.datamanager.data",
        "pipeline.datamanager.dataparser.data",
        "project_name",
        "vis",
        "timestamp",
        "data",
        "experiment_name",
        "log_gradients",
        "relative_model_dir",
        "save_only_latest_checkpoint",
        "steps_per_eval_image",
        "steps_per_eval_batch",
        "steps_per_eval_all_images",
        "steps_per_save",
        "prompt",
    ))
    _blacklist_prefixes = (
        "load_",
        "logging.",
        "viewer.",
    )
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if k in _blacklist:
            continue
        if any(k.startswith(p) for p in _blacklist_prefixes):
            continue
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k, separator=separator))
        else:
            flat[k] = format_value(v)
    return flat


class _CustomDataParser(DataParser):
    def __init__(self, dataset, dataparser_transform, dataparser_scale, config,  *args, **kwargs):
        del args, kwargs
        self.dataset = dataset
        self.dataparser_transform = dataparser_transform
        self.dataparser_scale = dataparser_scale
        super().__init__(config)

    @property
    def scene_box(self):
        aabb_scale = 1.5
        dataparser_class = type(self.config).__name__
        if dataparser_class == "BlenderDataParserConfig":
            aabb_scale = 1.5
        elif dataparser_class in {"ColmapDataParserConfig", "NerfstudioDataParserConfig"}:
            aabb_scale = 1
        else:
            raise ValueError(f"Unsupported dataparser class {dataparser_class}")
        return SceneBox(aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32))

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs) -> DataparserOutputs:
        del split, kwargs
        if self.dataset is None:
            assert self.dataparser_transform is not None, "dataparser_transform must be provided if dataset is None"
            assert self.dataparser_scale is not None, "dataparser_scale must be provided if dataset is None"
            # Return empty dataset
            cameras = NSCameras(
                camera_to_worlds=torch.zeros((0, 4, 4), dtype=torch.float32),
                fx=torch.zeros(0, dtype=torch.float32),
                fy=torch.zeros(0, dtype=torch.float32),
                cx=torch.zeros(0, dtype=torch.float32),
                cy=torch.zeros(0, dtype=torch.float32),
                distortion_params=torch.zeros((0, 6), dtype=torch.float32),
                width=torch.zeros(0, dtype=torch.int64),
                height=torch.zeros(0, dtype=torch.int64),
                camera_type=torch.zeros(0, dtype=torch.long),
            )
            metadata = {}
            return DataparserOutputs(
                [],  # image_names
                cameras, # cameras
                None,  # alpha_color
                self.scene_box,  # scene_box
                [],  # sampling masks
                metadata,
                dataparser_transform=self.dataparser_transform[..., :3, :].contiguous(),
                dataparser_scale=self.dataparser_scale,
            )

        image_names = [f"{i:06d}.png" for i in range(len(self.dataset["cameras"].poses))]
        npmap = {x.name.lower(): x.value for x in NPCameraType.__members__.values()}
        npmap["pinhole"] = npmap["perspective"]
        npmap["opencv"] = npmap["perspective"]
        npmap["opencv_fisheye"] = npmap["fisheye"]
        camtypes = [npmap[camera_model_from_int(int(self.dataset["cameras"].camera_models[i]))] for i in range(len(self.dataset["cameras"].poses))]

        poses = self.dataset["cameras"].poses.copy()
       
        # Convert from Opencv to OpenGL coordinate system
        poses[..., 0:3, 1:3] *= -1

        # in x,y,z order
        # assumes that the scene is centered at the origin
        dataparser_class = type(self.config).__name__
        alpha_color = None
        if dataparser_class == "BlenderDataParserConfig":
            from nerfstudio.utils.colors import get_color  # type: ignore
            alpha_color = self.config.alpha_color
            if alpha_color is not None:
                alpha_color = get_color(alpha_color)
            if self.dataparser_transform is None:
                self.dataparser_transform = torch.eye(4, dtype=torch.float32)
                self.dataparser_scale = 1.0
        elif dataparser_class in {"ColmapDataParserConfig", "NerfstudioDataParserConfig"}:
            if self.dataparser_transform is None:
                self.dataparser_transform, self.dataparser_scale = get_pose_transform(poses)
        else:
            raise ValueError(f"Unsupported dataparser class {dataparser_class}")

        th_poses = transform_poses(self.dataparser_transform, self.dataparser_scale, torch.from_numpy(poses).float())
        distortion_parameters = torch.from_numpy(_map_distortion_parameters(self.dataset["cameras"].distortion_parameters))
        cameras = NSCameras(
            camera_to_worlds=th_poses,
            fx=torch.from_numpy(self.dataset["cameras"].intrinsics[..., 0]).contiguous(),
            fy=torch.from_numpy(self.dataset["cameras"].intrinsics[..., 1]).contiguous(),
            cx=torch.from_numpy(self.dataset["cameras"].intrinsics[..., 2]).contiguous(),
            cy=torch.from_numpy(self.dataset["cameras"].intrinsics[..., 3]).contiguous(),
            distortion_params=distortion_parameters.contiguous(),
            width=torch.from_numpy(self.dataset["cameras"].image_sizes[..., 0]).long().contiguous(),
            height=torch.from_numpy(self.dataset["cameras"].image_sizes[..., 1]).long().contiguous(),
            camera_type=torch.tensor(camtypes, dtype=torch.long),
        )
        metadata = {}
        transform_matrix = self.dataparser_transform
        scale_factor = self.dataparser_scale

        if self.dataset.get("points3D_xyz") is not None:
            xyz = torch.from_numpy(self.dataset["points3D_xyz"]).float()

            # Transform poses using the dataparser transform
            xyz = torch.cat((xyz, torch.ones_like(xyz[..., :1])), -1) @ transform_matrix.T
            xyz = (xyz[..., :3] / xyz[..., 3:]).contiguous()
            xyz *= scale_factor
            metadata["points3D_xyz"] = xyz
            metadata["points3D_rgb"] = torch.from_numpy(self.dataset["points3D_rgb"])
        return DataparserOutputs(
            image_names,
            cameras,
            alpha_color,
            self.scene_box,
            image_names if self.dataset["sampling_masks"] else None,
            metadata,
            dataparser_transform=transform_matrix[..., :3, :].contiguous(),  # pylint: disable=protected-access
            dataparser_scale=scale_factor,
        )  # pylint: disable=protected-access


def transform_poses(dataparser_transform, dataparser_scale, poses):
    assert poses.dim() == 3
    poses = (
        dataparser_transform.to(poses.dtype)
        @ torch.cat([poses, torch.tensor([[[0, 0, 0, 1]]], dtype=poses.dtype).expand((len(poses), 1, 4))], -2)
    )[:, :3, :].contiguous()
    poses[:, :3, 3] *= dataparser_scale
    return poses


def get_pose_transform(poses):
    poses = np.copy(poses)
    lastrow = np.array([[[0, 0, 0, 1]]] * len(poses), dtype=poses.dtype)
    poses = np.concatenate([poses, lastrow], axis=-2)
    poses = poses[..., np.array([1, 0, 2, 3]), :]
    poses[..., 2, :] *= -1

    applied_transform = np.eye(4)[:3, :]
    applied_transform = applied_transform[np.array([1, 0, 2]), :]
    applied_transform[2, :] *= -1

    poses = torch.from_numpy(np.array(poses).astype(np.float32))
    poses, transform_matrix = camera_utils.auto_orient_and_center_poses(poses, method="up", center_method="poses")

    scale_factor = 1.0
    scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    poses[:, :3, 3] *= scale_factor

    applied_transform = torch.tensor(applied_transform, dtype=transform_matrix.dtype)
    transform_matrix = transform_matrix @ torch.cat([applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0)
    transform_matrix_extended = torch.cat([transform_matrix, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], -2)
    return transform_matrix_extended, scale_factor


class NerfStudio(Method):
    _default_nerfstudio_name: str = "nerfacto"
    _require_points3D: bool = False

    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None, 
                 config_overrides: Optional[dict] = None):
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        self.step = 0
        self._loaded_step = None
        _config_overrides = config_overrides
        nerfstudio_name = self._default_nerfstudio_name
        if config_overrides is not None:
            _config_overrides = (config_overrides or {}).copy()
            nerfstudio_name = _config_overrides.pop("method", nerfstudio_name)

        if checkpoint is not None:
            # Load nerfstudio checkpoint
            with open(os.path.join(checkpoint, "config.yml"), "r", encoding="utf8") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            model_path = os.path.join(checkpoint, config.relative_model_dir)
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model directory {model_path} does not exist")
            self._loaded_step = self.step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(model_path))[-1]
        else:
            config = copy.deepcopy(all_methods[nerfstudio_name])
        self._tmpdir = tempfile.TemporaryDirectory()
        self._mode = None
        self.dataparser_params = None

        if checkpoint is None:
            _config_overrides = (config_overrides or {}).copy()
            dataparser = _config_overrides.pop("pipeline.datamanager.dataparser", None)
            if dataparser is not None:
                from nerfstudio.configs.dataparser_configs import all_dataparsers  # type: ignore
                config.pipeline.datamanager.dataparser = all_dataparsers[dataparser]
            for k, v in (_config_overrides or {}).items():
                if not _config_safe_set(config, k, v):
                    raise ValueError(f"Invalid config key {k}")

        self.config = config
        self._original_config = copy.deepcopy(config)
        if checkpoint is not None:
            config.get_base_dir = lambda *_: Path(checkpoint)
            config.load_dir = config.get_checkpoint_dir()

        self._setup(train_dataset, config_overrides)

    @classmethod
    def get_method_info(cls):
        features = ("color",)
        if cls._require_points3D:
            features = features + ("points3D_xyz", "points3D_rgb")
        return MethodInfo(
            method_id="",  # Will be filled by the registry
            required_features=frozenset(features),
            supported_camera_models=frozenset(
                (
                    "pinhole",
                    "opencv",
                    "opencv_fisheye",
                )
            ),
            supported_outputs=("color", "depth", "accumulation"),
        )

    def get_info(self) -> ModelInfo:
        out = ModelInfo(
            loaded_step=self._loaded_step,
            loaded_checkpoint=str(self.checkpoint) if self.checkpoint is not None else None,
            num_iterations=self.config.max_num_iterations,
            hparams=flatten_hparams(self.config, separator="."),
            **self.get_method_info()
        )
        if hasattr(self.config.pipeline.datamanager, "train_num_rays_per_batch"):
            out["batch_size"] = int(self.config.pipeline.datamanager.train_num_rays_per_batch)
        if hasattr(self.config.pipeline.model, "eval_num_rays_per_chunk"):
            out["eval_batch_size"] = int(self.config.pipeline.model.eval_num_rays_per_chunk)
        return out

    @torch.no_grad()
    def render(self, camera: Cameras, *, options=None) -> RenderOutput:
        del options
        camera = camera.item()
        cameras = camera[None]
        poses = cameras.poses.copy()

        # Convert from Opencv to OpenGL coordinate system
        poses[..., 0:3, 1:3] *= -1

        poses = torch.from_numpy(poses)
        assert poses.dim() == 3
        train_dataparser_outputs = self._trainer.pipeline.datamanager.train_dataparser_outputs
        poses = transform_poses(train_dataparser_outputs.dataparser_transform, train_dataparser_outputs.dataparser_scale,  poses)
        intrinsics = torch.from_numpy(cameras.intrinsics)
        npmap = {x.name.lower(): x.value for x in NPCameraType.__members__.values()}
        npmap["pinhole"] = npmap["perspective"]
        npmap["opencv"] = npmap["perspective"]
        npmap["opencv_fisheye"] = npmap["fisheye"]
        camtypes = [npmap[camera_model_from_int(int(cameras.camera_models[i]))] for i in range(len(poses))]
        sizes = cameras.image_sizes
        distortion_parameters = torch.from_numpy(_map_distortion_parameters(cameras.distortion_parameters))
        ns_cameras = NSCameras(
            camera_to_worlds=poses.contiguous(),
            fx=intrinsics[..., 0].contiguous(),
            fy=intrinsics[..., 1].contiguous(),
            cx=intrinsics[..., 2].contiguous(),
            cy=intrinsics[..., 3].contiguous(),
            distortion_params=distortion_parameters.contiguous(),
            width=torch.from_numpy(sizes[..., 0]).long().contiguous(),
            height=torch.from_numpy(sizes[..., 1]).long().contiguous(),
            camera_type=torch.tensor(camtypes, dtype=torch.long),
        ).to(self._trainer.pipeline.device)
        self._trainer.pipeline.eval()

        assert len(poses) == 1
        outputs = self._trainer.pipeline.model.get_outputs_for_camera(ns_cameras)
        return self._remap_outputs(outputs)

    def _remap_outputs(self, outputs):
        def _remap_key(x):
            return x.replace("rgb", "color")
        return { _remap_key(k): v.detach().cpu().numpy() for k, v in outputs.items()}

    def _patch_dataparser(self, dataparser_cls, *, train_dataset, dataparser_transforms, dataparser_scale, config):
        del dataparser_cls
        del config
        return partial(_CustomDataParser, train_dataset, dataparser_transforms, dataparser_scale)

    def _patch_dataset(self, dataset_cls, train_dataset, config):
        del config
        class DatasetL(dataset_cls):
            def get_numpy_image(self, image_idx: int):
                return train_dataset["images"][image_idx]

            def get_image(self, image_idx: int):
                img = self.get_numpy_image(image_idx)
                img = convert_image_dtype(img, np.float32)
                image = torch.from_numpy(img)
                if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
                    alpha_color = self._dataparser_outputs.alpha_color
                    if isinstance(self._dataparser_outputs.alpha_color, str):
                        alpha_color = COLORS_DICT[alpha_color]
                    else:
                        alpha_color = torch.from_numpy(np.array(alpha_color, dtype=np.float32))
                    image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
                return image
        return DatasetL

    def _patch_datamanager(self, datamanager_cls, *, train_dataset=None, config):
        this = self

        class DM(datamanager_cls):  # pylint: disable=protected-access
            def __init__(self, *args, **kwargs):
                if train_dataset is None:
                    self.sample_train_cameras = lambda: [None]
                    super().__init__(*args, **kwargs)
                    del self.sample_train_cameras
                else:
                    super().__init__(*args, **kwargs)

            @property
            def dataset_type(self):
                dataset_type = getattr(self, "_idataset_type", InputDataset)
                return this._patch_dataset(dataset_type, train_dataset=train_dataset, config=config)

            @dataset_type.setter
            def dataset_type(self, value):
                self._idataset_type = value

            def create_eval_dataset(self, *args, **kwargs):
                del args, kwargs
                return []  # Needed for eval_unseen_cameras
        return DM

    def _patch_model(self, model_cls, *, config):
        del config
        class M(model_cls):  # pylint: disable=protected-access
            def load_state_dict(self, state_dict, *args, **kwargs):
                # Try fixing shapes
                for name, buf in self.named_buffers():
                    if name in state_dict and state_dict[name].shape != buf.shape:
                        buf.resize_(*state_dict[name].shape)
                for name, par in self.named_parameters():
                    if name in state_dict and state_dict[name].shape != par.shape:
                        par.data = state_dict[name].to(par.device)
                super().load_state_dict(state_dict, *args, **kwargs)
        return M

    def _patch_config(self, config, *, train_dataset, dataparser_transforms, dataparser_scale):
        # Fix for newer NS versions -> we replace the ParallelDataManager with VanillaDataManager
        dm = config.pipeline.datamanager
        if dm.__class__.__name__ == "ParallelDataManagerConfig":
            dm = VanillaDataManagerConfig(**{k.name: getattr(dm, k.name) for k in fields(VanillaDataManagerConfig)})
            dm._target = VanillaDataManager  # pylint: disable=protected-access
            config.pipeline.datamanager = dm
        del dm
        # Patch data manager
        datamanager_cls = config.pipeline.datamanager._target
        datamanager_cls = self._patch_datamanager(datamanager_cls, train_dataset=train_dataset, config=config)
        config.pipeline.datamanager._target = datamanager_cls
        # Patch data parser
        dataparser_cls = config.pipeline.datamanager.dataparser._target
        dataparser_cls = self._patch_dataparser(dataparser_cls, 
                                                config=config,
                                                train_dataset=train_dataset, 
                                                dataparser_transforms=dataparser_transforms, 
                                                dataparser_scale=dataparser_scale)
        config.pipeline.datamanager.dataparser._target  = dataparser_cls
        # Patch model
        model_cls = config.pipeline.model._target
        model_cls = self._patch_model(model_cls, config=config)
        config.pipeline.model._target = model_cls
        if hasattr(config.pipeline.model._target, "__name__"):  # Protection against mocks
            config.pipeline.model._target.__name__ = model_cls.__name__  # pylint: disable=protected-access
        # Fix rest of the config
        assert self._tmpdir is not None, "Method already closed"
        config.output_dir = Path(self._tmpdir.name)
        config.set_timestamp()
        config.vis = None
        # self.config.machine.device_type = "cuda"
        config.load_step = None
        return config

    def _setup(self, train_dataset: Optional[Dataset], config_overrides):
        del config_overrides
        train_dataset = None if train_dataset is None else train_dataset.copy()
        dataparser_transforms = dataparser_scale = None
        if self.checkpoint is None and train_dataset is None:
            raise RuntimeError("Either checkpoint or train_dataset must be provided")
        if self.checkpoint is not None:
            if os.path.exists(os.path.join(self.checkpoint, "dataparser_transforms.json")):
                with open(os.path.join(self.checkpoint, "dataparser_transforms.json"), "r", encoding="utf8") as f:
                    dataparser_params = json.load(f)
                    if "transform_base64" in dataparser_params:
                        dataparser_transforms = torch.from_numpy(numpy_from_base64(dataparser_params["transform_base64"]))
                    else:
                        warnings.warn("Checkpoint does not store the transforms in binary form. The results may not be precise.")
                        dataparser_transforms = torch.tensor(dataparser_params["transform"], dtype=torch.float32)
                    # Pad with [0 0 0 1]
                    dataparser_transforms = torch.cat([dataparser_transforms, torch.tensor([[0, 0, 0, 1]], dtype=torch.float32)], 0)[:4, :].contiguous()
                    dataparser_scale = dataparser_params["scale"]
            elif os.path.exists(os.path.join(self.checkpoint, "dataparser_params.pth")):
                # Older checkpoint version
                # TODO: remove this after upgrading all checkpoints
                warnings.warn("Older checkpoint version detected, please upgrade the checkpoint")
                _dataparser_params = torch.load(os.path.join(self.checkpoint, "dataparser_params.pth"))
                dataparser_transforms = _dataparser_params["dataparser_transform"]
                dataparser_scale = _dataparser_params["dataparser_scale"]
                if _dataparser_params["aabb_scale"] == 1.5:
                    logging.warning("Older checkpoint: blender detected")
                    from nerfstudio.configs.dataparser_configs import all_dataparsers  # type: ignore
                    dataparser = "blender-data"
                    self.config.pipeline.datamanager.dataparser = copy.deepcopy(all_dataparsers[dataparser])
                    self._original_config.pipeline.datamanager.dataparser = copy.deepcopy(all_dataparsers[dataparser])
                elif (
                        self.config.pipeline.model.use_appearance_embedding and
                        self.config.pipeline.model.camera_optimizer.mode == "off"):
                    logging.warning("Older checkpoint: mip360 detected")
            else:
                raise ValueError("No dataparser_transforms.json file found in the checkpoint directory")

        config = copy.deepcopy(self._original_config)
        config = self._patch_config(config, 
                                    train_dataset=train_dataset, 
                                    dataparser_transforms=dataparser_transforms, 
                                    dataparser_scale=dataparser_scale)
        self.config = config
        trainer = self.config.setup()
        trainer.setup()
        if self.checkpoint is not None:
            self.config.load_dir = Path(os.path.join(self.checkpoint, self.config.relative_model_dir))
            trainer._load_checkpoint()
        if train_dataset is not None:
            if getattr(self.config.pipeline.datamanager, "train_num_times_to_repeat_images", None) == -1:
                logging.debug("NerfStudio will cache all images, we will release the memory now")
                train_dataset["images"] = None  # type: ignore
        self._mode = "train"
        self._trainer = trainer

    def train_iteration(self, step: int):
        if self._mode != "train":
            raise RuntimeError("Method is not in train mode. Call setup_train first.")
        self.step = step

        self._trainer.pipeline.train()

        # training callbacks before the training iteration
        for callback in self._trainer.callbacks:
            callback.run_callback_at_location(step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION)

        # time the forward pass
        loss, loss_dict, metrics_dict = self._trainer.train_iteration(step)

        # training callbacks after the training iteration
        for callback in self._trainer.callbacks:
            callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

        metrics = metrics_dict
        metrics.update(loss_dict)
        metrics.update({"loss": loss})
        if hasattr(self.config.pipeline.datamanager, "train_num_rays_per_batch"):
            metrics.update({"num_rays": self.config.pipeline.datamanager.train_num_rays_per_batch})

        def detach(v):
            if torch.is_tensor(v):
                return v.detach().cpu().item()
            elif isinstance(v, np.ndarray):
                return v.item()
            return v

        self.step = step + 1
        return {k: detach(v) for k, v in metrics.items()}

    def save(self, path: str):
        """
        Save model.

        Args:
            path: Path to save.
        """
        assert isinstance(self._trainer, Trainer)
        os.makedirs(path, exist_ok=True)
        bckp = self._trainer.checkpoint_dir
        self._trainer.checkpoint_dir = path
        config_yaml_path = Path(path) / "config.yml"
        config_yaml_path.write_text(yaml.dump(self._original_config), "utf8")
        self._trainer.checkpoint_dir = Path(path) / self._original_config.relative_model_dir
        self._trainer.save_checkpoint(self.step)
        self._trainer.checkpoint_dir = bckp
        self._trainer.pipeline.datamanager.train_dataparser_outputs.save_dataparser_transform(Path(path) / "dataparser_transforms.json")

        # Extend dataparser transforms with reproducible transforms
        with Path(path).joinpath("dataparser_transforms.json").open("r+", encoding="utf8") as f:
            transforms = json.load(f)
            transforms["transform_base64"] = numpy_to_base64(self._trainer.pipeline.datamanager.train_dataparser_outputs.dataparser_transform.numpy())
            f.seek(0)
            f.truncate()
            json.dump(transforms, f, indent=2)

        # Note, since the torch checkpoint does not have deterministic SHA, we compute the SHA here.
        for fname in glob.glob(os.path.join(path, "**/*.ckpt"), recursive=True):
            ckpt = torch.load(fname, map_location="cpu")
            sha = get_torch_checkpoint_sha(ckpt)
            with open(fname + ".sha256", "w", encoding="utf8") as f:
                f.write(sha)

    def close(self):
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

# </nerfstudio.py>
# NOTE: Changes start from here
class WaterSplatting(NerfStudio):
    _default_nerfstudio_name: str = "water-splatting"
    _require_points3D: bool = True

    def _remap_outputs(self, outputs):
        return {
            "color": outputs["rgb"].detach().clamp(0, 1).cpu().numpy(),
            "accumulation": outputs["accumulation"].detach().squeeze(-1).cpu().numpy(),
            "depth": outputs["depth"].detach().squeeze(-1).cpu().numpy(),
            "color_object": outputs["rgb_object"].detach().clamp(0, 1).cpu().numpy(),
            "color_clear": outputs["rgb_clear"].detach().clamp(0, 1).cpu().numpy(),
            "color_medium": outputs["rgb_medium"].detach().clamp(0, 1).cpu().numpy(),
        }
