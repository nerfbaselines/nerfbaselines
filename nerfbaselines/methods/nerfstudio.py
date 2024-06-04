# pylint: disable=protected-access
import enum
import os
import dataclasses
from functools import partial
import logging
from dataclasses import fields
from pathlib import Path
import copy
import tempfile
from typing import Iterable, Optional, TypeVar, List, Tuple, Any, Sequence
import numpy as np
from nerfbaselines.types import Method, OptimizeEmbeddingsOutput, MethodInfo, ModelInfo
from nerfbaselines.types import Dataset, RenderOutput
from nerfbaselines.types import Cameras, camera_model_from_int
from nerfbaselines.utils import convert_image_dtype
from nerfbaselines.utils import cast_value, remap_error

import yaml
import torch
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras as NSCameras, CameraType as NPCameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataparserOutputs
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.trainer import TrainingCallbackLocation
from nerfstudio.engine.trainer import Trainer
from nerfstudio.configs.method_configs import all_methods
from nerfstudio.utils.colors import COLORS_DICT


T = TypeVar("T")


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
    def format_value(v, key=None):
        if isinstance(v, (str, float, int, bool, bytes, type(None))):
            return v
        if isinstance(v, (list, tuple)):
            return [format_value(x) for x in v]
        if isinstance(v, dict):
            return {k: format_value(x) for k, x in v.items()}
        if isinstance(v, type):
            return v.__module__ + ":" + v.__name__
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, enum.Enum):
            return v.name
        if dataclasses.is_dataclass(v):
            return flatten_hparams(v, separator=separator, _prefix=_prefix)
        return repr(v)
        raise ValueError(f"Unsupported type {type(v)} for key {key}")

    flat = {}
    if dataclasses.is_dataclass(hparams):
        hparams = {f.name: getattr(hparams, f.name) for f in dataclasses.fields(hparams)}
    for k, v in hparams.items():
        if _prefix:
            k = f"{_prefix}{separator}{k}"
        if isinstance(v, dict) or dataclasses.is_dataclass(v):
            flat.update(flatten_hparams(v, _prefix=k))
        else:
            flat[k] = format_value(v, key=k)
    return flat



def _get_config_dict(config):
    def dict_factory(items: List[Tuple[str, Any]]):
        def _safe_value(v):
            if isinstance(v, Path):
                return str(v)
            if isinstance(v, (bool, str, int, float, bytes, type(None))):
                return v
            if isinstance(v, np.ndarray):
                return v.tolist()
            if isinstance(v, torch.Tensor):
                return _safe_value(v.detach().cpu().numpy())
            if isinstance(v, enum.Enum):
                return v.name
            return str(v)
        return {k: _safe_value(v) for k, v in items}
    return dataclasses.asdict(config, dict_factory=dict_factory)

class _CustomDataParser(DataParser):
    def __init__(self, method, dataset, config, *args, **kwargs):
        self.dataset = dataset
        self.method = method
        super().__init__(config)

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs) -> DataparserOutputs:
        if split != "train" or self.dataset is None:
            aabb_scale = self.method.dataparser_params["aabb_scale"]
            scene_box = SceneBox(aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32))
            num_images = self.method.dataparser_params["num_images"]
            return DataparserOutputs(
                [Path("_.png") for _ in range(num_images)],
                NSCameras(
                    torch.zeros((num_images, 3, 4), dtype=torch.float32),
                    torch.zeros((num_images,), dtype=torch.float32),
                    torch.zeros((num_images,), dtype=torch.float32),
                    torch.zeros((num_images,), dtype=torch.float32),
                    torch.zeros((num_images,), dtype=torch.float32),
                    torch.zeros((num_images,), dtype=torch.long),
                    torch.zeros((num_images,), dtype=torch.long),
                    torch.zeros((num_images,), dtype=torch.float32),
                    torch.zeros((num_images,), dtype=torch.long),
                ),
                None,
                scene_box,
                None,
                {},
                dataparser_transform=self.method.dataparser_params["dataparser_transform"][..., :3, :].contiguous(),
                dataparser_scale=self.method.dataparser_params["dataparser_scale"],
            )
        image_names = [f"{i:06d}.png" for i in range(len(self.dataset["cameras"].poses))]
        camera_types = [NPCameraType.PERSPECTIVE for _ in range(len(self.dataset["cameras"].poses))]
        npmap = {x.name.lower(): x.value for x in NPCameraType.__members__.values()}
        npmap["pinhole"] = npmap["perspective"]
        npmap["opencv"] = npmap["perspective"]
        npmap["opencv_fisheye"] = npmap["fisheye"]
        camera_types = [npmap[camera_model_from_int(self.dataset["cameras"].camera_types[i])] for i in range(len(self.dataset["cameras"].poses))]

        poses = self.dataset["cameras"].poses.copy()
       
        # Convert from Opencv to OpenGL coordinate system
        poses[..., 0:3, 1:3] *= -1

        # in x,y,z order
        # assumes that the scene is centered at the origin
        if self.dataset["metadata"].get("name") == "blender":
            aabb_scale = 1.5
            self.method.dataparser_params = dict(dataparser_transform=torch.eye(4, dtype=torch.float32), dataparser_scale=1.0, aabb_scale=1.5, num_images=len(image_names))
        else:
            aabb_scale = 1
            if self.method.checkpoint is None:
                dp_trans, dp_scale = self.method._get_pose_transform(poses)
                self.method.dataparser_params = dict(
                    dataparser_transform=dp_trans,
                    dataparser_scale=dp_scale,
                    aabb_scale=1.0,
                    num_images=len(image_names),
                )

        if self.method.checkpoint is not None:
            assert self.method.dataparser_params is not None
        self.method._patch_dataparser_params()
        scene_box = SceneBox(aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32))
        th_poses = self.method._transform_poses(torch.from_numpy(poses).float())
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
            camera_type=torch.tensor(camera_types, dtype=torch.long),
        )
        metadata = {}
        transform_matrix = self.method.dataparser_params["dataparser_transform"]
        scale_factor = self.method.dataparser_params["dataparser_scale"]
        if self.method._require_points3D:
            assert self.dataset["points3D_xyz"] is not None, "Points3D are required but not provided"
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
            self.method.dataparser_params.get("alpha_color", None),
            scene_box,
            image_names if self.dataset["sampling_masks"] else None,
            metadata,
            dataparser_transform=transform_matrix[..., :3, :].contiguous(),  # pylint: disable=protected-access
            dataparser_scale=scale_factor,
        )  # pylint: disable=protected-access


class NerfStudio(Method):
    _method_name: str = "nerfstudio"
    _nerfstudio_name: Optional[str] = None
    _require_points3D: bool = False

    @remap_error
    def __init__(self, *,
                 checkpoint: Optional[str] = None,
                 train_dataset: Optional[Dataset] = None, 
                 config_overrides: Optional[dict] = None):
        assert self._nerfstudio_name is not None, "nerfstudio_name must be set in the subclass"
        self.checkpoint = str(checkpoint) if checkpoint is not None else None
        if checkpoint is not None:
            # Load nerfstudio checkpoint
            with open(os.path.join(checkpoint, "config.yml"), "r", encoding="utf8") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            self._original_config = copy.deepcopy(config)
            config.get_base_dir = lambda *_: Path(checkpoint)
            config.load_dir = config.get_checkpoint_dir()
        elif self._nerfstudio_name is not None:
            config = all_methods[self._nerfstudio_name]
            self._original_config = copy.deepcopy(config)
        else:
            raise ValueError("Either checkpoint or name must be provided")
        self.config = copy.deepcopy(config)
        self._trainer = None
        self._dm = None
        self.step = 0
        self._tmpdir = tempfile.TemporaryDirectory()
        self._mode = None
        self.dataparser_params = None

        if train_dataset is not None:
            self._apply_config_patch_for_dataset(self._original_config, train_dataset)
        for k, v in (config_overrides or {}).items():
            if not _config_safe_set(self._original_config, k, v):
                raise ValueError(f"Invalid config key {k}")
        if train_dataset is not None:
            self._setup_train(train_dataset)
        else:
            self._setup_eval()

    def _patch_dataparser_params(self):
        pass

    def _apply_config_patch_for_dataset(self, config, dataset: Dataset):
        # See https://github.com/nerfstudio-project/nerfstudio/tree/SIGGRAPH-2023-Code
        # NOTE: we change the average_init_density!!
        use_original_average_init_density = False
        if dataset["metadata"].get("name") == "blender":
            logging.info("Applying config patch for dataset type blender")
            if not _config_safe_set(config, "pipeline.model.near_plane", 2.0):
                logging.warning("Flag pipeline.model.near_plane is not set (required for blender-type dataset)")
            if not _config_safe_set(config, "pipeline.model.far_plane", 6.0):
                logging.warning("Flag pipeline.model.far_plane is not set (required for blender-type dataset)")
            if not _config_safe_set(config, "pipeline.datamanager.camera_optimizer.mode", "off") and not _config_safe_set(config, "pipeline.model.camera_optimizer.mode", "off"):
                logging.warning("Flag pipeline.datamanager.camera_optimizer.mode is not set (required for blender-type dataset)")
            if not _config_safe_set(config, "pipeline.model.use_appearance_embedding", False):
                logging.warning("Flag pipeline.model.use_appearance_embedding is not set (required for mipnerf360-type dataset)")
            if not _config_safe_set(config, "pipeline.model.background_color", "white"):
                logging.warning("Flag pipeline.model.background_color is not set (required for blender-type dataset)")
            if not _config_safe_set(config, "pipeline.model.proposal_initial_sampler", "uniform"):
                logging.warning("Flag pipeline.model.proposal_initial_sampler is not set (required for blender-type dataset)")
            if not _config_safe_set(config, "pipeline.model.distortion_loss_mult", 0.0):
                logging.warning("Flag pipeline.model.distortion_loss_mult is not set (required for blender-type dataset)")
            if not _config_safe_set(config, "pipeline.model.disable_scene_contraction", True):
                logging.warning("Flag pipeline.model.disable_scene_contraction is not set (required for blender-type dataset)")
            if not _config_safe_set(config, "pipeline.model.average_init_density", 1.0):
                logging.warning("Flag pipeline.model.average_init_density is not set (required for blender-type dataset)")

        if dataset["metadata"].get("name") in ("mipnerf360", "tanksandtemples"):
            old_num_iter = getattr(config, "max_num_iterations", 0)
            if old_num_iter < 70000:
                if not _config_safe_set(config, "max_num_iterations", 70000):
                    logging.warning("Flag max_num_iterations is not set (required for mipnerf360-type dataset)")
            if not _config_safe_set(config, "pipeline.datamanager.camera_optimizer.mode", "off") and not _config_safe_set(config, "pipeline.model.camera_optimizer.mode", "off"):
                logging.warning("Flag pipeline.datamanager.camera_optimizer.mode is not set (required for mipnerf360-type dataset)")
            if not _config_safe_set(config, "pipeline.model.use_appearance_embedding", False):
                logging.warning("Flag pipeline.model.use_appearance_embedding is not set (required for mipnerf360-type dataset)")
            if use_original_average_init_density:
                if not _config_safe_set(config, "pipeline.model.average_init_density", 1.0):
                    logging.warning("Flag pipeline.model.average_init_density is not set (required for mipnerf360-type dataset)")

        if dataset["metadata"].get("name") == "nerfstudio":
            if use_original_average_init_density:
                if not _config_safe_set(config, "pipeline.model.average_init_density", 1.0):
                    logging.warning("Flag pipeline.model.average_init_density is not set (required for nerfstudio-type dataset)")

    @property
    def batch_size(self):
        return self.config.pipeline.datamanager.train_num_rays_per_batch

    @classmethod
    def get_method_info(cls) -> MethodInfo:
        features = ("color",)
        if cls._require_points3D:
            features = features + ("points3D_xyz", "points3D_rgb")
        return MethodInfo(
            name=cls._method_name,
            required_features=frozenset(features),
            supported_camera_models=frozenset(
                (
                    "pinhole",
                    "opencv",
                    "opencv_fisheye",
                )
            ),
        )

    def get_info(self) -> ModelInfo:
        info = ModelInfo(
            loaded_step=None,
            loaded_checkpoint=str(self.checkpoint) if self.checkpoint is not None else None,
            num_iterations=self.config.max_num_iterations,
            batch_size=int(self.config.pipeline.datamanager.train_num_rays_per_batch),
            eval_batch_size=int(self.config.pipeline.model.eval_num_rays_per_chunk),
            hparams=flatten_hparams(self.config, separator="."),
            **self.get_method_info()
        )
        if self.checkpoint is not None:
            model_path = os.path.join(self.checkpoint, self.config.relative_model_dir)
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model directory {model_path} does not exist")
            info["loaded_step"] = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(model_path))[-1]
        return info

    @torch.no_grad()
    def render(self, cameras: Cameras, embeddings=None) -> Iterable[RenderOutput]:
        if embeddings is not None:
            raise NotImplementedError(f"Optimizing embeddings is not supported for method {self.get_method_info()['name']}")
        poses = cameras.poses.copy()

        # Convert from Opencv to OpenGL coordinate system
        poses[..., 0:3, 1:3] *= -1

        poses = torch.from_numpy(poses)
        assert poses.dim() == 3
        poses = self._transform_poses(poses)
        intrinsics = torch.from_numpy(cameras.intrinsics)
        camera_types = [NPCameraType.PERSPECTIVE for _ in range(len(poses))]
        npmap = {x.name.lower(): x.value for x in NPCameraType.__members__.values()}
        npmap["pinhole"] = npmap["perspective"]
        npmap["opencv"] = npmap["perspective"]
        npmap["opencv_fisheye"] = npmap["fisheye"]
        camera_types = [npmap[camera_model_from_int(cameras.camera_types[i])] for i in range(len(poses))]
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
            camera_type=torch.tensor(camera_types, dtype=torch.long),
        ).to(self._trainer.pipeline.device)
        self._trainer.pipeline.eval()
        global_i = 0
        for i in range(len(poses)):
            ray_bundle = ns_cameras.generate_rays(camera_indices=i, keep_shape=True)
            get_outputs = self._trainer.pipeline.model.get_outputs_for_camera_ray_bundle
            outputs = get_outputs(ray_bundle)
            global_i += int(sizes[i].prod(-1))
            color = self._trainer.pipeline.model.get_rgba_image(outputs)
            color = color.detach().cpu().numpy()
            out = {
                "color": color,
                "accumulation": outputs["accumulation"].detach().cpu().numpy(),
            }
            if "depth" in outputs:
                out["depth"] = outputs["depth"].view(*outputs["depth"].shape[:2]).detach().cpu().numpy()
            yield out
        self._trainer.pipeline.train()

    def _transform_poses(self, poses):
        assert poses.dim() == 3
        poses = (
            self.dataparser_params["dataparser_transform"].to(poses.dtype)
            @ torch.cat([poses, torch.tensor([[[0, 0, 0, 1]]], dtype=poses.dtype).expand((len(poses), 1, 4))], -2)
        )[:, :3, :].contiguous()
        poses[:, :3, 3] *= self.dataparser_params["dataparser_scale"]
        return poses

    def _get_pose_transform(self, poses):
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

    def _patch_datamanager(self, config, dataset=None, _images_holder=None):
        config.pipeline.datamanager.dataparser._target = partial(_CustomDataParser, self, dataset)

        dm = config.pipeline.datamanager
        if dm.__class__.__name__ == "ParallelDataManagerConfig":
            dm = VanillaDataManagerConfig(**{k.name: getattr(dm, k.name) for k in fields(VanillaDataManagerConfig)})
            dm._target = VanillaDataManager  # pylint: disable=protected-access
            config.pipeline.datamanager = dm

        def get_dataset_class(dataset_type):
            class DatasetL(dataset_type):
                def get_numpy_image(self, image_idx: int):
                    return _images_holder[0][image_idx]

                def get_image(self, image_idx: int):
                    img = self.get_numpy_image(image_idx)
                    img = convert_image_dtype(img, np.float32)
                    image = torch.from_numpy(img)
                    if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
                        if isinstance(self._dataparser_outputs.alpha_color, str):
                            alpha_color = COLORS_DICT[self._dataparser_outputs.alpha_color]
                        else:
                            alpha_color = torch.from_numpy(np.array(alpha_color, dtype=np.float32))
                        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
                    return image

            return DatasetL

        class DM(dm._target):  # pylint: disable=protected-access
            @property
            def dataset_type(self):
                return get_dataset_class(getattr(self, "_idataset_type", InputDataset))

            @dataset_type.setter
            def dataset_type(self, value):
                self._idataset_type = value

        if dataset is None:
            # setup_train is not called for eval dataset
            def setup_train(self, *args, **kwargs):
                if self.config.camera_optimizer is not None:
                    self.train_camera_optimizer = self.config.camera_optimizer.setup(num_cameras=self.train_dataset["cameras"].size, device=self.device)

            DM.setup_train = setup_train
            # setup_eval is not called for eval dataset
            DM.setup_eval = lambda *args, **kwargs: None
        config.pipeline.datamanager._target = DM  # pylint: disable=protected-access

    def _setup_train(self, train_dataset: Dataset):
        if self.checkpoint is not None:
            self.dataparser_params = torch.load(os.path.join(self.checkpoint, "dataparser_params.pth"), map_location="cpu")
        self.config = copy.deepcopy(self._original_config)
        # We use this hack to release the memory after the data was copied to cached dataloader
        images_holder = [train_dataset["images"]]

        self._patch_datamanager(self.config, train_dataset, images_holder)
        self.config.output_dir = Path(self._tmpdir.name)
        self.config.set_timestamp()
        self.config.vis = None
        self.config.machine.device_type = "cuda"
        self.config.load_step = None
        trainer = self.config.setup()
        trainer.setup()
        if self.checkpoint is not None:
            self.config.load_dir = Path(os.path.join(self.checkpoint, self.config.relative_model_dir))
            trainer._load_checkpoint()
        if getattr(self.config.pipeline.datamanager, "train_num_times_to_repeat_images", None) == -1:
            logging.debug("NerfStudio will cache all images, we will release the memory now")
            images_holder[0] = None
        self._mode = "train"
        self._trainer = trainer

    def _setup_eval(self):
        if self.checkpoint is None:
            raise RuntimeError("Checkpoint must be provided to setup_eval")
        self.dataparser_params = torch.load(os.path.join(self.checkpoint, "dataparser_params.pth"), map_location="cpu")
        self.config = copy.deepcopy(self._original_config)
        self.config.output_dir = Path(self._tmpdir.name)
        self._patch_datamanager(self.config, None, None)

        # Set eval batch size
        self.config.pipeline.model.eval_num_rays_per_chunk = 4096
        self.config.set_timestamp()
        self.config.vis = None
        self.config.machine.device_type = "cuda"
        self.config.load_step = None
        self.config.load_dir = Path(os.path.join(self.checkpoint, self.config.relative_model_dir))
        trainer = self.config.setup()
        trainer.setup()
        trainer._load_checkpoint()
        self._mode = "eval"
        self._trainer = trainer

    def _load_checkpoint(self):
        if self.checkpoint is not None:
            load_path = os.path.join(self.checkpoint, self.config.relative_model_dir, f"step-{self.get_info().loaded_step:09d}.ckpt")
            loaded_state = torch.load(load_path, map_location="cpu")
            self._trainer.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])

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
        metrics.update({"num_rays": self.config.pipeline.datamanager.train_num_rays_per_batch})

        def detach(v):
            if torch.is_tensor(v):
                return v.detach().cpu().item()
            elif isinstance(v, np.ndarray):
                return v.item()
            assert isinstance(v, (str, float, int))
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
        bckp = self._trainer.checkpoint_dir
        self._trainer.checkpoint_dir = path
        config_yaml_path = Path(path) / "config.yml"
        config_yaml_path.write_text(yaml.dump(self._original_config), "utf8")
        self._trainer.checkpoint_dir = Path(path) / self._original_config.relative_model_dir
        self._trainer.save_checkpoint(self.step)
        self._trainer.checkpoint_dir = bckp
        torch.save({k: v.cpu() if hasattr(v, "cpu") else v for k, v in self.dataparser_params.items()}, str(Path(path) / "dataparser_params.pth"))

    def close(self):
        self._tmpdir.cleanup()
        self._tmpdir = None

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

