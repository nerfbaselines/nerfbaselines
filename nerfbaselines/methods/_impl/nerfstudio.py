# pylint: disable=protected-access
import yaml
import os
from functools import partial
import logging
from dataclasses import fields
from pathlib import Path
import copy
import tempfile
from collections import defaultdict
from typing import Iterable, Optional
import numpy as np
from ...types import Method, ProgressCallback, CurrentProgress, MethodInfo
from ...types import Dataset
from ...cameras import CameraModel, Cameras

import torch
from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras as NSCameras, CameraType as NPCameraType
from nerfstudio.models.base_model import Model
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataparserOutputs
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig, VanillaDataManager, InputDataset
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.engine.trainer import TrainingCallbackLocation
from nerfstudio.engine.trainer import Trainer
from nerfstudio.configs.method_configs import all_methods


# Hack to add progress to existing models
@torch.no_grad()
def _hacked_get_outputs_for_camera_ray_bundle(self, camera_ray_bundle, update_callback: Optional[callable] = None):
    input_device = camera_ray_bundle.directions.device
    num_rays_per_chunk = self.config.eval_num_rays_per_chunk
    image_height, image_width = camera_ray_bundle.origins.shape[:2]
    num_rays = len(camera_ray_bundle)
    outputs_lists = defaultdict(list)
    for i in range(0, num_rays, num_rays_per_chunk):
        start_idx = i
        end_idx = i + num_rays_per_chunk
        ray_bundle = camera_ray_bundle.get_row_major_sliced_ray_bundle(start_idx, end_idx)
        # move the chunk inputs to the model device
        ray_bundle = ray_bundle.to(self.device)
        outputs = self.forward(ray_bundle=ray_bundle)
        for output_name, output in outputs.items():  # type: ignore
            if not isinstance(output, torch.Tensor):
                continue
            # move the chunk outputs from the model device back to the device of the inputs.
            outputs_lists[output_name].append(output.to(input_device))
        if update_callback:
            update_callback(min(num_rays, i + num_rays_per_chunk), num_rays)
    outputs = {}
    for output_name, outputs_list in outputs_lists.items():
        outputs[output_name] = torch.cat(outputs_list).view(image_height, image_width, -1)  # type: ignore
    return outputs


class NerfStudio(Method):
    nerfstudio_name: Optional[str] = None
    require_points3D: bool = False

    def __init__(self, nerfstudio_name: Optional[str] = None, checkpoint: str = None, require_points3D: Optional[bool] = None):
        self.checkpoint = checkpoint
        self.nerfstudio_name = nerfstudio_name or self.nerfstudio_name
        if checkpoint is not None:
            # Load nerfstudio checkpoint
            with open(os.path.join(checkpoint, "config.yml"), "r", encoding="utf8") as f:
                config = yaml.load(f, Loader=yaml.Loader)
            self._original_config = copy.deepcopy(config)
            config.get_base_dir = lambda *_: Path(checkpoint)
            config.load_dir = config.get_checkpoint_dir()
        elif self.nerfstudio_name is not None:
            config = all_methods[self.nerfstudio_name]
            self._original_config = copy.deepcopy(config)
        else:
            raise ValueError("Either checkpoint or name must be provided")
        self.config = copy.deepcopy(config)
        super().__init__(batch_size=self.config.pipeline.datamanager.train_num_rays_per_batch)
        self._trainer = None
        self._dm = None
        self.step = 0
        self._tmpdir = tempfile.TemporaryDirectory()
        self._mode = None
        self.dataparser_params = None
        self.require_points3D = require_points3D if require_points3D is not None else self.require_points3D

    @property
    def batch_size(self):
        return self.config.pipeline.datamanager.train_num_rays_per_batch

    def get_info(self) -> MethodInfo:
        features = ("images",)
        if self.require_points3D:
            features = features + ("points3D_xyz", "points3D_rgb")
        info = MethodInfo(
            loaded_step=None,
            num_iterations=self.config.max_num_iterations,
            required_features=frozenset(features),
            supports_undistortion=True,
            batch_size=self.config.pipeline.datamanager.train_num_rays_per_batch,
            eval_batch_size=self.config.pipeline.model.eval_num_rays_per_chunk,
        )
        if self.checkpoint is not None:
            model_path = os.path.join(self.checkpoint, self.config.relative_model_dir)
            if not os.path.exists(model_path):
                raise RuntimeError(f"Model directory {model_path} does not exist")
            info.loaded_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(model_path))[-1]
        return info

    def render(self, cameras: Cameras, progress_callback: Optional[ProgressCallback] = None) -> Iterable[np.ndarray]:
        if self._mode is None:
            self._setup_eval()
        poses = cameras.poses
        poses = torch.from_numpy(poses)
        assert poses.dim() == 3
        poses = self._transform_poses(poses)
        intrinsics = torch.from_numpy(cameras.intrinsics)
        camera_types = [NPCameraType.PERSPECTIVE for _ in range(len(poses))]
        npmap = {x.name.lower(): x.value for x in NPCameraType.__members__.values()}
        npmap["pinhole"] = npmap["perspective"]
        npmap["opencv"] = npmap["perspective"]
        npmap["opencv_fisheye"] = npmap["fisheye"]
        camera_types = [npmap[CameraModel(cameras.camera_types[i]).name.lower()] for i in range(len(poses))]
        sizes = cameras.image_sizes
        cameras = NSCameras(
            camera_to_worlds=poses.contiguous(),
            fx=intrinsics[..., 0].contiguous(),
            fy=intrinsics[..., 1].contiguous(),
            cx=intrinsics[..., 2].contiguous(),
            cy=intrinsics[..., 3].contiguous(),
            distortion_params=torch.from_numpy(cameras.distortion_parameters).contiguous(),
            width=torch.from_numpy(sizes[..., 0]).long().contiguous(),
            height=torch.from_numpy(sizes[..., 1]).long().contiguous(),
            camera_type=torch.tensor(camera_types, dtype=torch.long),
        )
        self._trainer.pipeline.eval()
        global_total = int(sizes.prod(-1).sum())
        global_i = 0
        if progress_callback:
            progress_callback(CurrentProgress(global_i, global_total, 0, len(poses)))
        for i in range(len(poses)):
            ray_bundle = cameras.generate_rays(camera_indices=i, keep_shape=True)
            get_outputs = self._trainer.pipeline.model.get_outputs_for_camera_ray_bundle
            if progress_callback and self._trainer.pipeline.model.__class__.get_outputs_for_camera_ray_bundle == Model.get_outputs_for_camera_ray_bundle:

                def local_progress(i, num_rays):
                    progress_callback(CurrentProgress(global_i + i, global_total, i, len(poses)))

                get_outputs = partial(_hacked_get_outputs_for_camera_ray_bundle, self._trainer.pipeline.model, update_callback=local_progress)
            outputs = get_outputs(ray_bundle)
            global_i += int(sizes[i].prod(-1))
            if progress_callback:
                progress_callback(CurrentProgress(global_i, global_total, i + 1, len(poses)))
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
            self.dataparser_params["dataparser_transform"]
            @ torch.cat([poses, torch.tensor([[[0, 0, 0, 1]]], dtype=self.dataparser_params["dataparser_transform"].dtype).expand((len(poses), 1, 4))], -2)
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

    def setup_train(self, train_dataset: Dataset, *, num_iterations: int):
        method = self
        if self.checkpoint is not None:
            self.dataparser_params = torch.load(os.path.join(self.checkpoint, "dataparser_params.pth"), map_location="cpu")
        self.config = copy.deepcopy(self._original_config)
        # We use this hack to release the memory after the data was copied to cached dataloader
        images_holder = [train_dataset.images]
        del train_dataset.images

        class CustomDataParser(DataParser):
            def __init__(self, config, *args, **kwargs):
                super().__init__(config)
                method._dp = self

            def _generate_dataparser_outputs(self, split: str = "train", **kwargs) -> DataparserOutputs:
                if split != "train":
                    return DataparserOutputs(
                        [],
                        NSCameras(
                            torch.zeros((1, 3, 4), dtype=torch.float32),
                            torch.zeros((1,), dtype=torch.float32),
                            torch.zeros((1,), dtype=torch.float32),
                            torch.zeros((1,), dtype=torch.float32),
                            torch.zeros((1,), dtype=torch.float32),
                            torch.zeros((1,), dtype=torch.long),
                            torch.zeros((1,), dtype=torch.long),
                            torch.zeros((1,), dtype=torch.float32),
                            torch.zeros((1,), dtype=torch.long),
                        ),
                        None,
                        None,
                        [],
                        {},
                    )
                image_names = [f"{i:06d}.png" for i in range(len(train_dataset.cameras.poses))]
                camera_types = [NPCameraType.PERSPECTIVE for _ in range(len(train_dataset.cameras.poses))]
                npmap = {x.name.lower(): x.value for x in NPCameraType.__members__.values()}
                npmap["pinhole"] = npmap["perspective"]
                npmap["opencv"] = npmap["perspective"]
                npmap["opencv_fisheye"] = npmap["fisheye"]
                camera_types = [npmap[CameraModel(train_dataset.cameras.camera_types[i]).name.lower()] for i in range(len(train_dataset.cameras.poses))]

                # in x,y,z order
                # assumes that the scene is centered at the origin
                if train_dataset.metadata.get("type") == "blender":
                    aabb_scale = 1.5
                    method.dataparser_params = dict(dataparser_transform=torch.eye(4, dtype=torch.float32), dataparser_scale=1.0, alpha_color="white")
                else:
                    aabb_scale = 1
                    if method.checkpoint is None:
                        dp_trans, dp_scale = method._get_pose_transform(train_dataset.cameras.poses)
                        method.dataparser_params = dict(
                            dataparser_transform=dp_trans,
                            dataparser_scale=dp_scale,
                        )

                if method.checkpoint is not None:
                    assert method.dataparser_params is not None
                scene_box = SceneBox(aabb=torch.tensor([[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32))
                th_poses = method._transform_poses(torch.from_numpy(train_dataset.cameras.poses).float())
                cameras = NSCameras(
                    camera_to_worlds=th_poses,
                    fx=torch.from_numpy(train_dataset.cameras.intrinsics[..., 0]).contiguous(),
                    fy=torch.from_numpy(train_dataset.cameras.intrinsics[..., 1]).contiguous(),
                    cx=torch.from_numpy(train_dataset.cameras.intrinsics[..., 2]).contiguous(),
                    cy=torch.from_numpy(train_dataset.cameras.intrinsics[..., 3]).contiguous(),
                    distortion_params=torch.from_numpy(train_dataset.cameras.distortion_parameters).contiguous(),
                    width=torch.from_numpy(train_dataset.cameras.image_sizes[..., 0]).long().contiguous(),
                    height=torch.from_numpy(train_dataset.cameras.image_sizes[..., 1]).long().contiguous(),
                    camera_type=torch.tensor(camera_types, dtype=torch.long),
                )
                metadata = {}
                if method.require_points3D:
                    metadata["points3D_xyz"] = torch.from_numpy(train_dataset.points3D_xyz).float()
                    metadata["points3D_rgb"] = torch.from_numpy(train_dataset.points3D_rgb)
                return DataparserOutputs(
                    image_names,
                    cameras,
                    method.dataparser_params.get("alpha_color", None),
                    scene_box,
                    image_names if train_dataset.sampling_masks else None,
                    metadata,
                    dataparser_transform=method.dataparser_params["dataparser_transform"][..., :3, :].contiguous(),  # pylint: disable=protected-access
                    dataparser_scale=method.dataparser_params["dataparser_scale"],
                )  # pylint: disable=protected-access

        self.config.pipeline.datamanager.dataparser._target = CustomDataParser  # pylint: disable=protected-access
        self.config.max_num_iterations = num_iterations

        dm = self.config.pipeline.datamanager
        if dm.__class__.__name__ == "ParallelDataManagerConfig":
            dm = VanillaDataManagerConfig(**{k.name: getattr(dm, k.name) for k in fields(VanillaDataManagerConfig)})
            dm._target = VanillaDataManager  # pylint: disable=protected-access
            self.config.pipeline.datamanager = dm

        class DM(dm._target):  # pylint: disable=protected-access
            @property
            def dataset_type(self):
                class DatasetL(getattr(self, "_idataset_type", InputDataset)):
                    def get_image(self, image_idx: int):
                        img = images_holder[0][image_idx]
                        if img.dtype == np.uint8:
                            img = img.astype(np.float32) / 255.0
                        image = torch.from_numpy(img)
                        if self._dataparser_outputs.alpha_color is not None and image.shape[-1] == 4:
                            image = image[:, :, :3] * image[:, :, -1:] + self._dataparser_outputs.alpha_color * (1.0 - image[:, :, -1:])
                        return image

                return DatasetL

            @dataset_type.setter
            def dataset_type(self, value):
                self._idataset_type = value

        self.config.output_dir = Path(self._tmpdir.name)
        self.config.pipeline.datamanager._target = DM  # pylint: disable=protected-access
        self.config.set_timestamp()
        self.config.vis = None
        self.config.machine.device_type = "cuda"
        self.config.load_step = None
        self._trainer = self.config.setup()
        self._trainer.setup()
        if self.checkpoint is not None:
            self.config.load_dir = Path(os.path.join(self.checkpoint, self.config.relative_model_dir))
            self._trainer._load_checkpoint()
        if getattr(self.config.pipeline.datamanager, "train_num_times_to_repeat_images", None) == -1:
            logging.debug("NerfStudio will cache all images, we will release the memory now")
            images_holder[0] = None
        self._mode = "train"

    def _setup_eval(self):
        if self.checkpoint is None:
            raise RuntimeError("Checkpoint must be provided to setup_eval")
        self.config = copy.deepcopy(self._original_config)
        self.config.output_dir = Path(self._tmpdir.name)

        class DM(self.config.pipeline.datamanager):
            def __init__(self, *args, **kwargs):
                pass

        self.config.pipeline.datamanager._target = DM  # pylint: disable=protected-access
        # Set eval batch size
        self.config.pipeline.model.eval_num_rays_per_chunk = 4096
        self.config.set_timestamp()
        self.config.vis = None
        self.config.machine.device_type = "cuda"
        self.config.load_step = None
        self.config.load_dir = Path(os.path.join(self.checkpoint, self.config.relative_model_dir))
        self._trainer = self.config.setup()
        self._trainer.setup()
        if self.checkpoint is not None:
            self._trainer._load_checkpoint()
        self.dataparser_params = torch.load(os.path.join(self.checkpoint, "dataparser_params.pth"), map_location="cpu")
        self._mode = "eval"

    def _load_checkpoint(self):
        if self.checkpoint is not None:
            load_path = os.path.join(self.checkpoint, self.config.relative_model_dir, f"step-{self.info.loaded_step:09d}.ckpt")
            loaded_state = torch.load(load_path, map_location="cpu")
            self._trainer.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
            print(loaded_state)

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
        if self._mode is None:
            self._setup_eval()
        assert isinstance(self._trainer, Trainer)
        bckp = self._trainer.checkpoint_dir
        self._trainer.checkpoint_dir = Path(path)
        config_yaml_path = Path(path) / "config.yml"
        config_yaml_path.write_text(yaml.dump(self._original_config), "utf8")
        self._trainer.checkpoint_dir = Path(os.path.join(path, self._original_config.relative_model_dir))
        self._trainer.save_checkpoint(self.step)
        self._trainer.checkpoint_dir = bckp
        torch.save({k: v.cpu() if hasattr(v, "cpu") else v for k, v in self.dataparser_params.items()}, os.path.join(path, "dataparser_params.pth"))

    def close(self):
        self._tmpdir.cleanup()
        self._tmpdir = None
