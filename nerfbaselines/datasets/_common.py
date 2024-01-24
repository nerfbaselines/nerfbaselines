import gc
import logging
import os
import struct
import numpy as np
import PIL.Image
from tqdm import tqdm
from ..types import Dataset
from .. import cameras
from ..utils import padded_stack


def single(xs):
    out = None
    for x in xs:
        if out is not None:
            raise ValueError("Expected single value, got multiple")
        out = x
    if out is None:
        raise ValueError("Expected single value, got none")
    return out


def _dataset_undistort_unsupported(dataset: Dataset, supported_camera_models):
    supported_models_int = set(x.value for x in supported_camera_models)
    undistort_tasks = []
    for i, camera in enumerate(dataset.cameras):
        if camera.camera_types.item() in supported_models_int:
            continue
        undistort_tasks.append((i, camera))
    if len(undistort_tasks) == 0:
        return False

    was_list = isinstance(dataset.images, list)
    new_images = list(dataset.images)
    new_sampling_masks = list(dataset.sampling_masks) if dataset.sampling_masks is not None else None
    dataset.images = new_images
    dataset.sampling_masks = new_sampling_masks

    # Release memory here
    gc.collect()

    for i, camera in tqdm(undistort_tasks, desc="undistorting images"):
        undistorted_camera = cameras.undistort_camera(camera)
        ow, oh = camera.image_sizes
        if dataset.file_paths is not None:
            dataset.file_paths[i] = os.path.join("/undistorted", os.path.split(dataset.file_paths[i])[-1])
        if dataset.sampling_mask_paths is not None:
            dataset.sampling_mask_paths[i] = os.path.join("/undistorted-masks", os.path.split(dataset.sampling_mask_paths[i])[-1])
        warped = cameras.warp_image_between_cameras(camera, undistorted_camera, new_images[i][:oh, :ow])
        new_images[i] = warped
        if new_sampling_masks is not None:
            warped = cameras.warp_image_between_cameras(camera, undistorted_camera, new_sampling_masks[i][:oh, :ow])
            new_sampling_masks[i] = warped
        # IMPORTANT: camera is modified in-place
        dataset.cameras[i] = undistorted_camera
    if not was_list:
        dataset.images = padded_stack(new_images)
        dataset.sampling_masks = padded_stack(new_sampling_masks) if new_sampling_masks is not None else None
    return True


def dataset_load_features(dataset: Dataset, required_features, supported_camera_models=None):
    images = []
    image_sizes = []
    for p in tqdm(dataset.file_paths, desc="loading images"):
        if str(p).endswith(".bin"):
            assert dataset.color_space is None or dataset.color_space == "linear"
            with open(p, "rb") as f:
                data_bytes = f.read()
                h, w = struct.unpack("ii", data_bytes[:8])
                image = np.frombuffer(data_bytes, dtype=np.float16, count=h * w * 4, offset=8).astype(np.float32).reshape([h, w, 4])
            dataset.color_space = "linear"
        else:
            assert dataset.color_space is None or dataset.color_space == "srgb"
            image = np.array(PIL.Image.open(p), dtype=np.uint8)
            dataset.color_space = "srgb"
        images.append(image)
        image_sizes.append([image.shape[1], image.shape[0]])

    dataset.images = images  # padded_stack(images)
    dataset.cameras = dataset.cameras.with_image_sizes(np.array(image_sizes, dtype=np.int32))
    if supported_camera_models is not None:
        if _dataset_undistort_unsupported(dataset, supported_camera_models):
            logging.warning("Some cameras models are not supported by the method. Images have been undistorted. Make sure to use the undistorted images for training.")
    return dataset


class DatasetNotFoundError(Exception):
    pass


class MultiDatasetError(DatasetNotFoundError):
    def __init__(self, errors, message):
        self.errors = errors
        self.message = message
        super().__init__(message + "\n" + "".join(f"\n  {name}: {error}" for name, error in errors.items()))

    def write_to_logger(self, color=True, terminal_width=None):
        if terminal_width is None:
            terminal_width = 120
            try:
                terminal_width = min(os.get_terminal_size().columns, 120)
            except OSError:
                pass
        message = self.message
        if color:
            message = "\33[0m\33[31m" + message + "\33[0m"
        for name, error in self.errors.items():
            prefix = f"   {name}: "
            mlen = terminal_width - len(prefix)
            prefixlen = len(prefix)
            if color:
                prefix = f"\33[96m{prefix}\33[0m"
            rows = [error[i : i + mlen] for i in range(0, len(error), mlen)]
            mdetail = f'\n{" "*prefixlen}'.join(rows)
            message += f"\n{prefix}{mdetail}"
        logging.error(message)
