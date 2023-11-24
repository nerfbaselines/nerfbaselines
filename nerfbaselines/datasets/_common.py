import struct
from typing import List
import numpy as np
import PIL.Image
from tqdm import tqdm
from ..types import Dataset


def single(xs):
    out = None
    for x in xs:
        if out is not None:
            raise ValueError("Expected single value, got multiple")
        out = x
    if out is None:
        raise ValueError("Expected single value, got none")
    return out


def padded_stack(tensors: List[np.ndarray]) -> np.ndarray:
    max_shape = tuple(max(s) for s in zip(*[x.shape for x in tensors]))
    out_tensors = []
    for x in tensors:
        pads = [(0, m - s) for s, m in zip(x.shape, max_shape)]
        out_tensors.append(np.pad(x, pads))
    return np.stack(out_tensors, 0)


def dataset_load_features(dataset: Dataset, required_features):
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
            image = np.array(PIL.Image.open(p).convert("RGB"), dtype=np.uint8)
            dataset.color_space = "srgb"
        images.append(image)
        image_sizes.append([image.shape[1], image.shape[0]])

    dataset.images = padded_stack(images)
    dataset.cameras = dataset.cameras.with_image_sizes(np.array(image_sizes, dtype=np.int32))

    if "sampling_masks" in required_features and dataset.sampling_mask_paths is not None:
        images = []
        for p in tqdm(dataset.file_paths, desc="loading masks"):
            image = np.array(PIL.Image.open(p).convert("L"), dtype=np.float32)
            images.append(image)
        dataset.sampling_masks = padded_stack(images)
    return dataset


class DatasetNotFoundError(Exception):
    pass
