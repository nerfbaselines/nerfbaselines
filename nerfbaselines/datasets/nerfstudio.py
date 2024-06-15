import os
import logging
from pathlib import Path
import math
import shutil
import json
import zipfile
from typing import Optional, List, Tuple, Dict, Union

import numpy as np
from PIL import Image

from ._colmap_utils import read_points3D_binary, read_points3D_text, read_images_binary, read_images_text
from ._common import DatasetNotFoundError, get_scene_scale, get_default_viewer_transform, new_dataset, dataset_index_select
from ..types import CameraModel, camera_model_to_int, FrozenSet, DatasetFeature, get_args, new_cameras
from ..io import wget


MAX_AUTO_RESOLUTION = 1600

CAMERA_MODEL_TO_TYPE: Dict[str, Optional[CameraModel]] = {
    "SIMPLE_PINHOLE": "pinhole",
    "PINHOLE": "pinhole",
    "SIMPLE_RADIAL": "opencv",
    "RADIAL": "opencv",
    "OPENCV": "opencv",
    "OPENCV_FISHEYE": "opencv_fisheye",
    "EQUIRECTANGULAR": None,
    "OMNIDIRECTIONALSTEREO_L": None,
    "OMNIDIRECTIONALSTEREO_R": None,
    "VR180_L": None,
    "VR180_R": None,
}


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)


def get_train_eval_split_fraction(image_filenames: List, train_split_fraction: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split fraction based on the number of images and the train split fraction.

    Args:
        image_filenames: list of image filenames
        train_split_fraction: fraction of images to use for training
    """

    # filter image_filenames and poses based on train/eval split percentage
    num_images = len(image_filenames)
    num_train_images = math.ceil(num_images * train_split_fraction)
    num_eval_images = num_images - num_train_images
    i_all = np.arange(num_images)
    i_train = np.linspace(0, num_images - 1, num_train_images, dtype=int)  # equally spaced training images starting and ending at 0 and num_images-1
    i_eval = np.setdiff1d(i_all, i_train)  # eval images are the remaining images
    assert len(i_eval) == num_eval_images

    return i_train, i_eval


def get_train_eval_split_filename(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the filename of the images.

    Args:
        image_filenames: list of image filenames
    """

    num_images = len(image_filenames)
    basenames = [os.path.basename(image_filename) for image_filename in image_filenames]
    i_all = np.arange(num_images)
    i_train = []
    i_eval = []
    for idx, basename in zip(i_all, basenames):
        # check the frame index
        if "train" in basename:
            i_train.append(idx)
        elif "eval" in basename:
            i_eval.append(idx)
        else:
            raise ValueError("frame should contain train/eval in its name to use this eval-frame-index eval mode")

    return np.array(i_train), np.array(i_eval)


def get_train_eval_split_interval(image_filenames: List, eval_interval: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split based on the interval of the images.

    Args:
        image_filenames: list of image filenames
        eval_interval: interval of images to use for eval
    """

    num_images = len(image_filenames)
    all_indices = np.arange(num_images)
    train_indices = all_indices[all_indices % eval_interval != 0]
    eval_indices = all_indices[all_indices % eval_interval == 0]
    i_train = train_indices
    i_eval = eval_indices

    return i_train, i_eval


def get_train_eval_split_all(image_filenames: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the train/eval split where all indices are used for both train and eval.

    Args:
        image_filenames: list of image filenames
    """
    num_images = len(image_filenames)
    i_all = np.arange(num_images)
    i_train = i_all
    i_eval = i_all
    return i_train, i_eval


def load_nerfstudio_dataset(path: Union[Path, str], split: str, downscale_factor: Optional[int] = None, features: Optional[FrozenSet[DatasetFeature]] = None, **kwargs):
    path = Path(path)
    downscale_factor_original = downscale_factor
    downscale_factor = None

    train_split_fraction: float = 0.9
    """The percentage of the dataset to use for training. Only used when eval_mode is train-split-fraction."""
    eval_interval: int = 8
    """The interval between frames to use for eval. Only used when eval_mode is eval-interval."""
    # depth_unit_scale_factor: float = 1e-3
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""

    # Literal["fraction", "filename", "interval", "all"]
    eval_mode = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.
    """

    def _get_fname(filepath: Path, data_dir: Path, downsample_folder_prefix="images_") -> Path:
        """Get the filename of the image file.
        downsample_folder_prefix can be used to point to auxiliary image data, e.g. masks

        filepath: the base file name of the transformations.
        data_dir: the directory of the data that contains the transform file
        downsample_folder_prefix: prefix of the newly generated downsampled images
        """
        nonlocal downscale_factor

        if downscale_factor is None:
            if downscale_factor_original is None:
                test_img = Image.open(data_dir / filepath)
                h, w = test_img.size
                max_res = max(h, w)
                df = 0
                while True:
                    if (max_res / 2 ** (df)) < MAX_AUTO_RESOLUTION:
                        break
                    if not (data_dir / f"{downsample_folder_prefix}{2**(df+1)}" / filepath.name).exists():
                        break
                    df += 1

                downscale_factor = 2**df
                logging.info(f"Auto image downscale factor of {downscale_factor}")
            else:
                downscale_factor = downscale_factor_original

        # pyright workaround
        assert downscale_factor is not None

        if downscale_factor > 1:
            return data_dir / f"{downsample_folder_prefix}{downscale_factor}" / filepath.name
        return data_dir / filepath

    assert path.exists(), f"Data directory {path} does not exist."

    if path.suffix == ".json":
        meta = load_from_json(path)
        data_dir = path.parent
    elif (path / "transforms.json").exists():
        meta = load_from_json(path / "transforms.json")
        data_dir = path
    else:
        raise DatasetNotFoundError(f"Could not find transforms.json in {path}")

    image_filenames: List[str] = []
    mask_filenames: List[str] = []
    depth_filenames = []
    poses = []

    fx_fixed = "fl_x" in meta
    fy_fixed = "fl_y" in meta
    cx_fixed = "cx" in meta
    cy_fixed = "cy" in meta
    height_fixed = "h" in meta
    width_fixed = "w" in meta
    distort_fixed = False
    for distort_key in ["k1", "k2", "k3", "k4", "p1", "p2"]:
        if distort_key in meta:
            distort_fixed = True
            break
    fx = []
    fy = []
    cx = []
    cy = []
    height = []
    width = []
    distort = []

    # sort the frames by fname
    fnames = []
    for frame in meta["frames"]:
        filepath = Path(frame["file_path"])
        fname = _get_fname(filepath, data_dir)
        fnames.append(fname)
    inds = np.argsort(fnames)
    frames = [meta["frames"][ind] for ind in inds]
    assert downscale_factor is not None, "downscale_factor should be set by now"

    for frame in frames:
        filepath = Path(frame["file_path"])
        fname = _get_fname(filepath, data_dir)

        if not fx_fixed:
            assert "fl_x" in frame, "fx not specified in frame"
            fx.append(float(frame["fl_x"]))
        if not fy_fixed:
            assert "fl_y" in frame, "fy not specified in frame"
            fy.append(float(frame["fl_y"]))
        if not cx_fixed:
            assert "cx" in frame, "cx not specified in frame"
            cx.append(float(frame["cx"]))
        if not cy_fixed:
            assert "cy" in frame, "cy not specified in frame"
            cy.append(float(frame["cy"]))
        if not height_fixed:
            assert "h" in frame, "height not specified in frame"
            height.append(int(frame["h"]))
        if not width_fixed:
            assert "w" in frame, "width not specified in frame"
            width.append(int(frame["w"]))
        if not distort_fixed:
            distort.append(
                np.array(
                    [
                        float(frame["k1"]) if "k1" in frame else 0.0,
                        float(frame["k2"]) if "k2" in frame else 0.0,
                        float(frame["p1"]) if "p1" in frame else 0.0,
                        float(frame["p2"]) if "p2" in frame else 0.0,
                        float(frame["k3"]) if "k3" in frame else 0.0,
                        float(frame["k4"]) if "k4" in frame else 0.0,
                    ],
                    dtype=np.float32,
                )
            )

        image_filenames.append(str(fname))
        poses.append(np.array(frame["transform_matrix"]))
        if "mask_path" in frame:
            mask_filepath = Path(frame["mask_path"])
            mask_fname = _get_fname(
                mask_filepath,
                data_dir,
                downsample_folder_prefix="masks_",
            )
            mask_filenames.append(str(mask_fname))

        if "depth_file_path" in frame:
            depth_filepath = Path(frame["depth_file_path"])
            depth_fname = _get_fname(depth_filepath, data_dir, downsample_folder_prefix="depths_")
            depth_filenames.append(depth_fname)

    assert len(mask_filenames) == 0 or (
        len(mask_filenames) == len(image_filenames)
    ), """
    Different number of image and mask filenames.
    You should check that mask_path is specified for every frame (or zero frames) in transforms.json.
    """
    assert len(depth_filenames) == 0 or (
        len(depth_filenames) == len(image_filenames)
    ), """
    Different number of image and depth filenames.
    You should check that depth_file_path is specified for every frame (or zero frames) in transforms.json.
    """

    has_split_files_spec = any(f"{split}_filenames" in meta for split in ("train", "val", "test"))
    if f"{split}_filenames" in meta:
        # Validate split first
        split_filenames = set(_get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"])
        unmatched_filenames = split_filenames.difference(image_filenames)
        if unmatched_filenames:
            raise RuntimeError(f"Some filenames for split {split} were not found: {unmatched_filenames}.")

        indices = [i for i, path in enumerate(image_filenames) if path in split_filenames]
        logging.warning(f"Dataset is overriding {split}_indices to {indices}")
        indices = np.array(indices, dtype=np.int32)
    elif has_split_files_spec:
        raise RuntimeError(f"The dataset's list of filenames for split {split} is missing.")
    else:
        # find train and eval indices based on the eval_mode specified
        if eval_mode == "fraction":
            i_train, i_eval = get_train_eval_split_fraction(image_filenames, train_split_fraction)
        elif eval_mode == "filename":
            i_train, i_eval = get_train_eval_split_filename(image_filenames)
        elif eval_mode == "interval":
            i_train, i_eval = get_train_eval_split_interval(image_filenames, eval_interval)
        elif eval_mode == "all":
            logging.warning("Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results.")
            i_train, i_eval = get_train_eval_split_all(image_filenames)
        else:
            raise ValueError(f"Unknown eval mode {eval_mode}")

        if split == "train":
            indices = i_train
        elif split in ["val", "test"]:
            indices = i_eval
        else:
            raise ValueError(f"Unknown dataparser split {split}")

    poses = np.array(poses).astype(np.float32)

    # if "orientation_override" in meta:
    #     orientation_method = meta["orientation_override"]
    #     CONSOLE.log(f"[yellow] Dataset is overriding orientation method to {orientation_method}")
    # else:
    #     orientation_method = self.config.orientation_method
    # poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
    #     poses,
    #     method=orientation_method,
    #     center_method=self.config.center_method,
    # )

    # # Scale poses
    # scale_factor = 1.0
    # if self.config.auto_scale_poses:
    #     scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
    # scale_factor *= self.config.scale_factor
    # poses[:, :3, 3] *= scale_factor

    # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.

    # # in x,y,z order
    # # assumes that the scene is centered at the origin
    # aabb_scale = 1.0
    # scene_box = SceneBox(
    #     aabb=torch.tensor(
    #         [[-aabb_scale, -aabb_scale, -aabb_scale], [aabb_scale, aabb_scale, aabb_scale]], dtype=torch.float32
    #     )
    # )

    if "camera_model" in meta:
        camera_type = CAMERA_MODEL_TO_TYPE.get(meta.get("camera_model"))
        if camera_type is None or camera_type not in get_args(CameraModel):
            raise NotImplementedError(f"Camera model {meta.get('camera_model')} is not supported.")
    else:
        if distort_fixed:
            has_distortion = any(meta[x] != 0.0 for x in ["k1", "k2", "p1", "p2", "k3", "k4"])
        else:
            has_distortion = any(np.any(x != 0.0) for x in distort)
        camera_type = "opencv" if has_distortion else "pinhole"

    fx = np.full((len(poses),), meta["fl_x"], dtype=np.float32) if fx_fixed else np.array(fx, dtype=np.float32)
    fy = np.full((len(poses),), meta["fl_y"], dtype=np.float32) if fy_fixed else np.array(fy, dtype=np.float32)
    cx = np.full((len(poses),), meta["cx"], dtype=np.float32) if cx_fixed else np.array(cx, dtype=np.float32)
    cy = np.full((len(poses),), meta["cy"], dtype=np.float32) if cy_fixed else np.array(cy, dtype=np.float32)
    height = np.full((len(poses),), meta["h"], dtype=np.int32) if height_fixed else np.array(height, dtype=np.int32)
    width = np.full((len(poses),), meta["w"], dtype=np.int32) if width_fixed else np.array(width, dtype=np.int32)
    if distort_fixed:
        distortion_params = np.repeat(
            np.array(
                [
                    float(meta["k1"]) if "k1" in meta else 0.0,
                    float(meta["k2"]) if "k2" in meta else 0.0,
                    float(meta["p2"]) if "p1" in meta else 0.0,
                    float(meta["p1"]) if "p2" in meta else 0.0,
                    float(meta["k3"]) if "k3" in meta else 0.0,
                    float(meta["k4"]) if "k4" in meta else 0.0,
                ]
            )[None, :],
            len(poses),
            0,
        )
    else:
        distortion_params = np.stack(distort, 0)

    c2w = poses[:, :3, :4]

    # Convert from OpenGL to OpenCV coordinate system
    c2w[0:3, 1:3] *= -1

    all_cameras = new_cameras(
        poses=c2w.astype(np.float32),
        intrinsics=np.stack([fx, fy, cx, cy], -1).astype(np.float32),
        camera_types=np.full((len(poses),), camera_model_to_int(camera_type), dtype=np.uint8),
        distortion_parameters=distortion_params.astype(np.float32),
        image_sizes=np.stack([height, width], -1).astype(np.int32),
        nears_fars=None,
    )

    # transform_matrix = torch.eye(4, dtype=torch.float32)
    # scale_factor = 1.0
    # if "applied_transform" in meta:
    #     applied_transform = torch.tensor(meta["applied_transform"], dtype=transform_matrix.dtype)
    #     transform_matrix = transform_matrix @ torch.cat(
    #         [applied_transform, torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype)], 0
    #     )
    #     transform_matrix
    # if "applied_scale" in meta:
    #     applied_scale = float(meta["applied_scale"])
    #     scale_factor *= applied_scale
    if downscale_factor > 1:
        images_root = data_dir / f"images_{downscale_factor}"
        sampling_masks_root = data_dir / f"masks_{downscale_factor}"
    else:
        images_root = data_dir
        sampling_masks_root = data_dir

    # "depth_filenames": depth_filenames if len(depth_filenames) > 0 else None,
    # "depth_unit_scale_factor": depth_unit_scale_factor,

    points3D_rgb = None
    points3D_xyz = None
    images_points3D_indices: Optional[List[np.ndarray]] = None
    if "points3D_xyz" in (features or {}):
        colmap_path = data_dir / "colmap" / "sparse" / "0"
        if not colmap_path.exists():
            colmap_path = data_dir / "sparse" / "0"
        elif not colmap_path.exists():
            colmap_path = data_dir / "sparse"
        elif not colmap_path.exists():
            colmap_path = data_dir
        if (colmap_path / "points3D.bin").exists():
            points3D = read_points3D_binary(str(colmap_path / "points3D.bin"))
        elif (colmap_path / "points3D.txt").exists():
            points3D = read_points3D_text(str(colmap_path / "points3D.txt"))
        else:
            raise RuntimeError(f"3D points are requested but not present in dataset {data_dir}")
        points3D_xyz = np.array([p.xyz for p in points3D.values()], dtype=np.float32)
        points3D_rgb = np.array([p.rgb for p in points3D.values()], dtype=np.uint8)

        # Transform xyz to match nerfstudio loader
        points3D_xyz = points3D_xyz[..., np.array([1, 0, 2])]
        points3D_xyz[..., 2] *= -1

        if "images_points3D_indices" in (features or {}):
            # TODO: Verify this feature is working well
            points3D_map = {k: i for i, k in enumerate(points3D.keys())}
            if (colmap_path / "points3D.bin").exists():
                images_colmap = read_images_binary(str(colmap_path / "images.bin"))
            elif (colmap_path / "points3D.txt").exists():
                images_colmap = read_images_text(str(colmap_path / "images.txt"))
            else:
                raise RuntimeError(f"3D points are requested but images.{{bin|txt}} not present in dataset {data_dir}")
            images_colmap_map = {}
            for image in images_colmap.values():
                images_colmap_map[image.name] = np.ndarray([points3D_map[x] for x in image.points3D_ids], dtype=np.int32)
            images_points3D_indices = []
            for impath in image_filenames:
                impath = os.path.relpath(impath, str(images_root))
                images_points3D_indices.append(images_colmap_map[impath])

    viewer_transform, viewer_pose = get_default_viewer_transform(all_cameras.poses, None)

    idx_tensor = np.array(indices, dtype=np.int32)

    # Get scene name (if official nerfstudio dataset)
    scene = None
    abspath = path.resolve()
    if len(abspath.parts) > 2 and abspath.parts[-1].lower() in nerfstudio_file_ids and abspath.parts[-2] == "nerfstudio":
        scene = abspath.parts[-1].lower()

    return dataset_index_select(
        new_dataset(
            cameras=all_cameras,
            image_paths=image_filenames,
            image_paths_root=str(images_root),
            sampling_mask_paths=mask_filenames if len(mask_filenames) > 0 else None,
            sampling_mask_paths_root=str(sampling_masks_root) if len(mask_filenames) > 0 else None,
            points3D_xyz=points3D_xyz,
            points3D_rgb=points3D_rgb,
            images_points3D_indices=images_points3D_indices,
            metadata={
                "name": "nerfstudio",
                "scene": scene,
                "expected_scene_scale": get_scene_scale(all_cameras, "object-centric") if split == "train" else None,
                "color_space": "srgb",
                "type": None,
                "evaluation_protocol": "default",
                "viewer_transform": viewer_transform,
                "viewer_initial_pose": viewer_pose,
                "downscale_factor": downscale_factor if downscale_factor > 1 else None,
            },
        ), idx_tensor)


def grab_file_id(zip_url: str) -> str:
    """Get the file id from the google drive zip url."""
    s = zip_url.split("/d/")[1]
    return s.split("/")[0]


nerfstudio_file_ids = {
    "bww_entrance": grab_file_id("https://drive.google.com/file/d/1ylkRHtfB3n3IRLf2wplpfxzPTq7nES9I/view?usp=sharing"),
    "campanelle":    grab_file_id("https://drive.google.com/file/d/13aOfGJRRH05pOOk9ikYGTwqFc2L1xskU/view?usp=sharing"),
    "desolation":   grab_file_id("https://drive.google.com/file/d/14IzOOQm9KBJ3kPbunQbUTHPnXnmZus-f/view?usp=sharing"),
    "library":      grab_file_id("https://drive.google.com/file/d/1Hjbh_-BuaWETQExn2x2qGD74UwrFugHx/view?usp=sharing"),
    # Poster is no longer available for download
    # "poster":       grab_file_id("https://drive.google.com/file/d/1dmjWGXlJnUxwosN6MVooCDQe970PkD-1/view?usp=sharing"),
    "redwoods2":    grab_file_id("https://drive.google.com/file/d/1rg-4NoXT8p6vkmbWxMOY6PSG4j3rfcJ8/view?usp=sharing"),
    "storefront":   grab_file_id("https://drive.google.com/file/d/16b792AguPZWDA_YC4igKCwXJqW0Tb21o/view?usp=sharing"),
    "vegetation":   grab_file_id("https://drive.google.com/file/d/1wBhLQ2odycrtU39y2akVurXEAt9SsVI3/view?usp=sharing"),
    "egypt":        grab_file_id("https://drive.google.com/file/d/1YktD85afw7uitC3nPamusk0vcBdAfjlF/view?usp=sharing"),
    "person":       grab_file_id("https://drive.google.com/file/d/1HsGMwkPu-R7oU7ySMdoo6Eppq8pKhHF3/view?usp=sharing"),
    "kitchen":      grab_file_id("https://drive.google.com/file/d/1IRmNyNZSNFidyj93Tt5DtaEU9h6eJdi1/view?usp=sharing"),
    "plane":        grab_file_id("https://drive.google.com/file/d/1tnv2NC2Iwz4XRYNtziUWvLJjObkZNo2D/view?usp=sharing"),
    "dozer":        grab_file_id("https://drive.google.com/file/d/1jQJPz5PhzTH--LOcCxvfzV_SDLEp1de3/view?usp=sharing"),
    "floating-tree":grab_file_id("https://drive.google.com/file/d/1mVEHcO2ep13WPx92IPDvdQg66vLQwFSy/view?usp=sharing"),
    "aspen":        grab_file_id("https://drive.google.com/file/d/1X1PQcji_QpxGfMxbETKMeK8aOnWCkuSB/view?usp=sharing"),
    "stump":        grab_file_id("https://drive.google.com/file/d/1yZFAAEvtw2hs4MXrrkvhVAzEliLLXPB7/view?usp=sharing"),
    "sculpture":    grab_file_id("https://drive.google.com/file/d/1CUU_k0Et2gysuBn_R5qenDMfYXEhNsd1/view?usp=sharing"),
    "giannini-hall":grab_file_id("https://drive.google.com/file/d/1UkjWXLN4qybq_a-j81FsTKghiXw39O8E/view?usp=sharing"),
}


def download_capture_name(output: Path, file_id_or_zip_url):
    """Download specific captures a given dataset and capture name."""
    target_path = str(output)
    download_path = Path(f"{target_path}.zip")
    tmp_path = target_path + ".tmp"
    shutil.rmtree(tmp_path, ignore_errors=True)
    os.makedirs(tmp_path, exist_ok=True)
    try:
        os.remove(download_path)
    except OSError:
        pass
    # if file_id_or_zip_url.endswith(".zip"):
    url = file_id_or_zip_url  # zip url
    wget(url, download_path)
    # else:
    #     try:
    #         import gdown
    #     except ImportError:
    #         logging.fatal("Please install gdown: pip install gdown")
    #         sys.exit(2)
    #     url = f"https://drive.google.com/uc?id={file_id_or_zip_url}"  # file id
    #     try:
    #         os.remove(download_path)
    #     except OSError:
    #         pass
    #     gdown.download(url, output=str(download_path))
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(tmp_path)
    inner_folders = os.listdir(tmp_path)
    assert len(inner_folders) == 1, "There is more than one folder inside this zip file."
    folder = os.path.join(tmp_path, inner_folders[0])
    shutil.rmtree(target_path, ignore_errors=True)
    shutil.move(folder, target_path)
    shutil.rmtree(tmp_path)
    os.remove(download_path)


def download_nerfstudio_dataset(path: str, output: Union[Path, str]):
    """
    Download data in the Nerfstudio format.
    If you are interested in the Nerfstudio Dataset subset from the SIGGRAPH 2023 paper,
    you can obtain that by using --capture-name nerfstudio-dataset or by visiting Google Drive directly at:
    https://drive.google.com/drive/folders/19TV6kdVGcmg3cGZ1bNIUnBBMD-iQjRbG?usp=drive_link.
    """
    output = Path(output)
    if not path.startswith("nerfstudio/") and path != "nerfstudio":
        raise DatasetNotFoundError("Dataset path must be equal to 'nerfstudio' or must start with 'nerfstudio/'.")
    if path == "nerfstudio":
        for x in nerfstudio_file_ids:
            download_nerfstudio_dataset(f"nerfstudio/{x}", output / x)
        return
    capture_name = path[len("nerfstudio/") :]
    if capture_name not in nerfstudio_file_ids:
        raise DatasetNotFoundError(f"Capture '{capture_name}' not a valid nerfstudio scene.")
    capture_url = f"https://huggingface.co/datasets/jkulhanek/nerfbaselines-data/resolve/main/nerfstudio/{capture_name}.zip?download=true"
    download_capture_name(output, capture_url)
    logging.info(f"Downloaded {path} to {output}")
