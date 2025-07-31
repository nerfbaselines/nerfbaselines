# Custom dataset loader

There are various data loaders implemented in NerfBaselines, however, if you have
a custom dataset and none of the loaders fit your needs, you can write your own loader.
In this guide, we will walk you through the process of writing a custom loader for a dataset
The dataset loader is a Python function that takes a path to the dataset, the split, optional keyword arguments, 
and returns a {class}`Dataset <nerfbaselines.Dataset>` dictionary.

## Creating new Dataset instance
The output of your dataset loader function should be the {class}`UnloadedDataset <nerfbaselines.UnloadedDataset>` dictionary.
There is a helper function called {func}`new_dataset <nerfbaselines.new_dataset>` to build the dataset dictionary, which you can use to simplify the process and which will ensure the correct structure.
In your loader function should need to generate at least the following:
- `image_paths`: a list of paths to the images. The paths are absolute.
- `cameras`: {class}`Cameras <nerfbaselines.Cameras>` object containing the camera poses and other camera parameters.

However, many other fields can be added, such as:
- `metadata`: A dictionary containing metadata about the dataset like dataset ID, scene name, and others (see [Using custom data](using-custom-data.md) for more information).
- `image_paths_root`: By default {func}`new_dataset <nerfbaselines.new_dataset>` will try to detect the common prefix for `image_paths`, however, you can specify it explicitly.
- `mask_paths`: A list of paths to the masks. The paths are absolute.
- `mask_paths_root`: Same as `image_paths_root`, but for `mask_paths`.
- `points3D_xyz`: A numpy array of shape `(N, 3)` containing the 3D points (e.g., COLMAP points3D.ply).
- `points3D_rgb`: A numpy array of shape `(N, 3)`, dtype `uint8`, containing the RGB values of the 3D points.
- `points3D_indices`: A list of numpy arrays of indices into `points3D_xyz` and `points3D_rgb` for each image.

In order to construct the {class}`Cameras <nerfbaselines.Cameras>`, you can use the {func}`new_cameras <nerfbaselines.new_cameras>` function. The function needs at least the following arguments:
- `poses`: A numpy array of shape `(N, 3, 4)` containing the camera poses (camera-to-world matrices with OpenCV coordinate system convention).
- `intrinsics`: A numpy array of shape `(N, 4)` containing the camera intrinsics in the form `[focal_x, focal_y, center_x, center_y]`.
- `image_sizes`: A numpy array of shape `(N, 2,)` containing the image sizes in the form `[width, height]`.
- `camera_models`: A numpy array of shape `(N,)` and dtype `np.uint8` containing the camera types (e.g., `perspective`, `opencv`, etc.). The camera types are converted to integers using the {func}`camera_model_to_int <nerfbaselines.camera_model_to_int>` function.

Optionally, you can also specify the following parameters:
- `distortion_parameters`: A numpy array of shape `(N, K)` containing the distortion parameters (k1, k2, p1, p2, k3, k4, k5, k6, ...). See OpenCV documentation for more information.
- `nears_fars`: A numpy array of shape `(N, 2)` containing the near and far planes for each camera.
- `metadata`: A numpy array of shape `(N, M)` containing metadata for each camera.


## Writing dataset loader function 
Let's assume you have a dataset with the following structure:
```
images/
    0.jpg
    1.jpg
    ...
cameras.csv  # Camera parameters as: (image_name, w, h, focal, ...3x4 pose matrix)
```

We will write a loader function that reads the camera parameters from the `cameras.csv` file and the images from the `images/` directory.
First, lets write a function to load the cameras from the `cameras.csv` file.
Let's create a `my_dataloader.py` file with the following content:
```python
import csv
import numpy as np
from nerfbaselines import new_cameras, new_dataset
from nerfbaselines.datasets import dataset_index_select

def _load_cameras(path):
    poses = []
    intrinsics = []
    image_sizes = []
    image_names = []

    with open(path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            image_names.append(row[0])
            w, h, focal, *pose = map(float, row[1:])
            poses.append(np.array(pose, dtype=np.float32).reshape(3, 4))
            intrinsics.append(np.array([focal, focal, w / 2, h / 2], dtype=np.float32))
            image_sizes.append(np.array([w, h], dtype=np.int32))
    return image_names, new_cameras(
        poses=np.stack(poses),
        intrinsics=np.stack(intrinsics),
        image_sizes=np.stack(image_sizes),
        camera_models=np.zeros(len(poses), dtype=np.uint8))
```

Now, we can write the loader function:
```python
def load_my_dataset(path, split, **kwargs):
    image_names, cameras = _load_cameras(os.path.join(path, "cameras.csv"))
    image_paths = [os.path.join(path, "images", name) for name in image_names]
    dataset = new_dataset(
        image_paths=image_paths,
        cameras=cameras,
        metadata={"id": "my-dataset", 
                  "type": "object-centric",
                  "scene": "my-scene"})

    # Now, we need to do train/test split
    test_indices = np.linspace(0, len(image_paths), 10, endpoint=False, dtype=np.int32)
    indices = test_indices if split == "test" else np.setdiff1d(np.arange(len(image_paths)), test_indices)
    return dataset_index_select(dataset, indices)
```

## Registering the loader
If you want the loader to be automatically used by NerfBaselines to load unknown datasets, you need to register the loader with NerfBaselines using the {func}`register <nerfbaselines.register>` function. This function should be called from a `spec` file (e.g., `my_dataloader_spec.py`).
```python
from nerfbaselines import register

register({
    "id": "my-dataset",
    "load_dataset_function": "my_dataloader:load_my_dataset"
})
```
The function will be automatically called for all unknown datasets (without an explicit `loader` field in their `nb-info.json` file).
If you believe the data loader would be useful for the community, consider contributing it to NerfBaselines by creating a pull request.
In that case, place the loader and the `spec` under `nerfbaselines/datasets` directory (see the example of `nerfbaselines/datasets/colmap.py`).
