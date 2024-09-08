# Adding new datasets

In this guide, we assume you already created a dataset and were able to train a model on it.
If this is not the case, please follow the [Using custom data](using-custom-data.md) and 
[Custom dataset loader](custom-dataset-loader) guides first.
Now, we will show you how to publish your dataset so that others can use it as well.
In order to do that, you need to publish the dataset first. Then, you can create
a `download_dataset` function, which will download the dataset from the source,
and finally, you can add the dataset to the list of available datasets by adding a `spec` file.

## Publishing the dataset
NerfBaselines does not assume any specific structure or format of the dataset, nor does it require
the dataset to be uploaded to any specific location. For example, you can host the dataset on your
own server, on Google Drive, or any other location. In this tutorial, we assume you have 
a COLMAP dataset (see [Using custom data](using-custom-data.md)). Therefore, you have a directory 
with the following structure:
```
images/
    0.jpg
    1.jpg
    ...
sparse/
    points3D.bin
    cameras.bin
    images.bin
```

Now, you can zip the dataset directory and upload it to your own server. We will assume you uploaded
the dataset to `https://example.com/datasets/my_dataset/my_scene.zip`.

## Creating the download function
In order to download the dataset, you need to create a Python function that will download the dataset.
Here we show one possible implementation of such a function. This function will download the dataset
from the specified URL and extract it to the specified output directory. The function will also add
a `nb-info.json` file to the root of the dataset directory. This file contains metadata about the dataset
such as the loader to use, the dataset ID, and the scene name. This file is used by NerfBaselines to
detect the format of the dataset.
Add the following code to a new Python file, e.g., `my_dataset.py`:
```python
import os
import zipfile
import requests
import tempfile
import shutil

def download_my_dataset(path, output):
    url = "https://example.com/datasets/my_dataset/my_scene.zip"
    # Users may call this function with the following arguments:
    #    path="my-dataset" - in case user wants to download the whole dataset
    #    path="my-dataset/my-scene" - in case user wants to download only a specific scene
    # Anything else is invalid
    assert path == "my-dataset/my-scene" or path == "my-dataset", f"Invalid path: {path}"

    if path == "my-dataset":
        # This is the case of full-dataset download
        output = os.path.join(output, "my-dataset")

    with tempfile.TemporaryDirectory() as tmp:
        output_zip = os.path.join(tmp, "my_scene.zip")
        output_dir = os.path.join(tmp, "my_scene")
        os.makedirs(output_dir, exist_ok=True)

        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()
        with open(output_zip, "wb") as f:
            f.write(response.content)

        # Extract it to the temporary directory
        with zipfile.ZipFile(output_zip, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        # Now, we will add a nb-info.json file to the root of the dataset directory
        with open(os.path.join(output_dir, "nb-info.json"), "w") as f:
            f.write('{"loader": "colmap", "id": "my-dataset", "scene": "my-scene"}')

        # Move the files to the output directory
        os.makedirs(os.path.dirname(output), exist_ok=True)
        if os.path.exists(output):
            shutil.rmtree(output)
        shutil.move(output_dir, output)
```

## Registering the dataset
In order to make the dataset available to NerfBaselines, you need to register the dataset with NerfBaselines.
This is done by adding a `spec` file. The `spec` file is a Python file that contains a {func}`nerfbaselines.register` call.
Let's create a new file, e.g., `my_dataset_spec.py`, with the following content:
```python
from nerfbaselines import register

register({
    "id": "my-dataset",
    "download_dataset_function": "my_dataset:download_my_dataset",
    "evaluation_protocol": "default",
    "metadata": {
        # ...
    }
})
```
For testing, you can temporarily notify NerfBaselines about the presence of the `spec` file by setting the environment
variable `NERFBASELINES_REGISTER`:
```bash
export NERFBASELINES_REGISTER="$PWD/my_dataset_spec.py"
```
Now, you can run the following command to download the dataset:
```bash
nerfbaselines download --data external://my-dataset/my-scene
```
Or use the dataset directly in `nerfbaselines` commands.

## Dataset metadata
The `metadata` field in the `spec` file can contain important information about the dataset.
This is required for the dataset to appear in the online benchmark. The metadata should contain the following fields:
- `id` (str): The unique identifier of the dataset.
- `name` (str): The (human readable) name of the dataset.
- `description` (str): A short description of the dataset.
- `paper_title` (optional str): The title of the paper where the dataset was introduced.
- `paper_authors` (optional List[str]): The authors of the paper.
- `paper_link` (optional str): The link to the paper.
- `link` (optional str): The link to the dataset webpage.
- `metrics` (List[str]): The list of metrics that are used to evaluate the dataset, e.g., `["psnr", "ssim", "lpips_vgg"]`.
- `default_metric` (str): The default metric to when sorting the datasets in the public benchmark.
- `scenes` (List[Dict[str, str]]): The list of scenes in the dataset. Each scene should contain the following fields:
  - `id` (str): The unique identifier of the scene.
  - `name` (str): The (human readable) name of the scene.

Here is an example of the metadata for the Mip-NeRF 360 dataset (with some scenes omitted):
```json
{
    "id": "mipnerf360",
    "name": "Mip-NeRF 360",
    "description": "Mip-NeRF 360 is a collection of four indoor and five outdoor object-centric scenes. The camera trajectory is an orbit around the object with fixed elevation and radius. The test set takes each n-th frame of the trajectory as test views.",
    "paper_title": "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields",
    "paper_authors": ["Jonathan T. Barron", "Ben Mildenhall", "Dor Verbin", "Pratul P. Srinivasan", "Peter Hedman"],
    "paper_link": "https://arxiv.org/pdf/2111.12077.pdf",
    "link": "https://jonbarron.info/mipnerf360/",
    "metrics": ["psnr", "ssim", "lpips_vgg"],
    "default_metric": "psnr",
    "scenes": [
        { "id": "garden", "name": "garden" },
        { "id": "bicycle", "name": "bicycle" }
    ]
}
```
