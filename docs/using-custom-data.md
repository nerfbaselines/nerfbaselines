# Using custom data

## Introduction
NerfBaselines implements various dataset loaders to help you use your data directly.
If your data is supported by one of the loaders, you can simply specify `--data` argument
and point it to the directory containing the dataset.
In this tutorial, we will show you how to use your custom data with NerfBaselines.
We will assume you use COLMAP to obtain the camera poses for your images.
If you decide to publish your dataset, follow the instructions in the [Adding new datasets](adding-new-datasets.md) guide.

## Obtaining camera poses
If you already have camera poses, you can skip this step. Otherwise, we will show you
how to obtain camera poses for your images using COLMAP.
However, you can also follow the great tutorial on [COLMAP's documentation](https://colmap.github.io/tutorial.html),
with instructions on how to run COLMAP from cli [here](https://colmap.github.io/cli.html).

Let's assume you created a directory for your dataset and placed the images in the `images/` directory.
We also assume you have COLMAP installed on your system.
```{note}
If you do not have COLMAP installed, you can follow the instruction [here](https://colmap.github.io/).
Or simply create a new Conda environment and install COLMAP using `conda install -c conda-forge colmap`.
```

Now, from the root of the dataset directory, run the following commands:
```bash
colmap feature_extractor \
  --database_path database.db \
  --image_path images

colmap exhaustive_matcher \
  --database_path database.db

mkdir sparse
colmap mapper \
  --database_path database.db \
  --image_path images \
  --output_path sparse
```

This will create the necessary files in the `sparse/` directory.
After this, the dataset is ready to be used with NerfBaselines.
However, if you want to customize the dataset, you can follow the next steps.

## Customizing the dataset
We assume you ran COLMAP on your images and have the following files:
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

Now, depending on your needs, you may want to add your custom `train`/`test` split.
In order to do that, you need to create `train_list.txt` and `test_list.txt` files,
where each line contains the image path relative to the `images/` directory.
In this example, the contents of the files could be:
```
0.jpg
1.jpg
...
```

```{note}
There should be no overlap between the images in `train_list.txt` and `test_list.txt`.
```

Since we will be using the existing COLMAP loader, we need to inform NerfBaselines
to use the loader. While NerfBaselines would be able to detect the correct format automatically,
it is always safer to be explicit, otherwise another loader might be used.
Therefore, we will add a `nb-info.json` file to the root of the dataset directory with the following content:
```json
{
  "loader": "colmap",
  "id": "my-dataset",
  "scene": "my-custom-scene"
}
```

Notice, how we also specified `id` and `scene`. These are the **metadata** of the dataset which
can be used by various methods to set the default parameters for the dataset.
There are other metadata fields that can be specified, such as:
 - `type`: the type of the dataset, e.g., `object-centric`, `forward-facing`, etc.
 - `downscale_factor`: the factor by which the images were downscaled (from the original resolution). Note, that this is only used for bookkeeping and does not affect the dataset loading.
 - `downscale_loaded_factor`: the factor by which the images will be downscaled when loaded. Note, that when this is set, the `downscale_factor` should be same or larger (in case of prior downscaling).

 ```{warning}
The `downscale_loaded_factor` can be used to downscale the images automatically when loading the dataset.
This can be useful for testing, however, when releasing the official dataset, we recommend to downscale the images
beforehand and release the smaller images. This will ensure consistent results and reduce the size of the dataset.
When resizing the images beforehand, please set `downscale_factor` to let the users know by how much the images were downscaled.
```
