# Quickstart
This guide will help you get started with NerfBaseline, but it assumes you have already installed the package. If you haven't, please refer to the [installation guide](installation.md).
In this guide, we show the full process of training a [Gaussian Splatting](methods.md#gaussian-splatting) model on the [Mip-NeRF 360](datasets.md#mipnerf360) dataset, garden scene.

## Public benchmark results
Before training a model, you can inspect the public benchmark to see the results of the implemented methods on standard datasets.
In our case, the results of the Gaussian Splatting method can be seen [here](https://nerfbaselines.github.io/m-gaussian-splatting).
If we want to skip the training and only inspect the public checkpoint, we can download the checkpoint from the website and extract it in the current working directory
(the next section can be skipped).

## Training a model
To train a model, you need to specify the method, the dataset, and optionally the backend.
The following command will start the training and evaluate on the test set at the end of the training:
```bash
nerfbaselines train --method gaussian-splatting --data external://mipnerf360/garden
```
The training script will try to detect the backend automatically, but if you want to choose a specific backend, you can specify it using the `--backend` argument:
```bash
nerfbaselines train --method gaussian-splatting --data external://mipnerf360/garden --backend docker
```

The training script will save all checkpoints, predictions, etc. into the current directory, but you can specify the output directory using the `--output` argument.
By default, the training will log all metrics into a `tensorboard` logger. You can specify a different logger using the `--logger` argument (e.g., `--logger wandb` for [Weights and Biases](https://wandb.ai/site)) - see [supported loggers](api/nerfbaselines.logging).

```{note}
For some official datasets (e.g., Mip-NeRF 360, NerfStudio, Blender, or Tanks and Temples), the datasets will be downloaded automatically if you use `external://dataset/scene` as the data argument.
See the ([list of datasets](datasets.md)) which can be downloaded automatically. 
If you want to your own data, point `--data` to the folder containing the dataset. 
The format of the datasets will be detected automatically.
```

After the training finished, you should see the following output in the output directory:
```
checkpoint-30000/
predictions-30000.tar.gz
output.zip
results-30000.json
tensorboard/
```

The `checkpoint-30000` folder contains the trained model, `predictions-30000.tar.gz` contains the predictions on the test set, and `results-30000.json` contains the evaluation results.
The `output.zip` file contains everything in a single archive and is used when uploading the results to the benchmark.

## Rendering images
To render images from a trained model, you can use the `render` command. Let's extend the previous example (assuming the checkpoint is in the current directory) and evaluate the model on the **train set**:
```bash
nerfbaselines render --checkpoint checkpoint-30000 --data external://mipnerf360/garden --split train --output predictions-train-30000.tar.gz
```
The `predictions-train-30000.tar.gz` file will contain the rendered images and the corresponding camera poses.

```{tip}
You can choose to output the results in a compressed archive (`.tar.gz`) or a folder. The output format is detected automatically based on the file extension.
```

```{note}
In all commands working with checkpoint, the method is detected automatically from the checkpoint folder, no need to specify it.
```


## Launching the viewer
To further inspect your trained model, you can launch the interactive viewer by running:
```bash
nerfbaselines viewer --checkpoint checkpoint-30000 --data external://mipnerf360/garden
```

The viewer will start in the background and will allow you to connect from your web browser (the URL will be printed in the terminal).
In the viewer, you can inspect the 3D scene, construct a camera trajectory, and render a video of the trajectory. You can also
save the camera trajectory to render it later using the `render-trajectory` command. In this walkthrough, we assume you created a camera trajectory
and saved it to a file `trajectory.json`.
For information on how to use the viewer, please refer to the {std:doc}`Viewer page <viewer>`.

```{tip}
By specifying the `--data` argument, the viewer will use the camera poses to perform gravity alignment and rescaling for a better viewing experience. 
It also enables visualizing the input camera frustums and input 3D point cloud. For some datasets (e.g., Photo Tourism datasets), it is required
if you want to visualize the differenct appearance embeddings of train images.
```

## Rendering a camera trajectory
To render a camera trajectory (e.g., created using the interactive viewer), use the following command:
```bash
nerfbaselines render-trajectory --checkpoint checkpoint-30000 --trajectory trajectory.json --output trajectory.mp4
```

The format is automatically detected based on the `--output` file extension and you can also save individual frames as images by specifying a folder (or a `.tar.gz`/`.zip`) instead of a file.
