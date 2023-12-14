# NerfBaselines
[![PyPI - Version](https://img.shields.io/pypi/v/nerfbaselines)](https://pypi.org/project/nerfbaselines/)
[![GitHub License](https://img.shields.io/badge/license-MIT-%2397ca00)](https://github.com/jkulhanek/nerfbaselines/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/nerfbaselines)](https://pepy.tech/project/nerfbaselines)


The goal of this project is to provide a simple uniform way to benchmark different NeRF methods on standard datasets to allow for an easy comparison.
The implemented methods use the original code published by the authors and, therefore, the resulting performance matches the original implementation.
DISCLAIMER: This project is at a very early stage of its development. Stay tuned!

## Getting started
Start by installing the `nerfbaselines` pip package on your host system.
```bash
pip install nerfbaselines
```
Now you can use the `nerfbaselines` cli to interact with NerfBaselines.

WARNING: the default installation only installs the core nerfbaselines package which does not depend on either PyTorch or JAX.
However, the LPIPS metric requires PyTorch to be installed and will be disabled otherwise. Similarly, if you install JAX and
have a GPU available, the dataloading and evaluation will be faster as some parts of the pipeline will be moved to GPU.
Therefore, we recommend installing the `extras` package by following the **Advanced installation** section.

The next step is to choose the backend which will be used to install different methods. At the moment there are the following backends implemented:
- **docker**: Offers good isolation, requires `docker` to be installed and the user to have access to it (being in the docker user group).
- **apptainer**: Similar level of isolation as `docker`, but does not require the user to have privileged access.
- **conda** (not recommended): Does not require docker/apptainer to be installed, but does not offer the same level of isolation and some methods require additional
dependencies to be installed. Also, some methods are not implemented for this backend because they rely on dependencies not found on `conda`.
- **python** (not recommended): Will run everything directly in the current environment. Everything needs to be installed in the environment for this backend to work.

The backend can be set as the `--backend <backend>` argument or using the `NB_BACKEND` environment variable.

## Advanced installation
The LPIPS metric requires PyTorch to be installed and will be disabled otherwise. Similarly, if you install JAX and
have a GPU available, the dataloading and evaluation will be faster as some parts of the pipeline will be moved to GPU.
In this section we describe how to install the packages required for LPIPS and accelerated dataloading.
We recommend this as the default installation (unless there is a reason for not installing PyTorch or JAX).
Select one of the following configurations:
- CPU-only install
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install jax[cpu]
pip install 'nerfbaselines[extras]'
```
- CUDA 11.8 install
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install jax[cuda11_pip]
pip install 'nerfbaselines[extras]'
```
- CUDA 12.1 install
```bash
pip install torch torchvision torchaudio
pip install jax[cuda12_pip]
pip install 'nerfbaselines[extras]'
```

## Downloading data
For some datasets, e.g. Mip-NeRF 360 or NerfStudio, the datasets can be downloaded automatically. You can specify the argument `--data external://dataset/scene` during training
or download the dataset beforehand by running `nerfbaselines download-dataset dataset/scene`.
Examples:
```bash
# Downloads the garden scene to the cache folder.
nerfbaselines download-dataset mipnerf360/garden

# Downloads all nerfstudio scenes to the cache folder.
nerfbaselines download-dataset nerfstudio

# Downloads kithen scene to folder kitchen
nerfbaselines download-dataset mipnerf360/kitchen -o kitchen
```

## Training
To start the training use the `nerfbaselines train --method <method> --data <data>` command. Use `--help` argument to learn about all implemented methods and supported features.

## Rendering
The `nerfbaselines render --checkpoint <checkpoint>` command can be used to render images from a trained checkpoint. Again, use `--help` to learn about the arguments.

## Interactive viewer
Given a trained checkpoint, the interactive viewer can be launched as follows:
```bash
nerfbaselines viewer --checkpoint <checkpoin> --data <dataset>
```
Even though the argument `--data <dataset>` is optional, it is recommended, as for some methods some viewers use the camera poses
to performa gravity alignement for better viewing experience.
Again, you can use the `--backend <backend>` flag or `NS_BACKEND=<backend>` environment variable to change the backend.


## Implementation status
Methods:
- [x] Nerfacto
- [x] Instant-NGP
- [x] Gaussian Splatting
- [x] Tetra-NeRF
- [x] Mip-NeRF 360
- [ ] NeRF
- [ ] Mip-NeRF
- [ ] Zip-NeRF

Datasets/features:
- [x] Mip-NeRF 360 dataset
- [x] Blender dataset
- [x] any COLMAP dataset
- [x] any NerfStudio dataset
- [x] automatic dataset download
- [x] interactive viewer
- [x] undistorting images for methods that do not support complex camera models (Gaussian Splatting)
- [x] logging to tensorboard, wandb
- [ ] Tanks and Temples
- [ ] LLFF dataset
- [ ] HDR images support
- [ ] RAW images support
- [ ] handling large datasets
- [ ] loading/creating camera trajectories in the interactive viewer

## Contributing
Contributions are very much welcome. Please open a PR with a dataset/method/feature that you want to contribute. The goal of this project is to slowly expand by implementing more and more methods.

## License
This project is licensed under the MIT license.

## Thanks
A big thanks to the authors of all implemented methods for the great work they have done.
