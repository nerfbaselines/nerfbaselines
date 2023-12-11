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

The next step is to choose the backend which will be used to install different methods. At the moment there are the following backends implemented:
- **docker**: Offers good isolation, requires `docker` to be installed and the user to have access to it (being in the docker user group).
- **apptainer**: Similar level of isolation as `docker`, but does not require the user to have privileged access.
- **conda** (not recommended): Does not require docker/apptainer to be installed, but does not offer the same level of isolation and some methods require additional
dependencies to be installed. Also, some methods are not implemented for this backend because they rely on dependencies not found on `conda`.
- **python** (not recommended): Will run everything directly in the current environment. Everything needs to be installed in the environment for this backend to work.

The backend can be set as the `--backend <backend>` argument or using the `NB_BACKEND` environment variable.

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
- [x] undistorting images for methods that do not support complex camera models (Gaussian Splatting)
- [ ] Tanks and Temples
- [ ] LLFF dataset
- [ ] HDR images support
- [ ] RAW images support
- [ ] handling large datasets

## Contributing
Contributions are very much welcome. Please open a PR with a dataset/method/feature that you want to contribute. The goal of this project is to slowly expand by implementing more and more methods.

## License
This project is licensed under the MIT license.

## Thanks
A big thanks to the authors of all implemented methods for the great work they have done.
