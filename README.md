<img width=112 height=112 align="left" src="assets/logo.png" />
<h1>
    <div>NerfBaselines</div>

[![PyPI - Version](https://img.shields.io/pypi/v/nerfbaselines)](https://pypi.org/project/nerfbaselines/)
[![GitHub License](https://img.shields.io/badge/license-MIT-%2397ca00)](https://github.com/jkulhanek/nerfbaselines/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/nerfbaselines)](https://pepy.tech/project/nerfbaselines)
</h1>

NerfBaselines is a framework for **evaluating and comparing existing NeRF and 3DGS methods**. Currently, most official implementations use different dataset loaders, evaluation protocols, and metrics, which renders benchmarking difficult. Therefore, this project aims to provide a **unified interface** for running and evaluating methods on different datasets in a consistent way using the same metrics. But instead of reimplementing the methods, **we use the official implementations** and wrap them so that they can be run easily using the same interface.

Please visit the <a href="https://jkulhanek.com/nerfbaselines">project page to see the results</a> of implemented methods on dataset benchmarks.<br/>

### [Project Page + Results](https://jkulhanek.com/nerfbaselines)

## Getting started
Start by installing the `nerfbaselines` pip package on your host system.
```bash
pip install nerfbaselines
```
Now you can use the `nerfbaselines` cli to interact with NerfBaselines.

The next step is to choose the backend which will be used to install different methods. At the moment there are the following backends implemented:
- **docker**: Offers good isolation, requires `docker` (with [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)) to be installed and the user to have access to it (being in the docker user group).
- **apptainer**: Similar level of isolation as `docker`, but does not require the user to have privileged access.
- **conda** (default): Does not require docker/apptainer to be installed, but does not offer the same level of isolation and some methods require additional
dependencies to be installed. Also, some methods are not implemented for this backend because they rely on dependencies not found on `conda`.
- **python**: Will run everything directly in the current environment. Everything needs to be installed in the environment for this backend to work.

The backend can be set as the `--backend <backend>` argument or using the `NERFBASELINES_BACKEND` environment variable.

## Downloading data
For some datasets, e.g. Mip-NeRF 360, NerfStudio, Blender, or Tanks and Temples, the datasets can be downloaded automatically. 
You can specify the argument `--data external://dataset/scene` during training
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
To start the training, use the `nerfbaselines train --method <method> --data <data>` command. Use `--help` argument to learn about all implemented methods and supported features.

## Rendering
The `nerfbaselines render --checkpoint <checkpoint>` command can be used to render images from a trained checkpoint. Again, use `--help` to learn about the arguments.

In order to render a camera trajectory (e.g., created using the interactive viewer), use the following command command:
```bash
nerfbaselines render-trajectory --checkpoint <checkpoint> --trajectory <trajectory> --output <output.mp4>
```

## Interactive viewer
Given a trained checkpoint, the interactive viewer can be launched as follows:
```bash
nerfbaselines viewer --checkpoint <checkpoin> --data <dataset>
```
Even though the argument `--data <dataset>` is optional, it is recommended, as the camera poses
are used to perform gravity alignment and rescaling for a better viewing experience.
It also enables visualizing the input camera frustums.

## Results
In this section, we present results of implemented methods on standard benchmark datasets. For detailed results, visit the project page:
[https://jkulhanek.com/nerfbaselines](https://jkulhanek.com/nerfbaselines)

### Mip-NeRF 360
Mip-NeRF 360 is a collection of four indoor and five outdoor object-centric scenes. The camera trajectory is an orbit around the object with fixed elevation and radius. The test set takes each n-th frame of the trajectory as test views.
Detailed results are available on the project page: [https://jkulhanek.com/nerfbaselines/mipnerf360](https://jkulhanek.com/nerfbaselines/mipnerf360)

| Method                                                                         |       PSNR |      SSIM |     LPIPS |        Time |   GPU mem. |
|:-------------------------------------------------------------------------------|-----------:|----------:|----------:|------------:|-----------:|
| [Zip-NeRF](https://jkulhanek.com/nerfbaselines/m-zipnerf)                      | **28.516** | **0.828** | **0.138** |  5h 30m 49s |    26.2 GB |
| [Mip-NeRF 360](https://jkulhanek.com/nerfbaselines/m-mipnerf360)               |   *27.670* |     0.792 |     0.196 |  7h 29m 42s |   127.0 GB |
| [Mip-Splatting](https://jkulhanek.com/nerfbaselines/m-mip-splatting)           |     27.498 |   *0.815* |     0.183 |      25m 1s |    11.0 GB |
| [Gaussian Splatting](https://jkulhanek.com/nerfbaselines/m-gaussian-splatting) |     27.439 |     0.814 |   *0.180* |     22m 45s |    11.1 GB |
| [NerfStudio](https://jkulhanek.com/nerfbaselines/m-nerfacto)                   |     26.348 |     0.730 |     0.257 |   *19m 50s* | **3.8 GB** |
| [Tetra-NeRF](https://jkulhanek.com/nerfbaselines/m-tetra-nerf)                 |     25.468 |     0.670 |     0.352 | 17h 32m 35s |    13.4 GB |
| [Instant NGP](https://jkulhanek.com/nerfbaselines/m-instant-ngp)               |     24.899 |     0.673 |     0.355 |  **4m 16s** |   *5.6 GB* |


### Blender
Blender (nerf-synthetic) is a synthetic dataset used to benchmark NeRF methods. It consists of 8 scenes of an object placed on a white background. Cameras are placed on a semi-sphere around the object.
Detailed results are available on the project page: [https://jkulhanek.com/nerfbaselines/blender](https://jkulhanek.com/nerfbaselines/blender)

| Method                                                                         |       PSNR |      SSIM |     LPIPS |       Time |   GPU mem. |
|:-------------------------------------------------------------------------------|-----------:|----------:|----------:|-----------:|-----------:|
| [Zip-NeRF](https://jkulhanek.com/nerfbaselines/m-zipnerf)                      | **33.670** | **0.973** | **0.020** | 5h 21m 57s |    26.2 GB |
| [Mip-Splatting](https://jkulhanek.com/nerfbaselines/m-mip-splatting)           |   *33.330* |     0.969 |     0.025 |     6m 49s |   *2.7 GB* |
| [Gaussian Splatting](https://jkulhanek.com/nerfbaselines/m-gaussian-splatting) |     33.308 |   *0.969* |   *0.023* |    *6m 6s* |     3.1 GB |
| [Instant NGP](https://jkulhanek.com/nerfbaselines/m-instant-ngp)               |     32.191 |     0.959 |     0.031 | **2m 23s** | **2.6 GB** |
| [Tetra-NeRF](https://jkulhanek.com/nerfbaselines/m-tetra-nerf)                 |     31.951 |     0.957 |     0.031 | 6h 53m 20s |    29.6 GB |
| [TensoRF](https://jkulhanek.com/nerfbaselines/m-tensorf)                       |     31.222 |     0.951 |     0.034 |      8m 9s |    16.9 GB |
| [Mip-NeRF 360](https://jkulhanek.com/nerfbaselines/m-mipnerf360)               |     30.345 |     0.951 |     0.038 | 3h 29m 39s |   114.8 GB |
| [NerfStudio](https://jkulhanek.com/nerfbaselines/m-nerfacto)                   |     29.191 |     0.941 |     0.049 |     9m 38s |     3.6 GB |


### Nerfstudio
Nerfstudio Dataset includes 10 in-the-wild captures obtained using either a mobile phone or a mirror-less camera with a fisheye lens. We processed the data using either COLMAP or the Polycam app to obtain camera poses and intrinsic parameters.
Detailed results are available on the project page: [https://jkulhanek.com/nerfbaselines/nerfstudio](https://jkulhanek.com/nerfbaselines/nerfstudio)

| Method                                                                         |       PSNR |      SSIM |     LPIPS |       Time |   GPU mem. |
|:-------------------------------------------------------------------------------|-----------:|----------:|----------:|-----------:|-----------:|
| [Zip-NeRF](https://jkulhanek.com/nerfbaselines/m-zipnerf)                      | **24.815** | **0.798** | **0.178** | 5h 21m 41s |    26.2 GB |
| [Instant NGP](https://jkulhanek.com/nerfbaselines/m-instant-ngp)               |   *20.653* |     0.601 |     0.452 | **4m 33s** | **4.2 GB** |
| [NerfStudio](https://jkulhanek.com/nerfbaselines/m-nerfacto)                   |     20.064 |   *0.617* |   *0.353* |  *13m 30s* |   *4.8 GB* |
| [Gaussian Splatting](https://jkulhanek.com/nerfbaselines/m-gaussian-splatting) |          - |         - |         - |          - |          - |
| [Mip-Splatting](https://jkulhanek.com/nerfbaselines/m-mip-splatting)           |          - |         - |         - |          - |          - |


## Implementation status
Methods:
- [x] NerfStudio (Nerfacto)
- [x] Instant-NGP
- [x] Gaussian Splatting
- [x] Mip-Splatting
- [x] Gaussian Opacity Fields
- [x] Tetra-NeRF
- [x] Mip-NeRF 360
- [x] Zip-NeRF
- [x] CamP
- [x] TensoRF
- [x] K-Planes
- [ ] Nerf-W (open source reimplementation)
- [ ] NeRF on-the-go
- [ ] TRIPS
- [ ] Mip-NeRF
- [ ] NeRF

Datasets/features:
- [x] Mip-NeRF 360 dataset
- [x] Blender dataset
- [x] any COLMAP dataset
- [x] any NerfStudio dataset
- [x] LLFF dataset
- [x] Tanks and Temples dataset
- [x] Photo Tourism dataset and evaluation protocol
- [x] Bundler dataset format
- [x] automatic dataset download
- [x] interactive viewer and trajectory editor
- [x] undistorting images for methods that do not support complex camera models (Gaussian Splatting)
- [x] logging to tensorboard, wandb
- [ ] HDR images support
- [ ] RAW images support

### Reproducing results
| Method                  | Mip-NeRF 360  | Blender    | NerfStudio | Tanks and Temples | LLFF    |
|:-----------------------:|:-------------:|:----------:|:----------:|:-----------------:|:-------:|
| NerfStudio              | 🥇 gold       | 🥇 gold    | ❔         | 🥇 gold           | ❌      |
| Instant-NGP             | 🥇 gold       | 🥇 gold    | 🥇 gold    | 🥇 gold           | ❌      |
| Gaussian Splatting      | 🥇 gold       | 🥇 gold    | ❌         | 🥇 gold           | ❌      |
| Mip-Splatting           | 🥇 gold       | 🥇 gold    | ❌         | 🥇 gold           | ❌      |
| Gaussian Opacity Fields | 🥇 gold       | 🥇 gold    | ❌         | 🥇 gold           | ❌      |
| Tetra-NeRF              | 🥈 silver     | 🥈 silver  | ❔         | ❔                | ❌      |
| Mip-NeRF 360            | 🥇 gold       | 🥇 gold    | ❔         | ❔                | ❌      |
| Zip-NeRF                | 🥇 gold       | 🥇 gold    | 🥇 gold    | 🥇 gold           | ❌      |
| CamP                    | ❔            | ❔         | ❔         | ❔                | ❌      |
| TensoRF                 | ❌            | 🥇 gold    | ❔         | ❔                | 🥇 gold |

## Contributing
Contributions are very much welcome. Please open a PR with a dataset/method/feature that you want to contribute. The goal of this project is to slowly expand by implementing more and more methods.

## License
This project is licensed under the [MIT license](https://raw.githubusercontent.com/jkulhanek/nerfbaselines/main/LICENSE)
Each implemented method is licensed under the license provided by the authors of the method.
For the currently implemented methods, the following licenses apply:
- NerfStudio: [Apache 2.0](https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/LICENSE)
- Instant-NGP: [custom, research purposes only](https://raw.githubusercontent.com/NVlabs/instant-ngp/master/LICENSE.txt) 
- Gaussian-Splatting: [custom, research purposes only](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/LICENSE.md)
- Mip-Splatting: [custom, research purposes only](https://raw.githubusercontent.com/autonomousvision/mip-splatting/main/LICENSE.md)
- Gaussian Opacity Fields: [custom, research purposes only](https://raw.githubusercontent.com/autonomousvision/gaussian-opacity-fields/main/LICENSE.md)
- Tetra-NeRF: [MIT](https://raw.githubusercontent.com/jkulhanek/tetra-nerf/master/LICENSE), [Apache 2.0](https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/LICENSE)
- Mip-NeRF 360: [Apache 2.0](https://raw.githubusercontent.com/google-research/multinerf/main/LICENSE)
- Zip-NeRF: [Apache 2.0](https://raw.githubusercontent.com/jonbarron/camp_zipnerf/main/LICENSE)
- CamP: [Apache 2.0](https://raw.githubusercontent.com/jonbarron/camp_zipnerf/main/LICENSE)

## Acknowledgements
A big thanks to the authors of all implemented methods for the great work they have done.
We would also like to thank the authors of [NerfStudio](https://github.com/nerfstudio-project/nerfstudio), 
especially Brent Yi, for [viser](https://github.com/nerfstudio-project/viser) - a great framework powering the viewer.
This work was supported by the Czech Science Foundation (GAČR) EXPRO (grant no. 23-07973X), the Grant Agency of the Czech Technical University in Prague (grant no. SGS24/095/OHK3/2T/13), and by the Ministry of Education, Youth and Sports of the Czech
Republic through the e-INFRA CZ (ID:90254).
