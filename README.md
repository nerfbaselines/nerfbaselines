<img width=112 height=112 align="left" src="assets/logo.png" />
<h1>
    <div>NerfBaselines</div>

[![PyPI - Version](https://img.shields.io/pypi/v/nerfbaselines)](https://pypi.org/project/nerfbaselines/)
[![GitHub License](https://img.shields.io/badge/license-MIT-%2397ca00)](https://github.com/nerfbaselines/nerfbaselines/blob/main/LICENSE)
[![Downloads](https://static.pepy.tech/badge/nerfbaselines)](https://pepy.tech/project/nerfbaselines)
</h1>

NerfBaselines is a framework for **evaluating and comparing existing NeRF and 3DGS methods**. Currently, most official implementations use different dataset loaders, evaluation protocols, and metrics, which renders benchmarking difficult. Therefore, this project aims to provide a **unified interface** for running and evaluating methods on different datasets in a consistent way using the same metrics. But instead of reimplementing the methods, **we use the official implementations** and wrap them so that they can be run easily using the same interface.

Please visit the <a href="https://nerfbaselines.github.io/">project page to see the results</a> of implemented methods on dataset benchmarks.<br/>

<h3>
<a href="https://nerfbaselines.github.io/">🌐 Web</a>  &nbsp;|&nbsp;
<a href="https://arxiv.org/pdf/2406.17345.pdf">📄 Paper</a> &nbsp;|&nbsp;
<a href="https://nerfbaselines.github.io/docs/">📚 Docs</a>
</h3>

## News
*[30/12/2024]* Added a new viewer implementation.</br>
*[22/09/2024]* Added mesh export for 2DGS, COLMAP, and GOF.</br>
*[17/09/2024]* Moved project to nerfbaselines/nerfbaselines repository.</br>
*[16/09/2024]* Added online demos and demo export for 3DGS-based methods. Check out the [benchmark page](https://nerfbaselines.github.io/).<br>
*[12/09/2024]* Added gsplat, 2D Gaussian Splatting, Scaffold-GS, and COLMAP MVS methods.<br>
*[09/09/2024]* Method and Dataset API refac in v1.2.x to simplify usage.<br>
*[28/08/2024]* Implemented faster communication protocols using shared memory.<br>
*[20/08/2024]* Added [documentation page](https://nerfbaselines.github.io/docs).

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
or download the dataset beforehand by running `nerfbaselines download-dataset external://dataset/scene`.
Examples:
```bash
# Downloads the garden scene to the cache folder.
nerfbaselines download-dataset external://mipnerf360/garden

# Downloads all nerfstudio scenes to the cache folder.
nerfbaselines download-dataset external://nerfstudio

# Downloads kithen scene to folder kitchen
nerfbaselines download-dataset external://mipnerf360/kitchen -o kitchen
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

![viewer-gui](https://github.com/user-attachments/assets/d0316bf1-e650-43a0-ad8f-3b69deeed471)


## Results
In this section, we present results of implemented methods on standard benchmark datasets. For detailed results, visit the project page:
[https://nerfbaselines.github.io](https://nerfbaselines.github.io/)

### Mip-NeRF 360
Mip-NeRF 360 is a collection of four indoor and five outdoor object-centric scenes. The camera trajectory is an orbit around the object with fixed elevation and radius. The test set takes each n-th frame of the trajectory as test views.
Detailed results are available on the project page: [https://nerfbaselines.github.io/mipnerf360](https://nerfbaselines.github.io/mipnerf360)

| Method                                                                               |       PSNR |      SSIM | LPIPS (VGG) |        Time | GPU mem. |
|:-------------------------------------------------------------------------------------|-----------:|----------:|------------:|------------:|---------:|
| [Zip-NeRF](https://nerfbaselines.github.io/m-zipnerf)                                | **28.553** | **0.829** |   **0.218** |  5h 30m 20s |  26.8 GB |
| [Scaffold-GS](https://nerfbaselines.github.io/m-scaffold-gs)                         |   *27.714* |     0.813 |       0.262 |     23m 28s |   8.7 GB |
| [Mip-NeRF 360](https://nerfbaselines.github.io/m-mipnerf360)                         |     27.681 |     0.792 |       0.272 | 30h 14m 36s |  33.6 GB |
| [3DGS-MCMC](https://nerfbaselines.github.io/m-3dgs-mcmc)                             |     27.571 |     0.798 |       0.281 |      35m 8s |  21.6 GB |
| [Mip-Splatting](https://nerfbaselines.github.io/m-mip-splatting)                     |     27.492 |     0.815 |       0.258 |     25m 37s |  11.0 GB |
| [Gaussian Splatting](https://nerfbaselines.github.io/m-gaussian-splatting)           |     27.434 |     0.814 |       0.257 |     23m 25s |  11.1 GB |
| [Gaussian Opacity Fields](https://nerfbaselines.github.io/m-gaussian-opacity-fields) |     27.421 |   *0.826* |     *0.234* |   1h 3m 54s |  28.4 GB |
| [gsplat](https://nerfbaselines.github.io/m-gsplat)                                   |     27.412 |     0.815 |       0.256 |     29m 19s |   8.3 GB |
| [2D Gaussian Splatting](https://nerfbaselines.github.io/m-2d-gaussian-splatting)     |     26.815 |     0.796 |       0.297 |     31m 10s |  13.2 GB |
| [NerfStudio](https://nerfbaselines.github.io/m-nerfacto)                             |     26.388 |     0.731 |       0.343 |   *19m 30s* | *5.9 GB* |
| [Instant NGP](https://nerfbaselines.github.io/m-instant-ngp)                         |     25.507 |     0.684 |       0.398 |  **3m 54s** |   7.8 GB |
| [COLMAP](https://nerfbaselines.github.io/m-colmap)                                   |     16.670 |     0.445 |       0.590 |  2h 52m 55s | **0 MB** |


### Blender
Blender (nerf-synthetic) is a synthetic dataset used to benchmark NeRF methods. It consists of 8 scenes of an object placed on a white background. Cameras are placed on a semi-sphere around the object. Scenes are licensed under various CC licenses.
Detailed results are available on the project page: [https://nerfbaselines.github.io/blender](https://nerfbaselines.github.io/blender)

| Method                                                                               |       PSNR |      SSIM | LPIPS (VGG) |        Time | GPU mem. |
|:-------------------------------------------------------------------------------------|-----------:|----------:|------------:|------------:|---------:|
| [Zip-NeRF](https://nerfbaselines.github.io/m-zipnerf)                                | **33.670** | **0.973** |   **0.036** |  5h 21m 57s |  26.2 GB |
| [Gaussian Opacity Fields](https://nerfbaselines.github.io/m-gaussian-opacity-fields) |   *33.451* |   *0.969* |       0.038 |     18m 26s |   3.1 GB |
| [Mip-Splatting](https://nerfbaselines.github.io/m-mip-splatting)                     |     33.330 |   *0.969* |       0.039 |      6m 49s |   2.7 GB |
| [Gaussian Splatting](https://nerfbaselines.github.io/m-gaussian-splatting)           |     33.308 |   *0.969* |     *0.037* |     *6m 6s* |   3.1 GB |
| [TensoRF](https://nerfbaselines.github.io/m-tensorf)                                 |     33.172 |     0.963 |       0.051 |     10m 47s |  16.4 GB |
| [Scaffold-GS](https://nerfbaselines.github.io/m-scaffold-gs)                         |     33.080 |     0.966 |       0.048 |       7m 4s |   3.7 GB |
| [3DGS-MCMC](https://nerfbaselines.github.io/m-3dgs-mcmc)                             |     33.068 |   *0.969* |       0.040 |      6m 13s |   3.9 GB |
| [K-Planes](https://nerfbaselines.github.io/m-kplanes)                                |     32.265 |     0.961 |       0.062 |     23m 58s |   4.6 GB |
| [Instant NGP](https://nerfbaselines.github.io/m-instant-ngp)                         |     32.198 |     0.959 |       0.055 |  **2m 23s** | *2.6 GB* |
| [Tetra-NeRF](https://nerfbaselines.github.io/m-tetra-nerf)                           |     31.951 |     0.957 |       0.056 |  6h 53m 20s |  29.6 GB |
| [gsplat](https://nerfbaselines.github.io/m-gsplat)                                   |     31.471 |     0.966 |       0.054 |     14m 45s |   2.8 GB |
| [Mip-NeRF 360](https://nerfbaselines.github.io/m-mipnerf360)                         |     30.345 |     0.951 |       0.060 |  3h 29m 39s | 114.8 GB |
| [NerfStudio](https://nerfbaselines.github.io/m-nerfacto)                             |     29.191 |     0.941 |       0.095 |      9m 38s |   3.6 GB |
| [NeRF](https://nerfbaselines.github.io/m-nerf)                                       |     28.723 |     0.936 |       0.092 | 23h 26m 30s |  10.2 GB |
| [COLMAP](https://nerfbaselines.github.io/m-colmap)                                   |     12.123 |     0.766 |       0.214 |  1h 20m 34s | **0 MB** |


### Tanks and Temples
Tanks and Temples is a benchmark for image-based 3D reconstruction. The benchmark sequences were acquired outside the lab, in realistic conditions. Ground-truth data was captured using an industrial laser scanner. The benchmark includes both outdoor scenes and indoor environments. The dataset is split into three subsets: training, intermediate, and advanced.
Detailed results are available on the project page: [https://nerfbaselines.github.io/tanksandtemples](https://nerfbaselines.github.io/tanksandtemples)

| Method                                                                               |       PSNR |      SSIM |     LPIPS |       Time | GPU mem. |
|:-------------------------------------------------------------------------------------|-----------:|----------:|----------:|-----------:|---------:|
| [Zip-NeRF](https://nerfbaselines.github.io/m-zipnerf)                                | **24.628** | **0.840** | **0.131** |  5h 44m 9s |  26.6 GB |
| [Mip-Splatting](https://nerfbaselines.github.io/m-mip-splatting)                     |   *23.930* |   *0.833* |     0.166 |    15m 56s |   7.3 GB |
| [Gaussian Splatting](https://nerfbaselines.github.io/m-gaussian-splatting)           |     23.827 |     0.831 |   *0.165* |  *13m 48s* |   6.9 GB |
| [Gaussian Opacity Fields](https://nerfbaselines.github.io/m-gaussian-opacity-fields) |     22.395 |     0.825 |     0.172 |    41m 21s |  24.1 GB |
| [NerfStudio](https://nerfbaselines.github.io/m-nerfacto)                             |     22.043 |     0.743 |     0.270 |    19m 27s | *3.7 GB* |
| [Instant NGP](https://nerfbaselines.github.io/m-instant-ngp)                         |     21.623 |     0.712 |     0.340 | **4m 27s** |   4.1 GB |
| [2D Gaussian Splatting](https://nerfbaselines.github.io/m-2d-gaussian-splatting)     |     21.535 |     0.768 |     0.281 |    15m 47s |   7.2 GB |
| [COLMAP](https://nerfbaselines.github.io/m-colmap)                                   |     11.919 |     0.436 |     0.606 | 5h 16m 11s | **0 MB** |


## Implementation status
| Method                    | Blender   | LLFF      | Mip-NeRF 360 | Nerfstudio | Photo Tourism | SeaThru-NeRF | Tanks and Temples |
|:------------------------- |:--------- |:--------- |:------------ |:---------- |:------------- |:------------ |:----------------- |
| 2D Gaussian Splatting     | 🥇 gold   | ❔        | 🥇 gold      | ❔         | ❔            | 🥇 gold      | 🥈 silver         |
| 3DGS-MCMC                 | 🥈 silver | ❔        | 🥇 gold      | ❔         | ❔            | 🥇 gold      | 🥇 gold           |
| CamP                      | ❔        | ❔        | ❔           | ❔         | ❔            | ❔           | ❔                |
| COLMAP                    | 🥇 gold   | ❔        | 🥇 gold      | 🥇 gold    | ❔            | ❔           | 🥇 gold           |
| Gaussian Opacity Fields   | 🥇 gold   | ❔        | 🥇 gold      | ❔         | ❔            | ❔           | 🥇 gold           |
| Gaussian Splatting        | 🥇 gold   | ❔        | 🥇 gold      | ❔         | ❔            | 🥇 gold      | 🥇 gold           |
| GS-W                      | ❔        | ❔        | ❔           | ❔         | ❔            | ❔           | ❔                |
| gsplat                    | 🥇 gold   | ❔        | 🥇 gold      | ❔         | 🥇 gold       | ❔           | 🥇 gold           |
| Instant NGP               | 🥇 gold   | ❔        | 🥇 gold      | 🥇 gold    | ❔            | ❔           | 🥇 gold           |
| K-Planes                  | 🥇 gold   | ❔        | ❔           | ❔         | 🥈 silver     | ❔           | ❔                |
| Mip-NeRF 360              | 🥇 gold   | ❔        | 🥇 gold      | ❔         | ❔            | ❔           | 🥇 gold           |
| Mip-Splatting             | 🥇 gold   | ❔        | 🥇 gold      | ❔         | ❔            | 🥇 gold      | 🥇 gold           |
| NeRF                      | 🥇 gold   | ❔        | ❔           | ❔         | ❔            | ❔           | ❔                |
| NeRF On-the-go            | ❔        | ❔        | ❔           | ❔         | ❔            | ❔           | ❔                |
| NeRF-W (reimplementation) | ❔        | ❔        | ❔           | ❔         | 🥇 gold       | ❔           | ❔                |
| NerfStudio                | 🥇 gold   | ❔        | 🥇 gold      | ❔         | ❔            | ❔           | 🥇 gold           |
| Scaffold-GS               | 🥇 gold   | ❔        | 🥇 gold      | ❔         | ❔            | 🥇 gold      | 🥇 gold           |
| SeaThru-NeRF              | ❔        | ❔        | ❔           | ❔         | ❔            | 🥇 gold      | ❔                |
| TensoRF                   | 🥇 gold   | 🥇 gold   | ❌           | ❔         | ❔            | ❔           | ❔                |
| Tetra-NeRF                | 🥈 silver | ❔        | 🥈 silver    | ❔         | ❔            | ❔           | ❔                |
| WildGaussians             | ❔        | ❔        | ❔           | ❔         | 🥇 gold       | ❔           | ❔                |
| Zip-NeRF                  | 🥇 gold   | ❌        | 🥇 gold      | 🥇 gold    | ❔            | ❔           | ❔                |



## Contributing
Contributions are very much welcome. Please open a PR with a dataset/method/feature that you want to contribute. The goal of this project is to slowly expand by implementing more and more methods.

## Citation
If you use this project in your research, please cite the following paper:
```bibtex
@article{kulhanek2024nerfbaselines,
  title={NerfBaselines: Consistent and Reproducible Evaluation of Novel View Synthesis Methods},
  author={Jonas Kulhanek and Torsten Sattler},
  year={2024},
  journal={arXiv},
}
```

## License
This project is licensed under the [MIT license](https://raw.githubusercontent.com/nerfbaselines/nerfbaselines/main/LICENSE)
Each implemented method is licensed under the license provided by the authors of the method.
For the currently implemented methods, the following licenses apply:
- 2D Gaussian Splatting: [custom, research only](https://raw.githubusercontent.com/hbb1/2d-gaussian-splatting/main/LICENSE.md)
- 3DGS-MCMC: [custom, research only](https://raw.githubusercontent.com/ubc-vision/3dgs-mcmc/refs/heads/main/LICENSE.md)
- CamP: [Apache 2.0](https://raw.githubusercontent.com/jonbarron/camp_zipnerf/main/LICENSE)
- COLMAP: [BSD](https://colmap.github.io/license.html)
- Gaussian Opacity Fields: [custom, research only](https://raw.githubusercontent.com/autonomousvision/gaussian-opacity-fields/main/LICENSE.md)
- Gaussian Splatting: [custom, research only](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/LICENSE.md)
- GS-W: unknown
- gsplat: [Apache 2.0](https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/LICENSE)
- Instant NGP: [custom, research only](https://raw.githubusercontent.com/NVlabs/instant-ngp/master/LICENSE.txt)
- K-Planes: [BSD 3](https://raw.githubusercontent.com/sarafridov/K-Planes/main/LICENSE)
- Mip-NeRF 360: [Apache 2.0](https://raw.githubusercontent.com/google-research/multinerf/main/LICENSE)
- Mip-Splatting: [custom, research only](https://raw.githubusercontent.com/autonomousvision/mip-splatting/main/LICENSE.md)
- NeRF-W (reimplementation): [MIT](https://raw.githubusercontent.com/kwea123/nerf_pl/master/LICENSE)
- NeRF: [MIT](https://github.com/bmild/nerf/blob/master/LICENSE)
- NerfStudio: [Apache 2.0](https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/LICENSE)
- Scaffold-GS: [custom, research only](https://raw.githubusercontent.com/city-super/Scaffold-GS/main/LICENSE.md)
- SeaThru-NeRF: [Apache 2.0](https://raw.githubusercontent.com/deborahLevy130/seathru_NeRF/master/LICENSE)
- TensoRF: [MIT](https://github.com/apchenstu/TensoRF/blob/main/LICENSE)
- Tetra-NeRF: [MIT](https://raw.githubusercontent.com/jkulhanek/tetra-nerf/master/LICENSE)
- WildGaussians: [MIT](https://raw.githubusercontent.com/jkulhanek/wild-gaussians/main/LICENSE), [custom, research only](https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/LICENSE.md)
- Zip-NeRF: [Apache 2.0](https://raw.githubusercontent.com/jonbarron/camp_zipnerf/main/LICENSE)


## Acknowledgements
A big thanks to the authors of all implemented methods for the great work they have done.
We would also like to thank the authors of [NerfStudio](https://github.com/nerfstudio-project/nerfstudio).
We also thank Mark Kellogg for the [3DGS web renderer](https://github.com/mkkellogg/GaussianSplats3D).
This work was supported by the Czech Science Foundation (GAČR) EXPRO (grant no. 23-07973X), the Grant Agency of the Czech Technical University in Prague (grant no. SGS24/095/OHK3/2T/13), and by the Ministry of Education, Youth and Sports of the Czech
Republic through the e-INFRA CZ (ID:90254).
