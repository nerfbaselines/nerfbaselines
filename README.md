# NerfBaselines
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

The backend can be set as the `--backend <backend>` argument or using the `NB_DEFAULT_BACKEND` environment variable.

## Training
To start the training use the `nerfbaselines train --method <method>` command. Use `--help` argument to learn about all implemented methods and supported features.

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
- [x] Mip-NeRF 360
- [x] any COLMAP datasets
- [ ] HDR images support
- [ ] RAW images support
- [ ] handling large datasets
- [ ] Tanks and Temples
- [ ] Blender
- [ ] any NerfStudio datasets

## Contributing
Contributions are very much welcome. Please open a PR with a dataset/method/feature that you want to contribute. The goal of this project is to slowly expand by implementing more and more methods.

## License
This project is licensed under the MIT license.

## Thanks
A big thanks to the authors of all implemented methods for the great work they have done.
