# Installation
Start by installing the `nerfbaselines` pip package on your system. We recommend using a virtual environment (venv or Conda) to avoid conflicts with other packages.
```{nerfbaselines-install}
```
Now you can use the `nerfbaselines` cli to interact with NerfBaselines ([see available commands](cli)).

**NerfBaselines** requires a backend in order to install all dependencies required by various methods. Currently there are the following backends implemented:
- **conda** (default): Does not require docker/apptainer to be installed, but does not offer the same level of isolation and some methods require additional
dependencies to be installed. Also, some methods are not implemented for this backend because they rely on dependencies not found on `conda`.
Please check the [list of methods](methods) first to see if your method is supported.
- **docker**: Offers good isolation, requires `docker` (with [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)) to be installed and the user to have access to it (being in the docker user group).
- **apptainer**: Similar level of isolation as `docker`, but does not require the user to have privileged access.
For all commands which use a backend, the backend can be set as the `--backend <backend>` argument or using the `NERFBASELINES_BACKEND` environment variable.
Next, we will describe the installation process for each backend.

```{warning}
The `conda` backend is not available on Windows or MacOS. This is because the `conda` dependencies for each method can only be installed on Linux (CUDA dependency). If you are on Windows, we strongly recommend installing `NerfBaselines` inside WSL2 and using the `conda` backend.
```

## Pre-requisites (NVIDIA GPU)
If you have an NVIDIA GPU and want to use it (with any backend), you need to install the latest NVIDIA drivers. The installation instruction can vary depending on your operating system.
Please follow the instructions on the [NVIDIA website](https://www.nvidia.com/Download/index.aspx) or your operating system distribution to install the latest drivers for your GPU.

## Conda
The `conda` backend is the default **(and recommended)** backend which is easy to install, but does not allow the same level of isolation and each method has to be built from source
which can be slow. Any Conda distribution can be used. If you haven't installed Conda yet, we recommend using the [miniforge](https://github.com/conda-forge/miniforge) distribution.
Navigate to the [miniforge install page](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install) and follow the instructions for your operating system. On Linux, you can use the following commands:
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

## Docker
In order to use the docker backend, you need to install `docker` and the [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit). To install `docker`, please follow the instructions on the [docker website](https://docs.docker.com/get-docker/). For the NVIDIA container toolkit, please follow the instructions on the [NVIDIA website](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Here, we give instructions for Ubuntu 22.04:
```bash
# Install docker
sudo apt-get update
sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

# Install NVIDIA container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
          sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
# Enable NVIDIA container toolkit in docker
sudo nvidia-ctk runtime configure --runtime=docker
# Restart docker
sudo systemctl restart docker
```

## Apptainer
The `apptainer` backend is similar to `docker`, but does not require the user to have privileged access (to install) and is often present in HPC. To install `apptainer`, please follow the instructions on the [apptainer website](https://apptainer.org/). Here, we give instructions for Ubuntu 22.04:
```bash
curl -s https://raw.githubusercontent.com/apptainer/apptainer/main/tools/install-unprivileged.sh | \
    bash -s - ~/.local

# Add the following line to your .bashrc, .bash_profile, or .zshrc
export PATH=$HOME/.local/bin:$PATH
```
