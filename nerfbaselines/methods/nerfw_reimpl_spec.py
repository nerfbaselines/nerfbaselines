import os
from nerfbaselines import register, MethodSpec


NeRFWReimplSpec: MethodSpec = {
    "method_class": ".nerfw_reimpl:NeRFWReimpl",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": r"""# Clone the repo.
git clone https://github.com/kwea123/nerf_pl
cd nerf_pl
# Switch to nerfw branch
git checkout 2dd2759619e435c66de48395b115207092967947

conda install -y 'mkl<2024.1' pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
pip install 'pip<24.1' 'setuptools<70.0.0'
pip install plyfile==0.8.1 \
    'pytorch-lightning==2.1.4' \
    'test-tube==0.7.5' \
    'kornia==0.7.2' \
    'opencv-python<=4.9.0.80' \
    'matplotlib==3.4.3' \
    'einops==0.8.0' \
    'torch-optimizer==0.3.0' \
    tqdm==4.67.1 \
    scikit-image==0.25.0 \
    importlib_metadata==8.5.0 \
    typing_extensions==4.12.2 \
    tensorboard==2.18.0 \
    wandb==0.19.1 \
    'Pillow<=11.1.0' \
    click==8.1.8 \
    mediapy==1.1.2 \
    lpips==0.1.4 \
    'pytest<=8.3.4' \
    scipy==1.13.1

conda install -y conda-build
conda develop "$PWD"
"""
    },
    "metadata": {
        "name": "NeRF-W (reimplementation)",
        "description": "Unofficial reimplementation of NeRF-W. Does not reach the performance reported in the original paper, but is widely used for benchmarking.",
        "licenses": [{"name": "MIT", "url": "https://raw.githubusercontent.com/kwea123/nerf_pl/master/LICENSE"}],
    },
    "id": "nerfw-reimpl",
    "implementation_status": {
        "phototourism": "reproducing",
    }
}

register(NeRFWReimplSpec)
