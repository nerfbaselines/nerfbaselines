import os
from ..registry import MethodSpec, register


NeRFWReimplSpec: MethodSpec = {
    "method": ".nerfw_reimpl:NeRFWReimpl",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": r"""# Clone the repo.
git clone https://github.com/kwea123/nerf_pl
cd nerf_pl
# Switch to nerfw branch
git checkout 2dd2759619e435c66de48395b115207092967947

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
pip install -U pip 'setuptools<70.0.0'
pip install plyfile==0.8.1 \
    'pytorch-lightning<1.0.0' \
    'test-tube<=0.7.5' \
    'kornia<=0.7.2' \
    'opencv-python<=4.9.0.80' \
    'matplotlib<=3.4.3' \
    'einops<=0.8.0' \
    'torch-optimizer<=0.3.0'
pip install lpips==0.1.4 importlib_metadata typing_extensions

conda install -y pip conda-build
conda develop "$PWD"
"""
    },
    "metadata": {
        "name": "NeRF-W (reimplementation)",
        "description": "Unofficial reimplementation of NeRF-W. Does not reach the performance reported in the original paper, but is widely used for benchmarking.",
    }
}

register(NeRFWReimplSpec, name="nerfw-reimpl")
