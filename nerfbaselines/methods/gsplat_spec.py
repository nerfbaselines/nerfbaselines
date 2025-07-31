import os
from nerfbaselines import register


_name = os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-")

# Scenes: Bicycle Bonsai Counter Garden Kitchen Room Stump
# PSNRs: 25.29 32.21 29.01 27.39 31.37 31.23 26.51
# SSIMs: 0.77 0.94 0.91 0.87 0.93 0.92 0.77
_paper_results = {
    "mipnerf360/bicycle": {"psnr": 25.29, "ssim": 0.77},
    "mipnerf360/bonsai": {"psnr": 32.21, "ssim": 0.94},
    "mipnerf360/counter": {"psnr": 29.01, "ssim": 0.91},
    "mipnerf360/garden": {"psnr": 27.39, "ssim": 0.87},
    "mipnerf360/kitchen": {"psnr": 31.37, "ssim": 0.93},
    "mipnerf360/room": {"psnr": 31.23, "ssim": 0.92},
    "mipnerf360/stump": {"psnr": 26.51, "ssim": 0.77},
}

register({
    "id": _name,
    "method_class": ".gsplat:GSplat",
    "conda": {
        "environment_name": _name,
        "python_version": "3.10",
        "install_script": r"""
git clone https://github.com/nerfstudio-project/gsplat.git
cd gsplat
git checkout cc800d7750a891ab8684eac4ddbcf90b79d16295
git submodule init
git submodule update --recursive
conda develop "$PWD/examples"

# Install build dependencies
conda install -y cuda-toolkit -c nvidia/label/cuda-11.8.0
pip install torch==2.3.0 torchvision==0.18.0 'numpy<2.0.0' --index-url https://download.pytorch.org/whl/cu118
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs"
if [[ "$(gcc -v 2>&1)" != *"gcc version 11"* ]]; then
  conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
  ln -s "$CC" "$CONDA_PREFIX/bin/gcc"
  ln -s "$CXX" "$CONDA_PREFIX/bin/g++"
  export CPATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPATH"
fi

# Install dependencies
pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
pip install opencv-python-headless==4.10.0.84 \
    -r examples/requirements.txt \
    plyfile==0.8.1 \
    mediapy==1.1.2 \
    scikit-image==0.21.0 \
    tqdm==4.66.2 \
    importlib_metadata==8.5.0 \
    typing_extensions==4.12.2 \
    wandb==0.19.1 \
    click==8.1.8 \
    Pillow==11.1.0 \
    matplotlib==3.9.4 \
    tensorboard==2.18.0 \
    pytest==8.3.4 \
    scipy==1.13.1

# Install and build gsplat
MAX_JOBS=8 pip install -e . --use-pep517 --no-build-isolation
python -c 'from gsplat import csrc'  # Test import

# Clear build dependencies
if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {} +
fi
""",
    },
    "metadata": {
        "name": "gsplat",
        "description": """gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. \
It is inspired by the 3DGS paper, but it is faster, more memory efficient, and with a growing list of new features. In NerfBaselines, \
the method was modified to enable appearance optimization, to support masks, and to support setting background color (which is required for the Blender dataset).""",
        "paper_title": "gsplat: An Open-Source Library for Gaussian Splatting",
        "paper_authors": [ "Vickie Ye", "Ruilong Li", "Justin Kerr", "Matias Turkulainen", "Brent Yi", "Zhuoyang Pan", "Otto Seiskari", "Jianbo Ye", "Jeffrey Hu", "Matthew Tancik", "Angjoo Kanazawa" ],
        "paper_link": "https://arxiv.org/pdf/2409.06765.pdf",
        "paper_results": _paper_results,
        "link": "https://docs.gsplat.studio/main/",
        "licenses": [{"name": "Apache 2.0", "url": "https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/LICENSE"}],
    },
    "presets": {
        "blender": { 
            "@apply": [{"dataset": "blender"}], 
            "init_type": "random", 
            "background_color": (1.0, 1.0, 1.0),
            "init_extent": 0.5,
        },
        "phototourism": { "@apply": [{"dataset": "phototourism"}], 
            "app_opt": True,  # Enable appearance optimization
            "steps_scaler": 3.333334,  # 100k steps
        },
    },
    "implementation_status": {
        "mipnerf360": "reproducing",
        "blender": "working",
        "tanksandtemples": "working",
        "phototourism": "working",
    }
})
