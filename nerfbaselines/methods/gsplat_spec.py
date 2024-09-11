import os
from nerfbaselines import register


_name = os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-")

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
conda install -y cuda-toolkit 'numpy<2.0.0' pytorch==2.1.2 torchvision==0.16.2 -c pytorch -c nvidia/label/cuda-11.8.0
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs"
if [ "$NERFBASELINES_DOCKER_BUILD" != "1" ]; then
conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
fi

# Install dependencies
pip install opencv-python-headless==4.10.0.84 -r examples/requirements.txt

# Install and build gsplat
pip install -e . --use-pep517 --no-build-isolation
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
It is inspired by the 3DGS paper, but it is faster, more memory efficient, and with a growing list of new features.""",
        "paper_title": "Mathematical Supplement for the gsplat Library",
        "paper_authors": [ "Vickie Ye", "Angjoo Kanazawa" ],
        "paper_link": "https://arxiv.org/pdf/2312.02121.pdf",
        "link": "https://docs.gsplat.studio/main/",
        "licenses": [{"name": "Apache 2.0", "url": "https://raw.githubusercontent.com/nerfstudio-project/gsplat/main/LICENSE"}],
    },
    "presets": {
        "blender": { 
            "@apply": [{"dataset": "blender"}], 
            "init_type": "random", 
        },
        "phototourism": { "@apply": [{"dataset": "phototourism"}], 
            "app_opt": True,  # Enable appearance optimization
            "steps_scaler": 3.333334,  # 100k steps
        },
    },
    "implementation_status": {
        "mipnerf360": "working",
        "blender": "working",
        "tanksandtemples": "working",
        "phototourism": "working",
    }
})
