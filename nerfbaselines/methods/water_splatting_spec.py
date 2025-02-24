import os
from nerfbaselines import register


_paper_results = {
    "seathru-nerf/iui3": { "psnr": 29.840, "ssim": 0.889, "lpips": 0.203 },
    "seathru-nerf/curasao": { "psnr": 32.203, "ssim": 0.948, "lpips": 0.116 },
    "seathru-nerf/japanese-gradens": { "psnr": 24.741, "ssim": 0.892, "lpips": 0.116 },
    "seathru-nerf/panama": { "psnr": 31.616, "ssim": 0.942, "lpips": 0.080 },
}


register({
    "id": "water-splatting",
    "method_class": ".water_splatting:WaterSplatting",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": r"""
git clone https://github.com/water-splatting/water-splatting.git
cd water-splatting
git checkout 5309df9003b9079769cc9b09eab7019a4025eefe
git submodule init
git submodule update --recursive

conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
pip install nerfstudio==1.1.4 'numpy<2.0.0' opencv-python-headless==4.10.0.84 torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
if [ "$NERFBASELINES_DOCKER_BUILD" != "1" ]; then
conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
fi
LIBRARY_PATH="$CONDA_PREFIX/lib/stubs" pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install and build water-splatting
pip install --no-use-pep517 -e .

# Remove build dependencies
conda uninstall -y gcc_linux-64 gxx_linux-64 make cmake

function nb-post-install () {
if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {} +
fi
}
""",
    },
    "metadata": {
        "name": "Water-Splatting",
        "description": """WaterSplatting combines 3DGS with volume rendering to enable water/fog modeling""",
        "paper_title": "WaterSplatting: Fast Underwater 3D Scene Reconstruction Using Gaussian Splatting",
        "paper_authors": [ "Huapeng Li", "Wenxuan Song" "Tianao Xu", "Alexandre Elsig", "Jonas Kulhanek" ],
        "paper_link": "https://water-splatting.github.io/paper.pdf",
        "paper_results": _paper_results,
        "link": "https://water-splatting.github.io/",
    },
    "presets": {},
    "implementation_status": {
        "seathru-nerf": "reproducing",
    }
})
