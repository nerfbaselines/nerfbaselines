import os
from nerfbaselines import register


register({
    "id": "2d-gaussian-splatting",
    "method_class": ".2d_gaussian_splatting:GaussianSplatting2D",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
cd 2d-gaussian-splatting
git checkout 19eb5f1e091a582e911b4282fe2832bac4c89f0f
git submodule update --recursive

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
pip install -U pip 'setuptools<70.0.0'
pip install plyfile==0.8.1 \
        mediapy==1.1.2 \
        open3d==0.18.0 \
        lpips==0.1.4 \
        scikit-image==0.21.0 \
        tqdm==4.66.2 \
        trimesh==4.3.2 \
        submodules/diff-surfel-rasterization \
        submodules/simple-knn \
        opencv-python-headless \
        importlib_metadata typing_extensions

conda develop .

function nb-post-install () {
if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {} +
# Replace all libs under $CONDA_PREFIX/lib with symlinks to pkgs/cuda-toolkit/targets/x86_64-linux/lib
for lib in "$CONDA_PREFIX"/lib/*.so*; do 
    if [ ! -f "$lib" ] || [ -L "$lib" ]; then continue; fi;
    lib="${lib%.so*}.so";libname=$(basename "$lib");
    tgt="$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/$libname"
    if [ -f "$tgt" ]; then echo "Deleting $lib"; rm "$lib"*; for tgtlib in "$tgt"*; do ln -s "$tgtlib" "$(dirname "$lib")"; done; fi;
done
fi
}
""",
    },
    "metadata": {
        "name": "2D Gaussian Splatting",
        "description": "2DGS adopts 2D oriented disks as surface elements and allows high-quality rendering with Gaussian Splatting. In NerfBaselines, we fixed bug with cx,cy, added appearance embedding optimization, and added support for sampling masks.",
        "paper_title": "2D Gaussian Splatting for Geometrically Accurate Radiance Fields",
        "paper_authors": ["Binbin Huang", "Zehao Yu", "Anpei Chen", "Andreas Geiger", "Shenghua Gao"],
        "paper_link": "https://arxiv.org/pdf/2403.17888.pdf",
        "link": "https://surfsplatting.github.io/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/hbb1/2d-gaussian-splatting/main/LICENSE.md"}],
    },
    "presets": {
        "blender": { "@apply": [{"dataset": "blender"}], "white_background": True, },
        "tanksandtemples": {
            "@apply": [{"dataset": "tanksandtemples"}],
            "depth_ratio": 1.0,
            "lambda_dist": 100,
        },
        "tanksandtemples-large": {
            "@apply": [
                {"dataset": "tanksandtemples", "scene": "meetingroom" },
                {"dataset": "tanksandtemples", "scene": "courthouse" },
                {"dataset": "tanksandtemples", "scene": "temple" },
                {"dataset": "tanksandtemples", "scene": "auditorium" },
                {"dataset": "tanksandtemples", "scene": "ballroom" },
                {"dataset": "tanksandtemples", "scene": "museum" },
                {"dataset": "tanksandtemples", "scene": "palace" },
            ],
            "lambda_dist": 10,
        },
        "large": {
            "@apply": [{"dataset": "phototourism"}],
            "@description": "A version of the method designed for large scenes.",
            "iterations": 100_000,
            "densify_from_iter": 1_500,
            "densify_until_iter": 50_000,
            "densification_interval": 300,
            "opacity_reset_interval": 10_000,
            "position_lr_max_steps": 100_000,
            "position_lr_final": 0.000_000_16,
            "position_lr_init": 0.000_016,
            "scaling_lr": 0.000_5,
        },
    },
    "implementation_status": {
        "blender": "working",
        "mipnerf360": "reproducing",
        "tanksandtemples": "working",
        "seathru-nerf": "working",
    }
})
