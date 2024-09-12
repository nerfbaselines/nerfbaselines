import os
from nerfbaselines import register, MethodSpec


ScaffoldGSSpec: MethodSpec = {
    "id": "scaffold-gs",
    "method_class": ".scaffold_gs:ScaffoldGS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone --recursive https://github.com/city-super/Scaffold-GS.git scaffold-gs
cd scaffold-gs
git checkout b9e6220d63fb66caf9b8dda05653d32a4a4fe38a

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge

pip install -U pip 'setuptools<70.0.0'
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install plyfile==0.8.1 tqdm submodules/diff-gaussian-rasterization submodules/simple-knn

conda develop .
pip install lpips==0.1.4 einops==0.8.0 laspy==2.5.4 jaxtyping==0.2.34 importlib_metadata typing_extensions
if ! python -c 'import cv2'; then pip install opencv-python-headless; fi

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
        "name": "Scaffold-GS",
        "description": """Scaffold-GS uses anchor points to distribute local 3D Gaussians, and predicts their attributes on-the-fly based on viewing direction and distance within the view frustum. In NerfBaselines, we fixed bug with cx,cy, added appearance embedding optimization, and added support for sampling masks.""",
        "paper_title": "Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering",
        "paper_authors": ["Tao Lu", "Mulin Yu", "Linning Xu", "Yuanbo Xiangli", "Limin Wang" , "Dahua Lin" "Bo Dai"],
        "paper_link": "https://arxiv.org/pdf/2312.00109.pdf",
        "paper_results": {},
        "link": "https://city-super.github.io/scaffold-gs/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/city-super/Scaffold-GS/main/LICENSE.md"}],
    },
    "presets": {
        "blender": { 
            "@apply": [{"dataset": "blender"}], 
            "white_background": True, 
            "voxel_size": 0.001,
            "update_init_factor": 4,
            "appearance_dim": 0,
            "ratio": 1,
        },
        "mipnerf360": {
            "@apply": [{"dataset": "mipnerf360"}], 
            "voxel_size": 0.001,
            "update_init_factor": 16,
            "appearance_dim": 0,
            "ratio": 1,
        },
        "tanksandtemples": {
            "@apply": [{"dataset": "tanksandtemples"}], 
            "voxel_size": 0.01,
            "update_init_factor": 16,
            "appearance_dim": 0,
            "ratio": 1,
        },
        "phototourism": {
            "@apply": [{"dataset": "phototourism"}], 
            "voxel_size": 0,
            "update_init_factor": 16,
            "appearance_dim": 32,
            "ratio": 1,

            # Make model larger
            "iterations": 100_000,  # 100k iterations
            "appearance_lr_max_steps": 100_000,  # 100k steps
            "position_lr_max_steps": 100_000,  # 100k steps
            "offset_lr_max_steps": 100_000,  # 100k steps
            "mlp_opacity_lr_max_steps": 100_000,  # 100k steps
            "mlp_cov_lr_max_steps": 100_000,  # 100k steps
            "mlp_color_lr_max_steps": 100_000,  # 100k steps
            "mlp_featurebank_lr_max_steps": 100_000,  # 100k steps
            "start_stat": 1500,
            "update_from": 4500,
            "update_interval": 300,
            "update_until": 50_000,
        },
    },
    "implementation_status": {
        "blender": "reproducing",
        "mipnerf360": "reproducing",
        "tanksandtemples": "working",
        "seathru-nerf": "working",
    }
}

register(ScaffoldGSSpec)
