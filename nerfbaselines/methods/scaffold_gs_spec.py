import os
from nerfbaselines import register, MethodSpec


# Blender scenes: Mic Chair Ship Materials Lego Drums Ficus Hotdog
# Blender PSNRs: 37.25 35.28 31.17 30.65 35.69 26.44 35.21 37.73
# Mipnerf360 scenes: bicycle garden stump room counter kitchen bonsai
# Mipnerf360 PSNRs: 24.50 27.17 26.27 31.93 29.34 31.30 32.70
# Mipnerf360 SSIMs: 0.705 0.842 0.784 0.925 0.914 0.928 0.946
# Mipnerf360 LPIPs: 0.306 0.146 0.284 0.202 0.191 0.126 0.185
# TanksAndTemples scenes: truck train
# TanksAndTemples PSNRs: 25.77 22.15
# TanksAndTemples SSIMs: 0.883 0.822
# TanksAndTemples LPIPs: 0.147 0.206
_paper_results = {
    "blender/mic": {"psnr": 37.25},
    "blender/chair": {"psnr": 35.28},
    "blender/ship": {"psnr": 31.17},
    "blender/materials": {"psnr": 30.65},
    "blender/lego": {"psnr": 35.69},
    "blender/drums": {"psnr": 26.44},
    "blender/ficus": {"psnr": 35.21},
    "blender/hotdog": {"psnr": 37.73},
    "mipnerf360/bicycle": {"psnr": 24.50, "ssim": 0.705, "lpips": 0.306},
    "mipnerf360/garden": {"psnr": 27.17, "ssim": 0.842, "lpips": 0.146},
    "mipnerf360/stump": {"psnr": 26.27, "ssim": 0.784, "lpips": 0.284},
    "mipnerf360/room": {"psnr": 31.93, "ssim": 0.925, "lpips": 0.202},
    "mipnerf360/counter": {"psnr": 29.34, "ssim": 0.914, "lpips": 0.191},
    "mipnerf360/kitchen": {"psnr": 31.30, "ssim": 0.928, "lpips": 0.126},
    "mipnerf360/bonsai": {"psnr": 32.70, "ssim": 0.946, "lpips": 0.185},
    "tanksandtemples/truck": {"psnr": 25.77, "ssim": 0.883, "lpips": 0.147},
    "tanksandtemples/train": {"psnr": 22.15, "ssim": 0.822, "lpips": 0.206},
}


ScaffoldGSSpec: MethodSpec = {
    "id": "scaffold-gs",
    "method_class": ".scaffold_gs:ScaffoldGS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/city-super/Scaffold-GS.git scaffold-gs
cd scaffold-gs
git checkout b9e6220d63fb66caf9b8dda05653d32a4a4fe38a
git submodule update --init --recursive

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'

pip install -U pip 'setuptools<70.0.0'
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html
pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
pip install plyfile==0.8.1 \
        mediapy==1.1.2 \
        open3d==0.18.0 \
        lpips==0.1.4 \
        scikit-image==0.21.0 \
        tqdm==4.66.2 \
        trimesh==4.3.2 \
        opencv-python-headless==4.10.0.84 \
        importlib_metadata==8.5.0 \
        typing_extensions==4.12.2 \
        wandb==0.19.1 \
        click==8.1.8 \
        Pillow==11.1.0 \
        matplotlib==3.9.4 \
        tensorboard==2.18.0 \
        scipy==1.13.1 \
        einops==0.8.0 \
        laspy==2.5.4 \
        jaxtyping==0.2.34 \
        'pytest<=8.3.4' \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        --no-build-isolation
conda develop .

if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -rf {} +
# Replace all libs under $CONDA_PREFIX/lib with symlinks to pkgs/cuda-toolkit/targets/x86_64-linux/lib
for lib in "$CONDA_PREFIX"/lib/*.so*; do 
    if [ ! -f "$lib" ] || [ -L "$lib" ]; then continue; fi;
    lib="${lib%.so*}.so";libname=$(basename "$lib");
    tgt="$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/$libname"
    if [ -f "$tgt" ]; then echo "Deleting $lib"; rm -rf "$lib"*; for tgtlib in "$tgt"*; do ln -s "$tgtlib" "$(dirname "$lib")"; done; fi;
done
fi
""",
    },
    "metadata": {
        "name": "Scaffold-GS",
        "description": """Scaffold-GS uses anchor points to distribute local 3D Gaussians, and predicts their attributes on-the-fly based on viewing direction and distance within the view frustum. In NerfBaselines, we fixed bug with cx,cy, added appearance embedding optimization, and added support for masks. Note, that we also implement a demo for the method, but it does not evaluate the MLP and the Gaussians are "baked" for specific viewing direction and appearance embedding (if enabled).""",
        "paper_title": "Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering",
        "paper_authors": ["Tao Lu", "Mulin Yu", "Linning Xu", "Yuanbo Xiangli", "Limin Wang" , "Dahua Lin" "Bo Dai"],
        "paper_link": "https://arxiv.org/pdf/2312.00109.pdf",
        "paper_results": _paper_results,
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
