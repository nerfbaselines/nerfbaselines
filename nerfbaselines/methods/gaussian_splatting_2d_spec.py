import os
from nerfbaselines import register

_note = """Authors evaluated on larger images which were downscaled to the target size (avoiding JPEG compression artifacts) instead of using the official provided downscaled images. As mentioned in the 3DGS paper, this increases results slightly ~0.5 dB PSNR."""
_tnt_note = """2DGS used different data pre-processing and train/test split for Tanks and Temples. It sets specific hyperparameters for each scene which may not be suitable with the public Tanks and Temples released by NerfBaselines. The results are not directly comparable and a hyperparameter tuning is needed to improve the results."""
_paper_results = {
    "blender/mic": { "psnr": 35.09 },
    "blender/chair": { "psnr": 35.05 },
    "blender/ship": { "psnr": 30.60 },
    "blender/materials": { "psnr": 29.74 },
    "blender/lego": { "psnr": 35.10 },
    "blender/drums": { "psnr": 26.05 },
    "blender/ficus": { "psnr": 35.57 },
    "blender/hotdog": { "psnr": 37.36 },
    "tanksandtemples/barn": { "psnr": 28.79, "note": _tnt_note },
    "tanksandtemples/caterpillar": { "psnr": 24.23, "note": _tnt_note },
    "tanksandtemples/courthouse": { "psnr": 23.51, "note": _tnt_note },
    "tanksandtemples/ignatius": { "psnr": 23.82, "note": _tnt_note },
    "tanksandtemples/meetingroom": { "psnr": 26.15, "note": _tnt_note },
    "tanksandtemples/truck": { "psnr": 26.85, "note": _tnt_note },
    "mipnerf360/bicycle": { "psnr": 24.87, "ssim": 0.752, "lpips": 0.218, "note": _note },
    "mipnerf360/flowers": { "psnr": 21.15, "ssim": 0.588, "lpips": 0.346, "note": _note },
    "mipnerf360/garden": { "psnr": 26.95, "ssim": 0.852, "lpips": 0.115, "note": _note },
    "mipnerf360/stump": { "psnr": 26.47, "ssim": 0.765, "lpips": 0.222, "note": _note },
    "mipnerf360/treehill": { "psnr": 22.27, "ssim": 0.627, "lpips": 0.329, "note": _note },
    "mipnerf360/room": { "psnr": 31.06, "ssim": 0.912, "lpips": 0.223, "note": _note },
    "mipnerf360/counter": { "psnr": 28.55, "ssim": 0.900, "lpips": 0.208, "note": _note },
    "mipnerf360/kitchen": { "psnr": 30.50, "ssim": 0.919, "lpips": 0.133, "note": _note },
    "mipnerf360/bonsai": { "psnr": 31.52, "ssim": 0.933, "lpips": 0.214, "note": _note},
}


register({
    "id": "2d-gaussian-splatting",
    "method_class": ".gaussian_splatting_2d:GaussianSplatting2D",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/hbb1/2d-gaussian-splatting.git
cd 2d-gaussian-splatting
git checkout 19eb5f1e091a582e911b4282fe2832bac4c89f0f
git submodule update --init --recursive

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
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
        pytest==8.3.4 \
        submodules/diff-surfel-rasterization \
        submodules/simple-knn \
        --no-build-isolation

conda develop .

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
""",
    },
    "metadata": {
        "name": "2D Gaussian Splatting",
        "description": "2DGS adopts 2D oriented disks as surface elements and allows high-quality rendering with Gaussian Splatting. In NerfBaselines, we fixed bug with cx,cy, added appearance embedding optimization, and added support for masks.",
        "paper_title": "2D Gaussian Splatting for Geometrically Accurate Radiance Fields",
        "paper_authors": ["Binbin Huang", "Zehao Yu", "Anpei Chen", "Andreas Geiger", "Shenghua Gao"],
        "paper_link": "https://arxiv.org/pdf/2403.17888.pdf",
        "paper_results": _paper_results,
        "link": "https://surfsplatting.github.io/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/hbb1/2d-gaussian-splatting/main/LICENSE.md"}],
    },
    "presets": {
        "blender": { "@apply": [{"dataset": "blender"}], "white_background": True, },
        "tanksandtemples": {
            "@apply": [{"dataset": "tanksandtemples"}],
            "depth_ratio": 1.0,
            "lambda_dist": 100,
            # Mesh generation related:
            "num_cluster": 1,
            "voxel_size": 0.004,
            "sdf_trunc": 0.016,
            "depth_trunc": 3.0,
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
            # Mesh generation related:
            "num_cluster": 1,
            "voxel_size": 0.006,
            "sdf_trunc": 0.024,
            "depth_trunc": 4.5,
        },
        "dtu": {
            "@apply": [{"dataset": "dtu"}],
            "depth_ratio": 1.0,
            "lambda_dist": 1000,
            # Mesh generation related:
            "num_cluster": 1,
            "voxel_size": 0.004,
            "sdf_trunc": 0.016,
            "depth_trunc": 3.0,
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
        "tanksandtemples": "working-not-reproducing",
        "seathru-nerf": "working",
    }
})
