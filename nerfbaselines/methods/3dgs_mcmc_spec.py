import os
from nerfbaselines import register

_note = """Authors evaluated on larger images which were downscaled to the target size (avoiding JPEG compression artifacts) instead of using the official provided downscaled images. As mentioned in the 3DGS paper, this increases results slightly ~0.5 dB PSNR."""
_blender_note = """Exact hyperparameters for Blender dataset are not provided in the released source code. The default parameters were used in NerfBaselines likely leading to worse results."""

_paper_results = {
    "blender/mic": {"psnr": 37.29, "ssim": 0.99, "lpips": 0.01, "note": _blender_note},
    "blender/ship": {"psnr": 30.82, "ssim": 0.91, "lpips": 0.12, "note": _blender_note},
    "blender/lego": {"psnr": 36.01, "ssim": 0.98, "lpips": 0.02, "note": _blender_note},
    "blender/chair": {"psnr": 36.51, "ssim": 0.99, "lpips": 0.02, "note": _blender_note},
    "blender/materials": {"psnr": 30.59, "ssim": 0.96, "lpips": 0.04, "note": _blender_note},
    "blender/hotdog": {"psnr": 37.82, "ssim": 0.99, "lpips": 0.02, "note": _blender_note},
    "blender/drums": {"psnr": 26.29, "ssim": 0.95, "lpips": 0.04, "note": _blender_note},
    "blender/ficus": {"psnr": 35.07, "ssim": 0.99, "lpips": 0.01, "note": _blender_note},
    "mipnerf360/counter": {"psnr": 29.51, "ssim": 0.92, "lpips": 0.22, "note": _note},
    "mipnerf360/stump": {"psnr": 27.80, "ssim": 0.82, "lpips": 0.19, "note": _note},
    "mipnerf360/kitchen": {"psnr": 32.27, "ssim": 0.94, "lpips": 0.14, "note": _note},
    "mipnerf360/bicycle": {"psnr": 26.15, "ssim": 0.81, "lpips": 0.18, "note": _note},
    "mipnerf360/bonsai": {"psnr": 32.88, "ssim": 0.95, "lpips": 0.22, "note": _note},
    "mipnerf360/room": {"psnr": 32.48, "ssim": 0.94, "lpips": 0.25, "note": _note},
    "mipnerf360/garden": {"psnr": 28.16, "ssim": 0.89, "lpips": 0.10, "note": _note},
    "tanksandtemples/train": {"psnr": 22.47, "ssim": 0.83, "lpips": 0.24},
    "tanksandtemples/truck": {"psnr": 26.11, "ssim": 0.89, "lpips": 0.14},
}


register({
    "id": "3dgs-mcmc",
    "method_class": ".3dgs_mcmc:GaussianSplattingMCMC",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/ubc-vision/3dgs-mcmc.git
cd 3dgs-mcmc
git checkout a22a24bd7e64b089d983bfcf52c906df7d46f25d
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
        submodules/diff-gaussian-rasterization \
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
        "name": "3DGS-MCMC",
        "description": "3DGS-MCMC reinterprets 3D Gaussian Splatting as MCMC sampling, introducing noise-based updates and removing heuristic cloning strategies, leading to improved rendering quality, efficient Gaussian use, and robustness to initialization. In NerfBaselines, we fixed bug with cx,cy, added appearance embedding optimization, and added support for masks and web demos.",
        "paper_title": "3D Gaussian Splatting as Markov Chain Monte Carlo",
        "paper_authors": "Shakiba Kheradmand, Daniel Rebain, Gopal Sharma, Weiwei Sun, Yang-Che Tseng, Hossam Isack, Abhishek Kar, Andrea Tagliasacchi, Kwang Moo Yi".split(", "),
        "paper_results": _paper_results,
        "paper_link": "https://ubc-vision.github.io/3dgs-mcmc/paper.pdf",
        "link": "https://ubc-vision.github.io/3dgs-mcmc/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/ubc-vision/3dgs-mcmc/refs/heads/main/LICENSE.md"}],
    },
    "presets": {
        "mipnerf360/kitchen": { "@apply": [{"dataset": "mipnerf360", "scene": "kitchen"}], "cap_max": 1800000 },
        "mipnerf360/bicycle": { "@apply": [{"dataset": "mipnerf360", "scene": "bicycle"}], "cap_max": 5900000 },
        "mipnerf360/bonsai": { "@apply": [{"dataset": "mipnerf360", "scene": "bonsai"}],   "cap_max": 1300000 },
        "mipnerf360/counter": { "@apply": [{"dataset": "mipnerf360", "scene": "counter"}], "cap_max": 1200000 },
        "mipnerf360/garden": { "@apply": [{"dataset": "mipnerf360", "scene": "garden"}],   "cap_max": 5200000 },
        "mipnerf360/kitchen": { "@apply": [{"dataset": "mipnerf360", "scene": "kitchen"}], "cap_max": 1800000 },
        "mipnerf360/room": { "@apply": [{"dataset": "mipnerf360", "scene": "room"}],       "cap_max": 1500000 },
        "mipnerf360/stump": { "@apply": [{"dataset": "mipnerf360", "scene": "stump"}],     "cap_max": 4750000 },
        "mipnerf360/flowers": { "@apply": [{"dataset": "mipnerf360", "scene": "flowers"}], "cap_max": 3700000 },
        "mipnerf360/treehill": { "@apply": [{"dataset": "mipnerf360", "scene": "treehill"}], "cap_max": 3800000 },

        "blender": { "@apply": [{"dataset": "blender"}], "white_background": True, },
        "blender/lego": { "@apply": [{"dataset": "blender", "scene": "lego"}], "cap_max": 325000 },
        "blender/ship": { "@apply": [{"dataset": "blender", "scene": "ship"}], "cap_max": 330000 },
        "blender/chair": { "@apply": [{"dataset": "blender", "scene": "chair"}], "cap_max": 270000 },
        "blender/materials": { "@apply": [{"dataset": "blender", "scene": "materials"}], "cap_max": 290000 },
        "blender/hotdog": { "@apply": [{"dataset": "blender", "scene": "hotdog"}], "cap_max": 150000 },
        "blender/drums": { "@apply": [{"dataset": "blender", "scene": "drums"}], "cap_max": 350000 },
        "blender/ficus": { "@apply": [{"dataset": "blender", "scene": "ficus"}], "cap_max": 300000 },
        "blender/mic": { "@apply": [{"dataset": "blender", "scene": "mic"}], "cap_max": 320000 },
    },
    "implementation_status": {
        "blender": "working-not-reproducing",
        "mipnerf360": "reproducing",
        "tanksandtemples": "working",
        "seathru-nerf": "working",
    }
})
