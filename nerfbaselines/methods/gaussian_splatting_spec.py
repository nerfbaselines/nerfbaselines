import os
from ..registry import MethodSpec, register

# NOTE: In the GS paper, they report two sets of numbers for Mip-NeRF 360, one 
# using the official downscaled images provided by Mip-NeRF 360 (in suppmat) and
# one where they downscale the images themselves (without storing them as JPEGs
# to avoid compression artifacts). This makes roughly 0.5 dB difference in PSNR.
# We use the official numbers for the downscaled JPEGs here to match most other
# papers. If you want to reach the higher PSNR, you should avoid storing the
# downscaled images as JPEGs.

paper_results = {
    # 360 scenes: bicycle flowers garden stump treehill room counter kitchen bonsai
    # 360 PSNRs: 25.246 21.520 27.410 26.550 22.490 30.632 28.700 30.317 31.980
    # 360 SSIMs: 0.771 0.605 0.868 0.775 0.638 0.914 0.905 0.922 0.938
    # 360 LPIPS: 0.205 0.336 0.103 0.210 0.317 0.220 0.204 0.129 0.205
    "mipnerf360/bicycle": {"psnr": 25.246, "ssim": 0.771, "lpips_vgg": 0.205},
    "mipnerf360/flowers": {"psnr": 21.520, "ssim": 0.605, "lpips_vgg": 0.336},
    "mipnerf360/garden": {"psnr": 27.410, "ssim": 0.868, "lpips_vgg": 0.103},
    "mipnerf360/stump": {"psnr": 26.550, "ssim": 0.775, "lpips_vgg": 0.210},
    "mipnerf360/treehill": {"psnr": 22.490, "ssim": 0.638, "lpips_vgg": 0.317},
    "mipnerf360/room": {"psnr": 30.632, "ssim": 0.914, "lpips_vgg": 0.220},
    "mipnerf360/counter": {"psnr": 28.700, "ssim": 0.905, "lpips_vgg": 0.204},
    "mipnerf360/kitchen": {"psnr": 30.317, "ssim": 0.922, "lpips_vgg": 0.129},
    "mipnerf360/bonsai": {"psnr": 31.980, "ssim": 0.938, "lpips_vgg": 0.205},

    # blender scenes: Mic Chair Ship Materials Lego Drums Ficus Hotdog
    # blender PSNRs: 35.36 35.83 30.80 30.00 35.78 26.15 34.87 37.72
    "blender/mic": {"psnr": 35.36},
    "blender/chair": {"psnr": 35.83},
    "blender/ship": {"psnr": 30.80},
    "blender/materials": {"psnr": 30.00},
    "blender/lego": {"psnr": 35.78},
    "blender/drums": {"psnr": 26.15},
    "blender/ficus": {"psnr": 34.87},
    "blender/hotdog": {"psnr": 37.72},
}


GaussianSplattingSpec: MethodSpec = {
    "method": ".gaussian_splatting:GaussianSplatting",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
git checkout 2eee0e26d2d5fd00ec462df47752223952f6bf4e

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
pip install -U pip 'setuptools<70.0.0'
pip install plyfile==0.8.1 tqdm submodules/diff-gaussian-rasterization submodules/simple-knn

conda install -c conda-forge -y nodejs==20.9.0
conda develop .
pip install lpips==0.1.4 importlib_metadata typing_extensions

function nb-post-install () {
if [ "$NB_DOCKER_BUILD" = "1" ]; then
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
        "name": "Gaussian Splatting",
        "description": """Official Gaussian Splatting implementation extended to support distorted camera models. It is fast to train (1 hous) and render (200 FPS).""",
        "paper_title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
        "paper_authors": ["Bernhard Kerbl", "Georgios Kopanas", "Thomas Leimkühler", "George Drettakis"],
        "paper_link": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf",
        "link": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/",
    },
    "dataset_overrides": {
        "blender": { "white_background": True, },
    },
}

register(GaussianSplattingSpec, name="gaussian-splatting", metadata={
    "paper_results": paper_results,
})
register(
    GaussianSplattingSpec,
    name="gaussian-splatting:large", 
    kwargs={
        "config_overrides": {
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
    metadata={
        "name": "Gaussian Splatting (large)",
        "description": """A version of Gaussian Splatting designed for larger scenes."""
    },
    dataset_overrides={
        "blender": {
            "white_background": True,
        },
    },
)
