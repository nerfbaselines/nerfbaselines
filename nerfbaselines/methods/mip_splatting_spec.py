import os
from nerfbaselines import register, MethodSpec


_MIPNERF360_NOTE = """Authors evaluated on larger images which were downscaled to the target size (avoiding JPEG compression artifacts) instead of using the official provided downscaled images. As mentioned in the 3DGS paper, this increases results slightly ~0.5 dB PSNR."""
_paper_results = {
    # Mip-NeRF 360
    # 360 scenes: bicycle flowers garden stump treehill room counter kitchen bonsai
    # 360 PSNRs: 25.72 21.93 27.76 26.94 22.98 31.74 29.16 31.55 32.31
    # 360 SSIMs: 0.780 0.623 0.875 0.786 0.655 0.928 0.916 0.933 0.948
    # 360 LPIPS: 0.206 0.331 0.103 0.209 0.320 0.192 0.179 0.113 0.173
    "mipnerf360/bicycle": {"psnr": 25.72, "ssim": 0.780, "lpips_vgg": 0.206, "note": _MIPNERF360_NOTE},
    "mipnerf360/flowers": {"psnr": 21.93, "ssim": 0.623, "lpips_vgg": 0.331, "note": _MIPNERF360_NOTE},
    "mipnerf360/garden": {"psnr": 27.76, "ssim": 0.875, "lpips_vgg": 0.103, "note": _MIPNERF360_NOTE},
    "mipnerf360/stump": {"psnr": 26.94, "ssim": 0.786, "lpips_vgg": 0.209, "note": _MIPNERF360_NOTE},
    "mipnerf360/treehill": {"psnr": 22.98, "ssim": 0.655, "lpips_vgg": 0.320, "note": _MIPNERF360_NOTE},
    "mipnerf360/room": {"psnr": 31.74, "ssim": 0.928, "lpips_vgg": 0.192, "note": _MIPNERF360_NOTE},
    "mipnerf360/counter": {"psnr": 29.16, "ssim": 0.916, "lpips_vgg": 0.179, "note": _MIPNERF360_NOTE},
    "mipnerf360/kitchen": {"psnr": 31.55, "ssim": 0.933, "lpips_vgg": 0.113, "note": _MIPNERF360_NOTE},
    "mipnerf360/bonsai": {"psnr": 32.31, "ssim": 0.948, "lpips_vgg": 0.173, "note": _MIPNERF360_NOTE},
}


MipSplattingSpec: MethodSpec = {
    "method_class": ".mip_splatting:MipSplatting",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """# Install mip-splatting
git clone https://github.com/autonomousvision/mip-splatting.git
cd mip-splatting
git checkout 746a17c9a906be256ed85b8fe18632f5d53e832d
# Remove unsupported (and unnecessary) open3d dependency
sed -i '/import open3d as o3d/d' train.py

conda install -y conda-build && conda develop .
conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
conda install -c conda-forge -y nodejs==20.9.0

pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
pip install \
        plyfile==0.8.1 \
        mediapy==1.1.2 \
        open3d==0.18.0 \
        ninja==1.11.1.3 \
        GPUtil==1.4.0 \
        einops==0.8.0 \
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
        'pytest<=8.3.4' \
        scipy==1.13.1

pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation

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
        "name": "Mip-Splatting",
        "description": """A modification of Gaussian Splatting designed to better handle aliasing artifacts.""",
        "paper_title": "Mip-Splatting: Alias-free 3D Gaussian Splatting",
        "paper_authors": ["Zehao Yu", "Anpei Chen", "Binbin Huang", "Torsten Sattler", "Andreas Geiger"],
        "paper_link": "https://arxiv.org/pdf/2311.16493.pdf",
        "paper_results": _paper_results,
        "link": "https://niujinshuchong.github.io/mip-splatting/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/autonomousvision/mip-splatting/main/LICENSE.md"}],
    },
    "presets": {
        "blender": { "@apply": [{"dataset": "blender"}], "white_background": True, },
        "large": {
            "@apply": [{"dataset": "phototourism"}],
            "@description": "A version of the method designed for large scenes.",
            "iterations": 300_000,
            "densify_from_iter": 5_000,
            "densify_until_iter": 150_000,
            "densification_interval": 1_000,
            "opacity_reset_interval": 30_000,
            "position_lr_max_steps": 300_000,
            "position_lr_final": 0.000_000_16,
            "position_lr_init": 0.000_016,
            "scaling_lr": 0.000_5,
        },
    },
    "id": "mip-splatting",
    "implementation_status": {
        "blender": "reproducing",
        "mipnerf360": "reproducing",
        "tanksandtemples": "working",
        "seathru-nerf": "working",
    }
}

register(MipSplattingSpec)
