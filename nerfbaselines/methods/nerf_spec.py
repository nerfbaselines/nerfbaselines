import os
from nerfbaselines import register, MethodSpec


_paper_results = {
    # Blender scenes: Chair Drums Ficus Hotdog Lego Materials Mic Ship
    # Blender PSNRs: 33.00 25.01 30.13 36.18 32.54 29.62 32.91 28.65
    # Blender SSIMs: 0.967 0.925 0.964 0.974 0.961 0.949 0.980 0.856
    # Blender LPIPS: 0.046 0.091 0.044 0.121 0.050 0.063 0.028 0.206
    "blender/chair": {"psnr": 33.00, "ssim": 0.967, "lpips_vgg": 0.046},
    "blender/drums": {"psnr": 25.01, "ssim": 0.925, "lpips_vgg": 0.091},
    "blender/ficus": {"psnr": 30.13, "ssim": 0.964, "lpips_vgg": 0.044},
    "blender/hotdog": {"psnr": 36.18, "ssim": 0.974, "lpips_vgg": 0.121},
    "blender/lego": {"psnr": 32.54, "ssim": 0.961, "lpips_vgg": 0.050},
    "blender/materials": {"psnr": 29.62, "ssim": 0.949, "lpips_vgg": 0.063},
    "blender/mic": {"psnr": 32.91, "ssim": 0.980, "lpips_vgg": 0.028},
    "blender/ship": {"psnr": 28.65, "ssim": 0.856, "lpips_vgg": 0.206},
}

NeRFSpec: MethodSpec = {
    "method_class": ".nerf:NeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.8",
        "install_script": r"""# Clone the repo.
git clone https://github.com/bmild/nerf
cd nerf
git checkout 18b8aebda6700ed659cb27a0c348b737a5f6ab60
# Allow depthmaps to be outputted
sed '239a\
    ret["depth"] = depth_map
' -i "$CONDA_PREFIX/src/nerf/run_nerf.py"

# Install requirements.
conda install -y imagemagick conda-build
pip install torch==2.3.0 torchvision==0.18.0 'numpy<2.0.0' --index-url https://download.pytorch.org/whl/cu118
# nvidia-pyindex modifies paths to install older TF with newer CUDA support
# However, we do not want to corrupt user's environment, so we set the options manually
# pip install -I nvidia-pyindex==1.0.9
pip install nvidia-tensorflow==1.15.5+nv22.10 --extra-index-url https://pypi.ngc.nvidia.com --no-cache-dir --trusted-host pypi.ngc.nvidia.com
pip install \
    configargparse==1.7 \
    'opencv-python-headless<=4.10.0.82' \
    imageio==2.34.1 \
    numpy==1.23.5 \
    'plyfile<=0.8.1' \
    'mediapy<=1.1.2' \
    'scikit-image<=0.21.0' \
    'tqdm<=4.66.2' \
    'importlib_metadata<=8.5.0' \
    'typing_extensions<=4.12.2' \
    'importlib_resources<=6.5.2' \
    'wandb<=0.19.1' \
    'click<=8.1.8' \
    'Pillow<=11.1.0' \
    'matplotlib<=3.9.4' \
    'tensorboard==1.15.0' \
    'pytest<=8.3.4' \
    'scipy<=1.13.1'

# Fix tensorboard being installed incorrectly
pip install tensorboard==1.15.0 --no-deps --force-reinstall

# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg==6.1.2' x264

conda develop "$PWD"
""",
    },
    "metadata": {
        "name": "NeRF",
        "description": "Original NeRF method representing radiance field using a large MLP.",
        "paper_title": "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        "paper_authors": ["Ben Mildenhall", "Pratul P. Srinivasan", "Matthew Tancik", "Jonathan T. Barron", "Ravi Ramamoorthi", "Ren Ng"],
        "paper_results": _paper_results,
        "paper_link": "https://arxiv.org/pdf/2003.08934.pdf",
        "link": "https://www.matthewtancik.com/nerf",
        "licenses": [{"name": "MIT", "url": "https://github.com/bmild/nerf/blob/master/LICENSE"}],
    },
    "presets": {
        "blender": { "@apply": [{"dataset": "blender"}], "config": "blender_config.txt" },
        "llff": { "@apply": [{"dataset": "llff"}], "config": "llff_config.txt" },
    },
    "id": "nerf",
    "implementation_status": {
        "blender": "reproducing",
    }
}

register(NeRFSpec)
