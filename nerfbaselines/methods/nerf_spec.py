import os
from ..registry import MethodSpec, register


paper_results = {
    # Blender scenes: Chair Drums Ficus Hotdog Lego Materials Mic Ship
    # Blender PSNRs: 33.00 25.01 30.13 36.18 32.54 29.62 32.91 28.65
    # Blender SSIMs: 0.967 0.925 0.964 0.974 0.961 0.949 0.980 0.856
    # Blender LPIPS: 0.046 0.091 0.044 0.121 0.050 0.063 0.028 0.206
    "blender/chair": {"psnr": 33.00, "ssim": 0.967, "lpips": 0.046},
    "blender/drums": {"psnr": 25.01, "ssim": 0.925, "lpips": 0.091},
    "blender/ficus": {"psnr": 30.13, "ssim": 0.964, "lpips": 0.044},
    "blender/hotdog": {"psnr": 36.18, "ssim": 0.974, "lpips": 0.121},
    "blender/lego": {"psnr": 32.54, "ssim": 0.961, "lpips": 0.050},
    "blender/materials": {"psnr": 29.62, "ssim": 0.949, "lpips": 0.063},
    "blender/mic": {"psnr": 32.91, "ssim": 0.980, "lpips": 0.028},
    "blender/ship": {"psnr": 28.65, "ssim": 0.856, "lpips": 0.206},
}


NeRFSpec: MethodSpec = {
    "method": ".nerf:NeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.7",
        "install_script": r"""# Clone the repo.
git clone https://github.com/bmild/nerf
cd nerf
git checkout 18b8aebda6700ed659cb27a0c348b737a5f6ab60
# Allow depthmaps to be outputted
sed '239a\
    ret["depth"] = depth_map
' -i "$CONDA_PREFIX/src/nerf/run_nerf.py"

# Install requirements.
# conda install -y pytorch==1.1.0 torchvision==0.3.0 -c pytorch
conda install -y numpy \
                 configargparse \
                 imagemagick \
                 cudatoolkit=10.0
conda install -y -c anaconda tensorflow-gpu==1.15 

# conda install -y cudatoolkit=11.8 tensorflow-gpu=2.12 \
#     numpy pip configargparse imagemagick conda-build
# 
# conda install -y pytorch=2.3.0 torchvision=0.18.0 pytorch-cuda=11.8 -c pytorch -c nvidia
# # conda install -y cudnn=8.8
# pip install imageio

# conda install -y \
#     cudnn=8.0 pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=11.0 tensorflow-gpu=2.8 conda-build \
#     numpy pip configargparse imagemagick \
#     -c pytorch
# pip install pytorch==1.1.0+cu100 torchvision==0.3.0+cu100 -f https://download.pytorch.org/whl/cu100/torch_stable.html
# pip install -y pytorch==1.1.0 torchvision==0.3.0
pip install torch==1.2.0 torchvision==0.4.0

# conda install -y pip conda-build
conda develop "$PWD"

python -m pip install --upgrade pip
function nb-post-install () {
    python -m pip uninstall -y pillow
    python -m pip install imageio==2.9.0 "pillow<7"
}
""",
    },
    "metadata": {
        "name": "NeRF",
        "description": "Original NeRF method representing radiance field using a large MLP.",
        "paper_title": "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis",
        "paper_authors": ["Ben Mildenhall", "Pratul P. Srinivasan", "Matthew Tancik", "Jonathan T. Barron", "Ravi Ramamoorthi", "Ren Ng"],
        "paper_link": "https://arxiv.org/pdf/2003.08934.pdf",
        "link": "https://www.matthewtancik.com/nerf",
    },
}

register(
    NeRFSpec,
    name="nerf",
)
