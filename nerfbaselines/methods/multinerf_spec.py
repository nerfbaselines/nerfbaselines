import os
from nerfbaselines import register, MethodSpec


_paper_results = {
    # 360 scenes: bicycle flowers garden stump treehill room counter kitchen bonsai
    # 360 PSNRs: 24.37 21.73 26.98 26.40 22.87 31.63 29.55 32.23 33.46
    # 360 SSIMs: 0.685 0.583 0.813 0.744 0.632 0.913 0.894 0.920 0.941
    # 360 LPIPS: 0.301 0.344 0.170 0.261 0.339 0.211 0.204 0.127 0.176
    "mipnerf360/bicycle": {"psnr": 24.37, "ssim": 0.685, "lpips_vgg": 0.301},
    "mipnerf360/flowers": {"psnr": 21.73, "ssim": 0.583, "lpips_vgg": 0.344},
    "mipnerf360/garden": {"psnr": 26.98, "ssim": 0.813, "lpips_vgg": 0.170},
    "mipnerf360/stump": {"psnr": 26.40, "ssim": 0.744, "lpips_vgg": 0.261},
    "mipnerf360/treehill": {"psnr": 22.87, "ssim": 0.632, "lpips_vgg": 0.339},
    "mipnerf360/room": {"psnr": 31.63, "ssim": 0.913, "lpips_vgg": 0.211},
    "mipnerf360/counter": {"psnr": 29.55, "ssim": 0.894, "lpips_vgg": 0.204},
    "mipnerf360/kitchen": {"psnr": 32.23, "ssim": 0.920, "lpips_vgg": 0.127},
    "mipnerf360/bonsai": {"psnr": 33.46, "ssim": 0.941, "lpips_vgg": 0.176},
}


MultiNeRFSpec: MethodSpec = {
    "method_class": ".multinerf:MultiNeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": """# Clone the repo.
git clone https://github.com/google-research/multinerf.git
cd multinerf
git checkout 5b4d4f64608ec8077222c52fdf814d40acc10bc1

conda install -y pip conda-build
conda develop "$PWD"

# Install requirements.
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda11_pip]==0.4.23" 'numpy<2' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install \
    "chex==0.1.85" \
    "dm-pix==0.4.2" \
    "ffmpeg" \
    'numpy<2' \
    "flax==0.7.5" \
    "gin-config==0.5.0" \
    "immutabledict==4.1.0" \
    "jax==0.4.23" \
    "jaxcam==0.1.1" \
    "jaxlib==0.4.23" \
    "mediapy==1.2.0" \
    "ml_collections" \
    "numpy==1.26.4" \
    "opencv-python-headless==4.9.0.80" \
    "pillow==10.2.0" \
    "rawpy==0.19.0" \
    "scipy==1.11.4" \
    "tensorboard==2.15.1" \
    "tensorflow==2.15.0" \
    "ml-dtypes==0.2.0" \
    "orbax-checkpoint==0.4.4" \
    plyfile==0.8.1 \
    scikit-image==0.21.0 \
    tqdm==4.66.2 \
    importlib_metadata==8.5.0 \
    typing_extensions==4.12.2 \
    wandb==0.19.1 \
    click==8.1.8 \
    Pillow==10.2.0 \
    'pytest<=8.3.4' \
    matplotlib==3.9.4

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
conda develop "$PWD/internal/pycolmap"
conda develop "$PWD/internal/pycolmap/pycolmap"

# Install default torch to compute metrics on cuda inside the container
pip install torch==2.2.0 torchvision==0.17.0 'numpy<2.0.0' --index-url https://download.pytorch.org/whl/cu118
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'

# Confirm that all the unit tests pass.
# ./scripts/run_all_unit_tests.sh
""",
    },
    "metadata": {
        "name": "Mip-NeRF 360",
        "description": """Official Mip-NeRF 360 implementation addapted to handle different camera distortion/intrinsic parameters.
It was designed for unbounded object-centric 360-degree capture and handles anti-aliasing well.
It is, however slower to train and render compared to other approaches.""",
        "paper_results": _paper_results,
        "paper_title": "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields",
        "paper_authors": ["Jonathan T. Barron", "Ben Mildenhall", "Dor Verbin", "Pratul P. Srinivasan", "Peter Hedman"],
        "paper_link": "https://arxiv.org/pdf/2111.12077.pdf",
        "link": "https://jonbarron.info/mipnerf360/",
        "licenses": [{"name": "Apache 2.0","url": "https://raw.githubusercontent.com/google-research/multinerf/main/LICENSE"}],
    },
    "presets": {
        "blender": { "@apply": [{"dataset": "blender"}], "base_config": "blender_256.gin" },
        "single-gpu": {
            "@description": "A version of the method designed for single GPU training (not official).",
            "Config.batch_size": 4096,
            "lr_init": 0.002 / 2,
            "lr_final": 0.00002 / 2,
            "Config.max_steps": 1_000_000,
        }
    },
    "id": "mipnerf360",
    "implementation_status": {
        "blender": "reproducing",
        "mipnerf360": "reproducing",
        "tanksandtemples": "working",
    }
}

register(MultiNeRFSpec)
