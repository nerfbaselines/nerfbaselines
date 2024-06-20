import os
from ..registry import MethodSpec, register


paper_results = {
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
    "method": ".multinerf:MultiNeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """# Clone the repo.
git clone https://github.com/google-research/multinerf.git
cd multinerf
git checkout 5b4d4f64608ec8077222c52fdf814d40acc10bc1

conda install -y pip conda-build
conda develop "$PWD"

# Install requirements.
python -m pip install --upgrade pip
python -m pip install --upgrade 'numpy<2.0.0' "jax[cuda11_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install \
    'numpy<=2.0.0' \
    flax==0.7.5 \
    opencv-python==4.9.0.80 \
    pillow==10.2.0 \
    tensorboard==2.15.1 \
    tensorflow==2.15.0.post1 \
    gin-config==0.5.0 \
    dm-pix==0.4.2 \
    rawpy==0.19.0 \
    mediapy==1.2.0 \
    'scipy<1.13.0'
# scipy 1.13.0 is not supported by the older jax
# https://github.com/google/jax/discussions/18995
# python -m pip install 'scipy<1.13.0'

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
conda develop "$PWD/internal/pycolmap"
conda develop "$PWD/internal/pycolmap/pycolmap"

# Confirm that all the unit tests pass.
# ./scripts/run_all_unit_tests.sh
""",
    },
    "metadata": {
        "name": "Mip-NeRF 360",
        "description": "",
        "paper_title": "Mip-NeRF 360: Unbounded Anti-Aliased Neural Radiance Fields",
        "paper_authors": ["Jonathan T. Barron", "Ben Mildenhall", "Dor Verbin", "Pratul P. Srinivasan", "Peter Hedman"],
        "paper_link": "https://arxiv.org/pdf/2111.12077.pdf",
        "link": "https://jonbarron.info/mipnerf360/",
    },
    "dataset_overrides": {
        "blender": { "base_config": "blender_256.gin" },
    },
}

register(
    MultiNeRFSpec,
    name="mipnerf360",
    metadata={
        "name": "Mip-NeRF 360",
        "description": """Official Mip-NeRF 360 implementation addapted to handle different camera distortion/intrinsic parameters.
It was designed for unbounded object-centric 360-degree capture and handles anti-aliasing well.
It is, however slower to train and render compared to other approaches.""",
        "paper_results": paper_results,
    },
)
register(
    MultiNeRFSpec,
    name="mipnerf360:single-gpu",
    kwargs={
        "config_overrides": {
            "Config.batch_size": 4096,
            "lr_init": 0.002 / 2,
            "lr_final": 0.00002 / 2,
            "Config.max_steps": 1_000_000,
        }
    },
    metadata={
        "name": "Mip-NeRF 360 (single GPU)",
        "description": """Mip-NeRF 360 implementation addapted to handle different camera distortion/intrinsic parameters.
This version is optimized for a single GPU and differs from the official hyperparameters!""",
    },
)
