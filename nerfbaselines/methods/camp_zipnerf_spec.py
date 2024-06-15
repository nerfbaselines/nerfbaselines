import os
from ..registry import MethodSpec, register


zipnerf_paper_results = {
    # 360 scenes: bicycle flowers garden stump treehill room counter kitchen bonsai
    # 360 PSNRs: 25.80 22.40 28.20 27.55 23.89 32.65 29.38 32.50 34.46
    # 360 SSIMs: 0.769 0.642 0.860 0.800 0.681 0.925 0.902 0.928 0.949
    # 360 LPIPS: 0.208 0.273 0.118 0.193 0.242 0.196 0.185 0.116 0.173
    "mipnerf360/bicycle": {"psnr": 25.80, "ssim": 0.769, "lpips_vgg": 0.208},
    "mipnerf360/flowers": {"psnr": 22.40, "ssim": 0.642, "lpips_vgg": 0.273},
    "mipnerf360/garden": {"psnr": 28.20, "ssim": 0.860, "lpips_vgg": 0.118},
    "mipnerf360/stump": {"psnr": 27.55, "ssim": 0.800, "lpips_vgg": 0.193},
    "mipnerf360/treehill": {"psnr": 23.89, "ssim": 0.681, "lpips_vgg": 0.242},
    "mipnerf360/room": {"psnr": 32.65, "ssim": 0.925, "lpips_vgg": 0.196},
    "mipnerf360/counter": {"psnr": 29.38, "ssim": 0.902, "lpips_vgg": 0.185},
    "mipnerf360/kitchen": {"psnr": 32.50, "ssim": 0.928, "lpips_vgg": 0.116},
    "mipnerf360/bonsai": {"psnr": 34.46, "ssim": 0.949, "lpips_vgg": 0.173},

    # blender scenes: chair drums ficus hotdog lego materials mic ship 
    # blender PSNRs: 34.84 25.84 33.90 37.14 34.84 31.66 35.15 31.38
    # blender SSIMs: 0.983 0.944 0.985 0.984 0.980 0.969 0.991 0.929
    # blender LPIPs: 0.017 0.050 0.015 0.020 0.019 0.032 0.007 0.091
    "blender/chair": {"psnr": 34.84, "ssim": 0.983, "lpips_vgg": 0.017},
    "blender/drums": {"psnr": 25.84, "ssim": 0.944, "lpips_vgg": 0.050},
    "blender/ficus": {"psnr": 33.90, "ssim": 0.985, "lpips_vgg": 0.015},
    "blender/hotdog": {"psnr": 37.14, "ssim": 0.984, "lpips_vgg": 0.020},
    "blender/lego": {"psnr": 34.84, "ssim": 0.980, "lpips_vgg": 0.019},
    "blender/materials": {"psnr": 31.66, "ssim": 0.969, "lpips_vgg": 0.032},
    "blender/mic": {"psnr": 35.15, "ssim": 0.991, "lpips_vgg": 0.007},
    "blender/ship": {"psnr": 31.38, "ssim": 0.929, "lpips_vgg": 0.091},
}


CamP_ZipNerfSpec: MethodSpec = {
    "method": ".camp_zipnerf:CamP_ZipNeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": """# Clone the repo.
git clone https://github.com/jonbarron/camp_zipnerf.git
cd camp_zipnerf
git checkout 16206bd88f37d5c727976557abfbd9b4fa28bbe1

# Prepare pip.
conda install -y pip conda-build

# Install requirements.
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda11_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install -r requirements.txt
# scipy 1.13.0 is not supported by the older jax
# https://github.com/google/jax/discussions/18995
python -m pip install 'scipy<1.13.0'

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
conda develop "$PWD"
conda develop "$PWD/internal/pycolmap"
conda develop "$PWD/internal/pycolmap/pycolmap"

# Confirm that all the unit tests pass.
# ./scripts/run_all_unit_tests.sh
""",
    },
    "metadata": {
    },
    "dataset_overrides": {
        "blender": {"base_config": "zipnerf/blender"},
    },
}

register(
    CamP_ZipNerfSpec,
    name="zipnerf",
    kwargs={
        "camp": False,
    },
    metadata={
        "name": "Zip-NeRF",
        "paper_title": "Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields",
        "paper_authors": ["Jonathan T. Barron", "Ben Mildenhall", "Dor Verbin", "Pratul P. Srinivasan", "Peter Hedman"],
        "paper_link": "https://arxiv.org/pdf/2304.06706.pdf",
        "link": "https://jonbarron.info/zipnerf/",
        "description": """Zip-NeRF is a radiance field method which addresses the aliasing problem in the case of hash-grid based methods (iNGP-based).
Instead of sampling along the ray it samples along a spiral path - approximating integration along the frustum. """,
        "paper_results": zipnerf_paper_results,
    },
)

register(
    CamP_ZipNerfSpec,
    name="camp",
    metadata={
        "name": "Zip-NeRF",
        "paper_title": "CamP: Camera Preconditioning for Neural Radiance Fields",
        "paper_authors": ["Keunhong Park", "Philipp Henzler", "Ben Mildenhall", "Jonathan T. Barron", "Ricardo Martin-Brualla"],
        "paper_link": "https://arxiv.org/pdf/2308.10902.pdf",
        "link": "https://camp-nerf.github.io/",
        "description": """CamP is an extension of Zip-NeRF which adds pose refinement to the training process. """,
    },
    kwargs={
        "camp": True,
    }
)
