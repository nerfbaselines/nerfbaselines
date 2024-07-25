import os
from ..registry import MethodSpec, register


paper_results = {
    "seathru-nerf/curasao": { "psnr": 30.48, "ssim": 0.87, "lpips": 0.20 },
    "seathru-nerf/panama": { "psnr": 27.89, "ssim": 0.83, "lpips": 0.22 },
    "seathru-nerf/japanese-gradens": { "psnr": 21.83, "ssim": 0.77, "lpips": 0.25 },
}


SeaThruNeRFSpec: MethodSpec = {
    "method": ".seathru_nerf:SeaThruNeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": """# Clone the repo.
git clone https://github.com/deborahLevy130/seathru_NeRF seathru-nerf
cd seathru-nerf
git checkout 3f4ebfe2c9dcb93af7916a3c7e7e196b9b956160

conda install -y pip conda-build
conda develop "$PWD"

# Install requirements.
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda11_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install \
    "chex==0.1.85" \
    "dm-pix==0.4.2" \
    "ffmpeg" \
    "flax==0.7.5" \
    "gin-config==0.5.0" \
    "immutabledict==4.1.0" \
    "jax==0.4.23" \
    "jaxcam==0.1.1" \
    "jaxlib==0.4.23" \
    "mediapy==1.2.0" \
    "ml_collections" \
    "numpy==1.26.4" \
    "opencv-python==4.9.0.80" \
    "pillow==10.2.0" \
    "rawpy==0.19.0" \
    "scipy==1.11.0" \
    "tensorboard==2.15.1" \
    "tensorflow==2.15.0" \
    "ml-dtypes==0.2.0" \
    "orbax-checkpoint==0.4.4"


# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
rm -rf ./internal/pycolmap
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
conda develop "$PWD/internal/pycolmap"
conda develop "$PWD/internal/pycolmap/pycolmap"

# Confirm that all the unit tests pass.
# ./scripts/run_all_unit_tests.sh
""",
    },
    "metadata": {
        "name": "SeaThru-NeRF",
        "paper_title": "SeaThru-NeRF: Neural Radiance Fields in Scattering Media",
        "paper_authors": ["Deborah Levy", "Amit Peleg", "Naama Pearl", "Dan Rosenbaum", "Derya Akkaynak", "Tali Treibitz", "Simon Korman"],
        "paper_link": "https://openaccess.thecvf.com/content/CVPR2023/papers/Levy_SeaThru-NeRF_Neural_Radiance_Fields_in_Scattering_Media_CVPR_2023_paper.pdf",
        "link": "https://sea-thru-nerf.github.io/",
        "licenses": [{"name": "Apache 2.0","url": "https://raw.githubusercontent.com/deborahLevy130/seathru_NeRF/master/LICENSE"}],
        "description": """Official SeaThru-NeRF implementation.
It is based on MipNeRF 360 and was disagned for underwater scenes.""",
        "paper_results": paper_results,
    },
}

register(SeaThruNeRFSpec, name="seathru-nerf")
