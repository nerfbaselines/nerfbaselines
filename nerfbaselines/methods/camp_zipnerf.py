import os
from ..registry import MethodSpec, register


CamP_ZipNerfSpec: MethodSpec = {
    "method": "._impl.camp_zipnerf:CamP_ZipNeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-3])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": """# Clone the repo.
git clone https://github.com/jonbarron/camp_zipnerf.git
cd camp_zipnerf

# Prepare pip.
conda install -y pip conda-build

# Install requirements.
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda11_pip]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
conda develop "$PWD"
conda develop "$PWD/internal/pycolmap"
conda develop "$PWD/internal/pycolmap/pycolmap"

# Confirm that all the unit tests pass.
# ./scripts/run_all_unit_tests.sh
""",
    },
    "metadata": {},
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
