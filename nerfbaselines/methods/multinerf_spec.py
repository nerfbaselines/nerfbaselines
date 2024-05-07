import os
from ..registry import MethodSpec, register


MultiNeRFSpec: MethodSpec = {
    "method": ".multinerf:MultiNeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """# Clone the repo.
git clone https://github.com/jkulhanek/multinerf.git
cd multinerf
git checkout 0e6699cc01eb3f0e77e0f7c15057a3ee29ad74ba

conda install -y pip conda-build
conda develop "$PWD"

# Install requirements.
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install -r requirements.txt

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
}

register(
    MultiNeRFSpec,
    name="mipnerf360",
    metadata={
        "name": "Mip-NeRF 360",
        "description": """Official Mip-NeRF 360 implementation addapted to handle different camera distortion/intrinsic parameters.
It was designed for unbounded object-centric 360-degree capture and handles anti-aliasing well.
It is, however slower to train and render compared to other approaches.""",
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