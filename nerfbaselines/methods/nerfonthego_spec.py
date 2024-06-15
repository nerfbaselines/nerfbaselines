import os
from ..registry import MethodSpec, register


paper_results = {
    # TODO: ...
}


register({
    "method": ".nerfonthego:NeRFOnthego",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """# Clone the repo.
git clone https://github.com/cvg/nerf-on-the-go.git
cd nerf-on-the-go
git checkout 1aba52266a9330095feab74f3a8e5d8a5f4c3ef3

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
""",
    },
    "metadata": {
        "name": "NeRF On-the-go",
        "description": "NeRF On-the-go enables novel view synthesis in in-the-wild scenes from casually captured images.",
        "paper_title": "NeRF On-the-go: Exploiting Uncertainty for Distractor-free NeRFs in the Wild",
        "paper_authors": ["Weining Ren", "Zihan Zhu", "Boyang Sun", "Julia Chen", "Marc Pollefeys", "Songyou Peng"],
        "paper_link": "https://arxiv.org/pdf/2405.18715.pdf",
        "link": "https://rwn17.github.io/nerf-on-the-go/",
        "paper_results": paper_results,
    },
}, name="nerfonthego")

