import os
from ..registry import MethodSpec, register

KPlanesSpec: MethodSpec = {
    "method": ".kplanes:KPlanes",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": """git clone https://github.com/sarafridov/K-Planes.git kplanes --recursive
cd kplanes
git checkout 7e3a82dbdda31eddbe2a160bc9ef89e734d9fc54

conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
conda install -y pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=11.8 'numpy<2.0.0' -c pytorch -c nvidia
if [ "$NB_DOCKER_BUILD" != "1" ]; then
    conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
fi
LIBRARY_PATH="$CONDA_PREFIX/lib/stubs" pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install tqdm pillow opencv-python pandas lpips==0.1.4 imageio torchmetrics scikit-image tensorboard matplotlib
conda install -y conda-build;conda develop .
pip install lpips==0.1.4 importlib_metadata typing_extensions

function nb-post-install () {
if [ "$NB_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {} +
fi
}
""",
    },
    "metadata": {
        "name": "K-Planes",
        "description": """K-Planes is a NeRF-based method representing d-dimensional space using 2 planes allowing for a seamless way to go from static (d=3) to dynamic (d=4) scenes.""",
        "paper_title": "K-Planes: Explicit Radiance Fields in Space, Time, and Appearance",
        "paper_authors": ["Sara Fridovich-Keil", "Giacomo Meanti", "Frederik Warburg", "Benjamin Recht", "Angjoo Kanazawa"],
        "paper_link": "https://arxiv.org/pdf/2301.10241",
        "link": "https://sarafridov.github.io/K-Planes/",
    },
    "dataset_overrides": {
        "blender": { "base_config": "NeRF/nerf_hybrid.py", },
        "phototourism/trevi-fountain": { "base_config": "Phototourism/trevi_hybrid.py", },
        "phototourism/brandenburg-gate": { "base_config": "Phototourism/brandenburg_hybrid.py", },
        "phototourism/sacre-coeur": { "base_config": "Phototourism/sacrecoeur_hybrid.py", },
    }
}

register(KPlanesSpec, name="kplanes", metadata={
})
