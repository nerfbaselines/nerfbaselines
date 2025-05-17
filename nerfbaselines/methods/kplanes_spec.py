import os
from nerfbaselines import register, MethodSpec

KPlanesSpec: MethodSpec = {
    "method_class": ".kplanes:KPlanes",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": """git clone https://github.com/sarafridov/K-Planes.git kplanes
cd kplanes
git checkout 7e3a82dbdda31eddbe2a160bc9ef89e734d9fc54
git submodule update --init --recursive

conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
pip install torch==2.3.0 torchvision==0.18.0 'numpy<2.0.0' --index-url https://download.pytorch.org/whl/cu118
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
if [[ "$(gcc -v 2>&1)" != *"gcc version 11"* ]]; then
    conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
    ln -s "$CC" "$CONDA_PREFIX/bin/gcc"
    ln -s "$CXX" "$CONDA_PREFIX/bin/g++"
    export CPATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPATH"
fi
_prefix="$CONDA_PREFIX";conda deactivate;conda activate "$_prefix" || exit 1
pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
LIBRARY_PATH="$CONDA_PREFIX/lib/stubs" pip install ninja 'git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch' --no-build-isolation
pip install tqdm pillow opencv-python pandas lpips==0.1.4 imageio torchmetrics scikit-image tensorboard matplotlib
conda install -y conda-build;conda develop .
pip install lpips==0.1.4 \
    plyfile==0.8.1 \
    mediapy==1.1.2 \
    scikit-image==0.21.0 \
    tqdm==4.66.2 \
    opencv-python-headless==4.10.0.84 \
    importlib_metadata==8.5.0 \
    typing_extensions==4.12.2 \
    wandb==0.19.1 \
    click==8.1.8 \
    Pillow==11.1.0 \
    matplotlib==3.9.4 \
    tensorboard==2.18.0 \
    'pytest<=8.3.4' \
    scipy==1.13.1

if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {} +
fi
""",
    },
    "metadata": {
        "name": "K-Planes",
        "description": """K-Planes is a NeRF-based method representing d-dimensional space using 2 planes allowing for a seamless way to go from static (d=3) to dynamic (d=4) scenes.""",
        "paper_title": "K-Planes: Explicit Radiance Fields in Space, Time, and Appearance",
        "paper_authors": ["Sara Fridovich-Keil", "Giacomo Meanti", "Frederik Warburg", "Benjamin Recht", "Angjoo Kanazawa"],
        "paper_link": "https://arxiv.org/pdf/2301.10241",
        "link": "https://sarafridov.github.io/K-Planes/",
        "licenses": [{"name": "BSD 3", "url": "https://raw.githubusercontent.com/sarafridov/K-Planes/main/LICENSE"}],
    },
    "presets": {
        "blender": { "@apply": [{"dataset": "blender"}], "config": "NeRF/nerf_hybrid.py", },
        "phototourism/trevi-fountain": { 
            "@apply": [{"dataset": "phototourism", "scene": "trevi-fountain"}],
            "config": "Phototourism/trevi_hybrid.py", },
        "phototourism/brandenburg-gate": { 
            "@apply": [{"dataset": "phototourism", "scene": "brandenburg-gate"}],
            "config": "Phototourism/brandenburg_hybrid.py", },
        "phototourism/sacre-coeur": {
            "@apply": [{"dataset": "phototourism", "scene": "sacre-coeur"}],
            "config": "Phototourism/sacrecoeur_hybrid.py", },
    },
    "id": "kplanes",
    "implementation_status": {
        "blender": "reproducing",
        "phototourism": "working-not-reproducing",
    },
}

register(KPlanesSpec)
