import os
from nerfbaselines import register, MethodSpec

OctreeGSSpec: MethodSpec = {
    "id": "octree-gs",
    "method_class": ".octree_gs:OctreeGS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/city-super/Octree-GS.git octreegs
cd octreegs
git checkout 7611e0febe9359d4e9f6b8aa83d304dc2d9366c8
git submodule update --init --recursive

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
conda install -y pytorch3d==0.7.7 -c pytorch3d
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'

pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'

pip install plyfile==0.8.1 \
        mediapy==1.1.2 \
        lpips==0.1.4 \
        scikit-image==0.21.0 \
        tqdm==4.66.2 \
        colorama==0.4.6 \
        opencv-python-headless==4.10.0.84 \
        importlib_metadata==8.5.0 \
        typing_extensions==4.12.2 \
        wandb==0.19.1 \
        click==8.1.8 \
        Pillow==11.1.0 \
        matplotlib==3.9.4 \
        tensorboard==2.18.0 \
        scipy==1.13.1 \
        einops==0.8.0 \
        laspy==2.5.4 \
        jaxtyping==0.2.34 \
        'pytest<=8.3.4' \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        --no-build-isolation \
        --use-pep517
pip install torch-scatter==2.1.2 \
        --no-build-isolation \
        --use-pep517 \
        -f https://data.pyg.org/whl/torch-2.0.1%2Bcu117.html
conda develop .

if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -rf {} +
# Replace all libs under $CONDA_PREFIX/lib with symlinks to pkgs/cuda-toolkit/targets/x86_64-linux/lib
for lib in "$CONDA_PREFIX"/lib/*.so*; do 
    if [ ! -f "$lib" ] || [ -L "$lib" ]; then continue; fi;
    lib="${lib%.so*}.so";libname=$(basename "$lib");
    tgt="$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/$libname"
    if [ -f "$tgt" ]; then echo "Deleting $lib"; rm -rf "$lib"*; for tgtlib in "$tgt"*; do ln -s "$tgtlib" "$(dirname "$lib")"; done; fi;
done
fi
""",
    },
    "metadata": {
        "name": "Octree-GS",
        "description": """An LOD-structured 3D Gaussian approach supporting level-of-detail decomposition for scene representation that contributes to the final rendering results.""",
        "paper_title": "Octree-GS: Towards Consistent Real-time Rendering with LOD-Structured 3D Gaussians",
        "paper_authors": ["Kerui Ren", "Lihan Jiang", "Tao Lu", "Mulin Yu", "Linning Xu", "Zhangkai Ni", "Bo Dai"],
        "paper_link": "https://arxiv.org/pdf/2403.17898.pdf",
        "link": "https://city-super.github.io/octree-gs/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/city-super/Octree-GS/refs/heads/main/LICENSE.md"}],
    },
    "presets": {
        "appearance": {
            "appearance_dim": 32,
        },
        "tanksandtemples": {
            "@apply": [{"dataset": "tanksandtemples"}],
            "visible_threshold": 0.9,
        },
        "deepblending": {
            "@apply": [{"dataset": "deepblending"}],
            "visible_threshold": 0.9,
        },
        "hierarchical-3dgs": {
            "@apply": [{"dataset": "hierarchical-3dgs"}],
            "visible_threshold": 0.01,
            "base_layer": 13,
        },
    },
}

register(OctreeGSSpec)
