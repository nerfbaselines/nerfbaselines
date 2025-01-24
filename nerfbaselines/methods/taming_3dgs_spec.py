import os
from nerfbaselines import register


_paper_note = """Experiments use the default 'ours' configuration. There is also 'big' configuration which uses more resources and achieves better results."""
_paper_results = {
        "mipnerf360/bicycle": { "psnr": 24.97, "note": _paper_note },
        "mipnerf360/bonsai": { "psnr": 31.8, "note": _paper_note },
        "mipnerf360/counter": { "psnr": 28.59, "note": _paper_note },
        "mipnerf360/flowers": { "psnr": 21.18, "note": _paper_note },
        "mipnerf360/garden": { "psnr": 27.45, "note": _paper_note },
        "mipnerf360/kitchen": { "psnr": 31.14, "note": _paper_note },
        "mipnerf360/room": { "psnr": 31.39, "note": _paper_note },
        "mipnerf360/stump": { "psnr": 26.04, "note": _paper_note },
        "mipnerf360/treehill": { "psnr": 23.04, "note": _paper_note },
}

register({
    "id": "taming-3dgs",
    "method_class": ".taming_3dgs:Taming3DGS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/humansensinglab/taming-3dgs.git
cd taming-3dgs
git checkout 446f2c0d50d082e660e5b899d304da5931351dec
git submodule update --recursive --init

git submodule update --init --recursive

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
conda install -c conda-forge -y nodejs==20.9.0
conda develop .

pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
pip install plyfile==0.8.1 \
        mediapy==1.1.2 \
        open3d==0.18.0 \
        lpips==0.1.4 \
        scikit-image==0.21.0 \
        tqdm==4.66.2 \
        trimesh==4.3.2 \
        opencv-python-headless==4.10.0.84 \
        importlib_metadata==8.5.0 \
        typing_extensions==4.12.2 \
        wandb==0.19.1 \
        click==8.1.8 \
        Pillow==11.1.0 \
        matplotlib==3.9.4 \
        tensorboard==2.18.0 \
        pytest==8.3.4 \
        websockets==14.2 \
        scipy==1.13.1 \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        --no-build-isolation
pip install \
    git+https://github.com/rahul-goel/fused-ssim@30fb258c8a38fe61e640c382f891f14b2e8b0b5a \
    --no-build-isolation

if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {} +
# Replace all libs under $CONDA_PREFIX/lib with symlinks to pkgs/cuda-toolkit/targets/x86_64-linux/lib
for lib in "$CONDA_PREFIX"/lib/*.so*; do 
    if [ ! -f "$lib" ] || [ -L "$lib" ]; then continue; fi;
    lib="${lib%.so*}.so";libname=$(basename "$lib");
    tgt="$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/$libname"
    if [ -f "$tgt" ]; then echo "Deleting $lib"; rm "$lib"*; for tgtlib in "$tgt"*; do ln -s "$tgtlib" "$(dirname "$lib")"; done; fi;
done
fi
""",
    },
    "metadata": {
        "name": "Taming 3DGS",
        "description": "Taming 3DGS improves the densification process to make the primitive count deterministic and implements several low-level optimizations for fast convergence",
        "paper_title": "Taming 3DGS: High-Quality Radiance Fields with Limited Resources",
        "paper_authors": ["Saswat Subhajyoti Mallick", "Rahul Goel", "Bernhard Kerbl", "Francisco Vicente Carrasco", "Markus Steinberger", "Fernando De La Torre"],
        "paper_link": "https://humansensinglab.github.io/taming-3dgs/docs/paper_lite.pdf",
        "paper_results": _paper_results,
        "link": "https://humansensinglab.github.io/taming-3dgs/",
        "licenses": [
            {"name": "MIT", "url": "https://raw.githubusercontent.com/humansensinglab/taming-3dgs/refs/heads/main/LICENSE.md" },
            {"name": "custom, research only", "url": "https://raw.githubusercontent.com/humansensinglab/taming-3dgs/refs/heads/main/LICENSE_ORIGINAL.md"}],
    },
    "presets": {
        "blender": { 
            "@apply": [{"dataset": "blender"}], 
            "white_background": True, 
            "sh_lower": True, 
            "densification_interval": 500},
        "big-mipnerf360": { 
            "mode": "final_count",
            "sh_lower": True,
            "budget": 4000000,
            "densification_interval": 100},
        "big-tanksandtemples": { 
            "mode": "final_count", 
            "sh_lower": True, 
            "densification_interval": 100, 
            "budget": 2000000},
        "big-mipnerf360/bicycle": { "budget": 5987095, },
        "big-mipnerf360/flowers": { "budget": 3618411, },
        "big-mipnerf360/garden": { "budget": 5728191, },
        "big-mipnerf360/stump": { "budget": 4867429, },
        "big-mipnerf360/treehill": { "budget": 3770257, },
        "big-mipnerf360/room": { "budget": 1548960, },
        "big-mipnerf360/counter": { "budget": 1190919, },
        "big-mipnerf360/kitchen": { "budget": 1803735, },
        "big-mipnerf360/bonsai": { "budget": 1252367, },
        "big-tanksandtemples/temple": { "budget": 2326100, },
        "big-tanksandtemples/playroom": { "budget": 3273600, },
        "big-tanksandtemples/train": { "budget": 1085480, },
        "big-tanksandtemples/truck": { "budget": 2584171, },
        "big-tanksandtemples/drjohnson": { "budget": 3273600, },

        "mipnerf360": {
            "@apply": [{"dataset": "mipnerf360"}],
            "mode": "multiplier",
            "budget": 15,
            "sh_lower": True,
            "densification_interval": 500
        },
        "mipnerf360/indoor": { 
            "@apply": [
                {"dataset": "mipnerf360", "scene": "bonsai"},
                {"dataset": "mipnerf360", "scene": "room"},
                {"dataset": "mipnerf360", "scene": "kitchen"},
                {"dataset": "mipnerf360", "scene": "counter"},
            ],
            "budget": 2, 
        },
        "tanksandtemples": {
            "@apply": [{"dataset": "tanksandtemples"}],
            "mode": "multiplier",
            "budget": 5,
            "sh_lower": True,
            "densification_interval": 500
        },
    },
    "implementation_status": {
        "blender": "working",
        "mipnerf360": "reproducing",
        "tanksandtemples": "working",
    }
})
