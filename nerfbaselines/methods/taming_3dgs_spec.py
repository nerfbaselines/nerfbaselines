import os
from nerfbaselines import register


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
        "name": "2D Gaussian Splatting",
        "description": "2DGS adopts 2D oriented disks as surface elements and allows high-quality rendering with Gaussian Splatting. In NerfBaselines, we fixed bug with cx,cy, added appearance embedding optimization, and added support for sampling masks.",
        "paper_title": "2D Gaussian Splatting for Geometrically Accurate Radiance Fields",
        "paper_authors": ["Binbin Huang", "Zehao Yu", "Anpei Chen", "Andreas Geiger", "Shenghua Gao"],
        "paper_link": "https://arxiv.org/pdf/2403.17888.pdf",
        "link": "https://surfsplatting.github.io/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/hbb1/2d-gaussian-splatting/main/LICENSE.md"}],
    },
    "presets": {
        "blender": { 
            "@apply": [{"dataset": "blender"}], 
            "white_background": True, 
            "sh_lower": True, 
            "mode": "final_count",
            "budget": 1000000,
            "densification_interval": 100},
        "mipnerf360": { 
            "@apply": [{"dataset": "mipnerf360"}],
            "mode": "final_count",
            "sh_lower": True,
            "budget": 4000000,
            "densification_interval": 100},
        "tanksandtemples": { 
            "@apply": [{"dataset": "tanksandtemples"}], 
            "mode": "final_count", 
            "sh_lower": True, 
            "densification_interval": 100, 
            "budget": 2000000},
        "mipnerf360/bicycle": { "@apply": [{"dataset": "mipnerf360", "scene": "bicycle"}], "budget": 5987095, },
        "mipnerf360/flowers": { "@apply": [{"dataset": "mipnerf360", "scene": "flowers"}], "budget": 3618411, },
        "mipnerf360/garden": { "@apply": [{"dataset": "mipnerf360", "scene": "garden"}], "budget": 5728191, },
        "mipnerf360/stump": { "@apply": [{"dataset": "mipnerf360", "scene": "stump"}], "budget": 4867429, },
        "mipnerf360/treehill": { "@apply": [{"dataset": "mipnerf360", "scene": "treehill"}], "budget": 3770257, },
        "mipnerf360/room": { "@apply": [{"dataset": "mipnerf360", "scene": "room"}], "budget": 1548960, },
        "mipnerf360/counter": { "@apply": [{"dataset": "mipnerf360", "scene": "counter"}], "budget": 1190919, },
        "mipnerf360/kitchen": { "@apply": [{"dataset": "mipnerf360", "scene": "kitchen"}], "budget": 1803735, },
        "mipnerf360/bonsai": { "@apply": [{"dataset": "mipnerf360", "scene": "bonsai"}], "budget": 1252367, },
        "tanksandtemples/temple": { "@apply": [{"dataset": "tanksandtemples", "scene": "temple"}], "budget": 2326100, },
        "tanksandtemples/playroom": { "@apply": [{"dataset": "tanksandtemples", "scene": "playroom"}], "budget": 3273600, },
        "tanksandtemples/train": { "@apply": [{"dataset": "tanksandtemples", "scene": "train"}], "budget": 1085480, },
        "tanksandtemples/truck": { "@apply": [{"dataset": "tanksandtemples", "scene": "truck"}], "budget": 2584171, },
        "tanksandtemples/drjohnson": { "@apply": [{"dataset": "tanksandtemples", "scene": "drjohnson"}], "budget": 3273600, },
    },
    "implementation_status": {}
})
