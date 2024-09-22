import os
from nerfbaselines import register


register({
    "id": "taming-3dgs",
    "method_class": ".taming_3dgs:Taming3DGS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone --recursive https://github.com/ubc-vision/3dgs-mcmc.git
cd taming-3dgs
git checkout a22a24bd7e64b089d983bfcf52c906df7d46f25d
git submodule update --recursive

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
pip install -U pip 'setuptools<70.0.0'
pip install plyfile==0.8.1 \
        mediapy==1.1.2 \
        open3d==0.18.0 \
        lpips==0.1.4 \
        scikit-image==0.21.0 \
        tqdm==4.66.2 \
        trimesh==4.3.2 \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        opencv-python-headless \
        importlib_metadata typing_extensions

conda develop .

function nb-post-install () {
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
}
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
        "blender": { "@apply": [{"dataset": "blender"}], "white_background": True, },
    },
    "implementation_status": {
        "blender": "working",
        "mipnerf360": "reproducing",
        "tanksandtemples": "working",
        "seathru-nerf": "working",
    }
})
