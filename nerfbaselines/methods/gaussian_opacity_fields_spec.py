import os
from ..registry import register, MethodSpec


GaussianOpacityFieldsSpec: MethodSpec = {
    "method": ".gaussian_opacity_fields:GaussianOpacityFields",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.8",
        "install_script": """# Install mip-splatting
git clone https://github.com/autonomousvision/gaussian-opacity-fields.git
cd gaussian-opacity-fields
git checkout 98d0858974437d329720727ee34e42e388425112
# Remove unsupported (and unnecessary) open3d dependency
sed -i '/import open3d as o3d/d' train.py

conda install -y conda-build
conda develop .

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 libcxx=17.0.6 -c conda-forge
conda install -c conda-forge -y nodejs==20.9.0
conda install -y -c conda-forge conda-forge::gmp==6.3.0 conda-forge::cgal==5.6.1

pip install -r requirements.txt
pip install -U pip 'setuptools<70.0.0'
pip install lpips==0.1.4

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/

# Add LD_LIBRARY_PATH to the environment
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LIBRARY_PATH"
export CPLUS_INCLUDE_PATH=${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH}
cd submodules/tetra-triangulation
cmake . && make && pip install -e . || exit 1

function nb-post-install () {
if [ "$NB_DOCKER_BUILD" = "1" ]; then
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
        "name": "Gaussian Opacity Fields",
        "description": """Improved Mip-Splatting with better geometry.""",
        "paper_title": "Gaussian Opacity Fields: Efficient and Compact Surface Reconstruction in Unbounded Scenes",
        "paper_authors": ["Zehao Yu", "Torsten Sattler", "Andreas Geiger"],
        "paper_link": "https://arxiv.org/pdf/2404.10772.pdf",
        "link": "https://niujinshuchong.github.io/gaussian-opacity-fields/",
    },
    "dataset_overrides": {
        "blender": { "white_background": True },
        "dtu": { "use_decoupled_appearance": True, "lambda_distortion": 100 },
        "tanksandtemples": { "use_decoupled_appearance": True },
        "phototourism": { "use_decoupled_appearance": True },
    },
}


register(GaussianOpacityFieldsSpec, name="gaussian-opacity-fields")
