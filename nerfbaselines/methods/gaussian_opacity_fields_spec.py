import os
from nerfbaselines import register, MethodSpec


GaussianOpacityFieldsSpec: MethodSpec = {
    "id": "gaussian-opacity-fields",
    "method_class": ".gaussian_opacity_fields:GaussianOpacityFields",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """# Install mip-splatting
git clone https://github.com/autonomousvision/gaussian-opacity-fields.git
cd gaussian-opacity-fields
git checkout 98d0858974437d329720727ee34e42e388425112
# Remove unsupported (and unnecessary) open3d dependency
sed -i '/import open3d as o3d/d' train.py

conda install -y conda-build && conda develop .
conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 libcxx=17.0.6 -c conda-forge
conda install -c conda-forge -y nodejs==20.9.0
conda install -y -c conda-forge conda-forge::gmp==6.3.0 conda-forge::cgal==5.6.1

pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
pip install \
        plyfile==0.8.1 \
        mediapy==1.1.2 \
        open3d==0.18.0 \
        ninja==1.11.1.3 \
        GPUtil==1.4.0 \
        einops==0.8.0 \
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
        scipy==1.13.1

pip install submodules/diff-gaussian-rasterization --no-build-isolation
pip install submodules/simple-knn --no-build-isolation

# Add LD_LIBRARY_PATH to the environment
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH" >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
export CUDA_HOME="$CONDA_PREFIX"
ln -s "$CC" "$CONDA_PREFIX/bin/gcc"
ln -s "$CXX" "$CONDA_PREFIX/bin/g++"
export CPATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:$LIBRARY_PATH"
export CPLUS_INCLUDE_PATH=${CUDA_HOME}/include:${CPLUS_INCLUDE_PATH}
cd submodules/tetra-triangulation
cmake . && make && pip install -e . || exit 1

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
        "name": "Gaussian Opacity Fields",
        "description": """Improved Mip-Splatting with better geometry.""",
        "paper_title": "Gaussian Opacity Fields: Efficient and Compact Surface Reconstruction in Unbounded Scenes",
        "paper_authors": ["Zehao Yu", "Torsten Sattler", "Andreas Geiger"],
        "paper_link": "https://arxiv.org/pdf/2404.10772.pdf",
        "link": "https://niujinshuchong.github.io/gaussian-opacity-fields/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/autonomousvision/gaussian-opacity-fields/main/LICENSE.md"}],
    },
    "presets": {
        "blender": { "@apply": [{"dataset": "blender"}], "white_background": True },
        "dtu": { "@apply": [{"dataset": "dtu"}], "use_decoupled_appearance": True, "lambda_distortion": 100 },
        "decoupled-appearance": {
            "@apply": [
                {"dataset": "phototourism"},
                {"dataset": "tanksandtemples"},
            ],
            "use_decoupled_appearance": True
        }
    },
    "implementation_status": {
        "blender": "working",
        "mipnerf360": "working",
        "tanksandtemples": "working",
    }
}


register(GaussianOpacityFieldsSpec)
