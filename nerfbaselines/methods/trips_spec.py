import os
from ..registry import MethodSpec, register


TRIPSSpec: MethodSpec = {
    "method": "trips_python:TRIPSMethod",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9.7",
        "install_script": r"""# Install trips
# Dependencies and environment setup
conda install -y cuda -c nvidia/label/cuda-11.8.0
conda install -y -c conda-forge cudnn=8.9
conda install -y -c conda-forge \
    glog=0.6.0 gflags=2.2.2 freeimage=3.17 tensorboard=2.8.0 \
    make=4.3 cmake=3.26.4 xorg-libx11=1.8.7 xorg-libxcursor=1.2.0 \
    xorg-libxrandr=1.5.2 xorg-libxinerama=1.1.5 xorg-libxext=1.3.4 xorg-libxi=1.7.10 \
    glew=2.1.0 openexr=3.2.2 zlib=1.2 ocl-icd-system jsoncpp=1.9.5 \
    gcc_linux-64=11.4 gxx_linux-64=11.4 git=2.34.1 \
    openmp=8.0.1 minizip=1 "unzip=6.0" \
    mesalib=24.0.2 mesa-libgl-cos7-x86_64=18.3.4 mesa-libgl-devel-cos7-x86_64=18.3.4
conda install -y -c intel mkl=2024.0.0 mkl-static=2024.0.0
conda remove -y protobuf libprotobuf
# Reactivate environment to apply changes to PATH
_prefix="$CONDA_PREFIX";conda deactivate;conda activate "$_prefix"
ln -s "$CC" "$CONDA_PREFIX/bin/gcc";ln -s "$CXX" "$CONDA_PREFIX/bin/g++"
export CPATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPATH"
mkdir -p "$CONDA_PREFIX/src"
rm -rf libtorch.zip; wget https://download.pytorch.org/libtorch/cu117/libtorch-shared-with-deps-1.13.1%2Bcu117.zip -O libtorch.zip
unzip -q libtorch.zip -d "$CONDA_PREFIX/src"; rm libtorch.zip
# Libcudnn is included in conda libs, remove it from libtorch
rm -rf "$CONDA_PREFIX/src/libtorch/lib/libcudnn"*
# Remove libstdc++ from conda to use the system one (see TRIPS README.md)
rm "$CONDA_PREFIX/lib/libstdc++.so"*

# Build protobuf (we need to compile it with CXXABI=0)
git clone https://github.com/protocolbuffers/protobuf.git "$CONDA_PREFIX/src/protobuf"
_pwd="$PWD";cd "$CONDA_PREFIX/src/protobuf"
git checkout v3.18.3; git submodule update --init --recursive
cmake "$CONDA_PREFIX/src/protobuf/cmake" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DCMAKE_CXX_STANDARD=14 \
    -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0'
cmake --build .
cd "$_pwd"

# Clone source code
git clone https://github.com/lfranke/TRIPS.git "$CONDA_PREFIX/src/TRIPS"
cd "$CONDA_PREFIX/src/TRIPS"
git checkout 9885967b3a0475f41dae230bb66777d70eae4955
git submodule update --init --recursive
mkdir -p "$CONDA_PREFIX/src/TRIPS/External"
ln -s "$CONDA_PREFIX/src/libtorch" "$CONDA_PREFIX/src/TRIPS/External/libtorch"
# Fix TRIPS to support older gpu
sed -i 's%^\#include <cuda/barrier>%//\#include <cuda/barrier>%g' "$CONDA_PREFIX/src/TRIPS/src/lib/rendering/RenderForward.cu"
sed -i 's%^\#include <cuda/pipeline>%//\#include <cuda/pipeline>%g' "$CONDA_PREFIX/src/TRIPS/src/lib/rendering/RenderForward.cu"

# Build
cmake -DCMAKE_PREFIX_PATH="$CONDA_PREFIX/src/libtorch" \
    -DProtobuf_USE_STATIC_LIBS=off \
    -DProtobuf_INCLUDE_DIR=${CONDA_PREFIX}/src/protobuf/src \
    -DProtobuf_PROTOC_EXECUTABLE=${CONDA_PREFIX}/src/protobuf/protoc \
    -DProtobuf_LIBRARY=${CONDA_PREFIX}/src/protobuf/libprotobuf.a \
    -DProtobuf_PROTOC_LIBRARY=${CONDA_PREFIX}/src/protobuf/libprotoc.a \
    -DCMAKE_CXX_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' \
    -DCMAKE_CUDA_FLAGS='-D_GLIBCXX_USE_CXX11_ABI=0' \
    -B "build" .
cmake --build "$CONDA_PREFIX/src/TRIPS/build" --verbose

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo "export PYTHONPATH=\\"$CONDA_PREFIX/src/TRIPS/build:\\$PYTHONPATH\\"" >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo "export PATH=\\"$CONDA_PREFIX/src/TRIPS/build:\\$PATH\\"" >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo "export LD_LIBRARY_PATH=\\"$CONDA_PREFIX/lib:\\$LD_LIBRARY_PATH\\"" >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

# Install PyTorch such that it can be used by NB
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
""",
    },
    "metadata": {
        "name": "TRIPS",
        "description": """TRIPS performs point splatting into feature pyramid processed using CNNs. Speed comparable to Gaussian Splatting""",
        "paper_title": "TRIPS: Trilinear Point Splatting for Real-Time Radiance Field Rendering",
        "paper_authors": [
            "Linus Franke", "Darius Rückert", "Laura Fink", "Marc Stamminger",
        ],
        "paper_link": "https://arxiv.org/pdf/2401.06003",
        "link": "https://lfranke.github.io/trips/",
    },
}

register(TRIPSSpec, name="trips")
