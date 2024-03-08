import os
from ..registry import MethodSpec, register


InstantNGPSpec: MethodSpec = {
    "method": "._impl.instant_ngp:InstantNGP",
    "conda": {
        "environment_name": os.path.split(__file__[:-3])[-1].replace("_", "-"),
        "python_version": "3.8",
        "install_script": """# Install ingp
# Dependencies and environment setup
conda install -y cudatoolkit-dev=11.7 \\
    make=4.3 cmake=3.28.3 xorg-libx11=1.8.7 xorg-libxcursor=1.2.0 \\
    xorg-libxrandr=1.5.2 xorg-libxinerama=1.1.5 xorg-libxext=1.3.4 xorg-libxi=1.7.10 \\
    glew=2.1.0 openexr=3.2.2 zlib=1.2 ocl-icd-system jsoncpp=1.9.5 \\
    gcc_linux-64=11 gxx_linux-64=11 binutils=2.40 \\
    mesalib=24.0.2 mesa-libgl-cos7-x86_64=18.3.4 mesa-libgl-devel-cos7-x86_64=18.3.4 \\
    libvulkan-headers=1.3.250.0 -c conda-forge
_prefix="$CONDA_PREFIX"
conda deactivate; conda activate "$_prefix"
ln -s "$CC" "$CONDA_PREFIX/bin/gcc"
ln -s "$CXX" "$CONDA_PREFIX/bin/g++"
export CPATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPATH"

# Clone source code
git clone --recursive https://github.com/NVlabs/instant-ngp.git
cd instant-ngp
git checkout cc749144b0665ff7adeee6c57787573fa3b45787
conda install -y conda-build && conda develop .

# Fix duplicated fmt in dependencies
rm -rf $CONDA_PREFIX/include/fmt

# Build
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j

# NOTE: torch is needed for nerfbaselines
conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo "export PYTHONPATH=\\"$CONDA_PREFIX/src/instant-ngp/build:\\$PYTHONPATH\\"" >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo "export PATH=\\"$CONDA_PREFIX/src/instant-ngp/build:\\$PATH\\"" >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
""",
    },
    "docker": {
        "environment_name": os.path.split(__file__[:-3])[-1].replace("_", "-"),
        "image": "kulhanek/ingp:latest",
        "python_path": "python3",
        "home_path": "/root",
    },
    "metadata": {
        "name": "Instant NGP",
        "description": """Instant-NGP is a method that uses hash-grid and a shallow MLP to accelerate training and rendering.
This method trains very fast (~6 min) and renders also fast ~3 FPS.""",
        "paper_title": "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding",
        "paper_authors": ["Thomas Müller", "Alex Evans", "Christoph Schied", "Alexander Keller"],
        "paper_link": "https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf",
        "link": "https://nvlabs.github.io/instant-ngp/",
    },
    "backends_order": ["docker", "conda"],
}

register(InstantNGPSpec, name="instant-ngp")