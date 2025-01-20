import os
from nerfbaselines import register, MethodSpec


_BLENDER_NOTE = """Instant-NGP trained and evaluated on black background instead of white."""
paper_results = {
    # blender scenes: Mic Ficus Chair Hotdog Materials Drums Ship Lego
    # blender PSNRs: 36.22 33.51 35.00 37.40 29.78 26.02 31.10 36.39 
    "blender/mic": {"psnr": 36.22, "note": _BLENDER_NOTE},
    "blender/ficus": {"psnr": 33.51, "note": _BLENDER_NOTE},
    "blender/chair": {"psnr": 35.00, "note": _BLENDER_NOTE},
    "blender/hotdog": {"psnr": 37.40, "note": _BLENDER_NOTE},
    "blender/materials": {"psnr": 29.78, "note": _BLENDER_NOTE},
    "blender/drums": {"psnr": 26.02, "note": _BLENDER_NOTE},
    "blender/ship": {"psnr": 31.10, "note": _BLENDER_NOTE},
    "blender/lego": {"psnr": 36.39, "note": _BLENDER_NOTE},
}


InstantNGPSpec: MethodSpec = {
    "method_class": ".instant_ngp:InstantNGP",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": r"""# Install ingp
# Dependencies and environment setup
conda install -y cuda-toolkit -c "nvidia/label/cuda-11.7.1"
conda install -y  \
    make=4.3 cmake=3.28.3 xorg-libx11=1.8.7 xorg-libxcursor=1.2.0 \
    xorg-libxrandr=1.5.2 xorg-libxinerama=1.1.5 xorg-libxext=1.3.4 xorg-libxi=1.7.10 \
    glew=2.1.0 openexr=3.2.2 zlib=1.2 ocl-icd-system jsoncpp=1.9.5 \
    gcc_linux-64=11 gxx_linux-64=11 binutils=2.40 \
    mesalib=24.0.2 mesa-libgl-cos7-x86_64=18.3.4 mesa-libgl-devel-cos7-x86_64=18.3.4 \
    libvulkan-headers=1.3.250.0 -c conda-forge
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
_prefix="$CONDA_PREFIX"
conda deactivate; conda activate "$_prefix"
ln -s "$CC" "$CONDA_PREFIX/bin/gcc"
ln -s "$CXX" "$CONDA_PREFIX/bin/g++"
export CPATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPATH"

# Clone source code
git clone https://github.com/NVlabs/instant-ngp.git
cd instant-ngp
git checkout cc749144b0665ff7adeee6c57787573fa3b45787
git submodule update --init --recursive
conda install -y conda-build && conda develop .
# Replace python version in CMakeLists.txt
sed -i "s/Python 3\.7/Python 3\.9/g" CMakeLists.txt

# Fix duplicated fmt in dependencies
rm -rf $CONDA_PREFIX/include/fmt

# Build
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"
cmake . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DPYTHON_EXECUTABLE="$CONDA_PREFIX/bin/python3.9" \
    -DPYTHON_LIBRARY="$CONDA_PREFIX/lib/libpython3.9.so" \
    -DPYTHON_INCLUDE_DIR="$CONDA_PREFIX/include/python3.9" \
    -B build
cmake --build build --config RelWithDebInfo -j

# NOTE: torch is needed for nerfbaselines
conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
pip install msgpack==1.0.8 \
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
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
echo "export PYTHONPATH=\"$CONDA_PREFIX/src/instant-ngp/build:\$PYTHONPATH\"" >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo "export LD_LIBRARY_PATH=\"$CONDA_PREFIX/src/instant-ngp/build:$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH\"" >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"
echo "export PATH=\"$CONDA_PREFIX/src/instant-ngp/build:\$PATH\"" >> "$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

# Test pyngp is available
# If not in CI, test the installation
if [ "$GITHUB_ACTIONS" != "true" ] && [ "$NERFBASELINES_DOCKER_BUILD" != "1" ]; then
    conda deactivate; conda activate "$_prefix"; 
    echo "Testing pyngp"
    LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/stubs" python -c "import pyngp;" || exit 1
fi
""",
    },
    "metadata": {
        "name": "Instant NGP",
        "description": """Instant-NGP is a method that uses hash-grid and a shallow MLP to accelerate training and rendering.
This method trains very fast (~6 min) and renders also fast ~3 FPS.""",
        "paper_title": "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding",
        "paper_authors": ["Thomas Müller", "Alex Evans", "Christoph Schied", "Alexander Keller"],
        "paper_link": "https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf",
        "link": "https://nvlabs.github.io/instant-ngp/",
        "paper_results": paper_results,
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/NVlabs/instant-ngp/master/LICENSE.txt"}],
    },
    "backends_order": ["docker", "conda"],
    "presets": {
        "blender": {
            "@apply": [{"dataset": "blender"}],
            "testbed.color_space": "SRGB",
            "testbed.nerf.cone_angle_constant": 0,
            "testbed.nerf.training.random_bg_color": False,
            "testbed.background_color": "1.0,1.0,1.0,1.0",
            "aabb_scale": None,
            "keep_coords": True,
        },
    },
    "id": "instant-ngp",
    "implementation_status": {
        "blender": "reproducing",
        "mipnerf360": "working",
        "tanksandtemples": "working",
        "nerfstudio": "working",
    },
}

register(InstantNGPSpec)
