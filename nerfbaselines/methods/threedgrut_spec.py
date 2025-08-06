import os
from nerfbaselines import register
from nerfbaselines.backends import CondaBackendSpec


_package_name = os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-")
_name = "3dgrut"
GIT_REPOSITORY = 'https://github.com/nv-tlabs/3dgrut.git'
GIT_REF = 'f41bc22936e2a8afee553b4c478c1c3a3fb9169e'

_conda_spec: CondaBackendSpec = CondaBackendSpec(
    environment_name=_name,
    python_version="3.11",
    install_script=rf"""
# Make sure gcc is at most 11 for nvcc compatibility
gcc_version=$(gcc -dumpversion)
if [ "$gcc_version" -gt 11 ]; then
    echo "[WARNING] Default gcc version $gcc_version is higher than 11. Installing GCC-11 to ensure compatibility with nvcc."
    conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
    _prefix="$CONDA_PREFIX";conda deactivate;conda activate "$_prefix"
    echo "[INFO] Installed GCC-11."
    gcc_version=$("$CC" -dumpversion)
    if [[ "$gcc_version" != 11* ]]; then
        echo "Failed to install GCC-11. Current version: $gcc_version"
        exit 1
    fi
elif [ "$gcc_version" -lt 11 ]; then
    echo "[WARNING] Default GCC version $gcc_version is lower than 11. This may cause compatibility issues with nvcc."
else
    echo "[INFO] Using default gcc version $gcc_version."
fi

echo "Cloning 3dgrut repository..."
git clone {GIT_REPOSITORY} {_name}
cd {_name}
git checkout {GIT_REF}
git submodule update --init --recursive

# Prepare GCC and ensure ffmpeg is installed
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
export LIBRARY_PATH="$CONDA_PREFIX/lib/stubs:$LIBRARY_PATH"

CUDA_VERSION="11.8.0"
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0";
conda env config vars set TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST

# Install CUDA and PyTorch dependencies
# CUDA 11.8 supports until compute capability 9.0
echo "Installing CUDA 11.8.0 ..."
conda install -y cuda-toolkit cmake ninja -c nvidia/label/cuda-11.8.0
pip install torch==2.1.2 torchvision==0.16.2 'numpy<2.0' --index-url https://download.pytorch.org/whl/cu118
pip3 install --find-links https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.1.2_cu118.html kaolin==0.17.0

# Install OpenGL headers for the playground
conda install -c conda-forge mesa-libgl-devel-cos7-x86_64 -y 

# Initialize git submodules and install Python requirements
git submodule update --init --recursive
pip install -r requirements.txt \
    'numpy<2.0.0' \
    plyfile==0.8.1 \
    mediapy==1.1.2 \
    scikit-image==0.21.0 \
    tqdm==4.66.2 \
    importlib_metadata==8.5.0 \
    typing_extensions==4.12.2 \
    wandb==0.19.1 \
    click==8.1.8 \
    Pillow==11.1.0 \
    matplotlib==3.9.4 \
    tensorboard==2.18.0 \
    pytest==8.3.4 \
    scipy==1.13.1 \
    opencv-python==4.11.0.86
pip install -e .
pip install slangtorch==1.3.11
echo "Dependencies installed successfully."

# Reload the environment variables
_prefix="$CONDA_PREFIX"
conda deactivate; conda activate "$_prefix"

# Build the native code
export MAX_JOBS=16
echo "Building native code with $MAX_JOBS jobs..."
python3 -c "
import hydra
from omegaconf import DictConfig, OmegaConf
from threedgrt_tracer.setup_3dgrt import setup_3dgrt
from threedgut_tracer.setup_3dgut import setup_3dgut
import threedgrut.utils.misc
with hydra.initialize(version_base=None, config_path='configs'):
    conf = hydra.compose(config_name='apps/colmap_3dgrt.yaml', overrides=[])
    setup_3dgrt(DictConfig(conf))
import lib3dgrt_cc as tdgrt
with hydra.initialize(version_base=None, config_path='configs'):
    conf = hydra.compose(config_name='apps/colmap_3dgut.yaml', overrides=[])
    setup_3dgut(DictConfig(conf))
import lib3dgut_cc as tdgut
" || echo "Failed to build native code. Please check the error messages above."
"""
)

register({
    "id": "3dgut",
    "method_class": f".{_package_name}:ThreeDGUT",
    "conda": _conda_spec,
    "metadata": {
        "name": "3dgut",
        "description": "3DGUT replaces traditional EWA splatting with an Unscented Transform to support nonlinear camera models and secondary effects like reflections, enabling flexible, distortion-aware rendering while retaining the speed of rasterization.",
        "paper_title": "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting",
        "paper_authors": ["Qi Wu", "Janick Martinez Esturo", "Ashkan Mirzaei", "Nicolas Moenne-Loccoz", "Zan Gojcic"],
        "paper_link": "https://arxiv.org/pdf/2412.12507.pdf",
        "paper_venue": "CVPR 2025",
        "link": "https://research.nvidia.com/labs/toronto-ai/3DGUT/",
        "licenses": [{"name": "Apache 2.0", "url": "https://raw.githubusercontent.com/nv-tlabs/3dgrut/refs/heads/main/LICENSE"}],
    },
    "presets": {
        "blender": {
            "@apply": [{"dataset": "blender"}],
            "config": "apps/nerf_synthetic_3dgut.yaml",
        },
    },
    "implementation_status": {}
})

register({
    "id": "3dgrt",
    "method_class": f".{_package_name}:ThreeDGRT",
    "conda": _conda_spec,
    "metadata": {
        "name": "3dgrt",
        "description": "The authors accelerate 3DGS rendering by replacing rasterization with a BVH-based GPU ray-tracing pipeline that wraps each particle in proxy meshes for fast ray–triangle tests, maintains raster-like speed while enabling ray-tracing effects and arbitrary cameras.",
        "paper_title": "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes",
        "paper_authors": ["Nicolas Moenne-Loccoz", "Ashkan Mirzaei", "Or Perel", "Riccardo de Lutio", "Janick Martinez Esturo", "Gavriel State", "Sanja Fidler", "Nicholas Sharp", "Zan Gojcic"],
        "paper_link": "https://arxiv.org/pdf/2407.07090.pdf",
        "paper_venue": "SIGGRAPH Asia 2024",
        "link": "https://gaussiantracer.github.io/",
        "licenses": [{"name": "Apache 2.0", "url": "https://raw.githubusercontent.com/nv-tlabs/3dgrut/refs/heads/main/LICENSE"}],
    },
    "presets": {
        "blender": {
            "@apply": [{"dataset": "blender"}],
            "config": "apps/nerf_synthetic_3dgrt.yaml",
        },
    },
    "implementation_status": {}
})
