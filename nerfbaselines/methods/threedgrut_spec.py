import os
from nerfbaselines import register
from nerfbaselines.backends import CondaBackendSpec


_package_name = os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-")
_name = "3dgrut"
GIT_REPOSITORY = 'https://github.com/nv-tlabs/3dgrut.git'
GIT_REF = 'f41bc22936e2a8afee553b4c478c1c3a3fb9169e'

_scannetpp_3dgut_note = (
    "The results are not part of the paper (only results for `Ours (sorted)` are reported), but are provided in the GitHub repository."
)
_scannetpp_3dgut_fps_note = f"{_scannetpp_3dgut_note} FPS was computed on a single NVIDIA RTX 5090 GPU"
_blender_3dgut_note = (
    "The method was trained and evaluated on black background instead of white, which is the default in NerfBaselines."
    " The results are not part of the paper (only results for `Ours (sorted)` are reported), but are provided in the GitHub repository."
)
_blender_3dgut_fps_note = f"{_blender_3dgut_note} FPS was computed on a single NVIDIA RTX 5090 GPU"
_3dgut_paper_results = {
    "blender/chair": {"psnr": 35.61, "ssim": 0.988, "fps": 599, "note": _blender_3dgut_note, "note_fps": _blender_3dgut_fps_note},
    "blender/drums": {"psnr": 25.99, "ssim": 0.953, "fps": 694, "note": _blender_3dgut_note, "note_fps": _blender_3dgut_fps_note},
    "blender/ficus": {"psnr": 36.43, "ssim": 0.988, "fps": 1053, "note": _blender_3dgut_note, "note_fps": _blender_3dgut_fps_note},
    "blender/hotdog": {"psnr": 38.11, "ssim": 0.986, "fps": 952, "note": _blender_3dgut_note, "note_fps": _blender_3dgut_fps_note},
    "blender/lego": {"psnr": 36.47, "ssim": 0.984, "fps": 826, "note": _blender_3dgut_note, "note_fps": _blender_3dgut_fps_note},
    "blender/materials": {"psnr": 30.39, "ssim": 0.960, "fps": 1000, "note": _blender_3dgut_note, "note_fps": _blender_3dgut_fps_note},
    "blender/mic": {"psnr": 36.32, "ssim": 0.992, "fps": 775, "note": _blender_3dgut_note, "note_fps": _blender_3dgut_fps_note},
    "blender/ship": {"psnr": 31.72, "ssim": 0.908, "fps": 870, "note": _blender_3dgut_note, "note_fps": _blender_3dgut_fps_note},
    "scannetpp/0a5c013435": {"psnr": 29.67, "ssim": 0.930, "fps": 389, "note": _scannetpp_3dgut_note, "note_fps": _scannetpp_3dgut_fps_note},
    "scannetpp/8d563fc2cc": {"psnr": 26.88, "ssim": 0.912, "fps": 439, "note": _scannetpp_3dgut_note, "note_fps": _scannetpp_3dgut_fps_note},
    "scannetpp/bb87c292ad": {"psnr": 31.58, "ssim": 0.941, "fps": 448, "note": _scannetpp_3dgut_note, "note_fps": _scannetpp_3dgut_fps_note},
    "scannetpp/d415cc449b": {"psnr": 28.12, "ssim": 0.871, "fps": 483, "note": _scannetpp_3dgut_note, "note_fps": _scannetpp_3dgut_fps_note},
    "scannetpp/e8ea9b4da8": {"psnr": 33.47, "ssim": 0.954, "fps": 394, "note": _scannetpp_3dgut_note, "note_fps": _scannetpp_3dgut_fps_note},
    "scannetpp/fe1733741f": {"psnr": 25.60, "ssim": 0.858, "fps": 450, "note": _scannetpp_3dgut_note, "note_fps": _scannetpp_3dgut_fps_note},
    "mipnerf360/bicycle": {"psnr": 24.21, "ssim": 0.741, "lpips": 0.202},
    "mipnerf360/bonsai": {"psnr": 32.17, "ssim": 0.941, "lpips": 0.226},
    "mipnerf360/counter": {"psnr": 29.03, "ssim": 0.908, "lpips": 0.197},
    "mipnerf360/garden": {"psnr": 26.90, "ssim": 0.851, "lpips": 0.121},
    "mipnerf360/kitchen": {"psnr": 31.23, "ssim": 0.926, "lpips": 0.126},
    "mipnerf360/stump": {"psnr": 26.51, "ssim": 0.768, "lpips": 0.222},
    "mipnerf360/flowers": {"psnr": 21.48, "ssim": 0.612, "lpips": 0.316},
    "mipnerf360/room": {"psnr": 31.64, "ssim": 0.919, "lpips": 0.218},
    "mipnerf360/treehill": {"psnr": 22.15, "ssim": 0.623, "lpips": 0.332},
}


_3dgrt_blender_note = "The method was trained and evaluated on black background instead of white, which is the default in NerfBaselines."
_3dgrt_paper_results = {
    "blender/chair": {"psnr": 36.02, "note": _3dgrt_blender_note},
    "blender/drums": {"psnr": 25.89, "note": _3dgrt_blender_note},
    "blender/ficus": {"psnr": 36.08, "note": _3dgrt_blender_note},
    "blender/hotdog": {"psnr": 37.63, "note": _3dgrt_blender_note},
    "blender/lego": {"psnr": 36.20, "note": _3dgrt_blender_note},
    "blender/materials": {"psnr": 30.17, "note": _3dgrt_blender_note},
    "blender/mic": {"psnr": 34.27, "note": _3dgrt_blender_note},
    "blender/ship": {"psnr": 30.77, "note": _3dgrt_blender_note},
}

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
    conf_3dgut = DictConfig(hydra.compose(config_name='apps/colmap_3dgut.yaml', overrides=[]))
with hydra.initialize(version_base=None, config_path='configs'):
    conf_3dgrt = DictConfig(hydra.compose(config_name='apps/colmap_3dgrt.yaml', overrides=[]))
try:
    setup_3dgut(conf_3dgut)
    setup_3dgrt(conf_3dgrt)
    import lib3dgut_cc as tdgut
    import lib3dgrt_cc as tdgrt
except ImportError as e:
    # ImportError: libcuda.so.1: cannot open shared object file: No such file or directory
    if 'libcuda.so.1' in str(e):
        print('NVIDIA driver not found. Skipping pre-building extension libraries.')
    else: raise e
"
"""
)

_long_description_3dgut = """
The official implementation implements pinhole cameras and opencv_fisheye camera models.
In NerfBaselines, we extend the implementation to also support opencv and full_opencv camera models (all currently supported by NerfBaselines).
Masks are also supported and we further extend the implementation to support arbitrary background colors.

The default configuration is `unsorted` (corresponds to `Ours` in the 3DGUT paper). There is also `sorted` preset which can be enabled by adding `--preset sorted` to the command line. The sorted preset corresponds to the `Ours (sorted)` results in the paper.
"""

register({
    "id": "3dgut",
    "method_class": f".{_package_name}:ThreeDGUT",
    "conda": _conda_spec,
    "metadata": {
        "name": "3DGUT",
        "description": "3DGUT replaces traditional EWA splatting with an Unscented Transform to support nonlinear camera models and secondary effects like reflections, enabling flexible, distortion-aware rendering while retaining the speed of rasterization.",
        "long_description": _long_description_3dgut,
        "paper_title": "3DGUT: Enabling Distorted Cameras and Secondary Rays in Gaussian Splatting",
        "paper_authors": ["Qi Wu", "Janick Martinez Esturo", "Ashkan Mirzaei", "Nicolas Moenne-Loccoz", "Zan Gojcic"],
        "paper_results": _3dgut_paper_results,
        "paper_link": "https://arxiv.org/pdf/2412.12507.pdf",
        "paper_venue": "CVPR 2025",
        "link": "https://research.nvidia.com/labs/toronto-ai/3DGUT/",
        "licenses": [{"name": "Apache 2.0", "url": "https://raw.githubusercontent.com/nv-tlabs/3dgrut/refs/heads/main/LICENSE"}],
    },
    "presets": {
        "blender": {
            "@apply": [{"dataset": "blender"}],
            "config": "paper/3dgut/unsorted_nerf_synthetic.yaml",
        },
        "scannetpp": {
            "@apply": [{"dataset": "scannetpp"}],
            "config": "paper/3dgut/unsorted_scannetpp.yaml",
        },
        "sorted": {
            "render.splat.k_buffer_size": 16
        },
    },
    "implementation_status": {}
})

_long_description_3dgrt = """
The official implementation implements pinhole cameras and opencv_fisheye camera models.
In NerfBaselines, we extend the implementation to also support opencv and full_opencv camera models (all currently supported by NerfBaselines).
Masks are also supported.

NOTE: In our experiments, 3DGRT always render only black images and no Gaussians get splitted/cloned.
"""

register({
    "id": "3dgrt",
    "method_class": f".{_package_name}:ThreeDGRT",
    "conda": _conda_spec,
    "metadata": {
        "name": "3DGRT",
        "description": "The authors accelerate 3DGS rendering by replacing rasterization with a BVH-based GPU ray-tracing pipeline that wraps each particle in proxy meshes for fast ray–triangle tests, maintains raster-like speed while enabling ray-tracing effects and arbitrary cameras.",
        "long_description": _long_description_3dgrt,
        "paper_title": "3D Gaussian Ray Tracing: Fast Tracing of Particle Scenes",
        "paper_authors": ["Nicolas Moenne-Loccoz", "Ashkan Mirzaei", "Or Perel", "Riccardo de Lutio", "Janick Martinez Esturo", "Gavriel State", "Sanja Fidler", "Nicholas Sharp", "Zan Gojcic"],
        "paper_link": "https://arxiv.org/pdf/2407.07090.pdf",
        "paper_results": _3dgrt_paper_results,
        "paper_venue": "SIGGRAPH Asia 2024",
        "link": "https://gaussiantracer.github.io/",
        "licenses": [{"name": "Apache 2.0", "url": "https://raw.githubusercontent.com/nv-tlabs/3dgrut/refs/heads/main/LICENSE"}],
    },
    "presets": {
        "blender": {
            "@apply": [{"dataset": "blender"}],
            "config": "paper/3dgrt/nerf_synthetic_ours.yaml",
        },
    },
    "implementation_status": {}
})
