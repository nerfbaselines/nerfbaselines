import os
from nerfbaselines import register, MethodSpec


_PHOTOTOURISM_DISCREPANCY_NOTE = """The original paper reports metrics for test images where the appearance embedding is estimated from the full test image, not just the left half as in the official evaluation protocol. The reported numbers are computed using the official evaluation protocol and are, therefore, lower than the numbers reported in the paper."""
_paper_results = {
    "phototourism/trevi-fountain": { "note": _PHOTOTOURISM_DISCREPANCY_NOTE, "psnr": 22.91, "ssim": 0.8014, "lpips": 0.1563 },
    "phototourism/sacre-coeur": { "note": _PHOTOTOURISM_DISCREPANCY_NOTE, "psnr": 23.24, "ssim": 0.8632, "lpips": 0.1300 },
    "phototourism/brandenburg-gate": { "note": _PHOTOTOURISM_DISCREPANCY_NOTE, "psnr": 27.96, "ssim": 0.9319, "lpips": 0.0862 },
}

GaussianSplattingWildSpec: MethodSpec = {
    "method_class": ".gaussian_splatting_wild:GaussianSplattingWild",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/EastbeanZhang/Gaussian-Wild.git
cd Gaussian-Wild
git checkout 79d9e6855298a2632b530644e52d1829c6356b08
git submodule update --init --recursive

# Fix the code, replace line 80 in scene/dataset_readers.py 
# from "intr = cam_intrinsics[extr.id]" to "intr = cam_intrinsics[extr.camera_id]"
sed -i '80s/.*/        intr = cam_intrinsics[extr.camera_id]/' scene/dataset_readers.py

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
pip install \
        pandas==2.2.2 \
        kornia==0.7.3 \
        plyfile==0.8.1 \
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
        scipy==1.13.1 \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        --no-build-isolation

conda install -c conda-forge -y nodejs==20.9.0
conda develop .
pip install lpips==0.1.4 importlib_metadata typing_extensions
if ! python -c 'import cv2'; then pip install opencv-python-headless; fi

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
        "name": "GS-W",
        "description": "Official GS-W implementation - 3DGS modified to handle appearance changes and transient objects. A reference view used to provide appearance conditioning. Note, that the method uses huge appearance embeddings (per-Gaussian) and appearance modeling has a large memory footprint.",
        "note": "The appearance embeddings are stored per Gaussian and therefore the memory consumption is huge when interpolating between two appearance embeddings. Furthermore, the training dataset is required during inference to compute the appearance embeddings and therefore is only enabled for default datasets.",
        "paper_title": "Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections",
        "paper_authors": ["Dongbin Zhang", "Chuming Wang", "Weitao Wang", "Peihao Li", "Minghan Qin", "Haoqian Wang"],
        "paper_results": _paper_results,
        "paper_link": "https://arxiv.org/pdf/2403.15704.pdf",
        "link": "https://eastbeanzhang.github.io/GS-W/",
        "licenses": [{"name": "unknown"}],
    },
    "id": "gaussian-splatting-wild",
}

register(GaussianSplattingWildSpec)
