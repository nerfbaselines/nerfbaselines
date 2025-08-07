import os
from nerfbaselines import register

GIT_REPOSITORY = "https://github.com/ForMyCat/SparseGS.git"
GIT_COMMIT = "95e7aef29c5562400d3b2b38cc7e90436a432b7c"
METHOD_ID = "sparsegs"

register({
    "id": METHOD_ID,
    "method_class": ".sparsegs:SparseGS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.10",
        "install_script": f"""git clone {GIT_REPOSITORY} sparsegs
cd sparsegs
git checkout {GIT_COMMIT}
git submodule update --init --recursive

# Fix broken sparsegs repo (missing code)
(
    git clone https://github.com/g-truc/glm.git submodules/diff-gaussian-rasterization-softmax/third_party/glm
    cd submodules/diff-gaussian-rasterization-softmax/third_party/glm
    git checkout 5c46b9c
    git submodule update --init --recursive
)

# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
conda install -y mkl==2023.1.0 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
conda install -y cuda -c nvidia/label/cuda-12.1.1

# Install PyTorch and other Python dependencies
pip install torch==2.2.1 torchvision==0.17.1 'numpy<2.0.0' --index-url https://download.pytorch.org/whl/cu121
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
        Pillow==11.0.0 \
        matplotlib==3.9.4 \
        tensorboard==2.18.0 \
        scipy==1.13.1 \
        pytest==8.3.4 \
        icecream==2.1.5 \
        diptest==0.10.0 \
        safetensors==0.4.1 \
        accelerate==0.26.0 \
        transformers==4.47.1 \
        diffusers==0.34.0 \
        torchmetrics==1.8.0 \
        submodules/diff-gaussian-rasterization-softmax \
        submodules/simple-knn \
        gdown \
        --no-build-isolation


echo "Downloading pretrained models..."
gdown '1BpUIkPQlY-Hbsg6jf4i4H-akpta_LiPZ' -O 'BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/latest_net_G.pth'
gdown '1-k4RVaIy4366CN7I7WWUt3RQ28G7kqml' -O 'BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/res101.pth'

conda develop .

if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {{}} +
# Replace all libs under $CONDA_PREFIX/lib with symlinks to pkgs/cuda-toolkit/targets/x86_64-linux/lib
for lib in "$CONDA_PREFIX"/lib/*.so*; do 
    if [ ! -f "$lib" ] || [ -L "$lib" ]; then continue; fi;
    lib="${{lib%.so*}}.so";libname=$(basename "$lib");
    tgt="$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/$libname"
    if [ -f "$tgt" ]; then echo "Deleting $lib"; rm "$lib"*; for tgtlib in "$tgt"*; do ln -s "$tgtlib" "$(dirname "$lib")"; done; fi;
done
fi
""",
    },
    "metadata": {
        "name": "SparseGS",
        "description": "SparseGS augments 3D Gaussian Splatting with depth-based priors, tailored depth rendering, a floater-pruning heuristic, and Unseen Viewpoint Regularization, letting it overcome “floaters” and background collapse when training views are scarce. Tested on Mip-NeRF360, LLFF, and DTU, it still trains quickly and renders in real time while reconstructing unbounded or forward-facing scenes from as few as 12 and 3 input images, respectively.",
        "paper_title": "SparseGS: Real-Time 360° Sparse View Synthesis using Gaussian Splatting",
        "paper_authors": "Haolin Xiong, Sairisheek Muttukuru, Rishi Upadhyay, Pradyumna Chari, Achuta Kadambi".split(", "),
        "paper_venue": "3DV 2025",
        "paper_link": "https://arxiv.org/pdf/2312.00206.pdf",
        "link": "https://formycat.github.io/SparseGS-Real-Time-360-Sparse-View-Synthesis-using-Gaussian-Splatting/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/ForMyCat/SparseGS/refs/heads/master/LICENSE.md"}],
    },
    "presets": {},
    "implementation_status": {}
})
