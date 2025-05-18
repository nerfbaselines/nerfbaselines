import os
from nerfbaselines import register, MethodSpec


note = r"""

In our implementation, we picked sensible defaults for new datasets as default which corresponds to the following:
    --cap_max=1300000 \
    --nu_degree=100 \
    --C_burnin=5e5 \
    --burnin_iterations=7000 \
    --iterations=40000
However, at least --cap_max have to be tuned for each dataset.
"""

StudentSplattingScoopingSpec: MethodSpec = {
    "id": "student-splatting-scooping",
    "method_class": ".student_splatting_scooping:StudentSplattingScooping",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/realcrane/student-splating-scooping.git student-splatting-scooping
cd student-splatting-scooping
git checkout bf6d26f0aa54663d9084c24d6121eab7705b9fdd
git submodule update --init --recursive

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
conda install -y pytorch3d==0.7.7 -c pytorch3d
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'

pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
pip install plyfile==0.8.1 \
        mediapy==1.1.2 \
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
        scipy==1.13.1 \
        einops==0.8.0 \
        laspy==2.5.4 \
        jaxtyping==0.2.34 \
        'pytest<=8.3.4' \
        submodules/diff-t-rasterization \
        submodules/simple-knn \
        --no-build-isolation
conda develop .

if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -rf {} +
# Replace all libs under $CONDA_PREFIX/lib with symlinks to pkgs/cuda-toolkit/targets/x86_64-linux/lib
for lib in "$CONDA_PREFIX"/lib/*.so*; do 
    if [ ! -f "$lib" ] || [ -L "$lib" ]; then continue; fi;
    lib="${lib%.so*}.so";libname=$(basename "$lib");
    tgt="$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/$libname"
    if [ -f "$tgt" ]; then echo "Deleting $lib"; rm -rf "$lib"*; for tgtlib in "$tgt"*; do ln -s "$tgtlib" "$(dirname "$lib")"; done; fi;
done
fi
""",
    },
    "metadata": {
        "name": "SSS",
        "description": """Planar-based Gaussian Splatting Reconstruction representation for efficient and high-fidelity surface reconstruction from multi-view RGB images without any geometric prior (depth or normal from pre-trained model).""",
        "paper_title": "PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction",
        "paper_authors": ["Danpeng Chen", "Hai Li", "Weicai Ye", "Yifan Wang", "Weijian Xie", "Shangjin Zhai", "Nan Wang", "Haomin Liu", "Hujun Bao", "Guofeng Zhang"],
        "paper_link": "https://zju3dv.github.io/pgsr/paper/pgsr.pdf",
        "link": "https://zju3dv.github.io/pgsr/",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/zju3dv/PGSR/refs/heads/main/LICENSE.md"}],
    },
    "presets": {
        "mipnerf360/bicycle": { 
           "@apply": [ {"dataset": "mipnerf360", "scene": "bicycle"}],
            "cap_max": 3000000,
            "C_burnin": 5e8,
            "burnin_iterations": 15000,
            "iterations": 30000,
        },
        "mipnerf360/bonsai": { 
           "@apply": [{"dataset": "mipnerf360", "scene": "bonsai"}],
            "cap_max": 1300000,
        },
        "mipnerf360/counter": { 
           "@apply": [ {"dataset": "mipnerf360", "scene": "counter"}],
            "cap_max": 1200000,
        },
        "mipnerf360/garden": { 
           "@apply": [ {"dataset": "mipnerf360", "scene": "garden"}],
            "cap_max": 3000000,
        },
        "mipnerf360/stump": {
           "@apply": [ {"dataset": "mipnerf360", "scene": "stump"}],
           "cap_max": 3000000,
           "burnin_iterations": 15000,
           "iterations": 20000
        },
        "tanksandtemples/train": {
           "@apply": [ {"dataset": "tanksandtemples", "scene": "train"}],
            "cap_max": 1100000,
        },
        "tanksandtemples/truck": {
           "@apply": [ {"dataset": "tanksandtemples", "scene": "truck"}],
            "cap_max": 1100000,
        },
    },
}

register(StudentSplattingScoopingSpec)
