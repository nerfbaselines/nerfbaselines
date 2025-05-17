import os
from nerfbaselines import register, MethodSpec


_nerfacto_paper_results = {
    # 360 scenes: bicycle garden stump room counter kitchen bonsai
    # 360 PSNRs (70k/5k): 24.08 / 22.36 26.47 / 24.05 24.78 / 18.94 30.89 / 29.36 27.20 / 25.92 30.29 / 28.17 32.16 / 28.98
    # 360 SSIMs (70k/5k): 0.599 / 0.474 0.774 / 0.617 0.662 / 0.364 0.896 / 0.866 0.843 / 0.776 0.890 / 0.838 0.933 / 0.880
    # 360 LPIPS (70k/5k): 0.422 / 0.551 0.235 / 0.385 0.380 / 0.669 0.296 / 0.302 0.314 / 0.346 0.190 / 0.223 0.197 / 0.252
    "mipnerf360/bicycle": {"psnr": 24.08, "ssim": 0.599, "lpips_vgg": 0.422},
    "mipnerf360/garden": {"psnr": 26.47, "ssim": 0.774, "lpips_vgg": 0.235},
    "mipnerf360/stump": {"psnr": 24.78, "ssim": 0.662, "lpips_vgg": 0.380},
    "mipnerf360/room": {"psnr": 30.89, "ssim": 0.896, "lpips_vgg": 0.296},
    "mipnerf360/counter": {"psnr": 27.20, "ssim": 0.843, "lpips_vgg": 0.314},
    "mipnerf360/kitchen": {"psnr": 30.29, "ssim": 0.890, "lpips_vgg": 0.190},
    "mipnerf360/bonsai": {"psnr": 32.16, "ssim": 0.933, "lpips_vgg": 0.197},
}


NerfStudioSpec: MethodSpec = {
    "method_class": ".nerfstudio:NerfStudio",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.10",
        "install_script": r"""
conda install -y cuda-toolkit -c nvidia/label/cuda-11.8.0
if [[ "$(gcc -v 2>&1)" != *"gcc version 11"* ]]; then
  conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
  ln -s "$CC" "$CONDA_PREFIX/bin/gcc"
  ln -s "$CXX" "$CONDA_PREFIX/bin/g++"
  export CPATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPATH"
fi
pip install torch==2.3.0 torchvision==0.18.0 'numpy<2.0.0' --index-url https://download.pytorch.org/whl/cu118
LIBRARY_PATH="$CONDA_PREFIX/lib/stubs" pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
if ! pip install open3d>=0.16.0; then
    wget -O open3d-0.18.0-py3-none-any.whl https://files.pythonhosted.org/packages/5c/ba/a4c5986951344f804b5cbd86f0a87d9ea5969e8d13f1e8913e2d8276e0d8/open3d-0.18.0-cp311-cp311-manylinux_2_27_x86_64.whl;
    pip install open3d-0.18.0-py3-none-any.whl;
    rm -rf open3d-0.18.0-py3-none-any.whl;
fi
pip install nerfstudio==0.3.4
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
git checkout 3a90cb529f893fbf89625a915a53a7a71b97a575
pip install \
    plyfile==1.1 \
    importlib_metadata==8.5.0 \
    typing_extensions==4.12.2 \
    wandb==0.19.1 \
    click==8.1.8 \
    Pillow==11.1.0 \
    tensorboard==2.18.0 \
    matplotlib==3.9.4 \
    'tqdm<=4.67.1' \
    'mediapy<=1.1.2' \
    'opencv-python-headless<=4.10.0.84' \
    'pytest<=8.3.4' \
    scipy==1.13.1 \
    -e .

# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'

if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {} +
fi
""",
    },
    "metadata": {
        "name": "NerfStudio",
        "description": """NerfStudio (Nerfacto) is a method based on Instant-NGP which combines several improvements from different papers to achieve good quality on real-world scenes captured under normal conditions. It is fast to train (12 min) and render speed is ~1 FPS.""",
        "paper_title": "Nerfstudio: A Modular Framework for Neural Radiance Field Development",
        "paper_authors": [
            "Matthew Tancik",
            "Ethan Weber",
            "Evonne Ng",
            "Ruilong Li",
            "Brent Yi",
            "Justin Kerr",
            "Terrance Wang",
            "Alexander Kristoffersen",
            "Jake Austin",
            "Kamyar Salahi",
            "Abhik Ahuja",
            "David McAllister",
            "Angjoo Kanazawa",
        ],
        "paper_link": "https://arxiv.org/pdf/2302.04264.pdf",
        "paper_results": _nerfacto_paper_results,
        "link": "https://docs.nerf.studio/",
        "licenses": [{"name": "Apache 2.0", "url": "https://raw.githubusercontent.com/nerfstudio-project/nerfstudio/main/LICENSE"}],
    },
    "presets": {
        "blender": {
            "@apply": [{"dataset": "blender"}],
            "pipeline.datamanager.dataparser": "blender-data",
            "pipeline.model.near_plane": 2.0,
            "pipeline.model.far_plane": 6.0,
            "pipeline.model.use_appearance_embedding": False,
            "pipeline.model.background_color": "white",
            "pipeline.model.proposal_initial_sampler": "uniform",
            "pipeline.model.distortion_loss_mult": 0.0,
            "pipeline.model.disable_scene_contraction": True,
            "pipeline.model.average_init_density": 1.0,
            "pipeline.model.camera_optimizer.mode": "off",
        },    
        "mipnerf360": { 
            "@apply": [{"dataset": "mipnerf360"}],
            "pipeline.model.use_appearance_embedding": False,
            "pipeline.model.camera_optimizer.mode": "off",
            "max_num_iterations": 70000,
        },
        "tanksandtemples": {
            "@apply": [{"dataset": "tanksandtemples"}],
            "pipeline.model.use_appearance_embedding": False,
            "pipeline.model.camera_optimizer.mode": "off",
            "max_num_iterations": 70000,
        },
        "big": {
            "@description": "Larger setup of Nerfacto model family. It has larger hashgrid and MLPs. It is slower to train and render, but it provides better quality.",
            "method": "nerfacto-big",
            "max_num_iterations": 100_000,
        },
        "huge": {
            "@description": "Largest setup of Nerfacto model family. It has larger hashgrid and MLPs. It is slower to train and render, but it provides better quality.",
            "method": "nerfacto-huge",
            "max_num_iterations": 100_000,
        },
    },
    "id": "nerfacto",
    "implementation_status": {
        "blender": "working",
        "mipnerf360": "reproducing",
        "tanksandtemples": "working",
    }
}

# Register nerfacto
register(NerfStudioSpec)
