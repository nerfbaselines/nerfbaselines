import os
from nerfbaselines import register, MethodSpec


_paper_results = {
    # blender scenes: chair drums ficus hotdog lego materials mic ship 
    # blender PSNRs: 35.05 25.01 33.31 36.16 34.75 29.30 35.49 31.13
    # blender SSIMs: 0.990 0.947 0.989 0.989 0.987 0.968 0.993 0.994
    "blender/chair": {"psnr": 35.05, "ssim": 0.990},
    "blender/drums": {"psnr": 25.01, "ssim": 0.947},
    "blender/ficus": {"psnr": 33.31, "ssim": 0.989},
    "blender/hotdog": {"psnr": 36.16, "ssim": 0.989},
    "blender/lego": {"psnr": 34.75, "ssim": 0.987},
    "blender/materials": {"psnr": 29.30, "ssim": 0.968},
    "blender/mic": {"psnr": 35.49, "ssim": 0.993},
    "blender/ship": {"psnr": 31.13, "ssim": 0.994},

    # 360 scenes: bicycle flowers garden stump treehill room counter kitchen bonsai
    # 360 PSNRs: 23.53 20.36 26.15 24.42 21.41 32.02 28.02 29.66 31.13
    # 360 SSIMs: 0.614 0.470 0.775 0.613 0.456 0.894 0.850 0.877 0.905
    # 360 LPIPS: 0.271 0.378 0.136 0.274 0.429 0.104 0.127 0.098 0.084
    "mipnerf360/bicycle": {"psnr": 23.53, "ssim": 0.614, "lpips_vgg": 0.271},
    "mipnerf360/flowers": {"psnr": 20.36, "ssim": 0.470, "lpips_vgg": 0.378},
    "mipnerf360/garden": {"psnr": 26.15, "ssim": 0.775, "lpips_vgg": 0.136},
    "mipnerf360/stump": {"psnr": 24.42, "ssim": 0.613, "lpips_vgg": 0.274},
    "mipnerf360/treehill": {"psnr": 21.41, "ssim": 0.456, "lpips_vgg": 0.429},
    "mipnerf360/room": {"psnr": 32.02, "ssim": 0.894, "lpips_vgg": 0.104},
    "mipnerf360/counter": {"psnr": 28.02, "ssim": 0.850, "lpips_vgg": 0.127},
    "mipnerf360/kitchen": {"psnr": 29.66, "ssim": 0.877, "lpips_vgg": 0.098},
    "mipnerf360/bonsai": {"psnr": 31.13, "ssim": 0.905, "lpips_vgg": 0.084},
}


TetraNeRFSpec: MethodSpec = {
    "method_class": ".tetranerf:TetraNeRF",
    "docker": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "image": "kulhanek/tetra-nerf:latest",
        "python_path": "python3",
        "home_path": "/home/user",
        "build_script": r"""
# kulhanek/tetra-nerf:latest includes:
#     torch

# Install ffmpeg
RUN sudo apt-get update && \
    sudo apt-get install -y --no-install-recommends ffmpeg && \
    sudo apt-get clean && \
    sudo rm -rf /var/lib/apt/lists/*

# Install pip dependencies
RUN pip install --upgrade pip && \
    pip install \
        'opencv-python-headless<=4.10.0.84' \
        "numpy<2.0.0" \
        "plyfile==0.8.1" \
        "mediapy<=1.1.2" \
        "scikit-image<=0.21.0" \
        "tqdm<=4.66.2" \
        "importlib_metadata==8.5.0" \
        "typing_extensions==4.12.2" \
        "wandb<=0.19.1" \
        "click<=8.1.8" \
        "Pillow<=11.1.0" \
        "matplotlib<=3.9.4" \
        'importlib-resources<=6.5.2' \
        'pytest<=8.3.4' \
        "tensorboard<=2.18.0" \
        "scipy<=1.13.1" && \
        pip cache purge

# Ensure nerfbaselines is executable
RUN echo -e '#!/usr/bin/env python3\nfrom nerfbaselines.__main__ import main;main()' > /home/user/.local/bin/nerfbaselines && \
    chmod +x /home/user/.local/bin/nerfbaselines
""",
    },
    "metadata": {
        "name": "Tetra-NeRF",
        "paper_title": "Tetra-NeRF: Representing Neural Radiance Fields Using Tetrahedra",
        "paper_authors": ["Jonas Kulhanek", "Torsten Sattler"],
        "paper_link": "https://arxiv.org/pdf/2304.09987.pdf",
        "paper_results": _paper_results,
        "link": "https://jkulhanek.com/tetra-nerf",
        "description": """Tetra-NeRF is a method that represents the scene as tetrahedral mesh obtained using Delaunay tetrahedralization. The input point cloud has to be provided (for COLMAP datasets the point cloud is automatically extracted). This is the official implementation
    from the paper.""",
        "licenses": [{"name": "MIT", "url":"https://raw.githubusercontent.com/jkulhanek/tetra-nerf/master/LICENSE"}],
    },
    "presets": {
        "blender": { "@apply": [{"dataset": "blender"}], "pipeline.datamanager.dataparser": "blender-data", },
        "latest": {
            "@description": "This variant of Tetra-NeRF uses biased sampling to speed-up training and rendering. It trains/renders almost twice as fast without sacrificing quality. WARNING: this variant is not the same as the one used in the Tetra-NeRF paper.",
            "method": "tetra-nerf",
        },
    },
    "id": "tetra-nerf",
    "implementation_status": {
        "mipnerf360": "working-not-reproducing",
        "blender": "working-not-reproducing",
    }
}

register(TetraNeRFSpec)
