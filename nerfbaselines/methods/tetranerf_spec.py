import os
from ..registry import MethodSpec, register


paper_results = {
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
    "method": ".tetranerf:TetraNeRF",
    "docker": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "image": "kulhanek/tetra-nerf:latest",
        "python_path": "python3",
        "home_path": "/home/user",
        "build_script": "",  # Force build
    },
    "kwargs": {
        "require_points3D": True,
        "nerfstudio_name": None,
    },
    "metadata": {
        "name": "Tetra-NeRF",
        "paper_title": "Tetra-NeRF: Representing Neural Radiance Fields Using Tetrahedra",
        "paper_authors": ["Jonas Kulhanek", "Torsten Sattler"],
        "paper_link": "https://arxiv.org/pdf/2304.09987.pdf",
        "link": "https://jkulhanek.com/tetra-nerf",
        "description": """Tetra-NeRF is a method that represents the scene as tetrahedral mesh obtained using Delaunay tetrahedralization. The input point cloud has to be provided (for COLMAP datasets the point cloud is automatically extracted). This is the official implementation
    from the paper.""",
    },
}

register(
    TetraNeRFSpec, 
    name="tetra-nerf", 
    kwargs={
        "nerfstudio_name": "tetra-nerf-original", 
        "require_points3D": True
    },
    metadata={
        "paper_results": paper_results,
    },
    dataset_overrides={
        "blender": { "pipeline.datamanager.dataparser": "blender-data", }
    }
)

register(
    TetraNeRFSpec,
    name="tetra-nerf:latest",
    kwargs={
        "nerfstudio_name": "tetra-nerf",
        "require_points3D": True,
    },
    metadata={
        "name": "Tetra-NeRF (latest)",
        "description": """This variant of Tetra-NeRF uses biased sampling to speed-up training and rendering. It trains/renders almost twice as fast without sacrificing quality. WARNING: this variant is not the same as the one used in the Tetra-NeRF paper.""",
    },
    dataset_overrides={
        "blender": { "pipeline.datamanager.dataparser": "blender-data", }
    }
)
