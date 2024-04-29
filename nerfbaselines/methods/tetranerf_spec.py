import os
from ..registry import MethodSpec, register


TetraNeRFSpec: MethodSpec = {
    "method": ".tetranerf:TetraNeRF",
    "docker": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "image": "kulhanek/tetra-nerf:latest",
        "python_path": "python3",
        "home_path": "/home/user",
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
    })

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
    })
