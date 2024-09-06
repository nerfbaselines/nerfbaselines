import os
from nerfbaselines import register, MethodSpec


ColmapMVSSpec: MethodSpec = {
    "method_class": ".colmap:ColmapMVS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": """# Install COLMAP from conda-forge.
conda install -y colmap=3.9.1 -c conda-forge
pip install opencv-python-headless pyrender==0.1.45 trimesh==4.4.8 pyopengl-accelerate==3.1.7
# For metric computation
pip install torch==2.3.1 torchvision==0.18.1 'numpy<2.0.0' --index-url https://download.pytorch.org/whl/cu118
""",
    },
    "metadata": {
        "name": "COLMAP",
        "description": """COLMAP Multi-View Stereo (MVS) is a general-purpose, end-to-end image-based 3D reconstruction pipeline.
It uses the point cloud if available, otherwise it runs a sparse reconstruction to obtained.
The reconstruction consists of a stereo matching step, followed by a multi-view stereo step to obtain a dense point cloud.
Finally, either Delaunay or Poisson meshing is used to obtain a mesh from the point cloud.
""",
        "paper_title": "Pixelwise View Selection for Unstructured Multi-View Stereo",
        "paper_authors": ["Johannes Lutz Schőnberger", "Enliang Zheng", "Marc Pollefeys", "Jan-Michael Frahm"],
        "paper_link": "https://demuc.de/papers/schoenberger2016mvs.pdf",
        "link": "https://colmap.github.io/",
        "licenses": [{"name": "BSD","url": "https://colmap.github.io/license.html"}],
    },
    "presets": {
        "blender": {
            "PoissonMeshing.trim": 5,
            "@apply": [{"dataset": "blender"}],
        }
    },
    "id": "colmap",
    "implementation_status": {
        "mipnerf360": "working",
        "blender": "working",
        "tanksandtemples": "working",
        "nerfstudio": "working",
    }
}

register(ColmapMVSSpec)
