import os
from nerfbaselines import register, MethodSpec


ColmapMVSSpec: MethodSpec = {
    "method_class": ".colmap:ColmapMVS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": """# Install COLMAP from conda-forge.
conda install -y colmap=3.9.1 -c conda-forge
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y ffmpeg=7.1.0
pip install \
    pyrender==0.1.45 \
    trimesh==4.4.8 \
    pyopengl-accelerate==3.1.7 \
    tqdm==4.67.1 \
    plyfile==1.1 \
    scikit-image==0.25.0 \
    opencv-python-headless==4.10.0.84 \
    importlib_metadata==8.5.0 \
    typing_extensions==4.12.2 \
    mediapy==1.1.2 \
    wandb==0.19.1 \
    click==8.1.8 \
    Pillow==11.1.0 \
    tensorboard==2.18.0 \
    matplotlib==3.9.4 \
    pytest==8.3.4 \
    scipy==1.13.1
pip install torch==2.3.1 \
    torchvision==0.18.1 \
    'numpy<2.0.0' --index-url https://download.pytorch.org/whl/cu118
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
