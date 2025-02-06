import os
from nerfbaselines import register, MethodSpec


F2NeRFSpec: MethodSpec = {
    "method_class": "f2nerf:F2NeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.11",
        "install_script": """
git clone https://github.com/Totoro97/f2-nerf.git f2nerf
cd f2nerf
git checkout 98f0daacb80e76724eb91519742c30fb35d0f72d
git submodule update --init --recursive

# Install dependencies
conda install -y cuda-toolkit -c "nvidia/label/cuda-11.7.1"
conda install -y  \
    make=4.3 cmake=3.28.3 \
    zlib=1.2 gcc_linux-64=11 gxx_linux-64=11 binutils=2.40 \
    -c conda-forge

# Build
cmake . -B build
cmake --build build --target main --config RelWithDebInfo -j

# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y ffmpeg=7.1.0
pip install \
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
pip install torch==1.13.1+cu117 \
        torchvision==0.14.1+cu117 \
        'numpy<2.0.0' \
        --extra-index-url https://download.pytorch.org/whl/cu117
""".strip(),
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
    "id": "f2nerf",
}

register(F2NeRFSpec)
