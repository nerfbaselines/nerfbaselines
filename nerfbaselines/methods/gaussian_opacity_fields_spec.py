import os
from ..registry import register, MethodSpec


GaussianOpacityFieldsSpec: MethodSpec = {
    "method": ".gaussian_opacity_fields:GaussianOpacityFields",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.8",
        "install_script": """# Install mip-splatting
git clone https://github.com/autonomousvision/gaussian-opacity-fields.git
cd gaussian-opacity-fields
git checkout 98d0858974437d329720727ee34e42e388425112
# Remove unsupported (and unnecessary) open3d dependency
sed -i '/import open3d as o3d/d' train.py

conda install -y conda-build
conda develop .

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
conda install -c conda-forge -y nodejs==20.9.0

pip install -r requirements.txt
pip install -U pip 'setuptools<70.0.0'
pip install lpips==0.1.4

(
cd submodules/tetra-triangulation
conda install -y cmake
conda install -y conda-forge::gmp
conda install -y conda-forge::cgal
cmake .
make
pip install -e .
) || exit 1

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
""",
    },
    "metadata": {
        "name": "Gaussian Opacity Fields",
        "description": """Improved Mip-Splatting with better geometry.""",
        "paper_title": "Gaussian Opacity Fields: Efficient and Compact Surface Reconstruction in Unbounded Scenes",
        "paper_authors": ["Zehao Yu", "Torsten Sattler", "Andreas Geiger"],
        "paper_link": "https://arxiv.org/pdf/2404.10772.pdf",
        "link": "https://niujinshuchong.github.io/gaussian-opacity-fields/",
    },
    "dataset_overrides": {
        "blender": { "white_background": True },
        "dtu": { "use_decoupled_appearance": True, "lambda_distortion": 100 },
        "tanksandtemples": { "use_decoupled_appearance": True },
        "phototourism": { "use_decoupled_appearance": True },
    },
}


register(GaussianOpacityFieldsSpec, name="gaussian-opacity-fields")
