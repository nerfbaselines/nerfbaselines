import os
from ..registry import MethodSpec, register


NerfStudioSpec: MethodSpec = {
    "method": ".nerfstudio:NerfStudio",
    "kwargs": {
        "nerfstudio_name": None,
        "require_points3D": False,
    },
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.10",
        "install_script": r"""
conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
conda install -y pytorch==2.3.0 torchvision==0.18.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
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
pip install -e .
""",
    },
    "metadata": {
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
        "link": "https://docs.nerf.studio/",
    },
}

# Register supported methods
register(
    NerfStudioSpec,
    name="nerfacto",
    kwargs={
        "nerfstudio_name": "nerfacto",
    },
    metadata={
        "name": "NerfStudio",
        "description": """NerfStudio (Nerfacto) is a method based on Instant-NGP which combines several improvements from different papers to achieve good quality on real-world scenes captured under normal conditions. It is fast to train (12 min) and render speed is ~1 FPS.""",
    },
)
register(
    NerfStudioSpec,
    name="nerfacto:big",
    kwargs={
        "nerfstudio_name": "nerfacto-big",
    },
    metadata={
        "name": "NerfStudio (Nerfacto-big)",
        "description": """Larger setup of Nerfacto model family. It has larger hashgrid and MLPs. It is slower to train and render, but it provides better quality.""",
    },
)
register(
    NerfStudioSpec,
    name="nerfacto:huge",
    kwargs={
        "nerfstudio_name": "nerfacto-huge",
    },
    metadata={
        "name": "NerfStudio (Nerfacto-huge)",
        "description": """Largest setup of Nerfacto model family. It has larger hashgrid and MLPs. It is slower to train and render, but it provides better quality.""",
    },
)
