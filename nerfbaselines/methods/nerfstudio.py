from typing import Optional
from ..registry import CondaMethod, DockerMethod, ApptainerMethod, LazyMethod, MethodSpec, DEFAULT_DOCKER_IMAGE


class NerfStudio(LazyMethod["._impl.nerfstudio", "NerfStudio"]):
    nerfstudio_name: Optional[str] = None
    require_points3D: bool = False


NerfStudioConda = CondaMethod.wrap(
    NerfStudio,
    conda_name="nerfstudio",
    python_version="3.10",
    install_script="""
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
conda install -y -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio==0.3.4
""",
)


NerfStudioSpec = MethodSpec(
    method=NerfStudio,
    conda=NerfStudioConda,
    docker=DockerMethod.wrap(NerfStudioConda, image=DEFAULT_DOCKER_IMAGE),
    apptainer=ApptainerMethod.wrap(NerfStudioConda, image="docker://" + DEFAULT_DOCKER_IMAGE),
    metadata={
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
)

# Register supported methods
NerfStudioSpec.register(
    "nerfacto",
    nerfstudio_name="nerfacto",
    metadata={
        "name": "NerfStudio",
        "description": """NerfStudio (Nerfacto) is a method based on Instant-NGP which combines several improvements from different papers to achieve good quality on real-world scenes captured under normal conditions. It is fast to train (12 min) and render speed is ~1 FPS.""",
    },
)
NerfStudioSpec.register(
    "nerfacto:big",
    nerfstudio_name="nerfacto-big",
    metadata={
        "name": "NerfStudio (Nerfacto-big)",
        "description": """Larger setup of Nerfacto model family. It has larger hashgrid and MLPs. It is slower to train and render, but it provides better quality.""",
    },
)
NerfStudioSpec.register(
    "nerfacto:huge",
    nerfstudio_name="nerfacto-huge",
    metadata={
        "name": "NerfStudio (Nerfacto-huge)",
        "description": """Largest setup of Nerfacto model family. It has larger hashgrid and MLPs. It is slower to train and render, but it provides better quality.""",
    },
)
