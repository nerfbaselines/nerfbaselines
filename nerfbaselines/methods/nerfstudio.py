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
)

# Register supported methods
NerfStudioSpec.register("nerfacto", nerfstudio_name="nerfacto")
NerfStudioSpec.register("nerfacto:big", nerfstudio_name="nerfacto-big")
NerfStudioSpec.register("nerfacto:huge", nerfstudio_name="nerfacto-huge")
