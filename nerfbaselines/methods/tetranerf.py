from ..backends import DockerMethod
from .nerfstudio import NerfStudio
from ..registry import MethodSpec


TetraNeRFSpec = MethodSpec(method=NerfStudio, docker=DockerMethod.wrap(NerfStudio, image="kulhanek/tetra-nerf:latest", python_path="python3", home_path="/home/user"))

TetraNeRFSpec.register("tetra-nerf", nerfstudio_name="tetra-nerf-original", require_points3D=True)
TetraNeRFSpec.register("tetra-nerf:latest", nerfstudio_name="tetra-nerf", require_points3D=True)
