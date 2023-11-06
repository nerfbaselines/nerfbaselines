import os
from ..backends import DockerMethod
from .nerfstudio import NerfStudio
from ..registry import MethodSpec


TetraNeRFSpec = MethodSpec(
    method=NerfStudio,
    docker=DockerMethod.wrap(
        NerfStudio,
        image="tetra-nerf:latest",
        python_path="python3",
        home_path="/home/user",
        mounts=[(os.path.expanduser("~/.cache/torch"), "/home/user/.cache/torch")]))

TetraNeRFSpec.register("tetranerf", method_name="tetranerf-original")
TetraNeRFSpec.register("tetranerf:latest", method_name="tetranerf")
