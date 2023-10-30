
from ..backends import DockerMethod


def build_tetranerf(config_name):
    return DockerMethod(image="tetra-nerf:latest", build_code=f"""
from tetranerf.nerfstudio.registration import {config_name}
trainer = {config_name}()

class TetraNeRF:
    def train_iteration(self, *args, **kwargs):
""")

TetraNeRF = build_tetranerf("tetranerf")
TetraNeRFOriginal = build_tetranerf("tetranerf_original")