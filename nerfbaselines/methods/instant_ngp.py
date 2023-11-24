from ..backends import DockerMethod
from ..registry import MethodSpec, LazyMethod


class InstantNGP(LazyMethod["._impl.instant_ngp", "InstantNGP"]):
    pass


InstantNGPSpec = MethodSpec(method=InstantNGP, docker=DockerMethod.wrap(InstantNGP, image="kulhanek/ingp:latest", python_path="python3", home_path="/root"))
InstantNGPSpec.register("instant-ngp")
