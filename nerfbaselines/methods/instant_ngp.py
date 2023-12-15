from ..backends import DockerMethod
from ..registry import MethodSpec, LazyMethod


class InstantNGP(LazyMethod["._impl.instant_ngp", "InstantNGP"]):
    pass


InstantNGPSpec = MethodSpec(method=InstantNGP, docker=DockerMethod.wrap(InstantNGP, image="kulhanek/ingp:latest", python_path="python3", home_path="/root"))
InstantNGPSpec.register(
    "instant-ngp",
    metadata={
        "name": "Instant NGP",
        "description": """Instant-NGP is a method that uses hash-grid and a shallow MLP to accelerate training and rendering.
This method trains very fast (~6 min) and renders also fast ~3 FPS.""",
        "paper_title": "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding",
        "paper_authors": ["Thomas Müller", "Alex Evans", "Christoph Schied", "Alexander Keller"],
        "paper_link": "https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf",
        "link": "https://nvlabs.github.io/instant-ngp/",
    },
)
