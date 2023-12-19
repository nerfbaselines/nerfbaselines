from ..backends import CondaMethod
from ..registry import MethodSpec, LazyMethod


class GaussianSplatting(LazyMethod["._impl.gaussian_splatting", "GaussianSplatting"]):
    pass


MethodSpec(
    GaussianSplatting,
    conda=CondaMethod.wrap(
        GaussianSplatting,
        conda_name="gaussian-splatting",
        install_script="""git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
git checkout 2eee0e26d2d5fd00ec462df47752223952f6bf4e
conda env update --file environment.yml --prune --prefix "$CONDA_PREFIX"
conda install -y conda-build
conda develop .
""",
    ),
).register(
    "gaussian-splatting",
    metadata={
        "name": "Gaussian Splatting",
        "description": """Official Gaussian Splatting implementation extended to support distorted camera models. It is fast to train (1 hous) and render (200 FPS).""",
        "paper_title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
        "paper_authors": ["Bernhard Kerbl", "Georgios Kopanas", "Thomas Leimkühler", "George Drettakis"],
        "paper_link": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf",
        "link": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/",
    },
)
