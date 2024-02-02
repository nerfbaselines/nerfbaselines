from ..registry import MethodSpecDict, register


GaussianSplattingSpecDict: MethodSpecDict = {
    "method": "._impl.gaussian_splatting:GaussianSplatting",
    "conda": {
        "conda_name": "gaussian-splatting",
        "install_script": """git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
git checkout 2eee0e26d2d5fd00ec462df47752223952f6bf4e
conda env update --file environment.yml --prune --prefix "$CONDA_PREFIX"
conda install -y conda-build
conda develop .
pip install importlib_metadata typing_extensions
""",
    },
    "metadata": {
        "name": "Gaussian Splatting",
        "description": """Official Gaussian Splatting implementation extended to support distorted camera models. It is fast to train (1 hous) and render (200 FPS).""",
        "paper_title": "3D Gaussian Splatting for Real-Time Radiance Field Rendering",
        "paper_authors": ["Bernhard Kerbl", "Georgios Kopanas", "Thomas Leimkühler", "George Drettakis"],
        "paper_link": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_low.pdf",
        "link": "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/",
    }
}

register(GaussianSplattingSpecDict, "gaussian-splatting")
register(
    GaussianSplattingSpecDict,
    "gaussian-splatting:large", 
    config_overrides={
        "iterations": 300_000,
        "densify_from_iter": 5_000,
        "densify_until_iter": 150_000,
        "densification_interval": 1_000,
        "opacity_reset_interval": 30_000,
        "position_lr_max_steps": 300_000,
        "position_lr_final": 0.000_000_16,
        "position_lr_init": 0.000_016,
        "scaling_lr": 0.000_5,
    },
    metadata={
        "name": "Gaussian Splatting (large)",
        "description": """A version of Gaussian Splatting designed for larger scenes."""
    },
)
