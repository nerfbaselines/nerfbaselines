import os
from ..registry import MethodSpec, register


GaussianSplattingSpec: MethodSpec = {
    "method": "._impl.gaussian_splatting:GaussianSplatting",
    "conda": {
        "environment_name": os.path.split(__file__[:-3])[-1].replace("_", "-"),
        "install_script": """git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
git checkout 2eee0e26d2d5fd00ec462df47752223952f6bf4e

conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
pip install plyfile==0.8.1 tqdm submodules/diff-gaussian-rasterization submodules/simple-knn

conda install -c conda-forge -y nodejs==20.9.0
conda develop .
pip install lpips==0.1.4 importlib_metadata typing_extensions
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

register(GaussianSplattingSpec, name="gaussian-splatting")
register(
    GaussianSplattingSpec,
    name="gaussian-splatting:large", 
    kwargs={
        "config_overrides": {
            "iterations": 100_000,
            "densify_from_iter": 1_500,
            "densify_until_iter": 50_000,
            "densification_interval": 300,
            "opacity_reset_interval": 10_000,
            "position_lr_max_steps": 100_000,
            "position_lr_final": 0.000_000_16,
            "position_lr_init": 0.000_016,
            "scaling_lr": 0.000_5,
        },
    },
    metadata={
        "name": "Gaussian Splatting (large)",
        "description": """A version of Gaussian Splatting designed for larger scenes."""
    },
)
