import os
from ..registry import register, MethodSpec


MipSplattingSpec: MethodSpec = {
    "method": "._impl.mip_splatting:MipSplatting",
    "conda": {
        "environment_name": os.path.split(__file__[:-3])[-1].replace("_", "-"),
        "python_version": "3.8",
        "install_script": """# Install mip-splatting
git clone https://github.com/autonomousvision/mip-splatting.git
cd mip-splatting
git checkout 746a17c9a906be256ed85b8fe18632f5d53e832d

conda install -y conda-build
conda develop .

conda install -y pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
conda install -c conda-forge -y nodejs==20.9.0

pip install -r requirements.txt
pip install lpips==0.1.4

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
""",
    },
    "metadata": {
        "name": "Mip-Splatting",
        "description": """A modification of Gaussian Splatting designed to better handle aliasing artifacts.""",
        "paper_title": "Mip-Splatting: Alias-free 3D Gaussian Splatting",
        "paper_authors": ["Zehao Yu", "Anpei Chen", "Binbin Huang", "Torsten Sattler", "Andreas Geiger"],
        "paper_link": "https://arxiv.org/pdf/2311.16493.pdf",
        "link": "https://niujinshuchong.github.io/mip-splatting/",
    },
}


register(MipSplattingSpec, name="mip-splatting")
register(
    MipSplattingSpec,
    name="mip-splatting:large", 
    kwargs={
        "config_overrides": {
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
    },
    metadata={
        "name": "Mip-Splatting (large)",
        "description": """A version of Mip-Splatting designed for larger scenes."""
    },
)
