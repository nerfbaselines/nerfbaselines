from ..registry import register, MethodSpecDict


MipSplattingSpecDict: MethodSpecDict = {
    "method": "._impl.mip_splatting:MipSplatting",
    "conda": {
        "conda_name": "mip-splatting",
        "python_version": "3.8",
        "install_script": """# Install mip-splatting
git clone https://github.com/autonomousvision/mip-splatting.git
cd mip-splatting
git checkout 746a17c9a906be256ed85b8fe18632f5d53e832d

conda install -y conda-build
conda develop .

conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 -c conda-forge

pip install -r requirements.txt

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


register(MipSplattingSpecDict, "mip-splatting")
register(
    MipSplattingSpecDict,
    "mip-splatting:large", 
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
        "name": "Mip-Splatting (large)",
        "description": """A version of Mip-Splatting designed for larger scenes."""
    },
)
