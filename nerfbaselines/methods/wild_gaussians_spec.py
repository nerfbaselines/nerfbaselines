from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nerfbaselines.registry import MethodSpec
else:
    MethodSpec = dict


WildGaussiansMethodSpec: MethodSpec = {
    "method": "wildgaussians.method:WildGaussians",
    "conda": {
        "environment_name": "wild-gaussians",
        "python_version": "3.11",
        "install_script": r"""
git clone https://github.com/jkulhanek/wild-gaussians.git
cd wild-gaussians
git checkout 47c24e823c00ec22d4b7383cc31d90de7eaae1f8
conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
if [ "$NB_DOCKER_BUILD" != "1" ]; then
conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
fi
pip install --upgrade pip
pip install -r requirements.txt
LIBRARY_PATH="$CONDA_PREFIX/lib/stubs" pip install -e ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
pip install -e .
"""
    },
    "dataset_overrides": {
        "phototourism": { "config": "phototourism.yml" },
        "nerfonthego": { "config": "nerfonthego.yml" },
        "nerfonthego-undistorted": { "config": "nerfonthego.yml" },
    },
    "metadata": {
        "name": "WildGaussians",
        "description": "WildGaussians adopts 3DGS to handle appearance changes and transient objects. After fixing appearance, it can be baked back into 3DGS.",
        "paper_title": "WildGaussians: 3D Gaussian Splatting in the Wild",
        "authors": ["Jonas Kulhanek", "Songyou Peng", "Zuzana Kukelova", "Marc Pollefeys", "Torsten Sattler"],
        "paper_link": "https://arxiv.org/pdf/2407.08447.pdf",
        "link": "https://wild-gaussians.github.io/",
        "licenses": [
            {"name": "MIT", "url": "https://raw.githubusercontent.com/jkulhanek/wild-gaussians/main/LICENSE"}, 
            {"name": "custom, research only", "url": "https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/main/LICENSE.md"}
        ],
        "output_artifacts": {
            "phototourism/trevi-fountain": { "link": "https://huggingface.co/jkulhanek/wild-gaussians/resolve/main/phototourism/trevi-fountain.zip" },
            "phototourism/sacre-coeur": { "link": "https://huggingface.co/jkulhanek/wild-gaussians/resolve/main/phototourism/sacre-coeur.zip" },
            "phototourism/brandenburg-gate": { "link": "https://huggingface.co/jkulhanek/wild-gaussians/resolve/main/phototourism/brandenburg-gate.zip" },
            "nerfonthego-undistorted/fountain": { "link": "https://huggingface.co/jkulhanek/wild-gaussians/resolve/main/nerfonthego-undistorted/fountain.zip" },
            "nerfonthego-undistorted/mountain": { "link": "https://huggingface.co/jkulhanek/wild-gaussians/resolve/main/nerfonthego-undistorted/mountain.zip" },
            "nerfonthego-undistorted/spot": { "link": "https://huggingface.co/jkulhanek/wild-gaussians/resolve/main/nerfonthego-undistorted/spot.zip" },
            "nerfonthego-undistorted/patio": { "link": "https://huggingface.co/jkulhanek/wild-gaussians/resolve/main/nerfonthego-undistorted/patio.zip" },
            "nerfonthego-undistorted/patio-high": { "link": "https://huggingface.co/jkulhanek/wild-gaussians/resolve/main/nerfonthego-undistorted/patio-high.zip" },
            "nerfonthego-undistorted/corner": { "link": "https://huggingface.co/jkulhanek/wild-gaussians/resolve/main/nerfonthego-undistorted/corner.zip" },
        },
    }
}

try:
    from nerfbaselines.registry import register
    register(WildGaussiansMethodSpec, name="wild-gaussians")
except ImportError:
    pass
