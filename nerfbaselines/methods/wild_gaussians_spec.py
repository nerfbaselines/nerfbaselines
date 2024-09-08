from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from nerfbaselines import MethodSpec
else:
    MethodSpec = dict


WildGaussiansMethodSpec: MethodSpec = {
    "method_class": "wildgaussians.method:WildGaussians",
    "conda": {
        "environment_name": "wild-gaussians",
        "python_version": "3.11",
        "install_script": r"""
git clone https://github.com/jkulhanek/wild-gaussians.git
cd wild-gaussians
git checkout 1149c8161dadb17ababb221782bcbf2f1c967960
conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
if [ "$NERFBASELINES_DOCKER_BUILD" != "1" ]; then
conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
fi
pip install --upgrade pip
pip install 'numpy<2.0.0' -r requirements.txt
LIBRARY_PATH="$CONDA_PREFIX/lib/stubs" pip install -e ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
pip install -e .
pip install opencv-python-headless

function nb-post-install () {
if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {} +
# Replace all libs under $CONDA_PREFIX/lib with symlinks to pkgs/cuda-toolkit/targets/x86_64-linux/lib
for lib in "$CONDA_PREFIX"/lib/*.so*; do 
    if [ ! -f "$lib" ] || [ -L "$lib" ]; then continue; fi;
    lib="${lib%.so*}.so";libname=$(basename "$lib");
    tgt="$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/$libname"
    if [ -f "$tgt" ]; then echo "Deleting $lib"; rm "$lib"*; for tgtlib in "$tgt"*; do ln -s "$tgtlib" "$(dirname "$lib")"; done; fi;
done
fi
}
"""
    },
    "presets": {
        "phototourism": { "@apply": [{"dataset": "phototourism"}], "config": "phototourism.yml" },
        "nerfonthego": { 
            "@apply": [
                {"dataset": "nerfonthego"},
                {"dataset": "nerfonthego-undistorted"},
            ], "config": "nerfonthego.yml" },
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
    },
    "id": "wild-gaussians",
    "supported_camera_models": ["pinhole",],
    "supported_outputs": ["color", "accumulation", "depth"],
    "required_features": ["color", "points3D_xyz"],
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
    "implementation_status": {
        "phototourism": "reproducing",
    },
}

try:
    from nerfbaselines import register
    register(WildGaussiansMethodSpec)
except ImportError:
    pass
