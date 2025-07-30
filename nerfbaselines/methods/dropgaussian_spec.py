import os
from nerfbaselines import register

_note = """Authors evaluated on larger images which were downscaled to the target size (avoiding JPEG compression artifacts) instead of using the official provided downscaled images. As mentioned in the 3DGS paper, this increases results slightly ~0.5 dB PSNR."""

_paper_results = {
}

GIT_REPOSITORY = "https://github.com/DCVL-3D/DropGaussian_release.git"
GIT_COMMIT_SHA = "c13da4aaddfed432fb556fdcf5d1d028b967ada8"
METHOD_ID = "dropgaussian"


register({
    "id": METHOD_ID,
    "method_class": f".{METHOD_ID}:DropGaussian",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": f"""git clone {GIT_REPOSITORY} {METHOD_ID}
cd {METHOD_ID}
git checkout {GIT_COMMIT_SHA}
git submodule update --init --recursive

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
pip install plyfile==0.8.1 \
        mediapy==1.1.2 \
        open3d==0.18.0 \
        lpips==0.1.4 \
        scikit-image==0.21.0 \
        tqdm==4.66.2 \
        trimesh==4.3.2 \
        opencv-python-headless==4.10.0.84 \
        importlib_metadata==8.5.0 \
        typing_extensions==4.12.2 \
        wandb==0.19.1 \
        click==8.1.8 \
        Pillow==11.1.0 \
        matplotlib==3.9.4 \
        tensorboard==2.18.0 \
        scipy==1.13.1 \
        pytest==8.3.4 \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        --no-build-isolation

conda develop .

if [ "$NERFBASELINES_DOCKER_BUILD" = "1" ]; then
# Reduce size of the environment by removing unused files
find "$CONDA_PREFIX" -name '*.a' -delete
find "$CONDA_PREFIX" -type d -name 'nsight*' -exec rm -r {{}} +
# Replace all libs under $CONDA_PREFIX/lib with symlinks to pkgs/cuda-toolkit/targets/x86_64-linux/lib
for lib in "$CONDA_PREFIX"/lib/*.so*; do 
    if [ ! -f "$lib" ] || [ -L "$lib" ]; then continue; fi;
    lib="${{lib%.so*}}.so";libname=$(basename "$lib");
    tgt="$CONDA_PREFIX/pkgs/cuda-toolkit/targets/x86_64-linux/lib/$libname"
    if [ -f "$tgt" ]; then echo "Deleting $lib"; rm "$lib"*; for tgtlib in "$tgt"*; do ln -s "$tgtlib" "$(dirname "$lib")"; done; fi;
done
fi
""",
    },
    "metadata": {
        "name": "DropGaussian",
        "description": "DropGaussian extends 3DGS with a simple regularization technique for sparse view novel view synthesis.",
        "paper_title": "DropGaussian: Structural Regularization for Sparse-view Gaussian Splatting",
        "paper_authors": "Hyunwoo Park, Gun Ryu, Wonjun Kim".split(", "),
        "paper_venue": "CVPR 2025",
        "paper_results": _paper_results,
        "paper_link": "https://openaccess.thecvf.com/content/CVPR2025/papers/Park_DropGaussian_Structural_Regularization_for_Sparse-view_Gaussian_Splatting_CVPR_2025_paper.pdf",
        "link": "https://github.com/DCVL-3D/DropGaussian_release",
        "licenses": [{"name": "custom, research only", "url": "https://raw.githubusercontent.com/DCVL-3D/DropGaussian_release/refs/heads/main/LICENSE_GAUSSIAN_SPLATTING.md"}, {"name": "Apache 2.0", "url": "https://raw.githubusercontent.com/DCVL-3D/DropGaussian_release/refs/heads/main/LICENSE"}],
    },
    "presets": {
    },
    "implementation_status": {}
})
