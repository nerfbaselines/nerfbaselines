import os
from nerfbaselines import register


_note = """Results in the paper were evaluated using different tau (0, 3, 6, 15),
where tau=0 is the slowest, but highest quality.
We choose tau=6 - consistent with most experiments in the paper,
providing a good trade-off between quality and speed.""".replace("\n", " ")
paper_results = {
    "hierarchical-3dgs/smallcity": { "psnr": 26.29, "ssim": 0.810, "lpips": 0.275, "note": _note },
    "hierarchical-3dgs/campus": { "psnr": 24.50, "ssim": 0.801, "lpips": 0.340, "note": _note },
}


long_description = """
The Hierarchical 3DGS implementation performs splitting the scene into regions, each
of which is optimized separately. This set is then merged into a single model. This
is only applied for larger scenes. In NerfBaselines integration, by default we do
not use the splitting - this matches the original implementation when applied to smaller scenes.
The splitting implementation is not yet available in NerfBaselines, but it is planned for the future.

Also, we added `depth_mode` option which allows to use different monodular depth predictors. Currently,
add `--preset depth-anything` to use Depth Anything V2 depth predictor, or use the default `dpt` model.

In order to enable appearance optimization, add `--preset exposure` to the command line. This 
will optimize a affine mapping for each image during the training to map the rendered colors.
This option is recommended for the scenes with strong lighting changes when rendering a video,
but it can decrease metrics - especially PSNR. By default exposure optimization is turned off.

Durring rendering, you can pass tau (float) in the rendering `options` to set the tau parameter
used by H3DGS renderer. The default value is 0 which means the finest set of Gaussians will
be used - highest quality, but slowest performance.
"""


register({
    "id": "hierarchical-3dgs",
    "method_class": ".hierarchical_3dgs:Hierarchical3DGS",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/graphdeco-inria/hierarchical-3d-gaussians.git
cd hierarchical-3d-gaussians
git checkout 85777b143010dedb7bc370a4591de3498fe878bb
git submodule update --recursive --init

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 'numpy<2.0.0' -c pytorch -c nvidia
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y 'ffmpeg<=7.1.0'
conda install -y gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
conda install -y cuda-toolkit -c "nvidia/label/cuda-11.7.1"
conda install -c conda-forge -y nodejs==20.9.0
conda develop .

# Fix gcc paths
_prefix="$CONDA_PREFIX"
conda deactivate; conda activate "$_prefix"
ln -s "$CC" "$CONDA_PREFIX/bin/gcc"
ln -s "$CXX" "$CONDA_PREFIX/bin/g++"
export CPATH="$CONDA_PREFIX/x86_64-conda-linux-gnu/sysroot/usr/include:$CPATH"

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
        pytest==8.3.4 \
        websockets==14.2 \
        timm==0.4.5 \
        scipy==1.13.1 \
        submodules/hierarchy-rasterizer \
        submodules/simple-knn \
        submodules/gaussianhierarchy \
        --no-build-isolation

(
cd submodules/gaussianhierarchy
cmake . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build -j --config Release
ln -s "$PWD/build/GaussianHierarchyCreator" "$CONDA_PREFIX/bin"
ln -s "$PWD/build/GaussianHierarchyMerger" "$CONDA_PREFIX/bin"
)

# Download checkpoints
mkdir -p submodules/Depth-Anything-V2/checkpoints/
curl -L https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true \
        -o submodules/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth
mkdir -p submodules/DPT/weights/
curl -L https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt \
        -o submodules/DPT/weights/dpt_large-midas-2f21e586.pt
conda develop $PWD/submodules/Depth-Anything-V2
conda develop $PWD/submodules/DPT

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
""",
    },
    "metadata": {
        "name": "H3DGS",
        "description": "H3DGS extends 3DGS with LOD rendering strategy based on hierarchical representation of the scene. For large scenes it splits it into chunks, optimize each separatedly and merge them into single model.",
        "paper_title": "A Hierarchical 3D Gaussian Representation for Real-Time Rendering of Very Large Datasets",
        "paper_authors": ["Bernhard Kerbl", "Andreas Meuleman", "Georgios Kopanas", "Michael Wimmer", "Alexandre Lanvin", "George Drettakis"],
        "paper_link": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/hierarchical-3d-gaussians_low.pdf",
        "link": "https://repo-sam.inria.fr/fungraph/hierarchical-3d-gaussians/",
        "licenses": [{
            "name": "custom, research only",
            "url": "https://raw.githubusercontent.com/graphdeco-inria/hierarchical-3d-gaussians/refs/heads/main/LICENSE.md"
        }, {
            "name": "custom, research only",
            "url": "https://raw.githubusercontent.com/graphdeco-inria/gaussian-splatting/refs/heads/main/LICENSE.md"
        }],
        "long_description": long_description,
        "paper_results": paper_results,
    },
    "presets": {
        "exposure": { "exposure_lr_init": 0.001, "exposure_lr_final": 0.0001 },
        "depth-anything": { "single.depth_mode": "depth-anything" },
        "tau-0": { "post.tau": "0" },
        "tau-3": { "post.tau": "3" },
        "tau-6": { "post.tau": "6" },
        "tau-6": { "post.tau": "15" },
    },
})
