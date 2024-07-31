import os
from ..registry import MethodSpec, register

paper_results = {
}


GaussianSplattingWildSpec: MethodSpec = {
    "method": ".gaussian_splatting_wild:GaussianSplattingWild",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """git clone https://github.com/EastbeanZhang/Gaussian-Wild.git --recursive
cd Gaussian-Wild
git checkout 79d9e6855298a2632b530644e52d1829c6356b08

# Fix the code, replace line 80 in scene/dataset_readers.py 
# from "intr = cam_intrinsics[extr.id]" to "intr = cam_intrinsics[extr.camera_id]"
sed -i '80s/.*/        intr = cam_intrinsics[extr.camera_id]/' scene/dataset_readers.py

conda install -y mkl==2023.1.0 pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 gcc_linux-64=11 gxx_linux-64=11 make=4.3 cmake=3.28.3 -c conda-forge
pip install -U pip 'setuptools<70.0.0' 'wheel==0.43.0'
pip install pandas==2.2.2 \
        plyfile==0.8.1 \
        kornia==0.7.3 \
        tqdm \
        submodules/diff-gaussian-rasterization \
        submodules/simple-knn \
        --no-build-isolation

conda install -c conda-forge -y nodejs==20.9.0
conda develop .
pip install lpips==0.1.4 importlib_metadata typing_extensions

function nb-post-install () {
if [ "$NB_DOCKER_BUILD" = "1" ]; then
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
""",
    },
    "metadata": {},
    "dataset_overrides": {},
}

register(GaussianSplattingWildSpec, name="gaussian-splatting-wild")

