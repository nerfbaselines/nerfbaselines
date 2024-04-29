import os
from ..registry import MethodSpec, register


NeRFSpec: MethodSpec = {
    "method": ".nerf:NeRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.7",
        "install_script": r"""# Clone the repo.
git clone https://github.com/bmild/nerf
cd nerf
git checkout 18b8aebda6700ed659cb27a0c348b737a5f6ab60
# Allow depthmaps to be outputted
sed '239a\
    ret["depth"] = depth_map
' -i "$CONDA_PREFIX/src/nerf/run_nerf.py"

conda install -y pip conda-build
conda develop "$PWD"

# Install requirements.
conda install -y numpy \
                 configargparse \
                 imagemagick \
                 cudatoolkit=10.0
conda install -y -c anaconda tensorflow-gpu==1.15 
conda install -y pytorch==1.1.0 torchvision==0.3.0 -c pytorch
python -m pip install --upgrade pip
function nb-post-install () {
    python -m pip uninstall -y pillow
    python -m pip install imageio==2.9.0 "pillow<7"
}
""",
    },
    "metadata": {
        "name": "NeRF",
        "description": "sdf",
        "paper_title": "sdf",
        "paper_authors": ["sdf"],
        "paper_link": "sdf",
        "link": "sdf",
    },
}

register(
    NeRFSpec,
    name="nerf",
)
