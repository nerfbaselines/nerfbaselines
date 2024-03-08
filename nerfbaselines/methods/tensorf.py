import os
from ..registry import register, MethodSpec


TensoRFSpec: MethodSpec = {
    "method": "._impl.tensorf:TensoRF",
    "conda": {
        "environment_name": os.path.split(__file__[:-3])[-1].replace("_", "-"),
        "python_version": "3.8",
        "install_script": """# Install TensoRF
git clone https://github.com/apchenstu/TensoRF.git tensorf
cd tensorf
git checkout 9370a87c88bf41b309da694833c81845cc960d50

conda install -y conda-build
conda develop .

conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y cudatoolkit-dev=11.7 -c conda-forge
pip install tqdm scikit-image opencv-python configargparse lpips imageio-ffmpeg kornia lpips tensorboard
pip install plyfile six
""",
    },
    "metadata": {
        "name": "TensoRF",
        "description": """TensoRF factorizes the radiance field into a multiple compact low-rank tensor components. It was designed and tester primarily on Blender, LLFF, and NSVF datasets.""",
        "paper_title": "TensoRF: Tensorial Radiance Fields",
        "paper_authors": ["Anpei Chen", "Zexiang Xu", "Andreas Geiger", "Jingyi Yu", "Hao Su"],
        "paper_link": "https://arxiv.org/pdf/2203.09517.pdf",
        "link": "https://apchenstu.github.io/TensoRF/",
    },
}

register(TensoRFSpec, name="tensorf")
