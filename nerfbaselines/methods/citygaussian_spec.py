from nerfbaselines import register, MethodSpec
from nerfbaselines.backends import CondaBackendSpec
import os


_conda_spec: CondaBackendSpec = {
        "environment_name": os.path.split(__file__[:-len("_spec.py")])[-1].replace("_", "-"),
        "python_version": "3.9",
        "install_script": """# Clone the repo.
git clone https://github.com/DekuLiuTesla/CityGaussian.git citygaussian
cd citygaussian
git checkout db21484dc262a446d12995633ac1b80bba44d4c9
git submodule update --init --recursive

# Prepare pip.
conda install -y pip conda-build -c conda-forge
# Install ffmpeg if not available
command -v ffmpeg >/dev/null || conda install -y ffmpeg=7.1.0

# Install build dependencies
conda install -y cuda-toolkit -c nvidia/label/cuda-11.8.0

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements/pyt201_cu118.txt
pip install -r requirements/CityGS.txt
pip install -r requirements.txt
""",
}


CityGaussianSpec: MethodSpec = {
    "id": "citygaussian",
    "method_class": ".citygaussian:CityGaussian",
    "conda": _conda_spec,
    "metadata": {
        "name": "CityGaussian",
        "paper_title": "",
        "paper_authors": [],
        "paper_link": "",
        "link": "",
        "description": "",
        "licenses": [],
    },
}

register(CityGaussianSpec)
