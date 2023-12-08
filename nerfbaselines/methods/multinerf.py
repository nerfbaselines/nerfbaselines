from ..registry import MethodSpec, CondaMethod, LazyMethod


class MultiNeRF(LazyMethod["._impl.multinerf", "MultiNeRF"]):
    batch_size: int = 16384
    num_iterations: int = 250_000
    learning_rate_multiplier: float = 1.0


MultiNeRFSpec = MethodSpec(
    method=MultiNeRF,
    conda=CondaMethod.wrap(
        MultiNeRF,
        conda_name="multinerf",
        python_version="3.9",
        install_script="""# Clone the repo.
git clone https://github.com/jkulhanek/multinerf.git
cd multinerf
git checkout 0e6699cc01eb3f0e77e0f7c15057a3ee29ad74ba

conda install -y pip conda-build
conda develop "$PWD"

# Install requirements.
python -m pip install --upgrade pip
python -m pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
python -m pip install -r requirements.txt

# Manually install rmbrualla's `pycolmap` (don't use pip's! It's different).
git clone https://github.com/rmbrualla/pycolmap.git ./internal/pycolmap
conda develop "$PWD/internal/pycolmap"
conda develop "$PWD/internal/pycolmap/pycolmap"

# Confirm that all the unit tests pass.
# ./scripts/run_all_unit_tests.sh
""",
    ),
)
MultiNeRFSpec.register("mipnerf360")
MultiNeRFSpec.register("mipnerf360:single-gpu", batch_size=4096, num_iterations=1_000_000, learning_rate_multiplier=1 / 2)
