import contextlib
from pathlib import Path
import subprocess
import os
from typing import Optional, List, Tuple, TYPE_CHECKING
import shlex
import nerfbaselines
from ..utils import cached_property
from ..types import NB_PREFIX, TypedDict
from ._docker import BASE_IMAGE
from ._conda import conda_get_install_script, conda_get_environment_hash, CondaBackendSpec
from ._rpc import RemoteProcessRPCBackend, get_safe_environment
from ._common import get_mounts
if TYPE_CHECKING:
    from ..registry import MethodSpec


class ApptainerBackendSpec(TypedDict, total=False):
    environment_name: Optional[str]
    image: Optional[str]
    home_path: str
    python_path: str
    default_cuda_archs: str
    conda_spec: Optional[CondaBackendSpec]


def apptainer_get_safe_environment():
    env = get_safe_environment()
    allowed = {"APPTAINER_IMAGES", "APPTAINER_CACHEDIR"}
    env.update({k: v for k, v in os.environ.items() if k in allowed})
    return env


def get_apptainer_spec(spec: 'MethodSpec') -> Optional[ApptainerBackendSpec]:
    apptainer_spec = spec.get("apptainer")
    if apptainer_spec is not None:
        return apptainer_spec

    docker_spec = spec.get("docker")
    if docker_spec is not None and docker_spec.get("image") is not None:
        return {
            **docker_spec,
            "image": "docker://" + docker_spec["image"]
        }

    conda_spec = spec.get("conda")
    if conda_spec is not None:
        return {
            "image": "docker://" + BASE_IMAGE,
            "environment_name": conda_spec.get("environment_name"),
            "conda_spec": conda_spec
        }
    return None


def _try_get_precompiled_docker_image(spec: ApptainerBackendSpec) -> Optional[str]:
    from ._docker import get_docker_image_name, DockerBackendSpec, docker_image_exists_remotely

    docker_image = spec.get("image")
    if docker_image is not None:
        if not docker_image.startswith("docker://"):
            return None
        docker_image = docker_image[len("docker://"):]
    docker_spec: DockerBackendSpec = {
        **spec,
        "image": docker_image,
    }
    docker_image_name = get_docker_image_name(docker_spec)
    if docker_image_exists_remotely(docker_image_name):
        return docker_image_name
    return None


def apptainer_run(spec: ApptainerBackendSpec, args, env,
                  mounts: Optional[List[Tuple[str, str]]] = None,
                  interactive: bool = False,
                  use_gpu: bool = True):
    os.makedirs(os.path.join(NB_PREFIX, "apptainer-conda-envs"), exist_ok=True)
    conda_cache = os.path.expanduser(env.get("CONDA_PKGS_DIRS", "~/.conda/pkgs"))
    os.makedirs(conda_cache, exist_ok=True)
    pip_cache = os.path.expanduser(env.get("PIP_CACHE_DIR", "~/.cache/pip"))
    os.makedirs(pip_cache, exist_ok=True)
    torch_home = os.path.expanduser(env.get("TORCH_HOME", "~/.cache/torch/hub"))
    os.makedirs(torch_home, exist_ok=True)
    image = spec.get("image") or f"docker://{BASE_IMAGE}"
    export_envs = ["TCNN_CUDA_ARCHITECTURES", "TORCH_CUDA_ARCH_LIST", "CUDAARCHS", "GITHUB_ACTIONS", "NB_AUTHKEY"]
    package_path = str(Path(nerfbaselines.__file__).absolute().parent.parent)
    return [
        "apptainer",
        "exec",
        # "--containall",
        "--cleanenv",
        "--writable-tmpfs",
        *(("--nv",) if use_gpu else ()),
        "--bind",
        "/tmp:/tmp",
        "--writable-tmpfs",
        "--no-home",
        "-H",
        (spec.get("home_path") or "/root"),
        "--workdir",
        os.getcwd(),
        "--mount",
        f'"src={shlex.quote(os.getcwd())}","dst={shlex.quote(os.getcwd())}"',
        "--mount",
        f'"src={shlex.quote(os.path.join(NB_PREFIX, "apptainer-conda-envs"))}","dst=/var/conda-envs"',
        "--mount",
        f'"src={shlex.quote(package_path)}","dst=/var/nb-package"',
        "--mount",
        f'"src={shlex.quote(NB_PREFIX)}",dst=/var/nb-prefix',
        "--mount",
        f'"src={shlex.quote(conda_cache)}",dst=/var/nb-conda-pkgs',
        "--mount",
        f'"src={shlex.quote(pip_cache)}",dst=/var/nb-pip-cache',
        "--mount",
        f'"src={shlex.quote(torch_home)}",dst=/var/nb-torch',
        *sum([["--mount", f'"src={shlex.quote(src)}","dst={shlex.quote(dst)}"'] for src, dst in mounts or []], []),
        *(sum((["--env", f"{name}={shlex.quote(env.get(name, ''))}"] for name in export_envs if name in env), [])),
        "--env", "PYTHONPATH=/var/nb-package",
        "--env",
        "NB_USE_GPU=" + ("1" if use_gpu else "0"),
        "--env",
        "CONDA_PKGS_DIRS=/var/nb-conda-pkgs",
        "--env",
        "NERFBASELINES_CONDA_ENVIRONMENTS=/var/conda-envs",
        "--env",
        "NERFBASELINES_PREFIX=/var/nb-prefix",
        "--env",
        "PIP_CACHE_DIR=/var/nb-pip-cache",
        "--env",
        "TORCH_HOME=/var/nb-torch",
        "--env",
        "COLUMNS=120",
        image,
    ] + args, env


@contextlib.contextmanager
def with_environ(env):
    environ_backup = os.environ.copy()
    try:
        os.environ.clear()
        os.environ.update(env)
        yield os.environ
    finally:
        os.environ.clear()
        os.environ.update(environ_backup)


class ApptainerBackend(RemoteProcessRPCBackend):
    name = "apptainer"

    def __init__(self, 
                 spec: ApptainerBackendSpec, 
                 address: str = "0.0.0.0", 
                 port: Optional[int] = None):
        self._spec = spec
        super().__init__(address=address, port=port)

    @cached_property
    def _docker_image_name(self):
        return _try_get_precompiled_docker_image(self._spec)

    def install(self):
        # Build the docker image if needed
        if self._docker_image_name is None:
            conda_spec = self._spec.get("conda_spec")
            if conda_spec is not None:
                with with_environ({**os.environ, "NERFBASELINES_CONDA_ENVIRONMENTS": "/var/conda-envs"}) as env:
                    args = ["bash", "-l", "-c", conda_get_install_script(conda_spec)]
                args, env = apptainer_run(
                    self._spec, 
                    args,
                    env=env,
                    mounts=get_mounts())
                subprocess.check_call(args, env=env)
            else:
                raise RuntimeError("Docker image is not available and apptainer image cannot be built.")
        else:
            # Pull the image and test
            args = [(self._spec.get("python_path") or "python"), "-c", "import nerfbaselines"]
            args, env = apptainer_run(
                self._spec, 
                args,                
                env=apptainer_get_safe_environment(),
                mounts=get_mounts())
            subprocess.check_call(args, env=env)

    def _launch_worker(self, args, env):
        # Run docker image
        conda_spec = self._spec.get("conda_spec")
        if conda_spec is not None:
            env_path = "/var/conda-envs"
            env_name = conda_spec["environment_name"]
            env_path = os.path.join(env_path, env_name, conda_get_environment_hash(conda_spec), env_name)
            args = [os.path.join(env_path, ".activate.sh")] + args
        return super()._launch_worker(*apptainer_run(
            self._spec, args, env, 
            mounts=get_mounts(), 
            interactive=False,
            use_gpu=os.getenv("GITHUB_ACTIONS") != "true"))

    def shell(self):
        # Run docker image
        env = apptainer_get_safe_environment()
        args = ["bash"]
        conda_spec = self._spec.get("conda_spec")
        if conda_spec is not None:
            env_path = "/var/conda-envs"
            env_name = conda_spec["environment_name"]
            env_path = os.path.join(env_path, env_name, conda_get_environment_hash(conda_spec), env_name)
            args = [os.path.join(env_path, ".activate.sh")] + args
        args, env = apptainer_run(
            self._spec, args, env, 
            mounts=get_mounts(),
            interactive=True,
            use_gpu=os.getenv("GITHUB_ACTIONS") != "true")
        os.execvpe(args[0], args, env)
