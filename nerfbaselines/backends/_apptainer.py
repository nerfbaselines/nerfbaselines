import sys
import io
import logging
import tempfile
import contextlib
from pathlib import Path
import subprocess
import os
from typing import Optional, List, Tuple, TYPE_CHECKING, cast
import shlex
import nerfbaselines
from ..types import NB_PREFIX, TypedDict, Required
from ._docker import BASE_IMAGE, get_docker_image_name, get_docker_spec
from ._conda import conda_get_install_script, conda_get_environment_hash, CondaBackendSpec
from ._rpc import RemoteProcessRPCBackend, get_safe_environment, customize_wrapper_separated_fs
from ._common import get_mounts
if TYPE_CHECKING:
    from ..registry import MethodSpec


class ApptainerBackendSpec(TypedDict, total=False):
    environment_name: Required[str]
    image: Optional[str]
    home_path: str
    python_path: str
    default_cuda_archs: str
    conda_spec: Optional[CondaBackendSpec]


def apptainer_get_safe_environment():
    env = get_safe_environment()
    allowed = {"APPTAINER_IMAGES", "APPTAINER_CACHEDIR", "CI", "NB_USE_GPU", "GITHUB_ACTIONS"}
    env.update({k: v for k, v in os.environ.items() if k in allowed})
    return env


def get_apptainer_spec(spec: 'MethodSpec') -> Optional[ApptainerBackendSpec]:
    apptainer_spec = spec.get("apptainer")
    if apptainer_spec is not None:
        return apptainer_spec

    docker_spec = get_docker_spec(spec)
    conda_spec = spec.get("conda")
    if docker_spec is not None:
        # Try to build apptainer spec from docker spec
        apptainer_spec = cast(ApptainerBackendSpec, {
            k: v for k, v in docker_spec.items() if k in ["environment_name", "home_path", "python_path", "default_cuda_archs"]
        })
        docker_image_name = get_docker_image_name(docker_spec)
        # If docker_image_name is the BASE_IMAGE, it be used, force conda build
        if docker_image_name is not None:
            apptainer_spec["image"] = "docker://" + docker_image_name
            # Fallback when image not found
            apptainer_spec["conda_spec"] = conda_spec
            return apptainer_spec

    if conda_spec is not None:
        environment_name = conda_spec.get("environment_name")
        assert environment_name is not None, "Environment name is not specified"
        return {
            "image": None,
            "environment_name": environment_name,
            "conda_spec": conda_spec,
        }
    return None


def _capture_subprocess_output(subprocess_args):
    import selectors
    process = subprocess.Popen(subprocess_args,
                               bufsize=1,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT,
                               universal_newlines=True)
    buf = io.StringIO()
    def handle_output(stream, mask):
        line = stream.readline()
        buf.write(line)
        sys.stdout.write(line)

    selector = selectors.DefaultSelector()
    assert process.stdout is not None, "stdout is None"
    selector.register(process.stdout, selectors.EVENT_READ, handle_output)

    while process.poll() is None:
        events = selector.select()
        for key, mask in events:
            callback = key.data
            callback(key.fileobj, mask)

    return_code = process.wait()
    selector.close()

    output = buf.getvalue()
    buf.close()
    return (return_code, output)


def _try_pull_docker_image(image: str) -> bool:
    status, output = _capture_subprocess_output(["apptainer", "exec", "--compat", image, "true"])
    if status == 0:
        return True
    if status != 0 and (
        "while making image from oci registry: error fetching image to cache:" in output.lower() or
        "failed to get checksum" in output.lower()
    ):
        # Image does not exist (likely)
        return False
    else:
        raise RuntimeError(f"Failed to pull docker image {image}")


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
    export_envs = ["TCNN_CUDA_ARCHITECTURES", "TORCH_CUDA_ARCH_LIST", "CUDAARCHS", "GITHUB_ACTIONS", "NB_AUTHKEY", "CI"]
    package_path = str(Path(nerfbaselines.__file__).absolute().parent.parent)

    return [
        "apptainer",
        "run",
        # "--containall",
        "--cleanenv",
        "--no-eval",
        "--writable-tmpfs",
        *(("--nv",) if use_gpu else ()),
        "--bind",
        "/tmp:/tmp",
        "--no-home",
        "-H",
        (spec.get("home_path") or "/root"),
        "--workdir",
        os.getcwd(),
        "--mount",
        f'"src={shlex.quote(os.getcwd())}","dst={shlex.quote(os.getcwd())}"',
        "--mount",
        f'"src={shlex.quote(os.path.join(NB_PREFIX, "apptainer-conda-envs"))}","dst=/var/apptainer-conda-envs"',
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
        *(sum((["--env", f"{name}={shlex.quote(shlex.quote(env.get(name, '')))}"] for name in export_envs if name in env), [])),
        ## "--env", "PYTHONPATH=/var/nb-package:${PYTHONPATH}",
        "--env",
        "NB_USE_GPU=" + ("1" if use_gpu else "0"),
        "--env",
        "CONDA_PKGS_DIRS=/var/nb-conda-pkgs",
        "--env",
        "NERFBASELINES_CONDA_ENVIRONMENTS=/var/apptainer-conda-envs",
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
        self._tmpdir = None
        self._applied_mounts = None
        self._installed = False
        super().__init__(address=address, port=port)

    def __enter__(self):
        super().__enter__()
        self._tmpdir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, *args):
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None
        self._applied_mounts = None
        super().__exit__(*args)

    def _customize_wrapper(self, ns):
        ns = super()._customize_wrapper(ns)
        assert self._tmpdir is not None, "Temporary directory is not initialized"
        customize_wrapper_separated_fs(self._tmpdir.name, "/var/nb-tmp", self._applied_mounts, ns)
        return ns

    def install(self):
        if self._installed:
            # Already installed
            return

        # Build the docker image if needed
        image = self._spec.get("image")
        if image is not None and image.startswith("docker://"):
            docker_conversion_disabled = os.environ.get("NERFBASELINES_APPTAINER_PREFER_CONDA", "0") == "1"
            if docker_conversion_disabled and self._spec.get("conda_spec") is not None:
                logging.info(f"Skipping docker image conversion for {image} because NERFBASELINES_APPTAINER_PREFER_CONDA is not set to 1.")
                image = None
            else:
                logging.info(f"Checking if docker image {image} is available. To disable this check, set NERFBASELINES_APPTAINER_PREFER_CONDA=1.")
                if not _try_pull_docker_image(image):
                    # Image does not exist (likely)
                    image = None

        if image != self._spec.get("image"):
            # Update the spec (remove missing docker image)
            self._spec = self._spec.copy()
            self._spec["image"] = image

        self._spec = self._spec.copy()
        self._spec["image"] = image

        if image is None:
            conda_spec = self._spec.get("conda_spec")
            if conda_spec is not None:
                logging.info("Docker image is not available. Trying to build apptainer-conda environment.")

                # First try to pull the base image
                if not _try_pull_docker_image(f"docker://{BASE_IMAGE}"):
                    base_latest = BASE_IMAGE[:BASE_IMAGE.rfind("-")]
                    logging.warning(f"Failed to pull base image {BASE_IMAGE}. Trying to pull {base_latest} instead.")
                    if not _try_pull_docker_image(f"docker://{base_latest}"):
                        raise RuntimeError(f"Failed to pull base image {BASE_IMAGE} or {base_latest}. Please install the latest version of NerfBaselines or switch backend.")
                    logging.info(f"Successfully pulled base image {base_latest}")
                    image = "docker://" + base_latest
                else:
                    image = "docker://" + BASE_IMAGE

                with with_environ({**os.environ, "NERFBASELINES_CONDA_ENVIRONMENTS": "/var/apptainer-conda-envs"}) as env:
                    args = ["bash", "-l", "-c", conda_get_install_script(conda_spec, package_path="/var/nb-package")]
                self._installed = True
                self._spec["image"] = image
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
            self._spec["image"] = image
            self._installed = True
            self._spec.pop("conda_spec", None)
            args = [(self._spec.get("python_path") or "python"), "-c", "import nerfbaselines"]
            args, env = apptainer_run(
                self._spec,
                args,                
                env=apptainer_get_safe_environment(),
                mounts=get_mounts())
            subprocess.check_call(args, env=env)

    def _launch_worker(self, args, env):
        # Run apptainer image
        if not self._installed:
            raise RuntimeError("Method is not installed. Please call install() first.")
        assert self._tmpdir is not None, "Temporary directory is not initialized"
        conda_spec = self._spec.get("conda_spec")
        if conda_spec is not None:
            env_path = "/var/apptainer-conda-envs"
            env_name = conda_spec["environment_name"]
            env_path = os.path.join(env_path, env_name, conda_get_environment_hash(conda_spec), env_name)
            args = [os.path.join(env_path, ".activate.sh")] + args
        self._applied_mounts = get_mounts()
        return super()._launch_worker(*apptainer_run(
            self._spec,
            args, env, 
            mounts=self._applied_mounts + [(self._tmpdir.name, "/var/nb-tmp")], 
            interactive=False,
            use_gpu=os.getenv("GITHUB_ACTIONS") != "true"))

    def shell(self):
        # Run apptainer image
        if not self._installed:
            raise RuntimeError("Method is not installed. Please call install() first.")
        env = apptainer_get_safe_environment()
        args = ["bash"]
        conda_spec = self._spec.get("conda_spec")
        if conda_spec is not None:
            env_path = "/var/apptainer-conda-envs"
            env_name = conda_spec["environment_name"]
            env_path = os.path.join(env_path, env_name, conda_get_environment_hash(conda_spec), env_name)
            args = [os.path.join(env_path, ".activate.sh")] + args
        args, env = apptainer_run(
            self._spec,
            args, env, 
            mounts=get_mounts(),
            interactive=True,
            use_gpu=os.getenv("GITHUB_ACTIONS") != "true")
        os.execvpe(args[0], args, env)
