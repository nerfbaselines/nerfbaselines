import shutil
import requests
from pathlib import Path
import json
import logging
import hashlib
import tempfile
import subprocess
import os
from typing import Optional, List, Tuple, Dict, Union, TYPE_CHECKING, cast
import shlex

import nerfbaselines

from ._conda import CondaBackendSpec, conda_get_environment_hash, conda_get_install_script
from ..types import NB_PREFIX, TypedDict, Required
from ..utils import get_package_dependencies, shlex_join
from ._rpc import RemoteProcessRPCBackend, get_safe_environment, customize_wrapper_separated_fs
from .._constants import DOCKER_REPOSITORY
from ._common import get_mounts
from .. import __version__
if TYPE_CHECKING:
    from ..registry import MethodSpec


EXPORT_ENVS = ["TCNN_CUDA_ARCHITECTURES", "TORCH_CUDA_ARCH_LIST", "CUDAARCHS", "GITHUB_ACTIONS", "NB_PORT", "NB_PATH", "NB_AUTHKEY", "NB_ARGS", "CI"]
DEFAULT_CUDA_ARCHS = "7.0 7.5 8.0 8.6+PTX"
DOCKER_TAG_HASH_LENGTH = 10


class DockerBackendSpec(TypedDict, total=False):
    environment_name: Required[str]
    image: Optional[str]
    home_path: str
    python_path: str
    default_cuda_archs: str
    conda_spec: CondaBackendSpec
    replace_user: bool
    build_script: str
    

def docker_get_environment_hash(spec: Union[DockerBackendSpec, Dict]):
    value = hashlib.sha256()

    def maybe_update(x):
        if x is not None:
            value.update(x.encode("utf8"))

    maybe_update(spec.get("image"))
    maybe_update(spec.get("home_path"))
    maybe_update(spec.get("default_cuda_archs"))
    maybe_update(spec.get("build_script"))
    maybe_update(",".join(get_package_dependencies(ignore_viewer=True)))
    maybe_update(_DEFAULT_TORCH_INSTALL_COMMAND)
    conda_spec = spec.get("conda_spec")
    if conda_spec:
        maybe_update(conda_get_environment_hash(conda_spec))
    return value.hexdigest()


_DEFAULT_TORCH_INSTALL_COMMAND = "torch==2.2.0 torchvision==0.17.0 'numpy<2.0.0' --index-url https://download.pytorch.org/whl/cu118"
BASE_IMAGE = f"{DOCKER_REPOSITORY}:base-{docker_get_environment_hash(cast(DockerBackendSpec, { 'default_cuda_archs': DEFAULT_CUDA_ARCHS }))[:DOCKER_TAG_HASH_LENGTH]}"


def get_docker_spec(spec: 'MethodSpec') -> Optional[DockerBackendSpec]:
    docker_spec: Optional[DockerBackendSpec] = spec.get("docker")
    if docker_spec is not None:
        return docker_spec
    conda_spec = spec.get("conda")
    if conda_spec is not None:
        docker_spec: Optional[DockerBackendSpec] = {
            "image": None,
            "environment_name": conda_spec["environment_name"],
            "conda_spec": conda_spec
        }
        return docker_spec
    return None


def docker_get_dockerfile(spec: DockerBackendSpec):
    from .. import registry

    image = spec.get("image")
    if image is None:
        script = Path(__file__).absolute().parent.joinpath("Dockerfile").read_text()
        script += "\n"
    else:
        script = f"FROM {image}\n"
    script += "LABEL maintainer=\"Jonas Kulhanek\"\n"
    script += f'LABEL com.nerfbaselines.version="{__version__}"\n'
    script += 'LABEL org.opencontainers.image.authors="jonas.kulhanek@live.com"\n'
    environment_name = spec.get("environment_name")
    if environment_name is not None:
        script += f'LABEL com.nerfbaselines.environment="{environment_name}"\n'

    package_path = "/var/nb-package"
    conda_spec = spec.get("conda_spec")
    build_script = spec.get("build_script")
    default_cuda_archs = spec.get("default_cuda_archs") or DEFAULT_CUDA_ARCHS
    tcnn_cuda_archs = default_cuda_archs.replace(".", "").replace("+PTX", "").replace(" ", ";")
    if build_script:
        run_command = f'set -e;export TORCH_CUDA_ARCH_LIST="{default_cuda_archs}";export TCNN_CUDA_ARCHITECTURES="{tcnn_cuda_archs}";export NB_DOCKER_BUILD=1;{build_script.rstrip()}'
        run_command = shlex.quote(run_command)
        out = ""
        end = ""
        for line in run_command.splitlines():
            out += end + line
            if line.endswith("\\"):
                end = "\\\\n\\\n"
            else:
                end = "\\n\\\n"
        run_command = out
        script += f'RUN /bin/bash -c "$(echo {run_command})" && \\\n'
        script += "bash -c 'rm -Rf /root/.cache/pip'\n"

    if conda_spec is not None:
        environment_name = conda_spec.get("environment_name")
        assert environment_name is not None, "CondaBackend requires environment_name to be specified"

        environment_path = f"/var/conda-envs/{environment_name}"
        script += "ENV NERFBASELINES_CONDA_ENVIRONMENTS=/var/conda-envs\n"

        shell_args = [os.path.join(environment_path, ".activate.sh")]

        # Add install conda script
        install_conda = conda_get_install_script(conda_spec, package_path, environment_path=environment_path)
        run_command = f'export TORCH_CUDA_ARCH_LIST="{default_cuda_archs}" && export TCNN_CUDA_ARCHITECTURES="{tcnn_cuda_archs}" && export NB_DOCKER_BUILD=1 && {install_conda.rstrip()}'
        run_command = shlex.quote(run_command)
        out = ""
        end = ""
        for line in run_command.splitlines():
            out += end + line
            if line.endswith("\\"):
                end = "\\\\n\\\n"
            else:
                end = "\\n\\\n"
        run_command = out

        script += f'RUN /bin/bash -c "$(echo {run_command})" && \\\n'
        script += shlex_join(shell_args) + " bash -c 'conda clean -afy && rm -Rf /root/.cache/pip'\n"
        # Fix permissions when changing the user inside the container
        script += "RUN chmod -R og=u /var/conda-envs\n"
        script += "ENTRYPOINT " + json.dumps(shell_args) + "\n"
        script += "SHELL " + json.dumps(shell_args) + "\n"

    else:
        # If not inside conda env, we install the dependencies
        python_path = spec.get("python_path") or "python"
        script += f"RUN if ! {python_path} -c 'import torch'; then {python_path} -m pip install --no-cache-dir " + _DEFAULT_TORCH_INSTALL_COMMAND + "; fi && \\\n"
        package_dependencies = get_package_dependencies()
        if package_dependencies:
            script += "    " + shlex_join([python_path, "-m", "pip", "--no-cache-dir", "install"] + package_dependencies)+ " && \\\n"
        script += f"    if ! {python_path} -c 'import cv2'; then {python_path} -m pip install --no-cache-dir opencv-python-headless; fi\n"
        script += f'RUN if ! nerfbaselines >/dev/null 2>&1; then echo -e \'#!/usr/bin/env {python_path}\\nfrom nerfbaselines.__main__ import main\\nif __name__ == "__main__":\\n  main()\\n\'>"/usr/bin/nerfbaselines" && chmod +x "/usr/bin/nerfbaselines" || echo "Failed to create nerfbaselines in the bin folder"; fi\n'

    script += "ENV NERFBASELINES_BACKEND=python\n"
    def is_method_allowed(method_spec: "MethodSpec"):
        docker_spec = get_docker_spec(method_spec)
        return docker_spec is not None and docker_spec.get("environment_name") == spec.get("environment_name")

    allowed_methods = ",".join((k for k, v in registry.methods_registry.items() if is_method_allowed(v)))
    script += f"ENV NERFBASELINES_ALLOWED_METHODS={allowed_methods}\n"
    script += f'ENV PYTHONPATH="{package_path}:$PYTHONPATH"\n'
    # Add nerfbaselines to the path
    script += 'CMD ["nerfbaselines"]\n'
    script += "COPY . " + package_path + "\n"
    return script


def _delete_old_docker_images(image_name: str):
    _image_name, tag = image_name.split(":")
    assert "-" in tag, f"Tag '{tag}' must contain a dash"
    prefix_name = image_name[:image_name.rfind("-")]
    for line in subprocess.check_output(f'docker images {prefix_name}-* --format json'.split()).splitlines():
        image = json.loads(line)
        if len(image["Tag"]) == DOCKER_TAG_HASH_LENGTH and image["Tag"] != tag:
            logging.info(f"Deleting old image {image['Tag']}")
            subprocess.check_call(f'docker rmi {image["ID"]}'.split())


def docker_image_exists_remotely(name: str):
    # Test if docker is available first
    # parts = name.split("/")
    # if len(parts) == 3 and parts[0] == "ghcr.io":
    #     hub, namespace, image = parts
    #     url = rf'https://{hub}/token?scope="repository:{namespace}/{image}:pull"'
    #     print(url)
    #     breakpoint()
    #     return requests.get(url).status_code == 200
    # elif len(parts) == 2:  # Docker Hub
    #     namespace, image = parts
    #     tag = "latest"
    #     if ":" in image:
    #         image, tag = image.split(":")
    #     response = requests.get(f'https://hub.docker.com/v2/repositories/{namespace}/{image}/tags/{tag}')
    #     return response.status_code == 200

    try:
        subprocess.check_call(["docker", "manifest", "inspect", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def _build_docker_image(name, dockerfile, skip_if_exists_remotely: bool = False, push: bool = False):
    if skip_if_exists_remotely:
        if docker_image_exists_remotely(name):
            logging.info(f"Image {name} already exists remotely, skipping build")
            return False
        logging.info(f"Image {name} does not exist remotely, building it locally")

    with tempfile.TemporaryDirectory() as tmpdir:
        package_path = str(Path(nerfbaselines.__file__).absolute().parent.parent)
        shutil.copytree(os.path.join(package_path, "nerfbaselines"), os.path.join(tmpdir, "nerfbaselines"))
        with open(os.path.join(tmpdir, "Dockerfile"), "w") as f:
            f.write(dockerfile)
        subprocess.check_call([
            "docker", "build", tmpdir, "-f", os.path.join(tmpdir, "Dockerfile"),
            "-t", name,
        ])
    logging.info(f'Created image "{name}"')

    if push:
        logging.info(f"Pushing image {name}")
        subprocess.check_call(["docker", "push", name])
    return True


def build_docker_image(spec: Optional['DockerBackendSpec'] = None,
                       skip_if_exists_remotely: bool = False, 
                       push: bool = False,
                       tag_latest: bool = False,
                       tag_latest_even_without_build: bool = False):
    if spec is None:
        name = BASE_IMAGE
        dockerfile = Path(__file__).absolute().parent.joinpath("Dockerfile").read_text()
    elif spec is not None and spec.get("image") is None or spec.get("conda_spec") is not None or spec.get("build_script") is not None:
        name = get_docker_image_name(spec)
        assert name is not None, "Docker image name must be specified"
        dockerfile = docker_get_dockerfile(spec)
    else:
        image = spec.get("image")
        raise ValueError(f"The docker image can only be pulled from {image}")
    was_built = _build_docker_image(
        name,
        dockerfile,
        skip_if_exists_remotely=skip_if_exists_remotely,
        push=push
    )
    if (tag_latest and was_built) or tag_latest_even_without_build:
        latest_name = name[:name.rfind("-")]
        logging.info(f"Tagging {name} as {latest_name}")
        if not was_built:
            subprocess.check_call(["docker", "pull", name])
        subprocess.check_call(["docker", "tag", name, latest_name])
        if push:
            logging.info(f"Pushing image {latest_name}")
            subprocess.check_call(["docker", "push", latest_name])
    _delete_old_docker_images(name)


def get_docker_image_name(spec: DockerBackendSpec):
    force_build = spec.get("build_script") is not None
    if spec.get("conda_spec") is None and not force_build:
        return spec.get("image")
    environment_hash = docker_get_environment_hash(spec)[:DOCKER_TAG_HASH_LENGTH]
    environment_name = spec.get("environment_name")
    assert environment_name is not None, "DockerBackend requires environment_name to be specified"
    return f"{DOCKER_REPOSITORY}:{environment_name}-{environment_hash}"

def get_docker_environments_to_build():
    from .. import registry

    methods = registry.get_supported_methods("docker")
    methods_to_install = {}
    for mname in methods:
        m = registry.get_method_spec(mname)
        spec = m.get("docker", m.get("conda"))
        if not spec:
            continue
        docker_spec = m.get("docker")
        if (
            docker_spec is not None and 
            docker_spec.get("image") is not None and 
            spec.get("build_script") is None
        ):
            # No build needed
            continue
        environment_id = spec.get("environment_name")
        if environment_id is None or environment_id in methods_to_install:
            continue
        methods_to_install[environment_id] = mname
    return list(methods_to_install.keys())


def docker_run_image(spec: DockerBackendSpec, 
                     args, 
                     env, 
                     mounts: Optional[List[Tuple[str, str]]] = None, 
                     ports: Optional[List[Tuple[int, int]]] = None, 
                     use_gpu: bool = True,
                     interactive: bool = True):
    image = get_docker_image_name(spec) or BASE_IMAGE
    os.makedirs(os.path.expanduser("~/.conda/pkgs"), exist_ok=True)
    torch_home = os.path.expanduser(os.environ.get("TORCH_HOME", "~/.cache/torch/hub"))
    os.makedirs(torch_home, exist_ok=True)
    replace_user = spec.get("replace_user")
    replace_user = replace_user if replace_user is not None else True
    package_path = str(Path(nerfbaselines.__file__).absolute().parent.parent)
    args = [
        "docker",
        "run",
        *(
            (
                "--user",
                ":".join(list(map(str, (os.getuid(), os.getgid())))),
                "-v=/etc/group:/etc/group:ro",
                "-v=/etc/passwd:/etc/passwd:ro",
                "-v=/etc/shadow:/etc/shadow:ro",
                "--env",
                f"HOME={shlex.quote(spec.get('home_path') or '/root')}",
            )
            if replace_user
            else ()
        ),
        *(
            (
                "--gpus",
                "all",
            )
            if use_gpu
            else ()
        ),
        "--shm-size=1g",
        "--ulimit", "memlock=-1",
        "--ulimit", "stack=67108864",
        "--workdir",
        os.getcwd(),
        "-v",
        shlex.quote(os.getcwd()) + ":" + shlex.quote(os.getcwd()),
        "-v",
        shlex.quote(NB_PREFIX) + ":/var/nb-prefix",
        "-v",
        shlex.quote(package_path) + ":/var/nb-package",
        "-v",
        shlex.quote(torch_home) + ":/var/nb-torch",
        *[f"-v={shlex.quote(src)}:{shlex.quote(dst)}" for src, dst in mounts or []],
        "--env", "CONDA_PKGS_DIRS=/var/nb-conda-pkgs",
        "--env", "PIP_CACHE_DIR=/var/nb-pip-cache",
        "--env", "TORCH_HOME=/var/nb-torch",
        "--env", "NERFBASELINES_PREFIX=/var/nb-prefix",
        "--env", "NB_USE_GPU=" + ("1" if use_gpu else "0"),
        *(sum((["--env", name] for name in env if name in EXPORT_ENVS), [])),
        *(sum((["-p", f"{ps}:{pd}"] for ps, pd in ports or []), [])),
        "--rm",
        "--network=host",
        ("-it" if interactive else "-i"),
        image,
    ] + args
    return args, env


class DockerBackend(RemoteProcessRPCBackend):
    name = "docker"

    def __init__(self, 
                 spec: DockerBackendSpec, 
                 address: str = "0.0.0.0", 
                 port: Optional[int] = None):
        self._spec = spec
        self._tmpdir = None
        self._applied_mounts = None
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
        # Build the docker image if needed
        image = self._spec.get("image")
        force_build = self._spec.get("build_script") is not None
        if image is None or self._spec.get("conda_spec") is not None or force_build:
            name = get_docker_image_name(self._spec) or BASE_IMAGE
            dockerfile = docker_get_dockerfile(self._spec)
            should_pull = False
            try:
                subprocess.check_call(["docker", "inspect", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except subprocess.CalledProcessError:
                try:
                    subprocess.check_call(["docker", "manifest", "inspect", name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    should_pull = True
                except subprocess.CalledProcessError:
                    logging.warning(f"Image {name} does not exist remotely. Building it locally.")
                    should_pull = False
            if should_pull:
                logging.info(f"Pulling image {name}")
                subprocess.check_call(["docker", "pull", name])
            else:
                _build_docker_image(name, dockerfile, skip_if_exists_remotely=False, push=False)
                _delete_old_docker_images(name)
        else:
            logging.info(f"Pulling image {image}")
            subprocess.check_call(["docker", "pull", image])

    def _launch_worker(self, args, env):
        assert self._tmpdir is not None, "Temporary directory is not initialized"
        # Run docker image
        self._applied_mounts = get_mounts()
        # Using network=host
        # forwarded_ports = get_forwarded_ports()
        # forwarded_ports.append((self._port, self._port))
        if args[0] == "python":
            args[0] = self._spec.get("python_path") or "python"
        return super()._launch_worker(*docker_run_image(
            self._spec, args, env, 
            mounts=self._applied_mounts + [(self._tmpdir.name, "/var/nb-tmp")], 
            ports=[],
            interactive=False,
            use_gpu=os.getenv("GITHUB_ACTIONS") != "true"))

    def shell(self):
        # Run docker image
        env = get_safe_environment()
        mounts = get_mounts()
        # Using network=host
        # forwarded_ports = get_forwarded_ports()
        args, env = docker_run_image(
            self._spec, ["/bin/bash"], env, 
            mounts=mounts,
            ports=[],
            interactive=True)
        os.execvpe(args[0], args, env)
