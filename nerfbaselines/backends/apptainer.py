import os
from typing import Optional, List, Tuple
import shlex
from ..communication import RemoteProcessMethod, PACKAGE_PATH, NB_PREFIX


class ApptainerMethod(RemoteProcessMethod):
    _local_address = "0.0.0.0"
    _export_envs = ["TCNN_CUDA_ARCHITECTURES", "TORCH_CUDA_ARCH_LIST", "CUDAARCHS", "GITHUB_ACTIONS", "NB_PORT", "NB_PATH", "NB_AUTHKEY", "NB_ARGS", "NB_PREFIX"]
    _package_path = "/var/nb-package"
    image: Optional[str] = None
    mounts: Optional[List[Tuple[str, str]]] = None
    home_path: str = "/root"
    environments_path: str = "/var/nb-prefix/apptainer-conda-envs"

    def __init__(self, *args, mounts: Optional[List[Tuple[str, str]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mounts = list((mounts or []) + (self.mounts or []))
        assert self.image is not None, "ApptainerMethod requires an image"

    @classmethod
    def _get_isolated_env(cls):
        out = super(ApptainerMethod, cls)._get_isolated_env()
        allowed = {"APPTAINER_IMAGES", "APPTAINER_CACHEDIR"}
        out.update({k: v for k, v in os.environ.items() if k in allowed})
        return out

    @property
    def shared_path(self) -> Optional[Tuple[str, str]]:
        if self._tmp_shared_dir is None:
            return None
        return (self._tmp_shared_dir.name, "/nb-shared")

    @classmethod
    def _get_install_args(cls) -> Optional[List[str]]:
        sub_args = super(ApptainerMethod, cls)._get_install_args()  # pylint: disable=assignment-from-none
        if sub_args is None:
            sub_args = ["true"]
        os.makedirs(os.path.join(NB_PREFIX, "apptainer-conda-envs"), exist_ok=True)
        conda_cache = os.path.expanduser(os.environ.get("CONDA_PKGS_DIRS", "~/.conda/pkgs"))
        os.makedirs(conda_cache, exist_ok=True)
        pip_cache = os.path.expanduser(os.environ.get("PIP_CACHE_DIR", "~/.cache/pip"))
        os.makedirs(pip_cache, exist_ok=True)
        torch_home = os.path.expanduser(os.environ.get("TORCH_HOME", "~/.cache/torch/hub"))
        os.makedirs(torch_home, exist_ok=True)
        use_gpu = True
        if os.getenv("GITHUB_ACTIONS") == "true":
            # GitHub Actions does not support GPU
            use_gpu = False
        return [
            "apptainer",
            "exec",
            # "--containall",
            "--cleanenv",
            *(("--nv",) if use_gpu else ()),
            "--bind",
            "/tmp:/tmp",
            "--writable-tmpfs",
            "--no-home",
            "-H",
            cls.home_path,
            "--workdir",
            os.getcwd(),
            "--bind",
            shlex.quote(os.getcwd()) + ":" + shlex.quote(os.getcwd()),
            "--bind",
            shlex.quote(PACKAGE_PATH) + ":" + shlex.quote(cls._package_path),
            "--bind",
            shlex.quote(NB_PREFIX) + ":/var/nb-prefix",
            "--bind",
            shlex.quote(conda_cache) + ":/var/nb-conda-pkgs",
            "--bind",
            shlex.quote(pip_cache) + ":/var/nb-pip-cache",
            "--bind",
            shlex.quote(torch_home) + ":/var/nb-torch",
            *[f"--bind={shlex.quote(src)}:{shlex.quote(dst)}" for src, dst in cls.mounts or []],
            "--env",
            "NB_USE_GPU=" + ("1" if use_gpu else "0"),
            "--env",
            "CONDA_PKGS_DIRS=/var/nb-conda-pkgs",
            "--env",
            "PIP_CACHE_DIR=/var/nb-pip-cache",
            "--env",
            "TORCH_HOME=/var/nb-torch",
            "--env",
            "NB_PREFIX=/var/nb-prefix",
            "--env",
            "COLUMNS=120",
            *(sum((["--env", f"{name}={shlex.quote(os.environ.get(name, ''))}"] for name in cls._export_envs if name in os.environ), [])),
            cls.image,
        ] + sub_args

    def _get_server_process_args(self, env, *args, **kwargs):
        python_args = super()._get_server_process_args(env, *args, **kwargs)
        os.makedirs(os.path.join(NB_PREFIX, "apptainer-conda-envs"), exist_ok=True)
        conda_cache = os.path.expanduser(env.get("CONDA_PKGS_DIRS", "~/.conda/pkgs"))
        os.makedirs(conda_cache, exist_ok=True)
        pip_cache = os.path.expanduser(env.get("PIP_CACHE_DIR", "~/.cache/pip"))
        os.makedirs(pip_cache, exist_ok=True)
        torch_home = os.path.expanduser(env.get("TORCH_HOME", "~/.cache/torch/hub"))
        os.makedirs(torch_home, exist_ok=True)
        use_gpu = True
        if os.getenv("GITHUB_ACTIONS") == "true":
            # GitHub Actions does not support GPU
            use_gpu = False
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
            self.home_path,
            "--workdir",
            os.getcwd(),
            "--bind",
            shlex.quote(os.getcwd()) + ":" + shlex.quote(os.getcwd()),
            "--bind",
            shlex.quote(PACKAGE_PATH) + ":" + shlex.quote(self._package_path),
            *(("--bind", shlex.quote(self.shared_path[0]) + ":" + shlex.quote(self.shared_path[1])) if self.shared_path is not None else []),
            "--bind",
            shlex.quote(NB_PREFIX) + ":/var/nb-prefix",
            "--bind",
            shlex.quote(conda_cache) + ":/var/nb-conda-pkgs",
            "--bind",
            shlex.quote(pip_cache) + ":/var/nb-pip-cache",
            "--bind",
            shlex.quote(torch_home) + ":/var/nb-torch",
            *[f"--bind={shlex.quote(src)}:{shlex.quote(dst)}" for src, dst in self.mounts or []],
            *([f"--bind={shlex.quote(str(self.checkpoint))}:{shlex.quote(str(self.checkpoint))}:ro"] if self.checkpoint is not None else []),
            *(sum((["--env", f"{name}={shlex.quote(env.get(name, ''))}"] for name in self._export_envs if name in env), [])),
            "--env",
            "NB_USE_GPU=" + ("1" if use_gpu else "0"),
            "--env",
            "CONDA_PKGS_DIRS=/var/nb-conda-pkgs",
            "--env",
            "NB_PREFIX=/var/nb-prefix",
            "--env",
            "PIP_CACHE_DIR=/var/nb-pip-cache",
            "--env",
            "TORCH_HOME=/var/nb-torch",
            "--env",
            "COLUMNS=120",
            self.image,
        ] + python_args
