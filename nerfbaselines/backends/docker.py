import os
from typing import Optional, List, Tuple
import shlex
from ..communication import RemoteProcessMethod, PACKAGE_PATH, NB_PREFIX


class ApptainerMethod(RemoteProcessMethod):
    _local_address = "0.0.0.0"
    image: str = None
    mounts: List[Tuple[str, str]] = None
    home_path: str = "/root"

    def __init__(self, *args,
                 image: Optional[str] = None,
                 mounts: Optional[List[Tuple[str, str]]] = None,
                 home_path: Optional[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if image is not None:
            self.image = image
        if home_path is not None:
            self.home_path = home_path
        self.mounts = list(mounts or self.mounts or [])
        assert self.image is not None, "ApptainerMethod requires an image"

    @property
    def shared_path(self) -> Optional[Tuple[str, str]]:
        if self._tmp_shared_dir is None:
            return None
        return (self._tmp_shared_dir.name, "/nb-shared")

    @property
    def environments_path(self):
        return os.path.join(NB_PREFIX, "apptainer-conda-envs")

    def _get_install_args(self) -> Optional[List[str]]:
        sub_args = super()._get_install_args()  # pylint: disable=assignment-from-none
        if sub_args is None:
            sub_args = ["true"]
        os.makedirs(os.path.expanduser("~/.conda/pkgs"), exist_ok=True)
        os.makedirs(os.path.expanduser("~/.cache/pip"), exist_ok=True)
        return ["apptainer", "exec", "--containall", "--cleanenv",
            "--nv",
            "--bind", shlex.quote(PACKAGE_PATH) + ":" + shlex.quote(PACKAGE_PATH),
            "--bind", shlex.quote(NB_PREFIX) + ":" + shlex.quote(NB_PREFIX),
            "--bind", shlex.quote(os.path.expanduser("~/.conda/pkgs")) + ":/var/nb-conda-pkgs",
            "--bind", shlex.quote(os.path.expanduser("~/.cache/pip")) + ":/var/nb-pip-cache",
            *[f"--bind={shlex.quote(src)}:{shlex.quote(dst)}" for src, dst in self.mounts],
            "--env", f"NB_PREFIX={shlex.quote(NB_PREFIX)}",
            "--env", "CONDA_PKGS_DIRS=/var/nb-conda-pkgs",
            "--env", "PIP_CACHE_DIR=/var/nb-pip-cache",
            self.image] + sub_args

    def _get_server_process_args(self, env, *args, **kwargs):
        python_args = super()._get_server_process_args(env, *args, **kwargs)
        os.makedirs(os.path.expanduser("~/.conda/pkgs"), exist_ok=True)
        return ["apptainer", "exec", "--containall", "--cleanenv",
            "--nv",
            "--bind", shlex.quote(PACKAGE_PATH) + ":" + shlex.quote(PACKAGE_PATH),
            "--bind", shlex.quote(self._tmp_shared_dir.name) + ":/nb-shared",
            "--bind", shlex.quote(NB_PREFIX) + ":" + shlex.quote(NB_PREFIX),
            *[f"--bind={shlex.quote(src)}:{shlex.quote(dst)}" for src, dst in self.mounts],
            *([f"--bind={shlex.quote(self.checkpoint)}:{shlex.quote(self.checkpoint)}:ro"] if self.checkpoint is not None else []),
            "--env", f"NB_PORT={env['NB_PORT']}",
            "--env", f"NB_PATH={env['NB_PATH']}",
            "--env", f"NB_AUTHKEY={env['NB_AUTHKEY']}",
            "--env", f"NB_ARGS={env['NB_ARGS']}",
            "--env", f"NB_PREFIX={shlex.quote(NB_PREFIX)}",
            self.image] + python_args


class DockerMethod(RemoteProcessMethod):
    _local_address = "0.0.0.0"
    image: str = None
    mounts: List[Tuple[str, str]] = None
    home_path: str = "/root"

    def __init__(self, *args,
                 image: Optional[str] = None,
                 mounts: Optional[List[Tuple[str, str]]] = None,
                 home_path: Optional[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        if image is not None:
            self.image = image
        if home_path is not None:
            self.home_path = home_path
        self.mounts = list(mounts or self.mounts or [])
        assert self.image is not None, "DockerMethod requires an image"

    @property
    def environments_path(self):
        return os.path.join(NB_PREFIX, "docker-conda-envs")

    @property
    def shared_path(self) -> Optional[Tuple[str, str]]:
        if self._tmp_shared_dir is None:
            return None
        return (self._tmp_shared_dir.name, "/nb-shared")

    def _get_install_args(self) -> Optional[List[str]]:
        sub_args = super()._get_install_args()  # pylint: disable=assignment-from-none
        if sub_args is None:
            sub_args = ["true"]
        os.makedirs(os.path.expanduser("~/.conda/pkgs"), exist_ok=True)
        os.makedirs(os.path.expanduser("~/.cache/pip"), exist_ok=True)
        uid_gid = ":".join(list(map(str, (os.getuid(), os.getgid()))))
        return ["bash", "-c", f"docker pull {shlex.quote(self.image)} && " + shlex.join([
            "docker", "run",
            "--user", uid_gid,
            "--gpus", "all",
            '-v=/etc/group:/etc/group:ro',
            '-v=/etc/passwd:/etc/passwd:ro',
            '-v=/etc/shadow:/etc/shadow:ro',
            "-v", shlex.quote(PACKAGE_PATH) + ":" + shlex.quote(PACKAGE_PATH),
            "-v", shlex.quote(NB_PREFIX) + ":" + shlex.quote(NB_PREFIX),
            "-v", shlex.quote(os.path.expanduser("~/.conda/pkgs")) + ":/var/nb-conda-pkgs",
            "-v", shlex.quote(os.path.expanduser("~/.cache/pip")) + ":/var/nb-pip-cache",
            *[f"-v={shlex.quote(src)}:{shlex.quote(dst)}" for src, dst in self.mounts],
            "--env", "CONDA_PKGS_DIRS=/var/nb-conda-pkgs",
            "--env", "PIP_CACHE_DIR=/var/nb-pip-cache",
            "--env", "HOME=/root",
            "--env", "NB_PREFIX",
            "--rm", "-it", self.image] + sub_args)
        ]

    def _get_server_process_args(self, *args, **kwargs):
        python_args = super()._get_server_process_args(*args, **kwargs)
        os.makedirs(os.path.expanduser("~/.conda/pkgs"), exist_ok=True)
        uid_gid = ":".join(list(map(str, (os.getuid(), os.getgid()))))
        return ["docker", "run", "--user", uid_gid,
            "--gpus", "all",
            '-v=/etc/group:/etc/group:ro',
            '-v=/etc/passwd:/etc/passwd:ro',
            '-v=/etc/shadow:/etc/shadow:ro',
            "-v", shlex.quote(PACKAGE_PATH) + ":" + shlex.quote(PACKAGE_PATH),
            "-v", shlex.quote(NB_PREFIX) + ":" + shlex.quote(NB_PREFIX),
            "-v", shlex.quote(self._tmp_shared_dir.name) + ":/nb-shared",
            *[f"-v={shlex.quote(src)}:{shlex.quote(dst)}" for src, dst in self.mounts],
            *([f"-v={shlex.quote(self.checkpoint)}:{shlex.quote(self.checkpoint)}:ro"] if self.checkpoint is not None else []),
            "--env", f"HOME={shlex.quote(self.home_path)}",
            "--env", "NB_PORT", "--env", "NB_PATH", "--env", "NB_AUTHKEY", "--env", "NB_ARGS",
            "--env", "NB_PREFIX", "-p", f"{self.connection_params.port}:{self.connection_params.port}",
            "--rm", "-i", self.image] + python_args 

    def get_dockerfile(self):
        sub_args = super()._get_install_args()  # pylint: disable=assignment-from-none
        script = f"FROM {self.image}\n"
        if sub_args:
            args_safe = []
            for arg in sub_args:  # pylint: disable=not-an-iterable
                arg = arg.replace(self.environments_path, "/var/nb-conda-envs")
                if "\n" in arg:
                    arg = shlex.quote(arg)
                    arg = arg.replace('\n', ' \\n\\\n')
                    args_safe.append(f'"$(echo {arg})"')
                else:
                    args_safe.append(shlex.quote(arg))
            script += "RUN " + " ".join(args_safe) + "\n"
        if self.python_path != "python":
            script += f'RUN ln -s "$(which {self.python_path})" "/usr/bin/python"' + "\n"
        env = self._get_isolated_env()
        env["_NB_IS_DOCKERFILE"] = "1"
        entrypoint = super()._get_server_process_args(env)
        script += "ENTRYPOINT [" + ", ".join("'" + x.rstrip("\n") + "'" for x in entrypoint) + "]\n"
        return script
