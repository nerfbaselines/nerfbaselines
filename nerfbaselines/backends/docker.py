import types
import os
from typing import Optional, List, Tuple
import shlex
from ..communication import RemoteProcessMethod, PACKAGE_PATH, NB_PREFIX
from ..utils import partialmethod
from .apptainer import ApptainerMethod


class DockerMethod(RemoteProcessMethod):
    _local_address = "0.0.0.0"
    _export_envs = ["TCNN_CUDA_ARCHITECTURES", "TORCH_CUDA_ARCH_LIST", "CUDAARCHS", "GITHUB_ACTIONS", "NB_PORT", "NB_PATH", "NB_AUTHKEY", "NB_ARGS"]
    _package_path = "/var/nb-package"
    _replace_user = True
    image: Optional[str] = None
    mounts: Optional[List[Tuple[str, str]]] = None
    home_path: str = "/root"
    environments_path: str = "/var/nb-prefix/docker-conda-envs"

    def __init__(self, *args, mounts: Optional[List[Tuple[str, str]]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mounts = list((mounts or []) + (self.mounts or []))
        assert self.image is not None, "DockerMethod requires an image"

    @classmethod
    def to_apptainer(cls):
        if cls == DockerMethod:
            return ApptainerMethod
        elif len(cls.__bases__) > 0 and DockerMethod == cls.__bases__[0]:
            bases = tuple(ApptainerMethod if b == DockerMethod else b for b in cls.__bases__)

            def build(ns):
                ns["__module__"] = cls.__module__
                ns["__doc__"] = cls.__doc__
                for k, v in cls.__dict__.items():
                    ns[k] = v
                if "__init__" in ns:
                    old_init = ns["__init__"]
                    kwargs = getattr(old_init, "__kwargs__", {})
                    if "image" in kwargs:
                        kwargs["image"] = "docker://" + kwargs["image"]
                    ns["__init__"] = partialmethod(ApptainerMethod.__init__, *getattr(old_init, "__args__", tuple()), **kwargs)
                ns["image"] = "docker://" + ns["image"]
                return ns

            return types.new_class(cls.__name__, bases=bases, exec_body=build)
        else:
            raise TypeError(f"Cannot convert {cls} to ApptainerMethod")

    @property
    def shared_path(self) -> Optional[Tuple[str, str]]:
        if self._tmp_shared_dir is None:
            return None
        return (self._tmp_shared_dir.name, "/nb-shared")

    @classmethod
    def _get_install_args(cls) -> Optional[List[str]]:
        assert cls.image is not None, "DockerMethod requires an image"
        sub_args = super(DockerMethod, cls)._get_install_args()  # pylint: disable=assignment-from-none
        if sub_args is None:
            sub_args = ["true"]
        os.makedirs(os.path.expanduser("~/.conda/pkgs"), exist_ok=True)
        os.makedirs(os.path.expanduser("~/.cache/pip"), exist_ok=True)
        torch_home = os.path.expanduser(os.environ.get("TORCH_HOME", "~/.cache/torch/hub"))
        os.makedirs(torch_home, exist_ok=True)
        os.makedirs(os.path.join(NB_PREFIX, "docker-conda-envs"), exist_ok=True)
        uid_gid = ":".join(list(map(str, (os.getuid(), os.getgid()))))
        use_gpu = True
        if os.getenv("GITHUB_ACTIONS") == "true":
            # GitHub Actions does not support GPU
            use_gpu = False
        return [
            "bash",
            "-c",
            f"docker pull {shlex.quote(cls.image)} && "
            + shlex.join(
                [
                    "docker",
                    "run",
                    *(
                        (
                            "--user",
                            uid_gid,
                            "-v=/etc/group:/etc/group:ro",
                            "-v=/etc/passwd:/etc/passwd:ro",
                            "-v=/etc/shadow:/etc/shadow:ro",
                            "--env",
                            f"HOME={shlex.quote(cls.home_path)}",
                        )
                        if cls._replace_user
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
                    "--workdir",
                    os.getcwd(),
                    "-v",
                    shlex.quote(os.getcwd()) + ":" + shlex.quote(os.getcwd()),
                    "-v",
                    shlex.quote(NB_PREFIX) + ":/var/nb-prefix",
                    "-v",
                    shlex.quote(PACKAGE_PATH) + ":" + shlex.quote(cls._package_path),
                    "-v",
                    shlex.quote(os.path.expanduser("~/.conda/pkgs")) + ":/var/nb-conda-pkgs",
                    "-v",
                    shlex.quote(os.path.expanduser("~/.cache/pip")) + ":/var/nb-pip-cache",
                    "-v",
                    shlex.quote(torch_home) + ":/var/nb-torch",
                    *[f"-v={shlex.quote(src)}:{shlex.quote(dst)}" for src, dst in cls.mounts or []],
                    "--env",
                    "NB_PREFIX=/var/nb-prefix",
                    "--env",
                    "CONDA_PKGS_DIRS=/var/nb-conda-pkgs",
                    "--env",
                    "PIP_CACHE_DIR=/var/nb-pip-cache",
                    "--env",
                    "TORCH_HOME=/var/nb-torch",
                    "--env",
                    "NB_USE_GPU=" + ("1" if use_gpu else "0"),
                    *(sum((["--env", name] for name in cls._export_envs), [])),
                    "--rm",
                    cls.image,
                ]
                + sub_args
            ),
        ]

    def _get_server_process_args(self, env, *args, **kwargs):
        python_args = super()._get_server_process_args(env, *args, **kwargs)
        os.makedirs(os.path.expanduser("~/.conda/pkgs"), exist_ok=True)
        torch_home = os.path.expanduser(os.environ.get("TORCH_HOME", "~/.cache/torch/hub"))
        os.makedirs(torch_home, exist_ok=True)
        os.makedirs(os.path.join(NB_PREFIX, "docker-conda-envs"), exist_ok=True)
        uid_gid = ":".join(list(map(str, (os.getuid(), os.getgid()))))
        use_gpu = True
        if os.getenv("GITHUB_ACTIONS") == "true":
            # GitHub Actions does not support GPU
            use_gpu = False
        return [
            "docker",
            "run",
            *(
                (
                    "--user",
                    uid_gid,
                    "-v=/etc/group:/etc/group:ro",
                    "-v=/etc/passwd:/etc/passwd:ro",
                    "-v=/etc/shadow:/etc/shadow:ro",
                    "--env",
                    f"HOME={shlex.quote(self.home_path)}",
                )
                if self._replace_user
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
            "--workdir",
            os.getcwd(),
            "-v",
            shlex.quote(os.getcwd()) + ":" + shlex.quote(os.getcwd()),
            "-v",
            shlex.quote(NB_PREFIX) + ":/var/nb-prefix",
            "-v",
            shlex.quote(PACKAGE_PATH) + ":" + shlex.quote(self._package_path),
            *(("-v", shlex.quote(self.shared_path[0]) + ":" + shlex.quote(self.shared_path[1])) if self.shared_path is not None else []),
            "-v",
            shlex.quote(torch_home) + ":/var/nb-torch",
            *[f"-v={shlex.quote(src)}:{shlex.quote(dst)}" for src, dst in self.mounts or []],
            *([f"-v={shlex.quote(str(self.checkpoint))}:{shlex.quote(str(self.checkpoint))}:ro"] if self.checkpoint is not None else []),
            "--env",
            "CONDA_PKGS_DIRS=/var/nb-conda-pkgs",
            "--env",
            "PIP_CACHE_DIR=/var/nb-pip-cache",
            "--env",
            "TORCH_HOME=/var/nb-torch",
            "--env",
            "NB_PREFIX=/var/nb-prefix",
            "--env",
            "NB_USE_GPU=" + ("1" if use_gpu else "0"),
            *(sum((["--env", name] for name in self._export_envs), [])),
            "-p",
            f"{self.connection_params.port}:{self.connection_params.port}",
            "--rm",
            ("-it" if env.get("_NB_IS_DOCKERFILE") == "1" else "-i"),
            self.image,
        ] + python_args

    @classmethod
    def get_dockerfile(cls):
        sub_args = super(DockerMethod, cls)._get_install_args()  # pylint: disable=assignment-from-none
        script = f"FROM {cls.image}\n"
        if sub_args:
            args_safe = []
            for arg in sub_args:  # pylint: disable=not-an-iterable
                if "\n" in arg:
                    arg = shlex.quote(arg)
                    arg = arg.replace("\n", " \\n\\\n")
                    args_safe.append(f'"$(echo {arg})"')
                else:
                    args_safe.append(shlex.quote(arg))
            script += "RUN " + " ".join(args_safe) + "\n"
        if cls.python_path != "python":
            script += f'RUN ln -s "$(which {cls.python_path})" "/usr/bin/python"' + "\n"
        env = cls._get_isolated_env()
        env["_NB_IS_DOCKERFILE"] = "1"
        entrypoint = super()._get_server_process_args(env)
        script += "ENTRYPOINT [" + ", ".join("'" + x.rstrip("\n") + "'" for x in entrypoint) + "]\n"
        return script
