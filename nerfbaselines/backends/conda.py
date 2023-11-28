import shlex
import os
from typing import Optional
import hashlib

from ..communication import RemoteProcessMethod, NB_PREFIX


class CondaMethod(RemoteProcessMethod):
    conda_name: Optional[str] = None
    environment: Optional[str] = None
    python_version: Optional[str] = None
    install_script: Optional[str] = None
    environments_path: str = os.path.join(NB_PREFIX, "conda-envs")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, python_path="python", **kwargs)
        assert self.conda_name is not None, "CondaMethod requires conda_name to be specified"

    @classmethod
    def get_environment_hash(cls):
        value = hashlib.sha256()
        if cls.python_version is not None:
            value.update(cls.python_version.encode("utf8"))
        if cls.environment is not None:
            value.update(cls.environment.encode("utf8"))
        if cls.install_script is not None:
            value.update(cls.install_script.encode("utf8"))
        return value.hexdigest()

    def _wrap_server_call(self, args):
        return

    @classmethod
    def _get_install_args(cls):
        assert cls.conda_name is not None, "CondaMethod requires conda_name to be specified"
        environment_hash = cls.get_environment_hash()
        env_root_path = os.path.join(cls.environments_path, cls.conda_name, environment_hash)
        env_path = os.path.join(env_root_path, ".e", cls.conda_name)
        args = []
        if cls.python_version is not None:
            args.append(f"python={cls.python_version}")
        sub_install = ""
        sub_install_args = super()._get_install_args()  # pylint: disable=assignment-from-none
        if sub_install_args:
            sub_install = shlex.join(sub_install_args)
        script = f"""set -eo pipefail
# Clear old environments
if [ -d {shlex.quote(os.path.dirname(env_root_path))} ]; then
    for hash in $(ls -1 {shlex.quote(os.path.dirname(env_root_path))}); do
        if [ "$hash" != {shlex.quote(environment_hash)} ]; then
            rm -rf {shlex.quote(os.path.dirname(env_root_path))}"/$hash"
        fi
    done
fi
# Create new environment
eval "$(conda shell.bash hook)"
if [ ! -e {shlex.quote(os.path.join(env_root_path, ".ack.txt"))} ]; then
rm -rf {shlex.quote(env_root_path)}
mkdir -p {shlex.quote(os.path.dirname(env_path))}
{shlex.join(["conda", "create", "--prefix", env_path, "-y"] + args)}
conda activate {shlex.quote(env_path)}
cd {shlex.quote(env_root_path)}
{cls.install_script}
touch {shlex.quote(os.path.join(env_root_path, ".ack.txt"))}
echo "0" > {shlex.quote(os.path.join(env_root_path, ".ack.txt"))}
fi
{sub_install}
"""
        return ["bash", "-c", script]

    def _get_server_process_args(self, env, *args, **kwargs):
        assert self.conda_name is not None, "CondaMethod requires conda_name to be specified"
        return [
            "bash",
            "-c",
            f"""eval "$(conda shell.bash hook)" && \
conda activate {os.path.join(self.environments_path, self.conda_name, self.get_environment_hash(), ".e", self.conda_name)} && \
exec {shlex.join(super()._get_server_process_args(env, *args, **kwargs))}
""",
        ]
